//! The conversion of a finished page, or band, to the session's output
//! profile (see [`SessionConfig::output_profile`](crate::SessionConfig::output_profile)),
//! or from the press's ink to the display under a soft proof (see
//! [`SessionConfig::proof`](crate::SessionConfig::proof)).

use std::sync::atomic::{AtomicBool, Ordering};

use pdf_interp::renderer::{CanvasBuffers, FinishedCanvas};
use pdf_interp::{IntentSet, OutputSpace, RenderingIntent, WorkingSpace};

use crate::{RasterError, RasterSession, RenderParams};

/// Refuse `params` asking for a conversion that `session` cannot make: an
/// intent or black-point compensation without an output profile, or
/// compensation under absolute colorimetry; and a CMYK page over a
/// transparent backdrop, or under an output profile that is not CMYK, and
/// an sRGB page under a CMYK one; under a soft proof, a CMYK page or a
/// transparent backdrop; the wide working space at 8 bits.
pub(crate) fn check_params(
    session: &RasterSession,
    params: &RenderParams,
) -> Result<(), RasterError> {
    if session.working_space.requested == Some(WorkingSpace::Wide)
        && params.depth == color::SampleDepth::Eight
    {
        return Err(RasterError::InvalidOptions(
            "the wide working space is composited at 16 bits, at which its levels are as \
             fine as sRGB's at 8: render the page at 16 bits (RenderParams::depth, rrocket \
             --depth 16)"
                .to_owned(),
        ));
    }
    if session.proof.is_some() {
        return check_proof(params);
    }
    if params.color == crate::ColorModel::Cmyk {
        check_cmyk(session, params)?;
    } else if session
        .output_profile
        .as_ref()
        .is_some_and(|p| p.space() == OutputSpace::Cmyk)
    {
        return Err(RasterError::InvalidOptions(
            "the session's output profile is CMYK, which separates CMYK pages: render the \
             pages as CMYK (RenderParams::with_cmyk, ColorModel::Cmyk)"
                .to_owned(),
        ));
    }
    if session.output_profile.is_none()
        && (params.intent.is_some() || params.black_point_compensation)
    {
        return Err(RasterError::InvalidOptions(
            "a rendering intent and black-point compensation apply to the conversion to an \
             output profile: give the session one (SessionConfig::output_profile, rrocket \
             --output-profile)"
                .to_owned(),
        ));
    }
    check_compensation(params)
}

/// Refuse `params` that a soft proof cannot render: CMYK pages, a
/// transparent backdrop, and compensation under absolute colorimetry.
fn check_proof(params: &RenderParams) -> Result<(), RasterError> {
    if params.color == crate::ColorModel::Cmyk {
        return Err(RasterError::InvalidOptions(
            "a soft proof converts the press's ink to the display: render RGB or gray pages, \
             or drop the proof for CMYK output"
                .to_owned(),
        ));
    }
    if params.transparent_background {
        return Err(RasterError::InvalidOptions(
            "a soft proof is painted on the press's opaque paper: render it without a \
             transparent backdrop"
                .to_owned(),
        ));
    }
    check_compensation(params)
}

/// Refuse black-point compensation under absolute colorimetry.
fn check_compensation(params: &RenderParams) -> Result<(), RasterError> {
    if params.black_point_compensation
        && params.intent == Some(RenderingIntent::AbsoluteColorimetric)
    {
        return Err(RasterError::InvalidOptions(
            "black-point compensation does not apply to absolute colorimetry, which reproduces \
             measured colour: choose another intent or drop the compensation"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Refuse a CMYK page that `session` and `params` cannot render.
fn check_cmyk(session: &RasterSession, params: &RenderParams) -> Result<(), RasterError> {
    if params.transparent_background {
        return Err(RasterError::InvalidOptions(
            "a CMYK page is opaque: render it without a transparent backdrop".to_owned(),
        ));
    }
    if session
        .output_profile
        .as_ref()
        .is_some_and(|p| p.space() != OutputSpace::Cmyk)
    {
        return Err(RasterError::InvalidOptions(
            "the session's output profile is RGB or gray, which converts sRGB pages: a CMYK \
             page takes a CMYK output profile or none"
                .to_owned(),
        ));
    }
    Ok(())
}

/// The working space a page of `session` rendered under `params` is
/// composited in, `preview` whether it is painted as an overprint
/// preview: sRGB for a preview, whose ink model is defined from sRGB;
/// otherwise the session's [`working_space`](crate::SessionConfig::working_space),
/// by default the wide space at 16 bits under an output profile and sRGB
/// otherwise.
pub(crate) fn working_space(
    session: &RasterSession,
    params: &RenderParams,
    preview: bool,
) -> WorkingSpace {
    if preview {
        return WorkingSpace::Srgb;
    }
    default_space(session, params)
}

/// The working space of a page of `session` under `params` that is not an
/// overprint preview.
fn default_space(session: &RasterSession, params: &RenderParams) -> WorkingSpace {
    match session.working_space.requested {
        Some(space) => space,
        None if session.output_profile.is_some() && params.depth == color::SampleDepth::Sixteen => {
            WorkingSpace::Wide
        }
        None => WorkingSpace::Srgb,
    }
}

/// The working space a session asked for, and the departures of its pages
/// from it reported so far, each once (see [`working_space_warnings`]).
#[derive(Debug, Default)]
pub(crate) struct SessionSpace {
    /// [`SessionConfig::working_space`](crate::SessionConfig::working_space).
    requested: Option<WorkingSpace>,
    /// An 8-bit page composited in sRGB under an output profile.
    narrow: AtomicBool,
    /// A wide page whose group declares a blending space.
    group: AtomicBool,
    /// An overprint-preview page composited in sRGB under an output
    /// profile.
    preview: AtomicBool,
}

impl SessionSpace {
    /// The space of a session asking for `requested`, nothing reported.
    pub(crate) fn new(requested: Option<WorkingSpace>) -> Self {
        Self {
            requested,
            ..Self::default()
        }
    }
}

/// Log, once per session each, the pages whose working space departs from
/// what an output profile promises (see [`working_space_warnings`]).
pub(crate) fn warn_working_space(
    session: &RasterSession,
    page: (u32, pdf::ObjectId),
    params: &RenderParams,
    canvas: &pdf_interp::CanvasRequest,
) {
    for warning in working_space_warnings(session, page, params, canvas) {
        log::warn!("{warning}");
    }
}

/// The warnings, each the first of its kind in the session, of page
/// `page_num` (object `page_id`) rendered under `params` on `canvas`
/// under an output profile: an 8-bit page composited in sRGB, unless the
/// session asked for sRGB; an overprint preview composited in sRGB where
/// the page would otherwise be wide; a wide page whose group declares a
/// blending space, which it blends in the working space instead.
pub(crate) fn working_space_warnings(
    session: &RasterSession,
    (page_num, page_id): (u32, pdf::ObjectId),
    params: &RenderParams,
    canvas: &pdf_interp::CanvasRequest,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if session.output_profile.is_none() {
        return warnings;
    }
    let once = |flag: &AtomicBool| !flag.swap(true, Ordering::Relaxed);
    let space = canvas.space();
    let preview = matches!(canvas, pdf_interp::CanvasRequest::OverprintPreview);
    if preview && default_space(session, params) == WorkingSpace::Wide {
        if once(&session.working_space.preview) {
            warnings.push(format!(
                "page {page_num} is painted as an overprint preview, whose ink model is \
                 defined from sRGB: it is composited in sRGB and its colours outside sRGB \
                 clip before the conversion to the output profile"
            ));
        }
    } else if space == WorkingSpace::Srgb
        && params.depth == color::SampleDepth::Eight
        && session.working_space.requested.is_none()
        && once(&session.working_space.narrow)
    {
        warnings.push(format!(
            "page {page_num} is composited in sRGB at 8 bits: its colours outside sRGB clip \
             before the conversion to the output profile; render at 16 bits \
             (RenderParams::depth, rrocket --depth 16) to composite in the wide working \
             space, or ask for sRGB (SessionConfig::working_space, rrocket --working-space \
             srgb) to keep this"
        ));
    }
    if space == WorkingSpace::Wide
        && pdf_interp::page_group_declares_space(&session.doc, page_id)
        && once(&session.working_space.group)
    {
        warnings.push(format!(
            "page {page_num}'s transparency group declares a blending colour space: it blends \
             in the wide working space instead"
        ));
    }
    warnings
}

/// The intent a page whose marking operators used `content` converts
/// under: `explicit` when given, else the page's one intent, else
/// relative colorimetric.  The second value is whether the page mixed
/// intents and none was given.
pub(crate) fn page_intent(
    explicit: Option<RenderingIntent>,
    content: IntentSet,
) -> (RenderingIntent, bool) {
    match (explicit, content.single()) {
        (Some(intent), _) | (None, Some(intent)) => (intent, false),
        (None, None) => (RenderingIntent::RelativeColorimetric, !content.is_empty()),
    }
}

/// `canvas`, rows of page `page_num` (object `page_id`) drawn under
/// `params` in the working space `space`, converted to the session's
/// output profile under the page's
/// intent (see [`page_intent`]) given its marking operators' `content`
/// intents; returned with the intent, or as it is with `None` without a
/// profile.  The first page of mixed intents in the session logs a
/// warning.  Under a soft proof the canvas is the press's ink, converted
/// to the display.  The colours of a CMYK page and of a proof separate
/// under each object's intent, or `params.intent` for all; the intent
/// returned for them is the page's ([`page_intent`]).
///
/// # Errors
///
/// [`RasterError::OutputProfile`] when the conversion cannot be built or
/// fails; [`RasterError::OutputIntent`] when the page has no press.
pub(crate) fn convert(
    session: &RasterSession,
    params: &RenderParams,
    (page_num, page_id): (u32, pdf::ObjectId),
    (canvas, content, space): (FinishedCanvas, IntentSet, WorkingSpace),
    buffers: &mut CanvasBuffers,
) -> Result<(FinishedCanvas, Option<RenderingIntent>), RasterError> {
    if let Some(proof) = &session.proof {
        let canvas = proof
            .for_page(&session.doc, page_id)?
            .convert(canvas, proof.intent, buffers)
            .map_err(RasterError::OutputProfile)?;
        return Ok((canvas, Some(page_intent(params.intent, content).0)));
    }
    let Some(profile) = &session.output_profile else {
        return Ok((canvas, None));
    };
    if canvas.is_cmyk() {
        return Ok((canvas, Some(page_intent(params.intent, content).0)));
    }
    let (intent, mixed) = page_intent(params.intent, content);
    if mixed && !session.mixed_intents_warned.swap(true, Ordering::Relaxed) {
        log::warn!(
            "page {page_num} paints under the rendering intents {content}; a page converts to \
             the output profile under one, so pages of mixed intents convert with \
             RelativeColorimetric (an explicit intent, rrocket --intent, chooses another)"
        );
    }
    let canvas = profile
        .convert(
            canvas,
            space,
            (intent, params.black_point_compensation),
            buffers,
        )
        .map_err(RasterError::OutputProfile)?;
    Ok((canvas, Some(intent)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_intent_is_the_explicit_one_else_the_content_s_one() {
        use RenderingIntent::{Perceptual, RelativeColorimetric, Saturation};
        let mut one = IntentSet::EMPTY;
        one.insert(Perceptual);
        let mut two = one;
        two.insert(Saturation);
        assert_eq!(
            page_intent(None, IntentSet::EMPTY),
            (RelativeColorimetric, false)
        );
        assert_eq!(page_intent(None, one), (Perceptual, false));
        assert_eq!(page_intent(None, two), (RelativeColorimetric, true));
        assert_eq!(page_intent(Some(Saturation), two), (Saturation, false));
    }

    /// A session of two pages under the RGB test printer profile; page 2's
    /// group declares a blending space.
    fn profiled_session() -> RasterSession {
        let content = crate::test_pdf::stream("0 g 0 0 1 1 re f");
        let pdf = crate::test_pdf::assemble(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 5 0 R >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /Contents 5 0 R \
             /Group << /S /Transparency /CS /DeviceRGB >> >>",
            &content,
        ]);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/icc/printer-rgb-v4.icc"
        );
        let profile = pdf_interp::OutputProfile::from_bytes(std::fs::read(path).unwrap()).unwrap();
        let mut config = crate::SessionConfig::with_policy(crate::BackendPolicy::CpuOnly);
        config.output_profile = Some(std::sync::Arc::new(profile));
        crate::open_session_from_bytes(pdf, &config).unwrap()
    }

    /// The working-space warnings of page `page_num` of `session` under
    /// `params` on `canvas`.
    fn warnings(
        session: &RasterSession,
        page_num: u32,
        params: &RenderParams,
        canvas: &pdf_interp::CanvasRequest,
    ) -> Vec<String> {
        let page_id = session.resolve_page(page_num).unwrap();
        working_space_warnings(session, (page_num, page_id), params, canvas)
    }

    #[test]
    fn each_working_space_warning_names_its_first_page_once() {
        use pdf_interp::CanvasRequest::{OverprintPreview, Rgb, WideRgb};
        let eight = RenderParams::new(1.0);
        let sixteen = RenderParams::new(1.0).with_depth(color::SampleDepth::Sixteen);
        let s = profiled_session();
        let first = warnings(&s, 1, &eight, &Rgb);
        assert!(first.len() == 1 && first[0].starts_with("page 1 is composited in sRGB at 8"));
        assert_eq!(warnings(&s, 2, &eight, &Rgb), Vec::<String>::new());
        let preview = warnings(&s, 2, &sixteen, &OverprintPreview);
        assert!(preview.len() == 1 && preview[0].starts_with("page 2 is painted as an overprint"));
        assert_eq!(
            warnings(&s, 1, &sixteen, &OverprintPreview),
            Vec::<String>::new()
        );
        assert_eq!(warnings(&s, 1, &sixteen, &WideRgb), Vec::<String>::new());
        let group = warnings(&s, 2, &sixteen, &WideRgb);
        assert!(group.len() == 1 && group[0].starts_with("page 2's transparency group"));
        assert_eq!(warnings(&s, 2, &sixteen, &WideRgb), Vec::<String>::new());
    }

    #[test]
    fn an_explicit_srgb_space_is_not_warned_about() {
        let mut s = profiled_session();
        s.working_space = SessionSpace::new(Some(WorkingSpace::Srgb));
        let eight = RenderParams::new(1.0);
        assert_eq!(
            warnings(&s, 1, &eight, &pdf_interp::CanvasRequest::Rgb),
            Vec::<String>::new()
        );
    }
}
