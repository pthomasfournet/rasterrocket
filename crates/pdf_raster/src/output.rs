//! The conversion of a finished page, or band, to the session's output
//! profile (see [`SessionConfig::output_profile`](crate::SessionConfig::output_profile)),
//! or from the press's ink to the display under a soft proof (see
//! [`SessionConfig::proof`](crate::SessionConfig::proof)).

use std::sync::atomic::Ordering;

use pdf_interp::renderer::{CanvasBuffers, FinishedCanvas};
use pdf_interp::{IntentSet, OutputSpace, RenderingIntent};

use crate::{RasterError, RasterSession, RenderParams};

/// Refuse `params` asking for a conversion that `session` cannot make: an
/// intent or black-point compensation without an output profile, or
/// compensation under absolute colorimetry; and a CMYK page over a
/// transparent backdrop, or under an output profile that is not CMYK, and
/// an sRGB page under a CMYK one; under a soft proof, a CMYK page or a
/// transparent backdrop.
pub(crate) fn check_params(
    session: &RasterSession,
    params: &RenderParams,
) -> Result<(), RasterError> {
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

/// The intent a CMYK page separates under: `params.intent`, else relative
/// colorimetry.  The separation is made as the page is painted, before
/// its content's intents are known.
pub(crate) fn cmyk_intent(params: &RenderParams) -> RenderingIntent {
    params
        .intent
        .unwrap_or(RenderingIntent::RelativeColorimetric)
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
/// `params`, converted to the session's output profile under the page's
/// intent (see [`page_intent`]) given its marking operators' `content`
/// intents; returned with the intent, or as it is with `None` without a
/// profile.  The first page of mixed intents in the session logs a
/// warning.  Under a soft proof the canvas is the press's ink, converted
/// to the display; the intent returned is the one its colours separated
/// under ([`cmyk_intent`]).
///
/// # Errors
///
/// [`RasterError::OutputProfile`] when the conversion cannot be built or
/// fails; [`RasterError::OutputIntent`] when the page has no press.
pub(crate) fn convert(
    session: &RasterSession,
    params: &RenderParams,
    (page_num, page_id): (u32, pdf::ObjectId),
    (canvas, content): (FinishedCanvas, IntentSet),
    buffers: &mut CanvasBuffers,
) -> Result<(FinishedCanvas, Option<RenderingIntent>), RasterError> {
    if let Some(proof) = &session.proof {
        let canvas = proof
            .for_page(&session.doc, page_id)?
            .convert(canvas, proof.intent, buffers)
            .map_err(RasterError::OutputProfile)?;
        return Ok((canvas, Some(cmyk_intent(params))));
    }
    let Some(profile) = &session.output_profile else {
        return Ok((canvas, None));
    };
    if canvas.is_cmyk() {
        return Ok((canvas, Some(cmyk_intent(params))));
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
        .convert(canvas, (intent, params.black_point_compensation), buffers)
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
}
