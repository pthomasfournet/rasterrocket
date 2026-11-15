//! The conversion of a finished page, or band, to the session's output
//! profile (see [`SessionConfig::output_profile`](crate::SessionConfig::output_profile)).

use std::sync::atomic::Ordering;

use pdf_interp::renderer::{CanvasBuffers, FinishedCanvas};
use pdf_interp::{IntentSet, RenderingIntent};

use crate::{RasterError, RasterSession, RenderParams};

/// Refuse `params` asking for a conversion that `session` cannot make: an
/// intent or black-point compensation without an output profile, or
/// compensation under absolute colorimetry.
pub(crate) fn check_params(
    session: &RasterSession,
    params: &RenderParams,
) -> Result<(), RasterError> {
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

/// `canvas`, rows of page `page_num` drawn under `params`, converted to the
/// session's output profile under the page's intent (see [`page_intent`])
/// given its marking operators' `content` intents; returned with the
/// intent, or as it is with `None` without a profile.  The first page of
/// mixed intents in the session logs a warning.
///
/// # Errors
///
/// [`RasterError::OutputProfile`] when the conversion cannot be built or
/// fails.
pub(crate) fn convert(
    session: &RasterSession,
    params: &RenderParams,
    page_num: u32,
    (canvas, content): (FinishedCanvas, IntentSet),
    buffers: &mut CanvasBuffers,
) -> Result<(FinishedCanvas, Option<RenderingIntent>), RasterError> {
    let Some(profile) = &session.output_profile else {
        return Ok((canvas, None));
    };
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
