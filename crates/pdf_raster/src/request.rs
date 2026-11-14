//! Per-call parameters of a session render: scales, device-pixel window,
//! backend override and cancellation.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::BackendPolicy;

#[cfg(doc)]
use crate::{RasterError, RasterSession, render_page_rgb, render_page_rgb_with};

/// A rectangle of device pixels within a page rendered at some scale.
///
/// Coordinates count from the top-left corner of the whole page, whose pixel
/// size at those scales is [`PageGeometry::size_px`](crate::PageGeometry::size_px).
/// Rendering a region draws exactly the pixels the whole-page render places
/// there, so regions that tile the page compose into the whole-page bitmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderRegion {
    /// Left edge, in pixels from the page's left edge.
    pub x: u32,
    /// Top edge, in pixels from the page's top edge.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl RenderRegion {
    /// The region at `(x, y)` of `width × height` pixels.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// A shareable flag that stops renders in progress.
///
/// Clones share one flag.  After [`cancel`](Self::cancel), a render holding a
/// clone stops before its next content-stream operator and returns
/// [`RasterError::Cancelled`]; a render started afterwards returns it without
/// drawing.  A single operator already executing (one image decode, one fill)
/// runs to completion first.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise the flag.  Idempotent; there is no way to lower it again.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether [`cancel`](Self::cancel) has been called on any clone.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// The shared flag, for the renderer's operator loop.
    pub(crate) fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}

/// Parameters of one [`render_page_rgb_with`] call.
///
/// Construct with [`RenderParams::new`] and the `with_*` methods; the struct
/// is `#[non_exhaustive]` so parameters can be added without a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RenderParams {
    /// Horizontal device pixels per PDF point: `x_dpi / 72`.  Must be
    /// positive and finite.
    pub scale_x: f64,
    /// Vertical device pixels per PDF point: `y_dpi / 72`.  Must be positive
    /// and finite.
    pub scale_y: f64,
    /// The window of the page to draw; `None` draws the whole page.
    ///
    /// The byte budget ([`SessionConfig::max_page_bytes`](crate::SessionConfig::max_page_bytes))
    /// applies to the region, not to the page, so a page too large to
    /// render whole can be rendered in regions; the side ceiling
    /// ([`MAX_PX_SIDE`](crate::MAX_PX_SIDE)) applies to the page.
    pub region: Option<RenderRegion>,
    /// Backend policy for this page's image decoders; `None` uses the
    /// session's.  [`BackendPolicy::CpuOnly`] skips GPU decoder setup for a
    /// page known to need none (e.g. one [`prescan_session`](crate::prescan_session)
    /// reports without images).
    pub policy: Option<BackendPolicy>,
    /// Cancellation flag polled during the render; `None` renders to completion.
    pub cancel: Option<CancelToken>,
    /// Render over a transparent backdrop instead of white paper.  The
    /// bitmap then has an alpha plane holding the page's coverage × opacity,
    /// its colour not premultiplied by it; a pixel nothing paints has colour
    /// and alpha 0.  `false` (the default) composites the page onto white
    /// and returns an opaque bitmap without an alpha plane.
    pub transparent_background: bool,
    /// Rows of each band of [`render_page_bands`](crate::render_page_bands);
    /// `None` sizes the bands by the session's
    /// [`band_bytes`](crate::SessionConfig::band_bytes).  Other entry
    /// points render the page whole and ignore it.
    pub band_rows: Option<NonZeroU32>,
}

impl RenderParams {
    /// Render the whole page at `scale` device pixels per point on both
    /// axes, with the session's backend policy.
    #[must_use]
    pub const fn new(scale: f64) -> Self {
        Self::new_xy(scale, scale)
    }

    /// Render the whole page at `scale_x` horizontal and `scale_y` vertical
    /// device pixels per point (non-square pixels, e.g. 300 × 150 dpi).
    #[must_use]
    pub const fn new_xy(scale_x: f64, scale_y: f64) -> Self {
        Self {
            scale_x,
            scale_y,
            region: None,
            policy: None,
            cancel: None,
            transparent_background: false,
            band_rows: None,
        }
    }

    /// Draw only `region` of the page.
    #[must_use]
    pub const fn with_region(mut self, region: RenderRegion) -> Self {
        self.region = Some(region);
        self
    }

    /// Use `policy` for this page's image decoders instead of the session's.
    #[must_use]
    pub const fn with_policy(mut self, policy: BackendPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Render over a transparent backdrop (see
    /// [`transparent_background`](Self::transparent_background)).
    #[must_use]
    pub const fn with_transparent_background(mut self) -> Self {
        self.transparent_background = true;
        self
    }

    /// The layout of the canvas the render allocates.
    #[must_use]
    pub const fn canvas_layout(&self) -> crate::CanvasLayout {
        crate::CanvasLayout::new(color::SampleDepth::Eight, self.transparent_background)
    }

    /// Render bands of `rows` rows (see [`band_rows`](Self::band_rows)).
    #[must_use]
    pub const fn with_band_rows(mut self, rows: NonZeroU32) -> Self {
        self.band_rows = Some(rows);
        self
    }

    /// Stop the render when `cancel` is cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_is_shared_by_clones() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(!token.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
        assert!(token.flag().load(Ordering::Relaxed));
    }

    #[test]
    fn params_builder_sets_each_field() {
        let token = CancelToken::new();
        let params = RenderParams::new(2.0)
            .with_region(RenderRegion::new(1, 2, 3, 4))
            .with_policy(BackendPolicy::CpuOnly)
            .with_cancel(token)
            .with_transparent_background()
            .with_band_rows(NonZeroU32::new(7).unwrap());
        assert_eq!(params.band_rows.map(NonZeroU32::get), Some(7));
        assert!((params.scale_x - 2.0).abs() < f64::EPSILON);
        assert!((params.scale_y - 2.0).abs() < f64::EPSILON);
        assert_eq!(params.region, Some(RenderRegion::new(1, 2, 3, 4)));
        assert_eq!(params.policy, Some(BackendPolicy::CpuOnly));
        assert!(params.cancel.is_some());
        assert!(params.transparent_background);
        let params = RenderParams::new_xy(3.0, 1.5);
        assert!(!params.transparent_background);
        assert_eq!(params.band_rows, None);
        assert!((params.scale_x - 3.0).abs() < f64::EPSILON);
        assert!((params.scale_y - 1.5).abs() < f64::EPSILON);
    }
}
