//! Per-call parameters of a session render: scales, device-pixel window,
//! backend override and cancellation.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color::SampleDepth;

use crate::BackendPolicy;
use pdf_interp::{PreserveBlack, RenderingIntent};

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
    /// The depth of the samples the page is rendered at: 8 bits (the
    /// default), or 16, which [`render_page_canvas`](crate::render_page_canvas)
    /// and [`render_page_bands`](crate::render_page_bands) return as
    /// 16-bit RGB.  A 16-bit render costs twice the bytes of an 8-bit one
    /// against the byte budget.  A page composited in the wide working
    /// space (see [`SessionConfig::working_space`](crate::SessionConfig::working_space))
    /// is painted at 16 bits at either depth, and costs the bytes of a
    /// 16-bit render; at 8 bits its result is rounded to 8 bits after the
    /// conversion to the output profile.
    pub depth: SampleDepth,
    /// The rendering intent of the conversion to the session's
    /// [`output_profile`](crate::SessionConfig::output_profile); `None`
    /// takes the one intent the page's marking operators used, else
    /// relative colorimetric.  Refused without an output profile.
    pub intent: Option<RenderingIntent>,
    /// Whether the conversion to the output profile compensates the black
    /// point (ISO 18619): the working space's black maps to the profile's
    /// darkest neutral.  Refused without an output profile, and under
    /// absolute colorimetry, which reproduces measured colour.
    pub black_point_compensation: bool,
    /// The preservation of black in the conversion of a colour given as
    /// the ink of another CMYK profile to the ink of the session's CMYK
    /// output profile, or of a soft proof's press; `None` converts it
    /// colorimetrically.  Refused without a CMYK page and a CMYK output
    /// profile or a soft proof, and under absolute colorimetry.
    pub preserve_black: Option<PreserveBlack>,
    /// The colour model of the page: sRGB (the default), or the process
    /// ink of a CMYK device, which [`render_page_canvas`](crate::render_page_canvas)
    /// and [`render_page_bands`](crate::render_page_bands) return as
    /// [`FinishedCanvas::Cmyk8`](crate::FinishedCanvas::Cmyk8) or
    /// `Cmyk16`.  A CMYK page is painted on an ink canvas, eight samples a
    /// pixel against the byte budget, and is opaque: refused with
    /// [`transparent_background`](Self::transparent_background), and with
    /// an output profile that is not CMYK.
    pub color: ColorModel,
}

/// The colour model a page is rendered in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorModel {
    /// sRGB, or the device values of the session's RGB or gray output
    /// profile.
    #[default]
    Rgb,
    /// The process ink of a CMYK device, `[C, M, Y, K]`, 0 no ink.  Colours
    /// given in `DeviceCMYK`, a four-component ICC space or as process
    /// colorants paint their components, those of an ICC space whose
    /// profile is not the session's CMYK output profile converted to its
    /// ink; spot colorants fold into their
    /// process equivalents; any other colour is separated, by the session's
    /// CMYK output profile when it has one and otherwise by the device
    /// conversion of ISO 32000-2 §10.4.2, under the graphics state's black
    /// generation and undercolour removal.  Overprint is simulated when the
    /// session simulates it.
    Cmyk,
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
            depth: SampleDepth::Eight,
            intent: None,
            black_point_compensation: false,
            preserve_black: None,
            color: ColorModel::Rgb,
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

    /// Render at samples of `depth` (see [`depth`](Self::depth)).
    #[must_use]
    pub const fn with_depth(mut self, depth: SampleDepth) -> Self {
        self.depth = depth;
        self
    }

    /// Convert to the output profile under `intent` (see
    /// [`intent`](Self::intent)).
    #[must_use]
    pub const fn with_intent(mut self, intent: RenderingIntent) -> Self {
        self.intent = Some(intent);
        self
    }

    /// Compensate the black point in the conversion to the output profile
    /// (see [`black_point_compensation`](Self::black_point_compensation)).
    #[must_use]
    pub const fn with_black_point_compensation(mut self) -> Self {
        self.black_point_compensation = true;
        self
    }

    /// Preserve black by `mode` in the conversion of another CMYK
    /// profile's ink (see [`preserve_black`](Self::preserve_black)).
    #[must_use]
    pub const fn with_preserve_black(mut self, mode: PreserveBlack) -> Self {
        self.preserve_black = Some(mode);
        self
    }

    /// Render the page as CMYK (see [`color`](Self::color)).
    #[must_use]
    pub const fn with_cmyk(mut self) -> Self {
        self.color = ColorModel::Cmyk;
        self
    }

    /// The layout of the canvas the render allocates in the sRGB working
    /// space: of ink for a CMYK page; a page simulating overprint is
    /// painted on ink too (see
    /// [`CanvasLayout::with_ink`](crate::CanvasLayout::with_ink)), and a
    /// page of the wide working space at 16 bits (see
    /// [`depth`](Self::depth)).
    #[must_use]
    pub const fn canvas_layout(&self) -> crate::CanvasLayout {
        crate::CanvasLayout::new(self.depth, self.transparent_background)
            .with_ink(matches!(self.color, ColorModel::Cmyk))
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
            .with_band_rows(NonZeroU32::new(7).unwrap())
            .with_depth(SampleDepth::Sixteen);
        assert_eq!(params.depth, SampleDepth::Sixteen);
        assert_eq!(params.canvas_layout().bytes_per_px(), 8);
        assert_eq!(params.color, ColorModel::Rgb);
        let cmyk = RenderParams::new(1.0).with_cmyk();
        assert_eq!(cmyk.color, ColorModel::Cmyk);
        assert_eq!(cmyk.canvas_layout().bytes_per_px(), 8);
        let wide = cmyk.with_depth(SampleDepth::Sixteen);
        assert_eq!(wide.canvas_layout().bytes_per_px(), 16);
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
        assert_eq!(params.depth, SampleDepth::Eight);
        assert!((params.scale_x - 3.0).abs() < f64::EPSILON);
        assert!((params.scale_y - 1.5).abs() < f64::EPSILON);
    }
}
