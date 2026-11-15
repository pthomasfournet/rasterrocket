//! Rendering a page in horizontal bands, so that a page larger than memory
//! is drawn a band at a time.
//!
//! Each band is a window of the page drawn by a renderer of its own over
//! the page's content, parsed once; the band holds the device pixels the
//! whole-page render places in its rows.  The renderers of a page share its
//! decoded images, forms, shadings and tiling cells, its tint-transform
//! budget, its spot colorants' slots and its diagnostics
//! ([`pdf_interp::renderer::PageState`]), and each starts its own operator
//! budget and deadline.

use std::num::NonZeroU32;

use color::{Cmyk8, Cmyk16, Gray8, Gray16, Rgb8, Rgb16};
use pdf_interp::renderer::{CanvasBuffers, FinishedCanvas, PageDiagnostics, PageState};
use raster::Bitmap;

use crate::render::{
    PageJob, allocation_failed, begin, check_aborted, check_strict_images, log_failures,
    page_content, page_view, render_page_canvas_with_geom, warn_unsupported,
};
use crate::{RasterError, RasterSession, RenderParams};

/// One band of a page render: rows `y .. y + bitmap.height()` of the view
/// drawn (the page, or the region of [`RenderParams::region`]), the full
/// width of the view.
#[derive(Clone, Copy)]
pub struct Band<'a> {
    /// The first row of the band, counted from the top of the view.
    pub y: u32,
    /// The band's pixels, at the render's depth, with an alpha plane over
    /// a transparent backdrop.
    pub bitmap: BandBitmap<'a>,
}

/// The pixels of a [`Band`], of the depth of [`RenderParams::depth`]:
/// sRGB, or the device values of the session's
/// [`output_profile`](crate::SessionConfig::output_profile), or the
/// process ink of a page rendered as CMYK.
#[derive(Clone, Copy)]
pub enum BandBitmap<'a> {
    /// 8-bit RGB.
    Rgb8(&'a Bitmap<Rgb8>),
    /// 16-bit RGB.
    Rgb16(&'a Bitmap<Rgb16>),
    /// 8-bit gray, of a gray output profile.
    Gray8(&'a Bitmap<Gray8>),
    /// 16-bit gray, of a gray output profile.
    Gray16(&'a Bitmap<Gray16>),
    /// 8-bit CMYK, 0 no ink.
    Cmyk8(&'a Bitmap<Cmyk8>),
    /// 16-bit CMYK, 0 no ink.
    Cmyk16(&'a Bitmap<Cmyk16>),
}

impl<'a> BandBitmap<'a> {
    /// The whole of `canvas` as a band.
    #[must_use]
    pub const fn of(canvas: &'a FinishedCanvas) -> Self {
        match canvas {
            FinishedCanvas::Rgb8(b) => Self::Rgb8(b),
            FinishedCanvas::Rgb16(b) => Self::Rgb16(b),
            FinishedCanvas::Gray8(b) => Self::Gray8(b),
            FinishedCanvas::Gray16(b) => Self::Gray16(b),
            FinishedCanvas::Cmyk8(b) => Self::Cmyk8(b),
            FinishedCanvas::Cmyk16(b) => Self::Cmyk16(b),
        }
    }

    /// The bitmap's `(width, height)` in pixels.
    #[must_use]
    pub const fn dims(self) -> (u32, u32) {
        match self {
            Self::Rgb8(b) => (b.width, b.height),
            Self::Rgb16(b) => (b.width, b.height),
            Self::Gray8(b) => (b.width, b.height),
            Self::Gray16(b) => (b.width, b.height),
            Self::Cmyk8(b) => (b.width, b.height),
            Self::Cmyk16(b) => (b.width, b.height),
        }
    }

    /// Rows of the band.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.dims().1
    }
}

/// The error of [`render_page_bands`]: the render's, or the sink's.
#[derive(Debug)]
pub enum BandError<E> {
    /// The page could not be rendered; the bands before it were delivered.
    Raster(RasterError),
    /// The sink refused a band; no later band was rendered.
    Sink(E),
}

impl<E> From<RasterError> for BandError<E> {
    fn from(e: RasterError) -> Self {
        Self::Raster(e)
    }
}

impl<E: std::fmt::Display> std::fmt::Display for BandError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Raster(e) => e.fmt(f),
            Self::Sink(e) => e.fmt(f),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for BandError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Raster(e) => Some(e),
            Self::Sink(e) => Some(e),
        }
    }
}

/// The rows of each band of a `width`-pixel view of `bytes_per_px` bytes a
/// pixel: `params.band_rows`, else as many as the session's
/// [`band_bytes`](crate::SessionConfig::band_bytes) hold, at least one.
fn band_rows(session: &RasterSession, params: &RenderParams, width: u32, bytes_per_px: u32) -> u32 {
    params.band_rows.map_or_else(
        || {
            let row = u64::from(width) * u64::from(bytes_per_px);
            u32::try_from(session.band_bytes / row.max(1))
                .unwrap_or(u32::MAX)
                .max(1)
        },
        NonZeroU32::get,
    )
}

/// Render page `page_num` (1-based) as `params` describes, in bands of
/// whole rows from the top, handing each band to `sink` in order; returns
/// the page's diagnostics.
///
/// The bands, stacked, are the bitmap [`render_page_canvas`] returns for
/// the same `params`, byte for byte.  Memory is one band (sized by
/// [`RenderParams::band_rows`] or the session's
/// [`band_bytes`](crate::SessionConfig::band_bytes)) and the page's decoded
/// resources, not the page.  The byte budget
/// ([`max_page_bytes`](crate::SessionConfig::max_page_bytes)) still bounds
/// the page (or region) drawn: it is the picture the bands make up.
///
/// A page that fits one band is rendered as [`render_page_rgb_with`]
/// renders it (on the GPU page device when the session has one); a page of
/// several bands is rendered on the CPU.  Every band executes the page's
/// whole content stream, so a page of `n` bands interprets its content `n`
/// times.
///
/// # Errors
///
/// [`BandError::Raster`] with the errors of [`render_page_rgb_with`], each
/// band failing as a page does; under a strict session an image failure
/// fails the band in which it is first met.  [`BandError::Sink`] with the
/// error of `sink`.
///
/// [`render_page_rgb_with`]: crate::render_page_rgb_with
/// [`render_page_canvas`]: crate::render_page_canvas
pub fn render_page_bands<E>(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
    mut sink: impl FnMut(Band<'_>) -> Result<(), E>,
) -> Result<PageDiagnostics, BandError<E>> {
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    let policy = begin(session, params)?;
    let bytes_per_px = crate::render::page_layout(session, page_id, params).bytes_per_px();
    let view = page_view(
        &geom,
        (params.scale_x, params.scale_y),
        params.region,
        bytes_per_px,
        session.max_page_bytes,
    )?;
    let rows = band_rows(session, params, view.width, bytes_per_px);
    if rows >= view.height {
        let (bitmap, diag) = render_page_canvas_with_geom(
            session,
            page_num,
            page_id,
            geom,
            params,
            &mut CanvasBuffers::default(),
        )?;
        sink(Band {
            y: 0,
            bitmap: BandBitmap::of(&bitmap),
        })
        .map_err(BandError::Sink)?;
        return Ok(diag);
    }

    let unsupported_features = pdf_interp::page_unsupported_features(&session.doc, page_id);
    warn_unsupported(session, page_num, &unsupported_features);
    let content = page_content(session, page_id)?;
    let ops = pdf_interp::parse_page_content(&session.doc, page_id, &content.bytes);

    let mut buffers = CanvasBuffers::default();
    let mut state: Option<PageState> = None;
    let mut intent = None;
    let mut y = 0;
    while y < view.height {
        let height = rows.min(view.height - y);
        let band_view = view.window(view.x, view.y + y, view.width, height);
        let job = PageJob::new(session, page_num, page_id, &band_view, &ops, params, policy);
        let mut renderer = job.renderer(std::mem::take(&mut buffers))?;
        if let Some(state) = state.take() {
            renderer.adopt_page_state(state);
        }
        job.execute(&mut renderer)?;
        check_aborted(&renderer)?;
        check_strict_images(session, renderer.image_failures())?;
        let (bitmap, next, mut spare) = renderer
            .try_finish_band(params.transparent_background)
            .map_err(allocation_failed)?;
        let (bitmap, output_intent) = crate::output::convert(
            session,
            params,
            page_num,
            (bitmap, next.content_intents()),
            &mut spare,
        )?;
        sink(Band {
            y,
            bitmap: BandBitmap::of(&bitmap),
        })
        .map_err(BandError::Sink)?;
        buffers = spare;
        bitmap.reclaim_into(&mut buffers);
        intent = output_intent;
        state = Some(next);
        y += height;
    }

    // The view has a row, so a band was drawn and `state` holds its page.
    let mut diag = state.map_or_else(PageDiagnostics::default, |s| s.diagnostics());
    log_failures(page_num, &content.failures);
    diag.content_failures = content.failures;
    log_failures(page_num, &diag.image_failures);
    diag.form_type = session.form_type;
    diag.unsupported_features = unsupported_features;
    diag.output_intent = intent;
    Ok(diag)
}
