//! Rendering one page of a [`RasterSession`] to an RGB bitmap.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use color::{Rgb8, SampleDepth};
use pdf_interp::renderer::{CanvasBuffers, FinishedCanvas};
use raster::Bitmap;

use crate::decoders::{DecoderReclaim, lend_decoders};
use crate::gray::{rgb_into_gray, rgb16_into_gray16};
use crate::{RasterError, RasterOptions, RasterSession, RenderParams, RenderRegion, RenderedPage};

/// Render one whole page to an RGB bitmap at `scale` device pixels per point.
///
/// Equivalent to [`render_page_rgb_with`] with [`RenderParams::new(scale)`](RenderParams::new).
/// `scale` is `dpi / 72.0`; for different horizontal and vertical
/// resolutions use [`RenderParams::new_xy`].
///
/// # Errors
///
/// As [`render_page_rgb_with`].
pub fn render_page_rgb(
    session: &RasterSession,
    page_num: u32,
    scale: f64,
) -> Result<Bitmap<Rgb8>, RasterError> {
    render_page_rgb_with(session, page_num, &RenderParams::new(scale))
}

/// Render page `page_num` (1-based) to an RGB bitmap as `params` describes:
/// the whole page or a [`RenderRegion`] of it, with an optional backend
/// override and [`CancelToken`](crate::CancelToken).
///
/// GPU image decoders are initialised lazily per calling thread on first use
/// and reused across pages — safe to call from multiple rayon threads
/// concurrently.  The bitmap is the region's size when a region is given and
/// the page's ([`PageGeometry::size_px`](crate::PageGeometry::size_px))
/// otherwise.
///
/// # Errors
///
/// - [`RasterError::InvalidOptions`] if a scale is ≤ 0 or non-finite, or the
///   region is empty or extends past the page.
/// - [`RasterError::Cancelled`] if the cancel token is raised before or
///   during the render.
/// - [`RasterError::BackendUnavailable`] if a forced backend fails to init on
///   this thread.
/// - [`RasterError::InvalidPageGeometry`] / [`RasterError::PageDegenerate`] /
///   [`RasterError::PageTooLarge`] / [`RasterError::PageExceedsCeiling`] /
///   [`RasterError::PageAllocationFailed`] / [`RasterError::PageOutOfRange`] /
///   [`RasterError::Pdf`] as documented on the error variants; the size
///   limits apply to the drawn region.
pub fn render_page_rgb_with(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<Bitmap<Rgb8>, RasterError> {
    render_page_rgb_with_diagnostics(session, page_num, params).map(|(bmp, _)| bmp)
}

/// [`render_page_rgb_with`], also returning the [`PageDiagnostics`] the
/// render collected.
///
/// [`PageDiagnostics`]: crate::PageDiagnostics
///
/// # Errors
///
/// As [`render_page_rgb_with`], and [`RasterError::InvalidOptions`] for
/// [`RenderParams::depth`] 16, which [`render_page_canvas`] renders.
pub fn render_page_rgb_with_diagnostics(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<(Bitmap<Rgb8>, crate::PageDiagnostics), RasterError> {
    if params.depth != SampleDepth::Eight {
        return Err(RasterError::InvalidOptions(
            "render_page_rgb returns 8-bit RGB: render 16-bit samples with render_page_canvas"
                .to_owned(),
        ));
    }
    let (canvas, diag) = render_page_canvas_with_diagnostics(session, page_num, params)?;
    Ok((rgb8(canvas)?, diag))
}

/// Render page `page_num` (1-based) as `params` describes, at the depth of
/// [`RenderParams::depth`]: [`FinishedCanvas::Rgb8`] at 8 bits, the bitmap
/// [`render_page_rgb_with`] returns, and [`FinishedCanvas::Rgb16`] at 16.
///
/// # Errors
///
/// As [`render_page_rgb_with`].
pub fn render_page_canvas(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<FinishedCanvas, RasterError> {
    render_page_canvas_with_diagnostics(session, page_num, params).map(|(canvas, _)| canvas)
}

/// [`render_page_canvas`], also returning the [`PageDiagnostics`] the
/// render collected.
///
/// [`PageDiagnostics`]: crate::PageDiagnostics
///
/// # Errors
///
/// As [`render_page_canvas`].
pub fn render_page_canvas_with_diagnostics(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<(FinishedCanvas, crate::PageDiagnostics), RasterError> {
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    render_page_canvas_with_geom(
        session,
        page_num,
        page_id,
        geom,
        params,
        &mut CanvasBuffers::default(),
    )
}

/// The 8-bit bitmap of `canvas`, rendered at 8 bits.
///
/// # Errors
///
/// [`RasterError::InvalidOptions`] for a 16-bit bitmap: the renderer
/// returns a bitmap of the depth it is asked for, so only a caller that
/// asked for 16 bits receives one; and for a gray bitmap, which a gray
/// output profile converts every page to.
pub(crate) fn rgb8(canvas: FinishedCanvas) -> Result<Bitmap<Rgb8>, RasterError> {
    match canvas {
        FinishedCanvas::Rgb8(bitmap) => Ok(bitmap),
        FinishedCanvas::Rgb16(_) => Err(RasterError::InvalidOptions(
            "a 16-bit page where an 8-bit one was rendered".to_owned(),
        )),
        FinishedCanvas::Gray8(_) | FinishedCanvas::Gray16(_) => Err(gray_output()),
        FinishedCanvas::Cmyk8(_) | FinishedCanvas::Cmyk16(_) => Err(cmyk_output()),
    }
}

/// The refusal of an RGB or gray result for a page rendered as CMYK.
pub(crate) fn cmyk_output() -> RasterError {
    RasterError::InvalidOptions(
        "the page is rendered as CMYK: render it with render_page_canvas or render_page_bands"
            .to_owned(),
    )
}

/// The refusal of an RGB result for a page a gray output profile converts
/// to gray.
pub(crate) fn gray_output() -> RasterError {
    RasterError::InvalidOptions(
        "the session's output profile is gray, so its pages are gray: render them with \
         render_page_canvas, render_page_bands or render_page_gray"
            .to_owned(),
    )
}

/// Render page `page_num` (1-based) of `session` to the grayscale
/// [`RenderedPage`] that [`raster_pdf`](crate::raster_pdf) yields for it.
///
/// Of `opts`, only the per-page fields apply: [`dpi`](RasterOptions::dpi),
/// [`depth`](RasterOptions::depth), [`deskew`](RasterOptions::deskew) and
/// [`cancel`](RasterOptions::cancel);
/// [`transparent_background`](RasterOptions::transparent_background) is
/// [`render_page_rgba`]'s.
/// The page selection fields are the caller's, and the fields mirrored on
/// [`SessionConfig`](crate::SessionConfig) were fixed when `session` was
/// opened.
///
/// # Errors
///
/// As [`render_page_rgb_with`], plus [`RasterError::Deskew`] when deskew
/// is asked of a page larger than
/// [`DESKEW_MAX_SIDE`](crate::deskew::DESKEW_MAX_SIDE) and
/// [`RasterError::BackendUnavailable`] when the GPU rotation of
/// [`BackendPolicy::ForceCuda`](crate::BackendPolicy::ForceCuda) fails (see
/// [`deskew::apply`](crate::deskew::apply)).
pub fn render_page_gray(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError> {
    let rgb_profile = session
        .output_profile
        .as_ref()
        .is_some_and(|p| p.space() == pdf_interp::OutputSpace::Rgb);
    if rgb_profile {
        return Err(RasterError::InvalidOptions(
            "a gray page of an RGB output profile's device values has no meaning: give the \
             session a gray output profile, or render the page in colour"
                .to_owned(),
        ));
    }
    let dpi = opts.dpi;
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    let mut params = RenderParams::new(f64::from(dpi) / 72.0).with_depth(opts.depth);
    params.cancel.clone_from(&opts.cancel);
    params.intent = opts.intent;
    params.black_point_compensation = opts.black_point_compensation;
    params.preserve_black = opts.preserve_black;
    let (canvas, diagnostics) = render_page_canvas_with_geom(
        session,
        page_num,
        page_id,
        geom,
        &params,
        &mut CanvasBuffers::default(),
    )?;
    let gray = match canvas {
        FinishedCanvas::Rgb8(rgb) => GrayCanvas::Eight(rgb_into_gray(rgb)),
        FinishedCanvas::Gray8(gray) => GrayCanvas::Eight(gray),
        FinishedCanvas::Rgb16(rgb) => GrayCanvas::Sixteen(rgb16_into_gray16(rgb)),
        FinishedCanvas::Gray16(gray) => GrayCanvas::Sixteen(gray),
        FinishedCanvas::Cmyk8(_) | FinishedCanvas::Cmyk16(_) => return Err(cmyk_output()),
    };
    Ok(match gray {
        GrayCanvas::Eight(mut gray) => {
            let skew = deskewed(&mut gray, opts.deskew, session.policy)?;
            let page = crate::page::gray8_into_rendered_page(gray, page_num, dpi, diagnostics);
            RenderedPage { skew, ..page }
        }
        GrayCanvas::Sixteen(mut gray) => {
            let skew = deskewed(&mut gray, opts.deskew, session.policy)?;
            let page = crate::page::gray16_into_rendered_page(gray, page_num, dpi, diagnostics);
            RenderedPage { skew, ..page }
        }
    })
}

/// A gray page at the depth it was rendered at.
enum GrayCanvas {
    Eight(Bitmap<color::Gray8>),
    Sixteen(Bitmap<color::Gray16>),
}

/// Deskew `gray` under `policy` when `deskew`; the skew measured.
fn deskewed<P: crate::deskew::DeskewPixel>(
    gray: &mut Bitmap<P>,
    deskew: bool,
    policy: crate::BackendPolicy,
) -> Result<Option<crate::deskew::Skew>, RasterError> {
    if !deskew {
        return Ok(None);
    }
    Ok(crate::deskew::apply(gray, policy)?)
}

/// Render page `page_num` (1-based) of `session` over a transparent
/// backdrop to an RGBA [`RenderedPage`].
///
/// The page is the one [`raster_pdf`](crate::raster_pdf) yields when
/// [`RasterOptions::transparent_background`] is set:
/// [`PixelFormat::Rgba8`](crate::PixelFormat::Rgba8), alpha not
/// premultiplied, or [`PixelFormat::Rgba16`](crate::PixelFormat::Rgba16)
/// at [`RasterOptions::depth`] 16.
///
/// Of `opts`, only [`dpi`](RasterOptions::dpi),
/// [`depth`](RasterOptions::depth) and [`cancel`](RasterOptions::cancel)
/// apply.
///
/// # Errors
///
/// As [`render_page_rgb_with`], plus [`RasterError::InvalidOptions`] when
/// [`deskew`](RasterOptions::deskew) is set: deskew operates on gray pages.
pub fn render_page_rgba(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError> {
    if opts.deskew {
        return Err(RasterError::deskew_needs_gray());
    }
    let dpi = opts.dpi;
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    let mut params = RenderParams::new(f64::from(dpi) / 72.0)
        .with_transparent_background()
        .with_depth(opts.depth);
    params.cancel.clone_from(&opts.cancel);
    params.intent = opts.intent;
    params.black_point_compensation = opts.black_point_compensation;
    params.preserve_black = opts.preserve_black;
    let (canvas, diagnostics) = render_page_canvas_with_geom(
        session,
        page_num,
        page_id,
        geom,
        &params,
        &mut CanvasBuffers::default(),
    )?;
    Ok(match canvas {
        FinishedCanvas::Rgb8(rgb) => {
            crate::page::rgb8_to_rgba_rendered_page(&rgb, page_num, dpi, diagnostics)
        }
        FinishedCanvas::Rgb16(rgb) => {
            crate::page::rgb16_to_rgba16_rendered_page(&rgb, page_num, dpi, diagnostics)
        }
        FinishedCanvas::Gray8(_) | FinishedCanvas::Gray16(_) => return Err(gray_output()),
        FinishedCanvas::Cmyk8(_) | FinishedCanvas::Cmyk16(_) => return Err(cmyk_output()),
    })
}

/// Render page `page_num` (1-based) of `session` to the CMYK
/// [`RenderedPage`] that [`raster_pdf`](crate::raster_pdf) yields for it
/// under [`ColorModel::Cmyk`](crate::ColorModel::Cmyk):
/// [`PixelFormat::Cmyk8`](crate::PixelFormat::Cmyk8), or
/// [`PixelFormat::Cmyk16`](crate::PixelFormat::Cmyk16) at
/// [`RasterOptions::depth`] 16.
///
/// Of `opts`, only [`dpi`](RasterOptions::dpi),
/// [`depth`](RasterOptions::depth), [`intent`](RasterOptions::intent),
/// [`black_point_compensation`](RasterOptions::black_point_compensation),
/// [`preserve_black`](RasterOptions::preserve_black) and
/// [`cancel`](RasterOptions::cancel) apply.
///
/// # Errors
///
/// As [`render_page_rgb_with`], plus [`RasterError::InvalidOptions`] when
/// [`deskew`](RasterOptions::deskew) or
/// [`transparent_background`](RasterOptions::transparent_background) is
/// set, and when the session's output profile is not CMYK.
pub fn render_page_cmyk(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError> {
    if opts.deskew {
        return Err(RasterError::deskew_needs_gray());
    }
    let dpi = opts.dpi;
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    let mut params = RenderParams::new(f64::from(dpi) / 72.0)
        .with_cmyk()
        .with_depth(opts.depth);
    params.transparent_background = opts.transparent_background;
    params.cancel.clone_from(&opts.cancel);
    params.intent = opts.intent;
    params.black_point_compensation = opts.black_point_compensation;
    params.preserve_black = opts.preserve_black;
    let (canvas, diagnostics) = render_page_canvas_with_geom(
        session,
        page_num,
        page_id,
        geom,
        &params,
        &mut CanvasBuffers::default(),
    )?;
    match canvas {
        FinishedCanvas::Cmyk8(cmyk) => Ok(crate::page::cmyk8_rendered_page(
            &cmyk,
            page_num,
            dpi,
            diagnostics,
        )),
        FinishedCanvas::Cmyk16(cmyk) => Ok(crate::page::cmyk16_rendered_page(
            &cmyk,
            page_num,
            dpi,
            diagnostics,
        )),
        other => Err(RasterError::InvalidOptions(format!(
            "a CMYK render finished as {other:?}: the renderer finishes a CMYK request as CMYK"
        ))),
    }
}

/// The view a render of `geom` at `scales` (horizontal, vertical) draws: the
/// whole page, or `region` of it.  The page's sides are held to
/// [`MAX_PX_SIDE`](crate::MAX_PX_SIDE), and the bitmap drawn, of
/// `bytes_per_px` bytes per pixel, to `max_bytes`.
pub(crate) fn page_view(
    geom: &pdf_interp::PageGeometry,
    scales: (f64, f64),
    region: Option<RenderRegion>,
    bytes_per_px: u32,
    max_bytes: u64,
) -> Result<pdf_interp::renderer::PageView, RasterError> {
    let whole = pdf_interp::renderer::PageView::whole(geom, scales);
    let (w_px, h_px) = (whole.page_width, whole.page_height);
    if w_px == 0 || h_px == 0 {
        return Err(RasterError::PageDegenerate {
            width: w_px,
            height: h_px,
        });
    }
    // The side ceiling applies to the page whatever is drawn of it: a
    // region's coordinates are page coordinates.  Checked before the
    // bitmap is allocated, never after.
    crate::page::check_sides(w_px, h_px)?;
    let Some(r) = region else {
        crate::page::validate_page_bytes(w_px, h_px, bytes_per_px, max_bytes)?;
        return Ok(whole);
    };
    if r.width == 0 || r.height == 0 {
        return Err(RasterError::InvalidOptions(format!(
            "render region {}×{} is empty",
            r.width, r.height
        )));
    }
    if u64::from(r.x) + u64::from(r.width) > u64::from(w_px)
        || u64::from(r.y) + u64::from(r.height) > u64::from(h_px)
    {
        return Err(RasterError::InvalidOptions(format!(
            "render region {}×{} at ({}, {}) extends past the {w_px}×{h_px} page",
            r.width, r.height, r.x, r.y
        )));
    }
    crate::page::validate_page_bytes(r.width, r.height, bytes_per_px, max_bytes)?;
    Ok(whole.window(r.x, r.y, r.width, r.height))
}

/// [`RasterError::InvalidOptions`] unless `scale` is a positive finite
/// number.
pub(crate) fn check_scale(scale: f64) -> Result<(), RasterError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(RasterError::InvalidOptions(format!(
            "scale must be a positive finite number, got {scale}"
        )))
    }
}

/// The group page `page_id` is painted as: always one over a
/// `transparent` backdrop, so that the page's alpha is kept.
fn page_group(
    doc: &pdf::Document,
    page_id: pdf::ObjectId,
    transparent: bool,
) -> Option<pdf_interp::resources::TransparencyGroupParams> {
    if transparent {
        Some(pdf_interp::transparent_page_group(doc, page_id))
    } else {
        pdf_interp::page_group(doc, page_id)
    }
}

/// The error of a page bitmap the allocator refused.
pub(crate) const fn allocation_failed(e: raster::BitmapAllocError) -> RasterError {
    RasterError::PageAllocationFailed {
        width: e.width,
        height: e.height,
    }
}

/// Inner implementation shared by every page render entry point.  The
/// page's canvas is built in `buffers`
/// ([`PageRenderer::try_with_page_group_in`]), and `buffers` is left
/// holding the storage the returned bitmap does not.
///
/// [`PageRenderer::try_with_page_group_in`]: pdf_interp::renderer::PageRenderer::try_with_page_group_in
///
/// `params.policy` overrides `session.policy` for GPU decoder selection only.
/// All other session state (CUDA context, Vulkan backend, VA-API queue) is unaffected.
pub(crate) fn render_page_canvas_with_geom(
    session: &RasterSession,
    page_num: u32,
    page_id: pdf::ObjectId,
    geom: pdf_interp::PageGeometry,
    params: &RenderParams,
    buffers: &mut CanvasBuffers,
) -> Result<(FinishedCanvas, pdf_interp::renderer::PageDiagnostics), RasterError> {
    let effective_policy = begin(session, params)?;
    let doc = &session.doc;
    let view = page_view(
        &geom,
        (params.scale_x, params.scale_y),
        params.region,
        page_layout(session, page_id, params)?.bytes_per_px(),
        session.max_page_bytes,
    )?;
    let unsupported_features = pdf_interp::page_unsupported_features(doc, page_id);
    warn_unsupported(session, page_num, &unsupported_features);
    let content = page_content(session, page_id)?;
    let ops = pdf_interp::parse_page_content(doc, page_id, &content.bytes);
    let content_failures = content.failures;

    // The view carries the full-precision scales that also sized the page,
    // so the CTM's translation terms are consistent with the bitmap extent.
    let job = PageJob::new(
        session,
        page_num,
        page_id,
        &view,
        &ops,
        params,
        effective_policy,
    )?;
    #[cfg(feature = "vulkan")]
    let (mut renderer, device_page) = crate::device_pages::render(&job, std::mem::take(buffers))?;
    #[cfg(not(feature = "vulkan"))]
    let mut renderer = job.run(std::mem::take(buffers))?;

    // A budget breach takes precedence over image failures: the page may not
    // have finished rendering at all.
    check_aborted(&renderer)?;
    log_failures(page_num, &content_failures);
    renderer.record_content_failures(content_failures);
    check_strict(session, &renderer)?;
    log_failures(page_num, renderer.image_failures());

    let (mut bitmap, mut diag, spare) = renderer
        .try_finish_keeping(params.transparent_background)
        .map_err(allocation_failed)?;
    *buffers = spare;
    crate::output::warn_group_spaces(session, page_num, diag.approximated_group_spaces);
    if job.gray {
        crate::gray::reduce_to_gray(&mut bitmap, job.space());
    }
    let (bitmap, output_intent) = crate::output::convert(
        session,
        params,
        (page_num, page_id),
        (bitmap, diag.content_intents, job.space()),
        buffers,
    )?;
    diag.output_intent = output_intent;
    diag.form_type = session.form_type;
    diag.unsupported_features = unsupported_features;
    #[cfg(feature = "vulkan")]
    {
        diag.device_page = device_page;
    }
    Ok((bitmap, diag))
}

/// The error of a render that `renderer` stopped early: cancelled, out of
/// memory, or over its operator budget or deadline.
pub(crate) fn check_aborted(
    renderer: &pdf_interp::renderer::PageRenderer<'_>,
) -> Result<(), RasterError> {
    match renderer.budget_status() {
        None => Ok(()),
        Some(pdf_interp::renderer::RenderAbort::Cancelled { .. }) => Err(RasterError::Cancelled),
        Some(&pdf_interp::renderer::RenderAbort::AllocationFailed { width, height }) => {
            Err(RasterError::PageAllocationFailed { width, height })
        }
        Some(reason) => Err(RasterError::PageBudgetExceeded(reason.to_string())),
    }
}

/// The error of a strict session for what `renderer` skipped:
/// [`RasterError::ImageDecodeFailed`] when an image of the page failed,
/// [`RasterError::PageBudgetExceeded`] when a form lay too deep to draw.
pub(crate) fn check_strict(
    session: &RasterSession,
    renderer: &pdf_interp::renderer::PageRenderer<'_>,
) -> Result<(), RasterError> {
    if !session.strict {
        return Ok(());
    }
    let failures = renderer.image_failures();
    if !failures.is_empty() {
        return Err(RasterError::ImageDecodeFailed(
            failures.iter().map(ToString::to_string).collect(),
        ));
    }
    if let Some(&(num, generation)) = renderer.forms_too_deep().first() {
        return Err(RasterError::PageBudgetExceeded(format!(
            "Form XObject ({num}, {generation}) lies more than {} forms deep; not drawn",
            crate::MAX_FORM_DEPTH
        )));
    }
    Ok(())
}

/// Log each image or content stream of page `page_num` that was skipped or
/// drawn in part.
pub(crate) fn log_failures<F: PartialFailure>(page_num: u32, failures: &[F]) {
    for failure in failures {
        if failure.partial() {
            log::warn!("page {page_num}: {failure}");
        } else {
            log::warn!("page {page_num}: skipped {failure}");
        }
    }
}

/// A failure that may have left part of its object drawn.
pub(crate) trait PartialFailure: std::fmt::Display {
    /// Whether the object was drawn from what decoded.
    fn partial(&self) -> bool;
}

impl PartialFailure for pdf_interp::renderer::ImageFailure {
    fn partial(&self) -> bool {
        self.partial
    }
}

impl PartialFailure for pdf_interp::renderer::ContentFailure {
    fn partial(&self) -> bool {
        self.partial
    }
}

/// The checks every render makes before it parses the page: the scales,
/// the depth and the cancellation token; returns the backend policy of
/// its image decoders.
pub(crate) fn begin(
    session: &RasterSession,
    params: &RenderParams,
) -> Result<crate::BackendPolicy, RasterError> {
    check_scale(params.scale_x)?;
    check_scale(params.scale_y)?;
    crate::output::check_params(session, params)?;
    if params
        .cancel
        .as_ref()
        .is_some_and(crate::CancelToken::is_cancelled)
    {
        return Err(RasterError::Cancelled);
    }
    Ok(params.policy.unwrap_or(session.policy))
}

/// The decoded content streams of page `page_id`, refused under a strict
/// session when any failed to decode.
pub(crate) fn page_content(
    session: &RasterSession,
    page_id: pdf::ObjectId,
) -> Result<pdf::PageContent, RasterError> {
    let content = session.doc.page_content(page_id)?;
    if session.strict && !content.failures.is_empty() {
        return Err(RasterError::ContentDecodeFailed(
            content.failures.iter().map(ToString::to_string).collect(),
        ));
    }
    Ok(content)
}

/// A page to render: its content, the view it is drawn in and how.
pub(crate) struct PageJob<'a> {
    pub(crate) session: &'a RasterSession,
    #[cfg_attr(
        not(feature = "vulkan"),
        expect(dead_code, reason = "read by the GPU page queue")
    )]
    pub(crate) page_num: u32,
    page_id: pdf::ObjectId,
    view: &'a pdf_interp::renderer::PageView,
    pub(crate) ops: &'a [pdf_interp::content::Operator<'a>],
    /// The page group the page is painted as.
    pub(crate) group: Option<pdf_interp::resources::TransparencyGroupParams>,
    /// The canvas the page is painted on.
    pub(crate) canvas: pdf_interp::CanvasRequest,
    /// The depth of the canvas's samples (see
    /// [`canvas_depth`](crate::output::canvas_depth)).
    pub(crate) depth: SampleDepth,
    /// Whether the page's transparency group composites in gray, on an
    /// RGB canvas, its finished colour reduced to gray.
    pub(crate) gray: bool,
    pub(crate) params: &'a RenderParams,
    /// The backend policy for GPU decoder selection.
    policy: crate::BackendPolicy,
}

impl<'a> PageJob<'a> {
    /// The job of drawing `ops`, the content of page `page_id`, in `view`
    /// as `params` asks.  The page is painted on an ink canvas, which
    /// simulates overprint, when the session simulates it and some
    /// graphics state the page draws with sets it (see [`canvas_request`]).
    ///
    /// # Errors
    ///
    /// As [`canvas_request`].
    pub(crate) fn new(
        session: &'a RasterSession,
        page_num: u32,
        page_id: pdf::ObjectId,
        view: &'a pdf_interp::renderer::PageView,
        ops: &'a [pdf_interp::content::Operator<'a>],
        params: &'a RenderParams,
        policy: crate::BackendPolicy,
    ) -> Result<Self, RasterError> {
        let doc = &session.doc;
        let canvas = canvas_request(session, page_id, params)?;
        crate::output::warn_working_space(session, (page_num, page_id), &canvas);
        let depth = crate::output::canvas_depth(&canvas, params);
        let gray = !canvas.is_ink()
            && pdf_interp::page_group_space(doc, page_id)
                == Some(pdf_interp::resources::GroupSpace::Gray);
        Ok(Self {
            session,
            page_num,
            page_id,
            view,
            ops,
            group: page_group(doc, page_id, params.transparent_background),
            canvas,
            depth,
            gray,
            params,
            policy,
        })
    }

    /// The working space the page is composited in.
    pub(crate) fn space(&self) -> pdf_interp::WorkingSpace {
        self.canvas.space()
    }

    /// A renderer of the page, configured from the session, its bitmap
    /// built in `buffers`.
    pub(crate) fn renderer(
        &self,
        buffers: CanvasBuffers,
    ) -> Result<pdf_interp::renderer::PageRenderer<'a>, RasterError> {
        let session = self.session;
        let mut renderer = pdf_interp::renderer::PageRenderer::try_with_page_group_in(
            self.view,
            &session.doc,
            self.page_id,
            self.group,
            &self.canvas,
            self.depth,
            buffers,
        )
        .map_err(allocation_failed)?;
        if let Some(cancel) = &self.params.cancel {
            renderer.set_cancel(cancel.flag());
        }
        renderer.set_antialias(session.text_antialias, session.vector_antialias);
        renderer.set_document_fonts(Arc::clone(&session.fonts));
        renderer.set_document_images(Arc::clone(&session.images));
        renderer.set_optional_content(Arc::clone(&session.optional_content));

        #[cfg(any(feature = "gpu-icc", feature = "cache"))]
        renderer.set_gpu_ctx(session.gpu_ctx.as_ref().map(Arc::clone));

        #[cfg(feature = "vulkan")]
        renderer.set_vk_backend(session.vk_backend.as_ref().map(Arc::clone));

        #[cfg(feature = "cache")]
        renderer.set_image_cache(session.image_cache.clone(), session.doc_id);
        Ok(renderer)
    }

    /// Execute the page's content and annotations on `renderer`.
    pub(crate) fn execute(
        &self,
        renderer: &mut pdf_interp::renderer::PageRenderer<'_>,
    ) -> Result<(), RasterError> {
        // RAII guard armed BEFORE lend_decoders, not after.  lend_decoders moves
        // handles out of TLS into the renderer one decoder at a time and is
        // fallible per step: a `?` early-return from a later `ensure_*` would
        // otherwise drop the renderer (and any handle already moved in) without
        // reclaim — a partial-move leak.  Arming the guard first makes lend
        // transactional: every exit from lend OR the render body — `?`-early-return,
        // normal return, or panic-unwind — drops the guard and runs reclaim exactly
        // once.  reclaim is safe after a partial lend: each decoder's
        // ensure+take+set runs atomically per feature, so earlier features are
        // fully lent (TLS cell `Ready(None)`, handle in renderer) and reclaim
        // restores them, while a feature whose `ensure_*` failed left its TLS cell
        // non-`Ready` and reclaim's `if let Ready` guard skips it (no clobber).
        let guard = DecoderReclaim(renderer);
        lend_decoders(self.session, guard.0, self.policy)?;
        guard.0.execute(self.ops);
        if self.session.annotations {
            guard.0.render_annotations(self.page_id);
        }
        Ok(())
    }

    /// A renderer that has executed the page.
    pub(crate) fn run(
        &self,
        buffers: CanvasBuffers,
    ) -> Result<pdf_interp::renderer::PageRenderer<'a>, RasterError> {
        let mut renderer = self.renderer(buffers)?;
        self.execute(&mut renderer)?;
        Ok(renderer)
    }
}

/// The canvas page `page_id` of `session` is painted on under `params`:
/// for a CMYK page an ink canvas finished as CMYK, simulating overprint
/// when the session does, whatever space the page's transparency group
/// declares; otherwise, when the session simulates overprint and some
/// graphics state the page draws with sets it, an ink canvas of the
/// process-ink model finished as sRGB, simulating overprint; when the
/// page's group composites in CMYK (ISO 32000-2 §11.4.7), an ink canvas of
/// its ICC profile finished as RGB of the working space, or of the
/// process-ink model finished as sRGB for `DeviceCMYK`; an RGB canvas of
/// the working space otherwise.
///
/// A CMYK page separates the colours that are not CMYK through the
/// session's CMYK output profile, under `params.intent`, else under the
/// intent each object is painted with, and its black-point compensation;
/// without a profile, by the device conversion.  Under a soft proof every page is a CMYK page
/// whose profile is the page's press.
///
/// # Errors
///
/// [`RasterError::OutputProfile`] when the profile's separation cannot be
/// built; [`RasterError::OutputIntent`] when the page has no press.
fn canvas_request(
    session: &RasterSession,
    page_id: pdf::ObjectId,
    params: &RenderParams,
) -> Result<pdf_interp::CanvasRequest, RasterError> {
    let press = match &session.proof {
        Some(proof) => Some(Arc::clone(proof.for_page(&session.doc, page_id)?.press())),
        None if params.color == crate::ColorModel::Cmyk => session.output_profile.clone(),
        None => None,
    };
    if session.proof.is_some() || params.color == crate::ColorModel::Cmyk {
        let space = crate::output::working_space(session, None);
        let separation = match press {
            Some(profile) => pdf_interp::OutputProfile::separation(
                &profile,
                space,
                (params.intent, params.black_point_compensation),
                params.preserve_black,
            )
            .map_err(RasterError::OutputProfile)?,
            None => pdf_interp::Separation::Device,
        };
        return Ok(pdf_interp::CanvasRequest::Cmyk {
            overprint: session.overprint,
            separation,
        });
    }
    let preview = session.overprint
        && pdf_interp::page_uses_overprint(&session.doc, page_id, session.annotations);
    let declared = pdf_interp::page_group_space(&session.doc, page_id);
    let cmyk_group = match declared {
        Some(pdf_interp::resources::GroupSpace::Cmyk { profile }) => Some(profile),
        _ => None,
    };
    Ok(match cmyk_group {
        Some(Some(profile)) if !preview => pdf_interp::CanvasRequest::ProfileRgb {
            profile,
            space: crate::output::working_space(session, declared),
        },
        _ if preview || cmyk_group.is_some() => {
            pdf_interp::CanvasRequest::ProcessRgb { overprint: preview }
        }
        _ => match crate::output::working_space(session, declared) {
            pdf_interp::WorkingSpace::Wide => pdf_interp::CanvasRequest::WideRgb,
            pdf_interp::WorkingSpace::Srgb => pdf_interp::CanvasRequest::Rgb,
        },
    })
}

/// The layout of the canvas page `page_id` of `session` is painted on
/// under `params`, which the byte budget counts: of ink for a page
/// painted on an ink canvas (see [`canvas_request`]), of 16-bit samples
/// for a canvas of the wide working space at either output depth (see
/// [`canvas_depth`](crate::output::canvas_depth)).
///
/// # Errors
///
/// As [`canvas_request`].
pub(crate) fn page_layout(
    session: &RasterSession,
    page_id: pdf::ObjectId,
    params: &RenderParams,
) -> Result<crate::CanvasLayout, RasterError> {
    let canvas = canvas_request(session, page_id, params)?;
    let depth = crate::output::canvas_depth(&canvas, params);
    Ok(crate::CanvasLayout::new(depth, params.transparent_background).with_ink(canvas.is_ink()))
}

/// Warn about the first page found to carry JavaScript and the first
/// carrying a 3D or multimedia annotation, each once per session.
pub(crate) fn warn_unsupported(
    session: &RasterSession,
    page_num: u32,
    features: &[pdf_interp::UnsupportedFeature],
) {
    if features
        .iter()
        .any(pdf_interp::UnsupportedFeature::is_javascript)
        && !session.javascript_warned.swap(true, Ordering::Relaxed)
    {
        log::warn!(
            "page {page_num}: document contains JavaScript (page or annotation actions); \
             JavaScript is ignored — rasterrocket does not execute scripts, \
             rendering static content only"
        );
    }
    if let Some(media) = features
        .iter()
        .find(|f| matches!(f, pdf_interp::UnsupportedFeature::MediaAnnotation { .. }))
        && !session.media_warned.swap(true, Ordering::Relaxed)
    {
        log::warn!(
            "page {page_num}: document contains 3D or multimedia annotations ({media}); \
             their interactive content is not rendered"
        );
    }
}
