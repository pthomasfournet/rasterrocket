//! Rendering one page of a [`RasterSession`] to an RGB bitmap.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use color::Rgb8;
use pdf_interp::renderer::CanvasBuffers;
use raster::Bitmap;

use crate::decoders::{DecoderReclaim, lend_decoders};
use crate::gray::rgb_into_gray;
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
///   [`RasterError::PageTooLarge`] / [`RasterError::PageAreaTooLarge`] /
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
/// As [`render_page_rgb_with`].
pub fn render_page_rgb_with_diagnostics(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<(Bitmap<Rgb8>, crate::PageDiagnostics), RasterError> {
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    render_page_rgb_with_geom(
        session,
        page_num,
        page_id,
        geom,
        params,
        &mut CanvasBuffers::default(),
    )
}

/// Render page `page_num` (1-based) of `session` to the grayscale
/// [`RenderedPage`] that [`raster_pdf`](crate::raster_pdf) yields for it.
///
/// Of `opts`, only the per-page fields apply: [`dpi`](RasterOptions::dpi),
/// [`deskew`](RasterOptions::deskew) and [`cancel`](RasterOptions::cancel);
/// [`transparent_background`](RasterOptions::transparent_background) is
/// [`render_page_rgba`]'s.
/// The page selection fields are the caller's, and the fields mirrored on
/// [`SessionConfig`](crate::SessionConfig) were fixed when `session` was
/// opened.
///
/// # Errors
///
/// As [`render_page_rgb_with`], plus [`RasterError::Deskew`] when deskew
/// rotation fails, which this release never reports (see
/// [`deskew::apply`](crate::deskew::apply)).
pub fn render_page_gray(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError> {
    let dpi = opts.dpi;
    let page_id = session.resolve_page(page_num)?;
    let geom = session.geometry_by_id(page_id)?;
    let mut params = RenderParams::new(f64::from(dpi) / 72.0);
    params.cancel.clone_from(&opts.cancel);
    let (rgb, diagnostics) = render_page_rgb_with_geom(
        session,
        page_num,
        page_id,
        geom,
        &params,
        &mut CanvasBuffers::default(),
    )?;
    let mut gray = rgb_into_gray(rgb);

    if opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    Ok(crate::page::gray8_into_rendered_page(
        gray,
        page_num,
        dpi,
        diagnostics,
    ))
}

/// Render page `page_num` (1-based) of `session` over a transparent
/// backdrop to an RGBA [`RenderedPage`].
///
/// The page is the one [`raster_pdf`](crate::raster_pdf) yields when
/// [`RasterOptions::transparent_background`] is set:
/// [`PixelFormat::Rgba8`](crate::PixelFormat::Rgba8), alpha not
/// premultiplied.
///
/// Of `opts`, only [`dpi`](RasterOptions::dpi) and
/// [`cancel`](RasterOptions::cancel) apply.
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
    let mut params = RenderParams::new(f64::from(dpi) / 72.0).with_transparent_background();
    params.cancel.clone_from(&opts.cancel);
    let (rgb, diagnostics) = render_page_rgb_with_geom(
        session,
        page_num,
        page_id,
        geom,
        &params,
        &mut CanvasBuffers::default(),
    )?;
    Ok(crate::page::rgb8_to_rgba_rendered_page(
        &rgb,
        page_num,
        dpi,
        diagnostics,
    ))
}

/// The view a render of `geom` at `scales` (horizontal, vertical) draws: the
/// whole page, or `region` of it.  Enforces the pixel-size limits on what is
/// drawn.
pub(crate) fn page_view(
    geom: &pdf_interp::PageGeometry,
    scales: (f64, f64),
    region: Option<RenderRegion>,
) -> Result<pdf_interp::renderer::PageView, RasterError> {
    let whole = pdf_interp::renderer::PageView::whole(geom, scales);
    let (w_px, h_px) = (whole.page_width, whole.page_height);
    if w_px == 0 || h_px == 0 {
        return Err(RasterError::PageDegenerate {
            width: w_px,
            height: h_px,
        });
    }
    let Some(r) = region else {
        // Reject oversized pages before the bitmap is allocated, never after,
        // via the shared guard (per-side limit, then total-area limit in u64).
        crate::page::validate_dimensions(w_px, h_px)?;
        return Ok(whole);
    };
    // A saturated side is not the page's true extent, so no window of it
    // can be placed.
    if w_px == u32::MAX || h_px == u32::MAX {
        return Err(RasterError::PageTooLarge {
            width: w_px,
            height: h_px,
        });
    }
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
    crate::page::validate_dimensions(r.width, r.height)?;
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
const fn allocation_failed(e: raster::BitmapAllocError) -> RasterError {
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
/// All other session state (GPU context for AA/ICC, VA-API queue) is unaffected.
pub(crate) fn render_page_rgb_with_geom(
    session: &RasterSession,
    page_num: u32,
    page_id: pdf::ObjectId,
    geom: pdf_interp::PageGeometry,
    params: &RenderParams,
    buffers: &mut CanvasBuffers,
) -> Result<(Bitmap<Rgb8>, pdf_interp::renderer::PageDiagnostics), RasterError> {
    check_scale(params.scale_x)?;
    check_scale(params.scale_y)?;
    let effective_policy = params.policy.unwrap_or(session.policy);
    if params
        .cancel
        .as_ref()
        .is_some_and(crate::CancelToken::is_cancelled)
    {
        return Err(RasterError::Cancelled);
    }

    let doc = &session.doc;

    let view = page_view(&geom, (params.scale_x, params.scale_y), params.region)?;

    let unsupported_features = pdf_interp::page_unsupported_features(doc, page_id);
    warn_unsupported(session, page_num, &unsupported_features);

    let content = doc.page_content(page_id)?;
    let ops = pdf_interp::parse_page_content(doc, page_id, &content.bytes);
    let content_failures = content.failures;
    if session.strict && !content_failures.is_empty() {
        return Err(RasterError::ContentDecodeFailed(
            content_failures.iter().map(ToString::to_string).collect(),
        ));
    }

    // The view carries the full-precision scales that also sized the page,
    // so the CTM's translation terms are consistent with the bitmap extent.
    // Overprint is simulated on an ink canvas, chosen for a page on which
    // some graphics state that is drawn sets it.
    let ink =
        session.overprint && pdf_interp::page_uses_overprint(doc, page_id, session.annotations);
    let job = PageJob {
        session,
        page_num,
        page_id,
        view: &view,
        ops: &ops,
        group: page_group(doc, page_id, params.transparent_background),
        ink,
        params,
        policy: effective_policy,
    };
    #[cfg(feature = "vulkan")]
    let (mut renderer, device_page) = crate::device_pages::render(&job, std::mem::take(buffers))?;
    #[cfg(not(feature = "vulkan"))]
    let mut renderer = job.run(std::mem::take(buffers))?;

    // A budget breach takes precedence over image failures: the page may not
    // have finished rendering at all.
    match renderer.budget_status() {
        None => {}
        Some(pdf_interp::renderer::RenderAbort::Cancelled { .. }) => {
            return Err(RasterError::Cancelled);
        }
        Some(&pdf_interp::renderer::RenderAbort::AllocationFailed { width, height }) => {
            return Err(RasterError::PageAllocationFailed { width, height });
        }
        Some(reason) => return Err(RasterError::PageBudgetExceeded(reason.to_string())),
    }

    for failure in &content_failures {
        if failure.partial {
            log::warn!("page {page_num}: {failure}");
        } else {
            log::warn!("page {page_num}: skipped {failure}");
        }
    }
    renderer.record_content_failures(content_failures);

    let failures = renderer.image_failures();
    if session.strict && !failures.is_empty() {
        return Err(RasterError::ImageDecodeFailed(
            failures.iter().map(ToString::to_string).collect(),
        ));
    }
    for failure in failures {
        if failure.partial {
            log::warn!("page {page_num}: {failure}");
        } else {
            log::warn!("page {page_num}: skipped {failure}");
        }
    }

    let (bitmap, mut diag, spare) = renderer
        .try_finish_keeping(params.transparent_background)
        .map_err(allocation_failed)?;
    *buffers = spare;
    diag.form_type = session.form_type;
    diag.unsupported_features = unsupported_features;
    #[cfg(feature = "vulkan")]
    {
        diag.device_page = device_page;
    }
    Ok((bitmap, diag))
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
    /// Whether the page is painted on an ink canvas.
    pub(crate) ink: bool,
    params: &'a RenderParams,
    /// The backend policy for GPU decoder selection.
    policy: crate::BackendPolicy,
}

impl<'a> PageJob<'a> {
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
            self.ink,
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

        #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
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

/// Warn about the first page found to carry JavaScript and the first
/// carrying a 3D or multimedia annotation, each once per session.
fn warn_unsupported(
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
