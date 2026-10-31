//! Rendering one page of a [`RasterSession`] to an RGB bitmap.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use color::Rgb8;
use raster::Bitmap;

use crate::decoders::{DecoderReclaim, lend_decoders};
use crate::gray::rgb_to_gray;
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
///   [`RasterError::PageOutOfRange`] / [`RasterError::Pdf`] as documented on
///   the error variants; the size limits apply to the drawn region.
pub fn render_page_rgb_with(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
) -> Result<Bitmap<Rgb8>, RasterError> {
    let page_id = session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&session.doc, page_id)?;
    render_page_rgb_with_geom(session, page_num, page_id, geom, params).map(|(bmp, _diag)| bmp)
}

/// Render page `page_num` (1-based) of `session` to the grayscale
/// [`RenderedPage`] that [`raster_pdf`](crate::raster_pdf) yields for it.
///
/// Of `opts`, only the per-page fields apply: [`dpi`](RasterOptions::dpi),
/// [`deskew`](RasterOptions::deskew) and [`cancel`](RasterOptions::cancel).  The page selection fields are
/// the caller's, and the fields mirrored on [`SessionConfig`](crate::SessionConfig)
/// were fixed when `session` was opened.
///
/// # Errors
///
/// As [`render_page_rgb_with`], plus [`RasterError::Deskew`] when deskew
/// rotation fails.
pub fn render_page_gray(
    session: &RasterSession,
    page_num: u32,
    opts: &RasterOptions,
) -> Result<RenderedPage, RasterError> {
    let dpi = opts.dpi;
    let page_id = session.resolve_page(page_num)?;
    let geom = pdf_interp::page_size_pts_by_id(&session.doc, page_id)?;
    let user_unit = geom.user_unit;
    let mut params = RenderParams::new(f64::from(dpi) / 72.0);
    params.cancel.clone_from(&opts.cancel);
    let (rgb, diagnostics) = render_page_rgb_with_geom(session, page_num, page_id, geom, &params)?;
    let mut gray = rgb_to_gray(&rgb);
    drop(rgb);

    if opts.deskew {
        crate::deskew::apply(&mut gray).map_err(|e| RasterError::Deskew(e.to_string()))?;
    }

    // `dpi` has passed the positive-finite scale check above; a product
    // beyond `f32::MAX` saturates to infinity rather than wrapping.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64 → f32 of a positive finite value saturates; UserUnit is within [0.1, 10]"
    )]
    let effective_dpi = (f64::from(dpi) * user_unit) as f32;

    Ok(crate::page::gray8_to_rendered_page(
        &gray,
        page_num,
        dpi,
        effective_dpi,
        diagnostics,
    ))
}

/// The view a render of `geom` at `scales` (horizontal, vertical) draws: the
/// whole page, or `region` of it.  Enforces the pixel-size limits on what is
/// drawn.
fn page_view(
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

/// Inner implementation shared by [`render_page_rgb_with`] and [`render_one`].
///
/// `params.policy` overrides `session.policy` for GPU decoder selection only.
/// All other session state (GPU context for AA/ICC, VA-API queue) is unaffected.
fn render_page_rgb_with_geom(
    session: &RasterSession,
    page_num: u32,
    page_id: pdf::ObjectId,
    geom: pdf_interp::PageGeometry,
    params: &RenderParams,
) -> Result<(Bitmap<Rgb8>, pdf_interp::renderer::PageDiagnostics), RasterError> {
    for scale in [params.scale_x, params.scale_y] {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(RasterError::InvalidOptions(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }
    }
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

    if !session.javascript_warned.load(Ordering::Relaxed)
        && pdf_interp::page_has_javascript(doc, page_id)
        && !session.javascript_warned.swap(true, Ordering::Relaxed)
    {
        log::warn!(
            "page {page_num}: document contains JavaScript (page or annotation actions); \
             JavaScript is ignored — rasterrocket does not execute scripts, \
             rendering static content only"
        );
    }

    let (ops, content_failures) = pdf_interp::parse_page_content_by_id(doc, page_id)?;
    if session.strict && !content_failures.is_empty() {
        return Err(RasterError::ContentDecodeFailed(
            content_failures.iter().map(ToString::to_string).collect(),
        ));
    }

    // The view carries the full-precision scales that also sized the page,
    // so the CTM's translation terms are consistent with the bitmap extent.
    let mut renderer = pdf_interp::renderer::PageRenderer::with_view(&view, doc, page_id);
    if let Some(cancel) = &params.cancel {
        renderer.set_cancel(cancel.flag());
    }
    renderer.set_antialias(session.text_antialias, session.vector_antialias);
    renderer.set_document_fonts(Arc::clone(&session.fonts));
    renderer.set_optional_content(Arc::clone(&session.optional_content));
    renderer.set_page_group(pdf_interp::page_group(doc, page_id));

    #[cfg(any(feature = "gpu-aa", feature = "gpu-icc", feature = "cache"))]
    renderer.set_gpu_ctx(session.gpu_ctx.as_ref().map(Arc::clone));

    #[cfg(feature = "vulkan")]
    renderer.set_vk_backend(session.vk_backend.as_ref().map(Arc::clone));

    #[cfg(feature = "cache")]
    renderer.set_image_cache(session.image_cache.as_ref().map(Arc::clone), session.doc_id);

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
    let guard = DecoderReclaim(&mut renderer);
    lend_decoders(session, guard.0, effective_policy)?;
    guard.0.execute(&ops);
    if session.annotations {
        guard.0.render_annotations(page_id);
    }
    // Drop reclaims exactly once and releases the &mut borrow, so `renderer`
    // is usable directly for the budget/image-failure/finish tail below.
    drop(guard);

    // A budget breach takes precedence over image failures: the page may not
    // have finished rendering at all.
    match renderer.budget_status() {
        None => {}
        Some(pdf_interp::renderer::RenderAbort::Cancelled { .. }) => {
            return Err(RasterError::Cancelled);
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

    let (bitmap, mut diag) = renderer.finish();
    diag.form_type = session.form_type;
    Ok((bitmap, diag))
}
