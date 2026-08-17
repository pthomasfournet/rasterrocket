//! GPU-accelerated fill helpers for [`PageRenderer`].
//!
//! All functions in this module are gated on `#[cfg(feature = "gpu-aa")]` and
//! take `renderer: &mut PageRenderer<'_>` as their first argument, delegating
//! from thin wrapper methods in `mod.rs`.

#[cfg(feature = "gpu-aa")]
use super::PageRenderer;

/// Clamped pixel-space bounding box for a GPU fill operation.
///
/// All fields are in device-pixel coordinates, clipped to the bitmap bounds.
/// `x` / `y` are the top-left corner; `w` / `h` are the extent.
#[cfg(feature = "gpu-aa")]
pub(super) struct GpuBbox {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) w: u32,
    pub(super) h: u32,
}

/// Shared preamble for GPU fill paths: flatten path, compute clamped bbox,
/// convert segments to packed f32.
///
/// Returns `None` if the path is empty, produces no segments, or the bbox
/// is degenerate/non-finite.  Otherwise returns `(segs_f32, bbox)`.
#[cfg(feature = "gpu-aa")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bbox is clamped to [0, bitmap.width/height] before cast; all values are \
              finite, non-negative, and ≤ u32::MAX"
)]
pub(super) fn gpu_fill_segs(
    renderer: &PageRenderer<'_>,
    path: &raster::path::Path,
) -> Option<(Vec<f32>, GpuBbox)> {
    use raster::xpath::XPath;

    use super::DEVICE_MATRIX;

    if path.pts.is_empty() {
        return None;
    }

    let xpath = XPath::new(path, &DEVICE_MATRIX, 0.1, true);
    if xpath.segs.is_empty() {
        return None;
    }

    // Pixel bbox from segment endpoints.
    let mut x_min_f = f64::INFINITY;
    let mut y_min_f = f64::INFINITY;
    let mut x_max_f = f64::NEG_INFINITY;
    let mut y_max_f = f64::NEG_INFINITY;
    for seg in &xpath.segs {
        x_min_f = x_min_f.min(seg.x0).min(seg.x1);
        y_min_f = y_min_f.min(seg.y0).min(seg.y1);
        x_max_f = x_max_f.max(seg.x0).max(seg.x1);
        y_max_f = y_max_f.max(seg.y0).max(seg.y1);
    }
    // Segment coords should always be finite; NaN/Inf would indicate a bug
    // in the path construction or CTM, so we log and bail rather than panic.
    if !x_min_f.is_finite() || !y_min_f.is_finite() || !x_max_f.is_finite() || !y_max_f.is_finite()
    {
        log::warn!(
            "gpu_fill_segs: non-finite segment bbox ({x_min_f}, {y_min_f}, {x_max_f}, {y_max_f}); skipping GPU path"
        );
        return None;
    }

    // Clamp to bitmap dimensions and quantise to integer pixels.
    let bmp_w = f64::from(renderer.bitmap.width);
    let bmp_h = f64::from(renderer.bitmap.height);
    x_min_f = x_min_f.max(0.0).floor();
    y_min_f = y_min_f.max(0.0).floor();
    x_max_f = x_max_f.min(bmp_w).ceil();
    y_max_f = y_max_f.min(bmp_h).ceil();
    if x_max_f <= x_min_f || y_max_f <= y_min_f {
        return None;
    }

    let bbox = GpuBbox {
        x: x_min_f as u32,
        y: y_min_f as u32,
        w: (x_max_f - x_min_f) as u32,
        h: (y_max_f - y_min_f) as u32,
    };

    // Pack segments as flat f32 [x0,y0,x1,y1].  The f64→f32 cast is
    // intentional: GPU kernels operate in f32, and precision loss is
    // sub-pixel at realistic PDF rasterisation DPIs.
    let segs_f32: Vec<f32> = xpath
        .segs
        .iter()
        .flat_map(|seg| [seg.x0 as f32, seg.y0 as f32, seg.x1 as f32, seg.y1 as f32])
        .collect();

    Some((segs_f32, bbox))
}

/// Paint a GPU-produced per-pixel coverage buffer into `renderer.bitmap`.
///
/// `coverage` must be exactly `bbox.w × bbox.h` bytes (one byte per pixel,
/// 0 = outside, 255 = inside), with `bbox` giving the top-left corner and
/// extent in device-pixel coordinates (already clamped to bitmap bounds).
///
/// Scans each row for contiguous non-zero spans, clips them to the active
/// clip region and bitmap bounds, then calls `pipe::render_span`.
#[cfg(feature = "gpu-aa")]
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "all coordinates bounded by bitmap dims which are derived from u32; \
              realistic pages are far below i32::MAX"
)]
pub(super) fn gpu_coverage_to_bitmap(
    renderer: &mut PageRenderer<'_>,
    coverage: &[u8],
    bbox: &GpuBbox,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) {
    let clip = renderer.gstate.current().clip.clone_shared();
    paint_coverage(&mut renderer.bitmap, &clip, coverage, bbox, pipe, src);
}

/// Render one clipped sub-span of a coverage row into `bitmap`.
#[cfg(feature = "gpu-aa")]
fn emit_span(
    bitmap: &mut raster::Bitmap<color::Rgb8>,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
    shape: &[u8],
    x0: i32,
    x1: i32,
    y: u32,
) {
    use raster::Pixel;
    use raster::pipe;

    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "x0/x1 are clamped to [0, bitmap.width) by the caller"
    )]
    {
        let (row_pixels, alpha_row) = bitmap.row_and_alpha_mut(y);
        let byte_off = x0 as usize * <color::Rgb8 as Pixel>::BYTES;
        let byte_end = (x1 as usize + 1) * <color::Rgb8 as Pixel>::BYTES;
        let alpha_range = x0 as usize..=x1 as usize;
        let dst_pixels = &mut row_pixels[byte_off..byte_end];
        let dst_alpha = alpha_row.map(|a| &mut a[alpha_range]);

        pipe::render_span::<color::Rgb8>(
            pipe,
            src,
            dst_pixels,
            dst_alpha,
            Some(shape),
            x0,
            x1,
            y as i32,
        );
    }
}

/// Clip-aware coverage painter backing [`gpu_coverage_to_bitmap`].
///
/// Split out from the `PageRenderer` wrapper so it can be exercised
/// directly against a hand-built `Bitmap` + `Clip` in unit tests.
#[cfg(feature = "gpu-aa")]
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "all coordinates bounded by bitmap dims which are derived from u32; \
              realistic pages are far below i32::MAX"
)]
fn paint_coverage(
    bitmap: &mut raster::Bitmap<color::Rgb8>,
    clip: &raster::clip::Clip,
    coverage: &[u8],
    bbox: &GpuBbox,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
) {
    use raster::clip::ClipResult;

    let GpuBbox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    } = *bbox;

    // The caller guarantees coverage.len() == bw * bh.  A mismatch would
    // mean a bug in the GPU dispatch layer — catch it in debug builds.
    debug_assert_eq!(
        coverage.len(),
        bw as usize * bh as usize,
        "coverage buffer length mismatch: expected {}×{}={}, got {}",
        bw,
        bh,
        bw as usize * bh as usize,
        coverage.len()
    );
    // bbox was clamped to bitmap dims in gpu_fill_segs; bx+bw ≤ bitmap.width,
    // by+bh ≤ bitmap.height.  Assert in debug builds.
    debug_assert!(
        bx.saturating_add(bw) <= bitmap.width && by.saturating_add(bh) <= bitmap.height,
        "GPU bbox ({bx},{by} {bw}×{bh}) extends outside bitmap ({}×{})",
        bitmap.width,
        bitmap.height
    );

    let bmp_w_i = bitmap.width as i32;

    for row in 0..bh {
        let y = by + row; // cannot exceed bitmap.height (clamped in gpu_fill_segs)
        let y_i = y as i32;

        let row_start = row as usize * bw as usize;
        let row_cov = &coverage[row_start..row_start + bw as usize];

        // Walk the row, collecting contiguous non-zero spans and emitting them.
        let mut span_start: Option<usize> = None;
        for col in 0..=bw as usize {
            let is_covered = col < bw as usize && row_cov[col] > 0;
            if is_covered {
                let _ = span_start.get_or_insert(col);
            } else if let Some(start) = span_start.take() {
                let x0 = bx as i32 + start as i32;
                let x1 = bx as i32 + col as i32 - 1;

                if clip.test_span(x0, x1, y_i) == ClipResult::AllOutside {
                    continue;
                }

                // Clamp to the bitmap and the clip rectangle, trimming the
                // coverage slice to match.
                let sx0 = x0.max(0).max(clip.x_min_i);
                let sx1 = x1.min(bmp_w_i - 1).min(clip.x_max_i);
                if sx0 > sx1 {
                    continue;
                }

                // trim_left = sx0 - x0 ≥ 0 and trim_right = x1 - sx1 ≥ 0 by the
                // clamps above; sx0 ≤ sx1 gives trim_left + trim_right =
                // (sx0-x0) + (x1-sx1) ≤ (x1-x0) = col-start-1, which is
                // < shape_slice.len() = col-start.  So the subtraction is safe.
                let shape_slice = &row_cov[start..col];
                let trim_left = (sx0 - x0) as usize;
                let trim_right = (x1 - sx1) as usize;
                let trimmed_shape = &shape_slice[trim_left..shape_slice.len() - trim_right];
                if trimmed_shape.is_empty() {
                    continue;
                }

                // Re-test after clamping: a span that merely straddled the
                // clip rectangle upgrades to AllInside once clamped, so only
                // path-clip (scanner) cases pay the per-pixel walk below.
                if clip.test_span(sx0, sx1, y_i) == ClipResult::AllInside {
                    emit_span(bitmap, pipe, src, trimmed_shape, sx0, sx1, y);
                    continue;
                }

                // Partial after clamping — a path-clip scanner is active.
                // Split the span into maximal per-pixel inside runs,
                // mirroring the CPU path's draw_span_clipped.  (Clip-path
                // edge fidelity is per-pixel here; the CPU AA path masks
                // per subsample.)
                let mut run_start: Option<i32> = None;
                for x in sx0..=sx1 + 1 {
                    let inside = x <= sx1 && clip.test(x, y_i);
                    if inside {
                        let _ = run_start.get_or_insert(x);
                    } else if let Some(rs) = run_start.take() {
                        let re = x - 1;
                        let off = (rs - sx0) as usize;
                        let run_shape = &trimmed_shape[off..off + (re - rs + 1) as usize];
                        emit_span(bitmap, pipe, src, run_shape, rs, re, y);
                    }
                }
            }
        }
    }
}

/// Attempt to rasterise `path` with the GPU 64-sample AA kernel.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if the area is below the dispatch threshold, the segment
/// list is empty, the bbox is non-finite, or the GPU call fails (warning
/// logged; CPU fill used as fallback).
#[cfg(feature = "gpu-aa")]
pub(super) fn try_gpu_aa_fill(
    renderer: &mut PageRenderer<'_>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
    ctx: &gpu::GpuCtx,
) -> bool {
    use gpu::GPU_AA_FILL_THRESHOLD;

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    if (bbox.w as usize * bbox.h as usize) < GPU_AA_FILL_THRESHOLD {
        return false;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let coverage = match ctx.aa_fill(
        &segs_f32,
        bbox.x as f32,
        bbox.y as f32,
        bbox.w,
        bbox.h,
        even_odd,
    ) {
        Ok(cov) => cov,
        Err(e) => {
            log::warn!("GPU AA fill failed, falling back to CPU: {e}");
            return false;
        }
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}

#[cfg(all(test, feature = "gpu-aa"))]
mod tests {
    use raster::clip::Clip;
    use raster::pipe::{PipeSrc, PipeState};
    use raster::types::BlendMode;
    use raster::xpath::XPath;

    use super::{GpuBbox, paint_coverage};

    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    fn opaque_pipe() -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: raster::state::TransferSet::identity_rgb(),
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        }
    }

    fn white_bitmap(w: u32, h: u32) -> raster::Bitmap<color::Rgb8> {
        let mut bitmap = raster::Bitmap::<color::Rgb8>::new(w, h, 1, false);
        bitmap.data_mut().fill(255);
        bitmap
    }

    fn pixel(bitmap: &raster::Bitmap<color::Rgb8>, x: usize, y: u32) -> [u8; 3] {
        let row = bitmap.row(y);
        [row[x].r, row[x].g, row[x].b]
    }

    /// A span straddling the clip rectangle must not paint outside it.
    #[test]
    fn paint_coverage_respects_clip_rectangle() {
        let mut bitmap = white_bitmap(64, 64);
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
        clip.clip_to_rect(10.0, 0.0, 20.0, 64.0);

        let bbox = GpuBbox {
            x: 0,
            y: 0,
            w: 32,
            h: 32,
        };
        let coverage = vec![255u8; 32 * 32];
        let pipe = opaque_pipe();
        let black = [0u8, 0, 0];
        let src = PipeSrc::Solid(&black);

        paint_coverage(&mut bitmap, &clip, &coverage, &bbox, &pipe, &src);

        for y in 0..32u32 {
            for x in 0..32usize {
                let px = pixel(&bitmap, x, y);
                if (10..=19).contains(&x) {
                    assert_eq!(px, [0, 0, 0], "({x}, {y}) inside clip must be painted");
                } else {
                    assert_eq!(
                        px,
                        [255, 255, 255],
                        "({x}, {y}) outside clip must stay white"
                    );
                }
            }
        }
    }

    /// A fill under a non-rectangular (scanner) clip must not paint outside
    /// the clip path, only inside it.
    #[test]
    fn paint_coverage_respects_clip_path() {
        let mut bitmap = white_bitmap(64, 64);
        let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);

        // Diamond centred on (16, 16): not axis-aligned, so clip_to_path
        // keeps it as a scanner instead of reducing to a rectangle.
        let mut builder = raster::path::PathBuilder::new();
        builder.move_to(16.0, 2.0).expect("move_to");
        builder.line_to(30.0, 16.0).expect("line_to");
        builder.line_to(16.0, 30.0).expect("line_to");
        builder.line_to(2.0, 16.0).expect("line_to");
        builder.close(false).expect("close");
        let diamond = builder.build();
        let xpath = XPath::new(&diamond, &IDENTITY, 0.1, true);
        clip.clip_to_path(&xpath, false);

        let bbox = GpuBbox {
            x: 0,
            y: 0,
            w: 32,
            h: 32,
        };
        let coverage = vec![255u8; 32 * 32];
        let pipe = opaque_pipe();
        let black = [0u8, 0, 0];
        let src = PipeSrc::Solid(&black);

        paint_coverage(&mut bitmap, &clip, &coverage, &bbox, &pipe, &src);

        // Sample points comfortably inside and outside the diamond so AA
        // edge coverage cannot blur the verdict.
        for (x, y) in [(16usize, 16u32), (16, 8), (16, 24), (9, 16), (23, 16)] {
            assert_eq!(
                pixel(&bitmap, x, y),
                [0, 0, 0],
                "({x}, {y}) inside the clip path must be painted"
            );
        }
        for (x, y) in [(3usize, 3u32), (28, 3), (3, 28), (28, 28), (0, 16), (31, 0)] {
            assert_eq!(
                pixel(&bitmap, x, y),
                [255, 255, 255],
                "({x}, {y}) outside the clip path must stay white"
            );
        }
    }
}

/// Attempt to rasterise `path` with the GPU tile-parallel analytical fill kernel.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if the area is below the dispatch threshold, the segment
/// list is empty, the bbox is non-finite, or the GPU call fails (warning
/// logged; caller falls through to AA or CPU fill).
#[cfg(feature = "gpu-aa")]
pub(super) fn try_gpu_tile_fill(
    renderer: &mut PageRenderer<'_>,
    path: &raster::path::Path,
    even_odd: bool,
    pipe: &raster::pipe::PipeState<'_>,
    src: &raster::pipe::PipeSrc<'_>,
    ctx: &gpu::GpuCtx,
) -> bool {
    use gpu::{GPU_TILE_FILL_THRESHOLD, build_tile_records};

    let Some((segs_f32, bbox)) = gpu_fill_segs(renderer, path) else {
        return false;
    };
    if (bbox.w as usize * bbox.h as usize) < GPU_TILE_FILL_THRESHOLD {
        return false;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "bbox.x/y are u32 bitmap coords; f32 precision is sub-pixel at realistic DPIs"
    )]
    let (records, tile_starts, tile_counts, grid_w) =
        build_tile_records(&segs_f32, bbox.x as f32, bbox.y as f32, bbox.w, bbox.h);

    let coverage = match ctx.tile_fill(
        &records,
        &tile_starts,
        &tile_counts,
        grid_w,
        bbox.w,
        bbox.h,
        even_odd,
    ) {
        Ok(cov) => cov,
        Err(e) => {
            log::warn!("GPU tile fill failed, falling back to AA fill: {e}");
            return false;
        }
    };

    gpu_coverage_to_bitmap(renderer, &coverage, &bbox, pipe, src);
    true
}
