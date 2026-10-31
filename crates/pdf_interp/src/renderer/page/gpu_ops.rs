//! GPU fill helpers for [`PageRenderer`].
//!
//! The device rasterises nothing of its own.  It evaluates the scanline
//! intersection table the CPU fill builds (`raster::fill::aa_scanner`) with
//! the CPU's span walk, returning per-pixel sub-sample counts, and those
//! counts are composited through the CPU's own `raster::fill::draw_aa_counts`.
//! Every stage that decides a pixel's value is therefore the CPU's code or a
//! byte-exact twin of it.
//!
//! All functions in this module are gated on `#[cfg(feature = "gpu-aa")]`.

#[cfg(feature = "gpu-aa")]
use gpu::{AaFillRect, AaScanTable, ScanIntersect};
#[cfg(feature = "gpu-aa")]
use raster::clip::Clip;
#[cfg(feature = "gpu-aa")]
use raster::pipe::{PipeSrc, PipeState};
#[cfg(feature = "gpu-aa")]
use raster::types::AA_SIZE;

#[cfg(feature = "gpu-aa")]
use super::{DEVICE_MATRIX, PageRenderer};

/// A fill prepared for device evaluation: the CPU scanner's intersection
/// table plus the pixel rectangle it is evaluated over.
#[cfg(feature = "gpu-aa")]
pub(super) struct GpuAaFill {
    /// Intersections grouped by AA row, sorted by `x0` within a row.
    pub(super) intersects: Vec<ScanIntersect>,
    /// Row offsets into `intersects`; `n_rows + 1` entries.
    pub(super) row_starts: Vec<u32>,
    /// AA row index of table row 0.
    pub(super) table_y0: i32,
    /// Pixel rectangle the kernel evaluates; within the bitmap.
    pub(super) rect: AaFillRect,
    /// Bitmap width × `AA_SIZE`.
    pub(super) aa_width: u32,
}

#[cfg(feature = "gpu-aa")]
impl GpuAaFill {
    pub(super) fn table(&self) -> AaScanTable<'_> {
        AaScanTable {
            intersects: &self.intersects,
            row_starts: &self.row_starts,
            y_min: self.table_y0,
        }
    }
}

/// Effective GPU AA-fill dispatch threshold: [`gpu::GPU_AA_FILL_THRESHOLD`]
/// unless `PDF_RASTER_AA_FILL_THRESHOLD` holds a decimal pixel area (`0` =
/// every eligible fill on the device).  Read once, on first use.
#[cfg(feature = "gpu-aa")]
pub(super) fn aa_fill_threshold() -> usize {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("PDF_RASTER_AA_FILL_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(gpu::GPU_AA_FILL_THRESHOLD)
    })
}

/// Build the device input for filling `path`, or `None` when the fill must
/// stay on the CPU.
///
/// `None` is returned when:
/// - the clip carries a path clip (the CPU masks those per sub-sample);
/// - the path is empty, scan-converts to nothing, or lies outside the bitmap;
/// - the bounding-box area is below `threshold`;
/// - some AA row's winding does not return to zero, in which case the
///   scanner's x bounds no longer bound the painted range.
///
/// The bounding box is the scanner's AA bounds divided by `AA_SIZE`, clamped
/// to the bitmap.  It is not clamped to the clip rectangle: the CPU applies
/// that clamp per row after finding the painted range, and
/// [`paint_coverage`] does the same.
#[cfg(feature = "gpu-aa")]
pub(super) fn prepare_aa_fill(
    clip: &Clip,
    bitmap_width: u32,
    bitmap_height: u32,
    path: &raster::path::Path,
    flatness: f64,
    even_odd: bool,
    threshold: usize,
) -> Option<GpuAaFill> {
    if !clip.is_rect() {
        return None;
    }
    let scanner = raster::fill::aa_scanner(path, &DEVICE_MATRIX, flatness, clip, even_odd)?;

    let width_i = i32::try_from(bitmap_width).ok()?;
    let height_i = i32::try_from(bitmap_height).ok()?;
    let x0 = (scanner.x_min / AA_SIZE).max(0);
    let x1 = (scanner.x_max / AA_SIZE).min(width_i - 1);
    let y0 = (scanner.y_min / AA_SIZE).max(0);
    let y1 = (scanner.y_max / AA_SIZE).min(height_i - 1);
    if x0 > x1 || y0 > y1 {
        return None;
    }
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 and y1 >= y0 checked above")]
    let rect = AaFillRect {
        x: x0,
        y: y0,
        width: (x1 - x0 + 1) as u32,
        height: (y1 - y0 + 1) as u32,
    };
    if rect.pixel_count()? < threshold {
        return None;
    }

    let row_starts = scanner.row_starts();
    let intersects = scanner.intersects();
    let eo_mask = if even_odd { 1i32 } else { !0i32 };
    let balanced = row_starts.windows(2).all(|w| {
        let sum = intersects[w[0] as usize..w[1] as usize]
            .iter()
            .fold(0i32, |acc, ix| acc.wrapping_add(ix.count));
        sum & eo_mask == 0
    });
    if !balanced {
        return None;
    }

    Some(GpuAaFill {
        intersects: intersects
            .iter()
            .map(|ix| ScanIntersect {
                x0: ix.x0,
                x1: ix.x1,
                count: ix.count,
            })
            .collect(),
        row_starts: row_starts.to_vec(),
        table_y0: scanner.y_min,
        rect,
        aa_width: bitmap_width.saturating_mul(AA_SIZE.unsigned_abs()),
    })
}

/// Composite device-computed coverage counts into `renderer.bitmap`.
///
/// `counts` must be exactly `rect.width × rect.height` bytes of sub-sample
/// counts (`0..=AA_SIZE²`); the slice is consumed as scratch.
#[cfg(feature = "gpu-aa")]
pub(super) fn gpu_coverage_to_bitmap(
    renderer: &mut PageRenderer<'_>,
    counts: &mut [u8],
    rect: AaFillRect,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
) {
    let clip = renderer.gstate.current().clip.clone_shared();
    paint_coverage(&mut renderer.bitmap, &clip, counts, rect, pipe, src);
}

/// Composite one rectangle of coverage counts, row by row, exactly as the
/// CPU fill loop does: the painted range of a row is its first and last
/// covered pixel, clamped to the clip rectangle, and the whole range
/// (interior zeros included) goes through `draw_aa_counts` as one span.
///
/// `rect` must lie within `bitmap` and the clip must be a rectangle within
/// the bitmap (both guaranteed by [`prepare_aa_fill`]).
#[cfg(feature = "gpu-aa")]
fn paint_coverage(
    bitmap: &mut raster::Bitmap<color::Rgb8>,
    clip: &Clip,
    counts: &mut [u8],
    rect: AaFillRect,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
) {
    let width = rect.width as usize;
    debug_assert_eq!(
        counts.len(),
        width * rect.height as usize,
        "coverage buffer does not match the fill rectangle"
    );
    debug_assert!(clip.is_rect(), "paint_coverage requires a rectangular clip");

    for (row, row_counts) in counts.chunks_exact_mut(width.max(1)).enumerate() {
        let Some(first) = row_counts.iter().position(|&c| c != 0) else {
            continue;
        };
        let last = row_counts
            .iter()
            .rposition(|&c| c != 0)
            .expect("a non-zero count exists");
        let y = rect.y + i32::try_from(row).expect("row fits i32");
        let x0 = (rect.x + i32::try_from(first).expect("column fits i32")).max(clip.x_min_i);
        let x1 = (rect.x + i32::try_from(last).expect("column fits i32")).min(clip.x_max_i);
        if x0 > x1 {
            continue;
        }
        #[expect(
            clippy::cast_sign_loss,
            reason = "x0 >= rect.x and x1 >= x0 by construction above"
        )]
        let span = &mut row_counts[(x0 - rect.x) as usize..=(x1 - rect.x) as usize];
        raster::fill::draw_aa_counts::<color::Rgb8>(bitmap, pipe, src, span, x0, x1, y);
    }
}

/// Attempt to rasterise `path` with the GPU scanline-AA kernel.
///
/// Returns `true` if the GPU path was taken (caller skips the CPU fill).
/// Returns `false` if [`prepare_aa_fill`] declines the fill or the GPU call
/// fails (warning logged; CPU fill used as fallback).
#[cfg(feature = "gpu-aa")]
pub(super) fn try_gpu_aa_fill(
    renderer: &mut PageRenderer<'_>,
    path: &raster::path::Path,
    even_odd: bool,
    flatness: f64,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    ctx: &gpu::GpuCtx,
) -> bool {
    let Some(fill) = prepare_aa_fill(
        &renderer.gstate.current().clip,
        renderer.bitmap.width,
        renderer.bitmap.height,
        path,
        flatness,
        even_odd,
        aa_fill_threshold(),
    ) else {
        return false;
    };

    let mut painted_any = false;
    let result = ctx.aa_fill_with(
        &fill.table(),
        fill.rect,
        fill.aa_width,
        even_odd,
        |row, counts| {
            let band = AaFillRect {
                y: fill.rect.y + i32::try_from(row).expect("row fits i32"),
                height: u32::try_from(counts.len() / fill.rect.width as usize)
                    .expect("band rows fit u32"),
                ..fill.rect
            };
            gpu_coverage_to_bitmap(renderer, counts, band, pipe, src);
            painted_any = true;
        },
    );
    match result {
        Ok(()) => true,
        // Compositing is not idempotent at partial coverage, so a CPU
        // fallback is only taken while no band has been painted.
        Err(e) if painted_any => {
            log::warn!("GPU AA fill failed part-way; the fill is incomplete: {e}");
            true
        }
        Err(e) => {
            log::warn!("GPU AA fill failed, falling back to CPU: {e}");
            false
        }
    }
}

#[cfg(all(test, feature = "gpu-aa"))]
mod tests {
    use raster::clip::Clip;
    use raster::path::{Path, PathBuilder};
    use raster::pipe::{PipeSrc, PipeState};
    use raster::types::BlendMode;
    use raster::xpath::XPath;

    use super::{DEVICE_MATRIX, paint_coverage, prepare_aa_fill};

    const W: u32 = 96;
    const H: u32 = 80;
    /// Dispatch threshold the eligibility tests run under: one 16×16 block.
    const THRESHOLD: usize = 256;

    fn pipe(a_input: u8) -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input,
            transfer: raster::TransferSet::identity_rgb(),
            backdrop: None,
        }
    }

    fn white_bitmap() -> raster::Bitmap<color::Rgb8> {
        let mut bitmap = raster::Bitmap::<color::Rgb8>::new(W, H, 1, false);
        bitmap.data_mut().fill(255);
        bitmap
    }

    fn page_clip() -> Clip {
        Clip::new(0.0, 0.0, f64::from(W), f64::from(H), true)
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Path {
        let mut b = PathBuilder::new();
        b.move_to(x0, y0).expect("move_to");
        b.line_to(x1, y0).expect("line_to");
        b.line_to(x1, y1).expect("line_to");
        b.line_to(x0, y1).expect("line_to");
        b.close(true).expect("close");
        b.build()
    }

    fn triangle() -> Path {
        let mut b = PathBuilder::new();
        b.move_to(7.3, 4.6).expect("move_to");
        b.line_to(88.9, 30.2).expect("line_to");
        b.line_to(21.4, 74.8).expect("line_to");
        b.close(true).expect("close");
        b.build()
    }

    fn circle() -> Path {
        // Four cubic arcs, kappa = 0.5523.
        let (cx, cy, r) = (47.37, 41.61, 30.9);
        let k = 0.5523 * r;
        let mut b = PathBuilder::new();
        b.move_to(cx + r, cy).expect("move_to");
        b.curve_to(cx + r, cy + k, cx + k, cy + r, cx, cy + r)
            .expect("curve_to");
        b.curve_to(cx - k, cy + r, cx - r, cy + k, cx - r, cy)
            .expect("curve_to");
        b.curve_to(cx - r, cy - k, cx - k, cy - r, cx, cy - r)
            .expect("curve_to");
        b.curve_to(cx + k, cy - r, cx + r, cy - k, cx + r, cy)
            .expect("curve_to");
        b.close(true).expect("close");
        b.build()
    }

    fn donut() -> Path {
        let mut b = PathBuilder::new();
        for (x0, y0, x1, y1) in [(10.5, 8.25, 85.75, 70.5), (30.2, 24.9, 66.1, 55.3)] {
            b.move_to(x0, y0).expect("move_to");
            b.line_to(x1, y0).expect("line_to");
            b.line_to(x1, y1).expect("line_to");
            b.line_to(x0, y1).expect("line_to");
            b.close(true).expect("close");
        }
        b.build()
    }

    /// Render `path` through the device model (prepare → host twin → paint)
    /// and through the CPU fill; both bitmaps must be byte-identical.
    fn assert_device_model_matches_cpu(
        name: &str,
        path: &Path,
        clip: &Clip,
        even_odd: bool,
        a_input: u8,
    ) {
        let pipe = pipe(a_input);
        let colour = [122u8, 117, 114];
        let src = PipeSrc::Solid(&colour);

        let fill = prepare_aa_fill(clip, W, H, path, 0.1, even_odd, THRESHOLD)
            .unwrap_or_else(|| panic!("{name}: fill must be device-eligible"));
        let mut counts = gpu::aa_fill_cpu(&fill.table(), fill.rect, fill.aa_width, even_odd);
        let mut device = white_bitmap();
        paint_coverage(&mut device, clip, &mut counts, fill.rect, &pipe, &src);

        let mut cpu = white_bitmap();
        if even_odd {
            raster::eo_fill(&mut cpu, clip, path, &pipe, &src, &DEVICE_MATRIX, 0.1, true);
        } else {
            raster::fill(&mut cpu, clip, path, &pipe, &src, &DEVICE_MATRIX, 0.1, true);
        }

        assert!(
            cpu.data().iter().any(|&b| b != 255),
            "{name}: CPU fill painted nothing"
        );
        let differing = cpu
            .data()
            .iter()
            .zip(device.data())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} bytes differ from the CPU fill"
        );
    }

    #[test]
    fn device_model_matches_cpu_on_fractional_rect() {
        let path = rect(20.31, 5.27, 30.22, 69.38);
        assert_device_model_matches_cpu("fractional rect", &path, &page_clip(), false, 255);
    }

    #[test]
    fn device_model_matches_cpu_on_slanted_and_curved_paths() {
        assert_device_model_matches_cpu("triangle", &triangle(), &page_clip(), false, 255);
        assert_device_model_matches_cpu("circle", &circle(), &page_clip(), false, 255);
    }

    #[test]
    fn device_model_matches_cpu_under_even_odd() {
        assert_device_model_matches_cpu("donut eo", &donut(), &page_clip(), true, 255);
        assert_device_model_matches_cpu("donut nz", &donut(), &page_clip(), false, 255);
    }

    #[test]
    fn device_model_matches_cpu_with_constant_alpha() {
        assert_device_model_matches_cpu("triangle alpha", &triangle(), &page_clip(), false, 128);
    }

    #[test]
    fn device_model_matches_cpu_under_rect_clip_and_off_page_geometry() {
        let mut clip = page_clip();
        clip.clip_to_rect(12.6, 9.4, 70.2, 60.7);
        assert_device_model_matches_cpu("clipped circle", &circle(), &clip, false, 255);

        let off_page = rect(-15.7, -9.2, 40.4, 33.6);
        assert_device_model_matches_cpu("off-page rect", &off_page, &page_clip(), false, 255);
        let past_edges = rect(60.3, 50.9, 130.0, 95.5);
        assert_device_model_matches_cpu("past-edge rect", &past_edges, &page_clip(), false, 255);
    }

    #[test]
    fn path_clips_stay_on_the_cpu() {
        let mut clip = page_clip();
        let mut b = PathBuilder::new();
        b.move_to(48.0, 2.0).expect("move_to");
        b.line_to(94.0, 40.0).expect("line_to");
        b.line_to(48.0, 78.0).expect("line_to");
        b.line_to(2.0, 40.0).expect("line_to");
        b.close(false).expect("close");
        let diamond = XPath::new(&b.build(), &DEVICE_MATRIX, 0.1, true);
        clip.clip_to_path(&diamond, false);
        assert!(!clip.is_rect(), "diamond must become a path clip");

        assert!(
            prepare_aa_fill(&clip, W, H, &circle(), 0.1, false, THRESHOLD).is_none(),
            "a fill under a path clip must not be dispatched to the device"
        );
    }

    #[test]
    fn default_threshold_keeps_every_fill_on_the_cpu() {
        let page = rect(0.5, 0.5, 95.5, 79.5);
        assert!(
            prepare_aa_fill(&page_clip(), W, H, &page, 0.1, false, THRESHOLD).is_some(),
            "a page-sized fill is device-eligible above the test threshold"
        );
        assert!(
            prepare_aa_fill(
                &page_clip(),
                W,
                H,
                &page,
                0.1,
                false,
                gpu::GPU_AA_FILL_THRESHOLD
            )
            .is_none(),
            "the shipped threshold leaves the device path dormant"
        );
    }

    #[test]
    fn small_and_empty_fills_stay_on_the_cpu() {
        let tiny = rect(10.0, 10.0, 20.0, 20.0);
        assert!(
            prepare_aa_fill(&page_clip(), W, H, &tiny, 0.1, false, THRESHOLD).is_none(),
            "a fill below the dispatch threshold stays on the CPU"
        );
        let outside = rect(200.0, 200.0, 300.0, 300.0);
        assert!(
            prepare_aa_fill(&page_clip(), W, H, &outside, 0.1, false, THRESHOLD).is_none(),
            "a fill outside the bitmap has nothing to dispatch"
        );
        assert!(
            prepare_aa_fill(
                &page_clip(),
                W,
                H,
                &PathBuilder::new().build(),
                0.1,
                false,
                THRESHOLD
            )
            .is_none(),
            "an empty path has nothing to dispatch"
        );
    }
}
