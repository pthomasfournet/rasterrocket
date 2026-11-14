//! Filled-path rasterization.
//!
//! Two public entry points:
//! - [`fill`] — non-zero winding fill
//! - [`eo_fill`] — even-odd fill
//!
//! Both entry points are thin wrappers around the private `fill_impl` which
//! does the actual work: build `XPath` → optionally AA-scale → build
//! `XPathScanner` → walk spans → clip → `pipe::render_bitmap_span`.
//!
//! # AA (vector antialias) mode
//!
//! When `vector_antialias` is `true` the path is scan-converted by
//! [`AreaRaster`] into each pixel's exact-area coverage over the clip
//! rectangle, the coverage of each row is scaled by the clip's
//! ([`Clip::mul_row`]), and the row is composited with the coverage as its
//! shape: runs of full coverage without a shape, the rest with one.

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::coverage::AreaRaster;
use crate::path::Path;
use crate::pipe::{self, PipeSrc, PipeState};
use crate::scanner::XPathScanner;
use crate::scanner::iter::ScanIterator;
use crate::simd;
use crate::types::floor_i32;
use crate::xpath::{PixelModel, XPath};
use color::Pixel;

/// Non-zero winding fill.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
pub fn fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
) {
    fill_impl::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        false,
    );
}

/// Even-odd fill.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
pub fn eo_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
) {
    fill_impl::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        true,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
fn fill_impl<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
) {
    if path.pts.is_empty() || bitmap.width == 0 {
        return;
    }

    if vector_antialias {
        area_fill::<P>(bitmap, clip, path, pipe, src, matrix, flatness, eo);
        return;
    }

    let mut xpath = XPath::new(path, matrix, flatness, true);
    xpath.cull_subpaths_outside(clip.x_min_i, clip.x_max_i, clip.y_min_i, clip.y_max_i);
    let scanner = XPathScanner::new(&xpath, eo, clip.y_min_i, clip.y_max_i);
    if scanner.is_empty() {
        return;
    }
    let clip_res = clip.test_rect(scanner.x_min, scanner.y_min, scanner.x_max, scanner.y_max);
    if clip_res == ClipResult::AllOutside {
        return;
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap.width ≤ i32::MAX in practice; zero checked above scanner.is_empty()"
    )]
    let width_i = bitmap.width as i32;

    // Iterate only over scanlines that have at least one intersection —
    // skips empty rows in the bounding box without touching the fill loop.
    for y in scanner.nonempty_rows() {
        #[expect(clippy::cast_sign_loss, reason = "cast after y < 0 guard")]
        if y < 0 || (y as u32) >= bitmap.height {
            continue;
        }
        for (x0, x1) in ScanIterator::new(&scanner, y) {
            // The span is clamped to the bitmap in both branches: a clip
            // rectangle need not lie within it.
            let (sx0, sx1) = if clip_res == ClipResult::AllInside {
                (x0.max(0), x1.min(width_i - 1))
            } else {
                (
                    x0.max(clip.x_min_i).max(0),
                    x1.min(clip.x_max_i).min(width_i - 1),
                )
            };
            if sx0 > sx1 {
                continue;
            }
            if clip_res == ClipResult::AllInside
                || clip.test_span(sx0, sx1, y) == ClipResult::AllInside
            {
                pipe::render_bitmap_span(bitmap, pipe, src, None, sx0, sx1, y);
            } else {
                for (rx0, rx1) in clip.inside_runs(sx0, sx1, y) {
                    pipe::render_bitmap_span(bitmap, pipe, src, None, rx0, rx1, y);
                }
            }
        }
    }
}

/// The exact-area rasteriser of `xpath` over its pixels inside the clip
/// rectangle and the `width × height` bitmap, its segments added; `None`
/// when no such pixel exists.
fn area_raster(xpath: &XPath, clip: &Clip, width: u32, height: u32) -> Option<AreaRaster> {
    let (lx, hx, ly, hy) = xpath.bounds()?;
    let last = |n: u32| i32::try_from(n).unwrap_or(i32::MAX).saturating_sub(1);
    let x0 = clip.x_min_i.max(0).max(floor_i32(lx));
    let y0 = clip.y_min_i.max(0).max(floor_i32(ly));
    let x1 = clip.x_max_i.min(last(width)).min(floor_i32(hx));
    let y1 = clip.y_max_i.min(last(height)).min(floor_i32(hy));
    if x0 > x1 || y0 > y1 {
        return None;
    }
    let mut raster = AreaRaster::new(x0, y0, x1, y1);
    raster.add_xpath(xpath);
    Some(raster)
}

/// Antialiased fill: exact-area coverage, scaled by the clip's, composited
/// row by row.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
fn area_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
    eo: bool,
) {
    let mut xpath = XPath::for_model(path, matrix, flatness, true, PixelModel::Area);
    xpath.cull_subpaths_outside(clip.x_min_i, clip.x_max_i, clip.y_min_i, clip.y_max_i);
    let Some(mut raster) = area_raster(&xpath, clip, bitmap.width, bitmap.height) else {
        return;
    };
    let clipped = !clip.is_rect();
    let mut scratch = Vec::new();
    raster.rows(eo, |y, x, coverage| {
        if clipped {
            scratch.clear();
            scratch.extend_from_slice(coverage);
            clip.mul_row(y, x, &mut scratch);
            draw_coverage_row::<P>(bitmap, pipe, src, &scratch, x, y);
        } else {
            draw_coverage_row::<P>(bitmap, pipe, src, coverage, x, y);
        }
    });
}

/// Runs of at least this many fully covered pixels are composited without
/// a shape; shorter ones stay in the surrounding shaped span.
const MIN_SOLID_RUN: usize = 16;

/// Composite `coverage[i]` as the shape of pixel `(x + i, y)`.  Runs of
/// [`MIN_SOLID_RUN`] or more fully covered pixels are composited with no
/// shape, uncovered runs as long are skipped, and the pixels between them
/// with their coverage as the shape.
fn draw_coverage_row<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    coverage: &[u8],
    x0: i32,
    row: i32,
) {
    let column = |at: usize| x0.saturating_add(i32::try_from(at).unwrap_or(i32::MAX));
    let run_of =
        |from: usize, value: u8| coverage[from..].iter().take_while(|&&c| c == value).count();
    let flush = |bitmap: &mut Bitmap<P>, from: Option<usize>, to: usize| {
        if let Some(from) = from
            && from < to
        {
            let shape = &coverage[from..to];
            pipe::render_bitmap_span::<P>(
                bitmap,
                pipe,
                src,
                Some(shape),
                column(from),
                column(to - 1),
                row,
            );
        }
    };
    let mut shaped_from = None;
    let mut at = 0;
    while at < coverage.len() {
        let value = coverage[at];
        if value == 0 || value == u8::MAX {
            let run = run_of(at, value);
            if run >= MIN_SOLID_RUN || (value == 0 && shaped_from.is_none()) {
                flush(bitmap, shaped_from.take(), at);
                if value == u8::MAX {
                    pipe::render_bitmap_span::<P>(
                        bitmap,
                        pipe,
                        src,
                        None,
                        column(at),
                        column(at + run - 1),
                        row,
                    );
                }
                at += run;
                continue;
            }
        }
        if shaped_from.is_none() {
            shaped_from = Some(at);
        }
        at += 1;
    }
    flush(bitmap, shaped_from, coverage.len());
}

/// Build the AA-space scanner for `path`: the geometry stage of an
/// antialiased fill.
///
/// The path is transformed by `matrix`, flattened with `flatness`, scaled by
/// [`AA_SIZE`], and scan-converted over the clip's AA row range
/// `[clip.y_min_i × AA_SIZE, (clip.y_max_i + 1) × AA_SIZE)`.  The scanner's
/// bounds are AA coordinates; dividing them by [`AA_SIZE`] gives the pixel
/// bounding box.
///
/// Returns `None` when the path has no points or the scanner covers no rows.
///
/// # Panics
///
/// Panics if the clip's AA row range overflows `i32`.
#[must_use]
pub fn aa_scanner(
    path: &Path,
    matrix: &[f64; 6],
    flatness: f64,
    clip: &Clip,
    eo: bool,
) -> Option<XPathScanner> {
    if path.pts.is_empty() {
        return None;
    }
    let mut xpath = XPath::new(path, matrix, flatness, true);
    xpath.aa_scale();
    let (y_min, y_max) = clip.aa_row_range();
    let (x_min, x_max) = clip.aa_column_range();
    xpath.cull_subpaths_outside(x_min, x_max, y_min, y_max);
    let scanner = XPathScanner::new(&xpath, eo, y_min, y_max);
    (!scanner.is_empty()).then_some(scanner)
}

/// Antialiased coverage of the device-space `xpath` over the pixels
/// `[0, width) × [0, height)` under the non-zero winding rule.
///
/// The result holds `width × height` bytes, row-major, each the exact-area
/// shape an antialiased [`fill`] of the same outline composites at that
/// pixel.  Placed at an integer pixel offset and composited as an 8-bit
/// glyph, the coverage reproduces such a fill under a clip it lies wholly
/// inside.
#[must_use]
pub fn aa_coverage(xpath: &XPath, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut coverage = vec![0u8; w * h];
    let whole = Clip::new(0.0, 0.0, f64::from(width), f64::from(height), true);
    let Some(mut raster) = area_raster(xpath, &whole, width, height) else {
        return coverage;
    };
    raster.rows(false, |y, x, row| {
        if let (Ok(y), Ok(x)) = (usize::try_from(y), usize::try_from(x)) {
            let start = y * w + x;
            let n = row.len().min(w - x);
            coverage[start..start + n].copy_from_slice(&row[..n]);
        }
    });
    coverage
}

// ── Span drawing helpers ──────────────────────────────────────────────────────

/// Composite one output row from per-pixel AA coverage counts.
///
/// `counts[i]` is the number of set AA sub-samples (`0..=AA_SIZE²`) of pixel
/// `x0 + i`; `counts.len()` must equal `x1 - x0 + 1`.  The counts are mapped
/// through `AA_GAMMA` in place and composited via `pipe::render_span` as one
/// span; a row whose counts are all zero leaves the bitmap untouched.
///
/// This is the compositing half of `draw_aa_line`; the GPU fill path
/// supplies counts computed on the device and joins the CPU pipeline here.
///
/// # Panics
///
/// Panics if `counts.len() != x1 - x0 + 1` or if any count exceeds
/// `AA_SIZE²`.
pub fn draw_aa_counts<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    counts: &mut [u8],
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 >= 0, "draw_aa_counts: x0={x0} is negative");
    debug_assert!(x0 <= x1, "draw_aa_counts: x0={x0} > x1={x1}");
    debug_assert!(y >= 0, "draw_aa_counts: y={y} is negative");
    #[expect(clippy::cast_sign_loss, reason = "x1 >= x0 >= 0 asserted above")]
    let expected = (x1 - x0 + 1) as usize;
    assert_eq!(
        counts.len(),
        expected,
        "draw_aa_counts: {} counts for a {expected}-pixel span",
        counts.len()
    );

    let shape = counts;
    if !simd::apply_aa_gamma(shape) {
        return;
    }

    pipe::render_bitmap_span::<P>(bitmap, pipe, src, Some(&*shape), x0, x1, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::path::PathBuilder;
    use crate::pipe::PipeSrc;
    use crate::testutil::{identity_matrix, make_clip, rect_path, simple_pipe};
    use color::Rgb8;

    /// The AA fill of `paths` in white on a black `w × h` bitmap, as the
    /// red channel per pixel, row-major.
    fn aa_render(paths: &[Path], w: u32, h: u32, eo: bool) -> Vec<u8> {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(w, h, 1, false);
        let white = [255u8; 3];
        let clip = Clip::new(0.0, 0.0, f64::from(w), f64::from(h), true);
        for path in paths {
            let draw = if eo { eo_fill::<Rgb8> } else { fill::<Rgb8> };
            draw(
                &mut bmp,
                &clip,
                path,
                &simple_pipe(),
                &PipeSrc::Solid(&white),
                &identity_matrix(),
                0.1,
                true,
            );
        }
        (0..h)
            .flat_map(|y| bmp.row(y).iter().map(|p| p.r).collect::<Vec<_>>())
            .collect()
    }

    /// A one-pixel band centred on a row boundary covers the rows on either
    /// side by half each.
    #[test]
    fn aa_band_on_a_row_boundary_covers_both_rows_equally() {
        let out = aa_render(&[rect_path(0.0, 9.5, 16.0, 10.5)], 16, 16, false);
        for x in 0..16 {
            let (above, below) = (out[9 * 16 + x], out[10 * 16 + x]);
            assert!(above.abs_diff(below) <= 1, "column {x}: {above} vs {below}");
            assert!(above.abs_diff(128) <= 1, "column {x}: {above}");
            assert_eq!(out[8 * 16 + x] | out[11 * 16 + x], 0, "column {x}");
        }
    }

    /// A rectangle offset by half a pixel covers its edge pixels by half
    /// and its corners by a quarter.
    #[test]
    fn aa_half_pixel_offset_rectangle_has_exact_edges() {
        let out = aa_render(&[rect_path(2.5, 2.5, 6.5, 6.5)], 9, 9, false);
        let at = |x: usize, y: usize| out[y * 9 + x];
        assert_eq!([at(2, 2), at(6, 2), at(2, 6), at(6, 6)], [64; 4], "corners");
        assert_eq!([at(4, 2), at(4, 6), at(2, 4), at(6, 4)], [128; 4], "edges");
        assert_eq!(at(4, 4), 255);
        assert_eq!(at(1, 4) | at(7, 4) | at(4, 1) | at(4, 7), 0);
    }

    /// Under even-odd the overlap of two subpaths is a hole, and a hole
    /// edge inside a pixel leaves the rest of the pixel covered.
    #[test]
    fn aa_even_odd_overlaps_are_holes_with_exact_edges() {
        let mut b = PathBuilder::new();
        for (x0, y0, x1, y1) in [
            (0.0, 0.0, 12.0, 12.0),
            (2.5, 2.5, 7.5, 7.5),
            (5.0, 5.0, 10.0, 10.0),
        ] {
            b.move_to(x0, y0).unwrap();
            b.line_to(x1, y0).unwrap();
            b.line_to(x1, y1).unwrap();
            b.line_to(x0, y1).unwrap();
            b.close(true).unwrap();
        }
        let path = b.build();
        let eo = aa_render(std::slice::from_ref(&path), 12, 12, true);
        let at = |x: usize, y: usize| eo[y * 12 + x];
        assert_eq!(at(1, 1), 255, "outer ring");
        assert_eq!(at(2, 3), 128, "hole edge inside the pixel");
        assert_eq!(at(3, 3), 0, "hole");
        assert_eq!(at(6, 6), 255, "the third square fills the hole back in");
        assert_eq!(at(8, 8), 0, "the third square cuts a hole in the outer one");
        let nz = aa_render(&[path], 12, 12, false);
        assert!(
            nz.iter().all(|&v| v == 255),
            "non-zero covers the whole square"
        );
    }

    /// The ink an AA fill lays down equals the area of its outline.
    #[test]
    fn aa_fill_ink_equals_area() {
        let mut b = PathBuilder::new();
        let tri = [(1.3, 0.7), (29.1, 6.4), (8.2, 27.8)];
        b.move_to(tri[0].0, tri[0].1).unwrap();
        b.line_to(tri[1].0, tri[1].1).unwrap();
        b.line_to(tri[2].0, tri[2].1).unwrap();
        b.close(true).unwrap();
        let out = aa_render(&[b.build()], 32, 32, false);
        let area = 0.5
            * (tri[1].0 - tri[0].0).mul_add(
                tri[2].1 - tri[0].1,
                -(tri[2].0 - tri[0].0) * (tri[1].1 - tri[0].1),
            );
        let ink: f64 = out.iter().map(|&v| f64::from(v) / 255.0).sum();
        assert!(
            (ink - area.abs()).abs() < 0.2,
            "ink {ink} vs area {}",
            area.abs()
        );
    }

    /// An AA fill covers the whole first sub-row of an edge that lies on a
    /// pixel boundary: the horizontal top edge and the vertical edges'
    /// starting sub-row carry winding count 0 but still bound the shape.
    #[test]
    fn aa_fill_covers_integer_top_edge_row() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let white = [255u8, 255, 255];
        fill::<Rgb8>(
            &mut bmp,
            &make_clip(8, 8),
            &rect_path(1.0, 2.0, 7.0, 6.0),
            &simple_pipe(),
            &PipeSrc::Solid(&white),
            &identity_matrix(),
            0.1,
            true,
        );
        let col: Vec<u8> = (0..8).map(|y| bmp.row(y)[3].r).collect();
        assert_eq!(
            col[..6],
            [0, 0, 255, 255, 255, 255],
            "rows 0..6 of column 3"
        );
    }

    /// Inclusive row range that received any ink, or `None` for a blank bitmap.
    fn inked_rows(bmp: &Bitmap<Rgb8>) -> Option<(u32, u32)> {
        let mut first = None;
        let mut last = None;
        for y in 0..bmp.height {
            if bmp
                .row(y)
                .iter()
                .any(|px| px.r != 0 || px.g != 0 || px.b != 0)
            {
                if first.is_none() {
                    first = Some(y);
                }
                last = Some(y);
            }
        }
        first.zip(last)
    }

    /// Inclusive column range that received any ink, or `None` for a blank bitmap.
    fn inked_cols(bmp: &Bitmap<Rgb8>) -> Option<(u32, u32)> {
        let mut first = None;
        let mut last = None;
        for y in 0..bmp.height {
            for (x, px) in bmp.row(y).iter().enumerate() {
                if px.r != 0 || px.g != 0 || px.b != 0 {
                    let x = u32::try_from(x).expect("bitmap column fits u32");
                    if first.is_none_or(|f| x < f) {
                        first = Some(x);
                    }
                    if last.is_none_or(|l| x > l) {
                        last = Some(x);
                    }
                }
            }
        }
        first.zip(last)
    }

    /// The coverage of an outline is the shape an antialiased fill of it
    /// composites: white filled onto black carries the shape byte as is.
    #[test]
    fn sloped_edges_fill_far_right_on_a_wide_page() {
        // A triangle past device column 8192, where antialiased coordinates
        // exceed 16.16 fixed point.
        let mut b = PathBuilder::new();
        b.move_to(9000.0, 10.0).unwrap();
        b.line_to(9100.0, 10.0).unwrap();
        b.line_to(9050.0, 100.0).unwrap();
        b.close(false).unwrap();
        let path = b.build();
        let xpath = XPath::new(&path, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true);
        let (w, h) = (9200, 120);
        let coverage = aa_coverage(&xpath, w, h);
        let at = |x: usize, y: usize| coverage[y * w as usize + x];
        assert_eq!(at(9050, 50), 255, "inside the triangle");
        assert_eq!(at(8191, 50), 0, "far left of it");
        assert_eq!(at(9150, 50), 0, "right of it");
    }

    #[test]
    fn aa_coverage_matches_an_aa_fill() {
        let mut b = PathBuilder::new();
        b.move_to(1.3, 0.6).unwrap();
        b.line_to(10.7, 3.2).unwrap();
        b.line_to(4.1, 9.4).unwrap();
        b.close(false).unwrap();
        let path = b.build();
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(12, 10, 1, false);
        fill::<Rgb8>(
            &mut bmp,
            &make_clip(12, 10),
            &path,
            &simple_pipe(),
            &PipeSrc::Solid(&[255, 255, 255]),
            &identity_matrix(),
            0.1,
            true,
        );
        let filled: Vec<u8> = (0..10)
            .flat_map(|y| bmp.row(y).iter().map(|p| p.r))
            .collect();
        let xpath = XPath::new(&path, &identity_matrix(), 0.1, true);
        let coverage = aa_coverage(&xpath, 12, 10);
        assert!(
            coverage.iter().any(|&v| v > 0 && v < 255),
            "edges are partial"
        );
        assert_eq!(coverage, filled);
    }

    /// An AA fill lands where its geometry is: `render_aa_line` takes a
    /// device-pixel row and expands it to the four AA sub-rows itself.
    #[test]
    fn aa_fill_rect_lands_on_its_own_geometry() {
        const W: u32 = 64;
        const H: u32 = 64;

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
        let clip = make_clip(W, H);
        let pipe = simple_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        // Deliberately away from the top edge: a row-space scaling bug cannot
        // masquerade as a correct result here.
        let path = rect_path(16.0, 40.0, 48.0, 56.0);

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            true,
        );

        let (r0, r1) = inked_rows(&bmp).expect("AA fill painted nothing at all");
        assert!(
            r0 >= 39 && r1 <= 57,
            "AA fill ink at rows {r0}..={r1}, expected to stay within 39..=57"
        );
        assert!(
            r1 - r0 >= 14,
            "AA fill covered only {} rows, expected ~16",
            r1 - r0 + 1
        );

        let (c0, c1) = inked_cols(&bmp).expect("AA fill painted nothing at all");
        assert!(
            c0 >= 15 && c1 <= 49,
            "AA fill ink at cols {c0}..={c1}, expected to stay within 15..=49"
        );
    }

    /// A fill sitting below a quarter of the page height must still render.
    ///
    /// With an AA-space row index the scanner row lookup ran off the end of the
    /// table for any geometry below `height / AA_SIZE`, so the fill silently
    /// produced nothing.
    #[test]
    fn aa_fill_low_on_the_page_is_not_dropped() {
        const W: u32 = 32;
        const H: u32 = 128;

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
        let clip = make_clip(W, H);
        let pipe = simple_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        // Entirely below H/AA_SIZE = 32.
        let path = rect_path(8.0, 96.0, 24.0, 120.0);

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            true,
        );

        let (r0, r1) = inked_rows(&bmp).expect("AA fill below H/4 rendered nothing");
        assert!(
            r0 >= 95 && r1 <= 121,
            "AA fill ink at rows {r0}..={r1}, expected to stay within 95..=121"
        );
    }

    /// AA and non-AA fills of the same rect must agree on where the shape is.
    ///
    /// AA differs from aliased rendering at the edges by design, but the two
    /// must not disagree about which part of the bitmap the shape occupies.
    #[test]
    fn aa_and_aliased_fill_agree_on_placement() {
        const W: u32 = 48;
        const H: u32 = 48;

        let clip = make_clip(W, H);
        let pipe = simple_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let path = rect_path(12.0, 20.0, 36.0, 40.0);
        let matrix = identity_matrix();

        let mut aliased: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
        fill::<Rgb8>(&mut aliased, &clip, &path, &pipe, &src, &matrix, 1.0, false);

        let mut aa: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
        fill::<Rgb8>(&mut aa, &clip, &path, &pipe, &src, &matrix, 1.0, true);

        let (ar0, ar1) = inked_rows(&aliased).expect("aliased fill painted nothing");
        let (br0, br1) = inked_rows(&aa).expect("AA fill painted nothing");

        // Allow one row of AA bleed on each side, no more.
        assert!(
            ar0.abs_diff(br0) <= 1 && ar1.abs_diff(br1) <= 1,
            "AA rows {br0}..={br1} disagree with aliased rows {ar0}..={ar1}"
        );
    }

    #[test]
    fn fill_rect_paints_solid() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        // Rect (1,1)→(5,5): every pixel its closed outline touches is
        // painted, rows 1..=5 and columns 1..=5; the horizontal edges carry
        // winding count 0 yet cover their own cells.
        let path = rect_path(1.0, 1.0, 5.0, 5.0);

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

        for y in 0..8u32 {
            for (x, px) in bmp.row(y).iter().enumerate() {
                let want = if (1..=5).contains(&y) && (1..=5).contains(&x) {
                    color
                } else {
                    [0; 3]
                };
                assert_eq!([px.r, px.g, px.b], want, "y={y} x={x}");
            }
        }
    }

    /// An aliased fill does not depend on the direction of its outline: a
    /// thin triangle whose edges share a cell on a scanline paints the
    /// same pixels traversed either way.
    #[test]
    fn aliased_fill_is_independent_of_orientation() {
        let triangle = |pts: [(f64, f64); 3]| {
            let mut b = PathBuilder::new();
            b.move_to(pts[0].0, pts[0].1).unwrap();
            b.line_to(pts[1].0, pts[1].1).unwrap();
            b.line_to(pts[2].0, pts[2].1).unwrap();
            b.close(false).unwrap();
            b.build()
        };
        let (a, b, c) = ((0.0, -0.0625), (0.0, 1.25), (5.0, 0.625));
        let draw = |path: &Path| {
            let mut bmp: Bitmap<Rgb8> = Bitmap::new(6, 2, 1, false);
            fill::<Rgb8>(
                &mut bmp,
                &make_clip(6, 2),
                path,
                &simple_pipe(),
                &PipeSrc::Solid(&[255, 255, 255]),
                &identity_matrix(),
                1.0,
                false,
            );
            (0..2)
                .flat_map(|y| bmp.row(y).iter().map(|p| p.r).collect::<Vec<_>>())
                .collect::<Vec<u8>>()
        };
        let forward = draw(&triangle([a, b, c]));
        assert_eq!(forward, draw(&triangle([c, b, a])));
        assert_eq!(forward[0], 255, "the triangle covers pixel (0, 0)");
    }

    #[test]
    fn fill_empty_path_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);
        let path = PathBuilder::new().build(); // empty

        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

        // Nothing should be painted.
        assert_eq!(bmp.row(0)[0].r, 0);
    }

    #[test]
    fn eo_fill_donut_leaves_interior_clear() {
        // Even-odd rule: a square inside a larger square leaves the inner area unfilled.
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, false);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // Outer square 1..8, inner square 3..6.
        let mut b = PathBuilder::new();
        b.move_to(1.0, 1.0).unwrap();
        b.line_to(8.0, 1.0).unwrap();
        b.line_to(8.0, 8.0).unwrap();
        b.line_to(1.0, 8.0).unwrap();
        b.close(true).unwrap();
        b.move_to(3.0, 3.0).unwrap();
        b.line_to(6.0, 3.0).unwrap();
        b.line_to(6.0, 6.0).unwrap();
        b.line_to(3.0, 6.0).unwrap();
        b.close(true).unwrap();
        let path = b.build();

        eo_fill::<Rgb8>(
            &mut bmp,
            &clip,
            &path,
            &pipe,
            &src,
            &identity_matrix(),
            1.0,
            false,
        );

        // Interior (4,4) should be unpainted.
        assert_eq!(bmp.row(4)[4].r, 0, "interior should be clear with EO rule");
        // Band (2, 2) should be painted.
        assert_eq!(bmp.row(2)[2].r, 255, "outer band should be painted");
    }

    #[test]
    fn aa_gamma_table_correct() {
        use crate::types::AA_GAMMA;
        // Validate every entry against the defining formula:
        // round((i / (AA_SIZE² )) ^ 1.5 * 255).  Derive the divisor from the
        // table's own length so the test stays correct if AA_SIZE changes.
        let max_idx = AA_GAMMA.len() - 1; // == AA_SIZE * AA_SIZE
        #[expect(
            clippy::cast_precision_loss,
            reason = "AA_GAMMA.len() ≤ 65 in practice; f64 represents it exactly"
        )]
        let divisor = max_idx as f64;
        for (i, &actual) in AA_GAMMA.iter().enumerate() {
            // i ∈ [0, max_idx]; (i/max_idx)^1.5 ∈ [0,1]; ×255 ∈ [0,255]; round
            // to non-negative integer in [0,255] — the `as u8` is provably in range.
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss,
                reason = "value is f64::round() of a [0,255]-bounded expression"
            )]
            let expected = ((i as f64 / divisor).powf(1.5) * 255.0).round() as u8;
            assert_eq!(actual, expected, "AA_GAMMA[{i}]: expected {expected}");
        }
    }

    #[test]
    fn scanner_produces_spans_for_rect() {
        use crate::scanner::XPathScanner;
        use crate::scanner::iter::ScanIterator;
        use crate::xpath::XPath;

        let path = rect_path(1.0, 1.0, 5.0, 5.0);
        let xpath = XPath::new(&path, &identity_matrix(), 1.0, true);
        let scanner = XPathScanner::new(&xpath, false, 0, 7);
        // Interior rows 2,3,4 should have spans; boundary rows 1,5 have horizontal
        // edges (count=0) and produce no interior spans.
        assert!(
            ScanIterator::new(&scanner, 2).next().is_some(),
            "no spans at y=2"
        );
        assert!(
            ScanIterator::new(&scanner, 3).next().is_some(),
            "no spans at y=3"
        );
        assert!(
            ScanIterator::new(&scanner, 4).next().is_some(),
            "no spans at y=4"
        );
    }

    /// An AA fill under a path clip paints the intersection of the fill and
    /// the clip.  A 10 × 10 square centred in a diamond clip must paint the
    /// square alone, not the diamond's width across every row it touches.
    #[test]
    fn aa_fill_under_path_clip_paints_the_intersection() {
        const W: u32 = 100;

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(W, W, 1, false);
        let mut clip = crate::clip::Clip::new(0.0, 0.0, f64::from(W), f64::from(W), true);
        let mut diamond = PathBuilder::new();
        diamond.move_to(50.0, 5.0).expect("move_to");
        diamond.line_to(95.0, 50.0).expect("line_to");
        diamond.line_to(50.0, 95.0).expect("line_to");
        diamond.line_to(5.0, 50.0).expect("line_to");
        diamond.close(false).expect("close");
        let xpath = crate::xpath::XPath::new(&diamond.build(), &identity_matrix(), 0.1, true);
        clip.clip_to_path(&xpath, false);

        let color = [255u8, 255, 255];
        fill::<Rgb8>(
            &mut bmp,
            &clip,
            &rect_path(45.0, 45.0, 55.0, 55.0),
            &simple_pipe(),
            &PipeSrc::Solid(&color),
            &identity_matrix(),
            1.0,
            true,
        );

        let mut painted = Vec::new();
        for y in 0..W {
            for (x, px) in bmp.row(y).iter().enumerate() {
                if px.r >= 128 {
                    painted.push((x, y));
                }
            }
        }
        assert_eq!(painted.len(), 100, "exactly the 10 × 10 square is painted");
        assert!(
            painted
                .iter()
                .all(|&(x, y)| (45..55).contains(&x) && (45..55).contains(&y)),
            "no pixel outside the square"
        );
    }
}
