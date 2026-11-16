//! Filled-path rasterization.
//!
//! Two public entry points:
//! - [`fill`] — non-zero winding fill
//! - [`eo_fill`] — even-odd fill
//!
//! Both entry points are thin wrappers around the private `fill_impl` which
//! does the actual work: build `XPath`, then either build `XPathScanner` →
//! walk spans → clip → `pipe::render_bitmap_span` (aliased), or the
//! exact-area coverage below (antialiased).
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
use crate::coverage::{AreaRaster, BLOCK, CoverageTable, run_len, uniform};
use crate::path::{Path, PathBuilder, collinear};
use crate::pipe::{self, PipeSrc, PipeState};
use crate::scanner::XPathScanner;
use crate::scanner::iter::ScanIterator;
use crate::stroke::{StrokeParams, stroke_outline};
use crate::types::{LineCap, LineJoin, floor_i32};
use crate::xpath::{PixelModel, XPath, XPathFlags};
use color::Pixel;
use color::convert::div255;

/// Non-zero winding fill.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
pub fn fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
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
        FillRule::NonZero,
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
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
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
        FillRule::EvenOdd,
    );
}

/// Fill the area `path` encloses, by the even-odd rule when `eo`, else
/// non-zero.
///
/// Unlike under [`fill`] and [`eo_fill`], a subpath that encloses no area
/// paints nothing.  It fills the outline of a stroke and the path a fill
/// and stroke of the same path paints.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
pub fn fill_area<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    eo: bool,
) {
    let rule = if eo {
        FillRule::EvenOddArea
    } else {
        FillRule::NonZeroArea
    };
    fill_impl::<P>(
        bitmap,
        clip,
        path,
        pipe,
        src,
        matrix,
        flatness,
        vector_antialias,
        rule,
    );
}

/// How [`fill_impl`] decides which points an outline paints.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FillRule {
    /// Non-zero winding; an antialiased fill also draws each subpath that
    /// encloses no area as a line ([`zero_area_lines`]).
    NonZero,
    /// Even-odd; zero-area subpaths as under `NonZero`.
    EvenOdd,
    /// Non-zero winding alone.
    NonZeroArea,
    /// Even-odd alone.
    EvenOddArea,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, geometry, compositing state and rasterisation options are independent inputs"
)]
fn fill_impl<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    matrix: &[f64; 6],
    flatness: f64,
    vector_antialias: bool,
    rule: FillRule,
) {
    if path.pts.is_empty() || bitmap.width == 0 {
        return;
    }
    let eo = matches!(rule, FillRule::EvenOdd | FillRule::EvenOddArea);

    if vector_antialias {
        let mut xpath = XPath::for_model(path, matrix, flatness, true, PixelModel::Area);
        // A line reaches half a pixel past its subpath, which culling would
        // drop when it lies just outside the clip.
        let lines = matches!(rule, FillRule::NonZero | FillRule::EvenOdd)
            .then(|| zero_area_lines(&xpath, flatness))
            .flatten();
        xpath.cull_subpaths_outside(clip.x_min_i, clip.x_max_i, clip.y_min_i, clip.y_max_i);
        area_fill::<P>(bitmap, clip, &xpath, pipe, src, eo, u8::MAX);
        if let Some(lines) = lines {
            area_fill::<P>(bitmap, clip, &lines, pipe, src, false, ZERO_AREA_WEIGHT);
        }
        return;
    }

    let mut xpath = XPath::new(path, matrix, flatness, true);
    xpath.cull_subpaths_outside(clip.x_min_i, clip.x_max_i, clip.y_min_i, clip.y_max_i);
    let scanner = match XPathScanner::new(&xpath, eo, clip.y_min_i, clip.y_max_i) {
        Ok(scanner) => scanner,
        Err(e) => {
            log::warn!("fill dropped: {e}");
            return;
        }
    };
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

/// The weight, out of 255, of the coverage of the line a zero-area
/// subpath draws: a quarter.
const ZERO_AREA_WEIGHT: u8 = u8::MAX / 4;

/// Antialiased fill of the device-space `xpath`: exact-area coverage,
/// scaled by `weight / 255` and by the clip's, composited row by row.
fn area_fill<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    xpath: &XPath,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    eo: bool,
    weight: u8,
) {
    let clipped = !clip.is_rect();
    area_rows(
        xpath,
        clip,
        bitmap.width,
        bitmap.height,
        eo,
        weight,
        |y, x, coverage| {
            if clipped {
                clip.mul_row(y, x, coverage);
            }
            draw_coverage_row::<P>(bitmap, pipe, src, coverage, x, y);
        },
    );
}

/// Call `emit(y, x, coverage)` for each row of the exact-area coverage of
/// the device-space `xpath` over its pixels inside the clip rectangle and
/// the `width × height` bitmap, scaled by `weight / 255`; `coverage[i]` is
/// the coverage of pixel `(x + i, y)` and may be changed in place.
fn area_rows(
    xpath: &XPath,
    clip: &Clip,
    width: u32,
    height: u32,
    eo: bool,
    weight: u8,
    mut emit: impl FnMut(i32, i32, &mut [u8]),
) {
    let Some(mut raster) = area_raster(xpath, clip, width, height) else {
        return;
    };
    raster.rows_mut(eo, |y, x, coverage| {
        if weight != u8::MAX {
            for c in coverage.iter_mut() {
                *c = div255(u32::from(*c) * u32::from(weight));
            }
        }
        emit(y, x, coverage);
    });
}

/// The coverage tables an antialiased fill of `path` composites, in order.
///
/// The fill is [`fill`] (or [`eo_fill`] when `eo`) under the rectangle of
/// `clip` on a `width × height` bitmap: the area, then the lines of its
/// zero-area subpaths when `with_lines` (only the area under
/// [`fill_area`]).  Compositing each in turn, its coverage as the shape, paints what the
/// fill paints where the clip is its rectangle.  Empty tables are left out.
#[expect(
    clippy::too_many_arguments,
    reason = "the geometry, clip, bitmap size and fill rule are independent inputs"
)]
#[must_use]
pub fn aa_fill_coverage(
    path: &Path,
    matrix: &[f64; 6],
    flatness: f64,
    clip: &Clip,
    width: u32,
    height: u32,
    eo: bool,
    with_lines: bool,
) -> Vec<CoverageTable> {
    if path.pts.is_empty() || width == 0 {
        return Vec::new();
    }
    let mut xpath = XPath::for_model(path, matrix, flatness, true, PixelModel::Area);
    let lines = with_lines
        .then(|| zero_area_lines(&xpath, flatness))
        .flatten();
    xpath.cull_subpaths_outside(clip.x_min_i, clip.x_max_i, clip.y_min_i, clip.y_max_i);
    let mut layers = vec![table_of(&xpath, clip, width, height, eo, u8::MAX)];
    if let Some(lines) = lines {
        layers.push(table_of(
            &lines,
            clip,
            width,
            height,
            false,
            ZERO_AREA_WEIGHT,
        ));
    }
    layers.retain(|t| !t.is_empty());
    layers
}

/// The rows of [`area_rows`] as a table.
fn table_of(
    xpath: &XPath,
    clip: &Clip,
    width: u32,
    height: u32,
    eo: bool,
    weight: u8,
) -> CoverageTable {
    let mut table = CoverageTable::empty();
    area_rows(xpath, clip, width, height, eo, weight, |y, x, coverage| {
        table.push_row(y, x, coverage);
    });
    table
}

/// The outline, in device space, of a stroke one pixel wide with butt caps
/// along every closed subpath of `xpath` that encloses no area but has an
/// extent (its points collinear); `None` when there is none.
///
/// A subpath's segments lie on one line, and the stroke follows the
/// disjoint intervals of the line they cover, each once, from the end with
/// the smaller coordinates: the outline does not depend on the subpath's
/// direction or starting point, and overlapping segments add no coverage.
/// A subpath of a single line (one segment and its closing one) has its
/// ends moved to the centres of the pixels holding them, so a line on a
/// pixel boundary covers the pixels after it rather than half of each
/// side.  A subpath that encloses an area contributes the spikes where it
/// folds back on itself ([`fold_spikes`]), which enclose none.
fn zero_area_lines(xpath: &XPath, flatness: f64) -> Option<XPath> {
    let mut lines = PathBuilder::new();
    let mut any = false;
    let mut segs = Vec::new();
    for subpath in xpath
        .segs
        .chunk_by(|_, next| !next.flags.contains(XPathFlags::SUBPATH))
    {
        segs.clear();
        segs.extend(subpath.iter().map(|s| ((s.x0, s.y0), (s.x1, s.y1))));
        let Some(intervals) = collinear::intervals(&segs) else {
            for (from, to) in fold_spikes(subpath) {
                any |= add_line(&mut lines, from, to, false);
            }
            continue;
        };
        let single = subpath.len() <= 2;
        for [from, to] in intervals {
            any |= add_line(&mut lines, from, to, single);
        }
    }
    if !any {
        return None;
    }
    let params = StrokeParams {
        line_width: 1.0,
        line_cap: LineCap::Butt,
        line_join: LineJoin::Miter,
        miter_limit: 10.0,
        flatness,
        stroke_adjust: false,
        line_dash: &[],
        line_dash_phase: 0.0,
        vector_antialias: true,
    };
    let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    stroke_outline(&lines.build(), &identity, &params)
}

/// The spikes of the closed subpath `segs`: at each vertex where the
/// subpath turns back along the line it came on, the part the two
/// segments share, from the vertex to the nearer of their other ends.
fn fold_spikes(segs: &[crate::xpath::XPathSeg]) -> Vec<(collinear::Point, collinear::Point)> {
    let directed = |s: &crate::xpath::XPathSeg| {
        if s.flags.contains(XPathFlags::FLIPPED) {
            ((s.x1, s.y1), (s.x0, s.y0))
        } else {
            ((s.x0, s.y0), (s.x1, s.y1))
        }
    };
    let mut spikes = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        let (prev, vertex) = directed(seg);
        let (_, next) = directed(&segs[(i + 1) % segs.len()]);
        let (ax, ay) = (prev.0 - vertex.0, prev.1 - vertex.1);
        let (bx, by) = (next.0 - vertex.0, next.1 - vertex.1);
        let (la, lb) = (ax.hypot(ay), bx.hypot(by));
        let cross = ax.mul_add(by, -(ay * bx));
        let dot = ax.mul_add(bx, ay * by);
        if la > 0.0 && lb > 0.0 && dot > 0.0 && cross.abs() <= 1e-9 * la * lb {
            spikes.push((vertex, if la <= lb { prev } else { next }));
        }
    }
    spikes
}

/// Append the open subpath `from → to` to `lines`, its ends moved to the
/// centres of their pixels when `centre`; whether it was added.
fn add_line(lines: &mut PathBuilder, from: (f64, f64), to: (f64, f64), centre: bool) -> bool {
    let place = |p: (f64, f64)| {
        if centre {
            (p.0.floor() + 0.5, p.1.floor() + 0.5)
        } else {
            p
        }
    };
    let (from, to) = (place(from), place(to));
    lines.move_to(from.0, from.1).is_ok() && lines.line_to(to.0, to.1).is_ok()
}

/// Composite `coverage[i]` as the shape of pixel `(x0 + i, row)`.
///
/// The row is walked in blocks of [`BLOCK`] pixels from the end of the
/// previous run: a block of full coverage starts a run composited with no
/// shape, a block of no coverage starts a run that is skipped, each run
/// extending as far as its value does; the pixels between runs are
/// composited with their coverage as the shape.
fn draw_coverage_row<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    coverage: &[u8],
    x0: i32,
    row: i32,
) {
    let column = |at: usize| x0.saturating_add(i32::try_from(at).unwrap_or(i32::MAX));
    let draw = |bitmap: &mut Bitmap<P>, from: usize, to: usize, solid: bool| {
        if from < to {
            let shape = (!solid).then(|| &coverage[from..to]);
            pipe::render_bitmap_span_coverage::<P>(
                bitmap,
                pipe,
                src,
                shape,
                column(from),
                column(to - 1),
                row,
            );
        }
    };
    let mut at = run_len(coverage, 0);
    let mut shaped_from = at;
    while at < coverage.len() {
        let block = &coverage[at..coverage.len().min(at + BLOCK)];
        let value = if uniform(block, u8::MAX) {
            u8::MAX
        } else if uniform(block, 0) {
            0
        } else {
            at += block.len();
            continue;
        };
        draw(bitmap, shaped_from, at, false);
        let end = at + run_len(&coverage[at..], value);
        if value == u8::MAX {
            draw(bitmap, at, end, true);
        }
        at = end;
        shaped_from = end;
    }
    draw(bitmap, shaped_from, coverage.len(), false);
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

    /// Compositing a coverage row by runs gives the pixels compositing it
    /// as one shaped span gives, at any constant alpha.
    #[test]
    fn coverage_rows_composite_as_one_shaped_span() {
        let mut coverage = vec![0u8; 7];
        coverage.extend([40, 255, 255, 90]);
        coverage.extend([255; 37]);
        coverage.extend([0; 20]);
        coverage.extend([128, 0, 255, 3]);
        coverage.extend([0; 16]);
        let n = u32::try_from(coverage.len()).unwrap();
        let x0 = 3;
        for alpha in [255, 128] {
            let pipe = crate::testutil::make_pipe(alpha, crate::types::BlendMode::Normal);
            let colour = [200u8, 30, 90];
            let src = PipeSrc::Solid(&colour);
            let mut runs: Bitmap<Rgb8> = Bitmap::new(n + 8, 1, 1, false);
            let mut whole: Bitmap<Rgb8> = Bitmap::new(n + 8, 1, 1, false);
            draw_coverage_row::<Rgb8>(&mut runs, &pipe, &src, &coverage, x0, 0);
            let x1 = x0 + i32::try_from(n).unwrap() - 1;
            pipe::render_bitmap_span::<Rgb8>(&mut whole, &pipe, &src, Some(&coverage), x0, x1, 0);
            assert_eq!(runs.row(0), whole.row(0), "alpha {alpha}");
        }
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

    /// A subpath that encloses no area fills a line one pixel wide at a
    /// quarter of the coverage, a single line through the centres of the
    /// pixels holding its ends; a stroke's outline with such a subpath
    /// fills nothing there.
    #[test]
    fn aa_fill_of_a_zero_area_subpath_draws_a_line() {
        let mut b = PathBuilder::new();
        b.move_to(2.0, 5.0).unwrap();
        b.line_to(14.0, 5.0).unwrap();
        b.move_to(3.0, 9.0).unwrap();
        b.line_to(13.0, 12.0).unwrap();
        b.line_to(8.0, 10.5).unwrap();
        let path = b.build();
        let out = aa_render(std::slice::from_ref(&path), 16, 16, false);
        let at = |x: usize, y: usize| out[y * 16 + x];
        assert_eq!(
            at(8, 5),
            ZERO_AREA_WEIGHT,
            "the horizontal line covers its row"
        );
        assert_eq!(at(8, 4) | at(8, 6), 0);
        let ink: f64 = out.iter().map(|&v| f64::from(v) / 255.0).sum();
        let diagonal = 10.0f64.hypot(3.0);
        assert!(
            (4.0f64.mul_add(ink, -12.0) - diagonal).abs() < 1.0,
            "ink {ink}"
        );

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 16, 1, false);
        fill_area::<Rgb8>(
            &mut bmp,
            &Clip::new(0.0, 0.0, 16.0, 16.0, true),
            &path,
            &simple_pipe(),
            &PipeSrc::Solid(&[255; 3]),
            &identity_matrix(),
            0.1,
            true,
            false,
        );
        assert!((0..16).all(|y| bmp.row(y).iter().all(|p| p.r == 0)));
    }

    /// A spike, where an outline that encloses an area runs out and back
    /// along one line, fills as a faint line too.
    #[test]
    fn aa_fill_of_a_spike_draws_a_line() {
        let mut b = PathBuilder::new();
        for (k, &(x, y)) in [
            (2.0, 2.0),
            (10.0, 2.0),
            (10.0, 10.0),
            (6.5, 10.0),
            (6.5, 14.0),
            (6.5, 10.0),
            (2.0, 10.0),
        ]
        .iter()
        .enumerate()
        {
            if k == 0 {
                b.move_to(x, y).unwrap();
            } else {
                b.line_to(x, y).unwrap();
            }
        }
        b.close(true).unwrap();
        let out = aa_render(&[b.build()], 16, 16, false);
        let at = |x: usize, y: usize| out[y * 16 + x];
        assert_eq!(at(6, 5), 255, "inside the square");
        assert!((1..255).contains(&at(6, 12)), "the spike: {}", at(6, 12));
        assert_eq!(at(9, 12), 0);
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

    /// An AA fill of a rectangle on pixel boundaries covers its first and
    /// last rows fully and nothing outside them.
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

    /// An AA fill lands where its geometry is, in rows and columns.
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

    /// A fill sitting below a quarter of the page height renders.
    #[test]
    fn aa_fill_low_on_the_page_is_not_dropped() {
        const W: u32 = 32;
        const H: u32 = 128;

        let mut bmp: Bitmap<Rgb8> = Bitmap::new(W, H, 1, false);
        let clip = make_clip(W, H);
        let pipe = simple_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        // Entirely below H / 4 = 32.
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
    fn scanner_produces_spans_for_rect() {
        use crate::scanner::XPathScanner;
        use crate::scanner::iter::ScanIterator;
        use crate::xpath::XPath;

        let path = rect_path(1.0, 1.0, 5.0, 5.0);
        let xpath = XPath::new(&path, &identity_matrix(), 1.0, true);
        let scanner = XPathScanner::new(&xpath, false, 0, 7).expect("small path");
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
