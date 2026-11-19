//! Exact-area coverage rasterisation of glyph outlines.
//!
//! [`rasterize`] turns a glyph [`Path`] (device pixels, y up, origin at the
//! pen position) into a [`GlyphBitmap`].  Curves are flattened to line
//! segments, which the exact-area rasteriser of path fills
//! ([`raster::coverage::AreaRaster`]) converts into each pixel's area inside
//! the outline under the non-zero winding rule.  Mono bitmaps set a pixel
//! when its coverage reaches one half.
//!
//! Every subpath is treated as closed.

use raster::coverage::AreaRaster;
use raster::path::{Path, PathPoint};

use crate::bitmap::GlyphBitmap;

/// Largest glyph bitmap width or height, in pixels.
pub(crate) const MAX_GLYPH_DIM: u32 = 8_192;

/// Largest glyph bitmap area, in pixels; it bounds the coverage bitmap at
/// 16 MiB.
pub(crate) const MAX_GLYPH_AREA: u64 = 1 << 24;

/// Maximum distance, in pixels, between a flattened curve and the curve.
const FLATTEN_TOLERANCE: f64 = 0.05;

/// Upper bound on the line segments one cubic flattens into.
const MAX_CURVE_SEGMENTS: u32 = 256;

/// Rasterise `path` into a coverage bitmap of at most `max_side` pixels a
/// side.
///
/// Returns `None` when the outline is empty, has zero width or height,
/// contains non-finite coordinates, or exceeds `max_side`,
/// [`MAX_GLYPH_DIM`] or [`MAX_GLYPH_AREA`].
#[must_use]
pub(crate) fn rasterize(path: &Path, aa: bool, max_side: u32) -> Option<GlyphBitmap> {
    let segments = flatten(path);
    let bounds = Bounds::of(&segments)?;
    let width = bounds.width()?;
    let height = bounds.height()?;
    let max_side = max_side.min(MAX_GLYPH_DIM);
    if width > max_side
        || height > max_side
        || u64::from(width) * u64::from(height) > MAX_GLYPH_AREA
    {
        log::debug!("font: glyph bitmap {width}×{height} exceeds the rasteriser budget");
        return None;
    }

    // Both sides lie within MAX_GLYPH_DIM, checked above.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "width and height are at most MAX_GLYPH_DIM"
    )]
    let mut raster = AreaRaster::new(0, 0, width as i32 - 1, height as i32 - 1);
    let (left, top) = (bounds.left, bounds.top);
    // Bitmap coordinates: x from the left side, y down from the top.
    let to_bitmap = |p: PathPoint| (p.x - left, top - p.y);
    for &(from, to) in &segments {
        let ((x0, y0), (x1, y1)) = (to_bitmap(from), to_bitmap(to));
        raster.add_line(x0, y0, x1, y1);
    }
    let (w, h) = (width as usize, height as usize);
    let row_bytes = if aa { w } else { w.div_ceil(8) };
    let mut data = vec![0u8; row_bytes * h];
    raster.rows(false, |y, x, coverage| {
        let (Ok(y), Ok(x)) = (usize::try_from(y), usize::try_from(x)) else {
            return;
        };
        let out = &mut data[y * row_bytes..(y + 1) * row_bytes];
        for (px, &c) in (x..w).zip(coverage) {
            if aa {
                out[px] = c;
            } else if c >= 128 {
                out[px / 8] |= 0x80 >> (px % 8);
            }
        }
    });

    // `left` and `top` are integral and bounded by the finite path extent
    // checked in `Bounds::of`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "left/top are integral pixel coordinates within i32 range (checked in Bounds::of)"
    )]
    let (left_i, top_i) = (left as i32, top as i32);
    Some(GlyphBitmap {
        x_off: -left_i,
        y_off: top_i,
        width,
        height,
        aa,
        data,
    })
}

/// Integer pixel bounds of a flattened outline: left/bottom rounded down,
/// right/top rounded up (y up).
struct Bounds {
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
}

impl Bounds {
    fn of(segments: &[(PathPoint, PathPoint)]) -> Option<Self> {
        let mut min = PathPoint::new(f64::INFINITY, f64::INFINITY);
        let mut max = PathPoint::new(f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in segments
            .iter()
            .flat_map(|&pair| <[PathPoint; 2]>::from(pair))
        {
            if !(p.x.is_finite() && p.y.is_finite()) {
                return None;
            }
            min = PathPoint::new(min.x.min(p.x), min.y.min(p.y));
            max = PathPoint::new(max.x.max(p.x), max.y.max(p.y));
        }
        let bounds = Self {
            left: min.x.floor(),
            right: max.x.ceil(),
            bottom: min.y.floor(),
            top: max.y.ceil(),
        };
        let limit = f64::from(i32::MAX / 2);
        (bounds.left >= -limit
            && bounds.right <= limit
            && bounds.bottom >= -limit
            && bounds.top <= limit)
            .then_some(bounds)
    }

    fn width(&self) -> Option<u32> {
        extent(self.right - self.left)
    }

    fn height(&self) -> Option<u32> {
        extent(self.top - self.bottom)
    }
}

/// A positive integral extent as `u32`; `None` for zero or out of range.
fn extent(v: f64) -> Option<u32> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v is integral and checked to lie in [1, u32::MAX]"
    )]
    (v >= 1.0 && v <= f64::from(u32::MAX)).then_some(v as u32)
}

/// Flatten every subpath of `path` into closed polylines.
fn flatten(path: &Path) -> Vec<(PathPoint, PathPoint)> {
    let pts = &path.pts;
    let flags = &path.flags;
    let mut segments = Vec::with_capacity(pts.len() * 2);
    let mut i = 0;
    while i < pts.len() {
        let start = pts[i];
        let mut cur = start;
        i += 1;
        while i < pts.len() && !flags[i].is_first() {
            if flags[i].is_curve() && i + 2 < pts.len() {
                flatten_cubic(cur, pts[i], pts[i + 1], pts[i + 2], &mut segments);
                cur = pts[i + 2];
                i += 3;
            } else {
                segments.push((cur, pts[i]));
                cur = pts[i];
                i += 1;
            }
        }
        if cur != start {
            segments.push((cur, start));
        }
    }
    segments
}

/// Flatten one cubic into uniformly parameterised line segments.  The
/// segment count follows Wang's bound, `n = ⌈√(3·d / (4·tol))⌉` with `d`
/// the largest second difference of the control polygon, which keeps every
/// segment within [`FLATTEN_TOLERANCE`] of the curve.
fn flatten_cubic(
    p0: PathPoint,
    p1: PathPoint,
    p2: PathPoint,
    p3: PathPoint,
    out: &mut Vec<(PathPoint, PathPoint)>,
) {
    let dd = |a: PathPoint, b: PathPoint, c: PathPoint| {
        let x = 2.0f64.mul_add(-b.x, a.x) + c.x;
        let y = 2.0f64.mul_add(-b.y, a.y) + c.y;
        x.hypot(y)
    };
    let d = dd(p0, p1, p2).max(dd(p1, p2, p3));
    let n_f = (0.75 * d / FLATTEN_TOLERANCE).sqrt().ceil();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "n_f is clamped to [1, MAX_CURVE_SEGMENTS] before the cast"
    )]
    let n = if n_f.is_finite() {
        n_f.clamp(1.0, f64::from(MAX_CURVE_SEGMENTS)) as u32
    } else {
        1
    };
    let step = 1.0 / f64::from(n);
    let mut prev = p0;
    for k in 1..=n {
        let t = f64::from(k) * step;
        let p = if k == n {
            p3
        } else {
            cubic_point(p0, p1, p2, p3, t)
        };
        out.push((prev, p));
        prev = p;
    }
}

/// Point at parameter `t` on the cubic Bézier `(p0, p1, p2, p3)`.
fn cubic_point(p0: PathPoint, p1: PathPoint, p2: PathPoint, p3: PathPoint, t: f64) -> PathPoint {
    let u = 1.0 - t;
    let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    PathPoint::new(
        b0.mul_add(p0.x, b1.mul_add(p1.x, b2.mul_add(p2.x, b3 * p3.x))),
        b0.mul_add(p0.y, b1.mul_add(p1.y, b2.mul_add(p2.y, b3 * p3.y))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::path::PathBuilder;

    /// [`super::rasterize`] under the rasteriser's own bounds alone.
    fn rasterize(path: &Path, aa: bool) -> Option<GlyphBitmap> {
        super::rasterize(path, aa, MAX_GLYPH_DIM)
    }

    fn polygon(points: &[(f64, f64)]) -> Path {
        let mut b = PathBuilder::new();
        b.move_to(points[0].0, points[0].1);
        for &(x, y) in &points[1..] {
            b.line_to(x, y).unwrap();
        }
        b.close(false).unwrap();
        b.build()
    }

    fn coverage_sum(bmp: &GlyphBitmap) -> f64 {
        bmp.data.iter().map(|&v| f64::from(v) / 255.0).sum()
    }

    #[test]
    fn pixel_aligned_square_is_fully_covered() {
        let bmp = rasterize(
            &polygon(&[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]),
            true,
        )
        .expect("square rasterises");
        assert_eq!((bmp.width, bmp.height, bmp.x_off, bmp.y_off), (4, 3, 0, 3));
        assert!(bmp.data.iter().all(|&v| v == 255), "{:?}", bmp.data);
    }

    #[test]
    fn winding_direction_does_not_matter() {
        let cw = rasterize(
            &polygon(&[(0.0, 0.0), (0.0, 3.0), (4.0, 3.0), (4.0, 0.0)]),
            true,
        )
        .unwrap();
        let ccw = rasterize(
            &polygon(&[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]),
            true,
        )
        .unwrap();
        assert_eq!(cw.data, ccw.data);
    }

    #[test]
    fn half_pixel_offset_square_has_exact_partial_coverage() {
        // A 2×2 square at (0.5, 0.5): a 3×3 bitmap whose corners are ¼
        // covered, edges ½ and centre fully.
        let bmp = rasterize(
            &polygon(&[(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
            true,
        )
        .unwrap();
        assert_eq!((bmp.width, bmp.height), (3, 3));
        let expect = [64u8, 128, 64, 128, 255, 128, 64, 128, 64];
        for (got, want) in bmp.data.iter().zip(expect) {
            assert!(got.abs_diff(want) <= 1, "{:?}", bmp.data);
        }
    }

    #[test]
    fn triangle_area_is_exact() {
        // Right triangle with legs 7 and 5: area 17.5.
        let bmp = rasterize(&polygon(&[(0.0, 0.0), (7.0, 0.0), (0.0, 5.0)]), true).unwrap();
        assert!(
            (coverage_sum(&bmp) - 17.5).abs() < 0.05,
            "{}",
            coverage_sum(&bmp)
        );
    }

    #[test]
    fn overlapping_same_direction_contours_do_not_double_cover() {
        let mut b = PathBuilder::new();
        for _ in 0..2 {
            b.move_to(0.0, 0.0);
            b.line_to(4.0, 0.0).unwrap();
            b.line_to(4.0, 4.0).unwrap();
            b.line_to(0.0, 4.0).unwrap();
            b.close(false).unwrap();
        }
        let bmp = rasterize(&b.build(), true).unwrap();
        assert!(bmp.data.iter().all(|&v| v == 255));
    }

    #[test]
    fn counter_with_opposite_winding_is_a_hole() {
        let mut b = PathBuilder::new();
        for &(x0, y0, x1, y1, rev) in &[(0.0, 0.0, 6.0, 6.0, false), (2.0, 2.0, 4.0, 4.0, true)] {
            let mut pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
            if rev {
                pts.reverse();
            }
            b.move_to(pts[0].0, pts[0].1);
            for &(x, y) in &pts[1..] {
                b.line_to(x, y).unwrap();
            }
            b.close(false).unwrap();
        }
        let bmp = rasterize(&b.build(), true).unwrap();
        // Rows run top-down; pixel (2..4, 2..4) in y-up is rows 2..4.
        for row in 0..6usize {
            for col in 0..6usize {
                let hole = (2..4).contains(&row) && (2..4).contains(&col);
                let v = bmp.data[row * 6 + col];
                assert_eq!(v, if hole { 0 } else { 255 }, "({col},{row})");
            }
        }
    }

    #[test]
    fn negative_coordinates_set_offsets() {
        let bmp = rasterize(
            &polygon(&[(-2.0, -1.0), (1.0, -1.0), (1.0, 2.0), (-2.0, 2.0)]),
            true,
        )
        .unwrap();
        assert_eq!((bmp.x_off, bmp.y_off, bmp.width, bmp.height), (2, 2, 3, 3));
    }

    #[test]
    fn steep_and_shallow_edges_agree_with_area() {
        // Parallelogram with base 10 and height 4 and slanted sides.
        let bmp = rasterize(
            &polygon(&[(0.0, 0.0), (10.0, 0.0), (13.3, 4.0), (3.3, 4.0)]),
            true,
        )
        .unwrap();
        assert!(
            (coverage_sum(&bmp) - 40.0).abs() < 0.05,
            "{}",
            coverage_sum(&bmp)
        );
    }

    #[test]
    fn curves_are_flattened_within_tolerance() {
        // Circle of radius 20 from four cubic arcs: area π·r².
        let k = 0.552_284_749_831 * 20.0;
        let mut b = PathBuilder::new();
        b.move_to(20.0, 0.0);
        b.curve_to(20.0, k, k, 20.0, 0.0, 20.0).unwrap();
        b.curve_to(-k, 20.0, -20.0, k, -20.0, 0.0).unwrap();
        b.curve_to(-20.0, -k, -k, -20.0, 0.0, -20.0).unwrap();
        b.curve_to(k, -20.0, 20.0, -k, 20.0, 0.0).unwrap();
        b.close(false).unwrap();
        let bmp = rasterize(&b.build(), true).unwrap();
        // Inscribed chords lose at most the tolerance band along the
        // perimeter: 2π·r·tol ≈ 6.3 px² of 1256.6.
        let lost = std::f64::consts::PI.mul_add(400.0, -coverage_sum(&bmp));
        assert!((0.0..6.3).contains(&lost), "lost {lost}");
    }

    #[test]
    fn mono_thresholds_at_half_coverage() {
        let bmp = rasterize(
            &polygon(&[(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
            false,
        )
        .unwrap();
        assert!(!bmp.aa);
        // Only the centre pixel (¼ corners are below, ½ edges are at the
        // threshold) — edges at exactly one half are set.
        assert_eq!(bmp.row_bytes(), 1);
        assert_eq!(bmp.data, vec![0b0100_0000, 0b1110_0000, 0b0100_0000]);
    }

    #[test]
    fn degenerate_and_oversized_outlines_are_rejected() {
        assert!(rasterize(&Path::new(), true).is_none());
        // Zero-height sliver.
        assert!(rasterize(&polygon(&[(0.0, 1.0), (5.0, 1.0), (2.0, 1.0)]), true).is_none());
        let huge = f64::from(MAX_GLYPH_DIM) + 1.0;
        assert!(
            rasterize(
                &polygon(&[(0.0, 0.0), (huge, 0.0), (huge, 1.0), (0.0, 1.0)]),
                true
            )
            .is_none()
        );
        assert!(rasterize(&polygon(&[(0.0, 0.0), (f64::NAN, 0.0), (1.0, 1.0)]), true).is_none());
    }
}
