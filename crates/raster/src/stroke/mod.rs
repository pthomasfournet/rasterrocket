//! Stroke rasterization.
//!
//! Public entry points are [`stroke`] and [`stroke_outline`], and the
//! geometry stages of a stroke, [`stroke_aa_scanner`] and [`hairline_runs`];
//! the helpers (`stroke_narrow`, `stroke_wide`, and the path-shaping
//! functions in the private `path` submodule) are crate-internal
//! implementation details.
//!
//! A stroke is flattened, dashed, then either drawn as a hairline (zero
//! width) or expanded into an outline and filled.

mod path;
use path::{MAX_DASHES, dash_count, dash_cycle, flatten_path, make_dashed_path, make_stroke_path};

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::fill::{aa_scanner, fill};
use crate::path::Path;
use crate::pipe::{PipeSrc, PipeState, render_bitmap_span};
use crate::scanner::XPathScanner;
use crate::types::{LineCap, LineJoin, floor_i32};
use crate::xpath::XPath;
use color::Pixel;

// ── Public API ────────────────────────────────────────────────────────────────

/// Parameters governing how a stroke is rendered.
///
/// The fields are the PDF graphics-state stroke parameters (ISO 32000-2 §8.4).
pub struct StrokeParams<'a> {
    /// Stroke line width in user space.
    pub line_width: f64,
    /// Line cap style (Butt / Round / Projecting).
    pub line_cap: LineCap,
    /// Line join style (Miter / Round / Bevel).
    pub line_join: LineJoin,
    /// Miter limit (dimensionless ratio).
    pub miter_limit: f64,
    /// Flatness for curve flattening (maximum chord deviation, device pixels).
    pub flatness: f64,
    /// Whether stroke adjustment is enabled.
    pub stroke_adjust: bool,
    /// Dash array; empty slice means solid line.
    pub line_dash: &'a [f64],
    /// Phase offset into the dash array.
    pub line_dash_phase: f64,
    /// Whether vector anti-aliasing is requested.
    pub vector_antialias: bool,
}

/// Top-level stroke entry point.
///
/// Steps:
/// 1. Flatten curves.
/// 2. Apply dashing if a dash array is set.
/// 3. Choose hairline (zero line-width) or wide rendering.
pub fn stroke<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
) {
    let Some(path2) = flattened_dashed(path, matrix, params) else {
        return;
    };

    // A zero line width is the thinnest line the device can draw
    // (ISO 32000-2 §8.4.3.2): a one-pixel hairline.
    if params.line_width == 0.0 {
        stroke_narrow::<P>(bitmap, clip, &path2, pipe, src, matrix, params.flatness);
    } else {
        stroke_wide::<P>(bitmap, clip, &path2, pipe, src, matrix, params);
    }
}

/// The region a stroke of `path` with a positive line width covers, as a
/// device-space path to fill with the nonzero winding rule; `None` for a
/// hairline (`line_width == 0`) or a stroke that covers nothing.
#[must_use]
pub fn stroke_outline(path: &Path, matrix: &[f64; 6], params: &StrokeParams<'_>) -> Option<XPath> {
    if params.line_width == 0.0 {
        return None;
    }
    let path = flattened_dashed(path, matrix, params)?;
    let outline = make_stroke_path(&path, params.line_width, params);
    Some(XPath::new(&outline, matrix, params.flatness, true))
}

/// `path` flattened and dashed per `params`; `None` when nothing remains.
fn flattened_dashed(path: &Path, matrix: &[f64; 6], params: &StrokeParams<'_>) -> Option<Path> {
    if path.pts.is_empty() {
        return None;
    }
    let path = flatten_path(path, params.flatness);
    let device_scale = matrix[0]
        .mul_add(matrix[3], -(matrix[1] * matrix[2]))
        .abs()
        .sqrt();
    let Some((dash, phase)) = dash_cycle(params.line_dash, params.line_dash_phase, device_scale)
    else {
        return Some(path);
    };
    let count = dash_count(&path, &dash);
    if count.is_nan() || count > MAX_DASHES {
        return Some(path);
    }
    let dashed = make_dashed_path(&path, &dash, phase);
    (!dashed.pts.is_empty()).then_some(dashed)
}

/// Draw zero-width (hairline) strokes: one pixel per scanline intersection.
///
/// Each segment of the flattened [`XPath`] is walked scanline-by-scanline,
/// drawing a single span (possibly one pixel wide) per row; see
/// [`narrow_runs`].
fn stroke_narrow<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    flatness: f64,
) {
    let (width, height) = (bitmap.width, bitmap.height);
    narrow_runs(clip, path, matrix, flatness, width, height, |x0, x1, y| {
        render_bitmap_span::<P>(bitmap, pipe, src, None, x0, x1, y);
    });
}

/// The pixel runs a hairline stroke of `path` paints on a `width × height`
/// bitmap under `clip`, in the order [`stroke`] paints them, as
/// `emit(x0, x1, y)` with `0 ≤ x0 ≤ x1 < width` and `0 ≤ y < height`.
///
/// Each run is composited with full coverage (no shape).  Runs of different
/// segments may overlap; each is composited on its own.  Nothing is emitted
/// for a positive line width.
pub fn hairline_runs(
    path: &Path,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
    clip: &Clip,
    width: u32,
    height: u32,
    emit: impl FnMut(i32, i32, i32),
) {
    if params.line_width != 0.0 {
        return;
    }
    let Some(path) = flattened_dashed(path, matrix, params) else {
        return;
    };
    narrow_runs(clip, &path, matrix, params.flatness, width, height, emit);
}

/// The AA-space scanner of a stroke of `path` with a positive line width.
///
/// It is the geometry stage ([`crate::fill::aa_scanner`]) of the non-zero
/// antialiased fill of the outline [`stroke`] paints.  `None` for a
/// hairline (`line_width == 0`) and for a stroke that covers no row.
#[must_use]
pub fn stroke_aa_scanner(
    path: &Path,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
    clip: &Clip,
) -> Option<XPathScanner> {
    if params.line_width == 0.0 {
        return None;
    }
    let path = flattened_dashed(path, matrix, params)?;
    let outline = make_stroke_path(&path, params.line_width, params);
    aa_scanner(&outline, matrix, params.flatness, clip, false)
}

/// Walk the segments of the flattened `path` scanline by scanline and emit
/// the pixel runs of a hairline (see [`narrow_span_runs`]).
fn narrow_runs(
    clip: &Clip,
    path: &Path,
    matrix: &[f64; 6],
    flatness: f64,
    width: u32,
    height: u32,
    mut emit: impl FnMut(i32, i32, i32),
) {
    // Build a non-closing XPath (no implicit close for hairlines).
    let xpath = XPath::new(path, matrix, flatness, false);
    let mut span = |x0: i32, x1: i32, y: i32, clip_res: ClipResult| {
        narrow_span_runs(clip, x0, x1, y, clip_res, width, height, &mut emit);
    };

    for seg in &xpath.segs {
        // Orient so that y0 <= y1.
        let (sx0, sy0, sx1, sy1) = if seg.y0 <= seg.y1 {
            (seg.x0, seg.y0, seg.x1, seg.y1)
        } else {
            (seg.x1, seg.y1, seg.x0, seg.y0)
        };

        let y0 = floor_i32(sy0);
        let y1 = floor_i32(sy1);
        let x0 = floor_i32(sx0);
        let x1 = floor_i32(sx1);

        let (xl, xr) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let clip_res = clip.test_rect(xl, y0, xr, y1);
        if clip_res == ClipResult::AllOutside {
            continue;
        }

        if y0 == y1 {
            // Horizontal or near-horizontal: draw one span.
            let (span_x0, span_x1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
            span(span_x0, span_x1, y0, clip_res);
        } else {
            // Sloped segment: walk scanlines.
            let dxdy = seg.dxdy;

            // The clip selects rows only: a first row below `y0` starts
            // where the segment crosses its top edge, every row above `y1`
            // ends where the segment crosses its bottom edge, and the
            // direction is that of the whole segment, all as in the
            // unclipped walk, so a drawn row has the unclipped row's pixels.
            let cy0 = y0.max(clip.y_min_i);
            let cy1 = y1.min(clip.y_max_i);
            let mut xa = if cy0 > y0 {
                floor_i32((f64::from(cy0) - sy0).mul_add(dxdy, sx0))
            } else {
                x0
            };
            let left_to_right = x0 <= x1;
            for y in cy0..=cy1 {
                let xb = if y < y1 {
                    floor_i32((f64::from(y) + 1.0 - sy0).mul_add(dxdy, sx0))
                } else if left_to_right {
                    x1 + 1
                } else {
                    x1 - 1
                };
                let (span_x0, span_x1) = if left_to_right {
                    if xa == xb { (xa, xa) } else { (xa, xb - 1) }
                } else if xa == xb {
                    (xa, xa)
                } else {
                    (xb + 1, xa)
                };
                span(span_x0, span_x1, y, clip_res);
                xa = xb;
            }
        }
    }
}

/// Wide stroke: expand the path into a filled outline and fill it.
fn stroke_wide<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    path: &Path,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    matrix: &[f64; 6],
    params: &StrokeParams<'_>,
) {
    let outline = make_stroke_path(path, params.line_width, params);
    fill::<P>(
        bitmap,
        clip,
        &outline,
        pipe,
        src,
        matrix,
        params.flatness,
        params.vector_antialias,
    );
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Emit the runs of the span `[x0, x1]` on row `y` (possibly a single
/// pixel) that lie on a `width × height` bitmap and inside the clip.
///
/// Used by [`narrow_runs`] for each hairline pixel-run.
#[expect(
    clippy::too_many_arguments,
    reason = "all parameters are required; splitting would add indirection"
)]
fn narrow_span_runs(
    clip: &Clip,
    x0: i32,
    x1: i32,
    y: i32,
    clip_res: ClipResult,
    width: u32,
    height: u32,
    emit: &mut impl FnMut(i32, i32, i32),
) {
    if y < 0 {
        return;
    }
    #[expect(clippy::cast_sign_loss, reason = "y >= 0 checked above")]
    if (y as u32) >= height {
        return;
    }
    // Clamp x to bitmap.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap width fits in i32 in practice"
    )]
    let width_i = width as i32;

    // The span is clamped to the bitmap in both branches: a clip rectangle
    // need not lie within it.
    let (sx0, sx1) = if clip_res == ClipResult::AllInside {
        (x0.max(0), x1.min(width_i - 1))
    } else {
        (
            x0.max(clip.x_min_i).max(0),
            x1.min(clip.x_max_i).min(width_i - 1),
        )
    };
    if sx0 > sx1 {
        return;
    }
    if clip_res == ClipResult::AllInside {
        emit(sx0, sx1, y);
    } else {
        for (rx0, rx1) in clip.inside_runs(sx0, sx1, y) {
            emit(rx0, rx1, y);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::path::PathBuilder;
    use crate::pipe::PipeSrc;
    use crate::testutil::{identity_matrix, make_clip, simple_pipe};
    use color::Rgb8;

    fn default_params<'a>() -> StrokeParams<'a> {
        StrokeParams {
            line_width: 0.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
            miter_limit: 10.0,
            flatness: 1.0,
            stroke_adjust: false,
            line_dash: &[],
            line_dash_phase: 0.0,
            vector_antialias: false,
        }
    }

    /// `stroke_narrow` should paint pixels along a diagonal line segment.
    #[test]
    fn stroke_narrow_draws_diagonal() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 16, 4, false);
        let clip = make_clip(16, 16);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // Build a diagonal path from (1,1) to (8,8).
        let mut b = PathBuilder::new();
        b.move_to(1.0, 1.0).unwrap();
        b.line_to(8.0, 8.0).unwrap();
        let path = b.build();

        // Flatten (it's already flat) and draw hairline.
        let flat = flatten_path(&path, 1.0);
        stroke_narrow::<Rgb8>(&mut bmp, &clip, &flat, &pipe, &src, &identity_matrix(), 1.0);

        // At least one pixel on the diagonal should be painted.
        let mut any_painted = false;
        for i in 1u32..9 {
            if bmp.row(i)[i as usize].r == 255 {
                any_painted = true;
                break;
            }
        }
        assert!(
            any_painted,
            "stroke_narrow should paint at least one diagonal pixel"
        );
    }

    /// A hairline cut short by the clip's bottom edge draws the rows it
    /// keeps as the unclipped hairline draws them.
    #[test]
    fn stroke_narrow_rows_are_unchanged_by_a_bottom_clip() {
        let draw = |clip_height: u32| {
            let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 16, 4, false);
            let clip = make_clip(16, clip_height);
            let pipe = simple_pipe();
            let color = [255u8, 0, 0];
            let src = PipeSrc::Solid(&color);
            for (x1, y1) in [(15.3, 6.6), (0.4, 7.9)] {
                let mut b = PathBuilder::new();
                b.move_to(4.2, 0.5).unwrap();
                b.line_to(x1, y1).unwrap();
                let flat = flatten_path(&b.build(), 1.0);
                stroke_narrow::<Rgb8>(&mut bmp, &clip, &flat, &pipe, &src, &identity_matrix(), 1.0);
            }
            (0..clip_height)
                .map(|y| bmp.row(y).iter().map(|p| p.r).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let whole = draw(16);
        for height in 1..8 {
            assert_eq!(
                draw(height),
                whole[..height as usize],
                "clip height {height}"
            );
        }
    }

    /// A hairline under a clip whose top or bottom edge lies inside a pixel
    /// row draws each kept row as the unclipped hairline draws it, for a
    /// steep leftward segment under the bottom edge and a shallow rightward
    /// one under the top edge.
    #[test]
    fn stroke_narrow_rows_are_unchanged_by_fractional_clip_edges() {
        let draw = |from: (f64, f64), to: (f64, f64), clip: &Clip| {
            let mut bmp: Bitmap<Rgb8> = Bitmap::new(24, 16, 4, false);
            let color = [255u8, 0, 0];
            let mut b = PathBuilder::new();
            b.move_to(from.0, from.1).unwrap();
            b.line_to(to.0, to.1).unwrap();
            let flat = flatten_path(&b.build(), 1.0);
            let src = PipeSrc::Solid(&color);
            stroke_narrow::<Rgb8>(
                &mut bmp,
                clip,
                &flat,
                &simple_pipe(),
                &src,
                &identity_matrix(),
                1.0,
            );
            (0..16)
                .map(|y| bmp.row(y).iter().map(|p| p.r).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let cases = [
            (
                (5.9, 0.0),
                (2.6, 10.0),
                Clip::new(0.0, 0.0, 24.0, 2.5, false),
            ),
            (
                (0.0, 0.0),
                (20.0, 10.0),
                Clip::new(0.0, 2.5, 24.0, 16.0, false),
            ),
        ];
        for (from, to, clip) in cases {
            let whole = draw(from, to, &make_clip(24, 16));
            let clipped = draw(from, to, &clip);
            for y in 0..16 {
                let kept = (clip.y_min_i..=clip.y_max_i).contains(&y);
                let row = usize::try_from(y).unwrap();
                let want = if kept {
                    whole[row].clone()
                } else {
                    vec![0; 24]
                };
                assert_eq!(clipped[row], want, "{from:?} to {to:?}, row {y}");
            }
        }
    }

    /// `make_stroke_path` must return a non-empty outline for a non-degenerate
    /// segment with positive line width.
    #[test]
    fn make_stroke_path_non_degenerate() {
        // A simple horizontal segment.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(10.0, 0.0).unwrap();
        let path = b.build();

        let params = StrokeParams {
            line_width: 2.0,
            ..default_params()
        };
        let outline = make_stroke_path(&path, 2.0, &params);
        assert!(
            !outline.pts.is_empty(),
            "make_stroke_path should return a non-empty path for a non-degenerate segment"
        );
        // The outline should have at least 4 points (a stroke rectangle).
        assert!(
            outline.pts.len() >= 4,
            "stroke outline should have at least 4 points, got {}",
            outline.pts.len()
        );
    }

    /// `make_dashed_path` must respect the dash array: segments alternate on/off.
    #[test]
    fn make_dashed_path_respects_dash_array() {
        // A horizontal line of length 20, dash array [4, 2] → on 4, off 2, ...
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(20.0, 0.0).unwrap();
        let path = b.build();

        let dash = [4.0_f64, 2.0];
        let dashed = make_dashed_path(&path, &dash, 0.0);

        // Result should be non-empty.
        assert!(
            !dashed.pts.is_empty(),
            "dashed path should not be empty for a long segment"
        );

        // All point x coordinates must be in [0, 20].
        for pt in &dashed.pts {
            assert!(
                pt.x >= -1e-9 && pt.x <= 20.0 + 1e-9,
                "dashed point x={} out of [0, 20]",
                pt.x
            );
        }

        // Should have multiple subpaths (at least 2 FIRST-flagged points).
        let first_count = dashed.flags.iter().filter(|f| f.is_first()).count();
        assert!(
            first_count >= 2,
            "dashed path should have at least 2 subpaths (on segments), got {first_count}"
        );
    }

    /// A dash cycle of zero length, or one shorter than a tenth of a device
    /// pixel, strokes solid.
    #[test]
    fn degenerate_dash_cycles_stroke_solid() {
        assert_eq!(dash_cycle(&[0.0], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[0.0, 0.0], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[1e-6], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[0.04, 0.04], 0.0, 1.0), None);
        assert!(dash_cycle(&[0.04, 0.04], 0.0, 4.0).is_some());
    }

    /// Non-finite or negative entries and a non-finite phase stroke solid.
    #[test]
    fn invalid_dash_entries_stroke_solid() {
        assert_eq!(dash_cycle(&[f64::NAN], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[f64::INFINITY, 2.0], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[3.0, -1.0], 0.0, 1.0), None);
        assert_eq!(dash_cycle(&[6.0], f64::INFINITY, 1.0), None);
        assert_eq!(dash_cycle(&[], 0.0, 1.0), None);
    }

    /// An odd-length array repeats with period twice its sum, and the phase
    /// is reduced into that period whatever its sign.
    #[test]
    fn dash_phase_reduces_modulo_the_full_cycle() {
        assert_eq!(dash_cycle(&[6.0], 7.0, 1.0), Some((vec![6.0, 6.0], 7.0)));
        assert_eq!(dash_cycle(&[6.0], -3.0, 1.0), Some((vec![6.0, 6.0], 9.0)));
        assert_eq!(
            dash_cycle(&[5.0, 2.0, 1.0], 0.0, 1.0),
            Some((vec![5.0, 2.0, 1.0, 5.0, 2.0, 1.0], 0.0))
        );
        assert_eq!(
            dash_cycle(&[3.0, 2.0], -4.0, 1.0),
            Some((vec![3.0, 2.0], 1.0))
        );
        let (_, phase) = dash_cycle(&[3.0, 2.0], 1e300, 1.0).expect("finite phase");
        assert!((0.0..5.0).contains(&phase), "phase {phase}");
    }

    /// `[6] 7 d` starts one unit into the off half of a 12-unit cycle, so the
    /// first dash runs from 5 to 11.
    #[test]
    fn odd_dash_array_with_phase_starts_off() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(20.0, 0.0).unwrap();
        let path = b.build();

        let (dash, phase) = dash_cycle(&[6.0], 7.0, 1.0).expect("dashed");
        let dashed = make_dashed_path(&path, &dash, phase);
        let xs: Vec<f64> = dashed.pts.iter().map(|p| p.x).collect();
        assert_eq!(xs, [5.0, 11.0, 17.0, 20.0]);
    }

    /// A path far longer than its dash cycle strokes solid instead of being
    /// cut into more than `MAX_DASHES` dashes.
    #[test]
    fn dash_count_beyond_the_budget_strokes_solid() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(1e9, 0.0).unwrap();
        let path = b.build();
        let params = StrokeParams {
            line_width: 1.0,
            line_dash: &[0.5, 0.5],
            ..default_params()
        };
        let stroked = flattened_dashed(&path, &identity_matrix(), &params).expect("solid");
        assert_eq!(stroked.pts.len(), 2, "the solid line itself");

        // Within the budget the same pattern still dashes.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.line_to(4.0, 0.0).unwrap();
        let short = flattened_dashed(&b.build(), &identity_matrix(), &params).expect("dashed");
        assert_eq!(short.pts.len(), 8, "four dashes of two points");
    }

    /// `flatten_path` must convert a curve to only straight segments (no CURVE flags).
    #[test]
    fn flatten_path_removes_curves() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0).unwrap();
        b.curve_to(1.0, 2.0, 3.0, 4.0, 4.0, 0.0).unwrap();
        let path = b.build();

        // The original path has CURVE flags.
        assert!(path.flags.iter().any(|f| f.is_curve()));

        let flat = flatten_path(&path, 1.0);
        // The flattened path must have no CURVE flags.
        assert!(
            flat.flags.iter().all(|f| !f.is_curve()),
            "flatten_path must remove all CURVE flags"
        );
        // The flattened path should have more than 2 points (the curve was subdivided).
        assert!(
            flat.pts.len() >= 2,
            "flattened curve should have at least 2 points"
        );
    }
}
