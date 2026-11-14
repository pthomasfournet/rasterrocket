//! Gouraud-shaded triangle fill.
//!
//! Rasterises a triangle with per-vertex colours by scanline sweep.  Each
//! scanline linearly interpolates colours from the left edge to the right edge,
//! then calls the compositing pipeline for the span.
//!
//! # Algorithm
//!
//! 1. Sort the three vertices by Y (insertion sort).
//! 2. Build `EdgeMap`s: `x` and per-channel colour as linear functions of Y.
//! 3. Sweep scanlines Y ∈ [y0, y2]; switch the active short edge at `y == y1`.
//! 4. Interpolate colour across each scanline and call `render_span`.
//!
//! # Colour space
//!
//! Vertex colours are `N` bytes, interpolated channel by channel: sRGB by
//! default.  [`gouraud_triangle_fill`] composites RGB vertices onto an
//! [`Rgb8`] bitmap; [`gouraud_triangle_spans`] yields the interpolated
//! spans of vertices of any width for a caller painting another pixel type.

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult};
use crate::pipe::{PipeSrc, PipeState, RowPattern, render_bitmap_span};
use color::Rgb8;
use color::convert::{lerp_u8, round_half_up_i32};

/// One vertex of a Gouraud-shaded triangle, of an `N`-byte colour.
#[derive(Copy, Clone, Debug)]
pub struct GouraudVertex<const N: usize = 3> {
    /// Device-space X coordinate (sub-pixel; rounded to nearest pixel).
    pub x: f64,
    /// Device-space Y coordinate.
    pub y: f64,
    /// The colour: sRGB bytes for `N = 3`, or the bytes of another pixel
    /// type.
    pub color: [u8; N],
}

/// Linear interpolation of a triangle edge as a function of scanline Y.
///
/// Both x and per-channel colour are expressed as `slope * y + offset`.
struct EdgeMap<const N: usize> {
    x_slope: f64,
    x_off: f64,
    c_slope: [f64; N],
    c_off: [f64; N],
}

impl<const N: usize> EdgeMap<N> {
    fn from_vertices(va: &GouraudVertex<N>, vb: &GouraudVertex<N>) -> Self {
        let ya = f64::from(round_half_up_i32(va.y));
        let yb = f64::from(round_half_up_i32(vb.y));
        let dy = yb - ya;
        if dy.abs() < 1e-9 {
            // Horizontal edge — slope undefined; hold start-vertex values.
            return Self {
                x_slope: 0.0,
                x_off: va.x,
                c_slope: [0.0; N],
                c_off: va.color.map(f64::from),
            };
        }
        let x_slope = (vb.x - va.x) / dy;
        let x_off = ya.mul_add(-x_slope, va.x);
        let c_slope: [f64; N] =
            std::array::from_fn(|ch| (f64::from(vb.color[ch]) - f64::from(va.color[ch])) / dy);
        let c_off: [f64; N] =
            std::array::from_fn(|ch| ya.mul_add(-c_slope[ch], f64::from(va.color[ch])));
        Self {
            x_slope,
            x_off,
            c_slope,
            c_off,
        }
    }

    /// Evaluate the edge at scanline `y` → (pixel x, interpolated colour).
    fn eval(&self, y: f64) -> (i32, [u8; N]) {
        let x = round_half_up_i32(self.x_slope.mul_add(y, self.x_off));
        let c: [u8; N] = std::array::from_fn(|ch| {
            #[expect(clippy::cast_possible_truncation, reason = "value clamped to 0..=255")]
            #[expect(clippy::cast_sign_loss, reason = "value clamped to 0.0..=255.0")]
            {
                self.c_slope[ch]
                    .mul_add(y, self.c_off[ch])
                    .clamp(0.0, 255.0) as u8
            }
        });
        (x, c)
    }
}

/// Pass the run `[run_x0, run_x1]` of `span_color`, of `n`-byte colours,
/// to `emit`.
///
/// `buf_x0` is the leftmost x whose colour is stored at `span_color[0]`.
/// `run_x0 >= buf_x0` must hold; violated only by a logic error in the caller.
fn emit_run(
    n: usize,
    emit: &mut impl FnMut(&[u8], i32, i32, i32),
    span_color: &[u8],
    buf_x0: i32,
    run_x0: i32,
    run_x1: i32,
    y: i32,
) {
    debug_assert!(
        run_x0 >= buf_x0,
        "emit_run: run_x0={run_x0} < buf_x0={buf_x0}"
    );
    debug_assert!(
        run_x0 <= run_x1,
        "emit_run: empty run x0={run_x0} x1={run_x1}"
    );
    debug_assert!(y >= 0, "emit_run: negative y={y}");

    #[expect(
        clippy::cast_sign_loss,
        reason = "run_x0 >= buf_x0 >= 0 (both clamped to 0 upstream)"
    )]
    let buf_off = (run_x0 - buf_x0) as usize * n;
    #[expect(clippy::cast_sign_loss, reason = "run_x1 >= run_x0")]
    let run_len = (run_x1 - run_x0 + 1) as usize;
    let buf_end = buf_off + run_len * n;
    assert!(
        buf_end <= span_color.len(),
        "emit_run: buf_end={buf_end} exceeds span_color.len()={} \
         (run_x0={run_x0} run_x1={run_x1} buf_x0={buf_x0})",
        span_color.len(),
    );
    emit(&span_color[buf_off..buf_end], run_x0, run_x1, y);
}

/// Fill a single Gouraud-shaded triangle into `bitmap`.
///
/// Pixels are clipped to `clip` and composited via `pipe`.
pub fn gouraud_triangle_fill(
    bitmap: &mut Bitmap<Rgb8>,
    origin: (i32, i32),
    clip: &Clip,
    pipe: &PipeState<'_>,
    v: [GouraudVertex; 3],
) {
    let (width, height) = (bitmap.width, bitmap.height);
    gouraud_triangle_spans(width, height, origin, clip, v, |colour, x0, x1, y| {
        render_bitmap_span::<Rgb8>(
            bitmap,
            pipe,
            &PipeSrc::Pattern(&RowPattern { data: colour }),
            None,
            x0,
            x1,
            y,
        );
    });
}

/// The spans of a single Gouraud-shaded triangle, clipped to `clip`.
///
/// The vertices lie on a pixel grid of which the `width × height` bitmap
/// is the window whose pixel `(0, 0)` is the grid's pixel `origin`; the
/// spans are computed on that grid and moved onto the bitmap last, so a
/// window receives the spans of the whole grid it covers.  `clip` is in
/// the bitmap's pixels.  `emit(colour, x0, x1, y)` receives the colour
/// bytes of the bitmap pixels `x0..=x1` of row `y`, `N` per pixel, in
/// scanline order.  [`gouraud_triangle_fill`] composites exactly these
/// spans.
pub fn gouraud_triangle_spans<const N: usize>(
    width: u32,
    height: u32,
    origin: (i32, i32),
    clip: &Clip,
    mut v: [GouraudVertex<N>; 3],
    mut emit: impl FnMut(&[u8], i32, i32, i32),
) {
    // A non-finite coordinate has no pixel position: the triangle is empty.
    if !v
        .iter()
        .all(|vert| vert.x.is_finite() && vert.y.is_finite())
    {
        return;
    }

    // --- Sort vertices by Y (insertion sort, 3 elements) ---
    if v[0].y > v[1].y {
        v.swap(0, 1);
    }
    if v[1].y > v[2].y {
        v.swap(1, 2);
        if v[0].y > v[1].y {
            v.swap(0, 1);
        }
    }
    // v[0].y ≤ v[1].y ≤ v[2].y.

    let [y0, y1, y2] = v.map(|vert| round_half_up_i32(vert.y));

    if y0 == y2 {
        return;
    } // all three vertices on the same scanline — degenerate

    let vx = v.map(|vert| round_half_up_i32(vert.x));
    let x_min = vx[0].min(vx[1]).min(vx[2]);
    let x_max = vx[0].max(vx[1]).max(vx[2]);
    let (ox, oy) = origin;
    // Single test_rect call; result used for both early-exit and per-scanline clip decisions.
    let clip_bbox = clip.test_rect(
        x_min.saturating_sub(ox),
        y0.saturating_sub(oy),
        x_max.saturating_sub(ox),
        y2.saturating_sub(oy),
    );
    if clip_bbox == ClipResult::AllOutside {
        return;
    }

    let long_edge = EdgeMap::from_vertices(&v[0], &v[2]);
    let upper_edge = EdgeMap::from_vertices(&v[0], &v[1]);
    let lower_edge = EdgeMap::from_vertices(&v[1], &v[2]);

    // Determine which side the long edge occupies.
    // When y0 == y1 (flat top), upper_edge has zero length — sample lower_edge
    // at y1+1 instead.  saturating_add handles the pathological y1 == i32::MAX
    // case by clamping the sample to i32::MAX; at that coordinate the bitmap
    // scan range (scan_y0..=scan_y2) is already empty, so the wrong edge
    // assignment never affects any painted pixel.
    let long_is_left = if y0 == y1 {
        let sample_y = y1.saturating_add(1);
        let (xl, _) = long_edge.eval(f64::from(sample_y));
        let (xs, _) = lower_edge.eval(f64::from(sample_y));
        xl <= xs
    } else {
        let (xl, _) = long_edge.eval(f64::from(y1));
        let (xs, _) = upper_edge.eval(f64::from(y1));
        xl <= xs
    };

    // The rows and columns, on the vertices' grid, inside both the clip
    // rectangle and the bitmap.  Columns and rows beyond `i32::MAX` lie
    // outside the `i32` scan range, so clamping the dimensions leaves the
    // painted pixels unchanged.
    let bmp_h = i32::try_from(height).unwrap_or(i32::MAX);
    let bmp_w = i32::try_from(width).unwrap_or(i32::MAX);
    let col_min = clip.x_min_i.max(0).saturating_add(ox);
    let col_max = clip.x_max_i.min(bmp_w - 1).saturating_add(ox);
    let scan_y0 = y0.max(clip.y_min_i.max(0).saturating_add(oy));
    let scan_y2 = y2.min(clip.y_max_i.min(bmp_h - 1).saturating_add(oy));

    // Span colour buffer, sized to the widest span the loop can emit.
    let max_span = usize::try_from(i64::from(col_max) - i64::from(col_min) + 1).unwrap_or(0);
    let mut span_color = vec![0u8; max_span * N];

    for y in scan_y0..=scan_y2 {
        let yf = f64::from(y);

        let short_edge = if y <= y1 { &upper_edge } else { &lower_edge };
        let (xl, cl) = long_edge.eval(yf);
        let (xs, cs) = short_edge.eval(yf);

        let (x_left, c_left, x_right, c_right) = if long_is_left {
            (xl, cl, xs, cs)
        } else {
            (xs, cs, xl, cl)
        };

        let sx0 = x_left.max(col_min);
        let sx1 = x_right.min(col_max);
        if sx0 > sx1 {
            continue;
        }

        #[expect(clippy::cast_sign_loss, reason = "sx1 >= sx0 (guarded above)")]
        let span_len = (sx1 - sx0 + 1) as usize;
        debug_assert!(
            span_len <= max_span,
            "span_len {span_len} exceeds pre-allocated max_span {max_span}"
        );
        let buf = &mut span_color[..span_len * N];

        let span_width = (f64::from(x_right) - f64::from(x_left)).max(1.0);
        for (px, out) in (sx0..=sx1).zip(buf.as_chunks_mut::<N>().0) {
            let t = ((f64::from(px) - f64::from(x_left)) / span_width).clamp(0.0, 1.0);
            #[expect(clippy::cast_sign_loss, reason = "t ∈ [0,1]")]
            #[expect(clippy::cast_possible_truncation, reason = "t * 256 ≤ 256")]
            let frac = (t * 256.0) as u32;
            for ((o, l), r) in out.iter_mut().zip(c_left).zip(c_right) {
                *o = lerp_u8(l, r, frac);
            }
        }

        // The span on the bitmap: within it, as the scan range is.
        let (bx0, bx1, by) = (sx0 - ox, sx1 - ox, y - oy);
        // Use per-pixel clip test only when the bbox test was inconclusive AND
        // the current span is not fully inside the clip.
        let needs_per_pixel_clip = clip_bbox != ClipResult::AllInside
            && clip.test_span(bx0, bx1, by) != ClipResult::AllInside;

        if needs_per_pixel_clip {
            for (rx0, rx1) in clip.inside_runs(bx0, bx1, by) {
                emit_run(N, &mut emit, &span_color, bx0, rx0, rx1, by);
            }
        } else {
            emit_run(N, &mut emit, &span_color, bx0, bx0, bx1, by);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::testutil::{make_clip, simple_pipe};

    fn vertex(x: f64, y: f64) -> GouraudVertex {
        GouraudVertex {
            x,
            y,
            color: [0; 3],
        }
    }

    /// Positions round half toward +∞ on both sides of the origin: `-0.7`
    /// lies nearest pixel `-1`, not `0`.
    #[test]
    fn negative_coordinates_round_toward_negative_infinity() {
        let vertical = EdgeMap::from_vertices(&vertex(-0.7, 0.0), &vertex(-0.7, 10.0));
        assert_eq!(vertical.eval(5.0).0, -1, "x = -0.7 rounds to -1");
        // The edge starts at row round(-0.7) = -1, so it spans 11 rows and
        // advances one pixel per row.
        let sloped = EdgeMap::from_vertices(&vertex(0.0, -0.7), &vertex(11.0, 10.0));
        assert_eq!(sloped.eval(4.0).0, 5, "five rows below the start row");
    }

    /// A vertex with a non-finite coordinate paints nothing.
    #[test]
    fn non_finite_vertex_paints_nothing() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 1, false);
        let mut tri = [vertex(1.0, 1.0), vertex(6.0, 1.0), vertex(1.0, 6.0)];
        for v in &mut tri {
            v.color = [255; 3];
        }
        tri[2].x = f64::INFINITY;
        gouraud_triangle_fill(&mut bmp, (0, 0), &make_clip(8, 8), &simple_pipe(), tri);
        assert!(bmp.data().iter().all(|&b| b == 0), "nothing painted");
    }

    /// All-white vertices: every painted pixel must be white.
    #[test]
    fn flat_white_triangle_paints_white() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, true);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex {
                x: 2.0,
                y: 2.0,
                color: [255, 255, 255],
            },
            GouraudVertex {
                x: 8.0,
                y: 2.0,
                color: [255, 255, 255],
            },
            GouraudVertex {
                x: 5.0,
                y: 7.0,
                color: [255, 255, 255],
            },
        ];
        gouraud_triangle_fill(&mut bmp, (0, 0), &clip, &pipe, tri);

        let row = bmp.row(3);
        assert_eq!(row[5].r, 255, "centroid R");
        assert_eq!(row[5].g, 255, "centroid G");
        assert_eq!(row[5].b, 255, "centroid B");
    }

    /// Centroid of a red-green-blue triangle ≈ average of vertex colours (~85 each).
    #[test]
    fn centroid_approximates_average_of_vertex_colors() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(12, 12, 4, true);
        let clip = make_clip(12, 12);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex {
                x: 1.0,
                y: 1.0,
                color: [255, 0, 0],
            },
            GouraudVertex {
                x: 10.0,
                y: 1.0,
                color: [0, 255, 0],
            },
            GouraudVertex {
                x: 5.5,
                y: 9.0,
                color: [0, 0, 255],
            },
        ];
        gouraud_triangle_fill(&mut bmp, (0, 0), &clip, &pipe, tri);

        let px = bmp.row(3)[5];
        assert!(px.r > 50 && px.r < 150, "centroid R={} expected ~85", px.r);
        assert!(px.g > 50 && px.g < 150, "centroid G={} expected ~85", px.g);
        assert!(px.b > 50 && px.b < 150, "centroid B={} expected ~85", px.b);
    }

    /// All vertices on the same scanline — degenerate, must not paint.
    #[test]
    fn degenerate_triangle_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex {
                x: 1.0,
                y: 4.0,
                color: [255, 0, 0],
            },
            GouraudVertex {
                x: 5.0,
                y: 4.0,
                color: [0, 255, 0],
            },
            GouraudVertex {
                x: 3.0,
                y: 4.0,
                color: [0, 0, 255],
            },
        ];
        gouraud_triangle_fill(&mut bmp, (0, 0), &clip, &pipe, tri);

        for y in 0..8u32 {
            for x in 0..8usize {
                assert_eq!(bmp.row(y)[x].r, 0, "y={y} x={x} should be zero");
            }
        }
    }

    /// Triangle entirely outside clip must not paint.
    #[test]
    fn triangle_outside_clip_is_noop() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(0.0, 0.0, 3.999, 3.999, false); // top-left 4×4 only
        let pipe = simple_pipe();

        let tri = [
            GouraudVertex {
                x: 5.0,
                y: 5.0,
                color: [255, 0, 0],
            },
            GouraudVertex {
                x: 7.0,
                y: 5.0,
                color: [0, 255, 0],
            },
            GouraudVertex {
                x: 6.0,
                y: 7.0,
                color: [0, 0, 255],
            },
        ];
        gouraud_triangle_fill(&mut bmp, (0, 0), &clip, &pipe, tri);

        for y in 0..8u32 {
            for x in 0..8usize {
                assert_eq!(bmp.row(y)[x].r, 0, "y={y} x={x} should be zero");
            }
        }
    }

    /// Both vertex orderings must produce a painted interior.
    #[test]
    fn vertex_order_both_paint_interior() {
        let tri_abc = [
            GouraudVertex {
                x: 2.0,
                y: 2.0,
                color: [255, 0, 0],
            },
            GouraudVertex {
                x: 8.0,
                y: 2.0,
                color: [0, 255, 0],
            },
            GouraudVertex {
                x: 5.0,
                y: 7.0,
                color: [0, 0, 255],
            },
        ];
        let tri_cba = [tri_abc[2], tri_abc[1], tri_abc[0]];

        let mut bmp1: Bitmap<Rgb8> = Bitmap::new(12, 10, 4, true);
        let mut bmp2: Bitmap<Rgb8> = Bitmap::new(12, 10, 4, true);
        let clip = make_clip(12, 10);
        let pipe = simple_pipe();

        gouraud_triangle_fill(&mut bmp1, (0, 0), &clip, &pipe, tri_abc);
        gouraud_triangle_fill(&mut bmp2, (0, 0), &clip, &pipe, tri_cba);

        let interior_painted = |bmp: &Bitmap<Rgb8>| {
            (3..7u32).any(|y| {
                (3..8usize).any(|x| {
                    let p = bmp.row(y)[x];
                    p.r > 0 || p.g > 0 || p.b > 0
                })
            })
        };
        assert!(interior_painted(&bmp1), "ABC order: interior not painted");
        assert!(interior_painted(&bmp2), "CBA order: interior not painted");
    }

    /// Flat-bottom triangle (y1 == y2) must paint correctly.
    #[test]
    fn flat_bottom_triangle_paints() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(10, 10, 4, true);
        let clip = make_clip(10, 10);
        let pipe = simple_pipe();

        // Apex at top, flat base at bottom.
        let tri = [
            GouraudVertex {
                x: 5.0,
                y: 1.0,
                color: [255, 0, 0],
            },
            GouraudVertex {
                x: 2.0,
                y: 7.0,
                color: [0, 255, 0],
            },
            GouraudVertex {
                x: 8.0,
                y: 7.0,
                color: [0, 0, 255],
            },
        ];
        gouraud_triangle_fill(&mut bmp, (0, 0), &clip, &pipe, tri);

        // Midpoint row should be painted.
        let mid = bmp.row(4);
        let any_painted = (2..8usize).any(|x| mid[x].r > 0 || mid[x].g > 0 || mid[x].b > 0);
        assert!(
            any_painted,
            "mid-row of flat-bottom triangle should be painted"
        );
    }

    /// A window of the vertices' grid receives the spans the whole grid
    /// places there: the vertices stay on the grid, and the window is given
    /// by its origin.
    #[test]
    fn a_window_receives_the_whole_grids_spans() {
        let tri = [
            GouraudVertex {
                x: 1.3,
                y: 0.7,
                color: [250, 10, 30],
            },
            GouraudVertex {
                x: 37.9,
                y: 11.2,
                color: [5, 240, 60],
            },
            GouraudVertex {
                x: 14.6,
                y: 28.4,
                color: [40, 70, 255],
            },
        ];
        let spans = |w: i32, h: i32, origin: (i32, i32)| {
            let (w, h) = (w.cast_unsigned(), h.cast_unsigned());
            let mut out = Vec::new();
            gouraud_triangle_spans(w, h, origin, &make_clip(w, h), tri, |c, x0, x1, y| {
                out.push((x0 + origin.0, x1 + origin.0, y + origin.1, c.to_vec()));
            });
            out
        };
        let whole = spans(40, 30, (0, 0));
        for (origin, w, h) in [((0, 9), 40, 1), ((5, 7), 20, 10), ((0, 13), 40, 17)] {
            let window = spans(w, h, origin);
            let (x_end, y_end) = (origin.0 + w, origin.1 + h);
            let expected: Vec<_> = whole
                .iter()
                .filter(|s| (origin.1..y_end).contains(&s.2))
                .filter_map(|(x0, x1, y, c)| {
                    let (a, b) = ((*x0).max(origin.0), (*x1).min(x_end - 1));
                    let skip = usize::try_from(a - x0).ok()? * 3;
                    let len = usize::try_from(b - a + 1).ok()? * 3;
                    Some((a, b, *y, c[skip..skip + len].to_vec()))
                })
                .collect();
            assert_eq!(window, expected, "window at {origin:?}");
        }
    }
}
