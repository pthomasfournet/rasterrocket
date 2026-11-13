//! Property tests of path filling and clipping.
//!
//! Paths are polygons and cubic curves with coordinates on a 1/16-pixel
//! grid, partly outside the bitmap, filled aliased and antialiased under
//! both winding rules.  Grid coordinates keep integer translation exact in
//! `f64`, so translated renders can be compared byte for byte.
//!
//! Each property runs the case count given to [`config`] unless
//! `PROPTEST_CASES` sets another.

use proptest::prelude::*;
use proptest::test_runner::Config;

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::fill::{eo_fill, fill};
use crate::path::{Path, PathBuilder};
use crate::pipe::PipeSrc;
use crate::testutil::simple_pipe;
use crate::types::{AA_GAMMA, AA_SIZE};
use crate::xpath::XPath;
use color::Rgb8;

/// The runner configuration: `cases` cases, or `PROPTEST_CASES` when set.
fn config(cases: u32) -> Config {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cases);
    Config {
        cases,
        ..Config::default()
    }
}

/// Bitmap sides, in pixels.
const SIDE: std::ops::RangeInclusive<u32> = 1..=40;

/// Grid steps per pixel.
const GRID: i32 = 16;

/// How far past each bitmap edge path coordinates reach, in pixels.
const MARGIN: i32 = 8;

type Pt = (f64, f64);

#[derive(Debug, Clone)]
enum Seg {
    Line(Pt),
    Curve(Pt, Pt, Pt),
}

#[derive(Debug, Clone)]
struct Subpath {
    start: Pt,
    segs: Vec<Seg>,
}

impl Subpath {
    /// The same outline traversed backwards.
    fn reversed(&self) -> Self {
        let mut points = vec![self.start];
        for seg in &self.segs {
            points.push(match seg {
                Seg::Line(p) | Seg::Curve(_, _, p) => *p,
            });
        }
        let segs = self
            .segs
            .iter()
            .enumerate()
            .rev()
            .map(|(i, seg)| match seg {
                Seg::Line(_) => Seg::Line(points[i]),
                Seg::Curve(c1, c2, _) => Seg::Curve(*c2, *c1, points[i]),
            })
            .collect();
        Self {
            start: *points.last().expect("the start is a point"),
            segs,
        }
    }

    fn translated(&self, dx: f64, dy: f64) -> Self {
        let t = |(x, y): Pt| (x + dx, y + dy);
        Self {
            start: t(self.start),
            segs: self
                .segs
                .iter()
                .map(|seg| match *seg {
                    Seg::Line(p) => Seg::Line(t(p)),
                    Seg::Curve(a, b, c) => Seg::Curve(t(a), t(b), t(c)),
                })
                .collect(),
        }
    }
}

fn build(subpaths: &[Subpath]) -> Path {
    let mut b = PathBuilder::new();
    for sp in subpaths {
        b.move_to(sp.start.0, sp.start.1).expect("move_to");
        for seg in &sp.segs {
            match *seg {
                Seg::Line((x, y)) => b.line_to(x, y),
                Seg::Curve((x1, y1), (x2, y2), (x3, y3)) => b.curve_to(x1, y1, x2, y2, x3, y3),
            }
            .expect("a current point");
        }
    }
    b.build()
}

/// A grid coordinate within `MARGIN` pixels of `[0, extent]`.
fn coord(extent: u32) -> impl Strategy<Value = f64> {
    let extent = i32::try_from(extent).expect("small");
    let lo = -MARGIN * GRID;
    let hi = (extent + MARGIN) * GRID;
    (lo..=hi).prop_map(|i| f64::from(i) / f64::from(GRID))
}

fn point(w: u32, h: u32) -> impl Strategy<Value = Pt> {
    (coord(w), coord(h))
}

fn seg(w: u32, h: u32, curves: bool) -> impl Strategy<Value = Seg> {
    let line = point(w, h).prop_map(Seg::Line);
    let curve = (point(w, h), point(w, h), point(w, h)).prop_map(|(a, b, c)| Seg::Curve(a, b, c));
    if curves {
        prop_oneof![2 => line, 1 => curve].boxed()
    } else {
        line.boxed()
    }
}

fn subpaths(w: u32, h: u32, curves: bool) -> impl Strategy<Value = Vec<Subpath>> {
    let sub = (point(w, h), prop::collection::vec(seg(w, h, curves), 1..8))
        .prop_map(|(start, segs)| Subpath { start, segs });
    prop::collection::vec(sub, 1..=3)
}

/// A rectangle with coordinates on the grid, possibly reaching outside
/// the bitmap and possibly empty.
fn rect(w: u32, h: u32) -> impl Strategy<Value = [f64; 4]> {
    (coord(w), coord(h), coord(w), coord(h)).prop_map(<[f64; 4]>::from)
}

/// Fill options: antialiased, even-odd.
fn mode() -> impl Strategy<Value = (bool, bool)> {
    (any::<bool>(), any::<bool>())
}

/// A scene: bitmap size, outline, clip rectangle, optional clip path, fill
/// options.
#[derive(Debug, Clone)]
struct Scene {
    w: u32,
    h: u32,
    path: Vec<Subpath>,
    clip_rect: [f64; 4],
    clip_path: Option<(Vec<Subpath>, bool)>,
    aa: bool,
    eo: bool,
}

impl Scene {
    fn clip(&self) -> Clip {
        let [x0, y0, x1, y1] = self.clip_rect;
        let mut clip = Clip::new(x0, y0, x1, y1, self.aa);
        if let Some((sub, eo)) = &self.clip_path {
            clip.clip_to_path(&XPath::new(&build(sub), &IDENTITY, 0.25, true), *eo);
        }
        clip
    }
}

fn scene(curves: bool, clip_paths: bool) -> impl Strategy<Value = Scene> {
    (SIDE, SIDE).prop_flat_map(move |(w, h)| {
        let clip_path = if clip_paths {
            prop::option::of((subpaths(w, h, curves), any::<bool>())).boxed()
        } else {
            Just(None).boxed()
        };
        (subpaths(w, h, curves), rect(w, h), clip_path, mode()).prop_map(
            move |(path, clip_rect, clip_path, (aa, eo))| Scene {
                w,
                h,
                path,
                clip_rect,
                clip_path,
                aa,
                eo,
            },
        )
    })
}

const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// The red channel of `path` filled white over black, one byte per pixel.
fn render(path: &Path, clip: &Clip, w: u32, h: u32, aa: bool, eo: bool) -> Vec<u8> {
    let mut bmp: Bitmap<Rgb8> = Bitmap::new(w, h, 1, false);
    let paint = if eo { eo_fill::<Rgb8> } else { fill::<Rgb8> };
    paint(
        &mut bmp,
        clip,
        path,
        &simple_pipe(),
        &PipeSrc::Solid(&[255, 255, 255]),
        &IDENTITY,
        0.25,
        aa,
    );
    (0..h)
        .flat_map(|y| bmp.row(y).iter().map(|p| p.r))
        .collect()
}

/// A clip of the whole `w × h` bitmap.
fn whole(w: u32, h: u32, aa: bool) -> Clip {
    Clip::new(0.0, 0.0, f64::from(w), f64::from(h), aa)
}

/// Pixel coordinates of index `i` in a `w`-wide raster.
fn at(i: usize, w: u32) -> (i32, i32) {
    let w = w as usize;
    (
        i32::try_from(i % w).expect("small"),
        i32::try_from(i / w).expect("small"),
    )
}

/// A polygon star-shaped about a centre: its vertices in strictly
/// increasing angle about the centre, each turn under π, so the outline is
/// simple and its winding number is 0 or ±1 everywhere.
///
/// Angular gaps are weights of 10 to 19 (any one under half their sum),
/// radii up to 24 px; vertices snap to the grid, and the few that the snap
/// takes out of order are rejected.
fn star(w: u32, h: u32) -> impl Strategy<Value = Subpath> {
    let vertex = (10u32..20, 1u32..=24 * 16);
    (point(w, h), prop::collection::vec(vertex, 3..10)).prop_filter_map(
        "vertices strictly ordered by angle, each turn under π",
        |(centre, spokes)| {
            let total: u32 = spokes.iter().map(|&(gap, _)| gap).sum();
            let snap = |v: f64| (v * f64::from(GRID)).round() / f64::from(GRID);
            let mut swept = 0;
            let vertices: Vec<Pt> = spokes
                .iter()
                .map(|&(gap, radius)| {
                    swept += gap;
                    let theta = std::f64::consts::TAU * f64::from(swept) / f64::from(total);
                    let r = f64::from(radius) / f64::from(GRID);
                    (
                        snap(r.mul_add(theta.cos(), centre.0)),
                        snap(r.mul_add(theta.sin(), centre.1)),
                    )
                })
                .collect();
            let angle = |p: &Pt| (p.1 - centre.1).atan2(p.0 - centre.0);
            let count = vertices.len();
            let simple = (0..count).all(|i| {
                let (p, q) = (&vertices[i], &vertices[(i + 1) % count]);
                if *p == centre {
                    return false;
                }
                let turn = (angle(q) - angle(p)).rem_euclid(std::f64::consts::TAU);
                turn > 1e-9 && turn < std::f64::consts::PI - 1e-9
            });
            let turns: f64 = (0..count)
                .map(|i| {
                    let (p, q) = (&vertices[i], &vertices[(i + 1) % count]);
                    (angle(q) - angle(p)).rem_euclid(std::f64::consts::TAU)
                })
                .sum();
            // The turns must sum to one revolution, not wrap twice.
            (simple && (turns - std::f64::consts::TAU).abs() < 1e-6).then(|| Subpath {
                start: vertices[0],
                segs: vertices[1..].iter().map(|&p| Seg::Line(p)).collect(),
            })
        },
    )
}

proptest! {
    #![proptest_config(config(2048))]

    /// A fill paints only pixels inside the clip: within the clip
    /// rectangle's pixel bounds, and, aliased, only pixels the clip's
    /// paths pass.
    #[test]
    fn fills_stay_inside_the_clip(s in scene(true, true)) {
        let clip = s.clip();
        let out = render(&build(&s.path), &clip, s.w, s.h, s.aa, s.eo);
        for (i, &v) in out.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let (x, y) = at(i, s.w);
            prop_assert!(
                (clip.x_min_i..=clip.x_max_i).contains(&x)
                    && (clip.y_min_i..=clip.y_max_i).contains(&y),
                "pixel ({}, {}) = {} outside the clip rectangle", x, y, v
            );
            if !s.aa {
                prop_assert!(clip.test(x, y), "pixel ({}, {}) outside the clip paths", x, y);
            }
        }
    }

    /// A clip rectangle masks the unclipped fill: inside it the pixels are
    /// the unclipped fill's, outside it they are untouched.
    #[test]
    fn a_clip_rectangle_masks_the_unclipped_fill(s in scene(true, false)) {
        let path = build(&s.path);
        let clip = s.clip();
        let clipped = render(&path, &clip, s.w, s.h, s.aa, s.eo);
        let free = render(&path, &whole(s.w, s.h, s.aa), s.w, s.h, s.aa, s.eo);
        for (i, (&c, &f)) in clipped.iter().zip(&free).enumerate() {
            let (x, y) = at(i, s.w);
            let inside = (clip.x_min_i..=clip.x_max_i).contains(&x)
                && (clip.y_min_i..=clip.y_max_i).contains(&y);
            prop_assert_eq!(c, if inside { f } else { 0 }, "pixel ({}, {})", x, y);
        }
    }

    /// Reversing every subpath changes no pixel under either rule.
    #[test]
    fn orientation_does_not_matter(s in scene(true, true)) {
        let clip = s.clip();
        let reversed: Vec<Subpath> = s.path.iter().map(Subpath::reversed).collect();
        prop_assert_eq!(
            render(&build(&s.path), &clip, s.w, s.h, s.aa, s.eo),
            render(&build(&reversed), &clip, s.w, s.h, s.aa, s.eo)
        );
    }

    /// Moving the outline and the clip by whole pixels moves the render.
    #[test]
    fn integer_translation_moves_the_render(
        s in scene(true, true),
        dx in -6i32..=6,
        dy in -6i32..=6,
    ) {
        let clip = s.clip();
        let moved: Vec<Subpath> =
            s.path.iter().map(|sp| sp.translated(f64::from(dx), f64::from(dy))).collect();
        let a = render(&build(&s.path), &clip, s.w, s.h, s.aa, s.eo);
        let b = render(&build(&moved), &clip.translated(dx, dy), s.w, s.h, s.aa, s.eo);
        let (w, h) = (i32::try_from(s.w).expect("small"), i32::try_from(s.h).expect("small"));
        for (i, &v) in a.iter().enumerate() {
            let (x, y) = at(i, s.w);
            let (mx, my) = (x + dx, y + dy);
            if (0..w).contains(&mx) && (0..h).contains(&my) {
                let j = usize::try_from(my * w + mx).expect("in range");
                prop_assert_eq!(b[j], v, "pixel ({}, {}) moved to ({}, {})", x, y, mx, my);
            }
        }
    }

    /// A simple polygon fills the same under both winding rules.
    #[test]
    fn simple_polygons_fill_alike_under_both_rules(
        (w, h, poly) in (SIDE, SIDE).prop_flat_map(|(w, h)| (Just(w), Just(h), star(w, h))),
        aa in any::<bool>(),
    ) {
        let path = build(&[poly]);
        let clip = whole(w, h, aa);
        prop_assert_eq!(
            render(&path, &clip, w, h, aa, false),
            render(&path, &clip, w, h, aa, true)
        );
    }

    /// A rectangle with integer corners covers exactly the cells its closed
    /// outline touches: aliased, the pixels `x0..=x1 × y0..=y1`; antialiased,
    /// the sub-samples `4·x0..=4·x1 × 4·y0..=4·y1`, so its interior pixels are
    /// fully covered and the column and row past its far edges a quarter.
    #[test]
    fn a_pixel_aligned_rectangle_covers_its_closed_cells(
        (w, h) in (SIDE, SIDE),
        (x0, y0) in (-4i32..40, -4i32..40),
        (dw, dh) in (0i32..24, 0i32..24),
        aa in any::<bool>(),
        eo in any::<bool>(),
        clockwise in any::<bool>(),
    ) {
        let (x1, y1) = (x0 + dw, y0 + dh);
        let (fx0, fy0, fx1, fy1) = (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
        let mut sp = Subpath {
            start: (fx0, fy0),
            segs: vec![Seg::Line((fx1, fy0)), Seg::Line((fx1, fy1)), Seg::Line((fx0, fy1))],
        };
        if clockwise {
            sp = sp.reversed();
        }
        let out = render(&build(&[sp]), &whole(w, h, aa), w, h, aa, eo);
        // Cells of `lo..=hi` (sub-sample units) inside the cell span `c`.
        let overlap = |c: std::ops::Range<i32>, lo: i32, hi: i32| {
            u8::try_from((c.end.min(hi + 1) - c.start.max(lo)).max(0)).expect("at most 4")
        };
        for (i, &v) in out.iter().enumerate() {
            let (x, y) = at(i, w);
            let want = if aa {
                let n = AA_SIZE;
                let cols = overlap(x * n..(x + 1) * n, x0 * n, x1 * n);
                let rows = overlap(y * n..(y + 1) * n, y0 * n, y1 * n);
                AA_GAMMA[usize::from(cols * rows)]
            } else if (x0..=x1).contains(&x) && (y0..=y1).contains(&y) {
                255
            } else {
                0
            };
            prop_assert_eq!(v, want, "pixel ({}, {})", x, y);
        }
    }

    /// Flattening keeps every edge finite and inside the bounding box of
    /// the transformed points, curve control points included.
    #[test]
    fn flattened_edges_stay_in_the_control_hull(
        path in subpaths(40, 40, true),
        m in prop::array::uniform4(-4.0f64..4.0),
        t in prop::array::uniform2(-100.0f64..100.0),
        flatness in prop_oneof![Just(0.0), 0.01f64..2.0],
    ) {
        let matrix = [m[0], m[1], m[2], m[3], t[0], t[1]];
        let path = build(&path);
        let xpath = XPath::new(&path, &matrix, flatness, true);
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in &path.pts {
            let x = matrix[0].mul_add(p.x, matrix[2].mul_add(p.y, matrix[4]));
            let y = matrix[1].mul_add(p.x, matrix[3].mul_add(p.y, matrix[5]));
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x), hi[1].max(y)];
        }
        let eps = 1e-6 * (1.0 + hi[0].abs().max(hi[1].abs()).max(lo[0].abs()).max(lo[1].abs()));
        for seg in &xpath.segs {
            prop_assert!(seg.is_finite(), "{:?}", seg);
            for (x, y) in [(seg.x0, seg.y0), (seg.x1, seg.y1)] {
                prop_assert!(
                    x >= lo[0] - eps && x <= hi[0] + eps && y >= lo[1] - eps && y <= hi[1] + eps,
                    "({}, {}) outside {:?}..{:?}", x, y, lo, hi
                );
            }
        }
    }
}
