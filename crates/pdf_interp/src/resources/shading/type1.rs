//! Function-based (Type 1) shadings (PDF §8.7.4.5.2).

use color::Sample;
use pdf::{Dictionary, Document};
use raster::pipe::Pattern;
use raster::shading::function::FunctionPattern;

use super::function::{Function, read_f64_array};
use super::{Canvas, ShadingColour, Unresolved, device_to_space, transform_point};
use crate::renderer::gstate::{ctm_multiply, ctm_transform};
use crate::resources::colorspace::EXACT_WORK_BUDGET;
use crate::resources::lattice::{Axis, for_each_corner};
use crate::resources::read_matrix;

/// Most samples of a Type 1 shading's device lattice (12 MiB of RGB).
const TYPE1_LATTICE_MAX: usize = 1 << 22;

/// Tolerance, in units of the Domain's extent, admitting a point on a
/// Type 1 shading's Domain edge despite rounding error.
const TYPE1_EDGE_TOLERANCE: f64 = 1e-9;

/// The device pixels a Type 1 shading is sampled at: every `step`-th
/// pixel from `(x0, y0)`, `cols × rows` of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lattice {
    x0: i64,
    y0: i64,
    step: usize,
    cols: usize,
    rows: usize,
}

impl Lattice {
    /// The lattice over the integer device pixels of `area`, the inclusive
    /// pixel bounds `[x0, y0, x1, y1]`: step 1 while the area holds at most
    /// `max_points` pixels, else the least step that keeps the lattice
    /// within it, the last row and column reaching the far edges.  `None`
    /// when the area is empty.  `max_points` is at least 4, which any area
    /// fits at a step of its longer side.
    fn covering([x0, y0, x1, y1]: [i64; 4], max_points: usize) -> Option<Self> {
        debug_assert!(max_points >= 4, "a lattice needs its four corners");
        let w = usize::try_from(x1 - x0 + 1).ok().filter(|&w| w > 0)?;
        let h = usize::try_from(y1 - y0 + 1).ok().filter(|&h| h > 0)?;
        let dims = |step: usize| ((w - 1).div_ceil(step) + 1, (h - 1).div_ceil(step) + 1);
        let fits = |step: usize| {
            let (cols, rows) = dims(step);
            cols.saturating_mul(rows) <= max_points
        };
        // `fits` is monotone in the step and holds at the longer side.
        let (mut lo, mut hi) = (1, w.max(h));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let (cols, rows) = dims(lo);
        Some(Self {
            x0,
            y0,
            step: lo,
            cols,
            rows,
        })
    }

    /// Device pixel of lattice point `(col, row)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the lattice lies within the canvas, far below 2^52 pixels"
    )]
    fn pixel(self, col: usize, row: usize) -> (f64, f64) {
        (
            self.x0 as f64 + (self.step * col) as f64,
            self.y0 as f64 + (self.step * row) as f64,
        )
    }

    /// The interpolation axes of device pixel `(px, py)`: the lattice
    /// columns and rows around it, clamped to the lattice.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the lattice lies within the canvas, far below 2^52 pixels"
    )]
    fn axes(self, px: f64, py: f64) -> [Axis; 2] {
        let step = self.step as f64;
        [
            Axis::at((px - self.x0 as f64) / step, self.cols, 1),
            Axis::at((py - self.y0 as f64) / step, self.rows, self.cols),
        ]
    }

    /// The window of the lattice points that interpolate the device pixels
    /// of `region`, the inclusive bounds `[x0, y0, x1, y1]`; an empty
    /// window when the region is empty.  [`Self::axes`] is monotone in
    /// each coordinate, so the axes of the region's corners bound those of
    /// every pixel within it.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the region holds pixels of the page, far below 2^52"
    )]
    fn window(self, [x0, y0, x1, y1]: [i64; 4]) -> Window {
        if x1 < x0 || y1 < y0 {
            return Window::EMPTY;
        }
        let [lo_x, lo_y] = self.axes(x0 as f64, y0 as f64);
        let [hi_x, hi_y] = self.axes(x1 as f64, y1 as f64);
        Window {
            col0: lo_x.lo,
            row0: lo_y.lo,
            cols: hi_x.hi - lo_x.lo + 1,
            rows: hi_y.hi - lo_y.lo + 1,
        }
    }
}

/// A rectangle of a [`Lattice`]'s points: `cols × rows` of them from
/// column `col0` and row `row0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    col0: usize,
    row0: usize,
    cols: usize,
    rows: usize,
}

impl Window {
    /// The window of no point.
    const EMPTY: Self = Self {
        col0: 0,
        row0: 0,
        cols: 0,
        rows: 0,
    };

    /// The window over every point of `lattice`.
    const fn whole(lattice: Lattice) -> Self {
        Self {
            col0: 0,
            row0: 0,
            cols: lattice.cols,
            rows: lattice.rows,
        }
    }

    /// Whether lattice point `(col, row)` lies in the window.
    const fn holds(self, col: usize, row: usize) -> bool {
        col >= self.col0
            && col < self.col0 + self.cols
            && row >= self.row0
            && row < self.row0 + self.rows
    }

    /// The lattice axes `[x, y]` re-indexed into the window's samples,
    /// stored row by row; `None` when a corner they name lies outside it.
    /// The positions and weights are unchanged.
    const fn rebase(self, [x, y]: [Axis; 2]) -> Option<[Axis; 2]> {
        if !(self.holds(x.lo, y.lo) && self.holds(x.hi, y.hi)) {
            return None;
        }
        Some([
            Axis {
                lo: x.lo - self.col0,
                hi: x.hi - self.col0,
                stride: 1,
                ..x
            },
            Axis {
                lo: y.lo - self.row0,
                hi: y.hi - self.row0,
                stride: self.cols,
                ..y
            },
        ])
    }
}

/// The inclusive integer pixel bounds `[x0, y0, x1, y1]` of the device
/// points `corners` that lie on `canvas`, widened by one pixel so a pixel
/// on the canvas edge has an interpolation neighbour; within `bbox`'s
/// corners too when given.  Empty bounds have `x1 < x0` or `y1 < y0`.
/// Every corner is finite.
fn type1_area(corners: &[(f64, f64)], bbox: Option<&[(f64, f64); 4]>, canvas: Canvas) -> [i64; 4] {
    let span = |points: &[(f64, f64)], axis: fn(&(f64, f64)) -> f64| {
        let lo = points.iter().map(axis).fold(f64::INFINITY, f64::min);
        let hi = points.iter().map(axis).fold(f64::NEG_INFINITY, f64::max);
        (lo.floor(), hi.ceil())
    };
    let (mut x0, mut x1) = span(corners, |c| c.0);
    let (mut y0, mut y1) = span(corners, |c| c.1);
    if let Some(bbox) = bbox {
        let (bx0, bx1) = span(bbox, |c| c.0);
        let (by0, by1) = span(bbox, |c| c.1);
        (x0, x1, y0, y1) = (x0.max(bx0), x1.min(bx1), y0.max(by0), y1.min(by1));
    }
    // The low edge is clamped to [-1, extent + 1] and the high edge to
    // [-2, extent]: an empty range stays empty.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "whole numbers clamped to within two of the canvas"
    )]
    let edge = |lo: f64, hi: f64, extent: f64| {
        let extent = extent.ceil();
        (
            lo.clamp(-1.0, extent + 1.0) as i64,
            hi.clamp(-2.0, extent) as i64,
        )
    };
    let (x0, x1) = edge(x0, x1, canvas.width);
    let (y0, y1) = edge(y0, y1, canvas.height);
    [x0, y0, x1, y1]
}

#[cfg(test)]
thread_local! {
    /// The Function evaluations [`sample_lattice`] has made on this thread.
    static EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The colours of a Type 1 shading at the points of `window` of
/// `lattice`, row by row: `function` at the shading-space point of each,
/// its fractions of the Domain `[x0, x1, y0, y1]` given by `domain_at`,
/// converted through `cs`.  At step 1 a point outside the Domain is never
/// read and is left black; at a larger step it takes the colour of the
/// nearest Domain point.
///
/// [`Unresolved::NoColour`], logged, when the function evaluates at none
/// of the lattice points it is evaluated at; points where it fails take
/// colour components 0, logged once.  The decision covers the whole
/// lattice, not the window alone: when the function fails at every point
/// of the window it reaches, the points outside the window are evaluated
/// until one succeeds.  A lattice with no point in the Domain evaluates
/// nothing and paints nothing.
fn sample_lattice<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    (lattice, window): (Lattice, Window),
    domain_at: &impl Fn(f64, f64) -> (f64, f64, bool),
    function: &Function,
    cs: C,
    n: usize,
    [xd0, xd1, yd0, yd1]: [f64; 4],
) -> Result<Box<[[S; N]]>, Unresolved> {
    // The function's outputs at lattice point `(col, row)`: `None` for a
    // point not evaluated, `Some(None)` where the function fails.
    let eval = |col: usize, row: usize| -> Option<Option<Vec<f64>>> {
        let (px, py) = lattice.pixel(col, row);
        let (tx, ty, inside) = domain_at(px, py);
        if lattice.step == 1 && !inside {
            return None;
        }
        let sx = tx.mul_add(xd1 - xd0, xd0);
        let sy = ty.mul_add(yd1 - yd0, yd0);
        #[cfg(test)]
        EVALUATIONS.with(|count| count.set(count.get() + 1));
        Some(function.eval_n(&[sx, sy], n))
    };
    let mut samples = Vec::with_capacity(window.cols * window.rows);
    let (mut evaluated, mut failures) = (0usize, 0usize);
    for row in window.row0..window.row0 + window.rows {
        for col in window.col0..window.col0 + window.cols {
            let Some(outputs) = eval(col, row) else {
                samples.push([S::ZERO; N]);
                continue;
            };
            evaluated += 1;
            let channels = outputs.unwrap_or_else(|| {
                failures += 1;
                vec![0.0; n]
            });
            samples.push(cs.convert(&channels));
        }
    }
    if failures == evaluated && window != Window::whole(lattice) {
        let outside = (0..lattice.rows)
            .flat_map(|row| (0..lattice.cols).map(move |col| (col, row)))
            .filter(|&(col, row)| !window.holds(col, row));
        for (col, row) in outside {
            match eval(col, row) {
                None => {}
                Some(None) => {
                    evaluated += 1;
                    failures += 1;
                }
                Some(Some(_)) => {
                    evaluated += 1;
                    break;
                }
            }
        }
    }
    if evaluated > 0 && failures == evaluated {
        log::warn!("shading/type1: Function evaluates at no sampled point — skipping");
        return Err(Unresolved::NoColour);
    }
    if failures > 0 {
        log::warn!(
            "shading/type1: Function fails at {failures} of {evaluated} sampled points — \
             painting their colour components as 0"
        );
    }
    Ok(samples.into_boxed_slice())
}

/// The Domain `[x0, x1, y0, y1]` of a Type 1 shading: `[0 1 0 1]` when
/// absent (PDF §8.7.4.5.2); a malformed one is logged and the default used.
fn type1_domain(sh: &Dictionary) -> [f64; 4] {
    const DEFAULT: [f64; 4] = [0.0, 1.0, 0.0, 1.0];
    if sh.get(b"Domain").is_none() {
        return DEFAULT;
    }
    read_f64_array(sh, b"Domain", 4)
        .and_then(|d| <[f64; 4]>::try_from(d.as_slice()).ok())
        .unwrap_or_else(|| {
            log::warn!("shading/type1: Domain is not four numbers — using [0 1 0 1]");
            DEFAULT
        })
}

/// A resolved Type 1 shading: its pattern and the heap bytes it holds.
type FunctionBased<S> = (Box<dyn Pattern<S> + Send + Sync>, usize);

/// A Type 1 shading ready to sample: its function, colour space and
/// output count, Domain `[x0, x1, y0, y1]`, and the matrix from shading
/// space to the device.
struct Type1<'a, C> {
    function: &'a Function,
    cs: C,
    n: usize,
    domain: [f64; 4],
    shading_ctm: [f64; 6],
}

pub(super) fn resolve_function_based<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    doc: &Document,
    sh: &Dictionary,
    cs: C,
    n: usize,
    ctm: &[f64; 6],
    canvas: Canvas,
    bbox: Option<&[(f64, f64); 4]>,
) -> Result<FunctionBased<S>, Unresolved> {
    let domain @ [xd0, xd1, yd0, yd1] = type1_domain(sh);
    if !domain.iter().all(|v| v.is_finite()) {
        log::warn!("shading/type1: non-finite Domain — skipping");
        return Err(Unresolved::Skipped);
    }
    if xd0 >= xd1 || yd0 >= yd1 {
        log::warn!("shading/type1: degenerate Domain [{xd0},{xd1}]×[{yd0},{yd1}] — skipping");
        return Err(Unresolved::Skipped);
    }
    let function = Function::load(doc, sh.get(b"Function").ok_or(Unresolved::Skipped)?);
    // The point budget is the smaller of a count and the conversion work
    // budget over the cost of one point.
    let point_cost = function.cost().saturating_add(cs.conversion_cost()).max(1);
    let affordable = usize::try_from(EXACT_WORK_BUDGET / point_cost).unwrap_or(usize::MAX);
    let shading = Type1 {
        function: &function,
        cs,
        n,
        domain,
        // PDF §8.7.4.5.2: `Matrix` maps shading space into the target
        // space the CTM maps to the device.
        shading_ctm: read_matrix(sh).map_or(*ctm, |m| ctm_multiply(&m, ctm)),
    };
    shading.pattern(canvas, bbox, TYPE1_LATTICE_MAX.min(affordable).max(4))
}

impl<C> Type1<'_, C> {
    /// The shading's pattern on `canvas`, within `bbox` when given, from at
    /// most `max_points` samples.
    ///
    /// The colour at shading-space point (x, y) is the conversion of
    /// Function(x, y).  It is sampled at the device pixels of a lattice
    /// over the part of the Domain's device bounding box on the canvas and
    /// within `bbox`: at every pixel, which the pattern then reads
    /// unchanged, unless that exceeds `max_points`, when every `step`-th
    /// pixel is sampled and the pixels between are interpolated.
    ///
    /// The lattice depends on the Domain, canvas and `bbox` alone.  With a
    /// `canvas.region`, only the lattice points that interpolate a pixel of
    /// the region are evaluated and held; the pattern paints those pixels
    /// as the whole lattice would, and leaves unpainted any pixel that
    /// interpolates a point not held.
    fn pattern<const N: usize, S: Sample>(
        &self,
        canvas: Canvas,
        bbox: Option<&[(f64, f64); 4]>,
        max_points: usize,
    ) -> Result<FunctionBased<S>, Unresolved>
    where
        C: ShadingColour<N, S>,
    {
        let [xd0, xd1, yd0, yd1] = self.domain;
        let page_h = canvas.height;
        let Some(inv) = device_to_space(&self.shading_ctm, page_h) else {
            log::warn!("shading/type1: degenerate Matrix × CTM — skipping");
            return Err(Unresolved::Skipped);
        };
        // The shading-space point of device pixel (px, py) as fractions of
        // the Domain, clamped to it, and whether it lies in the Domain (PDF
        // §8.7.4.5.2: points outside it are not painted).
        let domain_at = move |px: f64, py: f64| {
            let (sx, sy) = ctm_transform(&inv, px, py);
            let tx = (sx - xd0) / (xd1 - xd0);
            let ty = (sy - yd0) / (yd1 - yd0);
            let inside = |t: f64| (-TYPE1_EDGE_TOLERANCE..=1.0 + TYPE1_EDGE_TOLERANCE).contains(&t);
            (
                tx.clamp(0.0, 1.0),
                ty.clamp(0.0, 1.0),
                inside(tx) && inside(ty),
            )
        };
        let corner = |x, y| transform_point(&self.shading_ctm, x, y, page_h);
        let corners = [
            corner(xd0, yd0),
            corner(xd1, yd0),
            corner(xd0, yd1),
            corner(xd1, yd1),
        ];
        if corners
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            log::warn!("shading/type1: Domain maps to non-finite device coordinates — skipping");
            return Err(Unresolved::Skipped);
        }
        let Some(lattice) = Lattice::covering(type1_area(&corners, bbox, canvas), max_points)
        else {
            log::debug!(
                "shading/type1: the Domain covers no pixel of the canvas — nothing painted"
            );
            return Ok((Box::new(FunctionPattern::<N, S>::new(|_, _| None)), 0));
        };
        if lattice.step > 1 {
            log::debug!(
                "shading/type1: sampled every {} pixels, {} points within a budget of {max_points}",
                lattice.step,
                lattice.cols * lattice.rows
            );
        }
        let window = canvas
            .region
            .map_or_else(|| Window::whole(lattice), |region| lattice.window(region));
        let samples = sample_lattice(
            (lattice, window),
            &domain_at,
            self.function,
            self.cs,
            self.n,
            self.domain,
        )?;
        let held = std::mem::size_of_val(&*samples);
        let pattern = FunctionPattern::new(move |px, py| {
            if !domain_at(px, py).2 {
                return None;
            }
            let axes = window.rebase(lattice.axes(px, py))?;
            let level = |v: S| f64::from(u32::try_from(v.signed()).unwrap_or(0));
            let mut colour = [0.0_f64; N];
            for_each_corner(&axes, |index, weight| {
                for (acc, &v) in colour.iter_mut().zip(&samples[index]) {
                    *acc = weight.mul_add(level(v), *acc);
                }
            });
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a convex combination of samples, rounded, lies in the sample range"
            )]
            Some(colour.map(|v| S::from_signed(v.round() as i64)))
        });
        Ok((Box::new(pattern), held))
    }
}

#[cfg(test)]
mod tests {
    use color::SampleDepth;
    use pdf::{Object, Stream};

    use super::*;
    use crate::resources::colorspace::ColorSpace;
    use crate::resources::shading::ShadingCs;
    use crate::test_helpers::empty_doc;

    /// A Type 1 shading whose function evaluates nowhere paints nothing
    /// rather than a black rectangle.
    #[test]
    fn type1_with_an_unusable_function_is_skipped() {
        let doc = empty_doc();
        let mut exponential = Dictionary::new();
        exponential.set("FunctionType", Object::Integer(2));
        exponential.set("N", Object::Integer(1));
        let mut sh = Dictionary::new();
        sh.set("ShadingType", Object::Integer(1));
        // A Type 2 function takes one input; Type 1 supplies two.
        sh.set("Function", Object::Dictionary(exponential));
        let cs = ShadingCs {
            doc: &doc,
            space: &ColorSpace::DeviceGray,
            target: crate::ColourTarget::SRGB,
        };
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(
            resolve_function_based::<3, u8, _>(&doc, &sh, cs, 1, &ctm, CANVAS, None).err(),
            Some(Unresolved::NoColour)
        );
        // Read only where the Domain does not reach, it is skipped all the
        // same: the decision covers the whole lattice.
        let elsewhere = Canvas {
            region: Some([500, 500, 510, 510]),
            ..CANVAS
        };
        assert_eq!(
            resolve_function_based::<3, u8, _>(&doc, &sh, cs, 1, &ctm, elsewhere, None).err(),
            Some(Unresolved::NoColour)
        );
    }

    /// A Type 4 function of `(x, y)` returning grey `(x + y) / 2`.
    fn ramp_grey() -> Object {
        let program = b"{ add 0.5 mul }".to_vec();
        let mut dict = Dictionary::new();
        dict.set("FunctionType", Object::Integer(4));
        dict.set("Domain", reals(&[0.0, 1.0, 0.0, 1.0]));
        dict.set("Range", reals(&[0.0, 1.0]));
        dict.set("Length", Object::Integer(program.len().try_into().unwrap()));
        Object::Stream(Stream::new(dict, program))
    }

    /// The colours and coverage `pattern` paints in rows `y0..=y1`,
    /// columns `x0..=x1`.
    fn paint(pattern: &dyn Pattern, [x0, y0, x1, y1]: [i32; 4]) -> Vec<(Vec<u8>, Vec<u8>)> {
        let n = usize::try_from(x1 - x0 + 1).unwrap();
        (y0..=y1)
            .map(|y| {
                let (mut out, mut coverage) = (vec![0; n * 3], vec![0; n]);
                pattern.fill_span_coverage(y, x0, x1, &mut out, &mut coverage);
                (out, coverage)
            })
            .collect()
    }

    /// A Type 1 shading over the whole canvas, read in a 10 × 5 pixel
    /// region, evaluates its function only at the lattice points that
    /// region interpolates, and paints the region exactly as the shading
    /// sampled over the whole canvas does: at every pixel and on a coarse
    /// lattice.
    #[test]
    fn a_type1_shading_is_evaluated_only_where_its_region_is_read() {
        let doc = empty_doc();
        let function = Function::load(&doc, &ramp_grey());
        let shading = Type1 {
            function: &function,
            cs: ShadingCs {
                doc: &doc,
                space: &ColorSpace::DeviceGray,
                target: crate::ColourTarget::SRGB,
            },
            n: 1,
            domain: [0.0, 1.0, 0.0, 1.0],
            shading_ctm: [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
        };
        let region = [20, 30, 29, 34];
        let read = Canvas {
            region: Some(region.map(i64::from)),
            ..CANVAS
        };
        let evaluations = |canvas: Canvas, max_points: usize| {
            EVALUATIONS.with(|count| count.set(0));
            let (pattern, _) = shading
                .pattern::<3, u8>(canvas, None, max_points)
                .expect("resolves");
            (pattern, EVALUATIONS.with(std::cell::Cell::get))
        };
        // The window holds the upper neighbour of the last row and column.
        for (max_points, bound) in [(TYPE1_LATTICE_MAX, 11 * 6), (400, 4 * 3)] {
            let (whole, all) = evaluations(CANVAS, max_points);
            let (part, some) = evaluations(read, max_points);
            assert!(some <= bound, "{some} evaluations at budget {max_points}");
            assert!(all > 10 * some, "{all} evaluations over the canvas");
            assert_eq!(paint(&*part, region), paint(&*whole, region));
            assert_eq!(painted(&*part, 60, 0, 99), 0, "row 60 is not read");
        }
    }

    /// A 100 × 100 pixel canvas.
    const CANVAS: Canvas = Canvas {
        target: crate::ColourTarget::SRGB,
        width: 100.0,
        height: 100.0,
        depth: SampleDepth::Eight,
        region: None,
    };

    /// The array of the reals `values`.
    fn reals(values: &[f32]) -> Object {
        Object::Array(values.iter().copied().map(Object::Real).collect())
    }

    /// A Type 4 function of `(x, y)` returning grey 0.5.
    fn constant_grey() -> Object {
        let program = b"{ pop pop 0.5 }".to_vec();
        let mut dict = Dictionary::new();
        dict.set("FunctionType", Object::Integer(4));
        dict.set("Domain", reals(&[0.0, 1.0, 0.0, 1.0]));
        dict.set("Range", reals(&[0.0, 1.0]));
        dict.set("Length", Object::Integer(program.len().try_into().unwrap()));
        Object::Stream(Stream::new(dict, program))
    }

    /// A Type 1 shading of Domain `[0 1 0 1]` under `matrix`, painting
    /// grey 0.5.
    fn grey_type1(doc: &Document, matrix: [f32; 6]) -> Option<FunctionBased<u8>> {
        let constant = constant_grey();
        let mut sh = Dictionary::new();
        sh.set("ShadingType", Object::Integer(1));
        sh.set("Matrix", reals(&matrix));
        sh.set("Function", constant);
        let cs = ShadingCs {
            doc,
            space: &ColorSpace::DeviceGray,
            target: crate::ColourTarget::SRGB,
        };
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        resolve_function_based(doc, &sh, cs, 1, &ctm, CANVAS, None).ok()
    }

    /// The pixels of row `y` from `x0` to `x1` that `pattern` paints.
    fn painted(pattern: &dyn Pattern, y: i32, x0: i32, x1: i32) -> usize {
        let n = usize::try_from(x1 - x0 + 1).unwrap();
        let (mut out, mut coverage) = (vec![0; n * 3], vec![0; n]);
        pattern.fill_span_coverage(y, x0, x1, &mut out, &mut coverage);
        coverage.iter().filter(|&&c| c != 0).count()
    }

    /// Sampled every few pixels, a Domain rotated 45° paints the function's
    /// colour up to its edge: lattice points outside the Domain take the
    /// nearest Domain point's colour, never black.
    #[test]
    fn a_coarse_lattice_paints_the_function_up_to_a_rotated_edge() {
        let doc = empty_doc();
        let function = Function::load(&doc, &constant_grey());
        let shading = Type1 {
            function: &function,
            cs: ShadingCs {
                doc: &doc,
                space: &ColorSpace::DeviceGray,
                target: crate::ColourTarget::SRGB,
            },
            n: 1,
            domain: [0.0, 1.0, 0.0, 1.0],
            shading_ctm: [42.43, 42.43, -42.43, 42.43, 50.0, 20.0],
        };
        let (pattern, held) = shading
            .pattern::<3, u8>(CANVAS, None, 100)
            .expect("resolves");
        assert!(held <= 100 * 3, "{held} bytes for 100 points");
        let mut painted = 0;
        for y in 0..100 {
            let (mut out, mut coverage) = ([0; 300], [0; 100]);
            pattern.fill_span_coverage(y, 0, 99, &mut out, &mut coverage);
            for (x, (&c, rgb)) in coverage.iter().zip(out.as_chunks::<3>().0).enumerate() {
                if c != 0 {
                    painted += 1;
                    assert_eq!(*rgb, [128; 3], "pixel ({x}, {y})");
                }
            }
        }
        assert!(painted > 3000, "{painted} pixels painted");
    }

    /// A Domain between pixel coordinates holds no lattice point: the
    /// shading resolves and paints nothing, rather than being dropped as
    /// if its function failed.
    #[test]
    fn a_sub_pixel_type1_domain_resolves_and_paints_nothing() {
        let doc = empty_doc();
        let (pattern, _) = grey_type1(&doc, [0.4, 0.0, 0.0, 0.4, 10.2, 50.2]).expect("resolves");
        assert_eq!(painted(&*pattern, 49, 0, 99), 0);
    }

    /// A Domain wholly off the canvas resolves, holds nothing and paints
    /// nothing.
    #[test]
    fn an_off_canvas_type1_domain_holds_nothing() {
        let doc = empty_doc();
        let (pattern, held) =
            grey_type1(&doc, [10.0, 0.0, 0.0, 10.0, -5_000.0, 0.0]).expect("resolves");
        assert_eq!(held, 0);
        assert_eq!(painted(&*pattern, 50, 0, 99), 0);
    }

    /// The samples of a Domain covering the canvas are bounded by the
    /// canvas, however far the Domain extends.
    #[test]
    fn a_huge_type1_domain_is_sampled_over_the_canvas_only() {
        let doc = empty_doc();
        let (pattern, held) = grey_type1(&doc, [1e9, 0.0, 0.0, 1e9, -5e8, -5e8]).expect("resolves");
        assert!(held <= 102 * 102 * 3, "{held} bytes held");
        assert_eq!(painted(&*pattern, 50, 0, 99), 100);
    }

    #[test]
    fn a_lattice_covers_every_pixel_of_a_small_area() {
        let lattice = Lattice::covering([10, 5, 21, 9], TYPE1_LATTICE_MAX).unwrap();
        assert_eq!(
            lattice,
            Lattice {
                x0: 10,
                y0: 5,
                step: 1,
                cols: 12,
                rows: 5,
            }
        );
        assert_eq!(lattice.pixel(11, 4), (21.0, 9.0));
        let [x, y] = lattice.axes(13.0, 6.0);
        assert_eq!((x.lo, x.frac, y.lo, y.frac, y.stride), (3, 0.0, 1, 0.0, 12));
    }

    #[test]
    fn a_large_area_takes_the_least_step_within_the_budget() {
        let lattice = Lattice::covering([0, 0, 10_000, 6_000], TYPE1_LATTICE_MAX).unwrap();
        assert!(lattice.cols * lattice.rows <= TYPE1_LATTICE_MAX);
        let single = Lattice::covering([0, 0, 0, 0], 4).unwrap();
        assert_eq!((single.step, single.cols, single.rows), (1, 1, 1));
        let step = lattice.step;
        let dims = |s: usize| (10_000_usize.div_ceil(s) + 1) * (6_000_usize.div_ceil(s) + 1);
        assert!(
            dims(step - 1) > TYPE1_LATTICE_MAX,
            "step {step} is not the least"
        );
        let (far_x, far_y) = lattice.pixel(lattice.cols - 1, lattice.rows - 1);
        assert!(
            far_x >= 10_000.0 && far_y >= 6_000.0,
            "reaches the far edges"
        );
    }

    /// A thin area far longer than the budget still finds the least step.
    #[test]
    fn a_thin_area_takes_a_step_of_its_length_over_the_budget() {
        let lattice = Lattice::covering([0, 0, 99_999, 1], 100).unwrap();
        assert!(lattice.cols * lattice.rows <= 100);
        assert!(Lattice::covering([0, 0, 99_999, 1], 100).is_some());
        let dims = |s: usize| (99_999_usize.div_ceil(s) + 1) * 2;
        assert!(dims(lattice.step - 1) > 100);
    }

    #[test]
    fn an_empty_area_has_no_lattice() {
        assert_eq!(Lattice::covering([5, 0, 4, 3], 4), None);
        assert_eq!(Lattice::covering([0, 3, 4, 2], 4), None);
    }

    /// The sampled area is the Domain's box cut to the canvas, widened by
    /// one pixel, and to `BBox`; a box beside the canvas stays empty.
    #[test]
    fn the_type1_area_is_cut_to_the_canvas_and_bbox() {
        let domain = [
            (-50.5, -50.5),
            (500.5, -50.5),
            (-50.5, 500.5),
            (500.5, 500.5),
        ];
        assert_eq!(type1_area(&domain, None, CANVAS), [-1, -1, 100, 100]);
        let bbox = [(10.2, 20.0), (30.0, 20.0), (30.0, 40.7), (10.2, 40.7)];
        assert_eq!(type1_area(&domain, Some(&bbox), CANVAS), [10, 20, 30, 41]);
        let left = [(-900.0, 0.0), (-500.0, 0.0), (-900.0, 50.0), (-500.0, 50.0)];
        let [x0, _, x1, _] = type1_area(&left, None, CANVAS);
        assert!(x1 < x0, "a box left of the canvas is empty: {x0}..{x1}");
        let right = [(900.0, 0.0), (1500.0, 0.0), (900.0, 50.0), (1500.0, 50.0)];
        let [x0, _, x1, _] = type1_area(&right, None, CANVAS);
        assert!(x1 < x0, "a box right of the canvas is empty: {x0}..{x1}");
    }
}
