//! Vertex colours of mesh shadings (types 4–7).
//!
//! A mesh vertex carries either the shading colour space's components or,
//! when the shading has a `Function`, one parametric value `t` that the
//! function maps to colour components (PDF §8.7.4.5.5–8).  The two kinds are
//! [`CornerSource`] implementations: [`DirectColors`] converts components to
//! pixel bytes (sRGB, or ink) at the vertex and interpolates them;
//! [`ParamColors`] carries the parametric value through interpolation and
//! subdivision and converts it last, so the colour between vertices follows
//! the function rather than the chord of its end values.

use raster::shading::ColorRamp;
use raster::shading::gouraud::GouraudVertex;

use color::Sample;

use super::ShadingColour;

/// Subdivision tolerance on 8-bit canvases: three 8-bit levels.
const COLOR_DELTA_8: f64 = 3.0 / 255.0;

/// Subdivision tolerance on 16-bit canvases: a quarter of one 8-bit level
/// (about 64 sixteen-bit levels).  The linear interpolation inside a
/// triangle then deviates from the patch colour by less than the 8-bit
/// quantum, while the triangle count of a patch grows about twelvefold over
/// the 8-bit tolerance rather than reaching the per-patch subdivision limit.
const COLOR_DELTA_16: f64 = 1.0 / 1020.0;

/// Subdivision stops once every pair of corner colours differs by at most
/// this much per component, on the `[0, 1]` scale of a normalised
/// component: [`COLOR_DELTA_8`] for samples of up to 8 bits,
/// [`COLOR_DELTA_16`] for wider ones.
pub(super) fn color_delta<S: Sample>() -> f64 {
    if S::MAX.signed() > 255 {
        COLOR_DELTA_16
    } else {
        COLOR_DELTA_8
    }
}

/// Maximum subdivision depth of one mesh triangle.
const MAX_TRIANGLE_DEPTH: u8 = 6;

/// Subdivision of a mesh triangle or patch stops once its control points
/// span at most this many device pixels along both axes.
pub(super) const MIN_SUBDIVISION_EXTENT: f64 = 1.0;

/// Upper bound on the triangles of one mesh shading.  A patch record of a
/// few bytes subdivides into up to `2 · 4^6` triangles, so the stream length
/// does not bound the mesh; at about 100 bytes per indexed triangle this
/// bounds one mesh near 400 MiB.
pub(super) const MAX_MESH_TRIANGLES: usize = 1 << 22;

/// The triangles of a mesh being decoded, of colours of `N` samples of type
/// `S`, holding at most a fixed number.
pub(super) struct TriangleSink<const N: usize = 3, S: Sample = u8> {
    triangles: Vec<[GouraudVertex<N, S>; 3]>,
    limit: usize,
    full: bool,
}

impl<const N: usize, S: Sample> TriangleSink<N, S> {
    /// A sink holding at most [`MAX_MESH_TRIANGLES`] triangles.
    pub(super) const fn new() -> Self {
        Self::with_limit(MAX_MESH_TRIANGLES)
    }

    /// A sink holding at most `limit` triangles.
    pub(super) const fn with_limit(limit: usize) -> Self {
        Self {
            triangles: Vec::new(),
            limit,
            full: false,
        }
    }

    /// Append `triangle`, or drop it and mark the sink full when the sink
    /// holds its limit.
    pub(super) fn push(&mut self, triangle: [GouraudVertex<N, S>; 3]) {
        if self.triangles.len() < self.limit {
            self.triangles.push(triangle);
        } else {
            self.full = true;
        }
    }

    /// Whether a triangle has been dropped; decoding stops at this point.
    pub(super) const fn is_full(&self) -> bool {
        self.full
    }

    /// The triangles held, with a warning when some were dropped.
    pub(super) fn finish(self) -> Vec<[GouraudVertex<N, S>; 3]> {
        if self.full {
            log::warn!(
                "shading: mesh exceeds {} triangles — the remainder is not painted",
                self.limit
            );
        }
        self.triangles
    }
}

/// The larger of the x and y extents of `points`.
pub(super) fn extent(points: impl IntoIterator<Item = [f64; 2]>) -> f64 {
    let [x0, y0, x1, y1] = points.into_iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |[x0, y0, x1, y1], [x, y]| [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
    );
    (x1 - x0).max(y1 - y0)
}

/// The colour value held at a mesh corner while the mesh is decoded,
/// subdivided and interpolated.
pub(super) trait Corner: Copy {
    /// The value of a zero-initialised patch corner.
    const ZERO: Self;

    /// The value halfway between `a` and `b`.
    fn mid(a: Self, b: Self) -> Self;

    /// Largest per-component difference between `a` and `b`, in `[0, 1]`.
    fn delta(a: Self, b: Self) -> f64;
}

impl<const N: usize, S: Sample> Corner for [S; N] {
    const ZERO: Self = [S::ZERO; N];

    fn mid(a: Self, b: Self) -> Self {
        // lerp at half of LERP_ONE is round((a + b) / 2).
        std::array::from_fn(|i| S::lerp(a[i], b[i], S::LERP_ONE / 2))
    }

    fn delta(a: Self, b: Self) -> f64 {
        let level = |v: i64| f64::from(u32::try_from(v.unsigned_abs()).unwrap_or(u32::MAX));
        let max = level(S::MAX.signed());
        a.iter()
            .zip(b)
            .map(|(&x, y)| level(x.signed() - y.signed()) / max)
            .fold(0.0, f64::max)
    }
}

/// A parametric value normalised to its `Decode` range: `0` at the range's
/// first bound, `1` at its second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Param(pub(super) f64);

impl Corner for Param {
    const ZERO: Self = Self(0.0);

    fn mid(a: Self, b: Self) -> Self {
        Self(f64::midpoint(a.0, b.0))
    }

    fn delta(a: Self, b: Self) -> f64 {
        (a.0 - b.0).abs()
    }
}

/// How the colour data of a mesh vertex becomes a [`Corner`] and how a
/// corner becomes the `N` samples of a pixel: sRGB for `N = 3`, ink for
/// `N = 8`.
pub(super) trait CornerSource<const N: usize = 3> {
    /// The corner value this source produces.
    type Corner: Corner;
    /// The sample type of the pixels.
    type Sample: Sample;

    /// Whether a mesh triangle is subdivided until its corners are within
    /// [`color_delta`] before it is emitted.
    const SUBDIVIDE_TRIANGLES: bool;

    /// Number of colour fields each vertex carries in the stream.
    fn components(&self) -> usize;

    /// The corner value of a vertex whose colour fields decode to
    /// `components` (each mapped through its `Decode` range) and whose raw
    /// fields, divided by their maximum, are `normalised`.
    fn corner(&self, components: &[f64], normalised: &[f64]) -> Self::Corner;

    /// The pixel samples of `corner`.
    fn color(&self, corner: Self::Corner) -> [Self::Sample; N];
}

/// Vertices carrying the colour space's components, converted at the vertex
/// through `cs` to samples of type `S`.
pub(super) struct DirectColors<C, S> {
    pub(super) cs: C,
    pub(super) sample: std::marker::PhantomData<fn() -> S>,
}

impl<C, S> DirectColors<C, S> {
    /// Vertices converted through `cs`.
    pub(super) const fn new(cs: C) -> Self {
        Self {
            cs,
            sample: std::marker::PhantomData,
        }
    }
}

impl<const N: usize, S: Sample, C: ShadingColour<N, S>> CornerSource<N> for DirectColors<C, S> {
    type Corner = [S; N];
    type Sample = S;
    const SUBDIVIDE_TRIANGLES: bool = false;

    fn components(&self) -> usize {
        self.cs.channels()
    }

    fn corner(&self, components: &[f64], _normalised: &[f64]) -> [S; N] {
        self.cs.convert(components)
    }

    fn color(&self, corner: [S; N]) -> [S; N] {
        corner
    }
}

/// Vertices carrying one parametric value, coloured through `ramp`, which
/// samples the shading function across the `Decode` range of `t`.
pub(super) struct ParamColors<const N: usize = 3, S: Sample = u8> {
    pub(super) ramp: ColorRamp<N, S>,
}

impl<const N: usize, S: Sample> CornerSource<N> for ParamColors<N, S> {
    type Corner = Param;
    type Sample = S;
    const SUBDIVIDE_TRIANGLES: bool = true;

    fn components(&self) -> usize {
        1
    }

    fn corner(&self, _components: &[f64], normalised: &[f64]) -> Param {
        Param(normalised.first().copied().unwrap_or(0.0))
    }

    fn color(&self, corner: Param) -> [S; N] {
        let mut out = [S::default(); N];
        self.ramp.write(corner.0, &mut out);
        out
    }
}

/// A decoded mesh vertex in device space.
#[derive(Clone, Copy, Debug)]
pub(super) struct MeshVertex<C> {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) corner: C,
}

impl<C: Corner> MeshVertex<C> {
    fn mid(a: Self, b: Self) -> Self {
        Self {
            x: f64::midpoint(a.x, b.x),
            y: f64::midpoint(a.y, b.y),
            corner: C::mid(a.corner, b.corner),
        }
    }
}

/// Append triangle `tri` to `out` as Gouraud triangles.
///
/// With [`CornerSource::SUBDIVIDE_TRIANGLES`] the triangle splits at its
/// edge midpoints into four until its corners are within [`color_delta`],
/// it spans at most [`MIN_SUBDIVISION_EXTENT`] pixels, or
/// [`MAX_TRIANGLE_DEPTH`] is reached; the corner values interpolate linearly
/// across the triangle, so the midpoints are exact.
pub(super) fn emit_triangle<const N: usize, S: CornerSource<N>>(
    source: &S,
    tri: [MeshVertex<S::Corner>; 3],
    out: &mut TriangleSink<N, S::Sample>,
) {
    let limit = color_delta::<S::Sample>();
    let mut stack = vec![(tri, 0u8)];
    while !out.is_full()
        && let Some(([a, b, c], depth)) = stack.pop()
    {
        let spread = S::Corner::delta(a.corner, b.corner)
            .max(S::Corner::delta(b.corner, c.corner))
            .max(S::Corner::delta(c.corner, a.corner));
        if !S::SUBDIVIDE_TRIANGLES
            || depth >= MAX_TRIANGLE_DEPTH
            || spread <= limit
            || extent([a, b, c].map(|v| [v.x, v.y])) <= MIN_SUBDIVISION_EXTENT
        {
            out.push([a, b, c].map(|v| GouraudVertex {
                x: v.x,
                y: v.y,
                color: source.color(v.corner),
            }));
            continue;
        }
        let (ab, bc, ca) = (
            MeshVertex::mid(a, b),
            MeshVertex::mid(b, c),
            MeshVertex::mid(c, a),
        );
        let nd = depth + 1;
        stack.push(([a, ab, ca], nd));
        stack.push(([ab, b, bc], nd));
        stack.push(([ca, bc, c], nd));
        stack.push(([ab, bc, ca], nd));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f64, y: f64, s: f64) -> MeshVertex<Param> {
        MeshVertex {
            x,
            y,
            corner: Param(s),
        }
    }

    #[test]
    fn rgb_corner_delta_is_the_largest_channel_difference() {
        assert_eq!(<[u8; 3]>::delta([0, 0, 0], [0, 255, 51]), 1.0);
        assert_eq!(<[u8; 3]>::delta([7, 7, 7], [7, 7, 7]), 0.0);
    }

    #[test]
    fn parametric_triangle_subdivides_and_follows_the_ramp() {
        // Red → green → blue ramp; a triangle spanning the whole range.
        let source = ParamColors {
            ramp: ColorRamp::from_samples(vec![[255u8, 0, 0], [0, 255, 0], [0, 0, 255]])
                .expect("ramp"),
        };
        let mut sink = TriangleSink::new();
        emit_triangle(
            &source,
            [
                vertex(0.0, 0.0, 0.0),
                vertex(64.0, 0.0, 1.0),
                vertex(0.0, 64.0, 0.0),
            ],
            &mut sink,
        );
        let out = sink.finish();
        assert!(out.len() > 1, "a full-range triangle subdivides");
        // Some emitted vertex sits at s = 0.5, where the ramp is green.
        assert!(
            out.iter()
                .flatten()
                .any(|v| (v.x - 32.0).abs() < 1e-9 && v.y == 0.0 && v.color == [0, 255, 0]),
            "the edge midpoint takes the middle stop"
        );
    }

    fn full_range_ramp() -> ParamColors {
        ParamColors {
            ramp: ColorRamp::from_samples(vec![[255, 0, 0], [0, 0, 255]]).expect("ramp"),
        }
    }

    #[test]
    fn a_sub_pixel_triangle_is_not_subdivided() {
        let mut sink = TriangleSink::new();
        emit_triangle(
            &full_range_ramp(),
            [
                vertex(0.0, 0.0, 0.0),
                vertex(0.75, 0.0, 1.0),
                vertex(0.0, 0.75, 0.5),
            ],
            &mut sink,
        );
        assert_eq!(sink.finish().len(), 1);
    }

    #[test]
    fn subdivision_stops_when_the_sink_is_full() {
        let mut sink = TriangleSink::with_limit(10);
        emit_triangle(
            &full_range_ramp(),
            [
                vertex(0.0, 0.0, 0.0),
                vertex(512.0, 0.0, 1.0),
                vertex(0.0, 512.0, 0.5),
            ],
            &mut sink,
        );
        assert!(sink.is_full());
        assert_eq!(sink.finish().len(), 10);
    }

    #[test]
    fn direct_triangle_is_emitted_whole() {
        let doc = crate::test_helpers::empty_doc();
        let source = DirectColors::new(super::super::ShadingCs {
            doc: &doc,
            space: &super::super::ColorSpace::DeviceRgb,
        });
        let v = |x, color| MeshVertex {
            x,
            y: 0.0,
            corner: color,
        };
        let mut sink = TriangleSink::new();
        emit_triangle(
            &source,
            [
                v(0.0, [255u8, 0, 0]),
                v(9.0, [0, 0, 255]),
                v(4.0, [0, 255, 0]),
            ],
            &mut sink,
        );
        let out = sink.finish();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0][1].color, [0, 0, 255]);
    }
}
