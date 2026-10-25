//! Vertex colours of mesh shadings (types 4–7).
//!
//! A mesh vertex carries either the shading colour space's components or,
//! when the shading has a `Function`, one parametric value `t` that the
//! function maps to colour components (PDF §8.7.4.5.5–8).  The two kinds are
//! [`CornerSource`] implementations: [`DirectColors`] converts components to
//! sRGB at the vertex and interpolates in sRGB; [`ParamColors`] carries the
//! parametric value through interpolation and subdivision and converts it
//! last, so the colour between vertices follows the function rather than the
//! sRGB chord of its end values.

use raster::shading::ColorRamp;
use raster::shading::gouraud::GouraudVertex;

use super::ShadingCs;

/// Subdivision stops once every pair of corner colours differs by at most
/// this much per component (each component normalised to `[0, 1]`).
pub(super) const COLOR_DELTA: f64 = 3.0 / 255.0;

/// Maximum subdivision depth of one mesh triangle.
const MAX_TRIANGLE_DEPTH: u8 = 6;

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

impl Corner for [u8; 3] {
    const ZERO: Self = [0; 3];

    fn mid(a: Self, b: Self) -> Self {
        // lerp_u8(a, b, 128) = round((a + b) / 2).
        use color::convert::lerp_u8;
        [
            lerp_u8(a[0], b[0], 128),
            lerp_u8(a[1], b[1], 128),
            lerp_u8(a[2], b[2], 128),
        ]
    }

    fn delta(a: Self, b: Self) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, y)| f64::from(x.abs_diff(y)) / 255.0)
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
/// corner becomes sRGB.
pub(super) trait CornerSource {
    /// The corner value this source produces.
    type Corner: Corner;

    /// Whether a mesh triangle is subdivided until its corners are within
    /// [`COLOR_DELTA`] before it is emitted.
    const SUBDIVIDE_TRIANGLES: bool;

    /// Number of colour fields each vertex carries in the stream.
    fn components(&self) -> usize;

    /// The corner value of a vertex whose colour fields decode to
    /// `components` (each mapped through its `Decode` range) and whose raw
    /// fields, divided by their maximum, are `normalised`.
    fn corner(&self, components: &[f64], normalised: &[f64]) -> Self::Corner;

    /// The sRGB colour of `corner`.
    fn rgb(&self, corner: Self::Corner) -> [u8; 3];
}

/// Vertices carrying the colour space's components, converted at the vertex.
pub(super) struct DirectColors<'a> {
    pub(super) cs: ShadingCs<'a>,
}

impl CornerSource for DirectColors<'_> {
    type Corner = [u8; 3];
    const SUBDIVIDE_TRIANGLES: bool = false;

    fn components(&self) -> usize {
        self.cs.channels()
    }

    fn corner(&self, components: &[f64], _normalised: &[f64]) -> [u8; 3] {
        self.cs.to_rgb(components)
    }

    fn rgb(&self, corner: [u8; 3]) -> [u8; 3] {
        corner
    }
}

/// Vertices carrying one parametric value, coloured through `ramp`, which
/// samples the shading function across the `Decode` range of `t`.
pub(super) struct ParamColors {
    pub(super) ramp: ColorRamp,
}

impl CornerSource for ParamColors {
    type Corner = Param;
    const SUBDIVIDE_TRIANGLES: bool = true;

    fn components(&self) -> usize {
        1
    }

    fn corner(&self, _components: &[f64], normalised: &[f64]) -> Param {
        Param(normalised.first().copied().unwrap_or(0.0))
    }

    fn rgb(&self, corner: Param) -> [u8; 3] {
        let mut out = [0u8; 3];
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
/// edge midpoints into four until its corners are within [`COLOR_DELTA`] or
/// [`MAX_TRIANGLE_DEPTH`] is reached; the corner values interpolate linearly
/// across the triangle, so the midpoints are exact.
pub(super) fn emit_triangle<S: CornerSource>(
    source: &S,
    tri: [MeshVertex<S::Corner>; 3],
    out: &mut Vec<[GouraudVertex; 3]>,
) {
    let mut stack = vec![(tri, 0u8)];
    while let Some(([a, b, c], depth)) = stack.pop() {
        let spread = S::Corner::delta(a.corner, b.corner)
            .max(S::Corner::delta(b.corner, c.corner))
            .max(S::Corner::delta(c.corner, a.corner));
        if !S::SUBDIVIDE_TRIANGLES || depth >= MAX_TRIANGLE_DEPTH || spread <= COLOR_DELTA {
            out.push([a, b, c].map(|v| GouraudVertex {
                x: v.x,
                y: v.y,
                color: source.rgb(v.corner),
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
            ramp: ColorRamp::from_samples(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]])
                .expect("ramp"),
        };
        let mut out = Vec::new();
        emit_triangle(
            &source,
            [
                vertex(0.0, 0.0, 0.0),
                vertex(64.0, 0.0, 1.0),
                vertex(0.0, 64.0, 0.0),
            ],
            &mut out,
        );
        assert!(out.len() > 1, "a full-range triangle subdivides");
        // Some emitted vertex sits at s = 0.5, where the ramp is green.
        assert!(
            out.iter()
                .flatten()
                .any(|v| (v.x - 32.0).abs() < 1e-9 && v.y == 0.0 && v.color == [0, 255, 0]),
            "the edge midpoint takes the middle stop"
        );
    }

    #[test]
    fn direct_triangle_is_emitted_whole() {
        let source = DirectColors {
            cs: ShadingCs {
                doc: &crate::test_helpers::empty_doc(),
                space: &super::super::ColorSpace::DeviceRgb,
            },
        };
        let v = |x, color| MeshVertex {
            x,
            y: 0.0,
            corner: color,
        };
        let mut out = Vec::new();
        emit_triangle(
            &source,
            [
                v(0.0, [255, 0, 0]),
                v(9.0, [0, 0, 255]),
                v(4.0, [0, 255, 0]),
            ],
            &mut out,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0][1].color, [0, 0, 255]);
    }
}
