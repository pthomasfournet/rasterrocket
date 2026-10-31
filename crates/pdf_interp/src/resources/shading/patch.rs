//! Coons (Type 6, PDF §8.7.4.5.7) and tensor-product (Type 7, §8.7.4.5.8)
//! patch meshes.
//!
//! [`decode_patch_mesh`] reads the packed patch records and tessellates each
//! patch adaptively into Gouraud triangles.

use pdf::Dictionary;
use raster::shading::gouraud::GouraudVertex;

use super::mesh::{
    COLOR_DELTA, Corner, CornerSource, MIN_SUBDIVISION_EXTENT, TriangleSink, extent,
};
use super::stream::{BitReader, VertexFormat, parse_bits_per_flag};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum adaptive subdivision depth.
pub(super) const MAX_PATCH_DEPTH: u8 = 6;

// ── Patch struct ──────────────────────────────────────────────────────────────

/// A bicubic patch: 4×4 control points in device space + 2×2 corner colours.
///
/// Grid convention (PDF §8.7.4.5.6):
/// - `xy[row][col]` where row 0 = "first" edge (u=0), row 3 = "last" edge (u=1).
/// - `color[u_corner][v_corner]` where 0 = min, 1 = max.
///
/// Coordinates are already in device space (CTM applied + y-flipped).
#[derive(Clone, Copy)]
pub(super) struct Patch<C> {
    pub(super) xy: [[[f64; 2]; 4]; 4],
    pub(super) color: [[C; 2]; 2],
}

impl<C: Corner> Patch<C> {
    /// Create a zeroed patch (all points at origin, all corners [`Corner::ZERO`]).
    pub(super) const fn zero() -> Self {
        Self {
            xy: [[[0.0; 2]; 4]; 4],
            color: [[C::ZERO; 2]; 2],
        }
    }
}

// ── Patch math helpers ────────────────────────────────────────────────────────

/// Componentwise midpoint of two device-space points — exact, never overflows.
#[inline]
pub(super) const fn mid2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])]
}

/// Maximum per-component colour difference across all 4 corner pairs of the patch.
/// Used as the subdivision-stop criterion.
pub(super) fn color_delta<C: Corner>(p: &Patch<C>) -> f64 {
    let corners = [p.color[0][0], p.color[0][1], p.color[1][0], p.color[1][1]];
    let mut max_d = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            max_d = max_d.max(C::delta(corners[i], corners[j]));
        }
    }
    max_d
}

// ── Patch subdivision ─────────────────────────────────────────────────────────

/// Apply one step of de Casteljau to a row of 4 points, splitting at t=0.5.
///
/// Returns `(left, right)` where each is a 4-point row.
#[inline]
#[expect(
    clippy::similar_names,
    reason = "m01/m012/m0123 are standard de Casteljau intermediate-point names; renaming obscures the algorithm"
)]
pub(super) const fn casteljau_row(row: [[f64; 2]; 4]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let m01 = mid2(row[0], row[1]);
    let m12 = mid2(row[1], row[2]);
    let m23 = mid2(row[2], row[3]);
    let m012 = mid2(m01, m12);
    let m123 = mid2(m12, m23);
    let m0123 = mid2(m012, m123);
    ([row[0], m01, m012, m0123], [m0123, m123, m23, row[3]])
}

/// Split `p` at u=0.5 — applies de Casteljau along columns (u direction).
///
/// Returns `(lo_u, hi_u)` where `lo_u` covers u∈[0,0.5] and `hi_u` u∈[0.5,1].
pub(super) fn split_patch_u<C: Corner>(p: &Patch<C>) -> (Patch<C>, Patch<C>) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for col in 0..4 {
        let col_pts = [p.xy[0][col], p.xy[1][col], p.xy[2][col], p.xy[3][col]];
        let (l, r) = casteljau_row(col_pts);
        for row in 0..4 {
            lo.xy[row][col] = l[row];
            hi.xy[row][col] = r[row];
        }
    }
    // Interpolate corner colours along u.
    let c_mid_v0 = C::mid(p.color[0][0], p.color[1][0]);
    let c_mid_v1 = C::mid(p.color[0][1], p.color[1][1]);
    lo.color[0] = p.color[0];
    lo.color[1] = [c_mid_v0, c_mid_v1];
    hi.color[0] = [c_mid_v0, c_mid_v1];
    hi.color[1] = p.color[1];
    (lo, hi)
}

/// Split `p` at v=0.5 — applies de Casteljau along rows (v direction).
///
/// Returns `(lo_v, hi_v)` where `lo_v` covers v∈[0,0.5] and `hi_v` v∈[0.5,1].
pub(super) fn split_patch_v<C: Corner>(p: &Patch<C>) -> (Patch<C>, Patch<C>) {
    let mut lo = Patch::zero();
    let mut hi = Patch::zero();
    for row in 0..4 {
        let (l, r) = casteljau_row(p.xy[row]);
        lo.xy[row] = l;
        hi.xy[row] = r;
    }
    // Interpolate corner colours along v.
    let c_mid_u0 = C::mid(p.color[0][0], p.color[0][1]);
    let c_mid_u1 = C::mid(p.color[1][0], p.color[1][1]);
    lo.color[0][0] = p.color[0][0];
    lo.color[0][1] = c_mid_u0;
    lo.color[1][0] = p.color[1][0];
    lo.color[1][1] = c_mid_u1;
    hi.color[0][0] = c_mid_u0;
    hi.color[0][1] = p.color[0][1];
    hi.color[1][0] = c_mid_u1;
    hi.color[1][1] = p.color[1][1];
    (lo, hi)
}

// ── Tessellation ──────────────────────────────────────────────────────────────

/// Extract a corner device-space vertex from a patch, coloured by `source`.
///
/// `u=0` → row 0; `u=1` → row 3. `v=0` → col 0; `v=1` → col 3.
#[inline]
pub(super) fn corner_vertex<S: CornerSource>(
    source: &S,
    p: &Patch<S::Corner>,
    u: usize,
    v: usize,
) -> GouraudVertex {
    let row = u * 3;
    let col = v * 3;
    GouraudVertex {
        x: p.xy[row][col][0],
        y: p.xy[row][col][1],
        color: source.rgb(p.color[u][v]),
    }
}

/// Adaptively subdivide `patch` into Gouraud triangles and append them to `out`.
///
/// Subdivision stops when all corner colours are within [`COLOR_DELTA`] of each
/// other, the control points span at most [`MIN_SUBDIVISION_EXTENT`] pixels
/// (the patch lies in their convex hull), or [`MAX_PATCH_DEPTH`] is reached.
/// The work stack is explicit, so the native stack depth is constant.
pub(super) fn tessellate_patch<S: CornerSource>(
    source: &S,
    patch: Patch<S::Corner>,
    out: &mut TriangleSink,
) {
    // Pre-allocate the work stack.  In a DFS traversal of a 4-ary tree of depth D,
    // at most 3*D+1 nodes are simultaneously live on the stack (3 siblings waiting
    // at each level plus the children just pushed).  For MAX_PATCH_DEPTH=6 that is 19;
    // 24 is a safe round-up.
    let mut stack: Vec<(Patch<S::Corner>, u8)> =
        Vec::with_capacity(3 * MAX_PATCH_DEPTH as usize + 1);
    stack.push((patch, 0));

    while !out.is_full()
        && let Some((p, depth)) = stack.pop()
    {
        if depth >= MAX_PATCH_DEPTH
            || color_delta(&p) <= COLOR_DELTA
            || extent(p.xy.into_iter().flatten()) <= MIN_SUBDIVISION_EXTENT
        {
            // Leaf: emit 2 triangles covering the patch quad from its 4 corners.
            let v00 = corner_vertex(source, &p, 0, 0);
            let v01 = corner_vertex(source, &p, 0, 1);
            let v10 = corner_vertex(source, &p, 1, 0);
            let v11 = corner_vertex(source, &p, 1, 1);
            out.push([v00, v01, v10]);
            out.push([v01, v11, v10]);
        } else {
            // Split into 4 sub-patches and push them.
            let (pu0, pu1) = split_patch_u(&p);
            let (p00, p01) = split_patch_v(&pu0);
            let (p10, p11) = split_patch_v(&pu1);
            let nd = depth + 1;
            stack.push((p00, nd));
            stack.push((p01, nd));
            stack.push((p10, nd));
            stack.push((p11, nd));
        }
    }
}

// ── Patch assembly ────────────────────────────────────────────────────────────

/// Grid positions of the boundary control points in stream order:
/// `p00 p01 p02 p03 p13 p23 p33 p32 p31 p30 p20 p10` (PDF §8.7.4.5.7).
const BOUNDARY: [(usize, usize); 12] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (3, 2),
    (3, 1),
    (3, 0),
    (2, 0),
    (1, 0),
];

/// Grid positions of a tensor patch's interior points in stream order:
/// `p11 p12 p22 p21` (PDF §8.7.4.5.8).
const INTERIOR: [(usize, usize); 4] = [(1, 1), (1, 2), (2, 2), (2, 1)];

/// `Patch::color` positions of the corner colours in stream order, the
/// colours of `p00 p03 p33 p30`: colour `k` belongs to boundary point `3k`.
const CORNERS: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 1), (1, 0)];

/// A patch as its stream records give it: the boundary control points and
/// corner colours in stream order, and the interior points of a tensor
/// patch.
#[derive(Clone, Copy)]
pub(super) struct Outline<C> {
    boundary: [[f64; 2]; 12],
    interior: Option<[[f64; 2]; 4]>,
    colors: [C; 4],
}

impl<C: Corner> Outline<C> {
    /// The outline of a flag-0 record: `points` holds the 12 boundary
    /// points, then for a tensor patch the 4 interior points; `colors`
    /// holds the 4 corner colours.  `None` when a slice is short.
    pub(super) fn new(kind: PatchKind, points: &[[f64; 2]], colors: &[C]) -> Option<Self> {
        Some(Self {
            boundary: points.get(..12)?.try_into().ok()?,
            interior: kind.interior(points.get(12..)?),
            colors: colors.get(..4)?.try_into().ok()?,
        })
    }

    /// The outline of a record with flag `edge` (1–3) that continues this
    /// patch (PDF §8.7.4.5.7): its first four boundary points are this
    /// patch's boundary points `3·edge … 3·edge + 3` (cyclically) and its
    /// first two colours this patch's colours `edge` and `edge + 1`.
    /// `points` holds the other 8 boundary points, then for a tensor patch
    /// the 4 interior points; `colors` holds the other 2 colours.  `None`
    /// when a slice is short.
    pub(super) fn continued(
        &self,
        kind: PatchKind,
        edge: usize,
        points: &[[f64; 2]],
        colors: &[C],
    ) -> Option<Self> {
        let rest = points.get(..8)?;
        let mut boundary = [[0.0; 2]; 12];
        for (i, pt) in boundary.iter_mut().enumerate() {
            *pt = if i < 4 {
                self.boundary[(3 * edge + i) % 12]
            } else {
                rest[i - 4]
            };
        }
        Some(Self {
            boundary,
            interior: kind.interior(points.get(8..)?),
            colors: [
                self.colors[edge % 4],
                self.colors[(edge + 1) % 4],
                *colors.first()?,
                *colors.get(1)?,
            ],
        })
    }

    /// The control grid and corner colours, with a Coons patch's interior
    /// derived from its boundary.
    pub(super) fn patch(&self) -> Patch<C> {
        let mut p = Patch::zero();
        for (&(r, c), &pt) in BOUNDARY.iter().zip(&self.boundary) {
            p.xy[r][c] = pt;
        }
        for (&(u, v), &color) in CORNERS.iter().zip(&self.colors) {
            p.color[u][v] = color;
        }
        match self.interior {
            Some(interior) => {
                for (&(r, c), &pt) in INTERIOR.iter().zip(&interior) {
                    p.xy[r][c] = pt;
                }
            }
            None => fill_type6_interior(&mut p),
        }
        p
    }
}

/// Derive the 4 interior control points of a Coons patch from its boundary.
///
/// Standard Coons patch interior derivation (same formula for x and y independently).
pub(super) fn fill_type6_interior<C>(p: &mut Patch<C>) {
    for k in 0..2usize {
        // Snapshot boundary values before writing interior — avoids borrow conflict.
        let [p00, p01, p02, p03] = [p.xy[0][0][k], p.xy[0][1][k], p.xy[0][2][k], p.xy[0][3][k]];
        let [p10, p13] = [p.xy[1][0][k], p.xy[1][3][k]];
        let [p20, p23] = [p.xy[2][0][k], p.xy[2][3][k]];
        let [p30, p31, p32, p33] = [p.xy[3][0][k], p.xy[3][1][k], p.xy[3][2][k], p.xy[3][3][k]];
        // Coons formula: (-4a + 6(b+c) - 2(d+e) + 3(f+g) - h) / 9
        // Inner mul_add calls use FMA for reduced rounding error.
        p.xy[1][1][k] = ((-4.0f64).mul_add(p00, 6.0 * (p01 + p10))
            + (-2.0f64).mul_add(p03 + p30, 3.0 * (p31 + p13))
            - p33)
            / 9.0;
        p.xy[1][2][k] = ((-4.0f64).mul_add(p03, 6.0 * (p02 + p13))
            + (-2.0f64).mul_add(p00 + p33, 3.0 * (p32 + p10))
            - p30)
            / 9.0;
        p.xy[2][1][k] = ((-4.0f64).mul_add(p30, 6.0 * (p31 + p20))
            + (-2.0f64).mul_add(p33 + p00, 3.0 * (p01 + p23))
            - p03)
            / 9.0;
        p.xy[2][2][k] = ((-4.0f64).mul_add(p33, 6.0 * (p32 + p23))
            + (-2.0f64).mul_add(p30 + p03, 3.0 * (p02 + p20))
            - p00)
            / 9.0;
    }
}

// ── Top-level mesh decoder ────────────────────────────────────────────────────

/// The two patch mesh types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PatchKind {
    /// Type 6: the 12 boundary control points; the interior is derived.
    Coons,
    /// Type 7: all 16 control points.
    Tensor,
}

impl PatchKind {
    const fn shading_type(self) -> u8 {
        match self {
            Self::Coons => 6,
            Self::Tensor => 7,
        }
    }

    /// Control points a record carries: one patch from scratch (flag 0),
    /// or one sharing an edge, its four points, with the previous patch.
    const fn points(self, flag: u8) -> usize {
        let all = match self {
            Self::Coons => 12,
            Self::Tensor => 16,
        };
        if flag == 0 { all } else { all - 4 }
    }

    /// The interior points of a record whose points after the boundary
    /// are `rest`: the first four for a tensor patch, none for a Coons
    /// patch, whose interior derives from its boundary.
    fn interior(self, rest: &[[f64; 2]]) -> Option<[[f64; 2]; 4]> {
        match self {
            Self::Coons => None,
            Self::Tensor => rest.get(..4)?.try_into().ok(),
        }
    }
}

/// Decode a Type 6 or 7 patch mesh stream into `out`.
///
/// Each record (PDF §8.7.4.5.7) is a `BitsPerFlag`-bit flag, then 12 or 16
/// control points and 4 corner colours for flag 0, or 8 or 12 points and 2
/// colours for flags 1–3, which share an edge with the previous patch; every
/// record starts on a byte boundary.  A record with flag `f` in 1–3
/// continues the boundary of the previous patch from its corner `f`.  A
/// record that cannot be placed (a shared-edge flag with no previous patch,
/// or a flag above 3) is skipped.
pub(super) fn decode_patch_mesh<S: CornerSource>(
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    format: &VertexFormat<'_>,
    kind: PatchKind,
    out: &mut TriangleSink,
) {
    let shading_type = kind.shading_type();
    let Some(bpf) = parse_bits_per_flag(sh, shading_type) else {
        return;
    };
    let mut reader = BitReader::new(data);
    let mut fields = vec![0.0; source.components()];
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(16);
    let mut colors: Vec<S::Corner> = Vec::with_capacity(4);
    let mut prev: Option<Outline<S::Corner>> = None;

    while !out.is_full()
        && let Some(flag) = reader.read_bits(bpf)
    {
        // BitsPerFlag is at most 8.
        let Ok(flag) = u8::try_from(flag) else {
            break;
        };
        pts.clear();
        colors.clear();
        let n_colors = if flag == 0 { 4 } else { 2 };
        let complete = (0..kind.points(flag)).all(|_| {
            format
                .read_point(&mut reader)
                .map(|pt| pts.push(pt))
                .is_some()
        }) && (0..n_colors).all(|_| {
            format
                .read_corner(&mut reader, source, &mut fields)
                .map(|c| colors.push(c))
                .is_some()
        });
        if !complete {
            break;
        }
        reader.align();

        let outline = match (flag, &prev) {
            (0, _) => Outline::new(kind, &pts, &colors),
            (1..=3, Some(p)) => p.continued(kind, usize::from(flag), &pts, &colors),
            (1..=3, None) => {
                log::debug!(
                    "shading/type{shading_type}: flag {flag} with no previous patch — skipping record"
                );
                continue;
            }
            _ => {
                log::debug!("shading/type{shading_type}: unknown flag {flag} — skipping record");
                continue;
            }
        };

        let Some(outline) = outline else {
            break;
        };
        tessellate_patch(source, outline.patch(), out);
        prev = Some(outline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::too_many_arguments,
        reason = "test helper: 4 corner positions + 4 corner colours map naturally to 8 args; bundling into a struct just shifts the boilerplate to the call sites"
    )]
    fn flat_patch(
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        c00: [u8; 3],
        c01: [u8; 3],
        c10: [u8; 3],
        c11: [u8; 3],
    ) -> Patch<[u8; 3]> {
        // Bilinear interpolation of control points across the 4×4 grid.
        let mut p = Patch::zero();
        for r in 0..4u32 {
            let tr = f64::from(r) / 3.0;
            for c in 0..4u32 {
                let tc = f64::from(c) / 3.0;
                let (ri, ci) = (r as usize, c as usize);
                p.xy[ri][ci][0] = (x1 - x0).mul_add(tc, x0);
                p.xy[ri][ci][1] = (y1 - y0).mul_add(tr, y0);
            }
        }
        p.color[0][0] = c00;
        p.color[0][1] = c01;
        p.color[1][0] = c10;
        p.color[1][1] = c11;
        p
    }

    #[test]
    fn split_patch_u_flat_corners_unchanged() {
        let p = flat_patch(
            0.0,
            0.0,
            6.0,
            4.0,
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        );
        let (left, right) = split_patch_u(&p);
        // Top-left corner should be exactly preserved in both halves.
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[3][0], p.xy[3][0]);
        // The u-split seam (left's u=1 edge, right's u=0 edge) sits at the
        // y-midpoint of the patch's u=0 / u=3 control rows.
        let mid_y = f64::midpoint(p.xy[0][0][1], p.xy[3][0][1]);
        assert!(
            (left.xy[3][0][1] - mid_y).abs() < 1e-9,
            "left bottom y={} expected={}",
            left.xy[3][0][1],
            mid_y
        );
        assert!(
            (right.xy[0][0][1] - mid_y).abs() < 1e-9,
            "right top y={} expected={}",
            right.xy[0][0][1],
            mid_y
        );
        // Colors: left's u=1 edge should be midpoint of original u edges.
        for ch in 0..3 {
            let expected = u8::midpoint(p.color[0][0][ch], p.color[1][0][ch]);
            assert!(
                (i16::from(left.color[1][0][ch]) - i16::from(expected)).abs() <= 1,
                "color mismatch ch={ch}"
            );
        }
    }

    #[test]
    fn split_patch_v_flat_corners_unchanged() {
        let p = flat_patch(
            0.0,
            0.0,
            8.0,
            6.0,
            [10, 20, 30],
            [200, 100, 50],
            [30, 60, 90],
            [180, 90, 45],
        );
        let (left, right) = split_patch_v(&p);
        assert_eq!(left.xy[0][0], p.xy[0][0]);
        assert_eq!(right.xy[0][3], p.xy[0][3]);
        let mid_x = f64::midpoint(p.xy[0][0][0], p.xy[0][3][0]);
        assert!((left.xy[0][3][0] - mid_x).abs() < 1e-9);
        assert!((right.xy[0][0][0] - mid_x).abs() < 1e-9);
    }

    #[test]
    fn color_delta_uniform_is_zero() {
        let mut p = Patch::zero();
        let c = [128u8, 64, 32];
        p.color[0][0] = c;
        p.color[0][1] = c;
        p.color[1][0] = c;
        p.color[1][1] = c;
        assert_eq!(color_delta(&p), 0.0);
    }

    #[test]
    fn color_delta_max_diff_is_one() {
        let mut p = Patch::zero();
        p.color[0][0] = [0, 0, 0];
        p.color[0][1] = [255, 0, 0];
        p.color[1][0] = [0, 0, 0];
        p.color[1][1] = [0, 0, 0];
        let d = color_delta(&p);
        assert!((d - 1.0).abs() < 1e-9, "expected 1.0, got {d}");
    }

    #[test]
    fn tessellate_patch_uniform_emits_two_triangles() {
        // A patch where all corners have the same colour → depth=0 leaf immediately.
        let p = flat_patch(
            0.0,
            0.0,
            1.0,
            1.0,
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 128],
            [128, 128, 128],
        );
        let doc = crate::test_helpers::empty_doc();
        let source = super::super::mesh::DirectColors {
            cs: super::super::ShadingCs {
                doc: &doc,
                space: &super::super::ColorSpace::DeviceRgb,
            },
        };
        let mut sink = TriangleSink::new();
        tessellate_patch(&source, p, &mut sink);
        assert_eq!(
            sink.finish().len(),
            2,
            "expected exactly 2 triangles for uniform-colour patch"
        );
    }

    #[test]
    fn tessellate_patch_stops_at_the_pixel() {
        // Four distinct corner colours over half a pixel: one leaf.
        let p = flat_patch(
            0.0,
            0.0,
            0.5,
            0.5,
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        );
        let doc = crate::test_helpers::empty_doc();
        let source = super::super::mesh::DirectColors {
            cs: super::super::ShadingCs {
                doc: &doc,
                space: &super::super::ColorSpace::DeviceRgb,
            },
        };
        let mut sink = TriangleSink::new();
        tessellate_patch(&source, p, &mut sink);
        assert_eq!(sink.finish().len(), 2);
    }

    /// The boundary of the square `[x0, x0 + 3] × [y0, y0 + 3]` in stream
    /// order, one control point per unit.
    fn square(x0: f64, y0: f64) -> Vec<[f64; 2]> {
        [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3),
            (1, 3),
            (0, 3),
            (0, 2),
            (0, 1),
        ]
        .map(|(x, y)| [x0 + f64::from(x), y0 + f64::from(y)])
        .to_vec()
    }

    /// Flags 1, 2 and 3 continue the boundary from corners p03, p33 and
    /// p30 of the previous patch, with that corner's colour and the next.
    #[test]
    fn continued_outline_shares_the_flagged_edge() {
        let prev = Outline::new(
            PatchKind::Coons,
            &square(0.0, 0.0),
            &[[10u8; 3], [20; 3], [30; 3], [40; 3]],
        )
        .expect("outline");
        for (edge, shared, colors) in [
            (
                1,
                [[3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [3.0, 3.0]],
                [20, 30],
            ),
            (
                2,
                [[3.0, 3.0], [2.0, 3.0], [1.0, 3.0], [0.0, 3.0]],
                [30, 40],
            ),
            (
                3,
                [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
                [40, 10],
            ),
        ] {
            let next = prev
                .continued(
                    PatchKind::Coons,
                    edge,
                    &[[9.0, 9.0]; 8],
                    &[[50u8; 3], [60; 3]],
                )
                .expect("outline");
            assert_eq!(next.boundary[..4], shared, "flag {edge}");
            assert_eq!(
                next.colors,
                [[colors[0]; 3], [colors[1]; 3], [50; 3], [60; 3]],
                "flag {edge}"
            );
        }
    }

    /// A Coons patch beside a unit square, sharing its right edge: the
    /// continuation record lists the rest of the boundary from the end of
    /// the shared edge, so the patch is the square `[3, 6] × [0, 3]`.
    #[test]
    fn a_flag_1_record_continues_from_the_shared_edge() {
        let mut sh = Dictionary::new();
        for (k, v) in [
            ("BitsPerCoordinate", 8),
            ("BitsPerComponent", 8),
            ("BitsPerFlag", 8),
        ] {
            sh.set(k, pdf::Object::Integer(v));
        }
        let coord = |pts: &[[f64; 2]]| -> Vec<u8> {
            pts.iter()
                .flatten()
                .map(|&v| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "test coordinates are small non-negative integers"
                    )]
                    let b = v as u8;
                    b
                })
                .collect()
        };
        let mut data = vec![0];
        data.extend(coord(&square(0.0, 0.0)));
        data.extend([128; 4]);
        data.push(1);
        data.extend(coord(&[
            [4.0, 3.0],
            [5.0, 3.0],
            [6.0, 3.0],
            [6.0, 2.0],
            [6.0, 1.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [4.0, 0.0],
        ]));
        data.extend([128; 2]);

        let doc = crate::test_helpers::empty_doc();
        let source = super::super::mesh::DirectColors {
            cs: super::super::ShadingCs {
                doc: &doc,
                space: &super::super::ColorSpace::DeviceGray,
            },
        };
        let decode = [0.0, 255.0, 0.0, 255.0, 0.0, 1.0];
        // page_h = 0: device y is the negated shading y.
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let format = VertexFormat::parse(&sh, 6, &decode, &ctm, 0.0).expect("format");
        let mut sink = TriangleSink::new();
        decode_patch_mesh(&sh, &data, &source, &format, PatchKind::Coons, &mut sink);
        let tris = sink.finish();
        assert_eq!(tris.len(), 4, "two uniform patches, two triangles each");
        let corners = |t: &[GouraudVertex; 3]| t.map(|v| [v.x, -v.y]);
        // Leaf triangles are (p00, p03, p30) and (p03, p33, p30).
        assert_eq!(
            corners(&tris[2]),
            [[3.0, 0.0], [3.0, 3.0], [6.0, 0.0]],
            "the second patch covers [3, 6] × [0, 3]"
        );
        assert_eq!(corners(&tris[3]), [[3.0, 3.0], [6.0, 3.0], [6.0, 0.0]]);
    }

    #[test]
    fn build_patch_type6_interior_formula() {
        // Use a simple square boundary with known analytic interior.
        // p[0][0]=(0,0)  p[0][1]=(1,0) p[0][2]=(2,0) p[0][3]=(3,0)
        // p[1][3]=(3,1)  p[2][3]=(3,2)
        // p[3][3]=(3,3)  p[3][2]=(2,3) p[3][1]=(1,3) p[3][0]=(0,3)
        // p[2][0]=(0,2)  p[1][0]=(0,1)
        //
        // For a uniform-spaced rectilinear grid the Coons interior formula collapses to
        // simple bilinear: p[1][1]=(1,1), p[1][2]=(2,1), p[2][1]=(1,2), p[2][2]=(2,2).
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0], // p[0][0..3]
            [3.0, 1.0],
            [3.0, 2.0], // p[1][3], p[2][3]
            [3.0, 3.0],
            [2.0, 3.0],
            [1.0, 3.0],
            [0.0, 3.0], // p[3][3..0]
            [0.0, 2.0],
            [0.0, 1.0], // p[2][0], p[1][0]
        ];
        let colors: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
        let p = Outline::new(PatchKind::Coons, &pts, &colors)
            .expect("12 points and 4 colours")
            .patch();
        for (r, c, ex, ey) in [
            (1, 1, 1.0, 1.0),
            (1, 2, 2.0, 1.0),
            (2, 1, 1.0, 2.0),
            (2, 2, 2.0, 2.0),
        ] {
            assert!(
                (p.xy[r][c][0] - ex).abs() < 1e-9 && (p.xy[r][c][1] - ey).abs() < 1e-9,
                "interior[{r}][{c}] = {:?} expected ({ex},{ey})",
                p.xy[r][c]
            );
        }
    }
}
