//! Coons (Type 6, PDF §8.7.4.5.7) and tensor-product (Type 7, §8.7.4.5.8)
//! patch meshes.
//!
//! [`decode_patch_mesh`] reads the packed patch records and tessellates each
//! patch adaptively into Gouraud triangles.

use pdf::Dictionary;
use raster::shading::gouraud::GouraudVertex;

use super::mesh::{
    Corner, CornerSource, MIN_SUBDIVISION_EXTENT, TriangleSink, color_delta, extent,
};
use super::stream::{BitReader, VertexFormat, parse_bits_per_flag};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum adaptive subdivision depth.
pub(super) const MAX_PATCH_DEPTH: u8 = 6;

/// Largest distance, in device pixels, of a control point from the
/// corner quadrilateral of a patch drawn as that quadrilateral (see
/// [`is_flat`]).
const PATCH_FLATNESS: f64 = 0.25;

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
pub(super) fn corner_spread<C: Corner>(p: &Patch<C>) -> f64 {
    let corners = [p.color[0][0], p.color[0][1], p.color[1][0], p.color[1][1]];
    let mut max_d = 0.0_f64;
    for i in 0..4 {
        for j in (i + 1)..4 {
            max_d = max_d.max(C::delta(corners[i], corners[j]));
        }
    }
    max_d
}

/// The distance from `p` to the segment `a`–`b`.
fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len_sq = dx.mul_add(dx, dy * dy);
    let t = if len_sq > 0.0 {
        ((p[0] - a[0]).mul_add(dx, (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    f64::hypot(dx.mul_add(-t, p[0] - a[0]), dy.mul_add(-t, p[1] - a[1]))
}

/// Whether `point` lies inside the quadrilateral `quad` (even-odd rule) or
/// within `tol` of one of its sides.
fn within_quad(point: [f64; 2], quad: &[[f64; 2]; 4], tol: f64) -> bool {
    let mut inside = false;
    for k in 0..4 {
        let (start, end) = (quad[k], quad[(k + 1) % 4]);
        if segment_distance(point, start, end) <= tol {
            return true;
        }
        if (start[1] > point[1]) != (end[1] > point[1]) {
            let crossing =
                (end[0] - start[0]).mul_add((point[1] - start[1]) / (end[1] - start[1]), start[0]);
            if point[0] < crossing {
                inside = !inside;
            }
        }
    }
    inside
}

/// Whether the patch `p` is its corner quadrilateral to within
/// [`PATCH_FLATNESS`]: the inner control points of each boundary curve lie
/// within that distance of the side joining the curve's corners, so each
/// curve, inside the hull of its control points, does too; and the
/// interior control points lie in the quadrilateral or within that
/// distance of it, so the patch, inside the hull of all its control
/// points, does not fold out past its sides.  The test is independent of
/// how the control points parametrise the sides.
fn is_flat<C>(p: &Patch<C>) -> bool {
    let xy = &p.xy;
    let quad = [xy[0][0], xy[0][3], xy[3][3], xy[3][0]];
    let sides = [
        (quad[0], quad[1], [xy[0][1], xy[0][2]]),
        (quad[1], quad[2], [xy[1][3], xy[2][3]]),
        (quad[2], quad[3], [xy[3][2], xy[3][1]]),
        (quad[3], quad[0], [xy[2][0], xy[1][0]]),
    ];
    sides.iter().all(|&(a, b, inner)| {
        inner
            .iter()
            .all(|&c| segment_distance(c, a, b) <= PATCH_FLATNESS)
    }) && [xy[1][1], xy[1][2], xy[2][1], xy[2][2]]
        .iter()
        .all(|&c| within_quad(c, &quad, PATCH_FLATNESS))
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
pub(super) fn corner_vertex<const N: usize, S: CornerSource<N>>(
    source: &S,
    p: &Patch<S::Corner>,
    u: usize,
    v: usize,
) -> GouraudVertex<N, S::Sample> {
    let row = u * 3;
    let col = v * 3;
    GouraudVertex {
        x: p.xy[row][col][0],
        y: p.xy[row][col][1],
        color: source.color(p.color[u][v]),
    }
}

/// Adaptively subdivide `patch` into Gouraud triangles and append them to `out`.
///
/// Subdivision stops when all corner colours are within [`color_delta`] of
/// each other, levels of the pixels' samples, and the patch is its corner
/// quadrilateral to within [`PATCH_FLATNESS`] ([`is_flat`]); when the
/// control points span at most [`MIN_SUBDIVISION_EXTENT`] pixels (the patch
/// lies in their convex hull); or when [`MAX_PATCH_DEPTH`] is reached.
/// The work stack is explicit, so the native stack depth is constant.
pub(super) fn tessellate_patch<const N: usize, S: CornerSource<N>>(
    source: &S,
    patch: Patch<S::Corner>,
    out: &mut TriangleSink<N, S::Sample>,
) {
    // Pre-allocate the work stack.  In a DFS traversal of a 4-ary tree of depth D,
    // at most 3*D+1 nodes are simultaneously live on the stack (3 siblings waiting
    // at each level plus the children just pushed).  For MAX_PATCH_DEPTH=6 that is 19;
    // 24 is a safe round-up.
    let mut stack: Vec<(Patch<S::Corner>, u8)> =
        Vec::with_capacity(3 * MAX_PATCH_DEPTH as usize + 1);
    stack.push((patch, 0));
    let limit = color_delta::<S::Sample>();

    while !out.is_full()
        && let Some((p, depth)) = stack.pop()
    {
        if depth >= MAX_PATCH_DEPTH
            || (corner_spread(&p) <= limit && is_flat(&p))
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
pub(super) fn decode_patch_mesh<const N: usize, S: CornerSource<N>>(
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    format: &VertexFormat<'_>,
    kind: PatchKind,
    out: &mut TriangleSink<N, S::Sample>,
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
mod tests;
