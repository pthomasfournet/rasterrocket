//! Evaluation of an ICC profile's PCS → device tables at given values.
//!
//! A [`ProfileLut`] evaluates one `BToA` table (ICC.1:2022 §10.8, §10.10,
//! §10.12) in floating point at the values it is given, with no
//! resampling of its own: `lut8Type` and `lut16Type` as matrix (on PCS XYZ
//! input), input tables, CLUT and output tables; `lutBtoAType` as B
//! curves, matrix, M curves, CLUT and A curves.  One-
//! dimensional tables interpolate linearly; a CLUT interpolates
//! multilinearly in its cell, trilinearly for the three PCS inputs as
//! colour-management modules do for a table indexed by Lab or XYZ.  Every
//! stage's output is clamped to `[0, 1]`.
//!
//! The PCS side is encoded as the table type stores it: Lab as `L / 100`,
//! `(a + 128) / 255`, `(b + 128) / 255`, scaled by `65280 / 65535` in a
//! `lut16Type` (the legacy encoding); XYZ as `X · 32768 / 65535`
//! (`u1Fixed15Number`).  [`ProfileLut::to_device`] takes media-relative
//! D50 XYZ.

use moxcms::{ColorProfile, DataColorSpace, LutStore, LutType, LutWarehouse, RenderingIntent};

use crate::pcs::D50;
use crate::resources::icc_shaper::eval_trc;

/// The `BToA` table `intent` takes in `profile`: that of the intent (the
/// colorimetric one for both colorimetric intents), else the perceptual
/// one (ICC.1:2022 §8.10.2).
pub(crate) const fn table_of(
    profile: &ColorProfile,
    intent: RenderingIntent,
) -> Option<&LutWarehouse> {
    let (perceptual, colorimetric, saturation) = (
        &profile.lut_b_to_a_perceptual,
        &profile.lut_b_to_a_colorimetric,
        &profile.lut_b_to_a_saturation,
    );
    let own = match intent {
        RenderingIntent::Perceptual => perceptual,
        RenderingIntent::RelativeColorimetric | RenderingIntent::AbsoluteColorimetric => {
            colorimetric
        }
        RenderingIntent::Saturation => saturation,
    };
    match own {
        Some(lut) => Some(lut),
        None => perceptual.as_ref(),
    }
}

/// A one-dimensional curve: a sampled table or a tone curve.
#[derive(Debug, Clone)]
enum Curve {
    /// Samples spanning `[0, 1]`, interpolated linearly.
    Table(Vec<f64>),
    /// An ICC tone curve.
    Trc(moxcms::ToneReprCurve),
}

impl Curve {
    fn eval(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Self::Table(t) => interpolate(t, x),
            Self::Trc(c) => eval_trc(c, x).unwrap_or(x),
        }
        .clamp(0.0, 1.0)
    }
}

/// Linear interpolation in `table`, at least one sample, spanning
/// `[0, 1]`, at `x ∈ [0, 1]`.
fn interpolate(table: &[f64], x: f64) -> f64 {
    match table {
        [] => x,
        [v] => *v,
        _ => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a curve has far fewer than 2^52 samples"
            )]
            let pos = x * (table.len() - 1) as f64;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "pos lies in [0, len - 1]"
            )]
            let lo = (pos.floor() as usize).min(table.len() - 2);
            #[expect(clippy::cast_precision_loss, reason = "lo < len")]
            let f = pos - lo as f64;
            f.mul_add(table[lo + 1] - table[lo], table[lo])
        }
    }
}

/// The values of `store`, each in `[0, 1]`.
fn values(store: &LutStore) -> Vec<f64> {
    match store {
        LutStore::Store8(v) => v.iter().map(|&x| f64::from(x) / 255.0).collect(),
        LutStore::Store16(v) => v.iter().map(|&x| f64::from(x) / 65535.0).collect(),
    }
}

/// A colour lookup table of `outputs` values at each node of a grid of
/// `grid[i]` nodes along input `i`, the first input varying slowest.
#[derive(Debug, Clone)]
struct Clut {
    grid: Vec<usize>,
    outputs: usize,
    nodes: Vec<f64>,
}

impl Clut {
    /// The table of `nodes`, `None` when their count is not the grid's.
    fn new(grid: Vec<usize>, outputs: usize, nodes: Vec<f64>) -> Option<Self> {
        let count = grid.iter().try_fold(outputs, |n, &g| {
            (g >= 2).then(|| n.checked_mul(g)).flatten()
        })?;
        (nodes.len() == count).then_some(Self {
            grid,
            outputs,
            nodes,
        })
    }

    /// The multilinear interpolation of the cell around `input` into
    /// `out`.
    fn eval(&self, input: &[f64], out: &mut [f64]) {
        let n = self.grid.len();
        let mut base = 0usize;
        let mut stride = self.outputs;
        let mut cell = [(0usize, 0.0f64); 8];
        for i in (0..n).rev() {
            let g = self.grid[i];
            #[expect(clippy::cast_precision_loss, reason = "a grid has at most 255 nodes")]
            let pos = input[i].clamp(0.0, 1.0) * (g - 1) as f64;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "pos lies in [0, g - 1]"
            )]
            let lo = (pos.floor() as usize).min(g - 2);
            #[expect(clippy::cast_precision_loss, reason = "lo < 255")]
            let f = pos - lo as f64;
            base += lo * stride;
            cell[i] = (stride, f);
            stride *= g;
        }
        out.fill(0.0);
        for corner in 0..1usize << n {
            let mut weight = 1.0;
            let mut at = base;
            for (i, &(stride, f)) in cell.iter().enumerate().take(n) {
                if corner >> i & 1 == 1 {
                    weight *= f;
                    at += stride;
                } else {
                    weight *= 1.0 - f;
                }
            }
            if weight == 0.0 {
                continue;
            }
            for (o, &v) in out.iter_mut().zip(&self.nodes[at..at + self.outputs]) {
                *o = weight.mul_add(v, *o);
            }
        }
    }
}

/// An affine map `M · x + b` of three values.
#[derive(Debug, Clone, Copy)]
struct Affine {
    m: [[f64; 3]; 3],
    b: [f64; 3],
}

impl Affine {
    fn apply(&self, x: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|r| {
            let [a, b, c] = self.m[r];
            a.mul_add(x[0], b.mul_add(x[1], c.mul_add(x[2], self.b[r])))
                .clamp(0.0, 1.0)
        })
    }

    /// Whether the map is the identity within the resolution of the
    /// `s15Fixed16Number` it is stored in.
    fn is_identity(&self) -> bool {
        const EPS: f64 = 0.5 / 65536.0;
        self.b.iter().all(|b| b.abs() < EPS)
            && self.m.iter().enumerate().all(|(r, row)| {
                row.iter()
                    .enumerate()
                    .all(|(c, &v)| (v - f64::from(u8::from(r == c))).abs() < EPS)
            })
    }
}

/// The stages of a table between its device side and its PCS side, in
/// the order they apply.
#[derive(Debug, Clone)]
enum Stage {
    /// One curve per channel.
    Curves(Vec<Curve>),
    /// An affine map of three channels.
    Affine(Affine),
    /// A colour lookup table.
    Clut(Clut),
}

/// One `BToA` table of a profile, evaluated at given values.
#[derive(Debug, Clone)]
pub(crate) struct ProfileLut {
    /// The PCS of the profile.
    pcs: DataColorSpace,
    /// Whether Lab is encoded the legacy way of `lut16Type`.
    legacy_lab: bool,
    stages: Vec<Stage>,
}

impl ProfileLut {
    /// The `BToA` table `intent` takes in `profile` ([`table_of`]) for a
    /// profile of `device` channels; `None` when the profile has no such
    /// table, its PCS is neither Lab nor XYZ, or the table is malformed.
    pub(crate) fn new(
        profile: &ColorProfile,
        intent: RenderingIntent,
        device: usize,
    ) -> Option<Self> {
        if !matches!(profile.pcs, DataColorSpace::Lab | DataColorSpace::Xyz) {
            return None;
        }
        let (stages, legacy_lab) = match table_of(profile, intent)? {
            LutWarehouse::Lut(lut) => (
                lut_stages(lut, device, profile.pcs)?,
                lut.lut_type == LutType::Lut16,
            ),
            LutWarehouse::Multidimensional(m) => (mab_stages(m, device)?, false),
        };
        Some(Self {
            pcs: profile.pcs,
            legacy_lab,
            stages,
        })
    }

    /// The device values of the media-relative D50 XYZ `xyz` into `out`,
    /// one per device channel, each in `[0, 1]`.
    pub(crate) fn to_device(&self, xyz: [f64; 3], out: &mut [f64]) {
        let input = encode(xyz, self.pcs, self.legacy_lab);
        self.run(&input, out);
    }

    /// The stages applied to `input` into `out`.
    fn run(&self, input: &[f64], out: &mut [f64]) {
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        let mut len = input.len();
        a[..len].copy_from_slice(input);
        for stage in &self.stages {
            match stage {
                Stage::Curves(curves) => {
                    for (v, c) in a.iter_mut().zip(curves) {
                        *v = c.eval(*v);
                    }
                }
                Stage::Affine(m) => {
                    let y = m.apply([a[0], a[1], a[2]]);
                    a[..3].copy_from_slice(&y);
                }
                Stage::Clut(clut) => {
                    clut.eval(&a[..len], &mut b[..clut.outputs]);
                    len = clut.outputs;
                    for v in &mut b[..len] {
                        *v = v.clamp(0.0, 1.0);
                    }
                    a[..len].copy_from_slice(&b[..len]);
                }
            }
        }
        out.copy_from_slice(&a[..out.len()]);
    }
}

/// The stages of a `lut8Type` or `lut16Type` `BToA` table to `outputs`
/// device channels.
fn lut_stages(
    lut: &moxcms::LutDataType,
    outputs: usize,
    pcs: DataColorSpace,
) -> Option<Vec<Stage>> {
    let inputs = 3;
    if usize::from(lut.num_input_channels) != inputs
        || usize::from(lut.num_output_channels) != outputs
    {
        return None;
    }
    let tables = |store: &LutStore, entries: u16, n: usize| -> Option<Vec<Curve>> {
        let v = values(store);
        let e = usize::from(entries);
        (e >= 2 && v.len() >= e * n).then(|| {
            v.chunks_exact(e)
                .take(n)
                .map(|t| Curve::Table(t.to_vec()))
                .collect()
        })
    };
    let mut stages = Vec::new();
    let matrix = Affine {
        m: lut.matrix.v,
        b: [0.0; 3],
    };
    if pcs == DataColorSpace::Xyz && !matrix.is_identity() {
        stages.push(Stage::Affine(matrix));
    }
    stages.push(Stage::Curves(tables(
        &lut.input_table,
        lut.num_input_table_entries,
        inputs,
    )?));
    let grid = vec![usize::from(lut.num_clut_grid_points); inputs];
    stages.push(Stage::Clut(Clut::new(
        grid,
        outputs,
        values(&lut.clut_table),
    )?));
    stages.push(Stage::Curves(tables(
        &lut.output_table,
        lut.num_output_table_entries,
        outputs,
    )?));
    Some(stages)
}

/// The stages of a `lutBtoAType` table to `outputs` device channels.
fn mab_stages(m: &moxcms::LutMultidimensionalType, outputs: usize) -> Option<Vec<Stage>> {
    let inputs = 3;
    if usize::from(m.num_input_channels) != inputs || usize::from(m.num_output_channels) != outputs
    {
        return None;
    }
    let curves = |c: &[moxcms::ToneReprCurve], n: usize| -> Option<Option<Stage>> {
        match c.len() {
            0 => Some(None),
            len if len == n => Some(Some(Stage::Curves(
                c.iter().cloned().map(Curve::Trc).collect(),
            ))),
            _ => None,
        }
    };
    let a = curves(&m.a_curves, outputs)?;
    let b = curves(&m.b_curves, 3)?;
    let mc = curves(&m.m_curves, 3)?;
    let matrix = mc.as_ref().map(|_| {
        Stage::Affine(Affine {
            m: m.matrix.v,
            b: [m.bias.v[0], m.bias.v[1], m.bias.v[2]],
        })
    });
    let clut = match &m.clut {
        Some(store) => {
            let grid = m.grid_points[..inputs]
                .iter()
                .map(|&g| usize::from(g))
                .collect();
            Some(Stage::Clut(Clut::new(grid, outputs, values(store))?))
        }
        None if inputs == outputs => None,
        None => return None,
    };
    Some([b, matrix, mc, clut, a].into_iter().flatten().collect())
}

/// The scale of the legacy Lab encoding of `lut16Type`.
const LEGACY_LAB: f64 = 65280.0 / 65535.0;

/// The scale of the PCS XYZ encoding, `u1Fixed15Number`.
const XYZ_SCALE: f64 = 32768.0 / 65535.0;

/// CIE `f` of the Lab definition.
fn lab_f(t: f64) -> f64 {
    if t > 216.0 / 24389.0 {
        t.cbrt()
    } else {
        (24389.0f64 / 27.0).mul_add(t, 16.0) / 116.0
    }
}

/// `xyz` encoded for a table of PCS `pcs`.
fn encode(xyz: [f64; 3], pcs: DataColorSpace, legacy_lab: bool) -> [f64; 3] {
    if pcs == DataColorSpace::Xyz {
        return xyz.map(|v| (v * XYZ_SCALE).clamp(0.0, 1.0));
    }
    let [fx, fy, fz] = [0, 1, 2].map(|i| lab_f(xyz[i].max(0.0) / D50[i]));
    let lab = [
        116.0f64.mul_add(fy, -16.0) / 100.0,
        500.0f64.mul_add(fx - fy, 128.0) / 255.0,
        200.0f64.mul_add(fy - fz, 128.0) / 255.0,
    ];
    let scale = if legacy_lab { LEGACY_LAB } else { 1.0 };
    lab.map(|v| (v * scale).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pcs_encoding_of_white() {
        // White is L* 100, a* = b* = 0.
        let white = encode([0.9642, 1.0, 0.8249], DataColorSpace::Lab, false);
        assert!((white[0] - 1.0).abs() < 1e-12 && (white[1] - 128.0 / 255.0).abs() < 1e-12);
    }

    #[test]
    fn a_clut_interpolates_multilinearly() {
        // f(x, y) = x + 2y on a 2 × 3 grid, the first input slowest.
        let nodes = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0];
        let clut = Clut::new(vec![2, 3], 1, nodes).expect("a whole grid");
        let mut out = [0.0];
        clut.eval(&[0.25, 0.75], &mut out);
        assert!(
            (out[0] - 2.0f64.mul_add(0.75, 0.25)).abs() < 1e-12,
            "{out:?}"
        );
        assert!(Clut::new(vec![2, 3], 1, vec![0.0; 5]).is_none());
    }
}
