//! Evaluation of an ICC profile's device ↔ PCS tables at given values.
//!
//! A [`ProfileLut`] evaluates one `AToB` or `BToA` table (ICC.1:2022
//! §10.8, §10.10, §10.11, §10.12) in floating point at the values it is
//! given, with no resampling of its own: `lut8Type` and `lut16Type` as
//! matrix (on PCS XYZ input), input tables, CLUT and output tables;
//! `lutAtoBType` as A curves, CLUT, M curves, matrix and B curves;
//! `lutBtoAType` as B curves, matrix, M curves, CLUT and A curves.  One-
//! dimensional tables interpolate linearly.  A CLUT indexed by Lab
//! interpolates trilinearly in its cell; any other interpolates in the
//! tetrahedron of its last three inputs around the value (the simplex of
//! the cell visited in decreasing order of the fractional positions),
//! linearly along each input before them, and bilinearly or linearly with
//! fewer than three inputs.  A tetrahedron holds the cell's main diagonal
//! whole, so device values of equal components take the table's nodes of
//! equal components alone.  Every stage's output is clamped to `[0, 1]`.
//!
//! The PCS side is encoded as the table type stores it: Lab as `L / 100`,
//! `(a + 128) / 255`, `(b + 128) / 255`, scaled by `65280 / 65535` in a
//! `lut16Type` (the legacy encoding); XYZ as `X · 32768 / 65535`
//! (`u1Fixed15Number`).  [`ProfileLut::to_device`] takes and
//! [`ProfileLut::to_pcs`] returns media-relative D50 XYZ.

use moxcms::{ColorProfile, DataColorSpace, LutStore, LutType, LutWarehouse, RenderingIntent};

use crate::pcs::D50;
use crate::resources::icc_shaper::eval_trc;

/// The direction of a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// `AToB`: device values to the PCS.
    ToPcs,
    /// `BToA`: the PCS to device values.
    ToDevice,
}

/// The table `direction` takes under `intent` in `profile`: that of the
/// intent (the colorimetric one for both colorimetric intents), else the
/// perceptual one (ICC.1:2022 §8.10.2).
pub(crate) const fn table_of(
    profile: &ColorProfile,
    intent: RenderingIntent,
    direction: Direction,
) -> Option<&LutWarehouse> {
    let (perceptual, colorimetric, saturation) = match direction {
        Direction::ToPcs => (
            &profile.lut_a_to_b_perceptual,
            &profile.lut_a_to_b_colorimetric,
            &profile.lut_a_to_b_saturation,
        ),
        Direction::ToDevice => (
            &profile.lut_b_to_a_perceptual,
            &profile.lut_b_to_a_colorimetric,
            &profile.lut_b_to_a_saturation,
        ),
    };
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

/// The number of values at which [`ProfileLut::input_curves`] samples a
/// tone curve: 4 097, cells of 16 16-bit levels.
const TONE_CURVE_SAMPLES: usize = 4097;

impl Curve {
    /// The curve's values at evenly spaced inputs spanning `[0, 1]`: a
    /// table's own samples (two or more), else [`TONE_CURVE_SAMPLES`]
    /// values of [`Self::eval`].
    fn samples(&self) -> Vec<f64> {
        match self {
            Self::Table(t) if t.len() >= 2 => t.iter().map(|v| v.clamp(0.0, 1.0)).collect(),
            _ => (0..TONE_CURVE_SAMPLES)
                .map(|j| {
                    #[expect(clippy::cast_precision_loss, reason = "j < 4097")]
                    let x = j as f64 / (TONE_CURVE_SAMPLES - 1) as f64;
                    self.eval(x)
                })
                .collect(),
        }
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

/// The interpolation of a [`Clut`] within a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interpolation {
    /// Multilinear in every input.
    Multilinear,
    /// Tetrahedral in the last three inputs, linear along each before.
    Tetrahedral,
}

/// The largest number of outputs of a CLUT (ICC.1:2022 §10.12).
const MAX_OUTPUTS: usize = 16;

/// A colour lookup table of `outputs` values at each node of a grid of
/// `grid[i]` nodes along input `i`, the first input varying slowest.
#[derive(Debug, Clone)]
struct Clut {
    grid: Vec<usize>,
    outputs: usize,
    nodes: Vec<f64>,
    interpolation: Interpolation,
}

impl Clut {
    /// The table of `nodes`, `None` when their count is not the grid's or
    /// it has more than [`MAX_OUTPUTS`] outputs.  A table indexed by Lab
    /// (`lab_indexed`) interpolates multilinearly, any other
    /// tetrahedrally.
    fn new(grid: Vec<usize>, outputs: usize, nodes: Vec<f64>, lab_indexed: bool) -> Option<Self> {
        let count = grid.iter().try_fold(outputs, |n, &g| {
            (g >= 2).then(|| n.checked_mul(g)).flatten()
        })?;
        let interpolation = if lab_indexed {
            Interpolation::Multilinear
        } else {
            Interpolation::Tetrahedral
        };
        (nodes.len() == count && outputs <= MAX_OUTPUTS).then_some(Self {
            grid,
            outputs,
            nodes,
            interpolation,
        })
    }

    /// The interpolation of the cell around `input` into `out`.
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
        match self.interpolation {
            Interpolation::Multilinear => self.multilinear(&cell[..n], base, out),
            Interpolation::Tetrahedral => self.tetrahedral(&cell[..n], base, out),
        }
    }

    /// The multilinear interpolation in the cell of node `base`, whose
    /// inputs have the strides and fractional positions `cell`, into
    /// `out`.
    fn multilinear(&self, cell: &[(usize, f64)], base: usize, out: &mut [f64]) {
        out.fill(0.0);
        for corner in 0..1usize << cell.len() {
            let mut weight = 1.0;
            let mut at = base;
            for (i, &(stride, f)) in cell.iter().enumerate() {
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
            self.accumulate(weight, at, out);
        }
    }

    /// The interpolation in the cell of node `base` as
    /// [`Interpolation::Tetrahedral`] into `out`: along the first input
    /// between the interpolations of the faces at its two ends while more
    /// than three inputs remain; with three, the sum of the cell's nodes
    /// from `base` towards the opposite corner, one input at a time in
    /// decreasing order of fractional position, each weighted by the
    /// difference of successive positions; with fewer, multilinear.
    fn tetrahedral(&self, cell: &[(usize, f64)], base: usize, out: &mut [f64]) {
        match *cell {
            [(stride, f), ref rest @ ..] if rest.len() >= 3 => {
                self.tetrahedral(rest, base, out);
                if f > 0.0 {
                    let mut far = [0.0; MAX_OUTPUTS];
                    let far = &mut far[..self.outputs];
                    self.tetrahedral(rest, base + stride, far);
                    for (o, &v) in out.iter_mut().zip(&*far) {
                        *o = f.mul_add(v - *o, *o);
                    }
                }
            }
            [a, b, c] => {
                let mut order = [a, b, c];
                order.sort_unstable_by(|x, y| y.1.total_cmp(&x.1));
                out.fill(0.0);
                let mut at = base;
                let mut previous = 1.0;
                for (stride, f) in order {
                    self.accumulate(previous - f, at, out);
                    previous = f;
                    at += stride;
                }
                self.accumulate(previous, at, out);
            }
            _ => self.multilinear(cell, base, out),
        }
    }

    /// `out += weight · node(at)`.
    fn accumulate(&self, weight: f64, at: usize, out: &mut [f64]) {
        if weight == 0.0 {
            return;
        }
        for (o, &v) in out.iter_mut().zip(&self.nodes[at..at + self.outputs]) {
            *o = weight.mul_add(v, *o);
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

/// One table of a profile, evaluated at given values.
#[derive(Debug, Clone)]
pub(crate) struct ProfileLut {
    direction: Direction,
    /// The PCS of the profile.
    pcs: DataColorSpace,
    /// Whether Lab is encoded the legacy way of `lut16Type`.
    legacy_lab: bool,
    stages: Vec<Stage>,
}

impl ProfileLut {
    /// The table `direction` takes under `intent` in `profile`
    /// ([`table_of`]) for a profile of `device` channels; `None` when the
    /// profile has no such table, its PCS is neither Lab nor XYZ, or the
    /// table is malformed.
    pub(crate) fn new(
        profile: &ColorProfile,
        intent: RenderingIntent,
        direction: Direction,
        device: usize,
    ) -> Option<Self> {
        if !matches!(profile.pcs, DataColorSpace::Lab | DataColorSpace::Xyz) {
            return None;
        }
        let (inputs, outputs) = match direction {
            Direction::ToPcs => (device, 3),
            Direction::ToDevice => (3, device),
        };
        let (stages, legacy_lab) = match table_of(profile, intent, direction)? {
            LutWarehouse::Lut(lut) => (
                lut_stages(lut, inputs, outputs, profile.pcs, direction)?,
                lut.lut_type == LutType::Lut16,
            ),
            LutWarehouse::Multidimensional(m) => (
                mab_stages(m, inputs, outputs, profile.pcs, direction)?,
                false,
            ),
        };
        Some(Self {
            direction,
            pcs: profile.pcs,
            legacy_lab,
            stages,
        })
    }

    /// The device values of the media-relative D50 XYZ `xyz` into `out`,
    /// one per device channel, each in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// For a table to the PCS.
    pub(crate) fn to_device(&self, xyz: [f64; 3], out: &mut [f64]) {
        assert_eq!(self.direction, Direction::ToDevice, "a PCS → device table");
        let input = encode(xyz, self.pcs, self.legacy_lab);
        self.run(&input, out);
    }

    /// The media-relative D50 XYZ of the device values `device`.
    ///
    /// # Panics
    ///
    /// For a table to the device.
    pub(crate) fn to_pcs(&self, device: &[f64]) -> [f64; 3] {
        assert_eq!(self.direction, Direction::ToPcs, "a device → PCS table");
        let mut out = [0.0; 3];
        self.run(device, &mut out);
        decode(out, self.pcs, self.legacy_lab)
    }

    /// The curves a device → PCS table applies to its inputs before any
    /// other stage, each as its values at evenly spaced inputs spanning
    /// `[0, 1]`: a sampled table's own samples, a tone curve's at
    /// [`TONE_CURVE_SAMPLES`] values.  `None` when the table's first stage
    /// is not a set of curves, or each curve is the identity within half a
    /// 16-bit level at its samples.
    ///
    /// # Panics
    ///
    /// For a table to the device.
    pub(crate) fn input_curves(&self) -> Option<Vec<Vec<f64>>> {
        assert_eq!(self.direction, Direction::ToPcs, "a device → PCS table");
        let Some(Stage::Curves(curves)) = self.stages.first() else {
            return None;
        };
        let samples: Vec<Vec<f64>> = curves.iter().map(Curve::samples).collect();
        let identity = samples.iter().all(|s| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a curve has far fewer than 2^52 samples"
            )]
            let last = (s.len() - 1) as f64;
            s.iter().enumerate().all(|(j, &v)| {
                #[expect(clippy::cast_precision_loss, reason = "j < len")]
                let x = j as f64 / last;
                (v - x).abs() <= 0.5 / 65535.0
            })
        });
        (!identity).then_some(samples)
    }

    /// The number of nodes of the table's CLUT along each input, `None`
    /// for a table without one.
    pub(crate) fn clut_grid(&self) -> Option<&[usize]> {
        self.stages.iter().find_map(|stage| match stage {
            Stage::Clut(clut) => Some(&*clut.grid),
            _ => None,
        })
    }

    /// The media-relative D50 XYZ of the device values whose images under
    /// [`Self::input_curves`] are `shaped`: the stages after those curves
    /// applied to `shaped`.
    ///
    /// # Panics
    ///
    /// For a table to the device or one without input curves.
    pub(crate) fn to_pcs_shaped(&self, shaped: &[f64]) -> [f64; 3] {
        assert_eq!(self.direction, Direction::ToPcs, "a device → PCS table");
        assert!(
            matches!(self.stages.first(), Some(Stage::Curves(_))),
            "a table with input curves"
        );
        let mut out = [0.0; 3];
        self.run_from(1, shaped, &mut out);
        decode(out, self.pcs, self.legacy_lab)
    }

    /// The stages applied to `input` into `out`.
    fn run(&self, input: &[f64], out: &mut [f64]) {
        self.run_from(0, input, out);
    }

    /// The stages from the `first`th on applied to `input` into `out`.
    fn run_from(&self, first: usize, input: &[f64], out: &mut [f64]) {
        let mut a = [0.0f64; 16];
        let mut b = [0.0f64; 16];
        let mut len = input.len();
        a[..len].copy_from_slice(input);
        for stage in &self.stages[first..] {
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

/// The stages of a `lut8Type` or `lut16Type` table of `inputs` inputs and
/// `outputs` outputs.
fn lut_stages(
    lut: &moxcms::LutDataType,
    inputs: usize,
    outputs: usize,
    pcs: DataColorSpace,
    direction: Direction,
) -> Option<Vec<Stage>> {
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
    if direction == Direction::ToDevice && pcs == DataColorSpace::Xyz && !matrix.is_identity() {
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
        lab_indexed(pcs, direction),
    )?));
    stages.push(Stage::Curves(tables(
        &lut.output_table,
        lut.num_output_table_entries,
        outputs,
    )?));
    Some(stages)
}

/// The stages of a `lutAtoBType` or `lutBtoAType` table of `inputs`
/// inputs and `outputs` outputs.
fn mab_stages(
    m: &moxcms::LutMultidimensionalType,
    inputs: usize,
    outputs: usize,
    pcs: DataColorSpace,
    direction: Direction,
) -> Option<Vec<Stage>> {
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
    let device = match direction {
        Direction::ToPcs => inputs,
        Direction::ToDevice => outputs,
    };
    let a = curves(&m.a_curves, device)?;
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
            Some(Stage::Clut(Clut::new(
                grid,
                outputs,
                values(store),
                lab_indexed(pcs, direction),
            )?))
        }
        None if inputs == outputs => None,
        None => return None,
    };
    let stages = match direction {
        Direction::ToPcs => [a, clut, mc, matrix, b],
        Direction::ToDevice => [b, matrix, mc, clut, a],
    };
    Some(stages.into_iter().flatten().collect())
}

/// Whether the CLUT of a table `direction` of a profile of PCS `pcs` is
/// indexed by Lab.
fn lab_indexed(pcs: DataColorSpace, direction: Direction) -> bool {
    direction == Direction::ToDevice && pcs == DataColorSpace::Lab
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

/// The inverse of [`lab_f`].
fn lab_f_inv(f: f64) -> f64 {
    let t = f * f * f;
    if t > 216.0 / 24389.0 {
        t
    } else {
        116.0f64.mul_add(f, -16.0) * 27.0 / 24389.0
    }
}

/// `xyz` encoded for a table of PCS `pcs`, each encoded component clipped
/// to `[0, 1]`.  A component below 0 converts to Lab through the linear
/// branch of [`lab_f`], which extends below the origin, so a colour moved
/// below the PCS black keeps its `a*` and `b*` while its `L*` clips.
fn encode(xyz: [f64; 3], pcs: DataColorSpace, legacy_lab: bool) -> [f64; 3] {
    if pcs == DataColorSpace::Xyz {
        return xyz.map(|v| (v * XYZ_SCALE).clamp(0.0, 1.0));
    }
    let [fx, fy, fz] = [0, 1, 2].map(|i| lab_f(xyz[i] / D50[i]));
    let lab = [
        116.0f64.mul_add(fy, -16.0) / 100.0,
        500.0f64.mul_add(fx - fy, 128.0) / 255.0,
        200.0f64.mul_add(fy - fz, 128.0) / 255.0,
    ];
    let scale = if legacy_lab { LEGACY_LAB } else { 1.0 };
    lab.map(|v| (v * scale).clamp(0.0, 1.0))
}

/// The media-relative D50 XYZ `xyz` as Lab, encoded `L / 100`,
/// `(a + 128) / 255`, `(b + 128) / 255`, each clipped to `[0, 1]`.
pub(crate) fn lab_encoding(xyz: [f64; 3]) -> [f64; 3] {
    encode(xyz, DataColorSpace::Lab, false)
}

/// The media-relative D50 XYZ of `encoded`, Lab in the encoding of
/// [`lab_encoding`].
pub(crate) fn lab_decoding(encoded: [f64; 3]) -> [f64; 3] {
    decode(encoded, DataColorSpace::Lab, false)
}

/// The XYZ of `encoded`, a value of a table of PCS `pcs`.
fn decode(encoded: [f64; 3], pcs: DataColorSpace, legacy_lab: bool) -> [f64; 3] {
    if pcs == DataColorSpace::Xyz {
        return encoded.map(|v| v / XYZ_SCALE);
    }
    let scale = if legacy_lab { LEGACY_LAB } else { 1.0 };
    let [l, a, b] = encoded.map(|v| v / scale);
    let (l, a, b) = (
        l * 100.0,
        a.mul_add(255.0, -128.0),
        b.mul_add(255.0, -128.0),
    );
    let fy = (l + 16.0) / 116.0;
    let f = [fy + a / 500.0, fy, fy - b / 200.0];
    std::array::from_fn(|i| D50[i] * lab_f_inv(f[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pcs_encodings_round_trip() {
        for xyz in [
            [0.9642, 1.0, 0.8249],
            [0.2, 0.18, 0.1],
            [0.001, 0.0012, 0.0009],
        ] {
            for (pcs, legacy) in [
                (DataColorSpace::Lab, false),
                (DataColorSpace::Lab, true),
                (DataColorSpace::Xyz, false),
            ] {
                let back = decode(encode(xyz, pcs, legacy), pcs, legacy);
                for (a, b) in back.iter().zip(xyz) {
                    assert!((a - b).abs() < 1e-12, "{xyz:?} {pcs:?} {legacy}: {back:?}");
                }
            }
        }
        // White is L* 100, a* = b* = 0.
        let white = encode([0.9642, 1.0, 0.8249], DataColorSpace::Lab, false);
        assert!((white[0] - 1.0).abs() < 1e-12 && (white[1] - 128.0 / 255.0).abs() < 1e-12);
    }

    /// A PCS value below the origin, as a black-point map leaves the
    /// darkest colours, keeps the `a*` and `b*` of the linear branch of the
    /// CIE `f` while its `L*` clips to 0.
    #[test]
    fn a_colour_below_the_pcs_black_keeps_its_hue() {
        let xyz = [-0.0002, -0.0008, -0.0012];
        let lab = encode(xyz, DataColorSpace::Lab, false);
        let f = [0, 1, 2].map(|i| lab_f(xyz[i] / D50[i]));
        let [a, b] = [lab[1], lab[2]].map(|v| v.mul_add(255.0, -128.0));
        assert!(lab[0].abs() < 1e-12, "{lab:?}");
        assert!(500.0f64.mul_add(f[1] - f[0], a).abs() < 1e-9, "{lab:?}");
        assert!(200.0f64.mul_add(f[2] - f[1], b).abs() < 1e-9, "{lab:?}");
        assert!(a > 1.0, "a* of {lab:?}");
    }

    #[test]
    fn a_clut_interpolates_multilinearly() {
        // f(x, y) = x + 2y on a 2 × 3 grid, the first input slowest.
        let nodes = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0];
        let clut = Clut::new(vec![2, 3], 1, nodes, true).expect("a whole grid");
        let mut out = [0.0];
        clut.eval(&[0.25, 0.75], &mut out);
        assert!(
            (out[0] - 2.0f64.mul_add(0.75, 0.25)).abs() < 1e-12,
            "{out:?}"
        );
        assert!(Clut::new(vec![2, 3], 1, vec![0.0; 5], true).is_none());
    }

    /// A table of nodes `(x + y + z, x · y · z)` on a 3 × 3 × 3 grid, in
    /// `[0, 1]`, interpolated tetrahedrally: exact on the main diagonal
    /// whatever the off-diagonal nodes hold, and the cell's corners'
    /// convex combination elsewhere.
    #[test]
    fn a_device_clut_interpolates_tetrahedrally() {
        let mut nodes = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    let [x, y, z] = [x, y, z].map(|v| f64::from(v) / 2.0);
                    nodes.extend([(x + y + z) / 3.0, x * y * z]);
                }
            }
        }
        let clut = Clut::new(vec![3; 3], 2, nodes, false).expect("a whole grid");
        let mut out = [0.0; 2];
        clut.eval(&[0.3; 3], &mut out);
        // (0.3, 0.3, 0.3) lies 0.6 of the way from node 0 to node 1/2 of
        // the diagonal: 0.6 · (0.5, 0.125).
        assert!(
            (out[0] - 0.3).abs() < 1e-12 && (out[1] - 0.075).abs() < 1e-12,
            "{out:?}"
        );
        // Off the diagonal, fractions 0.8 ≥ 0.4 ≥ 0.2 in the first cell:
        // node 0, then (1, 0, 0), (1, 1, 0) and (1, 1, 1), halves.
        clut.eval(&[0.4, 0.2, 0.1], &mut out);
        let want = 0.4f64.mul_add(1.0 / 6.0, 0.2f64.mul_add(2.0 / 6.0, 0.2 * 3.0 / 6.0));
        assert!((out[0] - want).abs() < 1e-12, "{out:?} for {want}");
        assert!(0.2f64.mul_add(-0.125, out[1]).abs() < 1e-12, "{out:?}");
    }

    /// The version 4 RGB test printer maps equal device components to a
    /// neutral PCS, `a* = b* = 0` within the rounding of its model's
    /// matrix, at nodes and between them alike.
    #[test]
    fn a_device_table_keeps_equal_components_neutral() {
        for name in ["printer-rgb-v4"] {
            let path = format!(
                "{}/../../tests/fixtures/icc/{name}.icc",
                env!("CARGO_MANIFEST_DIR")
            );
            let profile =
                ColorProfile::new_from_slice(&std::fs::read(&path).unwrap()).expect("parses");
            for intent in [
                RenderingIntent::Perceptual,
                RenderingIntent::RelativeColorimetric,
                RenderingIntent::Saturation,
            ] {
                let lut = ProfileLut::new(&profile, intent, Direction::ToPcs, 3).expect("a table");
                for v in [0.03, 0.125, 0.2, 0.5, 0.77] {
                    let xyz = lut.to_pcs(&[v; 3]);
                    let lab = encode(xyz, DataColorSpace::Lab, false);
                    for c in &lab[1..] {
                        assert!(
                            (c - 128.0 / 255.0).abs() * 255.0 < 0.05,
                            "{name} {intent:?} {v}: {lab:?}"
                        );
                    }
                }
            }
        }
    }

    /// Four inputs interpolate linearly along the first between the
    /// tetrahedral interpolations of the other three.
    #[test]
    fn a_four_input_clut_is_linear_along_its_first_input() {
        let mut nodes = Vec::new();
        for w in 0..2 {
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        nodes.push(f64::from(w * 4 + x * y * z));
                    }
                }
            }
        }
        let nodes: Vec<f64> = nodes.iter().map(|v| v / 5.0).collect();
        let clut = Clut::new(vec![2; 4], 1, nodes, false).expect("a whole grid");
        let mut out = [0.0];
        clut.eval(&[0.25, 0.5, 0.5, 0.5], &mut out);
        assert!(
            (out[0] - 0.25f64.mul_add(4.0, 0.5) / 5.0).abs() < 1e-12,
            "{out:?}"
        );
    }

    /// A device → PCS table of two inputs: `curves`, then the CLUT of
    /// three nodes along each input whose XYZ is `(x, y, (x + y) / 2)`
    /// scaled to the PCS encoding.
    fn two_input_table(curves: Vec<Curve>) -> ProfileLut {
        let mut nodes = Vec::new();
        for x in 0..3 {
            for y in 0..3 {
                let [x, y] = [x, y].map(|v| f64::from(v) / 2.0);
                nodes.extend([x, y, f64::midpoint(x, y)].map(|v| v * XYZ_SCALE));
            }
        }
        let clut = Clut::new(vec![3, 3], 3, nodes, false).expect("a whole grid");
        ProfileLut {
            direction: Direction::ToPcs,
            pcs: DataColorSpace::Xyz,
            legacy_lab: false,
            stages: vec![Stage::Curves(curves), Stage::Clut(clut)],
        }
    }

    /// A table's input curves are its first stage's, a sampled table's
    /// own samples and a tone curve's at 4 097 values, `None` when each is
    /// the identity; the stages after them convert the curves' outputs as
    /// the whole table converts their inputs.
    #[test]
    fn the_input_curves_split_from_the_table() {
        let identity = two_input_table(vec![
            Curve::Table(vec![0.0, 0.5, 1.0]),
            Curve::Trc(moxcms::ToneReprCurve::Lut(Vec::new())),
        ]);
        assert!(identity.input_curves().is_none());
        assert_eq!(identity.clut_grid(), Some(&[3, 3][..]));
        let lut = two_input_table(vec![
            Curve::Table(vec![0.0, 0.1, 1.0]),
            Curve::Trc(moxcms::ToneReprCurve::Parametric(vec![2.0])),
        ]);
        let curves = lut.input_curves().expect("non-identity curves");
        assert_eq!(curves[0], [0.0, 0.1, 1.0]);
        assert_eq!(curves[1].len(), TONE_CURVE_SAMPLES);
        assert!((curves[1][2048] - 0.25).abs() < 1e-6, "{}", curves[1][2048]);
        for device in [[0.25, 0.5], [0.7, 0.1], [1.0, 0.0]] {
            let shaped = [interpolate(&curves[0], device[0]), device[1] * device[1]];
            let (a, b) = (lut.to_pcs(&device), lut.to_pcs_shaped(&shaped));
            for (a, b) in a.iter().zip(b) {
                assert!((a - b).abs() < 1e-6, "{device:?}: {a} {b}");
            }
        }
    }
}
