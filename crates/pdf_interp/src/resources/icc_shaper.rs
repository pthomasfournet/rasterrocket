//! Matrix/TRC ("shaper") ICC input profiles evaluated to sRGB.
//!
//! An RGB shaper profile maps a device sample to the D50 profile connection
//! space (PCS) by a per-channel tone reproduction curve (`rTRC`, `gTRC`,
//! `bTRC`) followed by the colorant matrix whose columns are `rXYZ`, `gXYZ`
//! and `bXYZ` (ICC.1:2010 §F.3).  A gray profile maps by `grayTRC` alone onto
//! the PCS luminance `Y` (§F.2).  The output side is sRGB: PCS XYZ goes
//! through the inverse of the D50-adapted sRGB colorant matrix and the
//! IEC 61966-2-1 encoding; a gray `Y` maps to the neutral `R = G = B`.
//!
//! [`ProfileConversion`] holds the evaluated form of one profile: a 256-entry
//! linearisation table per channel, the composite 3×3 matrix, and a shared
//! encoding table, so an 8-bit pixel costs three lookups, one matrix product
//! and three lookups.  A profile whose composite matrix is the identity
//! reduces further to one 256-entry table per channel, and one whose tables
//! are then the identity at 8-bit precision is reported as such so callers
//! skip the pass.
//!
//! Tags are read with `moxcms`; profiles without the full matrix/TRC tag set
//! (A2B/B2A table-based profiles), with a non-XYZ PCS, or that fail to parse
//! are [`ProfileConversion::Unsupported`].  Conversions are memoised
//! process-wide, keyed by the profile bytes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use moxcms::{ColorProfile, DataColorSpace, ToneReprCurve, Xyzd};
use pdf::{Document, ObjectId};

/// 3×3 matrix, row-major: `m[row][col]`.
pub(super) type Mat3 = [[f64; 3]; 3];

/// D50-adapted sRGB colorants (Bradford) as columns: the `rXYZ`, `gXYZ` and
/// `bXYZ` tags of the IEC 61966-2.1 profile.
const SRGB_D50_COLORANTS: Mat3 = [
    [0.436_065_7, 0.385_147_1, 0.143_066_4],
    [0.222_488_4, 0.716_873_2, 0.060_607_9],
    [0.013_916_0, 0.097_076_4, 0.714_096_1],
];

/// Maximum element-wise deviation of a composite matrix from the identity
/// for it to be treated as the identity.  Two encodings of the sRGB
/// colorants differ by under `1e-3` (s15Fixed16 rounding and adaptation
/// method); a deviation of `2e-3` moves a full-scale channel by half a level.
const IDENTITY_MATRIX_TOLERANCE: f64 = 2e-3;

/// Entries in the linear → sRGB encoding table.  The steepest segment of
/// the encoding has slope 12.92, so one table step is at most
/// `12.92 · 255 / (ENCODE_TABLE_LEN − 1) ≈ 0.2` output levels.
const ENCODE_TABLE_LEN: usize = 16_384;

/// Number of distinct profiles memoised before the cache is flushed.
const CACHE_CAPACITY: usize = 64;

// ── Curve and encoding maths ──────────────────────────────────────────────────

/// Evaluate an ICC tone reproduction curve at `x ∈ [0, 1]`.
///
/// - `curv` with no entries is the identity; with one entry it is a pure
///   power law whose exponent is that entry as `u8Fixed8Number`; with more it
///   is a table sampled at `n` equidistant inputs, interpolated linearly.
/// - `para` function types 0–4 (ICC.1:2010 Table 68), identified by their
///   parameter count `g`, `g a b`, `g a b c`, `g a b c d`, `g a b c d e f`.
///   A negative power base evaluates as zero.
///
/// Returns `None` for a parameter count outside the five defined types.
#[expect(
    clippy::many_single_char_names,
    reason = "g, a … f are the parameter names of ICC.1:2010 Table 68"
)]
pub(super) fn eval_trc(curve: &ToneReprCurve, x: f64) -> Option<f64> {
    let x = x.clamp(0.0, 1.0);
    match curve {
        ToneReprCurve::Lut(table) => Some(match table.as_slice() {
            [] => x,
            [gamma] => x.powf(f64::from(*gamma) / 256.0),
            samples => sampled_curve(samples, x),
        }),
        ToneReprCurve::Parametric(p) => {
            let p: Vec<f64> = p.iter().map(|&v| f64::from(v)).collect();
            let pow = |base: f64, g: f64| if base > 0.0 { base.powf(g) } else { 0.0 };
            match *p.as_slice() {
                [g] => Some(pow(x, g)),
                [g, a, b] => Some(if x >= -b / a {
                    pow(a.mul_add(x, b), g)
                } else {
                    0.0
                }),
                [g, a, b, c] => Some(if x >= -b / a {
                    pow(a.mul_add(x, b), g) + c
                } else {
                    c
                }),
                [g, a, b, c, d] => Some(if x >= d {
                    pow(a.mul_add(x, b), g)
                } else {
                    c * x
                }),
                [g, a, b, c, d, e, f] => Some(if x >= d {
                    pow(a.mul_add(x, b), g) + e
                } else {
                    c.mul_add(x, f)
                }),
                _ => None,
            }
        }
    }
}

/// Linear interpolation in a table of `n ≥ 2` samples spanning `[0, 1]`.
fn sampled_curve(samples: &[u16], x: f64) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a curv table has at most 2^32 entries; the position needs no more than f64 offers"
    )]
    let pos = x * (samples.len() - 1) as f64;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "pos ∈ [0, n − 1] after the clamp in the caller"
    )]
    let lo = (pos.floor() as usize).min(samples.len() - 2);
    #[expect(
        clippy::cast_precision_loss,
        reason = "lo < n, exactly representable as computed above"
    )]
    let frac = pos - lo as f64;
    let a = f64::from(samples[lo]);
    let b = f64::from(samples[lo + 1]);
    frac.mul_add(b - a, a) / 65_535.0
}

/// Linear → sRGB encoding per IEC 61966-2-1, input clamped to `[0, 1]`.
pub(super) fn srgb_encode(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        c.powf(1.0 / 2.4).mul_add(1.055, -0.055)
    }
}

/// `v ∈ [0, 1]` to the nearest 8-bit level.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to [0, 255] before the cast"
)]
fn unit_to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// `level / 255` for every 8-bit level.
fn level(i: u8) -> f64 {
    f64::from(i) / 255.0
}

/// The shared linear → 8-bit sRGB table, `ENCODE_TABLE_LEN` entries over `[0, 1]`.
fn encode_table() -> &'static [u8; ENCODE_TABLE_LEN] {
    static TABLE: OnceLock<Box<[u8; ENCODE_TABLE_LEN]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u8; ENCODE_TABLE_LEN]);
        for (i, e) in t.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "i < 16 384, exact in f64")]
            let lin = i as f64 / (ENCODE_TABLE_LEN - 1) as f64;
            *e = unit_to_u8(srgb_encode(lin));
        }
        t
    })
}

/// Index of the encoding-table entry nearest to linear value `v`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "v is clamped to [0, 1], so the product lies in [0, ENCODE_TABLE_LEN − 1]"
)]
const fn encode_index(v: f32) -> usize {
    v.clamp(0.0, 1.0)
        .mul_add((ENCODE_TABLE_LEN - 1) as f32, 0.5) as usize
}

pub(super) fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// Inverse by the adjugate; `None` for a singular or non-finite matrix.
pub(super) fn mat_inverse(m: &Mat3) -> Option<Mat3> {
    let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
        m[r0][c0].mul_add(m[r1][c1], -(m[r0][c1] * m[r1][c0]))
    };
    let c00 = cof(1, 2, 1, 2);
    let c01 = -cof(1, 2, 0, 2);
    let c02 = cof(1, 2, 0, 1);
    let det = m[0][2].mul_add(c02, m[0][0].mul_add(c00, m[0][1] * c01));
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = [
        [c00, -cof(0, 2, 1, 2), cof(0, 1, 1, 2)],
        [c01, cof(0, 2, 0, 2), -cof(0, 1, 0, 2)],
        [c02, -cof(0, 2, 0, 1), cof(0, 1, 0, 1)],
    ];
    Some(inv.map(|row| row.map(|v| v / det)))
}

/// XYZ (D50 PCS) → linear sRGB.
fn pcs_to_linear_srgb() -> Mat3 {
    mat_inverse(&SRGB_D50_COLORANTS).expect("the sRGB colorant matrix is non-singular")
}

fn is_identity(m: &Mat3) -> bool {
    m.iter().enumerate().all(|(r, row)| {
        row.iter().enumerate().all(|(c, &v)| {
            let target = if r == c { 1.0 } else { 0.0 };
            (v - target).abs() <= IDENTITY_MATRIX_TOLERANCE
        })
    })
}

// ── Evaluated conversions ─────────────────────────────────────────────────────

/// 8-bit gray → 8-bit sRGB gray: `table[v] = encode(grayTRC(v / 255))`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GrayShaper {
    table: [u8; 256],
}

impl GrayShaper {
    /// Rewrite one-byte-per-pixel gray samples in place.
    pub(super) fn convert_in_place(&self, samples: &mut [u8]) {
        for s in samples {
            *s = self.table[usize::from(*s)];
        }
    }

    pub(super) const fn convert(&self, v: u8) -> u8 {
        self.table[v as usize]
    }
}

/// 8-bit RGB → 8-bit sRGB through an RGB shaper profile.
#[derive(Debug, PartialEq)]
pub(super) enum RgbShaper {
    /// The composite matrix is the identity: one table per channel,
    /// `tables[c][v] = encode(TRC_c(v / 255))`.
    PerChannel(Box<[[u8; 256]; 3]>),
    /// `out = encode(matrix · (TRC_r(r), TRC_g(g), TRC_b(b)))`.
    Matrix {
        /// `linear[c][v] = TRC_c(v / 255)`.
        linear: Box<[[f32; 256]; 3]>,
        /// PCS → linear sRGB composed with the profile's colorant matrix.
        matrix: [[f32; 3]; 3],
    },
}

impl RgbShaper {
    /// Convert one pixel.
    pub(super) fn convert(&self, px: [u8; 3]) -> [u8; 3] {
        match self {
            Self::PerChannel(t) => [
                t[0][usize::from(px[0])],
                t[1][usize::from(px[1])],
                t[2][usize::from(px[2])],
            ],
            Self::Matrix { linear, matrix } => {
                Self::convert_matrix(linear, matrix, encode_table(), px)
            }
        }
    }

    #[inline]
    fn convert_matrix(
        linear: &[[f32; 256]; 3],
        m: &[[f32; 3]; 3],
        enc: &[u8; ENCODE_TABLE_LEN],
        px: [u8; 3],
    ) -> [u8; 3] {
        let r = linear[0][usize::from(px[0])];
        let g = linear[1][usize::from(px[1])];
        let b = linear[2][usize::from(px[2])];
        m.map(|row| enc[encode_index(row[2].mul_add(b, row[0].mul_add(r, row[1] * g)))])
    }

    /// Rewrite packed three-byte-per-pixel samples in place.  A trailing
    /// partial pixel is left untouched.
    pub(super) fn convert_in_place(&self, samples: &mut [u8]) {
        let pixels = samples.as_chunks_mut::<3>().0;
        match self {
            Self::PerChannel(t) => {
                for px in pixels {
                    *px = [
                        t[0][usize::from(px[0])],
                        t[1][usize::from(px[1])],
                        t[2][usize::from(px[2])],
                    ];
                }
            }
            Self::Matrix { linear, matrix } => {
                let enc = encode_table();
                for px in pixels {
                    *px = Self::convert_matrix(linear, matrix, enc, *px);
                }
            }
        }
    }
}

/// The evaluated form of one ICC profile for 8-bit input.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ProfileConversion {
    /// Not a matrix/TRC profile, or not a valid profile.
    Unsupported,
    /// A gray TRC profile; `None` when it equals the sRGB encoding at 8 bits.
    Gray(Option<Arc<GrayShaper>>),
    /// An RGB matrix/TRC profile; `None` when it equals sRGB at 8 bits.
    Rgb(Option<Arc<RgbShaper>>),
}

impl ProfileConversion {
    /// Evaluate `bytes` without consulting the cache.
    pub(super) fn from_profile_bytes(bytes: &[u8]) -> Self {
        let profile = match ColorProfile::new_from_slice(bytes) {
            Ok(p) => p,
            Err(e) => {
                log::debug!("icc: profile parse failed: {e}");
                return Self::Unsupported;
            }
        };
        if profile.pcs != DataColorSpace::Xyz {
            return Self::Unsupported;
        }
        match profile.color_space {
            DataColorSpace::Gray => profile
                .gray_trc
                .as_ref()
                .map_or(Self::Unsupported, gray_conversion),
            DataColorSpace::Rgb => rgb_conversion(&profile),
            _ => Self::Unsupported,
        }
    }

    /// Convert normalised components through the profile when it is a
    /// shaper profile of exactly `comps.len()` channels.
    ///
    /// Components are quantised to 8 bits, the precision of the tables.
    /// Returns `None` when the profile is unsupported or its channel count
    /// differs from `comps.len()`.
    pub(super) fn convert_components(&self, comps: &[f64]) -> Option<[u8; 3]> {
        let q = |v: f64| unit_to_u8(v);
        match (self, comps) {
            (Self::Gray(shaper), &[v]) => {
                let v = q(v);
                Some([shaper.as_ref().map_or(v, |s| s.convert(v)); 3])
            }
            (Self::Rgb(shaper), &[r, g, b]) => {
                let px = [q(r), q(g), q(b)];
                Some(shaper.as_ref().map_or(px, |s| s.convert(px)))
            }
            _ => None,
        }
    }
}

/// The conversion of a gray profile with tone curve `trc`; the identity
/// when every level maps to itself within one level.
fn gray_conversion(trc: &ToneReprCurve) -> ProfileConversion {
    let mut table = [0u8; 256];
    for (v, out) in (0u8..=255).zip(table.iter_mut()) {
        let Some(y) = eval_trc(trc, level(v)) else {
            return ProfileConversion::Unsupported;
        };
        *out = unit_to_u8(srgb_encode(y));
    }
    ProfileConversion::Gray(if is_identity_table(&table) {
        None
    } else {
        Some(Arc::new(GrayShaper { table }))
    })
}

fn is_identity_table(t: &[u8; 256]) -> bool {
    (0u8..=255).zip(t).all(|(v, &out)| v.abs_diff(out) <= 1)
}

const fn xyz_column(c: &Xyzd) -> [f64; 3] {
    [c.x, c.y, c.z]
}

/// The conversion of an RGB profile; unsupported when it lacks a TRC or a
/// non-singular colorant matrix, the identity when it equals sRGB at 8 bits.
fn rgb_conversion(p: &ColorProfile) -> ProfileConversion {
    rgb_shaper(p).map_or(ProfileConversion::Unsupported, ProfileConversion::Rgb)
}

/// `Err` for an unsupported profile, `Ok(None)` for the identity.
fn rgb_shaper(p: &ColorProfile) -> Result<Option<Arc<RgbShaper>>, ()> {
    let (Some(rt), Some(gt), Some(bt)) = (&p.red_trc, &p.green_trc, &p.blue_trc) else {
        return Err(());
    };
    let trcs = [rt, gt, bt];
    let cols = [
        xyz_column(&p.red_colorant),
        xyz_column(&p.green_colorant),
        xyz_column(&p.blue_colorant),
    ];
    let colorants: Mat3 = [0, 1, 2].map(|r| [cols[0][r], cols[1][r], cols[2][r]]);
    // A singular colorant matrix covers the all-zero default of absent tags.
    if mat_inverse(&colorants).is_none() {
        return Err(());
    }
    let composite = mat_mul(&pcs_to_linear_srgb(), &colorants);

    let mut linear = Box::new([[0f32; 256]; 3]);
    for (trc, lin) in trcs.iter().zip(linear.iter_mut()) {
        for (v, out) in (0u8..=255).zip(lin.iter_mut()) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a linear-light value needs no more than f32 precision"
            )]
            {
                *out = eval_trc(trc, level(v)).ok_or(())? as f32;
            }
        }
    }

    if !is_identity(&composite) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "matrix coefficients need no more than f32 precision"
        )]
        let matrix = composite.map(|row| row.map(|v| v as f32));
        return Ok(Some(Arc::new(RgbShaper::Matrix { linear, matrix })));
    }
    let mut tables = Box::new([[0u8; 256]; 3]);
    for (lin, table) in linear.iter().zip(tables.iter_mut()) {
        for (l, out) in lin.iter().zip(table.iter_mut()) {
            *out = unit_to_u8(srgb_encode(f64::from(*l)));
        }
    }
    Ok(if tables.iter().all(is_identity_table) {
        None
    } else {
        Some(Arc::new(RgbShaper::PerChannel(tables)))
    })
}

// ── Profile cache ─────────────────────────────────────────────────────────────

/// The conversion for `bytes`, memoised process-wide by profile content.
///
/// The cache holds at most [`CACHE_CAPACITY`] profiles and is flushed when
/// full; a profile is evaluated outside the lock, so concurrent first uses
/// of one profile may each evaluate it.
pub(super) fn conversion_for_profile(bytes: &[u8]) -> ProfileConversion {
    static CACHE: OnceLock<Mutex<HashMap<Box<[u8]>, ProfileConversion>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(bytes)
    {
        return hit.clone();
    }
    let conv = ProfileConversion::from_profile_bytes(bytes);
    let mut map = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if map.len() >= CACHE_CAPACITY {
        map.clear();
    }
    let _ = map.insert(bytes.into(), conv.clone());
    conv
}

/// The conversion for the ICC profile stream `stream_id`;
/// [`ProfileConversion::Unsupported`] when the stream cannot be read.
pub(super) fn conversion_for_stream(doc: &Document, stream_id: ObjectId) -> ProfileConversion {
    let Some(bytes) = fetch_profile_bytes(doc, stream_id) else {
        return ProfileConversion::Unsupported;
    };
    conversion_for_profile(&bytes)
}

/// Decompressed content of the ICC profile stream `stream_id`.
pub(super) fn fetch_profile_bytes(doc: &Document, stream_id: ObjectId) -> Option<Vec<u8>> {
    let obj = doc.get_object(stream_id).ok()?;
    obj.as_ref()
        .as_stream()?
        .decompressed_content()
        .map_err(|e| log::debug!("icc: profile stream decompression failed: {e}"))
        .ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `u8Fixed8Number` exponent 1.8 as written in a one-entry `curv`.
    const GAMMA_1_8: u16 = 461;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn profile_with(curve: &ToneReprCurve) -> Vec<u8> {
        let mut p = ColorProfile::new_srgb();
        p.red_trc = Some(curve.clone());
        p.green_trc = Some(curve.clone());
        p.blue_trc = Some(curve.clone());
        p.cicp = None;
        p.encode().expect("encode profile")
    }

    // ── Curves ────────────────────────────────────────────────────────────────

    #[test]
    fn empty_curv_is_identity() {
        let c = ToneReprCurve::Lut(vec![]);
        for x in [0.0, 0.25, 0.7, 1.0] {
            assert!(close(eval_trc(&c, x).unwrap(), x, 1e-12));
        }
    }

    #[test]
    fn one_entry_curv_is_u8fixed8_gamma() {
        let c = ToneReprCurve::Lut(vec![GAMMA_1_8]);
        let g = 461.0 / 256.0;
        assert!(close(eval_trc(&c, 0.5).unwrap(), 0.5f64.powf(g), 1e-12));
        assert!(close(eval_trc(&c, 1.0).unwrap(), 1.0, 1e-12));
        assert!(close(eval_trc(&c, 0.0).unwrap(), 0.0, 1e-12));
    }

    #[test]
    fn sampled_curv_interpolates_linearly() {
        let c = ToneReprCurve::Lut(vec![0, 16_384, 65_535]);
        assert!(close(
            eval_trc(&c, 0.5).unwrap(),
            16_384.0 / 65_535.0,
            1e-12
        ));
        let quarter = 8_192.0 / 65_535.0;
        assert!(close(eval_trc(&c, 0.25).unwrap(), quarter, 1e-12));
        assert!(close(eval_trc(&c, 1.0).unwrap(), 1.0, 1e-12));
        // Out-of-range input clamps to the domain.
        assert!(close(eval_trc(&c, 2.0).unwrap(), 1.0, 1e-12));
    }

    #[test]
    fn parametric_type0_is_pure_power() {
        let c = ToneReprCurve::Parametric(vec![2.2]);
        assert!(close(eval_trc(&c, 0.5).unwrap(), 0.5f64.powf(2.2), 1e-6));
    }

    #[test]
    fn parametric_type1_is_zero_below_the_threshold() {
        // y = (2x − 0.5)^2 for x ≥ 0.25, else 0.
        let c = ToneReprCurve::Parametric(vec![2.0, 2.0, -0.5]);
        assert!(close(eval_trc(&c, 0.1).unwrap(), 0.0, 1e-12));
        assert!(close(eval_trc(&c, 0.75).unwrap(), 1.0, 1e-6));
    }

    #[test]
    fn parametric_type2_adds_the_offset_everywhere() {
        // y = (2x − 0.5)^1 + 0.1 for x ≥ 0.25, else 0.1.
        let c = ToneReprCurve::Parametric(vec![1.0, 2.0, -0.5, 0.1]);
        assert!(close(eval_trc(&c, 0.0).unwrap(), 0.1, 1e-6));
        assert!(close(eval_trc(&c, 0.5).unwrap(), 0.6, 1e-6));
    }

    #[test]
    fn parametric_type3_is_the_srgb_decoding() {
        let c =
            ToneReprCurve::Parametric(vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45]);
        // Linear segment and power segment of IEC 61966-2-1.
        assert!(close(eval_trc(&c, 0.02).unwrap(), 0.02 / 12.92, 1e-6));
        let want = ((0.5 + 0.055) / 1.055f64).powf(2.4);
        assert!(close(eval_trc(&c, 0.5).unwrap(), want, 1e-5));
        // Round trip through the encoder.
        for v in [0.01, 0.2, 0.5, 0.9] {
            assert!(close(srgb_encode(eval_trc(&c, v).unwrap()), v, 1e-5));
        }
    }

    #[test]
    fn parametric_type4_uses_both_offsets() {
        // x ≥ 0.5: (x)^1 + 0.1; else 0.5x + 0.05.
        let c = ToneReprCurve::Parametric(vec![1.0, 1.0, 0.0, 0.5, 0.5, 0.1, 0.05]);
        assert!(close(eval_trc(&c, 0.2).unwrap(), 0.15, 1e-6));
        assert!(close(eval_trc(&c, 0.8).unwrap(), 0.9, 1e-6));
    }

    #[test]
    fn parametric_with_undefined_count_is_rejected() {
        assert_eq!(
            eval_trc(&ToneReprCurve::Parametric(vec![1.0, 2.0]), 0.5),
            None
        );
    }

    // ── Matrix ────────────────────────────────────────────────────────────────

    #[test]
    fn matrix_inverse_round_trips() {
        let inv = mat_inverse(&SRGB_D50_COLORANTS).unwrap();
        assert!(is_identity(&mat_mul(&inv, &SRGB_D50_COLORANTS)));
        assert!(mat_inverse(&[[0.0; 3]; 3]).is_none());
    }

    #[test]
    fn srgb_white_maps_to_white() {
        // The colorant columns of an RGB profile sum to the D50 PCS white.
        let m = pcs_to_linear_srgb();
        let d50 = [0.9642, 1.0, 0.8249];
        for row in m {
            let v: f64 = row.iter().zip(d50).map(|(a, b)| a * b).sum();
            assert!(close(v, 1.0, 1e-3), "{v}");
        }
    }

    // ── Profiles ──────────────────────────────────────────────────────────────

    #[test]
    fn srgb_profile_is_identity() {
        let bytes = ColorProfile::new_srgb().encode().unwrap();
        assert_eq!(
            ProfileConversion::from_profile_bytes(&bytes),
            ProfileConversion::Rgb(None)
        );
    }

    #[test]
    fn gamma_1_8_rgb_profile_lightens_mid_tones() {
        let bytes = profile_with(&ToneReprCurve::Lut(vec![GAMMA_1_8]));
        let ProfileConversion::Rgb(Some(shaper)) = ProfileConversion::from_profile_bytes(&bytes)
        else {
            panic!("gamma 1.8 on sRGB primaries is a per-channel shaper");
        };
        assert!(matches!(*shaper, RgbShaper::PerChannel(_)));
        // encode((128/255)^1.8) = 0.5740 → 146.
        let [r, g, b] = shaper.convert([128, 128, 128]);
        assert_eq!((r, g, b), (146, 146, 146));
        assert_eq!(shaper.convert([0, 0, 0]), [0, 0, 0]);
        assert_eq!(shaper.convert([255, 255, 255]), [255, 255, 255]);
    }

    #[test]
    fn wide_gamut_primaries_use_the_matrix() {
        // Adobe RGB (1998): wider green primary than sRGB.
        let mut p = ColorProfile::new_adobe_rgb();
        p.cicp = None;
        let bytes = p.encode().unwrap();
        let ProfileConversion::Rgb(Some(shaper)) = ProfileConversion::from_profile_bytes(&bytes)
        else {
            panic!("Adobe RGB needs a matrix");
        };
        assert!(matches!(*shaper, RgbShaper::Matrix { .. }));
        // Neutrals stay neutral; saturated green leaves the sRGB gamut and
        // clips with red at zero.
        let [r, g, b] = shaper.convert([128, 128, 128]);
        assert!(r.abs_diff(g) <= 1 && g.abs_diff(b) <= 1, "{r} {g} {b}");
        assert_eq!(shaper.convert([255, 255, 255]), [255, 255, 255]);
        let [r, g, _] = shaper.convert([0, 255, 0]);
        assert_eq!((r, g), (0, 255));
    }

    #[test]
    fn matrix_path_agrees_with_the_moxcms_transform() {
        use moxcms::{Layout, TransformOptions};
        let mut p = ColorProfile::new_adobe_rgb();
        p.cicp = None;
        let bytes = p.encode().unwrap();
        let ProfileConversion::Rgb(Some(shaper)) = ProfileConversion::from_profile_bytes(&bytes)
        else {
            panic!("Adobe RGB needs a matrix");
        };
        let src = ColorProfile::new_from_slice(&bytes).unwrap();
        let xform = src
            .create_transform_8bit(
                Layout::Rgb,
                &ColorProfile::new_srgb(),
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        for px in [
            [200, 100, 50],
            [30, 160, 220],
            [128, 128, 128],
            [10, 20, 30],
        ] {
            let mut want = [0u8; 3];
            xform.transform(&px, &mut want).unwrap();
            let got = shaper.convert(px);
            for (g, w) in got.iter().zip(want) {
                assert!(g.abs_diff(w) <= 2, "{px:?}: got {got:?}, moxcms {want:?}");
            }
        }
    }

    #[test]
    fn gray_profiles() {
        let mut p = ColorProfile::new_gray_with_gamma(1.0);
        p.gray_trc = Some(ToneReprCurve::Lut(vec![GAMMA_1_8]));
        let ProfileConversion::Gray(Some(shaper)) =
            ProfileConversion::from_profile_bytes(&p.encode().unwrap())
        else {
            panic!("gamma 1.8 gray is not sRGB");
        };
        assert_eq!(shaper.convert(128), 146);

        p.gray_trc = Some(ToneReprCurve::Parametric(vec![
            2.4,
            1.0 / 1.055,
            0.055 / 1.055,
            1.0 / 12.92,
            0.040_45,
        ]));
        assert_eq!(
            ProfileConversion::from_profile_bytes(&p.encode().unwrap()),
            ProfileConversion::Gray(None),
            "an sRGB-curve gray profile is the identity"
        );
    }

    #[test]
    fn garbage_is_unsupported() {
        assert_eq!(
            ProfileConversion::from_profile_bytes(b"not a profile"),
            ProfileConversion::Unsupported
        );
    }

    #[test]
    fn convert_components_checks_the_channel_count() {
        let gray = ProfileConversion::Gray(None);
        assert_eq!(gray.convert_components(&[0.5]), Some([128; 3]));
        assert_eq!(gray.convert_components(&[0.5, 0.5, 0.5]), None);
        let rgb = ProfileConversion::Rgb(None);
        assert_eq!(
            rgb.convert_components(&[1.0, 0.0, 0.5]),
            Some([255, 0, 128])
        );
        assert_eq!(
            ProfileConversion::Unsupported.convert_components(&[0.5]),
            None
        );
    }

    #[test]
    fn convert_in_place_matches_per_pixel() {
        let bytes = profile_with(&ToneReprCurve::Lut(vec![GAMMA_1_8]));
        let ProfileConversion::Rgb(Some(shaper)) = conversion_for_profile(&bytes) else {
            panic!("shaper expected");
        };
        let mut buf = vec![10, 20, 30, 128, 64, 200, 7];
        shaper.convert_in_place(&mut buf);
        assert_eq!(&buf[..3], &shaper.convert([10, 20, 30]));
        assert_eq!(&buf[3..6], &shaper.convert([128, 64, 200]));
        assert_eq!(buf[6], 7, "a trailing partial pixel is left alone");
        // The cached result is the same evaluation.
        assert_eq!(
            conversion_for_profile(&bytes),
            ProfileConversion::from_profile_bytes(&bytes)
        );
    }
}
