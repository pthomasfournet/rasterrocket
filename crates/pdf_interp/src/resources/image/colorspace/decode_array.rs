//! The `/Decode` array of an image (PDF §8.9.5.2).

use color::Sample;
use pdf::{Dictionary, Object};

/// The `/Decode` array of `dict` as `(dmin, dmax)` pairs for `components`
/// components.  `None` when the array is absent, holds fewer than
/// `2 × components` numbers, or holds a non-finite value (a malformed array
/// would otherwise map every sample to black through `NaN as u8 == 0`).
pub(super) fn parse_decode(dict: &Dictionary, components: usize) -> Option<Vec<(f64, f64)>> {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return None;
    };
    let vals: Vec<f64> = arr.iter().filter_map(pdf::Object::as_f64).collect();
    if vals.len() < components * 2 {
        if !arr.is_empty() {
            log::debug!(
                "image: Decode array has {} entries, need {}×2={} — ignoring (identity assumed)",
                arr.len(),
                components,
                components * 2,
            );
        }
        return None;
    }
    let pairs: Vec<(f64, f64)> = vals
        .as_chunks::<2>()
        .0
        .iter()
        .take(components)
        .map(|&pair| pair.into())
        .collect();
    if pairs
        .iter()
        .any(|&(d0, d1)| !d0.is_finite() || !d1.is_finite())
    {
        log::warn!(
            "image: Decode array contains non-finite value(s) — ignoring (identity assumed)"
        );
        return None;
    }
    Some(pairs)
}

/// The [`decode_lut`] of each pair of `decode`.
pub(super) fn decode_luts(decode: &[(f64, f64)]) -> Vec<[u8; 256]> {
    decode
        .iter()
        .map(|&(dmin, dmax)| decode_lut(dmin, dmax))
        .collect()
}

/// The value of the 16-bit sample `s` under the `/Decode` pair
/// `(dmin, dmax)`: `dmin + s · (dmax − dmin) / 65535`.
pub(super) fn decode_wide(s: u16, (dmin, dmax): (f64, f64)) -> f64 {
    s.to_unit().mul_add(dmax - dmin, dmin)
}

/// Table of the `/Decode` pair `(dmin, dmax)` over every 8-bit sample `s`:
/// `round(dmin·255 + s·(dmax − dmin))` clamped to `[0, 255]`, giving values
/// in `[dmin·255, dmax·255]` (or swapped when `dmax < dmin`).
fn decode_lut(dmin: f64, dmax: f64) -> [u8; 256] {
    std::array::from_fn(|i| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "i indexes a 256-entry table"
        )]
        let s = i as u8;
        let remapped = dmin.mul_add(255.0, f64::from(s) * (dmax - dmin));
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "remapped is clamped to [0.0, 255.0] before cast"
        )]
        let out = remapped.round().clamp(0.0, 255.0) as u8;
        out
    })
}

/// Tolerance within which two `Decode` values are equal.  Decode values
/// are parsed PDF reals, so exact values are typical.
const DECODE_EPS: f64 = 1e-9;

/// `true` when `(dmin, dmax)` is `(0, 1)`.
pub(super) fn is_identity_pair(dmin: f64, dmax: f64) -> bool {
    dmin.abs() < DECODE_EPS && (dmax - 1.0).abs() < DECODE_EPS
}

/// Whether `a` and `b` hold the same pairs, to within [`DECODE_EPS`].
pub(super) fn pairs_equal(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&(a0, a1), &(b0, b1))| {
            (a0 - b0).abs() < DECODE_EPS && (a1 - b1).abs() < DECODE_EPS
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::colorspace::test_support::decode_arr;

    #[test]
    fn parse_decode_requires_a_full_array() {
        let mut dict = Dictionary::new();
        assert_eq!(parse_decode(&dict, 1), None);
        dict.set("Decode", decode_arr(&[1]));
        assert_eq!(parse_decode(&dict, 1), None);
        dict.set("Decode", decode_arr(&[1, 0, 0, 1]));
        assert_eq!(parse_decode(&dict, 1), Some(vec![(1.0, 0.0)]));
        assert_eq!(parse_decode(&dict, 2), Some(vec![(1.0, 0.0), (0.0, 1.0)]));
    }

    #[test]
    fn decode_lut_maps_the_endpoints() {
        let lut = decode_lut(1.0, 0.0);
        assert_eq!((lut[0], lut[255], lut[64]), (255, 0, 191));
        let lut = decode_lut(0.0, 0.5);
        assert_eq!((lut[0], lut[255]), (0, 128));
    }
}
