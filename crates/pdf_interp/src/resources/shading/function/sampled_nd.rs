//! Type 0 (Sampled) function evaluation over an `m`-dimensional sample grid.
//!
//! PDF §7.10.2: each input is clipped to its `Domain` interval and mapped
//! through `Encode` onto the grid axis of `Size[i]` samples; the output is
//! the multilinear interpolation of the `2^m` surrounding samples, mapped
//! through `Decode` (default `Range`).  The first input varies fastest in
//! the sample stream.  `Order 3` (cubic) is evaluated as `Order 1`, which
//! the specification permits when cubic interpolation is unavailable.

use pdf::{Dictionary, Object};

use super::read_fn_domains;
use crate::resources::obj_to_f64;

/// Upper bound on the input dimension.  The interpolation visits `2^m`
/// grid corners; real tint transforms take a handful of colorants.
const MAX_INPUTS: usize = 16;

/// Evaluate a Type 0 function with dictionary `fn_dict` and decoded sample
/// stream `samples` at `inputs`, returning `n_out` outputs.  `None` when the
/// dictionary is malformed, `Size` does not have one entry per input, or
/// the stream is shorter than the grid it declares.
pub(super) fn eval_sampled_nd(
    fn_dict: &Dictionary,
    samples: &[u8],
    inputs: &[f64],
    n_out: usize,
) -> Option<Vec<f64>> {
    let dims = inputs.len();
    if dims == 0 || dims > MAX_INPUTS || n_out == 0 {
        return None;
    }
    let size: Vec<usize> = fn_dict
        .get(b"Size")?
        .as_array()?
        .iter()
        .map(|o| o.as_u32().map(|v| v as usize))
        .collect::<Option<_>>()?;
    if size.len() != dims || size.contains(&0) {
        return None;
    }
    let bps = fn_dict.get(b"BitsPerSample").and_then(Object::as_u32)?;
    if !matches!(bps, 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32) {
        return None;
    }
    let total_bits = size
        .iter()
        .try_fold(n_out, |acc, &axis| acc.checked_mul(axis))?
        .checked_mul(bps as usize)?;
    if samples.len().checked_mul(8)? < total_bits {
        log::warn!(
            "function: Type 0 sample stream truncated ({} bytes, grid needs {total_bits} bits)",
            samples.len()
        );
        return None;
    }

    let encode = read_pairs(fn_dict, b"Encode");
    let decode = read_pairs(fn_dict, b"Decode");
    let range = read_pairs(fn_dict, b"Range");
    let domains = read_fn_domains(fn_dict, dims);

    // Per axis: lower grid index, upper grid index, interpolation weight of
    // the upper index, and the axis stride in samples.
    let mut axes = Vec::with_capacity(dims);
    let mut stride = 1usize;
    for (axis, (&input, &(d0, d1))) in inputs.iter().zip(&domains).enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a grid axis that fits the sample stream is far below 2^52"
        )]
        let last = (size[axis] - 1) as f64;
        let (e0, e1) = encode.get(axis).copied().unwrap_or((0.0, last));
        let unit = if d1 > d0 {
            (input.clamp(d0, d1) - d0) / (d1 - d0)
        } else {
            0.0
        };
        let pos = unit.mul_add(e1 - e0, e0).clamp(0.0, last);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos is finite and clamped to [0, size - 1]"
        )]
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(size[axis] - 1);
        axes.push((lo, hi, pos - pos.floor(), stride));
        stride = stride.checked_mul(size[axis])?;
    }

    let max_sample = f64::from(u32::MAX >> (32 - bps));
    let mut out = vec![0.0; n_out];
    for corner in 0..1usize << dims {
        let mut weight = 1.0;
        let mut index = 0usize;
        for (bit, &(lo, hi, frac, axis_stride)) in axes.iter().enumerate() {
            let (grid_index, axis_weight) = if corner >> bit & 1 == 1 {
                (hi, frac)
            } else {
                (lo, 1.0 - frac)
            };
            weight *= axis_weight;
            index += grid_index * axis_stride;
        }
        if weight <= 0.0 {
            continue;
        }
        for (ch, acc) in out.iter_mut().enumerate() {
            let raw = read_sample(samples, (index * n_out + ch) * bps as usize, bps);
            *acc = weight.mul_add(f64::from(raw), *acc);
        }
    }

    Some(
        out.into_iter()
            .enumerate()
            .map(|(ch, v)| {
                let (lo, hi) = decode
                    .get(ch)
                    .or_else(|| range.get(ch))
                    .copied()
                    .unwrap_or((0.0, 1.0));
                (v / max_sample).mul_add(hi - lo, lo)
            })
            .collect(),
    )
}

/// Read `key` as consecutive `(lo, hi)` pairs; empty when absent.
fn read_pairs(dict: &Dictionary, key: &[u8]) -> Vec<(f64, f64)> {
    let vals: Vec<f64> = dict
        .get(key)
        .and_then(Object::as_array)
        .map(|a| a.iter().filter_map(obj_to_f64).collect())
        .unwrap_or_default();
    vals.as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| pair.into())
        .collect()
}

/// Read the `bps`-bit big-endian sample starting at bit `bit_off`.  The
/// caller has checked that the stream holds every sample of the grid.
fn read_sample(samples: &[u8], bit_off: usize, bps: u32) -> u32 {
    let mut v = 0u32;
    for k in 0..bps as usize {
        let bit = bit_off + k;
        let byte = samples.get(bit / 8).copied().unwrap_or(0);
        v = (v << 1) | u32::from(byte >> (7 - bit % 8) & 1);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[0 1 0 1 …]` with `pairs` unit intervals.
    fn unit_intervals(pairs: usize) -> Object {
        Object::Array(
            (0..pairs)
                .flat_map(|_| [Object::Integer(0), Object::Integer(1)])
                .collect(),
        )
    }

    fn dict(size: &[i64], bps: i64, range_len: usize) -> Dictionary {
        let mut d = Dictionary::new();
        d.set("FunctionType", Object::Integer(0));
        d.set("Domain", unit_intervals(size.len()));
        d.set("Range", unit_intervals(range_len));
        d.set(
            "Size",
            Object::Array(size.iter().map(|&s| Object::Integer(s)).collect()),
        );
        d.set("BitsPerSample", Object::Integer(bps));
        d
    }

    /// 2×2 grid, one output, first input fastest: samples at
    /// (0,0)=0, (1,0)=255, (0,1)=0, (1,1)=255 → output equals input 0.
    #[test]
    fn bilinear_grid_interpolates_along_the_fast_axis() {
        let d = dict(&[2, 2], 8, 1);
        let samples = [0u8, 255, 0, 255];
        for (x, y) in [(0.0, 0.0), (0.25, 0.9), (1.0, 0.5)] {
            let out = eval_sampled_nd(&d, &samples, &[x, y], 1).unwrap();
            assert!((out[0] - x).abs() < 1e-9, "({x},{y}) -> {out:?}");
        }
    }

    /// Two inputs, two outputs: output `k` copies input `k`.
    #[test]
    fn multi_output_grid_maps_each_axis() {
        let d = dict(&[2, 2], 8, 2);
        // (x,y) corners in stream order (0,0) (1,0) (0,1) (1,1).
        let samples = [0u8, 0, 255, 0, 0, 255, 255, 255];
        let out = eval_sampled_nd(&d, &samples, &[0.2, 0.7], 2).unwrap();
        assert!((out[0] - 0.2).abs() < 1e-9 && (out[1] - 0.7).abs() < 1e-9);
    }

    #[test]
    fn sub_byte_samples_are_read_msb_first() {
        let d = dict(&[2, 2], 4, 1);
        // Nibbles 0, F, 0, F.
        let samples = [0x0F, 0x0F];
        let out = eval_sampled_nd(&d, &samples, &[1.0, 0.0], 1).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn truncated_stream_and_size_mismatch_fail() {
        let d = dict(&[2, 2], 8, 1);
        assert!(eval_sampled_nd(&d, &[0, 255, 0], &[0.5, 0.5], 1).is_none());
        assert!(eval_sampled_nd(&d, &[0, 255, 0, 255], &[0.5, 0.5, 0.5], 1).is_none());
    }
}
