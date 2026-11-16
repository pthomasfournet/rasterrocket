//! The bounds every image decode observes: the declared dimensions, the
//! decoded-sample budget, and fallible allocation of sample buffers.

/// Largest `/Width` or `/Height` an image may declare.
const MAX_IMAGE_SIDE: i64 = 131_071;

/// Largest number of bytes of samples (`width × height × components`, two
/// bytes a sample kept at 16 bits) an image is stored or converted to:
/// 1 GiB.  A larger image is refused before any
/// allocation and reported as a decode failure.  A JPEG decode whose
/// decoder holds the frame's coefficients whole fits them in what the
/// samples leave of it.
pub(super) const MAX_DECODED_BYTES: u64 = 1 << 30;

/// Validate raw `i64` image dimensions and cast them to `u32`.
///
/// Returns `None` (caller logs and propagates) if either dimension is ≤ 0 or
/// exceeds [`MAX_IMAGE_SIDE`].  The product of two sides fits `u64`; the
/// decode budget ([`decode_budget`]) bounds it further.
///
/// The cast is safe: after the range check, each dimension is in
/// `[1, MAX_IMAGE_SIDE]`, which fits in `u32` without sign loss or
/// truncation.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "value range validated to [1, MAX_IMAGE_SIDE] — fits in u32 without loss"
)]
pub(crate) const fn validated_dims(w_raw: i64, h_raw: i64) -> Option<(u32, u32)> {
    if w_raw <= 0 || h_raw <= 0 || w_raw > MAX_IMAGE_SIDE || h_raw > MAX_IMAGE_SIDE {
        return None;
    }
    Some((w_raw as u32, h_raw as u32))
}

/// Whether a `width × height` image of `components` samples per pixel fits
/// [`MAX_DECODED_BYTES`]; the message names the excess otherwise.
pub(super) fn decode_budget(width: u32, height: u32, components: usize) -> Result<(), String> {
    decode_budget_of(width, height, components, MAX_DECODED_BYTES)
}

/// [`decode_budget`] against a budget of `budget` bytes.
pub(super) fn decode_budget_of(
    width: u32,
    height: u32,
    components: usize,
    budget: u64,
) -> Result<(), String> {
    let bytes = decoded_bytes(width, height, components);
    if bytes <= budget {
        return Ok(());
    }
    Err(format!(
        "image of {width}×{height}×{components} samples ({} MiB) exceeds the {} MiB decode budget",
        bytes >> 20,
        budget >> 20
    ))
}

/// The 8-bit samples of a `width × height` image of `components` samples
/// per pixel.
pub(super) fn decoded_bytes(width: u32, height: u32, components: usize) -> u64 {
    u64::from(width) * u64::from(height) * components as u64
}

/// A vector of `len` copies of `fill`, or `None` when the allocation
/// fails; a decoder refuses an image it cannot hold instead of aborting
/// the process.
pub(super) fn try_filled(len: usize, fill: u8) -> Option<Vec<u8>> {
    let mut v = try_with_capacity(len)?;
    v.resize(len, fill);
    Some(v)
}

/// An empty vector with room for `len` bytes, or `None` when the
/// allocation fails.
pub(super) fn try_with_capacity(len: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|e| log::warn!("image: cannot allocate {len} bytes: {e}"))
        .ok()?;
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_dims_rejects_zero() {
        assert!(validated_dims(0, 100).is_none());
        assert!(validated_dims(100, 0).is_none());
    }

    #[test]
    fn validated_dims_rejects_negative() {
        assert!(validated_dims(-1, 100).is_none());
        assert!(validated_dims(100, -1).is_none());
    }

    #[test]
    fn validated_dims_rejects_oversized() {
        assert!(validated_dims(MAX_IMAGE_SIDE + 1, 1).is_none());
        assert!(validated_dims(1, MAX_IMAGE_SIDE + 1).is_none());
    }

    #[test]
    fn validated_dims_accepts_boundary() {
        assert_eq!(validated_dims(1, 1), Some((1, 1)));
        assert_eq!(
            validated_dims(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
            Some((131_071, 131_071))
        );
    }

    #[test]
    fn decode_budget_bounds_the_sample_count() {
        assert!(decode_budget(131_071, 2, 1).is_ok());
        assert!(decode_budget(32_768, 32_768, 1).is_ok());
        assert!(decode_budget(32_768, 32_768, 3).is_err());
        assert!(decode_budget(65_536, 65_536, 1).is_err());
        let msg = decode_budget(65_536, 65_536, 1).unwrap_err();
        assert!(msg.contains("budget"), "{msg}");
    }
}
