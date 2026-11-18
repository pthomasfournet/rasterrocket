//! The word layout of pixels in device buffers.
//!
//! Kernels read and write device buffers in whole 32-bit words.  A device
//! pixel of three samples occupies `device_pixel_words(sample_bytes)`
//! words: its samples packed little-endian from the low byte of its first
//! word, the bytes past the third sample zero.  One-byte samples take one
//! word (`s0 | s1 << 8 | s2 << 16`), two-byte samples two
//! (`s0 | s1 << 16`, then `s2`).

/// The 32-bit words of one three-sample device pixel whose samples are
/// `sample_bytes` bytes each: 1 for 1-byte samples, 2 for 2-byte samples.
pub(crate) const fn device_pixel_words(sample_bytes: usize) -> usize {
    sample_bytes
}

/// Copies the three samples of each device pixel of `src`, the first
/// `3 × sample_bytes` bytes of each `4 × device_pixel_words(sample_bytes)`
/// bytes, into consecutive packed pixels of `dst`.
///
/// `dst.len()` decides the pixel count: `dst.len() / (3 × sample_bytes)`
/// pixels are copied, and `src` holds at least as many device pixels.
///
/// # Panics
///
/// When `dst` does not hold whole pixels or `src` holds fewer device
/// pixels than `dst` asks for.
pub(crate) fn unpad_pixels(src: &[u8], dst: &mut [u8], sample_bytes: usize) {
    let packed = 3 * sample_bytes;
    let device = 4 * device_pixel_words(sample_bytes);
    assert!(
        dst.len().is_multiple_of(packed),
        "dst holds whole pixels of {packed} bytes"
    );
    let n = dst.len() / packed;
    assert!(
        src.len() / device >= n,
        "src holds {n} device pixels of {device} bytes"
    );
    for (to, from) in dst.chunks_exact_mut(packed).zip(src.chunks_exact(device)) {
        to.copy_from_slice(&from[..packed]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_pixel_words_of_each_sample_width() {
        assert_eq!(device_pixel_words(1), 1);
        assert_eq!(device_pixel_words(2), 2);
    }

    #[test]
    fn unpad_pixels_keeps_the_samples_of_each_device_pixel() {
        let src8 = [1, 2, 3, 0, 4, 5, 6, 0];
        let mut dst8 = [0u8; 6];
        unpad_pixels(&src8, &mut dst8, 1);
        assert_eq!(dst8, [1, 2, 3, 4, 5, 6]);
        let src16 = [1, 2, 3, 4, 5, 6, 0, 0];
        let mut dst16 = [0u8; 6];
        unpad_pixels(&src16, &mut dst16, 2);
        assert_eq!(dst16, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn unpad_pixels_reads_no_more_than_dst_asks_for() {
        let src = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
        let mut dst = [0u8; 3];
        unpad_pixels(&src, &mut dst, 1);
        assert_eq!(dst, [1, 2, 3]);
    }
}
