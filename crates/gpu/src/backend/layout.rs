//! The word layout of pixels in device buffers.
//!
//! Kernels read and write device buffers in whole 32-bit words.  A device
//! pixel of three samples occupies `device_pixel_words(sample_bytes)`
//! words: its samples packed little-endian from the low byte of its first
//! word, the bytes past the third sample zero.  One-byte samples take one
//! word (`s0 | s1 << 8 | s2 << 16`), two-byte samples two
//! (`s0 | s1 << 16`, then `s2`).
//!
//! A byte buffer a kernel reads as words, such as a blit source, holds
//! `word_padded(len)` bytes: its `len` bytes, then up to three bytes the
//! kernel loads with the last word and never uses.
//!
//! Pad bytes of device buffers are zero on every backend: those of device
//! pixels are written as zero by the kernels that write the pixels, and
//! those of padded byte buffers by the upload that fills the buffer.  No
//! kernel result depends on their value.

/// The 32-bit words of one three-sample device pixel whose samples are
/// `sample_bytes` bytes each: 1 for 1-byte samples, 2 for 2-byte samples.
///
/// # Panics
///
/// For any other sample width (at compile time in a const context).
#[must_use]
pub const fn device_pixel_words(sample_bytes: usize) -> usize {
    match sample_bytes {
        1 => 1,
        2 => 2,
        _ => panic!("device pixels hold 1- or 2-byte samples"),
    }
}

/// The smallest multiple of 4 that is at least `len`: the bytes of a
/// buffer holding `len` bytes as whole 32-bit words.
///
/// # Panics
///
/// When the result exceeds `usize::MAX` (at compile time in a const
/// context).
#[must_use]
pub const fn word_padded(len: usize) -> usize {
    match len.checked_next_multiple_of(4) {
        Some(padded) => padded,
        None => panic!("word_padded: length overflows usize"),
    }
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
/// For a `sample_bytes` other than 1 or 2, when `dst` does not hold
/// whole pixels, or when `src` holds fewer device pixels than `dst` asks
/// for.
pub fn unpad_pixels(src: &[u8], dst: &mut [u8], sample_bytes: usize) {
    let device = 4 * device_pixel_words(sample_bytes);
    let packed = 3 * sample_bytes;
    let (src_len, dst_len) = (src.len(), dst.len());
    assert!(
        dst_len.is_multiple_of(packed),
        "unpad_pixels: dst length {dst_len} is not a whole number of \
         {packed}-byte pixels (sample_bytes {sample_bytes}, src length {src_len})"
    );
    let n = dst_len / packed;
    assert!(
        src_len / device >= n,
        "unpad_pixels: src length {src_len} holds fewer than the {n} \
         {device}-byte device pixels dst length {dst_len} asks for \
         (sample_bytes {sample_bytes})"
    );
    let src = &src[..n * device];
    if sample_bytes == 1 {
        unpad_words_8(src, dst);
    } else {
        unpad_words_16(src, dst);
    }
}

/// The little-endian word at byte `4 × i` of `s`.
fn word<const N: usize>(s: &[u8; N], i: usize) -> u32 {
    u32::from_le_bytes([s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]])
}

/// [`unpad_pixels`] of 1-byte samples, `src` holding exactly the device
/// pixels of `dst`: four device pixels, words `p0..p3`, pack into the
/// three words `p0 | p1 << 24`, `p1 >> 8 | p2 << 16`, `p2 >> 16 | p3 << 8`,
/// the pad byte of each `p` being past the shifted-out bits.
fn unpad_words_8(src: &[u8], dst: &mut [u8]) {
    let (dst_quads, dst_tail) = dst.as_chunks_mut::<12>();
    let (src_quads, src_tail) = src.as_chunks::<16>();
    for (d, s) in dst_quads.iter_mut().zip(src_quads) {
        let p = [word(s, 0), word(s, 1), word(s, 2), word(s, 3)];
        let packed = [
            p[0] & 0x00FF_FFFF | p[1] << 24,
            (p[1] >> 8) & 0xFFFF | p[2] << 16,
            (p[2] >> 16) & 0xFF | p[3] << 8,
        ];
        for (to, w) in d.as_chunks_mut::<4>().0.iter_mut().zip(packed) {
            *to = w.to_le_bytes();
        }
    }
    let (dst_px, _) = dst_tail.as_chunks_mut::<3>();
    for (to, from) in dst_px.iter_mut().zip(src_tail.as_chunks::<4>().0) {
        *to = [from[0], from[1], from[2]];
    }
}

/// [`unpad_pixels`] of 2-byte samples, `src` holding exactly the device
/// pixels of `dst`: two device pixels, words `(a0, b0)` and `(a1, b1)`,
/// pack into the three words `a0`, `b0 | a1 << 16`, `a1 >> 16 | b1 << 16`,
/// the pad bytes of each `b` being past the shifted-out bits.
fn unpad_words_16(src: &[u8], dst: &mut [u8]) {
    let (dst_pairs, dst_tail) = dst.as_chunks_mut::<12>();
    let (src_pairs, src_tail) = src.as_chunks::<16>();
    for (d, s) in dst_pairs.iter_mut().zip(src_pairs) {
        let [a0, b0, a1, b1] = [word(s, 0), word(s, 1), word(s, 2), word(s, 3)];
        let packed = [a0, b0 & 0xFFFF | a1 << 16, a1 >> 16 | b1 << 16];
        for (to, w) in d.as_chunks_mut::<4>().0.iter_mut().zip(packed) {
            *to = w.to_le_bytes();
        }
    }
    let (dst_px, _) = dst_tail.as_chunks_mut::<6>();
    for (to, from) in dst_px.iter_mut().zip(src_tail.as_chunks::<8>().0) {
        *to = [from[0], from[1], from[2], from[3], from[4], from[5]];
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
    #[should_panic(expected = "1- or 2-byte samples")]
    fn device_pixel_words_rejects_other_widths() {
        let _ = device_pixel_words(3);
    }

    #[test]
    fn word_padded_rounds_up_to_whole_words() {
        let padded: Vec<usize> = (0..=9).map(word_padded).collect();
        assert_eq!(padded, [0, 4, 4, 4, 4, 8, 8, 8, 8, 12]);
        assert_eq!(word_padded(usize::MAX - 3), usize::MAX - 3);
    }

    #[test]
    #[should_panic(expected = "length overflows usize")]
    fn word_padded_rejects_an_overflowing_length() {
        let _ = word_padded(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "dst length 4 is not a whole number of 3-byte pixels")]
    fn unpad_pixels_rejects_a_partial_dst_pixel() {
        unpad_pixels(&[0; 8], &mut [0; 4], 1);
    }

    #[test]
    #[should_panic(expected = "src length 7 holds fewer than the 2 4-byte device pixels")]
    fn unpad_pixels_rejects_a_short_src() {
        unpad_pixels(&[0; 7], &mut [0; 6], 1);
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
    fn unpad_pixels_matches_a_per_pixel_copy_at_every_tail_length() {
        for sample_bytes in [1, 2] {
            let device = 4 * device_pixel_words(sample_bytes);
            let packed = 3 * sample_bytes;
            for n in 0..=13 {
                let src: Vec<u8> = (0..n * device + 5)
                    .map(|i| u8::try_from((i * 37 + 11) % 251).unwrap())
                    .collect();
                let want: Vec<u8> = src
                    .chunks_exact(device)
                    .take(n)
                    .flat_map(|p| p[..packed].to_vec())
                    .collect();
                let mut got = vec![0xA5; n * packed];
                unpad_pixels(&src, &mut got, sample_bytes);
                assert_eq!(got, want, "{sample_bytes}-byte samples, {n} pixels");
            }
        }
    }

    #[test]
    fn unpad_pixels_reads_no_more_than_dst_asks_for() {
        let src = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0];
        let mut dst = [0u8; 3];
        unpad_pixels(&src, &mut dst, 1);
        assert_eq!(dst, [1, 2, 3]);
    }
}
