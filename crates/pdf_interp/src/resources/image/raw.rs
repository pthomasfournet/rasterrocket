//! Raw (unfiltered or transport-filtered) image decoding: expansion of
//! packed samples to one byte each.
//!
//! [`decode_raw`] dispatches on `BitsPerComponent` and yields the image's
//! samples in its own colour space; colour conversion is the caller's
//! (`ImageColour::convert`).

use pdf::{Dictionary, Object};

use super::bitpack::{downsample_16bpp, expand_nbpp, expand_nbpp_indexed, unpack_packed_bits};
use super::colorspace::{ImageColour, Layout, Samples};
use super::smask::{SoftMask, colour_key_mask_with};
use crate::resources::dict_ext::DictExt as _;

// ── Public entry point ────────────────────────────────────────────────────────

/// Expand raw (already-decompressed) pixel bytes to one byte per sample.
///
/// Supported `BitsPerComponent` values:
/// - 1  — packed MSB-first; 1 sample per component per pixel (PDF §8.9.3)
/// - 2  — 4 levels, scaled to 0x00/0x55/0xAA/0xFF
/// - 4  — 16 levels, scaled to 0x00…0xFF (value × 17)
/// - 8  — direct (common case)
/// - 16 — big-endian, high byte taken (linear 0–65535 → 0–255)
///
/// The palette indices of an `Indexed` image at 1, 2 or 4 bits are
/// unpacked without scaling.  A stencil (`is_mask`) decodes to
/// [`Layout::Mask`] bytes with its `/Decode` polarity applied.
///
/// A colour-key `/Mask` array (§8.9.6.4) on a non-stencil image becomes the
/// samples' mask, compared on the stored samples: see
/// [`raw_colour_key_mask`].  Data shorter than the dictionary declares is
/// padded and reported through [`Samples::partial`].
pub(super) fn decode_raw(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    dict: &Dictionary,
    colour: &ImageColour,
) -> Option<Samples> {
    // `BitsPerComponent` is optional for an image mask, which is 1 bit per
    // sample (§8.9.6.2); other images default to 8.
    let bpc = dict
        .get_i64(b"BitsPerComponent")
        .unwrap_or(if is_mask { 1 } else { 8 });

    if is_mask {
        // PDF 32000-1 §8.9.6.2: under the default `Decode [0 1]` a sample
        // of 0 marks the page with the fill colour and a sample of 1 leaves
        // it unchanged; `Decode [1 0]` swaps the two.  Every stencil source
        // (raw, JBIG2, CCITTFax) is normalised to the internal
        // `ImageColorSpace::Mask` bytes 0x00 = paint, 0xFF = transparent.
        // The bilevel codecs emit decoded sample 0 (black) as 0x00, so a raw
        // sample maps to the same byte value as a decoded bilevel sample.
        let mut out = decode_mask_raw(data, width, height, bpc, is_mask_inverted(dict))?;
        out.partial = short_stream(data, width, height, bpc, 1);
        return Some(out);
    }

    let components = colour.components();
    let mut samples = if colour.indexed() {
        expand_indices(data, width, height, bpc)?
    } else {
        expand_to_8bpc(data, width, height, bpc, components)?.into_owned()
    };
    // Exactly `width × height` pixels: 8-bit data may carry trailing bytes.
    samples.truncate(
        (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(components)?,
    );
    let mut out = Samples::cpu(
        width,
        height,
        Layout::Color(u8::try_from(components).ok()?),
        samples,
    );
    out.partial = short_stream(data, width, height, bpc, components);
    if let Some(Object::Array(ranges)) = dict.get(b"Mask") {
        out.smask = raw_colour_key_mask(data, width, height, bpc, components, ranges);
        if out.smask.is_none() {
            log::warn!(
                "image: colour-key /Mask malformed or mismatched — blitting image without mask"
            );
        }
    }
    Some(out)
}

/// The colour-key mask (§8.9.6.4) of a raw image with key ranges `ranges`.
///
/// The ranges are compared with the samples as stored: palette indices for
/// `Indexed`, colorant values before any colour conversion, values of the
/// image's own bit depth, and before `/Decode`.  Short data is
/// zero-extended as for decoding.  `None` for an unsupported depth or key
/// ranges that do not match the colour space.
fn raw_colour_key_mask(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    components: usize,
    ranges: &[Object],
) -> Option<SoftMask> {
    let Ok(bits @ (1 | 2 | 4 | 8 | 16)) = usize::try_from(bpc) else {
        return None;
    };
    let data = zero_extended(data, width, height, bpc, components)?;
    let w = usize::try_from(width).ok()?;
    // `zero_extended` checked that this product does not overflow.
    let row_bytes = (w * components * bits).div_ceil(8);
    colour_key_mask_with(ranges, components, width, height, |p, c| {
        let row = &data[(p / w) * row_bytes..];
        let i = (p % w) * components + c;
        match bits {
            8 => u32::from(row[i]),
            16 => u32::from(u16::from_be_bytes([row[2 * i], row[2 * i + 1]])),
            _ => {
                let bit = i * bits;
                let mask = (1u8 << bits) - 1;
                u32::from((row[bit / 8] >> (8 - bits - bit % 8)) & mask)
            }
        }
    })
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// The length of `height` rows of `width` pixels of `components` samples
/// at `bpc` bits each, each row padded to a whole byte (PDF §8.9.3).
/// `None` for an unsupported depth or a length that overflows `usize`.
fn declared_len(width: u32, height: u32, bpc: i64, components: usize) -> Option<usize> {
    let Ok(bits @ (1 | 2 | 4 | 8 | 16)) = usize::try_from(bpc) else {
        return None;
    };
    let row_bits = usize::try_from(width)
        .ok()?
        .checked_mul(components)?
        .checked_mul(bits)?;
    row_bits
        .div_ceil(8)
        .checked_mul(usize::try_from(height).ok()?)
}

/// What a stream shorter than [`declared_len`] lacks, for
/// [`Samples::partial`]; `None` for data of at least the full length.
fn short_stream(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    components: usize,
) -> Option<String> {
    let needed = declared_len(width, height, bpc, components)?;
    (data.len() < needed).then(|| {
        format!(
            "raw data short: {} of {needed} bytes; the missing samples are padding",
            data.len()
        )
    })
}

/// `data` extended with zero bytes to the length `height` rows of `width`
/// pixels of `components` samples at `bpc` bits each occupy, each row
/// padded to a whole byte (PDF §8.9.3).
///
/// A raw stream shorter than its dictionary declares (truncated or damaged
/// transport data) keeps the rows it holds; every missing sample is 0.
/// Data of at least the full length, or of an unsupported depth, is
/// borrowed unchanged.  `None` when the length overflows `usize`.
fn zero_extended(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    components: usize,
) -> Option<std::borrow::Cow<'_, [u8]>> {
    use std::borrow::Cow;
    if !matches!(bpc, 1 | 2 | 4 | 8 | 16) {
        return Some(Cow::Borrowed(data));
    }
    let needed = declared_len(width, height, bpc, components)?;
    let bits = bpc;
    if data.len() >= needed {
        return Some(Cow::Borrowed(data));
    }
    log::warn!(
        "image: raw data short ({} of {needed} bytes for {width}×{height}×{components} \
         at {bits} bpc), missing samples are 0",
        data.len()
    );
    let mut padded = Vec::with_capacity(needed);
    padded.extend_from_slice(data);
    padded.resize(needed, 0);
    Some(Cow::Owned(padded))
}

/// Expand `bpc`-bit samples of `components`-component pixels to one byte
/// per sample (PDF §8.9.3).  8-bit data of the full length is borrowed
/// unchanged; short data is first extended by [`zero_extended`].  `None`
/// for an unsupported depth.
fn expand_to_8bpc(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    components: usize,
) -> Option<std::borrow::Cow<'_, [u8]>> {
    use std::borrow::Cow;
    let full = zero_extended(data, width, height, bpc, components)?;
    if bpc == 8 {
        return Some(full);
    }
    let data = full.as_ref();
    Some(match bpc {
        // bpc=1 packs one bit per sample, one sample per component.
        // A row holds `width × components` samples (RGB: 3 bits/px,
        // CMYK: 4 bits/px).
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            let samples_per_row = width_usize.checked_mul(components)?;
            Cow::Owned(unpack_packed_bits(data, 1, samples_per_row, height, |v| {
                if v == 0 { 0x00 } else { 0xFF }
            })?)
        }
        2 => Cow::Owned(expand_nbpp::<2>(data, width, height, components)?),
        4 => Cow::Owned(expand_nbpp::<4>(data, width, height, components)?),
        16 => Cow::Owned(downsample_16bpp(data, width, height, components)?),
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            return None;
        }
    })
}

/// The palette indices of an `Indexed` image, one byte each, unscaled:
/// at 1, 2 or 4 bits they are unpacked MSB-first (PDF §8.6.6.3 allows
/// `bpc ∈ {1, 2, 4, 8}`); at 8 bits they are the data.  `None` for any
/// other depth.
fn expand_indices(data: &[u8], width: u32, height: u32, bpc: i64) -> Option<Vec<u8>> {
    let full = zero_extended(data, width, height, bpc, 1)?;
    match bpc {
        1 | 2 | 4 => {
            #[expect(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "bpc ∈ {1, 2, 4} — positive and fits u32"
            )]
            let bits = bpc as u32;
            expand_nbpp_indexed(&full, width, height, bits)
        }
        8 => Some(full.into_owned()),
        other => {
            log::debug!("image: Indexed with {other} bpc not supported");
            None
        }
    }
}

/// Apply an image dictionary's `/Decode` to the stencil output of a
/// bilevel codec (`JBIG2Decode`, `CCITTFaxDecode`), which leaves its
/// samples unmapped: a stencil ([`Layout::Mask`]) flips polarity under
/// `[1 0]`, as a raw stencil does (see [`is_mask_inverted`]).  Gray
/// samples are left for the colour conversion, which folds `/Decode` in.
pub(super) fn apply_stencil_decode(samples: &mut Samples, dict: &Dictionary) {
    if samples.layout != Layout::Mask || !is_mask_inverted(dict) {
        return;
    }
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            clippy::infallible_destructuring_match,
            reason = "the cache-feature build adds a device-resident arm; keep one shape"
        )
    )]
    let bytes = match &mut samples.data {
        super::ImageData::Cpu(bytes) => bytes,
        #[cfg(feature = "cache")]
        super::ImageData::Gpu(_) => return,
    };
    for b in bytes.iter_mut() {
        *b = !*b;
    }
}

/// Return `true` when the `ImageMask`'s `Decode` array is `[1, 0]` (inverted
/// stencil) per PDF 32000-1 §8.9.6.2.
///
/// Absent `/Decode` is the (non-inverted) default.  A present-but-malformed
/// `/Decode` on a stencil (not exactly two numeric entries, or values that
/// aren't a clean `[0 1]`/`[1 0]`) is treated as the non-inverted default but
/// logged: silently guessing polarity on a garbled stencil array is precisely
/// the inverted-output class this codebase guards against.
fn is_mask_inverted(dict: &Dictionary) -> bool {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return false;
    };
    let vals: Vec<f64> = arr.iter().filter_map(pdf::Object::as_f64).collect();
    match vals.as_slice() {
        // §8.9.6.2: only [0 1] (default) and [1 0] (inverted) are meaningful
        // for a 1-bpc stencil.  Use 0.5 thresholds so [0.0,1.0] reals as well
        // as exact integers classify correctly.
        [v0, v1] if *v0 < 0.5 && *v1 > 0.5 => false, // explicit [0 1] — default
        [v0, v1] if *v0 > 0.5 && *v1 < 0.5 => true,  // [1 0] — inverted
        _ => {
            log::debug!(
                "image: stencil /Decode {arr:?} is not [0 1] or [1 0] — assuming default polarity"
            );
            false
        }
    }
}

/// Decode a stencil mask (ImageMask=true) from raw data.
///
/// With the default `Decode [0 1]`, sample 0 maps to 0x00 (paint) and
/// sample 1 to 0xFF (transparent), per PDF 32000-1 §8.9.6.2.  When `invert`
/// is `true` (explicit `/Decode [1 0]`) the mapping is swapped.
///
/// Only `bpc == 1` is conformant for an `ImageMask` per §8.9.6.2; the `bpc == 8`
/// arm is a defensive fallback for non-conformant producers and is logged.
fn decode_mask_raw(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    invert: bool,
) -> Option<Samples> {
    match bpc {
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            // A truncated stream is padded with the transparent sample
            // value, so missing rows leave the page unchanged.
            let needed = width_usize
                .div_ceil(8)
                .checked_mul(usize::try_from(height).ok()?)?;
            let data: std::borrow::Cow<'_, [u8]> = if data.len() < needed {
                let mut padded = data.to_vec();
                padded.resize(needed, if invert { 0x00 } else { 0xFF });
                std::borrow::Cow::Owned(padded)
            } else {
                std::borrow::Cow::Borrowed(data)
            };
            // Sample 0 paints (0x00) and sample 1 is transparent (0xFF);
            // `invert` (explicit /Decode [1 0]) swaps them.
            let pixels = unpack_packed_bits(&data, 1, width_usize, height, |v| {
                let paint = if invert { v != 0 } else { v == 0 };
                if paint { 0x00 } else { 0xFF }
            })?;
            Some(Samples::mask(width, height, pixels))
        }
        8 => {
            // §8.9.6.2 mandates bpc=1 for an ImageMask; an 8-bpc stencil is a
            // producer bug.  Decode it defensively but say so rather than
            // mis-decoding silently.
            log::debug!(
                "image: ImageMask with non-conformant BitsPerComponent=8 \
                 (§8.9.6.2 requires 1) — decoding as 8-bpc stencil"
            );
            let expected = (width as usize).checked_mul(height as usize)?;
            // As at 1 bpc, a truncated stream is padded with the
            // transparent sample value.
            let mut pixels: Vec<u8> = if invert {
                data.iter().take(expected).map(|&b| 255 - b).collect()
            } else {
                data.iter().take(expected).copied().collect()
            };
            pixels.resize(expected, 0xFF);
            Some(Samples::mask(width, height, pixels))
        }
        other => {
            log::debug!("image: mask {other} bpc not yet implemented");
            None
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::ColorSpace;
    use crate::test_helpers::empty_doc;

    fn decode_dict(values: &[i64]) -> Dictionary {
        let mut dict = Dictionary::new();
        dict.set(
            "Decode",
            Object::Array(values.iter().map(|&v| Object::Integer(v)).collect()),
        );
        dict
    }

    /// Decode `data` as a `w × h` non-stencil image of `cs` at `bpc` bits.
    fn raw(cs: &str, bpc: i64, data: &[u8], w: u32, h: u32) -> Samples {
        let doc = empty_doc();
        let mut dict = Dictionary::new();
        dict.set(
            "ColorSpace",
            pdf::parse_object(cs.as_bytes(), &mut 0).unwrap(),
        );
        dict.set("BitsPerComponent", Object::Integer(bpc));
        let colour = ImageColour::resolve(&doc, &dict, None);
        decode_raw(data, w, h, false, &dict, &colour).expect("decodes")
    }

    #[test]
    fn stencil_decode_inverts_only_a_stencil() {
        let mut mask = Samples::mask(3, 1, vec![0x00, 0xFF, 0x00]);
        apply_stencil_decode(&mut mask, &decode_dict(&[1, 0]));
        assert_eq!(mask.data.as_cpu(), Some(&[0xFF, 0x00, 0xFF][..]));
        let mut gray = Samples::gray(3, 1, vec![0x00, 0xFF, 0x00]);
        apply_stencil_decode(&mut gray, &decode_dict(&[1, 0]));
        assert_eq!(gray.data.as_cpu(), Some(&[0x00, 0xFF, 0x00][..]));
        for dict in [Dictionary::new(), decode_dict(&[0, 1])] {
            let mut mask = Samples::mask(2, 1, vec![0x00, 0xFF]);
            apply_stencil_decode(&mut mask, &dict);
            assert_eq!(mask.data.as_cpu(), Some(&[0x00, 0xFF][..]));
        }
    }

    #[test]
    fn gray_8bpp_is_borrowed_unchanged() {
        let s = raw("/DeviceGray", 8, &[0, 128, 255], 3, 1);
        assert_eq!(s.layout, Layout::Color(1));
        assert_eq!(s.data.as_cpu().unwrap(), &[0, 128, 255]);
    }

    #[test]
    fn rgb_and_cmyk_keep_their_components() {
        let s = raw("/DeviceRGB", 8, &[255, 0, 0, 0, 255, 0, 0, 0, 255], 3, 1);
        assert_eq!(s.layout, Layout::Color(3));
        assert_eq!(s.data.len(), 9);
        let s = raw("/DeviceCMYK", 8, &[0, 0, 0, 0], 1, 1);
        assert_eq!(s.layout, Layout::Color(4));
    }

    /// Data past the declared length is dropped, so the samples hold
    /// exactly `width × height` pixels.
    #[test]
    fn trailing_bytes_are_dropped() {
        let s = raw("/DeviceRGB", 8, &[1, 2, 3, 4, 5, 6, 7, 8], 2, 1);
        assert_eq!(s.data.as_cpu().unwrap(), &[1, 2, 3, 4, 5, 6]);
        let s = raw(
            "[/Indexed /DeviceRGB 1 <ff000000ff00>]",
            8,
            &[0, 1, 1],
            2,
            1,
        );
        assert_eq!(s.data.as_cpu().unwrap(), &[0, 1]);
    }

    /// Short data keeps its samples; the missing ones are 0.
    #[test]
    fn too_short_data_is_zero_extended() {
        let s = raw("/DeviceRGB", 8, &[10, 20, 30], 2, 2);
        let mut expected = vec![0u8; 12];
        expected[..3].copy_from_slice(&[10, 20, 30]);
        assert_eq!(s.data.as_cpu().unwrap(), expected.as_slice());
    }

    #[test]
    fn gray_1bpp_expands_to_full_levels() {
        // 2-pixel Gray image at bpc=1: byte 0b10_000000 → pixels [0xFF, 0x00].
        let s = raw("/DeviceGray", 1, &[0b1000_0000], 2, 1);
        assert_eq!(s.data.as_cpu().unwrap(), &[0xFF, 0x00]);
    }

    #[test]
    fn rgb_1bpp_packs_three_bits_per_pixel() {
        // Byte 0b110_00000: R=1→0xFF, G=1→0xFF, B=0→0x00.
        let s = raw("/DeviceRGB", 1, &[0b1100_0000], 1, 1);
        assert_eq!(s.data.as_cpu().unwrap(), &[0xFF, 0xFF, 0x00]);
    }

    #[test]
    fn indexed_sub_byte_indices_are_not_scaled() {
        // Two 4-bit indices 0x1, 0xF stay 1 and 15.
        let s = raw("[/Indexed /DeviceRGB 15 <00>]", 4, &[0x1F], 2, 1);
        assert_eq!(s.layout, Layout::Color(1));
        assert_eq!(s.data.as_cpu().unwrap(), &[1, 15]);
        let s = raw("[/Indexed /DeviceRGB 1 <00>]", 1, &[0b1010_0000], 4, 1);
        assert_eq!(s.data.as_cpu().unwrap(), &[1, 0, 1, 0]);
    }

    #[test]
    fn indexed_components_is_one() {
        let doc = empty_doc();
        let mut dict = Dictionary::new();
        dict.set(
            "ColorSpace",
            pdf::parse_object(b"[/Indexed /DeviceRGB 1 <ff000000ff00>]", &mut 0).unwrap(),
        );
        let colour = ImageColour::resolve(&doc, &dict, None);
        assert!(colour.indexed());
        assert_eq!(colour.components(), 1);
        assert_eq!(colour.space().map(ColorSpace::ncomponents), Some(1));
    }

    #[test]
    fn colour_key_compares_stored_samples() {
        let doc = empty_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(4));
        dict.set(
            "Mask",
            Object::Array(vec![Object::Integer(15), Object::Integer(15)]),
        );
        let colour = ImageColour::resolve(&doc, &dict, None);
        let s = decode_raw(&[0xF0], 2, 1, false, &dict, &colour).unwrap();
        assert_eq!(s.data.as_cpu().unwrap(), &[0xFF, 0x00]);
        assert_eq!(s.smask.unwrap().alpha(), &[0x00, 0xFF]);
    }

    // ── Stencil masks ─────────────────────────────────────────────────────────

    /// A short 8-bpc stencil leaves its missing samples transparent.
    #[test]
    fn stencil_8bpc_short_buffer_pads_transparent() {
        let s = decode_mask_raw(&[0x00], 2, 1, 8, false).expect("short stencil decodes");
        assert_eq!(s.data.as_cpu().unwrap(), &[0x00, 0xFF]);
        let s = decode_mask_raw(&[0xFF], 2, 1, 8, true).expect("short stencil decodes");
        assert_eq!(s.data.as_cpu().unwrap(), &[0x00, 0xFF]);
    }

    #[test]
    fn stencil_8bpc_decode_inverted() {
        // With Decode=[1,0]: sample 0 → 0xFF (transparent), sample 255 → 0x00 (paint).
        let doc = empty_doc();
        let mut dict = decode_dict(&[1, 0]);
        dict.set("BitsPerComponent", Object::Integer(8));
        let colour = ImageColour::resolve(&doc, &dict, None);
        let s = decode_raw(&[0u8, 255], 2, 1, true, &dict, &colour).unwrap();
        assert_eq!(s.layout, Layout::Mask);
        assert_eq!(s.data.as_cpu().unwrap(), &[0xFF, 0x00]);
    }

    /// Build a dict for a 1-bpc raw `ImageMask`, optionally with a `/Decode`.
    fn stencil_dict(decode: Option<Vec<Object>>) -> Dictionary {
        let mut dict = Dictionary::new();
        dict.set("BitsPerComponent", Object::Integer(1));
        if let Some(d) = decode {
            dict.set("Decode", Object::Array(d));
        }
        dict
    }

    fn decode_stencil(dict: &Dictionary, data: &[u8], w: u32, h: u32) -> Vec<u8> {
        let doc = empty_doc();
        let colour = ImageColour::resolve(&doc, dict, None);
        decode_raw(data, w, h, true, dict, &colour)
            .expect("stencil decode should succeed")
            .data
            .as_cpu()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn stencil_1bpc_default_decode_sample0_is_paint() {
        // No /Decode: sample 0 paints (0x00), sample 1 is transparent (0xFF)
        // (§8.9.6.2).  Byte 0b1010_0000 over 4 px → trans, paint, trans, paint.
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0xFF, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn stencil_1bpc_explicit_decode_0_1_matches_default() {
        let dict = stencil_dict(Some(vec![Object::Integer(0), Object::Integer(1)]));
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0xFF, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn stencil_1bpc_decode_1_0_inverts_polarity() {
        // /Decode [1 0]: sample 1 paints, sample 0 is transparent.
        let dict = stencil_dict(Some(vec![Object::Integer(1), Object::Integer(0)]));
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn stencil_1bpc_decode_real_1_0_also_inverts() {
        let dict = stencil_dict(Some(vec![Object::Real(1.0), Object::Real(0.0)]));
        let px = decode_stencil(&dict, &[0b1000_0000u8], 2, 1);
        assert_eq!(px, vec![0x00, 0xFF]);
    }

    #[test]
    fn stencil_1bpc_all_zero_sample_paints() {
        let dict = stencil_dict(None);
        assert_eq!(decode_stencil(&dict, &[0x00u8], 1, 1), vec![0x00]);
        assert_eq!(decode_stencil(&dict, &[0x80u8], 1, 1), vec![0xFF]);
    }

    #[test]
    fn stencil_1bpc_row_padding_no_bleed_across_rows() {
        // 3-px-wide, 2-row stencil at 1 bpc.  Each row pads to a whole byte
        // (PDF §7.4.1), so the 4th–8th bits of byte 0 must NOT bleed into row 1.
        // Row 0 byte = 0b101_00000 (px: trans, paint, trans).
        // Row 1 byte = 0b010_00000 (px: paint, trans, paint).
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0b1010_0000u8, 0b0100_0000u8], 3, 2);
        assert_eq!(px, vec![0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn stencil_1bpc_short_buffer_pads_transparent() {
        // Samples past the end of a truncated stream are transparent under
        // either polarity.
        for (decode, first) in [
            (None, 0xFF),
            (Some(vec![Object::Integer(1), Object::Integer(0)]), 0x00),
        ] {
            let dict = stencil_dict(decode);
            let px = decode_stencil(&dict, &[0b1100_0000u8], 16, 2);
            assert_eq!(px.len(), 32);
            assert_eq!(&px[0..2], &[first, first]);
            assert!(px[8..].iter().all(|&b| b == 0xFF), "{px:?}");
        }
    }

    #[test]
    fn stencil_without_bits_per_component_is_one_bit() {
        let px = decode_stencil(&Dictionary::new(), &[0b0110_0000u8], 4, 1);
        assert_eq!(px, vec![0x00, 0xFF, 0xFF, 0x00]);
        let mut dict = Dictionary::new();
        dict.set(
            "Decode",
            Object::Array(vec![Object::Integer(1), Object::Integer(0)]),
        );
        let px = decode_stencil(&dict, &[0b0110_0000u8], 4, 1);
        assert_eq!(px, vec![0xFF, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn stencil_malformed_decode_falls_back_to_default_polarity() {
        // A single-element /Decode on a stencil is malformed; must fall back to
        // the non-inverted default rather than guessing an inverted polarity.
        let dict = stencil_dict(Some(vec![Object::Integer(1)]));
        let px = decode_stencil(&dict, &[0b1000_0000u8], 2, 1);
        assert_eq!(
            px,
            vec![0xFF, 0x00],
            "malformed stencil /Decode must use default (non-inverted) polarity"
        );
    }
}
