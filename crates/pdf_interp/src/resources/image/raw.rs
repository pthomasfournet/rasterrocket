//! Raw / `FlateDecode` image decoding.
//!
//! Entry point is [`decode_raw`], which dispatches on `BitsPerComponent` and
//! colour space.  Helper functions handle the common sub-cases: 8-bpp colour,
//! indexed palette expansion, and CMYK→RGB conversion.

use std::collections::HashMap;
use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use super::bitpack::{downsample_16bpp, expand_nbpp, expand_nbpp_indexed, unpack_packed_bits};
use super::colorspace::{ResolvedCs, indexed_palette, resolve_cs};
use super::{ImageColorSpace, ImageData, ImageDescriptor, ImageFilter};
use crate::resources::colorspace::{ColorSpace, needs_sample_conversion, resolve, samples_to_rgb};
use crate::resources::dict_ext::DictExt as _;

#[cfg(feature = "gpu-icc")]
use super::colorspace::extract_icc_bytes;
#[cfg(feature = "gpu-icc")]
use super::icc::IccClutCache;
#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

// ── Public entry point ────────────────────────────────────────────────────────

/// Expand raw (already-decompressed) pixel bytes into a normalised 8-bpp form.
///
/// Supported `BitsPerComponent` values:
/// - 1  — packed MSB-first; 1 sample per component per pixel (PDF §8.9.3)
/// - 2  — 4 levels, scaled to 0x00/0x55/0xAA/0xFF
/// - 4  — 16 levels, scaled to 0x00…0xFF (value × 17)
/// - 8  — direct (common case)
/// - 16 — big-endian, high byte taken (linear 0–65535 → 0–255)
///
/// `Indexed` images with bpc 1/2/4 unpack the sub-byte palette indices first.
/// CMYK images (bpc 8/16) are converted to RGB inline.
pub(super) fn decode_raw(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    dict: &Dictionary,
    #[cfg(feature = "gpu-icc")] icc: &mut super::IccHandles<'_>,
) -> Option<ImageDescriptor> {
    #[cfg(feature = "gpu-icc")]
    let (gpu_ctx, clut_cache) = (icc.gpu_ctx, icc.clut_cache.as_deref_mut());
    let bpc = dict.get_i64(b"BitsPerComponent").unwrap_or(8);

    if is_mask {
        // Stencil mask — always 1 byte per pixel, no colour space conversion.
        //
        // Polarity rationale (load-bearing — a v1 flood-fill regression came
        // from getting this backwards): this codebase normalises *every*
        // stencil source (JBIG2, CCITTFax, raw) to one internal
        // `ImageColorSpace::Mask` byte convention — 0x00 = paint, 0xFF =
        // transparent — where the *ink* bit (=1) is the paint bit.  The JBIG2
        // and CCITTFax decoders already emit that convention (bit 1 = black =
        // ink = paint).  Raw 1-bpc stencils are aligned to the same internal
        // convention so all three sources composite identically downstream and
        // match the reference renderers (mutool, pdftoppm) on real-world
        // producers, which overwhelmingly author raw stencils ink=1=paint.
        //
        // NB this is the *internal* convention, not a verbatim restatement of
        // PDF 32000-1 §8.9.6.2, whose literal default `Decode [0 1]` instead
        // says raw sample 0 = paint.  An explicit `/Decode [1 0]` on the
        // stencil flips polarity relative to this internal default; that is
        // what `is_mask_inverted` detects and `invert` carries through.
        let invert = is_mask_inverted(dict);
        return decode_mask_raw(data, width, height, bpc, invert);
    }

    // Resolve the ColorSpace entry, following indirect references.  We need the
    // raw Object to detect Indexed (which uses a different decode path) as well
    // as for subsequent color-space resolution.
    let cs_obj_raw = dict.get(b"ColorSpace");
    // Follow a top-level Reference so Indexed detection below works correctly.
    // `Document::get_object` returns `Arc<Object>`, so we keep the Arc alive in
    // a local binding and borrow through it for the match below.
    let cs_obj_arc: Option<Arc<Object>> = if let Some(Object::Reference(id)) = cs_obj_raw {
        doc.get_object(*id).ok()
    } else {
        None
    };
    let cs_obj: Option<&Object> = if matches!(cs_obj_raw, Some(Object::Reference(_))) {
        cs_obj_arc.as_deref()
    } else {
        cs_obj_raw
    };

    // Check for Indexed colour space: [/Indexed base hival lookup].
    // These images carry palette indices rather than component samples, so they
    // need special handling before the general path.
    if let Some(Object::Array(arr)) = cs_obj
        && arr
            .first()
            .and_then(|o| o.as_name())
            .is_some_and(|n| n == b"Indexed")
    {
        return decode_raw_indexed(doc, data, width, height, bpc, arr);
    }

    if let Some(o) = cs_obj {
        let space = resolve(doc, o);
        if needs_sample_conversion(&space) {
            return decode_raw_converted(doc, data, width, height, bpc, dict, &space);
        }
    }

    // General path: resolve the colour space to Gray / Rgb / Cmyk.
    let resolved = cs_obj.map_or(ResolvedCs::Gray, |o| resolve_cs(doc, o));

    // For ICCBased CMYK, extract the raw ICC profile bytes for the GPU CLUT path.
    // Only done when the resolved space is actually CMYK (avoids unnecessary work
    // for Gray/RGB ICCBased images).
    #[cfg(feature = "gpu-icc")]
    let icc_bytes: Option<Vec<u8>> = if resolved == ResolvedCs::Cmyk {
        cs_obj.and_then(|o| extract_icc_bytes(doc, o))
    } else {
        None
    };

    // Parse the Decode array once; used by all bpc arms after expansion to 8-bpc.
    // PDF §8.9.5.2: Decode has 2 values per component: [Dmin Dmax ...].
    // Identity is [0 1] per component — no-op so we skip the allocation.
    let decode_map = parse_decode(dict, resolved.components());

    // Macro to apply the Decode remap (if any) to an already-8bpc buffer
    // and then call decode_raw_8bpp.  The `pixels_expr` is evaluated once.
    macro_rules! decode_and_bpp {
        ($pixels_expr:expr) => {{
            let pixels: std::borrow::Cow<[u8]> = $pixels_expr;
            let remapped = apply_decode(pixels, &decode_map);
            decode_raw_8bpp(
                &remapped,
                width,
                height,
                resolved,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                icc_bytes.as_deref(),
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            )
        }};
    }

    decode_and_bpp!(expand_to_8bpc(
        data,
        width,
        height,
        bpc,
        resolved.components()
    )?)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Expand `bpc`-bit samples of `components`-component pixels to one byte
/// per sample (PDF §8.9.3).  8-bit data is borrowed unchanged.  `None` for
/// an unsupported depth or data shorter than the image.
fn expand_to_8bpc(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    components: usize,
) -> Option<std::borrow::Cow<'_, [u8]>> {
    use std::borrow::Cow;
    Some(match bpc {
        // bpc=1 packs one bit per sample, one sample per component.
        // Multi-component spaces (RGB: 3 bits/px, CMYK: 4 bits/px) must use
        // unpack_packed_bits with `components` samples per row — not expand_1bpp
        // which treats one bit as one pixel.
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            let samples_per_row = width_usize.checked_mul(components)?;
            Cow::Owned(unpack_packed_bits(data, 1, samples_per_row, height, |v| {
                if v == 0 { 0x00 } else { 0xFF }
            })?)
        }
        2 => Cow::Owned(expand_nbpp::<2>(data, width, height, components)?),
        4 => Cow::Owned(expand_nbpp::<4>(data, width, height, components)?),
        8 => Cow::Borrowed(data),
        16 => Cow::Owned(downsample_16bpp(data, width, height, components)?),
        other => {
            log::debug!("image: {other} bits-per-component not yet implemented");
            return None;
        }
    })
}

/// Decode a raw image whose colour space needs per-sample conversion
/// (see [`needs_sample_conversion`]): expand to one byte per component,
/// apply `Decode` for a `DeviceN` space, and convert each distinct sample
/// tuple once through the colour space's tint transform or Lab formulae.
fn decode_raw_converted(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    dict: &Dictionary,
    space: &ColorSpace,
) -> Option<ImageDescriptor> {
    /// Distinct sample tuples memoised before the table is reset; bounds
    /// memory on continuous-tone images.
    const MEMO_LIMIT: usize = 1 << 16;

    let n = space.ncomponents();
    let samples = expand_to_8bpc(data, width, height, bpc, n)?;
    let samples = if matches!(space, ColorSpace::DeviceN { .. }) {
        apply_decode(samples, &parse_decode(dict, n))
    } else {
        samples.into_owned()
    };
    let npixels = (width as usize).checked_mul(height as usize)?;
    let needed = npixels.checked_mul(n)?;
    let Some(samples) = samples.get(..needed) else {
        log::warn!(
            "image: raw data too short ({} bytes, need {needed})",
            samples.len()
        );
        return None;
    };

    let mut memo: HashMap<Box<[u8]>, [u8; 3]> = HashMap::new();
    let mut out = Vec::with_capacity(npixels.checked_mul(3)?);
    for px in samples.chunks_exact(n) {
        let rgb = if let Some(&rgb) = memo.get(px) {
            rgb
        } else {
            if memo.len() >= MEMO_LIMIT {
                memo.clear();
            }
            let rgb = samples_to_rgb(doc, space, px);
            let _ = memo.insert(px.into(), rgb);
            rgb
        };
        out.extend_from_slice(&rgb);
    }
    Some(ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Rgb,
        data: ImageData::Cpu(out),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Parse the `Decode` array into per-component `(dmin, dmax)` pairs.
///
/// Returns an empty `Vec` when the array is absent, has fewer than
/// `2 * components` entries (identity assumed by callers), or contains any
/// non-finite value (NaN/inf from a malformed PDF would silently blacken every
/// pixel via `NaN as u8 == 0`; warn and treat as identity instead).  Entries
/// that are already identity `(0.0, 1.0)` are preserved so that `apply_decode`
/// can detect and skip the no-op path without re-reading the dict.
pub(super) fn parse_decode(dict: &Dictionary, components: usize) -> Vec<(f64, f64)> {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return Vec::new();
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
        return Vec::new();
    }
    // Reject non-finite values produced by a malformed PDF.  A NaN or infinite
    // dmin/dmax would make `is_identity` return false (NaN comparisons), then
    // `mul_add` would produce NaN, and the final `NaN as u8` cast would silently
    // produce 0 (black) for every pixel.  Warn and fall back to identity instead.
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
        return Vec::new();
    }
    pairs
}

/// Apply the per-component Decode remap to an already-8bpc pixel buffer.
///
/// Each sample `s ∈ [0, 255]` is mapped to
/// `round(dmin * 255 + s * (dmax - dmin))`, clamped to `[0, 255]`.
/// Identity pairs `(0.0, 1.0)` are short-circuited per component.
///
/// Takes a `Cow<[u8]>` so borrowed slices (e.g. the 8-bpc path when the Decode
/// array is identity) avoid a heap copy: the identity fast-path calls
/// `into_owned()` which only allocates when a borrow was passed; non-identity
/// paths call `into_owned()` first and mutate in place.
pub(super) fn apply_decode(data: std::borrow::Cow<[u8]>, decode: &[(f64, f64)]) -> Vec<u8> {
    if is_identity_decode(decode) {
        // Identity — return the buffer as-is.  If the caller passed a borrowed
        // slice (e.g. the 8-bpc path) this avoids a copy entirely.
        return data.into_owned();
    }
    let mut pixels = data.into_owned();
    let components = decode.len();
    for chunk in pixels.chunks_mut(components) {
        for (s, &(dmin, dmax)) in chunk.iter_mut().zip(decode.iter()) {
            if is_identity_pair(dmin, dmax) {
                continue; // identity — leave sample unchanged
            }
            // `s` is in [0, 255]; output = dmin*255 + s*(dmax-dmin), giving
            // values in [dmin*255, dmax*255] (or swapped when dmax < dmin).
            // Clamp to [0.0, 255.0] before rounding and casting to u8.
            let remapped = dmin.mul_add(255.0, f64::from(*s) * (dmax - dmin));
            // clamp(0..=255) before cast: `as u8` after clamp is always in [0, 255].
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "remapped is clamped to [0.0, 255.0] before cast; round() output is non-negative"
            )]
            {
                *s = remapped.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    pixels
}

/// `true` when `(dmin, dmax)` is `(0, 1)` within 1e-9.  Decode values are
/// parsed PDF reals, so exact 0.0 and 1.0 are typical.
fn is_identity_pair(dmin: f64, dmax: f64) -> bool {
    const EPS: f64 = 1e-9;
    dmin.abs() < EPS && (dmax - 1.0).abs() < EPS
}

/// `true` when a parsed `Decode` map leaves every sample unchanged: empty
/// (absent or ignored array) or all pairs `(0, 1)`.
pub(super) fn is_identity_decode(decode: &[(f64, f64)]) -> bool {
    decode
        .iter()
        .all(|&(dmin, dmax)| is_identity_pair(dmin, dmax))
}

/// Apply an image dictionary's `/Decode` to the output of a bilevel codec
/// (`JBIG2Decode`, `CCITTFaxDecode`), which leaves its samples unmapped.
///
/// A stencil ([`ImageColorSpace::Mask`]) flips polarity under `[1 0]`, as a
/// raw stencil does (see [`is_mask_inverted`]); a grayscale image is remapped
/// through [`apply_decode`].  Other descriptors are left unchanged.
pub(super) fn apply_bilevel_decode(img: &mut ImageDescriptor, dict: &Dictionary) {
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            clippy::infallible_destructuring_match,
            reason = "the cache-feature build adds a device-resident arm; keep one shape"
        )
    )]
    let bytes = match &mut img.data {
        ImageData::Cpu(bytes) => bytes,
        #[cfg(feature = "cache")]
        ImageData::Gpu(_) => return,
    };
    match img.color_space {
        ImageColorSpace::Mask => {
            if is_mask_inverted(dict) {
                for b in bytes.iter_mut() {
                    *b = !*b;
                }
            }
        }
        ImageColorSpace::Gray => {
            let decode = parse_decode(dict, 1);
            if !is_identity_decode(&decode) {
                let samples = std::mem::take(bytes);
                *bytes = apply_decode(std::borrow::Cow::Owned(samples), &decode);
            }
        }
        ImageColorSpace::Rgb => {}
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
    // Reuse the same numeric extraction `parse_decode` uses so the two
    // `/Decode` readers cannot drift apart in how they coerce Integer/Real.
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
/// Internal stencil convention (NOT a verbatim PDF §8.9.6.2 restatement — see
/// the rationale block at the `decode_mask_raw` call site in [`decode_raw`]):
/// with the non-inverted default, raw bit 1 (= ink) maps to 0x00 (paint) and
/// raw bit 0 maps to 0xFF (transparent).  This deliberately matches the JBIG2
/// and `CCITTFax` decoder output (bit 1 = black = ink = paint) so every stencil
/// source composites identically, and matches mutool/pdftoppm on real-world
/// producers.
///
/// When `invert` is `true` (explicit `/Decode [1 0]`), polarity is flipped:
/// raw bit 0 maps to 0x00 (paint) and raw bit 1 maps to 0xFF (transparent).
///
/// Only `bpc == 1` is conformant for an `ImageMask` per §8.9.6.2; the `bpc == 8`
/// arm is a defensive fallback for non-conformant producers and is logged.
fn decode_mask_raw(
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    invert: bool,
) -> Option<ImageDescriptor> {
    match bpc {
        1 => {
            let width_usize = usize::try_from(width).ok()?;
            // Non-inverted default: ink bit (1) = paint (0x00), bit 0 =
            // transparent (0xFF) — aligned to the JBIG2/CCITT decoders, see the
            // call-site rationale.  `invert` (explicit /Decode [1 0]) swaps it.
            let pixels = unpack_packed_bits(data, 1, width_usize, height, |v| {
                let paint = if invert { v == 0 } else { v != 0 };
                if paint { 0x00 } else { 0xFF }
            })?;
            Some(ImageDescriptor {
                width,
                height,
                color_space: ImageColorSpace::Mask,
                data: ImageData::Cpu(pixels),
                smask: None,
                filter: ImageFilter::Raw,
            })
        }
        8 => {
            // §8.9.6.2 mandates bpc=1 for an ImageMask; an 8-bpc stencil is a
            // producer bug.  Decode it defensively (mutool/pdftoppm also do)
            // but say so rather than mis-decoding silently.
            log::debug!(
                "image: ImageMask with non-conformant BitsPerComponent=8 \
                 (§8.9.6.2 requires 1) — decoding as 8-bpc stencil"
            );
            let expected = (width as usize).checked_mul(height as usize)?;
            if data.len() < expected {
                log::warn!(
                    "image: mask raw data too short ({} bytes, need {expected})",
                    data.len()
                );
                return None;
            }
            let pixels: Vec<u8> = if invert {
                data[..expected].iter().map(|&b| 255 - b).collect()
            } else {
                data[..expected].to_vec()
            };
            Some(ImageDescriptor {
                width,
                height,
                color_space: ImageColorSpace::Mask,
                data: ImageData::Cpu(pixels),
                smask: None,
                filter: ImageFilter::Raw,
            })
        }
        other => {
            log::debug!("image: mask {other} bpc not yet implemented");
            None
        }
    }
}

/// Decode an 8-bpp non-indexed image for `resolved` colour space.
///
/// CMYK is converted to RGB.  Gray and RGB are returned as-is.
fn decode_raw_8bpp(
    data: &[u8],
    width: u32,
    height: u32,
    resolved: ResolvedCs,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<ImageDescriptor> {
    use super::codecs::cmyk_raw_to_rgb;

    let components = resolved.components();
    let npixels = (width as usize).checked_mul(height as usize)?;
    let expected = npixels.checked_mul(components)?;

    if data.len() < expected {
        log::warn!(
            "image: raw data too short ({} bytes, need {expected} for {width}×{height}×{components})",
            data.len()
        );
        return None;
    }

    let raw = &data[..expected];

    if resolved == ResolvedCs::Cmyk {
        // Raw PDF CMYK: 255 = full ink, 0 = no ink (not inverted).
        let rgb = cmyk_raw_to_rgb(
            raw,
            false,
            #[cfg(feature = "gpu-icc")]
            gpu_ctx,
            #[cfg(feature = "gpu-icc")]
            icc_bytes,
            #[cfg(feature = "gpu-icc")]
            clut_cache,
        )?;
        Some(ImageDescriptor {
            width,
            height,
            color_space: ImageColorSpace::Rgb,
            data: ImageData::Cpu(rgb),
            smask: None,
            filter: ImageFilter::Raw,
        })
    } else {
        Some(ImageDescriptor {
            width,
            height,
            color_space: resolved.to_image_cs(),
            data: ImageData::Cpu(raw.to_vec()),
            smask: None,
            filter: ImageFilter::Raw,
        })
    }
}

/// Decode a raw `Indexed` colour space image.
///
/// For bpc 1, 2, 4: index values are packed MSB-first, N bits per index.
/// For bpc 8: one byte per index (common case).
/// PDF spec §8.6.6.3 allows bpc ∈ {1, 2, 4, 8} for Indexed images.
fn decode_raw_indexed(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    bpc: i64,
    cs_arr: &[Object],
) -> Option<ImageDescriptor> {
    let indices_8bit: Vec<u8>;
    let indices: &[u8] = match bpc {
        1 | 2 | 4 => {
            // bpc ∈ {1, 2, 4} — positive and fits u32.
            #[expect(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "bpc ∈ {1, 2, 4} — positive and fits u32"
            )]
            let bits = bpc as u32;
            indices_8bit = expand_nbpp_indexed(data, width, height, bits)?;
            &indices_8bit
        }
        8 => data,
        other => {
            log::debug!("image: Indexed with {other} bpc not supported");
            return None;
        }
    };

    let (palette, out_cs) = indexed_palette(doc, cs_arr)?;
    let stride = out_cs.components();

    // `indexed_palette` guarantees stride ∈ {1, 3} and palette is a multiple of
    // stride with at least one entry, but we guard defensively.
    if stride == 0 || palette.is_empty() || palette.len() % stride != 0 {
        log::debug!(
            "image: Indexed palette invariant violated (len={}, stride={stride})",
            palette.len()
        );
        return None;
    }

    let npixels = (width as usize).checked_mul(height as usize)?;
    if indices.len() < npixels {
        log::warn!(
            "image: Indexed raw data too short ({} bytes, need {npixels})",
            indices.len()
        );
        return None;
    }

    let n_entries = palette.len() / stride; // ≥ 1 guaranteed by guards above
    let mut out = Vec::with_capacity(npixels.checked_mul(stride)?);
    for &idx in &indices[..npixels] {
        let i = usize::from(idx).min(n_entries - 1);
        out.extend_from_slice(&palette[i * stride..(i + 1) * stride]);
    }

    Some(ImageDescriptor {
        width,
        height,
        color_space: out_cs.to_image_cs(),
        data: ImageData::Cpu(out),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_helpers::empty_doc as make_doc;

    fn bilevel(color_space: ImageColorSpace, bytes: &[u8]) -> ImageDescriptor {
        ImageDescriptor {
            width: u32::try_from(bytes.len()).unwrap(),
            height: 1,
            color_space,
            data: ImageData::Cpu(bytes.to_vec()),
            smask: None,
            filter: ImageFilter::Raw,
        }
    }

    fn decode_dict(values: &[i64]) -> Dictionary {
        let mut dict = Dictionary::new();
        dict.set(
            "Decode",
            Object::Array(values.iter().map(|&v| Object::Integer(v)).collect()),
        );
        dict
    }

    #[test]
    fn bilevel_decode_inverts_stencil_and_gray_under_1_0() {
        for cs in [ImageColorSpace::Mask, ImageColorSpace::Gray] {
            let mut img = bilevel(cs, &[0x00, 0xFF, 0x00]);
            apply_bilevel_decode(&mut img, &decode_dict(&[1, 0]));
            assert_eq!(img.data.as_cpu(), Some(&[0xFF, 0x00, 0xFF][..]), "{cs:?}");
        }
    }

    #[test]
    fn bilevel_decode_default_or_absent_is_identity() {
        for dict in [Dictionary::new(), decode_dict(&[0, 1])] {
            for cs in [ImageColorSpace::Mask, ImageColorSpace::Gray] {
                let mut img = bilevel(cs, &[0x00, 0xFF]);
                apply_bilevel_decode(&mut img, &dict);
                assert_eq!(img.data.as_cpu(), Some(&[0x00, 0xFF][..]), "{cs:?}");
            }
        }
    }

    #[test]
    fn decode_raw_gray_8bpp() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![0u8, 128, 255];
        let desc = decode_raw(
            &doc,
            &data,
            3,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Gray);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0, 128, 255]);
    }

    #[test]
    fn decode_raw_rgb_8bpp() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let desc = decode_raw(
            &doc,
            &data,
            3,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.width, 3);
        assert_eq!(desc.data.len(), 9);
    }

    #[test]
    fn decode_raw_mask_ignores_cs() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("BitsPerComponent", Object::Integer(8));
        let data = vec![0u8, 255];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            true,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Mask);
    }

    #[test]
    fn decode_raw_mask_decode_inverted() {
        // bpc=8: Decode=[1,0] flips stencil polarity for 8-bpc masks.
        // With Decode=[1,0]: sample 0 → 0xFF (transparent), sample 255 → 0x00 (paint).
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(1.0), Object::Real(0.0)]),
        );
        // With Decode=[1,0]: sample 0 → 0xFF (transparent), sample 255 → 0x00 (paint).
        let data = vec![0u8, 255u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            true,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(
            pixels[0], 0xFF,
            "sample 0 with Decode=[1,0] must be transparent"
        );
        assert_eq!(
            pixels[1], 0x00,
            "sample 255 with Decode=[1,0] must be paint"
        );
    }

    #[test]
    fn decode_raw_cmyk_converts_to_rgb() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceCMYK".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // Single pixel: C=0, M=0, Y=0, K=0 → white (255, 255, 255).
        let data = vec![0u8, 0, 0, 0];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.data.as_cpu().unwrap(), &[255, 255, 255]);
    }

    #[test]
    fn decode_raw_too_short_returns_none() {
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // Claim 2×2 RGB but supply only 3 bytes (need 12).
        let data = vec![0u8, 0, 0];
        assert!(
            decode_raw(
                &doc,
                &data,
                2,
                2,
                false,
                &dict,
                #[cfg(feature = "gpu-icc")]
                &mut crate::resources::image::IccHandles::none(),
            )
            .is_none()
        );
    }

    #[test]
    fn decode_raw_gray_1bpp() {
        // 2-pixel Gray image at bpc=1: byte 0b10_000000 → pixels [0xFF, 0x00].
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(1));
        let data = vec![0b1000_0000u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Gray);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0xFF, 0x00]);
    }

    #[test]
    fn decode_raw_rgb_1bpp() {
        // 1-pixel RGB image at bpc=1: 3 bits packed MSB-first in one byte.
        // Byte 0b110_00000: R=1→0xFF, G=1→0xFF, B=0→0x00.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceRGB".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(1));
        let data = vec![0b1100_0000u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        assert_eq!(desc.data.as_cpu().unwrap(), &[0xFF, 0xFF, 0x00]);
    }

    #[test]
    fn decode_raw_gray_decode_inverted() {
        // A 2-pixel DeviceGray image with Decode=[1,0].
        // PDF §8.9.5.2: output = dmin*255 + s*(dmax-dmin) = 1*255 + s*(0-1) = 255 - s.
        // Sample 0 → output 255 (white); sample 255 → output 0 (black).
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(1.0), Object::Real(0.0)]),
        );
        let data = vec![0u8, 255u8]; // sample 0 → output 255, sample 255 → output 0
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(pixels[0], 255, "sample 0 with Decode=[1,0] must map to 255");
        assert_eq!(pixels[1], 0, "sample 255 with Decode=[1,0] must map to 0");
    }

    #[test]
    fn decode_raw_gray_decode_identity_unchanged() {
        // Explicit Decode=[0,1] is identity — output must equal input exactly.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(0.0), Object::Real(1.0)]),
        );
        let data = vec![128u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.data.as_cpu().unwrap()[0], 128);
    }

    #[test]
    fn decode_raw_gray_decode_integer_entries_accepted() {
        // Decode=[1, 0] written as Integer objects (not Real) must also invert.
        // Verifies that `parse_decode` / `Object::as_f64` handles Integer variants.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        dict.set(
            "Decode",
            Object::Array(vec![Object::Integer(1), Object::Integer(0)]),
        );
        let data = vec![0u8, 255u8];
        let desc = decode_raw(
            &doc,
            &data,
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        let pixels = desc.data.as_cpu().unwrap();
        assert_eq!(
            pixels[0], 255,
            "integer Decode=[1,0]: sample 0 must map to 255"
        );
        assert_eq!(
            pixels[1], 0,
            "integer Decode=[1,0]: sample 255 must map to 0"
        );
    }

    #[test]
    fn decode_raw_gray_decode_nonfinite_falls_back_to_identity() {
        // A malformed Decode array with a non-finite value must fall back to
        // identity rather than silently blackening every pixel.
        // The test uses NaN; the code path is the same for +/-inf.
        let doc = make_doc();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", Object::Name(b"DeviceGray".to_vec()));
        dict.set("BitsPerComponent", Object::Integer(8));
        // f32::NAN stored in Object::Real; as_f64 converts through f32 so it stays NaN.
        dict.set(
            "Decode",
            Object::Array(vec![Object::Real(f32::NAN), Object::Real(1.0)]),
        );
        let data = vec![128u8];
        let desc = decode_raw(
            &doc,
            &data,
            1,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        // Identity fallback: output must equal input.
        assert_eq!(
            desc.data.as_cpu().unwrap()[0],
            128,
            "non-finite Decode must fall back to identity"
        );
    }

    // ── 1-bpc raw stencil-mask polarity ───────────────────────────────────────

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
        let doc = make_doc();
        decode_raw(
            &doc,
            data,
            w,
            h,
            true,
            dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .expect("stencil decode should succeed")
        .data
        .as_cpu()
        .unwrap()
        .to_vec()
    }

    #[test]
    fn stencil_1bpc_default_decode_bit1_is_paint() {
        // No /Decode → internal default: ink bit (1) = paint (0x00), bit 0 =
        // transparent (0xFF).  Byte 0b1010_0000 over 4 px → paint, trans, paint, trans.
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn stencil_1bpc_explicit_decode_0_1_matches_default() {
        // Explicit /Decode [0 1] is the default — must equal the no-Decode case.
        let dict = stencil_dict(Some(vec![Object::Integer(0), Object::Integer(1)]));
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn stencil_1bpc_decode_1_0_inverts_polarity() {
        // /Decode [1 0] flips polarity: bit 0 = paint (0x00), bit 1 = transparent.
        let dict = stencil_dict(Some(vec![Object::Integer(1), Object::Integer(0)]));
        let px = decode_stencil(&dict, &[0b1010_0000u8], 4, 1);
        assert_eq!(px, vec![0xFF, 0x00, 0xFF, 0x00]);
    }

    #[test]
    fn stencil_1bpc_decode_real_1_0_also_inverts() {
        // /Decode [1.0 0.0] written as Real objects must also invert.
        let dict = stencil_dict(Some(vec![Object::Real(1.0), Object::Real(0.0)]));
        let px = decode_stencil(&dict, &[0b1000_0000u8], 2, 1);
        assert_eq!(px, vec![0xFF, 0x00]);
    }

    #[test]
    fn stencil_1bpc_all_zero_data_is_transparent_not_flood() {
        // The exact docmac repro shape: a 1×1 all-zero-bit stencil with the
        // default Decode must be transparent (0xFF), NOT paint — painting it
        // is the v1 full-page black-flood regression.
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0x00u8], 1, 1);
        assert_eq!(
            px,
            vec![0xFF],
            "all-zero default stencil must be transparent"
        );
    }

    #[test]
    fn stencil_1bpc_row_padding_no_bleed_across_rows() {
        // 3-px-wide, 2-row stencil at 1 bpc.  Each row pads to a whole byte
        // (PDF §7.4.1), so the 4th–8th bits of byte 0 must NOT bleed into row 1.
        // Row 0 byte = 0b101_00000 (px: paint, trans, paint).
        // Row 1 byte = 0b010_00000 (px: trans, paint, trans).
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0b1010_0000u8, 0b0100_0000u8], 3, 2);
        assert_eq!(px, vec![0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF]);
    }

    #[test]
    fn stencil_1bpc_short_buffer_zero_pads_gracefully() {
        // Buffer shorter than width×height: missing bits read as 0 (transparent
        // in default polarity) — graceful, no panic, no flood.
        let dict = stencil_dict(None);
        let px = decode_stencil(&dict, &[0b1100_0000u8], 16, 1);
        assert_eq!(px.len(), 16);
        assert_eq!(&px[0..2], &[0x00, 0x00]); // bits 1,1 → paint
        assert!(px[2..].iter().all(|&b| b == 0xFF)); // zero-padded → transparent
    }

    #[test]
    fn stencil_malformed_decode_falls_back_to_default_polarity() {
        // A single-element /Decode on a stencil is malformed; must fall back to
        // the non-inverted default rather than guessing an inverted polarity.
        let dict = stencil_dict(Some(vec![Object::Integer(1)]));
        let px = decode_stencil(&dict, &[0b1000_0000u8], 2, 1);
        assert_eq!(
            px,
            vec![0x00, 0xFF],
            "malformed stencil /Decode must use default (non-inverted) polarity"
        );
    }

    /// Two-colorant `[/Black /Spot]` `DeviceN` space whose Type 4 tint
    /// transform adds Black to K and maps the spot to the teal CMYK
    /// `(1, 0.08, 0.26, 0.38)` per unit tint.
    fn devicen_teal() -> (Document, Object) {
        const PROGRAM: &str = "{ exch 1 index 0.08 mul 2 index 0.26 mul 3 2 roll \
                               3 index 0.38 mul add dup 1 gt { pop 1 } if }";
        let body = format!(
            "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
             stream\n{PROGRAM}\nendstream",
            PROGRAM.len()
        );
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, &body)]);
        let cs = Object::Array(vec![
            Object::Name(b"DeviceN".to_vec()),
            Object::Array(vec![
                Object::Name(b"Black".to_vec()),
                Object::Name(b"Spot".to_vec()),
            ]),
            Object::Name(b"DeviceCMYK".to_vec()),
            Object::Reference((4, 0)),
        ]);
        (doc, cs)
    }

    fn teal_rgb() -> [u8; 3] {
        color::convert::cmyk_to_rgb_bytes(1.0, 0.08, 0.26, 0.38)
    }

    #[test]
    fn decode_raw_multi_colorant_devicen_converts_each_pixel() {
        let (doc, cs) = devicen_teal();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", cs);
        dict.set("BitsPerComponent", Object::Integer(8));
        // Pixels (Black, Spot): spot only, no ink, black only.
        let data = [0u8, 255, 0, 0, 255, 0];
        let desc = decode_raw(
            &doc,
            &data,
            3,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        let px = desc.data.as_cpu().unwrap();
        assert_eq!(px[0..3], teal_rgb());
        assert_eq!(px[3..6], [255, 255, 255]);
        assert_eq!(px[6..9], [0, 0, 0]);
    }

    #[test]
    fn decode_raw_indexed_over_multi_colorant_devicen_expands_the_palette() {
        let (doc, devicen) = devicen_teal();
        let mut dict = Dictionary::new();
        dict.set(
            "ColorSpace",
            Object::Array(vec![
                Object::Name(b"Indexed".to_vec()),
                devicen,
                Object::Integer(1),
                // Entry 0: full spot; entry 1: no ink.
                Object::String(vec![0, 255, 0, 0], pdf::StringFormat::Hexadecimal),
            ]),
        );
        dict.set("BitsPerComponent", Object::Integer(8));
        let desc = decode_raw(
            &doc,
            &[0u8, 1],
            2,
            1,
            false,
            &dict,
            #[cfg(feature = "gpu-icc")]
            &mut crate::resources::image::IccHandles::none(),
        )
        .unwrap();
        assert_eq!(desc.color_space, ImageColorSpace::Rgb);
        let px = desc.data.as_cpu().unwrap();
        assert_eq!(px[0..3], teal_rgb());
        assert_eq!(px[3..6], [255, 255, 255]);
    }
}
