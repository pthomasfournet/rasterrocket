//! Inline image decoding (`BI … ID … EI`) and parameter parsing.
#![expect(
    clippy::redundant_pub_crate,
    reason = "parse_inline_params is pub(crate) for prescan.rs; \
              redundant here because inline is a private module, but explicit visibility aids readability"
)]

use pdf::{Dictionary, Object};

use super::dispatch::{ImageStream, decode_image_stream};
use super::{ImageDecodeCtx, ImageDescriptor, validated_dims};
use crate::resources::dict_ext::DictExt;

#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;

#[cfg(test)]
use super::ImageColorSpace;

/// Decode an inline image (`BI … ID … EI`) from raw parameter and data bytes.
///
/// `params` is the raw content between `BI` and `ID`; it contains key/value
/// pairs using either the full PDF names (`/Width`, `/ColorSpace`, …) or the
/// abbreviated forms defined in PDF §8.9.7 Table 89 (`/W`, `/CS`, …).
/// `data` is the raw content between `ID` and `EI` (compressed or raw pixels).
///
/// A `/CS` that names neither a device space nor an abbreviation is looked up
/// in `ctx.color_spaces`, as is the base of an inline `[/Indexed …]` array
/// given by name; a name that resolves nowhere fails the decode.
///
/// Inline images cannot carry an `SMask` stream (they have no object identity),
/// so the returned descriptor always has `smask: None`.
///
/// GPU decoders are passed in for `DCTDecode` and `JPXDecode` streams.  The same
/// area threshold as `resolve_image` applies — images below
/// `super::GPU_JPEG_THRESHOLD_PX` (only present with the `nvjpeg` or `vaapi`
/// feature) use the CPU path regardless.  Most inline images are small (thumbnails, icons) and will fall
/// through to the CPU path; the parameters are plumbed through so that the
/// dispatch logic in `decode_dct` / `decode_jpx` can make the threshold decision
/// rather than unconditionally bypassing GPU decode for all inline images.
///
/// Returns `None` if the parameter block is unparseable, dimensions are
/// degenerate, a colour-space name is unresolvable, or the filter is
/// unsupported.
#[must_use]
pub fn decode_inline_image<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    ctx: &mut ImageDecodeCtx<'_>,
    params: &[u8],
    data: &[u8],
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<&mut JpegGpuDecoder<B>>,
) -> Option<ImageDescriptor> {
    let doc = ctx.doc;
    let mut dict = parse_inline_params(params);
    if let Err(name) = resolve_named_color_space(&mut dict, ctx.color_spaces) {
        log::warn!(
            "inline image: /ColorSpace {:?} is neither a device space nor a ColorSpace resource",
            String::from_utf8_lossy(&name)
        );
        return None;
    }

    let w_raw = dict.get_i64(b"Width")?;
    let h_raw = dict.get_i64(b"Height")?;
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("inline image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return None;
    };

    let is_mask = dict.get_bool(b"ImageMask").unwrap_or(false);
    let stream = pdf::Stream::new(dict, data.to_vec());
    let src = ImageStream {
        stream: &stream,
        width: w,
        height: h,
        is_mask,
    };
    // Inline images have no object identity, so the device image cache
    // never holds them.
    let mut img = decode_image_stream(
        ctx,
        &src,
        None,
        #[cfg(feature = "cache")]
        None,
        #[cfg(feature = "gpu-jpeg-huffman")]
        jpeg_gpu,
    )
    .map_err(|e| log::warn!("inline image: {e}"))
    .ok()?;

    if !is_mask
        && let Some(rewrite) = stream
            .dict
            .get(b"ColorSpace")
            .and_then(|cs| super::colorspace::sample_rewrite(doc, cs))
    {
        super::colorspace::apply_sample_rewrite(&mut img, &rewrite);
    }
    Some(img)
}

/// Replace a `/ColorSpace` name that is not a device space with the entry it
/// names in `color_spaces`, and likewise the base of an inline
/// `[/Indexed base hival lookup]` array (PDF §8.9.7).  Returns the name that
/// resolves nowhere as the error.
fn resolve_named_color_space(
    dict: &mut Dictionary,
    color_spaces: Option<&Dictionary>,
) -> Result<(), Vec<u8>> {
    let lookup = |name: &[u8]| -> Result<Option<Object>, Vec<u8>> {
        if matches!(name, b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK") {
            return Ok(None);
        }
        color_spaces
            .and_then(|d| d.get(name))
            .cloned()
            .map(Some)
            .ok_or_else(|| name.to_vec())
    };
    let Some(cs) = dict.get(b"ColorSpace") else {
        return Ok(());
    };
    let resolved = match cs {
        Object::Name(name) => lookup(name)?,
        Object::Array(arr) if arr.first().and_then(Object::as_name) == Some(b"Indexed") => {
            match arr.get(1) {
                Some(Object::Name(base)) => lookup(base)?.map(|base| {
                    let mut arr = arr.clone();
                    arr[1] = base;
                    Object::Array(arr)
                }),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(obj) = resolved {
        dict.set("ColorSpace", obj);
    }
    Ok(())
}

/// Parse the raw inline-image parameter block into a `pdf::Dictionary`.
///
/// Keys and values are in PDF syntax but may use the abbreviated forms from
/// PDF §8.9.7 Table 89 (e.g., `/W` → `/Width`, `/Fl` → `/FlateDecode`).
/// Unknown keys are passed through unchanged; unrecognised values are passed
/// through unchanged — the caller's `DictExt` helpers return `None` for
/// malformed entries, so no validation is needed here.
pub(crate) fn parse_inline_params(params: &[u8]) -> Dictionary {
    // Tokenise the parameter block with the content-stream tokenizer and
    // collect alternating key/value pairs.
    use crate::content::tokenizer::{Token, Tokenizer};

    let mut dict = Dictionary::new();
    let tokens: Vec<Token<'_>> = Tokenizer::new(params).collect();
    let mut i = 0;

    while i + 1 < tokens.len() {
        let Token::Name(key_bytes) = &tokens[i] else {
            i += 1;
            continue;
        };
        // Expand abbreviated key name → canonical PDF key.
        let canonical_key: &[u8] = expand_inline_key(key_bytes);
        let canonical_key_vec = canonical_key.to_vec();

        let value_obj = inline_token_to_object(&tokens[i + 1]);
        // Expand abbreviated filter/cs names inside the value.
        let value_obj = expand_inline_value(value_obj);

        dict.set(canonical_key_vec, value_obj);
        i += 2;
    }

    dict
}

/// The length in bytes of an inline image's data as its parameters
/// `dict` (keys expanded) determine it: `/Length` when present (PDF 2.0),
/// otherwise, for unfiltered data, `⌈W·BPC·n/8⌉·H` (§8.9.7), where `n` is
/// 1 for an image mask and the component count of a device or `/Indexed`
/// colour space.  `None` when the parameters leave it open: filtered data
/// without `/Length`, a colour space named from the resources, or
/// missing, non-positive or overflowing values.
pub(crate) fn inline_data_len(dict: &Dictionary) -> Option<usize> {
    if let Some(len) = dict.get_i64(b"Length") {
        return usize::try_from(len).ok();
    }
    if dict
        .get(b"Filter")
        .is_some_and(|f| !matches!(f, Object::Array(a) if a.is_empty()))
    {
        return None;
    }
    let dim = |key: &[u8]| {
        dict.get_i64(key)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
    };
    let (w, h) = (dim(b"Width")?, dim(b"Height")?);
    let (bpc, n) = if dict.get_bool(b"ImageMask").unwrap_or(false) {
        (1, 1)
    } else {
        let n = match dict.get(b"ColorSpace")? {
            Object::Name(name) => match name.as_slice() {
                b"DeviceGray" => 1,
                b"DeviceRGB" => 3,
                b"DeviceCMYK" => 4,
                _ => return None,
            },
            Object::Array(arr) if arr.first().and_then(Object::as_name) == Some(b"Indexed") => 1,
            _ => return None,
        };
        (dim(b"BitsPerComponent")?, n)
    };
    w.checked_mul(bpc)?
        .checked_mul(n)?
        .div_ceil(8)
        .checked_mul(h)
}

/// Map an abbreviated inline-image key to its canonical PDF name.
///
/// Abbreviations are defined in PDF §8.9.7 Table 89.  Unknown keys are
/// returned unchanged (as a byte slice borrow).
pub(super) const fn expand_inline_key(key: &[u8]) -> &[u8] {
    match key {
        b"W" => b"Width",
        b"H" => b"Height",
        b"CS" => b"ColorSpace",
        b"BPC" => b"BitsPerComponent",
        b"F" => b"Filter",
        b"DP" => b"DecodeParms",
        b"IM" => b"ImageMask",
        b"D" => b"Decode",
        b"I" => b"Interpolate",
        b"Intent" => b"Intent",
        b"L" => b"Length",
        other => other,
    }
}

/// Convert a content-stream [`Token`] to a `pdf::Object`.
pub(super) fn inline_token_to_object(tok: &crate::content::tokenizer::Token<'_>) -> Object {
    use crate::content::tokenizer::Token;
    match tok {
        Token::Name(n) => Object::Name(n.to_vec()),
        Token::Number(f) => {
            // Store as Integer when the value is a whole number, Real otherwise.
            // Upper bound 1e15 safely fits in i64 without f64 precision issues.
            let rounded = f.round();
            if (f - rounded).abs() < f64::EPSILON && rounded.abs() < 1e15 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "checked: value is a whole number with abs < 1e15; fits in i64"
                )]
                Object::Integer(rounded as i64)
            } else {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "f64 → f32: PDF numeric params are small; precision loss is acceptable"
                )]
                Object::Real(*f as f32)
            }
        }
        Token::Bool(b) => Object::Boolean(*b),
        Token::String(s) => Object::String(s.clone(), pdf::StringFormat::Literal),
        Token::Array(items) => Object::Array(items.iter().map(inline_token_to_object).collect()),
        // Inline image `/DP` (DecodeParms) is a dictionary, e.g.
        // `<</K -1 /Columns 277>>` for a CCITT G4 image mask.
        Token::Dict(items) => {
            let mut dict = Dictionary::new();
            for [key, value] in items.as_chunks::<2>().0 {
                if let Token::Name(key) = key {
                    dict.set(*key, inline_token_to_object(value));
                }
            }
            Object::Dictionary(dict)
        }
        _ => Object::Null,
    }
}

/// Expand abbreviated filter or colour-space names inside an `Object`.
///
/// Affects `Name` objects and arrays of names (filter arrays).  The canonical
/// names are the full PDF spelling; abbreviations come from PDF §8.9.7 Table 89
/// (for filters) and the colour-space name list.
pub(super) fn expand_inline_value(obj: Object) -> Object {
    match obj {
        Object::Name(ref n) => {
            let expanded = expand_inline_name(n);
            if expanded == n.as_slice() {
                obj
            } else {
                Object::Name(expanded.to_vec())
            }
        }
        Object::Array(arr) => Object::Array(
            arr.into_iter()
                .map(|o| match o {
                    Object::Name(n) => Object::Name(expand_inline_name(&n).to_vec()),
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

/// Expand an abbreviated inline-image name (filter or colour-space).
pub(super) const fn expand_inline_name(name: &[u8]) -> &[u8] {
    match name {
        // Filter abbreviations (PDF §8.9.7 Table 89).
        b"AHx" => b"ASCIIHexDecode",
        b"A85" => b"ASCII85Decode",
        b"LZW" => b"LZWDecode",
        b"Fl" => b"FlateDecode",
        b"RL" => b"RunLengthDecode",
        b"CCF" => b"CCITTFaxDecode",
        b"DCT" => b"DCTDecode",
        // Colour-space abbreviations.
        b"G" => b"DeviceGray",
        b"RGB" => b"DeviceRGB",
        b"CMYK" => b"DeviceCMYK",
        b"I" => b"Indexed",
        other => other,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pdf::Document;

    use crate::test_helpers::empty_doc;

    /// A context with no `ColorSpace` resources and every GPU decoder off.
    fn cpu_ctx(doc: &Document) -> ImageDecodeCtx<'_> {
        ctx_with_color_spaces(doc, None)
    }

    fn ctx_with_color_spaces<'a>(
        doc: &'a Document,
        color_spaces: Option<&'a Dictionary>,
    ) -> ImageDecodeCtx<'a> {
        ImageDecodeCtx {
            color_spaces,
            ..ImageDecodeCtx::cpu(doc)
        }
    }

    fn decode_with(
        color_spaces: &Dictionary,
        params: &[u8],
        data: &[u8],
    ) -> Option<ImageDescriptor> {
        let doc = empty_doc();
        decode_inline_image(
            &mut ctx_with_color_spaces(&doc, Some(color_spaces)),
            params,
            data,
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        )
    }

    /// `/CS /CS0` naming a `DeviceRGB` resource decodes three bytes per pixel.
    #[test]
    fn inline_image_named_color_space_resolves_through_resources() {
        let mut cs = Dictionary::new();
        cs.set("CS0", Object::Name(b"DeviceRGB".to_vec()));
        let data = [0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF];
        let img = decode_with(&cs, b"/W 2 /H 1 /CS /CS0 /BPC 8", &data).expect("decodes");
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu().unwrap(), &data.to_vec());
    }

    /// The base of an inline `[/I base hival lookup]` array may be a resource
    /// name; the palette then indexes into that space.
    #[test]
    fn inline_indexed_base_name_resolves_through_resources() {
        let mut cs = Dictionary::new();
        cs.set("Base", Object::Name(b"DeviceRGB".to_vec()));
        let params = b"/W 2 /H 1 /CS [/I /Base 1 <ff000000ff00>] /BPC 8";
        let img = decode_with(&cs, params, &[0, 1]).expect("decodes");
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu().unwrap(), &vec![0xFF, 0, 0, 0, 0xFF, 0]);
    }

    fn decode_cpu(params: &[u8], data: &[u8]) -> Option<ImageDescriptor> {
        let doc = empty_doc();
        decode_inline_image(
            &mut cpu_ctx(&doc),
            params,
            data,
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        )
    }

    /// Transport filters, alone or chained, decode to raw samples.
    #[test]
    fn inline_image_transport_filters() {
        let want = vec![0x00u8, 0x80, 0xFF, 0x40];
        for (params, data) in [
            (
                b"/W 2 /H 2 /CS /G /BPC 8 /F /RL".as_slice(),
                b"\x03\x00\x80\xff\x40\x80".as_slice(),
            ),
            (b"/W 2 /H 2 /CS /G /BPC 8 /F /AHx", b"0080FF40>"),
            (b"/W 2 /H 2 /CS /G /BPC 8 /F [/AHx /RL]", b"0300 80FF 4080>"),
        ] {
            let img = decode_cpu(params, data).expect("decodes");
            assert_eq!(
                img.data.as_cpu().unwrap(),
                &want,
                "{}",
                String::from_utf8_lossy(params)
            );
        }
    }

    /// A `/CS` name found in neither the device spaces nor the resources
    /// fails the decode instead of falling back to `DeviceGray`.
    #[test]
    fn inline_image_unknown_color_space_name_fails() {
        let data = [0u8; 6];
        assert!(decode_with(&Dictionary::new(), b"/W 2 /H 1 /CS /Nope /BPC 8", &data).is_none());
    }

    // ── parse_inline_params / expand_inline_key ───────────────────────────────

    #[test]
    fn inline_key_expansion_canonical() {
        assert_eq!(expand_inline_key(b"W"), b"Width");
        assert_eq!(expand_inline_key(b"H"), b"Height");
        assert_eq!(expand_inline_key(b"CS"), b"ColorSpace");
        assert_eq!(expand_inline_key(b"BPC"), b"BitsPerComponent");
        assert_eq!(expand_inline_key(b"F"), b"Filter");
        assert_eq!(expand_inline_key(b"DP"), b"DecodeParms");
        assert_eq!(expand_inline_key(b"IM"), b"ImageMask");
    }

    #[test]
    fn inline_key_unknown_passthrough() {
        // Unrecognised keys pass through unchanged.
        assert_eq!(expand_inline_key(b"Width"), b"Width");
        assert_eq!(expand_inline_key(b"Blah"), b"Blah");
    }

    #[test]
    fn inline_name_filter_expansion() {
        assert_eq!(expand_inline_name(b"Fl"), b"FlateDecode");
        assert_eq!(expand_inline_name(b"DCT"), b"DCTDecode");
        assert_eq!(expand_inline_name(b"CCF"), b"CCITTFaxDecode");
        assert_eq!(expand_inline_name(b"RL"), b"RunLengthDecode");
        assert_eq!(expand_inline_name(b"AHx"), b"ASCIIHexDecode");
        assert_eq!(expand_inline_name(b"A85"), b"ASCII85Decode");
        assert_eq!(expand_inline_name(b"LZW"), b"LZWDecode");
    }

    #[test]
    fn inline_name_cs_expansion() {
        assert_eq!(expand_inline_name(b"G"), b"DeviceGray");
        assert_eq!(expand_inline_name(b"RGB"), b"DeviceRGB");
        assert_eq!(expand_inline_name(b"CMYK"), b"DeviceCMYK");
    }

    #[test]
    fn parse_inline_params_basic() {
        use crate::resources::dict_ext::DictExt;
        // /W 4 /H 2 /CS /G /BPC 8 (abbreviated keys + abbreviated CS name)
        let params = b"/W 4 /H 2 /CS /G /BPC 8";
        let dict = parse_inline_params(params);
        assert_eq!(dict.get_i64(b"Width"), Some(4));
        assert_eq!(dict.get_i64(b"Height"), Some(2));
        assert_eq!(dict.get_i64(b"BitsPerComponent"), Some(8));
        // CS /G should expand to /DeviceGray.
        assert_eq!(
            dict.get(b"ColorSpace").and_then(|o| o.as_name()),
            Some(b"DeviceGray".as_ref())
        );
    }

    #[test]
    fn parse_inline_params_full_names() {
        use crate::resources::dict_ext::DictExt;
        // Full (non-abbreviated) names should also parse correctly.
        let params = b"/Width 3 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8";
        let dict = parse_inline_params(params);
        assert_eq!(dict.get_i64(b"Width"), Some(3));
        assert_eq!(dict.get_i64(b"Height"), Some(1));
        assert_eq!(dict.get_i64(b"BitsPerComponent"), Some(8));
    }

    #[test]
    fn parse_inline_params_decodeparms_dict() {
        use crate::resources::dict_ext::DictExt;
        // A CCITT G4 ImageMask inline image: the /DP value is an inline
        // dictionary `<</K -1 /Columns 277>>`.  It must round-trip into a
        // real Object::Dictionary so the codec sees /K = -1 (Group 4) rather
        // than defaulting to 0 (Group 3 1D) and producing zero rows.
        let params = b"/IM true /W 277 /H 102 /BPC 1 /F /CCF /DP<</K -1 /Columns 277>>";
        let dict = parse_inline_params(params);
        let dp = dict
            .get(b"DecodeParms")
            .and_then(Object::as_dict)
            .expect("DecodeParms must parse into a dictionary");
        assert_eq!(dp.get_i64(b"K"), Some(-1));
        assert_eq!(dp.get_i64(b"Columns"), Some(277));
    }

    #[test]
    fn parse_inline_params_decodeparms_dict_blackis1() {
        use crate::resources::dict_ext::DictExt;
        // Boolean and predictor keys inside the inline DecodeParms dict must
        // also survive the round-trip (regression guard for the wider class
        // of dropped inline-DecodeParms keys).
        let params = b"/W 8 /H 8 /BPC 1 /F /CCF /DP<</K 0 /BlackIs1 true /EncodedByteAlign true>>";
        let dict = parse_inline_params(params);
        let dp = dict
            .get(b"DecodeParms")
            .and_then(Object::as_dict)
            .expect("DecodeParms must parse into a dictionary");
        assert_eq!(dp.get_i64(b"K"), Some(0));
        assert_eq!(dp.get_bool(b"BlackIs1"), Some(true));
        assert_eq!(dp.get_bool(b"EncodedByteAlign"), Some(true));
    }

    // ── decode_inline_image ────────────────────────────────────────────────────

    #[test]
    fn inline_image_raw_gray() {
        // 2×2 raw DeviceGray image, 8 bpc, no filter.
        let params = b"/W 2 /H 2 /CS /G /BPC 8";
        let data = [0x00u8, 0x80, 0xFF, 0x40];
        let doc = empty_doc();
        let img = decode_inline_image(
            &mut cpu_ctx(&doc),
            params,
            &data,
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        )
        .expect("decode should succeed");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &data.to_vec());
    }

    #[test]
    fn inline_image_mask() {
        // 2×1 stencil.  PDF §8.9.6.2, default Decode [0 1]: sample 0
        // paints (0x00), sample 1 is transparent (0xFF).
        let params = b"/W 2 /H 1 /IM true /BPC 1";
        // byte 0b01000000: sample 0 then sample 1.
        let data = [0b0100_0000u8];
        let doc = empty_doc();
        let img = decode_inline_image(
            &mut cpu_ctx(&doc),
            params,
            &data,
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        )
        .expect("decode should succeed");
        assert_eq!(img.color_space, ImageColorSpace::Mask);
        let bytes = img.data.as_cpu().unwrap();
        assert_eq!(bytes[0], 0x00); // first pixel: paint
        assert_eq!(bytes[1], 0xFF); // second pixel: transparent
    }

    /// `/IM true` with no `/BPC`: an image mask is one bit per sample.
    #[test]
    fn inline_image_mask_without_bpc_is_one_bit() {
        let doc = empty_doc();
        let img = decode_inline_image(
            &mut cpu_ctx(&doc),
            b"/W 10 /H 2 /IM true",
            &[0b0100_0000, 0b0111_1111, 0b1111_1111, 0b1100_0000],
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
        )
        .expect("decode should succeed");
        assert_eq!(img.color_space, ImageColorSpace::Mask);
        let bytes = img.data.as_cpu().unwrap();
        assert_eq!(bytes.len(), 20);
        assert_eq!(&bytes[..3], &[0x00, 0xFF, 0x00]);
        assert_eq!(&bytes[8..10], &[0x00, 0xFF]);
        assert!(bytes[10..].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn inline_image_degenerate_dims() {
        let params = b"/W 0 /H 1 /CS /G /BPC 8";
        let doc = empty_doc();
        assert!(
            decode_inline_image(
                &mut cpu_ctx(&doc),
                params,
                &[],
                #[cfg(feature = "gpu-jpeg-huffman")]
                None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
            )
            .is_none()
        );
    }

    #[test]
    fn inline_image_missing_dims() {
        // No width/height → None.
        let params = b"/CS /G /BPC 8";
        let doc = empty_doc();
        assert!(
            decode_inline_image(
                &mut cpu_ctx(&doc),
                params,
                &[0u8; 4],
                #[cfg(feature = "gpu-jpeg-huffman")]
                None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
            )
            .is_none()
        );
    }
}
