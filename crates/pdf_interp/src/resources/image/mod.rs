//! PDF Image `XObject` resource extraction and decoding.
//!
//! [`resolve_image`] looks up a named `XObject` in the page resource dictionary
//! and, if it is an image, returns an [`ImageDescriptor`] with the decoded
//! pixel data ready for blitting.
//!
//! # Supported image types
//!
//! | Filter | `ImageMask` | Support |
//! |---|---|---|
//! | `CCITTFaxDecode` (`K<0`, Group 4 / T.6) | yes/no | yes |
//! | `CCITTFaxDecode` (`K=0`, Group 3 1D / T.4) | yes/no | yes |
//! | `FlateDecode` | yes/no | yes |
//! | none (raw) | yes/no | yes |
//! | `DCTDecode` (JPEG) | no | yes (CPU software or GPU nvJPEG) |
//! | `JPXDecode` (JPEG 2000) | no | yes (CPU software or GPU nvJPEG2000) |
//! | `JBIG2Decode` | yes/no | yes (via `hayro-jbig2`) |
//! | `CCITTFaxDecode` (`K>0`, Group 3 mixed 2D) | yes/no | yes (via `hayro-ccitt`) |
//!
//! # Pixel layout in `ImageDescriptor::data`
//!
//! | [`ImageColorSpace`] | Bytes per pixel | `0x00` meaning | `0xFF` meaning |
//! |---|---|---|---|
//! | `Gray` | 1 | black | white |
//! | `Rgb` | 3 | black (R=G=B=0) | white |
//! | `Mask` | 1 | paint with fill colour | transparent (leave background) |
//!
//! # nvJPEG acceleration (`nvjpeg` feature)
//!
//! When the crate is built with `--features nvjpeg`, `DCTDecode` streams with
//! pixel area ≥ `GPU_JPEG_THRESHOLD_PX` are decoded on the GPU via NVIDIA
//! nvJPEG instead of the CPU JPEG decoder.  Pass an `NvJpegDecoder` (from
//! `gpu::nvjpeg`) to [`resolve_image`] to enable this path; pass `None` for
//! CPU-only behaviour.
//!
//! # nvJPEG2000 acceleration (`nvjpeg2k` feature)
//!
//! When the crate is built with `--features nvjpeg2k`, `JPXDecode` streams
//! with pixel area ≥ `GPU_JPEG2K_THRESHOLD_PX` are decoded on the GPU via
//! NVIDIA nvJPEG2000 instead of the CPU JPEG 2000 decoder, when their
//! samples are one or three unsigned 8-bit components taken as they are
//! and their code-blocks use no coding style the library refuses.  A
//! [`DeviceJpx`] in [`ImageDecodeCtx::nvjpeg2k`] enables this path; `None`
//! is CPU-only behaviour.

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;

// ── Sub-modules ───────────────────────────────────────────────────────────────

pub(crate) mod bitpack;
pub(crate) mod codecs;
pub(crate) mod colorspace;
pub(crate) mod dispatch;
mod filters;
#[cfg(feature = "gpu-icc")]
pub(crate) mod icc_device;
pub(crate) mod inline;
mod limits;
pub(crate) mod raw;
pub(crate) mod smask;
mod thresholds;
mod types;

#[cfg(feature = "nvjpeg2k")]
pub(crate) use codecs::{DeviceJpx, Jpeg2kSource};
pub(crate) use filters::filter_name;
#[cfg(feature = "nvjpeg2k")]
pub(crate) use filters::image_filter_chain;
#[cfg(feature = "gpu-icc")]
pub(crate) use icc_device::{DeviceIccCounts, IccHandles};
pub(crate) use inline::decode_inline_image;
pub(crate) use limits::validated_dims;
use limits::{
    MAX_DECODED_BYTES, decode_budget, decode_budget_of, decoded_bytes, try_filled,
    try_with_capacity,
};
pub use smask::SoftMask;
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub(crate) use thresholds::GPU_JPEG_THRESHOLD_PX;
#[cfg(feature = "nvjpeg2k")]
pub(crate) use thresholds::{JpxSupply, draw_halves, jpx_area_dispatches, jpx_threshold};
pub(crate) use types::dominant_filter;
pub use types::{
    DecodedImage, IMAGE_FILTER_COUNT, ImageColorSpace, ImageData, ImageDecodeCtx, ImageDescriptor,
    ImageFilter, ImageRequest, ImageResolution, ImageTints, TintData,
};

// Visible only when cargo-fuzz sets `--cfg fuzzing`.  Not part of the public API.
// `pub(super)` codec functions cannot be re-exported across the crate boundary,
// so these thin wrappers bridge the visibility gap without widening it in normal
// builds.
#[cfg(fuzzing)]
#[doc(hidden)]
pub(crate) mod fuzz_entry {
    use pdf::{Document, Object};

    use super::codecs;

    /// Whether the stream decodes.
    #[must_use]
    pub fn decode_ccitt(
        data: &[u8],
        width: u32,
        height: u32,
        is_mask: bool,
        parms: Option<&Object>,
    ) -> bool {
        codecs::decode_ccitt(data, width, height, is_mask, parms).is_some()
    }

    /// Whether the stream decodes.
    #[must_use]
    pub fn decode_jbig2(
        doc: &Document,
        data: &[u8],
        width: u32,
        height: u32,
        is_mask: bool,
        parms: Option<&Object>,
    ) -> bool {
        codecs::decode_jbig2(doc, data, width, height, is_mask, parms).is_some()
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Look up a named `XObject` in `xobjects`, the `/XObject` resource
/// sub-dictionary in effect, and, if it is an image, decode and return its
/// pixels.
///
/// Returns:
/// - [`ImageResolution::Ok`] on success,
/// - [`ImageResolution::Absent`] if the object is an `XObject` of a subtype
///   with no raster appearance, or an image whose colour space paints
///   nothing (nothing to draw — skip silently),
/// - [`ImageResolution::DecodeFailed`] when the name does not resolve to a
///   readable `XObject` stream, the image dictionary is incomplete, or the
///   codec fails on the image or on its `/SMask` or stencil `/Mask`
///   (content is lost — surface to the caller).
#[must_use]
pub fn resolve_image(
    ctx: &mut ImageDecodeCtx<'_>,
    xobjects: &Dictionary,
    request: ImageRequest<'_>,
) -> ImageResolution {
    let doc = ctx.doc;
    let name = request.name;
    let unresolved = |what: &str| unresolved_image(name, what);
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            unused_variables,
            reason = "the object id keys the device image cache only"
        )
    )]
    let (stream_id, obj_arc) = match image_xobject(doc, xobjects, name) {
        Ok(found) => found,
        Err(Some(reason)) => return ImageResolution::DecodeFailed(reason),
        Err(None) => return ImageResolution::Absent,
    };
    let Some(stream) = obj_arc.as_ref().as_stream() else {
        return unresolved("object is not a stream");
    };

    let Some(w_raw) = stream.dict.get_i64(b"Width") else {
        return unresolved("image has no /Width");
    };
    let Some(h_raw) = stream.dict.get_i64(b"Height") else {
        return unresolved("image has no /Height");
    };
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        return unresolved(&format!("invalid dimensions {w_raw}×{h_raw}"));
    };

    let is_mask = stream.dict.get_bool(b"ImageMask").unwrap_or(false);
    let colour = colorspace::ImageColour::resolve(doc, &stream.dict, ctx.color_spaces);
    if !is_mask && colour.paints_nothing() {
        return ImageResolution::Absent;
    }
    let src = match dispatch::ImageStream::new(stream, w, h, is_mask, colour) {
        Ok(src) => src,
        Err(msg) => return unresolved(&msg),
    };

    let masks = ImageMasks::of(doc, &stream.dict, is_mask);
    let draw_size = masks.reduced_draw_size(request.draw_size);
    let ImageMasks {
        smask: smask_entry,
        mask: mask_entry,
        matte,
        ..
    } = masks;

    // The GPU blit composites cached images fully opaque and reads their
    // samples verbatim, so an image carrying a soft mask, an explicit mask,
    // or any colour conversion (a non-device space, or a `/Decode` remap)
    // stays host-resident: neither looked up (a content-hash hit from
    // another document would drop the mask or the conversion) nor
    // inserted.  The decode also bypasses the cache for an image drawn
    // below the resolution its frame header declares, for a canvas of the
    // wide working space, into which even device colours convert, and for
    // a 16-bit canvas, which the blit's 8-bit sampling does not serve.
    #[cfg(feature = "cache")]
    let dct_cache = if smask_entry.is_some()
        || mask_entry.is_some()
        || !src.colour.identity()
        || ctx.target.space != crate::WorkingSpace::Srgb
        || ctx.depth == color::SampleDepth::Sixteen
    {
        None
    } else {
        codecs::DctCacheCtx::from_resources(
            ctx.image_cache,
            ctx.doc_id,
            stream_id,
            request.draw_size,
        )
    };

    // The mask decodes first, so that its failure costs no base decode.
    let mask = match own_mask(ctx, smask_entry, mask_entry) {
        Ok(mask) => mask,
        Err(msg) => return unresolved(&msg),
    };

    let (mut img, partial, tints) = match dispatch::decode_image_stream(
        ctx,
        &src,
        draw_size,
        #[cfg(feature = "cache")]
        dct_cache.as_ref(),
    ) {
        Ok(decoded) => (decoded.image, decoded.partial, decoded.tints),
        Err(msg) => {
            log::warn!("image: /{}: {msg}", String::from_utf8_lossy(name));
            return ImageResolution::DecodeFailed(msg);
        }
    };

    // A colour-key mask, attached by the decode, is kept when no mask
    // stream replaces it.
    if mask.is_some() {
        img.smask = mask;
    }

    if let Some(matte) = &matte {
        smask::remove_matte(doc, ctx.target, src.colour.space(), matte, &mut img);
    }

    ImageResolution::Ok(DecodedImage {
        image: img,
        source_width: w,
        interpolate: stream.dict.get_bool(b"Interpolate").unwrap_or(false),
        own_soft_mask: matches!(smask_entry, Some(Object::Reference(_))),
        partial,
        tints,
        kept_tints: ctx.keep_tints,
        intent: ctx.target.intent.intent,
    })
}

/// [`ImageResolution::DecodeFailed`] for the image `XObject` `name`, with
/// the reason `what`, logged at warn.
fn unresolved_image(name: &[u8], what: &str) -> ImageResolution {
    ImageResolution::DecodeFailed(unresolved_reason(name, what))
}

/// The reason `what` an image `XObject` `name` cannot be drawn, logged at
/// warn.
fn unresolved_reason(name: &[u8], what: &str) -> String {
    log::warn!("image: /{}: {what}", String::from_utf8_lossy(name));
    what.to_owned()
}

/// The object id and the object of the image `XObject` stream `name` in
/// `xobjects`, the `/XObject` resource sub-dictionary in effect.
///
/// # Errors
/// `None` for an `XObject` of another subtype, which draws nothing; the
/// reason, logged, when `name` does not resolve to a readable image
/// stream.
fn image_xobject(
    doc: &Document,
    xobjects: &Dictionary,
    name: &[u8],
) -> Result<(ObjectId, std::sync::Arc<Object>), Option<String>> {
    let unresolved = |what: &str| Some(unresolved_reason(name, what));
    let Some(entry) = xobjects.get(name) else {
        return Err(unresolved("not found in the /XObject resources"));
    };
    let &Object::Reference(stream_id) = entry else {
        return Err(unresolved("resource entry is not an indirect reference"));
    };
    let obj = match doc.get_object(stream_id) {
        Ok(obj) => obj,
        Err(_) if doc.is_undefined(stream_id.0) => {
            return Err(unresolved(&format!(
                "object {} {} is not defined in the file",
                stream_id.0, stream_id.1
            )));
        }
        Err(e) => return Err(unresolved(&format!("object unreadable: {e}"))),
    };
    let Some(stream) = obj.as_ref().as_stream() else {
        return Err(unresolved("object is not a stream"));
    };
    match stream.dict.get_name(b"Subtype") {
        Some(b"Image") => Ok((stream_id, obj)),
        // A form reaches this path only when its stream could not be loaded.
        Some(b"Form") => Err(unresolved("form stream could not be loaded")),
        Some(_) => Err(None),
        None => Err(unresolved("no /Subtype")),
    }
}

/// The `/SMask` and `/Mask` entries of an image dictionary `dict`.  A
/// stencil (`is_mask`) has none (§8.9.6.2), and an entry that is null or
/// refers to an object the file does not define is absent (§7.3.10).  An
/// `/SMask` replaces a `/Mask` (§11.6.5.3).
/// The mask entries of an image dictionary that decide how its samples
/// decode.
struct ImageMasks<'d> {
    /// The defined `/SMask` entry.
    smask: Option<&'d Object>,
    /// The defined `/Mask` entry.
    mask: Option<&'d Object>,
    /// `/Mask` is a colour-key array that no `/SMask` replaces.
    colour_key: bool,
    /// The matte colour of a soft mask the image's samples are pre-blended
    /// with (§11.6.5.3).
    matte: Option<Vec<f64>>,
}

impl<'d> ImageMasks<'d> {
    /// The masks of the image dictionary `dict`; none for an image mask.
    fn of(doc: &Document, dict: &'d Dictionary, is_mask: bool) -> Self {
        let (smask, mask) = mask_entries(doc, dict, is_mask);
        let colour_key = smask.is_none() && matches!(mask, Some(Object::Array(_)));
        let matte = match smask {
            Some(Object::Reference(id)) => smask::matte(doc, *id),
            _ => None,
        };
        Self {
            smask,
            mask,
            colour_key,
            matte,
        }
    }

    /// The size a decode drawn at `draw_size` may reduce to.  A colour-key
    /// mask compares exact samples, which a reduced decode averages, and a
    /// matte is removed pixel by pixel on the mask's grid, so such an image
    /// decodes in full.  A mask stream otherwise keeps its own resolution
    /// and needs no such exclusion.
    fn reduced_draw_size(&self, draw_size: Option<(u32, u32)>) -> Option<(u32, u32)> {
        draw_size.filter(|_| self.matte.is_none() && !self.colour_key)
    }
}

/// The size the decode of the image dictionary `dict` drawn at `draw_size`
/// may reduce to: `draw_size`, unless the image's masks require a full
/// decode (see `ImageMasks::reduced_draw_size`).
#[cfg(feature = "nvjpeg2k")]
pub(crate) fn reduced_draw_size(
    doc: &Document,
    dict: &Dictionary,
    draw_size: Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    let is_mask = dict.get_bool(b"ImageMask").unwrap_or(false);
    ImageMasks::of(doc, dict, is_mask).reduced_draw_size(draw_size)
}

fn mask_entries<'d>(
    doc: &Document,
    dict: &'d Dictionary,
    is_mask: bool,
) -> (Option<&'d Object>, Option<&'d Object>) {
    if is_mask {
        return (None, None);
    }
    (
        defined(doc, dict.get(b"SMask")),
        defined(doc, dict.get(b"Mask")),
    )
}

/// `entry`, unless it is the null object, written as `null` or as a
/// reference to an object `doc` does not define (PDF §7.3.10): an entry
/// whose value is null is absent (§7.3.7).
fn defined<'o>(doc: &Document, entry: Option<&'o Object>) -> Option<&'o Object> {
    match entry {
        Some(Object::Null) => None,
        Some(Object::Reference(id)) if doc.is_undefined(id.0) => None,
        other => other,
    }
}

/// The image's own mask, from its `/SMask` entry `smask` or its stencil
/// `/Mask` entry `mask` (PDF §8.9.6, §11.6.5.2); `None` when it has none.
/// A colour-key `/Mask` array is applied by the base decode itself.
///
/// # Errors
/// The reason, naming the mask, when the mask fails to decode.  The mask is
/// part of the image's definition, so such an image cannot be drawn.
fn own_mask(
    ctx: &mut ImageDecodeCtx<'_>,
    smask: Option<&Object>,
    mask: Option<&Object>,
) -> Result<Option<SoftMask>, String> {
    let named = |kind: &'static str, id: ObjectId| {
        move |e: String| format!("{kind} (object {} {}): {e}", id.0, id.1)
    };
    match (smask, mask) {
        (Some(Object::Reference(id)), _) => smask::decode_smask(ctx, *id)
            .map(Some)
            .map_err(named("/SMask", *id)),
        (_, Some(Object::Reference(id))) => smask::decode_explicit_mask(ctx, *id)
            .map(Some)
            .map_err(named("/Mask", *id)),
        (_, Some(Object::Array(_)) | None) => Ok(None),
        (_, Some(_)) => {
            log::warn!("image: /Mask is neither a reference nor an array; ignored");
            Ok(None)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A colour-key `/Mask` on a JPEG 2000 image of more than 8 bits is
    /// compared in the codestream's own range: on the 16-bit gray fixture,
    /// whose second row holds 9362 and whose other rows lie outside
    /// [9000, 9500], the key masks that row alone.
    #[test]
    fn a_jpx_colour_key_compares_the_samples_at_their_own_precision() {
        let jpx = include_bytes!("testdata/gray16.jp2");
        let mut image = format!(
            "<< /Type /XObject /Subtype /Image /Width 8 /Height 8 /ColorSpace /DeviceGray \
             /BitsPerComponent 16 /Filter /JPXDecode /Mask [9000 9500] /Length {} >>\nstream\n",
            jpx.len()
        )
        .into_bytes();
        image.extend_from_slice(jpx);
        image.extend_from_slice(b"\nendstream");
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, image)]);
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((4, 0)));
        let ImageResolution::Ok(decoded) = resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        ) else {
            panic!("the image decodes");
        };
        let mask = decoded.image.smask.expect("the colour key masks");
        for (row, alpha) in mask.alpha().as_chunks::<8>().0.iter().enumerate() {
            let want = if row == 1 { 0 } else { 255 };
            assert!(alpha.iter().all(|&a| a == want), "row {row}: {alpha:?}");
        }
    }

    /// `/Im1`, object 4 of `doc`, resolved for a render at `depth`.
    fn resolve_im1(doc: &Document, depth: color::SampleDepth) -> DecodedImage {
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((4, 0)));
        let mut ctx = ImageDecodeCtx::cpu(doc);
        ctx.depth = depth;
        let ImageResolution::Ok(decoded) =
            resolve_image(&mut ctx, &xobjects, ImageRequest::full(b"Im1"))
        else {
            panic!("the image decodes");
        };
        decoded
    }

    /// A document whose object 4 is an image `XObject` of the dictionary
    /// entries `dict` and the stream bytes `data`.
    fn image_doc(dict: &str, data: &[u8]) -> Document {
        let mut image = format!(
            "<< /Type /XObject /Subtype /Image {dict} /Length {} >>\nstream\n",
            data.len()
        )
        .into_bytes();
        image.extend_from_slice(data);
        image.extend_from_slice(b"\nendstream");
        crate::test_helpers::make_doc_with_objects(&[(4, image)])
    }

    /// The colour key of a JPEG 2000 codestream of 4-bit samples compares
    /// them in their own range, `[15 15]` keying the value 15 out, at either
    /// depth.
    #[test]
    fn a_jpx_colour_key_compares_4_bit_samples_in_their_own_range() {
        let doc = image_doc(
            "/Width 4 /Height 2 /ColorSpace /DeviceGray /BitsPerComponent 4 \
             /Filter /JPXDecode /Mask [15 15]",
            include_bytes!("testdata/gray4.jp2"),
        );
        for depth in [color::SampleDepth::Eight, color::SampleDepth::Sixteen] {
            let mask = resolve_im1(&doc, depth)
                .image
                .smask
                .expect("the colour key masks");
            assert_eq!(mask.alpha(), [255, 255, 0, 0, 0, 255, 0, 255], "{depth:?}");
        }
    }

    /// At depth 16 a colour key still compares the samples in their own
    /// range: a 16-bit JPEG 2000 codestream and 16-bit raw samples, both
    /// kept at 16 bits, mask the samples inside `[9000 9500]`.
    #[test]
    fn a_colour_key_at_depth_16_compares_the_source_samples() {
        let sixteen = color::SampleDepth::Sixteen;
        let jpx = image_doc(
            "/Width 8 /Height 8 /ColorSpace /DeviceGray /BitsPerComponent 16 \
             /Filter /JPXDecode /Mask [9000 9500]",
            include_bytes!("testdata/gray16.jp2"),
        );
        let decoded = resolve_im1(&jpx, sixteen);
        assert!(decoded.image.data.as_cpu16().is_some(), "kept at 16 bits");
        let mask = decoded.image.smask.expect("the colour key masks");
        for (row, alpha) in mask.alpha().as_chunks::<8>().0.iter().enumerate() {
            let want = if row == 1 { 0 } else { 255 };
            assert!(alpha.iter().all(|&a| a == want), "row {row}: {alpha:?}");
        }
        let samples: Vec<u8> = [9362u16, 100, 9000, 9501]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let raw = image_doc(
            "/Width 4 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 16 \
             /Mask [9000 9500]",
            &samples,
        );
        let decoded = resolve_im1(&raw, sixteen);
        assert!(decoded.image.data.as_cpu16().is_some(), "kept at 16 bits");
        let mask = decoded.image.smask.expect("the colour key masks");
        assert_eq!(mask.alpha(), [0, 255, 0, 255]);
    }

    // ── Spot-colour images through the full XObject path ─────────────────────

    /// Decode `/Im1`, an image whose `ColorSpace` is `[/Separation /Black
    /// /DeviceCMYK <fn>]` with the tint transform `t ↦ (0, 0, 0, t)`, carrying
    /// `image_dict_extras` and the raw sample bytes `samples`.
    fn resolve_separation_black(image_dict_extras: &str, samples: &[u8]) -> ImageDescriptor {
        const PROGRAM: &str = "{dup 0 mul exch dup 0 mul exch dup 0 mul exch 1 mul }";
        let tint_fn = format!(
            "<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
             stream\n{PROGRAM}\nendstream",
            PROGRAM.len()
        )
        .into_bytes();
        let mut image = format!(
            "<< /Type /XObject /Subtype /Image /ColorSpace [/Separation /Black /DeviceCMYK 4 0 R] \
             {image_dict_extras} /Length {} >>\nstream\n",
            samples.len()
        )
        .into_bytes();
        image.extend_from_slice(samples);
        image.extend_from_slice(b"\nendstream");
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, tint_fn), (5, image)]);

        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((5, 0)));

        let resolution = resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        );
        match resolution {
            ImageResolution::Ok(decoded) => decoded.image,
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    #[test]
    fn separation_image_zero_tint_is_white() {
        // PDF §8.6.6.4: an 8-bit Separation sample is a tint — 0 is no
        // colorant (paper), 255 is full colorant.  Treating the bytes as
        // gray levels renders the page as its negative.
        let img =
            resolve_separation_black("/Width 3 /Height 1 /BitsPerComponent 8", &[0, 128, 255]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        let px = img.data.as_cpu().unwrap();
        let solid_k = color::cmyk::CORNERS[8][0];
        assert_eq!(px[0], 255, "tint 0 = no ink = white");
        assert_eq!(px[2], solid_k, "tint 1 = solid K");
        assert_eq!(
            px[1],
            color::cmyk::cmyk_to_rgb(0, 0, 0, 128)[0],
            "half tint is the half-K grey"
        );
    }

    #[test]
    fn separation_image_decode_array_applies_before_the_tint_transform() {
        // §8.9.5.2: Decode maps samples into the tint domain first, so
        // [1 0] turns sample 0 into full colorant.
        let img = resolve_separation_black(
            "/Width 2 /Height 1 /BitsPerComponent 8 /Decode [1 0]",
            &[0, 255],
        );
        assert_eq!(
            img.data.as_cpu().unwrap(),
            &[color::cmyk::CORNERS[8][0], 255]
        );
    }

    #[test]
    fn separation_image_one_bit_samples_follow_tint_polarity() {
        // One packed row, MSB first: only the first pixel carries colorant.
        let img = resolve_separation_black("/Width 8 /Height 1 /BitsPerComponent 1", &[0x80]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(px[0], color::cmyk::CORNERS[8][0], "set bit = tint 1 = ink");
        assert!(
            px[1..].iter().all(|&v| v == 255),
            "clear bits = tint 0 = paper"
        );
    }

    /// Resolve `/Im1`, object 4 of a document holding `image` (the whole
    /// stream object, dictionary and data).
    fn resolve_object(image: Vec<u8>) -> ImageResolution {
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, image)]);
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((4, 0)));
        resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        )
    }

    /// §8.9.5.1: an image's colour space may be of any type except
    /// `Pattern`, which has no components; such an image is refused, not
    /// converted through a zero-component sample layout.
    #[test]
    fn pattern_colour_space_image_is_refused() {
        let image = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                      /ColorSpace /Pattern /BitsPerComponent 8 /Length 2 >>\n\
                      stream\n\x10\x20\nendstream"
            .to_vec();
        match resolve_object(image) {
            ImageResolution::DecodeFailed(msg) => assert!(msg.contains("Pattern"), "{msg}"),
            ImageResolution::Ok(_) => panic!("a Pattern image must not decode"),
            ImageResolution::Absent => panic!("a Pattern image is a failure, not absent"),
        }
    }

    /// Resolve `/Im1`, object 5: a 2 × 1 gray image whose dictionary
    /// carries `mask_entry`, over object 4, the mask stream `mask`.
    fn resolve_with_mask(mask_entry: &str, mask: Vec<u8>) -> ImageResolution {
        let image = format!(
            "<< /Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceGray \
             /BitsPerComponent 8 {mask_entry} /Length 2 >>\nstream\n\x10\x20\nendstream"
        )
        .into_bytes();
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, mask), (5, image)]);
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((5, 0)));
        resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        )
    }

    /// A truncated `DCTDecode` stream.
    const CORRUPT_DCT: &[u8] = &[0xFF, 0xD8, 0xFF];

    /// An image whose `/SMask` fails to decode is an undecodable image,
    /// never drawn unmasked; the reason names the mask.
    #[test]
    fn undecodable_smask_fails_the_image() {
        let mut mask = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                         /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode \
                         /Length 3 >>\nstream\n"
            .to_vec();
        mask.extend_from_slice(CORRUPT_DCT);
        mask.extend_from_slice(b"\nendstream");
        match resolve_with_mask("/SMask 4 0 R", mask) {
            ImageResolution::DecodeFailed(msg) => {
                assert!(msg.starts_with("/SMask (object 4 0): "), "{msg}");
            }
            ImageResolution::Ok(_) => panic!("an image whose SMask fails must not be drawn"),
            ImageResolution::Absent => panic!("the image is a failure, not absent"),
        }
    }

    /// A stencil `/Mask` reference that is no stencil fails the image.
    #[test]
    fn unusable_stencil_mask_fails_the_image() {
        let mask = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                     /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 2 >>\n\
                     stream\n\x00\x00\nendstream"
            .to_vec();
        match resolve_with_mask("/Mask 4 0 R", mask) {
            ImageResolution::DecodeFailed(msg) => {
                assert!(
                    msg.contains("/Mask (object 4 0): not an ImageMask"),
                    "{msg}"
                );
            }
            ImageResolution::Ok(_) => panic!("an image whose /Mask fails must not be drawn"),
            ImageResolution::Absent => panic!("the image is a failure, not absent"),
        }
    }

    /// A mask reference to an object the file does not define is the null
    /// object: no mask, and the image is drawn (§7.3.10).
    #[test]
    fn mask_reference_to_an_undefined_object_is_no_mask() {
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray \
               /BitsPerComponent 8 /SMask 99 0 R /Length 1 >>\nstream\n\x10\nendstream"
                .to_vec(),
        )]);
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((4, 0)));
        match resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        ) {
            ImageResolution::Ok(decoded) => {
                assert!(decoded.image.smask.is_none());
                assert!(!decoded.own_soft_mask);
            }
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    /// A one-bit `/Mask` stream without `/ImageMask true` is read as the
    /// stencil its samples can only be.
    #[test]
    fn one_bit_mask_without_the_image_mask_flag_is_a_stencil() {
        let mask = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                     /ColorSpace /DeviceGray /BitsPerComponent 1 /Length 1 >>\n\
                     stream\n\x40\nendstream"
            .to_vec();
        match resolve_with_mask("/Mask 4 0 R", mask) {
            ImageResolution::Ok(decoded) => {
                let mask = decoded.image.smask.expect("the stencil is attached");
                assert_eq!(mask.alpha(), &[0xFF, 0x00]);
            }
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    /// A `null` `/Mask` is absent (§7.3.7): the image is drawn unmasked.
    #[test]
    fn null_mask_entry_is_no_mask() {
        match resolve_with_mask("/Mask null", b"null".to_vec()) {
            ImageResolution::Ok(decoded) => assert!(decoded.image.smask.is_none()),
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    /// A stencil carries no mask (§8.9.6.2): a stray `/SMask` on one is
    /// ignored, even one that does not decode.
    #[test]
    fn stencil_ignores_a_stray_smask() {
        let mut mask = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                         /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode \
                         /Length 3 >>\nstream\n"
            .to_vec();
        mask.extend_from_slice(CORRUPT_DCT);
        mask.extend_from_slice(b"\nendstream");
        let stencil = b"<< /Type /XObject /Subtype /Image /Width 8 /Height 1 /ImageMask true \
                        /SMask 4 0 R /Length 1 >>\nstream\n\x0F\nendstream"
            .to_vec();
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, mask), (5, stencil)]);
        let mut xobjects = Dictionary::new();
        xobjects.set("Im1", Object::Reference((5, 0)));
        match resolve_image(
            &mut ImageDecodeCtx::cpu(&doc),
            &xobjects,
            ImageRequest::full(b"Im1"),
        ) {
            ImageResolution::Ok(decoded) => {
                assert!(decoded.image.smask.is_none());
                assert!(!decoded.own_soft_mask);
            }
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    /// A decodable `/SMask` is attached to the image.
    #[test]
    fn decodable_smask_is_attached() {
        let mask = b"<< /Type /XObject /Subtype /Image /Width 2 /Height 1 \
                     /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 2 >>\n\
                     stream\n\x40\xC0\nendstream"
            .to_vec();
        match resolve_with_mask("/SMask 4 0 R", mask) {
            ImageResolution::Ok(decoded) => {
                let mask = decoded.image.smask.expect("the SMask is attached");
                assert_eq!(mask.alpha(), &[0x40, 0xC0]);
                assert!(decoded.own_soft_mask);
            }
            ImageResolution::DecodeFailed(msg) => panic!("decode failed: {msg}"),
            ImageResolution::Absent => panic!("image resource not found"),
        }
    }

    /// The budget charges the converted samples: 20 000 × 20 000 palette
    /// indices are 381 MiB, their RGB colours 1 144 MiB, over the 1 GiB
    /// budget, so the image is refused before any buffer is allocated.
    #[test]
    fn indexed_image_is_budgeted_by_its_rgb_output() {
        let image = b"<< /Type /XObject /Subtype /Image /Width 20000 /Height 20000 \
                      /ColorSpace [/Indexed /DeviceRGB 1 <ff000000ff00>] \
                      /BitsPerComponent 8 /Length 1 >>\nstream\n\x00\nendstream"
            .to_vec();
        match resolve_object(image) {
            ImageResolution::DecodeFailed(msg) => assert!(msg.contains("budget"), "{msg}"),
            ImageResolution::Ok(_) => panic!("the RGB output exceeds the budget"),
            ImageResolution::Absent => panic!("the image is a failure, not absent"),
        }
    }
}
