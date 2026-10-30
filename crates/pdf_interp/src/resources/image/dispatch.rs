//! Filter-chain dispatch from an image stream to its decoded samples.
//!
//! [`decode_image_stream`] is the one path from encoded stream bytes to an
//! [`ImageDescriptor`].  `XObject` images, inline images, soft masks and
//! stencil masks all decode through it, so each supports every filter chain
//! and `/DecodeParms` form the others do.

use std::borrow::Cow;

use pdf::Stream;

use super::codecs::{decode_ccitt, decode_dct, decode_jbig2, decode_jpx};
use super::raw::{apply_bilevel_decode, decode_raw};
use super::{
    ImageDecodeCtx, ImageDescriptor, ImageFilter, filter_name, image_filter_chain,
    is_transport_filter, terminal_decode_parms,
};

#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;

/// An image stream and the geometry its dictionary declares.
pub(super) struct ImageStream<'a> {
    /// The stream: its dictionary and its still-encoded bytes.
    pub stream: &'a Stream,
    /// `/Width`, validated.
    pub width: u32,
    /// `/Height`, validated.
    pub height: u32,
    /// `/ImageMask`: decode to stencil bytes ([`super::ImageColorSpace::Mask`]).
    pub is_mask: bool,
}

/// Decode `src` through its whole `/Filter` chain.
///
/// Every filter before the terminal one is a transport filter (PDF §7.4)
/// and is applied first; the terminal filter selects the codec.  A terminal
/// transport filter, or none, leaves raw samples for [`decode_raw`].  The
/// codec receives the `/DecodeParms` entry index-aligned to the terminal
/// filter.  `/Decode` is applied by the codec arms, so the samples returned
/// are final except for colour conversion.
///
/// `draw_size` is the device-pixel extent the image is drawn at, for a
/// codec that can reconstruct at reduced resolution; `None` decodes in
/// full.  `dct_cache` selects the device image cache entry of a
/// `DCTDecode` stream.  The error string names the stage that failed.
pub(super) fn decode_image_stream<
    #[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend,
>(
    ctx: &mut ImageDecodeCtx<'_>,
    src: &ImageStream<'_>,
    draw_size: Option<(u32, u32)>,
    #[cfg(feature = "cache")] dct_cache: Option<&super::codecs::DctCacheCtx<'_>>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<&mut JpegGpuDecoder<B>>,
) -> Result<ImageDescriptor, String> {
    let doc = ctx.doc;
    let dict = &src.stream.dict;
    let (w, h, is_mask) = (src.width, src.height, src.is_mask);

    let filter_obj = dict.get(b"Filter");
    let chain_len = filter_obj.map_or(0, |f| image_filter_chain(f).len());
    let filter = filter_obj.and_then(filter_name);
    let terminal_transport = filter.as_deref().is_some_and(is_transport_filter);

    // The codec input: the content with every transport filter applied.
    // The pass-through of the terminal image codec in `decompressed_content`
    // leaves its bytes unchanged.
    let input: Cow<'_, [u8]> = if chain_len > 1 || terminal_transport {
        match src.stream.decompressed_content() {
            Ok(data) => Cow::Owned(data),
            Err(e) => {
                let stage = if chain_len > 1 {
                    "chained filter".to_owned()
                } else {
                    filter.as_deref().unwrap_or_default().to_owned()
                };
                return Err(format!("{stage}: {e}"));
            }
        }
    } else {
        Cow::Borrowed(src.stream.content.as_slice())
    };
    let parms = terminal_decode_parms(dict, chain_len);

    let decoded = match filter.as_deref() {
        None => decode_raw(
            doc,
            &input,
            w,
            h,
            is_mask,
            dict,
            #[cfg(feature = "gpu-icc")]
            &mut ctx.icc,
        ),
        Some(f) if is_transport_filter(f) => decode_raw(
            doc,
            &input,
            w,
            h,
            is_mask,
            dict,
            #[cfg(feature = "gpu-icc")]
            &mut ctx.icc,
        ),
        Some("CCITTFaxDecode") => decode_ccitt(&input, w, h, is_mask, parms).map(|mut img| {
            apply_bilevel_decode(&mut img, dict);
            img
        }),
        Some("JBIG2Decode") => decode_jbig2(doc, &input, w, h, is_mask, parms).map(|mut img| {
            apply_bilevel_decode(&mut img, dict);
            img
        }),
        Some("DCTDecode") => decode_dct(
            &input,
            w,
            h,
            dict,
            ctx,
            #[cfg(feature = "gpu-jpeg-huffman")]
            jpeg_gpu,
            #[cfg(feature = "cache")]
            dct_cache,
        ),
        Some("JPXDecode") => {
            #[cfg(feature = "nvjpeg2k")]
            {
                decode_jpx(&input, w, h, draw_size, ctx.nvjpeg2k.as_deref_mut())
            }
            #[cfg(not(feature = "nvjpeg2k"))]
            {
                decode_jpx(&input, w, h, draw_size)
            }
        }
        Some(other) => return Err(format!("unsupported filter {other:?}")),
    };

    let mut img = decoded.ok_or_else(|| match filter.as_deref() {
        Some(codec @ ("CCITTFaxDecode" | "DCTDecode" | "JPXDecode" | "JBIG2Decode")) => {
            format!("{codec} decode failed")
        }
        _ => "raw/unfiltered decode failed".to_owned(),
    })?;
    img.filter = ImageFilter::from_filter_str(filter.as_deref());
    Ok(img)
}
