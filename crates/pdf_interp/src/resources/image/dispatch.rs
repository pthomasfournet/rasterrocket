//! Filter-chain dispatch from an image stream to its decoded, colour-converted
//! samples.
//!
//! [`decode_image_stream`] is the one path from encoded stream bytes to an
//! [`ImageDescriptor`].  `XObject` images, inline images, soft masks and
//! stencil masks all decode through it, so each supports every filter chain,
//! `/DecodeParms` form and colour space the others do.

use std::borrow::Cow;

use pdf::{Object, Stream};

use crate::resources::dict_ext::DictExt as _;

use super::codecs::{JpxOptions, decode_ccitt, decode_dct, decode_jbig2, decode_jpx};
use super::colorspace::{ImageColour, Samples};
use super::raw::{apply_stencil_decode, decode_raw};
use super::smask::{colour_key_mask, colour_key_mask_with};
use super::{
    ImageDecodeCtx, ImageDescriptor, ImageFilter, decode_budget, filter_name, image_filter_chain,
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
    /// The colour of the samples, resolved from the dictionary and the
    /// colour-space resources in effect.
    pub colour: ImageColour,
}

impl<'a> ImageStream<'a> {
    /// The stream of a `width × height` image whose samples are `colour`,
    /// or stencil bytes when `is_mask`, admitted for decoding.
    ///
    /// Refused, with the reason: a non-stencil image in a colour space
    /// without components (`Pattern`, which PDF §8.9.5.1 excludes from
    /// images), and an image whose stored or converted samples exceed the
    /// decode budget ([`super::MAX_DECODED_BYTES`]).  Nothing is allocated before this
    /// check.
    pub(super) fn new(
        stream: &'a Stream,
        width: u32,
        height: u32,
        is_mask: bool,
        colour: ImageColour,
    ) -> Result<Self, String> {
        if !is_mask && colour.components() == 0 {
            return Err(
                "a Pattern colour space has no components and cannot colour an image (PDF §8.9.5.1)"
                    .to_owned(),
            );
        }
        decode_budget(
            width,
            height,
            if is_mask {
                1
            } else {
                colour.budget_components()
            },
        )?;
        Ok(Self {
            stream,
            width,
            height,
            is_mask,
            colour,
        })
    }
}

/// A decoded, colour-converted image and what its stream lacked, if
/// anything (see [`Samples::partial`]).
pub(super) struct DecodedStream {
    pub image: ImageDescriptor,
    pub partial: Option<String>,
}

/// Decode `src` through its whole `/Filter` chain and convert its samples
/// to the colour the blit draws.
///
/// Every filter before the terminal one is a transport filter (PDF §7.4)
/// and is applied first; the terminal filter selects the codec.  A terminal
/// transport filter, or none, leaves raw samples for [`decode_raw`].  The
/// codec receives the `/DecodeParms` entry index-aligned to the terminal
/// filter.  A colour-key `/Mask` array is compared with the codec's
/// samples before `/Decode` and colour conversion; a raw image compares
/// its stored samples (see `decode_raw`).
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
) -> Result<DecodedStream, String> {
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
        None => decode_raw(&input, w, h, is_mask, dict, &src.colour),
        Some(f) if is_transport_filter(f) => decode_raw(&input, w, h, is_mask, dict, &src.colour),
        Some("CCITTFaxDecode") => decode_ccitt(&input, w, h, is_mask, parms).map(|mut s| {
            apply_stencil_decode(&mut s, dict);
            s
        }),
        Some("JBIG2Decode") => decode_jbig2(doc, &input, w, h, is_mask, parms).map(|mut s| {
            apply_stencil_decode(&mut s, dict);
            s
        }),
        Some("DCTDecode") => decode_dct(
            &input,
            w,
            h,
            ctx,
            #[cfg(feature = "gpu-jpeg-huffman")]
            jpeg_gpu,
            #[cfg(feature = "cache")]
            dct_cache,
        ),
        Some("JPXDecode") => {
            // An `/Indexed` space maps the codestream's samples through the
            // PDF palette (§8.6.6.3); any other space reads them as colour.
            let opts = JpxOptions {
                indices: !is_mask && src.colour.indexed(),
                smask_in_data: dict.get_i64(b"SMaskInData").unwrap_or(0),
            };
            #[cfg(feature = "nvjpeg2k")]
            let samples = decode_jpx(&input, w, h, draw_size, opts, ctx.nvjpeg2k.as_deref_mut());
            #[cfg(not(feature = "nvjpeg2k"))]
            let samples = decode_jpx(&input, w, h, draw_size, opts);
            Some(samples.map_err(|e| format!("JPXDecode: {e}"))?)
        }
        Some(other) => return Err(format!("unsupported filter {other:?}")),
    };

    let mut samples = decoded.ok_or_else(|| match filter.as_deref() {
        Some(codec @ ("CCITTFaxDecode" | "DCTDecode" | "JPXDecode" | "JBIG2Decode")) => {
            format!("{codec} decode failed")
        }
        _ => "raw/unfiltered decode failed".to_owned(),
    })?;
    let image_filter = ImageFilter::from_filter_str(filter.as_deref());

    // A codec's colour key is compared with its decoded samples; a raw
    // image's was compared with its stored samples in `decode_raw`.
    if !is_mask
        && !matches!(image_filter, ImageFilter::Raw | ImageFilter::Flate)
        && let Some(Object::Array(ranges)) = dict.get(b"Mask")
    {
        let bilevel = matches!(image_filter, ImageFilter::CcittFax | ImageFilter::Jbig2);
        samples.smask = codec_colour_key_mask(&samples, ranges, bilevel);
        if samples.smask.is_none() {
            log::warn!(
                "image: colour-key /Mask malformed or mismatched — blitting image without mask"
            );
        }
    }

    let partial = samples.partial.take();
    let mut image = src
        .colour
        .convert(
            doc,
            samples,
            #[cfg(feature = "gpu-icc")]
            &ctx.icc,
        )
        .ok_or_else(|| "colour conversion failed".to_owned())?;
    image.filter = image_filter;
    Ok(DecodedStream { image, partial })
}

/// The colour-key mask (§8.9.6.4) of a codec's host-resident samples.
/// A `bilevel` codec's samples are one bit each, expanded to `0x00` and
/// `0xFF`, so its ranges are compared with the bit values.
/// Device-resident samples cannot be range-tested; the mask is dropped
/// with a warning rather than silently.
fn codec_colour_key_mask(
    samples: &Samples,
    ranges: &[Object],
    bilevel: bool,
) -> Option<super::SoftMask> {
    let super::colorspace::Layout::Color(n) = samples.layout else {
        return None;
    };
    let Some(bytes) = samples.data.as_cpu() else {
        log::warn!(
            "image: colour-key /Mask on GPU-resident base image not supported — blitting image without mask"
        );
        return None;
    };
    if bilevel {
        let npix = (samples.width as usize).checked_mul(samples.height as usize)?;
        if bytes.len() < npix {
            return None;
        }
        return colour_key_mask_with(ranges, 1, samples.width, samples.height, |p, _| {
            u32::from(bytes[p] != 0)
        });
    }
    colour_key_mask(ranges, bytes, usize::from(n), samples.width, samples.height)
}
