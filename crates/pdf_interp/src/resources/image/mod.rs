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
mod resolve;
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
#[cfg(feature = "nvjpeg2k")]
pub(crate) use resolve::reduced_draw_size;
pub use resolve::resolve_image;
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
