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
//! NVIDIA nvJPEG2000 instead of the CPU JPEG 2000 decoder.  Pass an
//! `NvJpeg2kDecoder` (from `gpu::nvjpeg2k`) to [`resolve_image`] to enable this path; pass `None`
//! for CPU-only behaviour.  Only 1- and 3-component images are accelerated;
//! CMYK and other multi-channel images always fall through to the CPU path.

use std::borrow::Cow;

use pdf::{Dictionary, Document, Object};

use crate::resources::dict_ext::DictExt;

// ── GPU JPEG/JPEG2k acceleration ──────────────────────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::NvJpeg2kDecoder;

#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;

// ── GPU parallel-Huffman JPEG decoder ────────────────────────────────────────

#[cfg(feature = "gpu-jpeg-huffman")]
use gpu::jpeg_decoder::JpegGpuDecoder;

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
#[path = "../icc.rs"]
pub(crate) mod icc;

/// Minimum pixel area (width × height) for GPU-accelerated `DCTDecode`.
///
/// **Currently set to `u32::MAX` — nvJPEG dispatch is disabled on consumer
/// Blackwell.**  Verified on RTX 5070 (Blackwell, one NVJPG engine) running
/// the v0.6.0 baseline (10-corpus matrix, 2026-05-06): the GPU JPEG path was
/// 5–13× *slower* than 24-thread CPU zune-jpeg on every DCT-heavy corpus.
/// The mechanism is documented in the v0.6.0 baseline spec; the short
/// version is:
///
/// - `NVJPEG_BACKEND_HARDWARE` would serialise all 24 worker threads
///   through one on-die engine (consumer Blackwell has 1 NVJPG engine,
///   not multiple as on A100/H100/Ada datacenter parts).
/// - `NVJPEG_BACKEND_GPU_HYBRID` (currently selected; see
///   `crates/gpu/src/nvjpeg.rs`) only uses GPU for Huffman decoding when
///   batch size > 50–100; for single-image-per-call workloads it falls
///   back to CPU work behind a CUDA stream-sync wrapper, paying GPU-API
///   overhead for a CPU decode that is slower than calling zune-jpeg
///   directly.
/// - The batched nvJPEG API would fit, but requires collecting images
///   across pages, decoding them in batches of ≥ 100, then routing
///   results back to the per-page renderers — a major architectural
///   change not yet done.
///
/// On a datacenter GPU with multiple NVJPG engines (A100, H100, A30),
/// the `HARDWARE` backend should win at the original 512 × 512 threshold.
/// When that hardware becomes available the threshold should drop back
/// to `262_144`, the backend selector should be revisited, and the
/// matrix re-measured.
///
/// nvJPEG2000 (`GPU_JPEG2K_THRESHOLD_PX`) is *not* affected by this:
/// JPX decode on the same matrix was 1.7× faster on GPU than CPU on
/// corpus 10, validating the original threshold for JPEG 2000.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub const GPU_JPEG_THRESHOLD_PX: u32 = u32::MAX;

/// Minimum pixel area (width × height) for GPU parallel-Huffman JPEG decode.
///
/// Only 4:4:4 baseline JPEGs with 1 or 3 components are eligible.
/// Images below this threshold always use the CPU path.
///
/// Set to `u32::MAX` until a corpus benchmark confirms a crossover point —
/// mirrors the precedent set by `GPU_JPEG_THRESHOLD_PX` for nvJPEG.
///
/// Override at runtime for benchmarking: set the `PDF_RASTER_HUFFMAN_THRESHOLD`
/// environment variable to a decimal pixel-area value (e.g. `0` = always GPU,
/// `4294967295` = never GPU).  The env var is read once, lazily, on first use.
#[cfg(feature = "gpu-jpeg-huffman")]
pub const GPU_JPEG_HUFFMAN_THRESHOLD_PX: u32 = u32::MAX;

/// Returns the effective parallel-Huffman dispatch threshold: `GPU_JPEG_HUFFMAN_THRESHOLD_PX`
/// unless `PDF_RASTER_HUFFMAN_THRESHOLD` is set in the environment.
#[cfg(feature = "gpu-jpeg-huffman")]
pub(crate) fn huffman_threshold() -> u32 {
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<u32> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("PDF_RASTER_HUFFMAN_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(GPU_JPEG_HUFFMAN_THRESHOLD_PX)
    })
}

/// Minimum pixel area (width × height) for GPU-accelerated `JPXDecode`.
///
/// Below this threshold `PCIe` transfer overhead dominates and the CPU decoder
/// is faster.  512 × 512 = 262 144 pixels — same crossover as nvJPEG; JPEG 2000
/// decode is CPU-bound at similar pixel counts.
#[cfg(feature = "nvjpeg2k")]
pub const GPU_JPEG2K_THRESHOLD_PX: u32 = 262_144;

// ── Sub-modules ───────────────────────────────────────────────────────────────

pub(crate) mod bitpack;
pub(crate) mod codecs;
pub(crate) mod colorspace;
pub(crate) mod dispatch;
pub(crate) mod inline;
pub(crate) mod raw;
pub(crate) mod smask;

pub use inline::decode_inline_image;
pub use smask::SoftMask;

// Visible only when cargo-fuzz sets `--cfg fuzzing`.  Not part of the public API.
// `pub(super)` codec functions cannot be re-exported across the crate boundary,
// so these thin wrappers bridge the visibility gap without widening it in normal
// builds.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_entry {
    use pdf::{Document, Object};

    use super::codecs;

    /// Whether the stream decodes.
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

// ── Resolution tri-state ─────────────────────────────────────────────────────

/// Outcome of [`resolve_image`]: distinguishes a genuine decode failure from a
/// legitimately-absent resource so callers can surface errors rather than silently
/// rendering a blank region.
pub enum ImageResolution {
    /// Image decoded successfully.
    Ok(DecodedImage),
    /// The codec returned `None` or the stream was structurally invalid.  The
    /// page is incomplete; the message should be surfaced to the caller.
    DecodeFailed(String),
    /// The image has no raster appearance: the named `XObject` is of a
    /// non-image subtype (e.g. `/PS`), or the image's colour space paints
    /// nothing (every colorant `/None`, PDF §8.6.6.4).  Nothing is lost by
    /// skipping it.
    Absent,
}

/// A decoded image with the dictionary entries that govern its drawing.
pub struct DecodedImage {
    /// The decoded image.  A decoder that reconstructs at reduced resolution
    /// (see [`ImageRequest::draw_size`]) yields fewer pixels than the stream
    /// holds.
    pub image: ImageDescriptor,
    /// The stream's `/Width`: the source pixel count across, whatever
    /// resolution `image` was decoded at.
    pub source_width: u32,
    /// `/Interpolate`: the image asks to be interpolated when magnified
    /// (PDF 32000-1 §8.9.5.3).
    pub interpolate: bool,
    /// The image dictionary has an `/SMask` entry, which replaces the soft
    /// mask of the graphics state for this image (PDF 32000-1 Table 89).
    pub own_soft_mask: bool,
    /// What the stream lacked when `image` holds padding in place of
    /// samples the stream does not have (a stream shorter than its
    /// dictionary declares, or a codec that stopped short): the image is
    /// drawn, and the page is incomplete.
    pub partial: Option<String>,
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Colour space of the decoded image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// One byte per pixel, 0 = black, 255 = white.
    Gray,
    /// Three bytes per pixel: R, G, B.
    Rgb,
    /// 1-bit mask: pixel = 0 means "paint with current colour", 1 = transparent.
    /// Decoded as one byte per pixel, `0x00` paint and `0xFF` transparent; an
    /// intermediate byte `v` (from area-averaging reduction) paints with
    /// coverage `255 − v`.
    Mask,
}

impl ImageColorSpace {
    /// Decoded bytes per pixel after the codec produces an [`ImageDescriptor`].
    /// `Mask` is byte-expanded by the decoders (one byte per pixel, not bit-packed).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray | Self::Mask => 1,
            Self::Rgb => 3,
        }
    }
}

/// The compression filter used to store an image in the PDF stream.
///
/// Used by [`PageDiagnostics`](crate::renderer::page::PageDiagnostics) to
/// classify image content without re-reading the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum ImageFilter {
    /// `DCTDecode` — JPEG baseline or progressive.
    Dct = 0,
    /// `JPXDecode` — JPEG 2000.
    Jpx = 1,
    /// `CCITTFaxDecode` — Group 3 (T.4) or Group 4 (T.6) fax compression.
    CcittFax = 2,
    /// `JBIG2Decode` — JBIG2 bilevel compression.
    Jbig2 = 3,
    /// `FlateDecode` — zlib/deflate.
    Flate = 4,
    /// No filter — raw uncompressed pixels.
    Raw = 5,
}

impl ImageFilter {
    /// Map a PDF filter name string to the matching [`ImageFilter`] variant.
    ///
    /// Unrecognised or absent names map to [`ImageFilter::Raw`].  This is the
    /// single canonical mapping used by the renderer, the inline-image decoder,
    /// and the pre-scan pass.
    #[must_use]
    pub(crate) fn from_filter_str(name: Option<&str>) -> Self {
        match name {
            Some("DCTDecode") => Self::Dct,
            Some("JPXDecode") => Self::Jpx,
            Some("CCITTFaxDecode") => Self::CcittFax,
            Some("JBIG2Decode") => Self::Jbig2,
            Some("FlateDecode") => Self::Flate,
            _ => Self::Raw,
        }
    }
}

/// Number of [`ImageFilter`] variants — must match `filter_counts` array size in
/// `PageRenderer`.  The const assert in that module enforces this at compile time.
pub const IMAGE_FILTER_COUNT: usize = 6;

/// Backing storage for decoded image pixels.
///
/// Two variants depending on whether the `cache` feature is on:
///
/// - [`ImageData::Cpu`] always available; produced by every CPU
///   decode path (CCITT, JBIG2, raw, JPEG fallback).
/// - `ImageData::Gpu` (feature `cache`) — `Arc<CachedDeviceImage>`
///   handed back from the device image cache.  Produced by
///   `decode_dct` when a cache is wired in; consumed by the GPU
///   blit kernel which writes transformed pixels into the page's
///   `DevicePageBuffer` without a `PCIe` round-trip.
///
/// The enum is [`#[non_exhaustive]`] so future variants (e.g. an
/// `Arc<[u8]>` shared-host variant) can be added without breaking
/// downstream `match` consumers.
///
/// Hot-path access goes through [`Self::as_cpu`] which returns
/// `Option<&[u8]>` — `Some(&[..])` for the CPU variant, `None` for the
/// GPU variant.  GPU consumers must match on the variant explicitly to
/// reach the device pointer.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageData {
    /// Host-resident pixel bytes — layout defined by [`ImageColorSpace`].
    Cpu(Vec<u8>),
    /// Device-resident image, owned by the device image cache.  The `Arc`
    /// keeps the slab alive past in-flight kernel reads even if the
    /// cache evicts; see `gpu::cache::CachedDeviceImage` docs.
    #[cfg(feature = "cache")]
    Gpu(std::sync::Arc<gpu::cache::CachedDeviceImage>),
}

impl ImageData {
    /// Borrow the host-resident pixel bytes if this is a CPU-backed image.
    /// Returns `None` for variants whose pixels live in device memory.
    #[must_use]
    pub const fn as_cpu(&self) -> Option<&[u8]> {
        match self {
            Self::Cpu(bytes) => Some(bytes.as_slice()),
            #[cfg(feature = "cache")]
            Self::Gpu(_) => None,
        }
    }

    /// Length in bytes of the underlying pixel storage.  Cheap for either
    /// variant — no host memory access required for the GPU variant.
    #[must_use]
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            clippy::missing_const_for_fn,
            reason = "the cache-feature path calls a non-const cudarc method; keep one signature"
        )
    )]
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(bytes) => bytes.len(),
            #[cfg(feature = "cache")]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "vram_bytes returns u64 for forward-compat; 32-bit hosts will OOM long before this matters in practice"
            )]
            Self::Gpu(cached) => cached.vram_bytes() as usize,
        }
    }

    /// Whether the underlying pixel storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An image `XObject` to resolve, by resource name, and the size it will be
/// drawn at.
#[derive(Clone, Copy, Debug)]
pub struct ImageRequest<'a> {
    /// The `XObject` resource name.
    pub name: &'a [u8],
    /// Device-pixel extent `(width, height)` the image's unit square maps to,
    /// when known.  A decoder that can reconstruct at reduced resolution
    /// (JPEG 2000) decodes at no less than this along either axis; `None`
    /// decodes at full resolution.  Every other decoder ignores it, so their
    /// decodes serve draws of any size.
    pub draw_size: Option<(u32, u32)>,
}

impl<'a> ImageRequest<'a> {
    /// Request `name` at full resolution.
    #[must_use]
    pub const fn full(name: &'a [u8]) -> Self {
        Self {
            name,
            draw_size: None,
        }
    }
}

/// Decoded image data, ready for blitting onto the page bitmap.
///
/// See the module-level table for the exact byte layout per [`ImageColorSpace`].
#[derive(Debug)]
pub struct ImageDescriptor {
    /// Pixel width of the decoded image.
    pub width: u32,
    /// Pixel height of the decoded image.
    pub height: u32,
    /// Colour interpretation of `data`.
    pub color_space: ImageColorSpace,
    /// Pixel storage — host or (in future) device-resident.  Layout defined
    /// by `color_space` (see module doc).
    pub data: ImageData,
    /// Optional mask (`/SMask`, `/Mask`, or an embedded alpha channel) over
    /// the same unit square, at its own resolution, which need not be the
    /// image's.  `None` means every pixel is fully opaque.
    pub smask: Option<SoftMask>,
    /// The compression filter that was applied to this image in the PDF stream.
    pub filter: ImageFilter,
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
///   codec fails (content is lost — surface to the caller).
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "resource lookup, decode and mask attachment read as one sequence"
)]
pub fn resolve_image<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    ctx: &mut ImageDecodeCtx<'_>,
    xobjects: &Dictionary,
    request: ImageRequest<'_>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<&mut JpegGpuDecoder<B>>,
) -> ImageResolution {
    let doc = ctx.doc;
    let name = request.name;
    let unresolved = |what: &str| {
        let msg = format!("XObject /{}: {what}", String::from_utf8_lossy(name));
        log::warn!("image: {msg}");
        ImageResolution::DecodeFailed(msg)
    };
    let Some(entry) = xobjects.get(name) else {
        return unresolved("not found in the /XObject resources");
    };
    let &Object::Reference(stream_id) = entry else {
        return unresolved("resource entry is not an indirect reference");
    };
    // Bind the Arc to a local so the borrow into the stream below stays alive.
    let Ok(obj_arc) = doc.get_object(stream_id) else {
        return unresolved("object unreadable");
    };
    let Some(stream) = obj_arc.as_ref().as_stream() else {
        return unresolved("object is not a stream");
    };

    match stream.dict.get_name(b"Subtype") {
        Some(b"Image") => {}
        // A form reaches this path only when its stream could not be loaded.
        Some(b"Form") => return unresolved("form stream could not be loaded"),
        Some(_) => return ImageResolution::Absent,
        None => return unresolved("no /Subtype"),
    }

    let Some(w_raw) = stream.dict.get_i64(b"Width") else {
        return unresolved("image has no /Width");
    };
    let Some(h_raw) = stream.dict.get_i64(b"Height") else {
        return unresolved("image has no /Height");
    };
    let Some((w, h)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: degenerate dimensions {w_raw}×{h_raw}, skipping");
        return ImageResolution::DecodeFailed(format!(
            "image /{}: invalid dimensions {w_raw}×{h_raw}",
            String::from_utf8_lossy(name)
        ));
    };

    let is_mask = stream.dict.get_bool(b"ImageMask").unwrap_or(false);
    let colour = colorspace::ImageColour::resolve(doc, &stream.dict, ctx.color_spaces);
    if !is_mask && colour.paints_nothing() {
        return ImageResolution::Absent;
    }
    if let Err(msg) = decode_budget(w, h, if is_mask { 1 } else { colour.components() }) {
        return unresolved(&msg);
    }

    // The matte colour of a soft mask the image's samples are pre-blended
    // with (§11.6.5.3).
    let matte = match stream.dict.get(b"SMask") {
        Some(Object::Reference(id)) if !is_mask => smask::matte(doc, *id),
        _ => None,
    };

    // A colour-key mask compares exact samples, which a reduced decode
    // averages, and a matte is removed pixel by pixel on the mask's grid,
    // so such an image decodes in full.  A mask stream otherwise keeps its
    // own resolution and needs no such exclusion.
    let draw_size = request
        .draw_size
        .filter(|_| matte.is_none() && !matches!(stream.dict.get(b"Mask"), Some(Object::Array(_))));

    // The GPU blit composites cached images fully opaque and reads their
    // samples verbatim, so an image carrying a soft mask, an explicit mask,
    // or any colour conversion (a non-device space, or a `/Decode` remap)
    // stays host-resident: neither looked up (a content-hash hit from
    // another document would drop the mask or the conversion) nor
    // inserted.
    #[cfg(feature = "cache")]
    let dct_cache = if has_mask_entry(&stream.dict) || !colour.identity() {
        None
    } else {
        codecs::DctCacheCtx::from_resources(ctx.image_cache, ctx.doc_id, stream_id)
    };

    let src = dispatch::ImageStream {
        stream,
        width: w,
        height: h,
        is_mask,
        colour,
    };
    let (mut img, partial) = match dispatch::decode_image_stream(
        ctx,
        &src,
        draw_size,
        #[cfg(feature = "cache")]
        dct_cache.as_ref(),
        #[cfg(feature = "gpu-jpeg-huffman")]
        jpeg_gpu,
    ) {
        Ok(decoded) => (decoded.image, decoded.partial),
        Err(msg) => {
            log::warn!("image: /{}: {msg}", String::from_utf8_lossy(name));
            return ImageResolution::DecodeFailed(msg);
        }
    };

    // Resolve and decode the soft mask (`SMask`), if present.
    if let Some(Object::Reference(smask_id)) = stream.dict.get(b"SMask") {
        if let Some(mask) = smask::decode_smask(ctx, *smask_id) {
            img.smask = Some(mask);
        } else {
            // SMask decode failed.  Treat as a soft skip: the base image pixels
            // are valid so we return them without a mask rather than propagating
            // an error — an SMask failure should not discard an otherwise-correct
            // image from the page.  The renderer will blit the image fully opaque.
            log::warn!(
                "image: SMask (object {smask_id:?}) could not be decoded — blitting image without mask"
            );
        }
    } else if let Some(mask_obj) = stream.dict.get(b"Mask") {
        // PDF §8.9.6: an explicit /Mask, distinct from /SMask.  Two forms:
        //  - /Mask N 0 R           — §8.9.6.3 stencil mask image
        //  - /Mask [min max …]     — §8.9.6.4 colour-key masking, compared
        //    with the stored samples by the decode itself
        // SMask takes precedence (handled above); /Mask is only consulted when
        // no /SMask is present.  A decode failure is a soft skip — the base
        // image is still valid, so it is blitted fully opaque.
        match mask_obj {
            Object::Reference(mask_id) => {
                if let Some(mask) = smask::decode_explicit_mask(ctx, *mask_id) {
                    img.smask = Some(mask);
                } else {
                    log::warn!(
                        "image: explicit /Mask (object {mask_id:?}) could not be decoded — blitting image without mask"
                    );
                }
            }
            Object::Array(_) => {}
            _ => {
                log::warn!("image: /Mask is neither a reference nor an array — ignoring");
            }
        }
    }

    if let Some(matte) = &matte {
        smask::remove_matte(doc, src.colour.space(), matte, &mut img);
    }

    ImageResolution::Ok(DecodedImage {
        image: img,
        source_width: w,
        interpolate: stream.dict.get_bool(b"Interpolate").unwrap_or(false),
        own_soft_mask: !is_mask && matches!(stream.dict.get(b"SMask"), Some(Object::Reference(_))),
        partial,
    })
}

/// Whether an image dictionary carries a mask the blit must honour: an
/// `/SMask` stream reference, or a `/Mask` given as a stencil-mask stream
/// reference or a colour-key range array (PDF §8.9.6).
#[cfg(feature = "cache")]
fn has_mask_entry(dict: &Dictionary) -> bool {
    matches!(dict.get(b"SMask"), Some(Object::Reference(_)))
        || matches!(
            dict.get(b"Mask"),
            Some(Object::Reference(_) | Object::Array(_))
        )
}

// ── Shared helpers (used by multiple sub-modules) ─────────────────────────────

/// Largest `/Width` or `/Height` an image may declare.
pub const MAX_IMAGE_SIDE: i64 = 131_071;

/// Largest number of 8-bit samples (`width × height × components`) an
/// image is decoded to: 1 GiB.  A larger image is refused before any
/// allocation and reported as a decode failure.
pub const MAX_DECODED_BYTES: u64 = 1 << 30;

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
pub(super) const fn validated_dims(w_raw: i64, h_raw: i64) -> Option<(u32, u32)> {
    if w_raw <= 0 || h_raw <= 0 || w_raw > MAX_IMAGE_SIDE || h_raw > MAX_IMAGE_SIDE {
        return None;
    }
    Some((w_raw as u32, h_raw as u32))
}

/// Whether a `width × height` image of `components` samples per pixel fits
/// [`MAX_DECODED_BYTES`]; the message names the excess otherwise.
pub(super) fn decode_budget(width: u32, height: u32, components: usize) -> Result<(), String> {
    let bytes = u64::from(width) * u64::from(height) * components as u64;
    if bytes <= MAX_DECODED_BYTES {
        return Ok(());
    }
    Err(format!(
        "image of {width}×{height}×{components} samples ({} MiB) exceeds the {} MiB decode budget",
        bytes >> 20,
        MAX_DECODED_BYTES >> 20
    ))
}

/// A vector of `len` copies of `fill`, or `None` when the allocation
/// fails; a decoder refuses an image it cannot hold instead of aborting
/// the process.
pub(super) fn try_filled(len: usize, fill: u8) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|e| log::warn!("image: cannot allocate {len} bytes: {e}"))
        .ok()?;
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

/// GPU handles for ICC-based CMYK → RGB conversion.
#[cfg(feature = "gpu-icc")]
pub struct IccHandles<'a> {
    /// GPU context the CLUT conversion runs on; `None` converts on the CPU.
    /// The table itself is baked once per profile and document, with the
    /// profile's conversion.
    pub gpu_ctx: Option<&'a GpuCtx>,
}

#[cfg(feature = "gpu-icc")]
impl IccHandles<'_> {
    /// No GPU context: conversion runs on the CPU.
    #[must_use]
    pub const fn none() -> Self {
        Self { gpu_ctx: None }
    }
}

/// Everything an image decode reads besides the image's own bytes, apart
/// from the backend-generic GPU Huffman decoder (a separate argument).
///
/// Holds the document, the `ColorSpace` resources an inline image's `/CS`
/// name resolves through, the GPU decoders available to the image's filter
/// and the device image cache.  Each handle exists only with its feature;
/// `None` selects the CPU path or skips the cache.
pub struct ImageDecodeCtx<'a> {
    /// Document the image's indirect objects resolve in.
    pub doc: &'a Document,
    /// The `/ColorSpace` subdictionary of the resources in effect, through
    /// which an inline image's `/CS` name other than a device space resolves
    /// (PDF §8.9.7).
    pub color_spaces: Option<&'a Dictionary>,
    /// nvJPEG decoder for large `DCTDecode` images.
    #[cfg(feature = "nvjpeg")]
    pub nvjpeg: Option<&'a mut NvJpegDecoder>,
    /// VA-API JPEG queue for large `DCTDecode` images.
    #[cfg(feature = "vaapi")]
    pub vaapi: Option<&'a JpegQueueHandle>,
    /// nvJPEG2000 decoder for large `JPXDecode` images.
    #[cfg(feature = "nvjpeg2k")]
    pub nvjpeg2k: Option<&'a mut NvJpeg2kDecoder>,
    /// ICC conversion handles.
    #[cfg(feature = "gpu-icc")]
    pub icc: IccHandles<'a>,
    /// Device-resident image cache.
    #[cfg(feature = "cache")]
    pub image_cache: Option<&'a std::sync::Arc<gpu::cache::DeviceImageCache>>,
    /// Identity of the document within the cache.
    #[cfg(feature = "cache")]
    pub doc_id: Option<gpu::cache::DocId>,
}

impl<'a> ImageDecodeCtx<'a> {
    /// A context for `doc` with no colour-space resources and every GPU
    /// handle off.
    #[must_use]
    pub const fn cpu(doc: &'a Document) -> Self {
        Self {
            doc,
            color_spaces: None,
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(feature = "gpu-icc")]
            icc: IccHandles::none(),
            #[cfg(feature = "cache")]
            image_cache: None,
            #[cfg(feature = "cache")]
            doc_id: None,
        }
    }
}

/// `true` for a filter that transforms bytes without being an image codec
/// (PDF §7.4): an image ending in one decodes as raw samples after the whole
/// filter chain is applied.
pub(super) fn is_transport_filter(name: &str) -> bool {
    matches!(
        name,
        "FlateDecode" | "LZWDecode" | "ASCII85Decode" | "ASCIIHexDecode" | "RunLengthDecode"
    )
}

/// Collect every filter name from a `/Filter` entry, in application order,
/// abbreviations expanded.
///
/// Accepts a bare `Name` (one filter) or a `Name` array (a filter chain, PDF
/// §7.4: filters are listed in the order they were applied during encoding, so
/// decoding applies them left to right).  Non-name array entries are skipped.
/// An empty or absent array yields an empty vector (no filter).
pub(crate) fn image_filter_chain(obj: &Object) -> Vec<Cow<'_, str>> {
    match obj {
        Object::Name(n) => vec![full_filter_name(n)],
        Object::Array(arr) => arr
            .iter()
            .filter_map(|o| o.as_name().map(full_filter_name))
            .collect(),
        _ => Vec::new(),
    }
}

/// Extract the *terminal* image filter from a `/Filter` entry, its
/// abbreviation expanded.
///
/// Accepts a bare `Name`, a single-element `Name` array, or a multi-element
/// filter chain (e.g. `[/ASCII85Decode /CCITTFaxDecode]`).  For a chain, the
/// returned name is the LAST filter — the terminal image codec.  Earlier
/// filters in the chain are transport/prefilters (ASCII85, Flate, …) whose
/// bytes are produced by [`pdf::Stream::decompressed_content`] before the
/// terminal codec runs; see the dispatch in `resolve_image`.
///
/// An empty array (`Filter = []`) is treated as no filter (returns `None`)
/// and a debug message is emitted, matching the behaviour of an absent key.
pub(crate) fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(full_filter_name(n)),
        Object::Array(arr) => {
            if arr.is_empty() {
                log::debug!("image: Filter is an empty array — treating as no filter");
                return None;
            }
            arr.iter()
                .rev()
                .find_map(|o| o.as_name())
                .map(full_filter_name)
        }
        _ => None,
    }
}

/// The full name of the filter `name`, abbreviations expanded.
fn full_filter_name(name: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(pdf::filter_full_name(name))
}

/// Resolve the `/DecodeParms` entry that belongs to the *terminal* filter.
///
/// `/DecodeParms` is index-aligned to `/Filter` (PDF §7.4.1): entry `i` is the
/// parameters for filter `i`; a `null` entry or an absent key means "no
/// parameters".  When `/Filter` is a chain, the terminal codec
/// (`CCITTFaxDecode`, `JBIG2Decode`, …) must receive ITS entry — the last one —
/// not the whole array, and not a prefilter's entry.
///
/// Forms handled:
/// - `/DecodeParms` is a dict, `/Filter` is a single name → that dict.
/// - `/DecodeParms` is an array → the last array element (if a dict / non-null).
/// - absent / `null` / mismatched → `None`.
pub(super) fn terminal_decode_parms(dict: &Dictionary, filter_count: usize) -> Option<&Object> {
    match dict.get(b"DecodeParms")? {
        Object::Array(arr) => {
            // Prefer the entry index-aligned to the terminal (last) filter;
            // fall back to the last array element if lengths disagree.
            let idx = filter_count.checked_sub(1).filter(|&i| i < arr.len());
            let entry = idx.map_or_else(|| arr.last(), |i| arr.get(i))?;
            match entry {
                Object::Null => None,
                other => Some(other),
            }
        }
        Object::Null => None,
        other => Some(other),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
            #[cfg(feature = "gpu-jpeg-huffman")]
            None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
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

    // ── Chained-filter dispatch ────────────────────────────────────────────────

    #[test]
    fn filter_name_single_name() {
        let obj = Object::Name(b"FlateDecode".to_vec());
        assert_eq!(filter_name(&obj).as_deref(), Some("FlateDecode"));
    }

    #[test]
    fn filter_names_expand_abbreviations() {
        let obj = Object::Array(vec![
            Object::Name(b"A85".to_vec()),
            Object::Name(b"DCT".to_vec()),
        ]);
        assert_eq!(filter_name(&obj).as_deref(), Some("DCTDecode"));
        assert_eq!(image_filter_chain(&obj), ["ASCII85Decode", "DCTDecode"]);
    }

    #[test]
    fn filter_name_single_element_array() {
        let obj = Object::Array(vec![Object::Name(b"CCITTFaxDecode".to_vec())]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_chain_returns_terminal_codec() {
        // The scanned-book corpus case: [/ASCII85Decode /CCITTFaxDecode].
        // The terminal (last) filter is the image codec and must be dispatched
        // — previously this returned None and the page rendered blank.
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"CCITTFaxDecode".to_vec()),
        ]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_empty_array_is_none() {
        assert!(filter_name(&Object::Array(vec![])).is_none());
    }

    #[test]
    fn image_filter_chain_orders_left_to_right() {
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"FlateDecode".to_vec()),
        ]);
        let chain = image_filter_chain(&obj);
        assert_eq!(chain, vec!["ASCII85Decode", "FlateDecode"]);
    }

    #[test]
    fn image_filter_chain_single_name() {
        let obj = Object::Name(b"DCTDecode".to_vec());
        assert_eq!(image_filter_chain(&obj), vec!["DCTDecode"]);
    }

    #[test]
    fn terminal_decode_parms_picks_last_array_entry() {
        // [/ASCII85Decode /CCITTFaxDecode] with DecodeParms [null <</K -1>>]:
        // the CCITT codec must receive its OWN params (index 1), not null and
        // not the whole array.
        let mut ccitt = Dictionary::new();
        ccitt.set(b"K", Object::Integer(-1));
        let mut dict = Dictionary::new();
        dict.set(
            b"DecodeParms",
            Object::Array(vec![Object::Null, Object::Dictionary(ccitt)]),
        );
        let parms = terminal_decode_parms(&dict, 2).expect("terminal parms present");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"K"))
                .and_then(Object::as_i64),
            Some(-1)
        );
    }

    #[test]
    fn terminal_decode_parms_single_dict() {
        let mut ccitt = Dictionary::new();
        ccitt.set(b"Columns", Object::Integer(2480));
        let mut dict = Dictionary::new();
        dict.set(b"DecodeParms", Object::Dictionary(ccitt));
        let parms = terminal_decode_parms(&dict, 1).expect("single dict");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"Columns"))
                .and_then(Object::as_i64),
            Some(2480)
        );
    }

    #[test]
    fn terminal_decode_parms_null_and_absent_are_none() {
        let dict = Dictionary::new();
        assert!(terminal_decode_parms(&dict, 1).is_none());

        let mut with_null = Dictionary::new();
        with_null.set(
            b"DecodeParms",
            Object::Array(vec![Object::Null, Object::Null]),
        );
        assert!(terminal_decode_parms(&with_null, 2).is_none());
    }

    #[test]
    fn chained_ascii85_then_flate_roundtrip() {
        // A `[/ASCII85Decode /FlateDecode]` stream the way a PDF stores it:
        // the payload is flate-compressed, then ASCII85-encoded.  The fixture
        // bytes below are `"chained filter round-trip payload \x00\x01\x02\xff"`
        // run through zlib then Adobe ASCII85 (terminated with `~>`).
        // `decompressed_content` must undo both filters left to right and
        // recover the original payload — this is exactly the prefilter step
        // the scanned-book corpus case relies on (only the terminal codec
        // differs: CCITTFax there, treated as raw here).
        let original: &[u8] = b"chained filter round-trip payload \x00\x01\x02\xff";
        let ascii85: &[u8] = b"GaqFP8Bf:N9i4I)bUH+8;CK[@btHG9.EX2<-qLG`a\\PT-?smOA%fdE-%FY~>";

        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(vec![
                Object::Name(b"ASCII85Decode".to_vec()),
                Object::Name(b"FlateDecode".to_vec()),
            ]),
        );
        let stream = pdf::Stream::new(dict, ascii85.to_vec());
        let decoded = stream
            .decompressed_content()
            .expect("ASCII85 then Flate must round-trip");
        assert_eq!(decoded, original);
    }
}
