//! Compressed-image codec decoders: CCITT, DCT (JPEG), JPX (JPEG 2000), JBIG2.
//!
//! All functions are `pub(super)` — called exclusively from `super` (`image/mod.rs`)
//! and transitively from `inline.rs`.  Under `cfg(fuzzing)`, `decode_ccitt` and
//! `decode_jbig2` are additionally reachable through the `fuzz_entry` wrappers in
//! `image/mod.rs`, which bridge the visibility gap without widening it in normal builds.

use hayro_jbig2::Decoder as Jbig2Decoder;
use pdf::{Dictionary, Document, Object};

use super::{ImageColorSpace, ImageData, ImageDescriptor, ImageFilter};
use crate::resources::dict_ext::DictExt;

#[cfg(feature = "cache")]
use gpu::cache::{
    CachedDeviceImage, ContentHash, DeviceImageCache, DocId, ImageLayout as CacheImageLayout,
    InsertRequest, ObjId,
};
#[cfg(feature = "cache")]
use std::sync::Arc;

// ── GPU JPEG acceleration (nvJPEG and/or VA-API) ──────────────────────────────

#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::{Jpeg2kColorSpace as GpuJ2kCs, NvJpeg2kDecoder};

// ── GPU ICC CMYK→RGB acceleration ─────────────────────────────────────────────

#[cfg(feature = "gpu-icc")]
use gpu::GpuCtx;

#[cfg(feature = "gpu-icc")]
use super::icc::{self, IccClutCache};

// ── JPEG SOF peek ─────────────────────────────────────────────────────────────

/// Scan `data` for the first JPEG SOF marker and return the component count
/// (`Nf` field). Returns `None` if `data` is not a valid JPEG or the SOF
/// payload is too short. Zero allocations; stops at the first SOF found.
///
/// This avoids a full header-decode probe pass to determine the component count
/// before calling the main decode. The `gpu` crate has the same logic but is an
/// optional dependency (requires CUDA at link time), so we replicate the tiny
/// scan here for the unconditional CPU path.
fn jpeg_sof_components(data: &[u8]) -> Option<u8> {
    if data.get(0..2) != Some(&[0xFF, 0xD8]) {
        return None;
    }
    let mut pos = 2usize;
    loop {
        let ff_start = pos;
        while data.get(pos).copied() == Some(0xFF) {
            pos += 1;
        }
        if pos == ff_start {
            return None;
        }
        let &marker = data.get(pos)?;
        pos += 1;
        if matches!(marker, 0x01 | 0xD0..=0xD9) {
            continue;
        }
        let hi = *data.get(pos)?;
        let lo = *data.get(pos + 1)?;
        let seg_len = ((hi as usize) << 8) | (lo as usize);
        if seg_len < 2 {
            return None;
        }
        // SOF markers carry the component count at payload offset +5 (after P, Y, X).
        // Layout: length(2) P(1) Y(2) X(2) Nf(1) → Nf at pos+7.
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xCC) {
            return data.get(pos + 7).copied();
        }
        if marker == 0xDA {
            return None;
        }
        pos = pos.checked_add(seg_len)?;
    }
}

// ── CCITTFaxDecode ─────────────────────────────────────────────────────────────

/// Decode a `CCITTFaxDecode` stream (PDF §7.4.6).
///
/// `DecodeParms` keys honoured:
/// - `K`: `< 0` Group 4 (T.6), `0` Group 3 1-D (T.4), `> 0` Group 3 mixed
///   1-D/2-D (T.4 MR).
/// - `Columns`: pixels per encoded row; defaults to `width`.  Rows are
///   clamped or white-padded to `width` in the output.
/// - `Rows`: rows to decode; defaults to `height`.
/// - `EncodedByteAlign`: each encoded row starts on a byte boundary.  Applied
///   to Group 3 only: Group 4 streams that declare it are, in practice, not
///   row-aligned, and decoding them aligned loses all but the first rows.
/// - `EndOfLine`: EOL codes are required.  They are accepted either way.
/// - `EndOfBlock`: the stream may end with EOFB / RTC (default `true`).
/// - `BlackIs1`: 1 bits are black, so output samples are inverted.
///
/// The output is exactly `width × height` bytes, `0x00` black and `0xFF`
/// white (swapped under `BlackIs1`).  A stream that ends early or fails
/// part-way keeps its decoded rows and pads the rest white; one that yields
/// no row at all returns `None`.
pub(super) fn decode_ccitt(
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    use hayro_ccitt::{DecodeSettings, DecoderContext, EncodingMode};

    let parms_dict = parms.and_then(|o| o.as_dict());
    let flag =
        |key: &[u8], default: bool| parms_dict.and_then(|d| d.get_bool(key)).unwrap_or(default);
    let positive = |key: &[u8]| {
        parms_dict
            .and_then(|d| d.get_i64(key))
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
    };

    let k = parms_dict.and_then(|d| d.get_i64(b"K")).unwrap_or(0);
    let encoding = match k.cmp(&0) {
        std::cmp::Ordering::Less => EncodingMode::Group4,
        std::cmp::Ordering::Equal => EncodingMode::Group3_1D,
        std::cmp::Ordering::Greater => EncodingMode::Group3_2D {
            k: u32::try_from(k).ok()?,
        },
    };
    let black_is_1 = flag(b"BlackIs1", false);
    let settings = DecodeSettings {
        columns: positive(b"Columns").unwrap_or(width),
        rows: positive(b"Rows").unwrap_or(height),
        end_of_block: flag(b"EndOfBlock", true),
        end_of_line: flag(b"EndOfLine", false),
        rows_are_byte_aligned: k >= 0 && flag(b"EncodedByteAlign", false),
        encoding,
        invert_black: black_is_1,
    };

    let capacity = (width as usize).checked_mul(height as usize)?;
    // White in the output convention: 0xFF, or 0x00 when 1 bits are black.
    let white = if black_is_1 { 0x00 } else { 0xFF };
    let mut collector = HayroCcittCollector::new(capacity, width, white);
    let mut ctx = DecoderContext::new(settings);

    if let Err(e) = hayro_ccitt::decode(data, &mut collector, &mut ctx) {
        if collector.rows_decoded() == 0 {
            log::warn!("image: CCITTFaxDecode ({encoding:?}) failed: {e}");
            return None;
        }
        log::warn!(
            "image: CCITTFaxDecode ({encoding:?}) error after {}/{height} rows: {e}",
            collector.rows_decoded()
        );
    }

    let rows_decoded = collector.rows_decoded();
    let mut data_out = collector.finish();
    if data_out.len() > capacity {
        log::debug!(
            "image: CCITTFaxDecode: output too long ({} > {capacity}), truncating",
            data_out.len()
        );
        data_out.truncate(capacity);
    } else if data_out.len() < capacity {
        log::debug!("image: CCITTFaxDecode: got {rows_decoded}/{height} rows, padding remainder");
        data_out.resize(capacity, white);
    }

    let color_space = if is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width,
        height,
        color_space,
        data: ImageData::Cpu(data_out),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// [`hayro_ccitt::Decoder`] that collects pixel runs into one byte per pixel
/// (`0x00` black, `0xFF` white, as the decoder reports them).
///
/// Every row is exactly `width` bytes: runs past the end are clamped, and a
/// short row is filled with `pad`.
struct HayroCcittCollector {
    out: Vec<u8>,
    width: u32,
    pad: u8,
    col: u32,
    rows: u32,
}

impl HayroCcittCollector {
    fn new(capacity: usize, width: u32, pad: u8) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            width,
            pad,
            col: 0,
            rows: 0,
        }
    }

    const fn rows_decoded(&self) -> u32 {
        self.rows
    }

    /// The collected bytes, with a partial final row padded to `width`.
    fn finish(mut self) -> Vec<u8> {
        if self.col > 0 {
            self.fill_row();
        }
        self.out
    }

    fn fill_row(&mut self) {
        // `col <= width` holds: `push_pixels` clamps every run.
        let remaining = (self.width - self.col) as usize;
        self.out.extend(std::iter::repeat_n(self.pad, remaining));
        self.col = self.width;
    }
}

impl hayro_ccitt::Decoder for HayroCcittCollector {
    fn push_pixels(&mut self, white: bool, count: u32) {
        let n = count.min(self.width - self.col);
        let byte = if white { 0xFF } else { 0x00 };
        self.out.extend(std::iter::repeat_n(byte, n as usize));
        self.col += n;
    }

    fn next_line(&mut self) {
        self.fill_row();
        self.col = 0;
        self.rows += 1;
    }
}

// ── DCTDecode (JPEG) ──────────────────────────────────────────────────────────

/// Decode a `DCTDecode` (JPEG) stream.
///
/// `pdf_w` and `pdf_h` from the PDF stream dict are used only for a size sanity
/// check; the actual dimensions come from the JPEG SOF marker and are
/// authoritative.
///
/// # CMYK handling
///
/// JPEG CMYK images in PDF store ink densities in *inverted* form: a byte value
/// of 0 means *full ink*, 255 means *no ink*.  The JPEG decoder returns raw
/// bytes in this inverted form.  Converting to RGB:
///
/// ```text
/// R = (255 - C_raw) * (255 - K_raw) / 255
/// ```
///
/// where `C_raw` and `K_raw` are the raw (inverted) byte values.  The `255 -`
/// term converts each inverted value to a normal ink density before applying
/// the standard CMY+K → RGB formula.
#[expect(
    clippy::too_many_lines,
    reason = "GPU dispatch paths (nvjpeg, vaapi) plus the CPU decode path are each short; \
              combining them in one function keeps the fallback logic visible"
)]
#[cfg_attr(
    all(
        feature = "nvjpeg",
        feature = "vaapi",
        feature = "gpu-icc",
        feature = "cache"
    ),
    expect(
        clippy::too_many_arguments,
        reason = "decoder/cache handles are feature-gated cfg-args; bundling them \
                  into a struct would force every caller to construct a partially-\
                  populated context per call"
    )
)]
pub(super) fn decode_dct<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dict: &Dictionary,
    #[cfg(feature = "nvjpeg")] gpu: Option<&mut NvJpegDecoder>,
    #[cfg(feature = "vaapi")] vaapi: Option<&JpegQueueHandle>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<
        &mut gpu::jpeg_decoder::JpegGpuDecoder<B>,
    >,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
    #[cfg(feature = "cache")] cache_ctx: Option<&DctCacheCtx<'_>>,
) -> Option<ImageDescriptor> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace as ZColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    // Device-image-cache fast path: if a cache is wired in and the encoded
    // bytes are already cached (cross-document content-hash dedup),
    // return the device-resident handle with zero CPU decode work.
    // On miss, retain the precomputed BLAKE3 hash so the post-decode
    // insert at the bottom of this function doesn't re-hash the same
    // bytes — that would double the per-image hashing cost on cold
    // misses (~250 µs per 500 KB JPEG → ~250 ms per 1000-image render).
    #[cfg(feature = "cache")]
    let cache_miss_hash: Option<ContentHash> = match cache_ctx {
        Some(ctx) => match try_decode_dct_cached_lookup(data, pdf_w, pdf_h, ctx) {
            Ok(img) => return Some(img),
            Err(hash) => Some(hash),
        },
        None => None,
    };

    // `/Decode` remaps the decoded samples per colour-space component.  The
    // device decoders return final samples, so a non-identity remap keeps
    // the image on the CPU path, where it is applied before colour conversion.
    let decode = jpeg_sof_components(data)
        .map_or_else(Vec::new, |n| super::raw::parse_decode(dict, usize::from(n)));
    #[cfg(any(feature = "nvjpeg", feature = "vaapi", feature = "gpu-jpeg-huffman"))]
    let device_eligible = super::raw::is_identity_decode(&decode);

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    let area = pdf_w.saturating_mul(pdf_h);

    // Peek at the JPEG SOF marker to route to the correct decoder.
    // Prevents sending progressive JPEG to VA-API (baseline-only).
    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    let jpeg_variant = gpu::jpeg_sof_type(data);

    // ── nvJPEG fast path (baseline + progressive) ─────────────────────────────
    // The `>= GPU_JPEG_THRESHOLD_PX` comparison can be `false` for any input
    // when the threshold is `u32::MAX` (current consumer-Blackwell setting —
    // see the constant's docs). The expect is on the per-call site; the
    // threshold itself is not a constant truth.
    #[cfg(feature = "nvjpeg")]
    if device_eligible && let Some(dec) = gpu {
        let is_eligible = matches!(
            jpeg_variant,
            Some(gpu::JpegVariant::Baseline | gpu::JpegVariant::Progressive)
        );
        #[expect(
            clippy::absurd_extreme_comparisons,
            reason = "GPU_JPEG_THRESHOLD_PX is u32::MAX on consumer Blackwell to disable nvJPEG dispatch; tunable per hardware"
        )]
        let area_above_threshold = area >= super::GPU_JPEG_THRESHOLD_PX;
        if is_eligible
            && area_above_threshold
            && let Some(img) = decode_dct_gpu_path(data, pdf_w, pdf_h, dec)
        {
            return Some(img);
        }
    }

    // ── VA-API fast path (baseline only, via decode queue) ───────────────────
    // Progressive JPEG is skipped — VA-API VAEntrypointVLD supports baseline DCT only.
    // Jobs are routed through a dedicated worker thread (JpegQueueHandle) that owns
    // the VapiJpegDecoder, eliminating Mesa driver contention across Rayon workers.
    #[cfg(feature = "vaapi")]
    if device_eligible
        && let Some(handle) = vaapi
        && matches!(jpeg_variant, Some(gpu::JpegVariant::Baseline))
        && {
            #[expect(
                clippy::absurd_extreme_comparisons,
                reason = "see GPU_JPEG_THRESHOLD_PX docs — set to u32::MAX on consumer Blackwell"
            )]
            let above = area >= super::GPU_JPEG_THRESHOLD_PX;
            above
        }
        && let Some(img) = decode_dct_queue_path(data, pdf_w, pdf_h, handle)
    {
        return Some(img);
    }

    // ── GPU parallel-Huffman path ─────────────────────────────────────────────
    #[cfg(feature = "gpu-jpeg-huffman")]
    if device_eligible && let Some(dec) = jpeg_gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= super::huffman_threshold() {
            match dec.decode(data) {
                Ok(img) => {
                    let w = img.width;
                    let h = img.height;
                    if w == 0 || h == 0 {
                        log::warn!(
                            "GPU JPEG decoder returned zero-dimension image ({w}×{h}), skipping"
                        );
                    } else {
                        let n = (w as usize) * (h as usize) * 4;
                        let mut rgba = vec![0u8; n];
                        let downloaded = dec
                            .backend()
                            .download_async(&img.buffer, &mut rgba)
                            .and_then(|fence| dec.backend().wait_download(fence));
                        match downloaded {
                            Ok(()) => {
                                // The IDCT kernel always writes RGBA8. For
                                // grayscale (1-component) JPEGs it replicates
                                // luma into R, G, and B. Strip alpha to get the
                                // 3-byte-per-pixel RGB representation expected by
                                // the blit path, and preserve the correct
                                // color_space tag so downstream code branches
                                // correctly.
                                let (color_space, pixel_data) =
                                    if jpeg_sof_components(data) == Some(1) {
                                        let gray: Vec<u8> =
                                            rgba.chunks_exact(4).map(|px| px[0]).collect();
                                        (super::ImageColorSpace::Gray, gray)
                                    } else {
                                        let rgb: Vec<u8> = rgba
                                            .chunks_exact(4)
                                            .flat_map(|px| [px[0], px[1], px[2]])
                                            .collect();
                                        (super::ImageColorSpace::Rgb, rgb)
                                    };
                                return Some(ImageDescriptor {
                                    width: w,
                                    height: h,
                                    color_space,
                                    data: super::ImageData::Cpu(pixel_data),
                                    smask: None,
                                    filter: ImageFilter::Dct,
                                });
                            }
                            Err(e) => log::warn!("GPU JPEG download failed: {e}"),
                        }
                    }
                }
                Err(e) => log::debug!("GPU JPEG decode failed, falling back to CPU: {e}"),
            }
        }
    }

    // ── CPU path ──────────────────────────────────────────────────────────────
    let components = match jpeg_sof_components(data) {
        Some(n @ (1 | 3 | 4)) => n,
        Some(n) => {
            log::warn!("image: DCTDecode: unexpected component count {n}");
            return None;
        }
        None => {
            log::warn!("image: DCTDecode: not a valid JPEG stream");
            return None;
        }
    };

    // Choose the output colorspace to request from the decoder.
    let out_cs = match components {
        1 => ZColorSpace::Luma,
        3 => ZColorSpace::RGB,
        // CMYK — request raw CMYK output (4 bytes/pixel), convert to RGB below.
        _ => ZColorSpace::CMYK,
    };

    let options = DecoderOptions::default().jpeg_set_out_colorspace(out_cs);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);

    // An Adobe APP14 transform of 2 marks a four-component stream as YCCK.
    // The decoder has no YCCK → CMYK mapping, so request YCCK unchanged and
    // convert it below.
    let ycck = components == 4 && {
        decoder
            .decode_headers()
            .map_err(|e| log::warn!("image: DCTDecode header error: {e}"))
            .ok()?;
        decoder.input_colorspace() == Some(ZColorSpace::YCCK)
    };
    if ycck {
        decoder.set_options(options.jpeg_set_out_colorspace(ZColorSpace::YCCK));
    }

    let mut pixels = decoder
        .decode()
        .map_err(|e| log::warn!("image: DCTDecode decode error: {e}"))
        .ok()?;
    if ycck {
        ycck_to_adobe_cmyk(&mut pixels);
    }
    let pixels = super::raw::apply_decode(std::borrow::Cow::Owned(pixels), &decode);

    let jpeg_info = decoder.info().or_else(|| {
        log::warn!("image: DCTDecode: JPEG info unavailable after decode");
        None
    })?;
    let jw = u32::from(jpeg_info.width);
    let jh = u32::from(jpeg_info.height);

    if jw == 0 || jh == 0 {
        log::warn!("image: DCTDecode: JPEG reported zero dimensions {jw}×{jh}");
        return None;
    }

    // Validate the pixel buffer length against the dimensions we are about to use.
    // A mismatch means the JPEG SOF markers misreported dimensions; trust jw/jh
    // only when the buffer is consistent with them.
    let expected_pixels = (jw as usize)
        .checked_mul(jh as usize)
        .and_then(|n| n.checked_mul(usize::from(components)));
    if expected_pixels.is_none_or(|e| pixels.len() < e) {
        log::warn!(
            "image: DCTDecode: pixel buffer length {} inconsistent with \
             {jw}×{jh}×{components} — skipping",
            pixels.len()
        );
        return None;
    }

    if jw != pdf_w || jh != pdf_h {
        log::debug!(
            "image: DCTDecode: PDF dict says {pdf_w}×{pdf_h}, JPEG reports {jw}×{jh} — using JPEG dims"
        );
    }

    let cpu_desc = match out_cs {
        ZColorSpace::Luma => ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Gray,
            data: ImageData::Cpu(pixels),
            smask: None,
            filter: ImageFilter::Dct,
        },
        ZColorSpace::RGB => ImageDescriptor {
            width: jw,
            height: jh,
            color_space: ImageColorSpace::Rgb,
            data: ImageData::Cpu(pixels),
            smask: None,
            filter: ImageFilter::Dct,
        },
        ZColorSpace::CMYK => {
            // The samples are ink densities as the image dictionary
            // describes them.  An Adobe JPEG that stores inverted CMYK
            // carries `/Decode [1 0 1 0 1 0 1 0]`, applied above, so no
            // inversion is implied by the JPEG itself.  The JPEG's own
            // colour profile is not read and the PDF `ICCBased` stream is
            // not available here, so no CLUT is baked.
            let rgb = cmyk_raw_to_rgb(
                &pixels,
                false,
                #[cfg(feature = "gpu-icc")]
                gpu_ctx,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                clut_cache,
            )?;
            ImageDescriptor {
                width: jw,
                height: jh,
                color_space: ImageColorSpace::Rgb,
                data: ImageData::Cpu(rgb),
                smask: None,
                filter: ImageFilter::Dct,
            }
        }
        // out_cs is always Luma, RGB, or CMYK — set from the components match above.
        _ => unreachable!("DCTDecode: unexpected out_cs variant"),
    };

    // Insert into the cache so the next render of this PDF
    // (or a different PDF with the same image content) hits the
    // device-resident fast path instead of decoding again.  Reuse
    // the hash from the lookup-miss path to skip a second BLAKE3.
    #[cfg(feature = "cache")]
    if let (Some(ctx), Some(hash)) = (cache_ctx, cache_miss_hash) {
        return Some(try_decode_dct_cached_insert(cpu_desc, hash, ctx));
    }
    Some(cpu_desc)
}

/// Device-image-cache plumbing for `decode_dct`.  When set, [`decode_dct`]
/// (a) probes the cache by content hash before any decode work and,
/// on a miss, (b) inserts the decoded bytes after the CPU decode
/// completes so the next render hits the cache.
#[cfg(feature = "cache")]
pub(super) struct DctCacheCtx<'a> {
    /// Device-resident image cache.  Borrowed; the cache outlives the
    /// renderer for the duration of a `raster_pdf` call.
    pub cache: &'a Arc<DeviceImageCache>,
    /// Stable identifier for the source PDF.
    pub doc_id: DocId,
    /// Per-image PDF object id (the `XObject`'s stream object number).
    pub obj_id: ObjId,
}

#[cfg(feature = "cache")]
impl<'a> DctCacheCtx<'a> {
    /// Build the cache context from the renderer's cache reference,
    /// the document id, and the PDF object id of the image stream.
    ///
    /// Returns `None` when either the cache is absent or the `doc_id`
    /// is absent — in that case `decode_dct` skips both the lookup
    /// fast path and the post-decode insert.
    ///
    /// The `(u32, u16)` PDF `ObjectId` is mapped to `u32` by dropping
    /// the generation.  See [`gpu::cache::ObjId`] for the lossy-
    /// mapping rationale.
    pub(super) fn from_resources(
        cache: Option<&'a Arc<DeviceImageCache>>,
        doc_id: Option<DocId>,
        stream_id: pdf::ObjectId,
    ) -> Option<Self> {
        let cache = cache?;
        let doc_id = doc_id?;
        Some(Self {
            cache,
            doc_id,
            obj_id: ObjId(stream_id.0),
        })
    }
}

/// Probe the cache.  On a hit returns the device-resident
/// `ImageDescriptor`; on a miss returns `Err(ContentHash)` so the
/// caller can pass the already-computed hash to
/// [`try_decode_dct_cached_insert`] and avoid re-hashing the same
/// bytes after CPU decode.
#[cfg(feature = "cache")]
fn try_decode_dct_cached_lookup(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    ctx: &DctCacheCtx<'_>,
) -> Result<ImageDescriptor, ContentHash> {
    // Same-document fast path: alias by (DocId, ObjId) skips BLAKE3 hashing.
    if let Some(cached) = ctx.cache.lookup_by_id(ctx.doc_id, ctx.obj_id) {
        log::debug!(
            "decode_dct: cache hit by alias for ({:?}, {:?})",
            ctx.doc_id,
            ctx.obj_id,
        );
        return Ok(image_descriptor_from_cached(cached, pdf_w, pdf_h));
    }
    // Three-tier probe: VRAM → host RAM → disk (when enabled).
    // Re-binds the alias on hit so the same image referenced from a
    // later page goes through the alias fast path.
    let hash = DeviceImageCache::hash_bytes(data);
    if let Some(cached) = ctx
        .cache
        .lookup_by_hash_for_doc(ctx.doc_id, ctx.obj_id, &hash)
    {
        log::debug!("decode_dct: cache hit by content hash; alias rebound");
        return Ok(image_descriptor_from_cached(cached, pdf_w, pdf_h));
    }
    Err(hash)
}

/// Wrap a freshly-decoded JPEG into the cache, returning a
/// device-resident `ImageDescriptor`.  Skipped (returns the
/// caller's CPU descriptor) on insert error so the renderer
/// still gets pixels — typically a budget violation; not fatal.
///
/// `hash` should be the value returned from
/// [`try_decode_dct_cached_lookup`]'s `Err` arm; reusing it avoids
/// hashing the same bytes a second time.
#[cfg(feature = "cache")]
fn try_decode_dct_cached_insert(
    cpu_desc: ImageDescriptor,
    hash: ContentHash,
    ctx: &DctCacheCtx<'_>,
) -> ImageDescriptor {
    // Only cache RGB / Gray output (the layouts the blit kernel
    // can sample).  Mask images are rare in JPEG and stay on CPU.
    let layout = match cpu_desc.color_space {
        ImageColorSpace::Rgb => CacheImageLayout::Rgb,
        ImageColorSpace::Gray => CacheImageLayout::Gray,
        ImageColorSpace::Mask => return cpu_desc,
    };
    let Some(pixels) = cpu_desc.data.as_cpu() else {
        return cpu_desc;
    };
    let req = InsertRequest {
        doc: ctx.doc_id,
        obj: ctx.obj_id,
        hash,
        width: cpu_desc.width,
        height: cpu_desc.height,
        layout,
        pixels,
    };
    match ctx.cache.insert(req) {
        Ok(cached) => image_descriptor_from_cached(cached, cpu_desc.width, cpu_desc.height),
        Err(e) => {
            log::warn!(
                "decode_dct: cache insert failed ({e}); returning CPU descriptor — image will use CPU blit path"
            );
            cpu_desc
        }
    }
}

#[cfg(feature = "cache")]
fn image_descriptor_from_cached(
    cached: Arc<CachedDeviceImage>,
    pdf_w: u32,
    pdf_h: u32,
) -> ImageDescriptor {
    let color_space = match cached.layout {
        CacheImageLayout::Rgb => ImageColorSpace::Rgb,
        CacheImageLayout::Gray => ImageColorSpace::Gray,
        CacheImageLayout::Mask => ImageColorSpace::Mask,
    };
    // Prefer the cached entry's actual dimensions over the PDF
    // dict's claim — the dict can lie; the decoded image can't.
    let width = cached.width;
    let height = cached.height;
    if (width, height) != (pdf_w, pdf_h) {
        log::debug!(
            "decode_dct: cache hit dimensions {width}×{height} differ from PDF dict {pdf_w}×{pdf_h}; using cache",
        );
    }
    ImageDescriptor {
        width,
        height,
        color_space,
        data: ImageData::Gpu(cached),
        smask: None,
        filter: ImageFilter::Dct,
    }
}

/// Attempt JPEG decode via any [`gpu::GpuJpegDecoder`] implementation.
///
/// Returns `None` if the decoder rejects the image (unsupported encoding,
/// component count, or any transient decode error). The caller must then
/// fall through to the CPU path.
#[cfg(feature = "nvjpeg")]
fn decode_dct_gpu_path<D: gpu::GpuJpegDecoder>(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dec: &mut D,
) -> Option<ImageDescriptor> {
    use gpu::DecodedImage;

    let decoded: DecodedImage = dec
        .decode_jpeg(data, pdf_w, pdf_h)
        .map_err(|e| log::warn!("image: DCTDecode: GPU path failed: {e}"))
        .ok()?;

    let color_space = match decoded.components {
        1 => ImageColorSpace::Gray,
        3 => ImageColorSpace::Rgb,
        n => {
            log::warn!(
                "image: DCTDecode: GPU decoder returned unexpected component count {n}; falling back to CPU"
            );
            return None;
        }
    };

    Some(ImageDescriptor {
        width: decoded.width,
        height: decoded.height,
        color_space,
        data: ImageData::Cpu(decoded.data),
        filter: ImageFilter::Raw,
        smask: None,
    })
}

/// Attempt JPEG decode via a [`gpu::JpegQueueHandle`] (VA-API worker thread).
///
/// Mirrors [`decode_dct_gpu_path`] but routes through the dedicated OS worker
/// thread rather than calling the decoder directly on the Rayon thread.  Returns
/// `None` if the worker is unavailable or the decoder rejects the image; the
/// caller must then fall through to the CPU path.
#[cfg(feature = "vaapi")]
fn decode_dct_queue_path(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    handle: &gpu::JpegQueueHandle,
) -> Option<ImageDescriptor> {
    use gpu::DecodedImage;

    // Arc<[u8]> is required to send the JPEG bytes to the worker thread without
    // unsafe lifetime extension.  One memcpy of the JPEG payload; negligible vs
    // VCN hardware decode time (1–5 ms/frame for ≥512×512 images).
    let arc_data: std::sync::Arc<[u8]> = std::sync::Arc::from(data);
    let decoded: DecodedImage = handle.decode(arc_data, pdf_w, pdf_h)?;

    let color_space = match decoded.components {
        1 => ImageColorSpace::Gray,
        3 => ImageColorSpace::Rgb,
        n => {
            log::warn!(
                "image: DCTDecode: VA-API queue returned unexpected component count {n}; falling back to CPU"
            );
            return None;
        }
    };

    Some(ImageDescriptor {
        width: decoded.width,
        height: decoded.height,
        color_space,
        data: ImageData::Cpu(decoded.data),
        filter: ImageFilter::Raw,
        smask: None,
    })
}

/// Rewrite decoded YCCK samples, in place, as the CMYK samples the stream
/// encodes: `(Y, Cb, Cr, K)` → `(255 − R, 255 − G, 255 − B, K)`, where RGB is
/// the full-range Y′CbCr conversion of the first three channels.  The result
/// is in the same convention as a JPEG stored as CMYK and takes the same
/// path.  A trailing partial pixel is left unchanged.
fn ycck_to_adobe_cmyk(pixels: &mut [u8]) {
    for px in pixels.as_chunks_mut::<4>().0 {
        let (r, g, b) = color::convert::ycbcr_to_rgb(px[0], px[1], px[2]);
        px[0] = 255 - r;
        px[1] = 255 - g;
        px[2] = 255 - b;
    }
}

/// Convert a raw CMYK pixel buffer to RGB, dispatching to GPU when available.
///
/// `inverted = false` (raw images / `decode_raw_8bpp`): 0 = no ink, 255 = full ink.
/// `inverted = true` (JPEG CMYK): codec returns the complement convention (0 = full
/// ink, 255 = no ink); the complement is applied on-the-fly, avoiding a separate
/// allocation.
///
/// `icc_bytes` — raw ICC profile bytes extracted from an `ICCBased` colour space.
/// When provided (and the `gpu-icc` feature is active), a CLUT is baked from the
/// profile and used for the colour transform instead of the fast matrix approximation.
///
/// Returns `None` when the pixel buffer length is not a multiple of 4, or on
/// arithmetic overflow (degenerate image size).
#[expect(
    clippy::many_single_char_names,
    reason = "c/m/y/k/r/g/b are the canonical channel names for CMYK→RGB — renaming hurts clarity"
)]
pub(super) fn cmyk_raw_to_rgb(
    pixels: &[u8],
    inverted: bool,
    #[cfg(feature = "gpu-icc")] gpu_ctx: Option<&GpuCtx>,
    #[cfg(feature = "gpu-icc")] icc_bytes: Option<&[u8]>,
    #[cfg(feature = "gpu-icc")] clut_cache: Option<&mut IccClutCache>,
) -> Option<Vec<u8>> {
    if !pixels.len().is_multiple_of(4) {
        log::warn!(
            "image: cmyk_raw_to_rgb: pixel buffer length {} is not a multiple of 4",
            pixels.len()
        );
        return None;
    }

    // GPU path: delegate to GpuCtx which handles the dispatch-threshold check and
    // CPU fallback internally.  When ICC bytes are present, bake a CLUT for
    // profile-accurate conversion; fall back to the fast matrix approximation if
    // baking fails (e.g. corrupt profile or wrong colour space).
    #[cfg(feature = "gpu-icc")]
    if let Some(ctx) = gpu_ctx {
        // Complement JPEG CMYK pixels before GPU dispatch if needed.
        // A Cow avoids the allocation when `inverted = false` (the common case).
        let gpu_pixels: std::borrow::Cow<[u8]> = if inverted {
            pixels.iter().map(|&b| 255 - b).collect::<Vec<_>>().into()
        } else {
            pixels.into()
        };

        #[expect(
            clippy::option_if_let_else,
            reason = "branches are not symmetric (cached path omits .map(Into::into)); \
                      map_or_else would make this harder to read"
        )]
        let clut_arc: Option<std::sync::Arc<[u8]>> = icc_bytes.and_then(|bytes| {
            if let Some(cache) = clut_cache {
                icc::bake_cmyk_clut_cached(bytes, icc::DEFAULT_GRID_N, cache)
                    .map_err(|e| {
                        log::warn!("image: ICC CLUT bake failed, using matrix fallback: {e}");
                    })
                    .ok()
            } else {
                icc::bake_cmyk_clut(bytes, icc::DEFAULT_GRID_N)
                    .map_err(|e| {
                        log::warn!("image: ICC CLUT bake failed, using matrix fallback: {e}");
                    })
                    .ok()
                    .map(std::convert::Into::into)
            }
        });

        match ctx.icc_cmyk_to_rgb(
            &gpu_pixels,
            clut_arc.as_deref().map(|t| (t, icc::DEFAULT_GRID_N)),
        ) {
            Ok(rgb) => return Some(rgb),
            Err(e) => log::warn!("image: GPU CMYK→RGB failed, falling back to CPU: {e}"),
        }
    }

    // CPU path (also used below threshold by GpuCtx itself, but we land here
    // when the feature is disabled or no GpuCtx is provided).
    let npixels = pixels.len() / 4;
    let mut rgb = Vec::with_capacity(npixels.checked_mul(3)?);
    for chunk in pixels.as_chunks::<4>().0 {
        let [c, m, y, k] = if inverted {
            chunk.map(|v| 255 - v)
        } else {
            *chunk
        };
        let (r, g, b) = color::convert::cmyk_to_rgb_reflectance(c, m, y, k);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    Some(rgb)
}

// ── JPXDecode (JPEG 2000) ─────────────────────────────────────────────────────

/// Decode a `JPXDecode` (JPEG 2000) stream.
///
/// When the `nvjpeg2k` feature is active and `gpu` is `Some`, large images
/// (pixel area ≥ [`super::GPU_JPEG2K_THRESHOLD_PX`]) are decoded on the GPU via
/// nvJPEG2000.  All other images, and any image for which the GPU path fails
/// (unsupported component count, CUDA error, etc.), fall through to the CPU path.
///
/// PDF JPEG 2000 streams may be raw codestreams (`.j2k`) or full JP2 container
/// format (`.jp2`).  Both paths auto-detect the format from the stream.
///
/// Samples of any precision are scaled to 8 bits; sub-sampled components are
/// upsampled to the image grid.  An embedded alpha channel becomes
/// `ImageDescriptor::smask`.
pub(super) fn decode_jpx(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    draw_size: Option<(u32, u32)>,
    #[cfg(feature = "nvjpeg2k")] gpu: Option<&mut NvJpeg2kDecoder>,
) -> Option<ImageDescriptor> {
    // ── GPU fast path (nvjpeg2k feature, large images, 1- or 3-component only) ─
    #[cfg(feature = "nvjpeg2k")]
    if let Some(dec) = gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= super::GPU_JPEG2K_THRESHOLD_PX {
            if let Some(img) = decode_jpx_gpu(data, pdf_w, pdf_h, dec) {
                return Some(img);
            }
            log::debug!("image: JPXDecode: GPU path failed, retrying on CPU");
        }
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    // Skip the wavelet levels the drawn size cannot show: the decoder keeps
    // every level that leaves the image at least `draw_size` on both axes.
    let settings = hayro_jpeg2000::DecodeSettings {
        target_resolution: draw_size,
        ..Default::default()
    };
    let img = hayro_jpeg2000::Image::new(data, &settings)
        .map_err(|e| log::warn!("image: JPXDecode open error: {e}"))
        .ok()?;
    let (jw, jh) = (img.width(), img.height());
    if jw == 0 || jh == 0 {
        log::warn!("image: JPXDecode: JP2 reported zero dimensions {jw}×{jh}");
        return None;
    }
    if draw_size.is_none() && (jw != pdf_w || jh != pdf_h) {
        log::debug!(
            "image: JPXDecode: PDF dict says {pdf_w}×{pdf_h}, JP2 reports {jw}×{jh} — using JP2 dims"
        );
    }

    let channels = usize::from(img.color_space().num_channels());
    let has_alpha = img.has_alpha();
    let mut ctx = hayro_jpeg2000::DecoderContext::default();
    let samples = img
        .decode_u8(&mut ctx)
        .map_err(|e| log::warn!("image: JPXDecode decode error: {e}"))
        .ok()?;

    let npx = (jw as usize).checked_mul(jh as usize)?;
    let stride = channels + usize::from(has_alpha);
    if Some(samples.len()) != npx.checked_mul(stride) {
        log::warn!(
            "image: JPXDecode: {} samples for {jw}×{jh}×{stride}",
            samples.len()
        );
        return None;
    }

    // The alpha channel, when present, is the last sample of each pixel.
    let (colour, alpha) = if has_alpha {
        let colour = samples
            .chunks_exact(stride)
            .flat_map(|px| px[..channels].iter().copied())
            .collect();
        let alpha = samples
            .chunks_exact(stride)
            .map(|px| px[channels])
            .collect();
        (colour, Some(alpha))
    } else {
        (samples, None)
    };

    // An ICC-tagged or unrecognised colour space is read by its channel
    // count; the embedded profile is not applied.  CMYK samples are ink
    // densities (0 = no ink).
    let mut desc = match channels {
        1 => jpx_gray(jw, jh, colour),
        3 => jpx_rgb(jw, jh, colour),
        4 => {
            let rgb = cmyk_raw_to_rgb(
                &colour,
                false,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
                #[cfg(feature = "gpu-icc")]
                None,
            )?;
            jpx_rgb(jw, jh, rgb)
        }
        n => {
            log::warn!("image: JPXDecode: unsupported {n}-channel colour space");
            return None;
        }
    };
    desc.smask = alpha;
    Some(desc)
}

/// GPU-accelerated JPEG 2000 decode via nvJPEG2000.
///
/// Returns `None` for unsupported component counts (CMYK, LAB, N-channel) or
/// when any CUDA API call fails; the caller falls back to the CPU path in
/// both cases.
#[cfg(feature = "nvjpeg2k")]
fn decode_jpx_gpu(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    dec: &mut NvJpeg2kDecoder,
) -> Option<ImageDescriptor> {
    let img = match dec.decode_sync(data) {
        Ok(img) => img,
        // Unsupported component count (CMYK, Gray+Alpha, N-channel), sub-sampled
        // chroma, or a codestream type not supported by this nvJPEG2000 build
        // (HTJ2K, status 4) — all are expected; fall through to CPU silently.
        Err(
            gpu::nvjpeg2k::NvJpeg2kError::UnsupportedComponents(_)
            | gpu::nvjpeg2k::NvJpeg2kError::SubSampledComponents
            | gpu::nvjpeg2k::NvJpeg2kError::Nvjpeg2kStatus(4),
        ) => return None,
        Err(e) => {
            log::warn!("image: JPXDecode GPU: nvJPEG2000 error: {e}");
            return None;
        }
    };

    if img.width != pdf_w || img.height != pdf_h {
        log::debug!(
            "image: JPXDecode GPU: PDF dict says {pdf_w}×{pdf_h}, nvJPEG2000 reports {}×{} — using nvJPEG2000 dims",
            img.width,
            img.height,
        );
    }

    let color_space = match img.color_space {
        GpuJ2kCs::Gray => ImageColorSpace::Gray,
        GpuJ2kCs::Rgb => ImageColorSpace::Rgb,
    };

    Some(ImageDescriptor {
        width: img.width,
        height: img.height,
        color_space,
        data: ImageData::Cpu(img.data),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Wrap a decoded grayscale pixel buffer into an [`ImageDescriptor`].
#[inline]
pub(super) const fn jpx_gray(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Gray,
        data: ImageData::Cpu(data),
        smask: None,
        filter: ImageFilter::Raw,
    }
}

/// Wrap a decoded RGB pixel buffer into an [`ImageDescriptor`].
#[inline]
pub(super) const fn jpx_rgb(width: u32, height: u32, data: Vec<u8>) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space: ImageColorSpace::Rgb,
        data: ImageData::Cpu(data),
        smask: None,
        filter: ImageFilter::Raw,
    }
}

// ── JBIG2Decode ──────────────────────────────────────────────────────────────

/// Decode a `JBIG2Decode` stream via `hayro-jbig2`.
///
/// PDF JBIG2 uses the "embedded organisation" (Annex D.3 of T.88): the stream
/// contains page segments only; global segments live in a separate
/// `JBIG2Globals` stream referenced from `DecodeParms`.
///
/// The decoder produces one byte per pixel: 0x00 = black, 0xFF = white (for
/// grayscale images) or 0x00 = paint, 0xFF = transparent (for `ImageMask`).
/// JBIG2 convention is 0 = white, 1 = black; we invert to match the rest of
/// the image pipeline.
pub(super) fn decode_jbig2(
    doc: &Document,
    data: &[u8],
    width: u32,
    height: u32,
    is_mask: bool,
    parms: Option<&Object>,
) -> Option<ImageDescriptor> {
    // Resolve optional JBIG2Globals stream from DecodeParms.
    let globals_bytes: Option<Vec<u8>> = parms
        .and_then(|o| o.as_dict())
        .and_then(|d| d.get(b"JBIG2Globals"))
        .and_then(|o| {
            if let Object::Reference(id) = o {
                let g_obj = doc.get_object(*id).ok()?;
                let g_stream = g_obj.as_ref().as_stream()?;
                // JBIG2Globals streams are typically not compressed, but
                // decompressed_content handles both cases transparently.
                g_stream.decompressed_content().ok()
            } else {
                // PDF spec requires JBIG2Globals to be an indirect reference to
                // a stream object; any other form is malformed.
                log::warn!(
                    "image: JBIG2Decode: JBIG2Globals is not an indirect reference — \
                     ignoring globals"
                );
                None
            }
        });

    // Parse the embedded JBIG2 image (page segments + optional globals).
    let img = hayro_jbig2::Image::new_embedded(data, globals_bytes.as_deref())
        .map_err(|e| log::warn!("image: JBIG2Decode parse error: {e}"))
        .ok()?;

    let jw = img.width();
    let jh = img.height();

    // Validate decoded dimensions against the PDF image dict.
    if jw != width || jh != height {
        log::warn!(
            "image: JBIG2Decode dimension mismatch: image dict says {width}×{height}, JBIG2 stream says {jw}×{jh} — using stream dimensions"
        );
    }

    let n_pixels = (jw as usize).checked_mul(jh as usize)?;
    let mut collector = Jbig2Collector {
        data: Vec::with_capacity(n_pixels),
        n_pixels,
    };

    img.decode(&mut collector)
        .map_err(|e| log::warn!("image: JBIG2Decode decode error: {e}"))
        .ok()?;

    if collector.data.len() != n_pixels {
        log::warn!(
            "image: JBIG2Decode produced {} pixels, expected {n_pixels} — skipping",
            collector.data.len()
        );
        return None;
    }

    let cs = if is_mask {
        ImageColorSpace::Mask
    } else {
        ImageColorSpace::Gray
    };
    Some(ImageDescriptor {
        width: jw,
        height: jh,
        color_space: cs,
        data: ImageData::Cpu(collector.data),
        smask: None,
        filter: ImageFilter::Raw,
    })
}

/// Pixel collector for `hayro_jbig2::Decoder`.
///
/// JBIG2 convention: 0 = white, 1 = black.
/// `ImageColorSpace::Gray` convention: 0x00 = black, 0xFF = white.
/// `ImageColorSpace::Mask` convention: 0x00 = paint (== JBIG2 black), 0xFF = transparent.
///
/// Both output conventions share the same polarity flip: JBIG2 black (1) → 0x00,
/// JBIG2 white (0) → 0xFF.  `ImageColorSpace` selection is handled by the caller.
struct Jbig2Collector {
    data: Vec<u8>,
    /// Total expected pixel count — used to cap `push_pixel_chunk` so a stream
    /// whose run lengths aren't a multiple of 8 doesn't overflow the buffer.
    n_pixels: usize,
}

impl Jbig2Decoder for Jbig2Collector {
    fn push_pixel(&mut self, black: bool) {
        if self.data.len() < self.n_pixels {
            self.data.push(if black { 0x00 } else { 0xFF });
        }
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        let byte = if black { 0x00 } else { 0xFF };
        // chunk_count is in 8-pixel chunks; cap to the remaining pixels so that
        // run lengths that aren't an exact multiple of 8 don't produce a short
        // buffer (which would cause the caller to discard the image entirely).
        let remaining = self.n_pixels.saturating_sub(self.data.len());
        let n = (chunk_count as usize).saturating_mul(8).min(remaining);
        self.data.extend(std::iter::repeat_n(byte, n));
    }

    fn next_line(&mut self) {
        // Row boundary — nothing to do; pixels are already stored flat.
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack a string of `'0'`/`'1'` into bytes, MSB first, zero-padded.
    fn pack_bits(bits: &str) -> Vec<u8> {
        let bits: Vec<u8> = bits.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        bits.chunks(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u8, |acc, (i, &b)| acc | (u8::from(b == b'1') << (7 - i)))
            })
            .collect()
    }

    /// `DecodeParms` with the given integer and boolean entries.
    fn ccitt_parms(ints: &[(&[u8], i64)], flags: &[(&[u8], bool)]) -> Object {
        let mut d = Dictionary::new();
        for &(k, v) in ints {
            d.set(k, Object::Integer(v));
        }
        for &(k, v) in flags {
            d.set(k, Object::Boolean(v));
        }
        Object::Dictionary(d)
    }

    fn ccitt_bytes(stream: &str, w: u32, h: u32, parms: &Object) -> Vec<u8> {
        let img = decode_ccitt(&pack_bits(stream), w, h, false, Some(parms)).expect("decodes");
        img.data.as_cpu().expect("host-resident").to_vec()
    }

    /// Two rows of `white 2, black 4, white 2` (8 px).
    const TWO_ROWS: [u8; 16] = [
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, //
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,
    ];
    /// T.4 one-dimensional code for one `white 2, black 4, white 2` row.
    const ROW_1D: &str = "0111 011 0111";
    const EOL: &str = "000000000001";

    #[test]
    fn ccitt_group4_decodes_horizontal_and_vertical_modes() {
        // Row 0 against the imaginary white line: H(white 2, black 4), V0.
        // Row 1 against row 0: V0 three times.  Then EOFB.
        let stream = format!("001 0111 011 1   1 1 1   {EOL} {EOL}");
        let parms = ccitt_parms(&[(b"K", -1)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_group4_ignores_encoded_byte_align() {
        // The unaligned stream above, declared byte-aligned.
        let stream = format!("001 0111 011 1   1 1 1   {EOL} {EOL}");
        let parms = ccitt_parms(&[(b"K", -1)], &[(b"EncodedByteAlign", true)]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_group3_2d_decodes_one_dimensional_rows() {
        // K = 1: each row is EOL + tag bit 1 (1-D coded), then RTC.
        let stream = format!(
            "{EOL} 1 {ROW_1D} {EOL} 1 {ROW_1D} {}",
            format!("{EOL} 1 ").repeat(6)
        );
        let parms = ccitt_parms(&[(b"K", 1)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_encoded_byte_align_starts_each_row_on_a_byte() {
        // Group 3 1-D without EOLs: each 11-bit row is zero-filled to 16 bits.
        let stream = format!("{ROW_1D} 00000 {ROW_1D} 00000");
        let parms = ccitt_parms(&[(b"K", 0)], &[(b"EncodedByteAlign", true)]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);

        // Read unaligned, the fill bits are decoded as codes.
        let unaligned = ccitt_parms(&[(b"K", 0)], &[]);
        let misread = decode_ccitt(&pack_bits(&stream), 8, 2, false, Some(&unaligned))
            .and_then(|img| img.data.as_cpu().map(<[u8]>::to_vec));
        assert_ne!(misread.as_deref(), Some(&TWO_ROWS[..]));
    }

    #[test]
    fn ccitt_rows_narrower_than_the_image_are_padded_white() {
        // `Columns 6`: rows of white 2, black 4, padded to the 8-px width.
        let row = "0111 011";
        let stream = format!("{EOL} {row} {EOL} {row} {}", EOL.repeat(6));
        let parms = ccitt_parms(&[(b"K", 0), (b"Columns", 6)], &[]);
        assert_eq!(ccitt_bytes(&stream, 8, 2, &parms), TWO_ROWS);
    }

    #[test]
    fn ccitt_black_is_1_inverts_samples_and_padding() {
        // One decoded row of two; the missing row pads with white, which is
        // 0x00 once 1 bits mean black.
        let stream = format!("{EOL} {ROW_1D} {}", EOL.repeat(6));
        let parms = ccitt_parms(&[(b"K", 0)], &[(b"BlackIs1", true)]);
        let got = ccitt_bytes(&stream, 8, 2, &parms);
        assert_eq!(got[..8], [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]);
        assert_eq!(got[8..], [0x00; 8]);
    }

    #[test]
    fn ccitt_collector_clamps_a_run_that_overruns_the_row() {
        let mut c = HayroCcittCollector::new(8, 4, 0xFF);
        hayro_ccitt::Decoder::push_pixels(&mut c, false, 3);
        hayro_ccitt::Decoder::push_pixels(&mut c, true, 20);
        hayro_ccitt::Decoder::next_line(&mut c);
        assert_eq!(c.rows_decoded(), 1);
        assert_eq!(c.finish(), [0x00, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_grayscale() {
        // JBIG2: black=true → Gray 0x00, black=false → Gray 0xFF.
        let mut c = Jbig2Collector {
            data: Vec::new(),
            n_pixels: 2,
        };
        Jbig2Decoder::push_pixel(&mut c, true);
        Jbig2Decoder::push_pixel(&mut c, false);
        assert_eq!(c.data, [0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk() {
        // Pre-allocate 16 pixels, matching the `n_pixels` pattern used in production.
        let mut c = Jbig2Collector {
            data: Vec::with_capacity(16),
            n_pixels: 16,
        };
        // chunk_count=2 → 16 pixels of white (0xFF each).
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 16);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk_overrun() {
        // Regression test: chunk whose byte-count overshoots n_pixels by 7
        // (common when page dimensions aren't a multiple of 8).
        // Before the fix, this produced a short buffer and the image was dropped.
        let mut c = Jbig2Collector {
            data: Vec::with_capacity(10),
            n_pixels: 10,
        };
        // chunk_count=2 → 16 pixels, but n_pixels=10 — must cap at 10.
        Jbig2Decoder::push_pixel_chunk(&mut c, false, 2);
        assert_eq!(c.data.len(), 10);
        assert!(c.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn jbig2_decode_invalid_data_returns_none() {
        // Corrupt/empty JBIG2 data must not panic — it must return None.
        let doc = crate::test_helpers::empty_doc();
        let result = decode_jbig2(&doc, b"\x00\x01\x02\x03", 4, 4, false, None);
        assert!(result.is_none());
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    fn minimal_baseline_jpeg() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x05, // length = 5
            0x08, // precision
            0x00, 0x01, // height = 1
            0xFF, 0xD9, // EOI
        ]
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    fn minimal_progressive_jpeg() -> Vec<u8> {
        vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC2, // SOF2
            0x00, 0x05, // length = 5
            0x08, // precision
            0x00, 0x01, // height = 1
            0xFF, 0xD9, // EOI
        ]
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    #[test]
    fn jpeg_variant_baseline_detected() {
        let data = minimal_baseline_jpeg();
        assert_eq!(gpu::jpeg_sof_type(&data), Some(gpu::JpegVariant::Baseline));
    }

    #[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
    #[test]
    fn jpeg_variant_progressive_detected() {
        let data = minimal_progressive_jpeg();
        assert_eq!(
            gpu::jpeg_sof_type(&data),
            Some(gpu::JpegVariant::Progressive)
        );
    }

    // ── decode_dct_gpu_path ───────────────────────────────────────────────────

    #[cfg(feature = "nvjpeg")]
    mod gpu_path_tests {
        use super::*;

        struct AlwaysGray;

        impl gpu::GpuJpegDecoder for AlwaysGray {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![128u8; 4],
                    width: 2,
                    height: 2,
                    components: 1,
                })
            }
        }

        struct AlwaysFail;

        impl gpu::GpuJpegDecoder for AlwaysFail {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Err(gpu::GpuDecodeError::new(std::io::Error::other(
                    "simulated failure",
                )))
            }
        }

        struct AlwaysRgb;

        impl gpu::GpuJpegDecoder for AlwaysRgb {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![255u8; 12],
                    width: 2,
                    height: 2,
                    components: 3,
                })
            }
        }

        struct AlwaysCmyk;

        impl gpu::GpuJpegDecoder for AlwaysCmyk {
            fn decode_jpeg(
                &mut self,
                _data: &[u8],
                _width: u32,
                _height: u32,
            ) -> Result<gpu::DecodedImage, gpu::GpuDecodeError> {
                Ok(gpu::DecodedImage {
                    data: vec![0u8; 16],
                    width: 2,
                    height: 2,
                    components: 4,
                })
            }
        }

        #[test]
        fn generic_gpu_path_gray() {
            let mut dec = AlwaysGray;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            let img = result.expect("should succeed");
            assert_eq!(img.color_space, ImageColorSpace::Gray);
            assert_eq!(img.width, 2);
            assert_eq!(img.height, 2);
            assert_eq!(img.data.as_cpu().unwrap(), &vec![128u8; 4]);
        }

        #[test]
        fn generic_gpu_path_failure_returns_none() {
            let mut dec = AlwaysFail;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            assert!(result.is_none());
        }

        #[test]
        fn generic_gpu_path_rgb() {
            let mut dec = AlwaysRgb;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            let img = result.expect("should succeed");
            assert_eq!(img.color_space, ImageColorSpace::Rgb);
            assert_eq!(img.data.len(), 12);
        }

        #[test]
        fn generic_gpu_path_rejects_cmyk() {
            let mut dec = AlwaysCmyk;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            assert!(result.is_none());
        }
    }

    // ── jpx_gray / jpx_rgb helpers ────────────────────────────────────────────

    /// Decode a JPEG 2000 fixture (8 × 8) through `decode_jpx`.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn jpx_fixture(bytes: &[u8]) -> ImageDescriptor {
        decode_jpx(bytes, 8, 8, None).expect("decodes")
    }

    // Fixtures are lossless (reversible 5/3) encodes; the `.gray`, `.rgb`,
    // `.rgba` and `.gray16` files are the same images decoded by an
    // independent decoder.

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_draw_size_skips_levels_without_going_below_it() {
        let gray = include_bytes!("testdata/gray8.jp2");
        let full = decode_jpx(gray, 8, 8, None).expect("decodes");
        assert_eq!((full.width, full.height), (8, 8));

        let half = decode_jpx(gray, 8, 8, Some((4, 4))).expect("decodes");
        assert_eq!((half.width, half.height), (4, 4));
        assert_eq!(half.data.as_cpu().map(<[u8]>::len), Some(16));

        // 4 × 4 would be smaller than the drawn size, so no level is skipped.
        let five = decode_jpx(gray, 8, 8, Some((5, 5))).expect("decodes");
        assert_eq!((five.width, five.height), (8, 8));
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_gray_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("testdata/gray8.jp2"));
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("testdata/gray8.gray")[..])
        );
        assert!(img.smask.is_none());
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_raw_codestream_rgb_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("testdata/rgb8.j2k"));
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("testdata/rgb8.rgb")[..])
        );
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_alpha_channel_becomes_the_soft_mask() {
        let img = jpx_fixture(include_bytes!("testdata/rgba8.jp2"));
        let rgba = include_bytes!("testdata/rgba8.rgba");
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let alpha: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[3]).collect();
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu(), Some(&rgb[..]));
        assert_eq!(img.smask, Some(alpha));
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_16_bit_samples_scale_to_8_bit() {
        let img = jpx_fixture(include_bytes!("testdata/gray16.jp2"));
        let reference = include_bytes!("testdata/gray16.gray16");
        let got = img.data.as_cpu().expect("host-resident");
        assert_eq!(got.len(), 64);
        for (i, (&g, w)) in got.iter().zip(reference.as_chunks::<2>().0).enumerate() {
            let want = u16::from_be_bytes(*w) >> 8;
            assert!(
                u16::from(g).abs_diff(want) <= 1,
                "sample {i}: got {g}, 16-bit reference {}",
                u16::from_be_bytes(*w)
            );
        }
    }

    #[test]
    fn jpx_gray_smask_none_by_default() {
        let desc = jpx_gray(2, 2, vec![0u8; 4]);
        assert!(desc.smask.is_none(), "jpx_gray must not inject a smask");
    }

    #[test]
    fn jpx_rgb_smask_none_by_default() {
        let desc = jpx_rgb(2, 2, vec![0u8; 12]);
        assert!(desc.smask.is_none(), "jpx_rgb must not inject a smask");
    }

    // ── JPX-CMYK convention ───────────────────────────────────────────────────
    //
    // JPEG 2000 stores CMYK as direct ink density (0 = no ink), so
    // `decode_jpx` converts it with `inverted = false`, as for `DeviceCMYK`
    // raw images.  This pins that the two interpretations differ, so a
    // flipped flag would invert every CMYK JPX image.
    #[test]
    fn jpx_cmyk_uses_direct_ink_convention_not_jpeg_inverted() {
        // Pure-cyan ink at full density: direct convention (0 = no ink).
        let cyan_direct = [255u8, 0, 0, 0];

        let direct = cmyk_raw_to_rgb(
            &cyan_direct,
            false,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("direct CMYK→RGB");
        let inverted = cmyk_raw_to_rgb(
            &cyan_direct,
            true,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
            #[cfg(feature = "gpu-icc")]
            None,
        )
        .expect("inverted CMYK→RGB");

        // Full cyan ink absorbs red → R must be the smallest channel.
        let [r, g, b] = direct[..3] else {
            unreachable!("3 bytes per pixel")
        };
        assert!(
            r < g && r < b,
            "full cyan ink (direct convention) must darken red: got rgb=({r},{g},{b})"
        );

        // The inverted (JPEG) reading of the same bytes is a *different* colour;
        // if it were not, the convention choice would not matter and a flipped
        // flag would silently regress.
        assert_ne!(
            &direct[..3],
            &inverted[..3],
            "direct and JPEG-inverted CMYK readings must differ — \
             this is why decode_jpx_cmyk must pass inverted = false"
        );
    }
}
