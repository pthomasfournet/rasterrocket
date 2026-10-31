//! Compressed-image codec decoders: CCITT, DCT (JPEG), JPX (JPEG 2000), JBIG2.
//!
//! All functions are `pub(super)` — called exclusively from `super` (`image/mod.rs`)
//! and transitively from `inline.rs`.  Under `cfg(fuzzing)`, `decode_ccitt` and
//! `decode_jbig2` are additionally reachable through the `fuzz_entry` wrappers in
//! `image/mod.rs`, which bridge the visibility gap without widening it in normal builds.

use std::sync::Arc;

use hayro_jbig2::Decoder as Jbig2Decoder;
use pdf::{Document, Object};

#[cfg(feature = "cache")]
use super::ImageData;
use super::colorspace::{Layout, Samples};
use crate::resources::dict_ext::DictExt;

#[cfg(feature = "cache")]
use gpu::cache::{
    CachedDeviceImage, ContentHash, DeviceImageCache, DocId, ImageLayout as CacheImageLayout,
    InsertRequest, ObjId,
};

// ── GPU JPEG acceleration (nvJPEG and/or VA-API) ──────────────────────────────

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::{Jpeg2kColorSpace as GpuJ2kCs, NvJpeg2kDecoder};

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
/// - `EncodedByteAlign`: each encoded row starts on a byte boundary.  Group 3
///   streams honour it.  Group 4 streams that declare it occur both aligned
///   and unaligned, so a Group 4 stream is read unaligned and, if that stops
///   short, re-read aligned, keeping the reading that decodes more rows.
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
) -> Option<Samples> {
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
        std::cmp::Ordering::Greater => EncodingMode::Group3_2D,
    };
    let black_is_1 = flag(b"BlackIs1", false);
    let declared_aligned = flag(b"EncodedByteAlign", false);
    let rows = positive(b"Rows").unwrap_or(height);
    let columns = positive(b"Columns").unwrap_or(width);

    let capacity = (width as usize).checked_mul(height as usize)?;
    // White in the output convention: 0xFF, or 0x00 when 1 bits are black.
    let white = if black_is_1 { 0x00 } else { 0xFF };
    let run = |aligned: bool| {
        let mut settings = DecodeSettings::new(columns, rows);
        settings.end_of_block = flag(b"EndOfBlock", true);
        settings.end_of_line = flag(b"EndOfLine", false);
        settings.rows_are_byte_aligned = aligned;
        settings.encoding = encoding;
        let mut collector = HayroCcittCollector::new(capacity, width, columns, white)?;
        let mut ctx = DecoderContext::new(settings);
        let result = hayro_ccitt::decode(data, &mut collector, &mut ctx);
        Some((collector, result))
    };

    // Group 4 streams that declare `EncodedByteAlign` exist both aligned and
    // unaligned.  The unaligned reading is tried first; when it stops short,
    // the aligned reading is tried and whichever decodes more rows is kept.
    let (mut collector, mut result) = run(k >= 0 && declared_aligned)?;
    if k < 0 && declared_aligned && result.is_err() && collector.rows_decoded() < rows {
        let (aligned, aligned_result) = run(true)?;
        if aligned.rows_decoded() > collector.rows_decoded() {
            collector = aligned;
            result = aligned_result;
        }
    }

    if let Err(e) = result {
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

    let layout = if is_mask {
        Layout::Mask
    } else {
        Layout::Color(1)
    };
    let mut samples = Samples::cpu(width, height, layout, data_out);
    if rows_decoded < height {
        samples.partial = Some(format!(
            "CCITTFaxDecode: {rows_decoded} of {height} rows decoded; the rest are white"
        ));
    }
    Some(samples)
}

/// [`hayro_ccitt::Decoder`] that expands each row of colour changes into one
/// byte per pixel: `white` for white runs and its complement for black runs.
///
/// Every row is exactly `width` bytes.  Decoded rows span `columns` pixels:
/// a narrower row is padded with `white`, a wider one is clipped.
struct HayroCcittCollector {
    out: Vec<u8>,
    width: u32,
    /// The pixel at which an unterminated black run ends: `min(columns, width)`.
    row_end: u32,
    white: u8,
    rows: u32,
}

impl HayroCcittCollector {
    /// `None` when `capacity` bytes cannot be allocated.
    fn new(capacity: usize, width: u32, columns: u32, white: u8) -> Option<Self> {
        Some(Self {
            out: super::try_with_capacity(capacity)?,
            width,
            row_end: columns.min(width),
            white,
            rows: 0,
        })
    }

    const fn rows_decoded(&self) -> u32 {
        self.rows
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

impl hayro_ccitt::Decoder for HayroCcittCollector {
    fn push_row(&mut self, changes: &[u32]) {
        let start = self.out.len();
        self.out.resize(start + self.width as usize, self.white);
        let row = &mut self.out[start..];
        // Changes alternate white→black and black→white; an odd count leaves
        // the row black to its end.
        for run in changes.chunks(2) {
            let from = run[0].min(self.row_end) as usize;
            let to = run.get(1).map_or(self.row_end, |&x| x.min(self.row_end)) as usize;
            row[from..to].fill(!self.white);
        }
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
/// A four-component stream yields CMYK samples as the dictionary describes
/// them: an Adobe JPEG that stores inverted ink carries `/Decode [1 0 1 0 1
/// 0 1 0]`, which the colour conversion applies, so no inversion is implied
/// by the JPEG itself.  A YCCK stream is converted to CMYK here.
#[expect(
    clippy::too_many_lines,
    reason = "GPU dispatch paths (nvjpeg, vaapi) plus the CPU decode path are each short; \
              combining them in one function keeps the fallback logic visible"
)]
#[cfg_attr(
    not(feature = "nvjpeg"),
    expect(
        clippy::needless_pass_by_ref_mut,
        reason = "only the nvjpeg handle is borrowed mutably"
    )
)]
pub(super) fn decode_dct<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    #[cfg_attr(
        not(any(feature = "nvjpeg", feature = "vaapi")),
        expect(
            unused_variables,
            reason = "every handle in the context is feature-gated"
        )
    )]
    ctx: &mut super::ImageDecodeCtx<'_>,
    #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<
        &mut gpu::jpeg_decoder::JpegGpuDecoder<B>,
    >,
    #[cfg(feature = "cache")] cache_ctx: Option<&DctCacheCtx<'_>>,
) -> Option<Samples> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace as ZColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    #[cfg(feature = "nvjpeg")]
    let gpu = ctx.nvjpeg.as_deref_mut();
    #[cfg(feature = "vaapi")]
    let vaapi = ctx.vaapi;

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
    if let Some(dec) = gpu {
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
    if let Some(handle) = vaapi
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
    if let Some(dec) = jpeg_gpu {
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
                                return Some(if jpeg_sof_components(data) == Some(1) {
                                    let gray: Vec<u8> =
                                        rgba.as_chunks::<4>().0.iter().map(|px| px[0]).collect();
                                    Samples::gray(w, h, gray)
                                } else {
                                    let rgb: Vec<u8> = rgba
                                        .as_chunks::<4>()
                                        .0
                                        .iter()
                                        .flat_map(|&[r, g, b, _]| [r, g, b])
                                        .collect();
                                    Samples::cpu(w, h, Layout::Color(3), rgb)
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

    let cpu = Samples::cpu(jw, jh, Layout::Color(components), pixels);

    // Insert into the cache so the next render of this PDF
    // (or a different PDF with the same image content) hits the
    // device-resident fast path instead of decoding again.  Reuse
    // the hash from the lookup-miss path to skip a second BLAKE3.
    #[cfg(feature = "cache")]
    if let (Some(ctx), Some(hash)) = (cache_ctx, cache_miss_hash) {
        return Some(try_decode_dct_cached_insert(cpu, hash, ctx));
    }
    Some(cpu)
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
) -> Result<Samples, ContentHash> {
    // Same-document fast path: alias by (DocId, ObjId) skips BLAKE3 hashing.
    if let Some(cached) = ctx.cache.lookup_by_id(ctx.doc_id, ctx.obj_id) {
        log::debug!(
            "decode_dct: cache hit by alias for ({:?}, {:?})",
            ctx.doc_id,
            ctx.obj_id,
        );
        return Ok(samples_from_cached(cached, pdf_w, pdf_h));
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
        return Ok(samples_from_cached(cached, pdf_w, pdf_h));
    }
    Err(hash)
}

/// Wrap a freshly-decoded JPEG into the cache, returning
/// device-resident samples.  Skipped (returns the caller's CPU
/// samples) on insert error so the renderer still gets pixels —
/// typically a budget violation; not fatal.
///
/// `hash` should be the value returned from
/// [`try_decode_dct_cached_lookup`]'s `Err` arm; reusing it avoids
/// hashing the same bytes a second time.
#[cfg(feature = "cache")]
fn try_decode_dct_cached_insert(cpu: Samples, hash: ContentHash, ctx: &DctCacheCtx<'_>) -> Samples {
    // Only cache RGB / Gray samples (the layouts the blit kernel can
    // sample); CMYK samples await conversion and stay on the CPU.
    let layout = match cpu.layout {
        Layout::Color(3) => CacheImageLayout::Rgb,
        Layout::Color(1) => CacheImageLayout::Gray,
        _ => return cpu,
    };
    let Some(pixels) = cpu.data.as_cpu() else {
        return cpu;
    };
    let req = InsertRequest {
        doc: ctx.doc_id,
        obj: ctx.obj_id,
        hash,
        width: cpu.width,
        height: cpu.height,
        layout,
        pixels,
    };
    match ctx.cache.insert(req) {
        Ok(cached) => samples_from_cached(cached, cpu.width, cpu.height),
        Err(e) => {
            log::warn!(
                "decode_dct: cache insert failed ({e}); returning CPU samples — image will use CPU blit path"
            );
            cpu
        }
    }
}

#[cfg(feature = "cache")]
fn samples_from_cached(cached: Arc<CachedDeviceImage>, pdf_w: u32, pdf_h: u32) -> Samples {
    let layout = match cached.layout {
        CacheImageLayout::Rgb => Layout::Color(3),
        CacheImageLayout::Gray => Layout::Color(1),
        CacheImageLayout::Mask => Layout::Mask,
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
    Samples {
        width,
        height,
        layout,
        data: ImageData::Gpu(cached),
        smask: None,
        embedded_profile: None,
        partial: None,
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
) -> Option<Samples> {
    use gpu::DecodedImage;

    let decoded: DecodedImage = dec
        .decode_jpeg(data, pdf_w, pdf_h)
        .map_err(|e| log::warn!("image: DCTDecode: GPU path failed: {e}"))
        .ok()?;
    device_samples(decoded, "GPU decoder")
}

/// The samples of a device decoder's output, when it holds one or three
/// components; `what` names the decoder in the log otherwise.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
fn device_samples(decoded: gpu::DecodedImage, what: &str) -> Option<Samples> {
    let layout = match decoded.components {
        n @ (1 | 3) => Layout::Color(n),
        n => {
            log::warn!(
                "image: DCTDecode: {what} returned unexpected component count {n}; falling back to CPU"
            );
            return None;
        }
    };
    Some(Samples::cpu(
        decoded.width,
        decoded.height,
        layout,
        decoded.data,
    ))
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
) -> Option<Samples> {
    use gpu::DecodedImage;

    // Arc<[u8]> is required to send the JPEG bytes to the worker thread without
    // unsafe lifetime extension.  One memcpy of the JPEG payload; negligible vs
    // VCN hardware decode time (1–5 ms/frame for ≥512×512 images).
    let arc_data: std::sync::Arc<[u8]> = std::sync::Arc::from(data);
    let decoded: DecodedImage = handle.decode(arc_data, pdf_w, pdf_h)?;
    device_samples(decoded, "VA-API queue")
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
/// upsampled to the image grid.  `opts` carries what the image dictionary
/// decides about the codestream: see [`JpxOptions`].  An ICC profile the
/// codestream's colour specification carries is returned with the samples
/// (`Samples::embedded_profile`) for an image dictionary that names no
/// colour space.  The error names what the decoder refused.
pub(super) fn decode_jpx(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    draw_size: Option<(u32, u32)>,
    opts: JpxOptions,
    #[cfg(feature = "nvjpeg2k")] gpu: Option<&mut NvJpeg2kDecoder>,
) -> Result<Samples, String> {
    if opts.indices {
        return decode_jpx_indices(data);
    }

    // The headers: the codestream's colour specification, and the decoder
    // of the CPU path, which skips the wavelet levels the drawn size cannot
    // show (it keeps every level that leaves the image at least `draw_size`
    // on both axes).
    let settings = hayro_jpeg2000::DecodeSettings {
        target_resolution: draw_size,
        ..Default::default()
    };
    let img = hayro_jpeg2000::Image::new(data, &settings).map_err(|e| e.to_string())?;
    jpx_budget(&img)?;
    let embedded_profile = match img.color_space() {
        hayro_jpeg2000::ColorSpace::Icc { profile, .. } => Some(profile.clone()),
        _ => None,
    };

    // ── GPU fast path (nvjpeg2k feature, large images, 1- or 3-component only) ─
    // nvJPEG2000 rejects codestreams with an alpha channel, so the GPU result
    // never needs `/SMaskInData`.
    #[cfg(feature = "nvjpeg2k")]
    if let Some(dec) = gpu {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= super::GPU_JPEG2K_THRESHOLD_PX {
            if let Some(mut samples) = decode_jpx_gpu(data, pdf_w, pdf_h, dec) {
                samples.embedded_profile = embedded_profile;
                return Ok(samples);
            }
            log::debug!("image: JPXDecode: GPU path failed, retrying on CPU");
        }
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    let (jw, jh) = (img.width(), img.height());
    if jw == 0 || jh == 0 {
        return Err(format!("codestream reports zero dimensions {jw}×{jh}"));
    }
    if draw_size.is_none() && (jw != pdf_w || jh != pdf_h) {
        log::debug!(
            "image: JPXDecode: PDF dict says {pdf_w}×{pdf_h}, JP2 reports {jw}×{jh} — using JP2 dims"
        );
    }

    let channels = usize::from(img.color_space().num_channels());
    let has_alpha = img.has_alpha();
    let mut ctx = hayro_jpeg2000::DecoderContext::default();
    let samples = img.decode_u8(&mut ctx).map_err(|e| e.to_string())?;

    let stride = channels + usize::from(has_alpha);
    let expected = (jw as usize)
        .checked_mul(jh as usize)
        .and_then(|npx| npx.checked_mul(stride));
    if expected != Some(samples.len()) {
        return Err(format!("{} samples for {jw}×{jh}×{stride}", samples.len()));
    }

    // The alpha channel, when present, is the last sample of each pixel.
    // It is the image's soft mask only when `/SMaskInData` is nonzero
    // (§8.9.5); with 2 the colour channels are premultiplied by it, which
    // is the matte relation with every matte component 0, inverted in the
    // codestream's own colour space.
    let (colour, alpha) = if has_alpha {
        let mut colour: Vec<u8> = samples
            .chunks_exact(stride)
            .flat_map(|px| px[..channels].iter().copied())
            .collect();
        let alpha = (opts.smask_in_data != 0).then(|| {
            let alpha: Vec<u8> = samples
                .chunks_exact(stride)
                .map(|px| px[channels])
                .collect();
            if opts.smask_in_data == 2 {
                super::smask::unmatte(&mut colour, &alpha, &vec![0; channels]);
            }
            alpha
        });
        (colour, alpha)
    } else {
        (samples, None)
    };

    // The samples are read by channel count; CMYK samples are ink
    // densities (0 = no ink).
    let layout = match channels {
        1 => Layout::Color(1),
        3 => Layout::Color(3),
        4 => Layout::Color(4),
        n => return Err(format!("unsupported {n}-channel colour space")),
    };
    let mut samples = Samples::cpu(jw, jh, layout, colour);
    samples.smask = alpha.and_then(|a| super::SoftMask::new(jw, jh, a));
    samples.embedded_profile = embedded_profile;
    Ok(samples)
}

/// Decode a JPEG 2000 codestream's first component as palette indices
/// for an `/Indexed` PDF colour space: the samples unscaled, with any
/// palette in the JP2 container left unapplied, at full resolution (a
/// reduced decode would average indices).  The result is one index byte
/// per pixel.
fn decode_jpx_indices(data: &[u8]) -> Result<Samples, String> {
    let settings = hayro_jpeg2000::DecodeSettings {
        resolve_palette_indices: false,
        ..Default::default()
    };
    let img = hayro_jpeg2000::Image::new(data, &settings).map_err(|e| e.to_string())?;
    jpx_budget(&img)?;
    let (jw, jh) = (img.width(), img.height());
    let mut ctx = hayro_jpeg2000::DecoderContext::default();
    let decoded = img.decode(&mut ctx).map_err(|e| e.to_string())?;
    let samples = decoded
        .components()
        .first()
        .ok_or_else(|| "codestream has no component".to_owned())?
        .samples();
    let npx = (jw as usize).checked_mul(jh as usize);
    if jw == 0 || jh == 0 || npx != Some(samples.len()) {
        return Err(format!("{} index samples for {jw}×{jh}", samples.len()));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=255 before the cast"
    )]
    let indices = samples
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    Ok(Samples::gray(jw, jh, indices))
}

/// The decode budget ([`super::decode_budget`]) applied to the
/// dimensions and channel count a JPEG 2000 codestream declares, which
/// govern the decoder's allocations and may differ from the image
/// dictionary's.
fn jpx_budget(img: &hayro_jpeg2000::Image<'_>) -> Result<(), String> {
    let channels = usize::from(img.color_space().num_channels()) + usize::from(img.has_alpha());
    super::decode_budget(img.width(), img.height(), channels).map_err(|e| format!("codestream {e}"))
}

/// What the image dictionary decides about a JPEG 2000 codestream.
#[derive(Debug, Clone, Copy)]
pub(super) struct JpxOptions {
    /// The PDF colour space is `/Indexed`: decode the codestream's samples
    /// as palette indices for the caller to expand.
    pub indices: bool,
    /// `/SMaskInData` (§8.9.5): 0 ignores an alpha channel, 1 makes it the
    /// soft mask, 2 makes it the soft mask of colour premultiplied by it.
    pub smask_in_data: i64,
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
) -> Option<Samples> {
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

    let layout = match img.color_space {
        GpuJ2kCs::Gray => Layout::Color(1),
        GpuJ2kCs::Rgb => Layout::Color(3),
    };
    Some(Samples::cpu(img.width, img.height, layout, img.data))
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
) -> Option<Samples> {
    // Resolve the optional JBIG2Globals stream from DecodeParms.  Its
    // segments are decoded once per document and shared by every image
    // that refers to the stream.
    let settings = hayro_jbig2::DecodeSettings::default();
    let globals: Option<Arc<Option<hayro_jbig2::Globals>>> = match parms
        .and_then(|o| o.as_dict())
        .and_then(|d| d.get(b"JBIG2Globals"))
    {
        None => None,
        // `Some(None)` is an unreadable globals stream, which the image is
        // decoded without; `None` is a globals stream that fails to parse,
        // which fails the image.
        Some(&Object::Reference(id)) => Some(doc.derived(id, || {
            let Some(g_bytes) = doc
                .get_object(id)
                .ok()
                .and_then(|g| g.as_stream()?.decompressed_content().ok())
            else {
                return Some(None);
            };
            hayro_jbig2::Globals::new(&g_bytes, &settings)
                .map_err(|e| log::warn!("image: JBIG2Decode globals parse error: {e}"))
                .ok()
                .map(Some)
        })?),
        Some(_) => {
            // PDF spec requires JBIG2Globals to be an indirect reference to
            // a stream object; any other form is malformed.
            log::warn!(
                "image: JBIG2Decode: JBIG2Globals is not an indirect reference — \
                 ignoring globals"
            );
            None
        }
    };
    let img = hayro_jbig2::Page::new_embedded(
        data,
        globals.as_deref().and_then(Option::as_ref),
        &settings,
    )
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

    // The page size of the stream, not the dictionary's, sizes the
    // decoder's bitmap and the collected samples.
    if let Err(e) = super::decode_budget(jw, jh, 1) {
        log::warn!("image: JBIG2Decode page {e}");
        return None;
    }
    let n_pixels = (jw as usize).checked_mul(jh as usize)?;
    let mut collector = Jbig2Collector::new(n_pixels);

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

    let layout = if is_mask {
        Layout::Mask
    } else {
        Layout::Color(1)
    };
    Some(Samples::cpu(jw, jh, layout, collector.data))
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
    /// The page buffer could not be reserved; every pixel is dropped, so
    /// the collected length falls short and the image is refused.
    alloc_failed: bool,
}

impl Jbig2Collector {
    const fn new(n_pixels: usize) -> Self {
        Self {
            data: Vec::new(),
            n_pixels,
            alloc_failed: false,
        }
    }

    /// Reserves the whole page on the first emitted pixel, and reports
    /// whether the page buffer is held.  The decoder emits only after it
    /// has allocated the page bitmap, so `n_pixels` — taken from the
    /// stream's page information segment — is known to be a dimension the
    /// decoder accepted by the time it is reserved.
    fn reserve_page(&mut self) -> bool {
        if self.data.capacity() == 0
            && !self.alloc_failed
            && self.n_pixels > 0
            && let Err(e) = self.data.try_reserve_exact(self.n_pixels)
        {
            log::warn!(
                "image: JBIG2Decode: cannot allocate {} bytes: {e}",
                self.n_pixels
            );
            self.alloc_failed = true;
        }
        !self.alloc_failed
    }
}

impl Jbig2Decoder for Jbig2Collector {
    fn push_pixel(&mut self, black: bool) {
        if self.reserve_page() && self.data.len() < self.n_pixels {
            self.data.push(if black { 0x00 } else { 0xFF });
        }
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        if !self.reserve_page() {
            return;
        }
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
    use pdf::Dictionary;

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
    fn ccitt_group4_honours_encoded_byte_align_when_rows_are_aligned() {
        // The rows above, each padded to a byte boundary: row 0 is 11 bits
        // (+5 zero bits), row 1 is 3 bits (+5).  Read unaligned, the padding
        // is decoded as codes and the stream fails after row 0.
        let stream = format!("001 0111 011 1 00000   1 1 1 00000   {EOL} {EOL}");
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
    fn ccitt_collector_expands_changes_and_clamps_to_the_width() {
        let mut c = HayroCcittCollector::new(12, 4, 9, 0xFF).expect("collector");
        // Black from 0 to 3.
        hayro_ccitt::Decoder::push_row(&mut c, &[0, 3]);
        // Black from 2 to a change past the width (`Columns` > width).
        hayro_ccitt::Decoder::push_row(&mut c, &[2, 9]);
        // An odd count: black from 1 to the end of the row.
        hayro_ccitt::Decoder::push_row(&mut c, &[1]);
        assert_eq!(c.rows_decoded(), 3);
        assert_eq!(
            c.finish(),
            [
                0x00, 0x00, 0x00, 0xFF, //
                0xFF, 0xFF, 0x00, 0x00, //
                0xFF, 0x00, 0x00, 0x00,
            ]
        );
    }

    #[test]
    fn jbig2_collector_push_pixel_grayscale() {
        // JBIG2: black=true → Gray 0x00, black=false → Gray 0xFF.
        let mut c = Jbig2Collector::new(2);
        Jbig2Decoder::push_pixel(&mut c, true);
        Jbig2Decoder::push_pixel(&mut c, false);
        assert_eq!(c.data, [0x00, 0xFF]);
    }

    #[test]
    fn jbig2_collector_push_pixel_chunk() {
        let mut c = Jbig2Collector::new(16);
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
        let mut c = Jbig2Collector::new(10);
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

    #[test]
    fn jbig2_decode_oversized_page_returns_none() {
        // A page information segment declaring a 2^32 × 2^28 page: the
        // decoder rejects the bitmap, and no buffer of the declared size is
        // reserved before it does.
        let mut data = vec![
            0x00, 0x00, 0x00, 0x00, // segment number 0
            0x30, // type 48: page information
            0x00, // no referred-to segments
            0x01, // page association 1
            0x00, 0x00, 0x00, 0x13, // data length 19
        ];
        data.extend_from_slice(&0xFFFF_FFF0_u32.to_be_bytes()); // width
        data.extend_from_slice(&0x1000_0000_u32.to_be_bytes()); // height
        data.extend_from_slice(&[0; 8]); // resolutions unknown
        data.extend_from_slice(&[0x00, 0x00, 0x00]); // flags, striping
        let doc = crate::test_helpers::empty_doc();
        assert!(decode_jbig2(&doc, &data, 4, 4, false, None).is_none());
    }

    #[test]
    fn jbig2_page_over_the_decode_budget_returns_none() {
        // A 65 535 × 65 535 page, within the decoder's limits but 4 GiB of
        // samples, under a dictionary that declares 4 × 4: the stream's
        // page size is held to the budget before the decoder allocates.
        let mut data = vec![
            0x00, 0x00, 0x00, 0x00, // segment number 0
            0x30, // type 48: page information
            0x00, // no referred-to segments
            0x01, // page association 1
            0x00, 0x00, 0x00, 0x13, // data length 19
        ];
        data.extend_from_slice(&0xFFFF_u32.to_be_bytes()); // width
        data.extend_from_slice(&0xFFFF_u32.to_be_bytes()); // height
        data.extend_from_slice(&[0; 8]); // resolutions unknown
        data.extend_from_slice(&[0x00, 0x00, 0x00]); // flags, striping
        let doc = crate::test_helpers::empty_doc();
        assert!(decode_jbig2(&doc, &data, 4, 4, false, None).is_none());
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
            assert_eq!(img.layout, Layout::Color(1));
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
            assert_eq!(img.layout, Layout::Color(3));
            assert_eq!(img.data.len(), 12);
        }

        #[test]
        fn generic_gpu_path_rejects_cmyk() {
            let mut dec = AlwaysCmyk;
            let result = decode_dct_gpu_path(&[], 2, 2, &mut dec);
            assert!(result.is_none());
        }
    }

    // ── JPXDecode ─────────────────────────────────────────────────────────────

    /// Colour samples as stored, no alpha.
    #[cfg(not(feature = "nvjpeg2k"))]
    const PLAIN: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 0,
    };

    /// Colour samples as stored, an alpha channel as the soft mask.
    #[cfg(not(feature = "nvjpeg2k"))]
    const WITH_ALPHA: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 1,
    };

    /// Decode a JPEG 2000 fixture (8 × 8) through `decode_jpx`.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn jpx_fixture(bytes: &[u8]) -> Samples {
        decode_jpx(bytes, 8, 8, None, WITH_ALPHA).expect("decodes")
    }

    // Fixtures are lossless (reversible 5/3) encodes; the `.gray`, `.rgb`,
    // `.rgba` and `.gray16` files are the same images decoded by an
    // independent decoder.

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_draw_size_skips_levels_without_going_below_it() {
        let gray = include_bytes!("testdata/gray8.jp2");
        let full = decode_jpx(gray, 8, 8, None, PLAIN).expect("decodes");
        assert_eq!((full.width, full.height), (8, 8));

        let half = decode_jpx(gray, 8, 8, Some((4, 4)), PLAIN).expect("decodes");
        assert_eq!((half.width, half.height), (4, 4));
        assert_eq!(half.data.as_cpu().map(<[u8]>::len), Some(16));

        // 4 × 4 would be smaller than the drawn size, so no level is skipped.
        let five = decode_jpx(gray, 8, 8, Some((5, 5)), PLAIN).expect("decodes");
        assert_eq!((five.width, five.height), (8, 8));
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_gray_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("testdata/gray8.jp2"));
        assert_eq!(img.layout, Layout::Color(1));
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
        assert_eq!(img.layout, Layout::Color(3));
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
        assert_eq!(img.layout, Layout::Color(3));
        assert_eq!(img.data.as_cpu(), Some(&rgb[..]));
        let mask = img.smask.expect("alpha channel present");
        assert_eq!((mask.width(), mask.height()), (img.width, img.height));
        assert_eq!(mask.alpha(), &alpha[..]);
    }

    /// A codestream whose main header carries a CAP marker (HTJ2K) is
    /// refused with a diagnostic that names it.
    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_htj2k_codestream_is_refused_by_name() {
        // SOC and SIZ of the RGB fixture, then CAP (Lcap 8, Pcap bit 15:
        // Part 15, Ccap15 0x4000), then an EOC.
        let j2k = include_bytes!("testdata/rgb8.j2k");
        assert_eq!(&j2k[..4], &[0xFF, 0x4F, 0xFF, 0x51]);
        let lsiz = usize::from(u16::from_be_bytes([j2k[4], j2k[5]]));
        let mut ht = j2k[..4 + lsiz].to_vec();
        ht.extend_from_slice(&[0xFF, 0x50, 0x00, 0x08, 0x00, 0x02, 0x00, 0x00, 0x40, 0x00]);
        ht.extend_from_slice(&[0xFF, 0xD9]);
        let err = decode_jpx(&ht, 8, 8, None, PLAIN).expect_err("HTJ2K is not decoded");
        assert!(err.contains("HTJ2K"), "{err}");
    }

    /// A codestream whose SIZ declares 30 000 × 30 000 RGB samples, under a
    /// dictionary that declares 8 × 8, is refused by the decode budget
    /// before the decoder allocates its component buffers.
    #[test]
    fn jpx_codestream_over_the_decode_budget_is_refused() {
        let mut j2k = include_bytes!("testdata/rgb8.j2k").to_vec();
        assert_eq!(&j2k[..4], &[0xFF, 0x4F, 0xFF, 0x51]);
        // Xsiz, Ysiz, then XTsiz and YTsiz (one tile), after Lsiz and Rsiz.
        for offset in [8, 12, 24, 28] {
            j2k[offset..offset + 4].copy_from_slice(&30_000u32.to_be_bytes());
        }
        let opts = JpxOptions {
            indices: false,
            smask_in_data: 0,
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, opts, None).expect_err("over the budget");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, opts).expect_err("over the budget");
        assert!(err.contains("budget"), "{err}");
        let indexed = JpxOptions {
            indices: true,
            ..opts
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, indexed, None).expect_err("over the budget");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, indexed).expect_err("over the budget");
        assert!(err.contains("budget"), "{err}");
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
}
