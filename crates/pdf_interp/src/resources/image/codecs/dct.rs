//! `DCTDecode` (JPEG): the CPU decoder, the GPU decoders behind their
//! features, and the device image cache.

#[cfg(feature = "cache")]
use std::sync::Arc;

#[cfg(feature = "cache")]
use crate::resources::image::ImageData;
use crate::resources::image::colorspace::{Layout, Samples};

#[cfg(feature = "cache")]
use gpu::cache::{
    CachedDeviceImage, ContentHash, DeviceImageCache, DocId, ImageLayout as CacheImageLayout,
    InsertRequest, ObjId,
};

// ── JPEG SOF peek ─────────────────────────────────────────────────────────────

/// Scan `data` for the first JPEG SOF marker and return the component count
/// (`Nf` field). Returns `None` if `data` is not a valid JPEG or the SOF
/// payload is too short. Zero allocations; stops at the first SOF found.
///
/// This avoids a full header-decode probe pass to determine the component count
/// before calling the main decode. The `gpu` crate has the same logic but is an
/// optional dependency (requires CUDA at link time), so the CPU path carries
/// its own copy of the scan.
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
pub(in crate::resources::image) fn decode_dct<
    #[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend,
>(
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
    ctx: &mut crate::resources::image::ImageDecodeCtx<'_>,
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
        let area_above_threshold = area >= crate::resources::image::GPU_JPEG_THRESHOLD_PX;
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
            let above = area >= crate::resources::image::GPU_JPEG_THRESHOLD_PX;
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
        if area >= crate::resources::image::huffman_threshold() {
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

    // Validate the pixel buffer length against the dimensions used below.
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
pub(in crate::resources::image) struct DctCacheCtx<'a> {
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
    pub(in crate::resources::image) fn from_resources(
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

#[cfg(all(test, any(feature = "nvjpeg", feature = "vaapi")))]
mod tests {
    use super::*;

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

    #[test]
    fn jpeg_variant_baseline_detected() {
        let data = minimal_baseline_jpeg();
        assert_eq!(gpu::jpeg_sof_type(&data), Some(gpu::JpegVariant::Baseline));
    }

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
}
