//! `JPXDecode` (JPEG 2000): the CPU decoder and nvJPEG2000 behind its
//! feature.

use crate::resources::image::colorspace::{KeySamples, Layout, Samples};

#[cfg(feature = "nvjpeg2k")]
use gpu::nvjpeg2k::{Jpeg2kColorSpace as GpuJ2kCs, NvJpeg2kDecoder};

// ── JPXDecode (JPEG 2000) ─────────────────────────────────────────────────────

/// Decode a `JPXDecode` (JPEG 2000) stream.
///
/// When the `nvjpeg2k` feature is active and `gpu` is `Some`, large images
/// (pixel area ≥ `GPU_JPEG2K_THRESHOLD_PX`) are decoded on the GPU via
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
pub(in crate::resources::image) fn decode_jpx(
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
    jpx_budget(&img, 1)?;
    let embedded_profile = match img.color_space() {
        hayro_jpeg2000::ColorSpace::Icc { profile, .. } => Some(profile.clone()),
        _ => None,
    };

    // ── GPU fast path (nvjpeg2k feature, large images, 1- or 3-component only) ─
    // nvJPEG2000 rejects codestreams with an alpha channel, so the GPU result
    // never needs `/SMaskInData`.  Its samples are 8-bit: a codestream of
    // more than 8 bits kept at 16, or with a colour key, which compares the
    // samples at their own precision, decodes on the CPU.  Samples whose
    // two bytes each exceed the budget are kept at 8 bits.
    let wide = opts.wide
        && img.original_bit_depth() > 8
        && jpx_budget(&img, 2)
            .inspect_err(|why| {
                log::warn!("image: JPXDecode: {why} at 16 bits — decoding at 8 bits");
            })
            .is_ok();
    #[cfg(feature = "nvjpeg2k")]
    if let Some(dec) = gpu.filter(|_| !opts.colour_key && !wide) {
        let area = pdf_w.saturating_mul(pdf_h);
        if area >= crate::resources::image::GPU_JPEG2K_THRESHOLD_PX {
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
    // The samples are read by channel count; CMYK samples are ink
    // densities (0 = no ink).
    let layout = match channels {
        1 => Layout::Color(1),
        3 => Layout::Color(3),
        4 => Layout::Color(4),
        n => return Err(format!("unsupported {n}-channel colour space")),
    };
    let stride = channels + usize::from(has_alpha);
    let expected = (jw as usize)
        .checked_mul(jh as usize)
        .and_then(|npx| npx.checked_mul(stride));
    let mut ctx = hayro_jpeg2000::DecoderContext::default();
    let mut samples = if opts.colour_key || wide {
        let decoded = img.decode(&mut ctx).map_err(|e| e.to_string())?;
        let key_source = opts
            .colour_key
            .then(|| key_samples(&decoded, channels, (jw, jh)))
            .flatten();
        let mut samples = if wide {
            let wide = wide_samples(&decoded);
            check_len(wide.len(), expected, (jw, jh, stride))?;
            let (colour, alpha) = split_alpha(wide, channels, has_alpha, opts.smask_in_data);
            let mut samples = Samples::cpu16(jw, jh, layout, colour);
            samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new16(jw, jh, a));
            samples
        } else {
            let bytes = decoded.data_u8();
            check_len(bytes.len(), expected, (jw, jh, stride))?;
            let (colour, alpha) = split_alpha(bytes, channels, has_alpha, opts.smask_in_data);
            let mut samples = Samples::cpu(jw, jh, layout, colour);
            samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new(jw, jh, a));
            samples
        };
        samples.key_source = key_source;
        samples
    } else {
        let bytes = img.decode_u8(&mut ctx).map_err(|e| e.to_string())?;
        check_len(bytes.len(), expected, (jw, jh, stride))?;
        let (colour, alpha) = split_alpha(bytes, channels, has_alpha, opts.smask_in_data);
        let mut samples = Samples::cpu(jw, jh, layout, colour);
        samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new(jw, jh, a));
        samples
    };
    samples.embedded_profile = embedded_profile;
    Ok(samples)
}

/// `Err` naming the counts when `got` samples are not the `expected` of a
/// `w × h` image of `stride` samples a pixel.
fn check_len(
    got: usize,
    expected: Option<usize>,
    (w, h, stride): (u32, u32, usize),
) -> Result<(), String> {
    if expected == Some(got) {
        Ok(())
    } else {
        Err(format!("{got} samples for {w}×{h}×{stride}"))
    }
}

/// The colour samples of interleaved `samples` of `channels` colours and,
/// when `has_alpha`, an alpha channel last, and the alpha.  The alpha is
/// the image's soft mask only when `/SMaskInData` (`smask_in_data`) is
/// nonzero (§8.9.5); with 2 the colour channels are premultiplied by it,
/// which is the matte relation with every matte component 0, inverted in
/// the codestream's own colour space.
fn split_alpha<S: color::Sample>(
    samples: Vec<S>,
    channels: usize,
    has_alpha: bool,
    smask_in_data: i64,
) -> (Vec<S>, Option<Vec<S>>) {
    if !has_alpha {
        return (samples, None);
    }
    let stride = channels + 1;
    let mut colour: Vec<S> = samples
        .chunks_exact(stride)
        .flat_map(|px| px[..channels].iter().copied())
        .collect();
    let alpha = (smask_in_data != 0).then(|| {
        let alpha: Vec<S> = samples
            .chunks_exact(stride)
            .map(|px| px[channels])
            .collect();
        if smask_in_data == 2 {
            crate::resources::image::smask::unmatte(&mut colour, &alpha, &vec![S::ZERO; channels]);
        }
        alpha
    });
    (colour, alpha)
}

/// Every component of `decoded`, interleaved, each scaled from its own
/// precision to 16 bits: `round(v · 65535 / (2^bits − 1))`.
fn wide_samples(decoded: &hayro_jpeg2000::DecodedImage<'_>) -> Vec<u16> {
    let components = decoded.components();
    let len = components.first().map_or(0, |c| c.samples().len());
    let scales: Vec<f64> = components
        .iter()
        .map(|c| 65_535.0 / f64::from(u32::MAX >> (32 - u32::from(c.bit_depth()).clamp(1, 32))))
        .collect();
    let mut out = Vec::with_capacity(len * components.len());
    for i in 0..len {
        for (c, scale) in components.iter().zip(&scales) {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=65535 before the cast"
            )]
            out.push(
                (f64::from(c.samples()[i]) * scale)
                    .round()
                    .clamp(0.0, 65_535.0) as u16,
            );
        }
    }
    out
}

/// The samples a colour key compares for the first `channels` components
/// of `decoded`, a `w × h` image: `None` when every component is of 8
/// bits, whose 8-bit samples are their own; otherwise the components
/// interleaved at their own precision, each sample rounded and clamped to
/// `0..=2^bits − 1`, or why they cannot be kept — a component of more than
/// 16 bits, two bytes a sample over the decode budget, or an allocation
/// refused.
fn key_samples(
    decoded: &hayro_jpeg2000::DecodedImage<'_>,
    channels: usize,
    (w, h): (u32, u32),
) -> Option<KeySamples> {
    let components = &decoded.components()[..channels.min(decoded.components().len())];
    if components.iter().all(|c| c.bit_depth() == 8) {
        return None;
    }
    if let Some(c) = components.iter().find(|c| c.bit_depth() > 16) {
        return Some(KeySamples::Unusable(format!(
            "a {}-bit component exceeds the 16 bits a key sample holds",
            c.bit_depth()
        )));
    }
    if let Err(why) = crate::resources::image::decode_budget(w, h, 2 * components.len()) {
        return Some(KeySamples::Unusable(format!("key samples: {why}")));
    }
    let len = components.first().map_or(0, |c| c.samples().len());
    let mut out = Vec::new();
    if out.try_reserve_exact(len * components.len()).is_err() {
        return Some(KeySamples::Unusable(
            "key samples: allocation refused".to_owned(),
        ));
    }
    let maxima: Vec<f64> = components
        .iter()
        .map(|c| f64::from(u16::MAX >> (16 - c.bit_depth().clamp(1, 16))))
        .collect();
    for i in 0..len {
        for (c, &max) in components.iter().zip(&maxima) {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=2^bits − 1 with bits ≤ 16, which u16 holds"
            )]
            out.push(f64::from(c.samples()[i]).round().clamp(0.0, max) as u16);
        }
    }
    Some(KeySamples::Precise(out))
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
    jpx_budget(&img, 1)?;
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

/// The decode budget ([`crate::resources::image::decode_budget`]) applied to the
/// dimensions and channel count a JPEG 2000 codestream declares, which
/// govern the decoder's allocations and may differ from the image
/// dictionary's, at `bytes` bytes a sample.
fn jpx_budget(img: &hayro_jpeg2000::Image<'_>, bytes: usize) -> Result<(), String> {
    let channels = usize::from(img.color_space().num_channels()) + usize::from(img.has_alpha());
    crate::resources::image::decode_budget(img.width(), img.height(), channels * bytes)
        .map_err(|e| format!("codestream {e}"))
}

/// What the image dictionary decides about a JPEG 2000 codestream.
#[derive(Debug, Clone, Copy)]
pub(in crate::resources::image) struct JpxOptions {
    /// The PDF colour space is `/Indexed`: decode the codestream's samples
    /// as palette indices for the caller to expand.
    pub indices: bool,
    /// `/SMaskInData` (§8.9.5): 0 ignores an alpha channel, 1 makes it the
    /// soft mask, 2 makes it the soft mask of colour premultiplied by it.
    pub smask_in_data: i64,
    /// A colour-key `/Mask` compares the samples: a codestream of any
    /// component precision other than 8 bits keeps them at that precision
    /// as well ([`Samples::key_source`]).
    pub colour_key: bool,
    /// Keep the samples of a codestream of more than 8 bits at 16 bits.
    pub wide: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── JPXDecode ─────────────────────────────────────────────────────────────

    /// Colour samples as stored, no alpha.
    #[cfg(not(feature = "nvjpeg2k"))]
    const PLAIN: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 0,
        colour_key: false,
        wide: false,
    };

    /// Colour samples as stored, an alpha channel as the soft mask.
    #[cfg(not(feature = "nvjpeg2k"))]
    const WITH_ALPHA: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 1,
        colour_key: false,
        wide: false,
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
        let gray = include_bytes!("../testdata/gray8.jp2");
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
        let img = jpx_fixture(include_bytes!("../testdata/gray8.jp2"));
        assert_eq!(img.layout, Layout::Color(1));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../testdata/gray8.gray")[..])
        );
        assert!(img.smask.is_none());
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_raw_codestream_rgb_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("../testdata/rgb8.j2k"));
        assert_eq!(img.layout, Layout::Color(3));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../testdata/rgb8.rgb")[..])
        );
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_alpha_channel_becomes_the_soft_mask() {
        let img = jpx_fixture(include_bytes!("../testdata/rgba8.jp2"));
        let rgba = include_bytes!("../testdata/rgba8.rgba");
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

    /// An HTJ2K codestream (ISO/IEC 15444-15: CAP marker, HT code-blocks)
    /// decodes: the fixture's pixels, coded losslessly with the HT block
    /// coder.
    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_htj2k_codestream_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("../testdata/rgb8_ht.j2k"));
        assert_eq!(img.layout, Layout::Color(3));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../testdata/rgb8.rgb")[..])
        );
    }

    /// A codestream whose SIZ declares 30 000 × 30 000 RGB samples, under a
    /// dictionary that declares 8 × 8, is refused before the decoder
    /// allocates its component buffers.  The decoder's own sample limit
    /// (2^30, equal to the decode budget for 8-bit output) is checked
    /// first, when the header is parsed.
    #[test]
    fn jpx_codestream_over_the_sample_limit_is_refused() {
        let mut j2k = include_bytes!("../testdata/rgb8.j2k").to_vec();
        assert_eq!(&j2k[..4], &[0xFF, 0x4F, 0xFF, 0x51]);
        // Xsiz, Ysiz, then XTsiz and YTsiz (one tile), after Lsiz and Rsiz.
        for offset in [8, 12, 24, 28] {
            j2k[offset..offset + 4].copy_from_slice(&30_000u32.to_be_bytes());
        }
        let opts = JpxOptions {
            indices: false,
            smask_in_data: 0,
            colour_key: false,
            wide: false,
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, opts, None).expect_err("over the limit");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, opts).expect_err("over the limit");
        assert!(err.contains("too large"), "{err}");
        let indexed = JpxOptions {
            indices: true,
            ..opts
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, indexed, None).expect_err("over the limit");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, indexed).expect_err("over the limit");
        assert!(err.contains("too large"), "{err}");
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_16_bit_samples_scale_to_8_bit() {
        let img = jpx_fixture(include_bytes!("../testdata/gray16.jp2"));
        let reference = include_bytes!("../testdata/gray16.gray16");
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
