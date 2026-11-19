//! `JPXDecode` (JPEG 2000): the CPU decoder and nvJPEG2000 behind its
//! feature.

use crate::resources::image::colorspace::{KeySamples, Layout, Samples};
#[cfg(feature = "nvjpeg2k")]
use crate::resources::image::jpx_area_dispatches;

#[cfg(feature = "nvjpeg2k")]
mod device;

#[cfg(feature = "nvjpeg2k")]
pub(crate) use device::{DeviceJpx, Jpeg2kSource};

// ── JPXDecode (JPEG 2000) ─────────────────────────────────────────────────────

/// Decode a `JPXDecode` (JPEG 2000) stream.
///
/// When the `nvjpeg2k` feature is active and `gpu` is `Some`, an image of
/// at least `gpu`'s threshold in pixels whose codestream nvJPEG2000
/// reconstructs as the CPU decoder does (`device::takes`), which holds
/// all the tile data its headers declare, and whose draw the CPU decodes
/// at full resolution, is decoded on the device,
/// the decoder being obtained from `gpu` only then.  Every
/// other image, and one whose device decode fails, decodes on the CPU; a
/// codestream the device does not take is reported (`device::report`).
///
/// PDF JPEG 2000 streams may be raw codestreams (`.j2k`) or full JP2 container
/// format (`.jp2`).  Both paths auto-detect the format from the stream.
///
/// Samples of any precision are scaled to 8 bits, or to 16 for a
/// codestream of more than 8 bits under `opts.wide`; sub-sampled
/// components are upsampled to the image grid.  `opts` carries what the image dictionary
/// decides about the codestream: see [`JpxOptions`].  An ICC profile the
/// codestream's colour specification carries is returned with the samples
/// (`Samples::embedded_profile`) for an image dictionary that names no
/// colour space.  The error names what the decoder refused.
///
/// A codestream that ends early is judged by [`judged`]: one with a tile
/// that holds no data is an error, and the image is not drawn; one whose
/// every tile holds data, some of them cut short, is drawn from the
/// packets present and marked partial (`Samples::partial`).
pub(in crate::resources::image) fn decode_jpx(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    draw_size: Option<(u32, u32)>,
    opts: &JpxOptions,
    #[cfg(feature = "nvjpeg2k")] gpu: Option<&mut DeviceJpx<'_>>,
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
    let img = open_image(data, opts.channels, &settings)?;
    jpx_budget(&img, 1)?;
    let embedded_profile = match img.color_space() {
        hayro_jpeg2000::ColorSpace::Icc { profile, .. } => Some(profile.clone()),
        _ => None,
    };

    // ── Device path ──────────────────────────────────────────────────────────
    // A codestream of more than 8 bits is kept at 16 (samples whose two
    // bytes each exceed the budget are kept at 8); the device then returns
    // the components at their own precision, widened as the CPU path
    // widens them.  One under a colour key, which compares the samples at
    // their own precision, decodes on the CPU.
    let wide = opts.wide
        && img.original_bit_depth() > 8
        && jpx_budget(&img, 2)
            .inspect_err(|why| {
                log::warn!("image: JPXDecode: {why} at 16 bits — decoding at 8 bits");
            })
            .is_ok();
    #[cfg(feature = "nvjpeg2k")]
    if let Some(device) = gpu.filter(|_| !opts.colour_key)
        && jpx_area_dispatches(pdf_w, pdf_h, device.threshold())
        && let Ok(survey) = img.survey()
    {
        match device::takes(&img, &survey, wide) {
            // A codestream that ends early decodes on the CPU, which draws
            // the packets present and judges the loss (`judged`).
            Ok(()) if !survey.completeness.is_complete() => {}
            // The device decodes every resolution level; a draw the CPU
            // decodes at fewer is declined before the device decode.
            Ok(()) if draw_size.is_some() && !full_resolution(data, opts.channels, &img) => {
                device::report(&device::Decline::Reduced);
            }
            Ok(()) => {
                let cpu_size = (img.width(), img.height());
                if let Some(mut samples) =
                    device::decode(data, (pdf_w, pdf_h), cpu_size, wide, device)
                {
                    samples.embedded_profile = embedded_profile;
                    return Ok(samples);
                }
            }
            Err(why) => device::report(&why),
        }
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    let (jw, jh) = (img.width(), img.height());
    if jw == 0 || jh == 0 {
        return Err(format!("codestream reports zero dimensions {jw}×{jh}"));
    }
    if draw_size.is_none() && (jw != pdf_w || jh != pdf_h) {
        log::warn!(
            "image: JPXDecode: the image dictionary says {pdf_w}×{pdf_h} and the codestream \
             {jw}×{jh}; drawing at the codestream's size"
        );
    }

    let colour_channels = img.color_space().num_channels();
    let channels = usize::from(colour_channels);
    let has_alpha = img.has_alpha();
    // The samples are read by channel count; CMYK samples are ink
    // densities (0 = no ink).  Other counts are those of the image
    // dictionary's colour space.
    let layout = match colour_channels {
        n @ (1 | 3 | 4) => Layout::Color(n),
        n if opts.channels == Some(n) => Layout::Color(n),
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
            let (colour, alpha) = split_alpha(wide, channels, has_alpha, opts);
            let mut samples = Samples::cpu16(jw, jh, layout, colour);
            samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new16(jw, jh, a));
            samples
        } else {
            let bytes = decoded.data_u8();
            check_len(bytes.len(), expected, (jw, jh, stride))?;
            let (colour, alpha) = split_alpha(bytes, channels, has_alpha, opts);
            let mut samples = Samples::cpu(jw, jh, layout, colour);
            samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new(jw, jh, a));
            samples
        };
        samples.key_source = key_source;
        samples
    } else {
        let bytes = img.decode_u8(&mut ctx).map_err(|e| e.to_string())?;
        check_len(bytes.len(), expected, (jw, jh, stride))?;
        let (colour, alpha) = split_alpha(bytes, channels, has_alpha, opts);
        let mut samples = Samples::cpu(jw, jh, layout, colour);
        samples.smask = alpha.and_then(|a| crate::resources::image::SoftMask::new(jw, jh, a));
        samples
    };
    samples.partial = judged(ctx.completeness())?;
    samples.embedded_profile = embedded_profile;
    Ok(samples)
}

/// The completeness `c` of a decoded codestream as a draw decision: `Err`
/// when a tile holds no data, which would be drawn as the
/// zero-coefficient level (mid-grey) over its whole area; the reason the
/// image is partial when every tile holds data but some lost part of it
/// to the end of the codestream, their packets being drawn as decoded;
/// `None` when every tile is whole.
fn judged(c: hayro_jpeg2000::Completeness) -> Result<Option<String>, String> {
    if c.tiles_without_data > 0 {
        return Err(format!(
            "the codestream ends early: {} of its {} tiles hold no data",
            c.tiles_without_data, c.tiles
        ));
    }
    Ok((c.truncated_tiles > 0).then(|| {
        format!(
            "JPXDecode: the codestream ends early: {} of its {} tiles are cut short; the \
             image is drawn as far as it decodes",
            c.truncated_tiles, c.tiles
        )
    }))
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
/// the image's soft mask only when `/SMaskInData` (`opts.smask_in_data`)
/// is nonzero (§8.9.5); with 2 the colour channels are premultiplied by
/// it against the matte ([`matte_samples`]), a relation inverted in the
/// codestream's own colour space.
fn split_alpha<S: color::Sample>(
    samples: Vec<S>,
    channels: usize,
    has_alpha: bool,
    opts: &JpxOptions,
) -> (Vec<S>, Option<Vec<S>>) {
    let smask_in_data = opts.smask_in_data;
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
            let matte = matte_samples(opts.matte.as_deref(), channels);
            crate::resources::image::smask::unmatte(&mut colour, &alpha, &matte);
        }
        alpha
    });
    (colour, alpha)
}

/// The matte `matte` of a codestream of `channels` colour channels as
/// samples: each component, in 0..1, scaled to the sample range.  A matte
/// of another length, which §11.6.5.3 does not define, is 0 in every
/// component, with a warning, as is an absent one.
fn matte_samples<S: color::Sample>(matte: Option<&[f64]>, channels: usize) -> Vec<S> {
    match matte {
        Some(m) if m.len() == channels => m.iter().map(|&v| S::from_unit(v)).collect(),
        Some(m) => {
            log::warn!(
                "image: JPXDecode: /Matte has {} components, the codestream {channels} colour \
                 channels; matte 0 assumed",
                m.len()
            );
            vec![S::ZERO; channels]
        }
        None => vec![S::ZERO; channels],
    }
}

/// The factor [`widen`] scales a sample of `bits` bits by:
/// `65535 / (2^bits − 1)`.
fn widen_scale(bits: u8) -> f64 {
    65_535.0 / f64::from(u32::MAX >> (32 - u32::from(bits).clamp(1, 32)))
}

/// The sample `v` of a component of the precision `scale` stands for
/// ([`widen_scale`]) at 16 bits: `round(v · 65535 / (2^bits − 1))`,
/// clamped to `0..=65535`.
fn widen(v: f64, scale: f64) -> u16 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0..=65535 before the cast"
    )]
    let w = (v * scale).round().clamp(0.0, 65_535.0) as u16;
    w
}

/// Every component of `decoded`, interleaved, each scaled from its own
/// precision to 16 bits ([`widen`]).
fn wide_samples(decoded: &hayro_jpeg2000::DecodedImage<'_>) -> Vec<u16> {
    let components = decoded.components();
    let len = components.first().map_or(0, |c| c.samples().len());
    let scales: Vec<f64> = components
        .iter()
        .map(|c| widen_scale(c.bit_depth()))
        .collect();
    let mut out = Vec::with_capacity(len * components.len());
    for i in 0..len {
        for (c, &scale) in components.iter().zip(&scales) {
            out.push(widen(f64::from(c.samples()[i]), scale));
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
    let mut samples = Samples::gray(jw, jh, indices);
    samples.partial = judged(ctx.completeness())?;
    Ok(samples)
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

/// The codestream `data` read with `settings`.  The image dictionary's
/// colour space, when it names one, gives a raw codestream's colour
/// `channels`; a codestream that does not match them (or a JP2 file) is
/// read with its own colour specification.
fn open_image<'a>(
    data: &'a [u8],
    channels: Option<u8>,
    settings: &hayro_jpeg2000::DecodeSettings,
) -> Result<hayro_jpeg2000::Image<'a>, String> {
    channels
        .and_then(|n| hayro_jpeg2000::Image::from_codestream(data, n, settings).ok())
        .map_or_else(|| hayro_jpeg2000::Image::new(data, settings), Ok)
        .map_err(|e| e.to_string())
}

/// Whether `img`, the codestream `data` read for a draw, has the size of
/// the codestream read at full resolution: no resolution level is skipped.
/// A codestream whose full-resolution headers do not read is taken as
/// reduced.
#[cfg(feature = "nvjpeg2k")]
fn full_resolution(data: &[u8], channels: Option<u8>, img: &hayro_jpeg2000::Image<'_>) -> bool {
    open_image(data, channels, &hayro_jpeg2000::DecodeSettings::default())
        .is_ok_and(|full| (full.width(), full.height()) == (img.width(), img.height()))
}

/// What the image dictionary decides about a JPEG 2000 codestream.
#[derive(Debug, Clone)]
pub(in crate::resources::image) struct JpxOptions {
    /// The PDF colour space is `/Indexed`: decode the codestream's samples
    /// as palette indices for the caller to expand.
    pub indices: bool,
    /// `/SMaskInData` (§8.9.5): 0 ignores an alpha channel, 1 makes it the
    /// soft mask, 2 makes it the soft mask of colour premultiplied by it.
    pub smask_in_data: i64,
    /// The image dictionary's `/Matte` under `/SMaskInData 2`: the colour
    /// the samples are premultiplied against, one component in 0..1 per
    /// colour channel of the codestream (§11.6.5.3).  `None` is a matte of
    /// 0 in every component.
    pub matte: Option<Vec<f64>>,
    /// A colour-key `/Mask` compares the samples: a codestream of any
    /// component precision other than 8 bits keeps them at that precision
    /// as well ([`Samples::key_source`]).
    pub colour_key: bool,
    /// Keep the samples of a codestream of more than 8 bits at 16 bits.
    pub wide: bool,
    /// The components of the image dictionary's colour space, when it names
    /// one other than `/Indexed`: a raw codestream is decoded with that many
    /// colour channels, plus alpha when it has one more component, rather
    /// than with a colour space inferred from its components.
    pub channels: Option<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JPXDecode ─────────────────────────────────────────────────────────────

    /// Colour samples as stored, no alpha.
    const PLAIN: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 0,
        matte: None,
        colour_key: false,
        wide: false,
        channels: None,
    };

    /// Colour samples as stored, an alpha channel as the soft mask.
    #[cfg(not(feature = "nvjpeg2k"))]
    const WITH_ALPHA: JpxOptions = JpxOptions {
        indices: false,
        smask_in_data: 1,
        matte: None,
        colour_key: false,
        wide: false,
        channels: None,
    };

    /// Decode a JPEG 2000 fixture (8 × 8) through `decode_jpx`.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn jpx_fixture(bytes: &[u8]) -> Samples {
        decode_jpx(bytes, 8, 8, None, &WITH_ALPHA).expect("decodes")
    }

    // Fixtures are lossless (reversible 5/3) encodes; the `.gray`, `.rgb`,
    // `.rgba` and `.gray16` files are the same images decoded by an
    // independent decoder.

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_draw_size_skips_levels_without_going_below_it() {
        let gray = include_bytes!("../../testdata/gray8.jp2");
        let full = decode_jpx(gray, 8, 8, None, &PLAIN).expect("decodes");
        assert_eq!((full.width, full.height), (8, 8));

        let half = decode_jpx(gray, 8, 8, Some((4, 4)), &PLAIN).expect("decodes");
        assert_eq!((half.width, half.height), (4, 4));
        assert_eq!(half.data.as_cpu().map(<[u8]>::len), Some(16));

        // 4 × 4 would be smaller than the drawn size, so no level is skipped.
        let five = decode_jpx(gray, 8, 8, Some((5, 5)), &PLAIN).expect("decodes");
        assert_eq!((five.width, five.height), (8, 8));
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_gray_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("../../testdata/gray8.jp2"));
        assert_eq!(img.layout, Layout::Color(1));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../../testdata/gray8.gray")[..])
        );
        assert!(img.smask.is_none());
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_raw_codestream_rgb_matches_reference_decode() {
        let img = jpx_fixture(include_bytes!("../../testdata/rgb8.j2k"));
        assert_eq!(img.layout, Layout::Color(3));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../../testdata/rgb8.rgb")[..])
        );
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_alpha_channel_becomes_the_soft_mask() {
        let img = jpx_fixture(include_bytes!("../../testdata/rgba8.jp2"));
        let rgba = include_bytes!("../../testdata/rgba8.rgba");
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
        let img = jpx_fixture(include_bytes!("../../testdata/rgb8_ht.j2k"));
        assert_eq!(img.layout, Layout::Color(3));
        assert_eq!(
            img.data.as_cpu(),
            Some(&include_bytes!("../../testdata/rgb8.rgb")[..])
        );
    }

    /// A codestream that ends early: with its only tile cut short it is
    /// drawn as decoded and marked partial; ended before the tile's data
    /// it is not drawn.  `rgb8.j2k` holds one tile-part, its `SOT` at
    /// byte 119 and its `SOD` at 131.
    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn a_truncated_codestream_is_partial_or_refused() {
        let j2k = include_bytes!("../../testdata/rgb8.j2k");
        assert_eq!(&j2k[119..121], [0xFF, 0x90]);
        assert_eq!(&j2k[131..133], [0xFF, 0x93]);
        let decode = |len: usize| decode_jpx(&j2k[..len], 8, 8, None, &PLAIN);
        assert!(decode(j2k.len()).expect("whole").partial.is_none());
        let cut = decode(170).expect("cut short");
        let reason = cut.partial.expect("partial");
        assert!(
            reason.contains("1 of its 1 tiles are cut short"),
            "{reason}"
        );
        for len in [131, 133] {
            let err = decode(len).expect_err("no tile data");
            assert!(
                err.contains("1 of its 1 tiles hold no data"),
                "{len}: {err}"
            );
        }
    }

    /// Decode a 32 × 32 raw codestream under a dictionary whose colour space
    /// has `channels` components.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn raw_codestream(bytes: &[u8], channels: u8) -> Samples {
        let opts = JpxOptions {
            channels: Some(channels),
            ..PLAIN
        };
        decode_jpx(bytes, 32, 32, None, &opts).expect("decodes")
    }

    /// The `n` components at the centre of each 8 × 8 cell of a 32 × 32
    /// image, by cell row.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn cell_centres(samples: &Samples, n: usize) -> Vec<Vec<u8>> {
        let data = samples.data.as_cpu().expect("host samples");
        assert_eq!(data.len(), 32 * 32 * n);
        (0..4)
            .flat_map(|j| (0..4).map(move |i| (4 + 8 * i, 4 + 8 * j)))
            .map(|(x, y)| data[(y * 32 + x) * n..][..n].to_vec())
            .collect()
    }

    /// Whether every component of `got` is within 8 of `want`.
    #[cfg(not(feature = "nvjpeg2k"))]
    fn near(got: &[u8], want: &[u8]) -> bool {
        got.iter().zip(want).all(|(&g, &w)| g.abs_diff(w) <= 8)
    }

    /// A raw codestream under a two-component `DeviceN` space decodes as two
    /// colour channels, not as gray with alpha: a cyan ramp across and a
    /// magenta ramp down, cross-checked with a `mutool draw` render of the
    /// page.
    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_raw_codestream_takes_the_dictionary_channels() {
        let img = raw_codestream(include_bytes!("../../testdata/devicen2.j2k"), 2);
        assert_eq!(img.layout, Layout::Color(2));
        assert!(img.smask.is_none());
        let ramp = [29, 95, 162, 228];
        for (cell, got) in cell_centres(&img, 2).iter().enumerate() {
            let want = [ramp[cell % 4], ramp[cell / 4]];
            assert!(near(got, &want), "cell {cell}: {got:?} vs {want:?}");
        }
    }

    /// A raw codestream with subsampled second and third components under
    /// `DeviceRGB` keeps its samples as RGB, without the YCbCr reading of
    /// subsampled chroma: red across, green down, blue a checkerboard, as
    /// `mutool draw` renders the page.
    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_raw_codestream_with_subsampled_rgb_is_not_ycbcr() {
        let img = raw_codestream(include_bytes!("../../testdata/subsampled_rgb.j2k"), 3);
        assert_eq!(img.layout, Layout::Color(3));
        let (red, green) = ([28, 94, 160, 226], [26, 94, 162, 230]);
        for (cell, got) in cell_centres(&img, 3).iter().enumerate() {
            let (i, j) = (cell % 4, cell / 4);
            let blue = if (i + j) % 2 == 0 { 0 } else { 255 };
            let want = [red[i], green[j], blue];
            assert!(near(got, &want), "cell {cell}: {got:?} vs {want:?}");
        }
    }

    /// A codestream whose SIZ declares 30 000 × 30 000 RGB samples, under a
    /// dictionary that declares 8 × 8, is refused before the decoder
    /// allocates its component buffers.  The decoder's own sample limit
    /// (2^30, equal to the decode budget for 8-bit output) is checked
    /// first, when the header is parsed.
    #[test]
    fn jpx_codestream_over_the_sample_limit_is_refused() {
        let mut j2k = include_bytes!("../../testdata/rgb8.j2k").to_vec();
        assert_eq!(&j2k[..4], &[0xFF, 0x4F, 0xFF, 0x51]);
        // Xsiz, Ysiz, then XTsiz and YTsiz (one tile), after Lsiz and Rsiz.
        for offset in [8, 12, 24, 28] {
            j2k[offset..offset + 4].copy_from_slice(&30_000u32.to_be_bytes());
        }
        let opts = JpxOptions {
            indices: false,
            smask_in_data: 0,
            matte: None,
            colour_key: false,
            wide: false,
            channels: None,
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, &opts, None).expect_err("over the limit");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, &opts).expect_err("over the limit");
        assert!(err.contains("too large"), "{err}");
        let indexed = JpxOptions {
            indices: true,
            ..opts
        };
        #[cfg(feature = "nvjpeg2k")]
        let err = decode_jpx(&j2k, 8, 8, None, &indexed, None).expect_err("over the limit");
        #[cfg(not(feature = "nvjpeg2k"))]
        let err = decode_jpx(&j2k, 8, 8, None, &indexed).expect_err("over the limit");
        assert!(err.contains("too large"), "{err}");
    }

    #[cfg(not(feature = "nvjpeg2k"))]
    #[test]
    fn jpx_16_bit_samples_scale_to_8_bit() {
        let img = jpx_fixture(include_bytes!("../../testdata/gray16.jp2"));
        let reference = include_bytes!("../../testdata/gray16.gray16");
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

    /// At 16 bits an alpha channel is dropped under `/SMaskInData 0`,
    /// split off as the soft mask under 1, and under 2 also divided out of
    /// the colour premultiplied by it.
    #[test]
    fn split_alpha_follows_smask_in_data_at_16_bits() {
        let samples = vec![10_000u16, 20_000, 16_384, 65_535, 0, 65_535];
        let smask_in_data = |smask_in_data| JpxOptions {
            smask_in_data,
            ..PLAIN
        };
        let (colour, alpha) = split_alpha(samples.clone(), 2, true, &smask_in_data(0));
        assert_eq!((colour, alpha), (vec![10_000, 20_000, 65_535, 0], None));
        let (colour, alpha) = split_alpha(samples.clone(), 2, true, &smask_in_data(1));
        assert_eq!(colour, [10_000, 20_000, 65_535, 0]);
        assert_eq!(alpha.as_deref(), Some(&[16_384, 65_535][..]));
        let (colour, alpha) = split_alpha(samples, 2, true, &smask_in_data(2));
        assert_eq!(alpha.as_deref(), Some(&[16_384, 65_535][..]));
        assert!(
            colour[0].abs_diff(40_000) <= 1 && colour[1].abs_diff(65_535) <= 1,
            "{colour:?}"
        );
        assert_eq!(colour[2..], [65_535, 0]);
    }

    /// Under `/SMaskInData 2` the colour is un-premultiplied against the
    /// dictionary's `/Matte`, 0 in every component when it is absent or
    /// of a length other than the colour channels'.
    #[test]
    fn split_alpha_unmattes_against_the_dictionary_matte() {
        // One RGB pixel stored as m + α·(c − m) with c = (0, 0, 0),
        // m = (0, 1, 0) and α = 0.2: (0, 204, 0) at alpha 51.
        let samples = vec![0u8, 204, 0, 51];
        let with_matte = |matte: Option<Vec<f64>>| JpxOptions {
            smask_in_data: 2,
            matte,
            ..PLAIN
        };
        let (colour, _) = split_alpha(
            samples.clone(),
            3,
            true,
            &with_matte(Some(vec![0.0, 1.0, 0.0])),
        );
        assert_eq!(colour, [0, 0, 0]);
        let (colour, _) = split_alpha(samples.clone(), 3, true, &with_matte(None));
        assert_eq!(colour, [0, 255, 0]);
        let (colour, _) = split_alpha(samples, 3, true, &with_matte(Some(vec![1.0])));
        assert_eq!(colour, [0, 255, 0]);
    }
}
