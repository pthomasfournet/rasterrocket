//! `JPXDecode` on the device: the codestreams nvJPEG2000 reconstructs as
//! the CPU decoder does, and the decode itself, into 8-bit samples or, for
//! a codestream of more than 8 bits kept at 16, into 16-bit ones.

use std::sync::atomic::{AtomicU16, Ordering};

use gpu::nvjpeg2k::{Jpeg2kColorSpace, NvJpeg2kDecoder};

use super::{widen, widen_scale};
use crate::resources::image::colorspace::{Layout, Samples};

/// The code-block style bits (Table A.19) nvJPEG2000 parses: predictable
/// termination alone.  A codestream with any other bit set in a COD or
/// COC marker is refused by the library's parser
/// (`NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED`).
const DEVICE_CODE_BLOCK_STYLES: u8 = 0x10;

/// A supplier of a device JPEG 2000 decoder: `None` when none can be
/// built.
pub(crate) type Jpeg2kSource = fn() -> Option<NvJpeg2kDecoder>;

/// The device JPEG 2000 decoder a decode may use: the one attached, or,
/// when none is and a source remains, one the source supplies on first
/// use.  The source is consumed by that use, whatever it returns, so it
/// is called at most once.  Only an image whose area meets `threshold`
/// (`jpx_area_dispatches`) is decoded on it.
pub struct DeviceJpx<'a> {
    decoder: &'a mut Option<NvJpeg2kDecoder>,
    source: Option<&'a mut Option<Jpeg2kSource>>,
    threshold: u32,
}

impl<'a> DeviceJpx<'a> {
    /// The decoder in `decoder`, else one from the source in `source`, for
    /// images of at least `threshold` pixels; `None` when there is neither.
    pub fn new(
        decoder: &'a mut Option<NvJpeg2kDecoder>,
        source: Option<&'a mut Option<Jpeg2kSource>>,
        threshold: u32,
    ) -> Option<Self> {
        let source = source.filter(|s| s.is_some());
        (decoder.is_some() || source.is_some()).then_some(Self {
            decoder,
            source,
            threshold,
        })
    }

    /// The dispatch threshold of the decoder, in pixels.
    pub(super) const fn threshold(&self) -> u32 {
        self.threshold
    }

    /// The decoder, obtained from the source when none is attached.
    fn decoder(&mut self) -> Option<&mut NvJpeg2kDecoder> {
        if self.decoder.is_none()
            && let Some(source) = self.source.as_deref_mut().and_then(Option::take)
        {
            *self.decoder = source();
        }
        self.decoder.as_mut()
    }
}

/// Why nvJPEG2000 does not decode a codestream as the CPU decoder does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Decline {
    /// The samples of the image are not its components as reconstructed.
    NotPlain(hayro_jpeg2000::NotPlain),
    /// Code-block style bits other than predictable termination.
    CodeBlockStyle(u8),
    /// The irreversible 9-7 wavelet on components of more than 8 bits,
    /// whose samples the device rounds to integers before they are widened
    /// to 16 bits, where the CPU widens the reconstructed values.
    IrreversibleWide,
}

impl Decline {
    /// The bit of [`REPORTED`] that stands for the kind of the decline.
    const fn kind(&self) -> u16 {
        use hayro_jpeg2000::NotPlain as N;
        let bit = match self {
            Self::NotPlain(N::ColourConversion) => 0,
            Self::NotPlain(N::Alpha) => 1,
            Self::NotPlain(N::Palette) => 2,
            Self::NotPlain(N::ChannelDefinitions) => 3,
            Self::NotPlain(N::Components(_)) => 4,
            Self::NotPlain(N::Precision(_)) => 5,
            Self::NotPlain(N::Signed) => 6,
            Self::NotPlain(N::SubSampled) => 7,
            Self::CodeBlockStyle(_) => 8,
            Self::IrreversibleWide => 9,
        };
        1 << bit
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPlain(why) => why.fmt(f),
            Self::CodeBlockStyle(style) => write!(f, "code-block style {style:#04x}"),
            Self::IrreversibleWide => {
                f.write_str("the irreversible 9-7 wavelet on components of more than 8 bits")
            }
        }
    }
}

/// The kinds of [`Decline`] already reported in this process.
static REPORTED: AtomicU16 = AtomicU16::new(0);

/// `Err` naming the first property of `img` for which nvJPEG2000 would
/// not return the samples the CPU decoder returns, from its headers and
/// `survey` (see [`hayro_jpeg2000::Image::survey`]): it returns the
/// reconstructed components of a codestream of one or three unsigned
/// components on the image grid, of 8 bits, or of 9 to 16 bits when the
/// samples are kept at 16 (`wide`) and the reversible 5-3 wavelet is in
/// force throughout; and it parses no code-block style but predictable
/// termination.
pub(super) fn takes(
    img: &hayro_jpeg2000::Image<'_>,
    survey: &hayro_jpeg2000::Survey,
    wide: bool,
) -> Result<(), Decline> {
    if wide {
        img.plain_wide().map_err(Decline::NotPlain)?;
        if survey.irreversible {
            return Err(Decline::IrreversibleWide);
        }
    } else {
        img.plain_u8().map_err(Decline::NotPlain)?;
    }
    let styles = survey.code_block_styles & !DEVICE_CODE_BLOCK_STYLES;
    if styles != 0 {
        return Err(Decline::CodeBlockStyle(survey.code_block_styles));
    }
    Ok(())
}

/// Report `why` at warn the first time its kind declines an image, at
/// debug afterwards.
pub(super) fn report(why: &Decline) {
    let kind = why.kind();
    if REPORTED.fetch_or(kind, Ordering::Relaxed) & kind == 0 {
        log::warn!(
            "image: JPXDecode: nvJPEG2000 does not decode {why}; images with it decode on \
             the CPU (reported once)"
        );
    } else {
        log::debug!("image: JPXDecode: nvJPEG2000 does not decode {why}; decoding on the CPU");
    }
}

/// Decode the codestream `data` of a `pdf_w × pdf_h` image dictionary on
/// the device, obtaining the decoder from `device` first: into 8-bit
/// samples, or with `wide` into 16-bit ones, each component's samples
/// widened from its precision as the CPU path widens them.  `None`, after
/// a warning, when no decoder is available or the decode fails; the
/// caller decodes on the CPU.
pub(super) fn decode(
    data: &[u8],
    (pdf_w, pdf_h): (u32, u32),
    wide: bool,
    device: &mut DeviceJpx<'_>,
) -> Option<Samples> {
    let dec = device.decoder()?;
    let decoded = if wide {
        dec.decode_sync_u16(data).map(|img| {
            let samples = widened(&img.data, &img.precisions);
            (
                img.width,
                img.height,
                img.color_space,
                DeviceSamples::Wide(samples),
            )
        })
    } else {
        dec.decode_sync(data).map(|img| {
            (
                img.width,
                img.height,
                img.color_space,
                DeviceSamples::Narrow(img.data),
            )
        })
    };
    let (width, height, color_space, data) = match decoded {
        Ok(img) => img,
        Err(e) => {
            log::warn!("image: JPXDecode: nvJPEG2000 failed ({e}); decoding on the CPU");
            return None;
        }
    };
    if width != pdf_w || height != pdf_h {
        log::debug!(
            "image: JPXDecode: the image dictionary says {pdf_w}×{pdf_h} and nvJPEG2000 \
             {width}×{height}; drawing at the codestream's size"
        );
    }
    let layout = match color_space {
        Jpeg2kColorSpace::Gray => Layout::Color(1),
        Jpeg2kColorSpace::Rgb => Layout::Color(3),
    };
    Some(match data {
        DeviceSamples::Narrow(bytes) => Samples::cpu(width, height, layout, bytes),
        DeviceSamples::Wide(samples) => Samples::cpu16(width, height, layout, samples),
    })
}

/// The samples of a device decode.
enum DeviceSamples {
    /// 8-bit samples.
    Narrow(Vec<u8>),
    /// Samples widened to 16 bits.
    Wide(Vec<u16>),
}

/// Interleaved `samples` of components of `precisions` bits, each widened
/// to 16 bits ([`widen`]).
fn widened(samples: &[u16], precisions: &[u8]) -> Vec<u16> {
    let scales: Vec<f64> = precisions.iter().map(|&p| widen_scale(p)).collect();
    samples
        .chunks_exact(scales.len().max(1))
        .flat_map(|px| {
            px.iter()
                .zip(&scales)
                .map(|(&v, &s)| widen(f64::from(v), s))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(bytes: &[u8]) -> hayro_jpeg2000::Image<'_> {
        hayro_jpeg2000::Image::new(bytes, &hayro_jpeg2000::DecodeSettings::default())
            .expect("parses")
    }

    fn verdict(bytes: &[u8]) -> Result<(), Decline> {
        let img = image(bytes);
        let survey = img.survey().expect("tile-part headers parse");
        takes(&img, &survey, false)
    }

    /// `bytes`, a raw codestream, with the code-block style byte of its
    /// main-header COD marker set to `style`.
    fn with_style(bytes: &[u8], style: u8) -> Vec<u8> {
        let mut out = bytes.to_vec();
        // SOC, then marker segments: a marker and a length that counts
        // itself.
        let mut at = 2;
        while out[at..at + 2] != [0xFF, 0x52] {
            at += 2 + usize::from(u16::from_be_bytes([out[at + 2], out[at + 3]]));
        }
        // Marker, Lcod, Scod, SGcod (4 bytes), then levels, xcb, ycb, style.
        out[at + 2 + 2 + 1 + 4 + 3] = style;
        out
    }

    /// Codestreams of one or three unsigned 8-bit components, with no
    /// code-block style or predictable termination alone, are taken.
    #[test]
    fn plain_eight_bit_codestreams_are_taken() {
        let rgb = include_bytes!("../../testdata/rgb8.j2k");
        assert_eq!(verdict(rgb), Ok(()));
        assert_eq!(verdict(&with_style(rgb, 0x10)), Ok(()));
        assert_eq!(verdict(include_bytes!("../../testdata/gray8.jp2")), Ok(()));
    }

    /// Codestreams of 9 to 16 bits kept at 16 are taken under the
    /// reversible wavelet, and declined under the irreversible one; an
    /// 8-bit codestream kept at 16 is not a wide one.
    #[test]
    fn wide_codestreams_are_taken_under_the_reversible_wavelet_only() {
        use hayro_jpeg2000::NotPlain;
        let wide = |bytes: &[u8]| {
            let img = image(bytes);
            takes(&img, &img.survey().expect("tile-part headers parse"), true)
        };
        assert_eq!(wide(include_bytes!("../../testdata/gray16.jp2")), Ok(()));
        assert_eq!(
            wide(include_bytes!(
                "../../../../../../jpeg2000/tests/streams/codestream_precision_12.j2k"
            )),
            Ok(())
        );
        assert_eq!(
            wide(include_bytes!(
                "../../../../../../jpeg2000/tests/streams/quantization_irreversible_12bit.j2k"
            )),
            Err(Decline::IrreversibleWide)
        );
        assert_eq!(
            wide(include_bytes!("../../testdata/rgb8.j2k")),
            Err(Decline::NotPlain(NotPlain::Precision(8)))
        );
    }

    /// Every code-block style bit but predictable termination declines,
    /// as the library's parser refuses it.
    #[test]
    fn code_block_styles_other_than_predictable_termination_decline() {
        let rgb = include_bytes!("../../testdata/rgb8.j2k");
        for style in [0x01, 0x02, 0x04, 0x08, 0x20, 0x3E] {
            assert_eq!(
                verdict(&with_style(rgb, style)),
                Err(Decline::CodeBlockStyle(style)),
                "{style:#04x}"
            );
        }
    }

    /// Samples the device would return unscaled, unconverted or without
    /// their channel mapping decline.
    #[test]
    fn samples_other_than_the_plain_components_decline() {
        use hayro_jpeg2000::NotPlain;
        let gray16 = include_bytes!("../../testdata/gray16.jp2");
        assert_eq!(
            verdict(gray16),
            Err(Decline::NotPlain(NotPlain::Precision(16)))
        );
        // Sub-sampled chroma is read as sYCC unless the dictionary names
        // the colour channels.
        let subsampled = include_bytes!("../../testdata/subsampled_rgb.j2k");
        assert_eq!(
            verdict(subsampled),
            Err(Decline::NotPlain(NotPlain::ColourConversion))
        );
        let settings = hayro_jpeg2000::DecodeSettings::default();
        let rgb = hayro_jpeg2000::Image::from_codestream(subsampled, 3, &settings).expect("parses");
        let survey = rgb.survey().expect("tile-part headers parse");
        assert_eq!(
            takes(&rgb, &survey, false),
            Err(Decline::NotPlain(NotPlain::SubSampled))
        );
        assert_eq!(
            verdict(include_bytes!("../../testdata/gray4.jp2")),
            Err(Decline::NotPlain(NotPlain::Precision(4)))
        );
        assert_eq!(
            verdict(include_bytes!("../../testdata/rgba8.jp2")),
            Err(Decline::NotPlain(NotPlain::Alpha))
        );
    }

    /// The decoder is obtained from its source only for a codestream the
    /// device takes, after the checks that need no decoder; an image the
    /// device declines never builds one.
    #[test]
    fn the_source_is_called_only_for_a_codestream_the_device_takes() {
        use std::sync::atomic::AtomicUsize;
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        fn source() -> Option<NvJpeg2kDecoder> {
            assert_eq!(CALLS.fetch_add(1, Ordering::Relaxed), 0, "called once");
            None
        }
        let opts = super::super::JpxOptions {
            indices: false,
            smask_in_data: 0,
            matte: None,
            colour_key: false,
            wide: false,
            channels: None,
        };
        // Dictionary dimensions above any threshold; the codestreams are
        // 8 × 8.
        let decode = |bytes: &[u8]| {
            let mut decoder = None;
            let mut slot = Some(source as Jpeg2kSource);
            let mut device = DeviceJpx::new(&mut decoder, Some(&mut slot), 0).expect("a source");
            let samples =
                super::super::decode_jpx(bytes, 65_535, 65_535, None, &opts, Some(&mut device))
                    .expect("decodes on the CPU");
            assert!(samples.partial.is_none());
            slot.is_none()
        };
        let rgb = include_bytes!("../../testdata/rgb8.j2k");
        assert!(!decode(&with_style(rgb, 0x3E)), "declined: source kept");
        assert!(!decode(include_bytes!("../../testdata/gray16.jp2")));
        assert_eq!(CALLS.load(Ordering::Relaxed), 0);
        assert!(decode(rgb), "taken: source consumed");
        assert_eq!(CALLS.load(Ordering::Relaxed), 1);
    }

    /// A codestream that ends early is not decoded on the device: cut
    /// short, it is drawn from the CPU decode and marked partial; ended
    /// before its tile's data, it is refused.  `rgb8.j2k` holds one
    /// tile-part, its `SOD` at byte 131.
    #[test]
    fn a_truncated_codestream_is_judged_by_the_cpu_decode() {
        fn source() -> Option<NvJpeg2kDecoder> {
            panic!("no decoder is built for a codestream that ends early")
        }
        let rgb = include_bytes!("../../testdata/rgb8.j2k");
        let decode = |len: usize| {
            let mut decoder = None;
            let mut slot = Some(source as Jpeg2kSource);
            let mut device = DeviceJpx::new(&mut decoder, Some(&mut slot), 0).expect("a source");
            let opts = super::super::JpxOptions {
                indices: false,
                smask_in_data: 0,
                matte: None,
                colour_key: false,
                wide: false,
                channels: None,
            };
            super::super::decode_jpx(&rgb[..len], 65_535, 65_535, None, &opts, Some(&mut device))
        };
        let reason = decode(170).expect("cut short").partial.expect("partial");
        assert!(
            reason.contains("1 of its 1 tiles are cut short"),
            "{reason}"
        );
        let err = decode(133).expect_err("no tile data");
        assert!(err.contains("1 of its 1 tiles hold no data"), "{err}");
    }

    /// Without a decoder or a source there is no device to offer.
    #[test]
    fn no_device_without_a_decoder_or_a_source() {
        assert!(DeviceJpx::new(&mut None, None, 0).is_none());
        assert!(DeviceJpx::new(&mut None, Some(&mut None), 0).is_none());
    }

    /// Each kind of decline is reported at warn once.
    #[test]
    fn decline_kinds_are_distinct_bits() {
        use hayro_jpeg2000::NotPlain as N;
        let all = [
            Decline::NotPlain(N::ColourConversion),
            Decline::NotPlain(N::Alpha),
            Decline::NotPlain(N::Palette),
            Decline::NotPlain(N::ChannelDefinitions),
            Decline::NotPlain(N::Components(2)),
            Decline::NotPlain(N::Precision(4)),
            Decline::NotPlain(N::Signed),
            Decline::NotPlain(N::SubSampled),
            Decline::CodeBlockStyle(1),
            Decline::IrreversibleWide,
        ];
        let union = all.iter().fold(0u16, |acc, d| {
            assert_eq!(acc & d.kind(), 0, "{d}");
            acc | d.kind()
        });
        assert_eq!(union.count_ones() as usize, all.len());
    }

    /// The equivalence gate of the 16-bit device path: every codestream of
    /// 9 to 16 bits decodes on the device, widened, to the samples the CPU
    /// path returns at 16 bits, under the reversible wavelet; under the
    /// irreversible one the samples differ, and `takes` declines them.  The
    /// codestreams are the repository's and those of the directory
    /// `RR_JPX_GATE_DIR` names, if any.  Needs a CUDA device.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn wide_device_samples_equal_the_cpu_samples_under_the_reversible_wavelet() {
        use crate::resources::image::ImageData;
        const STREAMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../jpeg2000/tests/streams/");
        let mut fixtures: Vec<(String, Vec<u8>)> = [
            "codestream_precision_10.j2k",
            "codestream_precision_12.j2k",
            "codestream_precision_16.j2k",
            "packets_lblock_large.j2k",
            "packets_passes_codeword_long.j2k",
            "quantization_irreversible_12bit.j2k",
            "quantization_irreversible_16bit.j2k",
        ]
        .iter()
        .map(|n| {
            (
                (*n).to_owned(),
                std::fs::read(format!("{STREAMS}{n}")).expect("fixture"),
            )
        })
        .collect();
        fixtures.push((
            "gray16.jp2".to_owned(),
            include_bytes!("../../testdata/gray16.jp2").to_vec(),
        ));
        if let Ok(dir) = std::env::var("RR_JPX_GATE_DIR") {
            let mut extra: Vec<_> = std::fs::read_dir(dir)
                .expect("RR_JPX_GATE_DIR")
                .map(|e| e.expect("entry").path())
                .collect();
            extra.sort();
            for path in extra {
                let name = path
                    .file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned();
                fixtures.push((name, std::fs::read(&path).expect("read")));
            }
        }
        let opts = super::super::JpxOptions {
            indices: false,
            smask_in_data: 0,
            matte: None,
            colour_key: false,
            wide: true,
            channels: None,
        };
        let mut decoder =
            Some(NvJpeg2kDecoder::new(0).expect("gpu-validation needs a CUDA device"));
        let (mut reversible, mut irreversible) = (0, 0);
        for (name, bytes) in &fixtures {
            let img = image(bytes);
            let survey = img.survey().expect("tile-part headers parse");
            let verdict = takes(&img, &survey, true);
            if let Err(why) = &verdict
                && *why != Decline::IrreversibleWide
            {
                eprintln!("{name}: not compared ({why})");
                continue;
            }
            let (w, h) = (img.width(), img.height());
            let cpu = super::super::decode_jpx(bytes, w, h, None, &opts, None).expect("CPU decode");
            let mut device = DeviceJpx::new(&mut decoder, None, 0).expect("a decoder");
            let gpu = decode(bytes, (w, h), true, &mut device).expect("device decode");
            let (ImageData::Cpu16(cpu), ImageData::Cpu16(gpu)) = (&cpu.data, &gpu.data) else {
                panic!("{name}: 16-bit samples on both paths");
            };
            assert_eq!(cpu.len(), gpu.len(), "{name}: sample count");
            let differing = cpu.iter().zip(gpu).filter(|(a, b)| a != b).count();
            let max_delta = cpu
                .iter()
                .zip(gpu)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            let wavelet = if survey.irreversible { "9-7" } else { "5-3" };
            eprintln!(
                "{name}: {wavelet}, {differing} of {} samples differ (max delta {max_delta})",
                cpu.len()
            );
            if survey.irreversible {
                irreversible += 1;
                assert!(
                    differing > 0,
                    "{name}: equal under 9-7, which takes declines"
                );
                assert_eq!(takes(&img, &survey, true), Err(Decline::IrreversibleWide));
            } else {
                reversible += 1;
                assert_eq!(differing, 0, "{name}: the device differs under 5-3");
                assert_eq!(verdict, Ok(()), "{name}");
            }
        }
        assert!(
            reversible >= 5 && irreversible >= 2,
            "{reversible} / {irreversible}"
        );
    }
}
