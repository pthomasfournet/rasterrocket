//! `JPXDecode` on the device: the codestreams nvJPEG2000 reconstructs as
//! the CPU decoder does, and the decode itself.

use std::sync::atomic::{AtomicU16, Ordering};

use gpu::nvjpeg2k::{Jpeg2kColorSpace, NvJpeg2kDecoder};

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
/// is called at most once.
pub struct DeviceJpx<'a> {
    decoder: &'a mut Option<NvJpeg2kDecoder>,
    source: Option<&'a mut Option<Jpeg2kSource>>,
}

impl<'a> DeviceJpx<'a> {
    /// The decoder in `decoder`, else one from the source in `source`;
    /// `None` when there is neither.
    pub fn new(
        decoder: &'a mut Option<NvJpeg2kDecoder>,
        source: Option<&'a mut Option<Jpeg2kSource>>,
    ) -> Option<Self> {
        let source = source.filter(|s| s.is_some());
        (decoder.is_some() || source.is_some()).then_some(Self { decoder, source })
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
        };
        1 << bit
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPlain(why) => why.fmt(f),
            Self::CodeBlockStyle(style) => write!(f, "code-block style {style:#04x}"),
        }
    }
}

/// The kinds of [`Decline`] already reported in this process.
static REPORTED: AtomicU16 = AtomicU16::new(0);

/// `Err` naming the first property of `img` for which nvJPEG2000 would
/// not return the samples the CPU decoder returns, from its headers and
/// `survey` (see [`hayro_jpeg2000::Image::survey`]): it returns the
/// reconstructed components of a codestream of one or three unsigned
/// 8-bit components on the image grid, and parses no code-block style
/// but predictable termination.
pub(super) fn takes(
    img: &hayro_jpeg2000::Image<'_>,
    survey: &hayro_jpeg2000::Survey,
) -> Result<(), Decline> {
    img.plain_u8().map_err(Decline::NotPlain)?;
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
/// the device, obtaining the decoder from `device` first.  `None`, after a
/// warning, when no decoder is available or the decode fails; the caller
/// decodes on the CPU.
pub(super) fn decode(
    data: &[u8],
    pdf_w: u32,
    pdf_h: u32,
    device: &mut DeviceJpx<'_>,
) -> Option<Samples> {
    let dec = device.decoder()?;
    let img = match dec.decode_sync(data) {
        Ok(img) => img,
        Err(e) => {
            log::warn!("image: JPXDecode: nvJPEG2000 failed ({e}); decoding on the CPU");
            return None;
        }
    };
    if img.width != pdf_w || img.height != pdf_h {
        log::debug!(
            "image: JPXDecode: the image dictionary says {pdf_w}×{pdf_h} and nvJPEG2000 {}×{}; \
             drawing at the codestream's size",
            img.width,
            img.height,
        );
    }
    let layout = match img.color_space {
        Jpeg2kColorSpace::Gray => Layout::Color(1),
        Jpeg2kColorSpace::Rgb => Layout::Color(3),
    };
    Some(Samples::cpu(img.width, img.height, layout, img.data))
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
        takes(&img, &survey)
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
            takes(&rgb, &survey),
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
            let mut device = DeviceJpx::new(&mut decoder, Some(&mut slot)).expect("a source");
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

    /// Without a decoder or a source there is no device to offer.
    #[test]
    fn no_device_without_a_decoder_or_a_source() {
        assert!(DeviceJpx::new(&mut None, None).is_none());
        assert!(DeviceJpx::new(&mut None, Some(&mut None)).is_none());
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
        ];
        let union = all.iter().fold(0u16, |acc, d| {
            assert_eq!(acc & d.kind(), 0, "{d}");
            acc | d.kind()
        });
        assert_eq!(union.count_ones() as usize, all.len());
    }
}
