//! `JPXDecode` on the device: the codestreams nvJPEG2000 reconstructs as
//! the CPU decoder does, and the decode itself, into 8-bit samples or, for
//! a codestream of more than 8 bits kept at 16, into 16-bit ones.

use std::sync::atomic::{AtomicU32, Ordering};

use gpu::nvjpeg2k::{Jpeg2kColorSpace, NvJpeg2kDecoder, NvJpeg2kError};

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
    /// The irreversible 9-7 wavelet, under which the device's dequantised,
    /// inverse-transformed samples differ from the CPU decoder's by up to
    /// one level at 8 bits; at more than 8 bits it also rounds them to
    /// integers before they are widened to 16 bits, where the CPU widens
    /// the reconstructed values.
    Irreversible,
    /// A draw the CPU decodes at a reduced resolution level, of other
    /// dimensions than the device's full-resolution decode; decided from
    /// the headers, before the device decodes.
    Reduced,
    /// A device decode whose dimensions are not those of the CPU decode.
    SizeMismatch { device: (u32, u32), cpu: (u32, u32) },
    /// SOP marker segments allowed before the packets (`Scod` bit 1),
    /// present or not; the library's parser fails on the codestream.
    SopMarkers,
    /// Packet headers packed in PPM or PPT marker segments, which the
    /// library's parser refuses.
    PackedPacketHeaders,
    /// An RGN marker segment, which the library's parser refuses.
    RegionOfInterest,
    /// A CRG marker segment, which the library's parser refuses.
    ComponentRegistration,
    /// A marker of the range 0xFF30 to 0xFF3F, which the library's parser
    /// refuses.
    ReservedMarker,
    /// Components of one tile with differing numbers of decomposition
    /// levels, which the library's parser refuses.
    ComponentLevelsDiffer,
    /// The multiple component transformation selected for fewer than three
    /// components, which the library's parser refuses.
    IdleComponentTransform,
    /// A COD, COC, QCD or QCC marker segment in a tile-part other than the
    /// first of its tile, which the library's parser refuses.
    LateCodingStyle,
    /// A tile-part whose declared length ends before its SOD marker; the
    /// library's parser fails on the codestream.
    TilePartWithoutData,
    /// No EOC marker after a last tile-part of declared length; the
    /// library's parser fails on the codestream.
    Unterminated,
    /// An image area offset from the reference grid origin over more than
    /// one tile, on which the library's parser fails or which it decodes
    /// to other samples.
    TiledImageOffset,
}

impl Decline {
    /// The bit of [`REPORTED`] that stands for the kind of the decline.
    const fn kind(&self) -> u32 {
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
            Self::Irreversible => 9,
            Self::Reduced => 10,
            Self::SizeMismatch { .. } => 11,
            Self::SopMarkers => 12,
            Self::PackedPacketHeaders => 13,
            Self::RegionOfInterest => 14,
            Self::ComponentRegistration => 15,
            Self::ReservedMarker => 16,
            Self::ComponentLevelsDiffer => 17,
            Self::IdleComponentTransform => 18,
            Self::LateCodingStyle => 19,
            Self::TilePartWithoutData => 20,
            Self::Unterminated => 21,
            Self::TiledImageOffset => 22,
        };
        1 << bit
    }
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPlain(why) => why.fmt(f),
            Self::CodeBlockStyle(style) => write!(f, "code-block style {style:#04x}"),
            Self::Irreversible => f.write_str("the irreversible 9-7 wavelet"),
            Self::Reduced => f.write_str("a draw decoded at a reduced resolution level"),
            Self::SizeMismatch { device, cpu } => write!(
                f,
                "a codestream it decodes at {} × {} where the CPU decodes {} × {}",
                device.0, device.1, cpu.0, cpu.1
            ),
            Self::SopMarkers => f.write_str("SOP marker segments"),
            Self::PackedPacketHeaders => f.write_str("packed packet headers (PPM or PPT)"),
            Self::RegionOfInterest => f.write_str("a region of interest (RGN)"),
            Self::ComponentRegistration => f.write_str("a component registration (CRG)"),
            Self::ReservedMarker => f.write_str("a reserved marker (0xFF30 to 0xFF3F)"),
            Self::ComponentLevelsDiffer => {
                f.write_str("components of one tile with differing decomposition levels")
            }
            Self::IdleComponentTransform => {
                f.write_str("a component transformation selected for fewer than three components")
            }
            Self::LateCodingStyle => {
                f.write_str("a coding style in a tile-part other than the first of its tile")
            }
            Self::TilePartWithoutData => f.write_str("a tile-part that ends before its data"),
            Self::Unterminated => f.write_str("a codestream without an EOC marker"),
            Self::TiledImageOffset => {
                f.write_str("an image area offset from the grid origin over more than one tile")
            }
        }
    }
}

/// The kinds of [`Decline`] already reported in this process.
static REPORTED: AtomicU32 = AtomicU32::new(0);

/// `Err` naming the first property of `img` for which nvJPEG2000 would
/// not return the samples the CPU decoder returns, from its headers and
/// `survey` (see [`hayro_jpeg2000::Image::survey`]): it returns the
/// reconstructed components of a codestream of one or three unsigned
/// components on the image grid, of 8 bits, or of 9 to 16 bits when the
/// samples are kept at 16 (`wide`), under the reversible 5-3 wavelet in
/// force throughout; it parses no code-block style but predictable
/// termination; and it parses none of the markers and structures of the
/// variants of [`Decline`] from [`Decline::SopMarkers`] on.
pub(super) fn takes(
    img: &hayro_jpeg2000::Image<'_>,
    survey: &hayro_jpeg2000::Survey,
    wide: bool,
) -> Result<(), Decline> {
    if wide {
        img.plain_wide().map_err(Decline::NotPlain)?;
    } else {
        img.plain_u8().map_err(Decline::NotPlain)?;
    }
    if survey.irreversible {
        return Err(Decline::Irreversible);
    }
    let styles = survey.code_block_styles & !DEVICE_CODE_BLOCK_STYLES;
    if styles != 0 {
        return Err(Decline::CodeBlockStyle(survey.code_block_styles));
    }
    let structures = [
        (survey.sop_markers, Decline::SopMarkers),
        (survey.packed_packet_headers, Decline::PackedPacketHeaders),
        (survey.region_of_interest, Decline::RegionOfInterest),
        (
            survey.component_registration,
            Decline::ComponentRegistration,
        ),
        (survey.reserved_markers, Decline::ReservedMarker),
        (
            survey.component_levels_differ,
            Decline::ComponentLevelsDiffer,
        ),
        (
            survey.idle_component_transform,
            Decline::IdleComponentTransform,
        ),
        (survey.late_coding_style, Decline::LateCodingStyle),
        (survey.tile_part_without_data, Decline::TilePartWithoutData),
        (survey.unterminated, Decline::Unterminated),
        (
            survey.image_offset && survey.completeness.tiles > 1,
            Decline::TiledImageOffset,
        ),
    ];
    structures
        .into_iter()
        .find_map(|(present, why)| present.then_some(why))
        .map_or(Ok(()), Err)
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

/// The device decode of `codestream` by `dec`: into 8-bit samples, or with
/// `wide` into 16-bit ones, each component's samples widened from its
/// precision as the CPU path widens them.
fn device_decode(
    dec: &mut NvJpeg2kDecoder,
    codestream: &[u8],
    wide: bool,
) -> Result<(u32, u32, Jpeg2kColorSpace, DeviceSamples), NvJpeg2kError> {
    if wide {
        dec.decode_sync_u16(codestream).map(|img| {
            let samples = widened(&img.data, &img.precisions);
            (
                img.width,
                img.height,
                img.color_space,
                DeviceSamples::Wide(samples),
            )
        })
    } else {
        dec.decode_sync(codestream).map(|img| {
            (
                img.width,
                img.height,
                img.color_space,
                DeviceSamples::Narrow(img.data),
            )
        })
    }
}

/// Decode `codestream`, from its SOC marker, of a `pdf_w × pdf_h` image
/// dictionary on the device, obtaining the decoder from `device` first
/// ([`device_decode`]).  `None` when no decoder is available, and after a
/// warning when the decode fails or its dimensions are not `cpu_size`,
/// those of the CPU decode at the draw's resolution level; the caller
/// decodes on the CPU.
pub(super) fn decode(
    codestream: &[u8],
    (pdf_w, pdf_h): (u32, u32),
    cpu_size: (u32, u32),
    wide: bool,
    device: &mut DeviceJpx<'_>,
) -> Option<Samples> {
    let dec = device.decoder()?;
    let (width, height, color_space, data) = match device_decode(dec, codestream, wide) {
        Ok(img) => img,
        Err(e) => {
            log::warn!("image: JPXDecode: nvJPEG2000 failed ({e}); decoding on the CPU");
            return None;
        }
    };
    if (width, height) != cpu_size {
        report(&Decline::SizeMismatch {
            device: (width, height),
            cpu: cpu_size,
        });
        return None;
    }
    if width != pdf_w || height != pdf_h {
        log::debug!(
            "image: JPXDecode: the image dictionary says {pdf_w}×{pdf_h} and nvJPEG2000 \
             {width}×{height}; drawing at the codestream's size"
        );
    }
    log::debug!("image: JPXDecode: {width}×{height} decoded on nvJPEG2000");
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

    /// Codestreams of 8 bits under the irreversible wavelet decline, one
    /// component or three under the irreversible colour transform.
    #[test]
    fn eight_bit_codestreams_under_the_irreversible_wavelet_decline() {
        assert_eq!(
            verdict(include_bytes!(
                "../../../../../../jpeg2000/tests/streams/quantization_irreversible.j2k"
            )),
            Err(Decline::Irreversible)
        );
        assert_eq!(
            verdict(include_bytes!(
                "../../../../../../jpeg2000/tests/streams/mct_ict.j2k"
            )),
            Err(Decline::Irreversible)
        );
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
            Err(Decline::Irreversible)
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

    /// The decoder's test stream `name`.
    fn test_stream(name: &str) -> Vec<u8> {
        const STREAMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../jpeg2000/tests/streams/");
        std::fs::read(format!("{STREAMS}{name}")).expect("test stream")
    }

    /// Codestreams with a marker or structure the library's parser refuses
    /// or fails on decline, each for its own; streams that differ from one
    /// of them only by a structure the device decodes are taken.
    #[test]
    fn structures_the_device_does_not_parse_decline() {
        let declined = [
            ("packets_sop.j2k", Decline::SopMarkers),
            ("packets_sop_missing.j2k", Decline::SopMarkers),
            ("packets_ppm.j2k", Decline::PackedPacketHeaders),
            ("packets_ppt.j2k", Decline::PackedPacketHeaders),
            ("roi_reversible.j2k", Decline::RegionOfInterest),
            ("roi_tile_part_override.j2k", Decline::RegionOfInterest),
            ("codestream_crg.j2k", Decline::ComponentRegistration),
            ("codestream_reserved_marker.j2k", Decline::ReservedMarker),
            (
                "codestream_levels_per_component.j2k",
                Decline::ComponentLevelsDiffer,
            ),
            (
                "mct_one_component_ignored.j2k",
                Decline::IdleComponentTransform,
            ),
            (
                "tile_parts_cod_qcd_in_later_tile_part.j2k",
                Decline::LateCodingStyle,
            ),
            (
                "tile_parts_sot_only_tile_part.j2k",
                Decline::TilePartWithoutData,
            ),
            ("tile_parts_missing_eoc.j2k", Decline::Unterminated),
            ("tile_parts_garbage_before_eoc.j2k", Decline::Unterminated),
            ("codestream_offset_large.j2k", Decline::TiledImageOffset),
            (
                "codestream_tiles_offset_many.j2k",
                Decline::TiledImageOffset,
            ),
        ];
        for (name, why) in declined {
            assert_eq!(verdict(&test_stream(name)), Err(why), "{name}");
        }
        for name in [
            // SOP markers absent, EPH markers present.
            "packets_eph.j2k",
            // Decomposition levels that differ between tiles only.
            "codestream_tile_cod_fewer_levels.j2k",
            // Tile-parts of length 0, running to the end of the
            // codestream with no EOC marker; bytes after the EOC marker.
            "tile_parts_psot_zero.j2k",
            "tile_parts_psot_zero_padding.j2k",
            "tile_parts_bytes_after_eoc.j2k",
            // An image area offset from the grid origin in one tile.
            "codestream_offset.j2k",
            "codestream_single_row_odd_offset.j2k",
            // Several tile-parts of one tile, the coding style in the first.
            "tile_parts_split_resolution.j2k",
        ] {
            assert_eq!(verdict(&test_stream(name)), Ok(()), "{name}");
        }
    }

    /// The device is given the codestream of a JP2 file, from its SOC
    /// marker, whatever the file's boxes hold; a raw codestream is given
    /// whole.
    #[test]
    fn the_device_is_given_the_codestream_of_a_jp2_file() {
        let bytes = test_stream("jp2_box_longer_than_file.jp2");
        let img = image(&bytes);
        let codestream = img.codestream();
        assert_eq!(codestream[..2], [0xFF, 0x4F]);
        let at = bytes
            .windows(codestream.len())
            .position(|w| w == codestream)
            .expect("a slice of the file");
        assert!(at > 0, "after the file's boxes");
        let raw = include_bytes!("../../testdata/rgb8.j2k");
        assert_eq!(image(raw).codestream(), raw);
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
            Decline::Irreversible,
            Decline::Reduced,
            Decline::SizeMismatch {
                device: (2, 2),
                cpu: (1, 1),
            },
            Decline::SopMarkers,
            Decline::PackedPacketHeaders,
            Decline::RegionOfInterest,
            Decline::ComponentRegistration,
            Decline::ReservedMarker,
            Decline::ComponentLevelsDiffer,
            Decline::IdleComponentTransform,
            Decline::LateCodingStyle,
            Decline::TilePartWithoutData,
            Decline::Unterminated,
            Decline::TiledImageOffset,
        ];
        let union = all.iter().fold(0u32, |acc, d| {
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
        fixtures.extend(gate_dir_streams());
        let opts = jpx_opts(true);
        let mut decoder =
            Some(NvJpeg2kDecoder::new(0).expect("gpu-validation needs a CUDA device"));
        let (mut reversible, mut irreversible) = (0, 0);
        for (name, bytes) in &fixtures {
            let img = image(bytes);
            let survey = img.survey().expect("tile-part headers parse");
            let verdict = takes(&img, &survey, true);
            if let Err(why) = &verdict
                && *why != Decline::Irreversible
            {
                eprintln!("{name}: not compared ({why})");
                continue;
            }
            let (w, h) = (img.width(), img.height());
            let cpu = super::super::decode_jpx(bytes, w, h, None, &opts, None).expect("CPU decode");
            let mut device = DeviceJpx::new(&mut decoder, None, 0).expect("a decoder");
            let gpu =
                decode(img.codestream(), (w, h), (w, h), true, &mut device).expect("device decode");
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
                assert_eq!(takes(&img, &survey, true), Err(Decline::Irreversible));
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

    /// The options of a plain image, its samples kept at 16 bits when
    /// `wide`.
    #[cfg(feature = "gpu-validation")]
    const fn jpx_opts(wide: bool) -> super::super::JpxOptions {
        super::super::JpxOptions {
            indices: false,
            smask_in_data: 0,
            matte: None,
            colour_key: false,
            wide,
            channels: None,
        }
    }

    /// The files of the directory `RR_JPX_GATE_DIR` names, by name; none
    /// when it is unset.
    #[cfg(feature = "gpu-validation")]
    fn gate_dir_streams() -> Vec<(String, Vec<u8>)> {
        let Ok(dir) = std::env::var("RR_JPX_GATE_DIR") else {
            return Vec::new();
        };
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .expect("RR_JPX_GATE_DIR")
            .map(|e| e.expect("entry").path())
            .collect();
        paths.sort();
        paths
            .into_iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned();
                let bytes = std::fs::read(&path).expect("read");
                (name, bytes)
            })
            .collect()
    }

    /// The `JPXDecode` streams of the image `XObject`s the pages of the PDF
    /// at `path` name, each once, by object number; none when the file is
    /// absent.  A stream whose filter chain holds more than `JPXDecode` is
    /// skipped, named, since its content is not a codestream as stored.
    #[cfg(feature = "gpu-validation")]
    fn pdf_jpx_streams(path: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        use pdf::Object;
        if !path.exists() {
            eprintln!("skipping {}: absent", path.display());
            return Vec::new();
        }
        let doc = pdf::Document::open(path).expect("the PDF opens");
        let jpx = |dict: &pdf::Dictionary| match dict.get(b"Filter") {
            Some(Object::Name(n)) => n == b"JPXDecode",
            Some(Object::Array(a)) => {
                a.len() == 1 && a[0].as_name() == Some(b"JPXDecode".as_slice())
            }
            _ => false,
        };
        let chained = |dict: &pdf::Dictionary| {
            matches!(dict.get(b"Filter"), Some(Object::Array(a))
                if a.len() > 1 && a.iter().any(|f| f.as_name() == Some(b"JPXDecode".as_slice())))
        };
        let mut visited = std::collections::BTreeSet::new();
        let mut streams = Vec::new();
        for (_, page) in doc.get_pages() {
            let xobjects = doc
                .get_page_resource_dict(page, b"XObject")
                .expect("resources");
            for (_, value) in &xobjects {
                let Object::Reference(id) = value else {
                    continue;
                };
                if !visited.insert(id.0) {
                    continue;
                }
                if let Ok(obj) = doc.get_object(*id)
                    && let Object::Stream(stream) = &*obj
                {
                    if jpx(&stream.dict) {
                        streams.push((id.0, stream.content.clone()));
                    } else if chained(&stream.dict) {
                        eprintln!(
                            "{}: object {}: skipped, a filter chain",
                            path.display(),
                            id.0
                        );
                    }
                }
            }
        }
        streams.sort_by_key(|&(num, _)| num);
        streams
            .into_iter()
            .map(|(num, bytes)| (format!("object {num}"), bytes))
            .collect()
    }

    /// The files of the JPEG 2000 decoder's test streams, by name.
    #[cfg(feature = "gpu-validation")]
    fn decoder_test_streams() -> Vec<(String, Vec<u8>)> {
        const STREAMS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../jpeg2000/tests/streams/");
        let mut paths: Vec<_> = std::fs::read_dir(STREAMS)
            .expect("test streams")
            .map(|e| e.expect("entry").path())
            .collect();
        paths.sort();
        paths
            .iter()
            .map(|p| {
                let name = p.file_name().expect("name").to_string_lossy().into_owned();
                (name, std::fs::read(p).expect("read"))
            })
            .collect()
    }

    /// The samples of the codestream `bytes` of a `w × h` image at 8 bits,
    /// as the CPU path and as the device decode them: `(differing samples,
    /// largest difference, samples)`.  `None` when the CPU path refuses the
    /// codestream or the device decode fails.
    #[cfg(feature = "gpu-validation")]
    fn compare_eight_bit(
        bytes: &[u8],
        (w, h): (u32, u32),
        decoder: &mut Option<NvJpeg2kDecoder>,
    ) -> Option<(usize, u8, usize)> {
        use crate::resources::image::ImageData;
        let opts = jpx_opts(false);
        let cpu = super::super::decode_jpx(bytes, w, h, None, &opts, None).ok()?;
        let mut device = DeviceJpx::new(decoder, None, 0).expect("a decoder");
        let img = image(bytes);
        let gpu = decode(img.codestream(), (w, h), (w, h), false, &mut device)?;
        let (ImageData::Cpu(cpu), ImageData::Cpu(gpu)) = (&cpu.data, &gpu.data) else {
            panic!("8-bit samples on both paths");
        };
        assert_eq!(cpu.len(), gpu.len(), "sample count");
        let deltas = cpu.iter().zip(gpu.iter()).map(|(a, b)| a.abs_diff(*b));
        let differing = deltas.clone().filter(|&d| d > 0).count();
        Some((differing, deltas.max().unwrap_or(0), cpu.len()))
    }

    /// The equivalence gate of the 8-bit device path: every codestream
    /// `takes` accepts at 8 bits decodes on the device to the samples the
    /// CPU path returns, or the device decode fails and the CPU decodes
    /// it; and under the irreversible wavelet, which `takes` declines, the
    /// device samples differ for some codestream of each source compared.
    /// The codestreams are the JPEG 2000 decoder's test streams, the image
    /// streams of `corpus-10-scan-jbig2-jpx.pdf` when the fixture is
    /// present, and those of the directory `RR_JPX_GATE_DIR` names, if any.
    /// Needs a CUDA device.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn eight_bit_device_samples_equal_the_cpu_samples_or_decline() {
        /// Declined codestreams of a source decoded on both paths, at most.
        const DECLINED_COMPARED: usize = 4;
        const CORPUS_10: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/corpus-10-scan-jbig2-jpx.pdf"
        );
        let sources = [
            ("test streams", decoder_test_streams()),
            (
                "corpus-10",
                pdf_jpx_streams(std::path::Path::new(CORPUS_10)),
            ),
            ("RR_JPX_GATE_DIR", gate_dir_streams()),
        ];
        let mut decoder =
            Some(NvJpeg2kDecoder::new(0).expect("gpu-validation needs a CUDA device"));
        let mut taken = 0;
        for (source, fixtures) in &sources {
            let (mut declined, mut compared, mut differing_streams) = (0, 0, 0);
            for (name, bytes) in fixtures {
                let settings = hayro_jpeg2000::DecodeSettings::default();
                let Ok(img) = hayro_jpeg2000::Image::new(bytes, &settings) else {
                    continue;
                };
                let Ok(survey) = img.survey() else {
                    continue;
                };
                let verdict = takes(&img, &survey, false);
                match &verdict {
                    Ok(()) if survey.completeness.is_complete() => {}
                    Err(Decline::Irreversible) => {
                        declined += 1;
                        if compared == DECLINED_COMPARED {
                            continue;
                        }
                    }
                    _ => continue,
                }
                let size = (img.width(), img.height());
                let Some((differing, max_delta, samples)) =
                    compare_eight_bit(bytes, size, &mut decoder)
                else {
                    eprintln!("{source}: {name}: not decoded on both paths");
                    continue;
                };
                if verdict.is_ok() {
                    taken += 1;
                    assert_eq!(
                        differing, 0,
                        "{source}: {name}: taken, and {differing} samples differ \
                         (max delta {max_delta})"
                    );
                } else {
                    compared += 1;
                    differing_streams += usize::from(differing > 0);
                    eprintln!(
                        "{source}: {name}: 9-7, declined; {differing} of {samples} samples \
                         differ (max delta {max_delta})"
                    );
                }
            }
            eprintln!(
                "{source}: {} codestreams, {declined} declined under 9-7 ({compared} compared, \
                 {differing_streams} differing), {taken} taken in all so far",
                fixtures.len()
            );
            assert!(
                compared == 0 || differing_streams > 0,
                "{source}: every declined 9-7 codestream compared decodes equal"
            );
        }
        assert!(taken >= 20, "{taken} codestreams taken");
    }

    /// What became of a codestream offered to the device.
    #[cfg(feature = "gpu-validation")]
    #[derive(Debug)]
    enum Outcome {
        /// The headers or the tile-part headers do not parse.
        Unparsed,
        /// `takes` declines it.
        Declined(String),
        /// It ends early, so the CPU decodes it.
        Incomplete,
        /// The CPU path refuses it.
        CpuRefused,
        /// The device decode fails.
        DeviceFailed(String),
        /// The device decodes it to other samples or dimensions.
        DeviceDiffers(String),
        /// The device decodes it to the CPU's samples.
        DeviceOk,
    }

    /// The outcome of the codestream `bytes` on the device: kept at 16
    /// bits when its precision exceeds 8, as on a 16-bit canvas, else at 8.
    #[cfg(feature = "gpu-validation")]
    fn outcome(bytes: &[u8], decoder: &mut NvJpeg2kDecoder) -> Outcome {
        use crate::resources::image::ImageData;
        let settings = hayro_jpeg2000::DecodeSettings::default();
        let Ok(img) = hayro_jpeg2000::Image::new(bytes, &settings) else {
            return Outcome::Unparsed;
        };
        let Ok(survey) = img.survey() else {
            return Outcome::Unparsed;
        };
        let wide = img.original_bit_depth() > 8;
        match takes(&img, &survey, wide) {
            Err(Decline::Unterminated) | Ok(()) if !survey.completeness.is_complete() => {
                return Outcome::Incomplete;
            }
            Err(why) => return Outcome::Declined(why.to_string()),
            Ok(()) => {}
        }
        let (w, h) = (img.width(), img.height());
        let Ok(cpu) = super::super::decode_jpx(bytes, w, h, None, &jpx_opts(wide), None) else {
            return Outcome::CpuRefused;
        };
        let (dw, dh, _, data) = match device_decode(decoder, img.codestream(), wide) {
            Ok(d) => d,
            Err(e) => return Outcome::DeviceFailed(e.to_string()),
        };
        if (dw, dh) != (cpu.width, cpu.height) {
            return Outcome::DeviceDiffers(format!("{dw}×{dh}, CPU {}×{}", cpu.width, cpu.height));
        }
        let equal = match (&cpu.data, &data) {
            (ImageData::Cpu(a), DeviceSamples::Narrow(b)) => a == b,
            (ImageData::Cpu16(a), DeviceSamples::Wide(b)) => a == b,
            _ => false,
        };
        if equal {
            Outcome::DeviceOk
        } else {
            Outcome::DeviceDiffers("samples".to_owned())
        }
    }

    /// Every one of the JPEG 2000 decoder's test streams that `takes`
    /// accepts, complete and decoded by the CPU path, decodes on the device
    /// to the CPU's samples: kept at 16 bits when of more than 8, else at
    /// 8.  Needs a CUDA device.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn every_test_stream_taken_decodes_on_the_device() {
        let mut decoder = NvJpeg2kDecoder::new(0).expect("gpu-validation needs a CUDA device");
        let mut tally = std::collections::BTreeMap::<&str, usize>::new();
        let mut wrong = Vec::new();
        let mut streams = decoder_test_streams();
        streams.extend(gate_dir_streams());
        for (name, bytes) in streams {
            let (kind, detail) = match outcome(&bytes, &mut decoder) {
                Outcome::Unparsed => ("unparsed", String::new()),
                Outcome::Declined(why) => ("declined", why),
                Outcome::Incomplete => ("incomplete", String::new()),
                Outcome::CpuRefused => ("cpu-refused", String::new()),
                Outcome::DeviceOk => ("device-ok", String::new()),
                Outcome::DeviceFailed(e) => ("device-failed", e),
                Outcome::DeviceDiffers(what) => ("device-differs", what),
            };
            eprintln!("{name}: {kind} {detail}");
            if kind.starts_with("device-") && kind != "device-ok" {
                wrong.push(format!("{name}: {kind} {detail}"));
            }
            *tally.entry(kind).or_default() += 1;
        }
        eprintln!("{tally:?}");
        assert!(wrong.is_empty(), "{wrong:#?}");
        assert!(
            tally.get("device-ok").copied().unwrap_or(0) >= 20,
            "{tally:?}"
        );
    }

    /// A draw at half the image's size or less, which the CPU decodes at a
    /// reduced resolution level, returns the CPU's samples with a decoder
    /// attached; a larger draw decodes on the device, to the same samples.
    /// Needs a CUDA device.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn a_draw_the_cpu_decodes_reduced_returns_the_cpu_samples() {
        use crate::resources::image::ImageData;
        let bytes = include_bytes!("../../../../../../jpeg2000/tests/streams/mct_rct.j2k");
        let opts = jpx_opts(false);
        let mut decoder =
            Some(NvJpeg2kDecoder::new(0).expect("gpu-validation needs a CUDA device"));
        for draw in [(64, 48), (40, 30), (65, 49), (128, 96)] {
            let cpu = super::super::decode_jpx(bytes, 128, 96, Some(draw), &opts, None)
                .expect("CPU decode");
            let mut device = DeviceJpx::new(&mut decoder, None, 0).expect("a decoder");
            let got =
                super::super::decode_jpx(bytes, 128, 96, Some(draw), &opts, Some(&mut device))
                    .expect("decode");
            assert_eq!((got.width, got.height), (cpu.width, cpu.height), "{draw:?}");
            let (ImageData::Cpu(cpu), ImageData::Cpu(got)) = (&cpu.data, &got.data) else {
                panic!("8-bit samples");
            };
            assert!(cpu == got, "{draw:?}: the samples differ");
        }
    }
}
