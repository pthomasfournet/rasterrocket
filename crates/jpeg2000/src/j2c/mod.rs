mod arithmetic_decoder;
mod bitplane;
mod build;
mod codestream;
mod decode;
mod ht;
mod ht_tables;
mod idwt;
pub(crate) mod mct;
mod progression;
mod rect;
mod segment;
mod tag_tree;
mod tile;

use alloc::vec;
use alloc::vec::Vec;

use super::jp2::ImageBoxes;
use super::jp2::colr::{ColorSpace, ColorSpecificationBox, EnumeratedColorspace};
use crate::error::{FormatError, MarkerError, Result, TileError, ValidationError, bail};
use crate::j2c::codestream::markers;
use crate::reader::BitReader;
use crate::{DecodeSettings, Image, resolve_alpha_and_color_space};

use crate::math::{Level, SIMD_WIDTH, Simd, SimdBuffer, dispatch, f32x8};
pub(crate) use bitplane::BITPLANE_BIT_SIZE;
pub(crate) use codestream::Header;
pub use decode::DecoderContext;
pub(crate) use decode::{Deferred, decode};
pub use tile::{Completeness, Survey};

/// The [`Survey`] of the tile-parts of `data`, the codestream after the
/// main header `header`: the tile-part headers are parsed as a decode
/// parses them, and no packet is read.
pub(crate) fn survey(data: &[u8], header: &Header) -> Result<Survey> {
    let tiles = tile::parse(&mut BitReader::new(data), header)?;
    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }
    Ok(Survey::of(&tiles))
}

pub(crate) struct ParsedCodestream<'a> {
    pub(crate) header: Header,
    pub(crate) data: &'a [u8],
}

/// Decoded data for one JPEG2000 component.
#[derive(Debug, Clone)]
pub struct ComponentData {
    pub(crate) container: SimdBuffer<{ SIMD_WIDTH }>,
    pub(crate) bit_depth: u8,
}

impl ComponentData {
    /// The bit depth of this component.
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// The decoded samples of this component.
    pub fn samples(&self) -> &[f32] {
        self.container.truncated()
    }

    /// The samples plus `offset`, rounded to bytes as
    /// `DecodedImage::data_u8` rounds a single 8-bit component.  Each byte
    /// is the one adding `offset` in place and then converting would give.
    pub(crate) fn shifted_u8(&self, offset: f32) -> Vec<u8> {
        let samples = self.samples();
        let mut out = vec![0_u8; samples.len()];
        dispatch!(Level::new(), simd => shifted_u8_impl(simd, samples, offset, &mut out));
        out
    }
}

/// `samples` plus `offset`, rounded to `out` as `round_to_u8` rounds: blocks
/// of 16 through `round_and_interleave` (whose bytes equal
/// `round_to_u8`'s; scale 1 leaves every sample unchanged), the rest one by
/// one.
#[inline(always)]
fn shifted_u8_impl<S: Simd>(simd: S, samples: &[f32], offset: f32, out: &mut [u8]) {
    let blocks = samples.len().min(out.len()) / 16 * 16;
    for (block, out) in samples[..blocks]
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<16>().0)
    {
        let shifted = crate::math::join16(
            [0, 8].map(|half| f32x8::from_slice(simd, &block[half..half + 8]) + offset),
        );
        crate::math::round_and_interleave(simd, [&shifted], 1.0, out);
    }
    for (&v, out) in samples[blocks..].iter().zip(&mut out[blocks..]) {
        *out = crate::math::round_to_u8(v + offset);
    }
}

/// Parse a raw codestream.  With `color_channels`, the image has that many
/// colour channels, in the colour space of that channel count (gray, RGB,
/// CMYK or unknown), and an alpha channel when the codestream has one more
/// component; the codestream must have `color_channels` components or one
/// more.  Without, the colour space is inferred from the components.
pub(crate) fn parse<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
    color_channels: Option<u8>,
) -> Result<Image<'a>> {
    let parsed_codestream = parse_raw(stream, settings)?;
    let header = &parsed_codestream.header;
    let mut boxes = ImageBoxes::default();

    let (color_space, has_alpha) = if let Some(channels) = color_channels {
        let components = header.component_infos.len();
        let has_alpha = components == usize::from(channels) + 1;
        if channels == 0 || (components != usize::from(channels) && !has_alpha) {
            bail!(ValidationError::InvalidComponentMetadata);
        }

        let color_space = match channels {
            1 => crate::ColorSpace::Gray,
            3 => crate::ColorSpace::RGB,
            4 => crate::ColorSpace::CMYK,
            num_channels => crate::ColorSpace::Unknown { num_channels },
        };

        (color_space, has_alpha)
    } else {
        // If we are just decoding a raw codestream, we assume greyscale or
        // RGB. The exception are three components where the last two are
        // subsampled relative to the first one, since that's practically only
        // done for YCbCr.
        let subsampling = |c: &codestream::ComponentInfo| {
            (
                c.size_info.horizontal_resolution,
                c.size_info.vertical_resolution,
            )
        };
        let has_subsampled_chroma = matches!(
            header.component_infos.as_slice(),
            [y, cb, cr] if {
                let (y, cb, cr) = (subsampling(y), subsampling(cb), subsampling(cr));

                cb == cr && cb != y && cb.0 >= y.0 && cb.1 >= y.1
            }
        );

        let cs = if header.component_infos.len() < 3 {
            ColorSpace::Enumerated(EnumeratedColorspace::Greyscale)
        } else if has_subsampled_chroma {
            ColorSpace::Enumerated(EnumeratedColorspace::Sycc)
        } else {
            ColorSpace::Enumerated(EnumeratedColorspace::Srgb)
        };

        boxes.color_specification = Some(ColorSpecificationBox { color_space: cs });

        resolve_alpha_and_color_space(&boxes, &parsed_codestream.header, settings)?
    };

    Ok(Image {
        codestream: parsed_codestream.data,
        header: parsed_codestream.header,
        boxes,
        settings: *settings,
        color_space,
        has_alpha,
    })
}

pub(crate) fn parse_raw<'a>(
    stream: &'a [u8],
    settings: &DecodeSettings,
) -> Result<ParsedCodestream<'a>> {
    let mut reader = BitReader::new(stream);

    let marker = reader.read_marker()?;
    if marker != markers::SOC {
        bail!(MarkerError::Expected("SOC"));
    }

    let header = codestream::read_header(&mut reader, settings)?;
    let code_stream_data = reader.tail().ok_or(FormatError::MissingCodestream)?;

    Ok(ParsedCodestream {
        header,
        data: code_stream_data,
    })
}
