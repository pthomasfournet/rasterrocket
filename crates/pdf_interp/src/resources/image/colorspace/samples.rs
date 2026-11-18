//! The samples a codec yields, before colour conversion.

use crate::resources::image::{Bilevel, ImageData, SoftMask};

/// The sample layout a codec yields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::resources::image) enum Layout {
    /// Stencil bytes: `0x00` paints, `0xFF` is transparent.
    Mask,
    /// `n` interleaved 8-bit components per pixel in the image's colour
    /// space, before `/Decode`.
    Color(u8),
}

/// A codec's output: an image's samples before colour conversion.
#[derive(Debug)]
pub(in crate::resources::image) struct Samples {
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub data: ImageData,
    /// Alpha the codestream itself carries (JPEG 2000 under `/SMaskInData`),
    /// on the image's own grid.
    pub smask: Option<SoftMask>,
    /// The ICC profile of the codestream's own colour specification, for
    /// an image dictionary that names no colour space.
    pub embedded_profile: Option<Vec<u8>>,
    /// What the stream lacked when the samples are only partly its own:
    /// a stream shorter than its dictionary declares, or a codec that
    /// stopped short, leaves the missing samples padded.
    pub partial: Option<String>,
    /// The colour components at the codestream's own precision, when it
    /// is not 8 bits and a colour-key `/Mask` compares them in their own
    /// range (§8.9.6.4); `data` holds them scaled.
    pub key_source: Option<KeySamples>,
}

/// The samples a colour key compares, at the codestream's own precision.
#[derive(Debug)]
pub(in crate::resources::image) enum KeySamples {
    /// The colour components interleaved, at most 16 bits each.
    Precise(Vec<u16>),
    /// The samples could not be kept, for the reason given: the key is
    /// not applied.
    Unusable(String),
}

impl Samples {
    /// Host-resident samples of `layout`.
    pub(in crate::resources::image) const fn cpu(
        width: u32,
        height: u32,
        layout: Layout,
        data: Vec<u8>,
    ) -> Self {
        Self {
            width,
            height,
            layout,
            data: ImageData::Cpu(data),
            smask: None,
            embedded_profile: None,
            partial: None,
            key_source: None,
        }
    }

    /// Host-resident 16-bit samples of `layout`.
    pub(in crate::resources::image) fn cpu16(
        width: u32,
        height: u32,
        layout: Layout,
        data: Vec<u16>,
    ) -> Self {
        Self {
            data: ImageData::Cpu16(data),
            ..Self::cpu(width, height, layout, Vec::new())
        }
    }

    /// Host-resident bilevel samples of `layout`: `Mask`, or `Color(1)`.
    pub(in crate::resources::image) fn bilevel(layout: Layout, bits: Bilevel) -> Self {
        let (width, height) = (bits.width(), bits.height());
        Self {
            data: ImageData::Bilevel(Box::new(bits)),
            ..Self::cpu(width, height, layout, Vec::new())
        }
    }

    /// Stencil bytes.
    pub(in crate::resources::image) const fn mask(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self::cpu(width, height, Layout::Mask, data)
    }

    /// One component per pixel.
    pub(in crate::resources::image) const fn gray(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self::cpu(width, height, Layout::Color(1), data)
    }
}
