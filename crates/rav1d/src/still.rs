//! Single-picture decode: one call from the OBUs of a still image to its
//! planes.
//!
//! This module is not part of upstream `rav1d`; it is the entry point the
//! rasterrocket workspace uses (see `CHANGES.md`).  It is built on the safe
//! [`Decoder`] API of [`crate::rust_api`] and holds no `unsafe` code.

use std::fmt;

use crate::error::Rav1dError;
use crate::in_range::InRange;
use crate::rust_api::{Decoder, Picture, PixelLayout, PlanarImageComponent, Settings};

/// Chroma layout of a decoded picture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StillLayout {
    /// Luma alone.
    Mono,
    /// Chroma halved in both directions.
    I420,
    /// Chroma halved horizontally.
    I422,
    /// Chroma at luma resolution.
    I444,
}

impl StillLayout {
    /// The right shifts (horizontal, vertical) from luma to chroma
    /// coordinates; `(0, 0)` for [`Self::Mono`], which has no chroma.
    pub const fn chroma_shift(self) -> (u32, u32) {
        match self {
            Self::I420 => (1, 1),
            Self::I422 => (1, 0),
            Self::Mono | Self::I444 => (0, 0),
        }
    }
}

/// The colour description of the sequence header, as the code points of
/// ITU-T H.273 (2 is "unspecified" in the first three fields).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StillColor {
    /// Colour primaries.
    pub primaries: u8,
    /// Transfer characteristics.
    pub transfer: u8,
    /// Matrix coefficients.
    pub matrix: u8,
    /// Whether samples span the full range of the bit depth, as opposed to
    /// the limited ("studio") range.
    pub full_range: bool,
}

/// The samples of one plane, row-major without padding.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StillSamples {
    /// 8-bit pictures.
    U8(Vec<u8>),
    /// 10- and 12-bit pictures.
    U16(Vec<u16>),
}

impl StillSamples {
    /// The number of samples.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
        }
    }

    /// Whether the plane holds no samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One plane of a decoded picture.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StillPlane {
    /// Samples per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// `width * height` samples.
    pub samples: StillSamples,
}

/// A decoded still picture.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StillPicture {
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u8,
    /// Chroma layout.
    pub layout: StillLayout,
    /// Colour description of the sequence header.
    pub color: StillColor,
    /// The Y, U and V planes; U and V are empty for [`StillLayout::Mono`].
    pub planes: [StillPlane; 3],
}

/// Bounds on what [`decode_still`] may allocate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StillLimits {
    /// The most luma pixels (width × height) a frame header may declare; a
    /// larger frame fails before its planes are allocated.  Zero is no
    /// limit.
    pub max_pixels: u32,
}

/// Why [`decode_still`] failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StillError {
    /// The decoder rejected the data.
    Decode(Rav1dError),
    /// The data ended without yielding a picture.
    NoPicture,
    /// The picture's geometry or bit depth is outside what the decoder
    /// documents.
    Malformed,
}

impl fmt::Display for StillError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "{e}"),
            Self::NoPicture => f.write_str("no picture in the data"),
            Self::Malformed => f.write_str("inconsistent picture geometry"),
        }
    }
}

impl std::error::Error for StillError {}

impl From<Rav1dError> for StillError {
    fn from(e: Rav1dError) -> Self {
        Self::Decode(e)
    }
}

/// Decode the first picture of `obu`, a sequence of AV1 open bitstream
/// units (a sequence header followed by one frame, as a still image item
/// carries).
///
/// The decode runs on the calling thread.  Of a scalable stream, the highest
/// spatial layer of operating point 0 is returned; film grain is applied.
///
/// # Errors
///
/// [`StillError::Decode`] when the stream is malformed, unsupported or over
/// `limits`; [`StillError::NoPicture`] when it holds no complete frame.
pub fn decode_still(obu: &[u8], limits: StillLimits) -> Result<StillPicture, StillError> {
    let mut settings = Settings::new();
    settings.inner.n_threads = InRange::new(1).ok_or(Rav1dError::InvalidArgument)?;
    settings.inner.max_frame_delay = InRange::new(1).ok_or(Rav1dError::InvalidArgument)?;
    settings.set_apply_grain(true);
    settings.set_all_layers(false);
    settings.set_frame_size_limit(limits.max_pixels);
    // Diagnostics go to the caller as the error value, not to stderr.
    settings.inner.logger = None;
    let mut decoder = Decoder::with_settings(&settings)?;

    let mut sent = decoder.send_data(obu.into(), None, None, None);
    // Each pass either returns or consumes input, and input is finite; the
    // bound guards against a decoder that reports progress without making
    // any.
    for _ in 0..=obu.len() {
        match decoder.get_picture() {
            Ok(picture) => return own(&picture),
            Err(Rav1dError::TryAgain) => {}
            Err(e) => return Err(e.into()),
        }
        match sent {
            Err(Rav1dError::TryAgain) => sent = decoder.send_pending_data(),
            Err(e) => return Err(e.into()),
            Ok(()) => return Err(StillError::NoPicture),
        }
    }
    Err(StillError::NoPicture)
}

/// Copy `picture` out of the decoder's padded planes.
fn own(picture: &Picture) -> Result<StillPicture, StillError> {
    let (width, height) = (picture.width(), picture.height());
    if width == 0 || height == 0 {
        return Err(StillError::Malformed);
    }
    let bit_depth = picture.bits_per_component().ok_or(StillError::Malformed)?.0;
    let layout = match picture.pixel_layout() {
        PixelLayout::I400 => StillLayout::Mono,
        PixelLayout::I420 => StillLayout::I420,
        PixelLayout::I422 => StillLayout::I422,
        PixelLayout::I444 => StillLayout::I444,
    };
    let (shift_x, shift_y) = layout.chroma_shift();
    let chroma = if layout == StillLayout::Mono {
        (0, 0)
    } else {
        (
            (width + (1 << shift_x) - 1) >> shift_x,
            (height + (1 << shift_y) - 1) >> shift_y,
        )
    };
    let plane = |component, (w, h): (u32, u32)| own_plane(picture, component, w, h, bit_depth > 8);
    Ok(StillPicture {
        width,
        height,
        bit_depth,
        layout,
        color: StillColor {
            primaries: picture.color_primaries_code(),
            transfer: picture.transfer_code(),
            matrix: picture.matrix_code(),
            full_range: picture.full_range(),
        },
        planes: [
            plane(PlanarImageComponent::Y, (width, height))?,
            plane(PlanarImageComponent::U, chroma)?,
            plane(PlanarImageComponent::V, chroma)?,
        ],
    })
}

/// The top-left `width × height` samples of one component, whose rows are
/// `stride` bytes apart; `wide` samples are two native-endian bytes.
fn own_plane(
    picture: &Picture,
    component: PlanarImageComponent,
    width: u32,
    height: u32,
    wide: bool,
) -> Result<StillPlane, StillError> {
    let (w, h) = (width as usize, height as usize);
    let samples = w.checked_mul(h).ok_or(StillError::Malformed)?;
    if samples == 0 {
        let samples = if wide {
            StillSamples::U16(Vec::new())
        } else {
            StillSamples::U8(Vec::new())
        };
        return Ok(StillPlane {
            width,
            height,
            samples,
        });
    }
    let stride = picture.stride(component) as usize;
    let data = picture.plane(component);
    let row_bytes = if wide { w * 2 } else { w };
    // Row `h - 1` ends at `(h - 1) * stride + row_bytes`.
    let needed = (h - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(row_bytes))
        .ok_or(StillError::Malformed)?;
    if stride < row_bytes || data.len() < needed {
        return Err(StillError::Malformed);
    }
    let rows = data.chunks(stride).take(h).map(|row| &row[..row_bytes]);
    let samples = if wide {
        let mut out = Vec::with_capacity(samples);
        for row in rows {
            out.extend(
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&pair| u16::from_ne_bytes(pair)),
            );
        }
        StillSamples::U16(out)
    } else {
        let mut out = Vec::with_capacity(samples);
        for row in rows {
            out.extend_from_slice(row);
        }
        StillSamples::U8(out)
    };
    Ok(StillPlane {
        width,
        height,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sequence header and one frame: 8 × 4, 4:4:4, lossless, identity
    /// matrix, full range, encoded by libaom from the RGB picture
    /// `(30x, 60y, 255 − 30x)`.
    const OBU: &[u8] = &[
        0x0a, 0x08, 0x38, 0x08, 0x7e, 0xda, 0x02, 0x1a, 0x00, 0x40, 0x32, 0x37, 0x10, 0x00, 0x00,
        0x8b, 0xd6, 0xf8, 0xa0, 0x66, 0x94, 0x64, 0x43, 0x5a, 0x9d, 0x8d, 0xcc, 0xc7, 0xdd, 0x8d,
        0x72, 0xf9, 0x44, 0x46, 0xfa, 0x47, 0x3d, 0x14, 0xbb, 0xa8, 0xb5, 0x35, 0x17, 0x97, 0x30,
        0xb7, 0x2b, 0x5c, 0x21, 0x3a, 0xab, 0x5e, 0x8c, 0x65, 0x94, 0x07, 0x38, 0x8a, 0xcf, 0x7b,
        0x7a, 0x8c, 0x99, 0x9e, 0xfa, 0x40, 0x20,
    ];

    fn limits(max_pixels: u32) -> StillLimits {
        StillLimits { max_pixels }
    }

    #[test]
    fn a_lossless_picture_decodes_to_its_planes() {
        let picture = decode_still(OBU, limits(32)).unwrap();
        assert_eq!((picture.width, picture.height), (8, 4));
        assert_eq!((picture.bit_depth, picture.layout), (8, StillLayout::I444));
        assert_eq!((picture.color.matrix, picture.color.full_range), (0, true));
        // The identity matrix carries G, B, R as Y, U, V.
        let plane = |sample: fn(u8, u8) -> u8| StillPlane {
            width: 8,
            height: 4,
            samples: StillSamples::U8(
                (0..4)
                    .flat_map(|y| (0..8).map(move |x| sample(x, y)))
                    .collect(),
            ),
        };
        assert_eq!(picture.planes[0], plane(|_, y| y * 60));
        assert_eq!(picture.planes[1], plane(|x, _| 255 - x * 30));
        assert_eq!(picture.planes[2], plane(|x, _| x * 30));
    }

    #[test]
    fn a_frame_over_the_limit_is_refused() {
        assert_eq!(
            decode_still(OBU, limits(31)),
            Err(StillError::Decode(Rav1dError::OutOfRange))
        );
        assert!(decode_still(OBU, limits(0)).is_ok());
    }

    #[test]
    fn truncated_or_empty_data_is_an_error() {
        for cut in 0..OBU.len() {
            assert!(
                decode_still(&OBU[..cut], limits(32)).is_err(),
                "cut at {cut}"
            );
        }
    }
}
