//! Single-picture decode: one call from the OBUs of a still image to its
//! planes, borrowed from the decoder's picture buffer.
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

/// The samples of one plane: rows [`StillPlane::stride`] samples apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StillSamples<'a> {
    /// 8-bit pictures.
    U8(&'a [u8]),
    /// 10- and 12-bit pictures.
    U16(&'a [u16]),
}

/// One plane of a decoded picture, borrowed from the decoder's buffer.
///
/// Row `y` is the `width` samples from index `y * stride`; the samples
/// between rows are padding.  The slice ends with the last row, so it holds
/// `(height - 1) * stride + width` samples, or none for an absent plane.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StillPlane<'a> {
    /// Samples per row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The distance between the starts of consecutive rows, in samples.
    pub stride: usize,
    /// The samples.
    pub samples: StillSamples<'a>,
}

/// A decoded still picture.
#[derive(Clone, Debug)]
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
    /// The decoder's picture; [`check`] has verified that each plane holds
    /// the rows [`Self::plane`] hands out.
    picture: Picture,
}

impl StillPicture {
    /// The size of the chroma planes: `(0, 0)` for [`StillLayout::Mono`].
    pub const fn chroma_size(&self) -> (u32, u32) {
        if matches!(self.layout, StillLayout::Mono) {
            return (0, 0);
        }
        let (shift_x, shift_y) = self.layout.chroma_shift();
        (
            (self.width + (1 << shift_x) - 1) >> shift_x,
            (self.height + (1 << shift_y) - 1) >> shift_y,
        )
    }

    /// Plane `index`: 0 is Y, 1 is U, 2 is V.  U and V are empty for
    /// [`StillLayout::Mono`].
    ///
    /// # Panics
    ///
    /// If `index` is over 2.
    pub fn plane(&self, index: usize) -> StillPlane<'_> {
        let component = PlanarImageComponent::try_from(index).expect("a plane index of 0 to 2");
        let (width, height) = if index == 0 {
            (self.width, self.height)
        } else {
            self.chroma_size()
        };
        let wide = self.bit_depth > 8;
        if width == 0 || height == 0 {
            let samples = if wide {
                StillSamples::U16(&[])
            } else {
                StillSamples::U8(&[])
            };
            return StillPlane {
                width,
                height,
                stride: 0,
                samples,
            };
        }
        let stride_bytes = self.picture.stride(component) as usize;
        // `check` established that the plane holds this many samples.
        let stride = if wide { stride_bytes / 2 } else { stride_bytes };
        let len = (height as usize - 1) * stride + width as usize;
        let samples = if wide {
            StillSamples::U16(&self.picture.plane_wide(component)[..len])
        } else {
            StillSamples::U8(&self.picture.plane(component)[..len])
        };
        StillPlane {
            width,
            height,
            stride,
            samples,
        }
    }
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
/// `limits`; [`StillError::NoPicture`] when it holds no complete frame,
/// empty data included.
pub fn decode_still(obu: &[u8], limits: StillLimits) -> Result<StillPicture, StillError> {
    // The decoder's input validation aborts the process on empty data in a
    // debug build rather than returning its error.
    if obu.is_empty() {
        return Err(StillError::NoPicture);
    }
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
            Ok(picture) => return check(picture),
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

/// Wrap `picture` once its geometry is verified: a non-empty size, a bit
/// depth of 8, 10 or 12, and in every plane a stride of whole samples that
/// covers a row, with all rows inside the buffer.
fn check(picture: Picture) -> Result<StillPicture, StillError> {
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
    let still = StillPicture {
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
        picture,
    };
    let sample_bytes = if bit_depth > 8 { 2 } else { 1 };
    let components = [
        (PlanarImageComponent::Y, (width, height)),
        (PlanarImageComponent::U, still.chroma_size()),
        (PlanarImageComponent::V, still.chroma_size()),
    ];
    for (component, (w, h)) in components {
        let (w, h) = (w as usize, h as usize);
        if w == 0 || h == 0 {
            continue;
        }
        let stride = still.picture.stride(component) as usize;
        let row_bytes = w.checked_mul(sample_bytes).ok_or(StillError::Malformed)?;
        // Row `h - 1` ends at `(h - 1) * stride + row_bytes`.
        let needed = (h - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or(StillError::Malformed)?;
        if stride % sample_bytes != 0
            || stride < row_bytes
            || still.picture.plane(component).len() < needed
        {
            return Err(StillError::Malformed);
        }
    }
    Ok(still)
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
        assert_eq!(picture.chroma_size(), (8, 4));
        // The identity matrix carries G, B, R as Y, U, V.
        let samples: [fn(u8, u8) -> u8; 3] = [|_, y| y * 60, |x, _| 255 - x * 30, |x, _| x * 30];
        for (index, sample) in samples.into_iter().enumerate() {
            let plane = picture.plane(index);
            assert_eq!((plane.width, plane.height), (8, 4));
            let StillSamples::U8(data) = plane.samples else {
                panic!("an 8-bit picture has 8-bit samples")
            };
            assert_eq!(data.len(), 3 * plane.stride + 8);
            for y in 0..4u8 {
                let row = &data[usize::from(y) * plane.stride..][..8];
                let want: Vec<u8> = (0..8).map(|x| sample(x, y)).collect();
                assert_eq!(row, want, "plane {index} row {y}");
            }
        }
    }

    #[test]
    fn a_frame_over_the_limit_is_refused() {
        assert_eq!(
            decode_still(OBU, limits(31)).err(),
            Some(StillError::Decode(Rav1dError::OutOfRange))
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
