//! The rows of an image as the row-streamed writers take them: a band of
//! whole rows, from the top, in one of the layouts the writers store.
//!
//! A writer is told its image's size and layout when it is made, then takes
//! the image's bands in order ([`RowImage`]) and is finished once every row
//! is written.  A whole image is a single band, so writing an image whole
//! and writing it band by band produce the same bytes.

use color::{Gray8, Rgb8};
use raster::Bitmap;

use crate::{EncodeError, pack_bilevel_row, push_rgba};

/// A band of an image: whole rows, the full width of the image.
#[derive(Clone, Copy)]
pub enum RowImage<'a> {
    /// 8-bit RGB; RGBA, the alpha unassociated, when the bitmap has an
    /// alpha plane and the writer stores alpha.
    Rgb(&'a Bitmap<Rgb8>),
    /// 8-bit gray, 0 black.
    Gray(&'a Bitmap<Gray8>),
    /// One bit per pixel from gray luma, 1 for black (below
    /// [`crate::BLACK_BELOW`]), rows packed most significant bit first.
    Bilevel(&'a Bitmap<Gray8>),
}

/// The stored layout of an image's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowLayout {
    /// Three bytes a pixel.
    Rgb,
    /// Four bytes a pixel, the fourth the alpha plane's.
    Rgba,
    /// One byte a pixel.
    Gray,
    /// One bit a pixel, rows padded to a byte.
    Bilevel,
}

impl RowLayout {
    /// The bytes of a stored row of `width` pixels.
    #[must_use]
    pub const fn row_bytes(self, width: u32) -> usize {
        let w = width as usize;
        match self {
            Self::Rgb => w * 3,
            Self::Rgba => w * 4,
            Self::Gray => w,
            Self::Bilevel => w.div_ceil(8),
        }
    }
}

impl RowImage<'_> {
    /// Width and height in pixels.
    #[must_use]
    pub const fn size(&self) -> (u32, u32) {
        match self {
            Self::Rgb(b) => (b.width, b.height),
            Self::Gray(b) | Self::Bilevel(b) => (b.width, b.height),
        }
    }

    /// The layout the band is stored in when a writer keeps its alpha
    /// plane: RGBA for an RGB band with one.
    #[must_use]
    pub const fn layout(&self) -> RowLayout {
        match self {
            Self::Rgb(b) if b.has_alpha() => RowLayout::Rgba,
            Self::Rgb(_) => RowLayout::Rgb,
            Self::Gray(_) => RowLayout::Gray,
            Self::Bilevel(_) => RowLayout::Bilevel,
        }
    }

    /// Append the bytes of row `y` stored as `layout` to `out`: live
    /// pixels only.  An RGB band stored as [`RowLayout::Rgb`] drops its
    /// alpha plane; one stored as [`RowLayout::Rgba`] without one is
    /// opaque.
    ///
    /// # Panics
    ///
    /// When `layout` is not one of the band's: an RGB layout for a gray
    /// band or the reverse.  Writers check the band with [`Self::check`]
    /// first.
    pub(crate) fn row(&self, y: u32, layout: RowLayout, out: &mut Vec<u8>) {
        match (self, layout) {
            (Self::Rgb(b), RowLayout::Rgb) => {
                out.extend_from_slice(&b.row_bytes(y)[..b.width as usize * 3]);
            }
            (Self::Rgb(b), RowLayout::Rgba) => {
                let rgb = &b.row_bytes(y)[..b.width as usize * 3];
                match b.alpha_row(y) {
                    Some(alpha) => push_rgba(rgb, alpha, out),
                    None => {
                        for px in rgb.as_chunks::<3>().0 {
                            out.extend_from_slice(px);
                            out.push(u8::MAX);
                        }
                    }
                }
            }
            (Self::Gray(b) | Self::Bilevel(b), RowLayout::Gray) => {
                out.extend_from_slice(&b.row_bytes(y)[..b.width as usize]);
            }
            (Self::Gray(b) | Self::Bilevel(b), RowLayout::Bilevel) => {
                let start = out.len();
                out.resize(start + (b.width as usize).div_ceil(8), 0);
                pack_bilevel_row(&b.row_bytes(y)[..b.width as usize], &mut out[start..]);
            }
            _ => panic!("RowImage::row: layout {layout:?} does not store this band"),
        }
    }

    /// Check that this band continues an image of `width` pixels stored as
    /// `layout` with `rows_left` rows still to write.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a band of another width, of more
    /// rows than are left, or of colour for a gray image or the reverse.
    pub(crate) const fn check(
        &self,
        width: u32,
        layout: RowLayout,
        rows_left: u32,
    ) -> Result<(), EncodeError> {
        let (w, h) = self.size();
        if w != width {
            return Err(EncodeError::InvalidInput(
                "a band is not the width of the image it continues",
            ));
        }
        if h > rows_left {
            return Err(EncodeError::InvalidInput(
                "a band holds more rows than the image has left",
            ));
        }
        let fits = matches!(
            (self, layout),
            (Self::Rgb(_), RowLayout::Rgb | RowLayout::Rgba)
                | (
                    Self::Gray(_) | Self::Bilevel(_),
                    RowLayout::Gray | RowLayout::Bilevel
                )
        );
        if !fits {
            return Err(EncodeError::InvalidInput(
                "a band's pixels are not the colour model of the image it continues",
            ));
        }
        Ok(())
    }
}

/// The count of rows an image still takes, from its height down.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowCount {
    width: u32,
    height: u32,
    written: u32,
}

impl RowCount {
    /// A `width × height` image, no row written.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image.
    pub(crate) const fn new(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 {
            return Err(EncodeError::InvalidInput(
                "zero-width or zero-height image: there is nothing to encode",
            ));
        }
        Ok(Self {
            width,
            height,
            written: 0,
        })
    }

    /// Check `band` against the image and count its rows written.
    ///
    /// # Errors
    ///
    /// As [`RowImage::check`].
    pub(crate) fn take(
        &mut self,
        band: &RowImage<'_>,
        layout: RowLayout,
    ) -> Result<(), EncodeError> {
        band.check(self.width, layout, self.height - self.written)?;
        self.written += band.size().1;
        Ok(())
    }

    /// [`EncodeError::InvalidInput`] unless every row is written.
    pub(crate) const fn finished(self) -> Result<(), EncodeError> {
        if self.written == self.height {
            Ok(())
        } else {
            Err(EncodeError::InvalidInput(
                "the image was finished before all its rows were written",
            ))
        }
    }
}
