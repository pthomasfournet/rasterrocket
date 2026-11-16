//! A finished page: the bitmap a page canvas finishes as, of sRGB, of an
//! output profile's device values or of CMYK, and its narrowing from 16
//! to 8 bits.

use color::{Cmyk8, Cmyk16, Gray8, Gray16, Pixel, Rgb8, Rgb16, SampleDepth};
use raster::{Bitmap, BitmapAllocError, BitmapBuffers};

use super::CanvasBuffers;
use super::canvas::zero_transparent_colour;

/// A finished page, as sRGB, as device values or as CMYK.
///
/// The canvas is sRGB pixels at its depth, or, once converted to an output
/// profile ([`OutputProfile::convert`](crate::OutputProfile::convert)), the
/// profile's RGB or gray device values; an ink canvas finished as CMYK
/// ([`CanvasRequest::Cmyk`](crate::CanvasRequest::Cmyk)) is its process
/// ink.
pub enum FinishedCanvas {
    /// 8-bit RGB.
    Rgb8(Bitmap<Rgb8>),
    /// 16-bit RGB.
    Rgb16(Bitmap<Rgb16>),
    /// 8-bit gray, 0 black: the device values of a gray output profile.
    Gray8(Bitmap<Gray8>),
    /// 16-bit gray, 0 black: the device values of a gray output profile.
    Gray16(Bitmap<Gray16>),
    /// 8-bit CMYK, 0 no ink.
    Cmyk8(Bitmap<Cmyk8>),
    /// 16-bit CMYK, 0 no ink.
    Cmyk16(Bitmap<Cmyk16>),
}

impl std::fmt::Debug for FinishedCanvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (width, height) = self.dims();
        f.debug_struct("FinishedCanvas")
            .field("depth", &self.depth())
            .field("width", &width)
            .field("height", &height)
            .finish_non_exhaustive()
    }
}

impl FinishedCanvas {
    /// The bitmap's `(width, height)` in pixels.
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        match self {
            Self::Rgb8(b) => (b.width, b.height),
            Self::Rgb16(b) => (b.width, b.height),
            Self::Gray8(b) => (b.width, b.height),
            Self::Gray16(b) => (b.width, b.height),
            Self::Cmyk8(b) => (b.width, b.height),
            Self::Cmyk16(b) => (b.width, b.height),
        }
    }

    /// The depth of the bitmap's samples.
    #[must_use]
    pub const fn depth(&self) -> SampleDepth {
        match self {
            Self::Rgb8(_) | Self::Gray8(_) | Self::Cmyk8(_) => SampleDepth::Eight,
            Self::Rgb16(_) | Self::Gray16(_) | Self::Cmyk16(_) => SampleDepth::Sixteen,
        }
    }

    /// Whether the bitmap holds CMYK.
    #[must_use]
    pub const fn is_cmyk(&self) -> bool {
        matches!(self, Self::Cmyk8(_) | Self::Cmyk16(_))
    }

    /// Return the canvas's storage to `buffers` for the next canvas of its
    /// kind.
    pub fn reclaim_into(self, buffers: &mut CanvasBuffers) {
        match self {
            Self::Rgb8(b) => buffers.rgb.reclaim(b),
            Self::Rgb16(b) => buffers.rgb16.reclaim(b),
            Self::Gray8(b) => buffers.gray.reclaim(b),
            Self::Gray16(b) => buffers.gray16.reclaim(b),
            Self::Cmyk8(b) => buffers.cmyk.reclaim(b),
            Self::Cmyk16(b) => buffers.cmyk16.reclaim(b),
        }
    }

    /// Set the colour of every pixel whose alpha is 0 to 0 (see
    /// [`zero_transparent_colour`]).
    pub(super) fn zero_transparent_colour(&mut self) {
        match self {
            Self::Rgb8(b) => zero_transparent_colour(b),
            Self::Rgb16(b) => zero_transparent_colour(b),
            Self::Gray8(b) => zero_transparent_colour(b),
            Self::Gray16(b) => zero_transparent_colour(b),
            Self::Cmyk8(b) => zero_transparent_colour(b),
            Self::Cmyk16(b) => zero_transparent_colour(b),
        }
    }
    /// The canvas at 8 bits: a 16-bit canvas with every sample, its alpha
    /// included, narrowed to the nearest 8-bit level
    /// ([`color::Sample::from_sample`]), built in the 8-bit buffers of
    /// `buffers`, its own storage reclaimed into them; an 8-bit canvas as
    /// it is.  A pixel whose alpha narrows to 0 is 0.
    ///
    /// # Errors
    ///
    /// [`BitmapAllocError`] when the 8-bit bitmap cannot be allocated.
    pub fn narrowed(self, buffers: &mut CanvasBuffers) -> Result<Self, BitmapAllocError> {
        let mut narrow = match self {
            Self::Rgb8(_) | Self::Gray8(_) | Self::Cmyk8(_) => return Ok(self),
            Self::Rgb16(b) => Self::Rgb8(narrow::<Rgb16, Rgb8>(
                b,
                &mut buffers.rgb16,
                &mut buffers.rgb,
            )?),
            Self::Gray16(b) => Self::Gray8(narrow::<Gray16, Gray8>(
                b,
                &mut buffers.gray16,
                &mut buffers.gray,
            )?),
            Self::Cmyk16(b) => Self::Cmyk8(narrow::<Cmyk16, Cmyk8>(
                b,
                &mut buffers.cmyk16,
                &mut buffers.cmyk,
            )?),
        };
        narrow.zero_transparent_colour();
        Ok(narrow)
    }
}

/// `wide` narrowed to 8 bits in a bitmap built in `held`, its storage
/// then reclaimed into `spare` (see [`FinishedCanvas::narrowed`]).
fn narrow<P: Pixel<Sample = u16>, Q: Pixel<Sample = u8>>(
    wide: Bitmap<P>,
    spare: &mut BitmapBuffers<u16>,
    held: &mut BitmapBuffers,
) -> Result<Bitmap<Q>, BitmapAllocError> {
    let narrow = wide.try_converted_in(held.take_for(wide.has_alpha()))?;
    spare.reclaim(wide);
    Ok(narrow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_16_bit_canvas_narrows_to_the_nearest_8_bit_level() {
        let mut buffers = CanvasBuffers::default();
        let wide = Bitmap::<Rgb16>::from_packed(2, 1, vec![0, 128, 129, 65_535, 32_896, 257]);
        let FinishedCanvas::Rgb8(rgb) = FinishedCanvas::Rgb16(wide)
            .narrowed(&mut buffers)
            .expect("allocation")
        else {
            panic!("an RGB canvas narrows to RGB");
        };
        assert_eq!(rgb.data(), [0, 0, 1, 255, 128, 1]);
        assert!(
            buffers.rgb16.data.capacity() >= 6,
            "the 16-bit storage is kept"
        );
        let eight = FinishedCanvas::Cmyk8(Bitmap::from_packed(1, 1, vec![1, 2, 3, 4]));
        assert!(
            matches!(eight.narrowed(&mut buffers), Ok(FinishedCanvas::Cmyk8(b)) if b.data() == [1, 2, 3, 4])
        );
    }

    #[test]
    fn a_pixel_whose_alpha_narrows_to_0_is_0() {
        let mut wide = Bitmap::<Gray16>::new(2, 1, 1, true);
        wide.row_samples_mut(0)[..2].copy_from_slice(&[40_000, 40_000]);
        wide.alpha_plane_mut()
            .expect("an alpha plane")
            .copy_from_slice(&[128, 129]);
        let FinishedCanvas::Gray8(gray) = FinishedCanvas::Gray16(wide)
            .narrowed(&mut CanvasBuffers::default())
            .expect("allocation")
        else {
            panic!("a gray canvas narrows to gray");
        };
        assert_eq!(gray.data(), [0, 156]);
        assert_eq!(gray.alpha_plane(), Some(&[0, 1][..]));
    }
}
