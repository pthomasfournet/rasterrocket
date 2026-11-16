//! A finished page: the bitmap a page canvas finishes as, of RGB, of an
//! output profile's device values or of CMYK, and its narrowing from 16
//! to 8 bits.
//!
//! An ink canvas finishes as its process ink, its spots folded in, or as
//! RGB of a working space by the colour its ink shows ([`InkFinish`]).

use color::cmyk::{ink_to_cmyk_row, ink_to_cmyk_row16, ink_to_rgb_row, ink_to_rgb_row16};
use color::{Cmyk8, Cmyk16, Gray8, Gray16, Pixel, Rgb8, Rgb16, Sample, SampleDepth};
use raster::{Bitmap, BitmapAllocError, BitmapBuffers};

use super::canvas::{PageCanvas, zero_transparent_colour};
use super::ink_map::SpotTables;
use super::space_convert::{InkColour, fold_ink, into_space};
use super::{CanvasBuffers, PageRenderer};
use crate::{ColourIntent, ColourTarget, WorkingSpace};

/// What an ink canvas finishes as.
pub(super) enum InkFinish<'a> {
    /// Its process ink, the spots folded in.
    Cmyk,
    /// RGB of `space`: the colour of its ink, the spots folded in.
    Rgb {
        /// The colour the ink shows.
        colour: InkColour<'a>,
        /// The working space of the RGB.
        space: WorkingSpace,
    },
}

impl InkFinish<'_> {
    /// sRGB through the process-ink model.
    #[cfg(test)]
    pub(super) const SRGB: Self = Self::Rgb {
        colour: InkColour::Model,
        space: WorkingSpace::Srgb,
    };

    /// Whether the finish is sRGB through the process-ink model, which
    /// converts whole rows ([`ink_to_rgb_row`], [`ink_to_rgb_row16`]).
    const fn is_model_srgb(&self) -> bool {
        matches!(
            self,
            Self::Rgb {
                colour: InkColour::Model,
                space: WorkingSpace::Srgb,
            }
        )
    }

    /// The RGB of `space` of the ink pixels of `ink` into `out`, three
    /// samples a pixel, the spots folded in by `spots`.
    fn rgb_row<S: Sample>(&self, ink: &[S], spots: &SpotTables<'_>, out: &mut [S]) {
        let Self::Rgb { colour, space } = self else {
            return;
        };
        for (px, o) in ink
            .as_chunks::<8>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<3>().0)
        {
            let (from, rgb) = colour.rgb(fold_ink(px, spots));
            *o = into_space(from, *space, rgb);
        }
    }
}

impl<'doc> PageRenderer<'doc> {
    /// What the page's ink canvas finishes as: CMYK for a page finished
    /// as CMYK, RGB of the page's finishing space otherwise.
    pub(super) fn ink_finish(&self) -> InkFinish<'doc> {
        if self.ink.cmyk {
            return InkFinish::Cmyk;
        }
        let target = ColourTarget::new(ColourIntent::INITIAL, self.finish_space);
        InkFinish::Rgb {
            colour: InkColour::of(&self.ink.separator, self.resources.doc(), target),
            space: self.finish_space,
        }
    }
}

/// Convert the ink canvas `ink` into `out`, of its size and depth, one
/// row at a time through `row` (eight ink samples a pixel to the
/// `Q::NCOMPS` samples of `out`), copying its alpha.
fn ink_into<P: Pixel, Q: Pixel<Sample = P::Sample>>(
    ink: &Bitmap<P>,
    out: &mut Bitmap<Q>,
    row: impl Fn(&[P::Sample], &mut [P::Sample]),
) {
    let n = ink.width as usize;
    for y in 0..ink.height {
        row(
            &ink.row_samples(y)[..n * 8],
            &mut out.row_samples_mut(y)[..n * Q::NCOMPS],
        );
    }
    if let (Some(src), Some(dst)) = (ink.alpha_plane(), out.alpha_plane_mut()) {
        dst.copy_from_slice(src);
    }
}

/// A bitmap of pixels `Q` of the size and alpha of `ink`, built in
/// `held`, holding the conversion of `ink` through `row` (see
/// [`ink_into`]).
fn try_ink_into<P: Pixel, Q: Pixel<Sample = P::Sample>>(
    ink: &Bitmap<P>,
    held: &mut BitmapBuffers<P::Sample>,
    row: impl Fn(&[P::Sample], &mut [P::Sample]),
) -> Result<Bitmap<Q>, BitmapAllocError> {
    let alpha = ink.has_alpha();
    let mut out = Bitmap::<Q>::try_new_filled_in(
        held.take_for(alpha),
        ink.width,
        ink.height,
        1,
        alpha,
        P::Sample::default(),
    )?;
    ink_into(ink, &mut out, row);
    Ok(out)
}

impl PageCanvas {
    /// The canvas as RGB pixels at its depth, keeping its alpha plane, or
    /// an ink canvas as `finish` says (see [`Self::try_into_finished_in`]).
    ///
    /// # Panics
    ///
    /// When the allocator refuses the finished bitmap.
    pub(super) fn into_finished(
        self,
        spots: &SpotTables<'_>,
        finish: &InkFinish<'_>,
    ) -> FinishedCanvas {
        self.try_into_finished_in(spots, finish, &mut CanvasBuffers::default())
            .expect("the finished bitmap is allocated")
    }

    /// The canvas as pixels at its depth, keeping its alpha plane: an RGB
    /// canvas as it is; an ink canvas, its spot colourants `spots`, as
    /// `finish` says: its process ink with the spots folded in
    /// ([`ink_to_cmyk_row`], [`ink_to_cmyk_row16`]), or RGB, through
    /// [`ink_to_rgb_row`] or [`ink_to_rgb_row16`] for sRGB of the
    /// process-ink model.
    ///
    /// The bitmap of an ink canvas is built in the buffers of `buffers` of
    /// its model and depth ([`Bitmap::try_new_filled_in`]), and the ink
    /// canvas's storage is then reclaimed into its ink buffers; an RGB
    /// canvas is returned as it is and `buffers` are left as they are.
    ///
    /// # Errors
    ///
    /// The allocator refused the bitmap of an ink canvas.
    pub(super) fn try_into_finished_in(
        self,
        spots: &SpotTables<'_>,
        finish: &InkFinish<'_>,
        buffers: &mut CanvasBuffers,
    ) -> Result<FinishedCanvas, BitmapAllocError> {
        let (narrow, wide) = (&spots.narrow, &spots.wide);
        let cmyk = matches!(finish, InkFinish::Cmyk);
        let model = finish.is_model_srgb();
        Ok(match (self, cmyk) {
            (Self::Rgb(bitmap), _) => FinishedCanvas::Rgb8(bitmap),
            (Self::Rgb16(bitmap), _) => FinishedCanvas::Rgb16(bitmap),
            (Self::Ink(ink), false) => {
                let rgb = try_ink_into(&ink, &mut buffers.rgb, |i, o| {
                    if model {
                        ink_to_rgb_row(i, narrow, o);
                    } else {
                        finish.rgb_row(i, spots, o);
                    }
                })?;
                buffers.ink.reclaim(ink);
                FinishedCanvas::Rgb8(rgb)
            }
            (Self::Ink16(ink), false) => {
                let rgb = try_ink_into(&ink, &mut buffers.rgb16, |i, o| {
                    if model {
                        ink_to_rgb_row16(i, wide, o);
                    } else {
                        finish.rgb_row(i, spots, o);
                    }
                })?;
                buffers.ink16.reclaim(ink);
                FinishedCanvas::Rgb16(rgb)
            }
            (Self::Ink(ink), true) => {
                let out = try_ink_into(&ink, &mut buffers.cmyk, |i, o| {
                    ink_to_cmyk_row(i, narrow, o);
                })?;
                buffers.ink.reclaim(ink);
                FinishedCanvas::Cmyk8(out)
            }
            (Self::Ink16(ink), true) => {
                let out = try_ink_into(&ink, &mut buffers.cmyk16, |i, o| {
                    ink_to_cmyk_row16(i, wide, o);
                })?;
                buffers.ink16.reclaim(ink);
                FinishedCanvas::Cmyk16(out)
            }
        })
    }
}

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
