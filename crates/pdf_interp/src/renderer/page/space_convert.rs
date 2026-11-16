//! Pixels converted between the spaces a canvas holds its colours in.
//!
//! A canvas holds RGB of a working space ([`CanvasSpace::Rgb`]) or process
//! and spot ink ([`CanvasSpace::Ink`]).  A conversion passes through sRGB:
//! ink converts by the process-ink model with its spots folded in
//! ([`color::cmyk::fold_spots`], [`color::cmyk::cmyk_to_rgb_at`]), wide
//! values by [`working_space::to_srgb16`]; sRGB converts into the wide
//! space by [`working_space::from_srgb16`] and into ink by a
//! [`Separator`], whose spot channels it leaves empty.
//!
//! A reduction to gray takes the gray of the colour on the way: the
//! `DeviceRGB` → `DeviceGray` weights of an sRGB value
//! ([`color::convert::gray_at`]), the conventional `DeviceCMYK` →
//! `DeviceGray` conversion of an ink pixel
//! ([`color::cmyk::device_gray_at`]).  Both are linear in the samples
//! below the clamp of black, so a reduction commutes with compositing in
//! the `Normal` blend mode.
//!
//! A group's colour is straight, beside its alpha, so a converted pixel is
//! the group's colour converted; pixels of zero alpha stay empty.

use color::{DeviceN8, DeviceN16, Pixel, Rgb8, Rgb16, Sample};
use raster::Bitmap;
use raster::clip::Clip;
use raster::transparency::{GroupBitmap, GroupParams, begin_group_converted, discard_group};

use super::canvas::{CanvasGroup, PageCanvas};
use super::depth::{CanvasBackdrop, CanvasSample};
use super::ink_map::SpotTables;
use super::pixel_rect::PixelRect;
use crate::WorkingSpace;
use crate::renderer::color::no_spots;
use crate::separation::Separator;
use crate::working_space;

/// The space a canvas holds its colours in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CanvasSpace {
    /// RGB of a working space.
    Rgb(WorkingSpace),
    /// Process and spot ink.
    Ink,
}

/// The conversion of pixels of one canvas space into another, or into
/// the same one, through sRGB, reduced to gray on the way when `gray`.
pub(super) struct SpaceConversion<'a> {
    /// The space converted from.
    pub(super) from: CanvasSpace,
    /// The space converted into.
    pub(super) to: CanvasSpace,
    /// Whether the colour reduces to gray.
    pub(super) gray: bool,
    /// The conversion of sRGB into ink, in its own working space.
    pub(super) separator: &'a Separator,
    /// The process equivalents of the spot channels of an ink pixel.
    pub(super) spots: &'a SpotTables<'a>,
}

impl SpaceConversion<'_> {
    /// Convert one pixel: `src` holds the samples of a pixel of `from`,
    /// `dst` receives those of a pixel of `to`.
    pub(super) fn pixel<S: CanvasSample>(&self, src: &[S], dst: &mut [S]) {
        let srgb = match self.from {
            _ if self.gray => [gray_of(self.from, src, self.spots); 3],
            CanvasSpace::Ink => {
                let [c, m, y, k] = fold_ink(src, self.spots);
                color::cmyk::cmyk_to_rgb_at(c, m, y, k)
            }
            CanvasSpace::Rgb(space) => working_space::to_srgb(space, [src[0], src[1], src[2]]),
        };
        match self.to {
            CanvasSpace::Rgb(space) => {
                dst.copy_from_slice(&working_space::from_srgb::<S, S>(space, srgb));
            }
            CanvasSpace::Ink => {
                let rgb = working_space::from_srgb::<S, S>(self.separator.space(), srgb);
                dst.copy_from_slice(&no_spots(self.separator.ink(rgb)));
            }
        }
    }

    /// Convert the pixels of `region` of `src` into `dst`, a transparent
    /// bitmap of the same size, copying their alpha: each pixel of nonzero
    /// alpha converts, the others stay empty.
    fn region<P: Pixel, Q: Pixel<Sample = P::Sample>>(
        &self,
        src: &Bitmap<P>,
        dst: &mut Bitmap<Q>,
        region: PixelRect,
    ) where
        P::Sample: CanvasSample,
    {
        let (n, m) = (P::NCOMPS, Q::NCOMPS);
        let (x0, x1) = (region.x0 as usize, region.x1 as usize);
        let zero = P::Sample::default();
        for y in region.y0..region.y1 {
            let (colour, alpha) = (src.row_samples(y), src.alpha_row(y));
            let (out, out_alpha) = dst.row_and_alpha_mut(y);
            if let (Some(a), Some(o)) = (alpha, out_alpha) {
                o[x0..x1].copy_from_slice(&a[x0..x1]);
            }
            let mut last: Option<(&[P::Sample], usize)> = None;
            for x in x0..x1 {
                if alpha.is_some_and(|a| a[x] == zero) {
                    continue;
                }
                let px = &colour[x * n..(x + 1) * n];
                match last {
                    Some((seen, at)) if seen == px => {
                        out.copy_within(at..at + m, x * m);
                    }
                    _ => self.pixel(px, &mut out[x * m..(x + 1) * m]),
                }
                last = Some((px, x * m));
            }
        }
    }

    /// Convert the pixels of `region` of `bitmap` in place: `from` and `to`
    /// are spaces of one pixel type.  Pixels of zero alpha are left.
    fn in_place<P: Pixel>(&self, bitmap: &mut Bitmap<P>, region: PixelRect)
    where
        P::Sample: CanvasSample,
    {
        let n = P::NCOMPS;
        let (x0, x1) = (region.x0 as usize, region.x1 as usize);
        let zero = P::Sample::default();
        let mut buf = [P::Sample::default(); 8];
        for y in region.y0..region.y1 {
            let (colour, alpha) = bitmap.row_and_alpha_mut(y);
            for x in x0..x1 {
                if alpha.as_ref().is_some_and(|a| a[x] == zero) {
                    continue;
                }
                let px = &mut colour[x * n..(x + 1) * n];
                buf[..n].copy_from_slice(px);
                self.pixel(&buf[..n], px);
            }
        }
    }
}

/// The gray of the pixel `px` of a canvas of `space`, as an sRGB gray: the
/// `DeviceRGB` → `DeviceGray` weights of its sRGB value, or the
/// conventional `DeviceCMYK` → `DeviceGray` conversion of its process ink,
/// its spots folded in by `spots`.
pub(super) fn gray_of<S: CanvasSample>(space: CanvasSpace, px: &[S], spots: &SpotTables<'_>) -> S {
    match space {
        CanvasSpace::Ink => color::cmyk::device_gray_at(fold_ink(px, spots)),
        CanvasSpace::Rgb(space) => {
            color::convert::gray_at(working_space::to_srgb(space, [px[0], px[1], px[2]]))
        }
    }
}

/// The process ink of the ink pixel `px`, its spots folded in by `spots`
/// ([`color::cmyk::fold_spots`], [`color::cmyk::fold_spots16`]).
pub(super) fn fold_ink<S: Sample>(px: &[S], spots: &SpotTables<'_>) -> [S; 4] {
    S::as_u8(px).map_or_else(
        || {
            let wide = S::as_u16(px).expect("a canvas sample is 8 or 16 bits");
            let px: &[u16; 8] = wide.try_into().expect("an ink pixel has eight samples");
            color::cmyk::fold_spots16(px, &spots.wide).map(S::from_sample)
        },
        |bytes| {
            let px: &[u8; 8] = bytes.try_into().expect("an ink pixel has eight samples");
            color::cmyk::fold_spots(px, &spots.narrow).map(S::from_u8)
        },
    )
}

impl PageCanvas {
    /// Convert the pixels of `region` of this canvas, a transparency
    /// group's, into `dst`, a transparent canvas of the same depth and
    /// size, by `conversion` (see [`SpaceConversion`]).
    ///
    /// # Panics
    ///
    /// When `dst` is of another depth.
    pub(super) fn convert_region_into(
        &self,
        dst: &mut Self,
        region: PixelRect,
        conversion: &SpaceConversion<'_>,
    ) {
        match (self, dst) {
            (Self::Rgb(s), Self::Rgb(d)) => conversion.region::<Rgb8, Rgb8>(s, d, region),
            (Self::Rgb(s), Self::Ink(d)) => conversion.region::<Rgb8, DeviceN8>(s, d, region),
            (Self::Ink(s), Self::Rgb(d)) => conversion.region::<DeviceN8, Rgb8>(s, d, region),
            (Self::Ink(s), Self::Ink(d)) => conversion.region::<DeviceN8, DeviceN8>(s, d, region),
            (Self::Rgb16(s), Self::Rgb16(d)) => conversion.region::<Rgb16, Rgb16>(s, d, region),
            (Self::Rgb16(s), Self::Ink16(d)) => conversion.region::<Rgb16, DeviceN16>(s, d, region),
            (Self::Ink16(s), Self::Rgb16(d)) => conversion.region::<DeviceN16, Rgb16>(s, d, region),
            (Self::Ink16(s), Self::Ink16(d)) => {
                conversion.region::<DeviceN16, DeviceN16>(s, d, region);
            }
            _ => panic!("convert_region_into: the canvases differ in depth"),
        }
    }

    /// Convert the pixels of `region` in place by `conversion`, whose
    /// spaces are both of this canvas's pixel type.
    pub(super) fn convert_region(&mut self, region: PixelRect, conversion: &SpaceConversion<'_>) {
        super::canvas::with_canvas!(self, |b| conversion.in_place(b, region));
    }

    /// Open a transparency group over this canvas on `bitmap`, a
    /// transparent canvas of the same depth and size that holds the
    /// colours of another space: a non-isolated group's backdrop converts
    /// by `conversion` (see [`raster::transparency::begin_group_converted`]).
    ///
    /// # Panics
    ///
    /// When `bitmap` is of another depth.
    pub(super) fn begin_converted_group(
        &self,
        backdrop: Option<&CanvasBackdrop>,
        clip: &Clip,
        params: GroupParams,
        bitmap: Self,
        conversion: &SpaceConversion<'_>,
    ) -> CanvasGroup {
        macro_rules! open {
            ($p:expr, $b:expr, $s:ty, $variant:ident) => {
                CanvasGroup::$variant(begin_group_converted(
                    $p,
                    backdrop.and_then(<$s>::backdrop),
                    clip,
                    params,
                    $b,
                    |src, dst| conversion.pixel(src, dst),
                ))
            };
        }
        match (self, bitmap) {
            (Self::Rgb(p), Self::Rgb(b)) => open!(p, b, u8, Rgb),
            (Self::Ink(p), Self::Rgb(b)) => open!(p, b, u8, Rgb),
            (Self::Rgb(p), Self::Ink(b)) => open!(p, b, u8, Ink),
            (Self::Ink(p), Self::Ink(b)) => open!(p, b, u8, Ink),
            (Self::Rgb16(p), Self::Rgb16(b)) => open!(p, b, u16, Rgb16),
            (Self::Ink16(p), Self::Rgb16(b)) => open!(p, b, u16, Rgb16),
            (Self::Rgb16(p), Self::Ink16(b)) => open!(p, b, u16, Ink16),
            (Self::Ink16(p), Self::Ink16(b)) => open!(p, b, u16, Ink16),
            _ => panic!("begin_converted_group: the canvases differ in depth"),
        }
    }
}

impl CanvasGroup {
    /// Take the group's bitmap out as a canvas, an empty bitmap in its
    /// place until [`Self::put_bitmap`] returns it.
    pub(super) fn take_bitmap(&mut self) -> PageCanvas {
        fn take<P: Pixel>(g: &mut GroupBitmap<P>) -> Bitmap<P> {
            std::mem::replace(&mut g.bitmap, Bitmap::new(0, 0, 1, true))
        }
        match self {
            Self::Rgb(g) => PageCanvas::Rgb(take(g)),
            Self::Rgb16(g) => PageCanvas::Rgb16(take(g)),
            Self::Ink(g) => PageCanvas::Ink(take(g)),
            Self::Ink16(g) => PageCanvas::Ink16(take(g)),
        }
    }

    /// Give the group back the bitmap [`Self::take_bitmap`] took.
    ///
    /// # Panics
    ///
    /// When `canvas` is of another kind than the group.
    pub(super) fn put_bitmap(&mut self, canvas: PageCanvas) {
        match (self, canvas) {
            (Self::Rgb(g), PageCanvas::Rgb(b)) => g.bitmap = b,
            (Self::Rgb16(g), PageCanvas::Rgb16(b)) => g.bitmap = b,
            (Self::Ink(g), PageCanvas::Ink(b)) => g.bitmap = b,
            (Self::Ink16(g), PageCanvas::Ink16(b)) => g.bitmap = b,
            _ => panic!("put_bitmap: the canvas differs in kind from the group"),
        }
    }

    /// Close the group without compositing it: its bitmap, the region
    /// cleared (see [`raster::transparency::discard_group`]).
    pub(super) fn discard(self) -> PageCanvas {
        self.discard_with_clip().1
    }

    /// [`Self::discard`] with the clip saved when the group was opened.
    fn discard_with_clip(self) -> (Clip, PageCanvas) {
        match self {
            Self::Rgb(g) => {
                let (c, b) = discard_group(g);
                (c, PageCanvas::Rgb(b))
            }
            Self::Rgb16(g) => {
                let (c, b) = discard_group(g);
                (c, PageCanvas::Rgb16(b))
            }
            Self::Ink(g) => {
                let (c, b) = discard_group(g);
                (c, PageCanvas::Ink(b))
            }
            Self::Ink16(g) => {
                let (c, b) = discard_group(g);
                (c, PageCanvas::Ink16(b))
            }
        }
    }

    /// The group moved onto `bitmap`, a canvas holding its result in
    /// another space, with its region and saved clip and no backdrop, and
    /// the bitmap it was painted on, its region cleared.
    pub(super) fn rehome(self, bitmap: PageCanvas) -> (Self, PageCanvas) {
        const fn group<P: Pixel>(
            bitmap: Bitmap<P>,
            saved_clip: Clip,
            params: GroupParams,
        ) -> GroupBitmap<P> {
            GroupBitmap {
                bitmap,
                saved_clip,
                params,
                backdrop: None,
            }
        }
        let (x0, y0, x1, y1) = self.region();
        #[expect(
            clippy::cast_possible_wrap,
            reason = "the region lies within a canvas, whose dimensions are below 2^31"
        )]
        let params = GroupParams {
            x_min: x0 as i32,
            y_min: y0 as i32,
            x_max: x1 as i32 - 1,
            y_max: y1 as i32 - 1,
            isolated: true,
        };
        let (saved_clip, cleared) = self.discard_with_clip();
        let rehomed = match bitmap {
            PageCanvas::Rgb(b) => Self::Rgb(group(b, saved_clip, params)),
            PageCanvas::Rgb16(b) => Self::Rgb16(group(b, saved_clip, params)),
            PageCanvas::Ink(b) => Self::Ink(group(b, saved_clip, params)),
            PageCanvas::Ink16(b) => Self::Ink16(group(b, saved_clip, params)),
        };
        (rehomed, cleared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::page::canvas::CanvasKind;

    fn no_spot_tables() -> SpotTables<'static> {
        SpotTables {
            narrow: [None; 4],
            wide: [None; 4],
        }
    }

    fn convert<S: CanvasSample>(
        from: CanvasSpace,
        to: CanvasSpace,
        gray: bool,
        src: &[S],
        n: usize,
    ) -> Vec<S> {
        let spots = no_spot_tables();
        let separator = Separator::Model;
        let c = SpaceConversion {
            from,
            to,
            gray,
            separator: &separator,
            spots: &spots,
        };
        let mut out = vec![S::default(); n];
        c.pixel(src, &mut out);
        out
    }

    #[test]
    fn ink_converts_by_the_process_model() {
        let ink = [0u8, 255, 255, 0, 0, 0, 0, 0];
        let rgb = convert(
            CanvasSpace::Ink,
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            false,
            &ink,
            3,
        );
        assert_eq!(rgb, color::cmyk::cmyk_to_rgb(0, 255, 255, 0));
    }

    /// The inverse of the process-ink model reproduces a colour the model
    /// prints within four counts.
    #[test]
    fn srgb_round_trips_through_ink_within_the_inverse_tolerance() {
        for rgb in [
            [200u8, 30, 60],
            [10, 120, 200],
            [128, 128, 128],
            [255, 255, 255],
        ] {
            let ink = convert(
                CanvasSpace::Rgb(WorkingSpace::Srgb),
                CanvasSpace::Ink,
                false,
                &rgb,
                8,
            );
            assert_eq!(&ink[4..], &[0; 4], "no spot ink");
            let back = convert(
                CanvasSpace::Ink,
                CanvasSpace::Rgb(WorkingSpace::Srgb),
                false,
                &ink,
                3,
            );
            for (a, b) in rgb.iter().zip(&back) {
                assert!(a.abs_diff(*b) <= 4, "{rgb:?} → {ink:?} → {back:?}");
            }
        }
    }

    #[test]
    fn gray_reductions_take_the_device_gray() {
        let red = [255u8, 0, 0];
        let g = convert(
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            true,
            &red,
            3,
        );
        assert_eq!(g, [76, 76, 76]);
        let black = [0u8, 0, 0, 255, 0, 0, 0, 0];
        let g = convert(
            CanvasSpace::Ink,
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            true,
            &black,
            3,
        );
        assert_eq!(g, [0, 0, 0]);
    }

    #[test]
    fn a_neutral_keeps_its_value_into_the_wide_space() {
        let gray = [30_000u16; 3];
        let wide = convert(
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            CanvasSpace::Rgb(WorkingSpace::Wide),
            false,
            &gray,
            3,
        );
        assert_eq!(wide, gray);
        let red = [65_535u16, 0, 0];
        let wide = convert(
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            CanvasSpace::Rgb(WorkingSpace::Wide),
            false,
            &red,
            3,
        );
        let back = convert(
            CanvasSpace::Rgb(WorkingSpace::Wide),
            CanvasSpace::Rgb(WorkingSpace::Srgb),
            false,
            &wide,
            3,
        );
        assert!(
            red.iter().zip(&back).all(|(a, b)| a.abs_diff(*b) <= 32),
            "{back:?}"
        );
    }

    #[test]
    fn a_region_converts_painted_pixels_and_copies_alpha() {
        let mut src = PageCanvas::new(CanvasKind::Rgb, 3, 1, true);
        if let PageCanvas::Rgb(b) = &mut src {
            let (c, a) = b.row_and_alpha_mut(0);
            c[..3].copy_from_slice(&[200, 30, 60]);
            a.unwrap()[0] = 128;
            c[3..6].copy_from_slice(&[9, 9, 9]);
        }
        let mut dst = PageCanvas::new(CanvasKind::Ink, 3, 1, true);
        let spots = no_spot_tables();
        let conversion = SpaceConversion {
            from: CanvasSpace::Rgb(WorkingSpace::Srgb),
            to: CanvasSpace::Ink,
            gray: false,
            separator: &Separator::Model,
            spots: &spots,
        };
        src.convert_region_into(&mut dst, PixelRect::new(0, 0, 3, 1).unwrap(), &conversion);
        let PageCanvas::Ink(b) = dst else {
            unreachable!()
        };
        assert_eq!(b.alpha_row(0).unwrap(), &[128, 0, 0]);
        let want = no_spots(color::cmyk::rgb_to_ink([200, 30, 60]));
        assert_eq!(&b.row_samples(0)[..8], &want);
        assert_eq!(
            &b.row_samples(0)[8..],
            &[0; 16],
            "unpainted pixels stay empty"
        );
    }
}
