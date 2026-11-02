//! Tone operations on a page between rendering and encoding: conversion to
//! the colour model of the output, inversion, and the reduction to two
//! levels.

use color::{Gray8, Pixel};
use encode::BLACK_BELOW;
use raster::{Bitmap, HalftoneScreen, ScreenParams};

use crate::args::{Args, InvertWhen};
use crate::cleanup::Page;

/// A page is reversed out when more than `GROUND_NUMERATOR /
/// GROUND_DENOMINATOR` of its pixels share one dark tone band: a luma range
/// of [`GROUND_BAND`] levels below [`BLACK_BELOW`].  Light content on a
/// flat dark ground leaves the ground well over this fraction; a page of
/// dense black text inks well under a quarter of its area, and a dark
/// photograph or a page of saturated colours spreads its dark pixels over
/// many bands.
const GROUND_NUMERATOR: u64 = 3;
const GROUND_DENOMINATOR: u64 = 5;

/// The width in luma levels of the tone band of a dark ground.
const GROUND_BAND: usize = 32;

/// `page` made ready for encoding under `args`: in the colour model of the
/// output (luma for `--gray` and `--mono`, colour otherwise), inverted as
/// `--invert` selects, and for `--mono` reduced to two levels (see
/// [`two_levels`]).
pub(crate) fn prepare(page: Page, args: &Args) -> Page {
    let mut page = if args.luma_output() {
        Page::Gray(page.into_luma())
    } else {
        Page::Rgb(page.into_rgb())
    };
    let invert = match args.invert_when() {
        InvertWhen::Never => false,
        InvertWhen::Always => true,
        InvertWhen::Auto => page.with_luma(is_reversed_out),
    };
    if invert {
        match &mut page {
            Page::Rgb(rgb) => complement(rgb),
            Page::Gray(luma) | Page::Mono(luma) => complement(luma),
        }
    }
    if args.mono {
        Page::Mono(two_levels(&page.into_luma(), args.screen()))
    } else {
        page
    }
}

/// Whether `luma` is light content on a flat dark ground: see
/// [`GROUND_NUMERATOR`].  An empty page is not.
fn is_reversed_out(luma: &Bitmap<Gray8>) -> bool {
    let w = luma.width as usize;
    let mut histogram = [0u64; 256];
    for y in 0..luma.height {
        for &level in &luma.row_bytes(y)[..w] {
            histogram[usize::from(level)] += 1;
        }
    }
    let ground = histogram[..usize::from(BLACK_BELOW)]
        .windows(GROUND_BAND)
        .map(|band| band.iter().sum::<u64>())
        .max()
        .unwrap_or(0);
    let total = u64::from(luma.width) * u64::from(luma.height);
    ground * GROUND_DENOMINATOR > total * GROUND_NUMERATOR
}

/// Complement every sample of `bmp`: the negative of an additive or luma
/// image.
fn complement<P: Pixel>(bmp: &mut Bitmap<P>) {
    let live = bmp.width as usize * P::BYTES;
    for y in 0..bmp.height {
        for byte in &mut bmp.row_bytes_mut(y)[..live] {
            *byte = !*byte;
        }
    }
}

/// Reduce a luma bitmap to two levels, 0 (black) and 255 (white): through
/// the halftone `screen` when given, which renders intermediate grays as
/// dot patterns, else at the midpoint, values below [`BLACK_BELOW`] black.
pub(crate) fn two_levels(src: &Bitmap<Gray8>, screen: Option<ScreenParams>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    let mut screen = screen.map(HalftoneScreen::new);
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        match &mut screen {
            None => {
                for (dst_px, &luma) in dst_row.iter_mut().zip(src_row) {
                    *dst_px = if luma < BLACK_BELOW { 0 } else { 255 };
                }
            }
            Some(screen) => {
                // The screen tiles with a power-of-two period, so the
                // coordinates enter it modulo that period and may wrap.
                let row = y.cast_signed();
                for ((x, dst_px), &luma) in (0i32..).zip(dst_row.iter_mut()).zip(src_row) {
                    *dst_px = if screen.test(x, row, luma) { 255 } else { 0 };
                }
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    use super::{prepare, two_levels};
    use crate::args::args_with as args;
    use crate::cleanup::Page;

    #[test]
    fn mono_is_luma_black_ink_on_white() {
        let mut gray: Bitmap<Gray8> = Bitmap::new(3, 1, 1, false);
        gray.row_bytes_mut(0)[..3].copy_from_slice(&[0, 127, 128]);
        let mono = two_levels(&gray, None);
        assert_eq!(&mono.row_bytes(0)[..3], &[0, 0, 255]);
    }

    #[test]
    fn halftone_renders_gray_as_a_proportion_of_dots() {
        let screen = args(&["--mono", "--dither", "halftone", "-r", "200"]).screen();
        assert!(screen.is_some());
        for (level, low, high) in [(0u8, 0, 0), (64, 40, 90), (128, 100, 156), (255, 256, 256)] {
            let mut gray: Bitmap<Gray8> = Bitmap::new(16, 16, 1, false);
            for y in 0..16 {
                gray.row_bytes_mut(y)[..16].fill(level);
            }
            let mono = two_levels(&gray, screen);
            let white: usize = (0..16)
                .map(|y| {
                    mono.row_bytes(y)[..16]
                        .iter()
                        .map(|&v| usize::from(v / 255))
                        .sum::<usize>()
                })
                .sum();
            assert!(
                (low..=high).contains(&white),
                "level {level}: {white} of 256 white"
            );
        }
    }

    /// A gray page of ten pixels, `dark` of them black and the rest white.
    fn page(dark: usize) -> Page {
        let mut gray: Bitmap<Gray8> = Bitmap::new(10, 1, 4, false);
        for (i, px) in gray.row_bytes_mut(0)[..10].iter_mut().enumerate() {
            *px = if i < dark { 0 } else { 255 };
        }
        Page::Gray(gray)
    }

    fn luma(page: &Page) -> Vec<u8> {
        match page {
            Page::Gray(g) | Page::Mono(g) => g.row_bytes(0)[..10].to_vec(),
            Page::Rgb(_) => panic!("a luma page"),
        }
    }

    #[test]
    fn pages_take_the_colour_model_of_the_output() {
        assert!(matches!(
            prepare(page(2), &args(&["--mono"])),
            Page::Mono(_)
        ));
        assert!(matches!(
            prepare(page(2), &args(&["--gray"])),
            Page::Gray(_)
        ));
        let Page::Rgb(rgb) = prepare(page(2), &args(&[])) else {
            panic!("colour output takes a colour page");
        };
        assert_eq!(&rgb.row_bytes(0)[..9], &[0, 0, 0, 0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn invert_auto_turns_only_mostly_dark_pages() {
        let auto = args(&["--gray", "--invert=auto"]);
        // Six of ten on a black ground is not more than three fifths; seven
        // is.
        assert_eq!(luma(&prepare(page(6), &auto)), luma(&page(6)));
        assert_eq!(
            luma(&prepare(page(7), &auto)),
            luma(&page(3)).into_iter().rev().collect::<Vec<_>>()
        );
        assert_eq!(luma(&prepare(page(7), &args(&["--gray"]))), luma(&page(7)));
        let always = args(&["--gray", "--invert"]);
        assert_eq!(luma(&prepare(page(2), &always))[..3], [255, 255, 0]);
    }

    #[test]
    fn invert_auto_leaves_dark_pages_without_a_flat_ground() {
        // Ten dark pixels spread over 100 luma levels: dark, but no ground.
        let spread: Vec<u8> = (0..10).map(|i| i * 11).collect();
        let mut gray: Bitmap<Gray8> = Bitmap::new(10, 1, 4, false);
        gray.row_bytes_mut(0)[..10].copy_from_slice(&spread);
        let out = prepare(Page::Gray(gray), &args(&["--gray", "--invert=auto"]));
        assert_eq!(luma(&out), spread);
        // A dark navy ground, not black, still counts.
        let mut gray: Bitmap<Gray8> = Bitmap::new(10, 1, 4, false);
        gray.row_bytes_mut(0)[..10].copy_from_slice(&[40, 45, 50, 40, 44, 52, 41, 255, 255, 255]);
        let out = prepare(Page::Gray(gray), &args(&["--gray", "--invert=auto"]));
        assert_eq!(luma(&out)[7], 0);
    }

    #[test]
    fn colour_pages_invert_every_channel_and_turn_gray_for_gray_output() {
        let rgb = || {
            let mut rgb: Bitmap<Rgb8> = Bitmap::new(1, 1, 4, false);
            rgb.row_bytes_mut(0)[..3].copy_from_slice(&[10, 200, 0]);
            Page::Rgb(rgb)
        };
        let Page::Rgb(out) = prepare(rgb(), &args(&["--invert"])) else {
            panic!("colour output stays colour");
        };
        assert_eq!(&out.row_bytes(0)[..3], &[245, 55, 255]);
        assert!(matches!(prepare(rgb(), &args(&["--gray"])), Page::Gray(_)));
    }
}
