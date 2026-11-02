//! Tone operations on a page between rendering and encoding: conversion to
//! luma for the gray and bilevel outputs, inversion, and the reduction to
//! two levels.

use color::{Gray8, Pixel};
use pdf_raster::rgb_to_gray;
use raster::{Bitmap, HalftoneScreen, ScreenParams};

use crate::args::{Args, InvertWhen};
use crate::cleanup::Page;

/// Luma threshold separating dark ink from light paper (inclusive on the
/// light side).
const MONO_THRESHOLD: u8 = 128;

/// A page is dark when more than `DARK_NUMERATOR / DARK_DENOMINATOR` of its
/// pixels lie below [`MONO_THRESHOLD`].  A page of dense black text inks
/// well under a quarter of its area; light text reversed out of a dark
/// ground leaves the ground well over this fraction.
const DARK_NUMERATOR: u64 = 3;
const DARK_DENOMINATOR: u64 = 5;

/// `page` made ready for encoding under `args`: in luma when the output is
/// gray or bilevel, and inverted as `--invert` selects.
pub(crate) fn prepare(page: Page, args: &Args) -> Page {
    let mut page = match page {
        Page::Rgb(rgb) if args.gray || args.mono => Page::Gray(rgb_to_gray(&rgb)),
        page => page,
    };
    let invert = match args.invert_when() {
        InvertWhen::Never => false,
        InvertWhen::Always => true,
        InvertWhen::Auto => is_dark(&page),
    };
    if invert {
        match &mut page {
            Page::Rgb(rgb) => complement(rgb),
            Page::Gray(gray) => complement(gray),
        }
    }
    page
}

/// Whether `page` is dark: see [`DARK_NUMERATOR`].
fn is_dark(page: &Page) -> bool {
    let converted;
    let gray = match page {
        Page::Gray(gray) => gray,
        Page::Rgb(rgb) => {
            converted = rgb_to_gray(rgb);
            &converted
        }
    };
    let w = gray.width as usize;
    let dark: u64 = (0..gray.height)
        .map(|y| {
            gray.row_bytes(y)[..w]
                .iter()
                .filter(|&&luma| luma < MONO_THRESHOLD)
                .count() as u64
        })
        .sum();
    let total = u64::from(gray.width) * u64::from(gray.height);
    dark * DARK_DENOMINATOR > total * DARK_NUMERATOR
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

/// Reduce a grayscale bitmap to two levels, as luma 0 (black) and 255
/// (white): through the halftone `screen` when given, which renders
/// intermediate grays as dot patterns, else at the 50% midpoint, values
/// below [`MONO_THRESHOLD`] black.
pub(crate) fn gray_to_mono(src: &Bitmap<Gray8>, screen: Option<ScreenParams>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(src.width, src.height, 1, false);
    let w = src.width as usize;
    let mut screen = screen.map(HalftoneScreen::new);
    for y in 0..src.height {
        let src_row = &src.row_bytes(y)[..w];
        let dst_row = &mut dst.row_bytes_mut(y)[..w];
        match &mut screen {
            None => {
                for (dst_px, &luma) in dst_row.iter_mut().zip(src_row) {
                    *dst_px = if luma < MONO_THRESHOLD { 0 } else { 255 };
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
    use clap::Parser as _;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    use super::{gray_to_mono, prepare};
    use crate::args::Args;
    use crate::cleanup::Page;

    #[test]
    fn mono_is_luma_black_ink_on_white() {
        let mut gray: Bitmap<Gray8> = Bitmap::new(3, 1, 1, false);
        gray.row_bytes_mut(0)[..3].copy_from_slice(&[0, 127, 128]);
        let mono = gray_to_mono(&gray, None);
        assert_eq!(&mono.row_bytes(0)[..3], &[0, 0, 255]);
    }

    #[test]
    fn halftone_renders_gray_as_a_proportion_of_dots() {
        let screen = Args::parse_from([
            "rrocket", "--mono", "--dither", "halftone", "-r", "200", "in.pdf", "out",
        ])
        .screen();
        assert!(screen.is_some());
        for (level, low, high) in [(0u8, 0, 0), (64, 40, 90), (128, 100, 156), (255, 256, 256)] {
            let mut gray: Bitmap<Gray8> = Bitmap::new(16, 16, 1, false);
            for y in 0..16 {
                gray.row_bytes_mut(y)[..16].fill(level);
            }
            let mono = gray_to_mono(&gray, screen);
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
            Page::Gray(g) => g.row_bytes(0)[..10].to_vec(),
            Page::Rgb(_) => panic!("a gray page"),
        }
    }

    fn args(flags: &[&str]) -> Args {
        let mut argv = vec!["rrocket"];
        argv.extend_from_slice(flags);
        argv.extend(["in.pdf", "out"]);
        Args::parse_from(argv)
    }

    #[test]
    fn invert_auto_turns_only_mostly_dark_pages() {
        let auto = args(&["--gray", "--invert=auto"]);
        // Six of ten dark is not more than three fifths; seven is.
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
    fn colour_pages_invert_every_channel_and_turn_gray_for_gray_output() {
        let rgb = || {
            let mut rgb: Bitmap<Rgb8> = Bitmap::new(1, 1, 4, false);
            rgb.row_bytes_mut(0)[..3].copy_from_slice(&[10, 200, 0]);
            Page::Rgb(rgb)
        };
        match prepare(rgb(), &args(&["--invert"])) {
            Page::Rgb(out) => assert_eq!(&out.row_bytes(0)[..3], &[245, 55, 255]),
            Page::Gray(_) => panic!("colour output stays colour"),
        }
        assert!(matches!(prepare(rgb(), &args(&["--mono"])), Page::Gray(_)));
    }
}
