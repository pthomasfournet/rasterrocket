//! Property tests: every encoder's output, read back by an independent
//! decoder, holds the bitmap's live pixels.
//!
//! Bitmaps have random sizes, row padding (with garbage in the padding),
//! alpha planes and contents; bilevel pages are drawn as runs of random
//! lengths, long enough for every fax make-up code.  Each property runs the
//! case count given to `config` unless `PROPTEST_CASES` sets another.

use std::io::Cursor;
use std::num::NonZeroU32;

use color::cmyk::cmyk_to_rgb;
use color::{Cmyk8, DeviceN8, Gray8, Pixel, Rgb8};
use hayro_ccitt::{DecodeSettings, DecoderContext, EncodeScheme, Encoder, EncodingMode};
use proptest::prelude::*;
use raster::Bitmap;
use rasterrocket_encode::{
    BLACK_BELOW, JpegOptions, RowImage, TiffCompression, write_jpeg, write_pbm, write_pgm,
    write_png, write_ppm, write_tiff,
};

#[path = "../../../tests/support/proptest_config.rs"]
mod proptest_config;
use proptest_config::config;

/// A bitmap's construction: size, row padding, padding fill, contents and
/// alpha plane.
#[derive(Debug, Clone)]
struct Image {
    w: u32,
    h: u32,
    pad: usize,
    pad_fill: u8,
    /// `w × h × bytes-per-pixel` live bytes, row-major.
    live: Vec<u8>,
    alpha: Option<Vec<u8>>,
}

impl Image {
    fn bitmap<P: Pixel<Sample = u8>>(&self) -> Bitmap<P> {
        let mut bmp: Bitmap<P> = Bitmap::new_filled(
            self.w,
            self.h,
            self.pad,
            self.alpha.is_some(),
            self.pad_fill,
        );
        let row = self.w as usize * P::BYTES;
        for (y, src) in (0..self.h).zip(self.live.chunks_exact(row)) {
            bmp.row_bytes_mut(y)[..row].copy_from_slice(src);
        }
        if let Some(a) = &self.alpha {
            bmp.alpha_plane_mut().expect("allocated").copy_from_slice(a);
        }
        bmp
    }

    fn pixels(&self, bytes: usize) -> impl Iterator<Item = &[u8]> {
        self.live.chunks_exact(bytes)
    }
}

/// An image of `bytes`-byte pixels, up to `max_w × max_h`.
fn image(bytes: usize, max_w: u32, max_h: u32, alpha: bool) -> impl Strategy<Value = Image> {
    (
        1..=max_w,
        1..=max_h,
        1usize..=16,
        any::<u8>(),
        any::<bool>(),
    )
        .prop_flat_map(move |(w, h, pad, pad_fill, with_alpha)| {
            let n = w as usize * h as usize;
            let alpha = if alpha && with_alpha {
                prop::collection::vec(any::<u8>(), n).prop_map(Some).boxed()
            } else {
                Just(None).boxed()
            };
            (prop::collection::vec(any::<u8>(), n * bytes), alpha).prop_map(move |(live, alpha)| {
                Image {
                    w,
                    h,
                    pad,
                    pad_fill,
                    live,
                    alpha,
                }
            })
        })
}

/// A gray image of alternating black and white runs, each 1 to 3 000
/// pixels long but mostly short, `w` up to `max_w`.  A black run's level is
/// below [`BLACK_BELOW`], a white run's at or above it, often the level
/// next to the threshold.
fn bilevel(max_w: u32, max_h: u32) -> impl Strategy<Value = Image> {
    let len = prop_oneof![8 => 1usize..16, 3 => 16usize..300, 1 => 300usize..3000];
    let below = prop_oneof![Just(BLACK_BELOW - 1), 0..BLACK_BELOW];
    let at_or_above = prop_oneof![Just(BLACK_BELOW), BLACK_BELOW..=u8::MAX];
    let run = (len, below, at_or_above);
    (1..=max_w, 1..=max_h, 1usize..=16, any::<bool>()).prop_flat_map(move |(w, h, pad, first)| {
        let n = w as usize * h as usize;
        prop::collection::vec(run.clone(), 1..64).prop_map(move |runs| {
            let mut live = Vec::with_capacity(n);
            let mut black = first;
            for &(len, dark, light) in runs.iter().cycle() {
                if live.len() >= n {
                    break;
                }
                let gray = if black { dark } else { light };
                live.extend(std::iter::repeat_n(gray, len.min(n - live.len())));
                black = !black;
            }
            Image {
                w,
                h,
                pad,
                pad_fill: 0x5a,
                live,
                alpha: None,
            }
        })
    })
}

/// The body of a binary Netpbm file after its `magic`, size and (for
/// gray and colour) maxval lines, as written by this crate.
fn netpbm_body<'a>(file: &'a [u8], magic: &str, w: u32, h: u32, maxval: bool) -> &'a [u8] {
    let header = if maxval {
        format!("{magic}\n{w} {h}\n255\n")
    } else {
        format!("{magic}\n{w} {h}\n")
    };
    assert!(file.starts_with(header.as_bytes()), "header of {magic}");
    &file[header.len()..]
}

/// Rows of `w` pixels, packed one bit per pixel, most significant first,
/// 1 where `black` holds.
fn packed(live: &[u8], w: u32, black: impl Fn(u8) -> bool) -> Vec<u8> {
    let w = w as usize;
    let mut out = Vec::new();
    for row in live.chunks_exact(w) {
        let mut bytes = vec![0u8; w.div_ceil(8)];
        for (x, &v) in row.iter().enumerate() {
            if black(v) {
                bytes[x / 8] |= 0x80 >> (x % 8);
            }
        }
        out.extend(bytes);
    }
    out
}

fn png_decode(file: &[u8]) -> (png::ColorType, u32, u32, Vec<u8>) {
    let mut reader = png::Decoder::new(Cursor::new(file))
        .read_info()
        .expect("a PNG header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("a size")];
    let frame = reader.next_frame(&mut buf).expect("a PNG frame");
    buf.truncate(frame.buffer_size());
    assert_eq!(frame.bit_depth, png::BitDepth::Eight);
    (frame.color_type, frame.width, frame.height, buf)
}

fn tiff_decode(file: &[u8]) -> (tiff::ColorType, (u32, u32), Vec<u8>) {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(file)).expect("a TIFF header");
    let colortype = decoder.colortype().expect("a colour type");
    let dims = decoder.dimensions().expect("dimensions");
    let tiff::decoder::DecodingResult::U8(pixels) = decoder.read_image().expect("an image") else {
        panic!("8-bit samples expected");
    };
    (colortype, dims, pixels)
}

/// Rows decoded by the fax decoder, packed 1 black.
struct Rows {
    bytes: Vec<u8>,
    width: u32,
}

impl hayro_ccitt::Decoder for Rows {
    fn push_row(&mut self, changes: &[u32]) {
        let start = self.bytes.len();
        self.bytes
            .resize(start + self.width.div_ceil(8) as usize, 0);
        let ends = changes.iter().copied().chain([self.width]);
        for (i, (from, to)) in changes.iter().copied().zip(ends.skip(1)).enumerate() {
            if i % 2 == 0 {
                for x in from..to {
                    self.bytes[start + (x / 8) as usize] |= 0x80 >> (x % 8);
                }
            }
        }
    }
}

fn fax_scheme() -> impl Strategy<Value = EncodeScheme> {
    prop_oneof![
        Just(EncodeScheme::Group4),
        (prop::option::of(1u32..=8), any::<bool>()).prop_map(|(k, byte_aligned_eol)| {
            EncodeScheme::Group3 {
                k: k.and_then(NonZeroU32::new),
                byte_aligned_eol,
            }
        }),
    ]
}

fn tiff_compression() -> impl Strategy<Value = TiffCompression> {
    prop_oneof![
        Just(TiffCompression::None),
        Just(TiffCompression::PackBits),
        Just(TiffCompression::Lzw),
        Just(TiffCompression::Deflate),
    ]
}

proptest! {
    #![proptest_config(config(256))]

    /// PGM holds a gray bitmap's live bytes; PBM its luma below `BLACK_BELOW` as
    /// black bits; PNG the same gray bytes.
    #[test]
    fn gray_outputs_hold_the_pixels(img in image(1, 300, 240, false)) {
        let bmp = img.bitmap::<Gray8>();
        let mut pgm = Vec::new();
        write_pgm(&bmp, &mut pgm).expect("PGM");
        prop_assert_eq!(netpbm_body(&pgm, "P5", img.w, img.h, true), &img.live[..]);

        let mut pbm = Vec::new();
        write_pbm(&bmp, &mut pbm).expect("PBM");
        prop_assert_eq!(
            netpbm_body(&pbm, "P4", img.w, img.h, false),
            &packed(&img.live, img.w, |v| v < BLACK_BELOW)[..]
        );

        let mut file = Vec::new();
        write_png(&bmp, &mut file, None).expect("PNG");
        let (color, w, h, pixels) = png_decode(&file);
        prop_assert_eq!((color, w, h), (png::ColorType::Grayscale, img.w, img.h));
        prop_assert_eq!(pixels, img.live);
    }

    /// PPM and PNG hold an RGB bitmap's live bytes, PNG with its alpha
    /// plane interleaved when it has one; CMYK and `DeviceN` bitmaps hold
    /// the process-ink conversion of their process channels.
    #[test]
    fn colour_outputs_hold_the_pixels(
        rgb in image(3, 200, 120, true),
        cmyk in image(4, 64, 64, false),
        devicen in image(8, 48, 48, true),
    ) {
        let check = |img: &Image, file_ppm: &[u8], file_png: &[u8], want: Vec<u8>| {
            prop_assert_eq!(netpbm_body(file_ppm, "P6", img.w, img.h, true), &want[..]);
            let (color, w, h, pixels) = png_decode(file_png);
            prop_assert_eq!((w, h), (img.w, img.h));
            match &img.alpha {
                None => {
                    prop_assert_eq!(color, png::ColorType::Rgb);
                    prop_assert_eq!(pixels, want);
                }
                Some(alpha) => {
                    prop_assert_eq!(color, png::ColorType::Rgba);
                    let rgba: Vec<u8> = want
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .zip(alpha)
                        .flat_map(|(px, &a)| [px[0], px[1], px[2], a])
                        .collect();
                    prop_assert_eq!(pixels, rgba);
                }
            }
            Ok(())
        };
        let (mut ppm, mut png) = (Vec::new(), Vec::new());
        let bmp = rgb.bitmap::<Rgb8>();
        write_ppm(&bmp, &mut ppm).expect("PPM");
        write_png(&bmp, &mut png, None).expect("PNG");
        check(&rgb, &ppm, &png, rgb.live.clone())?;

        let to_rgb = |px: &[u8]| cmyk_to_rgb(px[0], px[1], px[2], px[3]);
        let (mut ppm, mut png) = (Vec::new(), Vec::new());
        let bmp = cmyk.bitmap::<Cmyk8>();
        write_ppm(&bmp, &mut ppm).expect("PPM");
        write_png(&bmp, &mut png, None).expect("PNG");
        check(&cmyk, &ppm, &png, cmyk.pixels(4).flat_map(to_rgb).collect())?;

        let (mut ppm, mut png) = (Vec::new(), Vec::new());
        let bmp = devicen.bitmap::<DeviceN8>();
        write_ppm(&bmp, &mut ppm).expect("PPM");
        write_png(&bmp, &mut png, None).expect("PNG");
        check(&devicen, &ppm, &png, devicen.pixels(8).flat_map(to_rgb).collect())?;
    }

    /// TIFF holds gray, RGB and RGBA pixels under every general-purpose
    /// compression.
    #[test]
    fn tiff_holds_gray_and_rgb_pixels(
        gray in image(1, 160, 96, false),
        rgb in image(3, 96, 64, true),
        compression in tiff_compression(),
    ) {
        let mut file = Vec::new();
        write_tiff(RowImage::Gray(&gray.bitmap()), (150.0, 150.0), compression, &mut file)
            .expect("gray TIFF");
        let (color, dims, pixels) = tiff_decode(&file);
        prop_assert_eq!((color, dims), (tiff::ColorType::Gray(8), (gray.w, gray.h)));
        prop_assert_eq!(pixels, gray.live);

        let mut file = Vec::new();
        write_tiff(RowImage::Rgb(&rgb.bitmap()), (72.0, 96.0), compression, &mut file)
            .expect("RGB TIFF");
        let (color, dims, pixels) = tiff_decode(&file);
        prop_assert_eq!(dims, (rgb.w, rgb.h));
        match &rgb.alpha {
            None => {
                prop_assert_eq!(color, tiff::ColorType::RGB(8));
                prop_assert_eq!(pixels, rgb.live);
            }
            Some(alpha) => {
                prop_assert_eq!(color, tiff::ColorType::RGBA(8));
                let rgba: Vec<u8> = rgb
                    .live
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(alpha)
                    .flat_map(|(px, &a)| [px[0], px[1], px[2], a])
                    .collect();
                prop_assert_eq!(pixels, rgba);
            }
        }
    }

    /// A bilevel TIFF holds the page's black pixels under every
    /// compression, Group 4 included; the reader returns `WhiteIsZero`
    /// samples inverted, its 1 bits white.
    #[test]
    fn bilevel_tiff_holds_the_black_pixels(
        page in bilevel(2600, 12),
        compression in prop_oneof![tiff_compression(), Just(TiffCompression::Group4)],
    ) {
        let mut file = Vec::new();
        write_tiff(RowImage::Bilevel(&page.bitmap()), (204.0, 196.0), compression, &mut file)
            .expect("bilevel TIFF");
        let (_, dims, pixels) = tiff_decode(&file);
        prop_assert_eq!(dims, (page.w, page.h));
        let black: Vec<u8> = pixels.iter().map(|b| !b).collect();
        // Bits past the last pixel of a row are not pixels.
        let width = page.w as usize;
        let mask = |row: &[u8]| {
            let mut row = row.to_vec();
            if !width.is_multiple_of(8) {
                *row.last_mut().expect("a byte") &= 0xff << (8 - width % 8);
            }
            row
        };
        let got: Vec<u8> = black.chunks_exact(width.div_ceil(8)).flat_map(mask).collect();
        prop_assert_eq!(got, packed(&page.live, page.w, |v| v < BLACK_BELOW));
    }

    /// The fax coder's output decodes back to its rows under every
    /// scheme: Group 3 one- and two-dimensional, any `k`, EOLs byte-aligned
    /// or not, and Group 4.
    #[test]
    fn fax_coding_round_trips(page in bilevel(3000, 10), scheme in fax_scheme()) {
        let rows = packed(&page.live, page.w, |v| v < BLACK_BELOW);
        let width = NonZeroU32::new(page.w).expect("non-zero width");
        let mut encoder = Encoder::new(width, scheme);
        for row in rows.chunks_exact(page.w.div_ceil(8) as usize) {
            encoder.encode_packed_row(row);
        }
        let data = encoder.finish();

        let mut settings = DecodeSettings::new(page.w, page.h);
        settings.strict = true;
        match scheme {
            EncodeScheme::Group4 => settings.encoding = EncodingMode::Group4,
            EncodeScheme::Group3 { k, byte_aligned_eol } => {
                settings.end_of_line = true;
                settings.rows_are_byte_aligned = byte_aligned_eol;
                settings.encoding = if k.is_some() {
                    EncodingMode::Group3_2D
                } else {
                    EncodingMode::Group3_1D
                };
            }
        }
        let mut decoded = Rows { bytes: Vec::new(), width: page.w };
        let _consumed = hayro_ccitt::decode(&data, &mut decoded, &mut DecoderContext::new(settings))
            .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        prop_assert_eq!(decoded.bytes, rows);
    }

    /// JPEG output of any size decodes to that size, and a flat image to
    /// its colour within the quantisation error of quality 100.
    #[test]
    fn jpeg_output_decodes_to_its_size(
        img in image(3, 40, 40, false),
        flat in any::<[u8; 3]>(),
        gray in any::<bool>(),
        progressive in any::<bool>(),
        optimize in any::<bool>(),
    ) {
        let options = JpegOptions { quality: 100, progressive, optimize };
        let channels = if gray { 1 } else { 3 };
        let n = img.w as usize * img.h as usize;
        let noise: Vec<u8> = img.live.iter().copied().take(n * channels).collect();
        let solid: Vec<u8> = flat[..channels].repeat(n);
        for (live, is_flat) in [(noise, false), (solid, true)] {
            let source = Image { live, ..img.clone() };
            let mut file = Vec::new();
            if gray {
                write_jpeg(&source.bitmap::<Gray8>(), options, &mut file, None)
            } else {
                write_jpeg(&source.bitmap::<Rgb8>(), options, &mut file, None)
            }
            .expect("JPEG");
            let mut decoder =
                zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&file));
            let pixels = decoder.decode().map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
            prop_assert_eq!(decoder.dimensions(), Some((img.w as usize, img.h as usize)));
            if is_flat {
                let step = pixels.len() / n;
                for px in pixels.chunks_exact(step) {
                    for (c, &want) in px.iter().zip(&flat[..channels]) {
                        prop_assert!(c.abs_diff(want) <= 3, "{:?} decoded as {:?}", flat, px);
                    }
                }
            }
        }
    }
}
