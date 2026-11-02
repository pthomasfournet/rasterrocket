//! Netpbm P4 (binary PBM) encoder.
//!
//! PBM stores 1-bit monochrome pixels, MSB first, rows padded to whole bytes.
//! Only `Gray8` / `Mono8` bitmaps are accepted.  Their values are luma, as
//! for every other encoder of this crate: a pixel below
//! [`crate::BLACK_BELOW`] is black (a 1 bit, per the P4 spec), any other
//! white.

use std::io::{self, Write};

use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::{EncodeError, pack_bilevel_row};

/// Write `bitmap` to `out` as a binary PBM (`P4`) image.
///
/// Input pixels are `Gray8` luma: below [`crate::BLACK_BELOW`] → black (1 bit), others → white
/// (0 bit).  Rows are MSB-packed and padded to whole bytes, matching the P4 spec.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for non-grayscale modes.
/// Returns [`EncodeError::Io`] on any I/O failure.
pub fn write_pbm<P: Pixel, W: Write>(bitmap: &Bitmap<P>, mut out: W) -> Result<(), EncodeError> {
    match P::MODE {
        PixelMode::Mono8 => {}
        PixelMode::Mono1
        | PixelMode::Rgb8
        | PixelMode::Bgr8
        | PixelMode::Xbgr8
        | PixelMode::Cmyk8
        | PixelMode::DeviceN8 => {
            return Err(EncodeError::UnsupportedMode(
                "write_pbm accepts only Gray8 (Mono8) bitmaps",
            ));
        }
    }

    write_pbm_header(&mut out, bitmap.width, bitmap.height)?;

    let w = bitmap.width as usize; // u32 → usize: lossless on all ≥32-bit targets
    let row_bytes_out = w.div_ceil(8); // packed byte count per output row (MSB first, P4 spec)
    let mut packed = vec![0u8; row_bytes_out];

    for y in 0..bitmap.height {
        let row = bitmap.row_bytes(y);
        let pixels = &row[..w]; // live pixels only (exclude stride padding)

        pack_bilevel_row(pixels, &mut packed);
        out.write_all(&packed)?;
    }

    out.flush()?;
    Ok(())
}

fn write_pbm_header<W: Write>(out: &mut W, width: u32, height: u32) -> io::Result<()> {
    writeln!(out, "P4")?;
    writeln!(out, "{width} {height}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};
    use raster::Bitmap;

    /// A `w × h` white bitmap.
    fn make_gray_bitmap(w: u32, h: u32) -> Bitmap<Gray8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            bmp.row_bytes_mut(y).fill(255);
        }
        bmp
    }

    fn gray_fill(bmp: &mut Bitmap<Gray8>, row: u32, values: &[u8]) {
        let r = bmp.row_bytes_mut(row);
        r[..values.len()].copy_from_slice(values);
    }

    #[test]
    fn pbm_header_format() {
        let bmp = make_gray_bitmap(8, 1);
        let mut out = Vec::new();
        write_pbm::<Gray8, _>(&bmp, &mut out).unwrap();
        assert!(
            out.starts_with(b"P4\n8 1\n"),
            "header: {:?}",
            &out[..12.min(out.len())]
        );
    }

    #[test]
    fn all_white_is_zero_byte() {
        // All pixels white (255) → packed byte = 0x00
        let bmp = make_gray_bitmap(8, 1);
        let mut out = Vec::new();
        write_pbm::<Gray8, _>(&bmp, &mut out).unwrap();
        let header_len = b"P4\n8 1\n".len();
        assert_eq!(out[header_len], 0x00, "all-white row must be 0x00");
    }

    #[test]
    fn all_black_is_ff_byte() {
        let mut bmp = make_gray_bitmap(8, 1);
        gray_fill(&mut bmp, 0, &[0u8; 8]);
        let mut out = Vec::new();
        write_pbm::<Gray8, _>(&bmp, &mut out).unwrap();
        let header_len = b"P4\n8 1\n".len();
        assert_eq!(out[header_len], 0xFF, "all-black row must be 0xFF");
    }

    #[test]
    fn alternating_checkerboard() {
        // Luma 0, 255, 0, 255, … → black, white, … → bits 1010_1010 = 0xAA;
        // 127 is the lightest black and 128 the darkest white.
        let mut bmp = make_gray_bitmap(8, 1);
        gray_fill(&mut bmp, 0, &[0, 255, 127, 128, 0, 200, 50, 255]);
        let mut out = Vec::new();
        write_pbm::<Gray8, _>(&bmp, &mut out).unwrap();
        let header_len = b"P4\n8 1\n".len();
        assert_eq!(out[header_len], 0xAA);
    }

    #[test]
    fn row_padding_when_width_not_multiple_of_8() {
        // 3 pixels wide → 1 packed byte per row; bits beyond pixel 2 must be 0.
        let mut bmp = make_gray_bitmap(3, 1);
        gray_fill(&mut bmp, 0, &[0, 0, 0]);
        let mut out = Vec::new();
        write_pbm::<Gray8, _>(&bmp, &mut out).unwrap();
        let header_len = b"P4\n3 1\n".len();
        // 3 black pixels → top 3 bits set: 1110_0000 = 0xE0
        assert_eq!(out[header_len], 0xE0, "3 black pixels must pack to 0xE0");
        assert_eq!(
            out.len(),
            header_len + 1,
            "3-pixel row occupies 1 packed byte"
        );
    }

    #[test]
    fn rgb8_returns_unsupported_error() {
        let bmp: Bitmap<Rgb8> = Bitmap::new(1, 1, 1, false);
        let result = write_pbm::<Rgb8, _>(&bmp, std::io::sink());
        assert!(matches!(result, Err(EncodeError::UnsupportedMode(_))));
    }
}
