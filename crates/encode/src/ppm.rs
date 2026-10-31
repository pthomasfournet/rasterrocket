//! Netpbm P6 (binary PPM) encoder.
//!
//! PPM stores RGB 8-bit pixels.  Pixel modes that are not natively RGB are
//! converted row by row:
//!
//! | Mode | Conversion |
//! |------|-----------|
//! | `Rgb8` | verbatim |
//! | `Bgr8` / `Xbgr8` | channel swap; the `Xbgr8` padding byte is dropped |
//! | `Cmyk8` | the process-ink model of `color::cmyk` |
//! | `DeviceN8` | the process channels only, as for `Cmyk8` |
//! | `Mono8` / `Mono1` | not supported — use [`write_pgm`][crate::write_pgm] |
//!
//! The output is a standard `P6` header followed by raw RGB bytes.

use std::io::{self, Write};

use color::cmyk::{cmyk_to_rgb, cmyk_to_rgb_row};
use color::{Pixel, PixelMode};
use raster::Bitmap;

use crate::EncodeError;

/// Write `bitmap` to `out` as a binary PPM (`P6`) image.
///
/// The sink `out` is consumed; wrap in `std::io::BufWriter` if buffering is
/// needed.
///
/// # Errors
///
/// Returns [`EncodeError::UnsupportedMode`] for grayscale or 1-bit modes —
/// use [`write_pgm`][crate::write_pgm] for those.
/// Returns [`EncodeError::Io`] on any I/O failure.
pub fn write_ppm<P: Pixel, W: Write>(bitmap: &Bitmap<P>, mut out: W) -> Result<(), EncodeError> {
    let source = RgbSource::of(P::MODE).ok_or(EncodeError::UnsupportedMode(
        "grayscale/mono bitmap: use write_pgm instead",
    ))?;
    write_ppm_header(&mut out, bitmap.width, bitmap.height)?;
    let mut rgb_row = vec![0u8; bitmap.width as usize * 3];
    for y in 0..bitmap.height {
        source.convert(bitmap.row_bytes(y), &mut rgb_row);
        out.write_all(&rgb_row)?;
    }
    out.flush()?;
    Ok(())
}

/// Write the P6 header.
fn write_ppm_header<W: Write>(out: &mut W, width: u32, height: u32) -> io::Result<()> {
    writeln!(out, "P6")?;
    writeln!(out, "{width} {height}")?;
    writeln!(out, "255")?;
    Ok(())
}

// ── RGB row conversion ────────────────────────────────────────────────────────

/// The byte layout of a pixel mode that converts to RGB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RgbSource {
    /// `[R, G, B]`, copied verbatim.
    Rgb,
    /// `[B, G, R]`, channel-swapped.
    Bgr,
    /// `[X, B, G, R]`; the padding byte is dropped.
    Xbgr,
    /// `[C, M, Y, K]`, through the process-ink model of `color::cmyk`.
    Cmyk,
    /// `[C, M, Y, K, S0, S1, S2, S3]`; the spot channels are dropped and the
    /// process channels converted as for `Cmyk`.
    DeviceN,
}

impl RgbSource {
    /// The layout of `mode`, or `None` for a grayscale or 1-bit mode, which
    /// carries no colour to convert.
    pub(crate) const fn of(mode: PixelMode) -> Option<Self> {
        match mode {
            PixelMode::Rgb8 => Some(Self::Rgb),
            PixelMode::Bgr8 => Some(Self::Bgr),
            PixelMode::Xbgr8 => Some(Self::Xbgr),
            PixelMode::Cmyk8 => Some(Self::Cmyk),
            PixelMode::DeviceN8 => Some(Self::DeviceN),
            PixelMode::Mono1 | PixelMode::Mono8 => None,
        }
    }

    /// Bytes per source pixel.
    const fn bytes(self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Xbgr | Self::Cmyk => 4,
            Self::DeviceN => 8,
        }
    }

    /// Convert the first `dst.len() / 3` pixels of the row `src` into `dst`.
    ///
    /// # Panics
    ///
    /// Panics if `dst.len()` is not a multiple of 3 or `src` is shorter
    /// than `dst.len() / 3` pixels.
    pub(crate) fn convert(self, src: &[u8], dst: &mut [u8]) {
        assert!(
            dst.len().is_multiple_of(3),
            "RgbSource::convert: RGB row length {} is not a multiple of 3",
            dst.len()
        );
        let src = &src[..dst.len() / 3 * self.bytes()];
        match self {
            Self::Rgb => dst.copy_from_slice(src),
            Self::Bgr => {
                for (o, &[b, g, r]) in dst.as_chunks_mut().0.iter_mut().zip(src.as_chunks().0) {
                    *o = [r, g, b];
                }
            }
            Self::Xbgr => {
                for (o, &[_, b, g, r]) in dst.as_chunks_mut().0.iter_mut().zip(src.as_chunks().0) {
                    *o = [r, g, b];
                }
            }
            Self::Cmyk => cmyk_to_rgb_row(src, dst),
            Self::DeviceN => {
                for (o, px) in dst
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(src.as_chunks::<8>().0)
                {
                    *o = cmyk_to_rgb(px[0], px[1], px[2], px[3]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Cmyk8, DeviceN8, Rgb8};
    use raster::Bitmap;

    fn make_rgb_bitmap(w: u32, h: u32, fill: [u8; 3]) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::new(w, h, 1, false);
        for y in 0..h {
            let row = bmp.row_bytes_mut(y);
            for chunk in row.as_chunks_mut::<3>().0 {
                *chunk = fill;
            }
        }
        bmp
    }

    /// Parse the P6 header and return the byte offset of the first pixel.
    fn header_len(out: &[u8]) -> usize {
        // Header format: "P6\n{w} {h}\n255\n" — find the third newline.
        let mut newlines = 0usize;
        for (i, &b) in out.iter().enumerate() {
            if b == b'\n' {
                newlines += 1;
                if newlines == 3 {
                    return i + 1;
                }
            }
        }
        panic!("malformed PPM header");
    }

    #[test]
    fn rgb_ppm_header_and_pixels() {
        let bmp = make_rgb_bitmap(2, 1, [255, 128, 0]);
        let mut out = Vec::new();
        write_ppm::<Rgb8, _>(&bmp, &mut out).unwrap();

        let expected_header = b"P6\n2 1\n255\n";
        assert!(
            out.starts_with(expected_header),
            "header mismatch: {:?}",
            &out[..expected_header.len().min(out.len())]
        );

        let pixels = &out[expected_header.len()..];
        assert_eq!(pixels.len(), 6, "2 pixels × 3 bytes");
        assert_eq!(&pixels[..3], &[255, 128, 0]);
        assert_eq!(&pixels[3..6], &[255, 128, 0]);
    }

    #[test]
    fn cmyk_black_converts_to_printed_black() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 255) = solid K, the printed black of process ink.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 255]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(&out[hlen..], &cmyk_to_rgb(0, 0, 0, 255), "solid K");
    }

    #[test]
    fn cmyk_white_converts_to_rgb_white() {
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(1, 1, 1, false);
        // CMYK (0, 0, 0, 0) = pure white.
        bmp.row_bytes_mut(0).copy_from_slice(&[0, 0, 0, 0]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(&out[hlen..], &[255, 255, 255], "CMYK white → RGB white");
    }

    #[test]
    fn devicen_uses_only_cmyk_portion() {
        let mut bmp: Bitmap<DeviceN8> = Bitmap::new(1, 1, 1, false);
        // DeviceN8: CMYK=(0,0,0,0) → white; spot channels ignored.
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[0, 0, 0, 0, 99, 99, 99, 99]);
        let mut out = Vec::new();
        write_ppm::<DeviceN8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        assert_eq!(
            &out[hlen..],
            &[255, 255, 255],
            "DeviceN spot channels must be ignored"
        );
    }

    #[test]
    fn stride_padding_not_written() {
        // width=1, pad=4 → stride=4 for Rgb8 (3 bytes/px rounded up to 4).
        let bmp: Bitmap<Rgb8> = Bitmap::new(1, 1, 4, false);
        let mut out = Vec::new();
        write_ppm::<Rgb8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        // Must be exactly 3 pixel bytes, not 4 (stride).
        assert_eq!(
            out.len() - hlen,
            3,
            "stride padding must not appear in PPM output"
        );
    }

    #[test]
    fn mono8_returns_unsupported_error() {
        use color::Gray8;
        let bmp: Bitmap<Gray8> = Bitmap::new(1, 1, 1, false);
        let mut out = Vec::new();
        let result = write_ppm::<Gray8, _>(&bmp, &mut out);
        assert!(
            matches!(result, Err(EncodeError::UnsupportedMode(_))),
            "Gray8 should return UnsupportedMode"
        );
    }

    #[test]
    fn cmyk_ppm_multi_pixel_layout() {
        // 2×2 with four distinct CMYK pixels — covers per-pixel CMYK→RGB
        // conversion across multiple rows and per-row destination stride.
        let mut bmp: Bitmap<Cmyk8> = Bitmap::new(2, 2, 1, false);
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[10, 0, 0, 0, 0, 20, 0, 0]);
        bmp.row_bytes_mut(1)
            .copy_from_slice(&[0, 0, 30, 0, 40, 50, 60, 0]);
        let mut out = Vec::new();
        write_ppm::<Cmyk8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        let pixels = &out[hlen..];
        assert_eq!(pixels.len(), 12, "2×2 × 3 RGB bytes");
        assert_eq!(&pixels[0..3], &cmyk_to_rgb(10, 0, 0, 0), "row 0 px 0");
        assert_eq!(&pixels[3..6], &cmyk_to_rgb(0, 20, 0, 0), "row 0 px 1");
        assert_eq!(&pixels[6..9], &cmyk_to_rgb(0, 0, 30, 0), "row 1 px 0");
        assert_eq!(&pixels[9..12], &cmyk_to_rgb(40, 50, 60, 0), "row 1 px 1");
    }

    #[test]
    fn devicen_ignores_spot_channels_per_pixel() {
        // 2×1 DeviceN8 (8 bytes/pixel = CMYK + 4 spot) verifying that the
        // spot bytes are skipped per pixel and only the CMYK portion drives
        // the RGB output.
        let mut bmp: Bitmap<DeviceN8> = Bitmap::new(2, 1, 1, false);
        // px0 CMYK=(10, 0, 0, 0), spot=99..; px1 CMYK=(0, 40, 80, 0), spot=88..
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[10, 0, 0, 0, 99, 99, 99, 99, 0, 40, 80, 0, 88, 88, 88, 88]);
        let mut out = Vec::new();
        write_ppm::<DeviceN8, _>(&bmp, &mut out).unwrap();
        let hlen = header_len(&out);
        let mut want = cmyk_to_rgb(10, 0, 0, 0).to_vec();
        want.extend_from_slice(&cmyk_to_rgb(0, 40, 80, 0));
        assert_eq!(
            &out[hlen..],
            want.as_slice(),
            "spot bytes must be skipped per pixel; only CMYK drives the RGB output"
        );
    }

    #[test]
    fn every_colour_layout_converts_one_pixel() {
        let cases: [(RgbSource, &[u8], [u8; 3]); 5] = [
            (RgbSource::Rgb, &[1, 2, 3], [1, 2, 3]),
            (RgbSource::Bgr, &[1, 2, 3], [3, 2, 1]),
            (RgbSource::Xbgr, &[9, 1, 2, 3], [3, 2, 1]),
            (
                RgbSource::Cmyk,
                &[10, 20, 30, 40],
                cmyk_to_rgb(10, 20, 30, 40),
            ),
            (
                RgbSource::DeviceN,
                &[10, 20, 30, 40, 99, 99, 99, 99],
                cmyk_to_rgb(10, 20, 30, 40),
            ),
        ];
        for (source, src, want) in cases {
            let mut dst = [0u8; 3];
            source.convert(src, &mut dst);
            assert_eq!(dst, want, "{source:?}");
        }
    }

    /// Bytes past the converted pixels, such as stride padding, are ignored.
    #[test]
    fn trailing_source_bytes_are_ignored() {
        let mut dst = [0u8; 6];
        RgbSource::Xbgr.convert(&[0, 1, 2, 3, 0, 4, 5, 6, 0xEE, 0xEE], &mut dst);
        assert_eq!(dst, [3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn grey_modes_have_no_rgb_layout() {
        assert_eq!(RgbSource::of(PixelMode::Mono8), None);
        assert_eq!(RgbSource::of(PixelMode::Mono1), None);
        assert_eq!(RgbSource::of(PixelMode::Xbgr8), Some(RgbSource::Xbgr));
    }
}
