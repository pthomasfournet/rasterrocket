//! Shared constructors and guards for [`RenderedPage`].
//!
//! Both the PDF render path and the comic-archive input path produce the same
//! grayscale `RenderedPage`; this module is the single definition of how one is
//! born from a `Bitmap<Gray8>` and of the pixel-dimension safety limits.

use color::{Gray8, Gray16, Rgb8, Rgb16, SampleDepth};
use raster::Bitmap;

use crate::{PageDiagnostics, PixelFormat, Pixels, RasterError, RenderedPage};

// ── Safety limits ───────────────────────────────────────────────────────────

/// Largest pixel width or height of a page accepted at any byte budget.
///
/// Every integer pixel coordinate up to 2^24 is exact in `f32`, the sum or
/// difference of two page coordinates and a small margin stays inside `i32`,
/// and the tiling-pattern lattice, enumerated within 2^30 px of its origin,
/// covers the whole page.  A page box of 14 400 units at the largest
/// `UserUnit` (10) is 2 000 in, 4.8 M px at 2 400 dpi.
pub const MAX_PX_SIDE: u32 = 1 << 24;

/// Default of [`SessionConfig::max_page_bytes`](crate::SessionConfig::max_page_bytes):
/// 1.8 GB, a 600 Mpx page of 8-bit RGB.
///
/// A0 at 600 dpi (19 860 × 28 080 ≈ 5.6e8 px) fits; a 30 000 × 30 000 RGB
/// page (2.7 GB) is refused before anything is allocated.
pub const DEFAULT_MAX_PAGE_BYTES: u64 = 1_800_000_000;

/// Default of [`SessionConfig::band_bytes`](crate::SessionConfig::band_bytes):
/// 64 MiB, 1 412 rows of a 44-inch-wide page at 360 dpi in 8-bit RGB.
pub const DEFAULT_BAND_BYTES: u64 = 64 << 20;

/// The pixel layout of a page canvas: RGB samples of `depth`, and an alpha
/// plane of the same depth over a transparent backdrop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CanvasLayout {
    /// Bits per sample.
    pub depth: SampleDepth,
    /// Whether the canvas has an alpha plane.
    pub alpha: bool,
}

impl CanvasLayout {
    /// The canvas of `depth`, with an alpha plane when `alpha`.
    #[must_use]
    pub const fn new(depth: SampleDepth, alpha: bool) -> Self {
        Self { depth, alpha }
    }

    /// Bytes a pixel of the canvas holds: 3 for 8-bit RGB, 4 with alpha;
    /// 6 and 8 at 16 bits.
    #[must_use]
    pub const fn bytes_per_px(self) -> u32 {
        (3 + self.alpha as u32) * self.depth.bytes()
    }
}

/// Reject a `width × height` picture of `bytes_per_px` bytes per pixel
/// whose side exceeds [`MAX_PX_SIDE`] or whose size exceeds `max_bytes`.
///
/// Shared by the PDF render path and the comic-archive decode path.  The
/// size is computed in `u64`, which no `u32 × u32 × u32` product of these
/// operands overflows (`2^24 · 2^24 · bytes_per_px` once the side check has
/// passed).
///
/// # Errors
///
/// [`RasterError::PageExceedsCeiling`] if either side exceeds
/// [`MAX_PX_SIDE`]; [`RasterError::PageTooLarge`] if
/// `width · height · bytes_per_px > max_bytes`.
pub fn validate_page_bytes(
    width: u32,
    height: u32,
    bytes_per_px: u32,
    max_bytes: u64,
) -> Result<(), RasterError> {
    check_sides(width, height)?;
    let bytes = u64::from(width) * u64::from(height) * u64::from(bytes_per_px);
    if bytes > max_bytes {
        return Err(RasterError::PageTooLarge {
            width,
            height,
            bytes,
            limit: max_bytes,
        });
    }
    Ok(())
}

/// [`RasterError::PageExceedsCeiling`] if either side exceeds [`MAX_PX_SIDE`].
pub(crate) const fn check_sides(width: u32, height: u32) -> Result<(), RasterError> {
    if width > MAX_PX_SIDE || height > MAX_PX_SIDE {
        return Err(RasterError::PageExceedsCeiling { width, height });
    }
    Ok(())
}

/// Flatten a tightly-or-padded `Bitmap<Gray8>` into a tight `width*height`
/// pixel buffer and assemble a [`RenderedPage`].
///
/// The single place that knows how `RenderedPage`'s pixel/stride/dpi fields are
/// populated, shared by every producer.
#[must_use]
pub fn gray8_to_rendered_page(
    bmp: &Bitmap<Gray8>,
    page_num: u32,
    dpi: f32,
    diagnostics: PageDiagnostics,
) -> RenderedPage {
    let w = bmp.width as usize;
    let mut pixels = Vec::with_capacity(w * bmp.height as usize);
    for y in 0..bmp.height {
        pixels.extend_from_slice(&bmp.row_bytes(y)[..w]);
    }
    RenderedPage {
        page_num,
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::U8(pixels),
        format: PixelFormat::Gray8,
        dpi,
        diagnostics,
    }
}

/// [`gray8_to_rendered_page`] taking the bitmap by value.
///
/// The page takes over a packed bitmap's buffer (`stride == width`, as
/// [`rgb_into_gray`](crate::rgb_into_gray) returns); `shrink_to_fit` then
/// releases its spare capacity, which an allocator may implement as a copy
/// into a `width * height` allocation. A padded bitmap is copied row by row.
#[must_use]
pub fn gray8_into_rendered_page(
    bmp: Bitmap<Gray8>,
    page_num: u32,
    dpi: f32,
    diagnostics: PageDiagnostics,
) -> RenderedPage {
    if bmp.stride != bmp.width as usize {
        return gray8_to_rendered_page(&bmp, page_num, dpi, diagnostics);
    }
    let (width, height) = (bmp.width, bmp.height);
    let mut pixels = bmp.into_data();
    pixels.shrink_to_fit();
    RenderedPage {
        page_num,
        width,
        height,
        pixels: Pixels::U8(pixels),
        format: PixelFormat::Gray8,
        dpi,
        diagnostics,
    }
}

/// Interleave an RGB bitmap and its alpha plane into a tight
/// `width * height * 4` RGBA buffer, alpha not premultiplied, and assemble
/// a [`PixelFormat::Rgba8`] [`RenderedPage`].  A bitmap without an alpha
/// plane is opaque: every alpha byte is 255.
pub(crate) fn rgb8_to_rgba_rendered_page(
    bmp: &Bitmap<Rgb8>,
    page_num: u32,
    dpi: f32,
    diagnostics: PageDiagnostics,
) -> RenderedPage {
    let row_bytes = bmp.width as usize * 4;
    let mut pixels = vec![0; row_bytes * bmp.height as usize];
    rgb_rows_to_rgba(bmp, &mut pixels, row_bytes);
    RenderedPage {
        page_num,
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::U8(pixels),
        format: PixelFormat::Rgba8,
        dpi,
        diagnostics,
    }
}

/// A [`PixelFormat::Gray16`] [`RenderedPage`] of `bmp`, its rows packed.
pub(crate) fn gray16_into_rendered_page(
    bmp: &Bitmap<Gray16>,
    page_num: u32,
    dpi: f32,
    diagnostics: PageDiagnostics,
) -> RenderedPage {
    let w = bmp.width as usize;
    let mut pixels = Vec::with_capacity(w * bmp.height as usize);
    for y in 0..bmp.height {
        pixels.extend_from_slice(&bmp.row_samples(y)[..w]);
    }
    RenderedPage {
        page_num,
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::U16(pixels),
        format: PixelFormat::Gray16,
        dpi,
        diagnostics,
    }
}

/// [`rgb8_to_rgba_rendered_page`] at 16 bits: a [`PixelFormat::Rgba16`]
/// page, alpha not premultiplied, opaque where the bitmap has no alpha
/// plane.
pub(crate) fn rgb16_to_rgba16_rendered_page(
    bmp: &Bitmap<Rgb16>,
    page_num: u32,
    dpi: f32,
    diagnostics: PageDiagnostics,
) -> RenderedPage {
    let w = bmp.width as usize;
    let mut pixels = Vec::with_capacity(w * 4 * bmp.height as usize);
    for y in 0..bmp.height {
        let alpha = bmp.alpha_row(y);
        for (x, rgb) in bmp.row_samples(y)[..w * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            pixels.extend_from_slice(rgb);
            pixels.push(alpha.map_or(u16::MAX, |a| a[x]));
        }
    }
    RenderedPage {
        page_num,
        width: bmp.width,
        height: bmp.height,
        pixels: Pixels::U16(pixels),
        format: PixelFormat::Rgba16,
        dpi,
        diagnostics,
    }
}

/// Write the pixels of `bmp` as RGBA to `out`, row `y` at
/// `out[y * stride..y * stride + width * 4]`, alpha not premultiplied; a
/// bitmap without an alpha plane is opaque.
///
/// `out` must hold `(height - 1) * stride + width * 4` bytes and
/// `stride ≥ width * 4`.
pub(crate) fn rgb_rows_to_rgba(bmp: &Bitmap<Rgb8>, out: &mut [u8], stride: usize) {
    let w = bmp.width as usize;
    for y in 0..bmp.height {
        raster::interleave_rgba(
            &bmp.row_bytes(y)[..w * 3],
            bmp.alpha_row(y),
            &mut out[y as usize * stride..][..w * 4],
        );
    }
}

/// Copy the RGB rows of `bmp` to `out`, row `y` at
/// `out[y * stride..y * stride + width * 3]`.
///
/// `out` must hold `(height - 1) * stride + width * 3` bytes and
/// `stride ≥ width * 3`.
pub(crate) fn rgb_rows_copy(bmp: &Bitmap<Rgb8>, out: &mut [u8], stride: usize) {
    let row = bmp.width as usize * 3;
    for y in 0..bmp.height {
        out[y as usize * stride..][..row].copy_from_slice(&bmp.row_bytes(y)[..row]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_canvas_pixel_holds_three_or_four_samples_of_its_depth() {
        let bytes = |depth, alpha| CanvasLayout::new(depth, alpha).bytes_per_px();
        assert_eq!(bytes(SampleDepth::Eight, false), 3);
        assert_eq!(bytes(SampleDepth::Eight, true), 4);
        assert_eq!(bytes(SampleDepth::Sixteen, false), 6);
        assert_eq!(bytes(SampleDepth::Sixteen, true), 8);
    }

    #[test]
    fn the_budget_bounds_the_bytes_of_the_page() {
        let limit = 3 * 1000 * 1000;
        assert!(validate_page_bytes(1000, 1000, 3, limit).is_ok());
        assert!(matches!(
            validate_page_bytes(1000, 1000, 4, limit),
            Err(RasterError::PageTooLarge {
                bytes: 4_000_000,
                limit: 3_000_000,
                ..
            })
        ));
        assert!(matches!(
            validate_page_bytes(1001, 1000, 3, limit),
            Err(RasterError::PageTooLarge { .. })
        ));
    }

    #[test]
    fn the_default_budget_keeps_the_rgb_page_bound() {
        assert!(validate_page_bytes(20_000, 30_000, 3, DEFAULT_MAX_PAGE_BYTES).is_ok());
        assert!(matches!(
            validate_page_bytes(30_000, 30_000, 3, DEFAULT_MAX_PAGE_BYTES),
            Err(RasterError::PageTooLarge { .. })
        ));
    }

    #[test]
    fn the_side_ceiling_holds_at_any_budget() {
        assert!(validate_page_bytes(MAX_PX_SIDE, 1, 3, u64::MAX).is_ok());
        for (w, h) in [
            (MAX_PX_SIDE + 1, 1),
            (1, MAX_PX_SIDE + 1),
            (u32::MAX, u32::MAX),
        ] {
            assert!(matches!(
                validate_page_bytes(w, h, 8, u64::MAX),
                Err(RasterError::PageExceedsCeiling { .. })
            ));
        }
    }

    #[test]
    fn constructor_flattens_and_fills_fields() {
        let mut bmp = Bitmap::<Gray8>::new(2, 2, 4, false); // row_pad 4 -> padded stride
        bmp.row_bytes_mut(0)[..2].copy_from_slice(&[10, 20]);
        bmp.row_bytes_mut(1)[..2].copy_from_slice(&[30, 40]);
        let page = gray8_to_rendered_page(&bmp, 7, 300.0, PageDiagnostics::default());
        assert_eq!(page.page_num, 7);
        assert_eq!((page.width, page.height), (2, 2));
        assert_eq!(
            page.pixels,
            Pixels::U8(vec![10, 20, 30, 40]),
            "must be tight, padding stripped"
        );
        assert_eq!(page.pixels.len(), 4);
        assert_eq!(page.format, PixelFormat::Gray8);
    }

    #[test]
    fn rgba_page_interleaves_the_alpha_plane() {
        let mut bmp = Bitmap::<Rgb8>::new(2, 1, 4, true);
        bmp.row_bytes_mut(0)[..6].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        bmp.alpha_plane_mut().unwrap().copy_from_slice(&[0, 200]);
        let page = rgb8_to_rgba_rendered_page(&bmp, 1, 72.0, PageDiagnostics::default());
        assert_eq!(page.format, PixelFormat::Rgba8);
        assert_eq!(page.pixels, Pixels::U8(vec![1, 2, 3, 0, 4, 5, 6, 200]));
        let opaque = Bitmap::<Rgb8>::new(1, 1, 1, false);
        let page = rgb8_to_rgba_rendered_page(&opaque, 1, 72.0, PageDiagnostics::default());
        assert_eq!(page.pixels.as_u8().unwrap()[3], u8::MAX);
    }

    #[test]
    fn sixteen_bit_pages_pack_their_samples() {
        let mut gray = Bitmap::<Gray16>::new(2, 1, 4, false);
        gray.row_samples_mut(0)[..2].copy_from_slice(&[1, 65_535]);
        let page = gray16_into_rendered_page(&gray, 1, 72.0, PageDiagnostics::default());
        assert_eq!(
            (page.format, page.pixels),
            (PixelFormat::Gray16, Pixels::U16(vec![1, 65_535]))
        );
        let mut rgb = Bitmap::<Rgb16>::new(1, 1, 1, true);
        rgb.row_samples_mut(0).copy_from_slice(&[1, 2, 3]);
        rgb.alpha_plane_mut().unwrap()[0] = 400;
        let page = rgb16_to_rgba16_rendered_page(&rgb, 1, 72.0, PageDiagnostics::default());
        assert_eq!(page.format, PixelFormat::Rgba16);
        assert_eq!(page.pixels, Pixels::U16(vec![1, 2, 3, 400]));
        assert_eq!(page.format.bytes_per_pixel(), 8);
    }

    #[test]
    fn owned_constructor_matches_borrowing_one() {
        for row_pad in [1, 4] {
            let mut bmp = Bitmap::<Gray8>::new(3, 2, row_pad, false);
            bmp.row_bytes_mut(0)[..3].copy_from_slice(&[1, 2, 3]);
            bmp.row_bytes_mut(1)[..3].copy_from_slice(&[4, 5, 6]);
            let want = gray8_to_rendered_page(&bmp, 2, 150.0, PageDiagnostics::default());
            let got = gray8_into_rendered_page(bmp, 2, 150.0, PageDiagnostics::default());
            assert_eq!((got.page_num, got.width, got.height), (2, 3, 2));
            assert_eq!(got.pixels, want.pixels, "row_pad {row_pad}");
        }
    }
}
