//! Shared constructors and guards for [`RenderedPage`].
//!
//! Both the PDF render path and the comic-archive input path produce the same
//! grayscale `RenderedPage`; this module is the single definition of how one is
//! born from a `Bitmap<Gray8>` and of the pixel-dimension safety limits.

use color::Gray8;
use raster::Bitmap;

use crate::{PageDiagnostics, RasterError, RenderedPage};

// ── Safety limits ───────────────────────────────────────────────────────────

/// Maximum pixel dimension (width or height) accepted for a rendered page.
///
/// Prevents absurdly large allocations from malformed or adversarial inputs.
/// 32 768 px at 150 DPI corresponds to roughly 366 inches (~9.3 metres).
pub const MAX_PX_DIMENSION: u32 = 32_768;

/// Maximum total pixel area (width × height) accepted for a rendered page.
///
/// [`MAX_PX_DIMENSION`] bounds each side independently but says nothing about
/// their product: a page whose width and height are *both* just under the
/// per-side limit (e.g. 30 000 × 30 000) passes the per-side check yet forces a
/// single ~2.7 GB RGB allocation — an unbounded-allocation soft-DoS that lives
/// *inside* the per-side limit. The total raster size is bounded here, not just
/// each side.
///
/// 600 000 000 px ≈ 600 MP ≈ 1.8 GiB at 3 bytes/px (RGB8). The headroom is
/// deliberate: the largest legitimate page expected is roughly A0
/// (841 × 1189 mm ≈ 33.1 × 46.8 in) at 600 DPI ≈ 19 860 × 28 080 ≈ 5.6e8 px,
/// which still fits with margin, while the absurd 30 000 × 30 000 = 9e8 px
/// case is rejected before any buffer is allocated. The product is computed in
/// `u64`: `MAX_PX_DIMENSION² = 32_768² ≈ 1.07e9` already overflows `u32`, so a
/// `u32` area computation would itself be a latent overflow bug — the wrap
/// could make a hostile page *pass*. `u64` cannot overflow for any
/// `u32 × u32` product and never panics.
pub const MAX_PX_AREA: u64 = 600_000_000;

/// Reject pixel dimensions that exceed the per-side or total-area safety limits.
///
/// Shared by the PDF render path and the comic-archive decode path so both
/// enforce the identical soft-DoS bound. `width`/`height` are pixel counts.
///
/// # Errors
///
/// [`RasterError::PageTooLarge`] if either side exceeds [`MAX_PX_DIMENSION`];
/// [`RasterError::PageAreaTooLarge`] if the area exceeds [`MAX_PX_AREA`].
pub fn validate_dimensions(width: u32, height: u32) -> Result<(), RasterError> {
    if width > MAX_PX_DIMENSION || height > MAX_PX_DIMENSION {
        return Err(RasterError::PageTooLarge { width, height });
    }
    let area = u64::from(width) * u64::from(height);
    if area > MAX_PX_AREA {
        return Err(RasterError::PageAreaTooLarge {
            width,
            height,
            area,
        });
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
    effective_dpi: f32,
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
        pixels,
        dpi,
        effective_dpi,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_in_bounds() {
        assert!(validate_dimensions(1024, 768).is_ok());
        assert!(validate_dimensions(MAX_PX_DIMENSION, 1).is_ok());
    }

    #[test]
    fn validate_rejects_oversized_side() {
        assert!(matches!(
            validate_dimensions(MAX_PX_DIMENSION + 1, 1),
            Err(RasterError::PageTooLarge { .. })
        ));
    }

    #[test]
    fn validate_rejects_oversized_area_within_sides() {
        let side = 30_000u32; // 30k <= MAX_PX_DIMENSION (32_768); 30k^2 = 9e8 > 6e8
        assert!(side <= MAX_PX_DIMENSION);
        assert!(matches!(
            validate_dimensions(side, side),
            Err(RasterError::PageAreaTooLarge { .. })
        ));
    }

    #[test]
    fn constructor_flattens_and_fills_fields() {
        let mut bmp = Bitmap::<Gray8>::new(2, 2, 4, false); // row_pad 4 -> padded stride
        bmp.row_bytes_mut(0)[..2].copy_from_slice(&[10, 20]);
        bmp.row_bytes_mut(1)[..2].copy_from_slice(&[30, 40]);
        let page = gray8_to_rendered_page(&bmp, 7, 300.0, 312.0, PageDiagnostics::default());
        assert_eq!(page.page_num, 7);
        assert_eq!((page.width, page.height), (2, 2));
        assert_eq!(
            page.pixels,
            vec![10, 20, 30, 40],
            "must be tight, padding stripped"
        );
        assert_eq!(page.pixels.len(), 4);
    }
}
