//! Rasterized glyph bitmap with owned pixel data.

/// A rasterized glyph, either 8-bit grey (anti-aliased) or 1-bit packed
/// (non-anti-aliased).
///
/// `x_off` is the signed horizontal distance from the left edge of the
/// bitmap to the pen origin (the left edge lies at `pen_x - x_off`), and
/// `y_off` is the signed distance from the baseline up to the top edge (the
/// top row lies at `pen_y - y_off` in y-down device space).
///
/// # Data layout
///
/// - Anti-aliased (`aa == true`): `width` bytes per row, `height` rows,
///   row stride = `width` (no padding).
/// - Mono (`aa == false`): `(width + 7) / 8` bytes per row (MSB-first packed),
///   `height` rows, row stride = `(width + 7) / 8`.
#[derive(Clone, Debug)]
pub struct GlyphBitmap {
    /// Signed horizontal offset from the left edge to the pen origin (pixels).
    pub x_off: i32,
    /// Signed vertical offset from baseline to top edge (pixels, up = positive).
    pub y_off: i32,
    /// Glyph width in pixels.
    pub width: u32,
    /// Glyph height in pixels.
    pub height: u32,
    /// `true` → 8-bit alpha per pixel; `false` → 1-bit packed MSB-first.
    pub aa: bool,
    /// Raw pixel data, owned.
    pub data: Vec<u8>,
}

impl GlyphBitmap {
    /// Bytes per row: `width` for AA, `width.div_ceil(8)` for mono.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        if self.aa {
            self.width as usize
        } else {
            (self.width as usize).div_ceil(8)
        }
    }

    /// Total expected data size in bytes.
    #[must_use]
    pub const fn data_len(&self) -> usize {
        self.row_bytes().saturating_mul(self.height as usize)
    }

    /// Return the slice for row `y` (0-indexed from the top).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height` or if `data` is shorter than expected.
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(
            y < self.height,
            "row {y} out of range (height {})",
            self.height
        );
        let rb = self.row_bytes();
        let start = y as usize * rb;
        &self.data[start..start + rb]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aa(w: u32, h: u32) -> GlyphBitmap {
        GlyphBitmap {
            x_off: 0,
            y_off: 0,
            width: w,
            height: h,
            aa: true,
            data: vec![0xAAu8; (w * h) as usize],
        }
    }

    fn make_mono(w: u32, h: u32) -> GlyphBitmap {
        let rb = w.div_ceil(8) as usize;
        GlyphBitmap {
            x_off: 0,
            y_off: 0,
            width: w,
            height: h,
            aa: false,
            data: vec![0xFFu8; rb * h as usize],
        }
    }

    #[test]
    fn aa_row_bytes() {
        assert_eq!(make_aa(10, 5).row_bytes(), 10);
    }

    #[test]
    fn mono_row_bytes_rounded_up() {
        assert_eq!(make_mono(9, 3).row_bytes(), 2); // (9+7)/8 = 2
        assert_eq!(make_mono(8, 1).row_bytes(), 1);
        assert_eq!(make_mono(1, 1).row_bytes(), 1);
    }

    #[test]
    fn data_len_matches_row_bytes_times_height() {
        let bmp = make_aa(4, 6);
        assert_eq!(bmp.data_len(), 24);
    }

    #[test]
    fn row_access_correct_slice() {
        let bmp = make_aa(3, 2);
        assert_eq!(bmp.row(0).len(), 3);
        assert_eq!(bmp.row(1).len(), 3);
    }

    #[test]
    #[should_panic(expected = "row 2 out of range")]
    fn row_oob_panics() {
        let _ = make_aa(3, 2).row(2);
    }
}
