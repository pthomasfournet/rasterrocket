//! Bitmap representation for JBIG2 decoding.
//!
//! "The variable whose value is the result of this decoding procedure is shown
//! in Table 3." (6.2.3)
//!
//! "GBREG - The decoded region bitmap." (Table 3)

use crate::budget::Budget;
use crate::decode::CombinationOperator;
use crate::error::{DecodeError, OverflowError, Result, bail};
use alloc::vec::Vec;

// To guard better against malicious files, we limit bitmap dimensions
// to `u16::MAX`, which should be more than enough.
pub(crate) const MAX_DIMENSION: u32 = u16::MAX as u32;

/// The underlying word type for packed pixel storage.
/// Change this to `u32` to switch back to 32-bit words.
pub(crate) type Word = u32;

pub(crate) const WORD_BITS: u32 = Word::BITS;
pub(crate) const WORD_BYTES: u32 = WORD_BITS / u8::BITS;
pub(crate) const WORD_SHIFT: u32 = WORD_BITS - 1;

/// A decoded bitmap with position information.
///
/// "Pixels decoded by the MMR decoder having the value 'black' shall be treated
/// as having the value 1. Pixels decoded by the MMR decoder having the value
/// 'white' shall be treated as having the value 0." (6.2.6)
#[derive(Debug, Default)]
pub(crate) struct Bitmap {
    /// Width in pixels.
    pub(crate) width: u32,
    /// Height in pixels.
    pub(crate) height: u32,
    /// Number of words per row.
    pub(crate) stride: u32,
    /// Packed pixel data, one bit per pixel, row-major order.
    /// Each row is padded to a word boundary.
    pub(crate) data: Vec<Word>,
    /// "This four-byte field gives the horizontal offset in pixels of the bitmap
    /// encoded in this segment relative to the page bitmap." (7.4.1.3)
    pub(crate) x_location: u32,
    /// "This four-byte field gives the vertical offset in pixels of the bitmap
    /// encoded in this segment relative to the page bitmap." (7.4.1.4)
    pub(crate) y_location: u32,
}

impl Bitmap {
    /// Create a new bitmap filled with white pixels (false), charged to
    /// `budget`.
    ///
    /// The bitmap is positioned at (0, 0).
    pub(crate) fn new(width: u32, height: u32, budget: &Budget) -> Result<Self> {
        Self::new_with(width, height, 0, 0, false, budget)
    }

    /// Like [`Bitmap::new`], but bounded by area (65 535² pixels) instead of
    /// by each side.
    pub(crate) fn new_wide(width: u32, height: u32, budget: &Budget) -> Result<Self> {
        if u64::from(width) * u64::from(height) > u64::from(MAX_DIMENSION).pow(2) {
            bail!(OverflowError::BitmapDimension);
        }

        let stride = width.div_ceil(WORD_BITS);

        Ok(Self {
            width,
            height,
            stride,
            data: budget.filled_vec(stride as usize * height as usize, 0)?,
            x_location: 0,
            y_location: 0,
        })
    }

    /// Create a new bitmap with full configuration, charged to `budget`.
    pub(crate) fn new_with(
        width: u32,
        height: u32,
        x_location: u32,
        y_location: u32,
        default_pixel: bool,
        budget: &Budget,
    ) -> Result<Self> {
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            bail!(OverflowError::BitmapDimension);
        }

        let stride = width.div_ceil(WORD_BITS);
        let default_word: Word = if default_pixel { !0 } else { 0 };
        let data = budget.filled_vec(stride as usize * height as usize, default_word)?;
        let mut bitmap = Self {
            width,
            height,
            stride,
            data,
            x_location,
            y_location,
        };

        if default_pixel {
            bitmap.clear_padding();
        }

        Ok(bitmap)
    }

    /// A copy of the bitmap, charged to `budget`.
    pub(crate) fn try_clone(&self, budget: &Budget) -> Result<Self> {
        let mut data = budget.vec_with_capacity(self.data.len())?;
        data.extend_from_slice(&self.data);

        Ok(Self {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data,
            x_location: self.x_location,
            y_location: self.y_location,
        })
    }

    pub(crate) fn reinitialize(
        &mut self,
        width: u32,
        height: u32,
        default_pixel: bool,
    ) -> Result<()> {
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            bail!(OverflowError::BitmapDimension);
        }

        let stride = width.div_ceil(WORD_BITS);
        let default_word: Word = if default_pixel { !0 } else { 0 };
        self.width = width;
        self.height = height;
        self.stride = stride;
        self.x_location = 0;
        self.y_location = 0;
        let len = stride as usize * height as usize;
        self.data.clear();
        if self.data.try_reserve_exact(len).is_err() {
            bail!(DecodeError::OutOfMemory);
        }
        self.data.resize(len, default_word);

        if default_pixel {
            self.clear_padding();
        }

        Ok(())
    }

    fn clear_padding(&mut self) {
        let trailing = self.width % WORD_BITS;
        if trailing == 0 {
            return;
        }

        let mask: Word = !0 << (WORD_BITS - trailing);
        for row in self.data.chunks_exact_mut(self.stride as usize) {
            if let Some(last) = row.last_mut() {
                *last &= mask;
            }
        }
    }

    /// Get a pixel value at (x, y).
    #[inline]
    pub(crate) fn get_pixel(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }

        let word_idx = (y * self.stride + x / WORD_BITS) as usize;
        let bit_pos = WORD_SHIFT - (x % WORD_BITS);
        ((self.data[word_idx] >> bit_pos) & 1) as u8
    }

    /// Set a pixel value at (x, y).
    #[inline]
    pub(crate) fn set_pixel(&mut self, x: u32, y: u32, value: u8) {
        if x >= self.width || y >= self.height {
            return;
        }

        let word_idx = (y * self.stride + x / WORD_BITS) as usize;
        let bit_pos = WORD_SHIFT - (x % WORD_BITS);
        self.data[word_idx] |= (value as Word) << bit_pos;
    }

    /// Combine another bitmap into this one at a specific location.
    ///
    /// "These operators describe how the segment's bitmap is to be combined with
    /// the page bitmap." (7.4.1.5)
    ///
    /// Pixels outside the destination bitmap are ignored.
    pub(crate) fn combine(&mut self, other: &Self, x: i32, y: i32, operator: CombinationOperator) {
        // TODO: Figure out how we can optimize this, especially since this is a
        // bottleneck for symbols/patterns.
        let dest_x_start = x.max(0);
        let dest_x_end = x.saturating_add(other.width as i32).min(self.width as i32);

        if dest_x_start >= dest_x_end {
            return;
        }

        let src_x_start = (dest_x_start - x) as u32;
        let dest_x_start = dest_x_start as u32;
        let dest_x_end = dest_x_end as u32;

        let (src_y_start, src_y_end) = Self::clipped_rows(self.height, other.height, y);

        for src_y in src_y_start..src_y_end {
            let dest_y = (i64::from(y) + i64::from(src_y)) as u32;

            let first_word = dest_x_start / WORD_BITS;
            let last_word = (dest_x_end - 1) / WORD_BITS;

            for word_idx in first_word..=last_word {
                let word_start_x = word_idx * WORD_BITS;
                let word_end_x = word_start_x + WORD_BITS;

                let px_start = dest_x_start.max(word_start_x);
                let px_end = dest_x_end.min(word_end_x);

                let bit_start = px_start - word_start_x;
                let bit_end = px_end - word_start_x;

                let mask: Word = if bit_end == WORD_BITS {
                    !0 >> bit_start
                } else {
                    (!0 >> bit_start) & !(!0 >> bit_end)
                };

                let src_x_for_range = src_x_start + (px_start - dest_x_start);
                let src_word_idx = src_x_for_range / WORD_BITS;
                let src_bit_offset = src_x_for_range % WORD_BITS;

                let src_word1 = other.get_word(src_y, src_word_idx);
                let src_word2 = other.get_word(src_y, src_word_idx + 1);

                let src_raw = if src_bit_offset == 0 {
                    src_word1
                } else {
                    (src_word1 << src_bit_offset) | (src_word2 >> (WORD_BITS - src_bit_offset))
                };
                let src_aligned = src_raw >> bit_start;

                let dest_idx = (dest_y * self.stride + word_idx) as usize;
                let dest_word = self.data[dest_idx];

                let result = match operator {
                    CombinationOperator::Or => dest_word | src_aligned,
                    CombinationOperator::And => dest_word & src_aligned,
                    CombinationOperator::Xor => dest_word ^ src_aligned,
                    CombinationOperator::Xnor => !(dest_word ^ src_aligned),
                    CombinationOperator::Replace => src_aligned,
                };

                self.data[dest_idx] = (dest_word & !mask) | (result & mask);
            }
        }
    }

    /// The rows of a source `src_height` rows tall, placed at row `y`, that
    /// fall inside a destination `dest_height` rows tall.
    fn clipped_rows(dest_height: u32, src_height: u32, y: i32) -> (u32, u32) {
        let start = (-i64::from(y)).clamp(0, i64::from(src_height));
        let end = (i64::from(dest_height) - i64::from(y)).clamp(start, i64::from(src_height));

        (start as u32, end as u32)
    }

    /// The number of pixels of this bitmap that [`Bitmap::combine`] writes
    /// when `other` is placed at (`x`, `y`).
    pub(crate) fn clipped_area(&self, other: &Self, x: i32, y: i32) -> u64 {
        let (x_start, x_end) = Self::clipped_rows(self.width, other.width, x);
        let (y_start, y_end) = Self::clipped_rows(self.height, other.height, y);

        u64::from(x_end - x_start) * u64::from(y_end - y_start)
    }

    #[inline]
    pub(crate) fn get_word(&self, row: u32, word_idx: u32) -> Word {
        if row >= self.height || word_idx >= self.stride {
            return 0;
        }

        self.data[(row * self.stride + word_idx) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_far_away() {
        let budget = Budget::for_page(8, 8);
        let mut page = Bitmap::new(8, 8, &budget).unwrap();
        let region = Bitmap::new_with(8, 8, 0, 0, true, &budget).unwrap();
        page.combine(&region, i32::MAX - 1, 0, CombinationOperator::Or);
        page.combine(&region, 0, i32::MAX - 1, CombinationOperator::Or);
        assert!(page.data.iter().all(|&w| w == 0));
        assert_eq!(page.clipped_area(&region, i32::MAX - 1, 0), 0);
        assert_eq!(page.clipped_area(&region, 0, i32::MIN), 0);
    }

    #[test]
    fn clipped_area_counts_the_pixels_inside_the_destination() {
        let budget = Budget::for_page(10, 10);
        let page = Bitmap::new(10, 10, &budget).unwrap();
        let symbol = Bitmap::new(4, 6, &budget).unwrap();

        assert_eq!(page.clipped_area(&symbol, 0, 0), 24);
        assert_eq!(page.clipped_area(&symbol, -1, 7), 9);
        assert_eq!(page.clipped_area(&symbol, 8, -5), 2);
        assert_eq!(page.clipped_area(&symbol, 10, 0), 0);
    }

    #[test]
    fn bitmaps_are_charged_to_the_budget() {
        let budget = Budget::with_limits(64, 0);

        assert!(Bitmap::new(33, 1, &budget).is_ok());
        assert_eq!(
            Bitmap::new(1, 1, &budget).unwrap_err(),
            crate::DecodeError::Overflow(OverflowError::MemoryBudget)
        );
    }
}
