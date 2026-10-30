//! Tiling pattern source for the compositing pipeline.
//!
//! [`TiledPattern`] holds a pre-rasterised tile bitmap and implements
//! [`crate::Pattern`] by repeating the tile across device space using modular
//! arithmetic.  One tile is computed once and then referenced immutably during
//! the fill/stroke operation.

use crate::pipe::Pattern;

/// A pre-rasterised tiling pattern.
///
/// The tile is a row-major RGB8 pixel buffer of `width × height` pixels,
/// anchored at `(phase_x, phase_y)` in device space.  `fill_span` tiles the
/// buffer by wrapping coordinates with `rem_euclid`.  An optional coverage
/// plane, one byte per tile pixel, gives the fraction of each pixel the
/// tile's content painted; pixels the content left unpainted leave the
/// destination unchanged.
///
/// # Invariants
///
/// - `width > 0` and `height > 0` (asserted in [`TiledPattern::new`])
/// - `pixels.len() == width * height * 3`
/// - `coverage`, when present, has `width * height` bytes, not all 255
pub struct TiledPattern {
    /// Rasterised tile pixels (RGB8, row-major).
    pixels: Vec<u8>,
    /// Coverage of each tile pixel; `None` when every pixel is fully covered.
    coverage: Option<Vec<u8>>,
    /// Tile width in device pixels.
    width: i32,
    /// Tile height in device pixels.
    height: i32,
    /// X offset of the pattern origin in device space.
    ///
    /// Used to phase the tiling so that the pattern is anchored at the correct
    /// position as specified by the pattern matrix translation.
    phase_x: i32,
    /// Y offset of the pattern origin in device space.
    phase_y: i32,
}

impl TiledPattern {
    /// Construct a new [`TiledPattern`] whose every pixel is fully covered.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is ≤ 0, or if `pixels.len()` does not
    /// equal `width * height * 3`.
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "width/height are asserted positive just above the cast"
    )]
    pub fn new(pixels: Vec<u8>, width: i32, height: i32, phase_x: i32, phase_y: i32) -> Self {
        assert!(
            width > 0,
            "TiledPattern: width must be positive, got {width}"
        );
        assert!(
            height > 0,
            "TiledPattern: height must be positive, got {height}"
        );
        let expected = width as usize * height as usize * 3;
        assert!(
            pixels.len() == expected,
            "TiledPattern: pixels.len() {} does not match width({width}) * height({height}) * 3 = {expected}",
            pixels.len()
        );
        Self {
            pixels,
            coverage: None,
            width,
            height,
            phase_x,
            phase_y,
        }
    }

    /// Attach a coverage plane, one byte per tile pixel in row-major order.
    ///
    /// A plane that covers every pixel fully is dropped, so the pattern then
    /// takes the fully covered compositing path.
    ///
    /// # Panics
    ///
    /// Panics if `coverage.len()` does not equal `width * height`.
    #[must_use]
    pub fn with_coverage(mut self, coverage: Vec<u8>) -> Self {
        assert_eq!(
            coverage.len() * 3,
            self.pixels.len(),
            "TiledPattern: coverage plane must hold one byte per tile pixel"
        );
        self.coverage = coverage.iter().any(|&c| c != 255).then_some(coverage);
        self
    }

    /// Pixel index of the start of the tile row holding device row `y`.
    #[expect(
        clippy::cast_sign_loss,
        reason = "the width is positive (asserted in new)"
    )]
    const fn tile_row(&self, y: i32) -> usize {
        (y - self.phase_y).rem_euclid(self.height) as usize * self.width as usize
    }

    /// Tile column holding device column `x`.
    const fn tile_col(&self, x: i32) -> usize {
        (x - self.phase_x).rem_euclid(self.width) as usize
    }
}

impl Pattern for TiledPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        let row = self.tile_row(y);
        for (x, px) in (x0..=x1).zip(out.as_chunks_mut::<3>().0) {
            let i = (row + self.tile_col(x)) * 3;
            px.copy_from_slice(&self.pixels[i..i + 3]);
        }
    }

    fn may_leave_unpainted(&self) -> bool {
        self.coverage.is_some()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], coverage: &mut [u8]) {
        self.fill_span(y, x0, x1, out);
        let Some(plane) = &self.coverage else {
            coverage.fill(255);
            return;
        };
        let row = self.tile_row(y);
        for (x, c) in (x0..=x1).zip(coverage) {
            *c = plane[row + self.tile_col(x)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×1 tile: red at column 0, green at column 1, phased by one pixel.
    fn two_by_one() -> TiledPattern {
        TiledPattern::new(vec![255, 0, 0, 0, 255, 0], 2, 1, 1, 0)
    }

    #[test]
    fn tiles_wrap_around_the_phase() {
        let mut out = [0u8; 4 * 3];
        two_by_one().fill_span(5, -1, 2, &mut out);
        assert_eq!(out, [255, 0, 0, 0, 255, 0, 255, 0, 0, 0, 255, 0]);
    }

    #[test]
    fn full_coverage_is_dropped() {
        let p = two_by_one().with_coverage(vec![255, 255]);
        assert!(!p.may_leave_unpainted());
    }

    #[test]
    fn coverage_repeats_with_the_tile() {
        let p = two_by_one().with_coverage(vec![0, 200]);
        assert!(p.may_leave_unpainted());
        let mut out = [0u8; 3 * 3];
        let mut coverage = [9u8; 3];
        p.fill_span_coverage(0, 0, 2, &mut out, &mut coverage);
        assert_eq!(coverage, [200, 0, 200]);
        assert_eq!(&out[..3], &[0, 255, 0]);
    }
}
