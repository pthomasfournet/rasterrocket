//! Tiling pattern source for the compositing pipeline.
//!
//! A tiling pattern paints its cell at every point of a lattice: cell
//! `(col, row)` lies at `origin + col · col_step + row · row_step` in device
//! space, the image of `(col · XStep, row · YStep)` under the pattern matrix
//! (ISO 32000-2 §8.7.3.1).  [`TiledPattern::compose`] composites a
//! rasterised cell at every lattice point whose cell meets a device
//! rectangle, and the result serves the pipeline as a [`Pattern`] over that
//! rectangle; [`TiledPattern::screen`] wraps a rectangle painted by other
//! means.  A lattice finer than a pixel is served as its mean colour and
//! coverage by [`TiledPattern::uniform`].

use std::ops::RangeInclusive;

use crate::bitmap::Bitmap;
use crate::pipe::{self, Pattern, PipeSrc, PipeState, RowPattern};
use crate::state::TransferSet;
use crate::types::BlendMode;
use color::Rgb8;

/// Upper bound on the lattice points enumerated for one pattern: cells
/// outside the rectangle cost an iteration each, and a strongly skewed
/// lattice enumerates many more points than it stamps.
const MAX_LATTICE_POINTS: u64 = 1 << 26;

/// Bound on a lattice coordinate: cells further from the origin than this
/// are not enumerated.
const LATTICE_LIMIT: f64 = (1u32 << 30) as f64;

/// The lattice of cell positions in device space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lattice {
    /// Device position of the top-left pixel of cell `(0, 0)`.
    pub origin: (f64, f64),
    /// Device displacement from a cell to the next along the pattern's x step.
    pub col_step: (f64, f64),
    /// Device displacement from a cell to the next along the pattern's y step.
    pub row_step: (f64, f64),
}

impl Lattice {
    /// Device position of the top-left pixel of cell `(col, row)`.
    #[must_use]
    pub fn position(&self, col: i32, row: i32) -> (f64, f64) {
        let (col, row) = (f64::from(col), f64::from(row));
        (
            col.mul_add(self.col_step.0, row.mul_add(self.row_step.0, self.origin.0)),
            col.mul_add(self.col_step.1, row.mul_add(self.row_step.1, self.origin.1)),
        )
    }

    /// The signed area of one lattice cell in device pixels.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.col_step
            .0
            .mul_add(self.row_step.1, -(self.col_step.1 * self.row_step.0))
    }

    /// The ranges of `col` and `row` whose cell, `cell_w × cell_h` pixels
    /// from its position, can meet the pixel rectangle `x0..x1 × y0..y1`.
    ///
    /// The ranges are the bounding box, in lattice coordinates, of the
    /// device rectangle grown by the cell size; a cell in them need not meet
    /// the rectangle.  `None` when the lattice is degenerate (a step that is
    /// zero, non-finite or parallel to the other) or the ranges are
    /// unbounded.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the bounds are clamped to ±2^30 before the casts"
    )]
    pub fn cells_meeting(
        &self,
        (x0, y0, x1, y1): (i32, i32, i32, i32),
        cell_w: u32,
        cell_h: u32,
    ) -> Option<(RangeInclusive<i32>, RangeInclusive<i32>)> {
        let det = self.determinant();
        if !det.is_finite() || det.abs() < 1e-9 {
            return None;
        }
        let (cs, rs) = (self.col_step, self.row_step);
        // A cell at position p covers p.x..p.x + cell_w, so it meets the
        // rectangle when p lies in the rectangle grown by the cell leftwards
        // and upwards.
        let left = f64::from(x0) - f64::from(cell_w);
        let top = f64::from(y0) - f64::from(cell_h);
        let (right, bottom) = (f64::from(x1), f64::from(y1));
        let mut cols = (f64::INFINITY, f64::NEG_INFINITY);
        let mut rows = (f64::INFINITY, f64::NEG_INFINITY);
        for (px, py) in [(left, top), (right, top), (left, bottom), (right, bottom)] {
            let (dx, dy) = (px - self.origin.0, py - self.origin.1);
            let col = dx.mul_add(rs.1, -(dy * rs.0)) / det;
            let row = cs.0.mul_add(dy, -(cs.1 * dx)) / det;
            cols = (cols.0.min(col), cols.1.max(col));
            rows = (rows.0.min(row), rows.1.max(row));
        }
        let bound = |v: f64| {
            v.is_finite()
                .then(|| v.clamp(-LATTICE_LIMIT, LATTICE_LIMIT))
        };
        let (c0, c1) = (bound(cols.0)?.floor() as i32, bound(cols.1)?.ceil() as i32);
        let (r0, r1) = (bound(rows.0)?.floor() as i32, bound(rows.1)?.ceil() as i32);
        Some((c0..=c1, r0..=r1))
    }
}

/// Source-over compositing at full opacity: the pipe that stamps a cell
/// onto the screen.
fn source_over() -> PipeState<'static> {
    PipeState {
        blend_mode: BlendMode::Normal,
        a_input: 255,
        overprint_mask: 0xFFFF_FFFF,
        overprint_additive: false,
        transfer: TransferSet::identity_rgb(),
        soft_mask: None,
        alpha0: None,
        knockout: false,
        knockout_opacity: 255,
        non_isolated_group: false,
    }
}

/// Composite `cell` onto `screen` with its top-left pixel at `(sx, sy)`,
/// restricted to the part of the cell inside the screen.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the overlap is clamped to the screen, whose dimensions are below 2^31"
)]
fn stamp(screen: &mut Bitmap<Rgb8>, cell: &Bitmap<Rgb8>, sx: i64, sy: i64, pipe: &PipeState<'_>) {
    let (sw, sh) = (i64::from(screen.width), i64::from(screen.height));
    let (cw, ch) = (i64::from(cell.width), i64::from(cell.height));
    let (x_a, x_b) = (sx.max(0), (sx + cw).min(sw));
    let (y_a, y_b) = (sy.max(0), (sy + ch).min(sh));
    if x_a >= x_b || y_a >= y_b {
        return;
    }
    let (cx_a, cx_b) = ((x_a - sx) as usize, (x_b - sx) as usize);
    for y in y_a..y_b {
        let cy = (y - sy) as u32;
        let shape = cell.alpha_row(cy).map(|a| &a[cx_a..cx_b]);
        if shape.is_some_and(|s| s.iter().all(|&a| a == 0)) {
            continue;
        }
        let src = RowPattern {
            data: &cell.row_bytes(cy)[cx_a * 3..cx_b * 3],
        };
        let (row, alpha) = screen.row_and_alpha_mut(y as u32);
        let dst = &mut row[x_a as usize * 3..x_b as usize * 3];
        let dst_alpha = alpha.map(|a| &mut a[x_a as usize..x_b as usize]);
        pipe::render_span::<Rgb8>(
            pipe,
            &PipeSrc::Pattern(&src),
            dst,
            dst_alpha,
            shape,
            x_a as i32,
            (x_b - 1) as i32,
            y as i32,
        );
    }
}

/// A tiling pattern resolved to device space.
///
/// The pattern is defined over one pixel rectangle: outside it, every pixel
/// is unpainted.  Callers paint within that rectangle.
pub struct TiledPattern {
    source: Source,
}

enum Source {
    /// The pattern composited over its rectangle, whose top-left pixel is
    /// `(x0, y0)`; `bitmap`'s alpha plane is the coverage of each pixel.
    Screen {
        bitmap: Bitmap<Rgb8>,
        x0: i32,
        y0: i32,
        /// Every pixel of the rectangle has coverage 255.
        fully_covered: bool,
    },
    /// The same colour and coverage at every pixel.
    Uniform { colour: [u8; 3], coverage: u8 },
}

impl TiledPattern {
    /// A pattern painting `colour` at `coverage` everywhere.
    #[must_use]
    pub const fn uniform(colour: [u8; 3], coverage: u8) -> Self {
        Self {
            source: Source::Uniform { colour, coverage },
        }
    }

    /// The pattern painted on `bitmap`, whose top-left pixel is device
    /// pixel `(x0, y0)` and whose alpha plane, when present, is the
    /// coverage of each pixel; a bitmap without one is fully covered.
    #[must_use]
    pub fn screen(bitmap: Bitmap<Rgb8>, x0: i32, y0: i32) -> Self {
        let fully_covered = bitmap
            .alpha_plane()
            .is_none_or(|a| a.iter().all(|&c| c == 255));
        Self {
            source: Source::Screen {
                bitmap,
                x0,
                y0,
                fully_covered,
            },
        }
    }

    /// Composite `cell` at every point of `lattice` whose cell meets the
    /// `width × height` pixel rectangle at `(x0, y0)`.  Each cell lands at
    /// its lattice position rounded to whole pixels.  `None` when the
    /// lattice is degenerate or enumerates more than [`MAX_LATTICE_POINTS`]
    /// points over the rectangle.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "cell positions lie within 2^30 of the origin (see Lattice::cells_meeting)"
    )]
    pub fn compose(
        cell: &Bitmap<Rgb8>,
        lattice: &Lattice,
        x0: i32,
        y0: i32,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let x1 = x0.checked_add(i32::try_from(width).ok()?)?;
        let y1 = y0.checked_add(i32::try_from(height).ok()?)?;
        let (cols, rows) = lattice.cells_meeting((x0, y0, x1, y1), cell.width, cell.height)?;
        let span = |r: &RangeInclusive<i32>| {
            u64::try_from(i64::from(*r.end()) - i64::from(*r.start()) + 1).ok()
        };
        if span(&cols)?.checked_mul(span(&rows)?)? > MAX_LATTICE_POINTS {
            return None;
        }
        let mut screen = Bitmap::new(width, height, 1, true);
        let pipe = source_over();
        for row in rows {
            for col in cols.clone() {
                let (px, py) = lattice.position(col, row);
                let sx = px.round() as i64 - i64::from(x0);
                let sy = py.round() as i64 - i64::from(y0);
                stamp(&mut screen, cell, sx, sy, &pipe);
            }
        }
        Some(Self::screen(screen, x0, y0))
    }
}

/// The screen row holding device row `y` and the screen columns of device
/// columns `x0..=x1` that fall within it, as `(row, local x0, offset, count)`:
/// `count` pixels from local column `local x0` serve output pixels from
/// `offset`.
fn screen_run(
    bitmap: &Bitmap<Rgb8>,
    sx0: i32,
    sy0: i32,
    y: i32,
    x0: i32,
    x1: i32,
) -> Option<(u32, usize, usize, usize)> {
    let row = u32::try_from(y.checked_sub(sy0)?).ok()?;
    if row >= bitmap.height {
        return None;
    }
    let (w, lx0, lx1) = (
        i64::from(bitmap.width),
        i64::from(x0) - i64::from(sx0),
        i64::from(x1) - i64::from(sx0),
    );
    let (a, b) = (lx0.max(0), lx1.min(w - 1));
    (a <= b).then(|| {
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "0 ≤ a ≤ b < the bitmap width, and a − lx0 ≤ the span length"
        )]
        let run = (row, a as usize, (a - lx0) as usize, (b - a + 1) as usize);
        run
    })
}

impl Pattern for TiledPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        match &self.source {
            Source::Uniform { colour, .. } => {
                for px in out.as_chunks_mut::<3>().0 {
                    *px = *colour;
                }
            }
            Source::Screen {
                bitmap,
                x0: sx0,
                y0: sy0,
                ..
            } => {
                out.fill(0);
                if let Some((row, lx, off, n)) = screen_run(bitmap, *sx0, *sy0, y, x0, x1) {
                    out[off * 3..(off + n) * 3]
                        .copy_from_slice(&bitmap.row_bytes(row)[lx * 3..(lx + n) * 3]);
                }
            }
        }
    }

    fn is_static_color(&self) -> bool {
        matches!(self.source, Source::Uniform { .. })
    }

    fn may_leave_unpainted(&self) -> bool {
        match &self.source {
            Source::Uniform { coverage, .. } => *coverage != 255,
            Source::Screen { fully_covered, .. } => !fully_covered,
        }
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], coverage: &mut [u8]) {
        self.fill_span(y, x0, x1, out);
        match &self.source {
            Source::Uniform { coverage: c, .. } => coverage.fill(*c),
            Source::Screen {
                bitmap,
                x0: sx0,
                y0: sy0,
                ..
            } => {
                coverage.fill(0);
                if let Some((row, lx, off, n)) = screen_run(bitmap, *sx0, *sy0, y, x0, x1) {
                    match bitmap.alpha_row(row) {
                        Some(alpha) => coverage[off..off + n].copy_from_slice(&alpha[lx..lx + n]),
                        None => coverage[off..off + n].fill(255),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 × 2 cell: red, opaque, except a transparent bottom-right pixel.
    fn cell() -> Bitmap<Rgb8> {
        let mut c = Bitmap::<Rgb8>::new(2, 2, 1, true);
        for y in 0..2 {
            c.row_bytes_mut(y).copy_from_slice(&[255, 0, 0, 255, 0, 0]);
        }
        c.alpha_plane_mut()
            .unwrap()
            .copy_from_slice(&[255, 255, 255, 0]);
        c
    }

    fn axis_aligned(step: f64) -> Lattice {
        Lattice {
            origin: (0.0, 0.0),
            col_step: (step, 0.0),
            row_step: (0.0, step),
        }
    }

    #[test]
    fn cells_meeting_covers_a_rotated_lattice() {
        // 45° lattice with a step of 10 px: positions (0, 0), (7.07, 7.07), …
        let l = Lattice {
            origin: (0.0, 0.0),
            col_step: (7.0710678, 7.0710678),
            row_step: (-7.0710678, 7.0710678),
        };
        let (cols, rows) = l.cells_meeting((0, 0, 20, 20), 2, 2).unwrap();
        assert!(cols.contains(&0) && cols.contains(&2), "{cols:?}");
        assert!(rows.contains(&-2) && rows.contains(&2), "{rows:?}");
        assert!(cols.end() - cols.start() < 10 && rows.end() - rows.start() < 10);
    }

    #[test]
    fn a_degenerate_lattice_has_no_cells() {
        let l = Lattice {
            origin: (0.0, 0.0),
            col_step: (1.0, 2.0),
            row_step: (2.0, 4.0),
        };
        assert!(l.cells_meeting((0, 0, 4, 4), 1, 1).is_none());
        assert!(TiledPattern::compose(&cell(), &axis_aligned(f64::NAN), 0, 0, 4, 4).is_none());
    }

    #[test]
    fn compose_stamps_cells_on_the_lattice_and_repeats_their_coverage() {
        let p = TiledPattern::compose(&cell(), &axis_aligned(2.0), 0, 0, 4, 2).unwrap();
        assert!(p.may_leave_unpainted());
        let mut out = [9u8; 4 * 3];
        let mut cov = [9u8; 4];
        p.fill_span_coverage(1, 0, 3, &mut out, &mut cov);
        assert_eq!(cov, [255, 0, 255, 0]);
        assert_eq!(&out[..3], &[255, 0, 0]);
    }

    #[test]
    fn compose_rounds_each_cell_to_its_own_position() {
        // A 2.5 px step: cells at 0, 2.5 → 3 (round half away from zero), 5.
        let p = TiledPattern::compose(&cell(), &axis_aligned(2.5), 0, 0, 7, 1).unwrap();
        let mut out = [0u8; 7 * 3];
        let mut cov = [0u8; 7];
        p.fill_span_coverage(0, 0, 6, &mut out, &mut cov);
        assert_eq!(cov, [255, 255, 0, 255, 255, 255, 255]);
    }

    #[test]
    fn overlapping_cells_composite_source_over() {
        // A 1 px step: the transparent pixel of one cell is covered by the
        // next cell's opaque pixel.
        let p = TiledPattern::compose(&cell(), &axis_aligned(1.0), 0, 0, 3, 2).unwrap();
        assert!(!p.may_leave_unpainted());
        let mut out = [0u8; 3 * 3];
        p.fill_span(1, 0, 2, &mut out);
        assert_eq!(out, [255, 0, 0, 255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn pixels_outside_the_rectangle_are_unpainted() {
        let p = TiledPattern::compose(&cell(), &axis_aligned(2.0), 10, 10, 2, 2).unwrap();
        let mut out = [7u8; 4 * 3];
        let mut cov = [7u8; 4];
        p.fill_span_coverage(10, 8, 11, &mut out, &mut cov);
        assert_eq!(cov, [0, 0, 255, 255]);
        p.fill_span_coverage(12, 10, 13, &mut out, &mut cov);
        assert_eq!(cov, [0, 0, 0, 0]);
    }

    #[test]
    fn uniform_serves_one_colour_and_coverage() {
        let p = TiledPattern::uniform([1, 2, 3], 100);
        assert!(p.is_static_color() && p.may_leave_unpainted());
        let mut out = [0u8; 2 * 3];
        let mut cov = [0u8; 2];
        p.fill_span_coverage(-5, 7, 8, &mut out, &mut cov);
        assert_eq!(out, [1, 2, 3, 1, 2, 3]);
        assert_eq!(cov, [100, 100]);
        assert!(!TiledPattern::uniform([0; 3], 255).may_leave_unpainted());
    }
}
