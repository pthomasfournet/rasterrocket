//! Separable fixed-point sampling grid for image blits.
//!
//! An image `Do` maps the unit square through the CTM onto the page.  Every
//! device pixel whose centre the image covers takes its colour from the
//! source samples around that centre's position in the image: the sample
//! whose footprint contains it (nearest neighbour), or, along an axis the
//! grid interpolates, the two samples whose centres bracket it, weighted
//! linearly by distance (bilinear).  The inverse affine map is separable, so
//! the source coordinate of device pixel `(dx, dy)` is `col[dx] + row[dy]`:
//! one table entry per bounding-box column plus one per row, summed per
//! pixel in integer arithmetic.
//!
//! The CPU sampler and the GPU blit kernel consume the same tables and
//! derive taps and weights from them by the same integer arithmetic, so
//! both paths produce identical pixels by construction.  The only
//! floating-point work is the table build, which runs once per image on the
//! host.
//!
//! # Fixed-point layout
//!
//! Source coordinates are Q32 (32 integer bits, 32 fraction bits) split into
//! an `i32` integer part and a `u32` fraction, so the per-pixel sum needs only
//! 32-bit integer operations on either side.  Each table entry is
//! [`ENTRY_WORDS`] `u32` words: `[x_hi, x_lo, y_hi, y_lo]`, with the `hi`
//! words carrying the two's-complement bit pattern of the `i32` integer part.
//! Adding a column entry to a row entry with carry from the fraction word
//! yields `floor(x_col + x_row)` exactly, where each term was itself
//! truncated to Q32 at build time.
//!
//! An interpolated axis measures the coordinate from sample centres, half a
//! sample below the sum, by subtracting `2^31` from the fraction with borrow
//! into the integer part; the top [`WEIGHT_BITS`] bits of the resulting
//! fraction weight the upper of the two samples.
//!
//! Integer parts are clamped to ±[`COORD_LIMIT`] at build time, so a
//! column + row sum can never overflow `i32`; a clamped entry still lies far
//! outside any image (dimensions are capped at 65 536 px) and is rejected by
//! the range check.

use raster::types::AA_SIZE;
use raster::{AaBuf, Clip, ClipResult, simd};

use super::image_reduce::{AXIS_TOLERANCE, REDUCTION_MARGIN};
use crate::renderer::gstate::{Ctm, ctm_transform};
use crate::resources::SoftMask;

/// `u32` words per table entry: `[x_hi, x_lo, y_hi, y_lo]`.
pub(super) const ENTRY_WORDS: usize = 4;

/// Magnitude cap on a table entry's integer part: `2^30 - 1`.  Two entries
/// plus a fraction carry sum to at most `2^31 - 1` — no `i32` overflow on
/// any consumer, including the kernels' plain `int` arithmetic — while a
/// clamped entry remains rejectable.
const COORD_LIMIT: f64 = 1_073_741_823.0;

/// Q32 fraction scale: `2^32`.
const FRACTION_SCALE: f64 = 4_294_967_296.0;

/// Fraction bits of a bilinear tap weight; the two weights along an axis
/// sum to `1 << WEIGHT_BITS`.
const WEIGHT_BITS: u32 = 8;

/// Half a sample in the Q32 fraction word.
const HALF_SAMPLE: u32 = 1 << 31;

/// Magnification (device pixels per sample) at or above which an image that
/// does not request interpolation keeps its samples as uniform blocks.
const MAGNIFICATION_LIMIT: f64 = 4.0;

/// Which image axes a [`SampleGrid`] interpolates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Filter {
    /// Interpolate between source columns.
    pub x: bool,
    /// Interpolate between source rows.
    pub y: bool,
}

impl Filter {
    /// Nearest-neighbour sampling on both axes.
    pub(super) const NEAREST: Self = Self { x: false, y: false };

    /// The filter for a `width × height` image drawn through `ctm`, whose
    /// dictionary sets `/Interpolate` to `interpolate`.
    ///
    /// An image drawn at or above [`MAGNIFICATION_LIMIT`] on either axis is
    /// interpolated only when it requests it.  Otherwise a rotated or
    /// skewed draw is interpolated on both axes, and a draw whose image axes
    /// map onto the device axes on each axis it magnifies: an axis drawn at
    /// one sample per device pixel or fewer (an image reduced for the draw
    /// is resampled to one per pixel first) keeps its samples.
    pub(super) fn for_draw(ctm: &Ctm, width: u32, height: u32, interpolate: bool) -> Self {
        let [a, b, c, d, _, _] = *ctm;
        let mag_x = a.hypot(b) / f64::from(width);
        let mag_y = c.hypot(d) / f64::from(height);
        if !(mag_x.is_finite() && mag_y.is_finite()) {
            return Self::NEAREST;
        }
        if !interpolate && mag_x.max(mag_y) >= MAGNIFICATION_LIMIT {
            return Self::NEAREST;
        }
        if !is_axis_aligned(ctm) {
            return Self { x: true, y: true };
        }
        Self {
            x: magnifies(mag_x),
            y: magnifies(mag_y),
        }
    }

    /// The blit kernel's encoding: [`gpu::blit::BLIT_FILTER_X`] and
    /// [`gpu::blit::BLIT_FILTER_Y`] bits.
    #[cfg(feature = "cache")]
    pub(super) const fn bits(self) -> u32 {
        use gpu::blit::{BLIT_FILTER_X, BLIT_FILTER_Y};
        (if self.x { BLIT_FILTER_X } else { 0 }) | (if self.y { BLIT_FILTER_Y } else { 0 })
    }
}

/// `true` when `ctm` maps the image axes onto the device axes, in either
/// order, to within [`AXIS_TOLERANCE`].
pub(super) fn is_axis_aligned(ctm: &Ctm) -> bool {
    let [a, b, c, d, _, _] = *ctm;
    (b.abs() <= AXIS_TOLERANCE && c.abs() <= AXIS_TOLERANCE)
        || (a.abs() <= AXIS_TOLERANCE && d.abs() <= AXIS_TOLERANCE)
}

/// `true` when an axis drawn at `mag` device pixels per sample magnifies
/// beyond the reduction margin.
fn magnifies(mag: f64) -> bool {
    mag > 1.0 + REDUCTION_MARGIN
}

/// The source samples one device pixel reads and their weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Taps {
    /// Left and right source columns; equal when the column is not blended.
    x: [u32; 2],
    /// Upper and lower source rows; equal when the row is not blended.
    y: [u32; 2],
    /// Weight of `x[1]`, out of `1 << WEIGHT_BITS`.
    wx: u32,
    /// Weight of `y[1]`, out of `1 << WEIGHT_BITS`.
    wy: u32,
}

impl Taps {
    /// The single sample `(x, y)`.
    #[cfg(test)]
    const fn point(x: u32, y: u32) -> Self {
        Self {
            x: [x, x],
            y: [y, y],
            wx: 0,
            wy: 0,
        }
    }

    /// The weighted mean of `fetch(x, y)` over the taps, rounded to nearest.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a weighted mean of u8 values with weights summing to 2^16 is at most 255 after the shift"
    )]
    pub(super) fn blend(&self, fetch: impl Fn(u32, u32) -> u8) -> u8 {
        if self.wx == 0 && self.wy == 0 {
            return fetch(self.x[0], self.y[0]);
        }
        let one = 1 << WEIGHT_BITS;
        let row = |y| {
            u32::from(fetch(self.x[0], y)) * (one - self.wx)
                + u32::from(fetch(self.x[1], y)) * self.wx
        };
        let sum = row(self.y[0]) * (one - self.wy) + row(self.y[1]) * self.wy;
        ((sum + (1 << (2 * WEIGHT_BITS - 1))) >> (2 * WEIGHT_BITS)) as u8
    }
}

/// Taps along one axis of `n` samples at integer part `i` and fraction
/// `frac` of a pixel centre's source coordinate, with `i` clamped onto the
/// axis.  When `blend` is set and `i` lies on the axis, the taps are the
/// samples whose centres bracket the coordinate, clamped to the axis, and
/// the weight is the coordinate's position between them; otherwise the
/// single sample `i`.
#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "each index is clamped to be non-negative before the u32 cast"
)]
fn axis_taps(i: i32, frac: u32, n: u32, blend: bool) -> ([u32; 2], u32) {
    let last = n - 1;
    if i < 0 {
        return ([0, 0], 0);
    }
    let i_u = i as u32;
    if i_u > last {
        return ([last, last], 0);
    }
    if !blend {
        return ([i_u, i_u], 0);
    }
    // Measured from sample centres: `i + frac − ½`, floored, with its
    // fraction's top bits as the weight of the upper sample.
    let lower = i - i32::from(frac < HALF_SAMPLE);
    let weight = frac.wrapping_sub(HALF_SAMPLE) >> (32 - WEIGHT_BITS);
    let lo = lower.max(0) as u32;
    let hi = ((lower + 1) as u32).min(last);
    // Past the outer sample centres both taps clamp to the edge sample.
    ([lo, hi], if lo == hi { 0 } else { weight })
}

/// Per-image sampling grid: the device-space bounding box plus the column
/// and row tables that map every pixel inside it to a source coordinate.
pub(super) struct SampleGrid {
    /// Inclusive left edge of the bounding box, page pixels.
    pub x0: u32,
    /// Inclusive top edge.
    pub y0: u32,
    /// Exclusive right edge.
    pub x1: u32,
    /// Exclusive bottom edge.
    pub y1: u32,
    /// Source image width in pixels.
    img_w: u32,
    /// Source image height in pixels.
    img_h: u32,
    /// Axes interpolated between samples.
    filter: Filter,
    /// `x1 - x0` entries of [`ENTRY_WORDS`] words each.
    cols: Vec<u32>,
    /// `y1 - y0` entries of [`ENTRY_WORDS`] words each.
    rows: Vec<u32>,
}

impl SampleGrid {
    /// Build the grid for an image of `img_w × img_h` pixels drawn through
    /// `ctm` onto a `page_w × page_h` bitmap whose y axis points down,
    /// interpolating the axes `filter` selects.
    ///
    /// Returns `None` when nothing can be drawn: a non-finite or singular
    /// CTM, or a bounding box that misses the page entirely.
    #[expect(
        clippy::many_single_char_names,
        clippy::similar_names,
        reason = "PDF CTM components a-f are spec terminology; kx_*/ky_* are the paired per-axis coefficients they produce"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bbox edges are clamped to [0, page dim] in i64 before the u32 cast"
    )]
    pub(super) fn new(
        ctm: &Ctm,
        page_w: u32,
        page_h: u32,
        img_w: u32,
        img_h: u32,
        filter: Filter,
    ) -> Option<Self> {
        if img_w == 0 || img_h == 0 {
            return None;
        }
        let page_h_f = f64::from(page_h);

        // Bounding box of the unit square's image, y-flipped into bitmap
        // space and clamped to the page.  `floor`/`ceil` of the corner
        // extremes covers every pixel whose area intersects the image, so
        // the centre-sampling test inside the loop never misses a pixel.
        let corners = [
            ctm_transform(ctm, 0.0, 0.0),
            ctm_transform(ctm, 1.0, 0.0),
            ctm_transform(ctm, 0.0, 1.0),
            ctm_transform(ctm, 1.0, 1.0),
        ];
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return None;
        }
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &corners {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            let y_dev = page_h_f - y;
            min_y = min_y.min(y_dev);
            max_y = max_y.max(y_dev);
        }
        let clamp = |v: f64, dim: u32| (v as i64).clamp(0, i64::from(dim)) as u32;
        let x0 = clamp(min_x.floor(), page_w);
        let x1 = clamp(max_x.ceil(), page_w);
        let y0 = clamp(min_y.floor(), page_h);
        let y1 = clamp(max_y.ceil(), page_h);
        if x0 >= x1 || y0 >= y1 {
            return None;
        }

        // Inverse map.  The CTM sends image space (u, v) to PDF user space:
        //   x     = a*u + c*v + e
        //   y_pdf = b*u + d*v + f
        // so with dx_rel = x - e and dy_rel = y_pdf - f:
        //   u = ( d*dx_rel - c*dy_rel) / det
        //   v = (-b*dx_rel + a*dy_rel) / det
        // The source coordinate is X = u * img_w and Y = (1 - v) * img_h
        // (image rows run top-down, PDF v runs bottom-up).  Both are affine
        // in (dx_rel, dy_rel), so each splits into a column term and a row
        // term; the `img_h` offset of Y lives in the row term.
        let [a, b, c, d, e, f] = *ctm;
        let det = a.mul_add(d, -(b * c));
        // `det` is finite: every corner was, so every CTM component is.
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        let w = f64::from(img_w);
        let h = f64::from(img_h);
        let kx_col = d * inv_det * w;
        let ky_col = b * inv_det * h;
        let kx_row = -c * inv_det * w;
        let ky_row = -a * inv_det * h;

        let mut cols = Vec::with_capacity((x1 - x0) as usize * ENTRY_WORDS);
        for dx in x0..x1 {
            let dx_rel = (f64::from(dx) + 0.5) - e;
            push_entry(&mut cols, kx_col * dx_rel, ky_col * dx_rel);
        }
        let mut rows = Vec::with_capacity((y1 - y0) as usize * ENTRY_WORDS);
        for dy in y0..y1 {
            let dy_rel = (page_h_f - (f64::from(dy) + 0.5)) - f;
            push_entry(&mut rows, kx_row * dy_rel, ky_row.mul_add(dy_rel, h));
        }

        Some(Self {
            x0,
            y0,
            x1,
            y1,
            img_w,
            img_h,
            filter,
            cols,
            rows,
        })
    }

    /// Axes interpolated between samples.
    #[cfg(feature = "cache")]
    pub(super) const fn filter(&self) -> Filter {
        self.filter
    }

    /// Column table, `(x1 - x0) * ENTRY_WORDS` words.
    #[cfg(any(feature = "cache", test))]
    pub(super) fn cols(&self) -> &[u32] {
        &self.cols
    }

    /// Row table, `(y1 - y0) * ENTRY_WORDS` words.
    #[cfg(any(feature = "cache", test))]
    pub(super) fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// Source taps for device pixel `(dx, dy)`, or `None` when the pixel
    /// centre falls outside the image.
    ///
    /// `dx`/`dy` must lie inside the bounding box.
    #[inline]
    #[expect(
        clippy::cast_sign_loss,
        reason = "the range check proves both coordinates non-negative before the u32 cast"
    )]
    pub(super) fn sample(&self, dx: u32, dy: u32) -> Option<Taps> {
        let (x, y) = self.coord(dx, dy);
        if x.0 < 0 || y.0 < 0 || x.0 as u32 >= self.img_w || y.0 as u32 >= self.img_h {
            return None;
        }
        Some(self.taps(x, y))
    }

    /// Source taps for device pixel `(dx, dy)`, clamped onto the image.
    ///
    /// A device pixel outside the bounding box is first clamped onto it.
    #[inline]
    pub(super) fn sample_clamped(&self, dx: u32, dy: u32) -> Taps {
        let (x, y) = self.coord(
            dx.clamp(self.x0, self.x1 - 1),
            dy.clamp(self.y0, self.y1 - 1),
        );
        self.taps(x, y)
    }

    /// Taps at source coordinate `(x, y)`, each an integer part and a
    /// fraction, clamped onto the image.
    #[inline]
    fn taps(&self, x: (i32, u32), y: (i32, u32)) -> Taps {
        let (xs, wx) = axis_taps(x.0, x.1, self.img_w, self.filter.x);
        let (ys, wy) = axis_taps(y.0, y.1, self.img_h, self.filter.y);
        Taps {
            x: xs,
            y: ys,
            wx,
            wy,
        }
    }

    /// Unbounded source coordinate of device pixel `(dx, dy)`, as an
    /// integer part and a Q32 fraction per axis: the column entry plus the
    /// row entry, with the fraction carry.
    #[inline]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "hi words carry an i32 bit pattern by construction"
    )]
    fn coord(&self, dx: u32, dy: u32) -> ((i32, u32), (i32, u32)) {
        let c = (dx - self.x0) as usize * ENTRY_WORDS;
        let r = (dy - self.y0) as usize * ENTRY_WORDS;
        let col = &self.cols[c..c + ENTRY_WORDS];
        let row = &self.rows[r..r + ENTRY_WORDS];
        let (frac_x, carry_x) = col[1].overflowing_add(row[1]);
        let x = (col[0] as i32)
            .wrapping_add(row[0] as i32)
            .wrapping_add(i32::from(carry_x));
        let (frac_y, carry_y) = col[3].overflowing_add(row[3]);
        let y = (col[2] as i32)
            .wrapping_add(row[2] as i32)
            .wrapping_add(i32::from(carry_y));
        ((x, frac_x), (y, frac_y))
    }
}

/// Alpha lookup for a masked image blit.
///
/// A mask on the image's own grid — the same dimensions drawn through the
/// same matrix — is read through the image's own taps.  Any other mask is
/// sampled through a [`SampleGrid`] of its own dimensions and matrix on the
/// same page, with the filter [`Filter::for_draw`] selects for it: each
/// device pixel reads the mask around its centre, so the mask is resolved
/// at device resolution whatever resolution the image was decoded or
/// resampled at.  Where the image admits a pixel centre that rounding
/// places just off the mask, the nearest mask pixel is used.
pub(super) struct MaskSampler<'a> {
    alpha: &'a [u8],
    width: usize,
    /// `None` when the mask shares the image's grid.
    grid: Option<SampleGrid>,
}

impl<'a> MaskSampler<'a> {
    /// Sampler for `mask` drawn through `mask_ctm`, over an image of
    /// `img_w × img_h` pixels drawn through `img_ctm` onto a
    /// `page_w × page_h` bitmap; `interpolate` is the image's
    /// `/Interpolate`.
    ///
    /// Returns `None` only where [`SampleGrid::new`] does for the mask: a
    /// degenerate matrix or a bounding box off the page.
    #[expect(
        clippy::float_cmp,
        reason = "the mask shares the image grid only when both are drawn through the identical matrix"
    )]
    #[expect(
        clippy::too_many_arguments,
        reason = "the mask and image placements are each a matrix plus dimensions"
    )]
    pub(super) fn new(
        mask: &'a SoftMask,
        mask_ctm: &Ctm,
        img_ctm: &Ctm,
        page_w: u32,
        page_h: u32,
        img_w: u32,
        img_h: u32,
        interpolate: bool,
    ) -> Option<Self> {
        let grid = if (mask.width(), mask.height()) == (img_w, img_h) && mask_ctm == img_ctm {
            None
        } else {
            let (w, h) = (mask.width(), mask.height());
            let filter = Filter::for_draw(mask_ctm, w, h, interpolate);
            Some(SampleGrid::new(mask_ctm, page_w, page_h, w, h, filter)?)
        };
        Some(Self {
            alpha: mask.alpha(),
            width: mask.width() as usize,
            grid,
        })
    }

    /// Alpha of device pixel `(dx, dy)`, whose image taps are `image`.
    ///
    /// `dx`/`dy` must lie inside the bounding box.
    #[inline]
    pub(super) fn alpha(&self, dx: u32, dy: u32, image: &Taps) -> u8 {
        let taps = self
            .grid
            .as_ref()
            .map_or(*image, |g| g.sample_clamped(dx, dy));
        taps.blend(|x, y| {
            self.alpha
                .get(y as usize * self.width + x as usize)
                .copied()
                .unwrap_or(u8::MAX)
        })
    }
}

/// Admission of a [`SampleGrid`]'s pixels by the graphics-state clip.
///
/// The grid's bounding box is classified once against the clip; rows and
/// then pixels are tested only as finely as that classification requires,
/// so an unclipped image costs one rectangle test and a clipped one costs a
/// span test per row plus a coverage computation on rows the clip edge
/// crosses.
pub(super) struct GridClip<'a> {
    clip: &'a Clip,
    bbox: ClipResult,
    /// Inclusive column span of the grid, device pixels.
    x0: i32,
    x1: i32,
    /// Sub-sample scratch for antialiased clips, allocated on the first row
    /// the clip edge crosses.
    aa_buf: Option<AaBuf>,
    /// Per-column coverage of the current partial row, indexed from `x0`.
    coverage: Vec<u8>,
}

impl<'a> GridClip<'a> {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "grid edges are clamped to the page, which fits i32"
    )]
    pub(super) fn new(clip: &'a Clip, grid: &SampleGrid) -> Self {
        let (x0, y0) = (grid.x0 as i32, grid.y0 as i32);
        let (x1, y1) = (grid.x1 as i32 - 1, grid.y1 as i32 - 1);
        Self {
            clip,
            bbox: clip.test_rect(x0, y0, x1, y1),
            x0,
            x1,
            aa_buf: None,
            coverage: Vec::new(),
        }
    }

    /// Classification of the whole grid bounding box.
    pub(super) const fn bbox(&self) -> ClipResult {
        self.bbox
    }

    /// Coverage for row `dy` of the grid, or `None` when the clip excludes
    /// the entire row.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "dy < grid.y1 ≤ page height, which fits i32"
    )]
    pub(super) fn row(&mut self, dy: u32) -> Option<RowClip<'_>> {
        let y = dy as i32;
        let res = if self.bbox == ClipResult::AllInside {
            ClipResult::AllInside
        } else {
            self.clip.test_span(self.x0, self.x1, y)
        };
        match res {
            ClipResult::AllOutside => None,
            ClipResult::AllInside => Some(RowClip::Full),
            ClipResult::Partial => {
                self.partial_coverage(y);
                Some(RowClip::Partial {
                    x0: self.x0,
                    coverage: &self.coverage,
                })
            }
        }
    }

    /// Fill `self.coverage` for a row the clip edge crosses.
    ///
    /// With an antialiased clip, every sub-sample of the grid's columns is
    /// set, intersected with the clip by [`Clip::clip_aa_line`], counted per
    /// pixel and mapped through the AA gamma table, exactly as a fill's
    /// coverage is.  A rectangle-only clip has integer bounds, so its columns
    /// come out fully covered or uncovered.  A non-antialiased clip uses the
    /// binary pixel test.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x0 ≥ 0 (grid clamped to the page) and the clipped span lies within [x0, x1]"
    )]
    fn partial_coverage(&mut self, y: i32) {
        let width = (self.x1 - self.x0 + 1) as usize;
        self.coverage.clear();
        self.coverage.resize(width, 0);
        if !self.clip.antialias {
            for (dx, cov) in (self.x0..=self.x1).zip(self.coverage.iter_mut()) {
                if self.clip.test(dx, y) {
                    *cov = 255;
                }
            }
            return;
        }
        let aa_size = AA_SIZE as usize;
        let buf = self
            .aa_buf
            .get_or_insert_with(|| AaBuf::new(self.x1 as usize + 1));
        buf.clear();
        for sub_row in 0..buf.height {
            buf.set_span(
                sub_row,
                self.x0 as usize * aa_size,
                (self.x1 as usize + 1) * aa_size,
            );
        }
        let (mut cx0, mut cx1) = (self.x0, self.x1);
        self.clip.clip_aa_line(buf, &mut cx0, &mut cx1, y);
        let (cx0, cx1) = (cx0.max(self.x0), cx1.min(self.x1));
        if cx0 > cx1 {
            return;
        }
        let rows = [
            buf.row_slice(0),
            buf.row_slice(1),
            buf.row_slice(2),
            buf.row_slice(3),
        ];
        let span = &mut self.coverage[(cx0 - self.x0) as usize..=(cx1 - self.x0) as usize];
        simd::aa_coverage_span(rows, cx0 as usize, span);
        // An uncovered span maps to zeros, which callers skip; the
        // any-covered flag adds nothing here.
        let _any_covered = simd::apply_aa_gamma(span);
    }
}

/// Coverage of the pixels on one grid row.
pub(super) enum RowClip<'a> {
    /// The whole row lies inside the clip.
    Full,
    /// The clip edge crosses the row; `coverage[i]` is the shape (0–255) of
    /// device column `x0 + i`.
    Partial { x0: i32, coverage: &'a [u8] },
}

impl RowClip<'_> {
    /// Clip coverage (0 = outside, 255 = inside) of device column `dx`.
    #[inline]
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "dx < grid.x1 ≤ page width, which fits i32; the index is checked by get"
    )]
    pub(super) fn coverage(&self, dx: u32) -> u8 {
        match self {
            Self::Full => 255,
            Self::Partial { x0, coverage } => coverage
                .get((dx as i32 - x0) as usize)
                .copied()
                .unwrap_or(0),
        }
    }
}

/// Append one `[x_hi, x_lo, y_hi, y_lo]` entry.
fn push_entry(table: &mut Vec<u32>, x: f64, y: f64) {
    let (x_hi, x_lo) = split_q32(x);
    let (y_hi, y_lo) = split_q32(y);
    table.extend_from_slice(&[x_hi, x_lo, y_hi, y_lo]);
}

/// Split a source coordinate into its Q32 integer and fraction words.
///
/// Non-finite input maps to the positive limit, which the range check
/// rejects.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to ±(2^30 - 1) so the floor fits i32; the fraction is in [0, 2^32)"
)]
fn split_q32(v: f64) -> (u32, u32) {
    let v = if v.is_finite() {
        v.clamp(-COORD_LIMIT, COORD_LIMIT)
    } else {
        COORD_LIMIT
    };
    let hi = v.floor();
    // `v - hi` is exact for |v| < 2^52 and lies in [0, 1).
    let lo = ((v - hi) * FRACTION_SCALE) as u32;
    ((hi as i32) as u32, lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference sampler: exact f64 inverse map at the pixel centre.
    #[expect(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::suboptimal_flops,
        reason = "test oracle: spelled as the textbook inverse-CTM formula, not for speed"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the range check proves x and y lie in [0, img dim) before the u32 cast"
    )]
    fn reference(
        ctm: &Ctm,
        page_h: u32,
        img_w: u32,
        img_h: u32,
        dx: u32,
        dy: u32,
    ) -> Option<(u32, u32)> {
        let [a, b, c, d, e, f] = *ctm;
        let det = a * d - b * c;
        let dx_rel = f64::from(dx) + 0.5 - e;
        let dy_rel = f64::from(page_h) - (f64::from(dy) + 0.5) - f;
        let u = (d * dx_rel - c * dy_rel) / det;
        let v = (-b * dx_rel + a * dy_rel) / det;
        let x = u * f64::from(img_w);
        let y = (1.0 - v) * f64::from(img_h);
        if x < 0.0 || y < 0.0 || x >= f64::from(img_w) || y >= f64::from(img_h) {
            return None;
        }
        Some((x as u32, y as u32))
    }

    fn assert_matches_reference(ctm: &Ctm, page: (u32, u32), img: (u32, u32)) {
        let grid =
            SampleGrid::new(ctm, page.0, page.1, img.0, img.1, Filter::NEAREST).expect("grid");
        let mut inside = 0;
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let got = grid.sample(dx, dy);
                let want =
                    reference(ctm, page.1, img.0, img.1, dx, dy).map(|(x, y)| Taps::point(x, y));
                assert_eq!(got, want, "pixel ({dx}, {dy})");
                inside += usize::from(got.is_some());
            }
        }
        assert!(inside > 0, "degenerate test geometry");
    }

    #[test]
    fn axis_aligned_fractional_placement_matches_reference() {
        // Fractional translation: the first bbox column starts before the
        // image edge, which the exact map must not snap to.
        let ctm = [1092.43, 0.0, 0.0, -1650.0, 91.285, 1650.0];
        assert_matches_reference(&ctm, (1275, 1650), (1844, 2828));
    }

    #[test]
    fn rotated_placement_matches_reference_and_rejects_outside_unit_square() {
        let ctm = [120.5, 30.25, -20.75, 150.5, 50.33, 40.17];
        let grid = SampleGrid::new(&ctm, 300, 300, 50, 80, Filter::NEAREST).expect("grid");
        let outside = (grid.y0..grid.y1)
            .flat_map(|dy| (grid.x0..grid.x1).map(move |dx| (dx, dy)))
            .filter(|&(dx, dy)| grid.sample(dx, dy).is_none())
            .count();
        assert!(
            outside > 0,
            "a rotated image must leave bbox corners untouched"
        );
        assert_matches_reference(&ctm, (300, 300), (50, 80));
    }

    #[test]
    fn flipped_and_upscaled_placement_matches_reference() {
        // Negative `a` mirrors horizontally; a 3×3 image over 200 px
        // exercises repeated source columns.
        let ctm = [-200.0, 0.0, 0.0, -100.0, 250.0, 150.0];
        assert_matches_reference(&ctm, (300, 200), (3, 3));
    }

    #[test]
    fn exact_integer_scale_is_a_pure_copy() {
        // 1:1 placement at an integer offset: the unit square lands on
        // user-space x 10..50, y 20..50, i.e. device rows 50..80 on a
        // 100 px page, and device pixel (10+k, 50+j) reads source (k, j).
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let grid = SampleGrid::new(&ctm, 100, 100, 40, 30, Filter::NEAREST).expect("grid");
        assert_eq!((grid.x0, grid.y0, grid.x1, grid.y1), (10, 50, 50, 80));
        for dy in 50..80 {
            for dx in 10..50 {
                assert_eq!(grid.sample(dx, dy), Some(Taps::point(dx - 10, dy - 50)));
            }
        }
    }

    #[test]
    fn partially_off_page_bbox_is_clamped() {
        let ctm = [100.0, 0.0, 0.0, -100.0, -30.0, 120.0];
        let grid = SampleGrid::new(&ctm, 50, 50, 10, 10, Filter::NEAREST).expect("grid");
        assert_eq!((grid.x0, grid.x1), (0, 50));
        assert_eq!((grid.y0, grid.y1), (0, 30));
        assert_eq!(grid.cols().len(), 50 * ENTRY_WORDS);
        assert_eq!(grid.rows().len(), 30 * ENTRY_WORDS);
    }

    #[test]
    fn degenerate_inputs_yield_no_grid() {
        let singular = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        assert!(SampleGrid::new(&singular, 10, 10, 4, 4, Filter::NEAREST).is_none());
        let non_finite = [f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(SampleGrid::new(&non_finite, 10, 10, 4, 4, Filter::NEAREST).is_none());
        let off_page = [10.0, 0.0, 0.0, -10.0, 500.0, 500.0];
        assert!(SampleGrid::new(&off_page, 10, 10, 4, 4, Filter::NEAREST).is_none());
        let empty_image = [10.0, 0.0, 0.0, -10.0, 0.0, 10.0];
        assert!(SampleGrid::new(&empty_image, 10, 10, 0, 4, Filter::NEAREST).is_none());
    }

    /// A clip rectangle covering the left half of a 1:1 grid: the bbox is
    /// partial, every row is partial, and coverage splits at the integer clip
    /// edge.
    #[test]
    fn grid_clip_rect_coverage_is_binary() {
        let ctm = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let grid = SampleGrid::new(&ctm, 10, 10, 10, 10, Filter::NEAREST).expect("grid");
        let clip = Clip::new(0.0, 0.0, 5.0, 10.0, true);
        let mut gc = GridClip::new(&clip, &grid);
        assert_eq!(gc.bbox(), ClipResult::Partial);
        let row = gc.row(3).expect("row 3 intersects the clip");
        assert_eq!(row.coverage(0), 255);
        assert_eq!(row.coverage(4), 255);
        assert_eq!(row.coverage(5), 0);
        assert_eq!(row.coverage(9), 0);
    }

    /// A clip rectangle above the grid: the bbox is outside and no row is
    /// admitted; one enclosing the grid: every row admits every column.
    #[test]
    fn grid_clip_bbox_extremes() {
        let ctm = [10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let grid = SampleGrid::new(&ctm, 10, 20, 10, 10, Filter::NEAREST).expect("grid");
        assert_eq!((grid.y0, grid.y1), (10, 20));

        let above = Clip::new(0.0, 0.0, 10.0, 10.0, true);
        let mut gc = GridClip::new(&above, &grid);
        assert_eq!(gc.bbox(), ClipResult::AllOutside);
        assert!(gc.row(10).is_none());

        let enclosing = Clip::new(0.0, 0.0, 10.0, 20.0, true);
        let mut gc = GridClip::new(&enclosing, &grid);
        assert_eq!(gc.bbox(), ClipResult::AllInside);
        let row = gc.row(15).expect("row inside the clip");
        assert!((0..10).all(|dx| row.coverage(dx) == 255));
    }

    /// A mask at four times the image's resolution is resolved per device
    /// pixel: under a 1:1 device mapping of the mask, each device pixel reads
    /// its own mask pixel, while the image repeats each sample four times.
    #[test]
    fn mask_sampler_resolves_a_finer_mask_per_device_pixel() {
        let ctm = [8.0, 0.0, 0.0, 8.0, 0.0, 0.0];
        let (img_w, img_h) = (2, 2);
        let alpha: Vec<u8> = (0..64).collect();
        let mask = SoftMask::new(8, 8, alpha).expect("8×8 mask");
        let grid = SampleGrid::new(&ctm, 8, 8, img_w, img_h, Filter::NEAREST).expect("grid");
        let sampler =
            MaskSampler::new(&mask, &ctm, &ctm, 8, 8, img_w, img_h, false).expect("sampler");
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let px = grid.sample(dx, dy).expect("inside");
                assert_eq!(px, Taps::point(dx / 4, dy / 4));
                let want = u8::try_from(dy * 8 + dx).unwrap();
                assert_eq!(sampler.alpha(dx, dy, &px), want, "pixel ({dx}, {dy})");
            }
        }
    }

    /// A mask on the image's grid is indexed by the image sample itself.
    #[test]
    fn mask_sampler_on_the_image_grid_reuses_the_image_sample() {
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let mask = SoftMask::new(
            40,
            30,
            (0..1200u32)
                .map(|i| u8::try_from(i % 251).unwrap())
                .collect(),
        )
        .expect("40×30 mask");
        let sampler =
            MaskSampler::new(&mask, &ctm, &ctm, 100, 100, 40, 30, false).expect("sampler");
        assert!(sampler.grid.is_none());
        assert_eq!(
            sampler.alpha(0, 0, &Taps::point(3, 2)),
            mask.alpha()[2 * 40 + 3]
        );
    }

    /// Clamping keeps a rounding-edge centre on the nearest mask pixel.
    #[test]
    fn sample_clamped_stays_on_the_image() {
        let ctm = [-200.0, 0.0, 0.0, -100.0, 250.0, 150.0];
        let grid = SampleGrid::new(&ctm, 300, 200, 3, 3, Filter::NEAREST).expect("grid");
        for dy in grid.y0..grid.y1 {
            for dx in grid.x0..grid.x1 {
                let taps = grid.sample_clamped(dx, dy);
                assert!(taps.x.iter().chain(&taps.y).all(|&v| v < 3));
                if let Some(p) = grid.sample(dx, dy) {
                    assert_eq!(p, taps);
                }
            }
        }
        // Device pixels outside the bounding box clamp onto its edge.
        assert_eq!(
            grid.sample_clamped(0, 0),
            grid.sample_clamped(grid.x0, grid.y0)
        );
        assert_eq!(
            grid.sample_clamped(299, 199),
            grid.sample_clamped(grid.x1 - 1, grid.y1 - 1)
        );
    }

    #[test]
    fn split_q32_round_trips_sign_and_fraction() {
        let (hi, lo) = split_q32(2.5);
        assert_eq!((hi.cast_signed(), lo), (2, 1 << 31));
        let (hi, lo) = split_q32(-0.25);
        assert_eq!((hi.cast_signed(), lo), (-1, 3 << 30));
        let (hi, lo) = split_q32(f64::INFINITY);
        assert_eq!((hi.cast_signed(), lo), ((1 << 30) - 1, 0));
        let (hi, lo) = split_q32(-1e300);
        assert_eq!((hi.cast_signed(), lo), (1 - (1 << 30), 0));
    }

    #[test]
    fn filter_selection() {
        let both = Filter { x: true, y: true };
        let draw = |ctm: Ctm, interpolate| Filter::for_draw(&ctm, 10, 10, interpolate);
        // One sample per pixel or fewer keeps the samples.
        assert_eq!(
            draw([10.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(draw([5.0, 0.0, 0.0, -5.0, 0.0, 5.0], true), Filter::NEAREST);
        // Below the magnification limit, each magnified axis blends.
        assert_eq!(draw([20.0, 0.0, 0.0, 20.0, 0.0, 0.0], false), both);
        assert_eq!(
            draw([20.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter { x: true, y: false }
        );
        // A quarter turn magnifying the image's rows only.
        assert_eq!(
            draw([0.0, 10.0, -30.0, 0.0, 0.0, 0.0], false),
            Filter { x: false, y: true }
        );
        // A sub-pixel skew still counts as axis-aligned.
        assert_eq!(
            draw([20.0, 0.4, 0.0, 10.0, 0.0, 0.0], false),
            Filter { x: true, y: false }
        );
        // At the limit, only a flagged image blends.
        assert_eq!(
            draw([40.0, 0.0, 0.0, 10.0, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(
            draw([40.0, 0.0, 0.0, 10.0, 0.0, 0.0], true),
            Filter { x: true, y: false }
        );
        // A rotated draw blends on both axes, at any scale below the limit.
        assert_eq!(draw([8.66, 5.0, -5.0, 8.66, 0.0, 0.0], false), both);
        assert_eq!(
            draw([86.6, 50.0, -50.0, 86.6, 0.0, 0.0], false),
            Filter::NEAREST
        );
        assert_eq!(draw([86.6, 50.0, -50.0, 86.6, 0.0, 0.0], true), both);
    }

    /// Interpolated taps bracket the pixel centre between sample centres,
    /// weighted by its position, and clamp to the edge samples outside them.
    #[test]
    fn bilinear_taps_ramp_between_sample_centres() {
        // Two samples over 100 px: centres at device x 25 and 75.
        let ctm = [100.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let filter = Filter { x: true, y: false };
        let grid = SampleGrid::new(&ctm, 100, 10, 2, 1, filter).expect("grid");
        let taps = |dx| grid.sample(dx, 5).expect("inside");
        assert_eq!(taps(10), Taps::point(0, 0));
        assert_eq!(taps(90), Taps::point(1, 0));
        let mid = taps(49);
        assert_eq!((mid.x, mid.wx), ([0, 1], 125));
        let ramp: Vec<u8> = (0..100)
            .map(|dx| taps(dx).blend(|x, _| [0, 255][x as usize]))
            .collect();
        assert!(ramp.windows(2).all(|p| p[0] <= p[1]));
        assert_eq!((ramp[24], ramp[75]), (0, 255));
        assert!((126..=130).contains(&ramp[50]), "{}", ramp[50]);
    }

    /// At one sample per pixel on an integer grid, every pixel centre is a
    /// sample centre: interpolation reproduces the samples exactly.
    #[test]
    fn bilinear_at_one_to_one_is_a_pure_copy() {
        let ctm = [40.0, 0.0, 0.0, 30.0, 10.0, 20.0];
        let filter = Filter { x: true, y: true };
        let grid = SampleGrid::new(&ctm, 100, 100, 40, 30, filter).expect("grid");
        for dy in 50..80 {
            for dx in 10..50 {
                let taps = grid.sample(dx, dy).expect("inside");
                assert_eq!((taps.wx, taps.wy), (0, 0));
                assert_eq!((taps.x[0], taps.y[0]), (dx - 10, dy - 50));
            }
        }
    }

    #[test]
    fn blend_rounds_to_nearest() {
        let taps = Taps {
            x: [0, 1],
            y: [0, 0],
            wx: 128,
            wy: 0,
        };
        assert_eq!(taps.blend(|x, _| [0, 255][x as usize]), 128);
        let corner = Taps {
            x: [0, 1],
            y: [0, 1],
            wx: 64,
            wy: 192,
        };
        // 0.25 · 0.75 of the (1, 1) sample.
        let v = corner.blend(|x, y| if (x, y) == (1, 1) { 255 } else { 0 });
        assert_eq!(v, 48);
    }
}
