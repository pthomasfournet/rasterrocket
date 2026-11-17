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
//! Integer parts are bounded by ±[`COORD_LIMIT`], so a column + row sum can
//! never overflow `i32`.  The sum of two entries is the exact coordinate only
//! while neither entry exceeds that bound.  A draw whose terms do — a sliver
//! of an image stretched across the page, whose per-axis terms grow with the
//! bounding box while their sum stays on the image — builds no tables: its
//! grid maps every pixel through the inverse matrix in `f64` instead, and only
//! the CPU sampler draws it.

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

mod filter;
mod grid;
mod grid_clip;
mod inverse;
mod mask;
mod taps;

pub(super) use filter::Filter;
pub(super) use grid::{GridView, SampleGrid};
pub(super) use grid_clip::GridClip;
pub(super) use mask::MaskSampler;
pub(super) use taps::{BlendedRows, sample_bytes};
