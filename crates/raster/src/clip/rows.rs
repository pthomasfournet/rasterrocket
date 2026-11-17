//! Row-by-row clip coverage over a fixed column span.

use super::{Clip, ClipResult};

// ── Row coverage ──────────────────────────────────────────────────────────────

/// Per-pixel coverage of a clip over a fixed column span, row by row.
///
/// The span's bounding box is classified once against the clip; rows and
/// then pixels are resolved only as finely as that classification requires.
/// A row the clip edge crosses yields the same 0–255 shape a fill's edge
/// would: the clip's exact-area coverage ([`Clip::mul_row`] of full
/// coverage).  A rectangle-only clip has integer bounds, so its columns come
/// out fully covered or uncovered, as do those of an aliased path clip.
pub struct ClipRows<'a> {
    clip: &'a Clip,
    bbox: ClipResult,
    /// Inclusive column span, clamped to non-negative columns.
    x0: i32,
    x1: i32,
    /// Per-column coverage of the current partial row, indexed from `x0`.
    coverage: Vec<u8>,
}

impl<'a> ClipRows<'a> {
    /// Coverage of `clip` over the inclusive rectangle `[x0, x1] × [y0, y1]`.
    /// Columns left of zero and right of the clip rectangle are outside.
    #[must_use]
    pub fn new(clip: &'a Clip, x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let x0 = x0.max(0);
        let bbox = if x0 > x1 || y0 > y1 {
            ClipResult::AllOutside
        } else {
            clip.test_rect(x0, y0, x1, y1)
        };
        Self {
            clip,
            bbox,
            x0,
            x1,
            coverage: Vec::new(),
        }
    }

    /// Classification of the whole rectangle.
    #[must_use]
    pub const fn bbox(&self) -> ClipResult {
        self.bbox
    }

    /// Coverage of row `y` over the span, or `None` when the clip excludes
    /// the entire row.
    pub fn row(&mut self, y: i32) -> Option<RowClip<'_>> {
        let res = match self.bbox {
            ClipResult::Partial => self.clip.test_span(self.x0, self.x1, y),
            other => other,
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
    #[expect(
        clippy::cast_sign_loss,
        reason = "0 ≤ x0 ≤ lo ≤ hi, and the clipped span lies within [lo, hi]"
    )]
    fn partial_coverage(&mut self, y: i32) {
        self.coverage.clear();
        // Columns outside the clip rectangle have no coverage, and
        // `RowClip::coverage` reads columns past the vector as 0, so the work
        // and the scratch are bounded by the rectangle, not by the span.
        let (lo, hi) = (
            self.x0.max(self.clip.x_min_i),
            self.x1.min(self.clip.x_max_i),
        );
        if lo > hi {
            return;
        }
        self.coverage.resize((hi - self.x0 + 1) as usize, 0);
        let span = &mut self.coverage[(lo - self.x0) as usize..];
        span.fill(u8::MAX);
        self.clip.mul_row(y, lo, span);
    }
}

/// Clip coverage of the pixels on one row of a [`ClipRows`] span.
#[derive(Clone, Copy, Debug)]
pub enum RowClip<'a> {
    /// The whole row lies inside the clip.
    Full,
    /// The clip edge crosses the row; `coverage[i]` is the shape (0–255) of
    /// device column `x0 + i`.
    Partial {
        /// Device column of `coverage[0]`.
        x0: i32,
        /// Per-column shape.
        coverage: &'a [u8],
    },
}

impl RowClip<'_> {
    /// Clip coverage (0 = outside, 255 = inside) of device column `x`.
    #[inline]
    #[must_use]
    pub fn coverage(&self, x: i32) -> u8 {
        match self {
            Self::Full => 255,
            Self::Partial { x0, coverage } => usize::try_from(x - x0)
                .ok()
                .and_then(|i| coverage.get(i))
                .copied()
                .unwrap_or(0),
        }
    }
}
