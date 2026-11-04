//! Stroked glyph outlines as cached coverage bitmaps.
//!
//! A text rendering mode that strokes (PDF §9.3.6) strokes every glyph
//! outline with the same parameters.  When that stroke is a solid, opaque,
//! undashed one in the normal blend mode, each glyph's stroke is rendered
//! once per glyph, stroke and quarter-pixel phase into an 8-bit coverage
//! bitmap and composited like a glyph bitmap thereafter.

use std::f64::consts::SQRT_2;

use font::bitmap::GlyphBitmap as CachedGlyph;
use font::key::FRACTION;
use raster::path::Path;
use raster::types::{BlendMode, LineCap, LineJoin};
use raster::{StrokeParams, stroke_outline};

use super::super::gstate::InterpGState;

/// Most pixels a cached stroke covers; a larger glyph strokes as a path.
const MAX_CACHED_AREA: f64 = (1 << 18) as f64;

/// The stroke of the graphics state as it applies to every glyph outline of
/// a text run, in device pixels with y down.
pub(super) struct GlyphStroke {
    /// The 2×2 part of the user-to-device matrix.
    matrix: [f64; 4],
    /// The inverse of `matrix`.
    inverse: [f64; 4],
    /// Line width in user space.
    line_width: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    miter_limit: f64,
    /// Flattening tolerance in user space.
    flatness: f64,
    /// Device-pixel distance past the outline the stroke can reach.
    reach: f64,
}

impl GlyphStroke {
    /// The stroke of `gs` under the user-to-device matrix `device`, or
    /// `None` when it cannot be cached per glyph: without antialiasing, with
    /// a pattern colour, a constant alpha below 1, a blend mode other than
    /// normal, a dash pattern, a line width that is not positive (a
    /// hairline), or a singular matrix.  Overlapping strokes of neighbouring
    /// glyphs composite as one painted area only under these conditions.
    pub(super) fn new(gs: &InterpGState, device: &[f64; 6], antialias: bool) -> Option<Self> {
        if !antialias
            || gs.stroke_pattern.is_some()
            || gs.stroke_alpha != u8::MAX
            || gs.blend_mode != BlendMode::Normal
            || !gs.dash.0.is_empty()
            || gs.line_width.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        {
            return None;
        }
        let [a, b, c, d, _, _] = *device;
        let det = a.mul_add(d, -(b * c));
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let reach = gs.line_width / 2.0 * a.hypot(b).max(c.hypot(d)) * gs.miter_limit.max(SQRT_2);
        if !reach.is_finite() {
            return None;
        }
        Some(Self {
            matrix: [a, b, c, d],
            inverse: [d / det, -b / det, -c / det, a / det],
            line_width: gs.line_width,
            line_cap: gs.line_cap,
            line_join: gs.line_join,
            miter_limit: gs.miter_limit,
            flatness: gs.flatness.max(0.1) / det.abs().sqrt(),
            reach,
        })
    }

    /// The bit patterns of every parameter the coverage depends on besides
    /// the glyph and its phase.
    pub(super) fn key(&self) -> [u64; 9] {
        let [m0, m1, m2, m3] = self.matrix.map(f64::to_bits);
        [
            m0,
            m1,
            m2,
            m3,
            self.line_width.to_bits(),
            self.miter_limit.to_bits(),
            self.flatness.to_bits(),
            self.line_cap as u64,
            self.line_join as u64,
        ]
    }

    /// The coverage of the stroke of `outline`, a glyph outline in device
    /// pixels relative to the pen with y up, placed `x_frac` quarter pixels
    /// right of the pen pixel's left edge; offsets are those of
    /// [`raster::fill_glyph`].  `None` for an outline with no points or a
    /// stroke covering more than [`MAX_CACHED_AREA`] pixels.
    pub(super) fn coverage(&self, outline: &Path, x_frac: u8) -> Option<CachedGlyph> {
        let shift = f64::from(x_frac) / f64::from(FRACTION);
        let (mut x_min, mut y_min) = (f64::INFINITY, f64::INFINITY);
        let (mut x_max, mut y_max) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &outline.pts {
            let (x, y) = (p.x + shift, -p.y);
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
        // One pixel of margin on each side holds the antialiased edge.
        let left = (x_min - self.reach).floor() - 1.0;
        let top = (y_min - self.reach).floor() - 1.0;
        let width = (x_max + self.reach).ceil() + 1.0 - left;
        let height = (y_max + self.reach).ceil() + 1.0 - top;
        // Comparisons with NaN are false, so a non-finite bound fails here.
        let fits = width > 0.0
            && height > 0.0
            && width * height <= MAX_CACHED_AREA
            && left.abs() <= MAX_CACHED_AREA
            && top.abs() <= MAX_CACHED_AREA;
        if !fits {
            return None;
        }
        // The outline in the bitmap's device space, mapped back to user
        // space: the stroker builds the outline there and maps it forward.
        let [i0, i1, i2, i3] = self.inverse;
        let to_user = [i0, i1, i2, i3, 0.0, 0.0];
        let user = Path {
            pts: outline
                .pts
                .iter()
                .map(|p| raster::xpath::transform(&to_user, p.x + shift - left, -p.y - top))
                .collect(),
            ..outline.clone()
        };
        let [m0, m1, m2, m3] = self.matrix;
        let params = StrokeParams {
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            flatness: self.flatness,
            line_dash: &[],
            line_dash_phase: 0.0,
            stroke_adjust: false,
            vector_antialias: true,
        };
        let xpath = stroke_outline(&user, &[m0, m1, m2, m3, 0.0, 0.0], &params)?;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "width and height are positive integers at most MAX_CACHED_AREA; \
                      left and top are integers of magnitude at most MAX_CACHED_AREA"
        )]
        let (width, height, x_off, y_off) =
            (width as u32, height as u32, -left as i32, -top as i32);
        let data = raster::fill::aa_coverage(xpath, width, height);
        Some(CachedGlyph {
            x_off,
            y_off,
            width,
            height,
            aa: true,
            data,
        })
    }
}
