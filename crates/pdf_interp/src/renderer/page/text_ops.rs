//! Text-rendering helpers extracted from `page/mod.rs`.
//!
//! These types and functions serve text showing: [`super::PageRenderer::show_text`]
//! and the `TJ` kerning offsets in the operator dispatch.

use crate::renderer::gstate::ctm_transform;

// ── Glyph record ──────────────────────────────────────────────────────────────

/// Holds rasterized glyph data for two-phase text rendering.
///
/// The bitmap is shared with the glyph cache rather than copied: the same
/// glyph typically recurs many times per page, and phase 2 only reads it.
pub(super) struct GlyphRecord {
    pub pen_x: i32,
    pub pen_y: i32,
    pub bitmap: std::sync::Arc<font::bitmap::GlyphBitmap>,
}

// ── Coordinate helpers ────────────────────────────────────────────────────────

/// Horizontal text-space advance of one glyph (PDF §9.4.4):
/// `(w × Tfs + Tc + Tw) × Th`, where `extra` is the word spacing when it
/// applies and `0.0` otherwise.
///
/// Evaluated as separate multiplies and adds.  The advance accumulates into
/// the text matrix glyph after glyph, and a fused multiply-add rounds
/// differently, which moves glyph pen positions by a device pixel.
#[expect(
    clippy::suboptimal_flops,
    reason = "unfused evaluation is the rounding the rendered pen positions depend on"
)]
pub(super) fn glyph_advance(
    width: f64,
    font_size: f64,
    char_spacing: f64,
    extra: f64,
    horiz_scaling: f64,
) -> f64 {
    (width * font_size + char_spacing + extra) * horiz_scaling
}

/// Translate text matrix `tm` by `tx` along its own x axis: `[1 0 0 1 tx 0] × Tm`.
///
/// Evaluated as separate multiplies and adds, for the reason given on
/// [`glyph_advance`].
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
pub(super) const fn translate_text_x(tm: [f64; 6], tx: f64) -> [f64; 6] {
    let [a, b, c, d, e, f] = tm;
    [a, b, c, d, e + tx * a, f + tx * b]
}

/// Translate text matrix `tm` by `ty` along its own y axis: `[1 0 0 1 0 ty] × Tm`,
/// the advance of vertical writing mode.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
pub(super) const fn translate_text_y(tm: [f64; 6], ty: f64) -> [f64; 6] {
    let [a, b, c, d, e, f] = tm;
    [a, b, c, d, e + ty * c, f + ty * d]
}

/// Map a text-space point through the text matrix and CTM to device-pixel
/// coordinates with y-flip (PDF origin = bottom-left; device origin = top-left).
///
/// `tm[6]` is `[a, b, c, d, e, f]` in PDF column-major form.
/// The full mapping is: `device = CTM × Tm × (tx, ty+rise)`.
#[expect(clippy::many_single_char_names, reason = "PDF matrix components")]
pub(super) fn text_to_device(
    ctm: &[f64; 6],
    tm: &[f64; 6],
    tx: f64,
    ty: f64,
    page_h: u32,
) -> (i32, i32) {
    // Apply text matrix: user_space = Tm * (tx, ty).
    let [a, b, c, d, e, f] = *tm;
    let ux = a.mul_add(tx, c * ty) + e;
    let uy = b.mul_add(tx, d * ty) + f;

    // Apply CTM: device = CTM * (ux, uy).
    let (dx, dy) = ctm_transform(ctm, ux, uy);

    // y-flip.
    let dy_flipped = f64::from(page_h) - dy;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "device pixels are always in [0, page_h|w]; round() output fits i32 for any real page"
    )]
    (dx.round() as i32, dy_flipped.round() as i32)
}
