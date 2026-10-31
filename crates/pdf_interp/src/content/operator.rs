//! Typed PDF content stream operator representation.
//!
//! [`decode`] consumes the pending operand stack and produces a strongly-typed
//! [`Operator`] value ready for dispatch to the renderer.

use std::borrow::Cow;

use super::operands::{
    ScnOperands, pop_bytes, pop_f64, pop_i32, pop_matrix, pop_n, pop_number_array, pop_scn, pop2,
    pop3, pop4,
};
use super::tokenizer::Token;

/// A decoded PDF content stream operator with its operands.
///
/// Variants follow the PDF 1.7 / ISO 32000-2 operator table (§8.2).
/// Operand order matches the PDF spec (left-to-right as written in the stream).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Operator<'a> {
    // ── Graphics state ────────────────────────────────────────────────────────
    /// `q` — push graphics state.
    Save,
    /// `Q` — pop graphics state.
    Restore,
    /// `cm a b c d e f` — concatenate matrix to CTM.
    ConcatMatrix([f64; 6]),
    /// `w` — set line width.
    SetLineWidth(f64),
    /// `J` — set line cap style (0 = butt, 1 = round, 2 = projecting square).
    SetLineCap(i32),
    /// `j` — set line join style (0 = miter, 1 = round, 2 = bevel).
    SetLineJoin(i32),
    /// `M` — set miter limit.
    SetMiterLimit(f64),
    /// `d [array] phase` — set dash pattern.
    SetDash {
        /// Dash array lengths in user space units.
        dashes: Vec<f64>,
        /// Phase offset into the dash array.
        phase: f64,
    },
    /// `ri name` — set colour rendering intent.
    SetRenderingIntent(Cow<'a, [u8]>),
    /// `i` — set flatness tolerance (0–100).
    SetFlatness(f64),
    /// `gs name` — apply named extended graphics state dictionary entry.
    SetExtGState(Cow<'a, [u8]>),

    // ── Path construction ─────────────────────────────────────────────────────
    /// `m x y` — begin new subpath at (x, y).
    MoveTo(f64, f64),
    /// `l x y` — append straight line to (x, y).
    LineTo(f64, f64),
    /// `c x1 y1 x2 y2 x3 y3` — append cubic Bézier (both control points explicit).
    CurveTo(f64, f64, f64, f64, f64, f64),
    /// `v x2 y2 x3 y3` — cubic Bézier; first control point = current point.
    CurveToV(f64, f64, f64, f64),
    /// `y x1 y1 x3 y3` — cubic Bézier; second control point = endpoint.
    CurveToY(f64, f64, f64, f64),
    /// `h` — close current subpath with a straight line to the start point.
    ClosePath,
    /// `re x y w h` — append rectangle as a complete subpath.
    Rectangle(f64, f64, f64, f64),

    // ── Path painting ─────────────────────────────────────────────────────────
    /// `S` — stroke path; clear path.
    Stroke,
    /// `s` — close and stroke; clear path.
    CloseStroke,
    /// `f` / `F` — fill path (non-zero winding rule); clear path.
    Fill,
    /// `f*` — fill path (even-odd rule); clear path.
    FillEvenOdd,
    /// `B` — fill then stroke (non-zero); clear path.
    FillStroke,
    /// `B*` — fill then stroke (even-odd); clear path.
    FillStrokeEvenOdd,
    /// `b` — close, fill, stroke (non-zero); clear path.
    CloseFillStroke,
    /// `b*` — close, fill, stroke (even-odd); clear path.
    CloseFillStrokeEvenOdd,
    /// `n` — end path without painting (used after clipping operators).
    EndPath,

    // ── Clipping ─────────────────────────────────────────────────────────────
    /// `W` — intersect clip path with current path (non-zero winding rule).
    Clip,
    /// `W*` — intersect clip path with current path (even-odd rule).
    ClipEvenOdd,

    // ── Text objects ──────────────────────────────────────────────────────────
    /// `BT` — begin text object; initialise text matrix to identity.
    BeginText,
    /// `ET` — end text object; discard text matrix.
    EndText,

    // ── Text state ────────────────────────────────────────────────────────────
    /// `Tf name size` — set font resource and size.
    SetFont {
        /// Font resource name (key into the page Font resource dict).
        name: Cow<'a, [u8]>,
        /// Font size in text space units.
        size: f64,
    },
    /// `Tc` — set character spacing (added to advance after each glyph).
    SetCharSpacing(f64),
    /// `Tw` — set word spacing (added to advance after ASCII SPACE, 0x20).
    SetWordSpacing(f64),
    /// `Tz` — set horizontal text scaling (percentage of normal width).
    SetHorizScaling(f64),
    /// `TL` — set text leading (vertical advance for `T*`, `'`, `"`).
    SetLeading(f64),
    /// `Ts` — set text rise (vertical offset from the baseline).
    SetTextRise(f64),
    /// `Tr` — set text rendering mode (0–7; controls fill/stroke/clip).
    SetTextRenderMode(i32),

    // ── Text positioning ──────────────────────────────────────────────────────
    /// `Td tx ty` — move text position by (tx, ty).
    TextMove(f64, f64),
    /// `TD tx ty` — move text position by (tx, ty) and set leading to −ty.
    TextMoveSetLeading(f64, f64),
    /// `Tm a b c d e f` — set the text matrix and text line matrix directly.
    SetTextMatrix([f64; 6]),
    /// `T*` — move to start of next line (equivalent to `0 −TL Td`).
    NextLine,

    // ── Text showing ──────────────────────────────────────────────────────────
    /// `Tj string` — show a string of glyphs.
    ShowText(Cow<'a, [u8]>),
    /// `TJ array` — show glyphs with individual horizontal adjustments.
    ShowTextArray(Vec<TextArrayElement<'a>>),
    /// `' string` — move to next line, then show string (equivalent to `T* Tj`).
    MoveNextLineShow(Cow<'a, [u8]>),
    /// `" aw ac string` — set word/char spacing, move to next line, show string.
    MoveNextLineShowSpaced {
        /// Word spacing to set before showing.
        aw: f64,
        /// Character spacing to set before showing.
        ac: f64,
        /// String to show.
        text: Cow<'a, [u8]>,
    },

    // ── Colour ────────────────────────────────────────────────────────────────
    /// `cs name` — set fill colour space.
    SetFillColorSpace(Cow<'a, [u8]>),
    /// `CS name` — set stroke colour space.
    SetStrokeColorSpace(Cow<'a, [u8]>),
    /// `sc` / `scn` — set fill colour (components in current fill space).
    SetFillColor(Vec<f64>),
    /// `SC` / `SCN` — set stroke colour (components in current stroke space).
    SetStrokeColor(Vec<f64>),
    /// `scn /Name [c1…cn]` — activate a named Pattern as the fill source.
    ///
    /// `components` carries optional tint values for uncoloured (`PaintType` 2)
    /// patterns; it is empty for coloured (`PaintType` 1) patterns.
    SetFillPattern {
        /// Key into the page `Pattern` resource dictionary.
        name: Cow<'a, [u8]>,
        /// Tint components for `PaintType` 2 patterns (may be empty).
        components: Vec<f64>,
    },
    /// `SCN /Name [c1…cn]` — activate a named Pattern as the stroke source.
    SetStrokePattern {
        /// Key into the page `Pattern` resource dictionary.
        name: Cow<'a, [u8]>,
        /// Tint components for `PaintType` 2 patterns (may be empty).
        components: Vec<f64>,
    },
    /// `g` — set fill colour to gray level (`DeviceGray` shorthand).
    SetFillGray(f64),
    /// `G` — set stroke colour to gray level (`DeviceGray` shorthand).
    SetStrokeGray(f64),
    /// `rg r g b` — set fill colour (`DeviceRGB` shorthand).
    SetFillRgb(f64, f64, f64),
    /// `RG r g b` — set stroke colour (`DeviceRGB` shorthand).
    SetStrokeRgb(f64, f64, f64),
    /// `k c m y k` — set fill colour (`DeviceCMYK` shorthand).
    SetFillCmyk(f64, f64, f64, f64),
    /// `K c m y k` — set stroke colour (`DeviceCMYK` shorthand).
    SetStrokeCmyk(f64, f64, f64, f64),

    // ── XObjects & images ─────────────────────────────────────────────────────
    /// `Do name` — paint an `XObject` (image or form `XObject`).
    PaintXObject(Cow<'a, [u8]>),
    /// `BI … ID … EI` — paint an inline image.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between `BI` and `ID`).
        params: Cow<'a, [u8]>,
        /// Raw image data bytes (between `ID` and `EI`).
        data: Cow<'a, [u8]>,
    },

    // ── Shading ───────────────────────────────────────────────────────────────
    /// `sh name` — paint a shading pattern.
    PaintShading(Cow<'a, [u8]>),

    // ── Type 3 glyph metrics ──────────────────────────────────────────────────
    /// `wx wy d0` — Type 3 coloured glyph: declares advance width (wx, wy).
    ///
    /// Must be the first operator in a Type 3 `CharProc`.  The glyph sets
    /// its own colours (PDF §9.6.5).
    Type3GlyphWidth(f64, f64),
    /// `wx wy llx lly urx ury d1` — Type 3 shape-only glyph with `BBox`.
    ///
    /// Must be the first operator in a Type 3 `CharProc`.  The glyph is a
    /// stencil painted in the text fill colour; its colour operators are
    /// ignored (PDF §9.6.5).  The `BBox` is a hint to the renderer for
    /// caching and clipping.
    Type3GlyphWidthBBox(f64, f64, f64, f64, f64, f64),

    // ── Marked content ────────────────────────────────────────────────────────
    /// `MP` / `DP` — marked-content points, with no rendering effect.
    MarkedContentPoint,
    /// `BMC` / `BDC` with a tag other than `/OC` — begin a marked-content
    /// sequence (PDF §14.6) with no rendering effect, closed by its `EMC`.
    BeginMarkedContent,
    /// `BDC /OC /Name` — begin an optional-content sequence (PDF §8.11.3.2).
    ///
    /// `Name` is a key into the `Properties` resource dictionary; the
    /// renderer resolves it to an optional content group and skips the
    /// enclosed content when that group is hidden.
    BeginOptionalContent(Cow<'a, [u8]>),
    /// `EMC` — end the innermost open marked-content sequence, optional or
    /// not.
    EndMarkedContent,

    // ── Compatibility sections ────────────────────────────────────────────────
    /// `BX` / `EX` — begin/end compatibility section (contents ignored).
    CompatibilitySection,

    // ── Unknown / unimplemented ───────────────────────────────────────────────
    /// An operator keyword the decoder does not recognise.
    Unknown(Cow<'a, [u8]>),
}

impl Operator<'_> {
    /// `true` for the operators that set a fill or stroke colour, colour
    /// space or pattern (PDF §8.6.8).
    #[must_use]
    pub const fn sets_colour(&self) -> bool {
        matches!(
            self,
            Self::SetFillColorSpace(_)
                | Self::SetStrokeColorSpace(_)
                | Self::SetFillColor(_)
                | Self::SetStrokeColor(_)
                | Self::SetFillPattern { .. }
                | Self::SetStrokePattern { .. }
                | Self::SetFillGray(_)
                | Self::SetStrokeGray(_)
                | Self::SetFillRgb(..)
                | Self::SetStrokeRgb(..)
                | Self::SetFillCmyk(..)
                | Self::SetStrokeCmyk(..)
        )
    }
}

/// Element of a `TJ` text-showing array.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TextArrayElement<'a> {
    /// A string of glyph codes to render.
    Text(Cow<'a, [u8]>),
    /// Horizontal adjustment in thousandths of a text-space unit.
    /// Negative values move the text origin to the right.
    Offset(f64),
}

impl TextArrayElement<'_> {
    /// The element with its string copied out of the content stream.
    #[must_use]
    pub fn into_owned(self) -> TextArrayElement<'static> {
        match self {
            Self::Text(text) => TextArrayElement::Text(Cow::Owned(text.into_owned())),
            Self::Offset(offset) => TextArrayElement::Offset(offset),
        }
    }
}

impl Operator<'_> {
    /// The operator with its names and strings copied out of the content
    /// stream, for an operator kept beyond the stream's buffer.
    #[must_use]
    pub fn into_owned(self) -> Operator<'static> {
        match self {
            Self::Save => Operator::Save,
            Self::Restore => Operator::Restore,
            Self::ConcatMatrix(a0) => Operator::ConcatMatrix(a0),
            Self::SetLineWidth(a0) => Operator::SetLineWidth(a0),
            Self::SetLineCap(a0) => Operator::SetLineCap(a0),
            Self::SetLineJoin(a0) => Operator::SetLineJoin(a0),
            Self::SetMiterLimit(a0) => Operator::SetMiterLimit(a0),
            Self::SetDash { dashes, phase } => Operator::SetDash { dashes, phase },
            Self::SetRenderingIntent(a0) => {
                Operator::SetRenderingIntent(Cow::Owned(a0.into_owned()))
            }
            Self::SetFlatness(a0) => Operator::SetFlatness(a0),
            Self::SetExtGState(a0) => Operator::SetExtGState(Cow::Owned(a0.into_owned())),
            Self::MoveTo(a0, a1) => Operator::MoveTo(a0, a1),
            Self::LineTo(a0, a1) => Operator::LineTo(a0, a1),
            Self::CurveTo(a0, a1, a2, a3, a4, a5) => Operator::CurveTo(a0, a1, a2, a3, a4, a5),
            Self::CurveToV(a0, a1, a2, a3) => Operator::CurveToV(a0, a1, a2, a3),
            Self::CurveToY(a0, a1, a2, a3) => Operator::CurveToY(a0, a1, a2, a3),
            Self::ClosePath => Operator::ClosePath,
            Self::Rectangle(a0, a1, a2, a3) => Operator::Rectangle(a0, a1, a2, a3),
            Self::Stroke => Operator::Stroke,
            Self::CloseStroke => Operator::CloseStroke,
            Self::Fill => Operator::Fill,
            Self::FillEvenOdd => Operator::FillEvenOdd,
            Self::FillStroke => Operator::FillStroke,
            Self::FillStrokeEvenOdd => Operator::FillStrokeEvenOdd,
            Self::CloseFillStroke => Operator::CloseFillStroke,
            Self::CloseFillStrokeEvenOdd => Operator::CloseFillStrokeEvenOdd,
            Self::EndPath => Operator::EndPath,
            Self::Clip => Operator::Clip,
            Self::ClipEvenOdd => Operator::ClipEvenOdd,
            Self::BeginText => Operator::BeginText,
            Self::EndText => Operator::EndText,
            Self::SetFont { name, size } => Operator::SetFont {
                name: Cow::Owned(name.into_owned()),
                size,
            },
            Self::SetCharSpacing(a0) => Operator::SetCharSpacing(a0),
            Self::SetWordSpacing(a0) => Operator::SetWordSpacing(a0),
            Self::SetHorizScaling(a0) => Operator::SetHorizScaling(a0),
            Self::SetLeading(a0) => Operator::SetLeading(a0),
            Self::SetTextRise(a0) => Operator::SetTextRise(a0),
            Self::SetTextRenderMode(a0) => Operator::SetTextRenderMode(a0),
            Self::TextMove(a0, a1) => Operator::TextMove(a0, a1),
            Self::TextMoveSetLeading(a0, a1) => Operator::TextMoveSetLeading(a0, a1),
            Self::SetTextMatrix(a0) => Operator::SetTextMatrix(a0),
            Self::NextLine => Operator::NextLine,
            Self::ShowText(a0) => Operator::ShowText(Cow::Owned(a0.into_owned())),
            Self::ShowTextArray(a0) => {
                Operator::ShowTextArray(a0.into_iter().map(TextArrayElement::into_owned).collect())
            }
            Self::MoveNextLineShow(a0) => Operator::MoveNextLineShow(Cow::Owned(a0.into_owned())),
            Self::MoveNextLineShowSpaced { aw, ac, text } => Operator::MoveNextLineShowSpaced {
                aw,
                ac,
                text: Cow::Owned(text.into_owned()),
            },
            Self::SetFillColorSpace(a0) => Operator::SetFillColorSpace(Cow::Owned(a0.into_owned())),
            Self::SetStrokeColorSpace(a0) => {
                Operator::SetStrokeColorSpace(Cow::Owned(a0.into_owned()))
            }
            Self::SetFillColor(a0) => Operator::SetFillColor(a0),
            Self::SetStrokeColor(a0) => Operator::SetStrokeColor(a0),
            Self::SetFillPattern { name, components } => Operator::SetFillPattern {
                name: Cow::Owned(name.into_owned()),
                components,
            },
            Self::SetStrokePattern { name, components } => Operator::SetStrokePattern {
                name: Cow::Owned(name.into_owned()),
                components,
            },
            Self::SetFillGray(a0) => Operator::SetFillGray(a0),
            Self::SetStrokeGray(a0) => Operator::SetStrokeGray(a0),
            Self::SetFillRgb(a0, a1, a2) => Operator::SetFillRgb(a0, a1, a2),
            Self::SetStrokeRgb(a0, a1, a2) => Operator::SetStrokeRgb(a0, a1, a2),
            Self::SetFillCmyk(a0, a1, a2, a3) => Operator::SetFillCmyk(a0, a1, a2, a3),
            Self::SetStrokeCmyk(a0, a1, a2, a3) => Operator::SetStrokeCmyk(a0, a1, a2, a3),
            Self::PaintXObject(a0) => Operator::PaintXObject(Cow::Owned(a0.into_owned())),
            Self::InlineImage { params, data } => Operator::InlineImage {
                params: Cow::Owned(params.into_owned()),
                data: Cow::Owned(data.into_owned()),
            },
            Self::PaintShading(a0) => Operator::PaintShading(Cow::Owned(a0.into_owned())),
            Self::Type3GlyphWidth(a0, a1) => Operator::Type3GlyphWidth(a0, a1),
            Self::Type3GlyphWidthBBox(a0, a1, a2, a3, a4, a5) => {
                Operator::Type3GlyphWidthBBox(a0, a1, a2, a3, a4, a5)
            }
            Self::MarkedContentPoint => Operator::MarkedContentPoint,
            Self::BeginMarkedContent => Operator::BeginMarkedContent,
            Self::BeginOptionalContent(a0) => {
                Operator::BeginOptionalContent(Cow::Owned(a0.into_owned()))
            }
            Self::EndMarkedContent => Operator::EndMarkedContent,
            Self::CompatibilitySection => Operator::CompatibilitySection,
            Self::Unknown(a0) => Operator::Unknown(Cow::Owned(a0.into_owned())),
        }
    }
}

/// Decode a pending operand stack and operator keyword into an [`Operator`].
///
/// `operands` is always cleared on return, regardless of success or failure,
/// so the stack is always empty and ready for the next operator.
#[expect(
    clippy::too_many_lines,
    reason = "operator dispatch table — splitting adds no clarity"
)]
pub(crate) fn decode<'a>(op: &'a [u8], operands: &mut Vec<Token<'a>>) -> Operator<'a> {
    let result = match op {
        // ── Graphics state ────────────────────────────────────────────────────
        b"q" => Operator::Save,
        b"Q" => Operator::Restore,
        b"cm" => Operator::ConcatMatrix(pop_matrix(operands)),
        b"w" => Operator::SetLineWidth(pop_f64(operands)),
        b"J" => Operator::SetLineCap(pop_i32(operands)),
        b"j" => Operator::SetLineJoin(pop_i32(operands)),
        b"M" => Operator::SetMiterLimit(pop_f64(operands)),
        b"d" => {
            // Stream order: `[dash array] phase d`
            // Stack order (LIFO): phase is on top, array below.
            let phase = pop_f64(operands);
            let dashes = pop_number_array(operands);
            Operator::SetDash { dashes, phase }
        }
        b"ri" => Operator::SetRenderingIntent(pop_bytes(operands)),
        b"i" => Operator::SetFlatness(pop_f64(operands)),
        b"gs" => Operator::SetExtGState(pop_bytes(operands)),

        // ── Path construction ─────────────────────────────────────────────────
        b"m" => {
            let (x, y) = pop2(operands);
            Operator::MoveTo(x, y)
        }
        b"l" => {
            let (x, y) = pop2(operands);
            Operator::LineTo(x, y)
        }
        b"c" => {
            let [x1, y1, x2, y2, x3, y3] = pop_n(operands);
            Operator::CurveTo(x1, y1, x2, y2, x3, y3)
        }
        b"v" => {
            let (x2, y2, x3, y3) = pop4(operands);
            Operator::CurveToV(x2, y2, x3, y3)
        }
        b"y" => {
            let (x1, y1, x3, y3) = pop4(operands);
            Operator::CurveToY(x1, y1, x3, y3)
        }
        b"h" => Operator::ClosePath,
        b"re" => {
            let (x, y, w, h) = pop4(operands);
            Operator::Rectangle(x, y, w, h)
        }

        // ── Path painting ─────────────────────────────────────────────────────
        b"S" => Operator::Stroke,
        b"s" => Operator::CloseStroke,
        b"f" | b"F" => Operator::Fill,
        b"f*" => Operator::FillEvenOdd,
        b"B" => Operator::FillStroke,
        b"B*" => Operator::FillStrokeEvenOdd,
        b"b" => Operator::CloseFillStroke,
        b"b*" => Operator::CloseFillStrokeEvenOdd,
        b"n" => Operator::EndPath,

        // ── Clipping ─────────────────────────────────────────────────────────
        b"W" => Operator::Clip,
        b"W*" => Operator::ClipEvenOdd,

        // ── Text objects ──────────────────────────────────────────────────────
        b"BT" => Operator::BeginText,
        b"ET" => Operator::EndText,

        // ── Text state ────────────────────────────────────────────────────────
        b"Tf" => {
            // Stream order: `name size Tf`
            let size = pop_f64(operands);
            let name = pop_bytes(operands);
            Operator::SetFont { name, size }
        }
        b"Tc" => Operator::SetCharSpacing(pop_f64(operands)),
        b"Tw" => Operator::SetWordSpacing(pop_f64(operands)),
        b"Tz" => Operator::SetHorizScaling(pop_f64(operands)),
        b"TL" => Operator::SetLeading(pop_f64(operands)),
        b"Ts" => Operator::SetTextRise(pop_f64(operands)),
        b"Tr" => Operator::SetTextRenderMode(pop_i32(operands)),

        // ── Text positioning ──────────────────────────────────────────────────
        b"Td" => {
            let (tx, ty) = pop2(operands);
            Operator::TextMove(tx, ty)
        }
        b"TD" => {
            let (tx, ty) = pop2(operands);
            Operator::TextMoveSetLeading(tx, ty)
        }
        b"Tm" => Operator::SetTextMatrix(pop_matrix(operands)),
        b"T*" => Operator::NextLine,

        // ── Text showing ──────────────────────────────────────────────────────
        b"Tj" => Operator::ShowText(pop_bytes(operands)),
        b"TJ" => Operator::ShowTextArray(pop_text_array(operands)),
        b"'" => Operator::MoveNextLineShow(pop_bytes(operands)),
        b"\"" => {
            // Stream order: `aw ac string "`
            let text = pop_bytes(operands);
            let ac = pop_f64(operands);
            let aw = pop_f64(operands);
            Operator::MoveNextLineShowSpaced { aw, ac, text }
        }

        // ── Colour ────────────────────────────────────────────────────────────
        b"cs" => Operator::SetFillColorSpace(pop_bytes(operands)),
        b"CS" => Operator::SetStrokeColorSpace(pop_bytes(operands)),
        b"sc" | b"scn" => match pop_scn(operands) {
            ScnOperands::Components(c) => Operator::SetFillColor(c),
            ScnOperands::Pattern { name, components } => {
                Operator::SetFillPattern { name, components }
            }
        },
        b"SC" | b"SCN" => match pop_scn(operands) {
            ScnOperands::Components(c) => Operator::SetStrokeColor(c),
            ScnOperands::Pattern { name, components } => {
                Operator::SetStrokePattern { name, components }
            }
        },
        b"g" => Operator::SetFillGray(pop_f64(operands)),
        b"G" => Operator::SetStrokeGray(pop_f64(operands)),
        b"rg" => {
            let (r, g, b) = pop3(operands);
            Operator::SetFillRgb(r, g, b)
        }
        b"RG" => {
            let (r, g, b) = pop3(operands);
            Operator::SetStrokeRgb(r, g, b)
        }
        b"k" => {
            let (c, m, y, k) = pop4(operands);
            Operator::SetFillCmyk(c, m, y, k)
        }
        b"K" => {
            let (c, m, y, k) = pop4(operands);
            Operator::SetStrokeCmyk(c, m, y, k)
        }

        // ── XObjects & images ─────────────────────────────────────────────────
        b"Do" => Operator::PaintXObject(pop_bytes(operands)),

        // ── Shading ───────────────────────────────────────────────────────────
        b"sh" => Operator::PaintShading(pop_bytes(operands)),

        // ── Type 3 glyph metrics ──────────────────────────────────────────────
        b"d0" => {
            let (wx, wy) = pop2(operands);
            Operator::Type3GlyphWidth(wx, wy)
        }
        b"d1" => {
            let [wx, wy, llx, lly, urx, ury] = pop_n(operands);
            Operator::Type3GlyphWidthBBox(wx, wy, llx, lly, urx, ury)
        }

        // ── Marked content ────────────────────────────────────────────────────
        // BDC with tag /OC begins an Optional Content Group region.
        // Operands on stack: /Tag (bottom) /PropertiesKey (top), or /Tag <<dict>>.
        // An optional-content property list is always named, because an
        // optional content group is an indirect object (PDF §8.11.3.2).
        b"BDC" => {
            let props = operands.pop();
            let tag = operands.pop();
            match (tag, props) {
                (Some(Token::Name(t)), Some(Token::Name(k))) if *t == *b"OC" => {
                    Operator::BeginOptionalContent(k)
                }
                _ => Operator::BeginMarkedContent,
            }
        }
        b"BMC" => Operator::BeginMarkedContent,
        b"EMC" => Operator::EndMarkedContent,
        b"MP" | b"DP" => Operator::MarkedContentPoint,

        // ── PDF compatibility sections (contents opaque to conforming readers) ─
        b"BX" | b"EX" => Operator::CompatibilitySection,

        _ => Operator::Unknown(Cow::Borrowed(op)),
    };

    operands.clear();
    result
}

/// Pop the top of the stack as a `TJ` text array.
fn pop_text_array<'a>(stack: &mut Vec<Token<'a>>) -> Vec<TextArrayElement<'a>> {
    match stack.pop() {
        Some(Token::Array(arr)) => arr
            .into_iter()
            .map(|t| match t {
                Token::String(s) => TextArrayElement::Text(s),
                Token::Number(n) => TextArrayElement::Offset(n),
                Token::Bool(b) => TextArrayElement::Offset(f64::from(u8::from(b))),
                _ => TextArrayElement::Offset(0.0),
            })
            .collect(),
        _ => Vec::new(),
    }
}
