//! Helper for loading (possibly variable) horizontal glyph metrics.

use raw::{
    tables::{hmtx::Hmtx, hvar::Hvar},
    types::{F2Dot14, GlyphId},
    FontRef, TableProvider,
};

/// Access to horizontal glyph metrics.
///
/// A font without a readable `hmtx` (or `hhea`) table has no metrics:
/// advances read as zero and [`lsb`](Self::lsb) returns `None`.  Font
/// programs embedded in PDF documents often omit both tables.
#[derive(Clone)]
pub(crate) struct GlyphHMetrics<'a> {
    pub hmtx: Option<Hmtx<'a>>,
    pub hvar: Option<Hvar<'a>>,
}

impl<'a> GlyphHMetrics<'a> {
    pub fn new(font: &FontRef<'a>) -> Self {
        // HVAR is only consulted alongside hmtx.
        let hmtx = font.hmtx().ok();
        let hvar = hmtx.as_ref().and_then(|_| font.hvar().ok());
        Self { hmtx, hvar }
    }

    /// Returns the advance width (in font units) for the given glyph and
    /// the location in variation space represented by the set of normalized
    /// coordinates in 2.14 fixed point.
    pub fn advance_width(&self, gid: GlyphId, coords: &'a [F2Dot14]) -> i32 {
        let Some(hmtx) = &self.hmtx else {
            return 0;
        };
        let mut advance = hmtx.advance(gid).unwrap_or_default() as i32;
        if let (false, Some(hvar)) = (coords.is_empty(), &self.hvar) {
            advance += hvar
                .advance_width_delta(gid, coords)
                .map(|delta| delta.to_i32())
                .unwrap_or(0);
        }
        advance
    }

    /// Returns the left side bearing (in font units) for the given glyph and
    /// the location in variation space represented by the set of normalized
    /// coordinates in 2.14 fixed point, or `None` when the font has no
    /// horizontal metrics.
    pub fn lsb(&self, gid: GlyphId, coords: &'a [F2Dot14]) -> Option<i32> {
        let hmtx = self.hmtx.as_ref()?;
        let mut lsb = hmtx.side_bearing(gid).unwrap_or_default() as i32;
        if let (false, Some(hvar)) = (coords.is_empty(), &self.hvar) {
            lsb += hvar
                .lsb_delta(gid, coords)
                .map(|delta| delta.to_i32())
                .unwrap_or(0);
        }
        Some(lsb)
    }
}
