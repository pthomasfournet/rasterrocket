//! Shaping: the glyphs and positions an sfnt's OpenType layout tables
//! (`GSUB`, `GPOS`) give a run of text, with the default features of the
//! run's script.

use harfrust::{Direction, ShapeOptions, ShaperData, UnicodeBuffer};

use super::FontProgram;

/// One glyph of a shaped run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    /// The glyph's composite-font selector, as the `*_by_gid` methods of
    /// [`crate::face::FontFace`] take it (see [`FontProgram::cid_to_gid`]).
    pub selector: u32,
    /// The advance after the glyph, in em.
    pub advance: f64,
    /// The displacement of the glyph from its pen position, in em, y up.
    pub offset: (f64, f64),
}

impl FontProgram {
    /// `text`, in logical order, shaped right to left when `rtl`, else left
    /// to right.  The script is the one `text`'s characters determine.  The
    /// glyphs are in visual order, left to right.
    ///
    /// `None` for a program that is not an sfnt, or when a shaped glyph has
    /// no selector (a CID-keyed program whose charset omits it).
    #[must_use]
    pub fn shape(&self, text: &str, rtl: bool) -> Option<Vec<ShapedGlyph>> {
        let font = self.sfnt()?;
        let data = self
            .shaper
            .get_or_init(|| Some(ShaperData::new(font)))
            .as_ref()?;
        let shaper = data.shaper(font).build();
        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(text);
        buffer.set_direction(if rtl {
            Direction::RightToLeft
        } else {
            Direction::LeftToRight
        });
        buffer.guess_segment_properties();
        let glyphs = shaper.shape(buffer, ShapeOptions::new());
        let em = |units: i32| f64::from(units) / self.upem;
        glyphs
            .glyph_infos()
            .iter()
            .zip(glyphs.glyph_positions())
            .map(|(info, pos)| {
                Some(ShapedGlyph {
                    selector: self.gid_to_selector(info.glyph_id)?,
                    advance: em(pos.x_advance),
                    offset: (em(pos.x_offset), em(pos.y_offset)),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::FontProgram;

    fn ahem() -> FontProgram {
        FontProgram::parse(font_test_data::AHEM.to_vec(), 0).expect("Ahem parses")
    }

    /// Ahem has no layout tables, so shaping keeps its charmap glyphs and
    /// `hmtx` advances.
    #[test]
    fn a_program_without_layout_tables_keeps_charmap_glyphs() {
        let program = ahem();
        let glyphs = program.shape("Ap", false).expect("an sfnt shapes");
        let expected: Vec<(u32, Option<f64>)> = "Ap"
            .chars()
            .map(|c| {
                let gid = program.char_index(u32::from(c));
                (gid, program.advance(gid))
            })
            .collect();
        let shaped: Vec<(u32, Option<f64>)> = glyphs
            .iter()
            .map(|g| (g.selector, Some(g.advance)))
            .collect();
        assert_eq!(shaped, expected);
        assert!(glyphs.iter().all(|g| g.offset == (0.0, 0.0)));
    }

    #[test]
    fn right_to_left_glyphs_come_in_visual_order() {
        let program = ahem();
        let ltr = program.shape("Ap", false).expect("an sfnt shapes");
        let mut rtl = program.shape("Ap", true).expect("an sfnt shapes");
        rtl.reverse();
        assert_eq!(rtl, ltr);
        assert_ne!(
            ltr[0].selector, ltr[1].selector,
            "the test needs two glyphs"
        );
    }
}
