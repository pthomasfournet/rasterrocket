//! Vertical glyph forms: the `GSUB` `vert` / `vrt2` single
//! substitutions of vertical writing.

use std::collections::HashMap;

use read_fonts::tables::gsub::{SingleSubst, SubstitutionSubtables};
use read_fonts::tables::layout::CoverageTable;
use read_fonts::types::{BigEndian, GlyphId, Tag};
use read_fonts::{FontRef, TableProvider};

use super::{FontProgram, Views};

impl FontProgram {
    /// The vertical form of glyph `gid` through the `GSUB` `vert` / `vrt2`
    /// single substitutions (rotated brackets and punctuation of vertical
    /// writing), or `gid` itself when the program defines none or the form
    /// it names is not a glyph of the program.
    #[must_use]
    pub fn vertical_gid(&self, gid: u32) -> u32 {
        self.vertical_forms
            .get_or_init(|| match self.data.borrow_dependent() {
                Views::Sfnt { font, outlines } => {
                    vertical_forms(font, &|gid| outlines.get(GlyphId::new(gid)).is_some())
                }
                _ => HashMap::new(),
            })
            .get(&gid)
            .copied()
            .unwrap_or(gid)
    }

    /// The composite-font selector of the vertical form of the glyph
    /// `selector` selects (see [`Self::cid_to_gid`]): a CID for a
    /// CID-keyed CFF program, else the glyph index.  `selector` itself
    /// when the glyph has no vertical form.
    #[must_use]
    pub fn vertical_selector(&self, selector: u32) -> u32 {
        let Some(gid) = self.cid_to_gid(selector) else {
            return selector;
        };
        let vertical = self.vertical_gid(gid);
        if vertical == gid {
            return selector;
        }
        self.gid_to_selector(vertical).unwrap_or(selector)
    }
}

/// Glyph index → vertical form from the single substitutions the `vert`
/// and `vrt2` features of `font`'s `GSUB` table apply (the first feature
/// listing a glyph with a form for which `is_glyph` holds wins).
fn vertical_forms(font: &FontRef<'_>, is_glyph: &dyn Fn(u32) -> bool) -> HashMap<u32, u32> {
    let mut forms = HashMap::new();
    let Ok(gsub) = font.gsub() else {
        return forms;
    };
    let (Ok(features), Ok(lookups)) = (gsub.feature_list(), gsub.lookup_list()) else {
        return forms;
    };
    let mut lookup_indices: Vec<u16> = Vec::new();
    for record in features.feature_records() {
        let tag = record.feature_tag();
        if (tag == Tag::new(b"vert") || tag == Tag::new(b"vrt2"))
            && let Ok(feature) = record.feature(features.offset_data())
        {
            lookup_indices.extend(feature.lookup_list_indices().iter().map(BigEndian::get));
        }
    }
    lookup_indices.sort_unstable();
    lookup_indices.dedup();
    for index in lookup_indices {
        let Ok(lookup) = lookups.lookups().get(usize::from(index)) else {
            continue;
        };
        let Ok(SubstitutionSubtables::Single(subtables)) = lookup.subtables() else {
            continue;
        };
        for subtable in subtables.iter().flatten() {
            match subtable {
                SingleSubst::Format1(single) => {
                    if let Ok(coverage) = single.coverage() {
                        let delta = single.delta_glyph_id().into();
                        add_forms(
                            &mut forms,
                            is_glyph,
                            &coverage,
                            delta_targets(&coverage, delta, 0xFFFF),
                        );
                    }
                }
                SingleSubst::Format3(single) => {
                    if let Ok(coverage) = single.coverage() {
                        let delta = single.delta_glyph_id().into();
                        add_forms(
                            &mut forms,
                            is_glyph,
                            &coverage,
                            delta_targets(&coverage, delta, 0xFF_FFFF),
                        );
                    }
                }
                SingleSubst::Format2(single) => {
                    if let Ok(coverage) = single.coverage() {
                        let targets = single.substitute_glyph_ids().iter();
                        add_forms(
                            &mut forms,
                            is_glyph,
                            &coverage,
                            targets.map(|to| to.get().to_u32()),
                        );
                    }
                }
                SingleSubst::Format4(single) => {
                    if let Ok(coverage) = single.coverage() {
                        let targets = single.substitute_glyph_ids().iter();
                        add_forms(
                            &mut forms,
                            is_glyph,
                            &coverage,
                            targets.map(|to| to.get().to_u32()),
                        );
                    }
                }
            }
        }
    }
    forms
}

/// Adds the forms of a single substitution: the covered glyphs, in
/// coverage order, map to `targets`.  A target for which `is_glyph` fails
/// is skipped; a glyph already mapped keeps its earlier form.
fn add_forms(
    forms: &mut HashMap<u32, u32>,
    is_glyph: &dyn Fn(u32) -> bool,
    coverage: &CoverageTable<'_>,
    targets: impl IntoIterator<Item = u32>,
) {
    for (glyph, to) in coverage.iter().zip(targets) {
        if is_glyph(to) {
            let _ = forms.entry(glyph.to_u32()).or_insert(to);
        }
    }
}

/// The targets of a delta single substitution, in coverage order: each
/// covered glyph plus `delta`, reduced to the glyph-index range of the
/// subtable format, whose largest index is the all-ones `mask`.
fn delta_targets<'a>(
    coverage: &CoverageTable<'a>,
    delta: i32,
    mask: u32,
) -> impl Iterator<Item = u32> + 'a {
    coverage
        .iter()
        .map(move |glyph| glyph.to_u32().wrapping_add_signed(delta) & mask)
}
