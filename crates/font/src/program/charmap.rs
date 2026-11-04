//! Character-code charmaps: the sfnt `cmap` subtable codes resolve
//! through, Unicode maps derived from glyph names, and the byte-keyed
//! symbolic lookups.

use read_fonts::tables::cmap::{Cmap, CmapSubtable, PlatformId};
use read_fonts::types::GlyphId;
use read_fonts::{FontRef, TableProvider};
use skrifa::{GlyphNameSource, GlyphNames, MetadataProvider};

use super::{Charmap, FontProgram, Outlines};

/// The sfnt `cmap` subtable a character code resolves through.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum SfntCmap {
    /// A Unicode subtable: codes are Unicode scalar values.
    Unicode(usize),
    /// No Unicode or Microsoft Symbol subtable, but `post` glyph names of
    /// which at least one maps to Unicode through the Adobe Glyph List:
    /// Unicode values resolve through those names.
    UnicodeFromNames,
    /// A (1,0) Macintosh Roman subtable: codes index it directly.
    MacRoman(usize),
    /// A (3,0) Microsoft Symbol subtable, whose entries usually sit in the
    /// 0xF000–0xF0FF private-use range (PDF §9.6.6.4).
    Symbol(usize),
    /// No usable subtable: every code resolves to `.notdef`.
    None,
}

impl FontProgram {
    /// `true` when this is an sfnt whose charmap is Unicode (a Unicode
    /// `cmap` subtable, or one derived from `post` glyph names), so
    /// [`Self::char_index`] takes Unicode scalar values.
    #[must_use]
    pub const fn has_unicode_cmap(&self) -> bool {
        matches!(
            self.outlines,
            Outlines::Sfnt {
                cmap: SfntCmap::Unicode(_) | SfntCmap::UnicodeFromNames,
                ..
            }
        )
    }

    /// `true` when the program is an sfnt with no `cmap` subtable keyed by
    /// character code (Unicode, (1,0) Macintosh Roman or (3,0) Microsoft
    /// Symbol).  Its glyphs are then reachable only by glyph index or
    /// through their `post` names; any Unicode charmap it answers
    /// [`Self::char_index`] with is derived from those names.
    #[must_use]
    pub fn lacks_cmap(&self) -> bool {
        match self.outlines {
            Outlines::Sfnt {
                cmap: SfntCmap::None,
                ..
            } => true,
            // Selected only when no Unicode or (3,0) subtable exists.
            Outlines::Sfnt {
                cmap: SfntCmap::UnicodeFromNames,
                ..
            } => !self
                .sfnt()
                .and_then(|font| font.cmap().ok())
                .is_some_and(|cmap| {
                    cmap.encoding_records()
                        .iter()
                        .any(|r| r.platform_id() == PlatformId::Macintosh && r.encoding_id() == 0)
                }),
            _ => false,
        }
    }

    /// Glyph index of character code `code` through the program's charmap,
    /// or 0 (`.notdef`) when it has none.
    ///
    /// The charmap is the one a simple font's codes resolve through when
    /// the PDF supplies no code → glyph table:
    ///
    /// - sfnt: a Unicode subtable (32-bit before 16-bit, last listed
    ///   first); else Unicode values derived from `post` names when the
    ///   font has no (3,0) subtable; else (1,0) Macintosh Roman, looked up
    ///   directly; else (3,0) Microsoft Symbol, looked up directly and then
    ///   with the 0xF000 private-use offset (PDF §9.6.6.4).
    /// - Type 1 and bare CFF: Unicode values derived from glyph names.
    #[must_use]
    pub fn char_index(&self, code: u32) -> u32 {
        match usize::try_from(code)
            .ok()
            .and_then(|i| self.byte_charmap.get(i))
        {
            Some(&gid) => gid,
            None => self.charmap_lookup(code),
        }
    }

    pub(super) fn charmap_lookup(&self, code: u32) -> u32 {
        match &self.outlines {
            Outlines::Sfnt { cmap, .. } => match *cmap {
                SfntCmap::Unicode(record) | SfntCmap::MacRoman(record) => {
                    self.cmap_subtable_lookup(record, code)
                }
                SfntCmap::Symbol(record) => match self.cmap_subtable_lookup(record, code) {
                    0 if code <= 0xFF => self.cmap_subtable_lookup(record, 0xF000 | code),
                    gid => gid,
                },
                SfntCmap::UnicodeFromNames => self.name_charmap_lookup(code),
                SfntCmap::None => 0,
            },
            Outlines::Cff => self.name_charmap_lookup(code),
            Outlines::Type1(font) => font.unicode_charmap().map(code).map_or(0, GlyphId::to_u32),
            // Glyphs are selected by CID only.
            Outlines::CidType1(_) => 0,
        }
    }

    /// Glyph index of the single-byte code `code` through the sfnt's
    /// byte-keyed `cmap` subtables, independent of the charmap
    /// [`Self::char_index`] uses: the (3,0) Microsoft Symbol subtable at
    /// `code`, then at `0xF000 + code`, then the (1,0) Macintosh Roman
    /// subtable at `code` (PDF §9.6.6.4, symbolic TrueType font).  0 when
    /// no such subtable maps the code, and for every non-sfnt program.
    #[must_use]
    pub fn code_index(&self, code: u8) -> u32 {
        let Some(cmap) = self.sfnt().and_then(|font| font.cmap().ok()) else {
            return 0;
        };
        let records = cmap.encoding_records();
        let find = |platform: PlatformId, encoding: u16| {
            records
                .iter()
                .position(|r| r.platform_id() == platform && r.encoding_id() == encoding)
        };
        let code = u32::from(code);
        let symbol = find(PlatformId::Windows, 0).map_or(0, |record| {
            match self.cmap_subtable_lookup(record, code) {
                0 => self.cmap_subtable_lookup(record, 0xF000 | code),
                gid => gid,
            }
        });
        if symbol != 0 {
            return symbol;
        }
        find(PlatformId::Macintosh, 0).map_or(0, |record| self.cmap_subtable_lookup(record, code))
    }

    fn cmap_subtable_lookup(&self, record: usize, code: u32) -> u32 {
        let Some(font) = self.sfnt() else {
            return 0;
        };
        font.cmap()
            .ok()
            .and_then(|cmap| {
                let subtable = cmap
                    .encoding_records()
                    .get(record)?
                    .subtable(cmap.offset_data())
                    .ok()?;
                subtable.map_codepoint(code)
            })
            .map_or(0, GlyphId::to_u32)
    }

    fn name_charmap_lookup(&self, code: u32) -> u32 {
        self.name_charmap
            .get_or_init(|| {
                let names = self.glyph_names();
                Charmap::from_glyph_names(
                    names
                        .iter()
                        .map(|(gid, name)| (GlyphId::new(*gid), name.as_ref())),
                )
            })
            .map(code)
            .map_or(0, GlyphId::to_u32)
    }
}

/// Choose the `cmap` subtable character codes resolve through.  When the
/// choice is [`SfntCmap::UnicodeFromNames`], the Unicode map built from the
/// `post` names comes back with it.
pub(super) fn select_sfnt_cmap(font: &FontRef<'_>) -> (SfntCmap, Option<Charmap>) {
    let Ok(cmap) = font.cmap() else {
        return post_name_charmap(font).map_or((SfntCmap::None, None), |map| {
            (SfntCmap::UnicodeFromNames, Some(map))
        });
    };
    if let Some(i) = unicode_subtable_index(&cmap) {
        return (SfntCmap::Unicode(i), None);
    }
    let records = cmap.encoding_records();
    let find = |platform: PlatformId, encoding: u16| {
        records
            .iter()
            .position(|r| r.platform_id() == platform && r.encoding_id() == encoding)
    };
    let symbol = find(PlatformId::Windows, 0);
    if symbol.is_none()
        && let Some(map) = post_name_charmap(font)
    {
        return (SfntCmap::UnicodeFromNames, Some(map));
    }
    let choice = find(PlatformId::Macintosh, 0)
        .map(SfntCmap::MacRoman)
        .or_else(|| symbol.map(SfntCmap::Symbol))
        .unwrap_or(SfntCmap::None);
    (choice, None)
}

/// Index of the encoding record of `cmap` whose subtable maps Unicode
/// scalar values: among the readable Unicode-platform, ISO, (3,1) and
/// (3,10) records, a full-repertoire subtable before a BMP one and, within
/// each class, the last listed.  Variation-sequence (format 14) subtables
/// map no characters and are skipped.
pub(crate) fn unicode_subtable_index(cmap: &Cmap<'_>) -> Option<usize> {
    let records = cmap.encoding_records();
    let format = |i: usize| records[i].subtable(cmap.offset_data()).ok();
    let is_unicode = |i: usize| {
        let r = &records[i];
        matches!(
            (r.platform_id(), r.encoding_id()),
            (PlatformId::Unicode | PlatformId::ISO, _) | (PlatformId::Windows, 1 | 10)
        ) && !matches!(format(i), Some(CmapSubtable::Format14(_)) | None)
    };
    let is_ucs4 = |i: usize| {
        let r = &records[i];
        match (r.platform_id(), r.encoding_id()) {
            (PlatformId::Windows, 10) | (PlatformId::Unicode, 4) => true,
            (PlatformId::Unicode, 6) => matches!(format(i), Some(CmapSubtable::Format13(_))),
            _ => false,
        }
    };
    (0..records.len())
        .rev()
        .find(|&i| is_unicode(i) && is_ucs4(i))
        .or_else(|| (0..records.len()).rev().find(|&i| is_unicode(i)))
}

/// Unicode map derived from the `post` glyph names, `None` when the table
/// names no glyph or no name maps to Unicode.
fn post_name_charmap(font: &FontRef<'_>) -> Option<Charmap> {
    if !font.post().is_ok_and(|post| post.num_names() != 0) {
        return None;
    }
    let names = font.glyph_names();
    if names.source() != GlyphNameSource::Post {
        return None;
    }
    let pairs: Vec<(GlyphId, String)> = named_glyphs(&names).collect();
    let map = Charmap::from_glyph_names(pairs.iter().map(|(gid, name)| (*gid, name.as_str())));
    let maps_any = map.iter().next().is_some();
    maps_any.then_some(map)
}

/// The (glyph, name) pairs of `names` in glyph order, without the
/// synthesized `gidN` names.
pub(super) fn named_glyphs<'a>(
    names: &GlyphNames<'a>,
) -> impl Iterator<Item = (GlyphId, String)> + 'a {
    names
        .iter()
        .filter(|(_, name)| !name.is_synthesized())
        .map(|(gid, name)| (gid, name.as_str().to_owned()))
}
