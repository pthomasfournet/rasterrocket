//! Parsed font programs.
//!
//! A [`FontProgram`] owns the bytes of one font program and answers every
//! question the renderer asks of it: which glyph a character code or glyph
//! name selects, the advance width of a glyph, and the glyph outline in
//! em units.  The container format is detected from the leading bytes, not
//! from the PDF font dictionary, because producers routinely embed one
//! format under another's stream key.
//!
//! | Format | Detected by | Outlines |
//! |--------|-------------|----------|
//! | sfnt (TrueType, OpenType, collections) | `00 01 00 00`, `true`, `OTTO`, `ttcf` | `glyf` / `CFF ` / `CFF2` |
//! | bare CFF (Type 1C, CID-keyed CFF) | major version 1 or 2, header size ≥ 4 | charstrings |
//! | Type 1 (PFA, PFB) | `%!` or the PFB segment marker `80 01` | eexec-encrypted charstrings |
//!
//! Outlines are unhinted.

use std::collections::HashMap;
use std::sync::OnceLock;

use read_fonts::ps::cff::CffFontRef;
use read_fonts::ps::charmap::Charmap;
use read_fonts::ps::type1::Type1Font;
use read_fonts::tables::cmap::{CmapSubtable, PlatformId};
use read_fonts::types::{GlyphId, Tag};
use read_fonts::{FontRef, TableProvider};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use skrifa::{GlyphNameSource, MetadataProvider};

use crate::engine::LoadError;

/// The container format of a font program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FontFormat {
    /// An sfnt container: TrueType (`glyf`) or OpenType (`CFF ` / `CFF2`)
    /// outlines, possibly inside a TrueType collection.
    Sfnt,
    /// A bare Compact Font Format program (PDF `FontFile3` with subtype
    /// `Type1C` or `CIDFontType0C`).
    Cff,
    /// A PostScript Type 1 program in PFA (ASCII) or PFB (segmented)
    /// layout.
    Type1,
}

impl FontFormat {
    /// Detect the format from the leading bytes of a font program.
    ///
    /// Returns `None` when the bytes match none of the supported formats.
    #[must_use]
    pub const fn detect(data: &[u8]) -> Option<Self> {
        match data {
            [0x00, 0x01, 0x00, 0x00, ..]
            | [b't', b'r', b'u', b'e', ..]
            | [b'O', b'T', b'T', b'O', ..]
            | [b't', b't', b'c', b'f', ..] => Some(Self::Sfnt),
            [0x80, 0x01, ..] | [b'%', b'!', ..] => Some(Self::Type1),
            // CFF header: major, minor, hdrSize, offSize.  CFF2 headers are
            // five bytes long.
            [1 | 2, _, hdr_size, ..] if *hdr_size >= 4 => Some(Self::Cff),
            _ => None,
        }
    }
}

/// The sfnt `cmap` subtable a character code resolves through.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SfntCmap {
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

/// Format-specific outline source.
enum Outlines {
    /// sfnt container; the face index selects a member of a collection.
    Sfnt { face_index: u32, cmap: SfntCmap },
    /// Bare CFF program.
    Cff,
    /// Type 1 program, decrypted and parsed once at load.
    Type1(Box<Type1Font>),
}

/// A parsed font program, shared by every [`crate::face::FontFace`] built
/// from the same bytes.
pub struct FontProgram {
    data: Box<[u8]>,
    outlines: Outlines,
    /// Outline coordinates are in units of `1 / upem` em.
    upem: f64,
    /// CID → glyph index for CID-keyed CFF programs (bare or in an sfnt):
    /// the charset maps glyph indices to CIDs.  `None` for every other
    /// program, whose glyph indices are used as given.
    cid_to_gid: Option<HashMap<u32, u32>>,
    /// Glyph indices of the single-byte codes 0–255 through the program's
    /// own charmap, resolved once at load.
    byte_charmap: Box<[u32; 256]>,
    /// Glyph name → glyph index, built on first use.
    names: OnceLock<HashMap<Box<str>, u32>>,
    /// Unicode → glyph index derived from glyph names, for programs whose
    /// charmap is built from names (Type 1, bare CFF, sfnt without a
    /// Unicode subtable).  Built on first use.
    name_charmap: OnceLock<Charmap>,
}

impl std::fmt::Debug for FontProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontProgram")
            .field("format", &self.format())
            .field("len", &self.data.len())
            .field("upem", &self.upem)
            .finish_non_exhaustive()
    }
}

impl FontProgram {
    /// Parse a font program.  `face_index` selects the member of a
    /// TrueType collection and is ignored for other formats.
    ///
    /// # Errors
    ///
    /// [`LoadError::UnsupportedFormat`] when the bytes match no supported
    /// format; [`LoadError::Malformed`] when they do but cannot be parsed
    /// or carry no outlines.
    pub fn parse(data: Vec<u8>, face_index: u32) -> Result<Self, LoadError> {
        let format = FontFormat::detect(&data).ok_or(LoadError::UnsupportedFormat)?;
        let data = data.into_boxed_slice();
        let name_charmap = OnceLock::new();
        let (outlines, upem, cid_to_gid) = match format {
            FontFormat::Sfnt => {
                let font = FontRef::from_index(&data, face_index)
                    .map_err(|e| LoadError::Malformed(format!("sfnt: {e}")))?;
                if font.outline_glyphs().format().is_none() {
                    return Err(LoadError::Malformed("sfnt has no glyph outlines".into()));
                }
                let upem = font
                    .head()
                    .map(|head| head.units_per_em())
                    .map_err(|e| LoadError::Malformed(format!("sfnt head: {e}")))?;
                let cid_to_gid = font
                    .table_data(Tag::new(b"CFF "))
                    .and_then(|cff| CffFontRef::new(cff.as_bytes(), 0, Some(i32::from(upem))).ok())
                    .and_then(|cff| cid_charset_map(&cff));
                let (cmap, post_charmap) = select_sfnt_cmap(&font);
                if let Some(charmap) = post_charmap {
                    let _ = name_charmap.set(charmap);
                }
                (
                    Outlines::Sfnt { face_index, cmap },
                    f64::from(upem),
                    cid_to_gid,
                )
            }
            FontFormat::Cff => {
                let cff = CffFontRef::new(&data, 0, None)
                    .map_err(|e| LoadError::Malformed(format!("CFF: {e}")))?;
                if cff.num_glyphs() == 0 {
                    return Err(LoadError::Malformed("CFF has no charstrings".into()));
                }
                let cid_to_gid = cid_charset_map(&cff);
                (Outlines::Cff, f64::from(cff.upem()), cid_to_gid)
            }
            FontFormat::Type1 => {
                let font = Type1Font::new(&data)
                    .map_err(|e| LoadError::Malformed(format!("Type 1: {e}")))?;
                let upem = f64::from(font.upem());
                (Outlines::Type1(Box::new(font)), upem, None)
            }
        };
        if !(upem.is_finite() && upem > 0.0) {
            return Err(LoadError::Malformed(format!("units per em {upem}")));
        }
        let mut program = Self {
            data,
            outlines,
            upem,
            cid_to_gid,
            byte_charmap: Box::new([0; 256]),
            names: OnceLock::new(),
            name_charmap,
        };
        let mut table = [0u32; 256];
        for (code, slot) in (0u32..).zip(table.iter_mut()) {
            *slot = program.charmap_lookup(code);
        }
        *program.byte_charmap = table;
        Ok(program)
    }

    /// The detected container format.
    #[must_use]
    pub const fn format(&self) -> FontFormat {
        match self.outlines {
            Outlines::Sfnt { .. } => FontFormat::Sfnt,
            Outlines::Cff => FontFormat::Cff,
            Outlines::Type1(_) => FontFormat::Type1,
        }
    }

    /// Bytes of the program, exactly as given to [`Self::parse`].
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

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

    fn charmap_lookup(&self, code: u32) -> u32 {
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
        }
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
                        .map(|(gid, name)| (GlyphId::new(*gid), name.as_str())),
                )
            })
            .map(code)
            .map_or(0, GlyphId::to_u32)
    }

    /// Glyph index of the glyph named `name` in the program's own glyph
    /// names (Type 1 `CharStrings`, CFF charset, sfnt `post` or CFF table),
    /// or 0 when absent.  The first glyph carrying the name wins.
    #[must_use]
    pub fn name_index(&self, name: &str) -> u32 {
        self.names
            .get_or_init(|| {
                let mut map = HashMap::new();
                for (gid, name) in self.glyph_names() {
                    let _ = map.entry(name.into_boxed_str()).or_insert(gid);
                }
                map
            })
            .get(name)
            .copied()
            .unwrap_or(0)
    }

    /// Every (glyph index, name) pair the program defines, in glyph order.
    /// Synthesized names (`gidN`) are not included.
    fn glyph_names(&self) -> Vec<(u32, String)> {
        match &self.outlines {
            Outlines::Sfnt { .. } => self.sfnt().map_or_else(Vec::new, |font| {
                font.glyph_names()
                    .iter()
                    .filter(|(_, name)| !name.is_synthesized())
                    .map(|(gid, name)| (gid.to_u32(), name.as_str().to_owned()))
                    .collect()
            }),
            Outlines::Cff => self.cff().map_or_else(Vec::new, |cff| {
                if cff.is_cid() {
                    return Vec::new();
                }
                let Some(charset) = cff.charset() else {
                    return Vec::new();
                };
                charset
                    .iter()
                    .filter_map(|(gid, sid)| {
                        let name = std::str::from_utf8(cff.string(sid)?).ok()?;
                        Some((gid.to_u32(), name.to_owned()))
                    })
                    .collect()
            }),
            Outlines::Type1(font) => font
                .glyph_names()
                .map(|(gid, name)| (gid.to_u32(), name.to_owned()))
                .collect(),
        }
    }

    /// Map a glyph selector from a composite font to a glyph index: for a
    /// CID-keyed CFF program the selector is a CID and resolves through the
    /// charset (CID 0 is `.notdef`); otherwise it is the glyph index.
    /// `None` when a CID has no glyph.
    #[must_use]
    pub fn cid_to_gid(&self, selector: u32) -> Option<u32> {
        match &self.cid_to_gid {
            Some(_) if selector == 0 => Some(0),
            Some(map) => map.get(&selector).copied(),
            None => Some(selector),
        }
    }

    /// Advance width of glyph `gid` in em units.  `None` when the glyph
    /// does not exist or the program records no advance for it.
    #[must_use]
    pub fn advance(&self, gid: u32) -> Option<f64> {
        let units = match &self.outlines {
            Outlines::Sfnt { .. } => {
                let font = self.sfnt()?;
                let hmtx = font.hmtx().ok()?;
                f64::from(hmtx.advance(GlyphId::new(gid))?)
            }
            Outlines::Cff | Outlines::Type1(_) => f64::from(self.draw(gid, &mut NullPen)??),
        };
        Some(units / self.upem)
    }

    /// Emit the unhinted outline of glyph `gid` to `pen` in font units
    /// (divide by [`Self::upem`] for em units, y up).
    ///
    /// Returns `None` when the glyph cannot be drawn (out of range, or a
    /// malformed charstring / glyph record), otherwise the advance width the
    /// charstring declares, in font units, when it declares one.
    pub fn draw(&self, gid: u32, pen: &mut impl OutlinePen) -> Option<Option<f32>> {
        let glyph = GlyphId::new(gid);
        match &self.outlines {
            Outlines::Sfnt { .. } => {
                let font = self.sfnt()?;
                let outline = font.outline_glyphs().get(glyph)?;
                let settings = DrawSettings::unhinted(Size::unscaled(), LocationRef::default());
                let metrics = outline.draw(settings, pen).ok()?;
                Some(metrics.advance_width)
            }
            Outlines::Cff => {
                let cff = self.cff()?;
                if gid >= cff.num_glyphs() {
                    return None;
                }
                let subfont = cff.subfont(cff.subfont_index(glyph)?, &[]).ok()?;
                cff.draw(&subfont, glyph, &[], None, pen).ok()
            }
            Outlines::Type1(font) => {
                if gid >= font.num_glyphs() {
                    return None;
                }
                font.draw(glyph, None, pen).ok()
            }
        }
    }

    /// Font units per em of the outlines [`Self::draw`] emits.
    #[must_use]
    pub const fn upem(&self) -> f64 {
        self.upem
    }

    fn sfnt(&self) -> Option<FontRef<'_>> {
        match self.outlines {
            Outlines::Sfnt { face_index, .. } => FontRef::from_index(&self.data, face_index).ok(),
            _ => None,
        }
    }

    fn cff(&self) -> Option<CffFontRef<'_>> {
        match self.outlines {
            Outlines::Cff => CffFontRef::new(&self.data, 0, None).ok(),
            _ => None,
        }
    }
}

/// CID → glyph index table of a CID-keyed CFF program, `None` for a
/// name-keyed one.
fn cid_charset_map(cff: &CffFontRef<'_>) -> Option<HashMap<u32, u32>> {
    if !cff.is_cid() {
        return None;
    }
    let charset = cff.charset()?;
    let mut map = HashMap::new();
    for (gid, cid) in charset.iter() {
        let _ = map
            .entry(u32::from(cid.to_u16()))
            .or_insert_with(|| gid.to_u32());
    }
    Some(map)
}

/// Choose the `cmap` subtable character codes resolve through.  When the
/// choice is [`SfntCmap::UnicodeFromNames`], the Unicode map built from the
/// `post` names comes back with it.
fn select_sfnt_cmap(font: &FontRef<'_>) -> (SfntCmap, Option<Charmap>) {
    let Ok(cmap) = font.cmap() else {
        return post_name_charmap(font).map_or((SfntCmap::None, None), |map| {
            (SfntCmap::UnicodeFromNames, Some(map))
        });
    };
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
    if let Some(i) = (0..records.len())
        .rev()
        .find(|&i| is_unicode(i) && is_ucs4(i))
    {
        return (SfntCmap::Unicode(i), None);
    }
    if let Some(i) = (0..records.len()).rev().find(|&i| is_unicode(i)) {
        return (SfntCmap::Unicode(i), None);
    }
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
    let pairs: Vec<(GlyphId, String)> = names
        .iter()
        .filter(|(_, name)| !name.is_synthesized())
        .map(|(gid, name)| (gid, name.as_str().to_owned()))
        .collect();
    let map = Charmap::from_glyph_names(pairs.iter().map(|(gid, name)| (*gid, name.as_str())));
    map.iter().next().is_some().then_some(map)
}

/// Pen that discards every command; used to evaluate charstring widths.
struct NullPen;

impl OutlinePen for NullPen {
    fn move_to(&mut self, _x: f32, _y: f32) {}
    fn line_to(&mut self, _x: f32, _y: f32) {}
    fn quad_to(&mut self, _cx0: f32, _cy0: f32, _x: f32, _y: f32) {}
    fn curve_to(&mut self, _cx0: f32, _cy0: f32, _cx1: f32, _cy1: f32, _x: f32, _y: f32) {}
    fn close(&mut self) {}
}
