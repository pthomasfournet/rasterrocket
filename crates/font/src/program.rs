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
//! | CID-keyed Type 1 (`CIDFontType0`) | `%!`, then a `CIDMap` and `StartData` section | charstrings per font dict, selected by CID |
//!
//! Outlines are unhinted, except for the TrueType programs whose glyphs
//! are assembled by their bytecode instructions
//! ([`FontProgram::requires_hinting`]): those are grid-fitted by the
//! TrueType interpreter at the size they are drawn at.

use std::collections::HashMap;
use std::sync::OnceLock;

use read_fonts::ps::cff::{CffFontRef, Subfont};
use read_fonts::ps::charmap::Charmap;
use read_fonts::ps::type1::Type1Font;
use read_fonts::ps::type1::cid::CidType1Font;
use read_fonts::tables::cmap::{CmapSubtable, PlatformId};
use read_fonts::types::{GlyphId, Tag};
use read_fonts::{FontRef, TableProvider};
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{
    DrawSettings, Engine, HintingInstance, HintingOptions, OutlineGlyphCollection,
    OutlineGlyphFormat, OutlinePen, Target,
};
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
    /// layout, name-keyed or CID-keyed.
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
    /// `hint_reliant` marks `glyf` outlines that are correct only after the
    /// glyph instructions run.
    Sfnt {
        face_index: u32,
        cmap: SfntCmap,
        hint_reliant: bool,
    },
    /// Bare CFF program.
    Cff,
    /// Type 1 program, decrypted and parsed once at load.
    Type1(Box<Type1Font>),
    /// CID-keyed Type 1 program (`CIDFontType0`), parsed and decrypted once
    /// at load.  Glyph indices are CIDs.
    CidType1(Box<CidType1Font>),
}

/// Views of the program bytes, parsed once at load so that drawing a
/// glyph reads only the glyph's own records.
#[expect(
    clippy::large_enum_variant,
    reason = "one per program, inside the self_cell's single heap allocation"
)]
enum Views<'a> {
    Sfnt {
        font: FontRef<'a>,
        outlines: OutlineGlyphCollection<'a>,
    },
    Cff(CffFontRef<'a>),
    /// Type 1 programs, which are parsed into owned structures.
    None,
}

self_cell::self_cell!(
    /// The program bytes and the [`Views`] that borrow them.
    struct ProgramData {
        owner: Box<[u8]>,
        #[covariant]
        dependent: Views,
    }
);

/// A parsed font program, shared by every [`crate::face::FontFace`] built
/// from the same bytes.
pub struct FontProgram {
    data: ProgramData,
    outlines: Outlines,
    /// The private dictionaries of a bare CFF program, indexed by font
    /// dictionary, each parsed on first use.  Empty for every other
    /// program.
    subfonts: Box<[OnceLock<Option<Subfont>>]>,
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
    /// Glyph indices of the codes 0–255 through the program's built-in
    /// encoding, 0 where it assigns none.  Built on first use.
    builtin_encoding: OnceLock<Box<[u32; 256]>>,
}

impl std::fmt::Debug for FontProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontProgram")
            .field("format", &self.format())
            .field("len", &self.data().len())
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
                let outline_glyphs = font.outline_glyphs();
                let hint_reliant = match outline_glyphs.format() {
                    None => {
                        return Err(LoadError::Malformed("sfnt has no glyph outlines".into()));
                    }
                    Some(OutlineGlyphFormat::Glyf) => outline_glyphs.require_interpreter(),
                    Some(_) => false,
                };
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
                    Outlines::Sfnt {
                        face_index,
                        cmap,
                        hint_reliant,
                    },
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
                let (outlines, upem) = parse_type1(&data)?;
                (outlines, f64::from(upem), None)
            }
        };
        if !(upem.is_finite() && upem > 0.0) {
            return Err(LoadError::Malformed(format!("units per em {upem}")));
        }
        let data = ProgramData::new(data, |data| match &outlines {
            Outlines::Sfnt { face_index, .. } => {
                FontRef::from_index(data, *face_index).map_or(Views::None, |font| Views::Sfnt {
                    outlines: font.outline_glyphs(),
                    font,
                })
            }
            Outlines::Cff => CffFontRef::new(data, 0, None).map_or(Views::None, Views::Cff),
            Outlines::Type1(_) | Outlines::CidType1(_) => Views::None,
        });
        let subfonts = match data.borrow_dependent() {
            Views::Cff(cff) => (0..cff.num_subfonts()).map(|_| OnceLock::new()).collect(),
            _ => Box::default(),
        };
        let mut program = Self {
            data,
            outlines,
            subfonts,
            upem,
            cid_to_gid,
            byte_charmap: Box::new([0; 256]),
            names: OnceLock::new(),
            name_charmap,
            builtin_encoding: OnceLock::new(),
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
            Outlines::Type1(_) | Outlines::CidType1(_) => FontFormat::Type1,
        }
    }

    /// Bytes of the program, exactly as given to [`Self::parse`].
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data.borrow_owner()
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

    /// Glyph index of character code `code` through the program's own
    /// built-in encoding (the `Encoding` of a Type 1 font dictionary or of
    /// a name-keyed CFF Top DICT), or `None` when the program has no
    /// built-in encoding or the encoding leaves `code` unassigned.  sfnt
    /// programs and CID-keyed programs (CFF or Type 1) carry no built-in
    /// encoding.
    #[must_use]
    pub fn builtin_encoding_index(&self, code: u8) -> Option<u32> {
        let table = self.builtin_encoding.get_or_init(|| {
            let mut table = Box::new([0; 256]);
            let mut fill = |map: &dyn Fn(u8) -> Option<GlyphId>| {
                for (code, slot) in (0..=u8::MAX).zip(table.iter_mut()) {
                    *slot = map(code).map_or(0, GlyphId::to_u32);
                }
            };
            match &self.outlines {
                Outlines::Sfnt { .. } | Outlines::CidType1(_) => {}
                Outlines::Cff => {
                    if let Some(cff) = self.cff().filter(|cff| !cff.is_cid())
                        && let Some(encoding) = cff.encoding()
                    {
                        fill(&|code| encoding.map(code));
                    }
                }
                Outlines::Type1(font) => {
                    if let Some(encoding) = font.encoding() {
                        fill(&|code| encoding.map(code));
                    }
                }
            }
            table
        });
        Some(table[usize::from(code)]).filter(|&gid| gid != 0)
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
            Outlines::CidType1(_) => Vec::new(),
        }
    }

    /// Map a glyph selector from a composite font to a glyph index: for a
    /// CID-keyed CFF program the selector is a CID and resolves through the
    /// charset (CID 0 is `.notdef`); otherwise it is the glyph index, which
    /// for a CID-keyed Type 1 program is the CID itself.  `None` when a CID
    /// has no glyph.
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
            Outlines::Cff | Outlines::Type1(_) | Outlines::CidType1(_) => {
                f64::from(self.draw(gid, &mut NullPen)??)
            }
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
                let outline = self.sfnt_outlines()?.get(glyph)?;
                let settings = DrawSettings::unhinted(Size::unscaled(), LocationRef::default());
                let metrics = outline.draw(settings, pen).ok()?;
                Some(metrics.advance_width)
            }
            Outlines::Cff => {
                let cff = self.cff()?;
                if gid >= cff.num_glyphs() {
                    return None;
                }
                let index = cff.subfont_index(glyph)?;
                let subfont = self
                    .subfonts
                    .get(usize::from(index))?
                    .get_or_init(|| cff.subfont(index, &[]).ok())
                    .as_ref()?;
                cff.draw(subfont, glyph, &[], None, pen).ok()
            }
            Outlines::Type1(font) => {
                if gid >= font.num_glyphs() {
                    return None;
                }
                font.draw(glyph, None, pen).ok()
            }
            // Out-of-range CIDs and CIDs without a glyph are errors.
            Outlines::CidType1(font) => font.draw(glyph, None, pen).ok(),
        }
    }

    /// `true` for a TrueType program whose glyphs take their shape from the
    /// glyph instructions rather than from the outline data alone, so that
    /// unhinted outlines are wrong (components overlap or sit off their
    /// strokes).  Such programs are recognised by family name and by the
    /// lengths and checksums of their `cvt `, `fpgm` and `prep` tables,
    /// which identify them after renaming or subsetting.
    #[must_use]
    pub const fn requires_hinting(&self) -> bool {
        matches!(
            self.outlines,
            Outlines::Sfnt {
                hint_reliant: true,
                ..
            }
        )
    }

    /// TrueType interpreter state for drawing at `ppem` pixels per em: the
    /// font program and control-value program have run, and the control
    /// values are scaled.  `None` when the program does not require hinting
    /// ([`Self::requires_hinting`]), when either program fails, or when it
    /// disables instructions at this size.
    ///
    /// The target is monochrome grid-fitting, under which the interpreter
    /// applies every instruction as written; the subpixel compatibility
    /// mode, which suppresses movement along the x axis, would leave the
    /// components unassembled.
    #[must_use]
    pub fn hinting_instance(&self, ppem: f32) -> Option<HintingInstance> {
        if !self.requires_hinting() {
            return None;
        }
        let outlines = self.sfnt_outlines()?;
        let options = HintingOptions {
            engine: Engine::Interpreter,
            target: Target::Mono,
        };
        let instance =
            HintingInstance::new(outlines, Size::new(ppem), LocationRef::default(), options)
                .ok()?;
        instance.is_enabled().then_some(instance)
    }

    /// Emit the outline of glyph `gid` grid-fitted by `instance`, in pixels
    /// at the instance's size (y up).  `None` when the glyph cannot be
    /// drawn.
    pub fn draw_hinted(
        &self,
        gid: u32,
        instance: &HintingInstance,
        pen: &mut impl OutlinePen,
    ) -> Option<()> {
        let outline = self.sfnt_outlines()?.get(GlyphId::new(gid))?;
        outline
            .draw(DrawSettings::hinted(instance, false), pen)
            .ok()
            .map(|_| ())
    }

    /// Font units per em of the outlines [`Self::draw`] emits.
    #[must_use]
    pub const fn upem(&self) -> f64 {
        self.upem
    }

    fn sfnt(&self) -> Option<&FontRef<'_>> {
        match self.data.borrow_dependent() {
            Views::Sfnt { font, .. } => Some(font),
            _ => None,
        }
    }

    fn sfnt_outlines(&self) -> Option<&OutlineGlyphCollection<'_>> {
        match self.data.borrow_dependent() {
            Views::Sfnt { outlines, .. } => Some(outlines),
            _ => None,
        }
    }

    fn cff(&self) -> Option<&CffFontRef<'_>> {
        match self.data.borrow_dependent() {
            Views::Cff(cff) => Some(cff),
            _ => None,
        }
    }
}

/// Parse a PostScript Type 1 program and return its outlines with their
/// units per em.  A program with the CID-keyed resource header is read as
/// CID-keyed; any other is read as name-keyed and, failing that, as
/// CID-keyed, which covers `CIDFont` programs written under a different
/// header comment.
fn parse_type1(data: &[u8]) -> Result<(Outlines, i32), LoadError> {
    let cid = |data: &[u8]| {
        CidType1Font::new(data).map(|font| {
            let upem = font.upem();
            (Outlines::CidType1(Box::new(font)), upem)
        })
    };
    if CidType1Font::is_cid_keyed(data) {
        return cid(data).map_err(|e| LoadError::Malformed(format!("CID-keyed Type 1: {e}")));
    }
    match Type1Font::new(data) {
        Ok(font) => {
            let upem = font.upem();
            Ok((Outlines::Type1(Box::new(font)), upem))
        }
        Err(e) => cid(data).map_err(|_| LoadError::Malformed(format!("Type 1: {e}"))),
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
