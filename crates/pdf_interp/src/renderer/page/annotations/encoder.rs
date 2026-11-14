//! The character codes that show Unicode text in a font, and the fallback
//! fonts that show what a font cannot encode.
//!
//! [`UnicodeEncoder`] inverts the code → Unicode relation of a font's
//! encoding: the glyph names of a simple font's codes (its base encoding
//! and `Differences`), and for a composite font its `ToUnicode` map or,
//! without one, its character collection's CID → Unicode table composed
//! with its `CMap`.  A character with several codes takes the lowest.
//!
//! The fallback fonts are non-embedded Type 0 fonts of the four Adobe
//! CJK character collections, encoded by their `UTF16-H` `CMaps`, and for
//! the characters these lack, the [`CharacterFonts`] of the appearance:
//! non-embedded simple fonts that each name up to 256 of its characters by
//! their glyph names (`uniXXXX`, `uXXXXX`).  They are drawn by the
//! substitute and fallback faces like any non-embedded font.  The
//! characters a text needs are known only once it is set, so their fonts
//! are added to the resources on demand:
//! [`FontTypesetter::missing_characters`] names those that were missing.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use pdf::{Dictionary, Object};

use crate::renderer::encodings::glyph_name_for_code;
use crate::renderer::fallback::first_scalar;
use crate::resources::cmap::CMap;
use crate::resources::cmap_predefined::{cid_to_unicode, predefined_cmap};
use crate::resources::font::FontDescriptor;

use super::super::PageRenderer;
use super::typeset::{Run, Typesetter};

/// The fallback fonts: resource name, `CIDSystemInfo` ordering and
/// encoding `CMap`.
const FALLBACKS: [(&[u8], &[u8], &[u8]); 4] = [
    (b"FallbackJapan1", b"Japan1", b"UniJIS-UTF16-H"),
    (b"FallbackGB1", b"GB1", b"UniGB-UTF16-H"),
    (b"FallbackCNS1", b"CNS1", b"UniCNS-UTF16-H"),
    (b"FallbackKorea1", b"Korea1", b"UniKS-UTF16-H"),
];

/// The codes that show each character in one font.
#[derive(Debug, Default)]
pub(super) struct UnicodeEncoder {
    codes: HashMap<char, Vec<u8>>,
}

impl UnicodeEncoder {
    /// The encoder of the font `desc`.
    pub(super) fn of(desc: &FontDescriptor) -> Self {
        let mut pairs: Vec<(u32, Vec<u8>)> = Vec::new();
        match &desc.cid_encoding {
            None => {
                for code in 0..=255u8 {
                    if let Some(u) =
                        glyph_name_for_code(&desc.differences, desc.base_encoding, code.into())
                            .and_then(font::glyph_name_to_unicode)
                    {
                        pairs.push((u, vec![code]));
                    }
                }
            }
            Some(enc) => {
                let cmap = &enc.encoding_cmap;
                if let Some(to_unicode) = &enc.to_unicode {
                    for (&code, &value) in &to_unicode.map {
                        if let (Some(u), Some(bytes)) =
                            (single_scalar(value), code_bytes(cmap, code))
                        {
                            pairs.push((u, bytes));
                        }
                    }
                } else if let Some(table) = desc.substitute.script.and_then(cid_to_unicode) {
                    let mut code_of_cid: HashMap<u32, u32> = HashMap::new();
                    if !cmap.identity {
                        for (&code, &cid) in &cmap.map {
                            let slot = code_of_cid.entry(cid).or_insert(code);
                            *slot = (*slot).min(code);
                        }
                    }
                    for (cid, unicodes) in table.iter() {
                        let code = if cmap.identity {
                            Some(cid)
                        } else {
                            code_of_cid.get(&cid).copied()
                        };
                        if let Some(bytes) = code.and_then(|c| code_bytes(cmap, c)) {
                            pairs.extend(unicodes.iter().map(|&u| (u, bytes.clone())));
                        }
                    }
                }
            }
        }
        let mut codes: HashMap<char, Vec<u8>> = HashMap::new();
        for (u, bytes) in pairs {
            let Some(c) = char::from_u32(u) else {
                continue;
            };
            let slot = codes.entry(c).or_insert_with(|| bytes.clone());
            if (bytes.len(), &bytes) < (slot.len(), &*slot) {
                *slot = bytes;
            }
        }
        Self { codes }
    }

    /// The codes of `c`, or `None` when the font cannot show it.
    pub(super) fn encode(&self, c: char) -> Option<&[u8]> {
        self.codes.get(&c).map(Vec::as_slice)
    }
}

/// The fallback fonts in the order they are tried for one text.
#[derive(Clone, Copy, Debug)]
pub(super) struct Fallbacks {
    /// Indices into [`FALLBACKS`].
    order: [usize; FALLBACKS.len()],
}

impl Fallbacks {
    /// The order for `text`: Japanese first when it holds kana, Korean
    /// first when it holds Hangul, else Simplified Chinese first; the other
    /// collections follow in [`FALLBACKS`] order.
    pub(super) fn for_text(text: &str) -> Self {
        use crate::resources::font_substitute::Script;
        let script = text
            .chars()
            .find_map(|ch| match Script::of_char(u32::from(ch)) {
                Some(s @ (Script::Japanese | Script::Korean)) => Some(s),
                _ => None,
            });
        let first = match script {
            Some(Script::Japanese) => 0,
            Some(Script::Korean) => 3,
            _ => 1,
        };
        let mut order = std::array::from_fn(|i| i);
        order[..=first].rotate_right(1);
        Self { order }
    }

    /// The first fallback that shows `c`: for an East Asian character, a
    /// CJK collection font that maps it to a glyph, else the appearance's
    /// [`CharacterFonts`].  Other characters the collections also hold,
    /// such as Greek, Cyrillic and symbols, are set in the character fonts,
    /// whose glyphs are proportional rather than the collections' full
    /// width.  `None` for a private-use character or a noncharacter, which
    /// no face is known to show.
    pub(super) fn find(self, c: char) -> Option<Fallback> {
        let mut units = [0u16; 2];
        let units = c.encode_utf16(&mut units);
        let key = units.iter().fold(0u32, |k, &u| (k << 16) | u32::from(u));
        let collection = east_asian(c)
            .then(|| {
                self.order
                    .iter()
                    .map(|&i| FALLBACKS[i])
                    .find(|(_, _, cmap)| {
                        predefined_cmap(cmap)
                            .is_some_and(|cmap| cmap.map.get(&key).is_some_and(|&cid| cid != 0))
                    })
            })
            .flatten();
        if let Some((name, _, _)) = collection {
            let codes = units.iter().flat_map(|u| u.to_be_bytes()).collect();
            return Some(Fallback::Collection { font: name, codes });
        }
        let scalar = u32::from(c);
        let private_or_non = matches!(scalar, 0xE000..=0xF8FF | 0xF_0000.. | 0xFDD0..=0xFDEF)
            || scalar & 0xFFFE == 0xFFFE;
        (!private_or_non).then_some(Fallback::Character)
    }
}

/// A fallback that shows a character (see [`Fallbacks::find`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Fallback {
    /// The CJK collection font `font`, with the character's codes there.
    Collection { font: &'static [u8], codes: Vec<u8> },
    /// The appearance's [`CharacterFonts`].
    Character,
}

/// Characters one font of [`CharacterFonts`] holds.
const FONT_CHARACTERS: usize = 256;

/// Most characters the [`CharacterFonts`] of one appearance hold; a
/// further character is set as `?`.
const MAX_CHARACTERS: usize = 64 * FONT_CHARACTERS;

/// The fonts that show the characters of one appearance that neither its
/// fonts nor a collection font show: standard Helvetica re-encoded by
/// `Differences`, code `k` of font `n` being the glyph named for the
/// character `256 × n + k` of the set (`uniXXXX`, `uXXXXX`), the set in
/// scalar order.  Each font is named by a hash of its characters, so a
/// name always stands for one encoding.
///
/// An appearance needing at most 256 such characters sets them all in
/// one font, so a word whose characters lie in different Unicode blocks,
/// such as an Arabic word with letters of the Arabic Supplement or a
/// joiner, is one show string, shaped as one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct CharacterFonts {
    /// The characters, in scalar order, at most [`MAX_CHARACTERS`].
    chars: Vec<char>,
}

impl CharacterFonts {
    /// Add `chars` to the set, up to [`MAX_CHARACTERS`] in all.
    pub(super) fn extend(&mut self, chars: impl IntoIterator<Item = char>) {
        self.chars.extend(chars);
        self.chars.sort_unstable();
        self.chars.dedup();
        self.chars.truncate(MAX_CHARACTERS);
    }

    /// The resource name of the font of `c` and its code there; `None`
    /// when the set lacks `c`.
    fn font_and_code(&self, c: char) -> Option<(Vec<u8>, u8)> {
        let k = self.chars.binary_search(&c).ok()?;
        let font = &self.chars[k - k % FONT_CHARACTERS..];
        let font = &font[..font.len().min(FONT_CHARACTERS)];
        Some((
            character_font_name(font),
            u8::try_from(k % FONT_CHARACTERS).ok()?,
        ))
    }

    /// Add the fonts to the font resource dictionary `fonts`.
    pub(super) fn add_to(&self, fonts: &mut Dictionary) {
        for chars in self.chars.chunks(FONT_CHARACTERS) {
            let mut differences = vec![Object::Integer(0)];
            differences.extend(chars.iter().map(|&c| {
                let scalar = u32::from(c);
                let name = if scalar <= 0xFFFF {
                    format!("uni{scalar:04X}")
                } else {
                    format!("u{scalar:X}")
                };
                Object::Name(name.into_bytes())
            }));
            let mut encoding = Dictionary::new();
            encoding.set(b"Type", Object::Name(b"Encoding".to_vec()));
            encoding.set(b"Differences", Object::Array(differences));
            let mut font = Dictionary::new();
            font.set(b"Type", Object::Name(b"Font".to_vec()));
            font.set(b"Subtype", Object::Name(b"Type1".to_vec()));
            font.set(b"BaseFont", Object::Name(b"Helvetica".to_vec()));
            font.set(b"Encoding", Object::Dictionary(encoding));
            fonts.set(character_font_name(chars), Object::Dictionary(font));
        }
    }
}

/// The resource name of the character font holding `chars`: `FallbackC`
/// and the 64-bit FNV-1a hash of their scalar values in hexadecimal.
fn character_font_name(chars: &[char]) -> Vec<u8> {
    let hash = chars
        .iter()
        .flat_map(|&c| u32::from(c).to_le_bytes())
        .fold(0xCBF2_9CE4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
        });
    format!("FallbackC{hash:016X}").into_bytes()
}

/// Whether `c` belongs to an East Asian script or the symbols and forms
/// that accompany it: Han, kana, Hangul, Bopomofo, CJK punctuation,
/// enclosed and compatibility characters, and half- and full-width forms.
const fn east_asian(c: char) -> bool {
    matches!(
        c as u32,
        0x1100..=0x11FF
            | 0x2E80..=0x2FDF
            | 0x3000..=0x33FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7FF
            | 0xF900..=0xFAFF
            | 0xFE30..=0xFE4F
            | 0xFF00..=0xFFEF
            | 0x2_0000..=0x3_FFFF
    )
}

/// Add the fallback fonts to the font resource dictionary `fonts`.
pub(super) fn add_fallback_fonts(fonts: &mut Dictionary) {
    for (name, ordering, cmap) in FALLBACKS {
        let mut info = Dictionary::new();
        info.set(
            b"Registry",
            Object::String(b"Adobe".to_vec(), pdf::StringFormat::Literal),
        );
        info.set(
            b"Ordering",
            Object::String(ordering.to_vec(), pdf::StringFormat::Literal),
        );
        info.set(b"Supplement", Object::Integer(0));
        let mut cid_font = Dictionary::new();
        cid_font.set(b"Type", Object::Name(b"Font".to_vec()));
        cid_font.set(b"Subtype", Object::Name(b"CIDFontType0".to_vec()));
        cid_font.set(b"BaseFont", Object::Name(name.to_vec()));
        cid_font.set(b"CIDSystemInfo", Object::Dictionary(info));
        let mut font = Dictionary::new();
        font.set(b"Type", Object::Name(b"Font".to_vec()));
        font.set(b"Subtype", Object::Name(b"Type0".to_vec()));
        font.set(b"BaseFont", Object::Name(name.to_vec()));
        font.set(b"Encoding", Object::Name(cmap.to_vec()));
        font.set(
            b"DescendantFonts",
            Object::Array(vec![Object::Dictionary(cid_font)]),
        );
        fonts.set(name, Object::Dictionary(font));
    }
}

/// A [`Typesetter`] over the fonts of the resources in effect in a page
/// renderer.  A character the requested font cannot encode is set in a
/// fallback font; one no fallback font covers is set as the requested
/// font's `?`.
pub(super) struct FontTypesetter<'r, 'p> {
    renderer: &'r mut PageRenderer<'p>,
    /// Encoders by font resource name.
    encoders: HashMap<Vec<u8>, UnicodeEncoder>,
    /// The character fonts in the resources.
    characters: &'r CharacterFonts,
    /// The characters set in a character font that `characters` lacks.
    missing: BTreeSet<char>,
    /// The characters set as `?`: those no font shows, and those beyond
    /// the [`MAX_CHARACTERS`] of the character fonts.
    unset: Unset,
    /// The state of inverting each font's embedded program.
    inversions: HashMap<Vec<u8>, Inversion>,
}

/// The state of inverting one font's embedded program (see
/// [`FontTypesetter::invert_program`]).
#[derive(Default)]
struct Inversion {
    /// Whether the inversion applies to the font; `None` until checked.
    applies: Option<bool>,
    /// The characters already looked up in the program.
    tried: std::collections::HashSet<char>,
    /// The inverse of the font's `CIDToGIDMap`, built when first needed.
    cid_of_gid: Option<HashMap<u32, u32>>,
}

impl<'r, 'p> FontTypesetter<'r, 'p> {
    /// A typesetter over the resources in effect in `renderer`, which hold
    /// the fonts of `characters`.
    pub(super) fn new(renderer: &'r mut PageRenderer<'p>, characters: &'r CharacterFonts) -> Self {
        Self {
            renderer,
            encoders: HashMap::new(),
            characters,
            missing: BTreeSet::new(),
            unset: Unset::default(),
            inversions: HashMap::new(),
        }
    }

    /// The characters set in a character font that the typesetter's
    /// [`CharacterFonts`] lack, each set as `?`; text set with any must be
    /// set again with them added to the set and its fonts to the
    /// resources.  When none are missing, the text is set as drawn, and
    /// the characters it sets as `?` are logged (see [`Unset::report`]).
    pub(super) fn missing_characters(self) -> BTreeSet<char> {
        if self.missing.is_empty() {
            self.unset.report();
        }
        self.missing
    }

    /// The encoder of the font resource `font`, holding the codes of the
    /// characters of `text` its encoding does not invert but its embedded
    /// program does (see [`Self::invert_program`]).
    fn encoder(&mut self, font: &[u8], text: &str) -> &UnicodeEncoder {
        if !self.encoders.contains_key(font) {
            let renderer = &mut *self.renderer;
            let active = renderer.font_cache.select(&renderer.resources, font);
            let _ = self
                .encoders
                .insert(font.to_vec(), UnicodeEncoder::of(&active.descriptor));
        }
        self.invert_program(font, text);
        &self.encoders[font]
    }

    /// Add to the encoder of `font` the codes of the characters of `text`
    /// it lacks, found through the font's embedded program, when `font` is
    /// a composite font with the identity `CMap`, no `ToUnicode` map and no
    /// character collection, whose codes therefore name glyphs alone: the
    /// program's Unicode charmap gives a character's glyph, and the
    /// inverse of `CIDToGIDMap` (the identity when absent) its CID, which
    /// is its code.  A glyph the program reaches only through its own CID
    /// mapping, as in a CID-keyed CFF program, is not inverted.
    fn invert_program(&mut self, font: &[u8], text: &str) {
        let inversion = self.inversions.entry(font.to_vec()).or_default();
        let Some(encoder) = self.encoders.get_mut(font) else {
            return;
        };
        if inversion.applies == Some(false) {
            return;
        }
        let wanted: Vec<char> = text
            .chars()
            .filter(|&c| {
                !c.is_control() && encoder.encode(c).is_none() && inversion.tried.insert(c)
            })
            .collect();
        if wanted.is_empty() {
            return;
        }
        let renderer = &mut *self.renderer;
        let active = renderer.font_cache.select(&renderer.resources, font);
        let descriptor = Arc::clone(&active.descriptor);
        let enc = descriptor.cid_encoding.as_ref().filter(|enc| {
            enc.encoding_cmap.identity
                && enc.to_unicode.is_none()
                && descriptor.substitute.script.is_none()
        });
        let face = enc.and_then(|_| {
            renderer
                .font_cache
                .get_or_load(&active, font, [12.0, 0.0, 0.0, 12.0])
                .filter(|face| !face.substitute && face.has_unicode_cmap())
        });
        let (Some(enc), Some(face)) = (enc, face) else {
            inversion.applies = Some(false);
            return;
        };
        inversion.applies = Some(true);
        for c in wanted {
            let gid = face.raw_get_char_index(u32::from(c));
            if gid == 0 || face.program().cid_to_gid(gid) != Some(gid) {
                continue;
            }
            let cid = enc.cid_to_gid.as_ref().map_or(Some(gid), |table| {
                inversion
                    .cid_of_gid
                    .get_or_insert_with(|| {
                        let mut inverse = HashMap::new();
                        for (cid, &g) in (0u32..).zip(table.iter()).skip(1) {
                            let _ = inverse.entry(g).or_insert(cid);
                        }
                        inverse
                    })
                    .get(&gid)
                    .copied()
            });
            if let Some(bytes) = cid.and_then(|cid| code_bytes(&enc.encoding_cmap, cid)) {
                let _ = encoder.codes.insert(c, bytes);
            }
        }
    }
}

impl Typesetter for FontTypesetter<'_, '_> {
    fn runs(&mut self, font: &[u8], text: &str) -> Vec<Run> {
        let fallbacks = Fallbacks::for_text(text);
        let characters = self.characters;
        let mut missing = std::mem::take(&mut self.missing);
        let mut unset = std::mem::take(&mut self.unset);
        let encoder = self.encoder(font, text);
        let question = encoder.encode('?').unwrap_or(b"?");
        let mut runs: Vec<Run> = Vec::new();
        for c in text.chars() {
            let fallback;
            let found = if c.is_control() {
                None
            } else {
                fallbacks.find(c)
            };
            let (run_font, codes): (&[u8], &[u8]) = if let Some(codes) = encoder.encode(c) {
                (font, codes)
            } else {
                match found {
                    Some(Fallback::Collection { font, codes }) => {
                        fallback = (font.to_vec(), codes);
                        (&fallback.0, &fallback.1)
                    }
                    Some(Fallback::Character) => {
                        if let Some((name, code)) = characters.font_and_code(c) {
                            fallback = (name, vec![code]);
                            (&fallback.0, &fallback.1)
                        } else {
                            if characters.chars.len() + missing.len() < MAX_CHARACTERS {
                                let _ = missing.insert(c);
                            } else {
                                let _ = unset.capped.insert(c);
                            }
                            (font, question)
                        }
                    }
                    None => {
                        let _ = unset.uncovered.insert(c);
                        (font, question)
                    }
                }
            };
            match runs.last_mut() {
                Some(run) if run.font == run_font => run.codes.extend_from_slice(codes),
                _ => runs.push(Run {
                    font: run_font.to_vec(),
                    codes: codes.to_vec(),
                }),
            }
        }
        self.missing = missing;
        self.unset = unset;
        runs
    }

    /// The font's `/Widths` (or `/W` for a composite font), else its
    /// program's advances.  A font the resources lack is not selected,
    /// since the font cache would keep the default font under its name;
    /// its codes advance half an em, for a layout that is set again.
    fn advance(&mut self, font: &[u8], codes: &[u8]) -> f64 {
        let renderer = &mut *self.renderer;
        if renderer.resources.font_entry(font).is_none() {
            return f64::from(u32::try_from(codes.len()).unwrap_or(u32::MAX)) / 2.0;
        }
        let active = renderer.font_cache.select(&renderer.resources, font);
        if let Some(enc) = &active.descriptor.cid_encoding {
            return enc
                .codes(codes)
                .map(|code| f64::from(enc.width_for_cid(code.value)) / 1000.0)
                .sum();
        }
        let mut total = 0.0;
        for &byte in codes {
            let code = u32::from(byte);
            total += match active.descriptor.width_for_code(code) {
                Some(w) => f64::from(w) / 1000.0,
                None => renderer
                    .font_cache
                    .get_or_load(&active, font, [12.0, 0.0, 0.0, 12.0])
                    .and_then(|face| face.glyph_advance(code))
                    .unwrap_or(0.5),
            };
        }
        total
    }
}

/// The characters an appearance's text sets as its font's `?`.
#[derive(Debug, Default)]
struct Unset {
    /// Characters no font shows: controls, private-use characters and
    /// noncharacters, and those the collection fonts omit.
    uncovered: BTreeSet<char>,
    /// Characters a character font would show, beyond the
    /// [`MAX_CHARACTERS`] the character fonts hold.
    capped: BTreeSet<char>,
}

impl Unset {
    /// Log the characters: at `warn` those beyond the cap, which is valid
    /// text dropped, at `debug` those no font shows.
    fn report(&self) {
        if let Some(&first) = self.capped.first() {
            log::warn!(
                "rasterrocket-interp: an appearance needs more than {MAX_CHARACTERS} characters \
                 its fonts lack; {} more (U+{:04X} first) are set as `?`",
                self.capped.len(),
                u32::from(first)
            );
        }
        if let Some(&first) = self.uncovered.first() {
            log::debug!(
                "rasterrocket-interp: no font shows {} characters of an appearance \
                 (U+{:04X} first); they are set as `?`",
                self.uncovered.len(),
                u32::from(first)
            );
        }
    }
}

/// The Unicode scalar value of a `ToUnicode` value that maps its code to
/// one character; `None` for a value of several characters, such as a
/// ligature's, whose code does not show its first character alone.
fn single_scalar(value: u32) -> Option<u32> {
    let scalar = first_scalar(value)?;
    // Above U+10FFFF the value packs two UTF-16 units; they are one
    // character only as a surrogate pair, which decodes above U+FFFF.
    (value <= 0x10_FFFF || scalar > 0xFFFF).then_some(scalar)
}

/// The bytes of `code` in a codespace range of `cmap` of its length; the
/// code as two bytes for an identity `CMap`.
fn code_bytes(cmap: &CMap, code: u32) -> Option<Vec<u8>> {
    (1..=4usize).find_map(|len| {
        if len < 4 && code >> (8 * len) != 0 {
            return None;
        }
        let bytes = &code.to_be_bytes()[4 - len..];
        cmap.codespaces
            .iter()
            .any(|r| {
                usize::from(r.bytes) == len
                    && bytes
                        .iter()
                        .zip(&r.lo)
                        .zip(&r.hi)
                        .all(|((b, lo), hi)| lo <= b && b <= hi)
            })
            .then(|| bytes.to_vec())
    })
}

#[cfg(test)]
mod tests {
    use super::super::typeset::Typesetter;
    use super::{
        CharacterFonts, Fallback, Fallbacks, FontTypesetter, MAX_CHARACTERS, UnicodeEncoder,
        code_bytes, single_scalar,
    };
    use crate::renderer::PageRenderer;
    use crate::resources::cmap::CMap;
    use crate::resources::cmap_predefined::predefined_cmap;
    use crate::resources::font::resolve_font;
    use crate::test_helpers::{make_doc_with_object, make_doc_with_objects};
    use pdf::{Dictionary, Object};

    /// The encoder of the font dictionary `font`.
    fn encoder(font: &str) -> UnicodeEncoder {
        let doc = make_doc_with_object(font);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        UnicodeEncoder::of(&resolve_font(&doc, &dict, Some((2, 0))))
    }

    /// The characters a typesetter over a page whose `/F1` is Helvetica,
    /// with the character fonts `characters`, sets as `?` for want of a
    /// font and beyond the character fonts' cap, setting `text`.
    fn unset(characters: &CharacterFonts, text: &str) -> (Vec<char>, Vec<char>) {
        let doc = make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /Font << /F1 5 0 R >> >> >>",
            ),
            (5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        ]);
        let mut renderer = PageRenderer::new(64, 32, &doc, (4, 0));
        let mut ts = FontTypesetter::new(&mut renderer, characters);
        let _ = ts.runs(b"F1", text);
        (
            ts.unset.uncovered.iter().copied().collect(),
            ts.unset.capped.iter().copied().collect(),
        )
    }

    #[test]
    fn characters_set_as_question_marks_are_kept_for_the_report() {
        let none = CharacterFonts::default();
        assert_eq!(unset(&none, "A\u{E000}B"), (vec!['\u{E000}'], vec![]));
        let mut full = CharacterFonts::default();
        full.extend((0x1000..).filter_map(char::from_u32).take(MAX_CHARACTERS));
        assert_eq!(unset(&full, "A\u{0628}"), (vec![], vec!['\u{0628}']));
    }

    #[test]
    fn a_simple_font_encodes_through_its_glyph_names() {
        let e = encoder(
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
             /Encoding << /BaseEncoding /WinAnsiEncoding /Differences [65 /uni65E5] >> >>",
        );
        assert_eq!(e.encode('é'), Some(&[0xE9][..]));
        assert_eq!(e.encode('\u{2014}'), Some(&[0x97][..]));
        assert_eq!(e.encode('日'), Some(&[65][..]), "through /Differences");
        assert_eq!(e.encode('A'), None, "its code shows another glyph");
        assert_eq!(e.encode('中'), None);
    }

    #[test]
    fn a_composite_font_encodes_through_its_collection_and_cmap() {
        let e = encoder(
            "<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiKakuGo-W5 \
             /Encoding /UniJIS-UCS2-H /DescendantFonts [<< /Type /Font \
             /Subtype /CIDFontType0 /BaseFont /HeiseiKakuGo-W5 /CIDSystemInfo \
             << /Registry (Adobe) /Ordering (Japan1) /Supplement 5 >> >>] >>",
        );
        // A code of the CID of each character; U+65E5 shares its CID with
        // KANGXI RADICAL SUN, whose lower code is chosen.
        let cmap = predefined_cmap(b"UniJIS-UCS2-H").expect("a predefined CMap");
        for c in ['日', 'の', 'A'] {
            let code = e.encode(c).expect("in Adobe-Japan1");
            let code = code.iter().fold(0u32, |v, &b| (v << 8) | u32::from(b));
            assert_eq!(cmap.lookup(code), cmap.lookup(u32::from(c)), "{c}");
        }
        assert_eq!(e.encode('한'), None, "Hangul is not in Adobe-Japan1");
    }

    #[test]
    fn a_ligature_code_does_not_encode_its_first_character() {
        let to_unicode = "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
                          3 beginbfchar\n<0001> <00660069>\n<0002> <0066>\n\
                          <0003> <D83DDE00>\nendbfchar\n";
        let doc = make_doc_with_objects(&[
            (
                4,
                "<< /Type /Font /Subtype /Type0 /BaseFont /F /Encoding /Identity-H \
                 /ToUnicode 5 0 R /DescendantFonts [<< /Type /Font /Subtype \
                 /CIDFontType2 /BaseFont /F /CIDSystemInfo << /Registry (Adobe) \
                 /Ordering (Identity) /Supplement 0 >> >>] >>"
                    .to_owned(),
            ),
            (
                5,
                format!(
                    "<< /Length {} >>\nstream\n{to_unicode}\nendstream",
                    to_unicode.len()
                ),
            ),
        ]);
        let dict = doc.get_dictionary((4, 0)).expect("a dictionary");
        let e = UnicodeEncoder::of(&resolve_font(&doc, &dict, Some((4, 0))));
        assert_eq!(e.encode('f'), Some(&[0, 2][..]), "not the fi ligature");
        assert_eq!(e.encode('\u{1F600}'), Some(&[0, 3][..]), "a surrogate pair");
        assert_eq!(single_scalar(0x0066_0069), None);
    }

    #[test]
    fn codes_take_the_length_of_their_codespace() {
        assert_eq!(code_bytes(&CMap::identity(), 0x41), Some(vec![0, 0x41]));
        assert_eq!(
            code_bytes(&CMap::identity(), 0x4E2D),
            Some(vec![0x4E, 0x2D])
        );
    }

    #[test]
    fn kana_select_the_japanese_fallback_and_han_alone_the_chinese() {
        let fallback = |c, text| Fallbacks::for_text(text).find(c).expect("a fallback");
        let font = |c, text| match fallback(c, text) {
            Fallback::Collection { font, .. } => font,
            Fallback::Character => panic!("{c:?} is in a collection"),
        };
        assert_eq!(
            fallback('日', "日本語の"),
            Fallback::Collection {
                font: b"FallbackJapan1",
                codes: vec![0x65, 0xE5]
            }
        );
        assert_eq!(font('日', "日本"), b"FallbackGB1");
        assert_eq!(font('한', "한국"), b"FallbackKorea1");
        // Arabic is in no collection.
        assert_eq!(fallback('\u{0627}', ""), Fallback::Character);
        // Private-use characters and noncharacters have no fallback.
        for c in ['\u{E000}', '\u{F0000}', '\u{FDD0}', '\u{FFFF}', '\u{1FFFE}'] {
            assert!(Fallbacks::for_text("").find(c).is_none(), "{c:?}");
        }
        assert_eq!(Fallbacks::for_text("한").order, [3, 0, 1, 2]);
        assert_eq!(Fallbacks::for_text("").order, [1, 0, 2, 3]);
    }

    /// Arabic letters of the main and supplement blocks and a joiner share
    /// one font, coded by their rank in scalar order; a font's name stands
    /// for its characters.
    #[test]
    fn the_characters_of_an_appearance_share_one_font() {
        let mut fonts = CharacterFonts::default();
        fonts.extend(['\u{0628}', '\u{200C}', '\u{0750}', '\u{08A0}', '\u{0628}']);
        let coded: Vec<_> = ['\u{0628}', '\u{0750}', '\u{08A0}', '\u{200C}']
            .map(|c| fonts.font_and_code(c).expect("in the set"))
            .into_iter()
            .collect();
        let name = &coded[0].0;
        assert!(coded.iter().all(|(font, _)| font == name));
        assert_eq!(
            coded.iter().map(|&(_, code)| code).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(fonts.font_and_code('\u{0627}'), None);

        let mut dict = Dictionary::new();
        fonts.add_to(&mut dict);
        let Some(Object::Dictionary(font)) = dict.get(name) else {
            panic!("the font is added under its name");
        };
        let Some(Object::Dictionary(encoding)) = font.get(b"Encoding") else {
            panic!("an encoding");
        };
        let Some(Object::Array(differences)) = encoding.get(b"Differences") else {
            panic!("differences");
        };
        let names: Vec<&[u8]> = differences[1..]
            .iter()
            .filter_map(|o| match o {
                Object::Name(n) => Some(n.as_slice()),
                _ => None,
            })
            .collect();
        assert_eq!(names, [&b"uni0628"[..], b"uni0750", b"uni08A0", b"uni200C"]);

        let mut more = fonts.clone();
        more.extend(['\u{0627}']);
        assert_ne!(more.font_and_code('\u{0628}').expect("in the set").0, *name);
    }

    /// Past 256 characters the set takes a second font.
    #[test]
    fn a_font_holds_256_characters() {
        let mut fonts = CharacterFonts::default();
        fonts.extend((0x0600..=0x0700).filter_map(char::from_u32));
        let first = fonts.font_and_code('\u{0600}').expect("in the set");
        let last = fonts.font_and_code('\u{06FF}').expect("in the set");
        let next = fonts.font_and_code('\u{0700}').expect("in the set");
        assert_eq!((first.0 == last.0, first.1, last.1), (true, 0, 255));
        assert_eq!(next.1, 0);
        assert_ne!(next.0, first.0);
    }
}
