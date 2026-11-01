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
//! the characters these lack, a non-embedded simple font per block of 256
//! Unicode scalar values whose code `c` is the glyph named for the block's
//! scalar value `c` (`uniXXXX`, `uXXXXX`).  They are drawn by the
//! substitute and fallback faces like any non-embedded font.  The block
//! fonts a text needs are known only once it is set, so they are added to
//! the resources on demand: [`FontTypesetter::missing_blocks`] names those
//! that were missing.

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

    /// The first fallback font that shows `c`, its codes there, and the
    /// Unicode block when it is a block font: for an East Asian character,
    /// a CJK collection font that maps it to a glyph, else the block font
    /// of `c`.  Other characters the collections also hold, such as Greek,
    /// Cyrillic and symbols, are set in their block font, whose glyphs are
    /// proportional rather than the collections' full width.  `None` for a
    /// private-use character or a noncharacter, which no face is known to
    /// show.
    pub(super) fn find(self, c: char) -> Option<(Vec<u8>, Vec<u8>, Option<u32>)> {
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
            return Some((name.to_vec(), codes, None));
        }
        let scalar = u32::from(c);
        let private_or_non = matches!(scalar, 0xE000..=0xF8FF | 0xF_0000.. | 0xFDD0..=0xFDEF)
            || scalar & 0xFFFE == 0xFFFE;
        if private_or_non {
            return None;
        }
        let [.., low] = scalar.to_be_bytes();
        Some((block_font_name(scalar >> 8), vec![low], Some(scalar >> 8)))
    }
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

/// The resource name of the fallback font of the Unicode block `block`,
/// the scalar values `block << 8` to `block << 8 | 0xFF`.
fn block_font_name(block: u32) -> Vec<u8> {
    format!("FallbackU{block:04X}").into_bytes()
}

/// Add the fallback fonts of the Unicode blocks `blocks` to the font
/// resource dictionary `fonts`: standard Helvetica re-encoded by
/// `Differences` so that code `c` is the glyph named for the block's
/// scalar value `c`.
pub(super) fn add_block_fonts(fonts: &mut Dictionary, blocks: impl IntoIterator<Item = u32>) {
    for block in blocks {
        let mut differences = vec![Object::Integer(0)];
        differences.extend((0..=0xFFu32).map(|low| {
            let scalar = block << 8 | low;
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
        fonts.set(block_font_name(block), Object::Dictionary(font));
    }
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
    /// The Unicode blocks whose fallback font text was set in, at most
    /// [`MAX_BLOCK_FONTS`].
    blocks: BTreeSet<u32>,
    /// Those of [`Self::blocks`] the resources lack.
    missing: BTreeSet<u32>,
    /// The state of inverting each font's embedded program.
    inversions: HashMap<Vec<u8>, Inversion>,
}

/// Most Unicode block fonts the text of one appearance is set in; a
/// character of a further block is set as `?`.
const MAX_BLOCK_FONTS: usize = 64;

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
    /// A typesetter over the resources in effect in `renderer`.
    pub(super) fn new(renderer: &'r mut PageRenderer<'p>) -> Self {
        Self {
            renderer,
            encoders: HashMap::new(),
            blocks: BTreeSet::new(),
            missing: BTreeSet::new(),
            inversions: HashMap::new(),
        }
    }

    /// The Unicode blocks whose fallback font text was set in but the
    /// resources lack; text set with any of them must be set again with
    /// their fonts added by [`add_block_fonts`].
    pub(super) fn missing_blocks(self) -> BTreeSet<u32> {
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
        let mut blocks = std::mem::take(&mut self.blocks);
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
            } else if let Some(found) = found.filter(|(.., block)| {
                block.is_none_or(|b| blocks.contains(&b) || blocks.len() < MAX_BLOCK_FONTS)
            }) {
                fallback = found;
                blocks.extend(fallback.2);
                (&fallback.0, &fallback.1)
            } else {
                (font, question)
            };
            match runs.last_mut() {
                Some(run) if run.font == run_font => run.codes.extend_from_slice(codes),
                _ => runs.push(Run {
                    font: run_font.to_vec(),
                    codes: codes.to_vec(),
                }),
            }
        }
        self.blocks = blocks;
        let resources = &self.renderer.resources;
        self.missing.extend(
            self.blocks
                .iter()
                .filter(|&&b| resources.font_entry(&block_font_name(b)).is_none()),
        );
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
    use super::{Fallbacks, UnicodeEncoder, code_bytes, single_scalar};
    use crate::resources::cmap::CMap;
    use crate::resources::cmap_predefined::predefined_cmap;
    use crate::resources::font::resolve_font;
    use crate::test_helpers::{make_doc_with_object, make_doc_with_objects};

    /// The encoder of the font dictionary `font`.
    fn encoder(font: &str) -> UnicodeEncoder {
        let doc = make_doc_with_object(font);
        let dict = doc.get_dictionary((2, 0)).expect("a dictionary");
        UnicodeEncoder::of(&resolve_font(&doc, &dict, Some((2, 0))))
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
        let (font, codes, block) = fallback('日', "日本語の");
        assert_eq!(
            (font.as_slice(), codes.as_slice(), block),
            (&b"FallbackJapan1"[..], &[0x65, 0xE5][..], None)
        );
        assert_eq!(fallback('日', "日本").0, b"FallbackGB1");
        assert_eq!(fallback('한', "한국").0, b"FallbackKorea1");
        // Arabic is in no collection: its block font, coded by the low byte.
        let (font, codes, block) = fallback('\u{0627}', "");
        assert_eq!(
            (font.as_slice(), codes.as_slice(), block),
            (&b"FallbackU0006"[..], &[0x27][..], Some(6))
        );
        // Private-use characters and noncharacters have no fallback.
        for c in ['\u{E000}', '\u{F0000}', '\u{FDD0}', '\u{FFFF}', '\u{1FFFE}'] {
            assert!(Fallbacks::for_text("").find(c).is_none(), "{c:?}");
        }
        assert_eq!(Fallbacks::for_text("한").order, [3, 0, 1, 2]);
        assert_eq!(Fallbacks::for_text("").order, [1, 0, 2, 3]);
    }
}
