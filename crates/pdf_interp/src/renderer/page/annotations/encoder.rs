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
//! CJK character collections, encoded by their `UTF16-H` `CMaps`; they are
//! drawn by the substitute and fallback faces like any non-embedded
//! composite font.

use std::collections::HashMap;

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
                            (first_scalar(value), code_bytes(cmap, code))
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

    /// The first fallback font that shows `c`, and its codes there.
    pub(super) fn find(self, c: char) -> Option<(&'static [u8], Vec<u8>)> {
        let mut units = [0u16; 2];
        let units = c.encode_utf16(&mut units);
        let key = units.iter().fold(0u32, |k, &u| (k << 16) | u32::from(u));
        let (name, _, _) = self
            .order
            .iter()
            .map(|&i| FALLBACKS[i])
            .find(|(_, _, cmap)| {
                predefined_cmap(cmap)
                    .is_some_and(|cmap| cmap.map.get(&key).is_some_and(|&cid| cid != 0))
            })?;
        Some((name, units.iter().flat_map(|u| u.to_be_bytes()).collect()))
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
}

impl<'r, 'p> FontTypesetter<'r, 'p> {
    /// A typesetter over the resources in effect in `renderer`.
    pub(super) fn new(renderer: &'r mut PageRenderer<'p>) -> Self {
        Self {
            renderer,
            encoders: HashMap::new(),
        }
    }

    /// The encoder of the font resource `font`.
    fn encoder(&mut self, font: &[u8]) -> &UnicodeEncoder {
        if !self.encoders.contains_key(font) {
            let renderer = &mut *self.renderer;
            let active = renderer.font_cache.select(&renderer.resources, font);
            let _ = self
                .encoders
                .insert(font.to_vec(), UnicodeEncoder::of(&active.descriptor));
        }
        &self.encoders[font]
    }
}

impl Typesetter for FontTypesetter<'_, '_> {
    fn runs(&mut self, font: &[u8], text: &str) -> Vec<Run> {
        let fallbacks = Fallbacks::for_text(text);
        let encoder = self.encoder(font);
        let question = encoder.encode('?').unwrap_or(b"?");
        let mut runs: Vec<Run> = Vec::new();
        for c in text.chars() {
            let fallback;
            let (run_font, codes): (&[u8], &[u8]) = if let Some(codes) = encoder.encode(c) {
                (font, codes)
            } else if let Some(found) = fallbacks.find(c) {
                fallback = found;
                (fallback.0, &fallback.1)
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
        runs
    }

    /// The font's `/Widths` (or `/W` for a composite font), else its
    /// program's advances.
    fn advance(&mut self, font: &[u8], codes: &[u8]) -> f64 {
        let renderer = &mut *self.renderer;
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
    use super::{Fallbacks, UnicodeEncoder, code_bytes};
    use crate::resources::cmap::CMap;
    use crate::resources::cmap_predefined::predefined_cmap;
    use crate::resources::font::resolve_font;
    use crate::test_helpers::make_doc_with_object;

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
    fn codes_take_the_length_of_their_codespace() {
        assert_eq!(code_bytes(&CMap::identity(), 0x41), Some(vec![0, 0x41]));
        assert_eq!(
            code_bytes(&CMap::identity(), 0x4E2D),
            Some(vec![0x4E, 0x2D])
        );
    }

    #[test]
    fn kana_select_the_japanese_fallback_and_han_alone_the_chinese() {
        let fallback = |c, text| Fallbacks::for_text(text).find(c);
        let (font, codes) = fallback('日', "日本語の").expect("covered");
        assert_eq!(
            (font, codes.as_slice()),
            (&b"FallbackJapan1"[..], &[0x65, 0xE5][..])
        );
        let (font, _) = fallback('日', "日本").expect("covered");
        assert_eq!(font, b"FallbackGB1");
        let (font, _) = fallback('한', "한국").expect("covered");
        assert_eq!(font, b"FallbackKorea1");
        assert!(
            fallback('\u{0627}', "").is_none(),
            "Arabic is in no collection"
        );
        assert_eq!(Fallbacks::for_text("한").order, [3, 0, 1, 2]);
        assert_eq!(Fallbacks::for_text("").order, [1, 0, 2, 3]);
    }
}
