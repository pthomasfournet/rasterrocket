//! Per-character fallback for fonts drawn by a substitute.
//!
//! A substitute covers the scripts of its own family; a code whose
//! character it lacks is drawn from an installed face that maps that
//! character ([`fallback_program`]).  The character of a code is the
//! Unicode value of its glyph name (simple font), of its CID in the font's
//! character collection, or of its code in the `ToUnicode` map (composite
//! font).
//!
//! Resolution is lazy: only the codes a text-showing operator shows are
//! looked up, each Unicode value once per font, so a composite font whose
//! `ToUnicode` map spans the whole code space costs nothing for the codes
//! it never shows.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use font::program::FontProgram;

use super::encodings::glyph_name_for_code;
use super::system_fonts::fallback_program;
use crate::resources::cmap::CMap;
use crate::resources::cmap_predefined::{CidToUnicode, cid_to_unicode};
use crate::resources::font::FontDescriptor;
use crate::resources::font_substitute::SubstituteStyle;

/// Where the Unicode value of a shown code comes from.
enum UnicodeSource {
    /// Simple font: the value of each code's glyph name, 0 when unknown.
    Names(Box<[u32; 256]>),
    /// Composite font of a known character collection: by CID.
    Collection(CidToUnicode),
    /// Composite font with a `ToUnicode` map: by code.
    ToUnicode(Arc<CMap>),
}

/// A fallback glyph: the slot of its program in
/// [`FallbackResolver::program`] and its glyph index there.
pub type FallbackGlyph = (usize, u32);

/// Fallback state of one font drawn by a substitute, shared by every page
/// that draws the font.
pub struct FallbackResolver {
    style: SubstituteStyle,
    /// Vertical writing: a fallback glyph is replaced by its vertical form.
    vertical: bool,
    /// The substitute program; a character it maps needs no fallback.
    primary: Arc<FontProgram>,
    source: UnicodeSource,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Programs that serve at least one character, in discovery order.
    programs: Vec<Arc<FontProgram>>,
    /// Resolution of each Unicode value looked up so far.
    glyphs: HashMap<u32, Option<FallbackGlyph>>,
}

impl FallbackResolver {
    /// The resolver of `desc` drawn by the substitute `primary`; `None`
    /// when no Unicode value of its codes is known (a composite font of no
    /// known collection and without a `ToUnicode` map).
    pub fn new(desc: &FontDescriptor, primary: Arc<FontProgram>) -> Option<Self> {
        let source = match &desc.cid_encoding {
            None => {
                let mut names = Box::new([0u32; 256]);
                for (code, slot) in names.iter_mut().enumerate() {
                    *slot = glyph_name_for_code(&desc.differences, desc.base_encoding, code)
                        .and_then(font::glyph_name_to_unicode)
                        .unwrap_or(0);
                }
                UnicodeSource::Names(names)
            }
            Some(enc) => desc
                .substitute
                .script
                .and_then(cid_to_unicode)
                .map_or_else(
                    || enc.to_unicode.clone().map(UnicodeSource::ToUnicode),
                    |collection| Some(UnicodeSource::Collection(collection)),
                )?,
        };
        Some(Self {
            style: desc.substitute,
            vertical: desc.cid_encoding.as_ref().is_some_and(|enc| enc.vertical),
            primary,
            source,
            state: Mutex::default(),
        })
    }

    /// The Unicode value the shown `code` (with CID `cid` in a composite
    /// font) stands for, when the font says.
    fn unicode(&self, code: u32, cid: u32) -> Option<u32> {
        let unicode = match &self.source {
            UnicodeSource::Names(names) => *names.get(usize::try_from(code).ok()?)?,
            UnicodeSource::Collection(table) => *table.get(&cid)?,
            UnicodeSource::ToUnicode(map) => first_scalar(*map.map.get(&code)?)?,
        };
        (unicode != 0).then_some(unicode)
    }

    /// The fallback glyph that draws the shown `code` (CID `cid`): the
    /// glyph of its character in the first installed program that maps
    /// it, when the substitute does not.  Each character is resolved once
    /// per font.
    pub fn glyph(&self, code: u32, cid: u32) -> Option<FallbackGlyph> {
        let unicode = self.unicode(code, cid)?;
        if let Some(&known) = self.lock().glyphs.get(&unicode) {
            return known;
        }
        // Searched outside the lock: a first search reads installed files.
        let found = (self.primary.char_index(unicode) == 0)
            .then(|| fallback_program(self.style, unicode))
            .flatten();
        let mut state = self.lock();
        if let Some(&known) = state.glyphs.get(&unicode) {
            return known;
        }
        let resolved = found.map(|program| {
            let mut gid = program.char_index(unicode);
            if self.vertical {
                gid = program.vertical_gid(gid);
            }
            let slot = state
                .programs
                .iter()
                .position(|p| Arc::ptr_eq(p, &program))
                .unwrap_or_else(|| {
                    state.programs.push(program);
                    state.programs.len() - 1
                });
            (slot, gid)
        });
        let _ = state.glyphs.insert(unicode, resolved);
        resolved
    }

    /// The program in `slot`.
    pub fn program(&self, slot: usize) -> Option<Arc<FontProgram>> {
        self.lock().programs.get(slot).cloned()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The first Unicode scalar value of a `ToUnicode` value: a scalar value,
/// or above U+10FFFF two UTF-16 units packed as `first << 16 | second`;
/// `None` for an unpaired surrogate.
pub const fn first_scalar(value: u32) -> Option<u32> {
    let (first, second) = if value > 0x10_FFFF {
        (value >> 16, value & 0xFFFF)
    } else {
        (value, 0)
    };
    match (first, second) {
        (0xD800..=0xDBFF, 0xDC00..=0xDFFF) => {
            Some(0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00))
        }
        (0xD800..=0xDFFF, _) => None,
        _ => Some(first),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_unicode_values_yield_their_first_scalar() {
        assert_eq!(first_scalar(0x0041), Some(0x41));
        assert_eq!(first_scalar(0x1_F600), Some(0x1_F600));
        assert_eq!(first_scalar(0x0066_0069), Some(0x66), "fi ligature");
        assert_eq!(first_scalar(0xD83D_DE00), Some(0x1_F600), "surrogate pair");
        assert_eq!(first_scalar(0xD83D), None, "unpaired high surrogate");
        assert_eq!(first_scalar(0xDE00_0041), None, "unpaired low surrogate");
    }
}
