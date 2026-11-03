//! Fallback faces compiled into the binary: script subsets of Noto faces
//! for Greek and Cyrillic, Hebrew, Arabic, Devanagari, Thai, Bengali,
//! Tamil, Armenian, Georgian and Ethiopic, sans-serif and serif, and three
//! symbol faces (`fonts/noto/`, SIL Open Font
//! License 1.1).
//!
//! They draw a character that neither a font's substitute nor any
//! installed face maps: [`super::system_fonts::FontLibrary::fallback_program`]
//! consults them only after every installed face.  Each is stored
//! zlib-compressed and inflated and parsed once per process, on the first
//! lookup that reaches it.

use std::sync::{Arc, OnceLock};

use font::program::FontProgram;

use crate::resources::font_substitute::{SubstituteFamily, SubstituteStyle};

/// One bundled face: its file name under `fonts/noto/`, its compressed
/// program and whether it is a serif design.
struct Bundled {
    #[cfg_attr(not(test), expect(dead_code, reason = "read by the tests only"))]
    file: &'static str,
    data: &'static [u8],
    serif: bool,
}

/// One bundled face: the zlib-compressed file `fonts/noto/<file>` and
/// whether it is a serif design.
macro_rules! face {
    ($file:literal, serif: $serif:literal) => {
        Bundled {
            file: $file,
            data: include_bytes!(concat!("../../fonts/noto/", $file)),
            serif: $serif,
        }
    };
}

/// The bundled faces; within a serif class, in lookup order.
const FACES: [Bundled; 23] = [
    face!("NotoSans-GreekCyrillic.ttf.z", serif: false),
    face!("NotoSerif-GreekCyrillic.ttf.z", serif: true),
    face!("NotoSansHebrew.ttf.z", serif: false),
    face!("NotoSerifHebrew.ttf.z", serif: true),
    face!("NotoSansArabic.ttf.z", serif: false),
    face!("NotoNaskhArabic.ttf.z", serif: true),
    face!("NotoSansDevanagari.ttf.z", serif: false),
    face!("NotoSerifDevanagari.ttf.z", serif: true),
    face!("NotoSansThai.ttf.z", serif: false),
    face!("NotoSerifThai.ttf.z", serif: true),
    face!("NotoSansBengali.ttf.z", serif: false),
    face!("NotoSerifBengali.ttf.z", serif: true),
    face!("NotoSansTamil.ttf.z", serif: false),
    face!("NotoSerifTamil.ttf.z", serif: true),
    face!("NotoSansArmenian.ttf.z", serif: false),
    face!("NotoSerifArmenian.ttf.z", serif: true),
    face!("NotoSansGeorgian.ttf.z", serif: false),
    face!("NotoSerifGeorgian.ttf.z", serif: true),
    face!("NotoSansEthiopic.ttf.z", serif: false),
    face!("NotoSerifEthiopic.ttf.z", serif: true),
    face!("NotoSansMath.ttf.z", serif: false),
    face!("NotoSansSymbols.ttf.z", serif: false),
    face!("NotoSansSymbols2.ttf.z", serif: false),
];

/// The bundled program of face `index`, inflated and parsed once per
/// process; `None`, logged, only if the bundled data is unusable.
fn program(index: usize) -> Option<Arc<FontProgram>> {
    static PROGRAMS: [OnceLock<Option<Arc<FontProgram>>>; FACES.len()] =
        [const { OnceLock::new() }; FACES.len()];
    PROGRAMS
        .get(index)?
        .get_or_init(|| {
            let data = pdf::inflate(FACES[index].data)
                .inspect_err(|e| log::error!("bundled_fonts: face {index}: {e}"))
                .ok()?;
            FontProgram::parse(data, 0)
                .inspect_err(|e| log::error!("bundled_fonts: face {index}: {e}"))
                .ok()
                .map(Arc::new)
        })
        .clone()
}

/// The bundled program that maps the character `c`, for a font of
/// `style`: faces of the style's serif class first, then the others.
/// `None` when no bundled face maps `c`.
pub(crate) fn bundled_program(style: SubstituteStyle, c: u32) -> Option<Arc<FontProgram>> {
    let want_serif = style.family == SubstituteFamily::Serif;
    let preferred = (0..FACES.len()).filter(|&i| FACES[i].serif == want_serif);
    let others = (0..FACES.len()).filter(|&i| FACES[i].serif != want_serif);
    preferred
        .chain(others)
        .find_map(|i| program(i).filter(|program| program.char_index(c) != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bundled face inflates and parses.
    #[test]
    fn bundled_faces_parse() {
        for index in 0..FACES.len() {
            assert!(program(index).is_some(), "face {index}");
        }
    }

    /// Each covered script resolves, in the serif class asked for when a
    /// face of that class covers it.
    #[test]
    fn bundled_faces_cover_their_scripts() {
        let sans = SubstituteStyle::default();
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..sans
        };
        for c in [
            0x03B1, 0x0416, 0x05D0, 0x0627, 0x0915, 0x0E01, 0x0995, 0x0B95, 0x0531, 0x10D0, 0x1200,
            0x2192, 0x2211,
        ] {
            let program = bundled_program(sans, c).unwrap_or_else(|| panic!("U+{c:04X}"));
            assert_ne!(program.char_index(c), 0, "U+{c:04X}");
        }
        for c in [0x03B1, 0x0995, 0x0B95, 0x0531, 0x10D0, 0x1200] {
            let sans = bundled_program(sans, c).expect("sans");
            let serif = bundled_program(serif, c).expect("serif");
            assert!(!Arc::ptr_eq(&sans, &serif), "U+{c:04X} has both classes");
        }
        assert!(bundled_program(sans, 0x4E2D).is_none(), "no CJK is bundled");
    }

    /// The compiled-in faces are exactly the `.ttf.z` files of
    /// `fonts/noto/`.
    #[test]
    fn every_bundled_file_is_compiled_in() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fonts/noto");
        let mut files: Vec<String> = std::fs::read_dir(dir)
            .expect("fonts/noto")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .into_string()
                    .expect("UTF-8")
            })
            .filter(|name| name.ends_with(".ttf.z"))
            .collect();
        files.sort();
        let mut compiled: Vec<&str> = FACES.iter().map(|face| face.file).collect();
        compiled.sort_unstable();
        assert_eq!(files, compiled);
    }

    /// The Indic faces map the dotted circle the shaper inserts for a
    /// broken cluster, and the Bengali and Tamil faces the danda and double
    /// danda that end their sentences.
    #[test]
    fn indic_faces_map_their_punctuation() {
        for (file, chars) in [
            ("NotoSansDevanagari.ttf.z", &[0x25CC][..]),
            ("NotoSerifDevanagari.ttf.z", &[0x25CC]),
            ("NotoSansBengali.ttf.z", &[0x0964, 0x0965, 0x25CC]),
            ("NotoSerifBengali.ttf.z", &[0x0964, 0x0965, 0x25CC]),
            ("NotoSansTamil.ttf.z", &[0x0964, 0x0965, 0x25CC]),
            ("NotoSerifTamil.ttf.z", &[0x0964, 0x0965, 0x25CC]),
        ] {
            let index = FACES.iter().position(|face| face.file == file).expect(file);
            let program = program(index).expect("parses");
            for &c in chars {
                assert_ne!(program.char_index(c), 0, "{file} maps U+{c:04X}");
            }
        }
    }
}
