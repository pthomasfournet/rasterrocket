//! Fallback faces compiled into the binary: script subsets of Noto faces
//! for Greek and Cyrillic, Hebrew, Arabic, Devanagari and Thai, sans-serif
//! and serif, and three symbol faces (`fonts/noto/`, SIL Open Font
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

/// One bundled face: its compressed program and whether it is a serif
/// design.
struct Bundled {
    data: &'static [u8],
    serif: bool,
}

/// The bundled faces; within a serif class, in lookup order.
const FACES: [Bundled; 13] = [
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSans-GreekCyrillic.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSerif-GreekCyrillic.ttf.z"),
        serif: true,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansHebrew.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSerifHebrew.ttf.z"),
        serif: true,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansArabic.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoNaskhArabic.ttf.z"),
        serif: true,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansDevanagari.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSerifDevanagari.ttf.z"),
        serif: true,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansThai.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSerifThai.ttf.z"),
        serif: true,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansMath.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansSymbols.ttf.z"),
        serif: false,
    },
    Bundled {
        data: include_bytes!("../../fonts/noto/NotoSansSymbols2.ttf.z"),
        serif: false,
    },
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
pub fn bundled_program(style: SubstituteStyle, c: u32) -> Option<Arc<FontProgram>> {
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
            0x03B1, 0x0416, 0x05D0, 0x0627, 0x0915, 0x0E01, 0x2192, 0x2211,
        ] {
            let program = bundled_program(sans, c).unwrap_or_else(|| panic!("U+{c:04X}"));
            assert_ne!(program.char_index(c), 0, "U+{c:04X}");
        }
        let sans_alpha = bundled_program(sans, 0x03B1).expect("alpha");
        let serif_alpha = bundled_program(serif, 0x03B1).expect("alpha");
        assert!(!Arc::ptr_eq(&sans_alpha, &serif_alpha));
        assert!(bundled_program(sans, 0x4E2D).is_none(), "no CJK is bundled");
    }
}
