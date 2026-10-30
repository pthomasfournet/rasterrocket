//! Installed fonts for the scripts the compiled-in substitutes do not
//! cover ([`Script`]): CJK, Arabic and Thai.
//!
//! The directories named by [`FONT_DIR_ENV`], then the standard font
//! directories, are walked once per process and every sfnt face is
//! indexed by family name and `OS/2` code-page ranges.  A request
//! selects, among the faces whose code-page bit covers the script, the
//! best match by a per-script preference list, serif class, weight and
//! slant.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use font::program::FontProgram;
use font::sfnt_info::{FaceInfo, face_infos};

use crate::resources::font_substitute::{Script, SubstituteFamily, SubstituteStyle};

/// Environment variable naming extra font directories, in the platform's
/// `PATH` syntax, searched before the standard directories.
pub const FONT_DIR_ENV: &str = "RROCKET_FONT_DIR";

/// The directories [`FONT_DIR_ENV`] names, in order; empty when unset.
pub fn env_font_dirs() -> Vec<PathBuf> {
    std::env::var_os(FONT_DIR_ENV)
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

/// Standard directories walked for fonts, in search order; `~` is the
/// home directory.
const FONT_DIRS: &[&str] = &[
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
    "C:/Windows/Fonts",
];

/// Greatest directory nesting walked below a font directory.
const MAX_DEPTH: u32 = 6;

/// Family-name fragments (lower case) of preferred faces per script, in
/// preference order.
const PREFERRED: &[(Script, &[&str])] = &[
    (
        Script::Japanese,
        &[
            "noto sans cjk jp",
            "noto serif cjk jp",
            "source han sans jp",
            "source han serif jp",
            "ipaexgothic",
            "ipaexmincho",
            "ipagothic",
            "ipamincho",
            "takaogothic",
            "takaomincho",
            "vl gothic",
            "droid sans japanese",
        ],
    ),
    (
        Script::SimplifiedChinese,
        &[
            "noto sans cjk sc",
            "noto serif cjk sc",
            "source han sans sc",
            "source han serif sc",
            "wenquanyi",
            "ar pl ukai cn",
            "ar pl uming cn",
            "ar pl sungtil gb",
            "ar pl kaitim gb",
        ],
    ),
    (
        Script::TraditionalChinese,
        &[
            "noto sans cjk tc",
            "noto serif cjk tc",
            "source han sans tc",
            "source han serif tc",
            "ar pl ukai tw",
            "ar pl uming tw",
            "ar pl kaitim big5",
            "ar pl mingti2l big5",
            "wenquanyi",
        ],
    ),
    (
        Script::Korean,
        &[
            "noto sans cjk kr",
            "noto serif cjk kr",
            "source han sans kr",
            "source han serif kr",
            "nanumgothic",
            "nanummyeongjo",
            "nanumbarungothic",
            "baekmuk gulim",
            "baekmuk batang",
            "undotum",
            "unbatang",
        ],
    ),
    (
        Script::Arabic,
        &[
            "noto sans arabic",
            "noto naskh arabic",
            "noto kufi arabic",
            "amiri",
            "scheherazade",
            "kacstone",
            "dejavu sans",
        ],
    ),
    (
        Script::Thai,
        &[
            "noto sans thai",
            "noto serif thai",
            "noto looped thai",
            "garuda",
            "norasi",
            "tlwg typo",
            "loma",
            "kinnari",
            "sawasdee",
            "waree",
            "umpush",
        ],
    ),
];

/// Family-name fragments that mark a serif design.
const SERIF_NAMES: &[&str] = &[
    "serif", "mincho", "ming", "song", "sung", "batang", "myeongjo", "naskh", "norasi", "kinnari",
];
/// Family-name fragments that mark a sans-serif design.
const SANS_NAMES: &[&str] = &[
    "sans", "gothic", "hei", "dotum", "gulim", "garuda", "loma", "waree", "kufi",
];

/// One installed face.
#[derive(Debug)]
struct Face {
    path: PathBuf,
    index: u32,
    /// Family name, lower case.
    family: String,
    code_page_ranges: [u32; 2],
    weight_class: u16,
    italic: bool,
    /// Serif design, when the name or `OS/2` class says.
    serif: Option<bool>,
}

impl Face {
    fn new(path: &Path, info: FaceInfo) -> Self {
        let mut family = info.family;
        family.make_ascii_lowercase();
        let serif = if SERIF_NAMES.iter().any(|f| family.contains(f)) {
            Some(true)
        } else if SANS_NAMES.iter().any(|f| family.contains(f)) {
            Some(false)
        } else {
            match info.family_class >> 8 {
                1..=7 => Some(true),
                8 => Some(false),
                _ => None,
            }
        };
        Self {
            path: path.to_path_buf(),
            index: info.index,
            family,
            code_page_ranges: info.code_page_ranges,
            weight_class: info.weight_class,
            italic: info.italic,
            serif,
        }
    }

    /// Whether the face's `OS/2` code pages include `script`.
    const fn covers(&self, script: Script) -> bool {
        let (range, bits): (usize, u32) = match script {
            Script::Japanese => (0, 1 << 17),
            Script::SimplifiedChinese => (0, 1 << 18),
            Script::Korean => (0, (1 << 19) | (1 << 21)),
            Script::TraditionalChinese => (0, 1 << 20),
            Script::Arabic => (0, 1 << 6),
            Script::Thai => (0, 1 << 16),
        };
        self.code_page_ranges[range] & bits != 0
    }
}

/// Every installed sfnt face, indexed once per process.
fn installed_faces() -> &'static [Face] {
    static FACES: OnceLock<Vec<Face>> = OnceLock::new();
    FACES.get_or_init(|| {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let mut faces = Vec::new();
        for dir in env_font_dirs() {
            walk(&dir, 0, &mut faces);
        }
        for dir in FONT_DIRS {
            let dir = match (dir.strip_prefix("~/"), &home) {
                (Some(rest), Some(home)) => home.join(rest),
                (Some(_), None) => continue,
                (None, _) => PathBuf::from(dir),
            };
            walk(&dir, 0, &mut faces);
        }
        log::debug!("system_fonts: indexed {} faces", faces.len());
        faces
    })
}

/// Index every sfnt file below `dir`, at most [`MAX_DEPTH`] levels down.
fn walk(dir: &Path, depth: u32, faces: &mut Vec<Face>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, faces);
            }
            continue;
        }
        let is_sfnt = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            e.eq_ignore_ascii_case("ttf")
                || e.eq_ignore_ascii_case("otf")
                || e.eq_ignore_ascii_case("ttc")
        });
        if is_sfnt {
            faces.extend(
                face_infos(&path)
                    .into_iter()
                    .map(|info| Face::new(&path, info)),
            );
        }
    }
}

/// The installed face that best serves `style` for `script`: among the
/// faces covering the script (all faces when none declares it), the one
/// with the closest serif class, then the earliest in the script's
/// preference list, then the nearest weight and slant.
fn select(script: Script, style: SubstituteStyle) -> Option<&'static Face> {
    let faces = installed_faces();
    let preferred = PREFERRED
        .iter()
        .find(|(s, _)| *s == script)
        .map_or(&[][..], |(_, names)| *names);
    let covering: Vec<&Face> = faces.iter().filter(|f| f.covers(script)).collect();
    let pool: Vec<&Face> = if covering.is_empty() {
        faces
            .iter()
            .filter(|f| preferred.iter().any(|p| f.family.contains(p)))
            .collect()
    } else {
        covering
    };
    let want_serif = style.family == SubstituteFamily::Serif;
    let want_weight: u16 = if style.bold { 700 } else { 400 };
    // A simple font of a non-CJK script mixes its script with Latin; a
    // face that also covers Latin-1 (code page bit 0) keeps both drawing.
    let want_latin = matches!(script, Script::Arabic | Script::Thai);
    pool.into_iter().min_by_key(|face| {
        let serif_rank = match face.serif {
            Some(serif) if serif == want_serif => 0,
            None => 1,
            Some(_) => 2,
        };
        let preference = preferred
            .iter()
            .position(|p| face.family.contains(p))
            .unwrap_or(preferred.len());
        (
            want_latin && face.code_page_ranges[0] & 1 == 0,
            serif_rank,
            preference,
            face.weight_class.abs_diff(want_weight),
            face.italic != style.italic,
        )
    })
}

/// The installed program for `style`, whose `script` must be set; `None`
/// when no installed face serves the script.  Resolved once per style per
/// process.
pub fn system_program(style: SubstituteStyle) -> Option<Arc<FontProgram>> {
    static CACHE: OnceLock<Mutex<HashMap<SubstituteStyle, Option<Arc<FontProgram>>>>> =
        OnceLock::new();
    let script = style.script?;
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(program) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&style)
    {
        return program.clone();
    }
    // Loaded outside the lock: a CJK collection is tens of megabytes.
    let program = select(script, style).and_then(|face| {
        std::fs::read(&face.path)
            .map_err(font::engine::LoadError::Io)
            .and_then(|data| FontProgram::parse(data, face.index))
            .inspect_err(|e| log::debug!("system_fonts: {}: {e}", face.path.display()))
            .ok()
            .map(Arc::new)
    });
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    cache.entry(style).or_insert(program).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(family: &str, class: i16, weight: u16, pages: u32) -> Face {
        Face::new(
            Path::new("/nonexistent"),
            FaceInfo {
                index: 0,
                family: family.to_owned(),
                code_page_ranges: [pages, 0],
                weight_class: weight,
                italic: false,
                family_class: class,
            },
        )
    }

    #[test]
    fn serif_class_comes_from_the_name_then_the_os2_class() {
        assert_eq!(face("Noto Serif CJK JP", 0, 400, 0).serif, Some(true));
        assert_eq!(face("IPAexGothic", 0, 400, 0).serif, Some(false));
        assert_eq!(face("Garuda", 0, 400, 0).serif, Some(false));
        assert_eq!(face("Unknown", 2 << 8, 400, 0).serif, Some(true));
        assert_eq!(face("Unknown", 8 << 8, 400, 0).serif, Some(false));
        assert_eq!(face("Unknown", 0, 400, 0).serif, None);
    }

    #[test]
    fn code_page_bits_identify_scripts() {
        let jp = face("x", 0, 400, 1 << 17);
        assert!(jp.covers(Script::Japanese));
        assert!(!jp.covers(Script::Korean));
        assert!(face("x", 0, 400, 1 << 21).covers(Script::Korean));
        assert!(face("x", 0, 400, 1 << 6).covers(Script::Arabic));
        assert!(face("x", 0, 400, 1 << 16).covers(Script::Thai));
    }

    /// With a Japanese face installed, both serif and sans requests
    /// resolve, and a serif request prefers a serif design.
    #[test]
    fn installed_japanese_face_resolves_by_serif_class() {
        let sans = SubstituteStyle {
            script: Some(Script::Japanese),
            ..SubstituteStyle::default()
        };
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..sans
        };
        let (Some(sans_face), Some(serif_face)) = (
            select(Script::Japanese, sans),
            select(Script::Japanese, serif),
        ) else {
            eprintln!("no Japanese font installed; skipping");
            return;
        };
        assert_ne!(sans_face.serif, Some(true), "{}", sans_face.family);
        assert_ne!(serif_face.serif, Some(false), "{}", serif_face.family);
        let program = system_program(serif).expect("loads");
        assert_ne!(
            program.char_index(0x65E5),
            0,
            "{} has 日",
            serif_face.family
        );
    }
}
