//! Installed fonts: the faces substitutes and per-character fallbacks are
//! drawn from.
//!
//! A [`FontLibrary`] walks a list of extra directories, then (unless
//! system fonts are off) the standard font directories, on first use and
//! indexes every sfnt face by family
//! name and `OS/2` code-page ranges.  A request for a script the
//! compiled-in substitutes do not cover ([`Script`]: CJK, Arabic, Thai)
//! selects, among the faces whose code-page bit covers the script, the
//! best match by a per-script preference list, serif class, weight and
//! slant.  A per-character request ([`FontLibrary::fallback_program`])
//! ranks every face the same way and takes the first whose `cmap` maps the
//! character; a face's `cmap` coverage is read on the first such request
//! that reaches it.
//!
//! Libraries are shared process-wide, one per list of extra directories
//! and system-font switch ([`FontLibrary::shared`]), so documents rendered
//! with the same sources index the installed fonts and parse each program
//! once.  With the `bundled-fonts` feature, faces compiled into the binary
//! are the last resort of a per-character request.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use font::program::FontProgram;
use font::sfnt_info::{Coverage, FaceInfo, face_infos, unicode_coverage};

use super::substitutes::INSTALLED_ROWS;
use crate::resources::font_substitute::{Script, SubstituteFamily, SubstituteStyle};

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

/// Family-name fragments (lower case) of preferred faces for a character
/// of no [`Script`], in preference order: families designed as broad
/// multi-script sets.
const PREFERRED_ANY: &[&str] = &[
    "noto sans",
    "noto serif",
    "dejavu sans",
    "dejavu serif",
    "freesans",
    "freeserif",
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
    /// Unicode coverage of the face's `cmap`, read on first use; `None`
    /// for a face without outlines or a Unicode subtable.
    coverage: OnceLock<Option<Coverage>>,
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
            coverage: OnceLock::new(),
        }
    }

    /// Whether the face's `cmap` maps the character `c`.
    fn maps(&self, c: u32) -> bool {
        self.coverage
            .get_or_init(|| unicode_coverage(&self.path, self.index))
            .as_ref()
            .is_some_and(|coverage| coverage.contains(c))
    }

    /// Ranking key of the face for `style` under the family-name
    /// preference list `preferred`; the least key is the best face.
    fn rank(&self, style: SubstituteStyle, preferred: &[&str]) -> (u8, usize, u16, bool) {
        let want_serif = style.family == SubstituteFamily::Serif;
        let want_weight: u16 = if style.bold { 700 } else { 400 };
        let serif_rank = match self.serif {
            Some(serif) if serif == want_serif => 0,
            None => 1,
            Some(_) => 2,
        };
        let preference = preferred
            .iter()
            .position(|p| self.family.contains(p))
            .unwrap_or(preferred.len());
        (
            serif_rank,
            preference,
            self.weight_class.abs_diff(want_weight),
            self.italic != style.italic,
        )
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

/// Key of one library: its extra directories and whether the standard
/// font directories are searched.
type LibraryKey = (Vec<PathBuf>, bool);

/// Key of one fallback ranking: the family class, weight and slant asked
/// for, and the script of the character.
type RankKey = (SubstituteFamily, bool, bool, Option<Script>);

/// The installed fonts of one list of extra directories and, when
/// `system` is set, the standard font directories.  See the module
/// documentation.
pub struct FontLibrary {
    /// Extra directories, walked before the standard ones.
    dirs: Vec<PathBuf>,
    /// Whether the standard font directories and the fixed locations of
    /// the metric-compatible substitutes are searched.
    system: bool,
    /// Every sfnt face found, sorted by path; built on first use.
    faces: OnceLock<Vec<Face>>,
    /// Program of each face, by index into `faces`, parsed on first use.
    programs: Mutex<HashMap<usize, Option<Arc<FontProgram>>>>,
    /// Face order of each fallback ranking, by index into `faces`.
    orders: Mutex<HashMap<RankKey, Arc<[usize]>>>,
    /// The installed metric-compatible substitute of each row of the
    /// substitute table, located on first use.
    substitutes: [OnceLock<Option<Arc<FontProgram>>>; INSTALLED_ROWS],
}

impl FontLibrary {
    fn new(dirs: Vec<PathBuf>, system: bool) -> Self {
        Self {
            dirs,
            system,
            faces: OnceLock::new(),
            programs: Mutex::default(),
            orders: Mutex::default(),
            substitutes: [const { OnceLock::new() }; INSTALLED_ROWS],
        }
    }

    /// The library searching the extra directories `dirs` (in order) and
    /// then, when `system` is set, the standard font directories; shared by
    /// every caller naming the same sources.
    #[must_use]
    pub fn shared(dirs: &[PathBuf], system: bool) -> Arc<Self> {
        static LIBRARIES: OnceLock<Mutex<HashMap<LibraryKey, Arc<FontLibrary>>>> = OnceLock::new();
        let mut libraries = LIBRARIES
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Arc::clone(
            libraries
                .entry((dirs.to_vec(), system))
                .or_insert_with(|| Arc::new(Self::new(dirs.to_vec(), system))),
        )
    }

    /// Every face of the library's directories, indexed on first use.
    fn faces(&self) -> &[Face] {
        self.faces.get_or_init(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from);
            let mut faces = Vec::new();
            for dir in &self.dirs {
                walk(dir, 0, &mut faces);
            }
            for dir in FONT_DIRS.iter().filter(|_| self.system) {
                let dir = match (dir.strip_prefix("~/"), &home) {
                    (Some(rest), Some(home)) => home.join(rest),
                    (Some(_), None) => continue,
                    (None, _) => PathBuf::from(dir),
                };
                walk(&dir, 0, &mut faces);
            }
            // Directory listings come in no fixed order; ties in every
            // ranking below resolve by path, so selection is the same on
            // every run.
            faces.sort_by(|a, b| a.path.cmp(&b.path).then(a.index.cmp(&b.index)));
            log::debug!("system_fonts: indexed {} faces", faces.len());
            faces
        })
    }

    /// The installed face that best serves `style` for `script`: among
    /// the faces covering the script (all faces when none declares it),
    /// the one with the closest serif class, then the earliest in the
    /// script's preference list, then the nearest weight and slant.
    fn select(&self, script: Script, style: SubstituteStyle) -> Option<usize> {
        let faces = self.faces();
        let preferred = preferred_names(Some(script));
        let covering: Vec<usize> = (0..faces.len())
            .filter(|&i| faces[i].covers(script))
            .collect();
        let pool: Vec<usize> = if covering.is_empty() {
            (0..faces.len())
                .filter(|&i| preferred.iter().any(|p| faces[i].family.contains(p)))
                .collect()
        } else {
            covering
        };
        // A simple font of a non-CJK script mixes its script with Latin; a
        // face that also covers Latin-1 (code page bit 0) keeps both drawing.
        let want_latin = matches!(script, Script::Arabic | Script::Thai);
        pool.into_iter().min_by_key(|&i| {
            let face = &faces[i];
            (
                want_latin && face.code_page_ranges[0] & 1 == 0,
                face.rank(style, preferred),
            )
        })
    }

    /// The program of face `index`, parsed once and shared by every
    /// request that selects the face.
    fn face_program(&self, index: usize) -> Option<Arc<FontProgram>> {
        let face = self.faces().get(index)?;
        if let Some(program) = self
            .programs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&index)
        {
            return program.clone();
        }
        // Parsed outside the lock: a CJK collection is tens of megabytes.
        let program = std::fs::read(&face.path)
            .map_err(font::engine::LoadError::Io)
            .and_then(|data| FontProgram::parse(data, face.index))
            .inspect_err(|e| log::debug!("system_fonts: {}: {e}", face.path.display()))
            .ok()
            .map(Arc::new);
        let mut programs = self.programs.lock().unwrap_or_else(PoisonError::into_inner);
        programs.entry(index).or_insert(program).clone()
    }

    /// The installed program for `style`, whose `script` must be set;
    /// `None` when no installed face serves the script.
    pub fn system_program(&self, style: SubstituteStyle) -> Option<Arc<FontProgram>> {
        let script = style.script?;
        self.select(script, style)
            .and_then(|index| self.face_program(index))
    }

    /// The installed program that draws the character `c` for a font of
    /// `style` whose substitute lacks it: among the faces whose `cmap` maps
    /// `c`, the first by the ranking of [`Self::select`] (serif class, the
    /// preference list of the character's script, weight, slant), ties
    /// going to the earlier path.  With the `bundled-fonts` feature, a
    /// bundled face that maps `c` when no installed face does.  `None`
    /// when no face maps `c`.
    pub fn fallback_program(&self, style: SubstituteStyle, c: u32) -> Option<Arc<FontProgram>> {
        let faces = self.faces();
        let script = Script::of_char(c);
        let key: RankKey = (style.family, style.bold, style.italic, script);
        let order = {
            let mut orders = self.orders.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(orders.entry(key).or_insert_with(|| {
                let preferred = preferred_names(script);
                let mut order: Vec<usize> = (0..faces.len()).collect();
                order.sort_by_key(|&i| faces[i].rank(style, preferred));
                order.into()
            }))
        };
        let installed = order.iter().filter(|&&i| faces[i].maps(c)).find_map(|&i| {
            self.face_program(i)
                .filter(|program| program.char_index(c) != 0)
        });
        #[cfg(feature = "bundled-fonts")]
        let installed = installed.or_else(|| super::bundled_fonts::bundled_program(style, c));
        installed
    }

    /// The installed substitute of row `row` of the substitute table,
    /// located by `locate` from the extra directories and the system-font
    /// switch on first use.
    pub(super) fn substitute(
        &self,
        row: usize,
        locate: impl FnOnce(&[PathBuf], bool) -> Option<Arc<FontProgram>>,
    ) -> Option<Arc<FontProgram>> {
        self.substitutes
            .get(row)?
            .get_or_init(|| locate(&self.dirs, self.system))
            .clone()
    }
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

/// The family-name preference list for script `script`, or for a
/// character of no known script.
fn preferred_names(script: Option<Script>) -> &'static [&'static str] {
    script
        .and_then(|script| PREFERRED.iter().find(|(s, _)| *s == script))
        .map_or(PREFERRED_ANY, |(_, names)| *names)
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
        let library = FontLibrary::shared(&[], true);
        let (Some(sans_face), Some(serif_face)) = (
            library.select(Script::Japanese, sans),
            library.select(Script::Japanese, serif),
        ) else {
            eprintln!("no Japanese font installed; skipping");
            return;
        };
        let (sans_face, serif_face) = (&library.faces()[sans_face], &library.faces()[serif_face]);
        assert_ne!(sans_face.serif, Some(true), "{}", sans_face.family);
        assert_ne!(serif_face.serif, Some(false), "{}", serif_face.family);
        let program = library.system_program(serif).expect("loads");
        assert_ne!(
            program.char_index(0x65E5),
            0,
            "{} has 日",
            serif_face.family
        );
    }

    /// One library serves every caller naming the same directories, and
    /// a different list gets its own, which indexes its directories first.
    #[test]
    fn libraries_are_shared_per_directory_list() {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../font_test_data/test_data/ttf"
        ));
        let standard = FontLibrary::shared(&[], true);
        assert!(Arc::ptr_eq(&standard, &FontLibrary::shared(&[], true)));
        let extra = FontLibrary::shared(std::slice::from_ref(&dir), true);
        assert!(!Arc::ptr_eq(&standard, &extra));
        assert!(Arc::ptr_eq(
            &extra,
            &FontLibrary::shared(std::slice::from_ref(&dir), true)
        ));
        assert_eq!(extra.dirs, std::slice::from_ref(&dir));
        assert!(
            extra.faces().iter().any(|f| f.path.starts_with(&dir)),
            "the extra directory is indexed"
        );
        assert!(!standard.faces().iter().any(|f| f.path.starts_with(&dir)));
    }
}
