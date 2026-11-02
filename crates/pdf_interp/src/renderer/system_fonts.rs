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
use std::time::SystemTime;

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
    "~/Library/Fonts",
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
            "hiragino sans",
            "hiragino kaku gothic pron",
            "hiragino kaku gothic pro",
            "hiragino mincho pron",
            "hiragino mincho pro",
            "yu gothic",
            "yu mincho",
            "meiryo",
            "ms gothic",
            "ms mincho",
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
            "pingfang sc",
            "hiragino sans gb",
            "heiti sc",
            "songti sc",
            "microsoft yahei",
            "dengxian",
            "simhei",
            "simsun",
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
            "pingfang tc",
            "heiti tc",
            "songti tc",
            "microsoft jhenghei",
            "mingliu",
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
            "apple sd gothic neo",
            "applemyungjo",
            "malgun gothic",
            "batang",
            "gulim",
            "dotum",
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
            "geeza pro",
            "al bayan",
            "arabic typesetting",
            "traditional arabic",
            "simplified arabic",
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
            "thonburi",
            "sukhumvit set",
            "leelawadee ui",
            "leelawadee",
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

/// Families (lower case) whose glyphs are placeholders standing for a
/// character's block, not drawings of it: never a substitute.
const PLACEHOLDER_FAMILIES: &[&str] = &[".lastresort", "lastresort"];

/// Family-name fragments that mark a serif design.
const SERIF_NAMES: &[&str] = &[
    "serif", "mincho", "ming", "song", "sung", "batang", "myeongjo", "myungjo", "naskh", "norasi",
    "kinnari",
];
/// Family-name fragments that mark a sans-serif design.
const SANS_NAMES: &[&str] = &[
    "sans", "gothic", "hei", "dotum", "gulim", "garuda", "loma", "waree", "kufi", "pingfang",
    "thonburi",
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
    /// The longest fragment of any script's preference list that the
    /// family name contains as whole words: the design the face is.
    listed_as: Option<&'static str>,
    /// Unicode coverage of the face's `cmap`, read on first use; `None`
    /// for a face without outlines or a Unicode subtable.
    coverage: OnceLock<Option<Coverage>>,
}

impl Face {
    fn new(path: &Path, info: FaceInfo) -> Self {
        let mut family = info.family;
        family.make_ascii_lowercase();
        let listed_as = PREFERRED
            .iter()
            .flat_map(|(_, names)| names.iter().copied())
            .filter(|name| names_words(&family, name))
            .max_by_key(|name| name.len());
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
            listed_as,
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

    /// Whether the face is one of the platform's private faces, whose
    /// family names start with `.`.
    fn hidden(&self) -> bool {
        self.family.starts_with('.')
    }

    /// Position of the face in the family-name preference list of
    /// `script` (see [`preferred_names`]), or the list's length when it is
    /// not listed.  For a script the face is the design it is listed as, so
    /// a family whose name extends a listed one by a further word
    /// ("hiragino sans gb" of "hiragino sans") is the longer design; for a
    /// character of no script, any listed name the family contains as
    /// whole words places it.
    fn preference(&self, script: Option<Script>) -> usize {
        let preferred = preferred_names(script);
        let position = match script {
            None => preferred.iter().position(|p| names_words(&self.family, p)),
            Some(_) => self
                .listed_as
                .and_then(|listed| preferred.iter().position(|p| *p == listed)),
        };
        position.unwrap_or(preferred.len())
    }

    /// Ranking key of the face for `style` under the family-name
    /// preference list of `script`; the least key is the best face.  For a
    /// script, a listed face ranks before every unlisted one: a face listed
    /// for the script is a design made for it, which an unlisted face of
    /// the requested serif class (a monospace or script face that happens
    /// to carry the characters) is not.  With `latin`, a face that also
    /// maps Latin comes next.  Then come the serif class, the preference,
    /// the weight and the slant.
    fn rank(&self, style: SubstituteStyle, script: Option<Script>, latin: bool) -> FaceRank {
        let want_serif = style.family == SubstituteFamily::Serif;
        let want_weight: u16 = if style.bold { 700 } else { 400 };
        let serif_rank = match self.serif {
            Some(serif) if serif == want_serif => 0,
            None => 1,
            Some(_) => 2,
        };
        let preference = self.preference(script);
        (
            script.is_some() && preference == preferred_names(script).len(),
            latin && !self.maps(u32::from('A')),
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

/// The ranking key of [`Face::rank`]: unlisted for the script, lacking
/// Latin, serif class, preference, weight distance, slant mismatch.
type FaceRank = (bool, bool, u8, usize, u16, bool);

/// The key by which [`FontLibrary::select`] chooses the face for `script`: a
/// visible face before a hidden one, then [`Face::rank`], asking for Latin
/// for Arabic and Thai, whose simple fonts mix the script with Latin.
fn select_key(face: &Face, style: SubstituteStyle, script: Script) -> (bool, FaceRank) {
    let latin = matches!(script, Script::Arabic | Script::Thai);
    (face.hidden(), face.rank(style, Some(script), latin))
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
pub(crate) struct FontLibrary {
    /// Extra directories, walked before the standard ones.
    dirs: Vec<PathBuf>,
    /// Whether the standard font directories and the fixed locations of
    /// the metric-compatible substitutes are searched.
    system: bool,
    /// Every sfnt face found, sorted by path, and the directories walked
    /// to find them; built on first use.
    index: OnceLock<Index>,
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
            index: OnceLock::new(),
            programs: Mutex::default(),
            orders: Mutex::default(),
            substitutes: [const { OnceLock::new() }; INSTALLED_ROWS],
        }
    }

    /// The library searching the extra directories `dirs` (in order) and
    /// then, when `system` is set, the standard font directories; shared by
    /// every caller naming the same sources.
    #[must_use]
    ///
    /// A library whose index is out of date — a directory it walked, or a
    /// root it found missing, has changed since — is replaced by a fresh
    /// one, so fonts installed or removed while the process runs are seen
    /// by the documents opened afterwards.  Callers holding the old library
    /// keep it.
    pub(crate) fn shared(dirs: &[PathBuf], system: bool) -> Arc<Self> {
        static LIBRARIES: OnceLock<Mutex<HashMap<LibraryKey, Arc<FontLibrary>>>> = OnceLock::new();
        let mut libraries = LIBRARIES
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let library = libraries
            .entry((dirs.to_vec(), system))
            .or_insert_with(|| Arc::new(Self::new(dirs.to_vec(), system)));
        if library.is_stale() {
            log::debug!("system_fonts: font directories changed; re-indexing");
            *library = Arc::new(Self::new(dirs.to_vec(), system));
        }
        let library = Arc::clone(library);
        drop(libraries);
        library
    }

    /// Whether the library was indexed and a directory it walked has
    /// changed since: its modification time differs, or it appeared or
    /// vanished.
    fn is_stale(&self) -> bool {
        self.index.get().is_some_and(|index| {
            index
                .directories
                .iter()
                .any(|(dir, stamp)| modified(dir) != *stamp)
        })
    }

    /// Every face of the library's directories, indexed on first use.
    fn faces(&self) -> &[Face] {
        &self
            .index
            .get_or_init(|| {
                let home = std::env::var_os("HOME").map(PathBuf::from);
                let mut index = Index::default();
                for dir in &self.dirs {
                    index.walk(dir, 0);
                }
                for dir in FONT_DIRS.iter().filter(|_| self.system) {
                    let dir = match (dir.strip_prefix("~/"), &home) {
                        (Some(rest), Some(home)) => home.join(rest),
                        (Some(_), None) => continue,
                        (None, _) => PathBuf::from(dir),
                    };
                    index.walk(&dir, 0);
                }
                // Directory listings come in no fixed order; ties in every
                // ranking below resolve by path, so selection is the same
                // on every run.
                index
                    .faces
                    .sort_by(|a, b| a.path.cmp(&b.path).then(a.index.cmp(&b.index)));
                log::debug!(
                    "system_fonts: indexed {} faces in {} directories",
                    index.faces.len(),
                    index.directories.len()
                );
                index
            })
            .faces
    }

    /// The installed face that best serves `style` for `script`: among
    /// the faces whose `cmap` maps the script's sample character (those
    /// declaring the script's code page first, every face when none of
    /// those maps it), the least by [`select_key`].  Code-page bits alone
    /// are not trusted: faces declare scripts they carry no glyphs for.
    fn select(&self, script: Script, style: SubstituteStyle) -> Option<usize> {
        let faces = self.faces();
        let sample = script_sample(script);
        let declaring: Vec<usize> = (0..faces.len())
            .filter(|&i| faces[i].covers(script) && faces[i].maps(sample))
            .collect();
        let pool: Vec<usize> = if declaring.is_empty() {
            (0..faces.len())
                .filter(|&i| faces[i].maps(sample))
                .collect()
        } else {
            declaring
        };
        pool.into_iter()
            .min_by_key(|&i| select_key(&faces[i], style, script))
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
    pub(crate) fn system_program(&self, style: SubstituteStyle) -> Option<Arc<FontProgram>> {
        let script = style.script?;
        self.select(script, style)
            .and_then(|index| self.face_program(index))
    }

    /// The installed program that draws the character `c` for a font of
    /// `style` whose substitute lacks it: among the faces whose `cmap` maps
    /// `c`, the first visible one by [`Face::rank`] for the character's
    /// script (listed for the script when it has one, serif class,
    /// preference, weight, slant), ties
    /// going to the earlier path; then, with the `bundled-fonts` feature,
    /// a bundled face that maps `c`; then a hidden installed face.  `None`
    /// when no face maps `c`.
    pub(crate) fn fallback_program(
        &self,
        style: SubstituteStyle,
        c: u32,
    ) -> Option<Arc<FontProgram>> {
        let faces = self.faces();
        let script = Script::of_char(c);
        let key: RankKey = (style.family, style.bold, style.italic, script);
        let order = {
            let mut orders = self.orders.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(orders.entry(key).or_insert_with(|| {
                let mut order: Vec<usize> = (0..faces.len()).collect();
                order.sort_by_key(|&i| faces[i].rank(style, script, false));
                order.into()
            }))
        };
        let installed = |hidden: bool| {
            order
                .iter()
                .filter(|&&i| faces[i].hidden() == hidden && faces[i].maps(c))
                .find_map(|&i| {
                    self.face_program(i)
                        .filter(|program| program.char_index(c) != 0)
                })
        };
        let visible = installed(false);
        #[cfg(feature = "bundled-fonts")]
        let visible = visible.or_else(|| super::bundled_fonts::bundled_program(style, c));
        visible.or_else(|| installed(true))
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

/// The faces of a library and the directories walked to find them.
#[derive(Default)]
struct Index {
    /// Every sfnt face found.
    faces: Vec<Face>,
    /// Each directory walked, with its modification time then; `None` for
    /// a root that could not be read.
    directories: Vec<(PathBuf, Option<SystemTime>)>,
}

impl Index {
    /// Add the faces of the font files in `dir` and, to [`MAX_DEPTH`],
    /// its subdirectories, recording each directory walked.
    fn walk(&mut self, dir: &Path, depth: u32) {
        self.directories.push((dir.to_path_buf(), modified(dir)));
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if depth < MAX_DEPTH {
                    self.walk(&path, depth + 1);
                }
                continue;
            }
            let is_sfnt = path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                e.eq_ignore_ascii_case("ttf")
                    || e.eq_ignore_ascii_case("otf")
                    || e.eq_ignore_ascii_case("ttc")
            });
            if is_sfnt {
                self.faces.extend(
                    face_infos(&path)
                        .into_iter()
                        .map(|info| Face::new(&path, info))
                        .filter(|face| !PLACEHOLDER_FAMILIES.contains(&face.family.as_str())),
                );
            }
        }
    }
}

/// The modification time of `dir`, `None` when it cannot be read.
fn modified(dir: &Path) -> Option<SystemTime> {
    std::fs::metadata(dir).and_then(|m| m.modified()).ok()
}

/// A character every face serving `script` maps: a kana, a common
/// ideograph, a syllable or a letter of the script.
const fn script_sample(script: Script) -> u32 {
    match script {
        Script::Japanese => 0x3042,
        Script::SimplifiedChinese | Script::TraditionalChinese => 0x4E2D,
        Script::Korean => 0xAC00,
        Script::Arabic => 0x0627,
        Script::Thai => 0x0E01,
    }
}

/// Whether `family` contains `name` as whole words: bounded on each side
/// by the string's end or a space.
fn names_words(family: &str, name: &str) -> bool {
    family.match_indices(name).any(|(at, _)| {
        let before = family[..at].chars().next_back();
        let after = family[at + name.len()..].chars().next();
        before.is_none_or(|c| c == ' ') && after.is_none_or(|c| c == ' ')
    })
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

    /// The best face of a candidate list for `style` under the preference
    /// list of `script`.
    fn best(faces: &[Face], script: Script, style: SubstituteStyle) -> &str {
        &faces
            .iter()
            .min_by_key(|f| select_key(f, style, script))
            .expect("candidates")
            .family
    }

    /// The installed Japanese faces of macOS: a named Hiragino family wins
    /// over Arial Unicode MS, which ties with it on class and weight, and
    /// over the hidden interface face of the same design.
    #[test]
    fn macos_japanese_faces_rank_hiragino_first() {
        let jp = 1 << 17;
        let faces = [
            face("Arial Unicode MS", 8 << 8, 400, jp),
            face(".Hiragino Kaku Gothic Interface", 0, 400, jp),
            face("Hiragino Sans GB", 0, 300, jp),
            face("Hiragino Sans", 0, 400, jp),
            face("Hiragino Mincho ProN", 0, 300, jp),
        ];
        let sans = SubstituteStyle {
            script: Some(Script::Japanese),
            ..SubstituteStyle::default()
        };
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..sans
        };
        assert_eq!(best(&faces, Script::Japanese, sans), "hiragino sans");
        assert_eq!(
            best(&faces, Script::Japanese, serif),
            "hiragino mincho pron"
        );
        // With only a hidden face and an unlisted visible one, the visible
        // one is drawn.
        let faces = [
            face(".Hiragino Kaku Gothic Interface", 0, 400, jp),
            face("Arial Unicode MS", 8 << 8, 400, jp),
        ];
        assert_eq!(best(&faces, Script::Japanese, sans), "arial unicode ms");
    }

    /// A family extending a listed name by a word is the longer design:
    /// "Hiragino Sans GB", a Chinese design, is not the Japanese "Hiragino
    /// Sans"; and a name matches only as whole words.
    #[test]
    fn listed_designs_match_whole_names() {
        let (japanese, chinese) = (Some(Script::Japanese), Some(Script::SimplifiedChinese));
        let unlisted = |script| preferred_names(script).len();
        let gb = face("Hiragino Sans GB", 0, 400, 0);
        assert_eq!(gb.preference(japanese), unlisted(japanese));
        assert!(gb.preference(chinese) < unlisted(chinese));
        assert!(face("Hiragino Sans", 0, 400, 0).preference(japanese) < unlisted(japanese));
        assert!(!names_words("notosans", "noto sans"));
        assert!(names_words("noto sans cjk jp", "noto sans"));
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

    /// For a character of no script, the serif class leads the fallback
    /// order: a listed sans-serif family does not displace an unlisted
    /// serif one for serif text.
    #[test]
    fn a_character_of_no_script_falls_back_by_serif_class_first() {
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..SubstituteStyle::default()
        };
        let faces = [
            face("DejaVu Sans", 8 << 8, 400, 0),
            face("Liberation Serif", 2 << 8, 400, 0),
        ];
        let first = faces
            .iter()
            .min_by_key(|f| f.rank(serif, None, false))
            .expect("candidates");
        assert_eq!(first.family, "liberation serif");
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

    /// A font installed into a library's directory while the process runs
    /// is seen by the next caller: the directory's change makes the
    /// indexed library stale, and `shared` replaces it.  An unchanged
    /// directory keeps its library.
    #[test]
    fn fonts_installed_after_indexing_are_seen() {
        let dir = std::env::temp_dir().join(format!("rr-font-install-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).expect("dir");
        // A modification time in the past, so that the install below
        // changes it whatever the file system's timestamp granularity.
        for d in [&dir, &sub] {
            std::fs::File::open(d)
                .and_then(|f| {
                    f.set_times(std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))
                })
                .expect("set mtime");
        }
        let before = FontLibrary::shared(std::slice::from_ref(&dir), false);
        assert!(before.faces().is_empty());
        assert!(Arc::ptr_eq(
            &before,
            &FontLibrary::shared(std::slice::from_ref(&dir), false)
        ));
        let ahem = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../font_test_data/test_data/ttf/ahem.ttf"
        );
        let _bytes = std::fs::copy(ahem, sub.join("ahem.ttf")).expect("install");
        let after = FontLibrary::shared(std::slice::from_ref(&dir), false);
        let indexed = after.faces().len();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!Arc::ptr_eq(&before, &after), "re-indexed");
        assert_eq!(indexed, 1);
        assert!(before.faces().is_empty(), "the old library is unchanged");
    }
}
