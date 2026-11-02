//! Substitute font programs for fonts without an embedded program.
//!
//! A font without a program is drawn with a substitute of its family class
//! and variant ([`SubstituteStyle`]): the installed metric-compatible face
//! (Liberation, Arial, Times New Roman, Courier New) when one is found in
//! an extra directory of the [`FontLibrary`] or on the fixed search path,
//! else the base-14 face compiled into the binary (`fonts/foxit/`).  The
//! Symbol and Dingbats families always draw with the compiled-in faces,
//! whose charsets carry the glyph names of their encodings.  An installed
//! substitute is resolved once per library (a library whose search
//! directories change is replaced), a compiled-in one once per process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use font::program::FontProgram;

use super::system_fonts::FontLibrary;
use crate::resources::font_substitute::{SubstituteFamily, SubstituteStyle};

/// Number of rows of [`INSTALLED`]: the three Latin family classes in
/// their four variants.
pub(crate) const INSTALLED_ROWS: usize = 12;

/// Row of [`EMBEDDED`] of the Symbol family.
const SYMBOL_ROW: usize = INSTALLED_ROWS;

/// Row of [`EMBEDDED`] of the Dingbats family.
const DINGBATS_ROW: usize = INSTALLED_ROWS + 1;

/// Installed typefaces per [`SubstituteStyle`], indexed by
/// [`substitute_index`]: `[metric-compatible free family stem, macOS
/// supplemental file stem, Windows file stem]`.
const INSTALLED: [[&str; 3]; INSTALLED_ROWS] = [
    ["LiberationSans-Regular", "Arial", "arial"],
    ["LiberationSans-Italic", "Arial Italic", "ariali"],
    ["LiberationSans-Bold", "Arial Bold", "arialbd"],
    ["LiberationSans-BoldItalic", "Arial Bold Italic", "arialbi"],
    ["LiberationSerif-Regular", "Times New Roman", "times"],
    ["LiberationSerif-Italic", "Times New Roman Italic", "timesi"],
    ["LiberationSerif-Bold", "Times New Roman Bold", "timesbd"],
    [
        "LiberationSerif-BoldItalic",
        "Times New Roman Bold Italic",
        "timesbi",
    ],
    ["LiberationMono-Regular", "Courier New", "cour"],
    ["LiberationMono-Italic", "Courier New Italic", "couri"],
    ["LiberationMono-Bold", "Courier New Bold", "courbd"],
    [
        "LiberationMono-BoldItalic",
        "Courier New Bold Italic",
        "courbi",
    ],
];

/// Compiled-in base-14 programs: the twelve rows of [`INSTALLED`], then
/// Symbol and Dingbats.
const EMBEDDED: [&[u8]; INSTALLED_ROWS + 2] = [
    include_bytes!("../../fonts/foxit/Sans.cff"),
    include_bytes!("../../fonts/foxit/SansItalic.cff"),
    include_bytes!("../../fonts/foxit/SansBold.cff"),
    include_bytes!("../../fonts/foxit/SansBoldItalic.cff"),
    include_bytes!("../../fonts/foxit/Serif.cff"),
    include_bytes!("../../fonts/foxit/SerifItalic.cff"),
    include_bytes!("../../fonts/foxit/SerifBold.cff"),
    include_bytes!("../../fonts/foxit/SerifBoldItalic.cff"),
    include_bytes!("../../fonts/foxit/Fixed.cff"),
    include_bytes!("../../fonts/foxit/FixedItalic.cff"),
    include_bytes!("../../fonts/foxit/FixedBold.cff"),
    include_bytes!("../../fonts/foxit/FixedBoldItalic.cff"),
    include_bytes!("../../fonts/foxit/Symbol.cff"),
    include_bytes!("../../fonts/foxit/Dingbats.cff"),
];

/// Row of [`EMBEDDED`] for `style`; rows below `INSTALLED.len()` are
/// also rows of [`INSTALLED`].  The Symbol and Dingbats families have one
/// face each.
const fn substitute_index(style: SubstituteStyle) -> usize {
    let family = match style.family {
        SubstituteFamily::Sans => 0,
        SubstituteFamily::Serif => 1,
        SubstituteFamily::Mono => 2,
        SubstituteFamily::Symbol => return SYMBOL_ROW,
        SubstituteFamily::Dingbats => return DINGBATS_ROW,
    };
    family * 4 + (style.bold as usize) * 2 + (style.italic as usize)
}

/// Candidate paths for one [`INSTALLED`] row, in search order: the free
/// family's file in each of the extra directories `dirs`, then, when
/// `system` is set, the fixed locations and the free family's file found
/// under [`UNIX_FONT_ROOTS`].  Each directory a candidate lies in, and
/// each directory searched below the roots, is reported to `watch`.
fn installed_candidates(
    [stem, mac, win]: [&str; 3],
    dirs: &[PathBuf],
    system: bool,
    watch: &mut dyn FnMut(&Path),
) -> Vec<PathBuf> {
    let file = format!("{stem}.ttf");
    let fixed = [
        format!("/usr/share/fonts/truetype/liberation/{file}"),
        format!("/usr/share/fonts/liberation/{file}"),
        format!("/System/Library/Fonts/Supplemental/{mac}.ttf"),
        format!("C:/Windows/Fonts/{win}.ttf"),
    ];
    let mut candidates: Vec<PathBuf> = dirs
        .iter()
        .map(|dir| dir.join(&file))
        .chain(fixed.into_iter().filter(|_| system).map(PathBuf::from))
        .collect();
    for candidate in &candidates {
        if let Some(dir) = candidate.parent() {
            watch(dir);
        }
    }
    if system {
        let roots = UNIX_FONT_ROOTS.map(PathBuf::from);
        candidates.extend(files_under(&roots, "Liberation", ".ttf", watch).remove(&file));
    }
    candidates
}

/// The directories whose subdirectories distributions install font
/// packages in, each under its own name (`truetype/liberation2`,
/// `liberation-sans`, …).
const UNIX_FONT_ROOTS: [&str; 2] = ["/usr/share/fonts", "/usr/local/share/fonts"];

/// Deepest directory level below a root that [`files_under`] reads.
const MAX_FONT_DIR_DEPTH: usize = 3;

/// The files under `roots`, to [`MAX_FONT_DIR_DEPTH`] levels of
/// directories, whose names start with `prefix` and end with `suffix`, by
/// file name; the first found keeps a name.  Unreadable directories are
/// skipped.  Each directory is reported to `watch` before it is read.
fn files_under(
    roots: &[PathBuf],
    prefix: &str,
    suffix: &str,
    watch: &mut dyn FnMut(&Path),
) -> HashMap<String, PathBuf> {
    let mut found = HashMap::new();
    let mut pending: Vec<(PathBuf, usize)> = roots.iter().map(|r| (r.clone(), 0)).collect();
    while let Some((dir, depth)) = pending.pop() {
        watch(&dir);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                if depth < MAX_FONT_DIR_DEPTH {
                    pending.push((path, depth + 1));
                }
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.starts_with(prefix)
                && name.ends_with(suffix)
            {
                let _ = found.entry(name.to_owned()).or_insert(path);
            }
        }
    }
    found
}

/// The installed face of row `row`, searched for in the extra directories
/// `dirs` first and then, when `system` is set, the fixed locations: the
/// first candidate that exists and parses; `None` when none does.  The
/// directories searched are reported to `watch`.
fn installed_program(
    row: usize,
    dirs: &[PathBuf],
    system: bool,
    watch: &mut dyn FnMut(&Path),
) -> Option<Arc<FontProgram>> {
    installed_candidates(INSTALLED[row], dirs, system, watch)
        .into_iter()
        .filter(|path| path.exists())
        .find_map(|path| {
            std::fs::read(&path)
                .map_err(font::engine::LoadError::Io)
                .and_then(|data| FontProgram::parse(data, 0))
                .inspect_err(|e| {
                    log::debug!("substitutes: installed font {}: {e}", path.display());
                })
                .ok()
        })
        .map(Arc::new)
}

/// The compiled-in program `bytes`, parsed.  `None`, logged, only if the
/// bundled data is unusable.
fn embedded_program(bytes: &'static [u8]) -> Option<Arc<FontProgram>> {
    FontProgram::parse(bytes.to_vec(), 0)
        .inspect_err(|e| log::error!("substitutes: bundled font program: {e}"))
        .ok()
        .map(Arc::new)
}

/// The substitute program for `style` from `library`: an installed face
/// of its script ([`FontLibrary::system_program`]) when it has one, else
/// the installed metric-compatible face of that exact family and variant,
/// else the compiled-in base-14 face.
pub(crate) fn substitute_program(
    style: SubstituteStyle,
    library: &FontLibrary,
) -> Option<Arc<FontProgram>> {
    static EMBEDDED_CACHE: [OnceLock<Option<Arc<FontProgram>>>; EMBEDDED.len()] =
        [const { OnceLock::new() }; EMBEDDED.len()];
    if style.script.is_some()
        && let Some(program) = library.system_program(style)
    {
        return Some(program);
    }
    let row = substitute_index(style);
    (row < INSTALLED_ROWS)
        .then(|| {
            library.substitute(row, |dirs, system, watch| {
                installed_program(row, dirs, system, watch)
            })
        })
        .flatten()
        .or_else(|| {
            EMBEDDED_CACHE[row]
                .get_or_init(|| embedded_program(EMBEDDED[row]))
                .clone()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use font::engine::{FaceParams, FontEngine};

    /// [`substitute_program`] from the library of the standard directories.
    fn substitute_program_default(style: SubstituteStyle) -> Option<Arc<FontProgram>> {
        substitute_program(style, &FontLibrary::shared(&[], true))
    }

    /// The Latin family classes in every variant.
    fn latin_styles() -> impl Iterator<Item = SubstituteStyle> {
        [
            SubstituteFamily::Sans,
            SubstituteFamily::Serif,
            SubstituteFamily::Mono,
        ]
        .into_iter()
        .flat_map(|family| {
            [(false, false), (false, true), (true, false), (true, true)]
                .into_iter()
                .map(move |(bold, italic)| SubstituteStyle {
                    family,
                    bold,
                    italic,
                    script: None,
                    standard: false,
                })
        })
    }

    fn all_styles() -> impl Iterator<Item = SubstituteStyle> {
        latin_styles().chain(
            [SubstituteFamily::Symbol, SubstituteFamily::Dingbats]
                .into_iter()
                .map(|family| SubstituteStyle {
                    family,
                    ..SubstituteStyle::default()
                }),
        )
    }

    /// Font files are found in the subdirectories distributions use, to
    /// the depth limit, and the first found keeps its name.
    #[test]
    fn font_files_are_found_in_package_subdirectories() {
        let root = std::env::temp_dir().join(format!("rr-font-roots-{}", std::process::id()));
        let deep = root.join("a/b/c/d");
        std::fs::create_dir_all(root.join("liberation-sans")).expect("dir");
        std::fs::create_dir_all(&deep).expect("dir");
        for path in [
            root.join("liberation-sans/LiberationSans-Regular.ttf"),
            root.join("liberation-sans/Other.ttf"),
            deep.join("LiberationMono-Regular.ttf"),
        ] {
            std::fs::write(path, b"").expect("file");
        }
        let found = files_under(
            std::slice::from_ref(&root),
            "Liberation",
            ".ttf",
            &mut |_| {},
        );
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(
            found.get("LiberationSans-Regular.ttf"),
            Some(&root.join("liberation-sans/LiberationSans-Regular.ttf"))
        );
        assert_eq!(found.len(), 1, "{found:?}");
    }

    /// A metric-compatible face installed into a searched directory after
    /// a substitute search came up empty is found by the next caller: the
    /// directory's change makes the library stale, though the library
    /// never built its face index.
    #[test]
    fn substitutes_installed_after_a_failed_search_are_found() {
        let dir =
            std::env::temp_dir().join(format!("rr-substitute-install-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        // A modification time in the past, so that the install below
        // changes it whatever the file system's timestamp granularity.
        std::fs::File::open(&dir)
            .and_then(|f| {
                f.set_times(
                    std::fs::FileTimes::new().set_modified(std::time::SystemTime::UNIX_EPOCH),
                )
            })
            .expect("set mtime");
        let style = SubstituteStyle::default();
        let before = FontLibrary::shared(std::slice::from_ref(&dir), false);
        let compiled_in = substitute_program(style, &before).expect("compiled-in face");
        assert_eq!(compiled_in.name_index("glyph243"), 0);
        let ahem = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../font_test_data/test_data/ttf/ahem.ttf"
        );
        let _bytes = std::fs::copy(ahem, dir.join("LiberationSans-Regular.ttf")).expect("install");
        let after = FontLibrary::shared(std::slice::from_ref(&dir), false);
        let installed = substitute_program(style, &after);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!Arc::ptr_eq(&before, &after), "the library was replaced");
        let installed = installed.expect("installed face");
        assert_ne!(
            installed.name_index("glyph243"),
            0,
            "the installed file draws"
        );
    }

    #[test]
    fn installed_rows_match_styles() {
        for style in latin_styles() {
            let stem = INSTALLED[substitute_index(style)][0];
            let family_name = match style.family {
                SubstituteFamily::Sans => "Sans-",
                SubstituteFamily::Serif => "Serif-",
                _ => "Mono-",
            };
            assert!(stem.contains(family_name), "{style:?} -> {stem}");
            assert_eq!(stem.contains("Bold"), style.bold, "{style:?} -> {stem}");
            assert_eq!(stem.contains("Italic"), style.italic, "{style:?} -> {stem}");
        }
    }

    /// Every compiled-in program parses and draws a glyph of its
    /// encoding with ink (`A` for the Latin faces, `alpha` for Symbol,
    /// `a1` for Dingbats), so a host without any installed font still
    /// renders text.
    #[test]
    fn embedded_programs_draw() {
        let engine = FontEngine::new(true);
        for (row, bytes) in EMBEDDED.iter().enumerate() {
            let program = embedded_program(bytes).unwrap_or_else(|| panic!("row {row} parses"));
            let name = match row {
                SYMBOL_ROW => "alpha",
                DINGBATS_ROW => "a1",
                _ => "A",
            };
            let gid = program.name_index(name);
            assert_ne!(gid, 0, "row {row} names {name}");
            let face = engine
                .load_program_face(
                    program,
                    FaceParams {
                        code_to_gid: vec![],
                        mat: [20.0, 0.0, 0.0, 20.0],
                    },
                )
                .expect("face at 20 px");
            let glyph = face.make_glyph_by_gid(gid, 0).expect("glyph bitmap");
            assert!(glyph.data.iter().any(|&c| c != 0), "row {row} draws {name}");
        }
    }

    /// The Symbol and Dingbats families never take an installed face:
    /// their codes name glyphs only the compiled-in faces carry.
    #[test]
    fn symbol_and_dingbats_are_always_compiled_in() {
        let symbol = substitute_program_default(SubstituteStyle {
            family: SubstituteFamily::Symbol,
            ..SubstituteStyle::default()
        })
        .expect("symbol");
        let dingbats = substitute_program_default(SubstituteStyle {
            family: SubstituteFamily::Dingbats,
            ..SubstituteStyle::default()
        })
        .expect("dingbats");
        assert_eq!(symbol.data(), EMBEDDED[SYMBOL_ROW]);
        assert_eq!(dingbats.data(), EMBEDDED[DINGBATS_ROW]);
        assert_ne!(symbol.name_index("therefore"), 0);
        assert_ne!(dingbats.name_index("a73"), 0);
    }

    /// Every style resolves to a program, installed or compiled in.
    #[test]
    fn every_style_has_a_substitute() {
        for style in all_styles() {
            assert!(substitute_program_default(style).is_some(), "{style:?}");
        }
    }

    /// A serif request resolves to a different program than the sans-serif
    /// default.
    #[test]
    fn serif_substitute_is_not_the_sans_program() {
        let serif = SubstituteStyle {
            family: SubstituteFamily::Serif,
            ..SubstituteStyle::default()
        };
        let serif = substitute_program_default(serif).expect("serif");
        let sans = substitute_program_default(SubstituteStyle::default()).expect("sans");
        assert!(!Arc::ptr_eq(&serif, &sans));
    }
}
