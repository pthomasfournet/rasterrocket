//! Tests of the installed-font library.

use super::*;

fn face(family: &str, class: i16, weight: u16, pages: u32) -> Face {
    Face::new(
        Path::new("/nonexistent"),
        FaceInfo {
            index: 0,
            family: family.to_owned(),
            postscript: None,
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
/// resolve; a serif request prefers a serif design whenever one of the
/// installed Japanese faces is serif, and otherwise takes a sans one.
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
    let serif_installed = library
        .faces()
        .iter()
        .any(|f| f.covers(Script::Japanese) && f.serif == Some(true));
    let sans_installed = library
        .faces()
        .iter()
        .any(|f| f.covers(Script::Japanese) && f.serif == Some(false));
    if sans_installed {
        assert_ne!(sans_face.serif, Some(true), "{}", sans_face.family);
    }
    if serif_installed {
        assert_ne!(serif_face.serif, Some(false), "{}", serif_face.family);
    }
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
        crate::test_helpers::backdate_dir(d);
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

/// A temporary directory, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A symbolic link to a directory is followed, but a directory reached
/// through several paths, or through a link to an ancestor, is read
/// once.
#[cfg(unix)]
#[test]
fn the_walk_reads_each_directory_once() {
    use std::os::unix::fs::symlink;
    let tmp = TempDir::new("rr-font-walk");
    let (root, other) = (tmp.0.join("root"), tmp.0.join("other"));
    std::fs::create_dir_all(root.join("a")).expect("dir");
    std::fs::create_dir_all(&other).expect("dir");
    std::fs::write(root.join("a/x.ttf"), b"").expect("file");
    std::fs::write(root.join("a/notes.txt"), b"").expect("file");
    std::fs::write(other.join("y.OTF"), b"").expect("file");
    symlink(&root, root.join("a/up")).expect("link");
    symlink(root.join("a"), root.join("b")).expect("link");
    symlink(&other, root.join("c")).expect("link");
    let walk = walk_font_dirs(&[root.clone(), root.join("a")], MAX_FONT_DIRS);
    let dirs: Vec<&Path> = walk.directories.iter().map(|(d, _)| d.as_path()).collect();
    assert_eq!(dirs, [root.as_path(), &root.join("a"), &root.join("c")]);
    assert!(walk.directories.iter().all(|(_, stamp)| stamp.is_some()));
    assert_eq!(walk.files, [root.join("a/x.ttf"), root.join("c/y.OTF")]);
}

/// The walk stops at its directory limit.
#[test]
fn the_walk_stops_at_the_directory_limit() {
    let tmp = TempDir::new("rr-font-walk-limit");
    std::fs::create_dir_all(tmp.0.join("a/b")).expect("dir");
    std::fs::write(tmp.0.join("top.ttc"), b"").expect("file");
    std::fs::write(tmp.0.join("a/b/deep.ttf"), b"").expect("file");
    let walk = walk_font_dirs(std::slice::from_ref(&tmp.0), 2);
    assert_eq!(walk.directories.len(), 2);
    assert_eq!(walk.files, [tmp.0.join("top.ttc")]);
    let walk = walk_font_dirs(std::slice::from_ref(&tmp.0), MAX_FONT_DIRS);
    assert_eq!(
        walk.files,
        [tmp.0.join("a/b/deep.ttf"), tmp.0.join("top.ttc")]
    );
}

/// A file is taken by its exact name, below a root only, the first by
/// path.
#[test]
fn named_files_are_found_below_the_roots() {
    let files = [
        "/elsewhere/LiberationSans-Regular.ttf",
        "/r/a/LiberationSans-Regular.ttf",
        "/r/b/LiberationSans-Regular.ttf",
        "/r/b/LiberationSansNarrow-Regular.ttf",
    ]
    .map(PathBuf::from);
    let found = first_named(&files, &["/r"], "LiberationSans-Regular.ttf");
    assert_eq!(found, Some(Path::new("/r/a/LiberationSans-Regular.ttf")));
    assert_eq!(first_named(&files, &["/r"], "Liberation"), None);
    assert_eq!(first_named(&files, &[], "LiberationSans-Regular.ttf"), None);
}

/// No standard directory depends on the working directory.
#[test]
fn standard_directories_are_absolute() {
    for dir in FONT_DIRS.iter().chain(UNIX_FONT_ROOTS) {
        assert!(
            dir.starts_with("~/") || Path::new(dir).is_absolute(),
            "{dir}"
        );
    }
}

/// Every substitute search of a library records the directories it
/// looks in once between them.
#[test]
fn substitute_searches_stamp_each_directory_once() {
    let tmp = TempDir::new("rr-font-stamps");
    let library = FontLibrary::new(vec![tmp.0.clone()], false);
    for family in [
        SubstituteFamily::Sans,
        SubstituteFamily::Serif,
        SubstituteFamily::Mono,
    ] {
        for (bold, italic) in [(false, false), (false, true), (true, false), (true, true)] {
            let style = SubstituteStyle {
                family,
                bold,
                italic,
                ..SubstituteStyle::default()
            };
            let _program = super::super::substitutes::substitute_program(style, None, &library);
        }
    }
    let stamps = library.stamps.lock().expect("stamps");
    assert_eq!(stamps.keys().collect::<Vec<_>>(), [&tmp.0]);
}
