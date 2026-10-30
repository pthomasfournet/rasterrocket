//! Font crate tests: every supported program format through the public
//! face API, glyph routing, and robustness on hostile input.

mod cid_type1;
mod formats;
mod hinting;
mod sfnt;
mod type1;

use raster::path::Path;

use crate::bitmap::GlyphBitmap;
use crate::engine::{FaceParams, FontEngine, LoadError};
use crate::face::FontFace;
use crate::program::FontFormat;

/// `NotoSansJP` (from the vendored font-test-data) subset to `A`, `M`, `Z`
/// with CIDs kept: glyph indices 1–3 carry CIDs 34, 46 and 59.
const CID_SUBSET: &[u8] = include_bytes!("data/notosansjp-cid-amz.otf");

/// A face for `data` at `px` pixels per em, identity charmap routing.
fn face_at(data: &[u8], px: f64) -> FontFace {
    try_face(data, [px, 0.0, 0.0, px]).expect("font loads")
}

fn try_face(data: &[u8], mat: [f64; 4]) -> Result<FontFace, LoadError> {
    FontEngine::new(true).load_memory_face(
        data.to_vec(),
        0,
        FaceParams {
            code_to_gid: Vec::new(),
            mat,
        },
    )
}

/// `(min_x, min_y, max_x, max_y)` over every point of `path`.
fn bounds(path: &Path) -> (f64, f64, f64, f64) {
    path.pts.iter().fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(a, b, c, d), p| (a.min(p.x), b.min(p.y), c.max(p.x), d.max(p.y)),
    )
}

fn subpaths(path: &Path) -> usize {
    path.flags.iter().filter(|f| f.is_first()).count()
}

/// Total coverage of a bitmap in pixels.
fn ink(bmp: &GlyphBitmap) -> f64 {
    bmp.data.iter().map(|&v| f64::from(v) / 255.0).sum()
}

fn same_bitmap(a: &GlyphBitmap, b: &GlyphBitmap) -> bool {
    (a.x_off, a.y_off, a.width, a.height, &a.data) == (b.x_off, b.y_off, b.width, b.height, &b.data)
}

/// Every font program the tests know, with its expected format.
fn sample_programs() -> Vec<(&'static str, Vec<u8>, FontFormat)> {
    vec![
        (
            "tinos (TrueType)",
            font_test_data::TINOS_SUBSET.to_vec(),
            FontFormat::Sfnt,
        ),
        (
            "symbolic box (TrueType)",
            sfnt::symbolic_box_font(),
            FontFormat::Sfnt,
        ),
        (
            "noto serif display (OpenType CFF)",
            font_test_data::NOTO_SERIF_DISPLAY_TRIMMED.to_vec(),
            FontFormat::Sfnt,
        ),
        (
            "noto sans jp (OpenType CID CFF)",
            font_test_data::NOTO_SANS_JP_CFF.to_vec(),
            FontFormat::Sfnt,
        ),
        (
            "bare CFF",
            sfnt::table(font_test_data::NOTO_SERIF_DISPLAY_TRIMMED, *b"CFF "),
            FontFormat::Cff,
        ),
        (
            "bare CID CFF",
            sfnt::table(font_test_data::NOTO_SANS_JP_CFF, *b"CFF "),
            FontFormat::Cff,
        ),
        (
            "noto serif PFA",
            font_test_data::type1::NOTO_SERIF_REGULAR_SUBSET_PFA.to_vec(),
            FontFormat::Type1,
        ),
        (
            "noto serif PFB",
            font_test_data::type1::NOTO_SERIF_REGULAR_SUBSET_PFB.to_vec(),
            FontFormat::Type1,
        ),
        ("synthetic PFA", type1::synthetic_pfa(), FontFormat::Type1),
        (
            "synthetic binary eexec",
            type1::synthetic_binary_eexec(),
            FontFormat::Type1,
        ),
        ("synthetic PFB", type1::synthetic_pfb(), FontFormat::Type1),
        (
            "hint-assembled (TrueType)",
            hinting::BY_TABLES.to_vec(),
            FontFormat::Sfnt,
        ),
        (
            "CID-keyed Type 1",
            cid_type1::CID_TYPE1.to_vec(),
            FontFormat::Type1,
        ),
    ]
}

#[test]
fn formats_are_detected_from_leading_bytes() {
    for (name, data, format) in sample_programs() {
        assert_eq!(FontFormat::detect(&data), Some(format), "{name}");
        assert_eq!(face_at(&data, 20.0).program().format(), format, "{name}");
    }
    assert_eq!(FontFormat::detect(b""), None);
    assert_eq!(FontFormat::detect(b"<html>"), None);
    assert!(matches!(
        try_face(b"GIF89a", [10.0, 0.0, 0.0, 10.0]),
        Err(LoadError::UnsupportedFormat)
    ));
}

#[test]
fn engine_shares_programs_between_faces() {
    let data = font_test_data::TINOS_SUBSET.to_vec();
    let mut engine = FontEngine::new(true);
    let params = |px: f64| FaceParams {
        code_to_gid: Vec::new(),
        mat: [px, 0.0, 0.0, px],
    };
    let a = engine
        .load_memory_face(data.clone(), 0, params(12.0))
        .unwrap();
    let b = engine.load_memory_face(data, 0, params(30.0)).unwrap();
    assert!(
        std::ptr::eq(a.program(), b.program()),
        "same bytes must share one program"
    );
    assert_ne!(a.id, b.id, "each face gets its own id");
    let other = engine
        .load_memory_face(sfnt::symbolic_box_font(), 0, params(12.0))
        .unwrap();
    assert!(!std::ptr::eq(a.program(), other.program()));
}

#[test]
fn degenerate_matrices_are_rejected() {
    let data = font_test_data::TINOS_SUBSET;
    for mat in [
        [0.0; 4],
        [10.0, 0.0, 0.0, 0.4],
        [f64::NAN, 0.0, 0.0, 10.0],
        [10.0, 0.0, 0.0, f64::INFINITY],
        [10.0, 0.0, 0.0, 70_000.0],
    ] {
        assert!(
            matches!(try_face(data, mat), Err(LoadError::DegenerateSize)),
            "{mat:?}"
        );
    }
}

#[test]
fn oversized_glyphs_have_paths_but_no_bitmap() {
    assert!(!face_at(font_test_data::TINOS_SUBSET, 2_000.0).prefers_path_fill());
    let face = face_at(font_test_data::TINOS_SUBSET, 60_000.0);
    assert!(face.prefers_path_fill());
    let gid = face.raw_get_char_index(u32::from('A'));
    assert!(gid != 0);
    assert!(
        face.make_glyph_by_gid(gid, 0).is_none(),
        "beyond the rasteriser budget"
    );
    let path = face
        .glyph_path_by_gid(gid)
        .expect("outline still available");
    assert!(bounds(&path).3 > 30_000.0);
}

// ── Routing ───────────────────────────────────────────────────────────────────

/// A `code_to_gid` table steers [`FontFace::make_glyph`] but never the
/// `*_by_gid` methods, which composite fonts call with an already resolved
/// glyph selector.
#[test]
fn by_gid_bypasses_code_to_gid() {
    let data = font_test_data::TINOS_SUBSET;
    let probe = face_at(data, 40.0);
    let gid_a = probe.raw_get_char_index(u32::from('A'));
    let gid_b = probe.raw_get_char_index(u32::from('B'));
    assert!(gid_a != 0 && gid_b != 0 && gid_a != gid_b);

    let mut hijack = vec![0u32; 256];
    hijack[65] = gid_b;
    let face = FontEngine::new(true)
        .load_memory_face(
            data.to_vec(),
            0,
            FaceParams {
                code_to_gid: hijack,
                mat: [40.0, 0.0, 0.0, 40.0],
            },
        )
        .unwrap();
    let via_code = face.make_glyph(65, 0).expect("code 65 renders B");
    let via_gid = face.make_glyph_by_gid(gid_a, 0).expect("gid renders A");
    let b = probe.make_glyph_by_gid(gid_b, 0).unwrap();
    assert!(same_bitmap(&via_code, &b));
    assert!(!same_bitmap(&via_code, &via_gid));
    assert!(face.glyph_path_by_gid(gid_a).is_some());
}

/// With no remap table the character-code and glyph-index routes reach the
/// same glyph.
#[test]
fn by_gid_and_charmap_agree_for_identity_face() {
    let face = face_at(font_test_data::TINOS_SUBSET, 40.0);
    let gid = face.raw_get_char_index(u32::from('a'));
    assert!(gid != 0);
    let via_code = face.make_glyph(u32::from('a'), 0).unwrap();
    let via_gid = face.make_glyph_by_gid(gid, 0).unwrap();
    assert!(same_bitmap(&via_code, &via_gid));
}

/// Composite fonts pass untrusted glyph selectors straight through; out of
/// range ones must come back as missing glyphs, never a panic.
#[test]
fn out_of_range_glyphs_are_missing_not_panics() {
    for (name, data, _) in sample_programs() {
        let face = face_at(&data, 30.0);
        for bogus in [u32::MAX, 1_000_000, 65_535] {
            assert!(face.make_glyph_by_gid(bogus, 0).is_none(), "{name} {bogus}");
            assert!(face.glyph_path_by_gid(bogus).is_none(), "{name} {bogus}");
            if let Some(adv) = face.glyph_advance_by_gid(bogus) {
                assert!(adv.is_finite(), "{name} {bogus}");
            }
            let _ = face.make_glyph(bogus, 0);
        }
    }
}

#[test]
fn sub_pixel_offset_shifts_coverage_right() {
    let face = face_at(&sfnt::symbolic_box_font(), 40.0);
    let whole = face.make_glyph(7, 0).unwrap();
    let half = face.make_glyph(7, 2).unwrap();
    // The 20 px wide box spans 21 columns at a half-pixel offset, its edge
    // columns half covered; total coverage is unchanged up to the rounding
    // of ½ to 128/255.
    assert_eq!(whole.width, 20);
    assert_eq!(half.width, 21);
    assert_eq!(half.data[0], 128);
    assert!((ink(&whole) - ink(&half)).abs() < 0.5);
}

// ── Hostile input ─────────────────────────────────────────────────────────────

/// Exercise every entry point on a possibly broken program.
fn poke(data: &[u8]) {
    let Ok(face) = try_face(data, [24.0, 3.0, -2.0, 24.0]) else {
        return;
    };
    for gid in [0, 1, 2, 3, 5, 34, 200] {
        let _ = face.make_glyph_by_gid(gid, 1);
        let _ = face.glyph_path_by_gid(gid);
        let _ = face.glyph_advance_by_gid(gid);
    }
    for code in [0, 7, 65, 0xF041, 0x10_FFFF] {
        let _ = face.make_glyph(code, 0);
        let _ = face.raw_get_char_index(code);
    }
    let _ = face.raw_get_name_index("A");
    let _ = face.has_unicode_cmap();
}

#[test]
fn truncated_programs_do_not_panic() {
    for (_, data, _) in sample_programs() {
        let step = (data.len() / 97).max(1);
        for len in (0..data.len()).step_by(step) {
            poke(&data[..len]);
        }
    }
}

#[test]
fn corrupted_programs_do_not_panic() {
    // Deterministic xorshift so failures reproduce.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for (_, data, _) in sample_programs() {
        for _ in 0..150 {
            let mut mutated = data.clone();
            for _ in 0..=next() % 8 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "values are reduced modulo the data length / 256"
                )]
                let (at, byte) = ((next() % mutated.len() as u64) as usize, next() as u8);
                mutated[at] = byte;
            }
            poke(&mutated);
        }
    }
}

/// Glyph names resolve to Unicode through the Adobe Glyph List (version
/// 2.0, which carries no ZapfDingbats `aN` name) and the `uni` / `u`
/// conventions; a `.variant` suffix is ignored.
#[test]
fn glyph_names_resolve_to_unicode() {
    use crate::glyph_name_to_unicode;
    assert_eq!(glyph_name_to_unicode("A"), Some(0x41));
    assert_eq!(glyph_name_to_unicode("alpha"), Some(0x3B1));
    assert_eq!(glyph_name_to_unicode("therefore"), Some(0x2234));
    assert_eq!(glyph_name_to_unicode("arrowright"), Some(0x2192));
    assert_eq!(glyph_name_to_unicode("a1"), None);
    assert_eq!(glyph_name_to_unicode("afii57415"), Some(0x627));
    assert_eq!(glyph_name_to_unicode("uni20AC"), Some(0x20AC));
    assert_eq!(glyph_name_to_unicode("u1F600"), Some(0x1F600));
    assert_eq!(glyph_name_to_unicode("A.sc"), Some(0x41));
    assert_eq!(glyph_name_to_unicode("gid42"), None);
}

/// The `GSUB` `vert` feature maps a glyph to its vertical form; glyphs
/// without one, and programs without the feature, map to themselves.
#[test]
fn vertical_forms_come_from_gsub() {
    let program =
        crate::program::FontProgram::parse(include_bytes!("data/vertical-forms.ttf").to_vec(), 0)
            .expect("parse");
    assert_eq!(program.name_index("square"), 1);
    assert_eq!(program.name_index("tall"), 2);
    assert_eq!(program.vertical_gid(1), 2);
    assert_eq!(program.vertical_gid(2), 2);
    assert_eq!(program.vertical_gid(3), 3);
    assert_eq!(program.vertical_selector(1), 2);
    assert_eq!(program.vertical_selector(3), 3);
    let plain =
        crate::program::FontProgram::parse(font_test_data::AHEM.to_vec(), 0).expect("parse");
    assert_eq!(plain.vertical_gid(58), 58);
    let cid = crate::program::FontProgram::parse(CID_SUBSET.to_vec(), 0).expect("parse");
    assert_eq!(cid.vertical_selector(34), 34);
}

/// The `OS/2` and `name` summary of a file's faces is read without
/// loading the file; a file that is not an sfnt yields nothing.
#[test]
fn sfnt_face_infos_summarise_a_file() {
    use crate::sfnt_info::face_infos;
    let dir = std::env::temp_dir().join(format!("rrocket-sfnt-info-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ahem = dir.join("ahem.ttf");
    std::fs::write(&ahem, font_test_data::AHEM).expect("write");
    let infos = face_infos(&ahem);
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].index, 0);
    assert_eq!(infos[0].family, "Ahem");
    assert_eq!(infos[0].weight_class, 400);
    assert!(!infos[0].italic);
    let text = dir.join("not-a-font.ttf");
    std::fs::write(&text, b"hello").expect("write");
    assert!(face_infos(&text).is_empty());
    assert!(face_infos(&dir.join("missing.ttf")).is_empty());
    std::fs::remove_dir_all(&dir).expect("cleanup");
}
