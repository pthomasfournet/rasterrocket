//! One test group per font program format a PDF can embed.

use super::{bounds, face_at, ink, same_bitmap, sfnt, subpaths, type1};
use crate::engine::{FaceParams, FontEngine};

/// A face whose device space equals font units for 1000-unit fonts.
const UNITS_1000: f64 = 1000.0;

// ── TrueType (FontFile2) ──────────────────────────────────────────────────────

#[test]
fn truetype_glyph_has_expected_size_and_position() {
    // Tinos: 2048 units per em, cap height ≈ 1341 units.
    let face = face_at(font_test_data::TINOS_SUBSET, 40.0);
    assert!(face.has_unicode_cmap());
    let a = face.make_glyph(u32::from('A'), 0).expect("A renders");
    // 1341 / 2048 · 40 ≈ 26.2 px tall, sitting on the baseline.
    assert!((26..=28).contains(&a.height), "height {}", a.height);
    assert!((26..=28).contains(&a.y_off), "top {}", a.y_off);
    assert!(
        i64::from(a.height) - i64::from(a.y_off) <= 1,
        "bottom at the baseline"
    );
    assert!(ink(&a) > 100.0);
    let advance = face.glyph_advance(u32::from('A')).expect("hmtx advance");
    assert!((0.6..0.8).contains(&advance), "advance {advance} em");
    // A space has an advance but no outline.
    assert!(face.make_glyph(u32::from(' '), 0).is_none());
    assert!(face.glyph_advance(u32::from(' ')).unwrap() > 0.0);
}

#[test]
fn symbolic_subset_resolves_through_mac_roman() {
    // (1,0) + (3,0) and no Unicode charmap: code 7 is the subset's real
    // code → glyph mapping through the Macintosh Roman subtable.
    let face = face_at(&sfnt::symbolic_box_font(), 40.0);
    assert!(!face.has_unicode_cmap());
    let g = face.make_glyph(7, 0).expect("code 7 resolves");
    // 500×700 units at 40 px/em: exactly 20×28 fully covered pixels.
    assert_eq!((g.width, g.height, g.x_off, g.y_off), (20, 28, 0, 28));
    assert!(g.data.iter().all(|&v| v == 255));
}

#[test]
fn symbol_cmap_codes_retry_in_private_use_area() {
    // Only (3,0): its entries sit at 0xF000 + code (PDF §9.6.6.4).
    let face = face_at(&sfnt::symbol_only_box_font(), 40.0);
    assert_eq!(face.raw_get_char_index(7), 1);
    assert!(face.make_glyph(7, 0).is_some());
    assert!(face.make_glyph(8, 0).is_none());
}

#[test]
fn post_names_stand_in_for_a_missing_unicode_cmap() {
    // (1,0) only, but `post` names GID 1 `A`: codes resolve as Unicode
    // through the glyph names.
    let face = face_at(&sfnt::named_box_font(&sfnt::mac_only_cmap()), 40.0);
    assert!(face.has_unicode_cmap());
    assert_eq!(face.raw_get_char_index(u32::from('A')), 1);
    assert_eq!(face.raw_get_name_index("A"), 1);
    assert_eq!(face.raw_get_name_index("B"), 0);
}

#[test]
fn code_index_reads_byte_subtables_beneath_a_name_charmap() {
    // The charmap is Unicode from `post` names, so code 7 is U+0007 to it;
    // the (1,0) subtable still maps code 7 to GID 1.
    let named = face_at(&sfnt::named_box_font(&sfnt::mac_only_cmap()), 40.0);
    assert_eq!(named.raw_get_char_index(7), 0);
    assert_eq!(named.raw_get_code_index(7), 1);
    assert_eq!(named.raw_get_code_index(8), 0);
    // (3,0) entries at 0xF000 + code.
    let symbol = face_at(&sfnt::symbol_only_box_font(), 40.0);
    assert_eq!(symbol.raw_get_code_index(7), 1);
    // A Unicode subtable is not byte-keyed.
    let unicode = face_at(font_test_data::TINOS_SUBSET, 40.0);
    assert_eq!(unicode.raw_get_code_index(b'A'), 0);
}

#[test]
fn post_names_outside_the_glyph_list_keep_mac_roman() {
    // A subset whose only glyph is named `circlesolid`: no name maps to
    // Unicode, so codes resolve through the (1,0) subtable directly.
    let face = face_at(&sfnt::custom_named_box_font(&sfnt::mac_only_cmap()), 40.0);
    assert!(!face.has_unicode_cmap());
    assert_eq!(face.raw_get_name_index("circlesolid"), 1);
    assert!(face.make_glyph(7, 0).is_some(), "code 7 through Mac Roman");
}

#[test]
fn truetype_without_horizontal_metrics_still_renders() {
    let full = face_at(font_test_data::TINOS_SUBSET, 40.0);
    let stripped = sfnt::rename_tables(
        font_test_data::TINOS_SUBSET,
        &[(b"hhea", b"hheb"), (b"hmtx", b"hmty")],
    );
    let bare = face_at(&stripped, 40.0);
    let code = u32::from('B');
    let a = full.make_glyph(code, 0).unwrap();
    let b = bare
        .make_glyph(code, 0)
        .expect("outline survives without hmtx");
    assert!(same_bitmap(&a, &b));
    assert!(bare.glyph_advance(code).is_none());
}

#[test]
fn system_fallback_font_loads_from_file() {
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ];
    let Some(path) = CANDIDATES.iter().find(|p| std::path::Path::new(p).exists()) else {
        eprintln!("no system font available; skipping");
        return;
    };
    let mut engine = FontEngine::new(true);
    let params = || FaceParams {
        code_to_gid: Vec::new(),
        mat: [24.0, 0.0, 0.0, 24.0],
    };
    let face = engine
        .load_file_face(path, 0, params())
        .expect("system font loads");
    assert!(face.has_unicode_cmap());
    for ch in ['H', 'e', 'g', '1'] {
        assert!(face.make_glyph(u32::from(ch), 0).is_some(), "{ch}");
    }
    let again = engine.load_file_face(path, 0, params()).unwrap();
    assert!(
        std::ptr::eq(face.program(), again.program()),
        "file program is cached"
    );
    assert!(
        engine
            .load_file_face("/nonexistent/font.ttf", 0, params())
            .is_err()
    );
}

// ── OpenType CFF (FontFile3 /OpenType) and bare CFF (FontFile3 /Type1C) ──────

#[test]
fn opentype_cff_glyphs_render_and_resolve_by_name() {
    let face = face_at(font_test_data::NOTO_SERIF_DISPLAY_TRIMMED, 40.0);
    // `post` is nameless; names come from the CFF charset.
    for (gid, name) in [(1, "i"), (2, "j"), (3, "k"), (4, "l")] {
        assert_eq!(face.raw_get_name_index(name), gid, "{name}");
        assert!(face.make_glyph_by_gid(gid, 0).is_some(), "{name}");
    }
    assert_eq!(face.raw_get_char_index(u32::from('k')), 3);
}

#[test]
fn bare_cff_matches_its_opentype_wrapper() {
    let cff = sfnt::table(font_test_data::NOTO_SERIF_DISPLAY_TRIMMED, *b"CFF ");
    let bare = face_at(&cff, 40.0);
    let wrapped = face_at(font_test_data::NOTO_SERIF_DISPLAY_TRIMMED, 40.0);
    assert!(!bare.has_unicode_cmap());
    for (gid, name) in [(1, "i"), (2, "j"), (3, "k"), (4, "l")] {
        assert_eq!(bare.raw_get_name_index(name), gid, "{name}");
        // The Unicode charmap of a bare CFF program derives from names.
        let code = u32::from(name.chars().next().unwrap());
        assert_eq!(bare.raw_get_char_index(code), gid, "{name}");
        let a = bare.make_glyph_by_gid(gid, 0).unwrap();
        let b = wrapped.make_glyph_by_gid(gid, 0).unwrap();
        assert!(same_bitmap(&a, &b), "{name}");
        let (wa, wb) = (
            bare.glyph_advance_by_gid(gid).unwrap(),
            wrapped.glyph_advance_by_gid(gid).unwrap(),
        );
        assert!((wa - wb).abs() < 1e-3, "{name}: {wa} vs {wb}");
    }
    assert_eq!(bare.raw_get_name_index("A"), 0);
}

// ── CID-keyed CFF (FontFile3 /CIDFontType0C) ─────────────────────────────────

/// Glyph index and CID of the glyph the Unicode charmap maps `code` to.
fn cid_of(face: &crate::face::FontFace, code: char) -> (u32, u32) {
    let gid = face.raw_get_char_index(u32::from(code));
    assert!(gid != 0, "{code} is mapped");
    (gid, gid_to_cid(gid))
}

/// The CID of glyph `gid` in [`super::CID_SUBSET`] (from its charset).
fn gid_to_cid(gid: u32) -> u32 {
    let table = sfnt::table(super::CID_SUBSET, *b"CFF ");
    let cff = read_fonts::ps::cff::CffFontRef::new(&table, 0, None).unwrap();
    let cid = cff
        .charset()
        .unwrap()
        .string_id(read_fonts::types::GlyphId::new(gid))
        .unwrap();
    u32::from(cid.to_u16())
}

#[test]
fn cid_keyed_cff_maps_cids_through_the_charset() {
    let otf = face_at(super::CID_SUBSET, 40.0);
    let bare = face_at(&sfnt::table(super::CID_SUBSET, *b"CFF "), 40.0);
    for ch in ['A', 'M', 'Z'] {
        let (gid, cid) = cid_of(&otf, ch);
        assert_ne!(gid, cid, "subset glyph order differs from CID order");
        let by_code = otf.make_glyph(u32::from(ch), 0).expect("Unicode route");
        for face in [&otf, &bare] {
            let by_cid = face.make_glyph_by_gid(cid, 0).expect("CID route");
            assert!(same_bitmap(&by_code, &by_cid), "{ch}");
            assert!(face.glyph_path_by_gid(cid).is_some());
            assert!(face.glyph_advance_by_gid(cid).unwrap() > 0.0);
        }
    }
    // CID 0 is .notdef; a CID absent from the subset is a missing glyph.
    assert!(bare.make_glyph_by_gid(9_999, 0).is_none());
}

#[test]
fn cid_keyed_and_sfnt_programs_have_no_builtin_encoding() {
    let bare = face_at(&sfnt::table(super::CID_SUBSET, *b"CFF "), 40.0);
    let wrapped = face_at(super::CID_SUBSET, 40.0);
    for code in [0, 65, 255] {
        assert_eq!(bare.raw_get_builtin_index(code), None);
        assert_eq!(wrapped.raw_get_builtin_index(code), None);
    }
}

// ── Type 1 (FontFile) ─────────────────────────────────────────────────────────

#[test]
fn type1_builtin_encoding_maps_codes_to_glyphs() {
    let standard = face_at(&type1::synthetic_pfa(), 40.0);
    let a = standard.raw_get_name_index("A");
    assert_ne!(a, 0);
    assert_eq!(standard.raw_get_builtin_index(65), Some(a));
    assert_eq!(standard.raw_get_builtin_index(33), None, "exclam is absent");

    // A custom encoding assigning code 33 to `B` and nothing else.
    let custom = String::from_utf8(type1::synthetic_pfa())
        .expect("PFA is ASCII")
        .replace(
            "/Encoding StandardEncoding def",
            "/Encoding 256 array 0 1 255 {1 index exch /.notdef put} for \
             dup 33 /B put readonly def",
        );
    let custom = face_at(custom.as_bytes(), 40.0);
    let b = custom.raw_get_name_index("B");
    assert_ne!(b, 0);
    assert_eq!(custom.raw_get_builtin_index(33), Some(b));
    assert_eq!(custom.raw_get_builtin_index(65), None);
}

#[test]
fn type1_pfa_and_pfb_render_identically() {
    let pfa = face_at(font_test_data::type1::NOTO_SERIF_REGULAR_SUBSET_PFA, 40.0);
    let pfb = face_at(font_test_data::type1::NOTO_SERIF_REGULAR_SUBSET_PFB, 40.0);
    for name in ["H", "f", "i", "x"] {
        let gid = pfa.raw_get_name_index(name);
        assert!(gid != 0, "{name}");
        assert_eq!(pfb.raw_get_name_index(name), gid);
        let code = u32::from(name.chars().next().unwrap());
        assert_eq!(
            pfa.raw_get_char_index(code),
            gid,
            "Unicode from glyph names"
        );
        let a = pfa.make_glyph_by_gid(gid, 0).unwrap();
        let b = pfb.make_glyph_by_gid(gid, 0).unwrap();
        assert!(same_bitmap(&a, &b), "{name}");
        assert!(ink(&a) > 20.0, "{name}");
    }
    // H: cap height ≈ 0.71 em → ≈ 28 px at 40 px/em.
    let h = pfa
        .make_glyph_by_gid(pfa.raw_get_name_index("H"), 0)
        .unwrap();
    assert!((26..=31).contains(&h.height), "{}", h.height);
}

#[test]
fn type1_layouts_parse_to_the_same_glyphs() {
    let variants = [
        type1::synthetic_pfa(),
        type1::synthetic_binary_eexec(),
        type1::synthetic_pfb(),
    ];
    let faces: Vec<_> = variants
        .iter()
        .map(|data| face_at(data, UNITS_1000))
        .collect();
    for name in ["A", "acute", "Aacute", "B", "C"] {
        let gid = faces[0].raw_get_name_index(name);
        assert!(gid != 0, "{name}");
        for face in &faces[1..] {
            assert_eq!(face.raw_get_name_index(name), gid, "{name}");
            let a = faces[0].glyph_path_by_gid(gid).unwrap();
            let b = face.glyph_path_by_gid(gid).unwrap();
            assert_eq!(a.pts, b.pts, "{name}");
        }
    }
}

fn synthetic_path(name: &str) -> raster::path::Path {
    let face = face_at(&type1::synthetic_pfa(), UNITS_1000);
    let gid = face.raw_get_name_index(name);
    assert!(gid != 0, "{name}");
    face.glyph_path_by_gid(gid)
        .unwrap_or_else(|| panic!("{name} has an outline"))
}

#[test]
fn type1_hint_replacement_keeps_the_outline() {
    let a = synthetic_path("A");
    assert_eq!(subpaths(&a), 1);
    let (x0, y0, x1, y1) = bounds(&a);
    assert!(
        (x0 - 50.0).abs() < 0.5 && (x1 - 650.0).abs() < 0.5,
        "{x0} {x1}"
    );
    assert!(y0.abs() < 0.5 && (y1 - 700.0).abs() < 0.5, "{y0} {y1}");
}

#[test]
fn type1_seac_composes_base_and_accent() {
    let composite = synthetic_path("Aacute");
    let base = synthetic_path("A");
    assert_eq!(subpaths(&composite), 2, "base contour + accent contour");
    let (bx0, by0, bx1, _) = bounds(&base);
    let (cx0, cy0, cx1, cy1) = bounds(&composite);
    assert!((cx0 - bx0).abs() < 0.5 && (cy0 - by0).abs() < 0.5);
    assert!(cx1 >= bx1 - 0.5);
    assert!((cy1 - 900.0).abs() < 0.5, "accent apex at 900, got {cy1}");
}

#[test]
fn type1_flex_becomes_two_curves() {
    let b = synthetic_path("B");
    assert_eq!(
        b.flags.iter().filter(|f| f.is_curve()).count(),
        4,
        "two cubics"
    );
    let dip = b
        .pts
        .iter()
        .any(|p| (p.x - 350.0).abs() < 0.5 && (p.y - 680.0).abs() < 0.5);
    assert!(dip, "flex joins at (350, 680): {:?}", b.pts);
    let (x0, y0, x1, y1) = bounds(&b);
    assert!(
        (x0 - 50.0).abs() < 0.5
            && (x1 - 650.0).abs() < 0.5
            && y0.abs() < 0.5
            && (y1 - 700.0).abs() < 0.5
    );
}

#[test]
fn type1_div_operator_sets_the_advance() {
    let face = face_at(&type1::synthetic_pfa(), UNITS_1000);
    let advance = face
        .glyph_advance_by_gid(face.raw_get_name_index("C"))
        .unwrap();
    assert!((advance - 0.5).abs() < 1e-3, "{advance}");
    let a = face
        .glyph_advance_by_gid(face.raw_get_name_index("A"))
        .unwrap();
    assert!((a - 0.7).abs() < 1e-3, "{a}");
}

#[test]
fn type1_glyphs_rasterise_with_full_interior_coverage() {
    let face = face_at(&type1::synthetic_pfa(), 100.0);
    let a = face
        .make_glyph_by_gid(face.raw_get_name_index("A"), 0)
        .unwrap();
    // 600×700 units at 0.1 px/unit: a 60×70 block starting 5 px right of
    // the pen.
    assert_eq!((a.width, a.height, a.x_off, a.y_off), (60, 70, -5, 70));
    assert!(a.data.iter().all(|&v| v == 255));
}
