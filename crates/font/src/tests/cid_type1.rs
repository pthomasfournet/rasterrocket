//! CID-keyed Type 1 programs (`FontFile` of a `CIDFontType0` font).
//!
//! The fixture (see `data/generate.py`) has 1000 units per em and five
//! CIDs: 0 `.notdef` without contours; 1 a square x 100..400, y 0..300;
//! 2 a triangle (50,0) (450,0) (250,400) drawn by a subroutine; 3 a diamond
//! x 300..700, y 0..400 in a second font dictionary whose charstrings are
//! stored unencrypted; 4 without a glyph.

use super::{bounds, face_at, ink, subpaths};
use crate::program::FontFormat;

pub(super) const CID_TYPE1: &[u8] = include_bytes!("data/cid-keyed-type1.cid");

/// `(min_x, min_y, max_x, max_y)` of CID `cid` at 40 px per em.
fn cid_bounds(data: &[u8], cid: u32) -> (f64, f64, f64, f64) {
    let face = face_at(data, 40.0);
    bounds(&face.glyph_path_by_gid(cid).expect("CID has an outline"))
}

fn close(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    [(a.0, b.0), (a.1, b.1), (a.2, b.2), (a.3, b.3)]
        .iter()
        .all(|(x, y)| (x - y).abs() < 1e-6)
}

#[test]
fn program_is_cid_keyed_type1() {
    let face = face_at(CID_TYPE1, 40.0);
    assert_eq!(face.program().format(), FontFormat::Type1);
    assert!(!face.program().requires_hinting());
    // Glyphs are selected by CID: there is no charmap and there are no
    // glyph names, and a CID selects itself.
    assert_eq!(face.raw_get_char_index(u32::from('A')), 0);
    assert_eq!(face.raw_get_name_index(".notdef"), 0);
    assert!(!face.has_unicode_cmap());
    assert_eq!(face.program().cid_to_gid(3), Some(3));
}

#[test]
fn cids_select_their_charstrings() {
    assert!(close(cid_bounds(CID_TYPE1, 1), (4.0, 0.0, 16.0, 12.0)));
    // The subroutine of the first font dictionary.
    assert!(close(cid_bounds(CID_TYPE1, 2), (2.0, 0.0, 18.0, 16.0)));
    // The second font dictionary, lenIV -1.
    assert!(close(cid_bounds(CID_TYPE1, 3), (12.0, 0.0, 28.0, 16.0)));
    let face = face_at(CID_TYPE1, 40.0);
    assert_eq!(subpaths(&face.glyph_path_by_gid(3).expect("outline")), 1);
}

#[test]
fn cid_bitmaps_cover_their_outlines() {
    let face = face_at(CID_TYPE1, 40.0);
    let square = face.make_glyph_by_gid(1, 0).expect("CID 1 renders");
    assert_eq!((square.width, square.height), (12, 12));
    assert!((ink(&square) - 144.0).abs() < 0.5);
    // Diamond: half of its 16 × 16 bounding box.
    let diamond = face.make_glyph_by_gid(3, 0).expect("CID 3 renders");
    assert!((ink(&diamond) - 128.0).abs() < 0.5);
}

#[test]
fn missing_cids_have_no_glyph() {
    let face = face_at(CID_TYPE1, 40.0);
    // CID 0 has no contours, CID 4 no charstring, CID 5 is past CIDCount.
    for cid in [0, 4, 5, 60_000] {
        assert!(face.make_glyph_by_gid(cid, 0).is_none(), "CID {cid}");
        assert!(face.glyph_path_by_gid(cid).is_none(), "CID {cid}");
    }
    assert_eq!(face.glyph_advance_by_gid(4), None);
}

#[test]
fn advances_come_from_hsbw() {
    let face = face_at(CID_TYPE1, 40.0);
    for cid in 0..4 {
        assert_eq!(face.glyph_advance_by_gid(cid), Some(1.0), "CID {cid}");
    }
}

#[test]
fn other_header_comments_are_accepted() {
    // The same program under a generic PostScript header.
    let header = b"%!PS-Adobe-3.0 Resource-CIDFont";
    assert!(CID_TYPE1.starts_with(header));
    let data = [&b"%!FontType1-1.0 CID"[..], &CID_TYPE1[header.len()..]].concat();
    assert!(close(cid_bounds(&data, 1), (4.0, 0.0, 16.0, 12.0)));
}
