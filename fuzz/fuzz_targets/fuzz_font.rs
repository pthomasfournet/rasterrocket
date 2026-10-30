#![no_main]

//! Embedded font programs (TrueType, OpenType/CFF, bare CFF, Type 1,
//! CID-keyed Type 1): format detection, table and charstring parsing,
//! charmaps, hinting and glyph rasterisation.
//!
//! Layout: `[face_index, size]`, then the font program.

use font::engine::{FaceParams, FontEngine};
use libfuzzer_sys::fuzz_target;

/// Character codes and glyph ids drawn per input.
const GLYPHS: u32 = 64;

fuzz_target!(|data: &[u8]| {
    let Some((sel, program)) = rasterrocket_fuzz::split_selectors::<2>(data) else {
        return;
    };
    let mut engine = FontEngine::new(sel[1] & 0x80 != 0);
    let px = f64::from(sel[1] & 0x7F) + 1.0;
    let params = FaceParams {
        code_to_gid: Vec::new(),
        mat: [px, 0.0, 0.0, px],
    };
    let Ok(face) = engine.load_memory_face(program.to_vec(), u32::from(sel[0] & 3), params) else {
        return;
    };
    for n in 0..GLYPHS {
        drop(face.make_glyph(n, 0));
        drop(face.glyph_path(n));
        std::hint::black_box(face.glyph_advance(n));
        drop(face.make_glyph_by_gid(n, 2));
    }
});
