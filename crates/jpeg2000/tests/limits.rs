//! Header fields that would make the decoder allocate without bound.
//!
//! Each test patches the SIZ or COD marker of `codestream_gray` (one 8-bit
//! component, 128 × 96, one tile, one layer, 64 × 64 code-blocks) and checks
//! that the image is refused with an error instead of being allocated.

use crate::*;

/// Offsets into `codestream_gray.j2k`: SIZ follows SOC at byte 2, COD
/// follows SIZ at byte 0x2d.
const XSIZ: usize = 8;
const YSIZ: usize = 12;
const XTSIZ: usize = 24;
const YTSIZ: usize = 28;
const COD_LAYERS: usize = 0x33;
const COD_XCB: usize = 0x37;
const COD_YCB: usize = 0x38;

fn gray_with(patches: &[(usize, &[u8])]) -> Vec<u8> {
    let mut data = read_stream("codestream_gray");
    for (offset, bytes) in patches {
        data[*offset..*offset + bytes.len()].copy_from_slice(bytes);
    }
    data
}

fn assert_too_large(data: &[u8]) {
    let result = Image::new(data, &DecodeSettings::default()).and_then(|image| {
        let mut ctx = DecoderContext::default();
        image.decode(&mut ctx).map(|_| ())
    });
    match result {
        Ok(()) => panic!("decoding succeeded, but should have failed"),
        Err(e) => assert_eq!(e.to_string(), "image is too large"),
    }
}

/// 256 × 256 one-pixel tiles: 65 536 tiles, one more than tile indices
/// can address.
#[test]
fn more_tiles_than_tile_indices() {
    let side = 256_u32.to_be_bytes();
    let one = 1_u32.to_be_bytes();
    assert_too_large(&gray_with(&[
        (XSIZ, &side),
        (YSIZ, &side),
        (XTSIZ, &one),
        (YTSIZ, &one),
    ]));
}

/// 32 769 × 32 769 samples, above the 2^30 decoded-sample limit.
#[test]
fn more_samples_than_limit() {
    let side = 32_769_u32.to_be_bytes();
    assert_too_large(&gray_with(&[
        (XSIZ, &side),
        (YSIZ, &side),
        (XTSIZ, &side),
        (YTSIZ, &side),
    ]));
}

/// 10 000 × 10 000 samples (within the sample limit) in 4 × 4
/// code-blocks: 6.25 million code-blocks in the one tile.
#[test]
fn more_code_blocks_than_limit() {
    let side = 10_000_u32.to_be_bytes();
    assert_too_large(&gray_with(&[
        (XSIZ, &side),
        (YSIZ, &side),
        (XTSIZ, &side),
        (YTSIZ, &side),
        (COD_XCB, &[0]),
        (COD_YCB, &[0]),
    ]));
}

/// 9 000 × 9 000 samples in 8 × 8 code-blocks with 32 layers: 1.27
/// million code-blocks (within the code-block limit) but 40 million
/// (code-block, layer) entries.
#[test]
fn more_code_block_layers_than_limit() {
    let side = 9_000_u32.to_be_bytes();
    assert_too_large(&gray_with(&[
        (XSIZ, &side),
        (YSIZ, &side),
        (XTSIZ, &side),
        (YTSIZ, &side),
        (COD_LAYERS, &32_u16.to_be_bytes()),
        (COD_XCB, &[1]),
        (COD_YCB, &[1]),
    ]));
}

/// 16 384 × 16 384 samples in 2 × 2 precincts, in a position-first
/// progression: 2^26 precincts at the top resolution alone, against a
/// limit of 2^22.
#[test]
fn more_precincts_than_limit() {
    const COD_LCOD: usize = 0x2f;
    const COD_SCOD: usize = 0x31;
    const COD_PROGRESSION: usize = 0x32;
    const COD_END: usize = 0x3b;
    let side = 16_384_u32.to_be_bytes();
    let mut data = gray_with(&[
        (XSIZ, &side),
        (YSIZ, &side),
        (XTSIZ, &side),
        (YTSIZ, &side),
        (COD_LCOD, &18_u16.to_be_bytes()),
        // Precincts defined, resolution-position-component-layer order.
        (COD_SCOD, &[1]),
        (COD_PROGRESSION, &[2]),
    ]);
    // One precinct size per resolution level, 2^1 × 2^1 each.
    data.splice(COD_END..COD_END, [0x11; 6]);
    assert_too_large(&data);
}
