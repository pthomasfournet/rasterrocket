//! The HT block coder (ISO/IEC 15444-15).

use crate::*;

#[test]
fn gray() {
    check("ht_gray", reference("main_r"));
}

#[test]
fn rgb_rct() {
    check("ht_rgb", reference("main_rgb"));
}

#[test]
fn blocks_4x4() {
    check("ht_blocks_4x4", reference("main_r"));
}

#[test]
fn odd_size() {
    check("ht_odd_size", reference("main_r_crop_5_3_61_47"));
}

#[test]
fn vertically_causal() {
    check("ht_causal_rgb", reference("main_rgb"));
}

#[test]
fn refinement_passes() {
    check("ht_refinement", reference("ht_refinement"));
}

#[test]
fn refinement_passes_vertically_causal() {
    // Irreversible with the ICT: the reference decoder rounds differently.
    check(
        "ht_refinement_causal",
        reference("ht_refinement_causal").tolerance(1),
    );
}

#[test]
fn quality_layers() {
    check("ht_layers", reference("ht_layers"));
}

#[test]
fn reduced_resolution() {
    check_with(
        "ht_gray",
        DecodeSettings {
            target_resolution: Some((32, 24)),
            ..Default::default()
        },
        reference("wavelet_reduced_resolution_2"),
    );
}

#[test]
fn mixed_with_annex_d_blocks() {
    check_error(
        "ht_mixed",
        DecodeSettings::default(),
        "mixing HT and classic code-blocks",
    );
}
