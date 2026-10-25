use super::{
    CORNER_TABLE, CORNERS, cmyk_f64_to_rgb, cmyk_to_rgb, cmyk_to_rgb_row, cmyk_to_rgb_vec,
    convert_scalar, unit_weight, weight,
};

/// Corner index `k·8 + c·4 + m·2 + y` → ink bytes `[c, m, y, k]`.
fn corner_inks(i: usize) -> [u8; 4] {
    let bit = |shift: usize| if (i >> shift) & 1 == 1 { 255 } else { 0 };
    [bit(2), bit(1), bit(0), bit(3)]
}

/// Exact multilinear interpolation of [`CORNERS`] at `t = x / 255`, unrounded.
fn exact(px: [u8; 4]) -> [f64; 3] {
    let t = px.map(|x| f64::from(x) / 255.0);
    let mut out = [0.0; 3];
    for (i, corner) in CORNERS.iter().enumerate() {
        let inks = corner_inks(i);
        let w: f64 = (0..4)
            .map(|a| if inks[a] == 255 { t[a] } else { 1.0 - t[a] })
            .product();
        for ch in 0..3 {
            out[ch] = w.mul_add(f64::from(corner[ch]), out[ch]);
        }
    }
    out
}

/// Deterministic pseudo-random bytes (LCG, high byte).
fn bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s.to_be_bytes()[0]
        })
        .collect()
}

/// Every (ink, K) byte pair on each of C, M and Y, plus a scrambled Y.
fn exhaustive_pairs() -> Vec<u8> {
    (0u32..65_536)
        .flat_map(|i| {
            let [ink, k, ..] = i.to_le_bytes();
            [ink, 255 - ink, ink.wrapping_mul(167), k]
        })
        .collect()
}

fn reference(cmyk: &[u8]) -> Vec<u8> {
    let mut rgb = vec![0u8; cmyk.len() / 4 * 3];
    convert_scalar(cmyk, &mut rgb);
    rgb
}

/// Relative luminance (IEC 61966-2-1 decoding, Rec. 709 weights).
fn luminance(rgb: [u8; 3]) -> f64 {
    let lin = |v: u8| {
        let v = f64::from(v) / 255.0;
        if v <= 0.040_45 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.0722f64.mul_add(
        lin(rgb[2]),
        0.2126f64.mul_add(lin(rgb[0]), 0.7152 * lin(rgb[1])),
    )
}

// ── Corners ───────────────────────────────────────────────────────────────────

#[test]
fn byte_corners_reproduce_table() {
    for (i, corner) in CORNERS.iter().enumerate() {
        let [c, m, y, k] = corner_inks(i);
        assert_eq!(
            cmyk_to_rgb(c, m, y, k),
            *corner,
            "corner {i} = cmyk({c},{m},{y},{k})"
        );
    }
}

#[test]
fn unit_corners_reproduce_table() {
    for (i, corner) in CORNERS.iter().enumerate() {
        let [c, m, y, k] = corner_inks(i).map(|v| f64::from(v) / 255.0);
        assert_eq!(cmyk_f64_to_rgb(c, m, y, k), *corner, "corner {i}");
    }
}

#[test]
fn corner_table_is_flattened_corners() {
    for (i, corner) in CORNERS.iter().enumerate() {
        assert_eq!(&CORNER_TABLE[i * 3..i * 3 + 3], corner);
    }
}

#[test]
fn paper_is_white_and_four_colour_solid_is_black() {
    assert_eq!(cmyk_to_rgb(0, 0, 0, 0), [255, 255, 255]);
    assert_eq!(cmyk_to_rgb(255, 255, 255, 255), [0, 0, 0]);
}

/// Solid process inks and their two-ink overprints are the printed colours,
/// not the ideal block filters of the additive complement.
#[test]
fn solids_and_overprints() {
    assert_eq!(cmyk_to_rgb(255, 0, 0, 0), [0, 158, 227], "cyan");
    assert_eq!(cmyk_to_rgb(0, 255, 0, 0), [229, 0, 125], "magenta");
    assert_eq!(cmyk_to_rgb(0, 0, 255, 0), [255, 237, 0], "yellow");
    assert_eq!(cmyk_to_rgb(0, 0, 0, 255), [23, 23, 23], "black");
    assert_eq!(cmyk_to_rgb(255, 255, 0, 0), [46, 35, 131], "blue overprint");
    assert_eq!(cmyk_to_rgb(255, 0, 255, 0), [0, 150, 62], "green overprint");
    assert_eq!(cmyk_to_rgb(0, 255, 255, 0), [227, 0, 11], "red overprint");
}

// ── Interpolation ─────────────────────────────────────────────────────────────

#[test]
fn weights_span_zero_to_256() {
    assert_eq!(weight(0), 0);
    assert_eq!(weight(127), 127);
    assert_eq!(weight(128), 129);
    assert_eq!(weight(255), 256);
    for x in 0..255u8 {
        assert!(
            weight(x) < weight(x + 1),
            "weight not strictly increasing at {x}"
        );
    }
}

/// Byte-quantised normalised inputs select the same weights as byte inputs,
/// so vector fills and image samples of one colour convert identically.
#[test]
fn unit_weights_match_byte_weights() {
    for x in 0..=255u8 {
        assert_eq!(unit_weight(f64::from(x) / 255.0), weight(x), "x = {x}");
    }
}

#[test]
fn unit_conversion_matches_bytes_on_a_grid() {
    let steps = [0u8, 1, 37, 64, 127, 128, 129, 200, 254, 255];
    for &c in &steps {
        for &m in &steps {
            for &y in &steps {
                for &k in &steps {
                    let unit = [c, m, y, k].map(|v| f64::from(v) / 255.0);
                    assert_eq!(
                        cmyk_f64_to_rgb(unit[0], unit[1], unit[2], unit[3]),
                        cmyk_to_rgb(c, m, y, k),
                        "cmyk({c},{m},{y},{k})"
                    );
                }
            }
        }
    }
}

/// The fixed-point evaluation stays within one level of the exact
/// multilinear interpolation.
#[test]
fn fixed_point_tracks_exact_interpolation() {
    let mut cmyk = bytes(4 * 50_000, 0x2468_ace0);
    cmyk.extend_from_slice(&exhaustive_pairs());
    for px in cmyk.as_chunks::<4>().0 {
        let got = cmyk_to_rgb(px[0], px[1], px[2], px[3]);
        let want = exact(*px);
        for ch in 0..3 {
            let err = (f64::from(got[ch]) - want[ch]).abs();
            assert!(
                err <= 1.0,
                "cmyk{px:?} channel {ch}: got {} want {:.3}",
                got[ch],
                want[ch]
            );
        }
    }
}

#[test]
fn process_blue_is_a_print_blue() {
    // 85% C + 69% M: a deep print blue, far from the additive (38, 79, 255).
    let rgb = cmyk_f64_to_rgb(0.85, 0.69, 0.0, 0.0);
    let want = exact([217, 176, 0, 0]);
    for ch in 0..3 {
        assert!(
            (f64::from(rgb[ch]) - want[ch]).abs() <= 1.0,
            "{rgb:?} vs {want:?}"
        );
    }
    assert!(rgb[2] < 180 && rgb[0] < 80 && rgb[1] < 90, "{rgb:?}");
}

/// A K-only ramp is an exactly neutral, monotone tone scale.
#[test]
fn k_only_ramp_is_monotone() {
    let mut prev = cmyk_to_rgb(0, 0, 0, 0);
    for k in 1..=255u8 {
        let cur = cmyk_to_rgb(0, 0, 0, k);
        for ch in 0..3 {
            assert!(
                cur[ch] <= prev[ch],
                "K ramp brightens at k={k}: {prev:?} → {cur:?}"
            );
        }
        assert!(
            cur[0] == cur[1] && cur[1] == cur[2],
            "K ramp leaves neutral at k={k}: {cur:?}"
        );
        prev = cur;
    }
}

/// Adding an ink to a K-free solid or overprint never makes it lighter, and
/// no corner carrying K is lighter than solid K alone.
#[test]
fn adding_ink_never_lightens_a_corner() {
    for (i, corner) in CORNERS[..8].iter().enumerate() {
        for bit in 0..4 {
            if i & (1 << bit) == 0 {
                let darker = CORNERS[i | (1 << bit)];
                assert!(
                    luminance(darker) <= luminance(*corner),
                    "corner {i} {corner:?} → {} {darker:?}",
                    i | (1 << bit)
                );
            }
        }
    }
    for (i, corner) in CORNERS[8..].iter().enumerate() {
        assert!(
            luminance(*corner) <= luminance(CORNERS[8]),
            "corner {} {corner:?} is lighter than solid K",
            i + 8
        );
    }
}

#[test]
fn unit_inputs_are_clamped_and_nan_is_no_ink() {
    assert_eq!(cmyk_f64_to_rgb(-1.0, 0.0, 0.0, 0.0), [255, 255, 255]);
    assert_eq!(cmyk_f64_to_rgb(0.0, 0.0, 0.0, 7.5), CORNERS[8]);
    assert_eq!(cmyk_f64_to_rgb(f64::NAN, 0.0, 0.0, 0.0), [255, 255, 255]);
    assert_eq!(cmyk_f64_to_rgb(1.0, 0.0, 0.0, f64::NAN), CORNERS[4]);
    assert_eq!(cmyk_f64_to_rgb(f64::INFINITY, 0.0, 0.0, 0.0), CORNERS[4]);
}

// ── Row conversion and SIMD parity ────────────────────────────────────────────

#[test]
fn row_matches_scalar_exhaustive_pairs() {
    let cmyk = exhaustive_pairs();
    assert_eq!(cmyk_to_rgb_vec(&cmyk), reference(&cmyk));
}

/// Every length from 0 to 40 pixels: whole blocks of every tier plus tails.
#[test]
fn row_matches_scalar_at_every_tail_length() {
    let all = bytes(4 * 40, 0x1357_9bdf);
    for n in 0..=40 {
        let cmyk = &all[..n * 4];
        assert_eq!(cmyk_to_rgb_vec(cmyk), reference(cmyk), "{n} pixels");
    }
}

#[test]
#[should_panic(expected = "not a multiple of 4")]
fn row_rejects_partial_pixel() {
    let mut rgb = [0u8; 3];
    cmyk_to_rgb_row(&[0, 0, 0], &mut rgb);
}

#[test]
#[should_panic(expected = "3 bytes per CMYK pixel")]
fn row_rejects_mismatched_output() {
    let mut rgb = [0u8; 4];
    cmyk_to_rgb_row(&[0, 0, 0, 0], &mut rgb);
}

/// Direct AVX-512 kernel call: the dispatcher only runs the highest tier
/// present, so each tier is also exercised on its own.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_block_matches_scalar() {
    if !(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi"))
    {
        return;
    }
    let mut cmyk = exhaustive_pairs();
    cmyk.extend_from_slice(&bytes(64 * 500, 0xdead_beef));
    let want = reference(&cmyk);
    let mut got = vec![0u8; want.len()];
    for (block, out) in cmyk
        .as_chunks::<64>()
        .0
        .iter()
        .zip(got.as_chunks_mut::<48>().0)
    {
        // SAFETY: avx512f + avx512bw + avx512vbmi detected above.
        #[expect(unsafe_code, reason = "direct call of a target-feature kernel")]
        unsafe {
            super::simd::x86::block_avx512(block, out);
        }
    }
    assert_eq!(got, want);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_block_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut cmyk = exhaustive_pairs();
    cmyk.extend_from_slice(&bytes(32 * 1000, 0x0bad_f00d));
    let want = reference(&cmyk);
    let mut got = vec![0u8; want.len()];
    for (block, out) in cmyk
        .as_chunks::<32>()
        .0
        .iter()
        .zip(got.as_chunks_mut::<24>().0)
    {
        // SAFETY: avx2 detected above.
        #[expect(unsafe_code, reason = "direct call of a target-feature kernel")]
        unsafe {
            super::simd::x86::block_avx2(block, out);
        }
    }
    assert_eq!(got, want);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_block_matches_scalar() {
    let mut cmyk = exhaustive_pairs();
    cmyk.extend_from_slice(&bytes(64 * 500, 0xdead_beef));
    let want = reference(&cmyk);
    let mut got = vec![0u8; want.len()];
    for (block, out) in cmyk
        .as_chunks::<64>()
        .0
        .iter()
        .zip(got.as_chunks_mut::<48>().0)
    {
        // SAFETY: NEON is mandatory on aarch64.
        #[expect(unsafe_code, reason = "direct call of a target-feature kernel")]
        unsafe {
            super::simd::arm::block_neon(block, out);
        }
    }
    assert_eq!(got, want);
}
