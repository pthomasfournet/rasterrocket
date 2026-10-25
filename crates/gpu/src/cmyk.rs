//! CPU fallback of the GPU CMYK→RGB entry points.
//!
//! The profile-free conversion is [`color::cmyk`] (SIMD row conversion);
//! the CLUT conversion is evaluated here with the same quadrilinear
//! interpolation as the GPU kernel.

// ── Public API ────────────────────────────────────────────────────────────────

/// CPU fallback for [`crate::GpuCtx::icc_cmyk_to_rgb`].
///
/// When `clut` is `None`, converts with the profile-free process-ink model
/// [`color::cmyk::cmyk_to_rgb_row`] (runtime-dispatched SIMD).
///
/// When `clut` is `Some((table, grid_n))`, evaluates the 4D CLUT using
/// quadrilinear interpolation — the same algorithm as the GPU kernel.
///
/// # Panics
///
/// Panics if `clut` is `Some((table, grid_n))` and either `grid_n < 2`
/// (fewer than 2 nodes per axis is degenerate and unusable for
/// interpolation) or `table.len() != grid_n⁴ × 3` (the table and grid size
/// disagree — reading through the mismatch would index out of bounds for a
/// short table and silently sample garbage for a long one).
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one function per CLUT/matrix branch with the interpolation inline — \
              splitting would scatter the shared index arithmetic"
)]
pub fn icc_cmyk_to_rgb_cpu(cmyk: &[u8], clut: Option<(&[u8], u32)>) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgb = vec![0u8; n * 3];

    match clut {
        None => {
            color::cmyk::cmyk_to_rgb_row(&cmyk[..n * 4], &mut rgb);
        }
        Some((table, grid_n)) => {
            // grid_n = 0 would cause underflow in `grid_n - 1` and panic via
            // `g - 1` index arithmetic below.  Public function: guard explicitly.
            assert!(
                grid_n >= 2,
                "icc_cmyk_to_rgb_cpu: grid_n must be ≥ 2, got {grid_n}"
            );
            let expected_len = (grid_n as usize)
                .checked_pow(4)
                .and_then(|nodes| nodes.checked_mul(3))
                .unwrap_or_else(|| {
                    panic!("grid_n({grid_n})^4*3 overflows usize — grid_n must be ≤ 255")
                });
            assert_eq!(
                table.len(),
                expected_len,
                "CLUT table length {got} ≠ grid_n({grid_n})^4*3={expected_len}",
                got = table.len(),
            );
            let g = grid_n as usize; // grid_n ≤ 255 from caller validation
            let g2 = g * g;
            let g3 = g2 * g;
            // grid_n ≤ 255 → (grid_n - 1) ≤ 254, exact in f32 (needs ≤ 8 mantissa bits).
            #[expect(
                clippy::cast_precision_loss,
                reason = "grid_n ≤ 255, fits exactly in f32 (8 bits < 23-bit mantissa)"
            )]
            let g1 = (grid_n - 1) as f32;
            let scale = g1 / 255.0;
            for (src, dst) in cmyk
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgb.as_chunks_mut::<3>().0)
            {
                let fc = f32::from(src[0]) * scale;
                let fm = f32::from(src[1]) * scale;
                let fy = f32::from(src[2]) * scale;
                let fk = f32::from(src[3]) * scale;

                // fc ∈ [0.0, g1] ⊂ [0.0, 254.0]; floor is non-negative and ≤ 254.
                // The sign-loss and truncation lints fire because `as usize` is UB
                // for negative or >usize::MAX floats; here neither can happen.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "fc/fm/fy/fk are in [0.0, 254.0]; floor is exact and non-negative"
                )]
                let (ic0, im0, iy0, ik0) = (
                    (fc as usize).min(g - 1),
                    (fm as usize).min(g - 1),
                    (fy as usize).min(g - 1),
                    (fk as usize).min(g - 1),
                );
                let ic1 = (ic0 + 1).min(g - 1);
                let im1 = (im0 + 1).min(g - 1);
                let iy1 = (iy0 + 1).min(g - 1);
                let ik1 = (ik0 + 1).min(g - 1);

                // Fractional weights: difference between float position and floored index.
                // ic0 ≤ 254 → ic0 as f32 is exact (fits in 8 mantissa bits).
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "ic0/im0/iy0/ik0 ≤ 254, exact in f32"
                )]
                let (wc, wm, wy, wk) = (
                    fc - ic0 as f32,
                    fm - im0 as f32,
                    fy - iy0 as f32,
                    fk - ik0 as f32,
                );

                let node = |ci: usize, mi: usize, yi: usize, ki: usize| -> [f32; 3] {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    [
                        f32::from(table[idx]),
                        f32::from(table[idx + 1]),
                        f32::from(table[idx + 2]),
                    ]
                };

                let lerp = |a: f32, b: f32, t: f32| t.mul_add(b - a, a);
                let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
                    [
                        lerp(a[0], b[0], t),
                        lerp(a[1], b[1], t),
                        lerp(a[2], b[2], t),
                    ]
                };

                // K=0 face: lerp Y → M → C
                let c0m0k0 = lerp3(node(ic0, im0, iy0, ik0), node(ic0, im0, iy1, ik0), wy);
                let c0m1k0 = lerp3(node(ic0, im1, iy0, ik0), node(ic0, im1, iy1, ik0), wy);
                let c1m0k0 = lerp3(node(ic1, im0, iy0, ik0), node(ic1, im0, iy1, ik0), wy);
                let c1m1k0 = lerp3(node(ic1, im1, iy0, ik0), node(ic1, im1, iy1, ik0), wy);
                let rk0 = lerp3(lerp3(c0m0k0, c0m1k0, wm), lerp3(c1m0k0, c1m1k0, wm), wc);

                // K=1 face: lerp Y → M → C
                let c0m0k1 = lerp3(node(ic0, im0, iy0, ik1), node(ic0, im0, iy1, ik1), wy);
                let c0m1k1 = lerp3(node(ic0, im1, iy0, ik1), node(ic0, im1, iy1, ik1), wy);
                let c1m0k1 = lerp3(node(ic1, im0, iy0, ik1), node(ic1, im0, iy1, ik1), wy);
                let c1m1k1 = lerp3(node(ic1, im1, iy0, ik1), node(ic1, im1, iy1, ik1), wy);
                let rk1 = lerp3(lerp3(c0m0k1, c0m1k1, wm), lerp3(c1m0k1, c1m1k1, wm), wc);

                let out = lerp3(rk0, rk1, wk);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0,255] before cast"
                )]
                {
                    dst[0] = out[0].clamp(0.0, 255.0).round() as u8;
                    dst[1] = out[1].clamp(0.0, 255.0).round() as u8;
                    dst[2] = out[2].clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    }

    rgb
}

#[cfg(test)]
mod tests {
    use super::icc_cmyk_to_rgb_cpu;
    use color::cmyk::{CORNER_TABLE, CORNERS, cmyk_to_rgb};

    /// Scalar reference: [`cmyk_to_rgb`] per pixel.
    fn reference(cmyk: &[u8]) -> Vec<u8> {
        cmyk.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| cmyk_to_rgb(px[0], px[1], px[2], px[3]))
            .collect()
    }

    fn pseudo_random(n: usize) -> Vec<u8> {
        let mut s = 0x1357_9bdf_u32;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                s.to_be_bytes()[0]
            })
            .collect()
    }

    #[test]
    fn icc_cmyk_matrix_white() {
        assert_eq!(icc_cmyk_to_rgb_cpu(&[0, 0, 0, 0], None), [255, 255, 255]);
    }

    #[test]
    fn icc_cmyk_matrix_solid_k_is_printed_black() {
        assert_eq!(icc_cmyk_to_rgb_cpu(&[0, 0, 0, 255], None), CORNERS[8]);
    }

    #[test]
    fn icc_cmyk_matrix_pure_cyan() {
        assert_eq!(icc_cmyk_to_rgb_cpu(&[255, 0, 0, 0], None), CORNERS[4]);
    }

    #[test]
    fn icc_cmyk_matrix_multi_pixel() {
        let rgb = icc_cmyk_to_rgb_cpu(&[0, 0, 0, 0, 255, 255, 255, 255], None);
        assert_eq!(&rgb[0..3], &[255, 255, 255]);
        assert_eq!(&rgb[3..6], &[0, 0, 0]);
    }

    /// The profile-free path is the `color::cmyk` row conversion, byte for
    /// byte, at block-multiple and tail lengths.
    #[test]
    fn icc_cmyk_matrix_matches_reference() {
        let cmyk = pseudo_random(4 * 1001);
        for n in [0, 1, 5, 8, 16, 40, 1001] {
            let px = &cmyk[..n * 4];
            assert_eq!(icc_cmyk_to_rgb_cpu(px, None), reference(px), "{n} pixels");
        }
    }

    /// The corner table is a valid 2⁴-node CLUT: the f32 CLUT evaluator
    /// reproduces the corners exactly and tracks the fixed-point model to
    /// within one level elsewhere.
    #[test]
    fn icc_cmyk_corner_table_as_clut_tracks_matrix_path() {
        let cmyk = pseudo_random(4 * 4096);
        let via_clut = icc_cmyk_to_rgb_cpu(&cmyk, Some((&CORNER_TABLE, 2)));
        let direct = icc_cmyk_to_rgb_cpu(&cmyk, None);
        for (i, (a, b)) in via_clut.iter().zip(&direct).enumerate() {
            assert!(a.abs_diff(*b) <= 1, "byte {i}: clut {a} vs matrix {b}");
        }
        for (i, corner) in CORNERS.iter().enumerate() {
            let bit = |s: usize| if (i >> s) & 1 == 1 { 255 } else { 0 };
            let px = [bit(2), bit(1), bit(0), bit(3)];
            assert_eq!(icc_cmyk_to_rgb_cpu(&px, Some((&CORNER_TABLE, 2))), corner);
        }
    }

    /// A (table, `grid_n`) pair whose lengths disagree must fail at the API
    /// boundary with a diagnostic naming both, not deep inside the
    /// interpolation closure with a bare slice index.
    #[test]
    #[should_panic(expected = "CLUT table length")]
    fn icc_cmyk_clut_rejects_mismatched_table_length() {
        let table = vec![0u8; 17usize.pow(4) * 3];
        let _ = icc_cmyk_to_rgb_cpu(&[0, 0, 0, 255], Some((&table, 33)));
    }

    #[test]
    fn icc_cmyk_clut_identity_corners() {
        // 2^4 = 16-node CLUT where output = matrix formula at corners.
        const G: u8 = 2;
        let g_usize = G as usize;
        let mut table = vec![0u8; g_usize * g_usize * g_usize * g_usize * 3];
        for ki in 0..G {
            for ci in 0..G {
                for mi in 0..G {
                    for yi in 0..G {
                        let idx = (usize::from(ki) * g_usize * g_usize * g_usize
                            + usize::from(ci) * g_usize * g_usize
                            + usize::from(mi) * g_usize
                            + usize::from(yi))
                            * 3;
                        // ki/ci/mi/yi ∈ {0, 1}; * 255 ∈ {0, 255} — fits in u8.
                        let c = ci * 255;
                        let m = mi * 255;
                        let y = yi * 255;
                        let k = ki * 255;
                        let inv_k = u32::from(255 - k);
                        // Each `((255 - x) * inv_k) / 255` ∈ [0, 255]; `try_into`
                        // makes the bound explicit so the lint stays satisfied.
                        table[idx] = u8::try_from((u32::from(255 - c) * inv_k) / 255)
                            .expect("by construction <=255");
                        table[idx + 1] = u8::try_from((u32::from(255 - m) * inv_k) / 255)
                            .expect("by construction <=255");
                        table[idx + 2] = u8::try_from((u32::from(255 - y) * inv_k) / 255)
                            .expect("by construction <=255");
                    }
                }
            }
        }
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 0], Some((&table, 2)));
        assert_eq!(rgb, [255, 255, 255], "white corner");
        let rgb = icc_cmyk_to_rgb_cpu(&[0u8, 0, 0, 255], Some((&table, 2)));
        assert_eq!(rgb, [0, 0, 0], "black corner");
    }
}
