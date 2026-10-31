//! CPU fallback of the GPU CMYK→RGB entry points.
//!
//! The profile-free conversion is [`color::cmyk`] (SIMD row conversion);
//! the CLUT conversion is evaluated here with the same quadrilinear
//! interpolation as the GPU kernel.

// ── Public API ────────────────────────────────────────────────────────────────

/// Check that `table` is a CLUT of `grid_n` nodes per axis: `grid_n` in
/// `[2, 255]` and `table.len() == grid_n⁴ × 3`.
///
/// # Errors
/// A message naming the violated condition and both values.
pub(crate) fn check_clut(table: &[u8], grid_n: u32) -> Result<(), String> {
    if !(2..=255).contains(&grid_n) {
        return Err(format!("CLUT grid_n must be in [2, 255], got {grid_n}"));
    }
    let expected = (grid_n as usize)
        .checked_pow(4)
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| format!("CLUT grid_n({grid_n})^4*3 overflows usize"))?;
    if table.len() != expected {
        return Err(format!(
            "CLUT table length {} ≠ grid_n({grid_n})^4*3={expected}",
            table.len()
        ));
    }
    Ok(())
}

/// CPU fallback for [`crate::GpuCtx::icc_cmyk_to_rgb`].
///
/// When `clut` is `None`, converts with the profile-free process-ink model
/// [`color::cmyk::cmyk_to_rgb_row`] (runtime-dispatched SIMD).
///
/// When `clut` is `Some((table, grid_n))`, evaluates the 4D CLUT using
/// quadrilinear interpolation — the arithmetic of the CUDA kernel.
///
/// # Panics
///
/// Panics if `clut` is `Some((table, grid_n))` and either `grid_n` lies
/// outside `[2, 255]` (fewer than 2 nodes per axis cannot be interpolated)
/// or `table.len() != grid_n⁴ × 3` (the table and grid size disagree —
/// reading through the mismatch would index out of bounds for a short table
/// and silently sample garbage for a long one).
#[must_use]
pub fn icc_cmyk_to_rgb_cpu(cmyk: &[u8], clut: Option<(&[u8], u32)>) -> Vec<u8> {
    let n = cmyk.len() / 4;
    let mut rgb = vec![0u8; n * 3];

    match clut {
        None => {
            color::cmyk::cmyk_to_rgb_row(&cmyk[..n * 4], &mut rgb);
        }
        Some((table, grid_n)) => {
            if let Err(e) = check_clut(table, grid_n) {
                panic!("icc_cmyk_to_rgb_cpu: {e}");
            }
            let lattice = Lattice::new(table, grid_n);
            for (src, dst) in cmyk
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgb.as_chunks_mut::<3>().0)
            {
                *dst = lattice.eval(*src);
            }
        }
    }

    rgb
}

/// A validated `grid_n⁴` RGB CLUT, indexed `(k·G³ + c·G² + m·G + y) · 3`.
struct Lattice<'a> {
    table: &'a [u8],
    /// Nodes per axis, in `[2, 255]`.
    g: usize,
    /// `(g − 1) / 255`: input byte to lattice coordinate.
    scale: f32,
}

impl<'a> Lattice<'a> {
    /// `table` and `grid_n` must have passed [`check_clut`].
    fn new(table: &'a [u8], grid_n: u32) -> Self {
        // grid_n ≤ 255 → (grid_n - 1) ≤ 254, exact in f32 (needs ≤ 8 mantissa bits).
        #[expect(
            clippy::cast_precision_loss,
            reason = "grid_n ≤ 255, fits exactly in f32 (8 bits < 23-bit mantissa)"
        )]
        let g1 = (grid_n - 1) as f32;
        Self {
            table,
            g: grid_n as usize,
            scale: g1 / 255.0,
        }
    }

    /// The RGB node at lattice indices `[c, m, y, k]`.
    fn node(&self, [ci, mi, yi, ki]: [usize; 4]) -> [f32; 3] {
        let g = self.g;
        let idx = (((ki * g + ci) * g + mi) * g + yi) * 3;
        [
            f32::from(self.table[idx]),
            f32::from(self.table[idx + 1]),
            f32::from(self.table[idx + 2]),
        ]
    }

    /// Interpolate the CMYK pixel `px`: lerp along Y, then M, then C, then K
    /// with fused multiply-adds, round half up and clamp to a byte.
    fn eval(&self, px: [u8; 4]) -> [u8; 3] {
        let last = self.g - 1;
        // Lattice coordinate, floor index (≤ last) and weight per channel.
        let axis = |v: u8| {
            let f = f32::from(v) * self.scale;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "f ∈ [0.0, 254.0]; the truncation is the floor"
            )]
            let i0 = (f as usize).min(last);
            #[expect(clippy::cast_precision_loss, reason = "i0 ≤ 254, exact in f32")]
            let w = f - i0 as f32;
            (i0, (i0 + 1).min(last), w)
        };
        let [c, m, y, k] = px.map(axis);
        let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            std::array::from_fn(|ch| t.mul_add(b[ch] - a[ch], a[ch]))
        };
        let pick = |(lo, hi, _): (usize, usize, f32), upper: bool| if upper { hi } else { lo };
        let face = |kk: bool| {
            let ym = |cc: bool, mm: bool| {
                let at = |yy| self.node([pick(c, cc), pick(m, mm), pick(y, yy), pick(k, kk)]);
                lerp3(at(false), at(true), y.2)
            };
            let c0 = lerp3(ym(false, false), ym(false, true), m.2);
            let c1 = lerp3(ym(true, false), ym(true, true), m.2);
            lerp3(c0, c1, c.2)
        };
        let out = lerp3(face(false), face(true), k.2);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to [0,255] before cast"
        )]
        out.map(|v| v.clamp(0.0, 255.0).round() as u8)
    }
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
