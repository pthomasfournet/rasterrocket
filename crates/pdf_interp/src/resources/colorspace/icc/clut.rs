//! CMYK → sRGB lookup tables baked from an ICC transform.
//!
//! A four-channel `moxcms` transform is evaluated on a regular `grid_n^4`
//! lattice; the result is a flat `u8` table indexed by
//! `(k·G³ + c·G² + m·G + y) · 3`, the layout of
//! `GpuCtx::icc_cmyk_to_rgb(..., Some((table, grid_n)))` and of the CPU
//! quadrilinear interpolation `gpu::icc_cmyk_to_rgb_cpu`.

use moxcms::{CmsError, Transform8BitExecutor};

/// Lattice nodes per axis: 17, the ICC minimum, gives a 17^4 × 3 =
/// 250 563-byte table; 33 nodes (≈ 3.4 MB) exceed the L2 cache of most GPUs.
pub const DEFAULT_GRID_N: u32 = 17;

/// Error returned by [`bake_cmyk_clut_with`].
#[derive(Debug)]
pub enum BakeError {
    /// `grid_n` is outside `[2, 255]`, or its table does not fit `usize`.
    InvalidGridSize(u32),
    /// The ICC profile is malformed, uses the wrong colour space, or the
    /// transform could not be built.
    Cms(CmsError),
}

impl std::fmt::Display for BakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGridSize(n) => {
                write!(f, "grid_n={n} is outside the valid range [2, 255]")
            }
            Self::Cms(e) => write!(f, "ICC CMS error: {e}"),
        }
    }
}

impl std::error::Error for BakeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cms(e) => Some(e),
            Self::InvalidGridSize(_) => None,
        }
    }
}

impl From<CmsError> for BakeError {
    fn from(e: CmsError) -> Self {
        Self::Cms(e)
    }
}

/// Map a grid index in `[0, grid_n-1]` to an 8-bit channel value in `[0, 255]`.
///
/// `i` must satisfy `i ≤ 254`; `step` = `255.0 / (grid_n - 1)`.
/// The multiply-then-round ensures the endpoint `i = grid_n-1` lands exactly on 255.
#[inline]
fn grid_to_u8(i: usize, step: f32) -> u8 {
    debug_assert!(i <= 254, "grid_to_u8: i={i} exceeds max index 254");
    // i ≤ 254, step ≤ 255.0 → product ≤ 64_770.0, well within f32 precision.
    // round() avoids accumulated floating-point drift near endpoints.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "i ≤ 254 (exact in f32); i*step ∈ [0.0, 255.0] after round(), cast is lossless"
    )]
    {
        (i as f32 * step).round() as u8
    }
}

/// Bake a CMYK → sRGB 8-bit transform (`Layout::Rgba` source, `Layout::Rgb`
/// destination) into a quadrilinear CLUT table.
///
/// `grid_n` — nodes per axis; must be in `[2, 255]`. Use [`DEFAULT_GRID_N`] (17)
/// for standard accuracy.
///
/// Returns a `Vec<u8>` of length `grid_n^4 * 3` with the sRGB output for each
/// CMYK lattice point, stored in `(k·G³+c·G²+m·G+y)*3` order.
///
/// # Errors
///
/// Returns [`BakeError::InvalidGridSize`] if `grid_n` is outside `[2, 255]`
/// or its table does not fit `usize`, or [`BakeError::Cms`] if the
/// transform fails.
#[expect(
    clippy::many_single_char_names,
    reason = "g/c/m/y/k are conventional CMYK grid-iteration names — clarity trumps length here"
)]
pub fn bake_cmyk_clut_with(
    xform: &Transform8BitExecutor,
    grid_n: u32,
) -> Result<Vec<u8>, BakeError> {
    if !(2..=255).contains(&grid_n) {
        return Err(BakeError::InvalidGridSize(grid_n));
    }

    let g = grid_n as usize; // grid_n ≤ 255
    let g2 = g * g;
    let g3 = g2 * g;
    // 255^4 · 3 exceeds a 32-bit usize.
    let total = g3
        .checked_mul(g)
        .and_then(|n| n.checked_mul(3))
        .ok_or(BakeError::InvalidGridSize(grid_n))?;

    // step = 255 / (grid_n - 1) maps grid index i ∈ [0, grid_n-1] to channel value [0, 255].
    // Using (grid_n - 1) as divisor ensures the last node (i = grid_n-1) maps exactly to 255,
    // not 255*(1 - 1/(grid_n-1)) ≈ 254.  grid_n ≥ 2 so denominator ≥ 1: no division by zero.
    #[expect(
        clippy::cast_precision_loss,
        reason = "grid_n ≤ 255, fits exactly in f32 (needs only 8 mantissa bits)"
    )]
    let step = 255.0_f32 / (grid_n - 1) as f32;

    // Process one K-slice (G^3 pixels) at a time to amortise per-call transform overhead.
    let mut src_buf = vec![0u8; g3 * 4];
    let mut dst_buf = vec![0u8; g3 * 3];
    let mut table = vec![0u8; total];

    for ki in 0..g {
        let k = grid_to_u8(ki, step);

        // Fill the K-slice input buffer: CMYK quads in (c, m, y) inner order.
        let mut off = 0;
        for ci in 0..g {
            let c = grid_to_u8(ci, step);
            for mi in 0..g {
                let m = grid_to_u8(mi, step);
                for yi in 0..g {
                    let y = grid_to_u8(yi, step);
                    src_buf[off] = c;
                    src_buf[off + 1] = m;
                    src_buf[off + 2] = y;
                    src_buf[off + 3] = k;
                    off += 4;
                }
            }
        }

        xform.transform(&src_buf[..g3 * 4], &mut dst_buf[..g3 * 3])?;

        // Scatter RGB triples from dst_buf into table in (k·G³+c·G²+m·G+y)*3 order.
        // The transform wrote pixels in the same (c, m, y) inner order we filled, so
        // src_off advances monotonically while the table index is computed by position.
        let mut src_off = 0;
        for ci in 0..g {
            for mi in 0..g {
                for yi in 0..g {
                    let idx = (ki * g3 + ci * g2 + mi * g + yi) * 3;
                    table[idx] = dst_buf[src_off];
                    table[idx + 1] = dst_buf[src_off + 1];
                    table[idx + 2] = dst_buf[src_off + 2];
                    src_off += 3;
                }
            }
        }
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::IccConversion;
    use super::*;

    /// A small CMYK (FOGRA-like) profile.
    const CMYK_PROFILE: &[u8] =
        include_bytes!("../../../../../jpeg2000/assets/CGATS001Compat-v2-micro.icc");

    fn cmyk_transform() -> Arc<IccConversion> {
        Arc::new(IccConversion::compile(CMYK_PROFILE, 4).expect("CMYK profile compiles"))
    }

    #[test]
    fn rejects_invalid_grid_sizes() {
        let conv = cmyk_transform();
        let IccConversion::Transform { xform, .. } = &*conv else {
            panic!("a CMYK profile compiles to a transform");
        };
        for grid in [0, 1, 256] {
            assert!(matches!(
                bake_cmyk_clut_with(&**xform, grid),
                Err(BakeError::InvalidGridSize(g)) if g == grid
            ));
        }
    }

    /// The table is baked once per conversion and shared; its no-ink corner
    /// is paper and its solid-K corner is dark.
    #[test]
    fn clut_is_baked_once_per_conversion() {
        let conv = cmyk_transform();
        let first = conv.cmyk_clut().expect("bakes");
        let second = conv.cmyk_clut().expect("bakes");
        assert!(Arc::ptr_eq(&first, &second));
        let g = DEFAULT_GRID_N as usize;
        assert_eq!(first.len(), g * g * g * g * 3);
        assert!(
            first[..3].iter().all(|&c| c > 240),
            "no ink is paper: {:?}",
            &first[..3]
        );
        let k = (g - 1) * g * g * g * 3;
        assert!(
            first[k..k + 3].iter().all(|&c| c < 60),
            "solid K is dark: {:?}",
            &first[k..k + 3]
        );
        let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
        let rgb = IccConversion::compile(&srgb, 3).expect("sRGB compiles");
        assert!(
            rgb.cmyk_clut().is_none(),
            "no table for a three-channel profile"
        );
    }
}
