//! Image rotation for deskew correction.
//!
//! Rotates a grayscale bitmap by a given angle (in degrees, CW positive)
//! using bilinear interpolation; the scalar kernel is in the `bilinear`
//! module.  Background pixels introduced by the rotation are white
//! (`Sample::MAX`, the scanner background convention).
//!
//! # Implementation
//!
//! **CPU path** (always available, 8 and 16 bits): per-output-pixel inverse
//! mapping with bilinear interpolation, one row at a time through the
//! fastest row kernel of the machine (`simd` module), each bit-exact with
//! the scalar kernel.
//!
//! **GPU path** (`gpu-deskew` feature, optional): CUDA NPP
//! `nppiRotate_8u_C1R_Ctx` and `nppiRotate_16u_C1R_Ctx` with hardware
//! bilinear via texture units.  ~0.3–0.5 ms for a 2550×3300 image
//! on RTX 5070.  Taken only under [`BackendPolicy::ForceCuda`], where a
//! failed GPU rotation, as without a CUDA device, is an error.  NPP's edge
//! rule is not the CPU's, so the samples whose source position lies within
//! 2 pixels of an edge of the sampled range are recomputed on the host
//! with the scalar kernel; the border is then the CPU's, and every other
//! sample is NPP's bilinear, within 2 levels of the CPU's at 8 bits and
//! 2 × 257 at 16 bits.

use color::Gray8;
use raster::Bitmap;

#[cfg(feature = "gpu-deskew")]
use super::bilinear::rotate_border_scalar;
use super::bilinear::{rotate_cpu_with, rotate_rows};
use super::simd::dispatch_rotate_row;
use super::{DeskewError, DeskewPixel};
use crate::BackendPolicy;

/// `img` rotated by `angle_deg` degrees (clockwise positive).
///
/// A new bitmap of the same dimensions holds the rotated content;
/// out-of-bounds source pixels are white (`Sample::MAX`).  The rotation
/// runs on the GPU when [`rotated_on_gpu`] gives a result under `policy`,
/// and on the CPU otherwise.
///
/// # Errors
///
/// As [`rotated_on_gpu`].
pub(crate) fn rotated<P: DeskewPixel>(
    img: &Bitmap<P>,
    angle_deg: f32,
    policy: BackendPolicy,
) -> Result<Bitmap<P>, DeskewError> {
    Ok(rotated_on_gpu(img, angle_deg, policy)?.unwrap_or_else(|| rotate_cpu(img, angle_deg)))
}

/// `img` rotated by `angle_deg` degrees clockwise on the GPU, or `None`
/// when [`rotates_on_gpu`] refuses `policy`; the caller then rotates on the
/// CPU.
///
/// # Errors
///
/// [`DeskewError::GpuRotation`] when `policy` requires the GPU rotation
/// and it fails.
#[cfg(feature = "gpu-deskew")]
pub(crate) fn rotated_on_gpu<P: DeskewPixel>(
    img: &Bitmap<P>,
    angle_deg: f32,
    policy: BackendPolicy,
) -> Result<Option<Bitmap<P>>, DeskewError> {
    if !rotates_on_gpu(policy) {
        return Ok(None);
    }
    rotate_gpu(img, angle_deg).map(Some)
}

/// No GPU rotation without the `gpu-deskew` feature: always `Ok(None)`.
#[cfg(not(feature = "gpu-deskew"))]
#[expect(
    clippy::unnecessary_wraps,
    reason = "the signature of the `gpu-deskew` build, whose GPU rotation can fail"
)]
pub(crate) const fn rotated_on_gpu<P: DeskewPixel>(
    _img: &Bitmap<P>,
    _angle_deg: f32,
    _policy: BackendPolicy,
) -> Result<Option<Bitmap<P>>, DeskewError> {
    Ok(None)
}

/// Whether `policy` rotates on the GPU.  The rotation is a CUDA kernel, so
/// only `ForceCuda` takes it: `Auto` creates no CUDA context for a page
/// whose CPU rotation costs less than the context's start-up, and
/// `CpuOnly`, `ForceVulkan` and `ForceVaapi` make no CUDA call.
#[cfg(feature = "gpu-deskew")]
const fn rotates_on_gpu(policy: BackendPolicy) -> bool {
    match policy {
        BackendPolicy::ForceCuda => true,
        BackendPolicy::Auto | BackendPolicy::CpuOnly | BackendPolicy::ForceVulkan => false,
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => false,
    }
}

// ── Public CPU path ───────────────────────────────────────────────────────────

/// CPU bilinear rotation (clockwise-positive).
///
/// Uses inverse mapping: for each output pixel `(ox, oy)`, compute the
/// corresponding source coordinates `(sx, sy)` by applying a CW rotation
/// matrix centred on the image centre, then bilinear-interpolate from the
/// four surrounding source pixels.
///
/// Out-of-bounds source pixels are white (`Sample::MAX`).  The last valid
/// sample origin is (w-1, h-1); on that boundary the 2×2 footprint is
/// clamped to the edge.
pub(crate) fn rotate_cpu<P: DeskewPixel>(src: &Bitmap<P>, angle_deg: f32) -> Bitmap<P> {
    rotate_cpu_with(src, angle_deg, dispatch_rotate_row)
}

/// [`rotate_cpu`] writing output row `y` to
/// `dst[y * stride..y * stride + src.width]`.
///
/// `dst` must hold `(src.height - 1) * stride + src.width` bytes and
/// `stride ≥ src.width`.
pub(crate) fn rotate_cpu_into(src: &Bitmap<Gray8>, angle_deg: f32, dst: &mut [u8], stride: usize) {
    rotate_rows(src, angle_deg, dst, stride, dispatch_rotate_row);
}

/// GPU rotation via CUDA NPP (`nppiRotate_8u_C1R_Ctx` for 8-bit samples,
/// `nppiRotate_16u_C1R_Ctx` for 16-bit ones), its border then recomputed
/// by [`rotate_border_scalar`].
#[cfg(feature = "gpu-deskew")]
pub(crate) fn rotate_gpu<P: DeskewPixel>(
    src: &Bitmap<P>,
    angle_deg: f32,
) -> Result<Bitmap<P>, DeskewError> {
    use color::Sample as _;

    let gpu_err = |e: gpu::npp_rotate::NppRotateError| DeskewError::GpuRotation(e.to_string());
    let (data, stride, w, h) = (src.data(), src.stride, src.width, src.height);
    // The rotation returns `width × height` samples with no row padding,
    // the layout of a bitmap with a row pad of 1.
    let mut dst = Bitmap::<P>::new(w, h, 1, false);
    let pixels: Vec<P::Sample> = if let Some(bytes) = P::Sample::as_u8(data) {
        let out = gpu::npp_rotate::rotate_gray8(bytes, stride, w, h, angle_deg).map_err(gpu_err)?;
        P::Sample::from_u8_slice(&out).map(<[_]>::to_vec)
    } else if let Some(wide) = P::Sample::as_u16(data) {
        let out = gpu::npp_rotate::rotate_gray16(wide, stride, w, h, angle_deg).map_err(gpu_err)?;
        P::Sample::from_u16_slice(&out).map(<[_]>::to_vec)
    } else {
        None
    }
    .ok_or_else(|| DeskewError::GpuRotation("a sample of neither 8 nor 16 bits".to_owned()))?;
    if pixels.len() != dst.data().len() {
        return Err(DeskewError::GpuRotation(format!(
            "GPU rotation returned {} samples for a {w}×{h} image ({} expected)",
            pixels.len(),
            dst.data().len()
        )));
    }
    dst.data_mut().copy_from_slice(&pixels);
    let stride = dst.stride;
    rotate_border_scalar(src, angle_deg, dst.data_mut(), stride);
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotating by 0° is a no-op (pixels should be unchanged).
    #[test]
    fn rotate_zero_is_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        // Write a checkerboard pattern.
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                *px = if (x + y as usize).is_multiple_of(2) {
                    0
                } else {
                    128
                };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        // Centre pixels should be identical; edges may differ (white fill vs original).
        let r_src = &src.row_bytes(32)[1..63];
        let r_dst = &rotated.row_bytes(32)[1..63];
        assert_eq!(r_src, r_dst, "centre row should be unchanged at 0°");

        // Pin the OOB-guard strict-inequality boundaries: at 0° the inverse
        // map is identity, so output `(0, y)` samples source `x0 = 0` and
        // output `(sx_max, y)` samples `x0 = sx_max = w - 1 = 63`.  If the
        // guard `<` → `<=`, x0=0 wrongly becomes OOB → 255.  If `>` → `>=`,
        // x0=63 wrongly becomes OOB → 255.
        assert_eq!(
            rotated.row_bytes(32)[0],
            src.row_bytes(32)[0],
            "x0 = 0 must be in-bounds (guards `x0 < 0`, not `x0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(32)[63],
            src.row_bytes(32)[63],
            "x0 = sx_max must be in-bounds (guards `x0 > sx_max`, not `x0 >= sx_max`)"
        );
        // And the y-axis boundaries: row 0 and row 63.
        assert_eq!(
            rotated.row_bytes(0)[32],
            src.row_bytes(0)[32],
            "y0 = 0 must be in-bounds (guards `y0 < 0`, not `y0 <= 0`)"
        );
        assert_eq!(
            rotated.row_bytes(63)[32],
            src.row_bytes(63)[32],
            "y0 = sy_max must be in-bounds (guards `y0 > sy_max`, not `y0 >= sy_max`)"
        );
    }

    /// At 0° the inverse map is the identity, so every pixel — including
    /// the last row and column — must survive: edge-clamped bilinear
    /// sampling degrades to nearest-neighbour exactly on the boundary
    /// instead of white-filling it.
    #[test]
    fn rotate_zero_preserves_last_row_and_column() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                *px = if (x + y as usize).is_multiple_of(2) {
                    0
                } else {
                    128
                };
            }
        }
        let rotated = rotate_cpu(&src, 0.0);
        for y in 0..64u32 {
            assert_eq!(
                src.row_bytes(y),
                rotated.row_bytes(y),
                "row {y} must be unchanged at 0°"
            );
        }
    }

    /// Of the policies, only `ForceCuda` rotates on the GPU.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn only_force_cuda_rotates_on_the_gpu() {
        assert!(rotates_on_gpu(BackendPolicy::ForceCuda));
        assert!(!rotates_on_gpu(BackendPolicy::Auto));
        assert!(!rotates_on_gpu(BackendPolicy::CpuOnly));
        assert!(!rotates_on_gpu(BackendPolicy::ForceVulkan));
        #[cfg(feature = "vaapi")]
        assert!(!rotates_on_gpu(BackendPolicy::ForceVaapi));
    }

    /// A policy other than `ForceCuda` yields no GPU rotation, with or
    /// without a CUDA device.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn rotated_on_gpu_is_none_unless_cuda_is_forced() {
        let img = Bitmap::<Gray8>::new_filled(64, 64, 1, false, 128);
        for policy in [
            BackendPolicy::Auto,
            BackendPolicy::CpuOnly,
            BackendPolicy::ForceVulkan,
            #[cfg(feature = "vaapi")]
            BackendPolicy::ForceVaapi,
        ] {
            assert!(
                matches!(rotated_on_gpu(&img, 3.0, policy), Ok(None)),
                "{policy:?}"
            );
        }
    }

    /// Under `ForceCuda` the rotation is the GPU's or an error, never a
    /// silent CPU rotation: with `CUDA_VISIBLE_DEVICES` empty it is
    /// [`DeskewError::GpuRotation`].
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn forced_cuda_rotation_is_the_gpus_or_an_error() {
        let img = Bitmap::<Gray8>::new_filled(64, 64, 1, false, 128);
        match rotated_on_gpu(&img, 3.0, BackendPolicy::ForceCuda) {
            Ok(Some(out)) => assert_eq!((out.width, out.height), (64, 64)),
            Err(DeskewError::GpuRotation(_)) => {}
            Ok(None) => panic!("ForceCuda declined the GPU rotation"),
            Err(e) => panic!("ForceCuda rotation failed with {e:?}"),
        }
        let rotated = rotated(&img, 3.0, BackendPolicy::ForceCuda);
        if std::env::var_os("CUDA_VISIBLE_DEVICES").is_some_and(|v| v.is_empty()) {
            assert!(
                matches!(rotated, Err(DeskewError::GpuRotation(_))),
                "no CUDA device, yet a ForceCuda rotation"
            );
        }
    }

    /// `rotated` at 0° does not panic.
    #[test]
    fn rotated_zero_no_panic() {
        let img = Bitmap::<Gray8>::new(32, 32, 1, false);
        let out = rotated(&img, 0.0, BackendPolicy::CpuOnly).expect("CPU rotation");
        assert_eq!((out.width, out.height), (32, 32));
    }

    /// `rotated` returns the rotated image: the pixel content changes after
    /// a non-trivial rotation.
    #[test]
    fn rotated_modifies_pixels() {
        // 32×32 image with a single white-stripe column on the left half so
        // any non-no-op rotation visibly redistributes the white.
        let mut img = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = img.row_bytes_mut(y);
            row[5] = 255; // vertical white line at x=5
        }
        let before_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        // 45° rotation breaks the vertical line into diagonal pieces.
        img = rotated(&img, 45.0, BackendPolicy::CpuOnly).expect("CPU rotation");
        let after_col5 = (0..32u32).map(|y| img.row_bytes(y)[5]).collect::<Vec<_>>();
        assert_ne!(
            before_col5, after_col5,
            "rotated at 45° must change the pixels, not no-op"
        );
    }

    /// The centre pixel is the rotation pivot — at any angle it maps to
    /// itself, so the rotated image's centre must equal the source's centre
    /// (within bilinear-rounding tolerance).  This pins the centre-pivot
    /// shift formulae (`cx`, `cy` and the offset terms in `sx_base`/
    /// `sy_base`): any operator swap that mis-locates the pivot would shift
    /// the centre pixel away from its source value.
    ///
    /// Uses a centre-asymmetric pattern (single bright dot at the centre,
    /// black elsewhere) so any pivot drift onto a neighbouring black pixel
    /// changes the centre value dramatically.  A smooth gradient would let
    /// pivot-off-by-one mutations slip through because the bilinear samples
    /// of an off-by-one pivot still approximate the same gradient value.
    ///
    /// Tests three angles (12°, 31.7°, 89°) so a mutation that happens to
    /// preserve the pivot at one specific angle still fails at another.
    #[test]
    fn rotate_cpu_centre_pixel_is_rotation_fixed_point() {
        // Mid-grey dot at the centre (not 255) so the OOB-white fallback
        // (255) on a coordinate corruption is visibly different from a true
        // centre sample — catches mutations that produce NaN source
        // coordinates and silently route through the OOB guard.
        const DOT: u8 = 128;
        // Odd dimensions so cx, cy land on integer pixel centres (16.0).
        let mut src = Bitmap::<Gray8>::new(33, 33, 1, false);
        src.row_bytes_mut(16)[16] = DOT;

        for angle in [12.0_f32, 31.7, 89.0] {
            let rotated = rotate_cpu(&src, angle);
            let dst_centre = rotated.row_bytes(16)[16];
            // True bilinear sample at the rotated centre averages the DOT
            // pixel with some-fraction-of-zero neighbours, landing in
            // [~80, 128] depending on angle.  An off-by-one pivot would
            // sample 4 zero pixels (~0); a NaN-coordinate mutation would
            // OOB-fallback to 255.  Bracket both failure modes.
            assert!(
                (50..=200).contains(&dst_centre),
                "centre pixel must be a true bilinear sample of the centre dot at {angle}°: \
                 got dst={dst_centre} (expected ~50..=200); \
                 ~0 = pivot drift, 255 = NaN/OOB"
            );
        }
    }

    /// `rotate_cpu(rotate_cpu(S, 180), 180)` must equal S on the interior,
    /// modulo small bilinear-rounding error.  Catches any inner-loop
    /// arithmetic mutation that doesn't self-invert (e.g. `*` → `+` would
    /// break the double-rotation identity).
    #[test]
    fn rotate_cpu_180_then_180_round_trips() {
        let mut src = Bitmap::<Gray8>::new(32, 32, 1, false);
        for y in 0..32u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // (x * 7 + y * 11) % 256 keeps the cast in [0, 255].
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]"
                )]
                {
                    *px = ((x * 7 + y as usize * 11) % 256) as u8;
                }
            }
        }

        let once = rotate_cpu(&src, 180.0);
        let twice = rotate_cpu(&once, 180.0);

        // Interior pixels only — borders accumulate two white-fill rounds.
        // Bilinear sampling at 180° hits half-pixel coordinates (cx/cy are
        // .5 in odd-sized images, 31.5 here), so a ±2 tolerance absorbs the
        // double-round error.
        let mut max_diff = 0i32;
        for y in 4..28u32 {
            let sr = &src.row_bytes(y)[4..28];
            let tr = &twice.row_bytes(y)[4..28];
            for (&a, &b) in sr.iter().zip(tr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(
            max_diff <= 2,
            "180°×2 round-trip diverges from identity: max diff {max_diff} > 2"
        );
    }

    /// A 1-pixel-wide source rotated by a small angle must have white-fill
    /// (255) at the diagonal corners where the inverse mapping falls
    /// outside the source.  Pins the OOB-guard bounds checks
    /// (`x0 < 0 || y0 < 0 || x0 > sx_max || y0 > sy_max`) by exercising
    /// pixels that depend on the strict `>` / `<` comparison — if any
    /// boundary flips to `>=` / `<=`, an in-range pixel turns white or
    /// vice versa.
    #[test]
    fn rotate_cpu_oob_guard_fills_diagonal_with_white() {
        // 8×8 source, fully black (all zeros).  After a 5° rotation, the
        // four corner regions of the output map to OOB source coordinates
        // and get filled with white.
        let src = Bitmap::<Gray8>::new(8, 8, 1, false);
        let rotated = rotate_cpu(&src, 5.0);

        // Corner pixels of the rotated 8×8, with cx = cy = 3.5 and the
        // valid sample range [0, 7] × [0, 7]: at 5° the CW inverse map
        // sends (0,0) to (0.32, -0.29) and (0,7) to (-0.29, 7.29) — a
        // negative coordinate makes both genuinely OOB, so they white-fill.
        // (7,0) maps to (7.29, 0.32) and (7,7) to (6.68, 7.29): both floor
        // into the last valid column/row, where the clamped bilinear
        // footprint samples the (black) edge pixels.
        assert_eq!(
            rotated.row_bytes(0)[0],
            255,
            "(0,0) corner must be white-filled (sy < 0 at 5°)"
        );
        assert_eq!(
            rotated.row_bytes(7)[0],
            255,
            "(0,7) corner must be white-filled (sx < 0 at 5°)"
        );
        assert_eq!(
            rotated.row_bytes(0)[7],
            0,
            "(7,0) corner floors into the last column and must sample black"
        );
        assert_eq!(
            rotated.row_bytes(7)[7],
            0,
            "(7,7) corner floors into the last row and must sample black"
        );

        // Centre pixel: maps to (3.5, 3.5) which IS in-bounds; sample of
        // black source = 0.  If the bounds-check inverted to <= / >=, the
        // centre would also be white.
        assert_eq!(
            rotated.row_bytes(4)[4],
            0,
            "centre pixel must sample in-bounds black source, not OOB white"
        );
    }

    /// Rotating 360° should return approximately the original image.
    #[test]
    fn rotate_full_circle_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x ≤ 63 so x * 4 ≤ 252; well below u8::MAX, no truncation.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 so x * 4 ≤ 252 < 256; fits u8 trivially"
                )]
                {
                    *px = (x * 4) as u8;
                }
            }
        }
        let rotated = rotate_cpu(&src, 360.0);
        // Bilinear + floating-point: allow ≤2 absolute difference at any pixel.
        let mut max_diff = 0i32;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let dr = &rotated.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(dr.iter()) {
                max_diff = max_diff.max((i32::from(a) - i32::from(b)).abs());
            }
        }
        assert!(max_diff <= 2, "360° rotation max diff {max_diff} > 2");
    }

    /// A xorshift64 generator of test inputs.
    #[cfg(feature = "gpu-deskew")]
    struct XorShift(u64);

    #[cfg(feature = "gpu-deskew")]
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        /// A value in `lo..=hi`.
        fn within(&mut self, lo: u32, hi: u32) -> u32 {
            let span = u64::from(hi - lo) + 1;
            lo + u32::try_from(self.next() % span).expect("below the span")
        }

        /// An angle in `[-limit, limit]` degrees.
        #[cfg(feature = "gpu-validation")]
        #[expect(
            clippy::cast_precision_loss,
            reason = "a 24-bit fraction is exact in f32"
        )]
        fn angle(&mut self, limit: f32) -> f32 {
            let unit = (self.next() >> 40) as f32 / (1u32 << 24) as f32;
            unit.mul_add(2.0 * limit, -limit)
        }
    }

    /// A test coordinate or dimension, below 2^16, as `f32`.
    #[cfg(feature = "gpu-deskew")]
    fn small(v: u32) -> f32 {
        f32::from(u16::try_from(v).expect("a test dimension"))
    }

    /// A `w × h` image of row pad `pad` whose samples are `sample()`.
    #[cfg(feature = "gpu-deskew")]
    fn image_of<P: DeskewPixel>(
        (w, h): (u32, u32),
        pad: usize,
        mut sample: impl FnMut() -> i64,
    ) -> Bitmap<P> {
        let mut img = Bitmap::<P>::new(w, h, pad, false);
        for y in 0..h {
            for px in img.row_samples_mut(y) {
                *px = color::Sample::from_signed(sample());
            }
        }
        img
    }

    /// The border pass rewrites, with the CPU rotation's samples, every
    /// sample whose source position lies within 1.5 pixels of an edge of
    /// the sampled range, and no sample it does not rewrite.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn the_border_pass_rewrites_the_border_with_the_cpus_samples() {
        /// A value no rotation of the test images yields.
        const MARK: u8 = 7;
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        for angle in [0.0, 0.3, -2.0, 6.9, -45.0, 90.0, 180.0] {
            let (w, h) = (rng.within(1, 90), rng.within(1, 90));
            let src: Bitmap<Gray8> = image_of((w, h), 1, || i64::from(rng.within(100, 200)));
            let cpu = rotate_cpu(&src, angle);
            let mut dst = vec![MARK; cpu.data().len()];
            rotate_border_scalar(&src, angle, &mut dst, cpu.stride);
            let (rad, cx, cy) = (
                angle.to_radians(),
                (small(w) - 1.0) * 0.5,
                (small(h) - 1.0) * 0.5,
            );
            let near = |v: f32, end: f32| v.abs() < 1.5 || (v - end).abs() < 1.5;
            for oy in 0..h {
                let (got, want) = (&dst[oy as usize * cpu.stride..], cpu.row_bytes(oy));
                for (ox, (&g, &c)) in (0u32..).zip(got.iter().zip(want)) {
                    let (dx, dy) = (small(ox) - cx, small(oy) - cy);
                    let sx = rad.cos().mul_add(dx, -(rad.sin() * dy)) + cx;
                    let sy = rad.sin().mul_add(dx, rad.cos() * dy) + cy;
                    let at = format!("{w}×{h} at {angle}°, ({ox}, {oy}), source ({sx}, {sy})");
                    if near(sx, small(w)) || near(sy, small(h)) {
                        assert_eq!(g, c, "{at}: not rewritten");
                    } else {
                        assert!(g == MARK || g == c, "{at}: {g} is neither");
                    }
                }
            }
        }
    }

    /// Every sample of the GPU rotation of random images of samples `P`,
    /// of random sizes, row pads and angles, lies within `bound` levels of
    /// the CPU rotation's, the border included; the histogram of the
    /// differences.
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_rotation_within<P: DeskewPixel>(
        seed: u64,
        bound: i64,
    ) -> std::collections::BTreeMap<i64, usize> {
        use color::Sample as _;

        let max = <P::Sample as color::Sample>::MAX.signed();
        let mut rng = XorShift(seed);
        let mut hist = std::collections::BTreeMap::new();
        let fixed = [
            0.0,
            0.01,
            -0.01,
            super::super::MAX_SEARCH_DEG,
            -super::super::MAX_SEARCH_DEG,
            45.0,
            -90.0,
            180.0,
        ];
        let random: Vec<f32> = (0..24)
            .map(|_| rng.angle(super::super::MAX_SEARCH_DEG))
            .collect();
        for angle in fixed.into_iter().chain(random) {
            let (w, h) = (rng.within(1, 257), rng.within(1, 257));
            let pad = [1, 4, 64][rng.within(0, 2) as usize];
            let src: Bitmap<P> = image_of((w, h), pad, || {
                i64::try_from(rng.next() % (max.unsigned_abs() + 1)).expect("a sample")
            });
            let cpu = rotate_cpu(&src, angle);
            let gpu = rotate_gpu(&src, angle).expect("GPU rotation");
            for y in 0..h {
                for (x, (c, g)) in (0..).zip(cpu.row_samples(y).iter().zip(gpu.row_samples(y))) {
                    let d = (c.signed() - g.signed()).abs();
                    assert!(
                        d <= bound,
                        "{w}×{h} (pad {pad}) at {angle}°: ({x}, {y}) CPU {c:?}, GPU {g:?}"
                    );
                    *hist.entry(d).or_insert(0) += 1;
                }
            }
        }
        hist
    }

    /// The GPU rotation of 8-bit images agrees with the CPU's to within
    /// 2 levels at every sample, the border included.
    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_vs_cpu_rotation_parity() {
        let hist = gpu_rotation_within::<Gray8>(0x2545_F491_4F6C_DD1D, 2);
        eprintln!("8-bit |CPU - GPU| histogram: {hist:?}");
    }

    /// The GPU rotation of 16-bit images agrees with the CPU's to within
    /// 2 levels of 65 535 at every sample, the border included, and keeps
    /// 16-bit levels.
    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_vs_cpu_rotation_parity_at_16_bits() {
        let hist = gpu_rotation_within::<color::Gray16>(0x1234_5678_9ABC_DEF1, 2);
        eprintln!("16-bit |CPU - GPU| histogram: {hist:?}");
        let mut src = Bitmap::<color::Gray16>::new(128, 128, 1, false);
        for y in 0..128u32 {
            for (x, px) in (0usize..).zip(src.row_samples_mut(y).iter_mut()) {
                *px = u16::try_from((x * 523 + y as usize * 787) % 65_536).expect("below 2^16");
            }
        }
        let gpu = rotate_gpu(&src, 2.0).expect("GPU rotate failed");
        assert!(gpu.data().iter().any(|&v| v % 257 != 0));
    }

    /// Under `ForceCuda` a 16-bit page rotates on the GPU, or the rotation
    /// is an error; other policies rotate on the CPU.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn a_16_bit_page_rotates_on_the_gpu_only_under_force_cuda() {
        let img = Bitmap::<color::Gray16>::new_filled(64, 64, 1, false, 30_000);
        assert!(matches!(
            rotated_on_gpu(&img, 3.0, BackendPolicy::CpuOnly),
            Ok(None)
        ));
        match rotated_on_gpu(&img, 3.0, BackendPolicy::ForceCuda) {
            Ok(Some(out)) => {
                assert_eq!((out.width, out.height), (64, 64));
                assert_eq!(
                    out.row_samples(32)[32],
                    30_000,
                    "the centre is the source's"
                );
                assert_eq!(out.row_samples(0)[0], u16::MAX, "a corner is white");
            }
            Err(DeskewError::GpuRotation(_)) => {}
            Ok(None) => panic!("ForceCuda declined the 16-bit GPU rotation"),
            Err(e) => panic!("ForceCuda rotation failed with {e:?}"),
        }
    }

    #[test]
    #[cfg(all(feature = "gpu-deskew", feature = "gpu-validation"))]
    fn gpu_rotate_zero_is_near_identity() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in (0usize..).zip(row.iter_mut()) {
                *px = u8::try_from((x * 4 + y as usize * 2) % 256).expect("below 256");
            }
        }
        let gpu = rotate_gpu(&src, 0.0).expect("GPU rotate 0° failed");
        let mut max_diff = 0u8;
        for y in 4..60u32 {
            let sr = &src.row_bytes(y)[4..60];
            let gr = &gpu.row_bytes(y)[4..60];
            for (&a, &b) in sr.iter().zip(gr.iter()) {
                max_diff = max_diff.max(a.abs_diff(b));
            }
        }
        assert!(max_diff <= 2, "GPU 0° rotation max diff {max_diff} > 2");
    }

    /// Structural shape test for `rotate_gpu` — runs without a CUDA device.
    ///
    /// On a no-GPU machine, `rotate_gpu` returns `Err`; on a GPU machine it
    /// returns `Ok` with a bitmap matching the source dimensions.  This test
    /// accepts both outcomes but verifies the dimensions when the call
    /// succeeds.
    ///
    /// Pins the four whole-function-body replacement mutants
    /// (`Ok(Bitmap::new())`, `Ok(Bitmap::from_iter(...))`, etc.) that
    /// otherwise survive: each substitutes an empty / default Bitmap with
    /// dimensions ≠ source, which the size assertion catches even without
    /// a CUDA device. Pixel-content parity is covered by
    /// `gpu_vs_cpu_rotation_parity` under the `gpu-validation` gate.
    #[test]
    #[cfg(feature = "gpu-deskew")]
    fn rotate_gpu_preserves_dimensions_or_errors() {
        let mut src = Bitmap::<Gray8>::new(64, 64, 1, false);
        for y in 0..64u32 {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // x + y < 128, fits u8 trivially; bytemuck-equivalent cast.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "x ≤ 63 and y ≤ 63 so x + y ≤ 126; well below u8::MAX"
                )]
                {
                    *px = (x + y as usize) as u8;
                }
            }
        }
        // No CUDA device → Err is the expected outcome; only the Ok arm needs
        // assertions (and is where cargo-mutants exercises rotate_gpu).
        if let Ok(out) = rotate_gpu(&src, 1.0) {
            assert_eq!(
                out.width, src.width,
                "rotate_gpu must return a bitmap with the source width"
            );
            assert_eq!(
                out.height, src.height,
                "rotate_gpu must return a bitmap with the source height"
            );
            assert_eq!(
                out.data().len(),
                (src.width as usize) * (src.height as usize),
                "rotate_gpu output buffer must have width*height bytes (tightly packed)"
            );
        }
    }

    /// Timing smoke-test: `rotate_cpu` on an 8.4 MP image (2900×2900).  The
    /// test always passes — the time is printed for manual inspection.  Run
    /// with `--nocapture` to see it.
    ///
    /// A4 at 300 DPI ≈ 2480×3508 (8.7 MP); 2900×2900 is a round-number proxy.
    /// Ryzen 9 9900X3D, release build, 1.5°: ~8.5 ms through the AVX-512
    /// tier, ~31 ms through the scalar loop.
    #[test]
    #[ignore = "manual perf smoke test; run with `--ignored --nocapture` for timing"]
    fn rotate_cpu_8mp_timing() {
        let w = 2900u32;
        let h = 2900u32;
        let mut src = Bitmap::<Gray8>::new(w, h, 1, false);
        for y in 0..h {
            let row = src.row_bytes_mut(y);
            for (x, px) in row.iter_mut().enumerate() {
                // % 256 keeps the value in [0, 255]; cast is exact.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "modulo 256 keeps the value in [0, u8::MAX]; cast is exact"
                )]
                {
                    *px = ((x + y as usize * 3) % 256) as u8;
                }
            }
        }

        let t0 = std::time::Instant::now();
        let _ = rotate_cpu(&src, 1.5);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!(
            "rotate_cpu 2900×2900 ({:.1} MP): {elapsed_ms:.2} ms",
            f64::from(w) * f64::from(h) / 1_000_000.0,
        );
        // Soft threshold: warn rather than fail so CI on slow machines doesn't flake.
        if elapsed_ms > 20.0 {
            eprintln!(
                "WARN: rotate_cpu exceeded 20 ms ({elapsed_ms:.2} ms) — \
                 check which row tier was dispatched on this machine"
            );
        }
    }
}
