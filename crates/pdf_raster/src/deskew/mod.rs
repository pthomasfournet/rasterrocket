//! Document deskew: detect skew angle and rotate to correct it.
//!
//! # Algorithm
//!
//! Two-phase pipeline:
//!
//! 1. **Detection** (`detect` module) — intensity-weighted projection-profile
//!    sweep.  Dark pixels are weighted by `255 − intensity` rather than binary
//!    foreground/background, so no binarisation threshold is needed.  AVX-512
//!    `VPSADBW` row sums, Rayon-parallel angle sweep, binary-search refinement
//!    to 0.01° convergence.
//!
//! 2. **Rotation** (`rotate` module) — bilinear rotation by the detected angle.
//!    For 8-bit gray: GPU path (CUDA NPP `nppiRotate`) when the `gpu-deskew`
//!    feature is enabled, the backend policy is [`BackendPolicy::ForceCuda`]
//!    and a CUDA device is available; CPU bilinear rotation otherwise.
//!    16-bit gray is measured on its 8-bit luma and rotated on the CPU at 16
//!    bits, through the scalar kernel the 8-bit SIMD tiers are exact with.
//!    Rotation is clockwise-positive: a detected CCW tilt of +θ° is corrected
//!    by a CW rotation by θ°.
//!
//! # Usage
//!
//! ```rust,no_run
//! use color::Gray8;
//! use raster::Bitmap;
//! use rasterrocket::{BackendPolicy, deskew};
//!
//! let mut page: Bitmap<Gray8> = todo!();
//! deskew::apply(&mut page, BackendPolicy::default()).expect("deskew failed");
//! ```

mod bilinear;
pub(crate) mod detect;
pub(crate) mod rotate;

use color::{Gray8, Gray16, Pixel, Sample};
use raster::Bitmap;

use crate::BackendPolicy;

/// Minimum skew magnitude that triggers rotation.
///
/// Angles smaller than this are considered measurement noise and are not
/// corrected.  Matches Tesseract's own skip threshold for consistency.
const MIN_CORRECT_DEG: f32 = 0.05;

/// Maximum skew magnitude searched.  Scanner jams rarely exceed ±10°; PDFs
/// with larger rotation use `/Rotate` which is handled at the page-geometry
/// level, not here.
pub(crate) const MAX_SEARCH_DEG: f32 = 7.0;

/// Largest width or height of an image deskew accepts.
///
/// The rotation's source coordinates are `f32`, whose resolution at
/// 2^15 px is 2^-9 px, finer than its 8-bit interpolation weights; the
/// AVX-512 rotation gathers bytes at `i32` offsets, which an image of
/// 2^15 × 2^15 bytes (2^30) leaves inside `i32`.
pub const DESKEW_MAX_SIDE: u32 = 32_768;

/// Error type for deskew operations: an image larger than
/// [`DESKEW_MAX_SIDE`], or a rotation that failed (see [`apply`]).
#[derive(Debug)]
pub struct DeskewError(pub(crate) String);

impl std::fmt::Display for DeskewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeskewError {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for color::Gray8 {}
    impl Sealed for color::Gray16 {}
}

/// A gray pixel [`apply`] deskews: [`Gray8`] or [`Gray16`].  Sealed.
pub trait DeskewPixel: Pixel + sealed::Sealed {
    /// The skew of `img` in degrees, measured on its 8-bit luma.
    #[doc(hidden)]
    fn skew_deg(img: &Bitmap<Self>) -> f32;

    /// `img` rotated clockwise by `angle_deg` degrees, under `policy`.
    #[doc(hidden)]
    fn rotated(img: &Bitmap<Self>, angle_deg: f32, policy: BackendPolicy) -> Bitmap<Self>;

    /// The sample as `f32` (exact).
    #[doc(hidden)]
    fn to_f32(s: Self::Sample) -> f32;
}

impl DeskewPixel for Gray8 {
    fn skew_deg(img: &Bitmap<Self>) -> f32 {
        detect::find_skew_deg(img)
    }

    fn rotated(img: &Bitmap<Self>, angle_deg: f32, policy: BackendPolicy) -> Bitmap<Self> {
        rotate::rotated(img, angle_deg, policy)
    }

    fn to_f32(s: u8) -> f32 {
        f32::from(s)
    }
}

impl DeskewPixel for Gray16 {
    /// Measured on the nearest 8-bit levels: the projection profile needs
    /// no more resolution than 8 bits gives.
    fn skew_deg(img: &Bitmap<Self>) -> f32 {
        let w = img.width as usize;
        let mut luma = Vec::with_capacity(w * img.height as usize);
        for y in 0..img.height {
            luma.extend(img.row_samples(y)[..w].iter().map(|&v| v.to_u8()));
        }
        detect::find_skew_deg(&Bitmap::<Gray8>::from_packed(img.width, img.height, luma))
    }

    /// On the CPU, whatever the policy: the GPU rotation is 8-bit.
    fn rotated(img: &Bitmap<Self>, angle_deg: f32, _policy: BackendPolicy) -> Bitmap<Self> {
        bilinear::rotate_scalar(img, angle_deg)
    }

    fn to_f32(s: u16) -> f32 {
        f32::from(s)
    }
}

/// Detect the skew angle of `img`, 8- or 16-bit gray, and rotate it
/// in-place to correct it.
///
/// An 8-bit page rotates on the GPU only under
/// [`BackendPolicy::ForceCuda`] in a build with the `gpu-deskew` feature,
/// on the CPU otherwise; a 16-bit page is measured on its nearest 8-bit
/// levels and rotated on the CPU at 16 bits.
///
/// Does nothing if the detected angle is below `MIN_CORRECT_DEG`.  A
/// failed detection counts as zero skew, and leaves the image as it is.
///
/// # Errors
///
/// [`DeskewError`] when a side of `img` exceeds [`DESKEW_MAX_SIDE`]; `img`
/// is then left as it is.  A GPU rotation that fails falls back to the CPU
/// rotation, which cannot fail.
pub fn apply<P: DeskewPixel>(
    img: &mut Bitmap<P>,
    policy: BackendPolicy,
) -> Result<(), DeskewError> {
    check_size(img)?;
    let angle_deg = P::skew_deg(img);
    if angle_deg.abs() >= MIN_CORRECT_DEG {
        *img = P::rotated(img, angle_deg, policy);
    }
    Ok(())
}

/// [`apply`] writing the corrected image to `out`, row `y` at
/// `out[y * stride..y * stride + img.width]`, and leaving `img` as it is.
/// A failed GPU rotation falls back to the CPU, which cannot fail.
///
/// `out` must hold `(img.height - 1) * stride + img.width` bytes and
/// `stride ≥ img.width`.
///
/// # Errors
///
/// As [`apply`]; `out` is then untouched.
pub(crate) fn apply_into(
    img: &Bitmap<Gray8>,
    out: &mut [u8],
    stride: usize,
    policy: BackendPolicy,
) -> Result<(), DeskewError> {
    check_size(img)?;
    let angle_deg = detect::find_skew_deg(img);
    if angle_deg.abs() < MIN_CORRECT_DEG {
        copy_rows(img, out, stride);
        return Ok(());
    }
    match rotate::rotated_on_gpu(img, angle_deg, policy) {
        Some(rotated) => copy_rows(&rotated, out, stride),
        None => rotate::rotate_cpu_into(img, angle_deg, out, stride),
    }
    Ok(())
}

/// [`DeskewError`] unless both sides of `img` are within [`DESKEW_MAX_SIDE`].
fn check_size<P: Pixel>(img: &Bitmap<P>) -> Result<(), DeskewError> {
    if img.width > DESKEW_MAX_SIDE || img.height > DESKEW_MAX_SIDE {
        return Err(DeskewError(format!(
            "deskew applies to pages up to {DESKEW_MAX_SIDE} px a side, and this one is \
             {}×{}; lower the resolution or turn deskew off",
            img.width, img.height
        )));
    }
    Ok(())
}

/// Copy the rows of `img` to `out`, row `y` at `out[y * stride..]`.
fn copy_rows(img: &Bitmap<Gray8>, out: &mut [u8], stride: usize) {
    let w = img.width as usize;
    for y in 0..img.height {
        out[y as usize * stride..][..w].copy_from_slice(&img.row_bytes(y)[..w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_past_the_deskew_bound_is_refused_and_left_as_it_is() {
        for (w, h) in [(DESKEW_MAX_SIDE + 1, 2), (2, DESKEW_MAX_SIDE + 1)] {
            let mut img = Bitmap::<Gray8>::new_filled(w, h, 1, false, 7);
            let err = apply(&mut img, BackendPolicy::CpuOnly).expect_err("over the bound");
            assert!(err.to_string().contains("32768 px a side"), "{err}");
            assert!(img.data().iter().all(|&v| v == 7));
            let mut out = vec![1u8; img.data().len()];
            assert!(apply_into(&img, &mut out, w as usize, BackendPolicy::CpuOnly).is_err());
            assert!(out.iter().all(|&v| v == 1), "out is untouched");
        }
        let mut img = Bitmap::<Gray8>::new_filled(DESKEW_MAX_SIDE, 2, 1, false, 255);
        assert!(apply(&mut img, BackendPolicy::CpuOnly).is_ok());
    }

    /// FNV-1a 64 of `bytes`.
    fn digest(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// A 600 × 400 page of text-like bars tilted by `deg`, on white.
    pub(crate) fn tilted_text(deg: f32) -> Bitmap<Gray8> {
        let (sin, cos) = deg.to_radians().sin_cos();
        let mut img = Bitmap::<Gray8>::new_filled(600, 400, 1, false, 250);
        for y in 0..400u32 {
            for (x, px) in img.row_bytes_mut(y).iter_mut().enumerate().take(600) {
                #[expect(clippy::cast_precision_loss, reason = "coordinates below 2^10")]
                let (dx, dy) = (x as f32 - 300.0, y as f32 - 200.0);
                let across = cos.mul_add(dy, -sin * dx) + 200.0;
                let along = cos.mul_add(dx, sin * dy) + 300.0;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the coordinates are clamped to [0, 1000] first"
                )]
                let (row, col) = (
                    across.clamp(0.0, 1000.0) as u32,
                    along.clamp(0.0, 1000.0) as u32,
                );
                if (40..360).contains(&row)
                    && (40..560).contains(&col)
                    && row % 16 < 7
                    && col % 23 != 0
                {
                    *px = 20;
                }
            }
        }
        img
    }

    /// The 8-bit rotation and deskew of fixed inputs, digested before the
    /// rotation became generic over the sample.
    #[test]
    fn eight_bit_deskew_matches_its_golden_digests() {
        let page = tilted_text(2.5);
        let mut fixed = tilted_text(2.5);
        apply(&mut fixed, BackendPolicy::CpuOnly).expect("within bounds");
        let got = [
            digest(rotate::rotate_cpu(&page, 3.25).data()),
            digest(rotate::rotate_cpu(&page, -6.0).data()),
            digest(fixed.data()),
        ];
        assert_ne!(fixed.data(), page.data(), "the page is deskewed");
        assert_eq!(got, GOLDEN, "{got:#x?}");
    }

    const GOLDEN: [u64; 3] = [
        0x7e57_dda9_e717_4282,
        0xe770_cd2a_b45a_c65e,
        0x8648_394c_d633_a04c,
    ];

    /// A 16-bit page is measured on its 8-bit levels and rotated at 16
    /// bits: widened from an 8-bit page, it finds the same angle, and its
    /// rotation rounds to within one 8-bit level of the 8-bit rotation.
    #[test]
    fn a_16_bit_page_deskews_like_its_8_bit_levels() {
        let eight = tilted_text(2.5);
        let wide: Vec<u16> = eight.data().iter().map(|&v| u16::from_u8(v)).collect();
        let mut sixteen = Bitmap::<Gray16>::from_packed(eight.width, eight.height, wide);
        let angle = Gray8::skew_deg(&eight);
        assert!(angle.abs() >= MIN_CORRECT_DEG, "{angle}");
        assert_eq!(Gray16::skew_deg(&sixteen).to_bits(), angle.to_bits());
        let mut fixed = tilted_text(2.5);
        apply(&mut fixed, BackendPolicy::CpuOnly).expect("within bounds");
        apply(&mut sixteen, BackendPolicy::ForceCuda).expect("within bounds");
        let mut finer = false;
        for (&deep, &narrow) in sixteen.data().iter().zip(fixed.data()) {
            assert!(deep.to_u8().abs_diff(narrow) <= 1, "{deep} vs {narrow}");
            finer |= !deep.is_multiple_of(257);
        }
        assert!(finer, "the blend keeps levels between the 8-bit ones");
        let mut big = Bitmap::<Gray16>::new(DESKEW_MAX_SIDE + 1, 2, 1, false);
        assert!(apply(&mut big, BackendPolicy::CpuOnly).is_err());
    }
}
