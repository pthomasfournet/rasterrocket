//! Document deskew: measure the skew of a gray page and rotate it to
//! correct it.
//!
//! # Algorithm
//!
//! 1. **Detection** (`detect` module): an intensity-weighted
//!    projection-profile sweep on a 4× reduction of the page, a confidence
//!    from the ratio of the best to the worst sweep score, and refinement
//!    on a 2× reduction.  Dark pixels weigh `255 − intensity`, so no
//!    binarisation threshold is needed.  A page without a confident angle
//!    within the sweep ([`detect`] returns `None`) is left as it is.
//!
//! 2. **Rotation** (`rotate` module): bilinear rotation by the detected
//!    angle, white outside the source.  For 8-bit gray: on the GPU (CUDA
//!    NPP `nppiRotate`) when the `gpu-deskew` feature is enabled, the
//!    backend policy is [`BackendPolicy::ForceCuda`] and a CUDA device is
//!    available; on the CPU otherwise.  16-bit gray is measured on its
//!    8-bit levels and rotated on the CPU at 16 bits, through the scalar
//!    kernel the 8-bit SIMD tiers are exact with.  Rotation is
//!    clockwise-positive: a detected counter-clockwise tilt of +θ° is
//!    corrected by a clockwise rotation by θ°.
//!
//! # Accuracy and limits
//!
//! The angle resolves about `2 / W` radians on a page `W` pixels wide:
//! 0.045° for an A4 page at 300 dpi, 0.09° at 150 dpi.  Skew is measured
//! within ±([`MAX_SEARCH_DEG`] − 0.25)°.  The measurement assumes lines of
//! text or other horizontal structure: a photograph, noise or a blank
//! page has no confident angle.
//!
//! # Usage
//!
//! ```rust,no_run
//! use color::Gray8;
//! use raster::Bitmap;
//! use rasterrocket::{BackendPolicy, deskew};
//!
//! let mut page: Bitmap<Gray8> = todo!();
//! match deskew::apply(&mut page, BackendPolicy::default()).expect("within DESKEW_MAX_SIDE") {
//!     Some(skew) => println!("{}° (confidence {})", skew.angle_deg, skew.confidence),
//!     None => println!("no confident angle; the page is unchanged"),
//! }
//! ```

mod bilinear;
pub(crate) mod detect;
pub(crate) mod rotate;

use color::{Gray8, Gray16, Pixel};
use raster::Bitmap;

use crate::BackendPolicy;

/// Least skew magnitude, in degrees, that [`apply`] corrects: about the
/// angular resolution of the measurement on a 300 dpi page (see the module
/// documentation).
pub const MIN_CORRECT_DEG: f32 = 0.05;

/// Largest skew magnitude searched, in degrees.  Quarter turns are the
/// page geometry's (`/Rotate`), not deskew's.
pub const MAX_SEARCH_DEG: f32 = 7.0;

/// Least confidence of a measured skew: the ratio of the best to the worst
/// score of the angle sweep.
pub const MIN_CONFIDENCE: f32 = 3.0;

/// A measured skew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Skew {
    /// The tilt of the page's lines in degrees, counter-clockwise
    /// positive, within ±([`MAX_SEARCH_DEG`] − 0.25).
    pub angle_deg: f32,
    /// The ratio of the best to the worst score of the angle sweep, at
    /// least [`MIN_CONFIDENCE`]: how much more the projection profile
    /// varies at the measured angle than at the least aligned one.
    pub confidence: f32,
}

/// Largest width or height of an image deskew accepts.
///
/// The rotation's source coordinates are `f32`, whose resolution at
/// 2^15 px is 2^-9 px, finer than its 8-bit interpolation weights; the
/// AVX-512 rotation kernel gathers bytes at `i32` offsets, which an image
/// of 2^15 × 2^15 bytes (2^30) leaves inside `i32`.
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
    /// The skew of `img`, measured on its 8-bit levels.
    #[doc(hidden)]
    fn measured(img: &Bitmap<Self>) -> Option<Skew>;

    /// `img` rotated clockwise by `angle_deg` degrees, under `policy`.
    #[doc(hidden)]
    fn rotated(img: &Bitmap<Self>, angle_deg: f32, policy: BackendPolicy) -> Bitmap<Self>;

    /// The sample as `f32` (exact).
    #[doc(hidden)]
    fn to_f32(s: Self::Sample) -> f32;
}

impl DeskewPixel for Gray8 {
    fn measured(img: &Bitmap<Self>) -> Option<Skew> {
        detect::detect(img)
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
    fn measured(img: &Bitmap<Self>) -> Option<Skew> {
        detect::detect(&img.converted())
    }

    /// On the CPU, whatever the policy: the GPU rotation is 8-bit.
    fn rotated(img: &Bitmap<Self>, angle_deg: f32, _policy: BackendPolicy) -> Bitmap<Self> {
        bilinear::rotate_scalar(img, angle_deg)
    }

    fn to_f32(s: u16) -> f32 {
        f32::from(s)
    }
}

/// The skew of `img`, 8- or 16-bit gray, or `None` when it has no
/// confident angle (see [`Skew`] and the module documentation).  A 16-bit
/// page is measured on its nearest 8-bit levels.
///
/// # Errors
///
/// [`DeskewError`] when a side of `img` exceeds [`DESKEW_MAX_SIDE`].
pub fn detect<P: DeskewPixel>(img: &Bitmap<P>) -> Result<Option<Skew>, DeskewError> {
    check_size(img)?;
    Ok(P::measured(img))
}

/// Measure the skew of `img`, 8- or 16-bit gray, and rotate it in place to
/// correct it; the skew measured ([`detect`]).
///
/// `img` is left as it is when the skew is `None` or its magnitude below
/// [`MIN_CORRECT_DEG`].
/// An 8-bit page rotates on the GPU only under
/// [`BackendPolicy::ForceCuda`] in a build with the `gpu-deskew` feature,
/// on the CPU otherwise; a 16-bit page is measured on its nearest 8-bit
/// levels and rotated on the CPU at 16 bits.
///
/// # Errors
///
/// [`DeskewError`] when a side of `img` exceeds [`DESKEW_MAX_SIDE`]; `img`
/// is then left as it is.  A GPU rotation that fails falls back to the CPU
/// rotation, which cannot fail.
pub fn apply<P: DeskewPixel>(
    img: &mut Bitmap<P>,
    policy: BackendPolicy,
) -> Result<Option<Skew>, DeskewError> {
    let skew = detect(img)?;
    if let Some(angle_deg) = correction(skew) {
        *img = P::rotated(img, angle_deg, policy);
    }
    Ok(skew)
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
) -> Result<Option<Skew>, DeskewError> {
    let skew = detect(img)?;
    let Some(angle_deg) = correction(skew) else {
        copy_rows(img, out, stride);
        return Ok(skew);
    };
    match rotate::rotated_on_gpu(img, angle_deg, policy) {
        Some(rotated) => copy_rows(&rotated, out, stride),
        None => rotate::rotate_cpu_into(img, angle_deg, out, stride),
    }
    Ok(skew)
}

/// The angle [`apply`] rotates by for `skew`: none without a skew or below
/// [`MIN_CORRECT_DEG`].
fn correction(skew: Option<Skew>) -> Option<f32> {
    skew.map(|s| s.angle_deg)
        .filter(|a| a.abs() >= MIN_CORRECT_DEG)
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
mod tests;
