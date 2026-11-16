//! GPU decoder abstraction traits.
//!
//! These traits provide a stable interface over hardware-specific decoder
//! backends so that callers do not depend directly on the concrete CUDA,
//! VA-API, or Metal types.
//!
//! # Design notes
//!
//! - All traits are `Send` but **not** `Sync`.  GPU contexts (CUDA, VA-API) are
//!   thread-affine: only the thread that created the context may use it.  Each
//!   Rayon worker thread owns its own `Box<dyn …>` instance; no shared state or
//!   locking is required.
//! - The trait definitions themselves are never `#[cfg]`-gated.  Only the
//!   concrete `impl` blocks are gated.  This keeps the public API stable across
//!   all feature combinations and lets callers write feature-independent code.
//! - Concrete error types are wrapped in [`GpuDecodeError`] at the trait
//!   boundary; inner variants retain the original error for inspection.
//!
//! # Current implementations
//!
//! | Trait | Impl | Feature flag |
//! |---|---|---|
//! | [`GpuJpegDecoder`] | `NvJpegDecoder` | `nvjpeg` |
//! | [`GpuJpegDecoder`] | `VapiJpegDecoder` | `vaapi` |
//! | [`GpuJpeg2kDecoder`] | `NvJpeg2kDecoder` | `nvjpeg2k` |

use std::fmt;

// ── Unified error type ────────────────────────────────────────────────────────

/// Unified error returned by all GPU decoder trait methods.
///
/// The inner `Box<dyn std::error::Error + Send + Sync>` carries the original
/// backend-specific error.  [`Display`](fmt::Display) delegates to the inner
/// error directly; [`std::error::Error::source`] returns `None` because the
/// inner message is already included in the `Display` output — exposing it via
/// `source` would cause `{:#}` alternate-display to print the message twice.
#[derive(Debug)]
pub struct GpuDecodeError(Box<dyn std::error::Error + Send + Sync + 'static>);

impl GpuDecodeError {
    /// Wrap any [`std::error::Error`] into a [`GpuDecodeError`].
    pub fn new<E: std::error::Error + Send + Sync + 'static>(e: E) -> Self {
        Self(Box::new(e))
    }
}

impl fmt::Display for GpuDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate directly — no prefix — so the inner message is the full message.
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for GpuDecodeError {}

// ── Decoded image result ──────────────────────────────────────────────────────

/// Decoded image pixels returned by the GPU decoder traits.
///
/// Pixels are always interleaved: 1 byte/pixel for gray, 3 bytes/pixel for RGB.
/// CMYK images are not decoded through this path — they fall through to the CPU.
#[derive(Debug)]
pub struct DecodedImage {
    /// Pixel bytes (interleaved, 8-bit per channel).
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of bytes per pixel (1 = gray, 3 = RGB).
    pub components: u8,
}

// ── GpuJpegDecoder ────────────────────────────────────────────────────────────

/// Hardware-accelerated JPEG baseline decoder.
///
/// Each instance is bound to one GPU context and one worker thread.  Do not
/// share instances across threads; create one per Rayon worker with
/// `ThreadLocal<Box<dyn GpuJpegDecoder>>`.
///
/// # CMYK limitation
///
/// JPEG CMYK images are **not** supported by any current implementation.
/// Both nvJPEG and VA-API hardware decoders handle YCbCr/grayscale only;
/// CMYK JPEG streams must be routed to the CPU decoder.  Implementations
/// return [`GpuDecodeError`] for CMYK rather than silently producing wrong
/// output.
pub trait GpuJpegDecoder: Send {
    /// Decode a JPEG bytestream to an interleaved Gray or RGB u8 buffer.
    ///
    /// `width` and `height` are the dimensions declared in the PDF stream
    /// dictionary and must match the JPEG SOF markers; an error is returned if
    /// they differ.
    ///
    /// Returns the decoded pixels plus the actual decoded dimensions (which may
    /// differ from `width`/`height` only if the decoder resamples).
    ///
    /// # Errors
    ///
    /// Returns [`GpuDecodeError`] on hardware decode failure, CMYK input,
    /// or dimension mismatch between the PDF dict and the JPEG SOF markers.
    fn decode_jpeg(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<DecodedImage, GpuDecodeError>;
}

// ── GpuJpeg2kDecoder ──────────────────────────────────────────────────────────

/// Hardware-accelerated JPEG 2000 decoder.
///
/// Same threading model as [`GpuJpegDecoder`]: one instance per worker thread,
/// `Send` but not `Sync`.
///
/// Only 1-component (gray) and 3-component (RGB) images are accelerated.
/// Sub-sampled chroma images and images with more than 3 components fall
/// through to the CPU decoder.
pub trait GpuJpeg2kDecoder: Send {
    /// Decode a JPEG 2000 bytestream to an interleaved Gray or RGB u8 buffer.
    ///
    /// # Errors
    ///
    /// Returns [`GpuDecodeError`] on hardware decode failure, unsupported
    /// component counts (not 1 or 3), or sub-sampled chroma images.
    fn decode_jpeg2k(&mut self, data: &[u8]) -> Result<DecodedImage, GpuDecodeError>;
}

// ── NvJpegDecoder impl ────────────────────────────────────────────────────────

/// Error for dimension mismatch between PDF dict and JPEG SOF markers.
#[cfg(feature = "nvjpeg")]
#[derive(Debug)]
struct DimensionMismatch {
    expected: (u32, u32),
    actual: (u32, u32),
}

#[cfg(feature = "nvjpeg")]
impl fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ew, eh) = self.expected;
        let (aw, ah) = self.actual;
        write!(
            f,
            "JPEG dimensions mismatch: PDF declared {ew}×{eh}, decoder returned {aw}×{ah}"
        )
    }
}

#[cfg(feature = "nvjpeg")]
impl std::error::Error for DimensionMismatch {}

#[cfg(feature = "nvjpeg")]
impl GpuJpegDecoder for crate::nvjpeg::NvJpegDecoder {
    fn decode_jpeg(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<DecodedImage, GpuDecodeError> {
        let decoded = self.decode_sync(data).map_err(GpuDecodeError::new)?;
        // Validate dimensions against PDF dictionary.
        if decoded.width != width || decoded.height != height {
            return Err(GpuDecodeError::new(DimensionMismatch {
                expected: (width, height),
                actual: (decoded.width, decoded.height),
            }));
        }
        let components = match decoded.color_space {
            crate::nvjpeg::JpegColorSpace::Gray => 1,
            crate::nvjpeg::JpegColorSpace::Rgb => 3,
        };
        Ok(DecodedImage {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            components,
        })
    }
}

// ── NvJpeg2kDecoder impl ──────────────────────────────────────────────────────

#[cfg(feature = "nvjpeg2k")]
impl GpuJpeg2kDecoder for crate::nvjpeg2k::NvJpeg2kDecoder {
    fn decode_jpeg2k(&mut self, data: &[u8]) -> Result<DecodedImage, GpuDecodeError> {
        let decoded = self.decode_sync(data).map_err(GpuDecodeError::new)?;
        let components = match decoded.color_space {
            crate::nvjpeg2k::Jpeg2kColorSpace::Gray => 1,
            crate::nvjpeg2k::Jpeg2kColorSpace::Rgb => 3,
        };
        Ok(DecodedImage {
            data: decoded.data,
            width: decoded.width,
            height: decoded.height,
            components,
        })
    }
}
