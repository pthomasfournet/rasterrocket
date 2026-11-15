//! Input validation errors.

use std::fmt;

/// Largest accepted image width or height, in pixels.
///
/// Coordinates are carried as `f64` and converted back to indices through
/// a clamp to `[0, 2³¹]`; every coordinate below this bound round-trips
/// exactly.
pub const MAX_DIMENSION: usize = 1 << 30;

/// An input image or argument that the stages cannot operate on.
///
/// Returned only for malformed input.  A detector that does not find what
/// it looks for is not an error: the image passes through unchanged and
/// the [`Report`](crate::Report) states why.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CameraError {
    /// The image has zero width or zero height.
    ZeroDimension {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
    },
    /// The width or height exceeds [`MAX_DIMENSION`].
    TooLarge {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
    },
    /// The row stride is smaller than the width.
    StrideTooSmall {
        /// Row stride in samples.
        stride: usize,
        /// Width in pixels.
        width: usize,
    },
    /// The buffer holds fewer than `stride · (height − 1) + width` samples.
    BufferTooShort {
        /// Buffer length in samples.
        len: usize,
        /// Required length in samples.
        required: usize,
    },
    /// `stride · (height − 1) + width`, or twice the pixel area, does not
    /// fit in `usize`.
    Overflow {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Row stride in samples.
        stride: usize,
    },
    /// A rectangle is empty or extends beyond the image.
    RectOutOfBounds {
        /// The rectangle.
        rect: crate::Rect,
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
    },
    /// A split column is not strictly inside the image.
    SplitOutOfRange {
        /// The requested split column.
        x: usize,
        /// Image width in pixels.
        width: usize,
    },
    /// An image passed to [`Report::replay`](crate::Report::replay),
    /// [`Background::normalize`](crate::Background::normalize) or
    /// [`BorderMask::whiten`](crate::BorderMask::whiten) differs in size
    /// from the image the measurement was taken on.
    SizeMismatch {
        /// `(width, height)` of the processed image.
        expected: (usize, usize),
        /// `(width, height)` of the image passed.
        actual: (usize, usize),
    },
    /// A size argument in pixels is zero.
    ZeroSize {
        /// The name of the argument.
        parameter: &'static str,
    },
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroDimension { width, height } => write!(
                f,
                "image is {width}×{height}; width and height must both be at least 1"
            ),
            Self::TooLarge { width, height } => write!(
                f,
                "image is {width}×{height}; width and height must not exceed {MAX_DIMENSION}"
            ),
            Self::StrideTooSmall { stride, width } => write!(
                f,
                "row stride {stride} is smaller than the width {width}; stride must be ≥ width"
            ),
            Self::BufferTooShort { len, required } => write!(
                f,
                "image buffer holds {len} samples; stride × (height − 1) + width requires {required}"
            ),
            Self::Overflow {
                width,
                height,
                stride,
            } => write!(
                f,
                "image of {width}×{height} with stride {stride} overflows usize; \
                 stride × (height − 1) + width and 2 × width × height must fit"
            ),
            Self::RectOutOfBounds {
                rect,
                width,
                height,
            } => write!(
                f,
                "rectangle {}×{} at ({}, {}) is empty or exceeds the {width}×{height} image; \
                 it must be non-empty and lie inside the image",
                rect.width, rect.height, rect.x, rect.y
            ),
            Self::SplitOutOfRange { x, width } => write!(
                f,
                "split column {x} is outside 1..{width}; both pages must be at least one column wide"
            ),
            Self::SizeMismatch { expected, actual } => write!(
                f,
                "image is {}×{}; the measurement was taken on a {}×{} image and applies only to that size",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::ZeroSize { parameter } => {
                write!(f, "{parameter} is 0; it must be at least 1 pixel")
            }
        }
    }
}

impl std::error::Error for CameraError {}
