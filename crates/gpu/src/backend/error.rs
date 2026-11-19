//! The backend-agnostic error type and its result alias.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

/// A backend-agnostic error type.
///
/// Variants exist for conditions that tests assert on (`matches!`) or
/// that benefit from typed log clarity. The long tail of errors goes
/// through [`BackendError::Other`] — Vulkan FFI failures, allocator
/// errors, init failures, numeric overflow, and free-form messages
/// constructed via [`BackendError::msg`].
///
/// `Display` is hand-rolled per variant; `Error::source` returns
/// `Some(&inner)` only for `Other` (typed variants are themselves the
/// source).
#[derive(Debug)]
pub enum BackendError {
    /// `alloc_*` was called with `size = 0`. `what` names the allocator
    /// (`"alloc_device"`, `"alloc_host_pinned"`, `"alloc_device_zeroed"`).
    ZeroSizeAlloc {
        /// Name of the rejecting allocator entry point, for diagnostic context.
        what: &'static str,
    },
    /// `vkCmdFillBuffer` requires a 4-byte-aligned size; `alloc_device_zeroed`
    /// (and other fill paths) surface this loudly rather than silently
    /// downgrading and leaving 1–3 trailing bytes non-zero.
    UnalignedFill {
        /// The offending size in bytes.
        size: u64,
        /// The backend's required alignment (always `4` for `vkCmdFillBuffer`).
        required_alignment: u64,
    },
    /// Recorder cap on descriptor sets per page was reached. `max` is the
    /// production limit (`MAX_DESC_SETS_PER_PAGE`).
    DescriptorPoolExhausted {
        /// Number of descriptor sets already allocated this page.
        allocated: u32,
        /// The backend's per-page cap.
        max: u32,
    },
    /// A `*Params::validate` invariant failed. `kind` names which
    /// invariant family (`"BlitInvariantViolation"`, …); `detail` is
    /// the human-readable "why", with the sizes involved when the
    /// invariant compares sizes.
    InvariantViolation {
        /// Stable name of the invariant family for `matches!` ergonomics.
        kind: &'static str,
        /// Human-readable description of the specific violation.
        detail: Cow<'static, str>,
    },
    /// A buffer the work needs exceeds a limit of the device: the work is
    /// too large for it, and the device itself is sound.
    DeviceLimit {
        /// The buffer, for diagnostic context.
        what: &'static str,
        /// Bytes the work needs.
        need: u64,
        /// The device limit's name.
        limit: &'static str,
        /// The device limit, in bytes.
        max: u64,
    },
    /// The device or the host had no memory left for a buffer the work
    /// needs: the work does not fit beside what is allocated now, and the
    /// device itself is sound.
    OutOfMemory {
        /// The buffer, for diagnostic context.
        what: &'static str,
        /// Bytes the buffer needs.
        bytes: u64,
    },
    /// A kernel reported, through its status words, an operation it could
    /// not complete exactly: the work's output is wrong and was discarded
    /// before reaching the caller.  The device's arithmetic is shown
    /// unreliable for that operation.
    KernelFault {
        /// The kernel, for diagnostic context.
        kernel: &'static str,
        /// The failed operation and its operands.
        detail: String,
    },
    /// An explicit device selector did not resolve to exactly one usable
    /// device, or no usable device exists.  Distinct from
    /// [`BackendError::Other`] so callers can treat a configuration mistake
    /// as fatal even where an unavailable backend would fall back silently.
    DeviceSelection {
        /// The kind of the reason.
        failure: SelectionFailure,
        /// The selector, the reason, and every enumerated device.
        message: String,
    },
    /// Catch-all for any other backend error: Vulkan FFI failures,
    /// allocator errors, init failures, numeric overflow, free-form
    /// `BackendError::msg(...)`. `source()` returns the inner error.
    Other(Box<dyn Error + Send + Sync + 'static>),
}

/// Why a device selection failed ([`BackendError::DeviceSelection`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionFailure {
    /// The selector or the selector list is malformed: empty, or naming
    /// one device twice.
    Invalid,
    /// No device matches the selector.
    NoMatch,
    /// More than one device matches the selector.
    Ambiguous,
    /// The device the selector names lacks something the backend needs.
    Ineligible,
    /// No enumerated device is a usable GPU.
    NoneUsable,
}

impl BackendError {
    /// Wrap an arbitrary error as a `BackendError::Other`.
    pub fn new<E: Error + Send + Sync + 'static>(e: E) -> Self {
        Self::Other(Box::new(e))
    }

    /// Build a `BackendError::Other` from a free-form message.
    ///
    /// Convenience for callsites that don't have an underlying [`Error`]
    /// to wrap and don't fit any typed variant. Routes through `Other`,
    /// so `source()` returns the carrier.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Other(Box::new(MsgError(message.into())))
    }

    /// An [`Self::InvariantViolation`] of the family `kind`.
    #[must_use]
    pub fn invariant(kind: &'static str, detail: impl Into<Cow<'static, str>>) -> Self {
        Self::InvariantViolation {
            kind,
            detail: detail.into(),
        }
    }
}

/// Internal carrier used by [`BackendError::msg`] so the inner type
/// satisfies `Error + Send + Sync + 'static`.
#[derive(Debug)]
struct MsgError(String);

impl fmt::Display for MsgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for MsgError {}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSizeAlloc { what } => {
                write!(f, "{what} called with size = 0; backends require size > 0")
            }
            Self::UnalignedFill {
                size,
                required_alignment,
            } => write!(
                f,
                "fill size {size} is not a multiple of {required_alignment}"
            ),
            Self::DescriptorPoolExhausted { allocated, max } => write!(
                f,
                "descriptor pool exhausted: {allocated} sets allocated this page (max {max})"
            ),
            Self::InvariantViolation { kind, detail } => write!(f, "{kind}: {detail}"),
            Self::DeviceLimit {
                what,
                need,
                limit,
                max,
            } => write!(
                f,
                "{what} need {need} bytes, above the device's {limit} of {max} bytes"
            ),
            Self::OutOfMemory { what, bytes } => {
                write!(f, "{what}: out of memory for {bytes} bytes")
            }
            Self::KernelFault { kernel, detail } => write!(f, "{kernel}: {detail}"),
            Self::DeviceSelection { message, .. } => f.write_str(message),
            Self::Other(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl Error for BackendError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Other(e) => Some(&**e),
            _ => None,
        }
    }
}

/// What a lost device means where the code runs, as the message of an
/// error that reports a lost device carries it.
///
/// Windows resets a GPU whose submission runs past its timeout detection
/// delay (TDR, 2 s by default), and every call on the device then fails
/// as lost.
#[cfg(any(feature = "vulkan", feature = "d3d12"))]
pub const DEVICE_LOST_CAUSE: &str = if cfg!(windows) {
    "the device was lost; on Windows the likely cause is a GPU timeout \
     (TDR: a submission that runs past about 2 s resets the device)"
} else {
    "the device was lost: a driver reset or a GPU fault"
};

/// Convenience alias for `Result<T, BackendError>`.
pub type Result<T> = std::result::Result<T, BackendError>;
