//! The error type of every rendering entry point.

use crate::page::MAX_PX_SIDE;

/// Errors returned by [`crate::raster_pdf`].
#[derive(Debug)]
#[non_exhaustive]
pub enum RasterError {
    /// [`RasterOptions`](crate::RasterOptions) fields violate documented constraints
    /// (e.g. `dpi ≤ 0`, `first_page > last_page`).
    InvalidOptions(String),
    /// The PDF could not be opened or parsed.
    Pdf(pdf_interp::InterpError),
    /// The requested page number is outside the document.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total number of pages in the document.
        total: u32,
    },
    /// The page has zero pixel width or height — malformed document.
    PageDegenerate {
        /// Width in pixels (0 when degenerate).
        width: u32,
        /// Height in pixels (0 when degenerate).
        height: u32,
    },
    /// The picture to be produced (the page, or the region drawn) holds more
    /// bytes than [`SessionConfig::max_page_bytes`](crate::SessionConfig::max_page_bytes).
    PageTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// `width × height × bytes per pixel` of the render's bitmap.
        bytes: u64,
        /// The budget it exceeds.
        limit: u64,
    },
    /// A side of the page exceeds [`MAX_PX_SIDE`], which no budget raises.
    PageExceedsCeiling {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// A bitmap the render of the drawn region needs could not be
    /// allocated: the region is within the size limits, but the allocator
    /// refused the buffers of its page bitmap, or of a bitmap drawn on it: a
    /// transparency group, soft mask, knockout coverage canvas, tiling-pattern
    /// cell or screen, Type 3 glyph canvas, or a snapshot of the canvas.
    PageAllocationFailed {
        /// Width of the refused bitmap in pixels.
        width: u32,
        /// Height of the refused bitmap in pixels.
        height: u32,
    },
    /// Deskew was refused: the page exceeds
    /// [`DESKEW_MAX_SIDE`](crate::deskew::DESKEW_MAX_SIDE).  A failed GPU
    /// rotation under [`BackendPolicy::ForceCuda`](crate::BackendPolicy::ForceCuda)
    /// is [`Self::BackendUnavailable`].
    Deskew(String),
    /// The row stride given for a caller-owned output buffer is shorter
    /// than one row of the page.
    OutputStrideTooSmall {
        /// The stride given, in bytes.
        stride: usize,
        /// The bytes of one row of the page: its width in pixels times
        /// the bytes per pixel of the output format.
        row_bytes: usize,
    },
    /// A caller-owned output buffer is shorter than the page at the stride
    /// given.
    OutputBufferTooSmall {
        /// Length of the buffer given, in bytes.
        len: usize,
        /// The bytes the page needs: `(height - 1) * stride + row_bytes`,
        /// or `usize::MAX` when that overflows.
        needed: usize,
    },
    /// A Page dictionary entry is structurally valid but outside permitted range
    /// (e.g. `UserUnit` outside `[0.1, 10.0]`).
    InvalidPageGeometry(String),
    /// A backend was required via [`BackendPolicy`](crate::BackendPolicy) but could not be initialised.
    BackendUnavailable(String),
    /// One or more images on the page could not be decoded or drawn, and the
    /// session is [strict](crate::SessionConfig::strict).  Carries one message
    /// per failed image.  Without `strict` these are reported in
    /// [`PageDiagnostics::image_failures`](crate::PageDiagnostics::image_failures)
    /// and the page still renders.
    ImageDecodeFailed(Vec<String>),
    /// One or more of the page's content streams could not be decoded, or
    /// decoded only in part, and the session is
    /// [strict](crate::SessionConfig::strict).  Carries one message per
    /// such stream.  Without `strict` these are reported in
    /// [`PageDiagnostics::content_failures`](crate::PageDiagnostics::content_failures)
    /// and the page renders from the data that decoded.
    ContentDecodeFailed(Vec<String>),
    /// The page render was aborted because it exceeded the per-page operator
    /// budget or wall-clock deadline.  The page is pathological (e.g.
    /// infinite-loop content stream or unbounded form-`XObject` recursion).
    /// The rendered bitmap is partial.
    PageBudgetExceeded(String),
    /// The render was stopped through its [`CancelToken`](crate::CancelToken).
    /// The rendered bitmap is partial and is discarded.
    Cancelled,
    /// The output profile admits no conversion for the page, or the
    /// conversion failed.
    OutputProfile(pdf_interp::OutputProfileError),
    /// A soft proof ([`SessionConfig::proof`](crate::SessionConfig::proof))
    /// found no output intent of a CMYK press for the document or page.
    OutputIntent(pdf_interp::OutputIntentError),
    /// The per-page render function panicked.
    ///
    /// Caught only under `panic = "unwind"` builds (e.g. test profiles).
    /// Under the crate's default `panic = "abort"` release profile the panic
    /// aborts the process before this variant can be constructed.
    RenderPanic {
        /// The 1-based page number that panicked.
        page: u32,
        /// The panic message extracted from the payload, if available.
        message: String,
    },
}

impl std::fmt::Display for RasterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidOptions(msg) => write!(f, "invalid raster options: {msg}"),
            // `e` is an `InterpError` that already formats its own labelled
            // message (e.g. "PDF error: …"); delegate so the prefix is not
            // applied twice.
            Self::Pdf(e) => write!(f, "{e}"),
            Self::PageOutOfRange { page, total } => {
                write!(
                    f,
                    "page {page} is out of range (document has {total} pages)"
                )
            }
            Self::PageDegenerate { width, height } => write!(
                f,
                "page has degenerate pixel dimensions {width}×{height} — \
                 PDF MediaBox may be malformed"
            ),
            Self::PageTooLarge {
                width,
                height,
                bytes,
                limit,
            } => write!(
                f,
                "a {width}×{height} page is {bytes} bytes, over the page budget of {limit} \
                 bytes; lower the resolution, raise the budget (`max_page_bytes`, \
                 rrocket --max-page-mib), or render the page in regions within it"
            ),
            Self::PageExceedsCeiling { width, height } => write!(
                f,
                "page pixel dimensions {width}×{height} exceed the per-side ceiling of \
                 {MAX_PX_SIDE} px; lower the resolution"
            ),
            Self::PageAllocationFailed { width, height } => write!(
                f,
                "cannot allocate a {width}×{height} bitmap for the page; lower the resolution, \
                 or render the page in smaller regions"
            ),
            Self::Deskew(msg) => write!(f, "deskew failed: {msg}"),
            Self::OutputStrideTooSmall { stride, row_bytes } => write!(
                f,
                "output stride {stride} is shorter than a page row of {row_bytes} bytes"
            ),
            Self::OutputBufferTooSmall { len, needed } => write!(
                f,
                "output buffer of {len} bytes is shorter than the {needed} bytes the page needs"
            ),
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
            Self::ImageDecodeFailed(v) => write!(f, "image decode failed: {}", v.join("; ")),
            Self::ContentDecodeFailed(v) => {
                write!(f, "content stream decode failed: {}", v.join("; "))
            }
            Self::PageBudgetExceeded(msg) => write!(f, "page render budget exceeded: {msg}"),
            Self::Cancelled => f.write_str("page render cancelled"),
            Self::OutputProfile(e) => write!(f, "output profile: {e}"),
            Self::OutputIntent(e) => write!(f, "soft proof: {e}"),
            Self::RenderPanic { page, message } => {
                write!(f, "page {page} render panicked: {message}")
            }
        }
    }
}

impl RasterError {
    /// The refusal of deskew for a render whose output is not gray.
    pub(crate) fn deskew_needs_gray() -> Self {
        Self::InvalidOptions(
            "deskew applies to gray pages only; render the page as gray \
             (render_page_gray, render_gray_into) or turn deskew off"
                .to_owned(),
        )
    }
}

impl std::error::Error for RasterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
            Self::OutputProfile(e) => Some(e),
            Self::OutputIntent(e) => Some(e),
            _ => None,
        }
    }
}

impl From<pdf_interp::InterpError> for RasterError {
    fn from(e: pdf_interp::InterpError) -> Self {
        match e {
            pdf_interp::InterpError::PageOutOfRange { page, total } => {
                Self::PageOutOfRange { page, total }
            }
            pdf_interp::InterpError::InvalidPageGeometry(msg) => Self::InvalidPageGeometry(msg),
            other => Self::Pdf(other),
        }
    }
}

/// A deskew error as the render's: an oversized page is
/// [`RasterError::Deskew`], and a failed forced GPU rotation is
/// [`RasterError::BackendUnavailable`], as a forced backend's other
/// failures are.
impl From<crate::deskew::DeskewError> for RasterError {
    fn from(e: crate::deskew::DeskewError) -> Self {
        match e {
            crate::deskew::DeskewError::TooLarge(msg) => Self::Deskew(msg),
            e @ crate::deskew::DeskewError::GpuRotation(_) => {
                Self::BackendUnavailable(e.to_string())
            }
        }
    }
}

/// Direct conversion from `pdf::PdfError` so call sites can write
/// `e.into()` instead of the two-level `RasterError::from(InterpError::from(e))`.
///
/// `PdfError::PageOutOfRange` carries a **0-based** index per the descender's
/// contract; the surrounding API surface is **1-based**.  Translate to the
/// 1-based [`RasterError::PageOutOfRange`] variant directly so the chained
/// conversion does not silently leak 0-based numbering through
/// `RasterError::Pdf(InterpError::Pdf(...))`.
impl From<pdf::PdfError> for RasterError {
    fn from(e: pdf::PdfError) -> Self {
        match e {
            pdf::PdfError::PageOutOfRange { page, total } => Self::PageOutOfRange {
                page: page.saturating_add(1),
                total,
            },
            other => Self::from(pdf_interp::InterpError::from(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{DEFAULT_MAX_PAGE_BYTES, MAX_PX_SIDE, validate_page_bytes};

    #[test]
    fn the_budget_message_names_the_budget_and_how_to_raise_it() {
        let msg = validate_page_bytes(30_000, 30_000, 3, DEFAULT_MAX_PAGE_BYTES)
            .expect_err("over the budget")
            .to_string();
        assert!(msg.contains("2700000000 bytes"), "{msg}");
        assert!(msg.contains("--max-page-mib"), "{msg}");
        assert!(msg.contains("render the page in regions"), "{msg}");
        let msg = validate_page_bytes(MAX_PX_SIDE + 1, 1, 3, u64::MAX)
            .expect_err("over the ceiling")
            .to_string();
        assert!(msg.contains("per-side ceiling of 16777216 px"), "{msg}");
    }

    /// An oversized deskew is a deskew error, a failed forced GPU rotation
    /// an unavailable backend.
    #[test]
    fn a_failed_gpu_rotation_is_an_unavailable_backend() {
        use crate::RasterError;
        use crate::deskew::DeskewError;
        let e = RasterError::from(DeskewError::TooLarge("too large".to_owned()));
        assert!(
            matches!(e, RasterError::Deskew(ref m) if m == "too large"),
            "{e}"
        );
        let e = RasterError::from(DeskewError::GpuRotation("no device".to_owned()));
        assert!(
            matches!(e, RasterError::BackendUnavailable(ref m) if m.contains("GPU rotation failed: no device")),
            "{e}"
        );
    }
}
