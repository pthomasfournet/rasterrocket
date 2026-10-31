//! The error type of every rendering entry point.

use crate::page::{MAX_PX_AREA, MAX_PX_DIMENSION};

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
    /// The computed pixel dimensions exceed the per-side safety limit.
    PageTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// The computed pixel area (width × height) exceeds the safety limit even
    /// though each side is within [`MAX_PX_DIMENSION`].
    PageAreaTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Total area in pixels (`width as u64 * height as u64`).
        area: u64,
    },
    /// Deskew rotation failed.
    Deskew(String),
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
            Self::PageTooLarge { width, height } => write!(
                f,
                "page pixel dimensions {width}×{height} exceed the per-side limit of \
                 {MAX_PX_DIMENSION} px; lower the resolution, or render the page in \
                 regions within the limit"
            ),
            Self::PageAreaTooLarge {
                width,
                height,
                area,
            } => write!(
                f,
                "page pixel area {width}×{height} = {area} px exceeds the limit of \
                 {MAX_PX_AREA} px; lower the resolution, or render the page in regions \
                 within the limit"
            ),
            Self::Deskew(msg) => write!(f, "deskew failed: {msg}"),
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "backend unavailable: {msg}"),
            Self::ImageDecodeFailed(v) => write!(f, "image decode failed: {}", v.join("; ")),
            Self::ContentDecodeFailed(v) => {
                write!(f, "content stream decode failed: {}", v.join("; "))
            }
            Self::PageBudgetExceeded(msg) => write!(f, "page render budget exceeded: {msg}"),
            Self::Cancelled => f.write_str("page render cancelled"),
            Self::RenderPanic { page, message } => {
                write!(f, "page {page} render panicked: {message}")
            }
        }
    }
}

impl std::error::Error for RasterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
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
    use crate::validate_dimensions;

    #[test]
    fn size_limit_messages_name_the_limit_and_region_rendering() {
        for err in [
            validate_dimensions(super::MAX_PX_DIMENSION + 1, 1),
            validate_dimensions(30_000, 30_000),
        ] {
            let msg = err.expect_err("over the limit").to_string();
            assert!(msg.contains("limit of"), "{msg}");
            assert!(msg.contains("render the page in regions"), "{msg}");
        }
    }
}
