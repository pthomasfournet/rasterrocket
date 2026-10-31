//! Page diagnostics: what a render drew, skipped or lost, and why a render
//! stopped early.

use pdf::ContentFailure;

use super::{MAX_FORM_DEPTH, PAGE_RENDER_DEADLINE_SECS, PageRenderer};
use crate::resources::ImageFilter;

/// Lightweight page metadata collected at zero extra cost during rendering.
///
/// Returned alongside the finished bitmap from [`PageRenderer::finish`].
/// Callers can use this to route pages to different OCR configurations without
/// a separate post-hoc analysis pass.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct PageDiagnostics {
    /// `true` when at least one image `XObject` or inline image was blitted.
    pub has_images: bool,
    /// `true` when at least one text-showing operator (`Tj`, `TJ`, `'`, `"`)
    /// was executed with a non-empty glyph sequence.  `false` on scan-only pages.
    pub has_vector_text: bool,
    /// The image filter seen most often on this page.
    ///
    /// `None` when no images were decoded (e.g. a pure-vector page).  When
    /// multiple filters are equally frequent the last one in [`ImageFilter`]
    /// discriminant order wins (Rust's `max_by_key` is last-wins on ties) —
    /// stable across identical pages.
    pub dominant_filter: Option<ImageFilter>,
    /// Estimated source pixels-per-inch of the dominant image, if any.
    ///
    /// Computed as `(image_width_px / page_width_pts) × 72`.  Useful as a hint
    /// for DPI auto-selection: rendering at the source PPI avoids upsampling a
    /// 67-PPI scan at 300 DPI (4.5× upscale with no information gain).
    ///
    /// `None` when no images were blitted or page geometry is unavailable.
    pub source_ppi_hint: Option<f32>,
    /// Images that were present but could not be drawn.  Each one was skipped
    /// and the page rendered without it, so the region it covers shows
    /// whatever was painted beneath.  Empty on a complete render.
    pub image_failures: Vec<ImageFailure>,
    /// Page content streams that could not be decoded, or only in part.  A
    /// failed one was skipped and the page rendered from the others; a
    /// partial one (`partial`) was executed up to where its data was lost.
    /// Either way, whatever the lost data would have drawn is missing.
    /// Empty on a complete render.
    pub content_failures: Vec<ContentFailure>,
    /// Font resource names (without the leading `/`) that `Tf` selected but
    /// the `/Font` resources in effect do not define.  Their text is drawn
    /// in Helvetica.  Empty when every font resolved.
    pub missing_fonts: Vec<String>,
    /// The document's interactive form type.  [`FormType::XfaFull`] marks
    /// a dynamic XFA form, whose rendered pages are the placeholder content
    /// stored in the file rather than the form.
    ///
    /// [`FormType::XfaFull`]: crate::FormType::XfaFull
    pub form_type: crate::FormType,
}

/// An image the renderer skipped because it could not be decoded or drawn,
/// or drew only partly because its stream lacked samples.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageFailure {
    /// `XObject` resource name without the leading `/` (e.g. `Im0`); `None`
    /// for an inline image.
    pub name: Option<String>,
    /// Codec or blit error, prefixed by the failing filter where known
    /// (e.g. `FlateDecode: invalid distance`); for a partial image, what
    /// the stream lacked.
    pub reason: String,
    /// The image was drawn from the samples its stream holds, with the
    /// missing ones padded; `false` when it was skipped altogether.
    pub partial: bool,
}

impl std::fmt::Display for ImageFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = if self.partial { " (partial)" } else { "" };
        match &self.name {
            Some(name) => write!(f, "image /{name}{what}: {}", self.reason),
            None => write!(f, "inline image{what}: {}", self.reason),
        }
    }
}

impl PageDiagnostics {
    /// Suggest a render DPI based on the source image resolution embedded in the PDF.
    ///
    /// Returns the source PPI rounded up to the nearest standard DPI step
    /// (72 / 96 / 150 / 200 / 300 / 400 / 600), clamped to `[min_dpi, max_dpi]`.
    ///
    /// Returns `None` when [`source_ppi_hint`](Self::source_ppi_hint) is absent
    /// (no images on the page), in which case the caller should use their default DPI.
    ///
    /// ## Rationale
    ///
    /// Many scanned PDFs store images at 62–67 PPI.  Rendering at 300 DPI upsamples
    /// 4.5× with no additional information.  Passing the suggested DPI to
    /// the renderer (`rasterrocket::RasterOptions::dpi`) renders at the minimum resolution that avoids
    /// upsampling, bounded by the caller's quality floor (`min_dpi`) and ceiling
    /// (`max_dpi`).
    ///
    /// Typical call: `diag.suggested_dpi(150.0, 300.0)` — never render below
    /// 150 DPI (Tesseract minimum for acceptable accuracy), never above 300 DPI.
    #[must_use]
    pub fn suggested_dpi(&self, min_dpi: f32, max_dpi: f32) -> Option<f32> {
        // Standard DPI steps in ascending order.  Any PPI above the last step
        // falls back to the last step (600), which is then clamped by max_dpi.
        const STEPS: [f32; 7] = [72.0, 96.0, 150.0, 200.0, 300.0, 400.0, 600.0];
        const STEPS_MAX: f32 = STEPS[STEPS.len() - 1]; // 600.0 — infallible const index
        let ppi = self.source_ppi_hint?;
        let stepped = STEPS
            .iter()
            .copied()
            .find(|&s| s >= ppi)
            .unwrap_or(STEPS_MAX);
        Some(stepped.clamp(min_dpi, max_dpi))
    }
}

/// Why a page render stopped before executing all of its operators.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderAbort {
    /// The page executed more operators than the per-page cap allows.
    OperatorBudget {
        /// Operators executed when the cap tripped.
        ops: u64,
    },
    /// The page ran past the per-page wall-clock deadline.
    Deadline {
        /// Operators executed when the deadline was detected.
        ops: u64,
    },
    /// Form `XObjects` nested deeper than the renderer's limit.
    FormDepth,
    /// The caller's cancellation flag was raised.
    Cancelled {
        /// Operators executed before the flag was observed.
        ops: u64,
    },
}

impl std::fmt::Display for RenderAbort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OperatorBudget { ops } => {
                write!(
                    f,
                    "page render exceeded operator budget ({ops} ops); aborting"
                )
            }
            Self::Deadline { ops } => write!(
                f,
                "page render exceeded {PAGE_RENDER_DEADLINE_SECS}s deadline after {ops} ops; aborting"
            ),
            Self::FormDepth => write!(
                f,
                "Form XObject nesting depth {MAX_FORM_DEPTH} exceeded; aborting page render"
            ),
            Self::Cancelled { ops } => write!(f, "page render cancelled after {ops} ops"),
        }
    }
}

impl PageRenderer<'_> {
    /// Images skipped so far on this page; also returned in
    /// [`PageDiagnostics::image_failures`] by [`finish`](Self::finish).
    #[must_use]
    pub fn image_failures(&self) -> &[ImageFailure] {
        &self.diag.image_failures
    }

    /// Content streams skipped on this page; also returned in
    /// [`PageDiagnostics::content_failures`] by [`finish`](Self::finish).
    #[must_use]
    pub fn content_failures(&self) -> &[ContentFailure] {
        &self.diag.content_failures
    }

    /// Record the page content streams that were skipped before execution.
    pub fn record_content_failures(&mut self, failures: Vec<ContentFailure>) {
        self.diag.content_failures.extend(failures);
    }

    /// Record an image that could not be drawn.  The image is skipped and the
    /// page render continues.
    pub(super) fn record_image_failure(&mut self, name: Option<&[u8]>, reason: String) {
        let failure = ImageFailure {
            name: name.map(|n| String::from_utf8_lossy(n).into_owned()),
            reason,
            partial: false,
        };
        log::debug!("rasterrocket-interp: skipped {failure}");
        self.diag.image_failures.push(failure);
    }

    /// Record an image drawn from a stream that lacked samples (see
    /// [`DecodedImage::partial`](crate::resources::DecodedImage::partial)).
    pub(super) fn record_partial_image(&mut self, name: Option<&[u8]>, reason: String) {
        let failure = ImageFailure {
            name: name.map(|n| String::from_utf8_lossy(n).into_owned()),
            reason,
            partial: true,
        };
        log::debug!("rasterrocket-interp: {failure}");
        self.diag.image_failures.push(failure);
    }

    /// Record a font resource `name` that `Tf` selected but the resources in
    /// effect do not define, once per page.
    pub(super) fn record_missing_font(&mut self, name: &[u8]) {
        let name = String::from_utf8_lossy(name);
        if self.diag.missing_fonts.iter().any(|n| *n == name) {
            return;
        }
        log::warn!("rasterrocket-interp: font resource /{name} is not defined — drawing Helvetica");
        self.diag.missing_fonts.push(name.into_owned());
    }
}
