//! Page diagnostics: what a render drew, skipped or lost, and why a render
//! stopped early.

use pdf::ContentFailure;

use super::{MAX_FORM_DEPTH, PAGE_RENDER_DEADLINE_SECS, PageRenderer};
use crate::resources::ImageFilter;

/// The source-resolution estimate of an image `image_width_px` samples
/// across on a page `page_width_pts` points wide before its rotation
/// (see [`crate::unrotated_width`]), as
/// [`PageDiagnostics::source_ppi_hint`] defines it:
/// `image_width_px / page_width_pts × 72`, the resolution of the image
/// were it to span the page's width.  `None` unless finite and positive.
pub(crate) fn source_ppi(image_width_px: u32, page_width_pts: f64) -> Option<f32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a resolution estimate needs no more than f32 precision; an overflow to \
                  infinity is rejected below"
    )]
    let ppi = (f64::from(image_width_px) / page_width_pts * 72.0) as f32;
    (ppi.is_finite() && ppi > 0.0).then_some(ppi)
}

/// The transparency-group colour spaces a page composited in an
/// approximation.
///
/// A CMYK ICC space over an RGB canvas composites as
/// `DeviceCMYK` does, through the process-ink model; a `CalRGB` or an RGB
/// ICC space that is not sRGB composites in the RGB working space of its
/// canvas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GroupSpaceApproximations {
    /// A CMYK ICC group space composited through the process-ink model.
    pub cmyk_profile: bool,
    /// A `CalRGB` or non-sRGB ICC RGB group space whose conversions do not
    /// build, composited in the canvas's RGB working space.
    pub rgb_profile: bool,
}

impl GroupSpaceApproximations {
    /// Record that a group of `space` was composited in an approximation.
    pub(crate) const fn record(&mut self, space: crate::resources::GroupSpace) {
        match space {
            crate::resources::GroupSpace::Cmyk { .. } => self.cmyk_profile = true,
            crate::resources::GroupSpace::Rgb(_) => self.rgb_profile = true,
            _ => {}
        }
    }

    /// The approximations of `self` and of `other`.
    #[must_use]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self {
            cmyk_profile: self.cmyk_profile || other.cmyk_profile,
            rgb_profile: self.rgb_profile || other.rgb_profile,
        }
    }

    /// Whether any group space was approximated.
    #[must_use]
    pub const fn any(self) -> bool {
        self.cmyk_profile || self.rgb_profile
    }
}

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
    /// Computed as `(image_width_px / page_width_pts) × 72` for the widest
    /// image on the page, its width in source samples whatever resolution
    /// it was decoded at, and the page box's width in points before the
    /// page's `/Rotate` and any session rotation: the extent along which an
    /// upright image's samples run.  Prescan and render measure against the
    /// same width.  Useful as a hint for DPI auto-selection: rendering at
    /// the source PPI avoids upsampling a 67-PPI scan at 300 DPI (4.5×
    /// upscale with no information gain).
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
    /// Form `XObjects` not drawn because they lie more than
    /// [`MAX_FORM_DEPTH`] forms deep, each once, by object
    /// identifier.  The rest of the page rendered without them.  Empty when
    /// every form was drawn.
    pub forms_too_deep: Vec<pdf::ObjectId>,
    /// Font resource names (without the leading `/`) that `Tf` selected but
    /// the `/Font` resources in effect do not define.  Their text is drawn
    /// in Helvetica.  Empty when every font resolved.
    pub missing_fonts: Vec<String>,
    /// Spot colorants painted on a page simulating overprint that found no
    /// spot channel, the four channels being held by other spots.  Each
    /// painted its alternate colour in process ink, so it overprints as
    /// that process colour rather than as a separate ink.  Empty when
    /// every spot had a channel, and on a page without overprint.
    pub unslotted_spots: Vec<String>,
    /// The transparency-group colour spaces the page composited in an
    /// approximation of their own (see [`GroupSpaceApproximations`]).
    pub approximated_group_spaces: GroupSpaceApproximations,
    /// The document's interactive form type.  [`FormType::XfaFull`] marks
    /// a dynamic XFA form, whose rendered pages are the placeholder content
    /// stored in the file rather than the form.
    ///
    /// [`FormType::XfaFull`]: crate::FormType::XfaFull
    pub form_type: crate::FormType,
    /// Features on this page that the renderer ignores: JavaScript in the
    /// page's or an annotation's actions, and 3D and multimedia
    /// annotations, in the order of
    /// [`page_unsupported_features`](crate::page_unsupported_features).
    /// The document-level ones are listed once per document, by
    /// [`document_unsupported_features`](crate::document_unsupported_features).
    /// Empty when the page uses none.
    pub unsupported_features: Vec<crate::UnsupportedFeature>,
    /// Images drawn through the blit kernel of the image cache's backend
    /// from device-resident samples.  Zero when the session holds no
    /// image cache, and for images the CPU draws: masked images, draws
    /// below the image's resolution, and draws beyond the blit tables.
    pub device_image_blits: u32,
    /// Images in an ICC-based CMYK space converted to RGB on a GPU
    /// backend.  Zero when the session holds no backend, and for images
    /// below the device's size threshold, which convert on the CPU with
    /// the same bytes.
    pub device_icc_conversions: u32,
    /// ICC CMYK conversions a GPU backend took and failed, each then
    /// performed on the CPU with the same bytes.  Non-zero means the
    /// device path is broken on this host; the first failure of the
    /// process is also logged at `warn`.
    pub device_icc_failures: u32,
    /// Whether the page was rendered whole on a GPU from its recorded
    /// paint commands, and why not when it was offered and declined.
    pub device_page: DevicePage,
    /// The rendering intents in force at the page's marking operators:
    /// the graphics state's (`ri`, `ExtGState` `RI`) for paths, text and
    /// shadings, an image's own `Intent` where it has one.  Empty on a
    /// page that marks nothing.
    pub content_intents: crate::IntentSet,
    /// The rendering intent the page was converted to its output profile
    /// under; `None` when it was not converted.
    pub output_intent: Option<crate::RenderingIntent>,
    /// The glyphs the page's text-showing operators showed, by text
    /// rendering mode and by why a glyph could not mark the page.  Zero
    /// from [`prescan_page`](crate::prescan_page), which executes nothing.
    pub glyphs: GlyphCounts,
}

/// The glyphs a page's text-showing operators (`Tj`, `TJ`, `'`, `"`)
/// showed: one per character code shown, each counted once.
///
/// A glyph in a painting mode (`Tr` 0, 1, 2, 4, 5, 6) is counted under its
/// mode only if it can mark the page; otherwise under the first of
/// [`zero_size`](Self::zero_size), [`unrendered`](Self::unrendered),
/// [`off_page`](Self::off_page) and [`transparent`](Self::transparent)
/// that applies.  A glyph in mode 3 or 7, which paints nothing, is counted
/// under its mode.  The counts are those of the page's content: the same
/// for a page rendered whole, in bands or as a region of it.
///
/// What can mark the page is judged from the glyph's position, size and
/// paint alone.  The clip, a soft mask, the blend mode and a colour equal
/// to the backdrop's are not considered, nor whether the glyph's outline
/// is empty (a space counts as drawn).  Glyphs inside a tiling pattern's
/// cell, a soft mask's group and a Type 3 glyph's description are not
/// counted; a Type 3 glyph is counted as one glyph of its string.  Text
/// in a substitute or fallback face is counted as any other.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GlyphCounts {
    /// Glyphs filled (`0 Tr`).
    pub fill: u32,
    /// Glyphs stroked (`1 Tr`).
    pub stroke: u32,
    /// Glyphs filled and stroked (`2 Tr`).
    pub fill_stroke: u32,
    /// Glyphs neither painted nor added to the clip (`3 Tr`): the mode of
    /// the text layer a recogniser lays over a scanned page.
    pub invisible: u32,
    /// Glyphs painted and added to the clip (`4`, `5` and `6 Tr`).
    pub clip_painted: u32,
    /// Glyphs only added to the clip (`7 Tr`).  A Type 3 glyph, whose
    /// description paints in every mode but 3, is counted under
    /// [`clip_painted`](Self::clip_painted) instead.
    pub clip_only: u32,
    /// Glyphs in a painting mode that draw nothing at the render's scale:
    /// a singular text rendering matrix (font size 0, `Tz 0`, a degenerate
    /// text matrix or CTM), or, for a font other than Type 3, an em below
    /// one device pixel.  The second case depends on the resolution.
    pub zero_size: u32,
    /// Glyphs in a painting mode that render nothing for want of a glyph:
    /// a code the face has no glyph for, a font program that cannot be
    /// loaded, a Type 3 code without a `CharProc`, and every code shown
    /// before any font is selected.
    pub unrendered: u32,
    /// Glyphs in a painting mode lying wholly outside the page box: the
    /// glyph's em square, widened by one em on each side, does not meet
    /// the page.
    pub off_page: u32,
    /// Glyphs in a painting mode whose paint marks nothing: each of the
    /// mode's paints (fill, stroke) has opacity 0 or a colour space whose
    /// colorants are all `None`.  A Type 3 glyph is transparent when the
    /// fill opacity is 0.
    pub transparent: u32,
}

impl GlyphCounts {
    /// Glyphs that can mark the page: those counted under a painting mode.
    #[must_use]
    pub const fn visible(&self) -> u32 {
        self.fill
            .saturating_add(self.stroke)
            .saturating_add(self.fill_stroke)
            .saturating_add(self.clip_painted)
    }

    /// Add the counts of `other` to these, saturating.
    pub(crate) const fn add(&mut self, other: &Self) {
        self.fill = self.fill.saturating_add(other.fill);
        self.stroke = self.stroke.saturating_add(other.stroke);
        self.fill_stroke = self.fill_stroke.saturating_add(other.fill_stroke);
        self.invisible = self.invisible.saturating_add(other.invisible);
        self.clip_painted = self.clip_painted.saturating_add(other.clip_painted);
        self.clip_only = self.clip_only.saturating_add(other.clip_only);
        self.zero_size = self.zero_size.saturating_add(other.zero_size);
        self.unrendered = self.unrendered.saturating_add(other.unrendered);
        self.off_page = self.off_page.saturating_add(other.off_page);
        self.transparent = self.transparent.saturating_add(other.transparent);
    }
}

/// How a page used the whole-page GPU raster.
///
/// A page offered to it is first interpreted into paint commands; one
/// that uses anything the raster does not render is then rendered on the
/// CPU, whole, as is a page the device fails on.  Every outcome yields the
/// same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DevicePage {
    /// The page was not offered to a device: the session renders every
    /// page on the CPU, or the device did not open, had not opened after
    /// the page waited for it, or was retired after a failure.  Also the outcome of a recording
    /// stopped by the page's work budget or by cancellation, which the
    /// caller reports as an error.
    #[default]
    NotOffered,
    /// The device rendered the page from `commands` paint commands.
    Rendered {
        /// Paint commands the page recorded.
        commands: u32,
    },
    /// The page paints with something the GPU raster does not render,
    /// named here; it was rendered on the CPU.
    Ineligible(&'static str),
    /// The device failed on the page, or the page exceeds the device's
    /// limits; it was rendered on the CPU.
    Failed,
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
    /// Whether the page was drawn with something left out or substituted:
    /// a skipped image or content stream, a form past the depth limit, a
    /// font that resolved to nothing, or an unsupported feature.
    #[must_use]
    pub const fn has_omissions(&self) -> bool {
        !(self.image_failures.is_empty()
            && self.content_failures.is_empty()
            && self.forms_too_deep.is_empty()
            && self.missing_fonts.is_empty()
            && self.unsupported_features.is_empty())
    }

    /// Whether the page shows a glyph that can mark it (see
    /// [`GlyphCounts::visible`]): text drawn on the page rather than only
    /// an invisible (`3 Tr`) or clip-only layer.
    #[must_use]
    pub const fn has_visible_text(&self) -> bool {
        self.glyphs.visible() > 0
    }

    /// Add the device ICC conversions `counts` recorded to the page's.
    #[cfg(feature = "gpu-icc")]
    pub(crate) const fn add_device_icc(
        &mut self,
        counts: &crate::resources::image::DeviceIccCounts,
    ) {
        self.device_icc_conversions = self
            .device_icc_conversions
            .saturating_add(counts.converted());
        self.device_icc_failures = self.device_icc_failures.saturating_add(counts.failed());
    }

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
    /// The caller's cancellation flag was raised.
    Cancelled {
        /// Operators executed before the flag was observed.
        ops: u64,
    },
    /// The allocator refused a bitmap the page needs: a transparency
    /// group, soft mask, knockout coverage canvas, tiling-pattern cell or
    /// screen, Type 3 glyph canvas, or a snapshot of the canvas.
    AllocationFailed {
        /// Width of the refused bitmap in pixels.
        width: u32,
        /// Height of the refused bitmap in pixels.
        height: u32,
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
            Self::Cancelled { ops } => write!(f, "page render cancelled after {ops} ops"),
            Self::AllocationFailed { width, height } => write!(
                f,
                "cannot allocate a {width}×{height} bitmap; aborting page render"
            ),
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

    /// Form `XObjects` skipped on this page for their nesting depth; also
    /// returned in [`PageDiagnostics::forms_too_deep`] by
    /// [`finish`](Self::finish).
    #[must_use]
    pub fn forms_too_deep(&self) -> &[pdf::ObjectId] {
        &self.diag.forms_too_deep
    }

    /// Record the form `id`, entered at [`MAX_FORM_DEPTH`] forms deep, as
    /// skipped, once per page.
    pub(super) fn record_form_too_deep(&mut self, id: pdf::ObjectId) {
        if self.diag.forms_too_deep.contains(&id) {
            return;
        }
        log::warn!(
            "rasterrocket-interp: Form XObject ({}, {}) lies more than {MAX_FORM_DEPTH} forms \
             deep — not drawn",
            id.0,
            id.1
        );
        self.diag.forms_too_deep.push(id);
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

    /// Record an image that could not be drawn, once per page.  The image
    /// is skipped and the page render continues.
    pub(super) fn record_image_failure(&mut self, name: Option<&[u8]>, reason: String) {
        self.push_image_failure(name, reason, false);
    }

    /// Record an image drawn from a stream that lacked samples (see
    /// [`DecodedImage::partial`](crate::resources::DecodedImage::partial)),
    /// once per page.
    pub(super) fn record_partial_image(&mut self, name: Option<&[u8]>, reason: String) {
        self.push_image_failure(name, reason, true);
    }

    /// Add an [`ImageFailure`] unless the page already records an equal one:
    /// an image drawn repeatedly fails the same way each time.
    fn push_image_failure(&mut self, name: Option<&[u8]>, reason: String, partial: bool) {
        let failure = ImageFailure {
            name: name.map(|n| String::from_utf8_lossy(n).into_owned()),
            reason,
            partial,
        };
        if self.diag.image_failures.contains(&failure) {
            return;
        }
        let skipped = if partial { "" } else { "skipped " };
        log::debug!("rasterrocket-interp: {skipped}{failure}");
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

#[cfg(test)]
mod omission_tests {
    use super::PageDiagnostics;

    #[test]
    fn has_omissions_is_false_by_default_and_true_when_a_field_is_set() {
        assert!(!PageDiagnostics::default().has_omissions());
        let d = PageDiagnostics {
            missing_fonts: vec!["F1".to_owned()],
            ..PageDiagnostics::default()
        };
        assert!(d.has_omissions());
    }
}
