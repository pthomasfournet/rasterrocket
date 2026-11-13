//! PDF content stream interpreter.
//!
//! # Architecture
//!
//! ```text
//! pdf::Document  →  page_content() + parse_page_content()  →  Vec<Operator>  →  renderer::PageRenderer
//!                →  prescan_page() →  PageDiagnostics  (no pixels decoded)
//! ```
//!
//! The [`content`] module handles tokenisation and operator decoding.
//! The [`renderer`] module rasterises operators to a pixel bitmap.
//! The `prescan` module classifies pages cheaply before rendering.

#[cfg(feature = "cache")]
pub mod cache;
pub mod content;
pub mod form_type;
pub(crate) mod optional_content;
pub mod page_group;
pub(crate) mod prescan;
pub mod renderer;
pub mod resources;
mod rotation;
#[cfg(test)]
mod test_helpers;
pub mod unsupported;

pub use form_type::{FormType, form_type};
pub use optional_content::{OptionalContent, RenderPurpose};
pub use page_group::{page_group, page_uses_overprint, transparent_page_group};
pub use prescan::prescan_page;
pub use rotation::{InvalidRotation, Rotation};
pub use unsupported::{
    MediaKind, ScriptLocation, UnsupportedFeature, document_unsupported_features,
    page_has_javascript, page_unsupported_features,
};

/// The effective device AA-fill dispatch threshold in pixels:
/// `gpu::GPU_AA_FILL_THRESHOLD` unless `PDF_RASTER_AA_FILL_THRESHOLD`
/// overrides it.  `usize::MAX` means no fill runs on a device.
#[cfg(feature = "gpu-aa")]
#[must_use]
pub fn aa_fill_threshold() -> usize {
    renderer::page::gpu_ops::aa_fill_threshold()
}

/// The effective parallel-Huffman JPEG dispatch threshold in pixels, unless
/// `PDF_RASTER_HUFFMAN_THRESHOLD` overrides it.  `u32::MAX` means no JPEG
/// decodes on a device.
#[cfg(feature = "gpu-jpeg-huffman")]
#[must_use]
pub fn huffman_threshold() -> u32 {
    resources::image::huffman_threshold()
}

/// A device kernel that a dispatch threshold can disable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceKernel {
    /// The antialiased path fill (`aa_fill_threshold`, feature `gpu-aa`).
    AaFill,
    /// The parallel-Huffman JPEG decoder (`huffman_threshold`, feature
    /// `gpu-jpeg-huffman`).
    HuffmanJpeg,
}

/// Whether `kernel` can dispatch at all.  A threshold equal to its type's
/// maximum disables its kernel, and a kernel not compiled into this build
/// is disabled.
#[cfg_attr(
    not(any(feature = "gpu-aa", feature = "gpu-jpeg-huffman")),
    expect(
        clippy::missing_const_for_fn,
        reason = "non-const only in builds that read a threshold"
    )
)]
#[must_use]
pub fn device_kernel_enabled(kernel: DeviceKernel) -> bool {
    match kernel {
        DeviceKernel::AaFill => {
            #[cfg(feature = "gpu-aa")]
            {
                aa_fill_threshold() != usize::MAX
            }
            #[cfg(not(feature = "gpu-aa"))]
            {
                false
            }
        }
        DeviceKernel::HuffmanJpeg => {
            #[cfg(feature = "gpu-jpeg-huffman")]
            {
                huffman_threshold() != u32::MAX
            }
            #[cfg(not(feature = "gpu-jpeg-huffman"))]
            {
                false
            }
        }
    }
}

/// Entry points for fuzz targets: the CCITT and JBIG2 codec wrappers and a
/// `CMap` parse-and-lookup driver.
///
/// Only compiled when `--cfg fuzzing` is set (cargo-fuzz sets this automatically).
/// Not part of the public API.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzz_helpers {
    pub use crate::resources::image::fuzz_entry::{decode_ccitt, decode_jbig2};

    /// Parse `data` as a `CMap` program, then split `data` itself into codes
    /// through the parsed code-space ranges and look each one up.
    pub fn parse_and_map_cmap(data: &[u8]) {
        let Some(cmap) = crate::resources::cmap::parse_cmap(data) else {
            return;
        };
        let mut rest = data;
        while !rest.is_empty() {
            let (code, len) = cmap.next_code(rest);
            let _ = std::hint::black_box(cmap.lookup(code));
            rest = &rest[usize::from(len.max(1)).min(rest.len())..];
        }
    }
}

use std::path::Path;

use pdf::{Document, OpenOptions};

/// Errors that can occur during PDF loading or content stream interpretation.
#[derive(Debug)]
#[non_exhaustive]
pub enum InterpError {
    /// The lazy PDF parser failed to load or parse the document.
    Pdf(pdf::PdfError),
    /// The requested page number is outside the document's page range.
    PageOutOfRange {
        /// The requested page (1-based).
        page: u32,
        /// Total number of pages in the document.
        total: u32,
    },
    /// A Page dictionary entry has a value that is structurally valid PDF but
    /// outside the range permitted by the spec or our safety limits.
    InvalidPageGeometry(String),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf(e) => write!(f, "PDF error: {e}"),
            Self::PageOutOfRange { page, total } => {
                write!(
                    f,
                    "page {page} is out of range (document has {total} pages)"
                )
            }
            Self::InvalidPageGeometry(msg) => write!(f, "invalid page geometry: {msg}"),
        }
    }
}

impl std::error::Error for InterpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pdf(e) => Some(e),
            Self::PageOutOfRange { .. } | Self::InvalidPageGeometry(_) => None,
        }
    }
}

impl From<pdf::PdfError> for InterpError {
    fn from(e: pdf::PdfError) -> Self {
        Self::Pdf(e)
    }
}

/// Open a PDF document from a file path.
///
/// The document still opens when it uses features the renderer ignores;
/// each one that changes what renders, or carries a script, is warned
/// about once (see [`document_unsupported_features`]):
/// - JavaScript at the catalog's `/OpenAction` or `/AA`, in the
///   `/Names /JavaScript` tree, or in `/AcroForm /AA`.  Scripts are never
///   executed, and an action counts only when it runs a script, never on
///   bare key presence;
/// - a dynamic XFA form, whose rendered pages are the placeholder content
///   stored in the file;
/// - a portable collection, whose embedded documents are not rendered.
///
/// Page-level and annotation features are not checked here: opening a
/// document reads no page objects.  [`page_unsupported_features`] lists a
/// page's.
///
/// An encrypted document is refused; see [`open_with`].
///
/// # Errors
/// - [`InterpError::Pdf`] if the file cannot be read, is not a valid PDF,
///   or is encrypted.
pub fn open(path: impl AsRef<Path>) -> Result<Document, InterpError> {
    open_with(path, &OpenOptions::default())
}

/// [`open`] with `options` deciding how an encrypted document is treated.
///
/// An encrypted document is refused unless
/// [`OpenOptions::decrypt_authorized`] is set, which must reflect a
/// deliberate caller decision (the CLI gates it behind an interactive
/// private-copy / liability waiver or an explicit operator bypass).  With it,
/// the empty user password and then [`OpenOptions::password`] unlock the
/// document, whose strings and streams are then decrypted in process as they
/// are read.  A password is an input to decryption, not an authorisation.
///
/// # Errors
/// - [`InterpError::Pdf`] if the file cannot be read or is not a valid PDF.
/// - [`InterpError::Pdf`] wrapping [`pdf::PdfError::EncryptedDocument`] if
///   the document is encrypted and decryption is not authorised, uses an
///   unsupported security handler, or needs a password other than the one
///   given.
pub fn open_with(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Document, InterpError> {
    let doc = Document::open_with(path.as_ref(), options)?;
    unsupported::warn_document_features(&doc);
    Ok(doc)
}

/// Open a PDF from in-memory bytes (no file path).
///
/// Applies the same post-open checks as [`open`].  For callers that already
/// hold the PDF bytes (e.g. extracted from an archive).  An encrypted
/// document is refused; see [`open_bytes_with`].
///
/// # Errors
/// - [`InterpError::Pdf`] if the bytes are not a parseable PDF, or are an
///   encrypted document.
pub fn open_bytes(bytes: Vec<u8>) -> Result<Document, InterpError> {
    open_bytes_with(bytes, &OpenOptions::default())
}

/// [`open_bytes`] with `options` deciding how an encrypted document is
/// treated, exactly as for [`open_with`].
///
/// # Errors
/// The errors of [`open_with`] other than file I/O.
pub fn open_bytes_with(bytes: Vec<u8>, options: &OpenOptions) -> Result<Document, InterpError> {
    let doc = Document::from_bytes_with(bytes, options)?;
    unsupported::warn_document_features(&doc);
    Ok(doc)
}

/// [`open_bytes_with`] for bytes the caller keeps shared, read in place
/// without a copy; see [`Document::from_shared`].
///
/// # Errors
/// The errors of [`open_with`] other than file I/O.
pub fn open_shared_with(
    bytes: impl AsRef<[u8]> + Send + Sync + 'static,
    options: &OpenOptions,
) -> Result<Document, InterpError> {
    let doc = Document::from_shared_with(bytes, options)?;
    unsupported::warn_document_features(&doc);
    Ok(doc)
}

/// Return the number of pages in `doc`.
///
/// Reads `/Pages /Count` directly (O(1) on well-formed PDFs); falls back to
/// the eager tree-walk count if the catalog entry is missing or malformed.
#[must_use]
pub fn page_count(doc: &Document) -> u32 {
    doc.page_count_fast()
}

/// Resolve a 1-based page number to its `pdf::ObjectId`.
///
/// Logarithmic in document page count: descends the `/Pages` tree using
/// each node's `/Count` to skip whole subtrees rather than walking every
/// leaf.
///
/// # Errors
/// [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document;
/// [`InterpError::Pdf`] if the page tree itself is malformed.
pub(crate) fn resolve_page_id(doc: &Document, page_num: u32) -> Result<pdf::ObjectId, InterpError> {
    let total = doc.page_count_fast();
    if page_num == 0 || page_num > total {
        return Err(InterpError::PageOutOfRange {
            page: page_num,
            total,
        });
    }
    // 1-based → 0-based.  Underflow guarded by `page_num != 0` above.
    doc.get_page(page_num - 1).map_err(InterpError::Pdf)
}

/// Page geometry: visible dimensions in PDF points, rotation, and user-space scale.
///
/// `width_pts` and `height_pts` are already adjusted for rotation — they describe the
/// output bitmap dimensions, not the raw `MediaBox`.  For example a landscape page with
/// `/Rotate 270` has `width_pts > height_pts` even though its `MediaBox` may be portrait.
///
/// Both dimensions are also already scaled by `user_unit` (PDF 1.6+ `UserUnit` key),
/// so they are physical points: a render at `dpi` has `dpi` pixels per physical inch.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PageGeometry {
    /// Width of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub width_pts: f64,
    /// Height of the rendered output in PDF points (after rotation and `UserUnit` scaling).
    pub height_pts: f64,
    /// Clockwise rotation in degrees; always one of 0, 90, 180, 270.
    pub rotate_cw: u16,
    /// `UserUnit` scale factor from the Page dictionary (PDF 1.6+, ISO 32000-2 §14.11.2).
    ///
    /// Specifies the size of one default user-space unit in 1/72-inch increments.
    /// `1.0` for the vast majority of documents.  Validated to `[0.1, 10.0]`.
    pub user_unit: f64,
    /// X-coordinate of the selected page box's lower-left corner in PDF user space (ISO 32000-2 §14.11.2).
    ///
    /// For the vast majority of documents this is `0.0`.  Non-zero for scanned PDFs
    /// whose `CropBox`/`MediaBox` does not start at the origin — e.g. a `CropBox`
    /// of `[36 36 576 756]` has `origin_x = 36`.  The initial CTM pre-translates by
    /// `(-origin_x, -origin_y)` so that the box's lower-left maps to device origin.
    pub origin_x: f64,
    /// Y-coordinate of the selected page box's lower-left corner in PDF user space.
    ///
    /// See `origin_x`.
    pub origin_y: f64,
    /// The page's `MediaBox` (own or inherited) as `[llx, lly, urx, ury]` in
    /// default user space, normalised so `llx < urx` and `lly < ury`; `None`
    /// when absent or unusable.
    pub media_box: Option<[f64; 4]>,
    /// The page's `CropBox` (own or inherited) as `[llx, lly, urx, ury]` in
    /// default user space, normalised like [`media_box`](Self::media_box) and
    /// not yet intersected with it; `None` when absent or unusable.
    pub crop_box: Option<[f64; 4]>,
}

impl PageGeometry {
    /// The pixel dimensions of the page rendered at `scale_x × scale_y`
    /// device pixels per point: `width_pts × scale_x` and
    /// `height_pts × scale_y`, each rounded to the nearest integer.
    ///
    /// A product beyond `u32::MAX` saturates to `u32::MAX`, and a NaN or
    /// non-positive product gives 0; callers reject both before allocating.
    #[must_use]
    pub fn size_px(&self, scale_x: f64, scale_y: f64) -> (u32, u32) {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "f64-to-u32 `as` saturates at both ends and maps NaN to 0, \
                      which is the documented contract"
        )]
        let px = |pts: f64, scale: f64| (pts * scale).round() as u32;
        (px(self.width_pts, scale_x), px(self.height_pts, scale_y))
    }
}

/// A page boundary box (ISO 32000-2 §14.11.2, Table 31).
///
/// Selects the region of the page that [`page_size_pts`] reports and the
/// renderer rasterises.  The default is [`Crop`](Self::Crop), the region a
/// conforming viewer displays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PageBox {
    /// The `MediaBox`: the full extent of the physical medium.  Inheritable.
    Media,
    /// The `CropBox`, intersected with the `MediaBox`; the `MediaBox` when
    /// absent.  Inheritable.
    #[default]
    Crop,
    /// The `BleedBox`, intersected with the `MediaBox`; the effective
    /// `CropBox` when absent.  Read from the page dictionary only.
    Bleed,
    /// The `TrimBox`, intersected with the `MediaBox`; the effective
    /// `CropBox` when absent.  Read from the page dictionary only.
    Trim,
    /// The `ArtBox`, intersected with the `MediaBox`; the effective
    /// `CropBox` when absent.  Read from the page dictionary only.
    Art,
}

impl PageBox {
    /// The page dictionary key of the box.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Media => "MediaBox",
            Self::Crop => "CropBox",
            Self::Bleed => "BleedBox",
            Self::Trim => "TrimBox",
            Self::Art => "ArtBox",
        }
    }
}

/// A page box as `(llx, lly, width, height)` in default user space, with a
/// finite lower-left corner and a positive finite extent.
type BoxRect = (f64, f64, f64, f64);

/// The number `o` holds, directly or through an indirect reference.
fn number(doc: &Document, o: &pdf::Object) -> Option<f64> {
    doc.resolve(o).ok().and_then(|o| resources::obj_to_f64(&o))
}

/// The rectangle `obj` describes, normalised so that `(llx, lly)` is the
/// minimum corner; `None` unless `obj` is an array of four finite numbers
/// (each direct or indirect) spanning a positive area.
fn parse_box(doc: &Document, obj: &pdf::Object) -> Option<BoxRect> {
    let number = |o| number(doc, o);
    let pdf::Object::Array(arr) = obj else {
        return None;
    };
    let [x0, y0, x1, y1] = arr.as_slice() else {
        return None;
    };
    let (x0, y0, x1, y1) = (number(x0)?, number(y0)?, number(x1)?, number(y1)?);
    // A coordinate beyond the f32 range is stored as an infinite Real; the
    // finiteness check keeps every derived width, height and origin finite,
    // as `build_initial_ctm` requires of its origin inputs.
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return None;
    }
    let w = (x1 - x0).abs();
    let h = (y1 - y0).abs();
    (w > 0.0 && h > 0.0).then(|| (x0.min(x1), y0.min(y1), w, h))
}

/// The intersection of `b` with the `MediaBox` `m` (ISO 32000-2 §14.11.2);
/// `None`, with a warning naming `page_box` and page object `page`, when
/// the intersection has no area.
fn clamp_to_media(b: BoxRect, m: BoxRect, page_box: PageBox, page: u32) -> Option<BoxRect> {
    let x0 = b.0.max(m.0);
    let y0 = b.1.max(m.1);
    let x1 = (b.0 + b.2).min(m.0 + m.2);
    let y1 = (b.1 + b.3).min(m.1 + m.3);
    let (w, h) = (x1 - x0, y1 - y0);
    if w > 0.0 && h > 0.0 {
        return Some((x0, y0, w, h));
    }
    log::warn!(
        "page object {page} {key} [{bx} {by} {bw}×{bh}] does not intersect \
         MediaBox [{mx} {my} {mw}×{mh}] (malformed PDF, §14.11.2)",
        key = page_box.key(),
        bx = b.0,
        by = b.1,
        bw = b.2,
        bh = b.3,
        mx = m.0,
        my = m.1,
        mw = m.2,
        mh = m.3,
    );
    None
}

/// The effective `CropBox` of page object `page`: the `CropBox` intersected
/// with the `MediaBox`; the `MediaBox` when the `CropBox` is absent or the
/// intersection is empty; the `CropBox` when the `MediaBox` is absent; and
/// `letter` at the origin when both are absent.  Every recovery from a
/// malformed page is logged.
fn effective_crop_box(
    media_box: Option<BoxRect>,
    crop_box: Option<BoxRect>,
    letter: (f64, f64),
    page: u32,
) -> BoxRect {
    match (media_box, crop_box) {
        (None, None) => {
            log::warn!(
                "page object {page} has no usable MediaBox or CropBox; assuming \
                 {}×{} (malformed PDF, §7.7.3.3)",
                letter.0,
                letter.1,
            );
            (0.0, 0.0, letter.0, letter.1)
        }
        (Some(m), None) => m,
        (None, Some(c)) => {
            log::warn!(
                "page object {page} has no usable MediaBox; falling back to CropBox \
                 for page geometry (malformed PDF, §7.7.3.3)",
            );
            c
        }
        (Some(m), Some(c)) => clamp_to_media(c, m, PageBox::Crop, page).unwrap_or_else(|| {
            log::warn!("page object {page}: using MediaBox for page geometry");
            m
        }),
    }
}

/// The rectangle of `page_box` on page object `page_id` (dictionary `dict`),
/// given the page's inherited `MediaBox` and `CropBox` and the `letter` size
/// assumed when both are absent; see [`page_size_pts`] for the rules.
fn select_box(
    doc: &Document,
    page_id: pdf::ObjectId,
    dict: &pdf::Dictionary,
    page_box: PageBox,
    (media_box, crop_box): (Option<BoxRect>, Option<BoxRect>),
    letter: (f64, f64),
) -> BoxRect {
    let crop_region = || effective_crop_box(media_box, crop_box, letter, page_id.0);
    match page_box {
        PageBox::Crop => crop_region(),
        PageBox::Media => media_box.unwrap_or_else(crop_region),
        PageBox::Bleed | PageBox::Trim | PageBox::Art => {
            // ISO 32000-2 Table 31: BleedBox, TrimBox and ArtBox are not
            // inheritable; only the page dictionary's own entry counts.
            let own = dict
                .get(page_box.key().as_bytes())
                .and_then(|o| doc.resolve(o).ok())
                .and_then(|o| parse_box(doc, &o));
            match (own, media_box) {
                (Some(b), Some(m)) => {
                    clamp_to_media(b, m, page_box, page_id.0).unwrap_or_else(|| {
                        log::warn!(
                            "page object {}: using the CropBox for page geometry",
                            page_id.0
                        );
                        crop_region()
                    })
                }
                (Some(b), None) => b,
                (None, _) => crop_region(),
            }
        }
    }
}

/// Return the geometry for page `page_num` (1-based), measured on `page_box`.
///
/// The selected box follows ISO 32000-2 §14.11.2: the `MediaBox` and
/// `CropBox` are inherited through the `/Parent` chain; the `BleedBox`,
/// `TrimBox` and `ArtBox` are read from the page dictionary only and default
/// to the effective `CropBox` when absent, unusable or disjoint from the
/// `MediaBox`; every box but the `MediaBox` is intersected with the
/// `MediaBox`.  An absent `MediaBox` is replaced by the `CropBox`.
///
/// Applies the `/Rotate` entry (multiples of 90° CW). Dimensions in the
/// returned struct are already swapped for 90°/270° rotations and scaled by
/// `UserUnit` so callers can use `width_pts × height_pts` directly as output
/// point dimensions.
///
/// When neither the `MediaBox` nor the `CropBox` can be read the page box is
/// US Letter (612 × 792 pt at the origin), still rotated by `/Rotate` and
/// scaled by `UserUnit`; a page object that is not a dictionary is unrotated
/// US Letter.
///
/// # Errors
///
/// - [`InterpError::PageOutOfRange`] if `page_num` is 0 or exceeds the document page count.
/// - [`InterpError::InvalidPageGeometry`] if `UserUnit` is present but outside `[0.1, 10.0]`.
pub fn page_size_pts(
    doc: &Document,
    page_num: u32,
    page_box: PageBox,
) -> Result<PageGeometry, InterpError> {
    let page_id = resolve_page_id(doc, page_num)?;
    page_size_pts_by_id(doc, page_id, page_box)
}

/// Like [`page_size_pts`] but takes a pre-resolved page object id.
///
/// Skips the page-tree descent.  Use when the caller already resolved the id —
/// the renderer resolves once at its entry point and reuses the id for both
/// `page_size_pts_by_id` and [`parse_page_content`].
///
/// # Errors
/// [`InterpError::InvalidPageGeometry`] if `UserUnit` is present but outside
/// `[0.1, 10.0]`.
#[expect(
    clippy::too_many_lines,
    reason = "box selection, /Rotate and UserUnit resolve one page dictionary in sequence; \
              each step is a short match whose result the next consumes"
)]
pub fn page_size_pts_by_id(
    doc: &Document,
    page_id: pdf::ObjectId,
    page_box: PageBox,
) -> Result<PageGeometry, InterpError> {
    let fallback = PageGeometry {
        width_pts: 612.0,
        height_pts: 792.0,
        rotate_cw: 0,
        user_unit: 1.0,
        origin_x: 0.0,
        origin_y: 0.0,
        media_box: None,
        crop_box: None,
    };

    let Ok(dict) = doc.get_dict(page_id) else {
        return Ok(fallback);
    };

    // ISO 32000-2 §7.7.3.4: MediaBox and CropBox are inheritable.
    let media_box = doc
        .get_inherited_page_attr(page_id, b"MediaBox")
        .and_then(|o| parse_box(doc, &o));
    let crop_box = doc
        .get_inherited_page_attr(page_id, b"CropBox")
        .and_then(|o| parse_box(doc, &o));
    let corners = |(x, y, w, h): BoxRect| [x, y, x + w, y + h];

    let (box_left, box_bottom, w_pts, h_pts) = select_box(
        doc,
        page_id,
        &dict,
        page_box,
        (media_box, crop_box),
        (fallback.width_pts, fallback.height_pts),
    );

    // ISO 32000-2 §7.7.3.4: /Rotate is inheritable.  Walk the /Parent chain
    // for the same reason as MediaBox/CropBox above.  A real number is read
    // truncated toward zero, as an integer-valued entry such as `90.0` is
    // meant.
    let rotate = doc.get_inherited_page_attr(page_id, b"Rotate");
    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64-to-i64 `as` truncates toward zero and saturates; the value is finite"
    )]
    let rotate_n = match &rotate {
        Some(pdf::Object::Integer(n)) => Some(*n),
        Some(pdf::Object::Real(r)) if r.is_finite() => Some(f64::from(*r) as i64),
        _ => None,
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rem_euclid(360) is always 0..=359, which fits u16"
    )]
    let rotate_cw: u16 = match (rotate_n, rotate) {
        (Some(n), _) => {
            // rem_euclid maps any sign into 0..=359 (so -90 → 270, 450 → 90).
            let normalized = n.rem_euclid(360) as u16;
            if !normalized.is_multiple_of(90) {
                // §7.7.3.4: /Rotate "shall be a multiple of 90".  A
                // non-multiple is malformed; snapping to the nearest lower
                // quadrant keeps the page upright-ish rather than rotating it
                // to a garbage angle, but the author's intent is unknowable —
                // say so instead of silently mis-rotating.
                log::warn!(
                    "/Rotate {n} on page object {} is not a multiple of 90 \
                     (malformed PDF, ISO 32000-2 §7.7.3.4); snapping to {}",
                    page_id.0,
                    normalized / 90 * 90,
                );
            }
            normalized / 90 * 90
        }
        // Absent /Rotate is the common, conforming case (no rotation) — silent.
        (None, None) => 0,
        // get_inherited_page_attr already resolves indirection, so any value
        // reaching here that is not a number is a malformed /Rotate type.
        (None, Some(other)) => {
            log::warn!(
                "/Rotate on page object {} is not a finite number (got {}); \
                 assuming 0 (malformed PDF, ISO 32000-2 §7.7.3.4)",
                page_id.0,
                other.enum_variant(),
            );
            0
        }
    };

    // UserUnit (PDF 1.6+, ISO 32000-2 §14.11.2): scales one default user-space unit
    // from 1/72 inch to UserUnit/72 inches.  Absent → 1.0.  Must be a finite positive
    // number.  We reject values outside [0.1, 10.0]: below 0.1 the page would be
    // unreasonably small at any practical DPI; above 10.0 the pixel dimensions would
    // exceed MAX_PX_DIMENSION at even modest DPI settings.  NaN/Inf in a Real object
    // are rejected explicitly because the range check (NaN comparisons are always false)
    // would otherwise silently pass them through.
    let user_unit: f64 = match dict.get(b"UserUnit") {
        None => 1.0, // absent — use default
        Some(obj) => {
            let Some(v) = number(doc, obj) else {
                return Err(InterpError::InvalidPageGeometry(format!(
                    "UserUnit on page object {} is not a number (got {})",
                    page_id.0,
                    obj.enum_variant()
                )));
            };
            if !v.is_finite() || !(0.1..=10.0).contains(&v) {
                return Err(InterpError::InvalidPageGeometry(format!(
                    "UserUnit {v} on page object {} is outside the valid range [0.1, 10.0]",
                    page_id.0,
                )));
            }
            v
        }
    };

    // Swap dimensions for 90°/270° so the caller gets output-bitmap dimensions directly.
    // Scale both dimensions by UserUnit so downstream only needs to apply dpi/72.
    let (width_pts, height_pts) = if rotate_cw == 90 || rotate_cw == 270 {
        (h_pts * user_unit, w_pts * user_unit)
    } else {
        (w_pts * user_unit, h_pts * user_unit)
    };

    // The origin is a user-space coordinate: the initial CTM scales user
    // space by dpi/72 × user_unit, so it pre-translates by (-origin_x,
    // -origin_y) in user-space units, not in points.
    let (origin_x, origin_y) = (box_left, box_bottom);

    Ok(PageGeometry {
        width_pts,
        height_pts,
        rotate_cw,
        user_unit,
        origin_x,
        origin_y,
        media_box: media_box.map(corners),
        crop_box: crop_box.map(corners),
    })
}

/// Parse `bytes`, the content of page `page_id` (from
/// [`pdf::Document::page_content`]), into its operators.
///
/// An inline image that names its colour space from the page's resources
/// has its data length computed through them.  The operators borrow
/// `bytes` wherever an operand needs no decoding.
#[must_use]
pub fn parse_page_content<'a>(
    doc: &Document,
    page_id: pdf::ObjectId,
    bytes: &'a [u8],
) -> Vec<content::Operator<'a>> {
    let resources = resources::PageResources::new(doc, page_id);
    let components = |name: &[u8]| resources.inline_image_components(name);
    content::parse_with(bytes, Some(&components))
}

// ── §7.7.3.4 inheritance + §14.11.2 clamp tests ──────────────────────────────

#[cfg(test)]
mod page_box_tests {
    use pdf::Document;

    use super::{PageBox, page_size_pts, page_size_pts_by_id};

    /// Build a minimal one-page PDF where the `MediaBox` is on the /Pages node
    /// (inherited), not on the page leaf.  The leaf has only /Type /Page and a
    /// /Parent back-reference.
    ///
    /// Object layout:
    ///   1 0 — Catalog (/Pages 2 0 R)
    ///   2 0 — Pages root (/`MediaBox` [0 0 500 700] /Kids [3 0 R] /Count 1)
    ///   3 0 — Page leaf (/Type /Page /Parent 2 0 R — no `MediaBox`)
    fn make_inherited_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 =
            "2 0 obj\n<</Type /Pages /MediaBox [0 0 500 700] /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("inherited-mediabox test PDF parse")
    }

    /// Build a one-page PDF where the page leaf has its own `MediaBox` [0 0 600 800]
    /// and a `CropBox` [0 0 700 900] that extends beyond the `MediaBox`.  After the
    /// §14.11.2 clamp the effective box must be [0 0 600 800] (the intersection).
    fn make_cropbox_exceeds_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        // CropBox [0 0 700 900] is larger than MediaBox [0 0 600 800] on all sides.
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R \
                    /MediaBox [0 0 600 800] /CropBox [0 0 700 900]>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("cropbox-exceeds-mediabox test PDF parse")
    }

    /// Build a one-page PDF where the page leaf has both `MediaBox` [0 0 612 792]
    /// and `CropBox` [72 72 540 720] (`CropBox` ⊆ `MediaBox` — the common born-digital
    /// case).  The result must match the `CropBox` exactly (pixel-neutral invariant).
    fn make_cropbox_inside_mediabox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R \
                    /MediaBox [0 0 612 792] /CropBox [72 72 540 720]>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("cropbox-inside-mediabox test PDF parse")
    }

    /// §7.7.3.4: a page leaf without its own `MediaBox` must inherit from the
    /// nearest ancestor /Pages node.  Before this fix the result would have
    /// been the hardcoded 612×792 fallback.
    #[test]
    fn inherited_mediabox_produces_correct_size() {
        let doc = make_inherited_mediabox_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (500.0, 700.0),
            "inherited MediaBox [0 0 500 700] must yield 500×700, got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
    }

    /// §7.7.3.4 negative: before the fix, a leaf without its own `MediaBox` would
    /// have returned the 612×792 fallback instead of the inherited value.  This
    /// assertion confirms the old wrong behavior is gone.
    #[test]
    fn inherited_mediabox_is_not_fallback() {
        let doc = make_inherited_mediabox_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        assert_ne!(
            (geom.width_pts, geom.height_pts),
            (612.0, 792.0),
            "page with inherited MediaBox must NOT return the 612×792 default fallback"
        );
    }

    /// §14.11.2: `CropBox` extending beyond `MediaBox` must be clamped to `MediaBox`.
    /// Effective box = intersection([0 0 700 900], [0 0 600 800]) = [0 0 600 800].
    #[test]
    fn cropbox_exceeding_mediabox_is_clamped() {
        let doc = make_cropbox_exceeds_mediabox_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (600.0, 800.0),
            "CropBox larger than MediaBox must be clamped to MediaBox; got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
        // Origin must be the intersection lower-left (0,0 in this case).
        assert_eq!(
            (geom.origin_x, geom.origin_y),
            (0.0, 0.0),
            "clamped origin must be intersection lower-left"
        );
    }

    /// Pixel-neutrality invariant: `CropBox` ⊆ `MediaBox` → result equals the
    /// `CropBox` exactly (no change from pre-fix behavior for the common case).
    /// `CropBox` [72 72 540 720] inside `MediaBox` [0 0 612 792] → 468×648.
    #[test]
    fn cropbox_inside_mediabox_is_unchanged() {
        let doc = make_cropbox_inside_mediabox_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        // w = 540-72 = 468, h = 720-72 = 648
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (468.0, 648.0),
            "CropBox inside MediaBox must be unchanged; got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
        assert_eq!(
            (geom.origin_x, geom.origin_y),
            (72.0, 72.0),
            "origin must be CropBox lower-left when CropBox is inside MediaBox"
        );
    }

    /// Verify that `page_size_pts_by_id` with a pre-resolved id gives the same
    /// answer as `page_size_pts` (exercises the by-id entry point for inheritance).
    #[test]
    fn page_size_pts_by_id_agrees_with_page_size_pts() {
        let doc = make_inherited_mediabox_doc();
        let geom_num = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        let page_id = doc.get_page(0).expect("get_page(0) failed");
        let geom_id =
            page_size_pts_by_id(&doc, page_id, PageBox::Crop).expect("page_size_pts_by_id failed");
        assert_eq!(
            (geom_num.width_pts, geom_num.height_pts),
            (geom_id.width_pts, geom_id.height_pts)
        );
    }

    /// Build a hostile one-page PDF whose /Parent chain is a cycle and which
    /// has NO `MediaBox` anywhere, so resolving the inheritable box forces a full
    /// traversal of the ring.  The page leaf's /Parent points at the /Pages
    /// node, and the /Pages node's /Parent points back at the page leaf —
    /// `page leaf 3 0 ⇄ pages 2 0`.  A naive walker loops forever / overflows;
    /// the depth + visited-set guard must terminate fast and fall back.
    fn make_cyclic_parent_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        // /Pages node points back DOWN to the page leaf via /Parent — a ring.
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1 /Parent 3 0 R>>\nendobj\n";
        // Page leaf has no MediaBox; its /Parent is the /Pages node.
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("cyclic-parent test PDF parse")
    }

    /// A page with no box anywhere on its /Parent chain is assumed to be US
    /// Letter, and the /Rotate it inherits still turns it.
    #[test]
    fn missing_boxes_keep_inherited_rotate() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Rotate 90 /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R>>\nendobj\n";
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        let doc = Document::from_bytes_owned(bytes).expect("test PDF parse");

        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts");
        assert_eq!(geom.rotate_cw, 90);
        assert_eq!((geom.width_pts, geom.height_pts), (792.0, 612.0));
        assert_eq!((geom.origin_x, geom.origin_y), (0.0, 0.0));
    }

    /// `DoS` guard: a cyclic /Parent ring must terminate (no hang, no
    /// stack-overflow) and fall back to the 612×792 default rather than
    /// spinning.  This is the campaign's recurring hostile-PDF class.
    #[test]
    fn cyclic_parent_chain_terminates_with_fallback() {
        let doc = make_cyclic_parent_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop)
            .expect("page_size_pts must not hang/panic on a cycle");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (612.0, 792.0),
            "cyclic /Parent with no MediaBox must fall back to 612×792, got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
    }

    /// Build a PDF where `MediaBox` [0 0 600 800] and `CropBox` [900 900 1000 1000]
    /// are disjoint (`CropBox` entirely outside `MediaBox`).  §14.11.2: the
    /// intersection is empty, so the effective box must fall back to `MediaBox` —
    /// never a zero/negative-area box and never the 612×792 default.
    fn make_disjoint_cropbox_doc() -> Document {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R \
                    /MediaBox [0 0 600 800] /CropBox [900 900 1000 1000]>>\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let xref_start = off3 + obj3.len();
        let xref = format!(
            "xref\n0 4\n\
             0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes();
        Document::from_bytes_owned(bytes).expect("disjoint-cropbox test PDF parse")
    }

    /// §14.11.2 degenerate case: `CropBox` disjoint from `MediaBox` → empty
    /// intersection → fall back to `MediaBox` (not a zero-area box, not 612×792).
    #[test]
    fn disjoint_cropbox_falls_back_to_mediabox() {
        let doc = make_disjoint_cropbox_doc();
        let geom = page_size_pts(&doc, 1, PageBox::Crop).expect("page_size_pts failed");
        assert_eq!(
            (geom.width_pts, geom.height_pts),
            (600.0, 800.0),
            "disjoint CropBox must fall back to MediaBox 600×800, got {}×{}",
            geom.width_pts,
            geom.height_pts
        );
        assert_eq!(
            (geom.origin_x, geom.origin_y),
            (0.0, 0.0),
            "fallback origin must be the MediaBox lower-left"
        );
    }

    /// The geometry of page object `(4, 0)` with dictionary entries `page`
    /// and indirect objects 5, 6, … `objects`.
    fn geometry_of(page: &str, objects: &[&str]) -> super::PageGeometry {
        geometry_on(page, objects, PageBox::Crop)
    }

    /// As [`geometry_of`], measured on `page_box`.
    fn geometry_on(page: &str, objects: &[&str], page_box: PageBox) -> super::PageGeometry {
        let mut all = vec![(4, format!("<< /Type /Page {page} >>"))];
        all.extend((5..).zip(objects.iter().map(|o| (*o).to_owned())));
        let doc = crate::test_helpers::make_doc_with_objects(&all);
        page_size_pts_by_id(&doc, (4, 0), page_box).expect("geometry")
    }

    /// `(width_pts, height_pts, origin_x, origin_y)` of `geom`.
    fn extent(geom: &super::PageGeometry) -> (f64, f64, f64, f64) {
        (
            geom.width_pts,
            geom.height_pts,
            geom.origin_x,
            geom.origin_y,
        )
    }

    const ALL_BOXES: [PageBox; 5] = [
        PageBox::Media,
        PageBox::Crop,
        PageBox::Bleed,
        PageBox::Trim,
        PageBox::Art,
    ];

    #[test]
    fn media_and_crop_boxes_are_inherited() {
        let page = "/Parent 5 0 R";
        let pages = "<< /Type /Pages /MediaBox [0 0 600 800] /CropBox [50 60 550 760] >>";
        let media = geometry_on(page, &[pages], PageBox::Media);
        assert_eq!(extent(&media), (600.0, 800.0, 0.0, 0.0));
        let crop = geometry_on(page, &[pages], PageBox::Crop);
        assert_eq!(extent(&crop), (500.0, 700.0, 50.0, 60.0));
    }

    #[test]
    fn each_leaf_box_is_selected_with_its_own_origin() {
        let page = "/MediaBox [0 0 600 800] /CropBox [10 10 590 790] \
                    /BleedBox [20 30 580 770] /TrimBox [40 50 560 750] /ArtBox [300 400 100 200]";
        let expected = [
            (PageBox::Media, (600.0, 800.0, 0.0, 0.0)),
            (PageBox::Crop, (580.0, 780.0, 10.0, 10.0)),
            (PageBox::Bleed, (560.0, 740.0, 20.0, 30.0)),
            (PageBox::Trim, (520.0, 700.0, 40.0, 50.0)),
            (PageBox::Art, (200.0, 200.0, 100.0, 200.0)),
        ];
        for (page_box, want) in expected {
            assert_eq!(
                extent(&geometry_on(page, &[], page_box)),
                want,
                "{page_box:?}"
            );
        }
    }

    #[test]
    fn inherited_bleed_trim_and_art_boxes_are_ignored() {
        let page = "/Parent 5 0 R /MediaBox [0 0 600 800] /CropBox [10 20 590 780]";
        let pages = "<< /Type /Pages /BleedBox [0 0 100 100] /TrimBox [0 0 100 100] \
                     /ArtBox [0 0 100 100] >>";
        for page_box in [PageBox::Bleed, PageBox::Trim, PageBox::Art] {
            let geom = geometry_on(page, &[pages], page_box);
            assert_eq!(extent(&geom), (580.0, 760.0, 10.0, 20.0), "{page_box:?}");
        }
    }

    #[test]
    fn absent_or_unusable_leaf_boxes_default_to_the_crop_box() {
        let crop = "/MediaBox [0 0 600 800] /CropBox [72 72 540 720]";
        for page_box in [PageBox::Bleed, PageBox::Trim, PageBox::Art] {
            let geom = geometry_on(crop, &[], page_box);
            assert_eq!(extent(&geom), (468.0, 648.0, 72.0, 72.0), "{page_box:?}");
            let unusable = format!("{crop} /{} [0 0 /W 100]", page_box.key());
            let geom = geometry_on(&unusable, &[], page_box);
            assert_eq!(extent(&geom), (468.0, 648.0, 72.0, 72.0), "{page_box:?}");
        }
        // Without a CropBox the default is the MediaBox.
        let geom = geometry_on("/MediaBox [0 0 600 800]", &[], PageBox::Trim);
        assert_eq!(extent(&geom), (600.0, 800.0, 0.0, 0.0));
    }

    #[test]
    fn leaf_boxes_are_intersected_with_the_media_box() {
        let page = "/MediaBox [0 0 600 800] /TrimBox [-50 100 300 900]";
        let geom = geometry_on(page, &[], PageBox::Trim);
        assert_eq!(extent(&geom), (300.0, 700.0, 0.0, 100.0));
    }

    #[test]
    fn a_leaf_box_disjoint_from_the_media_box_falls_back_to_the_crop_box() {
        let page = "/MediaBox [0 0 600 800] /CropBox [10 10 500 700] /BleedBox [900 900 1000 1000]";
        let geom = geometry_on(page, &[], PageBox::Bleed);
        assert_eq!(extent(&geom), (490.0, 690.0, 10.0, 10.0));
    }

    #[test]
    fn the_media_box_ignores_a_disjoint_crop_box() {
        let page = "/MediaBox [5 5 605 805] /CropBox [900 900 1000 1000]";
        let geom = geometry_on(page, &[], PageBox::Media);
        assert_eq!(extent(&geom), (600.0, 800.0, 5.0, 5.0));
    }

    #[test]
    fn the_media_box_falls_back_to_the_crop_box_when_absent() {
        let geom = geometry_on("/CropBox [10 10 110 210]", &[], PageBox::Media);
        assert_eq!(extent(&geom), (100.0, 200.0, 10.0, 10.0));
    }

    #[test]
    fn a_leaf_box_may_be_indirect() {
        let geom = geometry_on(
            "/MediaBox [0 0 600 800] /ArtBox 5 0 R",
            &["[100 100 200 300]"],
            PageBox::Art,
        );
        assert_eq!(extent(&geom), (100.0, 200.0, 100.0, 100.0));
    }

    #[test]
    fn rotate_applies_to_every_box() {
        let page = "/MediaBox [0 0 600 800] /CropBox [0 0 500 700] /BleedBox [0 0 400 600] \
                    /TrimBox [0 0 300 500] /ArtBox [0 0 200 400] /Rotate 90";
        let unrotated = [
            (600.0, 800.0),
            (500.0, 700.0),
            (400.0, 600.0),
            (300.0, 500.0),
            (200.0, 400.0),
        ];
        for (page_box, (w, h)) in ALL_BOXES.into_iter().zip(unrotated) {
            let geom = geometry_on(page, &[], page_box);
            assert_eq!(geom.rotate_cw, 90);
            assert_eq!((geom.width_pts, geom.height_pts), (h, w), "{page_box:?}");
        }
    }

    #[test]
    fn the_crop_box_is_the_default() {
        assert_eq!(PageBox::default(), PageBox::Crop);
    }

    #[test]
    fn the_page_boxes_are_reported_whichever_box_is_selected() {
        let page = "/MediaBox [0 0 600 800] /CropBox [10 10 590 790] /TrimBox [40 50 560 750]";
        for page_box in ALL_BOXES {
            let geom = geometry_on(page, &[], page_box);
            assert_eq!(geom.media_box, Some([0.0, 0.0, 600.0, 800.0]));
            assert_eq!(geom.crop_box, Some([10.0, 10.0, 590.0, 790.0]));
        }
    }

    #[test]
    fn a_box_with_a_non_numeric_entry_is_unusable() {
        let geom = geometry_of("/MediaBox [10 10 /W 792]", &[]);
        assert_eq!(geom.media_box, None);
        assert_eq!((geom.width_pts, geom.height_pts), (612.0, 792.0));
    }

    #[test]
    fn box_entries_and_user_unit_may_be_indirect() {
        let geom = geometry_of("/MediaBox [0 0 5 0 R 400] /UserUnit 6 0 R", &["300", "2"]);
        assert_eq!(geom.user_unit, 2.0);
        assert_eq!((geom.width_pts, geom.height_pts), (600.0, 800.0));
    }

    #[test]
    fn a_real_rotate_is_read_as_an_integer() {
        let geom = geometry_of("/MediaBox [0 0 300 400] /Rotate 90.0", &[]);
        assert_eq!(geom.rotate_cw, 90);
        assert_eq!((geom.width_pts, geom.height_pts), (400.0, 300.0));
    }
}

#[cfg(test)]
mod open_bytes_tests {
    use super::{open_bytes, page_count};

    /// A minimal valid single-page PDF (one empty 612×792-pt page).
    fn minimal_pdf() -> Vec<u8> {
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn open_bytes_parses_an_in_memory_pdf() {
        let doc = open_bytes(minimal_pdf()).expect("minimal in-memory PDF must parse");
        assert_eq!(
            page_count(&doc),
            1,
            "the in-memory PDF has exactly one page"
        );
    }

    #[test]
    fn open_bytes_rejects_non_pdf_bytes() {
        // `Document` is not `Debug`, so match the result rather than
        // `expect_err` (which would need to format the `Ok` value).
        match open_bytes(b"not a pdf at all".to_vec()) {
            Ok(_) => panic!("garbage bytes must not parse as a PDF"),
            Err(super::InterpError::Pdf(_)) => {}
            Err(other) => panic!("garbage bytes must surface as InterpError::Pdf, got {other:?}"),
        }
    }

    #[test]
    fn a_parser_error_is_the_source_of_the_interp_error() {
        let Err(e) = super::open("/nonexistent/rasterrocket-test.pdf") else {
            panic!("a missing file must not open");
        };
        let source = std::error::Error::source(&e).expect("the parser error is the source");
        let parser = source
            .downcast_ref::<pdf::PdfError>()
            .expect("the source is a pdf::PdfError");
        assert!(
            matches!(parser, pdf::PdfError::Io(_)),
            "a missing file is an I/O error, got {parser:?}"
        );
    }

    #[test]
    fn a_device_kernel_is_enabled_only_below_its_maximum_threshold() {
        use super::{DeviceKernel, device_kernel_enabled};
        #[cfg(feature = "gpu-aa")]
        let aa = super::aa_fill_threshold() != usize::MAX;
        #[cfg(not(feature = "gpu-aa"))]
        let aa = false;
        #[cfg(feature = "gpu-jpeg-huffman")]
        let huffman = super::huffman_threshold() != u32::MAX;
        #[cfg(not(feature = "gpu-jpeg-huffman"))]
        let huffman = false;
        assert_eq!(device_kernel_enabled(DeviceKernel::AaFill), aa);
        assert_eq!(device_kernel_enabled(DeviceKernel::HuffmanJpeg), huffman);
    }
}
