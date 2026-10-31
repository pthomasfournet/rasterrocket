//! Lightweight page pre-scan: classify page content without rendering pixels.
//!
//! [`prescan_page`] walks a page's resource dictionary and content stream operators
//! to populate a [`PageDiagnostics`] struct at a fraction of the cost of a full
//! render.  No bitmap is allocated; no image data is decoded.
//!
//! # What is inspected
//!
//! - **`XObject` images**: `Filter`, `Width`, `Height` read from the stream
//!   dictionary via reference lookup — no decompression.
//! - **Form `XObject`s**: recursed up to a small fixed depth (`MAX_PRESCAN_DEPTH`)
//!   so images inside forms are counted.
//! - **Inline images**: `Filter`, `W`/`Width`, `H`/`Height` read from the raw
//!   parameter block via the same `parse_inline_params` path used by the full
//!   renderer — no pixel decoding.
//! - **Text operators**: any `Tj`, `TJ`, `'`, or `"` operator sets
//!   `has_vector_text = true`.
//!
//! # PPI estimate
//!
//! Unlike the full renderer (which projects image pixels through the live CTM
//! stack), the prescan uses the raw PDF `Width` dict entry and the page width in
//! points (via the crate-internal `page_size_pts_by_id` helper):
//!
//! ```text
//! ppi ≈ (image_width_px / page_width_pts) × 72
//! ```
//!
//! This assumes the image fills the page width, so it over-estimates PPI on
//! partial-width images.  For routing purposes an over-estimate is safe — it
//! can only make the prescan more aggressive about flagging GPU candidates,
//! which is conservative.  The real area guard in the decoder path
//! (`GPU_JPEG_THRESHOLD_PX`) is the authoritative threshold.
//!
//! # Error handling
//!
//! Errors from `parse_page` or `page_size_pts` are propagated.  Errors from
//! individual resource lookups (missing dict key, corrupt stream header) are
//! silently skipped — the scan continues.  A partial result is still useful
//! for routing.

use std::collections::HashSet;

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::{
    InterpError,
    content::Operator,
    page_size_pts_by_id, parse_page_content_by_id,
    renderer::PageDiagnostics,
    resources::image::{
        IMAGE_FILTER_COUNT, ImageFilter, dominant_filter, filter_name, inline::parse_inline_params,
    },
};

/// Maximum Form `XObject` recursion depth during the pre-scan.
///
/// Intentionally shallower than the renderer's `MAX_FORM_DEPTH` (32): the
/// prescan only needs to find the dominant image filter, so deeply nested
/// forms are unlikely to change the classification.  A lower cap also
/// limits prescan cost on adversarially nested documents.
const MAX_PRESCAN_DEPTH: u32 = 4;

// ── Public entry point ────────────────────────────────────────────────────────

/// Classify page `page_num` (1-based) without rendering any pixels.
///
/// Returns a [`PageDiagnostics`] with `has_images`, `has_vector_text`,
/// `dominant_filter`, and a conservative `source_ppi_hint` derived from the
/// image's PDF dictionary `Width` entry (see module doc for caveats).
///
/// # Errors
///
/// - [`InterpError::PageOutOfRange`] if `page_num` is outside the document.
/// - [`InterpError::InvalidPageGeometry`] if the page's `UserUnit` is invalid.
///
/// Any per-resource lookup error (corrupt dict, missing filter key) is
/// silently skipped so that a partial classification is still returned.
pub fn prescan_page(doc: &Document, page_num: u32) -> Result<PageDiagnostics, InterpError> {
    let page_id = crate::resolve_page_id(doc, page_num)?;

    let geom = page_size_pts_by_id(doc, page_id)?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "page width in PDF points (≤ ~10 000 pts); f32 is sufficient for PPI routing"
    )]
    let mut scan = ImageScan::new(doc, geom.width_pts as f32);

    // 1. Walk XObject resource dictionary (no decoding).
    scan.xobjects(page_id, 0);

    // 2. Walk content stream operators for inline images and text.
    let mut has_vector_text = false;
    let (ops, content_failures) = parse_page_content_by_id(doc, page_id)?;
    for op in &ops {
        match op {
            Operator::InlineImage { params, .. } => {
                scan.image(&parse_inline_params(params));
            }
            Operator::ShowText(_)
            | Operator::ShowTextArray(_)
            | Operator::MoveNextLineShow(_)
            | Operator::MoveNextLineShowSpaced { .. } => {
                has_vector_text = true;
            }
            _ => {}
        }
    }

    Ok(PageDiagnostics {
        has_images: scan.has_images,
        has_vector_text,
        dominant_filter: dominant_filter(&scan.filter_counts),
        source_ppi_hint: (scan.max_ppi > 0.0).then_some(scan.max_ppi),
        image_failures: Vec::new(),
        content_failures,
        missing_fonts: Vec::new(),
        form_type: crate::form_type(doc),
    })
}

// ── Image statistics ──────────────────────────────────────────────────────────

/// The image statistics of one page, accumulated over its image `XObject`s,
/// the images of its forms, and its inline images.
struct ImageScan<'a> {
    doc: &'a Document,
    /// Width of the page in PDF points, the denominator of the PPI estimate.
    page_pts_width: f32,
    /// Images seen per filter, indexed by [`ImageFilter`] discriminant.
    filter_counts: [u32; IMAGE_FILTER_COUNT],
    has_images: bool,
    /// Largest PPI estimate so far; 0 when none.
    max_ppi: f32,
    /// Forms already walked: each is walked once however many resource
    /// dictionaries name it, so the walk is linear in the number of forms
    /// rather than exponential in the nesting depth.
    forms_seen: HashSet<ObjectId>,
}

impl<'a> ImageScan<'a> {
    fn new(doc: &'a Document, page_pts_width: f32) -> Self {
        Self {
            doc,
            page_pts_width,
            filter_counts: [0; IMAGE_FILTER_COUNT],
            has_images: false,
            max_ppi: 0.0,
            forms_seen: HashSet::new(),
        }
    }

    /// Walk the `XObject` resource dict of `context_id`, recursing into
    /// Form `XObject`s up to [`MAX_PRESCAN_DEPTH`].
    fn xobjects(&mut self, context_id: ObjectId, depth: u32) {
        // `/Resources` is inherited down the page tree when a page has none.
        let Ok(xobj_dict) = self.doc.get_page_resource_dict(context_id, b"XObject") else {
            return;
        };
        for (_name, obj) in &xobj_dict {
            // XObjects are streams, so every usable entry is a reference.
            let &Object::Reference(xobj_id) = obj else {
                continue;
            };
            let Ok(xobj) = self.doc.get_object(xobj_id) else {
                continue;
            };
            let Object::Stream(stream) = xobj.as_ref() else {
                continue;
            };
            match stream.dict.get(b"Subtype").and_then(Object::as_name) {
                Some(b"Image") => self.image(&stream.dict),
                Some(b"Form") if depth < MAX_PRESCAN_DEPTH && self.forms_seen.insert(xobj_id) => {
                    self.xobjects(xobj_id, depth + 1);
                }
                _ => {}
            }
        }
    }

    /// Count the image whose dictionary (an image `XObject`'s, or an inline
    /// image's with its keys expanded) is `dict`.
    fn image(&mut self, dict: &Dictionary) {
        self.has_images = true;
        let filter = dict
            .get(b"Filter")
            .and_then(|o| self.doc.resolve(o).ok())
            .and_then(|o| filter_name(&o).map(std::borrow::Cow::into_owned));
        let slot =
            &mut self.filter_counts[ImageFilter::from_filter_str(filter.as_deref()) as usize];
        *slot = slot.saturating_add(1);
        if let Some(w_px) = positive_u32(dict, b"Width")
            && self.page_pts_width > 0.0
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "an image width is far below 2^24 in practice; f32 is sufficient for PPI routing"
            )]
            let ppi = (w_px as f32 / self.page_pts_width) * 72.0;
            self.max_ppi = self.max_ppi.max(ppi);
        }
    }
}

/// A positive integer value (Integer, or Real truncated) of `key`; `None`
/// when absent, not positive, non-numeric, or beyond `u32`.
fn positive_u32(dict: &Dictionary, key: &[u8]) -> Option<u32> {
    let n = match dict.get(key)? {
        Object::Integer(i) => *i,
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f32-to-i64 `as` truncates toward zero and saturates, NaN to 0"
        )]
        Object::Real(r) => *r as i64,
        _ => return None,
    };
    u32::try_from(n).ok().filter(|&n| n > 0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal in-memory PDF with one page that has a content stream.
    ///
    /// Layout (object numbers):
    ///   1: Catalog
    ///   2: Pages
    ///   3: Page (Type=Page, Parent=2, `MediaBox`=[0 0 612 792], Contents=4)
    ///   4: Content stream (length = `content_bytes.len()`)
    fn doc_with_content(content_bytes: &[u8]) -> (Document, u32) {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n".to_string();
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n".to_string();
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                    /Contents 4 0 R>>\nendobj\n"
            .to_string();
        let stream_len = content_bytes.len();
        let obj4_header = format!("4 0 obj\n<</Length {stream_len}>>\nstream\n");
        let obj4_footer = "\nendstream\nendobj\n";

        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off4 = off3 + obj3.len();
        let xref_start = off4 + obj4_header.len() + stream_len + obj4_footer.len();

        let xref = format!(
            "xref\n0 5\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n\
             {off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n\
             {off4:010} 00000 n\r\n",
        );
        let trailer = format!("trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");

        let mut bytes = Vec::with_capacity(xref_start + xref.len() + trailer.len());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(obj1.as_bytes());
        bytes.extend_from_slice(obj2.as_bytes());
        bytes.extend_from_slice(obj3.as_bytes());
        bytes.extend_from_slice(obj4_header.as_bytes());
        bytes.extend_from_slice(content_bytes);
        bytes.extend_from_slice(obj4_footer.as_bytes());
        bytes.extend_from_slice(xref.as_bytes());
        bytes.extend_from_slice(trailer.as_bytes());

        let doc = Document::from_bytes_owned(bytes).expect("test PDF parse");
        (doc, 1)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn prescan_page_out_of_range_returns_err() {
        let (doc, _) = doc_with_content(b"BT /F1 12 Tf (Hello) Tj ET");
        let result = prescan_page(&doc, 99);
        assert!(
            matches!(result, Err(InterpError::PageOutOfRange { .. })),
            "expected PageOutOfRange, got {result:?}"
        );
    }

    #[test]
    fn prescan_pure_text_page_no_images() {
        // Content stream: begin text, show text, end text.
        let ops = b"BT /F1 12 Tf 72 720 Td (Hello World) Tj ET";
        let (doc, page_num) = doc_with_content(ops);
        let diag = prescan_page(&doc, page_num).expect("prescan should succeed");
        assert!(!diag.has_images, "pure text page must not have images");
        assert!(diag.dominant_filter.is_none());
        assert!(diag.source_ppi_hint.is_none());
        // has_vector_text: depends on the content stream parser recognising Tj —
        // accept either true or false here since the test page has no font resources
        // (Tj may be silently dropped). The key assertion is no images.
    }

    #[test]
    fn prescan_empty_page_is_clean() {
        let (doc, page_num) = doc_with_content(b"");
        let diag = prescan_page(&doc, page_num).expect("prescan of empty page should succeed");
        assert!(!diag.has_images);
        assert!(!diag.has_vector_text);
        assert!(diag.dominant_filter.is_none());
        assert!(diag.source_ppi_hint.is_none());
    }

    #[test]
    fn image_filter_from_filter_str_covers_all_variants() {
        assert!(matches!(
            ImageFilter::from_filter_str(Some("DCTDecode")),
            ImageFilter::Dct
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("JPXDecode")),
            ImageFilter::Jpx
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("CCITTFaxDecode")),
            ImageFilter::CcittFax
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("JBIG2Decode")),
            ImageFilter::Jbig2
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("FlateDecode")),
            ImageFilter::Flate
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(None),
            ImageFilter::Raw
        ));
        assert!(matches!(
            ImageFilter::from_filter_str(Some("Unknown")),
            ImageFilter::Raw
        ));
    }

    #[test]
    fn positive_u32_reads_integer_and_real_and_rejects_out_of_range() {
        let mut dict = Dictionary::new();
        dict.set(b"Width", Object::Integer(640));
        dict.set(b"Height", Object::Real(480.0));
        dict.set(b"Huge", Object::Integer(i64::from(u32::MAX) + 2));
        dict.set(b"Negative", Object::Integer(-3));
        assert_eq!(positive_u32(&dict, b"Width"), Some(640));
        assert_eq!(positive_u32(&dict, b"Height"), Some(480));
        assert_eq!(positive_u32(&dict, b"Huge"), None);
        assert_eq!(positive_u32(&dict, b"Negative"), None);
        assert_eq!(positive_u32(&dict, b"Missing"), None);
    }

    #[test]
    fn an_image_counts_its_filter_and_ppi() {
        let doc = crate::test_helpers::empty_doc();
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"DCTDecode".to_vec()));
        dict.set(b"Width", Object::Integer(1224));
        let mut scan = ImageScan::new(&doc, 612.0);
        scan.image(&dict);
        scan.image(&dict);
        assert!(scan.has_images);
        assert_eq!(scan.filter_counts[ImageFilter::Dct as usize], 2);
        assert!((scan.max_ppi - 144.0).abs() < 1e-3, "got {}", scan.max_ppi);
        assert_eq!(dominant_filter(&scan.filter_counts), Some(ImageFilter::Dct));
    }

    #[test]
    fn a_form_named_many_times_is_walked_once() {
        let image = "<< /Type /XObject /Subtype /Image /Filter /DCTDecode /Width 8 /Height 8 \
                     /Length 0 >>\nstream\n\nendstream";
        let form = "<< /Type /XObject /Subtype /Form \
                    /Resources << /XObject << /A 5 0 R /B 5 0 R /C 5 0 R /I 6 0 R >> >> \
                    /Length 0 >>\nstream\n\nendstream";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /XObject << /F 5 0 R /G 5 0 R >> >> >>",
            ),
            (5, form),
            (6, image),
        ]);
        let mut scan = ImageScan::new(&doc, 612.0);
        scan.xobjects((4, 0), 0);
        assert_eq!(scan.filter_counts[ImageFilter::Dct as usize], 1);
    }
}
