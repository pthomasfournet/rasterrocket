//! `/Linearized` (Fast Web View) detection.
//!
//! Per ISO 32000-1 Annex F.3.3, a linearized PDF places a `/Linearized`
//! dictionary as the first indirect object of the file, beginning within its
//! first 1024 bytes, and records the file's length in `/L`.  When the length
//! differs, the linearization parameters no longer describe the file.
//!
//! [`LinearizationHints`] is public API of the parser; the renderer does not
//! consult it.  The contents of the Page Offset Hint Table (§ F.4.5) are not
//! parsed: page lookup always descends the page tree, and the page count is
//! the tree's.

use crate::{document::Document, lexer::skip_ws, object::Object, object::parse_indirect_object};

/// Parsed `/Linearized` parameters.
#[derive(Debug, Clone, Copy)]
pub struct LinearizationHints {
    /// `/N` — total number of pages.
    pub page_count: u32,
    /// `/O` — object number of the first page (page 0).
    pub first_page_obj: u32,
    /// `/H[0]` — position of the Page Offset Hint Table stream in
    /// [`Document::bytes`]: the offset written in the file plus the bytes
    /// that precede the `%PDF-` header, if any.
    pub hint_stream_offset: u64,
    /// `/H[1]` — length of the Page Offset Hint Table stream.
    pub hint_stream_length: u64,
}

impl LinearizationHints {
    /// The linearization parameters of `doc`: `Some` only when the file's
    /// first indirect object is a `/Linearized` dictionary carrying `/N`,
    /// `/O`, `/H[0]` and `/H[1]`, and an `/L` equal to the length of the file
    /// from its `%PDF-` header on.
    pub(crate) fn load(doc: &Document) -> Option<Self> {
        let bytes = doc.bytes();
        let header = find_header(bytes)?;
        let dict = first_object(&bytes[header..])?;
        if !dict.contains_key(b"Linearized") {
            return None;
        }
        let file_length = u64::try_from(bytes.len() - header).ok()?;
        match dict.get(b"L").and_then(Object::as_u64) {
            Some(length) if length == file_length => {}
            Some(length) => {
                log::debug!(
                    "LinearizationHints: /L {length} differs from the file length \
                     {file_length}; the parameters do not describe the file"
                );
                return None;
            }
            None => {
                log::debug!("LinearizationHints: /Linearized dict has no integer /L");
                return None;
            }
        }
        let hints = parse_dict(&dict, header);
        if hints.is_none() {
            log::debug!("LinearizationHints: /Linearized dict lacks /N, /O or /H");
        }
        hints
    }
}

/// The bytes of the file within which the `%PDF-` header and the
/// `/Linearized` dictionary each begin.
const FIRST_OBJECT_WINDOW: usize = 1024;

/// The position of the `%PDF-` header within the file's first
/// [`FIRST_OBJECT_WINDOW`] bytes.
fn find_header(bytes: &[u8]) -> Option<usize> {
    bytes[..bytes.len().min(FIRST_OBJECT_WINDOW)]
        .windows(5)
        .position(|w| w == b"%PDF-")
}

/// The first indirect object of `file`, the file from its header on, read
/// in place when it is a dictionary beginning within the first
/// [`FIRST_OBJECT_WINDOW`] bytes.  The header line and any other comments
/// before it are skipped as white space.
fn first_object(file: &[u8]) -> Option<crate::dictionary::Dictionary> {
    let mut at = 0;
    skip_ws(file, &mut at);
    if at >= FIRST_OBJECT_WINDOW {
        return None;
    }
    match parse_indirect_object(file, at)?.1 {
        Object::Dictionary(dict) => Some(dict),
        _ => None,
    }
}

/// Pure parsing of a `/Linearized` dict into `LinearizationHints`, for a
/// file whose written offsets are relative to position `base`.
/// Returns `None` if any required key is missing or malformed.
fn parse_dict(lin_dict: &crate::dictionary::Dictionary, base: usize) -> Option<LinearizationHints> {
    let page_count = lin_dict.get(b"N").and_then(Object::as_u32)?;
    let first_page_obj = lin_dict.get(b"O").and_then(Object::as_u32)?;

    // /H is an array of [offset, length, ...].  First pair is mandatory
    // (Page Offset Hint Table); subsequent pairs are optional.  Borrow the
    // array — we only read [0] and [1], no need to clone the Vec.
    let h_array = lin_dict.get(b"H").and_then(Object::as_array)?;
    let hint_stream_offset = h_array
        .first()
        .and_then(Object::as_u64)?
        .checked_add(u64::try_from(base).ok()?)?;
    let hint_stream_length = h_array.get(1).and_then(Object::as_u64)?;

    Some(LinearizationHints {
        page_count,
        first_page_obj,
        hint_stream_offset,
        hint_stream_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    /// A PDF of `objects`, `(number, body)` in file order, with its
    /// cross-reference table and a trailer naming `root`.  A `/L 0000000000`
    /// in a body becomes the file's length.
    fn pdf(objects: &[(u32, &str)], root: u32) -> Vec<u8> {
        let mut out = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
        let size = objects.iter().map(|&(n, _)| n).max().unwrap() + 1;
        let mut offsets = vec![None; size as usize];
        for &(n, body) in objects {
            offsets[n as usize] = Some(out.len());
            out.extend_from_slice(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {size}\n").as_bytes());
        for offset in offsets {
            let entry = offset.map_or_else(
                || "0000000000 65535 f\r\n".to_owned(),
                |o| format!("{o:010} 00000 n\r\n"),
            );
            out.extend_from_slice(entry.as_bytes());
        }
        out.extend_from_slice(
            format!("trailer\n<</Size {size} /Root {root} 0 R>>\nstartxref\n{xref}\n%%EOF\n")
                .as_bytes(),
        );
        let length = format!("/L {:010}", out.len());
        if let Some(at) = out.windows(13).position(|w| w == b"/L 0000000000") {
            out[at..at + 13].copy_from_slice(length.as_bytes());
        }
        out
    }

    const LINEARIZED: &str = "<</Linearized 1 /L 0000000000 /N 100 /O 5 /H [12345 678]>>";
    const CATALOG: &str = "<</Type /Catalog /Pages 3 0 R>>";
    const PAGES: &str = "<</Type /Pages /Kids [4 0 R] /Count 1>>";
    const PAGE: &str = "<</Type /Page /Parent 3 0 R /MediaBox [0 0 612 792]>>";

    fn load(bytes: Vec<u8>) -> Option<LinearizationHints> {
        LinearizationHints::load(&Document::from_bytes_owned(bytes).unwrap())
    }

    #[test]
    fn a_file_without_the_dict_first_is_not_linearized() {
        let plain = pdf(&[(2, CATALOG), (3, PAGES), (4, PAGE)], 2);
        assert!(load(plain).is_none());
        // Object 1 is the dict but not the file's first object.
        let late = pdf(&[(2, CATALOG), (1, LINEARIZED), (3, PAGES), (4, PAGE)], 2);
        assert!(load(late).is_none());
    }

    #[test]
    fn the_first_object_is_the_dict_whatever_its_number() {
        for number in [1, 7] {
            let bytes = pdf(
                &[(number, LINEARIZED), (2, CATALOG), (3, PAGES), (4, PAGE)],
                2,
            );
            let hints = load(bytes.clone()).expect("Some(hints)");
            assert_eq!(hints.page_count, 100);
            assert_eq!(hints.first_page_obj, 5);
            assert_eq!(hints.hint_stream_offset, 12345);
            assert_eq!(hints.hint_stream_length, 678);

            // Bytes before the header shift every written offset, the hint
            // stream's included; /L counts from the header.
            let prefix = b"Header: x\n";
            let hints = load([&prefix[..], &bytes].concat()).expect("Some(hints) behind a prefix");
            assert_eq!(hints.hint_stream_offset, 12345 + prefix.len() as u64);
        }
    }

    #[test]
    fn a_file_updated_after_linearization_is_not_linearized() {
        let mut bytes = pdf(&[(1, LINEARIZED), (2, CATALOG), (3, PAGES), (4, PAGE)], 2);
        bytes.extend_from_slice(b"% appended\n");
        assert!(load(bytes).is_none());
    }

    #[test]
    fn a_dict_missing_a_required_entry_yields_none() {
        let bytes = pdf(
            &[
                (1, "<</Linearized 1 /L 0000000000>>"),
                (2, CATALOG),
                (3, PAGES),
                (4, PAGE),
            ],
            2,
        );
        assert!(load(bytes).is_none());
    }

    #[test]
    fn an_implausible_count_yields_to_the_tree() {
        for count in ["2000000000", "0"] {
            let pages = PAGES.replace("/Count 1", &format!("/Count {count}"));
            let bytes = pdf(&[(2, CATALOG), (3, &pages), (4, PAGE)], 2);
            let doc = Document::from_bytes_owned(bytes).unwrap();
            assert_eq!(doc.page_count_fast(), 1, "/Count {count}");
        }
    }

    #[test]
    fn the_page_count_is_the_trees_not_the_dicts() {
        let bytes = pdf(&[(1, LINEARIZED), (2, CATALOG), (3, PAGES), (4, PAGE)], 2);
        let doc = Document::from_bytes_owned(bytes).unwrap();
        assert_eq!(doc.linearization_hints().unwrap().page_count, 100);
        assert_eq!(doc.page_count_fast(), 1);
    }
}
