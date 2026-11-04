//! `/Linearized` (Fast Web View) hint-table detection.
//!
//! Per PDF 1.7 § F.3.3, a linearized PDF places a `/Linearized` dict as the
//! first indirect object in the file, within its first 1024 bytes, and
//! records the file's length in `/L`.  A file whose length differs has been
//! updated since it was linearized, and its linearization no longer holds.
//!
//! This module reads the dictionary's `/N`, `/O` and `/H` entries.  The
//! contents of the Page Offset Hint Table (§ F.4.5) are not parsed; page
//! lookup always descends the page tree, and the page count is the tree's.

use crate::{document::Document, object::Object};

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
    /// `/O`, `/H[0]` and `/H[1]`, and an `/L` equal to the file's length.
    /// A document that is not linearized, was updated after it was, or
    /// whose dictionary lacks one of those entries, yields `None`.
    pub(crate) fn load(doc: &Document) -> Option<Self> {
        let lin_obj = find_linearized_object(doc)?;
        let dict = lin_obj.as_dict()?;
        let file_length = doc.bytes().len() - doc.base();
        if dict.get(b"L").and_then(Object::as_u64) != u64::try_from(file_length).ok() {
            log::debug!(
                "LinearizationHints: /L differs from the file length {file_length}; the file \
                 was updated after linearization",
            );
            return None;
        }
        let hints = parse_dict(dict, doc.base());
        if hints.is_none() {
            log::debug!(
                "LinearizationHints: /Linearized dict present but required keys missing; \
                 falling back to descent",
            );
        }
        hints
    }
}

/// The bytes after the header in which the `/Linearized` dict begins.
const FIRST_OBJECT_WINDOW: usize = 1024;

/// The first indirect object of the file, when it is a `/Linearized`
/// dict.  Any failure to find, read or destructure it yields `None`, since
/// a non-linearized PDF is not an error.
fn find_linearized_object(doc: &Document) -> Option<std::sync::Arc<Object>> {
    let body = &doc.bytes()[doc.base()..];
    let id = first_object_id(&body[..body.len().min(FIRST_OBJECT_WINDOW)])?;
    let obj = doc.get_object(id).ok()?;
    obj.as_dict()?.contains_key(b"Linearized").then_some(obj)
}

/// The `(number, generation)` of the first `N G obj` header in `head`, the
/// file from its `%PDF-` header on, after the header line and any comment
/// lines and white space.
fn first_object_id(head: &[u8]) -> Option<(u32, u16)> {
    let mut at = 0;
    loop {
        at = skip_white(head, at);
        if head.get(at) != Some(&b'%') {
            break;
        }
        while head.get(at).is_some_and(|&b| b != b'\r' && b != b'\n') {
            at += 1;
        }
    }
    let (num, at) = integer(head, at)?;
    let gen_at = skip_white(head, at);
    if gen_at == at {
        return None;
    }
    let (generation, at) = integer(head, gen_at)?;
    let obj_at = skip_white(head, at);
    if obj_at == at || !head[obj_at..].starts_with(b"obj") {
        return None;
    }
    Some((u32::try_from(num).ok()?, u16::try_from(generation).ok()?))
}

/// The position of the first byte at or after `at` that is not white space.
fn skip_white(bytes: &[u8], mut at: usize) -> usize {
    while bytes
        .get(at)
        .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\0'))
    {
        at += 1;
    }
    at
}

/// The unsigned integer of the digits at `at`, and the position after them.
fn integer(bytes: &[u8], at: usize) -> Option<(u64, usize)> {
    let digits = bytes[at..]
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .count();
    let text = std::str::from_utf8(&bytes[at..at + digits]).ok()?;
    Some((text.parse().ok()?, at + digits))
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
    fn the_page_count_is_the_trees_not_the_dicts() {
        let bytes = pdf(&[(1, LINEARIZED), (2, CATALOG), (3, PAGES), (4, PAGE)], 2);
        let doc = Document::from_bytes_owned(bytes).unwrap();
        assert_eq!(doc.linearization_hints().unwrap().page_count, 100);
        assert_eq!(doc.page_count_fast(), 1);
    }

    #[test]
    fn object_headers_are_read_after_comments_and_white_space() {
        assert_eq!(
            first_object_id(b"%PDF-1.7\n%\xE2\xE3\n\r\n 12 3 obj<<>>"),
            Some((12, 3))
        );
        assert_eq!(first_object_id(b"%PDF-1.7\n12 0 obj"), Some((12, 0)));
        assert_eq!(first_object_id(b"%PDF-1.7\n120 obj"), None);
        assert_eq!(first_object_id(b"%PDF-1.7\n12 0obj"), None);
        assert_eq!(first_object_id(b"%PDF-1.7\nxref"), None);
        assert_eq!(first_object_id(b"%PDF-1.7\n"), None);
    }
}
