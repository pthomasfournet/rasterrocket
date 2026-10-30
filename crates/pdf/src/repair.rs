//! Cross-reference reconstruction from a scan of the file body.
//!
//! The repair path for a file whose cross-reference chain cannot be read.
//! One pass over the bytes locates every `N G obj` header and `trailer`
//! keyword.  The object after a header is parsed far enough to step over its
//! body, so an `obj` inside a string or a stream body is not a header; a
//! `/Type /ObjStm` container is expanded into entries for the objects it
//! holds, and the dictionaries of `trailer` keywords and `/Type /XRef`
//! streams are merged into one trailer.  Later bytes supersede earlier ones,
//! the incremental-update rule.

use std::collections::{HashMap, HashSet};

use crate::{
    dictionary::Dictionary,
    error::PdfError,
    lexer::{is_ws, skip_ws},
    object::{
        MAX_STREAM_SCAN, Object, ObjectId, StreamBody, parse_dictionary, parse_object, stream_body,
    },
    objstm::ObjStm,
    xref::{XrefEntry, XrefTable},
};

/// Most entries a reconstruction records; the strict parser's sanity limit.
const SANITY_LIMIT: usize = 10_000_000;

/// What a scan of the file body found.
pub(crate) struct Scan {
    /// Every object number seen, mapped to its last definition.
    pub entries: HashMap<u32, XrefEntry>,
    /// The `trailer` and `/Type /XRef` dictionaries merged in file order.
    pub trailer: Dictionary,
    /// The last usable catalogue (`/Type /Catalog` with `/Pages`) seen.
    catalog: Option<ObjectId>,
    /// Object numbers of `/Type /ObjStm` containers seen.
    containers: HashSet<u32>,
    /// A position from which no `endstream` keyword is found.
    no_endstream_from: Option<usize>,
}

/// Rebuild a cross-reference table from a scan of the file body.
///
/// The table is shaped exactly like a strict parse, so the rest of the reader
/// is unaware repair happened.  The trailer's `/Root` is kept when it names
/// a scanned object; otherwise the last usable catalogue seen is substituted.
///
/// # Errors
/// [`PdfError::MissingXref`] when the file holds no object headers, more
/// than the sanity limit of them, or no usable catalogue.
pub(crate) fn reconstruct_xref(data: &[u8]) -> Result<XrefTable, PdfError> {
    let scan = scan_objects(data)?;
    if scan.entries.is_empty() {
        return Err(PdfError::MissingXref(
            "no recoverable objects found while repairing".into(),
        ));
    }
    let named = match scan.trailer.get(b"Root") {
        Some(Object::Reference(id)) if scan.entries.contains_key(&id.0) => Some(*id),
        _ => None,
    };
    let Some(root) = named.or(scan.catalog) else {
        return Err(PdfError::MissingXref(
            "no trailer or /Catalog object recoverable while repairing".into(),
        ));
    };
    let mut trailer = scan.trailer;
    trailer.set(b"Root", Object::Reference(root));
    Ok(XrefTable {
        entries: scan.entries,
        trailer,
    })
}

/// Scan `data` for object headers and trailers.
///
/// Bounded: every byte is examined once by the anchor scan and at most once
/// more by the object parser, since a parsed object moves the scan past its
/// body, and a position no `endstream` follows is remembered so unterminated
/// streams are not rescanned.
///
/// # Errors
/// [`PdfError::MissingXref`] when the file holds more object headers than
/// the sanity limit.
pub(crate) fn scan_objects(data: &[u8]) -> Result<Scan, PdfError> {
    let mut scan = Scan {
        entries: HashMap::new(),
        trailer: Dictionary::new(),
        catalog: None,
        containers: HashSet::new(),
        no_endstream_from: None,
    };
    let mut i = 0usize;
    while i < data.len() {
        match data[i] {
            b'o' if data[i..].starts_with(b"obj") && token_ends_at(data, i + 3) => {
                if let Some((id, header_start)) = scan_obj_header_backwards(data, i) {
                    i = scan.add_object(data, id, header_start, i + 3)?;
                } else {
                    i += 3;
                }
            }
            b't' if data[i..].starts_with(b"trailer") => {
                let mut pos = i + 7;
                skip_ws(data, &mut pos);
                if data[pos..].starts_with(b"<<")
                    && let Some(dict) = parse_dictionary(data, &mut pos)
                {
                    scan.merge_trailer(&dict);
                    i = pos;
                } else {
                    i += 7;
                }
            }
            _ => i += 1,
        }
    }
    Ok(scan)
}

impl Scan {
    /// Record object `id` whose header starts at `header_start` and whose
    /// body follows `pos`; return the position to resume scanning at.
    fn add_object(
        &mut self,
        data: &[u8],
        id: ObjectId,
        header_start: usize,
        pos: usize,
    ) -> Result<usize, PdfError> {
        let _ = self.entries.insert(
            id.0,
            XrefEntry::Direct {
                offset: header_start as u64,
                generation: id.1,
            },
        );
        if self.entries.len() > SANITY_LIMIT {
            return Err(PdfError::MissingXref(
                "reconstruction exceeded object sanity limit".into(),
            ));
        }

        let mut at = pos;
        skip_ws(data, &mut at);
        if !data[at..].starts_with(b"<<") {
            // A scalar, array or string body: step over it when it parses.
            return Ok(parse_object(data, &mut at).map_or(pos, |_| at));
        }
        let Some(dict) = parse_dictionary(data, &mut at) else {
            return Ok(pos);
        };
        let past_last_endstream = self.no_endstream_from.is_some_and(|from| at >= from);
        let body = if past_last_endstream && stream_follows(data, at) {
            StreamBody::Unterminated { start: at }
        } else {
            stream_body(data, &dict, at)
        };
        match body {
            StreamBody::None => {
                self.note_catalog(id, &dict);
                Ok(at)
            }
            StreamBody::Unterminated { start } => {
                if data.len() - start <= MAX_STREAM_SCAN {
                    self.no_endstream_from = Some(start);
                }
                Ok(start)
            }
            StreamBody::Found { body, end } => {
                match dict.get(b"Type").and_then(Object::as_name) {
                    Some(b"ObjStm") => self.expand_objstm(id.0, &data[body], &dict),
                    Some(b"XRef") => self.merge_trailer(&dict),
                    _ => {}
                }
                Ok(end)
            }
        }
    }

    /// Record the objects held by object stream `container`, whose decoded
    /// body is `content`, as compressed entries.  An object a later header
    /// defines directly, or that is itself a container, keeps its direct
    /// entry; a compressed entry otherwise supersedes an earlier direct one
    /// of generation 0.
    fn expand_objstm(&mut self, container: u32, content: &[u8], dict: &Dictionary) {
        let Ok(objstm) = ObjStm::open(content, dict) else {
            return;
        };
        let _ = self.containers.insert(container);
        for (index, &(num, _)) in objstm.directory().iter().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                break;
            };
            if self.containers.contains(&num)
                || matches!(
                    self.entries.get(&num),
                    Some(XrefEntry::Direct { generation, .. }) if *generation > 0
                )
            {
                continue;
            }
            let _ = self
                .entries
                .insert(num, XrefEntry::InObjStm { container, index });
            if let Some(Object::Dictionary(d)) = objstm.parse_at(index as usize) {
                self.note_catalog((num, 0), &d);
            }
        }
    }

    /// Merge `dict` into the trailer; its keys supersede earlier ones.
    fn merge_trailer(&mut self, dict: &Dictionary) {
        for (k, v) in dict {
            let _ = self.trailer.insert(k.clone(), v.clone());
        }
    }

    /// Remember `id` when `dict` is a usable catalogue: `/Type /Catalog`
    /// with `/Pages`.  A bare catalogue stub without `/Pages` is not one,
    /// so a truncated file is diagnosed as such rather than as a document
    /// with no pages.
    fn note_catalog(&mut self, id: ObjectId, dict: &Dictionary) {
        if dict.get(b"Type").and_then(Object::as_name) == Some(b"Catalog")
            && dict.get(b"Pages").is_some()
        {
            self.catalog = Some(id);
        }
    }
}

/// Whether the `stream` keyword follows `pos` after whitespace.
fn stream_follows(data: &[u8], pos: usize) -> bool {
    let mut p = pos;
    skip_ws(data, &mut p);
    data[p..].starts_with(b"stream")
}

/// Whether a keyword ending at `end` is followed by whitespace, a
/// delimiter, or the end of the data.
fn token_ends_at(data: &[u8], end: usize) -> bool {
    data.get(end)
        .is_none_or(|&b| is_ws(b) || matches!(b, b'<' | b'[' | b'(' | b'/' | b'%'))
}

/// Given the byte offset of the `obj` keyword, walk backwards over the
/// `<n> <g> ` prefix and return `((obj_num, gen), header_start_offset)`.
fn scan_obj_header_backwards(data: &[u8], obj_kw: usize) -> Option<(ObjectId, usize)> {
    let mut p = obj_kw;
    // Skip the single space(s) before "obj".
    while p > 0 && is_ws(data[p - 1]) {
        p -= 1;
    }
    // Generation number.
    let gen_end = p;
    while p > 0 && data[p - 1].is_ascii_digit() {
        p -= 1;
    }
    if p == gen_end {
        return None;
    }
    let generation: u16 = std::str::from_utf8(&data[p..gen_end]).ok()?.parse().ok()?;
    while p > 0 && is_ws(data[p - 1]) {
        p -= 1;
    }
    // Object number.
    let num_end = p;
    while p > 0 && data[p - 1].is_ascii_digit() {
        p -= 1;
    }
    if p == num_end {
        return None;
    }
    let num: u32 = std::str::from_utf8(&data[p..num_end]).ok()?.parse().ok()?;
    Some(((num, generation), p))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header offset of `needle` in `hay`.
    fn offset_of(hay: &[u8], needle: &[u8]) -> u64 {
        hay.windows(needle.len())
            .position(|w| w == needle)
            .expect("needle present") as u64
    }

    /// `N 0 obj <dict> stream <body> endstream endobj`.
    fn stream_object(num: u32, dict: &str, body: &[u8]) -> Vec<u8> {
        let mut out =
            format!("{num} 0 obj\n<<{dict}/Length {}>>\nstream\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendstream\nendobj\n");
        out
    }

    /// An object stream numbered `container` holding `objects`.
    fn objstm_object(container: u32, objects: &[(u32, &str)]) -> Vec<u8> {
        let mut directory = String::new();
        let mut bodies = String::new();
        for (num, body) in objects {
            directory.push_str(&format!("{num} {} ", bodies.len()));
            bodies.push_str(body);
            bodies.push('\n');
        }
        let first = directory.len();
        stream_object(
            container,
            &format!("/Type/ObjStm/N {}/First {first}", objects.len()),
            format!("{directory}{bodies}").as_bytes(),
        )
    }

    /// Incremental-update semantics: when an object number is defined more
    /// than once, the *last* (highest-offset) definition must win.  Picking
    /// the earlier version is the classic silent-wrong-recovery bug.
    #[test]
    fn reconstruct_picks_last_object_definition() {
        // No xref/startxref at all so the strict parse fails and repair runs.
        // Object 4 is defined twice: an old stub, then the live version.
        let body = b"%PDF-1.4\n\
4 0 obj<</stale true>>endobj\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]>>endobj\n\
4 0 obj<</live true>>endobj\n";
        let table = reconstruct_xref(body).expect("repair must succeed");
        let XrefEntry::Direct { offset, .. } = table.get(4).expect("obj 4 present") else {
            panic!("obj 4 must be a Direct entry");
        };
        assert_eq!(
            offset,
            offset_of(body, b"4 0 obj<</live"),
            "obj 4 must resolve to the LAST definition, not the stale one"
        );
    }

    /// A hostile file that is millions of `1 0 obj` tokens must terminate
    /// quickly: the scan is a single linear pass and duplicate object
    /// numbers collapse to one map entry (the cap is never the thing
    /// that saves us here — the linear bound is).
    #[test]
    fn reconstruct_obj_flood_is_bounded() {
        let mut body = b"%PDF-1.4\n".to_vec();
        // ~2 million repetitions of the same header → ~16 MB of input.
        for _ in 0..2_000_000 {
            body.extend_from_slice(b"1 0 obj ");
        }
        let start = std::time::Instant::now();
        // No /Root recoverable, so this returns Err — the point is that it
        // *returns* (single pass, one collapsed entry) instead of hanging.
        let _ = reconstruct_xref(&body);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "obj-flood scan must be linear, took {:?}",
            start.elapsed()
        );
    }

    /// A flood of stream objects with no `endstream` anywhere must not
    /// rescan the rest of the file for every one of them.
    #[test]
    fn unterminated_stream_flood_is_bounded() {
        let mut body = b"%PDF-1.4\n".to_vec();
        for _ in 0..200_000 {
            body.extend_from_slice(b"1 0 obj<</Length 5>>stream\n");
        }
        let start = std::time::Instant::now();
        let _ = reconstruct_xref(&body);
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "unterminated-stream scan must be linear, took {:?}",
            start.elapsed()
        );
    }

    /// When two usable `/Type /Catalog` objects exist (incremental update
    /// replaced the catalogue), the synthetic trailer must point at the
    /// one written LAST in the file, never a stale earlier copy.
    #[test]
    fn reconstruct_synthetic_root_prefers_last_catalog() {
        let body = b"%PDF-1.7\n\
9 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]>>endobj\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n";
        // Obj 1's catalog is written after obj 9's — it is the live one.
        let table = reconstruct_xref(body).expect("repair must succeed");
        assert_eq!(
            table.trailer.get(b"Root"),
            Some(&Object::Reference((1, 0))),
            "must select the last-written /Catalog (obj 1)"
        );
    }

    /// The objects an `/ObjStm` holds are recovered as compressed entries,
    /// and a catalogue among them serves as the root.
    #[test]
    fn objstm_members_are_recovered_as_compressed_entries() {
        let mut body = b"%PDF-1.5\n".to_vec();
        body.extend_from_slice(&objstm_object(
            5,
            &[
                (1, "<</Type/Catalog/Pages 2 0 R>>"),
                (2, "<</Type/Pages/Kids[3 0 R]/Count 1>>"),
                (3, "<</Type/Page/Parent 2 0 R/MediaBox[0 0 10 10]>>"),
            ],
        ));
        let table = reconstruct_xref(&body).expect("repair must succeed");
        assert!(matches!(
            table.get(3),
            Some(XrefEntry::InObjStm {
                container: 5,
                index: 2
            })
        ));
        assert!(matches!(table.get(5), Some(XrefEntry::Direct { .. })));
        assert_eq!(table.trailer.get(b"Root"), Some(&Object::Reference((1, 0))));
    }

    /// The dictionary of a `/Type /XRef` stream is a trailer: its `/Root`
    /// and `/Info` are harvested.
    #[test]
    fn xref_stream_dictionary_is_harvested_as_trailer() {
        let mut body = b"%PDF-1.5\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n\
8 0 obj<</Producer(x)>>endobj\n"
            .to_vec();
        body.extend_from_slice(&stream_object(
            9,
            "/Type/XRef/Size 10/W[1 1 1]/Root 1 0 R/Info 8 0 R",
            b"\x00\x00\x00",
        ));
        let table = reconstruct_xref(&body).expect("repair must succeed");
        assert_eq!(table.trailer.get(b"Root"), Some(&Object::Reference((1, 0))));
        assert_eq!(table.trailer.get(b"Info"), Some(&Object::Reference((8, 0))));
    }

    /// A trailer whose `/Root` names no scanned object is not trusted; the
    /// catalogue found in the body is used instead.
    #[test]
    fn trailer_root_naming_no_object_is_replaced() {
        let body = b"%PDF-1.4\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n\
trailer\n<</Root 77 0 R/Info 8 0 R>>\n";
        let table = reconstruct_xref(body).expect("repair must succeed");
        assert_eq!(table.trailer.get(b"Root"), Some(&Object::Reference((1, 0))));
        assert_eq!(table.trailer.get(b"Info"), Some(&Object::Reference((8, 0))));
    }

    /// `N G obj` inside a stream body (an embedded file, say) or inside a
    /// string is not an object header.
    #[test]
    fn headers_inside_bodies_and_strings_are_not_objects() {
        let mut body = b"%PDF-1.4\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n\
3 0 obj(see 98 0 obj for details)endobj\n"
            .to_vec();
        body.extend_from_slice(&stream_object(
            4,
            "",
            b"%PDF-1.4\n99 0 obj<</Embedded true>>endobj\n",
        ));
        let table = reconstruct_xref(&body).expect("repair must succeed");
        assert!(table.get(98).is_none(), "header inside a string");
        assert!(table.get(99).is_none(), "header inside a stream body");
        assert!(table.get(3).is_some() && table.get(4).is_some());
    }

    /// A catalogue written after an unterminated stream is still found:
    /// the no-`endstream` memo only short-circuits stream bodies.
    #[test]
    fn catalog_after_unterminated_stream_is_found() {
        let body = b"%PDF-1.4\n\
4 0 obj<</Length 5>>stream\nabcde\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n";
        let table = reconstruct_xref(body).expect("repair must succeed");
        assert_eq!(table.trailer.get(b"Root"), Some(&Object::Reference((1, 0))));
    }

    /// An object defined directly after the container that also holds it
    /// keeps the direct definition; one defined directly before it is
    /// superseded by the compressed one.
    #[test]
    fn direct_and_compressed_definitions_follow_file_order() {
        let mut body =
            b"%PDF-1.5\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n6 0 obj(before)endobj\n"
                .to_vec();
        body.extend_from_slice(&objstm_object(5, &[(6, "(in stream)"), (7, "(in stream)")]));
        body.extend_from_slice(b"7 0 obj(after)endobj\n");
        let scan = scan_objects(&body).expect("scan");
        assert!(matches!(
            scan.entries.get(&6),
            Some(XrefEntry::InObjStm { .. })
        ));
        assert!(matches!(
            scan.entries.get(&7),
            Some(XrefEntry::Direct { .. })
        ));
    }
}
