//! Cross-reference table parser.
//!
//! Reads the xref section(s) from the end of the file, follows `/Prev` chains
//! for incremental updates, and produces a flat `XrefTable` mapping object
//! numbers to their locations.  Handles all three formats:
//!
//! * Traditional ASCII xref tables (PDF ≤ 1.4)
//! * Binary cross-reference streams (PDF 1.5+, `/Type /XRef`)
//! * Hybrid files (ASCII table with `/XRefStm` pointer)

use std::collections::HashMap;

use crate::{
    dictionary::Dictionary,
    error::PdfError,
    lexer::{is_ws, parse_u64, skip_ws},
    object::{Object, Stream, parse_object},
    repair::reconstruct_xref,
};

/// Where to find an indirect object in the file.
#[derive(Debug, Clone, Copy)]
pub(crate) enum XrefEntry {
    /// Object lives at `offset` bytes from the start of the file.
    Direct {
        offset: u64,
        /// Generation number — carried through xref repair so a synthetic
        /// trailer can reference the recovered catalogue with its true
        /// generation.
        generation: u16,
    },
    /// Object is inside an object stream; `container` is the object number of
    /// the `ObjStm`, `index` is the 0-based position within it.
    InObjStm { container: u32, index: u32 },
    /// The object number is free (ISO 32000-1 §7.5.4): the newest section
    /// that mentions it released it, so no older section's entry applies.
    Free,
}

/// Complete cross-reference table built from all xref sections in the file.
/// Later xref sections override earlier ones (incremental update semantics):
/// sections are read newest first and the first entry seen for an object
/// number — in-use or free — is the one kept.
#[derive(Debug, Default)]
pub(crate) struct XrefTable {
    pub entries: HashMap<u32, XrefEntry>,
    /// The document trailer dictionary bytes.  We keep the raw bytes and parse
    /// on demand so this module does not depend on a full object graph.
    pub trailer: Dictionary,
}

impl XrefTable {
    pub(crate) fn get(&self, id: u32) -> Option<XrefEntry> {
        self.entries.get(&id).copied()
    }
}

/// Parse all xref sections starting from the `startxref` offset at the end of
/// the file, following `/Prev` chains, for a file whose `%PDF-` header starts
/// at byte `header`.
///
/// Returns the table and the base its byte offsets are relative to.  Offsets
/// are measured from the start of the file (ISO 32000-2 §7.5.5).  When they
/// do not lead to a cross-reference chain and the header is not at byte 0,
/// they are read relative to the header: bytes prepended to a finished file
/// (a transport header, a byte-order mark) shift every position by the
/// header's offset.
pub(crate) fn read_xref(data: &[u8], header: usize) -> Result<(XrefTable, usize), PdfError> {
    let absolute = match read_xref_strict(data) {
        // A strict parse that resolved a `/Root` is authoritative — this is
        // every well-formed file, byte-unchanged.
        Ok(table) if table.trailer.contains_key(b"Root") => return Ok((table, 0)),
        other => other,
    };
    if header > 0
        && let Some(body) = data.get(header..)
        && let Ok(table) = read_xref_strict(body)
        && table.trailer.contains_key(b"Root")
    {
        return Ok((table, header));
    }
    match absolute {
        // Strict parse succeeded structurally but the trailer carries no
        // usable `/Root` (an xref section we located but could not resolve
        // to a catalogue).  Try the repair scan as a *bonus*: only adopt it
        // when it yields something better, otherwise keep the strict table
        // so existing higher-level diagnostics are unchanged.
        Ok(strict_table) => Ok((reconstruct_xref(data).unwrap_or(strict_table), 0)),
        // Strict parse failed outright (no startxref, bad offset, …).  Many
        // real-world "broken" PDFs — interrupted writes, hand-edited files,
        // naive concatenations — still contain intact object bodies and a
        // recoverable catalogue, so scan the body for object headers and
        // rebuild.  If repair also fails, surface the
        // original strict diagnostic (the more actionable of the two).
        Err(strict_err) => reconstruct_xref(data)
            .map(|table| (table, 0))
            .map_err(|_| strict_err),
    }
}

/// Read the chain of xref sections from `startxref`, newest first.  Each
/// section contributes its own entries, then those of its `/XRefStm` stream,
/// before its `/Prev` section is read (ISO 32000-1 §7.5.8.4).  The walk is a
/// loop over offsets already seen, so neither a cyclic nor a long chain
/// recurses.
fn read_xref_strict(data: &[u8]) -> Result<XrefTable, PdfError> {
    let mut table = XrefTable::default();
    let mut visited = std::collections::HashSet::new();
    let mut next = Some(find_startxref(data)?);
    while let Some(offset) = next
        && visited.insert(offset)
    {
        let trailer = read_xref_section(data, offset, &mut table)?;
        merge_trailer(&trailer, &mut table);
        if let Some(stm) = offset_entry(&trailer, b"XRefStm")
            && visited.insert(stm)
        {
            let stream_trailer = read_xref_section(data, stm, &mut table)?;
            merge_trailer(&stream_trailer, &mut table);
        }
        next = offset_entry(&trailer, b"Prev");
    }
    Ok(table)
}

/// Read the entries of the xref section at `offset` into `table` (entries
/// already present win) and return the section's trailer dictionary.
fn read_xref_section(
    data: &[u8],
    offset: u64,
    table: &mut XrefTable,
) -> Result<Dictionary, PdfError> {
    let Some(off) = usize::try_from(offset).ok().filter(|&off| off < data.len()) else {
        return Err(PdfError::BadXref(format!(
            "offset {offset} beyond end of file"
        )));
    };

    // Detect format: "xref" keyword → traditional table; anything else must be
    // an indirect object (the xref stream).
    if data[off..].starts_with(b"xref") {
        read_traditional_xref(data, off, table)
    } else {
        read_xref_stream(data, off, table)
    }
}

// ── Traditional ASCII xref table ─────────────────────────────────────────────

fn read_traditional_xref(
    data: &[u8],
    start: usize,
    table: &mut XrefTable,
) -> Result<Dictionary, PdfError> {
    let mut pos = start + 4; // skip "xref"
    skip_ws(data, &mut pos);

    // Parse subsections: each is "<first_obj> <count>" followed by entries.
    while pos < data.len() {
        // Peek ahead: if we see "trailer" we're done with entries.
        if data[pos..].starts_with(b"trailer") {
            pos += 7;
            break;
        }

        let first_raw = parse_u64(data, &mut pos).ok_or_else(|| {
            PdfError::BadXref("expected object number in subsection header".into())
        })?;
        let first = u32::try_from(first_raw).map_err(|_| {
            PdfError::BadXref(format!(
                "subsection first object {first_raw} exceeds u32::MAX"
            ))
        })?;
        skip_ws(data, &mut pos);
        let count = parse_u64(data, &mut pos)
            .ok_or_else(|| PdfError::BadXref("expected count in subsection header".into()))?;
        // Sanity cap: no real PDF has more than 10 M objects.
        let Some(count) = u32::try_from(count).ok().filter(|&c| c <= 10_000_000) else {
            return Err(PdfError::BadXref(format!(
                "xref subsection count {count} exceeds sanity limit"
            )));
        };
        // Consume the line ending after the subsection header.
        while pos < data.len() && matches!(data[pos], b'\r' | b'\n') {
            pos += 1;
        }

        for i in 0..count {
            // Each entry is exactly 20 bytes: "nnnnnnnnnn ggggg n/f\r\n"
            // We are lenient and scan for the next token rather than fixed-stride.
            skip_ws(data, &mut pos);
            let offset = parse_u64(data, &mut pos)
                .ok_or_else(|| PdfError::BadXref(format!("bad entry offset at pos {pos}")))?;
            skip_ws(data, &mut pos);
            let generation = parse_u64(data, &mut pos)
                .ok_or_else(|| PdfError::BadXref(format!("bad generation at pos {pos}")))?;
            skip_ws(data, &mut pos);
            // A missing flag (end of input) reads as free; `pos` stays within
            // `0..=data.len()`.
            let flag = data.get(pos).copied().unwrap_or(b'f');
            pos = (pos + 1).min(data.len());
            // Skip up to the end of this entry line (handles \r, \n, \r\n, space+\r etc.)
            while pos < data.len() && matches!(data[pos], b' ' | b'\r' | b'\n') {
                pos += 1;
            }

            let obj_num = first.checked_add(i).ok_or_else(|| {
                PdfError::BadXref(format!("object number overflow: first={first} i={i}"))
            })?;
            // Only insert if not already present (newer xref sections win).
            // A generation above `u16::MAX` names no object that can exist;
            // the entry is dropped rather than truncated.  A free entry is
            // kept as such so that no older section can redefine the object.
            let entry = match flag {
                b'n' => u16::try_from(generation)
                    .ok()
                    .map(|generation| XrefEntry::Direct { offset, generation }),
                b'f' => Some(XrefEntry::Free),
                _ => None,
            };
            if let Some(entry) = entry {
                let _ = table.entries.entry(obj_num).or_insert(entry);
            }
        }
    }

    // Parse the trailer dictionary.
    skip_ws(data, &mut pos);
    parse_trailer_dict(data, &mut pos)
}

fn parse_trailer_dict(data: &[u8], pos: &mut usize) -> Result<Dictionary, PdfError> {
    skip_ws(data, pos);
    // The trailer dict starts with "<<".
    if !data[*pos..].starts_with(b"<<") {
        return Err(PdfError::BadXref(format!(
            "expected '<<' for trailer dict at pos {pos}"
        )));
    }

    let obj = parse_object(data, pos)
        .ok_or_else(|| PdfError::BadXref("could not parse trailer dictionary".into()))?;
    match obj {
        Object::Dictionary(d) => Ok(d),
        other => Err(PdfError::BadXref(format!(
            "trailer is not a dict: {other:?}"
        ))),
    }
}

/// Merge `dict` into `table.trailer`; existing keys win, as newer sections
/// are merged first.
fn merge_trailer(dict: &Dictionary, table: &mut XrefTable) {
    for (k, v) in dict {
        if !table.trailer.contains_key(k) {
            let _ = table.trailer.insert(k.clone(), v.clone());
        }
    }
}

/// The byte offset a trailer's `/Prev` or `/XRefStm` entry names.  A missing
/// entry, a non-integer, zero or a negative value all mean "no such section".
fn offset_entry(dict: &Dictionary, key: &[u8]) -> Option<u64> {
    match dict.get(key) {
        Some(Object::Integer(n)) => u64::try_from(*n).ok().filter(|&off| off > 0),
        _ => None,
    }
}

// ── Cross-reference stream (PDF 1.5+) ────────────────────────────────────────

fn read_xref_stream(
    data: &[u8],
    start: usize,
    table: &mut XrefTable,
) -> Result<Dictionary, PdfError> {
    // Parse the indirect object header: "<id> <gen> obj"
    let mut pos = start;
    skip_ws(data, &mut pos);
    let _ = parse_u64(data, &mut pos); // object number — discard
    skip_ws(data, &mut pos);
    let _ = parse_u64(data, &mut pos); // generation — discard
    skip_ws(data, &mut pos);
    if !data[pos..].starts_with(b"obj") {
        return Err(PdfError::BadXref(format!(
            "expected 'obj' keyword at offset {start}"
        )));
    }
    pos += 3;
    skip_ws(data, &mut pos);

    // parse_object returns Object::Stream when it finds the "stream" keyword.
    let dict_obj = parse_object(data, &mut pos)
        .ok_or_else(|| PdfError::BadXref("could not parse xref stream dict".into()))?;
    let Object::Stream(Stream {
        dict,
        content: raw_stream,
    }) = dict_obj
    else {
        return Err(PdfError::BadXref(
            "xref stream object is not a Stream".into(),
        ));
    };
    // Decode the stream (almost always `FlateDecode`).
    let stream_bytes = decode_xref_stream(&raw_stream, &dict)?;

    // Read /W field widths.
    let Some(Object::Array(w)) = dict.get(b"W") else {
        return Err(PdfError::BadXref("xref stream: missing /W array".into()));
    };
    if w.len() < 3 {
        return Err(PdfError::BadXref("xref stream: /W needs 3 entries".into()));
    }
    let (w0, w1, w2) = (
        xref_field_width(&w[0])?,
        xref_field_width(&w[1])?,
        xref_field_width(&w[2])?,
    );
    // Each width is at most `MAX_XREF_FIELD_WIDTH`, so the sum cannot overflow.
    let entry_len = w0 + w1 + w2;
    if entry_len == 0 {
        return Err(PdfError::BadXref("xref stream: zero-width entry".into()));
    }

    let index_pairs = xref_stream_index(&dict)?;

    // Decode binary entries.
    let mut byte_pos = 0usize;
    let mut pair_idx = 0;
    while pair_idx + 1 < index_pairs.len() {
        let first_obj = index_pairs[pair_idx];
        let count = index_pairs[pair_idx + 1];
        pair_idx += 2;

        // Sanity cap to defeat malformed PDFs claiming billions of entries.
        if count > 10_000_000 {
            return Err(PdfError::BadXref(format!(
                "xref stream subsection count {count} exceeds sanity limit"
            )));
        }

        for i in 0..count {
            if byte_pos + entry_len > stream_bytes.len() {
                break;
            }

            let type_val = if w0 == 0 {
                1 // default type is 1 per spec
            } else {
                read_be_uint(&stream_bytes[byte_pos..byte_pos + w0])
            };
            let f1 = read_be_uint(&stream_bytes[byte_pos + w0..byte_pos + w0 + w1]);
            let f2 = if w2 == 0 {
                0
            } else {
                read_be_uint(&stream_bytes[byte_pos + w0 + w1..byte_pos + w0 + w1 + w2])
            };
            byte_pos += entry_len;

            let Some(obj_num) = first_obj.checked_add(i) else {
                break; // object number overflow — stop processing this subsection
            };
            if table.entries.contains_key(&obj_num) {
                continue; // newer xref section already has this
            }

            // A field value outside its target type names no object that
            // can exist; such an entry is dropped rather than truncated onto
            // a different object.
            let entry = match type_val {
                0 => Some(XrefEntry::Free),
                1 => u16::try_from(f2).ok().map(|generation| XrefEntry::Direct {
                    offset: f1,
                    generation,
                }),
                2 => u32::try_from(f1)
                    .ok()
                    .zip(u32::try_from(f2).ok())
                    .map(|(container, index)| XrefEntry::InObjStm { container, index }),
                // Other types are ignored per spec.
                _ => None,
            };
            if let Some(entry) = entry {
                let _ = table.entries.insert(obj_num, entry);
            }
        }
    }

    Ok(dict)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// The `/Index` array of a cross-reference stream as flat
/// `[first, count, first, count, …]` pairs; absent, it defaults to one
/// subsection `[0 /Size]`.
fn xref_stream_index(dict: &Dictionary) -> Result<Vec<u32>, PdfError> {
    let index_pairs: Vec<u32> = if let Some(Object::Array(a)) = dict.get(b"Index") {
        a.iter()
            .map(|o| {
                o.as_u32().ok_or_else(|| {
                    PdfError::BadXref(format!("xref stream /Index: bad integer {o:?}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let size = dict
            .get(b"Size")
            .ok_or_else(|| PdfError::BadXref("xref stream: missing /Size".into()))?;
        let size_u32 = size
            .as_u32()
            .ok_or_else(|| PdfError::BadXref(format!("xref stream /Size: bad integer {size:?}")))?;
        vec![0, size_u32]
    };
    if !index_pairs.len().is_multiple_of(2) {
        return Err(PdfError::BadXref(
            "xref stream: /Index must have even number of entries".into(),
        ));
    }
    Ok(index_pairs)
}

/// Widest `/W` field accepted, in bytes: every field value is read into a
/// `u64`.
const MAX_XREF_FIELD_WIDTH: usize = 8;

/// Validate one `/W` entry of a cross-reference stream: a non-negative
/// integer no wider than [`MAX_XREF_FIELD_WIDTH`].  A non-integer entry is
/// read as width 0 (the field is absent).
fn xref_field_width(obj: &Object) -> Result<usize, PdfError> {
    let Object::Integer(n) = obj else {
        return Ok(0);
    };
    usize::try_from(*n)
        .ok()
        .filter(|&w| w <= MAX_XREF_FIELD_WIDTH)
        .ok_or_else(|| {
            PdfError::BadXref(format!(
                "xref stream: /W width {n} outside 0..={MAX_XREF_FIELD_WIDTH}"
            ))
        })
}

/// Decode an xref stream's raw bytes.  Almost always `FlateDecode`; may be
/// uncompressed (no /Filter key).
fn decode_xref_stream(raw: &[u8], dict: &Dictionary) -> Result<Vec<u8>, PdfError> {
    use crate::stream::apply_flate;

    // /Filter is optional, a Name, or a single-element array of a Name.
    let filter: Option<&[u8]> = match dict.get(b"Filter") {
        Some(Object::Name(n)) => Some(n),
        Some(Object::Array(a)) if a.len() == 1 => a[0].as_name(),
        _ => None,
    };

    match filter {
        None => Ok(raw.to_vec()),
        Some(b"FlateDecode") => {
            let params = dict
                .get(b"DecodeParms")
                .map(|o| o as &dyn crate::stream::DictLookup);
            apply_flate(raw, params).map_err(|e| PdfError::DecodeFailed(e.to_string()))
        }
        Some(other) => Err(PdfError::DecodeFailed(format!(
            "unsupported xref stream filter: {}",
            String::from_utf8_lossy(other)
        ))),
    }
}

/// Read a big-endian unsigned integer from `bytes` (1–8 bytes).
fn read_be_uint(bytes: &[u8]) -> u64 {
    let mut val = 0u64;
    for &b in bytes {
        val = (val << 8) | u64::from(b);
    }
    val
}

/// Find the byte offset stored after the last `startxref` keyword in the file.
/// Scans the final 2048 bytes to handle files with trailing garbage.
fn find_startxref(data: &[u8]) -> Result<u64, PdfError> {
    let scan_start = data.len().saturating_sub(2048);
    let tail = &data[scan_start..];

    // Find the last occurrence of "startxref".
    let needle = b"startxref";
    let rel_pos = tail
        .windows(needle.len())
        .rposition(|w| w == needle)
        .ok_or_else(|| PdfError::BadXref("'startxref' not found in last 2048 bytes".into()))?;

    let mut pos = scan_start + rel_pos + needle.len();
    // Skip whitespace (should be a single newline).
    while pos < data.len() && is_ws(data[pos]) {
        pos += 1;
    }
    parse_u64(data, &mut pos)
        .ok_or_else(|| PdfError::BadXref("no integer after 'startxref'".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_startxref_basic() {
        let data = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000009 65535 f\r\ntrailer\n<<>>\nstartxref\n9\n%%EOF";
        let off = find_startxref(data).unwrap();
        assert_eq!(off, 9);
    }

    #[test]
    fn read_be_uint_cases() {
        assert_eq!(read_be_uint(&[0x00, 0x01, 0x23]), 0x0123);
        assert_eq!(read_be_uint(&[0xFF]), 0xFF);
        assert_eq!(read_be_uint(&[]), 0);
    }

    #[test]
    fn xref_subsection_count_capped() {
        // count = 99_999_999 exceeds the 10 M sanity limit and must error,
        // not allocate ~2 GB of HashMap entries.
        let data = b"\
%PDF-1.4
xref
0 99999999
0000000000 65535 f\r
trailer
<<>>
startxref
9
%%EOF";
        let err = read_xref(data, 0).unwrap_err();
        match err {
            PdfError::BadXref(s) => assert!(s.contains("sanity limit"), "got: {s}"),
            other => panic!("expected BadXref, got {other:?}"),
        }
    }

    #[test]
    fn xref_negative_prev_does_not_recurse() {
        // /Prev = -1 used to be cast as u64 = 0xFFFF_FFFF_FFFF_FFFF and trigger
        // a "beyond end of file" error from a bogus offset; now it is skipped.
        let data = b"\
%PDF-1.4
xref
0 1
0000000000 65535 f\r
trailer
<</Size 1 /Prev -1>>
startxref
9
%%EOF";
        // Should succeed: /Prev=-1 is treated as no previous xref.
        let (table, _) = read_xref(data, 0).expect("xref parse");
        assert!(matches!(table.get(0), Some(XrefEntry::Free)));
        assert_eq!(table.entries.len(), 1);
    }

    /// Build a file whose only xref section is an uncompressed cross-reference
    /// stream with the given `/W` and `/Index` source text and entry bytes.
    fn xref_stream_pdf(w: &str, index: &str, entries: &[u8]) -> Vec<u8> {
        let mut data = b"%PDF-1.5\n".to_vec();
        let xref_off = data.len();
        data.extend_from_slice(
            format!(
                "9 0 obj\n<</Type/XRef/Size 10/W{w}/Index{index}/Length {}>>\nstream\n",
                entries.len()
            )
            .as_bytes(),
        );
        data.extend_from_slice(entries);
        data.extend_from_slice(
            format!("\nendstream\nendobj\nstartxref\n{xref_off}\n%%EOF").as_bytes(),
        );
        data
    }

    /// A negative `/W` width must be rejected; casting it to `usize` made
    /// the field offsets wrap and the entry slicing panic.
    #[test]
    fn xref_stream_negative_width_is_rejected() {
        let data = xref_stream_pdf("[-1 1 1]", "[0 1]", &[1, 0, 0]);
        let err = read_xref_strict(&data).unwrap_err();
        assert!(
            matches!(&err, PdfError::BadXref(s) if s.contains("/W")),
            "got {err:?}"
        );
        // The public entry point must not panic either.
        let _ = read_xref(&data, 0);
    }

    /// Widths whose sum overflows `usize`, or any single width wider than a
    /// `u64`, must be rejected rather than wrapping the entry length.
    #[test]
    fn xref_stream_oversized_width_is_rejected() {
        for w in ["[9223372036854775807 9223372036854775807 2]", "[1 9 1]"] {
            let data = xref_stream_pdf(w, "[0 1]", &[1, 0, 0]);
            let err = read_xref_strict(&data).unwrap_err();
            assert!(
                matches!(&err, PdfError::BadXref(s) if s.contains("/W")),
                "/W {w}: got {err:?}"
            );
        }
    }

    /// Entry fields wider than their target type must not be truncated: a
    /// type-2 container number of `2^32 + 4` used to alias object 4, and a
    /// type-1 generation of `65536 + 7` used to alias generation 7.
    #[test]
    fn xref_stream_out_of_range_fields_are_dropped() {
        // `/W [1 8 5]`: one type byte, an 8-byte field 1, a 5-byte field 2.
        let mut entries = Vec::new();
        let mut entry = |kind: u8, f1: u64, f2: u64| {
            entries.push(kind);
            entries.extend_from_slice(&f1.to_be_bytes());
            entries.extend_from_slice(&f2.to_be_bytes()[3..]);
        };
        entry(2, (1 << 32) + 4, 0); // object 1
        entry(2, 4, (1 << 32) + 1); // object 2
        entry(1, 9, 65_536 + 7); // object 3
        entry(1, 9, 0); // object 4: well-formed control

        let data = xref_stream_pdf("[1 8 5]", "[1 4]", &entries);
        let table = read_xref_strict(&data).expect("xref stream parses");
        assert!(table.get(1).is_none(), "container 2^32+4 must not alias 4");
        assert!(table.get(2).is_none(), "index 2^32+1 must not alias 1");
        assert!(table.get(3).is_none(), "generation 65543 must not alias 7");
        assert!(
            matches!(
                table.get(4),
                Some(XrefEntry::Direct {
                    offset: 9,
                    generation: 0
                })
            ),
            "well-formed entry must survive"
        );
    }

    /// A classic xref section `xref\n<first> <count>\n<entries>` followed by
    /// `trailer <dict>`, as one byte string.
    fn classic_section(first: u32, entries: &[(u64, u16, u8)], trailer: &str) -> Vec<u8> {
        let mut s = format!("xref\n{first} {}\n", entries.len());
        for (offset, generation, flag) in entries {
            s.push_str(&format!(
                "{offset:010} {generation:05} {} \n",
                *flag as char
            ));
        }
        s.push_str(&format!("trailer\n{trailer}\n"));
        s.into_bytes()
    }

    /// Append `section` to `data` and return its offset.
    fn append(data: &mut Vec<u8>, section: &[u8]) -> usize {
        let off = data.len();
        data.extend_from_slice(section);
        off
    }

    /// A hybrid-reference file: object 7 is defined at `old`, then
    /// redefined at `new` by an update whose `/XRefStm` stream carries the
    /// new entry while its `/Prev` section carries the old one.  Returns the
    /// file and the two header offsets.
    fn hybrid_file() -> (Vec<u8>, u64, u64) {
        let mut data = b"%PDF-1.7\n".to_vec();
        let old = append(&mut data, b"7 0 obj\n(old)\nendobj\n");
        let xref1 = append(
            &mut data,
            &classic_section(7, &[(old as u64, 0, b'n')], "<</Size 8/Root 1 0 R>>"),
        );
        let new = append(&mut data, b"7 0 obj\n(new)\nendobj\n");
        let mut stream =
            b"9 0 obj\n<</Type/XRef/Size 10/W[1 4 1]/Index[7 1]/Length 6>>\nstream\n\x01".to_vec();
        stream.extend_from_slice(&(new as u32).to_be_bytes());
        stream.extend_from_slice(b"\x00\nendstream\nendobj\n");
        let stm = append(&mut data, &stream);
        let xref2 = append(
            &mut data,
            &classic_section(
                9,
                &[(stm as u64, 0, b'n')],
                &format!("<</Size 10/Root 1 0 R/Prev {xref1}/XRefStm {stm}>>"),
            ),
        );
        data.extend_from_slice(format!("startxref\n{xref2}\n%%EOF").as_bytes());
        (data, old as u64, new as u64)
    }

    /// In a hybrid file the `/XRefStm` stream is consulted before the
    /// `/Prev` chain (ISO 32000-1 §7.5.8.4), so its entry for a redefined
    /// object shadows the older section's.
    #[test]
    fn hybrid_xrefstm_entries_shadow_prev_section() {
        let (data, old, new) = hybrid_file();
        let table = read_xref_strict(&data).expect("hybrid file parses");
        let Some(XrefEntry::Direct { offset, .. }) = table.get(7) else {
            panic!("object 7 must be a direct entry, got {:?}", table.get(7));
        };
        assert_eq!(
            offset, new,
            "the /XRefStm entry wins over /Prev (old at {old})"
        );
    }

    /// A newer section's free entry is recorded, so the older section's
    /// in-use entry for the same object cannot resurrect it.
    #[test]
    fn free_entry_shadows_older_definition() {
        let mut data = b"%PDF-1.7\n".to_vec();
        let body = append(&mut data, b"4 0 obj\n(body)\nendobj\n");
        let xref1 = append(
            &mut data,
            &classic_section(4, &[(body as u64, 0, b'n')], "<</Size 5/Root 1 0 R>>"),
        );
        let xref2 = append(
            &mut data,
            &classic_section(
                4,
                &[(0, 1, b'f')],
                &format!("<</Size 5/Root 1 0 R/Prev {xref1}>>"),
            ),
        );
        data.extend_from_slice(format!("startxref\n{xref2}\n%%EOF").as_bytes());
        let table = read_xref_strict(&data).expect("file parses");
        assert!(
            matches!(table.get(4), Some(XrefEntry::Free)),
            "object 4 must be free, got {:?}",
            table.get(4)
        );
    }

    /// A type-0 entry in a newer cross-reference stream frees the object
    /// against an older section's in-use entry.
    #[test]
    fn xref_stream_free_entry_shadows_older_definition() {
        let mut data = b"%PDF-1.5\n".to_vec();
        let body = append(&mut data, b"4 0 obj\n(body)\nendobj\n");
        let xref1 = append(
            &mut data,
            &classic_section(4, &[(body as u64, 0, b'n')], "<</Size 5/Root 1 0 R>>"),
        );
        let stm = append(
            &mut data,
            format!(
                "9 0 obj\n<</Type/XRef/Size 10/W[1 1 1]/Index[4 1]/Prev {xref1}/Root 1 0 R\
                 /Length 3>>\nstream\n\x00\x00\x00\nendstream\nendobj\n"
            )
            .as_bytes(),
        );
        data.extend_from_slice(format!("startxref\n{stm}\n%%EOF").as_bytes());
        let table = read_xref_strict(&data).expect("file parses");
        assert!(
            matches!(table.get(4), Some(XrefEntry::Free)),
            "object 4 must be free, got {:?}",
            table.get(4)
        );
    }

    /// A `/Prev` chain of any length is walked without recursion: a file of
    /// tens of thousands of sections parses rather than exhausting the stack.
    #[test]
    fn long_prev_chain_is_walked_iteratively() {
        let mut data = b"%PDF-1.7\n".to_vec();
        let mut prev: Option<usize> = None;
        for _ in 0..50_000 {
            let trailer = prev.map_or_else(
                || "<</Size 1>>".to_owned(),
                |p| format!("<</Size 1/Prev {p}>>"),
            );
            prev = Some(append(&mut data, &classic_section(0, &[], &trailer)));
        }
        data.extend_from_slice(format!("startxref\n{}\n%%EOF", prev.unwrap()).as_bytes());
        let table = read_xref_strict(&data).expect("chain parses");
        assert!(table.entries.is_empty());
    }

    /// A traditional entry whose generation exceeds `u16::MAX` is dropped
    /// instead of being truncated onto another generation.
    #[test]
    fn traditional_xref_out_of_range_generation_is_dropped() {
        let data = b"%PDF-1.4\n\
xref\n\
0 2\n\
0000000009 65543 n\r\n\
0000000009 00000 n\r\n\
trailer\n\
<</Size 2>>\n\
startxref\n\
9\n\
%%EOF";
        let table = read_xref_strict(data).expect("xref parses");
        assert!(table.get(0).is_none(), "generation 65543 must be dropped");
        assert!(table.get(1).is_some(), "well-formed entry must survive");
    }

    /// A traditional table that ends immediately after an entry's
    /// generation number (no `n`/`f` flag, no trailer) must yield an error;
    /// the flag skip used to step the cursor one byte past the end of the
    /// buffer and the trailer lookup then sliced out of range.
    #[test]
    fn traditional_xref_truncated_after_generation_does_not_panic() {
        let mut data = b"%PDF-1.4\nstartxref\n".to_vec();
        let tail = b"\n%%EOF\n";
        // The table follows the startxref pointer; compute its offset.
        let digits_len = 2;
        let xref_off = data.len() + digits_len + tail.len();
        data.extend_from_slice(format!("{xref_off}").as_bytes());
        assert_eq!(data.len() + tail.len(), xref_off);
        data.extend_from_slice(tail);
        data.extend_from_slice(b"xref\n0 1\n0000000000 65535");
        assert!(read_xref_strict(&data).is_err());
        let _ = read_xref(&data, 0);
    }
}
