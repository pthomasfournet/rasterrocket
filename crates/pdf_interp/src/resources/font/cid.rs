//! Type 0 composite fonts: the `Encoding` `CMap`, the descendant
//! `CIDFont`'s program, `CIDToGIDMap` and per-CID metrics (PDF §9.7).

use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use super::{
    BaseEncoding, FontDescriptor, PdfFontKind, classify_kind, embedded_program, empty_differences,
    number, number_array, width_units,
};
use crate::resources::cmap::{CMap, DecodedCode, parse_cmap};
use crate::resources::cmap_predefined::predefined_cmap;
use crate::resources::dict_ext::DictExt;
use crate::resources::font_substitute::{Script, ordering_script, substitute_style};

/// CID encoding information for Type 0 composite fonts.
///
/// Type 0 fonts encode character codes as 1–4 byte sequences.  The `Encoding`
/// `CMap` maps byte sequences to CIDs; the `CIDToGIDMap` (or identity) then maps
/// CIDs to glyph indices.
#[derive(Debug)]
pub struct CidEncoding {
    /// The `Encoding` `CMap`: splits strings into codes and maps each to a
    /// CID.  The identity `CMap` for `Identity-H` / `Identity-V` and for an
    /// `Encoding` that cannot be resolved.
    pub encoding_cmap: Arc<CMap>,

    /// Precomputed CID → GID lookup from `CIDToGIDMap`.
    ///
    /// `None` means Identity (CID == GID, the common case for OpenType-CFF and
    /// TrueType `CIDFonts`).
    pub cid_to_gid: Option<Vec<u32>>,

    /// Default advance width in thousandths of a text-space unit (`DW` key).
    pub default_width: i32,

    /// Explicit per-CID advance widths from the `W` array.
    pub widths: CidMetrics<i32>,

    /// Vertical writing mode: the `Encoding` `CMap` has `WMode` 1.
    pub vertical: bool,

    /// `DW2`: the default position vector's y component and the default
    /// vertical advance, in thousandths of a text-space unit (PDF §9.7.4.3;
    /// default `[880 -1000]`).
    pub default_vertical: [i32; 2],

    /// `W2`: explicit per-CID vertical metrics `[w1y, vx, vy]`.
    pub vertical_metrics: CidMetrics<[i32; 3]>,

    /// The font's `ToUnicode` `CMap` (code → Unicode), read for a font
    /// without a program: a substitute selects its glyphs by Unicode value,
    /// and the collection's CID → Unicode table is the only other source.
    pub to_unicode: Option<Arc<CMap>>,
}

/// Per-CID values of a `W` or `W2` array: `(first_cid, values)` segments,
/// the value of CID `first_cid + k` at index `k`.  Where segments overlap,
/// the earlier one in the array applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CidMetrics<T> {
    segments: Vec<(u32, Vec<T>)>,
    /// The segments ascend by first CID without overlapping, so a lookup
    /// bisects instead of scanning.
    sorted: bool,
}

impl<T> CidMetrics<T> {
    /// The metrics of `segments`, in array order.
    #[must_use]
    pub fn new(segments: Vec<(u32, Vec<T>)>) -> Self {
        let sorted = segments.windows(2).all(|pair| {
            let [(first, values), (next, _)] = pair else {
                return true;
            };
            u64::from(*first) + values.len() as u64 <= u64::from(*next)
        });
        Self { segments, sorted }
    }

    /// The value of `cid`, `None` when no segment covers it.
    #[must_use]
    pub fn get(&self, cid: u32) -> Option<&T> {
        if self.sorted {
            let after = self.segments.partition_point(|(first, _)| *first <= cid);
            Self::value_in(self.segments.get(after.checked_sub(1)?)?, cid)
        } else {
            self.segments
                .iter()
                .find_map(|segment| Self::value_in(segment, cid))
        }
    }

    /// The value of `cid` in the segment `(first, values)`, if it covers
    /// `cid`.
    fn value_in((first, values): &(u32, Vec<T>), cid: u32) -> Option<&T> {
        values.get(usize::try_from(cid.checked_sub(*first)?).ok()?)
    }

    /// The `(first_cid, values)` segments, in array order.
    #[must_use]
    pub fn segments(&self) -> &[(u32, Vec<T>)] {
        &self.segments
    }

    /// Each value mapped through `f`.
    #[must_use]
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> CidMetrics<U> {
        CidMetrics {
            segments: self
                .segments
                .into_iter()
                .map(|(first, values)| (first, values.into_iter().map(&mut f).collect()))
                .collect(),
            sorted: self.sorted,
        }
    }
}

/// The vertical metrics of one CID in thousandths of a text-space unit
/// (PDF §9.7.4.3): the vertical advance `w1y` and the position vector
/// `(vx, vy)` from the horizontal origin to the vertical origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerticalMetrics {
    /// Vertical advance (negative: downwards).
    pub w1y: i32,
    /// Position vector x component.
    pub vx: i32,
    /// Position vector y component.
    pub vy: i32,
}

impl CidEncoding {
    /// The vertical metrics of `cid`: its `W2` entry, else the defaults
    /// — `DW2`'s advance, half the horizontal width for `vx`, `DW2`'s
    /// `vy`.
    #[must_use]
    pub fn vertical_metrics_for_cid(&self, cid: u32) -> VerticalMetrics {
        if let Some(&[w1y, vx, vy]) = self.vertical_metrics.get(cid) {
            return VerticalMetrics { w1y, vx, vy };
        }
        VerticalMetrics {
            w1y: self.default_vertical[1],
            vx: self.width_for_cid(cid) / 2,
            vy: self.default_vertical[0],
        }
    }

    /// Return the advance width for `cid` in thousandths of a text-space unit.
    ///
    /// Searches the explicit `W` table first; falls back to `default_width`.
    #[must_use]
    pub fn width_for_cid(&self, cid: u32) -> i32 {
        self.widths.get(cid).copied().unwrap_or(self.default_width)
    }

    /// Decode `bytes` into character codes through the `Encoding` `CMap`,
    /// each with its CID and byte length.
    pub fn codes<'a>(&'a self, bytes: &'a [u8]) -> impl Iterator<Item = DecodedCode> + 'a {
        self.encoding_cmap.iter_codes(bytes)
    }

    /// The glyph index `cid` selects through `cid_to_gid`; identity without
    /// a table, 0 (`.notdef`) for a CID past the table's end.
    #[must_use]
    pub fn gid_for_cid(&self, cid: u32) -> u32 {
        // CIDs are 16-bit values (PDF spec §9.7.4); usize is always ≥ 16 bits.
        self.cid_to_gid
            .as_ref()
            .map_or(cid, |table| table.get(cid as usize).copied().unwrap_or(0))
    }
}

/// Resolve a `Type0` composite font dictionary into a [`FontDescriptor`].
///
/// # Structure
///
/// ```text
/// Type0 dict
///   Encoding: /Identity-H | /Identity-V | stream (CMap)
///   DescendantFonts: [ref]          ← array with exactly one entry
///     Subtype: CIDFontType0 | CIDFontType2
///     FontDescriptor: ref           ← embedded font bytes live here
///     DW: integer                   ← default advance width
///     W: array                      ← per-CID widths
///     CIDToGIDMap: /Identity | ref  ← CID → GID (CIDFontType2 only)
///   ToUnicode: stream               ← optional; read for a substitute
/// ```
pub(super) fn resolve_type0_font(doc: &Document, dict: &Dictionary) -> FontDescriptor {
    // 1. Encoding CMap: maps char codes to CIDs.
    let encoding_cmap = extract_type0_encoding_cmap(doc, dict);

    // 2. DescendantFonts: the actual CIDFont dict with bytes and widths.
    let descendant = extract_descendant(doc, dict);

    // 3. The program embedded in the CIDFont's FontDescriptor.
    let program = descendant.as_ref().and_then(|d| embedded_program(doc, d));

    // 4. CIDToGIDMap (CIDFontType2 only; CIDFontType0/CFF uses identity).
    let cid_to_gid = descendant.as_ref().and_then(|d| extract_cid_to_gid(doc, d));

    // 5. CIDFont metrics: DW and W, and DW2 and W2 for vertical writing.
    let (default_width, widths) = descendant.as_ref().map_or_else(
        || (1000, CidMetrics::new(Vec::new())),
        |d| extract_cid_widths(doc, d),
    );
    let (default_vertical, vertical_metrics) = descendant.as_ref().map_or_else(
        || ([880, -1000], CidMetrics::new(Vec::new())),
        |d| extract_cid_vertical_metrics(doc, d),
    );

    // 6. Determine font kind from CIDFont Subtype.
    let kind = descendant
        .as_ref()
        .map_or(PdfFontKind::Other, classify_kind);

    // 7. The `ToUnicode` map a substitute draws through.
    let to_unicode = if program.is_none() {
        dict.get(b"ToUnicode")
            .and_then(|obj| embedded_cmap(doc, obj, 0))
            .map(Arc::new)
    } else {
        None
    };

    // 8. The script of a substitute: the CIDFont's ordering, else the
    // CMap's, else the family name.
    let mut substitute = substitute_style(doc, dict, descendant.as_ref());
    if let Some(script) = descendant
        .as_ref()
        .and_then(|d| ordering_script(doc, d))
        .or_else(|| {
            encoding_cmap
                .ordering
                .as_deref()
                .and_then(|o| Script::from_ordering(o.as_bytes()))
        })
    {
        substitute.script = Some(script);
    }

    FontDescriptor {
        kind,
        program,
        widths: vec![],
        first_char: 0,
        // Type 0 fonts use `CidEncoding::width_for_cid` exclusively;
        // `missing_width` is never consulted but must be initialised.
        missing_width: 0,
        differences: empty_differences(),
        // Unused for Type 0 (CID path bypasses name resolution entirely).
        base_encoding: BaseEncoding::Standard,
        cid_encoding: Some(CidEncoding {
            vertical: encoding_cmap.vertical,
            encoding_cmap,
            cid_to_gid,
            default_width,
            widths,
            default_vertical,
            vertical_metrics,
            to_unicode,
        }),
        type3: None,
        substitute,
    }
}

/// Greatest `UseCMap` nesting followed through embedded `CMap` streams.
const MAX_USECMAP_DEPTH: u32 = 4;

/// Resolve the `Encoding` entry of a Type 0 font to its [`CMap`]: a
/// predefined `CMap` by name, or an embedded stream.  The identity `CMap`
/// when the entry is absent, names an unknown `CMap`, or holds a stream
/// that does not parse.
fn extract_type0_encoding_cmap(doc: &Document, dict: &Dictionary) -> Arc<CMap> {
    let Some(enc) = dict.get(b"Encoding") else {
        log::warn!("font: Type0 font has no Encoding — treating as Identity-H");
        return Arc::new(CMap::identity());
    };
    match enc {
        Object::Name(n) => named_cmap(n),
        Object::Reference(_) | Object::Stream(_) => embedded_cmap(doc, enc, 0).map_or_else(
            || {
                log::warn!("font: Type0 Encoding CMap stream could not be parsed");
                Arc::new(CMap::identity())
            },
            Arc::new,
        ),
        _ => {
            log::warn!("font: Type0 Encoding is not a Name or stream — treating as identity");
            Arc::new(CMap::identity())
        }
    }
}

/// The `CMap` a name denotes: identity for `Identity-H` / `Identity-V`,
/// else the predefined `CMap` of that name, else identity with a warning.
fn named_cmap(name: &[u8]) -> Arc<CMap> {
    if name == b"Identity-H" {
        return Arc::new(CMap::identity());
    }
    if name == b"Identity-V" {
        return Arc::new(CMap::identity_vertical());
    }
    predefined_cmap(name).unwrap_or_else(|| {
        log::warn!(
            "font: unknown predefined CMap /{} — treating as identity",
            String::from_utf8_lossy(name)
        );
        Arc::new(CMap::identity())
    })
}

/// Parse an embedded `CMap` stream (`obj` is the stream or a reference to
/// it) and apply its dictionary's `WMode` and `UseCMap`, a name or
/// another stream, at nesting `depth`.
fn embedded_cmap(doc: &Document, obj: &Object, depth: u32) -> Option<CMap> {
    let resolved = doc.resolve(obj).ok()?;
    let stream = resolved.as_stream()?;
    let content = stream.decompressed_content().ok()?;
    let mut cmap = parse_cmap(&content)?;
    if let Some(mode) = stream.dict.get_i64(b"WMode") {
        cmap.vertical = mode == 1;
    }
    match stream.dict.get(b"UseCMap") {
        Some(Object::Name(n)) => cmap.inherit(&named_cmap(n)),
        Some(parent @ (Object::Reference(_) | Object::Stream(_))) if depth < MAX_USECMAP_DEPTH => {
            if let Some(parent) = embedded_cmap(doc, parent, depth + 1) {
                cmap.inherit(&parent);
            }
        }
        _ => {}
    }
    Some(cmap)
}

/// Resolve the first entry of `DescendantFonts` to a dictionary.
///
/// The PDF spec requires exactly one descendant; we take the first and warn if
/// the array is absent or empty.
fn extract_descendant(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    let Some(arr_obj) = dict.get(b"DescendantFonts") else {
        log::warn!("font: Type0 font has no DescendantFonts — cannot load face");
        return None;
    };
    match arr_obj {
        Object::Array(a) => {
            if a.is_empty() {
                log::warn!("font: Type0 DescendantFonts array is empty");
                return None;
            }
            crate::resources::resolve_dict(doc, &a[0])
        }
        Object::Reference(id) => {
            let obj = doc.get_object(*id).ok()?;
            let arr = obj.as_array()?;
            if arr.is_empty() {
                log::warn!("font: Type0 DescendantFonts array is empty");
                return None;
            }
            crate::resources::resolve_dict(doc, &arr[0])
        }
        _ => None,
    }
}

/// Parse the `CIDToGIDMap` stream from a `CIDFontType2` (TrueType) descendant.
///
/// The map is a byte array of 2-byte big-endian GIDs indexed by CID.
/// `/Identity` (or absent) means CID == GID; we represent that as `None`.
const MAX_CID_TO_GID_BYTES: usize = 65536 * 2;
fn extract_cid_to_gid(doc: &Document, descendant: &Dictionary) -> Option<Vec<u32>> {
    let obj = descendant.get(b"CIDToGIDMap")?;
    match obj {
        Object::Reference(id) => {
            let stream_obj = doc.get_object(*id).ok()?;
            let stream = stream_obj.as_stream()?;
            let bytes = stream.decompressed_content().ok()?;

            // A CIDToGIDMap covers CIDs 0..65535 at most → max 65536 × 2 = 131072 bytes.
            // Reject oversized maps to prevent memory exhaustion from malicious PDFs.
            if bytes.len() > MAX_CID_TO_GID_BYTES {
                log::warn!(
                    "font: CIDToGIDMap stream is {} bytes — exceeds {MAX_CID_TO_GID_BYTES} limit, ignoring",
                    bytes.len()
                );
                return None;
            }

            // CIDToGIDMap is pairs of bytes: GID[cid] = (bytes[2*cid] << 8) | bytes[2*cid+1].
            if bytes.len() % 2 != 0 {
                log::warn!(
                    "font: CIDToGIDMap stream length {} is odd — ignoring",
                    bytes.len()
                );
                return None;
            }
            let table: Vec<u32> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u32::from(pair[0]) << 8 | u32::from(pair[1]))
                .collect();
            Some(table)
        }
        _ => None,
    }
}

/// Greatest number of per-CID values materialised from one `W` or `W2`
/// array.  The range form `cfirst clast v…` expands to `clast − cfirst + 1`
/// values, so a few bytes of input could otherwise request gigabytes; CIDs
/// are 16-bit (PDF §9.7.4.3), so no font needs more than this many.
const MAX_CID_METRIC_ENTRIES: usize = 0x1_0000;

/// The CID a `W` / `W2` element denotes: a non-negative integer within
/// `u32`.
fn cid_number(doc: &Document, obj: &Object) -> Option<u32> {
    let n = number(doc, obj)?;
    (n.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&n)).then(|| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "n is an integer within the u32 range"
        )]
        let cid = n as u32;
        cid
    })
}

/// Parse a `W` (`N` = 1) or `W2` (`N` = 3) array into `(first_cid,
/// values)` segments of `N` metrics per CID.
///
/// The grammar (PDF §9.7.4.3) is `[c [v…]  cfirst clast v1…vN  …]`, and
/// every element, the inner arrays' included, may be indirect.  A
/// malformed entry (an element that is no CID, a descending or wider than
/// 16-bit range) is skipped; an empty inner array yields no segment, so
/// the segment count stays bounded by [`MAX_CID_METRIC_ENTRIES`].  `key`
/// names the array in diagnostics.
fn parse_cid_metrics<const N: usize>(
    doc: &Document,
    arr: &[Object],
    key: &str,
) -> Vec<(u32, Vec<[i32; N]>)> {
    let metric = |obj: &Object| number(doc, obj).map_or(0, width_units);
    let mut segments = Vec::new();
    let mut materialised = 0usize;
    let mut i = 0;
    while i < arr.len() {
        let Some(first) = cid_number(doc, &arr[i]) else {
            log::warn!("font: {key} entry {i} is not a CID — skipping it");
            i += 1;
            continue;
        };
        let Some(second) = arr.get(i + 1).and_then(|o| doc.resolve(o).ok()) else {
            break;
        };
        let values: Vec<[i32; N]> = match &*second {
            Object::Array(inner) => {
                i += 2;
                inner
                    .as_chunks::<N>()
                    .0
                    .iter()
                    .map(|chunk| chunk.each_ref().map(metric))
                    .collect()
            }
            last if number(doc, last).is_some() => {
                let Some(value) = arr.get(i + 2..i + 2 + N) else {
                    break;
                };
                let value: [i32; N] = std::array::from_fn(|k| metric(&value[k]));
                i += 2 + N;
                match cid_number(doc, last).filter(|&last| last >= first) {
                    Some(last) if last - first < 0x1_0000 => {
                        vec![value; (last - first) as usize + 1]
                    }
                    _ => {
                        log::warn!(
                            "font: {key} range from CID {first} is degenerate — skipping it"
                        );
                        continue;
                    }
                }
            }
            _ => {
                i += 2;
                continue;
            }
        };
        if values.is_empty() {
            continue;
        }
        if materialised.saturating_add(values.len()) > MAX_CID_METRIC_ENTRIES {
            log::warn!(
                "font: {key} exceeds {MAX_CID_METRIC_ENTRIES} CID entries (the 16-bit CID \
                 space) — ignoring the rest"
            );
            break;
        }
        materialised += values.len();
        segments.push((first, values));
    }
    segments
}

/// The array at `dict[key]`, direct or indirect, parsed by
/// [`parse_cid_metrics`]; no segments when it is absent or no array.
fn cid_metrics<const N: usize>(
    doc: &Document,
    dict: &Dictionary,
    key: &str,
) -> CidMetrics<[i32; N]> {
    let segments = dict
        .get(key.as_bytes())
        .and_then(|o| doc.resolve(o).ok())
        .and_then(|o| {
            o.as_array()
                .map(|arr| parse_cid_metrics::<N>(doc, arr, key))
        })
        .unwrap_or_default();
    CidMetrics::new(segments)
}

/// `DW2` and `W2` of a `CIDFont` dictionary: `([vy, w1y], metrics)`, with
/// the spec default `[880 -1000]` for an absent or malformed `DW2`.
fn extract_cid_vertical_metrics(
    doc: &Document,
    dict: &Dictionary,
) -> ([i32; 2], CidMetrics<[i32; 3]>) {
    let dw2 = match number_array(doc, dict.get(b"DW2")).as_slice() {
        &[vy, w1y, ..] => [width_units(vy), width_units(w1y)],
        _ => [880, -1000],
    };
    (dw2, cid_metrics::<3>(doc, dict, "W2"))
}

/// `DW` and `W` of a `CIDFont` dictionary: `(default_width, widths)`, the
/// default 1000 when `DW` is absent (PDF §9.7.4.3).
fn extract_cid_widths(doc: &Document, dict: &Dictionary) -> (i32, CidMetrics<i32>) {
    let dw = dict
        .get(b"DW")
        .and_then(|o| number(doc, o))
        .map_or(1000, width_units);
    (dw, cid_metrics::<1>(doc, dict, "W").map(|[w]| w))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::tests::make_dict;
    use super::*;
    use crate::test_helpers::{empty_doc, make_doc_with_object};

    #[test]
    fn cid_widths_resolve_indirect_w_array() {
        // Real-world CIDFonts (Word/Acrobat output) emit `/W` as an indirect
        // reference (`/W 119 0 R`).  PDF §9.7.4.3: the value is a (possibly
        // indirect) array.  MuPDF reads it via `pdf_dict_get` which resolves
        // references transparently (source_pdf/pdf-font.c:1266).  Without
        // resolving, every per-CID width is dropped and all glyphs advance by
        // `DW` (1000/em), piling text into an unreadable scramble.
        let doc = make_doc_with_object("[ 3 3 250 5 [ 500 600 ] ]");
        let dict = make_dict(&[
            (b"DW", Object::Integer(1000)),
            (b"W", Object::Reference((2, 0))),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000);
        // `3 3 250` (range form) → CID 3 width 250; `5 [500 600]` → CIDs 5,6.
        assert_eq!(segs.segments(), vec![(3, vec![250]), (5, vec![500, 600])]);
    }

    #[test]
    fn cid_widths_resolve_indirect_dw() {
        // `/DW` may also be an indirect reference; it must resolve too, else
        // the default advance silently falls back to the 1000 hardcoded
        // default instead of the document's intended em width.
        let doc = make_doc_with_object("742");
        let dict = make_dict(&[(b"DW", Object::Reference((2, 0)))]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 742);
        assert_eq!(segs.segments().len(), 0);
    }

    #[test]
    fn cid_widths_inline_w_array_unaffected() {
        // Inline (non-reference) `/W` arrays — the case for already-passing
        // CID PDFs — must keep working: `doc.resolve` is a transparent
        // pass-through for a non-reference Object, so this proves the indirect
        // fix does not regress the inline path.
        let doc = empty_doc();
        let dict = make_dict(&[
            (b"DW", Object::Integer(500)),
            (
                b"W",
                Object::Array(vec![
                    Object::Integer(10),
                    Object::Array(vec![Object::Integer(333), Object::Integer(444)]),
                ]),
            ),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 500);
        assert_eq!(segs.segments(), vec![(10, vec![333, 444])]);
    }

    #[test]
    fn cid_widths_resolve_indirect_inner_w_elements() {
        // PDF §9.7.4.3: *any* number in `/W` may be an indirect reference —
        // including the elements of the nested width array and the range
        // `last_cid`/`w`.  Resolving only the top-level `/W` and the leading
        // CID leaves an inner `5 0 R`/`6 0 R` width collapsing to 0-advance
        // (overlapping glyphs — the scramble defect on a different input).
        // This proves inner elements resolve too.
        //
        // obj 4 = `/W` array: `1 [ 5 0 R 6 0 R ]  20 7 0 R  8 0 R`
        //   → CID 1 widths [333, 444] (indirect), CIDs 20..21 width 555 from
        //     an indirect last_cid (obj 7 = 21) and indirect range width
        //     (obj 8 = 555).  Object ids MUST stay contiguous (4,5,6,7,8) so
        //     the single-subsection test xref stays valid.
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, "[ 1 [ 5 0 R 6 0 R ] 20 7 0 R 8 0 R ]"),
            (5, "333"),
            (6, "444"),
            (7, "21"),
            (8, "555"),
        ]);
        let dict = make_dict(&[(b"W", Object::Reference((4, 0)))]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000, "DW absent → spec default");
        assert_eq!(
            segs.segments(),
            vec![(1, vec![333, 444]), (20, vec![555, 555])],
            "indirect inner width-array elements and indirect range last_cid/w \
             must resolve, not collapse to 0"
        );
    }

    #[test]
    fn cid_widths_cyclic_w_ref_is_loud_graceful() {
        // A self-referential `/W` (`/W 4 0 R; 4 0 obj 4 0 R`) is the
        // recursive-resolve DoS class.  `Document::resolve` is depth-bounded
        // (32) and returns an error on the cycle; `extract_cid_widths` maps
        // that to the empty-segment fallback (graceful — glyphs advance by DW,
        // never a panic, hang, or OOM).
        let doc = crate::test_helpers::make_doc_with_objects(&[(4, "4 0 R")]);
        let dict = make_dict(&[
            (b"DW", Object::Integer(600)),
            (b"W", Object::Reference((4, 0))),
        ]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 600, "DW still honoured when /W is a broken cycle");
        assert!(
            segs.segments().is_empty(),
            "cyclic /W must degrade to no per-CID widths, not hang or panic"
        );
    }

    #[test]
    fn cid_widths_malformed_w_array_is_loud_graceful() {
        // Hostile/corrupt `/W`, parsed strictly left-to-right per the PDF
        // §9.7.4.3 grammar.  Sequence (with how the cursor consumes it):
        //   Name        → bad leading element, skip 1
        //   5 [ -7 9 ]  → well-formed width-array form: CID 5 widths [-7, 9]
        //                 (a negative width is kept as is, it is
        //                 NOT silently zeroed or panicked)
        //   50 40 200   → range form, cfirst(50) > clast(40) → degenerate,
        //                 logged + skipped, all 3 consumed
        //   99          → dangling leading CID with no value → loop breaks
        // No panic, no silent corruption of the one valid entry.
        let doc = empty_doc();
        let dict = make_dict(&[(
            b"W",
            Object::Array(vec![
                Object::Name(b"NotACid".to_vec()),
                Object::Integer(5),
                Object::Array(vec![Object::Integer(-7), Object::Integer(9)]),
                Object::Integer(50),
                Object::Integer(40),
                Object::Integer(200),
                Object::Integer(99),
            ]),
        )]);
        let (dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(dw, 1000);
        assert_eq!(segs.segments(), vec![(5, vec![-7, 9])]);
    }

    /// Empty inner arrays add no segment (a `W` of a million `c []` pairs
    /// would otherwise be scanned per glyph), a real-valued `clast` is a
    /// CID, and where segments overlap the earlier one applies.
    #[test]
    fn cid_widths_skip_empty_runs_and_keep_array_precedence() {
        let doc = empty_doc();
        let mut w = Vec::new();
        for _ in 0..1000 {
            w.extend([Object::Integer(7), Object::Array(vec![])]);
        }
        w.extend([
            Object::Integer(10),
            Object::Real(12.0),
            Object::Integer(300),
            Object::Integer(11),
            Object::Array(vec![Object::Integer(400), Object::Integer(500)]),
        ]);
        let dict = make_dict(&[(b"W", Object::Array(w))]);
        let (_, widths) = extract_cid_widths(&doc, &dict);
        assert_eq!(
            widths.segments(),
            vec![(10, vec![300; 3]), (11, vec![400, 500])]
        );
        assert_eq!(widths.get(11), Some(&300), "the earlier segment applies");
        assert_eq!(widths.get(13), None);
    }

    /// A lookup in ascending, disjoint segments bisects to the same
    /// answer a scan gives, at every boundary.
    #[test]
    fn cid_metrics_lookup_matches_a_scan() {
        let segments = vec![
            (0, vec![1, 2]),
            (5, vec![3]),
            (6, vec![4, 5, 6]),
            (100, vec![7]),
        ];
        let metrics = CidMetrics::new(segments.clone());
        assert!(metrics.sorted);
        for cid in 0u32..=101 {
            let scan = segments.iter().find_map(|(first, values)| {
                values.get(usize::try_from(cid.checked_sub(*first)?).ok()?)
            });
            assert_eq!(metrics.get(cid), scan, "CID {cid}");
        }
        assert!(!CidMetrics::new(vec![(5, vec![1, 2]), (6, vec![3])]).sorted);
    }

    #[test]
    fn cid_widths_oversized_range_is_capped() {
        // A chain of maximal ranges must not amplify a tiny `/W` into
        // gigabytes of Vec<i32>.  Two back-to-back full-16-bit ranges request
        // 2 × 65 536 slots; the second crosses MAX_CID_METRIC_ENTRIES and is
        // truncated (loud-graceful) rather than OOMing.
        let doc = empty_doc();
        let dict = make_dict(&[(
            b"W",
            Object::Array(vec![
                Object::Integer(0),
                Object::Integer(0xFFFF),
                Object::Integer(500),
                Object::Integer(0x1_0000),
                Object::Integer(0x1_FFFF),
                Object::Integer(600),
            ]),
        )]);
        let (_dw, segs) = extract_cid_widths(&doc, &dict);
        assert_eq!(
            segs.segments().len(),
            1,
            "second range exceeds the entry cap"
        );
        assert_eq!(segs.segments()[0].0, 0);
        assert_eq!(segs.segments()[0].1.len(), 0x1_0000);
    }

    #[test]
    fn cid_vertical_metrics_parse_both_w2_forms() {
        // `1 [ -900 500 800 ]  5 6 -1000 250 880`
        let doc = empty_doc();
        let dict = make_dict(&[
            (
                b"DW2",
                Object::Array(vec![Object::Integer(850), Object::Integer(-950)]),
            ),
            (
                b"W2",
                Object::Array(vec![
                    Object::Integer(1),
                    Object::Array(vec![
                        Object::Integer(-900),
                        Object::Integer(500),
                        Object::Integer(800),
                    ]),
                    Object::Integer(5),
                    Object::Integer(6),
                    Object::Integer(-1000),
                    Object::Integer(250),
                    Object::Integer(880),
                ]),
            ),
        ]);
        let (dw2, segments) = extract_cid_vertical_metrics(&doc, &dict);
        assert_eq!(dw2, [850, -950]);
        assert_eq!(
            segments.segments(),
            vec![(1, vec![[-900, 500, 800]]), (5, vec![[-1000, 250, 880]; 2])]
        );
        let enc = CidEncoding {
            encoding_cmap: Arc::new(CMap::identity_vertical()),
            cid_to_gid: None,
            default_width: 600,
            widths: CidMetrics::new(Vec::new()),
            vertical: true,
            default_vertical: dw2,
            vertical_metrics: segments,
            to_unicode: None,
        };
        assert_eq!(
            enc.vertical_metrics_for_cid(1),
            VerticalMetrics {
                w1y: -900,
                vx: 500,
                vy: 800
            }
        );
        assert_eq!(enc.vertical_metrics_for_cid(6).vx, 250);
        // Defaults: DW2's advance and vy, half the horizontal width for vx.
        assert_eq!(
            enc.vertical_metrics_for_cid(9),
            VerticalMetrics {
                w1y: -950,
                vx: 300,
                vy: 850
            }
        );
    }

    #[test]
    fn vertical_metrics_defaults_without_dw2() {
        let doc = empty_doc();
        let (dw2, segments) = extract_cid_vertical_metrics(&doc, &make_dict(&[]));
        assert_eq!(dw2, [880, -1000]);
        assert_eq!(segments.segments(), []);
    }
}
