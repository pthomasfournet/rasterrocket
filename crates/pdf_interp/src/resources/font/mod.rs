//! PDF font resource extraction.
//!
//! Given a PDF font dictionary, [`resolve_font`] extracts everything the
//! renderer needs to draw with it: the embedded program, parsed, the font
//! kind, the encoding, the widths and the class of its substitute.
//!
//! # PDF font taxonomy
//!
//! ```text
//! Subtype
//!   Type1   → PostScript Type 1 / CFF
//!   MMType1 → Multiple Master (treated as Type 1)
//!   TrueType → TrueType / OpenType-TT
//!   Type0   → CIDFont composite (descendant font carries the actual bytes)
//!   Type3   → paint-procedure glyph (content streams, no font program)
//!   CIDFontType0 / CIDFontType2 → only appear as DescendantFonts entries
//! ```
//!
//! Embedded bytes live under `FontDescriptor` → `FontFile` (Type 1),
//! `FontFile2` (TrueType), or `FontFile3` (CFF/OpenType-CFF).
//!
//! When no program is embedded (standard 14, or unembedded third-party
//! fonts), or the embedded one does not parse, the renderer draws a
//! substitute of the same family class and variant: an installed
//! metric-compatible face, else a base-14 face compiled into the binary.

use std::sync::Arc;

use font::program::FontProgram;
use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;
use crate::resources::font_substitute::{
    SubstituteFamily, SubstituteStyle, differences_script, substitute_style,
};

mod cid;
mod type3;

pub(crate) use cid::CidEncoding;
#[cfg(test)]
pub(crate) use cid::CidMetrics;
pub(crate) use type3::{Type3Data, Type3Glyph};

use cid::resolve_type0_font;
use type3::resolve_type3_font;

// ── Public types ──────────────────────────────────────────────────────────────

/// The PDF font subtype family, which decides how simple-font character
/// codes are resolved to glyphs (see the renderer's font cache).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PdfFontKind {
    /// Type 1 / Type 1C (CFF-wrapped).
    Type1,
    /// TrueType / OpenType-TT.
    TrueType,
    /// Anything else (OpenType-CFF, CID, unknown).
    Other,
}

/// The base encoding a simple font's char codes resolve through before any
/// `Differences` overrides (PDF §9.6.6).  Determines the code→glyph-name
/// table the font cache uses when resolving glyphs by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BaseEncoding {
    /// `/StandardEncoding` — Adobe Standard.  Also the spec default for a
    /// non-symbolic simple font with no explicit base encoding.
    Standard,
    /// `/WinAnsiEncoding` — Windows Code Page 1252 superset.
    WinAnsi,
    /// `/MacRomanEncoding` — Mac OS Roman.
    MacRoman,
    /// No base encoding named: the embedded font program's built-in
    /// encoding (PDF §9.6.6.1, Table 112).  A code the built-in encoding
    /// leaves unassigned, or a font without an embedded Type 1 / CFF
    /// program, resolves through `StandardEncoding`.
    FontProgram,
    /// The built-in encoding of the standard Symbol font (Annex D.5): the
    /// base of a non-embedded Symbol font, whatever encoding name the
    /// dictionary carries.
    Symbol,
    /// The built-in encoding of the standard `ZapfDingbats` font (Annex
    /// D.6): the base of a non-embedded `ZapfDingbats` font, as for
    /// [`Self::Symbol`].
    ZapfDingbats,
}

/// Everything needed to load a font face for one PDF font resource.
#[derive(Debug)]
pub struct FontDescriptor {
    /// PDF font subtype family; selects the code → glyph resolution path.
    pub kind: PdfFontKind,
    /// The embedded font program, parsed once for every page that draws
    /// the font; `None` when the font embeds none or its program does not
    /// parse, and a substitute draws it.
    pub program: Option<Arc<FontProgram>>,
    /// Per-character advance widths in thousandths of a text-space unit,
    /// indexed by `char_code - first_char`.  Empty for CID/Type0 fonts.
    pub widths: Vec<i32>,
    /// First char code covered by `widths`.
    pub first_char: u32,
    /// `MissingWidth` from the `FontDescriptor` sub-dict — the advance width
    /// (in thousandths of a text-space unit) used for char codes outside the
    /// `[first_char, first_char + widths.len())` range.  Default `0` per PDF
    /// §9.8.2.
    pub missing_width: i32,
    /// Per-character glyph name overrides from the PDF `Encoding/Differences`
    /// array.  Index = char code (0–255); `None` = inherit from base encoding.
    /// Used by `FontCache` (in `rasterrocket_interp::renderer::font_cache`) to resolve
    /// names → Unicode → GID through the font program's charmap at
    /// face-load time.
    pub differences: Box<[Option<Box<str>>; 256]>,
    /// Base encoding for simple fonts: the code→glyph-name table that applies
    /// before `Differences` overrides.  `StandardEncoding` per PDF spec when
    /// no `/BaseEncoding` is named.  Unused for Type 0 / Type 3.
    pub base_encoding: BaseEncoding,
    /// CID encoding for Type 0 composite fonts.  `Some` when `Subtype` is
    /// `Type0`; `None` for all simple fonts (Type1, TrueType, `MMType1`).
    pub cid_encoding: Option<CidEncoding>,
    /// Type 3 glyph data.  `Some` only when `Subtype` is `Type3`; `None` for
    /// all fonts with a font program.  When `Some`, `program` is `None` and `kind`
    /// is `Other` (Type 3 has no font program; glyphs are content streams).
    pub type3: Option<Type3Data>,
    /// Typeface class of the substitute drawn when `program` is `None`.
    pub substitute: SubstituteStyle,
}

impl FontDescriptor {
    /// Return the simple-font advance width (in thousandths of a text-space
    /// unit) for `char_code`, or `None` if the `Widths` array is empty.
    ///
    /// PDF §9.2.4: the `Widths` array is the **authoritative** width source
    /// for simple fonts in the `[FirstChar, LastChar]` range.  Outside that
    /// range, `MissingWidth` applies.  The whole array is allowed to be
    /// absent only for the 14 standard fonts (Helvetica, Times-Roman, …),
    /// where the advance width of the substitute face's glyph is the
    /// fallback — that's what `None` here signals to the caller.
    ///
    /// Not for Type 0 (use `CidEncoding::width_for_cid`) or Type 3
    /// (use `Type3Data::width`).
    #[must_use]
    pub fn width_for_code(&self, char_code: u32) -> Option<i32> {
        if self.widths.is_empty() {
            return None;
        }
        // Below FirstChar → out-of-range below; use MissingWidth.
        // At-or-above FirstChar: look up; out-of-range above also uses MissingWidth.
        let Some(idx) = (char_code as usize).checked_sub(self.first_char as usize) else {
            return Some(self.missing_width);
        };
        Some(self.widths.get(idx).copied().unwrap_or(self.missing_width))
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// The descriptor drawn for a `Tf` whose resource name is absent from the
/// `/Font` dictionary or names an object that is not a font dictionary:
/// a non-embedded Helvetica, whose substitute supplies the widths.
#[must_use]
pub(crate) fn fallback_font(doc: &Document) -> FontDescriptor {
    let mut dict = Dictionary::new();
    dict.set(b"Type", Object::Name(b"Font".to_vec()));
    dict.set(b"Subtype", Object::Name(b"Type1".to_vec()));
    dict.set(b"BaseFont", Object::Name(b"Helvetica".to_vec()));
    resolve_font(doc, &dict, None)
}

/// Extract a [`FontDescriptor`] from a PDF font dictionary; `id` is the
/// dictionary's object identifier when it is an indirect object.
///
/// This is infallible: if any optional field is missing we fill in a sensible
/// default so the renderer can always produce *something*, even if it is wrong.
#[must_use]
pub fn resolve_font(doc: &Document, dict: &Dictionary, id: Option<ObjectId>) -> FontDescriptor {
    let subtype = dict.get_name(b"Subtype").unwrap_or(b"");

    if matches!(subtype, b"Type0") {
        return resolve_type0_font(doc, dict);
    }
    if matches!(subtype, b"Type3") {
        return resolve_type3_font(doc, dict, id);
    }

    let kind = classify_kind(dict);
    let program = embedded_program(doc, dict);
    let (first_char, real_widths) = extract_widths(doc, dict);
    let widths: Vec<i32> = real_widths.into_iter().map(width_units).collect();
    let missing_width = extract_missing_width(doc, dict);
    let (differences, mut base_encoding) = extract_encoding(doc, dict);
    let mut substitute = substitute_style(doc, dict, Some(dict));

    // A symbolic standard font drawn by its substitute keeps its built-in
    // encoding as the base: its codes denote symbols, not Latin letters,
    // and only `Differences` may reassign them.  A substitute's script is
    // also read from the glyph names `Differences` assigns.
    if program.is_none() {
        match substitute.family {
            SubstituteFamily::Symbol => base_encoding = BaseEncoding::Symbol,
            SubstituteFamily::Dingbats => base_encoding = BaseEncoding::ZapfDingbats,
            SubstituteFamily::Sans | SubstituteFamily::Serif | SubstituteFamily::Mono => {}
        }
        if substitute.script.is_none() {
            // Codes the font encodes without a `Differences` entry are
            // Latin content through the base encoding.
            let last = first_char.saturating_add(u32::try_from(widths.len()).unwrap_or(u32::MAX));
            let base_codes = (first_char..last)
                .filter(|&code| differences.get(code as usize).is_some_and(Option::is_none))
                .count();
            substitute.script = differences_script(&differences[..], base_codes);
        }
    }

    FontDescriptor {
        kind,
        program,
        widths,
        first_char,
        missing_width,
        differences,
        base_encoding,
        cid_encoding: None,
        type3: None,
        substitute,
    }
}

/// `MissingWidth` of the font descriptor (not of the font dictionary
/// itself); 0 when absent (PDF §9.8.2).
fn extract_missing_width(doc: &Document, dict: &Dictionary) -> i32 {
    resolve_fd(doc, dict)
        .as_ref()
        .and_then(|fd| fd.get(b"MissingWidth"))
        .and_then(|w| number(doc, w))
        .map_or(0, width_units)
}

// ── Kind classification ───────────────────────────────────────────────────────

fn classify_kind(dict: &Dictionary) -> PdfFontKind {
    let subtype = dict.get_name(b"Subtype").unwrap_or(b"");

    match subtype {
        b"Type1" | b"MMType1" => PdfFontKind::Type1,
        b"TrueType" => PdfFontKind::TrueType,
        _ => PdfFontKind::Other,
    }
}

// ── Embedded byte extraction ──────────────────────────────────────────────────

/// The embedded program of the font (or `CIDFont`) dictionary `dict`,
/// parsed; `None` when it embeds none, or (logged) when it does not parse.
///
/// Every `FontFile*` key is tried, the one the subtype names first
/// (`FontFile` for Type 1, `FontFile2` otherwise): the program loader
/// sniffs the actual format, so a Type 1 dictionary whose descriptor holds
/// a TrueType program under `FontFile2` still draws with it.
fn embedded_program(doc: &Document, dict: &Dictionary) -> Option<Arc<FontProgram>> {
    let bytes = extract_bytes(doc, dict)?;
    FontProgram::parse(bytes, 0)
        .inspect_err(|e| {
            log::warn!("font: embedded font program unusable ({e}) — drawing a substitute");
        })
        .ok()
        .map(Arc::new)
}

/// The `FontDescriptor` dictionary of the font (or `CIDFont`) dictionary
/// `dict`, direct or indirect.
pub(crate) fn resolve_fd(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    super::resolve_dict(doc, dict.get(b"FontDescriptor")?)
}

/// Maximum decompressed font stream size (64 MiB).
///
/// Protects against deflate bombs embedded in malicious PDFs.  Real font
/// programs are measured in kilobytes; 64 MiB is a generous upper bound.
const MAX_FONT_BYTES: usize = 64 * 1024 * 1024;

/// Read and decompress a stream at `dict[key]`, following an indirect ref if
/// necessary.  Returns `None` on any error or if the stream exceeds
/// [`MAX_FONT_BYTES`] after decompression.
fn read_stream(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<Vec<u8>> {
    // For inline objects we can read the stream directly; for references the
    // resolved Arc<Object> must outlive the &Stream borrow, so we keep it.
    let direct = dict.get(key)?;
    let resolved;
    let stream = match direct {
        Object::Reference(id) => {
            resolved = doc.get_object(*id).ok()?;
            resolved.as_stream()?
        }
        other => other.as_stream()?,
    };
    let bytes = stream.decompressed_content().ok()?;
    if bytes.len() > MAX_FONT_BYTES {
        log::warn!(
            "font: embedded stream for key {} is {} bytes — exceeds {MAX_FONT_BYTES} limit, skipping",
            String::from_utf8_lossy(key),
            bytes.len(),
        );
        return None;
    }
    Some(bytes)
}

// ── Advance-width extraction ──────────────────────────────────────────────────

/// The finite number `obj` holds, through an indirect reference when it
/// is one; `None` for a non-number, a non-finite real or a broken
/// reference.  Any number of a font dictionary may be indirect.
fn number(doc: &Document, obj: &Object) -> Option<f64> {
    match obj {
        Object::Reference(_) => doc.resolve(obj).ok()?.as_f64(),
        direct => direct.as_f64(),
    }
    .filter(|n| n.is_finite())
}

/// The numbers of the array `obj`, itself possibly indirect, each read by
/// [`number`]; 0 for an element that is no number.  Empty when `obj` is
/// absent or no array.
fn number_array(doc: &Document, obj: Option<&Object>) -> Vec<f64> {
    obj.and_then(|o| doc.resolve(o).ok())
        .and_then(|arr| {
            arr.as_array()
                .map(|arr| arr.iter().map(|o| number(doc, o).unwrap_or(0.0)).collect())
        })
        .unwrap_or_default()
}

/// A glyph metric in thousandths of a text-space unit, rounded to the
/// nearest unit and saturated to the `i32` range.
#[must_use]
pub(crate) const fn width_units(w: f64) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a float-to-int `as` saturates at the target range, the clamp intended here"
    )]
    let units = w.round() as i32;
    units
}

/// `FirstChar` and `Widths` of a simple font dictionary, the widths in
/// glyph space (thousandths of a text-space unit except for Type 3).
///
/// `(0, [])` when `Widths` is absent; also when `FirstChar` is not a code
/// of a single-byte encoding (0–255), since the array cannot then be
/// aligned to the codes it describes.
fn extract_widths(doc: &Document, dict: &Dictionary) -> (u32, Vec<f64>) {
    let first = dict.get(b"FirstChar").map(|o| number(doc, o));
    let first = match first {
        None => 0,
        Some(Some(n)) if n.fract() == 0.0 && (0.0..=255.0).contains(&n) => {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "n is an integer in 0..=255"
            )]
            let code = n as u32;
            code
        }
        Some(invalid) => {
            log::warn!("font: FirstChar {invalid:?} is not a code in 0..=255 — ignoring Widths");
            return (0, Vec::new());
        }
    };
    (first, number_array(doc, dict.get(b"Widths")))
}

// ── Char → glyph-index mapping ────────────────────────────────────────────────

/// Empty differences table — all `None` (inherit from base encoding).
fn empty_differences() -> Box<[Option<Box<str>>; 256]> {
    Box::new(std::array::from_fn(|_| None))
}

/// `Differences` and base encoding of a simple font's `Encoding` entry: a
/// name or a dictionary, either possibly indirect.  An absent entry, or
/// one of another type, gives no differences and
/// [`BaseEncoding::FontProgram`].
fn extract_encoding(
    doc: &Document,
    dict: &Dictionary,
) -> (Box<[Option<Box<str>>; 256]>, BaseEncoding) {
    let encoding = dict.get(b"Encoding").and_then(|o| doc.resolve(o).ok());
    match encoding.as_deref() {
        Some(Object::Dictionary(d)) => (
            parse_differences(doc, d),
            parse_base_encoding(d.get_name(b"BaseEncoding")),
        ),
        // A bare named encoding (`/WinAnsiEncoding`, `/MacRomanEncoding`, …):
        // no Differences, base encoding selected by the name.
        Some(Object::Name(n)) => (empty_differences(), parse_base_encoding(Some(n))),
        _ => (empty_differences(), BaseEncoding::FontProgram),
    }
}

/// Map a PDF base-encoding name to [`BaseEncoding`].  Absent → the font
/// program's built-in encoding; `MacExpertEncoding` (small capitals and
/// figures the base tables do not carry) → `WinAnsiEncoding`, whose
/// Latin range it shares; any other name → `StandardEncoding`.
const fn parse_base_encoding(name: Option<&[u8]>) -> BaseEncoding {
    match name {
        None => BaseEncoding::FontProgram,
        Some(b"WinAnsiEncoding" | b"MacExpertEncoding") => BaseEncoding::WinAnsi,
        Some(b"MacRomanEncoding") => BaseEncoding::MacRoman,
        Some(_) => BaseEncoding::Standard,
    }
}

/// The bytes of the program embedded in the font descriptor of the font
/// (or `CIDFont`) dictionary `dict`, in the key order of
/// [`embedded_program`]; `None` when it embeds none.
fn extract_bytes(doc: &Document, dict: &Dictionary) -> Option<Vec<u8>> {
    let fd = resolve_fd(doc, dict)?;
    let priority: &[&[u8]] = match classify_kind(dict) {
        PdfFontKind::Type1 => &[b"FontFile", b"FontFile3", b"FontFile2"],
        _ => &[b"FontFile2", b"FontFile3", b"FontFile"],
    };
    for key in priority {
        if let Some(bytes) = read_stream(doc, &fd, key) {
            return Some(bytes);
        }
    }
    None
}

/// Parse the `Differences` array, direct or indirect, of a dictionary-form
/// `Encoding` object.
///
/// The array has the form `[code /Name /Name … code /Name …]`: each number
/// sets the code of the next name, and each name is assigned to the
/// current code, which then advances (PDF §9.6.6.1).  A number that is no
/// code of a single-byte encoding (0–255) discards the names after it up
/// to the next number, as does running past code 255.
fn parse_differences(doc: &Document, dict: &Dictionary) -> Box<[Option<Box<str>>; 256]> {
    let mut table = empty_differences();
    let Some(arr) = dict.get(b"Differences").and_then(|o| doc.resolve(o).ok()) else {
        return table;
    };
    let Some(arr) = arr.as_array() else {
        return table;
    };

    // The code of the next name; `None` while names are being discarded.
    let mut code: Option<usize> = Some(0);
    for obj in arr {
        if let Object::Name(name) = obj {
            if let Some(slot) = code.and_then(|c| table.get_mut(c)) {
                *slot = Some(String::from_utf8_lossy(name).into());
            }
            code = code.map(|c| c + 1);
        } else if let Some(n) = number(doc, obj) {
            code = (n.fract() == 0.0 && (0.0..256.0).contains(&n)).then(|| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "n is an integer in 0..256"
                )]
                let c = n as usize;
                c
            });
            if code.is_none() {
                log::warn!("font: Differences code {n} is outside 0..=255 — skipping its names");
            }
        }
    }

    table
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn make_dict(pairs: &[(&[u8], Object)]) -> Dictionary {
        let mut d = Dictionary::new();
        for (k, v) in pairs {
            d.set(*k, v.clone());
        }
        d
    }

    #[test]
    fn classify_type1() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::Type1);
    }

    #[test]
    fn classify_truetype() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"TrueType".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::TrueType);
    }

    #[test]
    fn classify_unknown_is_other() {
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type0".to_vec()))]);
        assert_eq!(classify_kind(&d), PdfFontKind::Other);
    }

    #[test]
    fn widths_extracted() {
        let d = make_dict(&[
            (b"FirstChar", Object::Integer(32)),
            (
                b"Widths",
                Object::Array(vec![
                    Object::Integer(278),
                    Object::Integer(500),
                    Object::Integer(556),
                ]),
            ),
        ]);
        let (fc, ws) = extract_widths(&empty_doc(), &d);
        assert_eq!(fc, 32);
        assert_eq!(ws, vec![278.0, 500.0, 556.0]);
    }

    #[test]
    fn widths_absent_returns_empty() {
        let d = make_dict(&[]);
        let (fc, ws) = extract_widths(&empty_doc(), &d);
        assert_eq!(fc, 0);
        assert_eq!(ws.len(), 0);
    }

    /// Build a `FontDescriptor` with explicit widths for testing the
    /// `width_for_code` lookup paths.  The fields not exercised by the
    /// lookup are filled with stubs.
    fn descriptor_with_widths(
        first_char: u32,
        widths: Vec<i32>,
        missing_width: i32,
    ) -> FontDescriptor {
        FontDescriptor {
            kind: PdfFontKind::Type1,
            program: None,
            widths,
            first_char,
            missing_width,
            differences: empty_differences(),
            base_encoding: BaseEncoding::Standard,
            cid_encoding: None,
            type3: None,
            substitute: SubstituteStyle::default(),
        }
    }

    #[test]
    fn width_for_code_in_range() {
        let d = descriptor_with_widths(32, vec![278, 500, 556], 0);
        assert_eq!(d.width_for_code(32), Some(278));
        assert_eq!(d.width_for_code(33), Some(500));
        assert_eq!(d.width_for_code(34), Some(556));
    }

    #[test]
    fn width_for_code_above_range_returns_missing_width() {
        let d = descriptor_with_widths(32, vec![278, 500, 556], 250);
        // 35 is just past LastChar = first_char + len - 1 = 34.
        assert_eq!(d.width_for_code(35), Some(250));
        assert_eq!(d.width_for_code(255), Some(250));
    }

    #[test]
    fn width_for_code_below_range_returns_missing_width() {
        // PDF §9.2.4: below FirstChar is also "out of range" → MissingWidth.
        let d = descriptor_with_widths(32, vec![278, 500, 556], 250);
        assert_eq!(d.width_for_code(0), Some(250));
        assert_eq!(d.width_for_code(31), Some(250));
    }

    #[test]
    fn width_for_code_empty_widths_returns_none() {
        // Standard 14 fonts (Helvetica etc.) pre-PDF-1.5 can omit Widths;
        // None signals to the caller to fall back to the font program's advance.
        let d = descriptor_with_widths(0, vec![], 0);
        assert_eq!(d.width_for_code(65), None);
    }

    #[test]
    fn width_units_round_and_saturate() {
        assert_eq!(width_units(0.0), 0);
        assert_eq!(width_units(277.832), 278);
        assert_eq!(width_units(-1000.0), -1000);
        assert_eq!(width_units(1e300), i32::MAX);
        assert_eq!(width_units(-1e300), i32::MIN);
    }

    /// `Widths` written as an indirect array, the form most producers
    /// emit, and indirect elements inside it are read; a real-valued
    /// width keeps its value rather than truncating.
    #[test]
    fn widths_resolve_indirect_arrays_and_elements() {
        let doc =
            crate::test_helpers::make_doc_with_objects(&[(4, "[ 278 5 0 R 556.5 ]"), (5, "500")]);
        let d = make_dict(&[
            (b"FirstChar", Object::Integer(32)),
            (b"Widths", Object::Reference((4, 0))),
        ]);
        assert_eq!(extract_widths(&doc, &d), (32, vec![278.0, 500.0, 556.5]));
    }

    /// A `FirstChar` that is no single-byte code cannot align `Widths`:
    /// the widths are dropped rather than shifted onto other codes.
    #[test]
    fn widths_with_an_out_of_range_first_char_are_ignored() {
        for first in [
            Object::Integer(-1),
            Object::Integer(1 << 32),
            Object::Real(3.5),
        ] {
            let d = make_dict(&[
                (b"FirstChar", first.clone()),
                (b"Widths", Object::Array(vec![Object::Integer(500)])),
            ]);
            assert_eq!(extract_widths(&empty_doc(), &d), (0, vec![]), "{first:?}");
        }
    }

    #[test]
    fn extract_missing_width_inline_dict() {
        let fd = make_dict(&[(b"MissingWidth", Object::Integer(750))]);
        let parent = make_dict(&[(b"FontDescriptor", Object::Dictionary(fd))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 750);
    }

    #[test]
    fn extract_missing_width_absent_returns_zero() {
        // FontDescriptor sub-dict exists but no MissingWidth key.
        let fd = make_dict(&[]);
        let parent = make_dict(&[(b"FontDescriptor", Object::Dictionary(fd))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 0);
    }

    #[test]
    fn extract_missing_width_no_descriptor_returns_zero() {
        // Whole FontDescriptor key absent → default 0.
        let parent = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        let doc = empty_doc();
        assert_eq!(extract_missing_width(&doc, &parent), 0);
    }

    #[test]
    fn width_for_code_zero_width_in_range_is_distinct_from_missing() {
        // A literal 0 in the Widths array means "glyph has no advance"
        // (zero-width joiner, combining marks). MUST NOT fall through to
        // MissingWidth — they're different signals.
        let d = descriptor_with_widths(32, vec![278, 0, 556], 999);
        assert_eq!(d.width_for_code(33), Some(0), "in-range 0 must be returned");
        assert_eq!(
            d.width_for_code(35),
            Some(999),
            "out-of-range must return MissingWidth, not 0"
        );
    }

    use crate::test_helpers::empty_doc;

    /// `Differences` assigns names from each code on; an out-of-range
    /// code discards its names instead of letting them overwrite the
    /// previous run, and an indirect array is read.
    #[test]
    fn differences_follow_codes_and_skip_out_of_range_runs() {
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            "[ 65 /A /B 300 /X /Y 254 /y /z /overflow ]",
        )]);
        let enc = make_dict(&[(b"Differences", Object::Reference((4, 0)))]);
        let table = parse_differences(&doc, &enc);
        assert_eq!(table[65].as_deref(), Some("A"));
        assert_eq!(table[66].as_deref(), Some("B"));
        assert_eq!(table[67], None, "the names after 300 are discarded");
        assert_eq!(table[254].as_deref(), Some("y"));
        assert_eq!(table[255].as_deref(), Some("z"));
        assert_eq!(table.iter().flatten().count(), 4);
    }

    /// An embedded program that does not parse leaves the descriptor
    /// without one, so it is drawn, and classified, as a font without a
    /// program: a symbolic standard font keeps its built-in encoding.
    #[test]
    fn unparseable_embedded_program_is_drawn_as_a_substitute() {
        let doc = crate::test_helpers::make_doc_with_stream(b"not a font program", "");
        let fd = make_dict(&[(b"FontFile2", Object::Reference((2, 0)))]);
        let d = make_dict(&[
            (b"Subtype", Object::Name(b"TrueType".to_vec())),
            (b"BaseFont", Object::Name(b"Symbol".to_vec())),
            (b"FontDescriptor", Object::Dictionary(fd)),
        ]);
        assert!(extract_bytes(&doc, &d).is_some(), "the stream is embedded");
        let desc = resolve_font(&doc, &d, None);
        assert!(desc.program.is_none());
        assert_eq!(desc.base_encoding, BaseEncoding::Symbol);
    }

    #[test]
    fn no_embedded_bytes_returns_none() {
        // A minimal font dict with no FontDescriptor.
        let d = make_dict(&[(b"Subtype", Object::Name(b"Type1".to_vec()))]);
        // We need a Document to call extract_bytes, but for simple cases without
        // embedded streams (no FontDescriptor key) the result is None regardless
        // of what's in the document.
        let doc = empty_doc();
        assert!(extract_bytes(&doc, &d).is_none());
    }
}
