//! PDF object model and single-object parser.
//!
//! [`Object`] has one variant per PDF object type (ISO 32000-1 §7.3).  The
//! parser is a hand-rolled recursive descent over a borrowed byte slice.

use crate::dictionary::Dictionary;
use crate::lexer::{
    parse_u64, read_hex_string, read_literal_string, read_name, scan_token, skip_ws,
};

/// A PDF indirect-object identifier `(object_number, generation)`.
///
/// In practice generation numbers are always 0 in modern PDFs.
pub type ObjectId = (u32, u16);

/// A PDF object value.
#[derive(Debug, Clone, PartialEq)]
pub enum Object {
    /// The `null` object.
    Null,
    /// `true` or `false`.
    Boolean(bool),
    /// An integer literal.
    Integer(i64),
    /// A real literal; always finite.
    Real(f32),
    /// A name, without the leading `/` and with `#xx` escapes decoded.
    Name(Vec<u8>),
    /// A string with escapes decoded, and the syntax it was written in.
    String(Vec<u8>, StringFormat),
    /// An array of objects.
    Array(Vec<Self>),
    /// A dictionary.
    Dictionary(Dictionary),
    /// A stream: dictionary plus undecoded body.
    Stream(Stream),
    /// Indirect reference — resolved lazily by `Document::get_object`.
    Reference(ObjectId),
}

/// Syntax a string object was written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringFormat {
    /// A `(literal)` string.
    Literal,
    /// A `<hex>` string.
    Hexadecimal,
}

/// A PDF stream: a dictionary plus raw (undecoded) bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Stream {
    /// The stream dictionary.
    pub dict: Dictionary,
    /// Raw (compressed) stream bytes.
    pub content: Vec<u8>,
}

impl Stream {
    /// A stream with dictionary `dict` and undecoded body `content`.
    #[must_use]
    pub const fn new(dict: Dictionary, content: Vec<u8>) -> Self {
        Self { dict, content }
    }

    /// Apply the stream's `/Filter` chain to its raw `content` and return the
    /// decoded bytes.
    ///
    /// Image filters (`DCTDecode`, `JBIG2Decode`, `JPXDecode`, `CCITTFaxDecode`)
    /// are passed through unchanged — the codec layer in `rasterrocket-interp` decodes
    /// those.
    ///
    /// # Errors
    /// Returns the underlying decode error string when a filter step fails.
    pub fn decompressed_content(&self) -> Result<Vec<u8>, String> {
        crate::stream::decode_stream(&self.content, &self.dict)
    }
}

impl Object {
    // ── Typed accessors: `Some` only when the variant matches ────────────────

    /// The value of a `Boolean`.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// The value of an `Integer`, or a `Real` truncated toward zero
    /// (saturating at the `i64` range).
    #[must_use]
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "truncation toward zero is the documented conversion; \
                          `as` saturates out-of-range values"
            )]
            Self::Real(r) => Some(*r as i64),
            _ => None,
        }
    }

    /// Strict integer-only accessor.  Returns `Some` only for `Object::Integer`
    /// or `Object::Real` whose value is exactly representable as an integer
    /// (i.e. `r.fract() == 0.0`).  Used by [`Self::as_u32`] / [`Self::as_u64`]
    /// so they don't silently truncate `100.5` to `100` for dict keys whose
    /// spec'd domain is integer (`/Count`, `/N`, `/O`, byte offsets, …).
    fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the value is integral, so only out-of-range magnitudes \
                          change, and those saturate"
            )]
            Self::Real(r) if r.is_finite() && r.fract() == 0.0 => Some(*r as i64),
            _ => None,
        }
    }

    /// Numeric accessor that rejects negatives and `> u32::MAX`.  Convenient
    /// for PDF dict keys whose spec'd domain is non-negative `Integer`
    /// (`/Count`, `/N`, `/O`, page numbers, …).  Returns `None` for any
    /// non-integer, non-numeric, or out-of-range value.
    ///
    /// `Object::Real` values are accepted only when the value is exactly
    /// representable as an integer — `100.0` ✓, `100.5` ✗.  Dict keys
    /// that lie about their type get rejected rather than silently
    /// truncated.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        u32::try_from(self.as_integer()?).ok()
    }

    /// Same as [`Self::as_u32`] but for byte offsets and lengths that
    /// genuinely need the full `u64` range.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        u64::try_from(self.as_integer()?).ok()
    }

    /// The value of a `Real`, or an `Integer` rounded to the nearest `f32`.
    #[must_use]
    pub const fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Real(r) => Some(*r),
            #[expect(
                clippy::cast_precision_loss,
                reason = "rounding to the nearest f32 is the documented conversion"
            )]
            Self::Integer(n) => Some(*n as f32),
            _ => None,
        }
    }

    /// [`Self::as_f32`] widened to `f64`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.as_f32().map(f64::from)
    }

    /// The bytes of a `Name`.
    #[must_use]
    pub fn as_name(&self) -> Option<&[u8]> {
        match self {
            Self::Name(n) => Some(n),
            _ => None,
        }
    }

    /// The bytes of a `String`.
    #[must_use]
    pub fn as_str(&self) -> Option<&[u8]> {
        match self {
            Self::String(s, _) => Some(s),
            _ => None,
        }
    }

    /// The items of an `Array`.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The dictionary of a `Dictionary`, or of a `Stream`.
    #[must_use]
    pub const fn as_dict(&self) -> Option<&Dictionary> {
        match self {
            Self::Dictionary(d) => Some(d),
            Self::Stream(s) => Some(&s.dict),
            _ => None,
        }
    }

    /// The stream of a `Stream`.
    #[must_use]
    pub const fn as_stream(&self) -> Option<&Stream> {
        match self {
            Self::Stream(s) => Some(s),
            _ => None,
        }
    }

    /// The target of a `Reference`.
    #[must_use]
    pub const fn as_reference(&self) -> Option<ObjectId> {
        match self {
            Self::Reference(id) => Some(*id),
            _ => None,
        }
    }

    /// Return the variant name as a static string, so error messages can
    /// describe a value's kind without its full `Debug` representation.
    #[must_use]
    pub const fn enum_variant(&self) -> &'static str {
        match self {
            Self::Null => "Null",
            Self::Boolean(_) => "Boolean",
            Self::Integer(_) => "Integer",
            Self::Real(_) => "Real",
            Self::Name(_) => "Name",
            Self::String(_, _) => "String",
            Self::Array(_) => "Array",
            Self::Dictionary(_) => "Dictionary",
            Self::Stream(_) => "Stream",
            Self::Reference(_) => "Reference",
        }
    }
}

// ── Object parser ─────────────────────────────────────────────────────────────

/// Check that a declared stream `/Length` is consistent with the data: the
/// byte just past `start + len` (after optional EOL whitespace) must be the
/// `endstream` keyword.  Used to decide whether to trust `/Length` or fall
/// back to scanning, so that a stale or indirect `/Length` cannot truncate
/// (or over-read) a stream body.
fn length_lands_on_endstream(data: &[u8], start: usize, len: usize) -> bool {
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    if end > data.len() {
        return false;
    }
    let mut p = end;
    // PDF §7.3.8.1: an EOL may sit between the data and `endstream`.
    while matches!(data.get(p), Some(b'\r' | b'\n' | b' ' | b'\t')) {
        p += 1;
    }
    data[p..].starts_with(b"endstream")
}

/// Trim the single EOL delimiter (CRLF, LF, or CR) that PDF writers insert
/// between the stream data and the `endstream` keyword (PDF §7.3.8.1).  `kw`
/// is the index of the `e` in `endstream`; the returned index is the true end
/// of the body.  Lower-bounded by `start` so an empty stream stays empty.
const fn trim_eol_before(data: &[u8], start: usize, kw: usize) -> usize {
    let mut end = kw;
    if end > start && data[end - 1] == b'\n' {
        end -= 1;
        if end > start && data[end - 1] == b'\r' {
            end -= 1;
        }
    } else if end > start && data[end - 1] == b'\r' {
        end -= 1;
    }
    end
}

/// A scan for `endstream` must be bounded: when the keyword is missing or
/// far away (truncated object, or binary data with no terminator) an
/// unbounded `windows().position()` over a memory-mapped multi-GB file would
/// do O(file size) work *per object* during page resolution.  No legitimate
/// single *undecoded* stream body exceeds this; it tracks the 512 MiB
/// aggregate decoded-content cap in `document.rs` with generous headroom for
/// the compressed-to-decoded ratio, so a real stream is always found while a
/// missing terminator fails fast and loudly instead of degrading throughput.
const MAX_STREAM_SCAN: usize = 1usize << 31; // 2 GiB

/// Deepest array/dictionary nesting [`parse_object`] descends into.  Deeper
/// input fails to parse; the bound keeps recursion within a fixed stack
/// budget on hostile input.
const MAX_NESTING: usize = 256;

/// Scan forward from `start` for the true end of the stream body.
///
/// The hard part is *which* `endstream` is the terminator: `/Length` is
/// indirect (the dvips / pdfTeX `N 0 R` idiom) and unresolvable here, so the
/// declared length cannot disambiguate, and a compressed binary body can
/// legitimately contain the literal bytes `endstream`.  Taking the first raw
/// match would truncate such a stream to a corrupt prefix — silently blank,
/// the exact regression the indirect-`/Length` fix exists to prevent.
///
/// A well-formed indirect object always closes `…data EOL endstream EOL
/// endobj` (PDF §7.3.8 / §7.3.10), so the terminator is the `endstream` that
/// is **EOL-preceded and followed — after whitespace — by `endobj`**.  A
/// false `endstream` embedded in compressed data essentially never satisfies
/// both.  Preference order:
///   1. first EOL-preceded `endstream` immediately followed by `endobj`
///      (the strong terminator — every well-formed object has exactly this,
///      so stop at the first to keep the scan O(stream) not O(file));
///   2. else last EOL-preceded `endstream` (writers always EOL-precede it; a
///      stray in-body hit is followed by more data then the real terminator,
///      so the *last* such hit is the safer pick);
///   3. else first raw `endstream` (degenerate input — recover *something*
///      rather than dropping the whole page).
///
/// Returns `None` if no `endstream` keyword is found within
/// [`MAX_STREAM_SCAN`] of `start` (truncated / malformed object, or no
/// terminator), so the caller fails loudly rather than scanning unbounded.
fn find_endstream(data: &[u8], start: usize) -> Option<usize> {
    const KW: &[u8] = b"endstream";
    if start > data.len() {
        return None;
    }
    let scan_end = start.saturating_add(MAX_STREAM_SCAN).min(data.len());
    let window = &data[start..scan_end];

    let mut first_raw: Option<usize> = None;
    let mut last_eol_preceded: Option<usize> = None;

    for rel in 0..window.len().saturating_sub(KW.len() - 1) {
        if &window[rel..rel + KW.len()] != KW {
            continue;
        }
        let kw = start + rel;
        if first_raw.is_none() {
            first_raw = Some(kw);
        }
        // A delimiter EOL must sit immediately before a terminating keyword.
        if kw == start || !matches!(data[kw - 1], b'\r' | b'\n') {
            continue;
        }
        last_eol_preceded = Some(kw);
        // …and the object must continue with `endobj` after the keyword and
        // any intervening whitespace, or this only looks like a terminator.
        let mut p = kw + KW.len();
        while matches!(
            data.get(p),
            Some(b'\r' | b'\n' | b' ' | b'\t' | b'\x0c' | b'\0')
        ) {
            p += 1;
        }
        if data[p..].starts_with(b"endobj") {
            // Strong terminator — done; no need to scan the rest of the file.
            return Some(trim_eol_before(data, start, kw));
        }
    }

    let kw = last_eol_preceded.or(first_raw)?;
    Some(trim_eol_before(data, start, kw))
}

/// Parse one PDF object starting at `*pos` in `data`.  Advances `*pos` past
/// the object.  Does NOT parse the `<id> <gen> obj … endobj` wrapper — call
/// `parse_indirect_object` for that.
///
/// Returns `None` only on genuine EOF or if the first non-whitespace token is
/// not a valid object start, or when arrays and dictionaries nest deeper
/// than [`MAX_NESTING`] (an over-deep array keeps the items parsed before
/// the limit).
pub fn parse_object(data: &[u8], pos: &mut usize) -> Option<Object> {
    parse_object_nested(data, pos, 0)
}

/// [`parse_object`] for an object enclosed in `depth` arrays/dictionaries.
fn parse_object_nested(data: &[u8], pos: &mut usize, depth: usize) -> Option<Object> {
    skip_ws(data, pos);

    let b = *data.get(*pos)?;

    match b {
        b'/' => {
            let name = read_name(data, pos);
            Some(Object::Name(name))
        }
        b'(' => {
            let s = read_literal_string(data, pos);
            Some(Object::String(s, StringFormat::Literal))
        }
        b'<' if data.get(*pos + 1) == Some(&b'<') => {
            // Dictionary or stream.
            let dict = parse_dict(data, pos, depth)?;
            // Peek past whitespace to see if "stream" follows.
            let mut peek = *pos;
            skip_ws(data, &mut peek);
            if data[peek..].starts_with(b"stream") {
                peek += 6;
                // Consume exactly one newline after "stream".
                if data.get(peek) == Some(&b'\r') {
                    peek += 1;
                }
                if data.get(peek) == Some(&b'\n') {
                    peek += 1;
                }
                let stream_start = peek.min(data.len());
                // Determine stream length.  `/Length` is authoritative when it
                // is a *direct* integer, but the dvips / pdfTeX idiom writes it
                // as a forward indirect reference (`/Length N 0 R`) whose value
                // object appears *after* the stream — unresolvable here because
                // `parse_object` has no `Document`.  In that case (and whenever
                // the declared length does not actually land on `endstream`),
                // fall back to scanning for the `endstream` keyword, which is
                // what every production PDF reader does for robustness.
                let declared_len = dict
                    .get(b"Length")
                    .and_then(Object::as_i64)
                    .and_then(|n| usize::try_from(n).ok());

                let stream_end = match declared_len {
                    Some(len) if length_lands_on_endstream(data, stream_start, len) => {
                        stream_start.saturating_add(len).min(data.len())
                    }
                    // Missing, indirect (`N 0 R`), negative, or inconsistent
                    // `/Length`: locate the real end by searching for the
                    // `endstream` keyword and trimming the EOL that precedes it.
                    // No `endstream` within the scan bound means the object is
                    // truncated or the file is not a valid PDF; the body is
                    // dropped with a warning rather than silently.
                    _ => find_endstream(data, stream_start).unwrap_or_else(|| {
                        log::warn!(
                            "stream at offset {stream_start}: no `endstream` \
                             terminator within {MAX_STREAM_SCAN} bytes \
                             (truncated or malformed object); content dropped"
                        );
                        stream_start
                    }),
                };
                let content = data[stream_start..stream_end].to_vec();
                // Advance pos past "endstream".
                *pos = stream_end;
                skip_ws(data, pos);
                if data[*pos..].starts_with(b"endstream") {
                    *pos += 9;
                }
                Some(Object::Stream(Stream::new(dict, content)))
            } else {
                Some(Object::Dictionary(dict))
            }
        }
        b'<' => {
            let s = read_hex_string(data, pos);
            Some(Object::String(s, StringFormat::Hexadecimal))
        }
        b'[' => {
            if depth >= MAX_NESTING {
                return None;
            }
            *pos += 1;
            let mut items = Vec::new();
            loop {
                skip_ws(data, pos);
                match data.get(*pos) {
                    None => break,
                    Some(b']') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        if let Some(item) = parse_object_nested(data, pos, depth + 1) {
                            items.push(item);
                        } else {
                            break;
                        }
                    }
                }
            }
            Some(Object::Array(items))
        }
        _ => {
            // Number, bool, null, reference, or keyword.
            parse_scalar(data, pos)
        }
    }
}

fn parse_scalar(data: &[u8], pos: &mut usize) -> Option<Object> {
    let word = scan_token(data, pos);
    match word {
        b"true" => return Some(Object::Boolean(true)),
        b"false" => return Some(Object::Boolean(false)),
        b"null" => return Some(Object::Null),
        _ => {}
    }

    // Try to parse as a number.  If it looks like an integer, peek ahead to
    // check for the "X Y R" reference pattern.
    if let Ok(s) = std::str::from_utf8(word) {
        if let Ok(n) = s.parse::<i64>() {
            // Peek for "gen R" to form an indirect reference. Only valid when
            // the object number fits in u32 and is non-negative; otherwise we
            // fall through and emit a plain Integer.
            if let Ok(num) = u32::try_from(n) {
                let mut peek = *pos;
                skip_ws(data, &mut peek);
                let gen_tok = scan_token(data, &mut peek);
                if let Some(generation) = std::str::from_utf8(gen_tok)
                    .ok()
                    .and_then(|s| s.parse::<i64>().ok())
                    .and_then(|g| u16::try_from(g).ok())
                {
                    skip_ws(data, &mut peek);
                    let r_tok = scan_token(data, &mut peek);
                    if r_tok == b"R" {
                        *pos = peek;
                        return Some(Object::Reference((num, generation)));
                    }
                }
            }
            return Some(Object::Integer(n));
        }
        if let Some(r) = parse_real(s) {
            return Some(Object::Real(r));
        }
    }

    None
}

/// Parse a real-number token: an optional sign, decimal digits with at most
/// one `.`, and an optional `e`/`E` exponent.  Spellings of infinity and NaN
/// are not numbers in PDF syntax and yield `None`; a numeric token beyond the
/// `f32` range saturates to `±f32::MAX`, so every `Real` is finite.
fn parse_real(s: &str) -> Option<f32> {
    let numeric = s.bytes().any(|b| b.is_ascii_digit())
        && s.bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'.' | b'+' | b'-' | b'e' | b'E'));
    if !numeric {
        return None;
    }
    let r = s.parse::<f32>().ok()?;
    Some(if r.is_infinite() {
        f32::MAX.copysign(r)
    } else {
        r
    })
}

/// Parse the dictionary at `*pos`, itself enclosed in `depth`
/// arrays/dictionaries.
fn parse_dict(data: &[u8], pos: &mut usize, depth: usize) -> Option<Dictionary> {
    if depth >= MAX_NESTING {
        return None;
    }
    // Caller has ensured data[*pos..] starts with "<<".
    *pos += 2;
    let mut dict = Dictionary::new();

    loop {
        skip_ws(data, pos);
        if data[*pos..].starts_with(b">>") {
            *pos += 2;
            break;
        }
        if *pos >= data.len() {
            break;
        }
        // Key must be a Name.
        if *data.get(*pos)? != b'/' {
            // Malformed dict — skip one byte and retry.
            *pos += 1;
            continue;
        }
        let key = read_name(data, pos);
        // Value is any object.
        skip_ws(data, pos);
        let value = parse_object_nested(data, pos, depth + 1)?;
        // A repeated key keeps its last value.
        let _ = dict.insert(key, value);
    }

    Some(dict)
}

/// Parse a complete indirect object (`<id> <gen> obj … endobj`) at `offset`.
/// Returns `(object_id, object_value)`.
pub(crate) fn parse_indirect_object(data: &[u8], offset: usize) -> Option<(ObjectId, Object)> {
    let mut pos = offset;
    skip_ws(data, &mut pos);

    let id_num = u32::try_from(parse_u64(data, &mut pos)?).ok()?;
    skip_ws(data, &mut pos);
    let gen_num = u16::try_from(parse_u64(data, &mut pos)?).ok()?;
    skip_ws(data, &mut pos);

    if !data[pos..].starts_with(b"obj") {
        return None;
    }
    pos += 3;

    let obj = parse_object(data, &mut pos)?;

    Some(((id_num, gen_num), obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &[u8]) -> Object {
        let mut pos = 0;
        parse_object(src, &mut pos).expect("parse failed")
    }

    #[test]
    fn integer() {
        assert_eq!(parse(b"42"), Object::Integer(42));
        assert_eq!(parse(b"-7"), Object::Integer(-7));
    }

    #[test]
    fn real() {
        assert!(matches!(parse(b"3.14"), Object::Real(_)));
    }

    #[test]
    fn boolean() {
        assert_eq!(parse(b"true"), Object::Boolean(true));
        assert_eq!(parse(b"false"), Object::Boolean(false));
    }

    /// `as_u32` accepts integer-valued reals (`100.0` → `Some(100)`) but
    /// rejects fractional reals (`100.5` → `None`).  Dict keys whose
    /// spec'd domain is integer (`/Count`, `/N`, `/O`, byte offsets, …)
    /// must not silently truncate.
    #[test]
    fn as_u32_rejects_fractional_real() {
        assert_eq!(Object::Integer(42).as_u32(), Some(42));
        assert_eq!(Object::Real(42.0).as_u32(), Some(42));
        assert_eq!(Object::Real(42.5).as_u32(), None);
        assert_eq!(Object::Real(-1.0).as_u32(), None);
        assert_eq!(Object::Real(f32::NAN).as_u32(), None);
        assert_eq!(Object::Real(f32::INFINITY).as_u32(), None);
        assert_eq!(Object::Integer(-1).as_u32(), None);
        assert_eq!(Object::Boolean(true).as_u32(), None);
    }

    #[test]
    fn as_u64_rejects_fractional_real() {
        assert_eq!(Object::Integer(42).as_u64(), Some(42));
        assert_eq!(Object::Real(12345.0).as_u64(), Some(12345));
        assert_eq!(Object::Real(12345.5).as_u64(), None);
        assert_eq!(Object::Integer(-1).as_u64(), None);
    }

    #[test]
    fn null() {
        assert_eq!(parse(b"null"), Object::Null);
    }

    #[test]
    fn name() {
        assert_eq!(parse(b"/Type"), Object::Name(b"Type".to_vec()));
    }

    #[test]
    fn name_with_escape() {
        assert_eq!(parse(b"/#46oo"), Object::Name(b"Foo".to_vec()));
    }

    #[test]
    fn literal_string() {
        assert_eq!(
            parse(b"(hello)"),
            Object::String(b"hello".to_vec(), StringFormat::Literal)
        );
    }

    #[test]
    fn hex_string() {
        assert_eq!(
            parse(b"<48656c6c6f>"),
            Object::String(b"Hello".to_vec(), StringFormat::Hexadecimal)
        );
    }

    #[test]
    fn array() {
        let obj = parse(b"[1 2 3]");
        assert_eq!(
            obj,
            Object::Array(vec![
                Object::Integer(1),
                Object::Integer(2),
                Object::Integer(3)
            ])
        );
    }

    #[test]
    fn dictionary() {
        let obj = parse(b"<</Type /Page /Width 100>>");
        let dict = obj.as_dict().unwrap();
        assert_eq!(dict.get(b"Type").unwrap().as_name().unwrap(), b"Page");
        assert_eq!(dict.get(b"Width").unwrap().as_i64().unwrap(), 100);
    }

    #[test]
    fn indirect_reference() {
        assert_eq!(parse(b"5 0 R"), Object::Reference((5, 0)));
    }

    #[test]
    fn indirect_object() {
        let src = b"1 0 obj\n42\nendobj";
        let (id, obj) = parse_indirect_object(src, 0).unwrap();
        assert_eq!(id, (1, 0));
        assert_eq!(obj, Object::Integer(42));
    }

    #[test]
    fn stream_with_negative_length_recovers_via_endstream_scan() {
        // /Length = -1 used to overflow when cast to usize, then was clamped
        // to 0 (silent data loss).  Now an invalid `/Length` falls back to
        // scanning for the `endstream` keyword, recovering the real body.
        let src = b"<</Length -1>>\nstream\nXY\nendstream";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert_eq!(s.content, b"XY"),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_with_indirect_length_recovers_via_endstream_scan() {
        // dvips / pdfTeX idiom: `/Length N 0 R` is a forward indirect
        // reference, unresolvable at parse time.  Must fall back to the
        // `endstream` scan instead of yielding an empty (white-page) body.
        let src = b"<</Length 6 0 R>>\nstream\nhello world\nendstream";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert_eq!(s.content, b"hello world"),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_with_valid_direct_length_is_byte_exact() {
        // The fast path must be unchanged for well-formed PDFs: a correct
        // direct `/Length` is trusted verbatim (no endstream rescan).
        let src = b"<</Length 5>>\nstream\nABCDE\nendstream";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert_eq!(s.content, b"ABCDE"),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn stream_with_stale_short_length_falls_back_to_scan() {
        // A `/Length` that does not land on `endstream` (stale after the file
        // was hand-edited) must not silently truncate the body.
        let src = b"<</Length 2>>\nstream\nABCDEFG\nendstream";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert_eq!(s.content, b"ABCDEFG"),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn indirect_length_stream_with_literal_endstream_in_binary_body() {
        // The v1 silent-blank sin: an indirect-`/Length` stream (dvips / arXiv
        // idiom) whose *compressed* body legitimately contains the literal
        // bytes `endstream`.  Taking the first raw match would truncate the
        // body to a corrupt prefix → decode failure → blank page.  The real
        // terminator is the one that is EOL-preceded and `endobj`-followed.
        let mut src = Vec::new();
        src.extend_from_slice(b"<</Length 9 0 R>>\nstream\n");
        src.extend_from_slice(b"PART-ONE\x00\x01endstream\x02\x03PART-TWO");
        src.extend_from_slice(b"\nendstream\nendobj");
        let mut pos = 0;
        let obj = parse_object(&src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert_eq!(
                s.content, b"PART-ONE\x00\x01endstream\x02\x03PART-TWO",
                "false `endstream` in binary body truncated the stream"
            ),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn indirect_length_stream_with_no_endstream_terminates_bounded() {
        // Truncated object: indirect `/Length`, no `endstream` anywhere.  Must
        // not scan unbounded / hang; `find_endstream` returns `None` and the
        // caller drops the body (and logs loudly) rather than spinning.
        let src = b"<</Length 9 0 R>>\nstream\nthis body never terminates at all";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        match obj {
            Object::Stream(s) => assert!(
                s.content.is_empty(),
                "no terminator must yield an empty body, not a guess"
            ),
            other => panic!("expected Stream, got {other:?}"),
        }
    }

    #[test]
    fn find_endstream_scan_is_bounded() {
        // Directly assert the scan is bounded: no `endstream` in a buffer
        // larger than the keyword returns `None` without scanning forever.
        let big = vec![b'A'; 4 * 1024 * 1024];
        assert_eq!(find_endstream(&big, 0), None);
        // Start past EOF is rejected, not an out-of-bounds slice.
        assert_eq!(find_endstream(b"abc", 99), None);
    }

    #[test]
    fn find_endstream_prefers_eol_and_endobj_anchored_terminator() {
        // An early EOL-preceded `endstream` that is NOT `endobj`-followed is
        // body data; the real terminator is the `endobj`-anchored one.
        let data = b"data\nendstream more data here\nendstream\nendobj";
        let end = find_endstream(data, 0).expect("found");
        assert_eq!(&data[0..end], b"data\nendstream more data here");
    }

    #[test]
    fn parse_indirect_object_rejects_oversized_id() {
        // Object ID > u32::MAX must be rejected, not silently truncated.
        let src = b"99999999999 0 obj\n42\nendobj";
        assert!(parse_indirect_object(src, 0).is_none());
    }

    #[test]
    fn reference_with_oversized_id_falls_back_to_integer() {
        // "X Y R" where X overflows u32 must NOT be recognised as a Reference.
        let src = b"99999999999 0 R";
        let mut pos = 0;
        let obj = parse_object(src, &mut pos).expect("parse");
        assert_eq!(obj, Object::Integer(99_999_999_999));
    }

    /// Container nesting depth of `obj` (a scalar is depth 0).
    fn nesting_depth(obj: &Object) -> usize {
        match obj {
            Object::Array(items) => 1 + items.iter().map(nesting_depth).max().unwrap_or(0),
            Object::Dictionary(d) => 1 + d.iter().map(|(_, v)| nesting_depth(v)).max().unwrap_or(0),
            _ => 0,
        }
    }

    /// Hostile nesting depth must not overflow the stack: a dictionary
    /// nested past `MAX_NESTING` fails to parse, and an array stops
    /// descending at the limit.
    #[test]
    fn deep_nesting_does_not_overflow_the_stack() {
        let mut src = b"<</K ".repeat(1_000_000);
        src.extend_from_slice(&b">>".repeat(1_000_000));
        assert!(parse_object(&src, &mut 0).is_none());

        let mut src = b"[".repeat(1_000_000);
        src.extend_from_slice(&b"]".repeat(1_000_000));
        let obj = parse_object(&src, &mut 0).expect("partial array");
        assert!(nesting_depth(&obj) <= MAX_NESTING);
    }

    /// Nesting up to `MAX_NESTING` levels still parses.
    #[test]
    fn nesting_at_the_limit_parses() {
        let mut src = b"[".repeat(MAX_NESTING);
        src.extend_from_slice(&b"]".repeat(MAX_NESTING));
        let mut obj = parse(&src);
        for _ in 1..MAX_NESTING {
            let Object::Array(mut items) = obj else {
                panic!("expected an array");
            };
            obj = items.pop().expect("one nested item");
        }
        assert_eq!(obj, Object::Array(Vec::new()));
    }

    /// Spellings of infinity and NaN are not PDF numbers; `str::parse::<f32>`
    /// accepts them, so they used to become non-finite `Real`s.
    #[test]
    fn non_finite_spellings_are_not_numbers() {
        for src in [&b"inf"[..], b"-inf", b"+infinity", b"NaN", b"nan"] {
            let mut pos = 0;
            let obj = parse_object(src, &mut pos);
            assert!(
                obj.is_none(),
                "{:?} parsed as {obj:?}",
                String::from_utf8_lossy(src)
            );
        }
    }

    /// A numeric token beyond the `f32` range saturates instead of becoming
    /// infinite; ordinary reals are unaffected.
    #[test]
    fn out_of_range_real_saturates() {
        assert_eq!(parse(b"1e999"), Object::Real(f32::MAX));
        assert_eq!(parse(b"-1e999"), Object::Real(-f32::MAX));
        assert_eq!(parse(b"4."), Object::Real(4.0));
        assert_eq!(parse(b"-.002"), Object::Real(-0.002));
    }
}
