//! PDF object model and single-object parser.
//!
//! [`Object`] has one variant per PDF object type (ISO 32000-1 §7.3).  The
//! parser is a hand-rolled recursive descent over a borrowed byte slice.

use crate::dictionary::Dictionary;
use crate::lexer::{
    is_delim, is_number_byte, is_ws, parse_number, parse_u64, read_hex_string, read_literal_string,
    read_name, scan_token, skip_ws,
};

/// A PDF indirect-object identifier `(object_number, generation)`.
///
/// In practice generation numbers are always 0 in modern PDFs.
pub type ObjectId = (u32, u16);

/// The largest number a file can give an object or a reference.
///
/// A larger number in an object header, a reference or a cross-reference
/// entry is not an object number: the header and the entry are ignored and
/// the `N G R` is read as integers.  The numbers above this one identify
/// page-tree nodes written as direct dictionaries (see
/// [`crate::Document::get_page`]), so they never collide with an object of
/// the file.
pub const MAX_OBJECT_NUMBER: u32 = (1 << 31) - 1;

/// `n` as an object number: `None` when it is negative or larger than
/// [`MAX_OBJECT_NUMBER`].
pub(crate) fn object_number<N: TryInto<u32>>(n: N) -> Option<u32> {
    n.try_into().ok().filter(|&num| num <= MAX_OBJECT_NUMBER)
}

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
    /// A filter that met corrupt or missing data keeps the bytes it decoded
    /// before it; that truncation is logged.
    ///
    /// # Errors
    /// Returns the underlying decode error string when a filter step fails.
    pub fn decompressed_content(&self) -> Result<Vec<u8>, String> {
        let decoded = crate::stream::decode_stream(&self.content, &self.dict)?;
        if let Some(reason) = &decoded.truncated {
            log::warn!("stream: {reason}");
        }
        Ok(decoded.bytes)
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
    keyword_at(data, p, b"endstream")
}

/// Whether keyword `kw` starts at `pos` and ends at whitespace, a delimiter
/// or the end of the data.
fn keyword_at(data: &[u8], pos: usize, kw: &[u8]) -> bool {
    data.get(pos..).is_some_and(|rest| rest.starts_with(kw))
        && data
            .get(pos + kw.len())
            .is_none_or(|&b| is_ws(b) || is_delim(b))
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
pub(crate) const MAX_STREAM_SCAN: usize = 1usize << 31; // 2 GiB

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
/// endobj` (PDF §7.3.8 / §7.3.10), so the terminator is the `endstream`
/// **followed — after whitespace — by `endobj`**.  A false `endstream`
/// embedded in compressed data essentially never satisfies that.
/// Preference order:
///   1. first `endstream` keyword followed by `endobj`, whatever byte
///      precedes it (the strong terminator — every well-formed object has
///      exactly this, so stop at the first to keep the scan O(stream) not
///      O(file));
///   2. else last EOL-preceded `endstream` keyword (writers always
///      EOL-precede it; a stray in-body hit is followed by more data then
///      the real terminator, so the *last* such hit is the safer pick);
///   3. else first `endstream` keyword;
///   4. else, when no `endstream` keyword precedes it, the object's own
///      `endobj` between whitespace, before the `obj` of another object:
///      the object closes without the stream's terminator;
///   5. else the first `N G obj` header beginning a line met before any
///      `endstream` keyword: a body never extends into another object, so it ends
///      before that header;
///   6. else the first `endstream` bytes in any context (degenerate input —
///      recover *something* rather than dropping the whole page).
///
/// `endstream` is a keyword when a whitespace or delimiter byte, or the end
/// of the data, follows it (`endstream.` is body data).  Outside rule 1 it
/// also needs a byte other than a delimiter before it (`(endstream` opens a
/// string; `ETendstream` ends a body that omits the EOL).
///
/// Once a candidate or an object header is known, the scan ends at the
/// next `endobj` or `stream` keyword: past them lies another object, whose
/// `endstream` cannot terminate this body.  A header alone does not end the
/// scan, since a body may hold the text `N G obj` (a string `(1 0 obj)`).  The work and the body are
/// thereby bounded by the distance to the next object, so a file of many
/// streams that share one terminator, or none of whose terminators is
/// followed by `endobj`, is scanned in time linear in its size rather than
/// once to the end of the file per stream.
///
/// Returns `None` if none of `endstream`, `endobj` or an object header is
/// found within [`MAX_STREAM_SCAN`] of `start` (truncated / malformed
/// object, or no terminator), so the caller fails loudly rather than
/// scanning unbounded.
fn find_endstream(data: &[u8], start: usize) -> Option<usize> {
    const KW: &[u8] = b"endstream";
    if start > data.len() {
        return None;
    }
    let scan_end = start.saturating_add(MAX_STREAM_SCAN).min(data.len());
    let window = &data[start..scan_end];

    // The first `endstream` that counts as a keyword, and the first run of
    // its bytes in any context.
    let mut first_raw: Option<usize> = None;
    let mut first_bytes: Option<usize> = None;
    let mut last_eol_preceded: Option<usize> = None;

    let keyword_at_rel = |rel: usize, kw: &[u8]| keyword_at(data, start + rel, kw);
    let byte_before = |rel: usize| (start + rel).checked_sub(1).map(|before| data[before]);
    // Whether an `endobj` may still end the body: no other object's
    // `obj` keyword lies between it and the body.
    let mut endobj_ends_body = true;
    // The start of the next object's `N G obj` header.
    let mut next_header: Option<usize> = None;
    for rel in 0..window.len() {
        let rest = &window[rel..];
        if !rest.starts_with(KW) {
            let ws_before = byte_before(rel).is_none_or(is_ws);
            if ws_before && keyword_at_rel(rel, b"obj") {
                if first_raw.is_none() {
                    endobj_ends_body = false;
                }
                if next_header.is_none() && first_raw.is_none() {
                    next_header = object_header_start(data, start, start + rel);
                }
            }
            let found = first_raw.or(first_bytes).is_some();
            // Past a header with no `endstream` keyword before it, that
            // object's `stream` or `endobj` keyword shows the header opens
            // another object: the body ends before it.
            let after_header = next_header.is_some() && first_raw.is_none();
            if after_header
                && ws_before
                && keyword_at_rel(rel, b"endobj")
                && data.get(start + rel + 6).is_none_or(|&b| is_ws(b))
            {
                break;
            }
            if after_header
                && keyword_at_rel(rel, b"stream")
                && byte_before(rel).is_some_and(|b| is_ws(b) || b == b'>')
            {
                break;
            }
            if keyword_at_rel(rel, b"endobj") {
                // A whole-word `endobj` before any `endstream` keyword ends
                // the body itself.
                if first_raw.is_none()
                    && endobj_ends_body
                    && ws_before
                    && data.get(start + rel + 6).is_none_or(|&b| is_ws(b))
                {
                    log::warn!(
                        "stream at offset {start}: `endobj` before any `endstream`; \
                         the body ends at `endobj`"
                    );
                    return Some(trim_eol_before(data, start, start + rel));
                }
                if found {
                    break;
                }
            }
            let opens_stream = rest.starts_with(b"stream") && !window[..rel].ends_with(b"end");
            if found && opens_stream {
                break;
            }
            continue;
        }
        let kw = start + rel;
        if first_bytes.is_none() {
            first_bytes = Some(kw);
        }
        if !keyword_at_rel(rel, KW) {
            continue;
        }
        // A keyword the object's `endobj` follows after whitespace is the
        // terminator, whatever byte of the body precedes it: done, with no
        // need to scan the rest of the file.
        let mut p = kw + KW.len();
        while matches!(
            data.get(p),
            Some(b'\r' | b'\n' | b' ' | b'\t' | b'\x0c' | b'\0')
        ) {
            p += 1;
        }
        if data[p..].starts_with(b"endobj") {
            return Some(trim_eol_before(data, start, kw));
        }
        // Otherwise `endstream` right after a delimiter is text, as in a
        // string `(endstream is …)`; after data glued to it, as in
        // `ETendstream`, a candidate.
        if byte_before(rel).is_some_and(is_delim) {
            continue;
        }
        if first_raw.is_none() {
            first_raw = Some(kw);
        }
        // A delimiter EOL before the keyword makes it the likelier one.
        if kw > start && matches!(data[kw - 1], b'\r' | b'\n') {
            last_eol_preceded = Some(kw);
        }
    }

    if let Some(kw) = last_eol_preceded.or(first_raw) {
        return Some(trim_eol_before(data, start, kw));
    }
    if let Some(header) = next_header {
        log::warn!(
            "stream at offset {start}: no `endstream` before the next object's header \
             at offset {header}; the body ends there"
        );
        return Some(trim_eol_before(data, start, header));
    }
    first_bytes.map(|kw| trim_eol_before(data, start, kw))
}

/// The start of the object header `N G obj` whose `obj` keyword is at
/// `obj`, when the two tokens before it are unsigned integers separated by
/// whitespace and the header begins a line: at `lo`, the start of the
/// scan, or after a carriage return or line feed.
fn object_header_start(data: &[u8], lo: usize, obj: usize) -> Option<usize> {
    let mut p = obj;
    // Whitespace, the generation, whitespace, the object number, backwards.
    for _ in 0..2 {
        let ws_end = p;
        while p > lo && is_ws(data[p - 1]) {
            p -= 1;
        }
        if p == ws_end {
            return None;
        }
        let digits_end = p;
        while p > lo && data[p - 1].is_ascii_digit() {
            p -= 1;
        }
        if p == digits_end {
            return None;
        }
    }
    (p == lo || matches!(data[p - 1], b'\r' | b'\n')).then_some(p)
}

/// Where the body of a stream object lies, relative to its dictionary.
pub(crate) enum StreamBody {
    /// No `stream` keyword follows the dictionary: a plain dictionary.
    None,
    /// `stream` follows, but no `endstream` terminates the body within
    /// [`MAX_STREAM_SCAN`] of `start`, where the body begins.
    Unterminated { start: usize },
    /// The body's byte range, and the position after `endstream`.
    Found {
        body: std::ops::Range<usize>,
        end: usize,
    },
}

/// Locate the body of the stream whose dictionary `dict` ends at `pos`.
///
/// `/Length` is authoritative when it is a *direct* integer that lands on
/// `endstream`; the dvips / pdfTeX idiom writes it as a forward indirect
/// reference (`/Length N 0 R`) whose value object appears *after* the
/// stream — unresolvable here, without a `Document`.  In that case, and
/// whenever the declared length does not land on `endstream`, the real end
/// is found by searching for the keyword and trimming the EOL before it.
pub(crate) fn stream_body(data: &[u8], dict: &Dictionary, pos: usize) -> StreamBody {
    let mut peek = pos;
    skip_ws(data, &mut peek);
    if !data[peek..].starts_with(b"stream") {
        return StreamBody::None;
    }
    peek += 6;
    // Consume exactly one newline after "stream", and the spaces or tabs
    // some writers put before it.
    let blanks = data[peek..]
        .iter()
        .take_while(|&&b| matches!(b, b' ' | b'\t'))
        .count();
    if matches!(data.get(peek + blanks), Some(b'\r' | b'\n')) {
        peek += blanks;
    }
    if data.get(peek) == Some(&b'\r') {
        peek += 1;
    }
    if data.get(peek) == Some(&b'\n') {
        peek += 1;
    }
    let start = peek.min(data.len());
    let declared_len = dict
        .get(b"Length")
        .and_then(Object::as_i64)
        .and_then(|n| usize::try_from(n).ok());
    let stream_end = match declared_len {
        Some(len) if length_lands_on_endstream(data, start, len) => {
            start.saturating_add(len).min(data.len())
        }
        _ => match find_endstream(data, start) {
            Some(end) => end,
            None => return StreamBody::Unterminated { start },
        },
    };
    let mut end = stream_end;
    skip_ws(data, &mut end);
    if data[end..].starts_with(b"endstream") {
        end += 9;
    }
    StreamBody::Found {
        body: start..stream_end,
        end,
    }
}

/// Parse the dictionary at `*pos`, which the caller has checked starts
/// with `<<`, leaving `*pos` after its `>>`; a stream body that follows is
/// not consumed.
pub(crate) fn parse_dictionary(data: &[u8], pos: &mut usize) -> Option<Dictionary> {
    parse_dict(data, pos, 0)
}

/// Parse one PDF object starting at `*pos` in `data`.  Advances `*pos` past
/// the object.  Does NOT parse the `<id> <gen> obj … endobj` wrapper — call
/// `parse_indirect_object` for that.
///
/// Returns `None` only on genuine EOF or if the first non-whitespace token is
/// not a valid object start, or when arrays and dictionaries nest deeper
/// than 256 levels (an over-deep array keeps the items parsed before
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
            match stream_body(data, &dict, *pos) {
                StreamBody::None => Some(Object::Dictionary(dict)),
                // No `endstream` within the scan bound means the object is
                // truncated or the file is not a valid PDF; the body is
                // dropped with a warning rather than silently.
                StreamBody::Unterminated { start } => {
                    log::warn!(
                        "stream at offset {start}: no `endstream` terminator within \
                         {MAX_STREAM_SCAN} bytes (truncated or malformed object); \
                         content dropped"
                    );
                    *pos = start;
                    Some(Object::Stream(Stream::new(dict, Vec::new())))
                }
                StreamBody::Found { body, end } => {
                    let content = data[body].to_vec();
                    *pos = end;
                    Some(Object::Stream(Stream::new(dict, content)))
                }
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
            // `n` is an object number (see `object_number`); otherwise we
            // fall through and emit a plain Integer.
            if let Some(num) = object_number(n) {
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
    // A malformed number token (`612.0.`, `--5`) has the value of its
    // longest well-formed prefix, as in content streams.
    if word.iter().any(u8::is_ascii_digit)
        && let Some(value) = parse_number(word)
    {
        log::debug!(
            "malformed number {:?} read as {value}",
            String::from_utf8_lossy(word)
        );
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the value is clamped to the f32 range; only precision is lost"
        )]
        return Some(Object::Real(value as f32));
    }

    None
}

/// Parse a real-number token: an optional sign, decimal digits with at most
/// one `.`, and an optional `e`/`E` exponent.  Spellings of infinity and NaN
/// are not numbers in PDF syntax and yield `None`; a numeric token beyond the
/// `f32` range saturates to `±f32::MAX`, so every `Real` is finite.
fn parse_real(s: &str) -> Option<f32> {
    let numeric = s.bytes().any(|b| b.is_ascii_digit()) && s.bytes().all(is_number_byte);
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
///
/// A key whose value is not an object (a bare keyword, a stray delimiter) is
/// dropped together with the offending token, and the remaining entries are
/// read.  An object-level keyword (`endobj`, `stream`, `endstream`, or the
/// `obj` or `trailer` that opens what follows an unterminated dictionary)
/// where a key or value is expected ends the dictionary without being
/// consumed.
/// `None` only when the dictionary, or a container value inside it, nests
/// deeper than [`MAX_NESTING`].
fn parse_dict(data: &[u8], pos: &mut usize, depth: usize) -> Option<Dictionary> {
    if depth >= MAX_NESTING {
        return None;
    }
    // Caller has ensured data[*pos..] starts with "<<".
    *pos += 2;
    let mut dict = Dictionary::new();

    loop {
        skip_ws(data, pos);
        if *pos >= data.len() {
            break;
        }
        if data[*pos..].starts_with(b">>") {
            *pos += 2;
            break;
        }
        if at_object_keyword(data, *pos) {
            break;
        }
        // Key must be a Name.
        if data[*pos] != b'/' {
            // Malformed dict — skip one byte and retry.
            *pos += 1;
            continue;
        }
        let key = read_name(data, pos);
        // Value is any object.
        skip_ws(data, pos);
        let value_start = *pos;
        if let Some(value) = parse_object_nested(data, pos, depth + 1) {
            // A repeated key keeps its last value.
            let _ = dict.insert(key, value);
            continue;
        }
        // Only a container nested past the limit fails to parse as a
        // container; that failure is the whole dictionary's.
        if data[value_start..].starts_with(b"<<") || data.get(value_start) == Some(&b'[') {
            return None;
        }
        *pos = value_start;
        if data[*pos..].starts_with(b">>") || at_object_keyword(data, *pos) {
            continue;
        }
        // Step over the token that is not a value; a lone delimiter is one
        // byte.  At the end of the data there is nothing to step over.
        if scan_token(data, pos).is_empty() && *pos < data.len() {
            *pos += 1;
        }
    }

    Some(dict)
}

/// Whether the token at `pos` is a keyword that ends an indirect object,
/// opens a stream body, or opens the next indirect object or the trailer,
/// so no dictionary entry can continue past it.
///
/// `endobj`, `stream` and `endstream` count wherever they start; `obj` and
/// `trailer` only at a token boundary, as the `obj` of the next object's
/// `N G obj` header or the `trailer` of an unterminated object's file.
/// Reads at most one byte before `pos` and one past the longest keyword, so
/// a caller stepping through a long token byte by byte stays linear in its
/// length.
fn at_object_keyword(data: &[u8], pos: usize) -> bool {
    let keyword_here = |kw: &[u8]| keyword_at(data, pos, kw);
    let token_starts = pos
        .checked_sub(1)
        .and_then(|before| data.get(before))
        .is_none_or(|&b| is_ws(b) || is_delim(b));
    [b"endobj".as_slice(), b"stream", b"endstream"]
        .into_iter()
        .any(keyword_here)
        || (token_starts
            && [b"obj".as_slice(), b"trailer"]
                .into_iter()
                .any(keyword_here))
}

/// Parse a complete indirect object (`<id> <gen> obj … endobj`) at `offset`.
/// Returns `(object_id, object_value)`.
pub(crate) fn parse_indirect_object(data: &[u8], offset: usize) -> Option<(ObjectId, Object)> {
    let mut pos = offset;
    skip_ws(data, &mut pos);

    let id_num = object_number(parse_u64(data, &mut pos)?)?;
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

    /// Without an `endobj`-anchored terminator, the scan ends at the next
    /// object's keywords: the body is cut at this object's own `endstream`,
    /// never at a later object's.
    #[test]
    fn find_endstream_stops_at_the_next_object() {
        let data = b"abc\nendstream\n2 0 obj <</Length 1 0 R>> stream\nxyz\nendstream\n";
        let end = find_endstream(data, 0).expect("found");
        assert_eq!(&data[..end], b"abc");
        let data = b"abc\nendstream junk endobj more\nendstream\n";
        let end = find_endstream(data, 0).expect("found");
        assert_eq!(&data[..end], b"abc");
    }

    /// The content of the stream object `6 0 obj <</Length len>> stream<eol>
    /// body`.
    fn stream_content(len: usize, eol: &str, body: &str) -> Vec<u8> {
        let src = format!("6 0 obj\n<</Length {len}>>\nstream{eol}{body}");
        let Some((_, Object::Stream(s))) = parse_indirect_object(src.as_bytes(), 0) else {
            panic!("a stream: {src:?}");
        };
        s.content
    }

    /// pdfium's seven `/Length` recovery cases (its `bug_524043_*` pixel
    /// tests) and the `ETendstream` one (`bug_551258_1`): the body around a
    /// `/Length` that is right, or that misses `endstream`, against keywords
    /// in and after the data.
    #[test]
    fn stream_length_recovery_cases() {
        let head = "BT\n20 50 Td\n/F1 12 Tf\n(endobj is text) Tj\n";
        let tail = "0 50 Td\n/F2 13 Tf\n(second) Tj\nET";
        let both = |mid: &str| format!("{head}{mid}\n{tail}");
        let cut = head.len() + 3;
        // 1: a /Length that lands on the real `endstream` is kept.
        let body = both("endstream");
        let src = format!("{body}\nendstream\nendobj\n");
        assert_eq!(stream_content(body.len(), "\n", &src), body.as_bytes());
        // 2: a wrong /Length; the `endobj`-anchored terminator wins.
        assert_eq!(stream_content(cut, "\n", &src), body.as_bytes());
        // 3: `eendstream` in the data is not a terminator either.
        let body = both("eendstream");
        let src = format!("{body}\nendstream\nendobj\n");
        assert_eq!(stream_content(cut, "\n", &src), body.as_bytes());
        // 4: `endstream.` is data, also where /Length lands on it, and so is
        // `(endstream`; the object's `endobj` ends the body.
        let body = format!("{}(endstream) Tj\n", both("endstream."));
        let src = format!("{body}\nendobj\n");
        assert_eq!(stream_content(cut, "\n", &src), body.as_bytes());
        assert_eq!(stream_content(head.len(), "\n", &src), body.as_bytes());
        // 5: `endstream%` is a candidate, but the later EOL-preceded one wins.
        let body = both("endstream%");
        let src = format!("{body}\nendstream\nxref\n");
        assert_eq!(stream_content(cut, "\n", &src), body.as_bytes());
        // 6: `endstream+` is no keyword, and nothing terminates the stream:
        // its bytes end the body as a last resort.
        let src = format!("{}\nxref\ntrailer\n", both("endstream+"));
        assert_eq!(stream_content(cut, "\n", &src), head.trim_end().as_bytes());
        // 7: a lone `endstream` in the data, nothing after: cut there.
        let src = format!("{}\nxref\ntrailer\n", both("endstream"));
        assert_eq!(stream_content(cut, "\n", &src), head.trim_end().as_bytes());
        // `stream`, a space, then the EOL: /Length counts from the line
        // after, and lands on an `endstream` glued to the data.
        let body = both("endstream");
        let src = format!("{body}endstream\nendobj\n");
        assert_eq!(stream_content(body.len(), " \n", &src), body.as_bytes());
    }

    /// A body ending in a delimiter byte glued to `endstream` (compressed
    /// data under an indirect /Length) ends there when `endobj` follows,
    /// also past an earlier EOL-preceded `endstream` in the data.
    #[test]
    fn endobj_anchored_terminator_after_a_delimiter_byte() {
        let data = b"x\x9c/(\nendstream z/endstream\nendobj\n";
        let end = find_endstream(data, 0).expect("found");
        assert_eq!(&data[..end], b"x\x9c/(\nendstream z/");
    }

    /// An `endobj` ends a body without `endstream` only as a whole word and
    /// before another object's header; a later object's `endobj` does not,
    /// and the body ends before that object's header instead.
    #[test]
    fn endobj_ends_a_body_only_within_its_object() {
        let data = b"abc\nendobj\n2 0 obj";
        assert_eq!(find_endstream(data, 0), Some(3));
        assert_eq!(find_endstream(b"(endobj) x>>endobj\n", 0), None);
        assert_eq!(find_endstream(b"abc\n2 0 obj\n<<>>\nendobj\n", 0), Some(3));
    }

    /// Without an `endstream` keyword, a body ends before the next `N G obj`
    /// header; `obj` after anything but two integers is body data.
    #[test]
    fn next_object_header_ends_a_body_without_endstream() {
        let data = b"abc\r\n12 0 obj <</Length 1>>stream\nxyz\nendstream\nendobj\n";
        assert_eq!(find_endstream(data, 0), Some(3));
        let data = b"a obj 0 obj x1 0 obj -1 0 obj\nrest\nendstream\nendobj\n";
        let end = find_endstream(data, 0).expect("found");
        assert_eq!(&data[..end], b"a obj 0 obj x1 0 obj -1 0 obj\nrest");
        // An `endstream` keyword before the header still wins.
        let data = b"abc\nendstream junk\n3 0 obj\n";
        assert_eq!(find_endstream(data, 0), Some(3));
    }

    /// The text `N G obj` inside a body does not end it: a content stream
    /// under an indirect `/Length` showing the string `(1 0 obj)`, and
    /// binary data holding a header-like run, keep their whole body.
    #[test]
    fn header_text_inside_a_body_does_not_end_it() {
        let body = b"BT /F1 12 Tf 72 712 Td (1 0 obj) Tj 0 -14 Td (2 0 obj <<) Tj ET";
        let mut src = b"4 0 obj\n<</Length 5 0 R>>\nstream\n".to_vec();
        src.extend_from_slice(body);
        src.extend_from_slice(b"\nendstream\nendobj\n5 0 obj\n71\nendobj\n");
        let Some((_, Object::Stream(s))) = parse_indirect_object(&src, 0) else {
            panic!("a stream");
        };
        assert_eq!(s.content, body);

        let body = b"x\x9c\x01\n1 0 obj \x7f\xfe/\x00stream\x03\n3 0 obj\xff\x10";
        let mut src = b"4 0 obj\n<</Length 5 0 R>>\nstream\n".to_vec();
        src.extend_from_slice(body);
        src.extend_from_slice(b"\nendstream\nendobj\n");
        let Some((_, Object::Stream(s))) = parse_indirect_object(&src, 0) else {
            panic!("a stream");
        };
        assert_eq!(s.content, body);
    }

    /// Header text within a line, followed by `stream` and `endobj` text,
    /// keeps the body whole; a header beginning a line, followed by that
    /// object's `stream` and no `endstream`, ends it.
    #[test]
    fn only_a_header_beginning_a_line_ends_a_body() {
        let body = b"BT (1 0 obj) Tj ( stream) Tj ( endobj) Tj ET";
        let mut src = b"4 0 obj\n<</Length 5 0 R>>\nstream\n".to_vec();
        src.extend_from_slice(body);
        src.extend_from_slice(b"\nendstream\nendobj\n");
        let Some((_, Object::Stream(s))) = parse_indirect_object(&src, 0) else {
            panic!("a stream");
        };
        assert_eq!(s.content, body);

        let src = b"4 0 obj\n<</Length 5 0 R>>\nstream\nabc\n1 0 obj\n<<>>\nstream\nxyz\n";
        let Some((_, Object::Stream(s))) = parse_indirect_object(src, 0) else {
            panic!("a stream");
        };
        assert_eq!(s.content, b"abc");
    }

    /// Stream objects with a wrong `/Length` and no terminator of their own,
    /// followed by one shared `endstream endobj`, each hold only their own
    /// data: no body runs on to the shared terminator.
    #[test]
    fn streams_sharing_one_terminator_keep_their_own_bodies() {
        const K: usize = 400;
        const PAD: usize = 20_000;
        let mut src = Vec::new();
        let mut offsets = Vec::new();
        for i in 0..K {
            offsets.push(src.len());
            src.extend_from_slice(format!("{} 0 obj <</Length 1>>stream\n", i + 1).as_bytes());
            src.extend(std::iter::repeat_n(b'A', PAD));
            src.push(b'\n');
        }
        src.extend_from_slice(b"endstream\nendobj\n");
        let start = std::time::Instant::now();
        for (i, &offset) in offsets.iter().enumerate() {
            let Some((id, Object::Stream(s))) = parse_indirect_object(&src, offset) else {
                panic!("stream {i}");
            };
            assert_eq!(id.0 as usize, i + 1);
            assert_eq!(s.content.len(), PAD, "object {}", i + 1);
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "took {:?}",
            start.elapsed()
        );
    }

    /// A flood of streams whose terminators are never followed by
    /// `endobj` parses in time linear in its length.
    #[test]
    fn stream_flood_without_endobj_is_linear() {
        let one = b"1 0 obj <</Length 9 0 R>> stream\nbody\nendstream\n";
        let src = one.repeat(100_000);
        let start = std::time::Instant::now();
        let mut pos = 0;
        while let Some((_, obj)) = parse_indirect_object(&src, pos) {
            let Object::Stream(s) = obj else {
                panic!("a stream");
            };
            assert_eq!(s.content, b"body");
            pos += one.len();
            if pos >= src.len() {
                break;
            }
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}",
            start.elapsed()
        );
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

    /// Numbers above `MAX_OBJECT_NUMBER` name neither an object nor a
    /// reference; the largest object number does both.
    #[test]
    fn numbers_above_the_largest_object_number_are_no_object_numbers() {
        let parse = |src: &[u8]| parse_object(src, &mut 0).expect("parse");
        assert_eq!(parse(b"2147483648 0 R"), Object::Integer(2_147_483_648));
        assert_eq!(
            parse(b"2147483647 0 R"),
            Object::Reference((MAX_OBJECT_NUMBER, 0))
        );
        assert!(parse_indirect_object(b"2147483648 0 obj\n1\nendobj", 0).is_none());
        assert!(parse_indirect_object(b"2147483647 0 obj\n1\nendobj", 0).is_some());
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

    /// The dictionary `src` parses to, as `(key, value)` pairs sorted by key.
    fn entries(src: &[u8]) -> Vec<(Vec<u8>, Object)> {
        let Object::Dictionary(d) = parse(src) else {
            panic!("expected a dictionary");
        };
        let mut v: Vec<_> = d.iter().map(|(k, o)| (k.clone(), o.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// A value that is not an object drops its key and the offending token;
    /// the entries around it survive.
    #[test]
    fn unparsable_value_drops_only_its_key() {
        let expect = vec![
            (b"A".to_vec(), Object::Integer(1)),
            (b"C".to_vec(), Object::Name(b"D".to_vec())),
        ];
        for bad in [&b"bar"[..], b")", b"}", b"]", b"R"] {
            let mut src = b"<< /A 1 /B ".to_vec();
            src.extend_from_slice(bad);
            src.extend_from_slice(b" /C /D >>");
            assert_eq!(entries(&src), expect, "{}", String::from_utf8_lossy(bad));
        }
        // A key with no value before `>>`.
        assert_eq!(
            entries(b"<< /A 1 /B >>"),
            vec![(b"A".to_vec(), Object::Integer(1))]
        );
    }

    /// A dictionary cut off after a key, or inside a key's value, ends at
    /// the end of the data with the entries read so far.
    #[test]
    fn truncated_dictionary_ends_at_end_of_data() {
        for src in [&b"<< /A 1 /B"[..], b"<< /A 1 /B ", b"<</A 1/B"] {
            let mut pos = 0;
            let Some(Object::Dictionary(d)) = parse_object(src, &mut pos) else {
                panic!("expected a dictionary: {}", String::from_utf8_lossy(src));
            };
            assert_eq!(d.get(b"A"), Some(&Object::Integer(1)));
            assert!(pos <= src.len());
        }
    }

    /// `endobj` or `stream` where an entry is expected ends the dictionary
    /// unconsumed, so an unclosed dictionary neither swallows the next
    /// object nor hides its stream body.
    #[test]
    fn object_keyword_ends_an_unclosed_dictionary() {
        let src = b"<< /A 1 /B endobj 4 0 obj << /C 2 >>";
        let mut pos = 0;
        let Some(Object::Dictionary(d)) = parse_object(src, &mut pos) else {
            panic!("expected a dictionary");
        };
        assert_eq!(d.len(), 1);
        assert!(src[pos..].starts_with(b"endobj"));

        let src = b"<< /Length 3 /B stream\nabc\nendstream";
        let Object::Stream(stream) = parse(src) else {
            panic!("expected a stream");
        };
        assert_eq!(stream.content, b"abc");
        assert_eq!(stream.dict.get(b"Length"), Some(&Object::Integer(3)));
    }

    /// The `obj` of the next object's header and a `trailer` keyword end an
    /// unclosed dictionary unconsumed; within a longer token they do not.
    #[test]
    fn next_object_or_trailer_ends_an_unclosed_dictionary() {
        for (src, rest) in [
            (&b"<< /A 1 /B\n9 0 obj [1] endobj"[..], &b"obj"[..]),
            (b"<< /A 1 /B 9 0 obj [1] endobj", b"obj"),
            (b"<< /A 1 [2]trailer << /Root 3 0 R >>", b"trailer"),
        ] {
            let mut pos = 0;
            let Some(Object::Dictionary(d)) = parse_object(src, &mut pos) else {
                panic!("expected a dictionary: {}", String::from_utf8_lossy(src));
            };
            assert_eq!(d.get(b"A"), Some(&Object::Integer(1)));
            assert!(!d.contains_key(b"Root"));
            assert!(
                src[pos..].starts_with(rest),
                "{}",
                String::from_utf8_lossy(src)
            );
        }
        assert_eq!(
            entries(b"<< /A 1 xobj /B 2 /C trailers >>"),
            vec![
                (b"A".to_vec(), Object::Integer(1)),
                (b"B".to_vec(), Object::Integer(2)),
            ]
        );
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

    /// A malformed number token has the value of its longest well-formed
    /// prefix, as in content streams, so an array or dictionary holding one
    /// keeps the elements after it.
    #[test]
    fn malformed_number_keeps_the_rest_of_its_array() {
        let Object::Array(items) = parse(b"[0 0 612.0. 792]") else {
            panic!("an array");
        };
        assert_eq!(
            items,
            [
                Object::Integer(0),
                Object::Integer(0),
                Object::Real(612.0),
                Object::Integer(792)
            ]
        );
        for (src, value) in [
            (&b"--5"[..], -5.0),
            (b"1.2.3", 1.2),
            (b"5-", 5.0),
            (b"1e", 1.0),
        ] {
            assert_eq!(
                parse(src),
                Object::Real(value),
                "{}",
                String::from_utf8_lossy(src)
            );
        }
        let entries = entries(b"<< /W [500 --5 600] /X 1 >>");
        assert_eq!(
            entries,
            [
                (
                    b"W".to_vec(),
                    Object::Array(vec![
                        Object::Integer(500),
                        Object::Real(-5.0),
                        Object::Integer(600)
                    ])
                ),
                (b"X".to_vec(), Object::Integer(1)),
            ]
        );
        // A token without a digit is still no number.
        let mut pos = 0;
        assert!(parse_object(b"-.", &mut pos).is_none());
    }

    /// A dictionary that runs into a long token that is no key (stream data
    /// after a lost `>>`) is stepped through in time linear in its length:
    /// a megabyte of it finishes, where rescanning the token at every byte
    /// would not.
    #[test]
    fn dictionary_over_long_junk_token_is_linear() {
        let mut src = b"<< /A 1 ".to_vec();
        src.resize(src.len() + (1 << 20), b'7');
        let mut pos = 0;
        let Some(Object::Dictionary(dict)) = parse_object(&src, &mut pos) else {
            panic!("a dictionary");
        };
        assert_eq!(dict.get(b"A"), Some(&Object::Integer(1)));
    }

    /// The object keywords end a dictionary only as whole tokens.
    #[test]
    fn object_keywords_end_a_dictionary_as_whole_tokens() {
        for (src, keys) in [
            (&b"<< /A 1 endobj /B 2 >>"[..], 1),
            (b"<< /A 1 stream\n", 1),
            (b"<< /A 1 endstream", 1),
            (b"<< /A 1 streamer /B 2 >>", 2),
            (b"<< /A 1 endobjx /B 2 >>", 2),
        ] {
            let mut pos = 0;
            let obj = parse_object(src, &mut pos).expect("an object");
            let dict = obj.as_dict().expect("a dictionary or stream");
            assert_eq!(dict.len(), keys, "{}", String::from_utf8_lossy(src));
        }
    }
}
