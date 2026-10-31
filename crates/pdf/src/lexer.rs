//! Shared byte-level scanner primitives used by both the object parser and
//! the content-stream tokenizer.
//!
//! All functions operate on a `(src: &[u8], pos: &mut usize)` pair so callers
//! can drive parsing without wrapping state in a struct.

/// PDF whitespace characters (spec §7.2.2).
#[inline]
pub(crate) const fn is_ws(b: u8) -> bool {
    matches!(b, b'\0' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

/// PDF delimiter characters (spec §7.2.2).
#[inline]
pub(crate) const fn is_delim(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

/// Hex nibble value; non-hex bytes return 0.
#[inline]
pub(crate) const fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

/// Advance past whitespace and `%`-comments.
pub(crate) fn skip_ws(src: &[u8], pos: &mut usize) {
    loop {
        match src.get(*pos).copied() {
            Some(b) if is_ws(b) => *pos += 1,
            Some(b'%') => {
                *pos += 1;
                while *pos < src.len() && !matches!(src[*pos], b'\r' | b'\n') {
                    *pos += 1;
                }
            }
            _ => break,
        }
    }
}

/// Read a `/Name` token starting at `pos` (which points at the `/`).
/// Returns the name bytes (without the leading `/`) as a slice into `src`.
/// Decodes `#xx` escape sequences into owned `Vec<u8>` when needed.
pub(crate) fn read_name(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '/'
    let start = *pos;
    while *pos < src.len() {
        let b = src[*pos];
        if is_ws(b) || is_delim(b) {
            break;
        }
        *pos += 1;
    }
    decode_name(&src[start..*pos]).into_owned()
}

/// The bytes of the name whose source text (without the leading `/`) is
/// `raw`, with each `#xx` escape decoded (ISO 32000-2 §7.3.5).
///
/// Borrows `raw` when it holds no `#`, the common case.
#[must_use]
pub fn decode_name(raw: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if !raw.contains(&b'#') {
        return std::borrow::Cow::Borrowed(raw);
    }
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'#' && i + 2 < raw.len() {
            out.push((hex_nibble(raw[i + 1]) << 4) | hex_nibble(raw[i + 2]));
            i += 3;
        } else {
            out.push(raw[i]);
            i += 1;
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Read a literal `(string)` starting at `pos` (pointing at `(`).
/// Returns decoded bytes (escape sequences resolved, nesting handled).
pub(crate) fn read_literal_string(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '('
    let mut out = Vec::new();
    let mut depth = 1usize;
    while *pos < src.len() {
        let b = src[*pos];
        *pos += 1;
        match b {
            b'(' => {
                depth += 1;
                out.push(b'(');
            }
            b')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                out.push(b')');
            }
            b'\\' => {
                if *pos >= src.len() {
                    break;
                }
                let esc = src[*pos];
                *pos += 1;
                match esc {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'b' => out.push(b'\x08'),
                    b'f' => out.push(b'\x0C'),
                    b'(' => out.push(b'('),
                    b')' => out.push(b')'),
                    b'\\' => out.push(b'\\'),
                    b'\r'
                        // Line continuation — CRLF or bare CR.
                        if src.get(*pos) == Some(&b'\n') => {
                        *pos += 1;
                    }
                    b'\n' => {} // Line continuation.
                    d if d.is_ascii_digit() && d < b'8' => {
                        // Octal escape: up to 3 digits.
                        let mut val = u16::from(d - b'0');
                        for _ in 0..2 {
                            match src.get(*pos) {
                                Some(&c) if c.is_ascii_digit() && c < b'8' => {
                                    val = val * 8 + u16::from(c - b'0');
                                    *pos += 1;
                                }
                                _ => break,
                            }
                        }
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "an octal escape above 0o377 is malformed; its high-order bit is discarded"
                        )]
                        out.push(val as u8);
                    }
                    _ => {} // Unknown escape — ignore backslash per spec.
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Read a hex `<string>` starting at `pos` (pointing at `<`).
/// Returns decoded bytes. Trailing unpaired nibble is zero-padded.
pub(crate) fn read_hex_string(src: &[u8], pos: &mut usize) -> Vec<u8> {
    *pos += 1; // skip '<'
    decode_hex_digits(src, pos)
}

/// Decode hexadecimal digit pairs from `pos` up to and including the next
/// `>`, or to the end of `src`, skipping PDF whitespace; `*pos` ends after
/// the `>`.  A final unpaired digit is read as if followed by `0`, and a
/// byte that is not a hex digit reads as `0` (see [`hex_nibble`]).
///
/// Shared by hex strings (ISO 32000-2 §7.3.4.3) and the `ASCIIHexDecode`
/// filter (§7.4.2), which have the same syntax.
pub(crate) fn decode_hex_digits(src: &[u8], pos: &mut usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut hi: Option<u8> = None;
    while *pos < src.len() {
        let b = src[*pos];
        *pos += 1;
        if b == b'>' {
            break;
        }
        if is_ws(b) {
            continue;
        }
        let nibble = hex_nibble(b);
        match hi {
            None => hi = Some(nibble),
            Some(h) => {
                out.push((h << 4) | nibble);
                hi = None;
            }
        }
    }
    if let Some(h) = hi {
        out.push(h << 4);
    }
    out
}

/// Whether `b` can occur in a number token: digits, signs, the decimal
/// point, and the exponent markers `e`/`E`.  Exponents are not PDF syntax
/// (ISO 32000-2 §7.3.3) and are accepted as an extension.
#[inline]
pub(crate) const fn is_number_byte(b: u8) -> bool {
    matches!(b, b'0'..=b'9' | b'+' | b'-' | b'.' | b'e' | b'E')
}

/// The value of a number token, or `None` when `word` is not one.
///
/// A word is a number token when it starts with a digit, sign or decimal
/// point and consists of digits, signs, decimal points and the exponent
/// markers `e`/`E` only.  A well-formed token
/// has its exact value; a malformed one (`--5`, `.5.`, `5-`, `34.5-6`) has
/// the value of its longest prefix of the form `[+-]digits[.digits]`, or 0
/// when that prefix holds no digit.  Values are clamped to the
/// single-precision range of ISO 32000-2 Annex C, so the result is finite.
#[must_use]
pub fn parse_number(word: &[u8]) -> Option<f64> {
    let first = *word.first()?;
    if !matches!(first, b'0'..=b'9' | b'+' | b'-' | b'.')
        || !word.iter().all(|&b| is_number_byte(b))
    {
        return None;
    }
    // The bytes are ASCII, so every prefix is UTF-8.
    let parse = |w: &[u8]| std::str::from_utf8(w).ok()?.parse::<f64>().ok();
    let value = parse(word).unwrap_or_else(|| {
        let digits = |from: usize| {
            word[from..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count()
        };
        let mut end = usize::from(matches!(first, b'+' | b'-'));
        end += digits(end);
        if word.get(end) == Some(&b'.') {
            end += 1;
            end += digits(end);
        }
        parse(&word[..end]).unwrap_or(0.0)
    });
    let max = f64::from(f32::MAX);
    Some(value.clamp(-max, max))
}

/// Scan a run of non-whitespace, non-delimiter bytes; return it as a slice.
/// Does NOT advance past the terminator.
pub(crate) fn scan_token<'a>(src: &'a [u8], pos: &mut usize) -> &'a [u8] {
    let start = *pos;
    while *pos < src.len() {
        let b = src[*pos];
        if is_ws(b) || is_delim(b) {
            break;
        }
        *pos += 1;
    }
    &src[start..*pos]
}

/// Parse a decimal non-negative integer at `pos`.  Returns `None` if no
/// digit is present.  Does not advance past the terminator.
pub(crate) fn parse_u64(src: &[u8], pos: &mut usize) -> Option<u64> {
    let start = *pos;
    while *pos < src.len() && src[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return None;
    }
    std::str::from_utf8(&src[start..*pos])
        .ok()
        .and_then(|s| s.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_formed_numbers_parse_exactly() {
        for (word, value) in [
            (&b"42"[..], 42.0),
            (b"-7", -7.0),
            (b"+3.5", 3.5),
            (b".5", 0.5),
            (b"4.", 4.0),
            (b"-.002", -0.002),
            (b"1e3", 1000.0),
        ] {
            assert_eq!(parse_number(word), Some(value), "{word:?}");
        }
    }

    /// A malformed number token takes the value of its longest well-formed
    /// prefix, or 0.
    #[test]
    fn malformed_numbers_take_their_longest_valid_prefix() {
        for (word, value) in [
            (&b"--5"[..], 0.0),
            (b".5.", 0.5),
            (b"5-", 5.0),
            (b"34.5-6", 34.5),
            (b"-", 0.0),
            (b".", 0.0),
            (b"1e", 1.0),
            (b"1.2.3", 1.2),
        ] {
            assert_eq!(parse_number(word), Some(value), "{word:?}");
        }
    }

    /// Words outside the number alphabet, including spellings of infinity
    /// and NaN, are not numbers; out-of-range values are clamped.
    #[test]
    fn non_numbers_are_rejected_and_values_stay_finite() {
        for word in [&b"inf"[..], b"-inf", b"NaN", b"e5", b"re", b"5a", b""] {
            assert_eq!(parse_number(word), None, "{word:?}");
        }
        let max = f64::from(f32::MAX);
        assert_eq!(parse_number(b"1e400"), Some(max));
        assert_eq!(parse_number(b"-1e400"), Some(-max));
    }
}
