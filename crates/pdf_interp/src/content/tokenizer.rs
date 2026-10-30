//! PDF content stream tokenizer.
//!
//! Converts raw content stream bytes into a sequence of [`Token`]s.
//! The PDF content stream grammar is simple: zero or more operand tokens
//! followed by a single operator keyword, repeated until end-of-stream.
//!
//! Inline image data (`BI … ID … EI`) is handled as a single [`Token::InlineImage`]
//! so the caller never sees a bare `BI` operator keyword.

use crate::resources::image::inline::{inline_data_len, parse_inline_params};

/// A single token from a PDF content stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    /// Integer or real number operand.
    Number(f64),
    /// Boolean literal (`true` / `false`).
    Bool(bool),
    /// Name operand (`/Foo`) — slice into the source buffer, no leading `/`.
    Name(&'a [u8]),
    /// Literal string operand (`(…)`) or hex string (`<…>`), decoded.
    String(Vec<u8>),
    /// Array operand (`[…]`), elements already decoded.
    Array(Vec<Self>),
    /// Dictionary operand (`<< … >>`): keys and values alternate, each
    /// already decoded; well-formed input has a `Name` at every even index.
    Dict(Vec<Self>),
    /// Operator keyword — slice into the source buffer.
    Op(&'a [u8]),
    /// Inline image: parameter dictionary bytes + raw pixel data bytes.
    InlineImage {
        /// Raw bytes of the inline-image parameter dict (between `BI` and `ID`).
        params: &'a [u8],
        /// Raw image data bytes (between `ID` and `EI`).
        data: &'a [u8],
    },
}

/// Maximum array/dictionary nesting depth read as structure.
///
/// An opener at this depth yields an empty container and its contents are
/// read as elements of the enclosing container, so recursion depth (and stack
/// use) is bounded by this constant regardless of input.
const MAX_NESTING: usize = 256;

/// The component count of a colour space named from the `/ColorSpace`
/// resources, for an inline image's data length; `None` for a name the
/// resources do not define.
pub type InlineComponents<'a> = &'a dyn Fn(&[u8]) -> Option<usize>;

/// Iterates over tokens in a PDF content stream byte slice.
///
/// The tokenizer borrows its source slice for its lifetime; zero-copy for
/// names and operator keywords.
pub struct Tokenizer<'a> {
    src: &'a [u8],
    pos: usize,
    /// Number of arrays and dictionaries currently open.
    depth: usize,
    /// Resolves a named colour space of an inline image to its component
    /// count (see [`inline_data_len`]).
    inline_components: Option<InlineComponents<'a>>,
}

impl<'a> Tokenizer<'a> {
    /// Create a tokenizer over `src`.
    #[must_use]
    pub const fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            depth: 0,
            inline_components: None,
        }
    }

    /// The same tokenizer, resolving an inline image's named colour space
    /// through `components` when computing the image's data length.
    #[must_use]
    pub const fn with_inline_components(mut self, components: InlineComponents<'a>) -> Self {
        self.inline_components = Some(components);
        self
    }

    fn remaining(&self) -> &'a [u8] {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    const fn advance(&mut self) {
        if self.pos < self.src.len() {
            self.pos += 1;
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                // PDF §7.2.2: NUL (0x00) is also whitespace.
                Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C' | b'\0') => self.advance(),
                Some(b'%') => {
                    // Comment runs to end of line (CR, LF, or CRLF).
                    while !matches!(self.peek(), None | Some(b'\r' | b'\n')) {
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn read_name(&mut self) -> &'a [u8] {
        // Caller has verified the current byte is '/'; skip it.
        self.advance();
        let start = self.pos;
        while let Some(b) = self.peek() {
            if is_delimiter(b) || is_whitespace(b) {
                break;
            }
            self.advance();
        }
        &self.src[start..self.pos]
    }

    fn read_literal_string(&mut self) -> Vec<u8> {
        // Caller has verified the current byte is '('; skip it.
        self.advance();
        let mut out = Vec::new();
        let mut depth = 1usize;
        while let Some(b) = self.peek() {
            self.advance();
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
                b'\\' => match self.peek() {
                    Some(b'n') => {
                        self.advance();
                        out.push(b'\n');
                    }
                    Some(b'r') => {
                        self.advance();
                        out.push(b'\r');
                    }
                    Some(b't') => {
                        self.advance();
                        out.push(b'\t');
                    }
                    Some(b'b') => {
                        self.advance();
                        out.push(b'\x08');
                    }
                    Some(b'f') => {
                        self.advance();
                        out.push(b'\x0C');
                    }
                    Some(b'(') => {
                        self.advance();
                        out.push(b'(');
                    }
                    Some(b')') => {
                        self.advance();
                        out.push(b')');
                    }
                    Some(b'\\') => {
                        self.advance();
                        out.push(b'\\');
                    }
                    Some(b'\r') => {
                        self.advance();
                        // CRLF line continuation — skip both bytes.
                        if self.peek() == Some(b'\n') {
                            self.advance();
                        }
                    }
                    Some(b'\n') => {
                        self.advance();
                    }
                    Some(d) if d.is_ascii_digit() && d < b'8' => {
                        // Up to 3 octal digits (PDF §7.3.4.2).
                        // Max octal value is \377 (255); mask to u8 is safe.
                        let mut val: u16 = 0;
                        for _ in 0..3 {
                            match self.peek() {
                                Some(c) if c.is_ascii_digit() && c < b'8' => {
                                    val = val * 8 + u16::from(c - b'0');
                                    self.advance();
                                }
                                _ => break,
                            }
                        }
                        // Truncate to low byte; values >255 are malformed PDF.
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "intentional: malformed PDF octal values >255 are truncated per spec leniency"
                        )]
                        out.push(val as u8);
                    }
                    _ => {} // Unrecognised escape — PDF spec says ignore the backslash.
                },
                other => out.push(other),
            }
        }
        out
    }

    fn read_hex_string(&mut self) -> Vec<u8> {
        // Caller has verified the current byte is '<'; skip it.
        self.advance();
        let mut out = Vec::new();
        let mut hi: Option<u8> = None;
        while let Some(b) = self.peek() {
            self.advance();
            if b == b'>' {
                break;
            }
            if is_whitespace(b) {
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
        // A trailing unpaired nibble is treated as the high nibble with low=0
        // (PDF §7.3.4.3).
        if let Some(h) = hi {
            out.push(h << 4);
        }
        out
    }

    fn read_array(&mut self) -> Vec<Token<'a>> {
        // Caller has verified the current byte is '['; skip it.
        self.advance();
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            match self.peek() {
                None => break,
                Some(b']') => {
                    self.advance();
                    break;
                }
                _ => {
                    if let Some(tok) = self.next_token() {
                        items.push(tok);
                    }
                }
            }
        }
        items
    }

    /// Read a `<< … >>` dictionary; the current bytes are `<<`.
    ///
    /// Entries are read with [`Self::next_token`], so strings, hex strings,
    /// arrays and nested dictionaries inside it are delimited by their own
    /// grammar rather than by counting angle brackets.  A stray `>` is
    /// skipped; end of input closes the dictionary.
    fn read_dict(&mut self) -> Vec<Token<'a>> {
        self.pos += 2;
        let mut items = Vec::new();
        loop {
            self.skip_whitespace_and_comments();
            if self.remaining().starts_with(b">>") {
                self.pos += 2;
                break;
            }
            match self.peek() {
                None => break,
                Some(b'>') => self.advance(),
                _ => {
                    if let Some(tok) = self.next_token() {
                        items.push(tok);
                    }
                }
            }
        }
        items
    }

    /// Read an inline image block starting immediately after the `BI` keyword.
    ///
    /// The PDF spec (§8.9.7) requires:
    /// - Parameter key/value pairs between `BI` and `ID`
    /// - A single whitespace byte immediately after `ID`
    /// - Image data bytes up to `EI`, which must be preceded by whitespace
    ///
    /// When the parameters determine the data length (see
    /// [`inline_data_len`]) and `EI` follows that many bytes, the data is
    /// exactly those bytes; otherwise it ends at the first `EI` preceded by
    /// whitespace.
    ///
    /// On a malformed stream (no `ID` or no `EI`), returns the bytes consumed
    /// so far rather than panicking.
    fn read_inline_image(&mut self) -> Token<'a> {
        // Scan for the `ID` keyword. We do NOT call skip_whitespace_and_comments
        // here because we need the raw param bytes to include any whitespace
        // that is part of the dict; `param_end` is set at the start of `ID`.
        let param_start = self.pos;
        let param_end = loop {
            if self.pos >= self.src.len() {
                break self.pos;
            }
            if self.remaining().starts_with(b"ID")
                && self
                    .remaining()
                    .get(2)
                    .is_none_or(|&b| is_whitespace(b) || is_delimiter(b))
            {
                let end = self.pos;
                self.pos += 2; // consume `ID`
                // Consume exactly one whitespace byte after `ID` (PDF §8.9.7).
                if matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n' | b'\x0C')) {
                    self.advance();
                }
                break end;
            }
            self.pos += 1;
        };

        let data_start = self.pos;
        let params = &self.src[param_start..param_end];

        // Data whose length the parameters determine ends there, when `EI`
        // follows it after optional whitespace; its bytes may then contain
        // `EI` themselves.
        let declared_end = inline_data_len(&parse_inline_params(params), self.inline_components)
            .and_then(|len| data_start.checked_add(len))
            .filter(|&end| end <= self.src.len());
        if let Some(end) = declared_end
            && let Some(after) = self.ei_at(end)
        {
            self.pos = after;
            return Token::InlineImage {
                params,
                data: &self.src[data_start..end],
            };
        }

        // Otherwise scan for `EI` preceded by whitespace. The byte *before*
        // `EI` is whitespace and is not part of the image data.
        let data_end = loop {
            if self.pos >= self.src.len() {
                break self.pos;
            }
            // Check that the byte before `EI` is whitespace; at pos==data_start
            // there is no preceding byte, so we treat that as whitespace (pos 0
            // relative to data).
            let prev_is_ws = self.pos == data_start || is_whitespace(self.src[self.pos - 1]);
            if prev_is_ws
                && self.remaining().starts_with(b"EI")
                && self
                    .remaining()
                    .get(2)
                    .is_none_or(|&b| is_whitespace(b) || is_delimiter(b))
            {
                // data_end is the position of the preceding whitespace byte,
                // which is not part of the image data.
                // `EI` directly at `data_start` ends empty data.
                let end = self.pos.saturating_sub(1).max(data_start);
                self.pos += 2; // consume `EI`
                break end;
            }
            self.pos += 1;
        };

        Token::InlineImage {
            params,
            data: &self.src[data_start..data_end],
        }
    }

    /// The position just past an `EI` keyword that follows `at` after
    /// optional whitespace, or `None` when none does.
    fn ei_at(&self, at: usize) -> Option<usize> {
        let rest = self.src.get(at..)?;
        let ws = rest.iter().take_while(|&&b| is_whitespace(b)).count();
        let after = at + ws + 2;
        (rest[ws..].starts_with(b"EI")
            && self
                .src
                .get(after)
                .is_none_or(|&b| is_whitespace(b) || is_delimiter(b)))
        .then_some(after)
    }

    /// Return the next token, or `None` at end of stream.
    ///
    /// Skipped bytes (stray `]`, lone delimiters) are consumed in a loop, not
    /// by re-entering this method, so an adversarial stream of millions of
    /// stray delimiters cannot drive unbounded call depth and a stack overflow.
    pub fn next_token(&mut self) -> Option<Token<'a>> {
        loop {
            self.skip_whitespace_and_comments();
            let b = self.peek()?;

            // Each loop iteration either returns a token or strictly advances
            // `self.pos` before re-scanning (the stray-`]` and empty-word skip
            // paths both `advance()`), guaranteeing termination on any finite
            // input. `before` anchors the per-iteration progress assertions.
            let before = self.pos;

            match b {
                b'/' => return Some(Token::Name(self.read_name())),
                b'(' => return Some(Token::String(self.read_literal_string())),
                b'<' if self.src.get(self.pos + 1) == Some(&b'<') => {
                    if self.depth >= MAX_NESTING {
                        self.pos += 2;
                        return Some(Token::Dict(Vec::new()));
                    }
                    self.depth += 1;
                    let items = self.read_dict();
                    self.depth -= 1;
                    return Some(Token::Dict(items));
                }
                b'<' => return Some(Token::String(self.read_hex_string())),
                b'[' => {
                    if self.depth >= MAX_NESTING {
                        self.advance();
                        return Some(Token::Array(Vec::new()));
                    }
                    self.depth += 1;
                    let items = self.read_array();
                    self.depth -= 1;
                    return Some(Token::Array(items));
                }
                b']' => {
                    // Stray closing bracket — skip and re-scan on the next
                    // loop iteration. Looping (not recursing) keeps a stream of
                    // millions of stray `]` from overflowing the stack.
                    self.advance();
                    debug_assert!(self.pos > before, "tokenizer must advance past stray ']'");
                }
                _ => {
                    // Number, boolean, or operator keyword — all start with an
                    // ASCII character that is not a delimiter.
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if is_whitespace(c) || is_delimiter(c) {
                            break;
                        }
                        self.advance();
                    }
                    let word = &self.src[start..self.pos];

                    // An unmatched delimiter (e.g. a stray `>` left over when
                    // the `<<…>>` scanner's depth counter is thrown off by a
                    // hex-string `<…>` nested inside a marked-content dict)
                    // yields an empty word: the loop above breaks immediately
                    // on the delimiter. Skip the byte and re-scan on the next
                    // iteration rather than emitting a zero-length `Op` without
                    // advancing — that stalled the tokenizer and grew the `ops`
                    // Vec in `parse` without bound until OOM.
                    if word.is_empty() {
                        self.advance();
                        debug_assert!(
                            self.pos > before,
                            "tokenizer must advance past unrecognised delimiter"
                        );
                        continue;
                    }

                    if word == b"true" {
                        return Some(Token::Bool(true));
                    }
                    if word == b"false" {
                        return Some(Token::Bool(false));
                    }

                    // A word of number characters is an operand even when
                    // malformed, never an operator.
                    if let Some(n) = pdf::parse_number(word) {
                        return Some(Token::Number(n));
                    }

                    // Inline image: consume params + data into a single token so
                    // the caller never has to handle a bare `BI` operator.
                    if word == b"BI" {
                        return Some(self.read_inline_image());
                    }

                    return Some(Token::Op(word));
                }
            }
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

/// Returns `true` for the six PDF whitespace characters (PDF §7.2.2).
const fn is_whitespace(b: u8) -> bool {
    matches!(b, b'\0' | b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
}

/// Returns `true` for the ten PDF delimiter characters (PDF §7.2.2).
const fn is_delimiter(b: u8) -> bool {
    matches!(
        b,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

/// Convert a single ASCII hex character to its nibble value.
/// Non-hex characters return 0 (matches Acrobat's lenient behaviour).
const fn hex_nibble(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(src: &[u8]) -> Vec<Token<'_>> {
        Tokenizer::new(src).collect()
    }

    #[test]
    fn numbers() {
        let t = tokens(b"1 2.5 -3 +4.0");
        assert_eq!(t[0], Token::Number(1.0));
        assert_eq!(t[1], Token::Number(2.5));
        assert_eq!(t[2], Token::Number(-3.0));
        assert_eq!(t[3], Token::Number(4.0));
    }

    #[test]
    fn operator() {
        let t = tokens(b"1 0 0 1 100 200 cm");
        assert_eq!(t.last(), Some(&Token::Op(b"cm")));
        assert_eq!(t.len(), 7);
    }

    #[test]
    fn name() {
        let t = tokens(b"/Helvetica");
        assert_eq!(t[0], Token::Name(b"Helvetica"));
    }

    #[test]
    fn literal_string() {
        let t = tokens(b"(hello world)");
        assert_eq!(t[0], Token::String(b"hello world".to_vec()));
    }

    #[test]
    fn literal_string_nested_parens() {
        let t = tokens(b"(a(b)c)");
        assert_eq!(t[0], Token::String(b"a(b)c".to_vec()));
    }

    #[test]
    fn literal_string_octal_escape() {
        // \110 = 0o110 = 72 = 'H'
        let t = tokens(b"(\\110i)");
        assert_eq!(t[0], Token::String(b"Hi".to_vec()));
    }

    #[test]
    fn hex_string() {
        let t = tokens(b"<48656c6c6f>");
        assert_eq!(t[0], Token::String(b"Hello".to_vec()));
    }

    #[test]
    fn hex_string_trailing_nibble() {
        // <9> → high nibble 9, low nibble 0 → 0x90
        let t = tokens(b"<9>");
        assert_eq!(t[0], Token::String(vec![0x90]));
    }

    #[test]
    fn comment_skipped() {
        let t = tokens(b"% this is a comment\n1 2 m");
        assert_eq!(t.len(), 3);
        assert_eq!(t[2], Token::Op(b"m"));
    }

    #[test]
    fn array() {
        let t = tokens(b"[1 2 3]");
        assert_eq!(
            t[0],
            Token::Array(vec![
                Token::Number(1.0),
                Token::Number(2.0),
                Token::Number(3.0)
            ])
        );
    }

    #[test]
    fn booleans() {
        let t = tokens(b"true false");
        assert_eq!(t[0], Token::Bool(true));
        assert_eq!(t[1], Token::Bool(false));
    }

    #[test]
    fn path_sequence() {
        // 100 200 m 300 400 l S → 4 numbers + 3 ops = 7 tokens
        let t = tokens(b"100 200 m 300 400 l S");
        assert_eq!(t.len(), 7);
        assert_eq!(t[6], Token::Op(b"S"));
    }

    #[test]
    fn inline_image() {
        // PDF spec §8.9.7: EI must be preceded by a whitespace byte.
        let src = b"BI /W 1 /H 1 /CS /G /BPC 8 ID \xFF EI";
        let t = tokens(src);
        assert_eq!(t.len(), 1);
        match &t[0] {
            Token::InlineImage { data, .. } => assert_eq!(*data, b"\xFF"),
            other => panic!("expected InlineImage, got {other:?}"),
        }
    }

    #[test]
    fn inline_image_data_may_contain_ei() {
        let src = b"BI /W 4 /H 1 /CS /G /BPC 8 ID  EI \nEI Q";
        let t = tokens(src);
        assert_eq!(t.len(), 2);
        match &t[0] {
            Token::InlineImage { data, .. } => assert_eq!(*data, b" EI "),
            other => panic!("expected InlineImage, got {other:?}"),
        }
        assert_eq!(t[1], Token::Op(b"Q"));
    }

    #[test]
    fn inline_image_declared_length_not_followed_by_ei_falls_back_to_the_scan() {
        // Four bytes declared, two present before `EI`.
        let src = b"BI /W 4 /H 1 /CS /G /BPC 8 ID ab EI Q";
        let t = tokens(src);
        assert_eq!(t.len(), 2);
        match &t[0] {
            Token::InlineImage { data, .. } => assert_eq!(*data, b"ab"),
            other => panic!("expected InlineImage, got {other:?}"),
        }
    }

    #[test]
    fn inline_image_empty_data() {
        let t = tokens(b"BI ID EI Q");
        assert_eq!(t.len(), 2);
        match &t[0] {
            Token::InlineImage { data, .. } => assert_eq!(data.len(), 0),
            other => panic!("expected InlineImage, got {other:?}"),
        }
    }

    #[test]
    fn empty_stream() {
        assert_eq!(tokens(b""), vec![]);
    }

    #[test]
    fn only_comment() {
        assert_eq!(tokens(b"% nothing\n"), vec![]);
    }

    /// A lone `>` in a content stream (left over from a `<<…>>` dict whose
    /// `>>` depth-counting was thrown off by a nested `<hex-string>`) must not
    /// cause an infinite loop.  The stray `>` is a delimiter and the default
    /// arm of `next_token` would produce an empty word without advancing `pos`,
    /// stalling the tokenizer forever.  The fix: skip the unrecognised byte
    /// and continue.
    #[test]
    fn stray_gt_is_skipped_without_infinite_loop() {
        // `>` is a delimiter; the tokenizer must skip it and still return
        // the following tokens normally.
        let t = tokens(b"> /Name cm");
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], Token::Name(b"Name"));
        assert_eq!(t[1], Token::Op(b"cm"));
    }

    /// Reproduces the glfmm livret OOM: a `<</ActualText<FEFF00A0>>>` marked-
    /// content dict is followed by real operators.  The `<<…>>` scanner exits
    /// one `>` early (the hex-string's closing `>` matches before the outer
    /// dict's `>>`), leaving a lone `>` that previously caused an infinite loop.
    #[test]
    fn marked_content_dict_with_hex_string_terminates() {
        let src = b"/Span<</ActualText<FEFF00A0>>> BDC\n( )Tj\nEMC\n";
        // Should terminate (finite token count) — the exact token sequence is
        // not load-bearing, only that we don't loop forever.
        let t = tokens(src);
        assert_ne!(t.len(), 0);
        // The last real operators must be present.
        let ops: Vec<_> = t
            .iter()
            .filter_map(|tok| {
                if let Token::Op(kw) = tok {
                    Some(*kw)
                } else {
                    None
                }
            })
            .collect();
        assert!(ops.contains(&b"BDC".as_ref()), "BDC must appear: {ops:?}");
        assert!(ops.contains(&b"Tj".as_ref()), "Tj must appear: {ops:?}");
        assert!(ops.contains(&b"EMC".as_ref()), "EMC must appear: {ops:?}");
    }

    /// `>>`, `]`, `)` and `>` are delimiters, so the operator that follows
    /// them needs no whitespace (ISO 32000-2 §7.2.3).
    #[test]
    fn operator_directly_after_closing_delimiter() {
        assert_eq!(
            tokens(b"/P <</MCID 0 >>BDC"),
            vec![
                Token::Name(b"P"),
                Token::Dict(vec![Token::Name(b"MCID"), Token::Number(0.0)]),
                Token::Op(b"BDC"),
            ]
        );
        assert_eq!(
            tokens(b"[(a)1]TJ"),
            vec![
                Token::Array(vec![Token::String(b"a".to_vec()), Token::Number(1.0)]),
                Token::Op(b"TJ"),
            ]
        );
        assert_eq!(
            tokens(b"(a)Tj<62>Tj"),
            vec![
                Token::String(b"a".to_vec()),
                Token::Op(b"Tj"),
                Token::String(b"b".to_vec()),
                Token::Op(b"Tj"),
            ]
        );
    }

    /// Hex strings and nested dictionaries inside a dictionary close where
    /// their own grammar says, not where an angle-bracket count would.
    #[test]
    fn dict_with_hex_string_and_nested_dict() {
        assert_eq!(
            tokens(b"/Span<</ActualText<FEFF00A0>/A<</B[1]>>>>BDC EMC"),
            vec![
                Token::Name(b"Span"),
                Token::Dict(vec![
                    Token::Name(b"ActualText"),
                    Token::String(vec![0xFE, 0xFF, 0x00, 0xA0]),
                    Token::Name(b"A"),
                    Token::Dict(vec![
                        Token::Name(b"B"),
                        Token::Array(vec![Token::Number(1.0)])
                    ]),
                ]),
                Token::Op(b"BDC"),
                Token::Op(b"EMC"),
            ]
        );
    }

    /// A long run of stray delimiters (the exact bytes an unbalanced `<<…>>`
    /// scanner can leave behind, repeated) must not recurse per byte. Skipping
    /// is done by looping inside `next_token`, not by re-entering it, so even a
    /// half-million stray delimiters tokenize on a 256 KiB stack without a
    /// stack overflow. Run on a deliberately small-stack thread: before the
    /// loop conversion this recursed once per skipped byte and aborted.
    #[test]
    fn millions_of_stray_delimiters_do_not_overflow_the_stack() {
        let handle = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let src = vec![b'>'; 500_000];
                Tokenizer::new(&src).count()
            })
            .expect("spawn small-stack tokenizer thread");
        let n = handle
            .join()
            .expect("tokenizer must not overflow the stack on stray delimiters");
        assert_eq!(n, 0, "all stray `>` are skipped, yielding no tokens");
    }

    /// A malformed number is an operand with the value of its longest
    /// well-formed prefix, never an operator that would clear the operands
    /// before it; a word outside the number alphabet is an operator.
    #[test]
    fn malformed_numbers_are_operands() {
        assert_eq!(
            tokens(b"10 --5 80- .5. inf re"),
            vec![
                Token::Number(10.0),
                Token::Number(0.0),
                Token::Number(80.0),
                Token::Number(0.5),
                Token::Op(b"inf"),
                Token::Op(b"re"),
            ]
        );
    }

    /// Nesting depth of `tok` (a non-container token is depth 0).
    fn nesting_depth(tok: &Token<'_>) -> usize {
        match tok {
            Token::Array(items) | Token::Dict(items) => {
                1 + items.iter().map(nesting_depth).max().unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// A million unclosed `[` or `<<` must not recurse once per opener: past
    /// `MAX_NESTING` an opener yields an empty container, so the tokenizer
    /// finishes on a 2 MiB stack.
    #[test]
    fn unbounded_nesting_does_not_overflow_the_stack() {
        for opener in [&b"["[..], &b"<<"[..]] {
            let mut src = opener.repeat(1_000_000);
            src.extend_from_slice(b" 0 g");
            let handle = std::thread::Builder::new()
                .stack_size(2 * 1024 * 1024)
                .spawn(move || {
                    let toks: Vec<Token<'_>> = Tokenizer::new(&src).collect();
                    toks.iter().map(nesting_depth).max().unwrap_or(0)
                })
                .expect("spawn tokenizer thread");
            let depth = handle
                .join()
                .expect("tokenizer must not overflow the stack on deep nesting");
            assert_eq!(
                depth,
                MAX_NESTING + 1,
                "openers past the cap yield empty containers"
            );
        }
    }

    /// Nesting up to `MAX_NESTING` levels is read as structure, and the
    /// tokens after a balanced over-deep run are unaffected.
    #[test]
    fn nesting_at_the_cap_is_preserved() {
        let mut src = b"[".repeat(MAX_NESTING);
        src.extend_from_slice(&b"]".repeat(MAX_NESTING));
        src.extend_from_slice(b" 1 g");
        let toks = tokens(&src);
        assert_eq!(nesting_depth(&toks[0]), MAX_NESTING);
        assert_eq!(&toks[1..], &[Token::Number(1.0), Token::Op(b"g")]);
    }
}
