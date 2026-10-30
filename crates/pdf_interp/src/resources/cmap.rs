//! PDF `CMap` parser and decoder — character code → `CID` and
//! `ToUnicode` mappings (PDF 32000-1 §9.7.5, §9.7.6).
//!
//! A Type 0 composite font maps *character codes* (1–4 bytes each) to
//! *CIDs* through its `Encoding` `CMap`: a stream, or the name of a
//! predefined `CMap` ([`super::cmap_predefined`]).  The CIDs select glyphs
//! through the descendant font's `CIDToGIDMap` or charset.  A `ToUnicode`
//! `CMap` has the same syntax with Unicode values.
//!
//! # `CMap` syntax
//!
//! ```text
//! /90ms-RKSJ-H usecmap        % inherit another CMap's ranges and mappings
//! begincodespacerange
//! <00> <80>                   % single-byte codes
//! <8140> <FFFF>               % two-byte codes
//! endcodespacerange
//! begincidchar
//! <20> 32                     % code 0x20 → CID 32
//! endcidchar
//! begincidrange
//! <0020> <0080> 100           % codes 0x0020–0x0080 → CIDs 100–196
//! endcidrange
//! beginbfchar                 % ToUnicode variant
//! <0041> <0041>
//! endbfchar
//! beginbfrange
//! <0020> <007E> <0020>
//! endbfrange
//! ```
//!
//! # Code extraction
//!
//! The codespace ranges partition the byte string into codes
//! (§9.7.6.3).  A range is byte-wise rectangular: a code of its length
//! belongs to it when every byte lies between the corresponding bytes of
//! its bounds.  At each position the shortest length whose prefix is a
//! member of a range is taken; when no length matches, the length of the
//! shortest range that contains the first byte, else one byte.

use std::collections::HashMap;
use std::sync::Arc;

use super::cmap_predefined::predefined_cmap_nested;

// ── Public types ──────────────────────────────────────────────────────────────

/// One codespace range (§9.7.6.2): codes of `bytes` bytes whose every
/// byte lies within the corresponding bytes of `lo` and `hi`.  Only the
/// first `bytes` entries of `lo` and `hi` are meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Codespace {
    /// Code length in bytes, `1..=4`.
    pub bytes: u8,
    /// Lower bound, big-endian, left-aligned.
    pub lo: [u8; 4],
    /// Upper bound, big-endian, left-aligned.
    pub hi: [u8; 4],
}

impl Codespace {
    /// The full range of `bytes`-byte codes.
    const fn full(bytes: u8) -> Self {
        Self {
            bytes,
            lo: [0; 4],
            hi: [0xFF; 4],
        }
    }

    /// Whether `prefix`, the leading bytes of a code, lies within the
    /// range byte-wise.
    fn contains_prefix(&self, prefix: &[u8]) -> bool {
        prefix.len() <= usize::from(self.bytes)
            && prefix
                .iter()
                .zip(&self.lo)
                .zip(&self.hi)
                .all(|((b, lo), hi)| lo <= b && b <= hi)
    }
}

/// One character code extracted from a string, with the value it maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedCode {
    /// The code, big-endian.
    pub code: u32,
    /// The CID (or Unicode value) the code maps to; 0 when unmapped.
    pub value: u32,
    /// The number of bytes the code occupied, `1..=4`.
    pub len: u8,
}

/// A resolved `CMap`: the codespace ranges that split a string into codes,
/// and the value (CID or Unicode) of each code.
#[derive(Debug, Clone, Default)]
pub struct CMap {
    /// Codespace ranges, in definition order.
    pub codespaces: Vec<Codespace>,
    /// Code → CID (or Unicode) for every explicitly mapped code.
    pub map: HashMap<u32, u32>,
    /// A code absent from `map` maps to itself (`Identity-H` / `Identity-V`,
    /// and every `CMap` built on them with `usecmap`).
    pub identity: bool,
}

impl CMap {
    /// The two-byte identity `CMap` (`Identity-H` / `Identity-V`).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            codespaces: vec![Codespace::full(2)],
            map: HashMap::new(),
            identity: true,
        }
    }

    /// The value `code` maps to: its explicit mapping, else itself for an
    /// identity `CMap`, else 0.
    #[must_use]
    pub fn lookup(&self, code: u32) -> u32 {
        self.map
            .get(&code)
            .copied()
            .unwrap_or(if self.identity { code } else { 0 })
    }

    /// The code at the start of `bytes` and its length, by the extraction
    /// rules of §9.7.6.3.  `bytes` must not be empty.
    #[must_use]
    pub fn next_code(&self, bytes: &[u8]) -> (u32, u8) {
        let available = bytes.len().min(4);
        let full_match = (1..=available).find(|&n| {
            self.codespaces
                .iter()
                .any(|r| usize::from(r.bytes) == n && r.contains_prefix(&bytes[..n]))
        });
        let len = full_match.unwrap_or_else(|| {
            self.codespaces
                .iter()
                .filter(|r| r.contains_prefix(&bytes[..1]))
                .map(|r| usize::from(r.bytes))
                .min()
                .unwrap_or(1)
                .min(available)
        });
        #[expect(clippy::cast_possible_truncation, reason = "len is at most 4")]
        (bytes_to_u32(&bytes[..len]), len as u8)
    }

    /// Decode every code in `bytes`.
    #[must_use]
    pub const fn iter_codes<'a>(&'a self, bytes: &'a [u8]) -> CMapIter<'a> {
        CMapIter { cmap: self, bytes }
    }

    /// Incorporate `parent` as by `usecmap`: its codespace ranges and
    /// mappings apply wherever this `CMap` defines none of its own.
    pub fn inherit(&mut self, parent: &Self) {
        for range in &parent.codespaces {
            if !self.codespaces.contains(range) {
                self.codespaces.push(*range);
            }
        }
        for (&code, &value) in &parent.map {
            let _ = self.map.entry(code).or_insert(value);
        }
        self.identity |= parent.identity;
    }
}

/// Iterator over the codes of a byte string, produced by [`CMap::iter_codes`].
pub struct CMapIter<'a> {
    cmap: &'a CMap,
    bytes: &'a [u8],
}

impl Iterator for CMapIter<'_> {
    type Item = DecodedCode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            return None;
        }
        let (code, len) = self.cmap.next_code(self.bytes);
        self.bytes = &self.bytes[usize::from(len)..];
        Some(DecodedCode {
            code,
            value: self.cmap.lookup(code),
            len,
        })
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Greatest `usecmap` nesting followed while parsing.
const MAX_USECMAP_DEPTH: u32 = 4;

/// Parse a `CMap` stream.
///
/// Handles encoding `CMaps` (`begincidchar` / `begincidrange`) and
/// `ToUnicode` `CMaps` (`beginbfchar` / `beginbfrange`), and `usecmap` of
/// a predefined `CMap`.  A `CMap` that declares no codespace range takes
/// the full range of every code length its mappings use.
///
/// Returns `None` when the stream contains no recognised directive, in
/// which case callers fall back to an identity mapping.
#[must_use]
pub fn parse_cmap(stream: &[u8]) -> Option<CMap> {
    parse_cmap_nested(stream, 0)
}

/// [`parse_cmap`] at `usecmap` nesting `depth`.
pub(super) fn parse_cmap_nested(stream: &[u8], depth: u32) -> Option<CMap> {
    let tokens = tokenise(stream);
    let mut cmap = CMap::default();
    // Code lengths used by the mappings, for a `CMap` without codespace
    // ranges: bit `n - 1` for an `n`-byte code.
    let mut code_lengths = 0u8;
    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i] {
            b"begincodespacerange" => {
                i += 1;
                while i < tokens.len() && tokens[i] != b"endcodespacerange" {
                    if let (Some(lo), Some(hi)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_hex_string(tokens.get(i + 1).copied()),
                    ) {
                        if let Some(range) = codespace(&lo, &hi) {
                            cmap.codespaces.push(range);
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            b"begincidchar" | b"beginbfchar" => {
                i += 1;
                while i < tokens.len() && !is_section_boundary(tokens[i]) {
                    if let (Some(code), Some(value)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_value(tokens.get(i + 1).copied()),
                    ) {
                        code_lengths |= length_bit(&code);
                        let _ = cmap.map.insert(bytes_to_u32(&code), value);
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                // A section that ends at the next section's start (no end
                // keyword) must not consume that keyword.
                if tokens
                    .get(i)
                    .is_none_or(|t| *t != b"endcidchar" && *t != b"endbfchar")
                {
                    continue;
                }
            }
            b"begincidrange" | b"beginbfrange" => {
                i += 1;
                while i < tokens.len() && !is_section_boundary(tokens[i]) {
                    if let (Some(lo_b), Some(hi_b), Some(base)) = (
                        parse_hex_string(tokens.get(i).copied()),
                        parse_hex_string(tokens.get(i + 1).copied()),
                        parse_value(tokens.get(i + 2).copied()),
                    ) {
                        code_lengths |= length_bit(&lo_b);
                        let lo = bytes_to_u32(&lo_b);
                        let hi = bytes_to_u32(&hi_b);
                        // A range wider than the 16-bit CID space is corrupt.
                        if hi >= lo && hi - lo < 0x1_0000 {
                            for offset in 0u32..=(hi - lo) {
                                let _ = cmap.map.insert(lo + offset, base.saturating_add(offset));
                            }
                        } else {
                            log::warn!("cmap: ignoring degenerate range {lo:04X}–{hi:04X}");
                        }
                        i += 3;
                    } else {
                        i += 1;
                    }
                }
                if tokens
                    .get(i)
                    .is_none_or(|t| *t != b"endcidrange" && *t != b"endbfrange")
                {
                    continue;
                }
            }
            b"usecmap" => {
                let parent = i
                    .checked_sub(1)
                    .and_then(|p| tokens[p].strip_prefix(b"/"))
                    .and_then(|name| parent_cmap(name, depth));
                match parent {
                    Some(parent) => cmap.inherit(&parent),
                    None => log::warn!("cmap: usecmap of an unknown or too deeply nested CMap"),
                }
            }
            _ => {}
        }
        i += 1;
    }

    if cmap.map.is_empty() && cmap.codespaces.is_empty() && !cmap.identity {
        return None;
    }
    if cmap.codespaces.is_empty() {
        // No range declared: every code length the mappings use, in full;
        // single-byte when the stream maps nothing.
        let lengths = if code_lengths == 0 { 1 } else { code_lengths };
        cmap.codespaces = (1..=4u8)
            .filter(|n| lengths & (1 << (n - 1)) != 0)
            .map(Codespace::full)
            .collect();
    }
    Some(cmap)
}

/// The `CMap` a `usecmap` at nesting `depth` names: identity, or a
/// predefined `CMap`.
fn parent_cmap(name: &[u8], depth: u32) -> Option<Arc<CMap>> {
    if depth >= MAX_USECMAP_DEPTH {
        return None;
    }
    match name {
        b"Identity-H" | b"Identity-V" => Some(Arc::new(CMap::identity())),
        _ => predefined_cmap_nested(name, depth + 1),
    }
}

/// A codespace range from its bounds; `None` when they are empty, longer
/// than four bytes, or of different lengths.
fn codespace(lo: &[u8], hi: &[u8]) -> Option<Codespace> {
    if lo.is_empty() || lo.len() > 4 || lo.len() != hi.len() {
        return None;
    }
    let mut range = Codespace::full(u8::try_from(lo.len()).ok()?);
    range.lo[..lo.len()].copy_from_slice(lo);
    range.hi[..hi.len()].copy_from_slice(hi);
    Some(range)
}

/// Bit of `code_lengths` for a code of `code.len()` bytes (clamped to 1–4).
fn length_bit(code: &[u8]) -> u8 {
    1 << (code.len().clamp(1, 4) - 1)
}

/// Whether `tok` starts or ends a mapping section (a missing end keyword
/// must not let one section consume the next).
const fn is_section_boundary(tok: &[u8]) -> bool {
    matches!(
        tok,
        b"endcidchar"
            | b"endbfchar"
            | b"endcidrange"
            | b"endbfrange"
            | b"begincidchar"
            | b"beginbfchar"
            | b"begincidrange"
            | b"beginbfrange"
            | b"endcmap"
    )
}

// ── Tokeniser ─────────────────────────────────────────────────────────────────

/// Split a `CMap` stream into tokens: keywords and names, `<hex>` strings,
/// decimal integers, and opaque `(...)` strings and `[...]` arrays.
/// Comments (`%` to end of line) are stripped.  The stream is bytes, not
/// text: comments and string literals may hold any byte.
#[expect(
    clippy::too_many_lines,
    reason = "match dispatch over 7 byte classes — extracting sub-functions would split tightly coupled index arithmetic"
)]
fn tokenise(bytes: &[u8]) -> Vec<&[u8]> {
    let mut tokens: Vec<&[u8]> = Vec::with_capacity(256);
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            // Whitespace, the second `>` of `>>`, PostScript procedure
            // braces and stray closers are all delimiters that carry no
            // token.  Each must be consumed here: the keyword arm stops at
            // them without advancing.
            b' ' | b'\t' | b'\r' | b'\n' | b'\0' | b'\x0C' | b'>' | b')' | b']' | b'{' | b'}' => {
                i += 1;
            }

            b'%' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }

            // Hex string <...>
            b'<' => {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
                tokens.push(&bytes[start..i]);
            }

            // PostScript literal string (...), captured as an opaque token.
            b'(' => {
                let start = i;
                let mut depth = 0u32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'(' => {
                            depth += 1;
                            i += 1;
                        }
                        b')' => {
                            // Depth never wraps on unbalanced input: at 0 the
                            // string ends.
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        // An escaped byte; a lone `\` at the end stops.
                        b'\\' => i += if i + 1 < bytes.len() { 2 } else { 1 },
                        _ => i += 1,
                    }
                }
                tokens.push(&bytes[start..i]);
            }

            // Array literal [...], captured as an opaque token, nested
            // arrays included.
            b'[' => {
                let start = i;
                let mut depth = 0u32;
                while i < bytes.len() {
                    match bytes[i] {
                        b'[' => {
                            depth += 1;
                            i += 1;
                        }
                        b']' => {
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                            depth -= 1;
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => i += 1,
                    }
                }
                tokens.push(&bytes[start..i]);
            }

            // Keyword, name or decimal integer.
            _ => {
                let start = i;
                while i < bytes.len()
                    && !matches!(
                        bytes[i],
                        b' ' | b'\t'
                            | b'\r'
                            | b'\n'
                            | b'\0'
                            | b'\x0C'
                            | b'<'
                            | b'>'
                            | b'('
                            | b')'
                            | b'['
                            | b']'
                            | b'{'
                            | b'}'
                            | b'%'
                    )
                {
                    i += 1;
                }
                if i > start {
                    tokens.push(&bytes[start..i]);
                }
            }
        }
    }

    tokens
}

// ── Token parsers ─────────────────────────────────────────────────────────────

/// Parse a `<hex>` token into raw bytes.  `None` for a non-hex token, an
/// empty string, or invalid hex content.
fn parse_hex_string(tok: Option<&[u8]>) -> Option<Vec<u8>> {
    let inner = tok?.strip_prefix(b"<")?.strip_suffix(b">")?;
    // Whitespace is permitted inside a hex string (PDF §7.3.4.3).
    let digits: Vec<u8> = inner
        .iter()
        .copied()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if digits.is_empty() {
        return None;
    }
    // An odd digit count is left-padded with a zero nibble.
    let padded;
    let digits: &[u8] = if digits.len().is_multiple_of(2) {
        &digits
    } else {
        padded = [b"0".as_slice(), &digits].concat();
        &padded
    };
    digits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| Some((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?))
        .collect()
}

/// Parse a value token: a decimal integer or a hex string `<...>` (a
/// big-endian Unicode value in a `bfrange` / `bfchar`).
fn parse_value(tok: Option<&[u8]>) -> Option<u32> {
    let tok = tok?;
    if tok.starts_with(b"<") {
        parse_hex_string(Some(tok)).map(|b| bytes_to_u32(&b))
    } else {
        std::str::from_utf8(tok).ok()?.parse::<u32>().ok()
    }
}

/// Interpret `bytes` as a big-endian unsigned integer, truncated to its
/// first four bytes (codes are at most four bytes, PDF §9.7.6).
fn bytes_to_u32(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(4)
        .fold(0u32, |acc, &b| (acc << 8) | u32::from(b))
}

/// The value of one ASCII hex digit; `None` for any other byte.
const fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `(code, value, len)` of every code in `bytes`.
    fn decode(cmap: &CMap, bytes: &[u8]) -> Vec<(u32, u32, u8)> {
        cmap.iter_codes(bytes)
            .map(|c| (c.code, c.value, c.len))
            .collect()
    }

    #[test]
    fn parse_cid_char_single_byte() {
        // Full CMap stream with a `<<...>>` dict literal — exercises the `>`
        // skip in the tokeniser.
        let stream = br"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def
/CMapName /Identity-H def
/CMapType 1 def
1 begincodespacerange
<00> <FF>
endcodespacerange
3 begincidchar
<20> 32
<41> 65
<61> 97
endcidchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.codespaces, vec![Codespace::full(1)]);
        assert_eq!(cmap.map.get(&0x20), Some(&32));
        assert_eq!(cmap.map.get(&0x41), Some(&65));
        assert_eq!(cmap.map.get(&0x61), Some(&97));
        assert_eq!(cmap.map.get(&0x00), None);
    }

    #[test]
    fn parse_cid_range_two_byte() {
        let stream = br"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0020> <0022> 100
endcidrange
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.codespaces, vec![Codespace::full(2)]);
        assert_eq!(cmap.map.get(&0x0020), Some(&100));
        assert_eq!(cmap.map.get(&0x0021), Some(&101));
        assert_eq!(cmap.map.get(&0x0022), Some(&102));
        assert_eq!(cmap.map.get(&0x0023), None);
    }

    #[test]
    fn parse_bfchar() {
        let stream = br"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0041> <0041>
<0042> <0042>
endbfchar
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&0x0041));
        assert_eq!(cmap.map.get(&0x0042), Some(&0x0042));
    }

    #[test]
    fn parse_bfrange() {
        let stream = br"
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfrange
<0041> <0043> <0041>
endbfrange
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&0x0041));
        assert_eq!(cmap.map.get(&0x0042), Some(&0x0042));
        assert_eq!(cmap.map.get(&0x0043), Some(&0x0043));
    }

    #[test]
    fn degenerate_range_skipped() {
        let stream = br"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0050> <0040> 0
endcidrange
        ";
        // hi < lo: codespace is valid but range block is empty.
        let cmap = parse_cmap(stream).expect("should parse with empty map");
        assert!(cmap.map.is_empty());
    }

    #[test]
    fn iter_codes_two_byte() {
        let stream = br"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidrange
<0041> <0043> 10
endcidrange
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        let bytes: &[u8] = &[0x00, 0x41, 0x00, 0x42, 0x00, 0x43];
        assert_eq!(
            decode(&cmap, bytes),
            vec![(0x0041, 10, 2), (0x0042, 11, 2), (0x0043, 12, 2)]
        );
    }

    #[test]
    fn iter_codes_missing_maps_to_zero() {
        let stream = br"
1 begincodespacerange
<00> <FF>
endcodespacerange
1 begincidchar
<41> 65
endcidchar
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        let bytes: &[u8] = &[0x41, 0x42]; // 0x42 not in map
        assert_eq!(decode(&cmap, bytes), vec![(0x41, 65, 1), (0x42, 0, 1)]);
    }

    /// Codespace ranges of one and two bytes split a string by the
    /// shortest matching length: `41 8140 42` is three codes (§9.7.6.3).
    #[test]
    fn mixed_width_codespaces_split_by_range() {
        let stream = br"
2 begincodespacerange
<00> <80>
<8140> <FFFF>
endcodespacerange
1 begincidrange
<41> <43> 1
endcidrange
1 begincidchar
<8140> 4
endcidchar
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.codespaces.len(), 2);
        assert_eq!(
            decode(&cmap, &[0x41, 0x81, 0x40, 0x42]),
            vec![(0x41, 1, 1), (0x8140, 4, 2), (0x42, 2, 1)]
        );
    }

    /// A code in no range takes the length of the shortest range that
    /// contains its first byte, else one byte; it maps to 0.  A string
    /// that ends inside a code yields the remaining bytes as one code.
    #[test]
    fn invalid_codes_follow_the_partial_match_rule() {
        let stream = br"
2 begincodespacerange
<00> <80>
<8140> <9FFC>
endcodespacerange
1 begincidrange
<41> <43> 1
endcidrange
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        // 0x81 0x20: second byte below 0x40 — still a two-byte code.
        assert_eq!(decode(&cmap, &[0x81, 0x20]), vec![(0x8120, 0, 2)]);
        // 0xE0: in no range at all — one byte.
        assert_eq!(
            decode(&cmap, &[0xE0, 0x41]),
            vec![(0xE0, 0, 1), (0x41, 1, 1)]
        );
        // A dangling lead byte.
        assert_eq!(
            decode(&cmap, &[0x41, 0x81]),
            vec![(0x41, 1, 1), (0x81, 0, 1)]
        );
    }

    /// Ranges are rectangular byte-wise: `<8140> <9FFC>` excludes `0x9000`
    /// even though it lies between the bounds as an integer.
    #[test]
    fn codespace_membership_is_byte_wise() {
        let range = codespace(&[0x81, 0x40], &[0x9F, 0xFC]).expect("range");
        assert!(range.contains_prefix(&[0x90, 0x40]));
        assert!(!range.contains_prefix(&[0x90, 0x00]));
        assert!(range.contains_prefix(&[0x90]));
        assert!(!range.contains_prefix(&[0x80]));
        assert!(codespace(&[0x00], &[0xFF, 0xFF]).is_none());
        assert!(codespace(&[], &[]).is_none());
    }

    /// A stream holding bytes that are not UTF-8 (in a comment here) still
    /// parses: the syntax is byte-oriented.
    #[test]
    fn non_utf8_bytes_do_not_reject_the_stream() {
        let stream = b"% caf\xE9 \xFF\xFE\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n1 begincidrange\n<0041> <0043> 1\nendcidrange\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(decode(&cmap, &[0x00, 0x42]), vec![(0x42, 2, 2)]);
    }

    /// `/Identity-H usecmap` makes every code map to itself unless
    /// overridden.
    #[test]
    fn usecmap_identity_inherits_the_identity_mapping() {
        let stream = b"/Identity-H usecmap\n1 begincidchar\n<0041> 4\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.codespaces, vec![Codespace::full(2)]);
        assert_eq!(
            decode(&cmap, &[0x00, 0x41, 0x00, 0x42]),
            vec![(0x41, 4, 2), (0x42, 0x42, 2)]
        );
    }

    /// `usecmap` of a predefined `CMap` brings in its ranges and mappings;
    /// the child's own entries win.
    #[test]
    fn usecmap_predefined_inherits_ranges_and_mappings() {
        let stream = b"/90ms-RKSJ-H usecmap\n1 begincidchar\n<41> 9999\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.codespaces.len(), 4);
        let parent = predefined_cmap_nested(b"90ms-RKSJ-H", 0).expect("predefined");
        assert_eq!(cmap.lookup(0x41), 9999);
        assert_eq!(cmap.lookup(0x42), parent.lookup(0x42));
        assert_eq!(cmap.lookup(0x889F), parent.lookup(0x889F));
    }

    /// A `CMap` without codespace ranges takes the code lengths its
    /// mappings use.
    #[test]
    fn missing_codespace_is_inferred_from_the_mappings() {
        let two = parse_cmap(b"1 begincidrange\n<0041> <0043> 1\nendcidrange\n").expect("parse");
        assert_eq!(two.codespaces, vec![Codespace::full(2)]);
        let one = parse_cmap(b"1 begincidchar\n<41> 1\nendcidchar\n").expect("parse");
        assert_eq!(one.codespaces, vec![Codespace::full(1)]);
    }

    #[test]
    fn empty_stream_returns_none() {
        assert!(parse_cmap(b"").is_none());
    }

    #[test]
    fn missing_end_keyword_does_not_consume_next_section() {
        // A truncated stream with no endcidchar — the parser must not eat the
        // begincidrange block that follows.
        let stream = br"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 begincidchar
<0041> 10
begincidrange
<0042> <0043> 20
endcidrange
        ";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x0041), Some(&10));
        assert_eq!(cmap.map.get(&0x0042), Some(&20));
        assert_eq!(cmap.map.get(&0x0043), Some(&21));
    }

    #[test]
    fn escaped_backslash_at_eof_does_not_panic() {
        // A PS string ending with a lone backslash must not advance past EOF.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<20> 32\nendcidchar\n(test\\";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }

    #[test]
    fn odd_length_hex_string_left_padded() {
        // <1> should be treated as <01> per PDF spec §7.3.4.3.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<1> 99\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x01), Some(&99));
    }

    #[test]
    fn invalid_hex_chars_rejected() {
        // A hex string containing non-hex characters must not produce garbage output.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n1 begincidchar\n<ZZ> 99\nendcidchar\n";
        // <ZZ> fails to parse, so the entry is skipped. The map remains empty
        // but we still get a valid CMap (codespace was found).
        let cmap = parse_cmap(stream).expect("should parse");
        assert!(cmap.map.is_empty());
    }

    #[test]
    fn ps_string_unbalanced_close_paren_does_not_hang() {
        // A PS string token with more `)` than `(` must terminate rather than
        // wrapping the depth counter.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n(unbalanced)) 1 begincidchar\n<20> 32\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }

    #[test]
    fn postscript_procedure_braces_skipped() {
        // `{...}` PostScript procedure bodies appear in Type 4 function CMaps.
        // Braces must be treated as delimiters so `{begincodespacerange}` is
        // not tokenised as a single keyword.
        let stream = b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n{beginbfchar} 1 begincidchar\n<20> 32\nendcidchar\n";
        let cmap = parse_cmap(stream).expect("should parse");
        assert_eq!(cmap.map.get(&0x20), Some(&32));
    }
}
