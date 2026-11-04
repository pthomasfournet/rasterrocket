# Changes from upstream

This crate is a local fork of `read-fonts`, vendored from fontations commit
`1d1672367c7909013b4e366085d68d5ad612ee67` (2026-11-02, after the 0.46.0
release). It is not published and changes are not sent upstream.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.99), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields and dependencies are resolved to
  values; `publish = false`; `font-test-data` points at the copy in
  `crates/font_test_data` (same upstream commit). The benchmarks and their
  `criterion` / `rand` dev-dependencies are removed.

- Fixed-width byte groups are read with `slice::as_chunks` (stable since
  Rust 1.88) instead of `chunks_exact` with a constant size, as clippy
  requires once the minimum toolchain allows it: the checksum in
  `tables.rs`, inline instruction operands, and the `I16` / `I32` delta
  runs in `tables/variations.rs`. The `post` name iterator's checkpoint
  test uses `is_multiple_of`, and `ps::type1::mul_div_wide` divides with
  `checked_div`. Behaviour is unchanged.

- The licence files replace upstream's symlinks to the repository root.

## Type 1 fonts embedded in PDF

- `ps::type1`: an `/Encoding` value the parser does not recognize (a
  named encoding other than Standard / Expert / ISOLatin1, or a malformed
  array) no longer rejects the whole font; the font loads with no built-in
  encoding. A PDF simple font supplies its own encoding, so the outlines
  are still reachable by glyph name. Test: `unrecognized_encoding_keeps_font`.
- `ps::type1::verify_header`: any `%!` header comment is accepted, not only
  `%!PS-AdobeFont` and `%!FontType` (for example `%!PS-Adobe-3.0
  Resource-Font`, which some producers write into `FontFile` streams). The
  dictionary parse that follows still rejects non-fonts. Test:
  `resource_font_header_accepted`.
- Two parser tests upstream does not have: a procedure whose last token
  ends at the closing brace, as in the `/OtherSubrs` entry `dup 3 {3}
  executeonly put` of TeX-derived fonts
  (`parse_procs_with_tokens_ending_at_the_brace`), and a font of 1 000 000
  nested braces, which a recursive procedure scan overflowed the stack on
  (`parse_deeply_nested_procs_without_recursion`).

## CID-keyed Type 1 fonts

- New `ps::type1::cid::CidType1Font`: a reader for CID-keyed Type 1 fonts
  (CIDFontType 0, Adobe Technical Note #5014), which a PDF embeds as
  `FontFile` of a `CIDFontType0` descendant font. It parses the cleartext
  header (`CIDMapOffset`, `FDBytes`, `GDBytes`, `CIDCount`, `FDArray`),
  locates the binary or `(Hex)` data section after `StartData`, reads
  each font dictionary's `FontMatrix`, `lenIV` and subroutines
  (`SubrMapOffset`, `SDBytes`, `SubrCount`), and decrypts every charstring
  with its dictionary's `lenIV`; charstrings run through the existing
  Type 1 interpreter with their dictionary's subroutines, and `seac` codes
  are CIDs. Font dictionaries are delimited by `%ADOBeginFontDict`
  comments as in FreeType's `cid` driver. Differences from FreeType, all
  more lenient: any `%!` header is accepted (FreeType requires
  `%!PS-Adobe-3.0 Resource-CIDFont`, which `CidType1Font::is_cid_keyed`
  tests for), and a dictionary key outside a declared dictionary or an
  unreadable `FontMatrix` is ignored instead of failing the font. Errors
  are reported with the `model::Error` kinds the Type 1 reader uses. Tests:
  `ps::type1::cid::tests::*` (in `src/tests/test_cid_type1.rs`).
- `CidType1Font`: decoded subroutines and charstrings are bounded by the
  length of the data section. The CIDMap and the subroutine maps are
  offsets into that section, and the ranges of different CIDs (or the
  subroutine maps of different font dictionaries) may overlap, so
  decoding each one separately cost memory quadratic in the input (4 096
  CIDs over a 16 KiB section asked for 30 MiB; a few MiB of input, for
  terabytes). Font dictionaries with the same subroutine map and `lenIV`
  share one decoded set, CIDs with the same charstring range share one
  decoded charstring, and once the distinct ranges exceed the section's
  length a further charstring leaves its CID without a glyph and further
  subroutines reject the font. Well-formed fonts decode each byte once
  and are unaffected. Tests:
  `overlapping_charstrings_are_bounded_by_the_data_section`,
  `identical_charstring_ranges_share_one_copy`,
  `font_dicts_naming_one_subr_map_share_their_subrs`.

## CFF charsets

- `ps/cff/charset.rs`: the range iterator of format 1 and 2 charsets ended
  the whole iteration at the first glyph whose string identifier exceeds
  `u16` (a range starting near 65535), while `Charset::glyph_id` skips such
  a range and keeps counting glyphs. The iterator now skips those glyphs,
  so the two agree and the glyphs of later ranges are reached. Test:
  `iteration_skips_identifiers_past_u16_and_agrees_with_lookup`.
- `ps/cff/encoding.rs`: a test that a predefined encoding, and a
  supplement, naming a glyph past the font's glyph count map to nothing
  (`predefined_codes_stop_at_the_glyph_count`); upstream's charset bound
  provides the behaviour.

## Hint-reliant ("tricky") TrueType detection

- `tables::glyf::hint_reliant`: the `cvt ` / `fpgm` / `prep`
  identification compares the three table lengths with the list first and
  computes the checksums only when they match an entry, so ordinary fonts
  pay no scan of their hinting programs. The result is unchanged.

## Rebase history

- 2026-11-04: rebased from fontations `cd4aaef57` (read-fonts 0.41.0) onto
  `1d1672367` (after 0.46.0). Upstream had by then restructured the crate
  (the `model` module with `Font` and the hinting error types and
  outline-to-path conversion from skrifa; the `ps` module's `cs`, `error`
  and `Charmap` became `charstring`, `model::Error` and `UnicodeCharmap`)
  and taken over hint-reliant detection from skrifa, deriving the family
  name as FreeType does, which this fork had done in skrifa. Every upstream
  fix this fork had brought over by hand (glyf point reading and flag
  repeats, bitmap locations, charstring work bounds, CFF charset, FDSelect,
  encoding and BCD real fixes, `VarLenArray::get`, the Type 1 procedure
  scan, `kerx` format 2 classes) is upstream's own code now. Rendering
  changes this brings that were deliberately left out before: cmap format
  4 `idRangeOffset` handling (#2376), CFF charset lookups bounded by the
  glyph count (#2379, #2397), Type 1 parsing fixes (#2257), bitmap table
  overflows (#2053), and the Adobe Glyph List extended with the ITC Zapf
  Dingbats names (`a1` maps to U+2701), as FreeType's synthesized Unicode
  cmap does (#2378).
