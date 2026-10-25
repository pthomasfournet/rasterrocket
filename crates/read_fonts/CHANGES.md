# Changes from upstream

This crate is a local fork of `read-fonts`, vendored from fontations commit
`cd4aaef57887dd1d8a5875e38c406843e66c9cab` (2026-07-03, the 0.41.0 release).
It is not published and changes are not sent upstream.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: workspace-inherited fields resolved to values; `publish =
  false`; `font-test-data` points at the copy in `crates/font_test_data`
  (same upstream commit as the `skrifa` fork). The benchmarks and their
  `criterion` / `rand` dev-dependencies are removed.

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
- `ps::type1::Parser::read_proc`: a token ending at the closing brace, as
  in the `/OtherSubrs` entry `dup 3 {3} executeonly put` of TeX-derived
  fonts, consumed the brace; the procedure then ran on through the
  following entries and the font was rejected for lacking `/CharStrings`.
  The closing brace is now checked before each token is read. Test:
  `parse_procs_with_tokens_ending_at_the_brace`.

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
  unreadable `FontMatrix` is ignored instead of failing the font. Tests:
  `ps::type1::cid::tests::*`.
