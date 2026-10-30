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
  The closing brace was then checked before each token was read; that
  change is superseded by upstream's byte-level procedure scan (#1993,
  under "Upstream fixes brought over"), which handles the same input.
  Test: `parse_procs_with_tokens_ending_at_the_brace`.

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
  `ps::type1::cid::tests::*` (in `src/tests/test_cid_type1.rs`).

## Upstream fixes brought over

Fixes to code this fork's users reach (glyf outlines, the CFF, CFF2 and
Type 1 parsers and charstring interpreter) from upstream commits after the
base, applied as upstream wrote them with their tests. The base in
`UPSTREAM.toml` is unchanged; `reviewed` records how far upstream history
was examined.

- `26503f2a0` (#1965) `tables::glyf::SimpleGlyph::read_points_fast`: a
  simple glyph with zero points and trailing padding indexed `flags[0]`
  of an empty slice (a panic in release builds). Found by the `fuzz_font`
  target within 605 runs.
- `f5ab31fe7` (#2052) `PointIter::advance_flags`: the repeat count is
  widened so its `+ 1` cannot overflow.
- `dd9af718b` (#1977) `BitmapSize::location`: an error instead of a
  panic on truncated subtable data.
- `d8769631a` (#2000) charstring evaluation: a bound on the work one
  charstring may ask for.
- `d0f5d071f` (#2016) CFF charset: overflow panics in range arithmetic.
- `10b5d6959` (#2026) CFF FDSelect: out-of-range font dictionary indices.
- `b08c6cce2` (#2051) `VarLenArray::get`: reads bounded by the data.
- `2a6bdb532` (#2359) CFF DICT: a BCD real whose exponent overflows is
  consumed whole instead of desynchronising the operand stream.
- `f95a2db5b` (#1993) `ps::type1::Parser`: procedures are scanned byte by
  byte with a nesting counter instead of recursing once per `{`, so a
  font of nested braces no longer overflows the stack (1 000 000 levels
  aborted the test process before; test
  `parse_deeply_nested_procs_without_recursion`); comments and literal
  strings inside a procedure no longer end it early; the `RD` / `-|`
  lookahead no longer tokenizes binary data; a literal string no longer
  consumes the byte after its closing parenthesis. Hand-applied: the
  procedure scan replaces this fork's earlier `read_proc` change.

## Upstream review (2026-10-30)

Upstream history through `42e0f877d` (2026-10-27, crates.io 0.46.0) was
reviewed by commit. Besides the fixes above, nothing is brought over:

- The rebase onto 0.46 is deferred to the next font work. Upstream has
  restructured the crate since the base (`model` / `OutlineGlyphs`, the
  hinting error types and outline-to-path conversion moved in from
  skrifa, the `ps` module reorganised), which this fork's Type 1 and
  CID-keyed Type 1 changes would have to be re-applied onto.
- Left out as behaviour changes that need their own render comparison:
  #2376 (cmap format 4: an `idRangeOffset` before the glyph id array maps
  nothing instead of the first entry), #2379 / #2397 (CFF charset lookups
  stop at the glyph count), #2257 (Type 1 parsing fixes), #2053 (bitmap
  table overflows; this fork's users draw outlines only).
- Not reached by this fork's users: layout, AAT, COLR, SVG, cmap formats
  14 / 15, DMAP, subsetting and glyph closure.

