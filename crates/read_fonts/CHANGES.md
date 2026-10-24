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
