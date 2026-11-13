# Changes from upstream

This crate is a local fork of `zune-jpeg`, vendored from upstream commit
`31d81fed7551c8ccea456d9d8e2b1fd8bebb6995` (2026-03-26, crate version
0.5.15, the release on crates.io). It is not published and changes are not
sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- Kept: `src/`, `tests/`, `README.md`, `Changelog.md`, `Benches.md` and the
  three licence files (upstream's symlinks to its repository root replaced
  by the files themselves). Left out: `fuzz/` (upstream's fuzz targets) and
  `.gitignore`.

- `Cargo.toml`: `rust-version` is inherited from the workspace, so every
  crate of the repository builds with the same minimum toolchain;
  `publish = false`; `zune-core` is the crates.io release instead of
  upstream's sibling directory; the unused `zune-ppm` dev-dependency is
  dropped; the lint table is empty, so the crate is linted with clippy's
  defaults.

- `tests/`: the images the suites read from upstream's repository-level
  `test-images/jpeg/` directory are copied into `tests/images/` and the
  `include_bytes!` paths point there.

- `src/lib.rs`: the crate-level `#![warn(clippy::pedantic, ...)]` block and
  the stray `#![macro_use]` are removed, and the `idct` module is private
  under every configuration (upstream made it public under `cfg(fuzzing)`
  for the fuzz target left out above).

- Clippy's default findings are fixed in place, none changing behaviour:
  `div_ceil` for the hand-written round-up divisions, `as_chunks` /
  `as_chunks_mut` for `chunks_exact` with a constant size (and the array
  conversions that made redundant), a derived `Default`, `sort_by_key`,
  `if let` for `Option::map` with a unit closure, a slice fill for a
  hand-counted loop in the Huffman lookup table, `slice::swap` in a NEON
  test, and doc-comment layout.

- The sources are formatted with the workspace's rustfmt settings
  (upstream uses nightly-only options).

## Changes in this fork
