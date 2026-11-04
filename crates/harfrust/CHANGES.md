# Changes from upstream

This crate is a local fork of `harfrust`, vendored from harfrust commit
`bdd55a548fb3baaaed0e0633402909858858170a` (2026-11-02, after the 0.15.0
release). It is not published and changes are not sent upstream.

It is built against the read-fonts fork in `crates/read_fonts` (upstream
read-fonts 0.46), and the two are rebased together.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.99), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields are resolved to values; `publish =
  false`; `read-fonts` is the fork in `crates/read_fonts`. The benchmarks
  and the dev-dependencies of upstream's test suite and comparison tools
  are dropped. `[lints] workspace = true` is removed, so the repository's
  own lint levels do not apply to upstream's code; `exclude` is removed,
  nothing outside the crate being vendored; `readme` points at the crate's
  own `README.md`, upstream's repository README with a provenance note and
  without its logo and badges. `[package.metadata.release] release =
  false` keeps the crate out of `cargo release`, so its version stays that
  of the upstream base.

- Upstream's test suite (`harfrust/tests`, about 6 000 shaping cases with
  their fonts) is not vendored. The unit tests are, and some embed fonts
  from outside `src` (13 files, about 1 MB): the ten they read from
  upstream's `tests/fonts` are copied to the same paths, and
  `Roboto-Regular.ttf` and `NotoSansDevanagari-Regular.ttf` (upstream
  `benches/fonts`) and one fuzzer case (upstream `fuzz/corpus/fuzz_shape`)
  to `tests/fonts/benches` and `tests/fonts/fuzz`, with the
  `include_bytes!` paths changed to match.

## Lints under the 1.99 minimum toolchain

Behaviour is unchanged by each of these.

- `unicode/mod.rs`: `compose_hangul` and `decompose_hangul` test their
  index arithmetic with `is_multiple_of`.
- `ot/lookup/gpos.rs`: value-record fields are read with `as_chunks`.

The crate's unit tests run in CI with the read-fonts and skrifa fork tests.

## Tests

Two unit tests this fork added before the rebase, for upstream fixes it
had brought over, are kept: `buffer::tests::move_to_charges_the_operation_limit`
(upstream `c3270f1`) and
`unicode::tests::mongolian_free_variation_selectors_are_default_ignorable`
(upstream `bf81a6b`).

## Rebase history

- 2026-11-04: rebased from harfrust `60b28ea22` (0.12.0) onto `bdd55a548`
  (after 0.15.0). Every upstream fix this fork had brought over by hand
  (syllable and operation bounds, `kern` / `kerx` / `morx` scan bounds and
  index fixes, cursive and mark attachment fixes, the Indic, Thai, Telugu,
  USE and Mongolian fixes, mark offsets and multiple-substitution
  ligature components) is upstream's own code now. Upstream's later
  changes come with it: the module reorganisation (`src/hb/*` to `src/*`),
  the `Buffer` / `ShaperFont` / `shape` API in place of `UnicodeBuffer` /
  `ShaperData`, lookups cached with the font tables, saturating position
  arithmetic, Hangul, AAT `kerx` format 1 and `mort`, Arabic fallback
  synthesis, Unicode 18 data and HarfBuzz's OpenType rounding.
