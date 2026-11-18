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

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.100), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields are resolved to values; `publish =
  false`; `read-fonts` is the fork in `crates/read_fonts`. The benchmarks
  and the dev-dependencies of upstream's test suite and comparison tools
  are dropped. `[lints] workspace = true` is emptied to `[lints]`, so the repository's
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

## Lints required since the 1.99 minimum toolchain

Behaviour is unchanged by each of these.

- `unicode/mod.rs`: `compose_hangul` and `decompose_hangul` test their
  index arithmetic with `is_multiple_of`.
- `ot/lookup/gpos.rs`: value-record fields are read with `as_chunks`.
- `aat/morx.rs` (test of the morx and mort glyph-set caches): lookup
  segments are rewritten with `as_chunks_mut`.

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

## Upstream review (2026-11-14)

Upstream history through `26f5a7682` (2026-11-14, after the 0.16.0
release) was reviewed by pull request and brought over by hand, each
pull request one commit, in upstream's order; the base in
`UPSTREAM.toml` is unchanged.  Only `src` is brought over: upstream's
`harfrust/tests` suite, its fonts and its C API crate are not vendored.
Test fonts the unit tests read are the ones already copied (see
Vendoring); the one new `include_bytes!` of `benches/fonts` is
redirected to `tests/fonts/benches`.

Ported:

- #583 (47c4405, 20ca114, 51b1107, 5bf032d, d5536c6): lookup variations
  bounded, duplicate stage lookups merged, feature indices resolved in
  one pass, GSUB extensions of the wrong type skipped.
- #584 (617e7ff, b22c6ca, db1b1e9, 58dd665, 61defdf): cluster-merge
  walks charged, live items only looped over, glyph set built in
  ascending order.
- #587 (f322786, f223e41, 58b97e9, 826b840, 4425de7, baab979, b60aa1a,
  f7026ef): syllable look-ahead bounded and shaper edge cases.
- #582 (a676781, 7a7f9a3, 24ab0af, 30082b0, f80aa9e, 0932d23, c9a72d8):
  iterative cursive chains, charged GPOS walks, mark offsets.
- #588 (9e3553c, 4eb450a, b901148, 9fd171d, b84f0a2, e0ab4ce, ed622a1,
  bc96fd4, b26b469, d06db47, c1c66a7): AAT budgets and bounds.
- #589 (d9e1a95, de00e11, f06f33b, 8620539, 07cfb3c, 8aa1353, 3ee2d01):
  matching walks charged (mostly replaced by #601).
- #585 (06da5f9): `Buffer::clear` restores what an unwound shape leaves.
- #595: wide glyph ids are out of bounds of legacy kern class tables.
- #596: AAT action handlers outlined; no output change.
- #598 (30ac5cc): equal-flag AAT feature ranges merged, map compilation
  budgeted.
- #599: AAT tables with more than 1024 subtables ignored.
- #601: the shaping budget is charged per lookup pass.
- #603: syllables cut only when one reaches the length limit.
- #606: zero-context matching for new-spec Indic scripts.
- #604 (b575a26, eac44a0, e7d2ee7): skip, skipped-record and ligature
  walks charged again, matching stops once the budget is spent.
- #610 (13acef1, aec3d28, 18e692e, c6721e6): per-syllable matching
  stops before skipped glyphs, kerx cross-direction mark offsets,
  duplicate trak entries averaged, Arabic PUA mappings refreshed.  The
  exact float comparison in `aat/trak.rs` carries no lint allowance,
  `clippy::float_cmp` not being enabled here.
- #611: morx language matching on raw ltag bytes, integral F48Dot16
  values beyond i32 scaled through the float path, cursive anchor
  rounding, a vertical-advance fallback to the line metrics when `vmtx`
  is empty, a kerx and an Arabic fix.
- #602 (b6ec0f891): signed AAT states, together with the read-fonts
  fork's port of read-fonts #2455 in the same commit, whose API it
  adapts to.  The workspace's read-fonts version bump in #602 has no
  counterpart: the fork is a path dependency.

Covered: the merge commits 0460dbc, 06585ea, 4654c82, 687453e, c1660b6,
8f87c79, e55117d, 4472417, 859d44f (`d09ed51`), 26f5a76 and the
`Merge … into …` commits (848bbfa, bc71392, 95a2ac9, 0800fc3, d4294c5,
2f8df34, 65651e1) by their constituent commits above.

Not applicable:

- #450 (8e006a0, `icu_properties` 2.3): the `icu` feature is not
  enabled by any crate of the workspace.
- d29d54c (shaping-suite refresh): the suite is not vendored.
- #611's CI workflow change: not vendored.

Rejected: none.

## Upstream review (2026-11-18)

Upstream history after `26f5a7682` through `f33cff93f` (2026-11-16, on
0.16.0) was reviewed; the base in `UPSTREAM.toml` is unchanged.

Ported:

- #612 (f33cff93f): `morx` / `mort` chain subtables and `kerx` format 1
  and 4 subtables whose state machine can act only on a glyph of a
  completer class are skipped when the buffer holds none; a machine whose
  end-of-text or out-of-bounds transition acts, a legacy one, or one with
  fewer than 4 or more than 512 classes is not gated.  Applied as
  upstream wrote it, with its unit tests, which read the already copied
  `TestMORXOne.ttf`.
