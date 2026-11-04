# Changes from upstream

This crate is a local fork of `harfrust`, vendored from harfrust commit
`60b28ea22b5261710018d69c168a762bcb28794c` (2026-07-03, the 0.12.0 release).
It is not published and changes are not sent upstream.

0.12.0 is the release built on read-fonts 0.41, the base of the read-fonts
fork in `crates/read_fonts`. Moving to a later release goes with rebasing that
fork.

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
  and their dev-dependencies are dropped. `[lints] workspace = true` is
  removed, so the repository's own lint levels do not apply to upstream's
  code; `exclude` is removed, nothing outside the crate being vendored;
  `readme` points at the crate's own `README.md`.

- Upstream's test suite (`harfrust/tests`, about 6 000 shaping cases with
  their fonts) is not vendored. It passed in full on this fork's sources
  linked against the read-fonts fork when the fork was made.

## Lints under the 1.99 minimum toolchain

Behaviour is unchanged by each of these.

- `hb/face.rs`: `shape`, re-exported only under `experimental_font_api`,
  is compiled only under that feature.
- `hb/glyph_metrics.rs`: two side-bearing lookups return through `?`.
- `hb/unicode.rs`: `compose` returns through `?` and initialises its result
  in one expression; `compose_hangul` and `decompose_hangul` test their
  index arithmetic with `is_multiple_of`.

The crate's own unit tests, which need no external fonts and cover these
functions, run in CI with the read-fonts and skrifa fork tests.

## Upstream fixes brought over

Upstream commits to `harfrust/` after the base, applied by hand to the
0.12.0 module layout (upstream has since moved `src/hb/*` to `src/*`), with
upstream's unit tests where they need no external font. The base in
`UPSTREAM.toml` is unchanged; `reviewed` records how far upstream history
was examined.

Bounds on work:

- `0bab619` A syllable is cut after 64 glyphs (`MAX_SYLLABLE_LENGTH`),
  bounding the reordering work an over-long run of marks asks for, and
  `c2f8ca4` (#570) records each cut in the glyphs' syllable serial so the
  Indic, Khmer, Myanmar and USE passes before and after substitution see
  the same pieces, which are kept in one cluster. Output changes only for
  syllables longer than 64 glyphs. Test:
  `buffer::tests::next_syllable_is_bounded`.
- `c3270f1` (#429) `move_to` charges the glyphs it copies to the buffer's
  operation limit, which bounds the quadratic work of a `morx` insertion
  far behind its mark. Test:
  `buffer::tests::move_to_charges_the_operation_limit` (this fork).
- `b6dbcf1` `set_masks` stops walking the buffer once the operation limit
  is spent. Test:
  `buffer::tests::set_masks_stops_once_the_operation_limit_is_spent`.
- `708294f` (#417) Legacy `kern` resumes after the glyphs the skipping
  iterator has already passed instead of rescanning them, which was
  quadratic in a run of ignored marks.
- `b3c2212` (#573) `kerx` and `morx` initial-glyph scans stop at class
  65 535, the largest a `u16` class reaches, whatever class count the
  table declares. Tests: `aat::layout_kerx_table::tests`,
  `aat::layout_morx_table::tests`.
- `0ea68e9` (#484) A GSUB or GPOS lookup type is kept as `u16`, so an
  out-of-range type is rejected instead of being truncated to a valid one
  and run over unrelated bytes.

Panics and out-of-range indexing:

- `996f4f1` (#411) A cursive attachment whose distance does not fit the
  `i16` chain is left unattached, and the chain walk stops at the end of
  the buffer instead of indexing past it. Test: `ot::gpos::cursive::tests`.
- `b609640` (#480) A mark more than `i16::MAX` glyphs after its base is
  left unattached instead of being placed on the wrong glyph.
- `44490af` (#416) A `morx` insertion at the end of the text marks the
  inserted glyph instead of indexing past the input.
- `443dd21` (#568) `morx` insertion indices are summed in `usize`, so a
  list of 65 536 or more glyphs no longer wraps to its start.
- `fd029d6` Fallback mark positioning halves an offset before negating
  it, so `i32::MIN` does not overflow.

Shaping fixes in the scripts this fork shapes; output changes where the
fix applies:

- `41258d7` (#483) `would_apply` of a format 3 sequence context matches
  glyph `i` against coverage `i`; the Indic shaper's below-base and
  post-base consonant tests (`blwf`, `pstf`, `vatu`) saw no format 3
  lookup apply before.
- `f814104` Thai: decomposing SARA AM keeps every above-base mark before
  it when moving Nikhahit, instead of repeating the first.
- `1bd58d1` Telugu length marks (U+0C55, U+0C56) take modified combining
  classes 4 and 5, so they reorder before the virama and keep their
  relative order. Test: `unicode::tests`.
- `f396022` USE: pre-base vowels in symbol clusters are reordered. Test:
  `ot_shaper_use::tests`.
- `bf81a6b` U+180F (Mongolian free variation selector four) is default
  ignorable. Test: `unicode::tests` (this fork).
- `4af3297` A mark takes its base's cross-stream offset, through the
  base's cursive chain, when it is attached, and only the main-direction
  offset is propagated afterwards (HarfBuzz 6b6f0d97).
- `9c522f9`, `4290992` The pieces of a glyph decomposed by a multiple
  substitution count as one ligature component, and a mark attaches to
  the first piece or to the ligature it joined (HarfBuzz issue 4969).

Verification: upstream's shaping suite at the base (6 027 cases), with the
cases these commits add to it and their fonts, was run against this fork's
sources in a scratch workspace: all 6 042 pass, `khmer_misc_046` to `048`
with the expectations `4af3297` updates. Without the added cases, those
three are the only base cases whose output changes.

## Upstream review (2026-11-04)

Upstream history through `bba5d0ff3` (2026-11-01, crates.io 0.15.0) was
reviewed by commit. Besides the fixes above, nothing is brought over:

- A rebase onto 0.15 goes with the read-fonts fork's rebase: upstream has
  reorganised the modules (`8b61fb0`, `dd31b6b`), replaced the buffer
  types and entry points (`fa84dc7`, `64e8fca`, `af0d3bf`), moved to
  read-fonts 0.43 and 0.46 (`ee63876`, `8001cea`, `10b4a42`), turned buffer
  indexing into `Option`s (`864e824`, `4a18f6d`) and cached parsed lookups
  with the font tables (`9550595` and the commits around it).
- Unreached by this fork's users, who shape one fresh buffer per run of
  one detected script, at the default location of an installed or bundled
  face, without features, flags or a variation-selector fallback glyph:
  the stale cursor before a dotted circle in a reused buffer (`0196951`),
  the variation-selector fallback (`ca987b0`), the native direction of a
  run without a script (`278c861`), unsafe-to-concat and tatweel flags
  (`6b73f21`, `c56badb`, `c9ff3cb`, `d6861eb`, `6c67092`), Hangul
  (`b02ad22`, `000ab49`, `37c2883`), vertical origins (`904c25b`),
  negative variable advances (`1141212`), feature variations selected at
  the default location (`dfa9693`, which also needs the read-fonts
  rebase), and the preprocessing budgets and class-matrix checks
  (`f5f6f80`, `f325ef9`, `88d8525`, `e118fd7`, `7eed82f`) that guard caches
  added after the base.
- Overflow hardening of positions that wrap rather than panic in release
  builds and need out-of-range font values: the saturating-arithmetic
  series (`94c4438` to `3d529d8`, `7aa82bf`).
- Features and behaviour changes outside the scripts shaped here: AAT
  `kerx` format 1 and tuple kerning (`03af308`, `930682b`, `e4c93a4`,
  `233fb9f`), `mort` shaping and language-specific morph features
  (`c770a1f`, `c4f98f5`), Arabic fallback synthesis (`4befd3f`, `34ae8a9`,
  `3b5763a`), Unicode 18 data (`41c79c8`, `0f51460`), OpenType rounding at
  other scales (`b6100b2`), language and script tag mapping (`19cf334`,
  `2acba18`), ICU data, the C API, tracing, glyph-name caching and
  performance work.
