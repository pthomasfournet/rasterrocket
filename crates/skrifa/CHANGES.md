# Changes from upstream

This crate is a local fork of `skrifa`, vendored from fontations commit
`1d1672367c7909013b4e366085d68d5ad612ee67` (2026-11-02, after the 0.49.0
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
  values; `publish = false`; `read-fonts` is the fork in
  `crates/read_fonts` and `font-test-data` the copy in
  `crates/font_test_data`.

- The licence files replace upstream's symlinks to the repository root.

- `outline::glyf::hint::math::mul_div_no_round` divides with
  `checked_div(..).unwrap_or(..)` where upstream tests the divisor first
  (same result; the workspace's `clippy -D warnings` gate rejects the
  manual form).

## Font programs embedded in PDF

- `outline::metrics::GlyphHMetrics`: a font without a readable `hmtx` or
  `hhea` table keeps its `glyf`, `CFF` and `VARC` outlines; advances read
  as zero. Before, the whole outline collection was empty for such fonts.
  The `glyf` loader takes the side bearing to be the glyph's `xMin` when
  there are no metrics, so outlines are not shifted. Test:
  `outline::tests::outlines_without_horizontal_metrics`.

## Glyph name iteration

- `GlyphNames::iter` over a version 2.0 `post` table splits the string
  data once and indexes it, in time linear in the glyph count whatever
  order the name indices come in. Upstream's iterator
  (`read_fonts::tables::post::GlyphNames`) rescans from one of 16
  checkpoints for a name index below the previous one, and skips a glyph
  whose name cannot be read, which shifts every later name onto the wrong
  glyph in `GlyphNames::iter`. Here such a glyph gets a synthesized name,
  as `GlyphNames::get` gives it. Test:
  `glyph_name::tests::post_iter_matches_table_lookup`.

## Rebase history

- 2026-11-04: rebased from fontations `b373cdbc6` (skrifa 0.44.0) onto
  `1d1672367` (after 0.49.0). Hint-reliant detection moved to read-fonts
  upstream, which now derives the family name as FreeType does, as this
  fork did (the length-first table check is kept, see the read-fonts
  fork's `CHANGES.md`). Every upstream fix this fork had brought over by
  hand (TrueType interpreter bounds and arithmetic, the composite
  component limit, VARC bounds) is upstream's own code now. Rendering
  changes this brings that were deliberately left out before: #2040 (the
  TrueType interpreter's parity checks ignore fractional bits), which
  affects only the hint-reliant fonts this workspace hints; the autohinter
  and colour-glyph changes are not reached by this workspace's renderer.
