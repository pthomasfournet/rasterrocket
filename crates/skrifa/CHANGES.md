# Changes from upstream

This crate is a local fork of `skrifa`, vendored from fontations commit
`b373cdbc6043a791fb23b3e0f413649458f5e63a` (2026-07-04, the 0.44.0 release).
It is not published and changes are not sent upstream.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: workspace-inherited fields resolved to values; `publish =
  false`; `read-fonts` is the fork in `crates/read_fonts` and
  `font-test-data` the copy in `crates/font_test_data`.

- The licence files replace upstream's symlinks to the repository root.

## Font programs embedded in PDF

- `outline::metrics::GlyphHMetrics`: a font without a readable `hmtx` or
  `hhea` table keeps its `glyf`, `CFF` and `VARC` outlines; advances read
  as zero. Before, the whole outline collection was empty for such fonts.
  The `glyf` loader takes the side bearing to be the glyph's `xMin` when
  there are no metrics, so outlines are not shifted. Test:
  `outline::tests::outlines_without_horizontal_metrics`.

## Hint-reliant ("tricky") TrueType detection

- `outline::hint_reliant`: the family name compared against the list is
  now derived as FreeType derives it — name IDs 21, 16, 1 in that order
  (16, 1 for a WWS-only font), the record chosen by FreeType's platform /
  language preference, and every UTF-16 code unit or byte outside 32..=127
  read as `?`. Before, only name ID 1 was read, a name longer than 18
  characters never matched (FreeType matches by substring, so a subset tag
  plus a long family such as `ABCDEF+DFKaiShu-SB-Estd-BF` was missed), and
  non-ASCII characters were truncated to their low byte instead of reading
  as the `?` the `HuaTianKaiTi?` / `HuaTianSongTi?` entries expect.
- The `cvt ` / `fpgm` / `prep` identification compares table lengths first
  and computes a table's checksum only when its length matches an entry,
  so ordinary fonts pay no table scan. The name and checksum lists are
  unchanged and identical to FreeType 2.13.2's. Tests:
  `outline::hint_reliant::tests::*`.

## Glyph name iteration

- `GlyphNames::iter` over a version 2.0 `post` table splits the string
  data once and indexes it. Before, each name was found by
  `Post::glyph_name`, which walks every Pascal string ahead of it, so
  listing all names took time quadratic in the glyph count (about 70% of
  render time on documents whose embedded TrueType subsets carry thousands
  of `post` names). `GlyphNames::get` is unchanged. Test:
  `glyph_name::tests::post_iter_matches_table_lookup`.

## Upstream fixes brought over

Fixes to code this fork's users reach (the TrueType bytecode interpreter,
glyf and VARC outlines) from upstream commits after the base, applied as
upstream wrote them with their tests. The base in `UPSTREAM.toml` is
unchanged; `reviewed` records how far upstream history was examined.

- `28ad033c8` (#1963) TrueType interpreter: a bound on the instructions
  skipped by `IF` / `ELSE` / `FDEF` scanning.
- `93d40d25f` (#1964) TrueType interpreter: out-of-bounds indexing and
  overflowing arithmetic in `data`, `outline`, `math`, `round` and `zone`.
  `math::mul_div_no_round` divides with `checked_div(..).unwrap_or(..)`
  where upstream tests the divisor first (same result; the workspace's
  `clippy -D warnings` gate rejects the manual form).
- `8c1fedf93` (#1978) glyf: at most 2048 components are processed per
  composite outline, against exponential fan-out.
- `487a3335d` (#2011) TrueType interpreter: wrapping arithmetic in `MIRP`
  and every rounding mode.
- `f3edd9d2d` (#2039) TrueType interpreter: overflow in `line_vector`.
- `867fe4d6e` (#2014) VARC: bounded condition-evaluation recursion.
- `f43403c06` (#2012) VARC: a null `MultiItemVariationStore` offset no
  longer panics.
- `e829e72a3` (#2032) VARC: a bound on the edges processed per glyph.

## Upstream review (2026-10-30)

Upstream history through `8cf9cbf41` (2026-10-29, crates.io 0.49.0) was
reviewed by commit. Besides the fixes above, nothing is brought over:

- The rebase onto 0.49 is deferred to the next font work. Upstream has
  moved hint-reliant detection, hinting error types and outline-to-path
  conversion into read-fonts and reorganised the PostScript support,
  which this fork's hint-reliant and metrics changes would have to be
  re-applied onto.
- Left out as a behaviour change that needs its own render comparison:
  #2040 (TrueType interpreter parity checks ignore fractional bits).
- Not reached by this fork's users, who hint with the interpreter only:
  the autohinter fixes (#1980, #1987, #1988, #1989, #2044 and the work
  budgets), COLR (#2009, #2031), layout, and colour or bitmap glyphs.

