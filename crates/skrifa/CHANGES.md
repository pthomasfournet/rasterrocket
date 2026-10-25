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
