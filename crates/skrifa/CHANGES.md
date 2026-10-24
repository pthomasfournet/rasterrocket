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
