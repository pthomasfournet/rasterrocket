# Changes from upstream

This crate is a local fork of `hayro-ccitt`, vendored from upstream commit
`79926198e557e06fea8a8edcb0bc577eb0130ec5` (2026-10-23). It is not published
and changes are not sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: workspace-inherited fields resolved to values; `publish =
  false`; upstream's workspace lint set is not inherited, so the crate is
  linted with clippy's defaults.

- `LICENSE-APACHE`, `LICENSE-MIT`: upstream's symlinks to its repository
  root replaced by the files themselves.

The base is ahead of the 0.4.0 release on crates.io by its later fixes
(Group 4 EOFB handling, a shift overflow on data ending within a row, a
strict mode) and by a changed sink API: `Decoder::push_row` receives each
row as its colour-change positions instead of runs of pixels.
