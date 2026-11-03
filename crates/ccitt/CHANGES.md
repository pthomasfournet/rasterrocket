# Changes from upstream

This crate is a local fork of `hayro-ccitt`, vendored from upstream commit
`79926198e557e06fea8a8edcb0bc577eb0130ec5` (2026-10-23). It is not published
and changes are not sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Encoder

- `encode.rs` (new): `Encoder` codes rows given as colour-change positions,
  the form `Decoder::push_row` receives, as Group 3 one-dimensional (MH),
  Group 3 two-dimensional (MR, period `k: NonZeroU32`) with an EOL before
  every row (optionally byte-aligned), or Group 4 (MMR) ended by EOFB.  No
  RTC is written.  The width is a `NonZeroU32`, so construction cannot
  fail.  `changes_of_row` derives the changes from packed pixels, skipping
  whole bytes of one colour, and `Encoder::encode_packed_row` codes a
  packed row through a reused change buffer.  Two-dimensional coding finds
  b1 with a cursor that only advances, linear in the row's changes.  Run
  codes are indexed by run in the shared tables, whose layout a constant
  assertion checks.  The run-length tables in `tables.rs` are shared with
  the decoder and became `pub(crate)`; the decoder is unchanged.  Tests
  compare the output with bit strings written from the T.4 and T.6 code
  tables, independent of `tables.rs`.

## Upstream changes brought over

- `4f7d16fab` (#1733): the decoder's per-row colour-change list grows
  with `try_reserve`, and a failed allocation is the new
  `DecodeError::OutOfMemory` instead of an abort.

Upstream history through `4f7d16fab` (2026-11-02) was reviewed on
2026-11-03. Left out: #1731 (`DecodeSettings` made explicit, its types
exhaustive: an API change with no decoding change) and #1732 (version
bumps and changelogs).

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.99), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields are resolved to values; `publish =
  false`; upstream's workspace lint set is not inherited, so the crate is
  linted with clippy's defaults.

- `LICENSE-APACHE`, `LICENSE-MIT`: upstream's symlinks to its repository
  root replaced by the files themselves.

The base is ahead of the 0.4.0 release on crates.io by its later fixes
(Group 4 EOFB handling, a shift overflow on data ending within a row, a
strict mode) and by a changed sink API: `Decoder::push_row` receives each
row as its colour-change positions instead of runs of pixels.
