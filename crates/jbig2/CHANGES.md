# Changes from upstream

This crate is a local fork of `hayro-jbig2`, vendored from upstream commit
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
  false`; `hayro-ccitt` resolved to the local fork in `crates/ccitt`;
  upstream's workspace lint set is not inherited, so the crate is linted
  with clippy's defaults plus upstream's `too_many_arguments` allowance.

- `examples/` dropped; `LICENSE-APACHE`, `LICENSE-MIT`: upstream's
  symlinks to its repository root replaced by the files themselves.

The base is ahead of the 0.3.1 release on crates.io by its later fixes
(among them a panic on a symbol dictionary that reuses the contexts of one
with another template, integer overflows on extreme values, and recovery
of the regions before a truncated or failing segment), a byte-at-a-time
fast path for uniform runs in generic regions, a strict mode, and a changed
API: embedded global segments are parsed once into `Globals` and an image
is opened with `Page::new_embedded`.
