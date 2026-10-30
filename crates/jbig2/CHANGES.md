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

## Upstream fixes brought over

Upstream commits to `hayro-jbig2/` after the base, applied as upstream
wrote them with their tests. The base in `UPSTREAM.toml` is unchanged;
`reviewed` records how far upstream history was examined.

- `96789708f` (#1629) A symbol dictionary may define and export up to
  2^20 symbols each (was 65 535 in total). Streams with more symbols,
  which used to fail, now decode; the `many_symbols` tests of
  `tests/symbol.rs` and `tests/huffman.rs` are no longer ignored.
- `86c0da15f` (#1630) A text region or symbol dictionary refers to at most
  2^20 symbols across the dictionaries it names.
- `44d78c7ec` (#1631) A height-class collective bitmap is bounded by area
  (65 535² pixels) rather than by its width, which legitimately exceeds
  65 535 when many symbols share one height class.
- `49fd125c0` (#1640) The uniform-byte fast path of generic regions is
  restated (`skip_mps` renamed `decode_mps_run`); decoding is unchanged.
- `5a4ddbb86` (#1671) A page wider or taller than 65 535 pixels is
  rejected when its page information segment is parsed, before any
  region is decoded. Test (this fork):
  `page::page_larger_than_bitmap_limit_is_rejected`.

## Upstream review (2026-10-30)

Upstream history through `a543cdba5` (2026-10-29, crates.io 0.3.1) was
reviewed by commit; the five commits above are every change to
`hayro-jbig2/` after the base, and all are brought over.
