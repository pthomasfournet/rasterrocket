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

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.100), so
  every crate of the repository builds with the same minimum toolchain;
  the other workspace-inherited fields are resolved to values; `publish =
  false`; `hayro-ccitt` resolved to the local fork in `crates/ccitt`;
  upstream's workspace lint set is not inherited, so the crate is linted
  with clippy's defaults plus upstream's `too_many_arguments` allowance.
  The pattern-shift table in `decode/pattern.rs` iterates a pattern's
  rows by slice rather than by index (clippy's `needless_range_loop`
  from Rust 1.100); behaviour is unchanged.

- `examples/` dropped; `LICENSE-APACHE`, `LICENSE-MIT`: upstream's
  symlinks to its repository root replaced by the files themselves.

The base is ahead of the 0.3.1 release on crates.io by its later fixes
(among them a panic on a symbol dictionary that reuses the contexts of one
with another template, integer overflows on extreme values, and recovery
of the regions before a truncated or failing segment), a byte-at-a-time
fast path for uniform runs in generic regions, a strict mode, and a changed
API: embedded global segments are parsed once into `Globals` and an image
is opened with `Page::new_embedded`.

## Changes in this fork

- An MMR-coded generic region whose decoder fails to allocate memory
  (`hayro_ccitt::DecodeError::OutOfMemory`) fails with the new
  `DecodeError::OutOfMemory` in strict and lenient mode alike, and the
  page loop does not recover an immediate region that failed this way;
  the error reaches `Page::decode`'s caller. Other MMR failures keep
  upstream's behaviour: an error in strict mode, the rows decoded so far
  in lenient mode. Output changes only when an allocation fails. Tests:
  `decode::generic::tests`.
- Decode budget (`src/budget.rs`). Every bitmap, symbol copy, pattern
  table, halftone gray-scale image and skip mask is charged, before it
  is allocated, to a cumulative memory budget, and the allocation itself
  is fallible (`DecodeError::OutOfMemory`). A page may allocate 16 times
  its area plus 2^26 bits besides the page bitmap, at most 2^33 bits (1
  GiB); the global segments of a file or a `JBIG2Globals` stream may
  allocate 2^31 bits. Decoding a symbol instance of a text region or of
  a refinement/aggregate symbol charges 64 pixels, whether or not it
  writes any, plus the pixels it writes; a halftone region charges grid
  cells × pattern area; all to a compositing budget of 64 page areas
  plus 2^28 pixels, at most 2^36 (2^32 for the globals). An exhausted
  budget fails the segment with `OverflowError::MemoryBudget` or
  `OverflowError::WorkBudget`, and like `DecodeError::OutOfMemory` the
  error fails `Page::decode` in lenient mode too, where other errors of
  an immediate region drop the region and keep the page: once spent, a
  budget would fail every later region of the page as well. Over the 579
  JBIG2 documents of the archive corpus (first 40 pages each, 21 275
  images) the largest page used 21 % of its memory budget and 1.6 % of
  its compositing budget, and the largest globals 2 % of theirs; no
  output changes. Tests: `tests/budget.rs`, `budget::tests`,
  `bitmap::tests`, `decode::text::tests` (instance and pixel work),
  `decode::symbol::tests` (every allocation of a dictionary, including
  its collective bitmap and exported copies); the exhausted-decoder test
  of `tests/malformed.rs` declares a region that the budget admits. The
  budget's counters are `Cell`s, which leaves the auto traits of the
  public types as upstream has them with the `std` feature:
  `DecoderContext` is `Send` but not `Sync` (its lazily built standard
  Huffman tables already were not), and `Globals` is `Send + Sync`. A
  compile-time assertion in `lib.rs` pins both. Every fallible
  reservation, of a charged vector or of the reused page bitmap, goes
  through `budget::try_reserve`. Every budget is constructed explicitly:
  `Globals` decode under the globals budget, and a `DecoderContext`
  starts with an empty budget that each page replaces with its own, so a
  decode path that skipped that step would fail with a budget error
  rather than run under the globals limits.
- The limit of 32 symbol instances per byte of segment data applies to a
  symbol dictionary as a whole: its aggregate symbols (6.5.8.2) draw on
  one allowance of 32 per byte of the dictionary's data, where upstream
  gave each aggregate the whole allowance. A dictionary whose aggregates
  declare more instances in total fails with
  `SymbolError::TooManyInstances`. Tests: `decode::symbol::tests`.
- `Bitmap::combine` iterates only the source rows that fall inside the
  destination (`Bitmap::clipped_span`, which also clips the columns
  counted by `Bitmap::clipped_area`), so a symbol placed far outside its
  region costs nothing per row. A refinement/aggregate symbol (6.5.8.2)
  looks its referenced symbols up in the input and new symbol lists
  (`SymbolSet`) instead of copying both into a new list per symbol,
  which was quadratic in the dictionary size. `Bitmap` is no longer
  `Clone`, so every copy goes through the charged `Bitmap::try_clone`.
  Output is unchanged.
- The first adaptive template pixel of a pattern dictionary's collective
  bitmap is placed at x = -HDPW (6.7.5) for every pattern width: the AT
  pixel x offset is an `i16`. Upstream negated `HDPW as i8`, which
  overflowed at HDPW = 128 (a panic with overflow checks) and put the
  pixel at 256 - HDPW for HDPW > 128, decoding those dictionaries with the
  wrong context. Output changes only for patterns 128 or more pixels
  wide. Tests: `decode::pattern::tests`,
  `halftone::patterns_at_least_128_pixels_wide`.

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
- `90e1a3a96` (#1773) A symbol dictionary's export table has at most
  2 × symbols + 2 runs, and `ec28315a6` (#1774) at most as many empty
  height classes as new symbols; a stream of zero-length runs or of
  empty height classes, which looped without end, fails with
  `SymbolError::Invalid`. Tests: the two `symbol_dictionary_empty_*`
  streams of `tests/malformed.rs`.
- `9214724ae` (#1776) An uncompressed height-class collective bitmap is
  read from the segment before it is allocated.
- `14fe0d9c6` (#1778), adapted: the exported-symbol list reserves (and
  charges to the budget) no more than the dictionary's symbol count, and
  an export table that exports more symbols than declared fails at the
  run that overshoots instead of after the copies, so the charged
  reservation is never outgrown. The new-symbol lists keep their
  reservation by the declared count, charged to the budget, which
  bounds it; reserving less would let them grow uncharged.
- `c04e1e46b` (#1797), adapted: a height-class collective bitmap, the
  one bitmap bounded by area rather than by its sides, is also at most
  2^22 pixels wide (and 65 535 high), which keeps its coordinates plus
  offsets within `i32`. Every other bitmap keeps the 65 535-pixel side
  limit, tighter than upstream's 2^22. Test: `bitmap::tests`.

## Upstream review (2026-10-30)

Upstream history through `a543cdba5` (2026-10-29, crates.io 0.3.1) was
reviewed by commit; the five commits above are every change to
`hayro-jbig2/` after the base, and all are brought over.

On 2026-11-03 it was reviewed through `3f3825c3e` (2026-11-02, crates.io
0.4.0). Nothing is brought over: #1731 adapts the MMR region decoder to
an upstream `hayro-ccitt` API change that the ccitt fork leaves out, and
#1732 is a version bump and changelog.
