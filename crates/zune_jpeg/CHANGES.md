# Changes from upstream

This crate is a local fork of `zune-jpeg`, vendored from upstream commit
`9706202c268cd69276a4a1144261a988a0d77440` (2026-05-01, crate version 0.5.15
plus unreleased changes). It is not published and changes are not sent
upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits to the crate,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- The fork was first vendored at the 0.5.15 release (upstream commit
  `31d81fed`, 2026-03-26), then moved to the base above. The base is ahead
  of the release by: arithmetic entropy decoding (SOF9 and SOF10, the
  `arith` feature); a bounds check missing from the AVX2 and NEON
  YCbCr → RGB conversions; a corrected component stride for unusual
  sampling factors and the up-sampling scratch space sized for them; no
  panic on an unusual component count, nor on an oversized DC prediction;
  wrapping multiplication when a coefficient is dequantised. It predates upstream's resumable (incremental) decoding, which this
  project does not use; upstream history after the base has not been
  reviewed as a whole, and the entries below name the later upstream
  commits whose fixes were brought over.

- Kept: `src/`, `tests/`, `README.md`, `Changelog.md`, `Benches.md` and the
  three licence files (upstream's symlinks to its repository root replaced
  by the files themselves). Left out: `fuzz/` (upstream's fuzz targets) and
  `.gitignore`.

- `Cargo.toml`: `rust-version` is inherited from the workspace, so every
  crate of the repository builds with the same minimum toolchain;
  `publish = false`; `zune-core` is the crates.io release instead of
  upstream's sibling directory (the crate builds against 0.5.3 unchanged);
  the unused `zune-ppm` dev-dependency is dropped; the lint table is
  empty, so the crate is linted with clippy's defaults.

- `tests/`: the images the suites read from upstream's repository-level
  `test-images/jpeg/` directory are copied into `tests/images/` and the
  `include_bytes!` paths point there.

- `src/lib.rs`: the crate-level `#![warn(clippy::pedantic, ...)]` block and
  the stray `#![macro_use]` are removed, and the `idct` module is private
  under every configuration (upstream made it public under `cfg(fuzzing)`
  for the fuzz target left out above).

- Clippy's default findings are fixed in place, none changing behaviour:
  `as_chunks` / `as_chunks_mut` for `chunks_exact` with a constant size
  (and the array conversions that made redundant), a redundant reference
  in a `write!` argument, and doc-comment layout.

- The sources are formatted with the workspace's rustfmt settings
  (upstream uses nightly-only options).

## Changes in this fork

- **Sequential frames coded in several scans decode correctly (output
  changes).** A baseline or extended-sequential frame whose components are
  spread over several scans is buffered whole and decoded scan by scan.
  Three defects, all in `mcu.rs`:

  - The entropy decoder finds the marker that follows a scan before the
    decoder reaches it (the bit reader up to eight bytes early), and the
    row loop acted on it at the end of the row in progress: a scan whose
    last rows fit in those bytes lost them (a flat 16×16 image with one
    scan per component decoded its second block row as zero
    coefficients). A marker that ends the scan is now acted on after the
    scan's last row, or when the scan's data is used up. The same held
    for a single-scan frame followed by a marker other than EOI.
  - A scan of one component of a subsampled frame was read with the MCU
    row count and each row placed `vertical_sample` block rows apart; the
    scan has the component's own number of block rows, one after the
    other. A scan of several components (`Y` then `Cb Cr`, say) was read
    one block per component in raster order; it is coded in MCUs.
  - A scan whose last block leaves no marker in the lookahead ended the
    decode; the marker after it is now read. Each scan starts its own
    restart interval.

  Upstream fixed the first two later (commits `90b6bbca`, `0346b875`); the
  code here is ours. `tests/scan_layouts.rs` decodes one set of
  coefficients in every scan layout and requires identical pixels.

- **Arithmetic decoding, three fixes brought over from upstream.**
  `BitStream::marker_ends_data` (upstream `52300d39`): a row of an
  arithmetic-coded scan is not cut short at a marker, the decoder being
  fed zero bytes until the scan is complete. The decoder is fed zeros
  after an acknowledged EOI as after any marker (`99a07c79`). A
  progressive AC scan in which the decoder finds no end of band keeps the
  coefficients decoded so far and skips the scan's remaining blocks
  instead of failing the image (`8cccb0b0`).
