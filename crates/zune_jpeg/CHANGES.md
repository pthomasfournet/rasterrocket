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
  project does not use. Upstream history after the base is reviewed up to
  commit `ab211bb1` (2026-11-11, 0.5.16-rc3): "Upstream review" below
  lists what was brought over and what was left out.

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

- **Arithmetic decoding is on by default** (`arith` is an optional
  feature upstream). `tests/arithmetic.rs` decodes arithmetic-coded
  lossless transcodes of Huffman-coded files (sequential, restart
  intervals, a scan per component, progressive; four subsamplings, gray,
  four components) and requires the Huffman decode within one level, and
  damages every byte of six such streams without a panic.
  `tests/arithmetic_progressive.rs` and its images are upstream's (commit
  `8cccb0b0`), its strict decodes of the two jpegsuite streams changed (see
  below).

- **Arithmetic decoding, three fixes brought over from upstream.**
  `BitStream::marker_ends_data` (upstream `52300d39`): a row of an
  arithmetic-coded scan is not cut short at a marker, the decoder being
  fed zero bytes until the scan is complete. The decoder is fed zeros
  after an acknowledged EOI as after any marker (`99a07c79`). A
  progressive AC scan in which the decoder finds no end of band keeps the
  coefficients decoded so far and skips the scan's remaining blocks
  instead of failing the image (`8cccb0b0`); here (ours) the rows from the
  abandoned block row on do not count as intact for `intact_rows`, and a
  strict decode fails, as libjpeg reports the band a bad arithmetic code.
  Upstream's `tests/arithmetic_progressive.rs` decoded its two jpegsuite
  streams with such bands strictly; it now requires the strict decode to
  fail and the other to match libjpeg's pixels with fewer rows than the
  height intact (the streams' arithmetic decode differs from their Huffman
  transcodes by up to 54 levels: the band's later blocks are lost). Test:
  `arithmetic::an_ac_band_without_an_end_of_band_is_not_intact`.

- **`JpegDecoder::intact_rows` (new API): how much of an image was decoded
  from data that was present.** The number of leading image rows the last
  decode produced from entropy-coded data that was all there and well
  formed: the height for a complete stream, with or without its EOI; fewer
  when a scan's data stops (at the end of the input or at a marker)
  before the scan's last block, when the entropy decoder met an error a
  decode that is not strict carries on from, or when the scans decoded
  leave a component uncoded or a coefficient of a progressive frame short
  of its full precision. The Huffman bit reader counts the bits it makes
  up (`fill_bits`, a `u64`: zeros past a marker, an EOI or the end of the
  input), which are the lowest of its buffer, so one has been consumed exactly
  when fewer bits are left; the row loops ask after every row. Within
  eight bytes of the end of the input the buffer is filled by a separate
  cold routine that counts the bytes it reads past the end, so the hot
  refill is unchanged. An arithmetic-coded scan that stops at a marker
  short of its blocks is not detected (the arithmetic decoder reads zeros
  past a marker as a matter of course). `tests/intact_rows.rs` cuts noise
  images at every byte of their scan data. The row index passed to
  `decode_mcu_width` and `inner_decode_mcu_width` is named `row` (upstream:
  `mcu_height`).

- **Streams that stop short: a decode that is not strict draws more of
  them (output changes for such streams only).** Past the end of the input
  the bit reader supplies zeros, as it does past a marker, instead of
  failing once its buffer is empty: the row in progress is completed and
  the row loop stops at the next. `overread_by` counts bytes read past the
  end only (it counted the last byte of the input too, which stopped a
  complete single-scan stream without EOI one row early when its last
  rows were a few bytes long). A marker segment between scans that cannot
  be read, and a row that fails in a frame of several scans, end the image
  with the scans decoded so far drawn, where they failed the decode or
  left the image blank. So does a progressive frame of more scans than
  `max_scans` (100 by default), with no row counted intact; a strict decode
  still fails (`invalid_images::a_progressive_frame_past_the_scan_limit_is_drawn_partial`).
  The marker after the first scan of a progressive
  frame is read from the stream when the entropy decoder has not run into
  it (it was an error).

- **A DNL segment after the first scan is read and skipped** (it was an
  error, and ended a decode that is not strict with the scan's rows after
  the one the bit reader found it in reported damaged). The frame keeps
  the height of its header; a header height of zero is still refused.
  Within a scan's data the segment waits for the scan's last row, as the
  markers between scans do. Test:
  `intact_rows::a_dnl_segment_after_the_first_scan_is_skipped`.

- **`JpegDecoder::set_max_coefficient_bytes` (new API): a limit on the
  coefficients a frame holds whole.** A progressive frame, and a
  sequential frame whose components are coded in several scans, keep two
  bytes per sample of every component until the last scan, besides the
  output; a few dozen bytes of headers can declare a frame of gigabytes.
  Over the limit the decode fails before the coefficients are allocated,
  strict or not. The default is no limit, so decodes are unchanged
  otherwise. Both allocation sites share `JpegDecoder::coefficient_planes`.
  Test: `invalid_images::coefficients_over_the_limit_are_not_allocated`.

## Upstream review (9706202c..ab211bb1)

The 139 upstream commits to the crate after the base were read; the fixes
that apply to this fork's code are brought over by hand, each with a test
that failed before. Complete, well-formed streams decode to the same bytes
as before unless an entry says otherwise.

### Brought over

- **AC run past the end of a block** (`8800effb`, `168a3b7b`). A run that
  takes the zig-zag position past 63 was written at `position & 63` on the
  Huffman slow paths (baseline and progressive), over the DC or a low
  frequency, and at 63 on the fast paths, so one damaged stream decoded
  two ways. On every path, skipped blocks included, a strict decode now
  fails and any other puts the coefficient on position 63; the row counts
  as damaged for `intact_rows` (ours). Output changes for such streams
  only. Test: `invalid_images::ac_run_past_end_of_block` (upstream's,
  with the `intact_rows` checks added).

- **Huffman tables are checked scan by scan** (`88f5b1d2`). The tables of
  every component were required before the first scan, which refused a
  frame whose later scan uses tables defined between the scans. A later
  scan whose tables are missing ends a decode that is not strict with the
  earlier scans drawn and no row reported whole (ours). Tests:
  `non_interleaved::decode_non_interleaved_with_luma_last` (upstream's
  image), `non_interleaved::scan_with_undefined_tables`.

- **A scan of a component already coded is not decoded** (after `0041cb1f`,
  `590214e4`, `e62542ac`; the code is ours). A sequential frame codes each
  component once; decoding a further scan of it replaced the component
  (with gray when the scan had no data), and walked every row of blocks
  however little data the scan had: 1,000 empty scans appended to a frame
  cost 1,000 passes over it. A strict decode fails; any other ends the
  image there, complete. That bounds the scans of a frame by its
  components, so upstream's limit of `max_scans` scans is not needed.
  Output changes for such streams only. Tests:
  `non_interleaved::scan_after_the_last_component_is_not_part_of_the_image`,
  `invalid_images::scans_after_the_last_component_are_not_walked`.

- **Gray output of a vertically subsampled frame coded in several scans**
  (`393425a6`). The rows of MCUs were cut from the component buffer by its
  length, which is double for such a decode: a sequential frame panicked,
  a progressive one drew the wrong rows. Test:
  `color_output::luma_matches_y_of_ycbcr_for_vertically_sampled_images`.

- **Components the output colour space does not use** (`94661f44`,
  `24a891ae`): their empty buffers are neither carried over nor
  up-sampled (a panic for a subsampled frame decoded to `LumaA`, now the
  error other layouts give). Tests:
  `color_output::unsupported_lumaa_output_*`.

- **Sampling factors that are not 1 or 2 beside a vertically sampled
  component** (`d94b25ad`): the rows held back until the next row of MCUs
  are the component's up-sampled ones, not its coefficients. Output
  changes for such frames. Test:
  `color_output::generic_upsampling_preserves_delayed_boundary_rows`.

- **CMYK and YCCK to RGBA** (`fe9d4da7`): rows and pixels are four bytes
  wide and alpha is 255. Tests: `color_output::cmyk_rgba_*`,
  `color_output::ycck_rgba_*`.

- **The input colour space is decided when the headers are read**
  (`9a8663b0`, `2fa0e417`; the rules are ours). It was settled during the
  decode, so `input_colorspace()` after `decode_headers()` said CMYK for
  an RGB frame with an Adobe transform of 0, and an APP14 segment after
  the frame header overrode the component count (a gray frame became a
  one-band image). The component count comes first (one component is
  gray, four are YCCK with transform 2 and CMYK otherwise), then the
  Adobe transform, then component identifiers `R`, `G`, `B` without an
  APP14 segment. Output changes for two layouts: an APP14 segment after
  the frame header of a gray or four-component frame, and three
  components named `R`, `G`, `B` with a transform of 1 or 2 (now YCbCr, as
  the segment says; they were taken for RGB). Tests: `app14` (upstream's
  suite and images, with `single_component_is_luma_with_any_app14_transform`).

- **A sequential Huffman-coded stream without its tables** (`aa4b64f7`)
  uses the tables of section K.3, as only a Motion JPEG frame with an
  `AVI1` segment did. Tests: `abbreviated`.

- **A data byte of 0xFF at the very end of the input** (after `d0e4fa32`,
  `95c0fa06`; the code is ours): the zero read past the end is not a
  stuffed byte, so the 0xFF counts as made-up bits for `intact_rows`.
  Test: `intact_rows::a_cut_after_a_data_byte_of_ff_is_not_whole`.

### Already here, in this fork's own code

- Sequential frames in several scans (`0346b875`, `90b6bbca`, `dc4211c6`)
  and the restart interval starting again at every scan (`ab211bb1`): see
  "Changes in this fork". Upstream's tests for them
  (`read_ahead_marker_does_not_drop_final_scan_row`,
  `restart_interval_resets_at_each_non_interleaved_scan`, the DC table
  selectors of `50974519`) pass on the fork.
- Arithmetic decoding (`52300d39`, `99a07c79`, `8cccb0b0`).
- Streams that stop short (`5fb24de4`, `c7b40903`, `5a5f9822`, `bcb686e6`,
  and their tests `73d77c56`, `c727560d`): `intact_rows` and the bit
  reader's count of made-up bits.
- `d133c8e4` restores a guard the base still has; `961f9324` and
  `2e540da3` change one table index and change it back.

### Left out

- **Resumable (incremental) decoding, cancellation, raw-plane, iMCU-row
  and scanline output**, with their fixes, tests, documentation and fuzz
  targets: APIs this project does not use, and most of the changes to the
  scan loops. `cd0e42f1` `6f862f29` `26b51679` `9944127e` `18cebfa3`
  `fb0f27bb` `4b609830` `26797e76` `babf5c25` `8016f59d` `8620bb88`
  `ebaefe98` `7e0999fe` `d221107a` `d77ed371` `441101c4` `a15916bf`
  `ea004bee` `43624422` `cd53f20f` `6dc33271` `96b4dbfd` `07f19feb`
  `2a7551a2` `b4aa76c3` `be589016` `9d2ba803` `9260d033` `dc61956e`
  `36f8e8d2` `077efb86` `a61dae59` `a1751bc5` `bd1f8193` `f0358ad0`
  `e096f7b8` `367588ef` `6f82b89e` `8e55ebdc` `969e2ac1` `6faddab4`
  `67f5b9f4` `ca04b109` `3c70c145` `7d7b1bbc` `20f759e2` `0f018e2e`
  `9f731656` `70b4ef1e` `2f393b08` `8ada51ac` `b09ebae8` `d320cc22`
  `d4edde40` `ed488353` `ef427a5e` `ec30f3c0` `c2bac6ab` `3f553ce2`
  `bfb334d8` `9e97be00` `21405d9f` `1fffc827` `249738f4` `50273ca5`
  `9b8ef2e9` `f4011ced` `9e623cc6` `2f9086b0` `dc9d4510` `083c0d33`
  `fb9fc04f` `c3b9adbe` `c65df8ee` `c5ddc597` `58d2d171` `49855297`
  `ba9fc244`. (`8620bb88` also reads every marker segment whole before
  parsing it; the checks are the same.)
- **A more accurate inverse DCT and YCbCr conversion** (`8385ce39`,
  `80193182`, `3820eb5f`, `c0b6a3fd`): 13-bit constants and other rounding,
  which change the samples of every image by a level here and there. Not
  a defect fix; to be taken, if at all, as a measured change of its own.
- **Features**: frames whose height comes from a DNL marker (`506a2af3`,
  `8f1d3832`; the PDF interpreter writes the image dictionary's height into
  the frame header, and the DNL segment is then read and skipped, see
  "Changes in this fork"), header parsing of lossless and 12-bit frames, which still
  do not decode (`23de4803`, `e668c57a`, `8cbb0712`), YCCK to CMYK output
  (`b39e0123`; the interpreter converts YCCK itself), `into_inner`
  (`cf820760`), SIMD for BGR and BGRA output (`f51cff38`, `ea8efb73`).
- **No effect here**: `2be3e47d` (speed of AC refinement only), `2d27aa26`
  (refuses an ICC chunk numbered out of range, which decodes today),
  `43aa7a29` (a segment cut short fails either way; matters to a retry),
  `b36e7eb1` (an overflow check on an offset the fork does not compute;
  the block position is bounds-checked), `e63a8d95` (restructuring),
  `7f54114c`, `b7689b86`, `5ea4f97a`, `bcb3a944`, `44d44e34`, `ca69cf57`
  (tidying, tests), the releases `a0793d5b`, `eff6a857`, `4a9c0c50`,
  `c1c7f497`, and the merges `35667b2f`, `c302881e`.
