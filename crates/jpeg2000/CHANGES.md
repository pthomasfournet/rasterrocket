# Changes from upstream

This crate is a local fork of `hayro-jpeg2000`, vendored from upstream commit
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
  false`; the `image` integration is no longer a default feature (the test
  suite enables it with `--features image`).

- `LICENSE-APACHE`, `LICENSE-MIT`: upstream's symlinks to its repository
  root replaced by the files themselves.

## Diagnostics

- `j2c/codestream.rs`, `error.rs`: a CAP marker (0xFF50, ISO/IEC 15444-15
  HTJ2K) in the main header fails with `MarkerError::UnsupportedHtj2k`,
  whose message names HTJ2K, instead of the generic "unsupported marker".
  Decoding is otherwise unchanged; HTJ2K code-blocks are not decoded.

## Memory traffic under parallel decoding

Measured on 24 concurrent decodes of a 2 641 × 4 278 RGB 9/7 image (the
layered-scan background typical of scanned books) and of a 2 600 × 4 000 lossy
photo, against OpenJPEG under the same load. Each change keeps output
byte-identical on the benchmark images and passes the full test suite.

- `idwt::filter_vertical`: process the image in strips of 32 columns, copied
  into a contiguous buffer that stays in cache for all lifting steps, instead
  of running each lifting step over the whole image. Columns filter
  independently, so every sample sees the same operations in the same order.
  Scan layer 2.13× → 1.81× OpenJPEG's time.
- `mct::apply_inverse`: add the DC level shift of the first three components
  to the inverse-MCT results as they are stored; `apply_sign_shift` then runs
  only on the remaining components. 1.81× → 1.51×.
- `decode_tile`: stop re-zeroing the coefficient buffer before each
  component. Every code block of a decoded sub-band writes all of its
  coefficients and code blocks partition the sub-band, so stale values are
  never read. `TileDecompositions::coefficient_count` lost its only reader and
  is removed. 1.51× → 1.42× (photo 1.50×).
- `decode::store`: when the IDWT output covers exactly the component's whole
  image (one tile, no sub-sampling, no offset), swap it into the channel
  instead of copying it row by row; the IDWT reuses the channel's previous
  allocation. 1.42× → 1.25× (photo 1.27×).
- `Image::decode_u8` (new): the bytes of `decode` + `data_u8`. For three
  8-bit components without alpha, palette, channel definitions or a converted
  colour space, the inverse MCT and DC level shift are computed per 8-sample
  chunk and rounded straight into interleaved RGB, instead of being written
  back to the component buffers and read again. `mct` now has one
  `inverse_chunk` kernel shared by both paths; `j2c::decode` takes
  `defer_mct` and returns the pending transform. A new test checks
  `decode_u8` against `decode` + `data_u8` on every stream in the suite.
  1.25× → 1.08× (photo 1.21×).
- `Image::decode_u8`: the fused path also takes one 8-bit component under
  the same conditions. `j2c::decode` leaves that component's DC level shift
  unapplied and returns it (`Deferred::Shift`, beside `Deferred::Mct`, which
  replaces the returned `Option<PendingMct>`), and `ComponentData::shifted_u8`
  adds it as each sample is rounded to a byte, so the shifted samples are
  never written back. Output is byte-identical (the `decode_u8` equivalence
  test covers the single-component streams of the suite). On a 66-page
  grayscale layered scan (JPX background and foreground, JBIG2 mask), 300
  dpi, 24 threads: 0.70 s → 0.66 s and 0.76 s → 0.74 s in two interleaved
  10-run sets; one thread 3.74 s → 3.61 s. No change at 150 dpi, where the
  images are decoded reduced.

## Release tooling

- `Cargo.toml`: `[package.metadata.release] release = false`, so
  `cargo release` leaves this crate's version (the upstream base's) alone and
  creates no tag for it.

## Lints

- `Cargo.toml` allows `clippy::too_many_arguments` for upstream code we have
  not changed (`bitplane.rs`).

## Upstream changes brought over

Upstream commits to `hayro-jpeg2000/` after the base, applied as upstream
wrote them. The base in `UPSTREAM.toml` is unchanged; `reviewed` records
how far upstream history was examined.

- `cabf8edf4` (#1618), `c5956d7f1` (#1624), `7acf4212e` (#1627),
  `955629bb2` (#1628): `interleave_and_convert`, the conversion behind
  `DecodedImage::data_u8` (and `decode_u8` outside its fused path), rounds
  and interleaves 16 samples at a time with explicit SIMD for one to four
  components of one bit depth, scaling by `255 / (2^depth - 1)`. Output
  is byte-identical on every stream of the test suite; `data_u8` over the
  suite went from 387 ms to 117 ms for 1 GB of output.
- `a553afcb4` (#1648): the `logging` feature, its optional `log`
  dependency and the warnings behind it are removed. The fork never
  enabled the feature, so no build changes behaviour.

## Upstream review (2026-10-30)

Upstream history through `a543cdba5` (2026-10-29, crates.io 0.4.1) was
reviewed by commit; the five commits above are every change to
`hayro-jpeg2000/` after the base, and all are brought over (outside
the crate, #1618 and #1648 also touch upstream crates this fork does not
carry).

## Hardening

Limits and checks against codestreams whose header fields make the decoder
abort the process (by an allocation failure or a panic) instead of
returning an error. Found by fuzzing `Image::new` with `decode` and
`decode_u8`, seeded with the test-suite streams. Decoded output of every
image within the limits is unchanged.

- `j2c/codestream.rs`: an image is refused with `ImageTooLarge` when it
  has more than 65 535 tiles (tile indices, `Isot`, range over 0..=65534),
  more than 2^18 (tile, component) pairs (each tile copies the coding
  parameters of every component), or more than 2^30 decoded samples summed
  over its components at the decoded resolution (each is held as an
  `f32`). A few header bytes could otherwise size the tile list or the
  component buffers at hundreds of gigabytes. Tests:
  `limits::more_tiles_than_tile_indices`, `limits::more_samples_than_limit`.
- `j2c/build.rs`: one tile holds at most 2^24 precincts, 2^24 code-blocks
  and 2^26 (code-block, layer) entries over all its components and
  resolutions; a tile beyond them fails with `ImageTooLarge`. These
  structures are built for every resolution, skipped ones included, so
  small code-blocks or precincts could ask for billions of entries within
  the sample limit. Tests: `limits::more_code_blocks_than_limit`,
  `limits::more_code_block_layers_than_limit`.
