# Changes from upstream

This crate is a local fork of `hayro-jpeg2000`, vendored from upstream commit
`79926198e557e06fea8a8edcb0bc577eb0130ec5` (2026-10-23). It is not published
and changes are not sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `Cargo.toml`: workspace-inherited fields resolved to values; `publish =
  false`; the `image` integration is no longer a default feature (the test
  suite enables it with `--features image`).

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
