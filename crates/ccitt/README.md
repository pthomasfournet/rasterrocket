# hayro-ccitt

> **Local fork.** This directory is the rasterrocket workspace's local fork
> of [`hayro-ccitt`](https://github.com/LaurenzV/hayro/tree/main/hayro-ccitt), based on an upstream
> commit made after its 0.4.0 release. That base commit is recorded in
> [`UPSTREAM.toml`](UPSTREAM.toml); every change made here, and every
> upstream fix brought over since the base, is listed in
> [`CHANGES.md`](CHANGES.md). The crate is not published (`publish = false`)
> and changes are not sent upstream.
>
> `rasterrocket-interp` uses it to decode `CCITTFaxDecode` images, and the
> local `hayro-jbig2` fork uses it for MMR-coded regions.
>
> Everything below is upstream's README, kept as written. Its links to
> crates.io, docs.rs and the upstream issue tracker describe the upstream
> crate, not this fork.

<!-- cargo-rdme start -->

A decoder for CCITT fax-encoded images.

This crate implements the CCITT Group 3 and Group 4 fax compression algorithms
as defined in ITU-T Recommendations T.4 and T.6. These encodings are commonly
used for bi-level (black and white) images in PDF documents and fax transmissions.

The main entry point is the [`decode`] function, which takes encoded data, a
[`DecoderContext`], and outputs each decoded row as a list of color changes
through a [`Decoder`] trait that can be implemented according to your needs.

The crate is `no_std` compatible but requires an allocator to be available.

## Safety
Unsafe code is forbidden via a crate-level attribute.

## License
Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

[`decode`]: https://docs.rs/hayro-ccitt/latest/hayro_ccitt/fn.decode.html
[`Decoder`]: https://docs.rs/hayro-ccitt/latest/hayro_ccitt/trait.Decoder.html
[`DecoderContext`]: https://docs.rs/hayro-ccitt/latest/hayro_ccitt/struct.DecoderContext.html

<!-- cargo-rdme end -->
