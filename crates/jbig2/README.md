# hayro-jbig2

> **Local fork.** This directory is the rasterrocket workspace's local fork
> of [`hayro-jbig2`](https://github.com/LaurenzV/hayro/tree/main/hayro-jbig2), based on an upstream
> commit made after its 0.3.1 release. That base commit is recorded in
> [`UPSTREAM.toml`](UPSTREAM.toml); every change made here, and every
> upstream fix brought over since the base, is listed in
> [`CHANGES.md`](CHANGES.md). The crate is not published (`publish = false`)
> and changes are not sent upstream.
>
> `rasterrocket-interp` uses it to decode `JBIG2Decode` images; its
> `hayro-ccitt` dependency is the local fork in `crates/ccitt`.
>
> Everything below is upstream's README, kept as written. Its links to
> crates.io, docs.rs and the upstream issue tracker describe the upstream
> crate, not this fork.

A memory-safe, pure-Rust JBIG2 decoder.

`hayro-jbig2` decodes JBIG2 images as specified in ITU-T T.88 (also known as
ISO/IEC 14492). JBIG2 is a bi-level image compression standard commonly used
in PDF documents for compressing scanned text documents.

## Status

This crate is currently under development.

## Safety
By default, the crate has the `simd` feature enabled, which uses the
[`fearless_simd`](https://github.com/linebender/fearless_simd) crate to accelerate decoding of JBIG2 images with halftoning. 
If you want to eliminate any usage of unsafe in this crate as well as its dependencies, you can simply disable this
feature and still get very good performance in the vast majority of cases. Unsafe code is forbidden
via a crate-level attribute.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
