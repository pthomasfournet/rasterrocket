# skrifa

> **Local fork.** This directory is the rasterrocket workspace's local fork
> of [`skrifa`](https://github.com/googlefonts/fontations/tree/main/skrifa), based on upstream
> release 0.44.0. The upstream base commit is recorded in
> [`UPSTREAM.toml`](UPSTREAM.toml); every change made here, and every
> upstream fix brought over since the base, is listed in
> [`CHANGES.md`](CHANGES.md). The crate is not published (`publish = false`)
> and changes are not sent upstream.
>
> `rasterrocket-font` uses it for glyph outlines, metrics and TrueType
> hinting. It depends on the local `read-fonts` fork in `crates/read_fonts`,
> and the workspace `[patch.crates-io]` table puts this fork in place of the
> registry crate for every dependent.
>
> Everything below is upstream's README, kept as written. Its links to
> crates.io, docs.rs and the upstream issue tracker describe the upstream
> crate, not this fork.

This crate aims to be a robust, ergonomic, high performance library for reading
OpenType fonts. It is built on top of the
[read-fonts](https://github.com/googlefonts/fontations/tree/main/read-fonts)
low level parsing library and is also part of the
[oxidize](https://github.com/googlefonts/oxidize) project.

## Features

### Metadata

The following information is currently exposed:

* Global font metrics with variation support (units per em, ascender,
descender, etc)
* Glyph metrics with variation support (advance width, left side-bearing, etc)
* Codepoint to nominal glyph identifier mapping
    * Unicode variation sequences
* Localized strings
* Attributes (stretch, style and weight)
* Variation axes and named instances
    * Conversion from user coordinates to normalized design coordinates

Future goals include:

* Color palettes
* Embedded bitmap strikes

### Glyph formats

| Source | Decoding | Variations | Hinting |
|--------|----------|------------|---------|
| glyf   | ✔️      | ✔️         | ✔️     |
| CFF    | ✔️      | -          | ✔️      |
| CFF2   | ✔️      | ✔️         | ✔️     |
| COLRv0 | ✔️      | -          | -       |
| COLRv1 | ✔️      | ✔️         | -       |
| EBDT   | ✔️*     | -          | -       |
| CBDT   | ✔️*     | -          | -       |
| sbix   | ✔️*     | -          | -       |

\* Raw support available through the `read-fonts` crate.

## Panicking

This library should not panic regardless of API misuse or use of
corrupted/malicious font files. Please file an issue if this occurs.

## The name?

Following along with our theme, *skrifa* is Old Norse for "write" or "it is
written." And so it is named.

## Safety

Unsafe code is forbidden by a `#![forbid(unsafe_code)]` attribute in the root
of the library.
