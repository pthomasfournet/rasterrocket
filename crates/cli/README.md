# rasterrocket-cli (`rrocket`)

The command-line renderer: PDF pages, and comic archives (`.cbz`, `.cb7`,
`.cbt`, `.cbr`), to PPM, PGM, PBM, PNG, JPEG or TIFF files, to one fax document
(`--fax`, sent with `--fax-to`), or to a printer through CUPS (`--print`), with
flags for resolution, page range, scaling and output format. Pages render at 8 or 16 bits per sample (`--depth`),
as CMYK TIFF or JPEG (`--cmyk`, `--jpegcmyk`), and through an output ICC
profile (`--output-profile`, `--intent`, `--bpc`) that is embedded in the
files. A page larger than memory renders and is written in bands
(`--band-mib`), within a page budget (`--max-page-mib`).

```bash
cargo build --release -p rasterrocket-cli            # CPU only
target/release/rrocket -r 300 --gray --png document.pdf ./out
```

A bare output prefix such as `out` writes to a temporary RAM directory (under
`/dev/shm`, where the system has one) that is removed when `rrocket` exits; use a path (`./out`) or `--no-ram` to keep the
files; a `--fax` document is always written to disk, and `--print` writes no
files.

GPU features are passed by name and forwarded to the library
(`--features "nvjpeg2k,gpu-icc,cache"`, `--features vulkan`). The default
`av1-asm` feature builds the AV1 decoder's assembly kernels for AVIF comic
pages, which on x86 needs `nasm` on `PATH`. The default
`bundled-fonts` feature compiles in Noto fallback faces for Greek, Cyrillic,
Hebrew, Arabic, Devanagari, Thai, Bengali, Tamil, Armenian, Georgian,
Ethiopic and symbols (~0.75 MB). `--no-default-features` omits both;
`--no-default-features --features bundled-fonts` drops the assembly kernels
alone. Every flag, the
size limits and the exit codes are in the
[CLI reference](../../docs/cli-reference.md). Build from this repository;
the crates.io release (1.0.1) is abandoned.
