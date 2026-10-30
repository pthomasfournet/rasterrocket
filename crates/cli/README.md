# rasterrocket-cli (`rrocket`)

The command-line renderer: PDF pages, and comic archives (`.cbz`, `.cb7`,
`.cbt`), to PPM, PGM, PBM or PNG files, with `pdftoppm`-style flags.

```bash
cargo build --release -p rasterrocket-cli            # CPU only
target/release/rrocket -r 300 --gray --png document.pdf ./out
```

A bare output prefix such as `out` writes to a temporary RAM directory that is
removed when `rrocket` exits; use a path (`./out`) or `--no-ram` to keep the
files. JPEG and TIFF output are not implemented yet.

GPU features are passed by name and forwarded to the library
(`--features "nvjpeg2k,gpu-icc,cache"`, `--features vulkan`). Every flag, the
size limits and the exit codes are in the
[CLI reference](../../docs/cli-reference.md). Build from this repository;
the crates.io release (1.0.1) is abandoned.
