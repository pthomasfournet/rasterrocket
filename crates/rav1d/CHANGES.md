# Changes from upstream

This crate is a local fork of `rav1d`, the Rust port of the dav1d AV1
decoder, vendored from upstream commit
`2e7216c20ac2a75de204d6c8d7268b75a45959b4` (2026-10-27, crate version 1.1.1).
It is not published and changes are not sent upstream.

Every change keeps decoded output byte-identical to the vendored baseline
unless the entry says otherwise.

## Vendoring

- `UPSTREAM.toml` records the base (repository, path, commit, date,
  version). `scripts/check_vendored.py`, run weekly by the
  `Vendored upstream` workflow, reports upstream commits under `src/`,
  newer crates.io releases and RustSec advisories after that base. Bring
  wanted upstream changes over by hand and update `UPSTREAM.toml`.

- Kept: the Rust sources (`src/**/*.rs`), the assembly sources the `asm`
  feature builds (`src/x86/*.asm`, `src/ext/x86/x86inc.asm`,
  `src/arm/**/*.S` and the header they include, `src/arm/asm-offsets.h`),
  `build.rs`, `COPYING`, `README.md`, `THANKS.md`, and `doc/PATENTS` as
  `PATENTS`.

- Left out, none of it part of the library build: the C sources and headers
  of dav1d the port was made from (`*.c`, `*.h`, the `ppc`, `riscv`,
  `loongarch` and `win32` directories, which hold nothing else), the meson
  build (`meson.build`, `meson_options.txt`, `*.in`), the command-line tools
  (`tools/`), the C examples (`examples/`), the test suite and its scripts
  (`tests/`), packaging (`package/`), the remaining documentation (`doc/`),
  CI configuration (`.github/`, `.gitlab-ci.yml`), `benchmark.py`,
  `retranspile.sh`, `gcovr.cfg`, `NEWS.dav1d`, `CONTRIBUTING.md`,
  `Cargo.lock`, `rust-toolchain.toml` (a nightly pin; the library builds on
  stable for x86-64 and aarch64) and `rustfmt.toml` (nightly-only options;
  the vendored files are already formatted and stable `rustfmt` leaves them
  unchanged).

- `Cargo.toml`: `rust-version` is inherited from the workspace (1.100);
  `publish = false`; the workspace declaration (the crate was its own
  workspace root, with `tools`), the `exclude` list and the build profiles
  are gone, a member's profiles being ignored; the library is an `rlib`
  alone (upstream also builds a `staticlib` for C callers); no bench
  target.

- `Cargo.toml`: the build dependencies `cc` and `nasm-rs` and the
  dependency `raw-cpuid` are optional and enabled by the `asm` feature,
  their only user, so that a build without `asm` compiles no build tool and
  needs no assembler.  `asm` stays a default feature, as upstream: an x86
  build then needs the `nasm` executable on `PATH`, an arm build assembles
  its `.S` sources with the C compiler `cc` finds.
  `--no-default-features --features bitdepth_8,bitdepth_16` is the build
  without assembly.

- `paste` (unmaintained, RUSTSEC-2024-0436, which the workspace's
  `cargo deny` gate refuses) is replaced by its maintained successor
  `pastey`, a drop-in: `use paste::paste` becomes `use pastey::paste` in
  `src/itx.rs`, `src/wedge.rs` and `src/include/common/bitdepth.rs`.

- `Cargo.toml`: an explicit `[lints.clippy]` table.  The crate is linted
  at clippy's defaults, the workspace lint set not inherited; upstream's
  crate-level `#![expect(...)]` list in `src/lib.rs` is kept as is, and two
  lints that clippy 1.100 adds findings for in unchanged upstream code are
  allowed in the manifest: `chunks_exact_to_as_chunks` and
  `manual_is_multiple_of`.
