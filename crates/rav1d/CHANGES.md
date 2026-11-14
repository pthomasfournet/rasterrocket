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

## Single-picture decode

- `src/still.rs` (new), re-exported at the crate root: `decode_still(obu,
  StillLimits) -> Result<StillPicture, StillError>` decodes the first
  picture of a run of OBUs on the calling thread (one worker, no frame
  delay), with film grain applied, the highest spatial layer of operating
  point 0 alone, the frame size limit from `StillLimits::max_pixels`, and
  no logger (failures are the returned error, nothing is written to
  stderr).  `StillPicture` keeps the decoder's picture and carries the
  size, bit depth, chroma layout and the sequence header's colour
  description as ITU-T H.273 code points with the range flag;
  `StillPicture::plane` lends each plane from the decoder's buffer as
  `&[u8]` (8-bit pictures) or `&[u16]` (10 and 12 bits) with its stride in
  samples, the geometry having been checked once when the picture was
  wrapped.  The module has no `unsafe` code: it is built on the `Decoder`
  and `Picture` types of `rust_api`.

  The planes were first handed out as tight owned copies; lending them
  removes a copy of every sample (2000 × 3000 4:2:0 page through the comic
  crate, one thread, Ryzen 9 9900X3D, medians of 21: 34.0 → 31.9 ms at 8
  bits, 37.9 → 35.8 ms at 10).

- `src/still.rs`: `still_layout(obu) -> Result<StillLayout, StillError>`,
  re-exported at the crate root, walks the OBU framing and parses each
  sequence header with the decoder's own `rav1d_parse_sequence_header`,
  without decoding, and returns the widest chroma layout any of them
  declares.  `StillLayout` derives `PartialOrd`/`Ord`, its variants ordered
  by the chroma samples they carry.  A caller bounds the planes
  `decode_still` will allocate by that layout rather than by what a
  container declares of the stream.

- `src/still.rs`: `decode_still` returns `StillError::NoPicture` for empty
  data before the decoder sees it; `Decoder::send_data` validates its input
  with `validate_input!`, which aborts the process in a debug build.

- `src/rust_api.rs`: `Picture::color_primaries_code`, `transfer_code`,
  `matrix_code` and `full_range`, the sequence header's colour description
  as code points, beside the existing accessors that return `av-data`
  enums; `Picture::plane_wide`, a plane as `&[u16]`, for pictures of more
  than 8 bits (the existing `Picture::plane` gives their bytes).
