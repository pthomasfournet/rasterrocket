# rasterrocket on Linux

Linux is the primary platform: every feature, including every GPU backend,
is developed and tested on it. Current Linux testing is on an x86-64 machine
with an i7-8700K (AVX2), an RTX 2080 SUPER and a UHD 630; results up to
2026-11 also come from a Ryzen 9 9900X3D with an RTX 5070 and the CPU's AMD
Raphael iGPU, which now runs Windows. Other hardware is untested; see
[Tested hardware](../../README.md#tested-hardware).

- **CI:** each push runs the test suite on x86-64 (`ubuntu-latest`),
  except the GPU tests, which need a device. On arm64 (`ubuntu-24.04-arm`)
  it runs clippy and the tests of the NEON code paths; that runner is the
  only aarch64 Linux machine tested (no Raspberry Pi, Graviton or other
  arm64 server). A separate workflow
  reports line coverage of the CPU-only suites, and each release tag builds
  a PGO + BOLT x86-64 binary (`scripts/release_pgo_build.sh`), uploaded as
  a workflow artifact.
- **CPU:** x86-64 uses AVX2, or AVX-512 when built with
  `-C target-cpu=native` on a CPU that has it, with a scalar fallback.
  aarch64 uses NEON.

## Install

```bash
sudo apt install build-essential nasm   # the C linker and, on x86, the assembler (Debian, Ubuntu)
git clone https://github.com/pthomasfournet/rasterrocket
cd rasterrocket
cargo build --release -p rasterrocket-cli
```

You also need Rust 1.100 or newer, from [rustup.rs](https://rustup.rs). The
binary is `target/release/rrocket`. `nasm` assembles the x86 kernels of the
AV1 decoder behind AVIF comic pages; an arm64 build does not use it, and
`--no-default-features --features bundled-fonts` builds without those
kernels on any CPU.

## GPU

All GPU features are optional, and all of them fall back to the CPU when no
device is present.

| Backend | Feature flags | Needs | Tested on |
|---|---|---|---|
| NVIDIA CUDA 13 | `nvjpeg2k`, `gpu-icc`, `gpu-deskew`, `cache`, … | CUDA 13 toolkit and driver | RTX 5070 (`sm_120`); RTX 2080 SUPER (`sm_75`) from an nvcc 12.4 build on driver 570 (a CUDA 13 build needs driver 580 or later; a CUDA 13 build on its current driver is untested); other NVIDIA GPUs untested |
| Vulkan compute | `vulkan` | the LunarG SDK (`slangc`) to build; a Vulkan 1.3 driver to run | RTX 5070 and RTX 2080 SUPER (NVIDIA driver), AMD Raphael iGPU (RADV), llvmpipe; Intel (ANV) untested |
| VA-API JPEG decode | `vaapi` | `libva` | AMD Raphael iGPU (VCN 4.0, Mesa radeonsi); Intel untested |

See [Getting Started](../getting-started.md#gpu-optional) for what each
backend accelerates, and which kernels are wired in but off by default.

## Fonts

Non-embedded fonts are drawn from the installed fonts, without fontconfig.
rrocket searches:

- `/usr/share/fonts`
- `/usr/local/share/fonts`
- `~/.fonts`
- `~/.local/share/fonts`
- any directory named by `RROCKET_FONT_DIR`

The standard PDF fonts are drawn with the metric-compatible Liberation
faces when they are installed (`fonts-liberation` on Debian and Ubuntu,
`liberation-fonts` on Fedora, `ttf-liberation` on Arch), in whichever
subdirectory of `/usr/share/fonts` or `/usr/local/share/fonts` the
distribution puts them, and otherwise with faces built into the binary.
CJK text needs an installed CJK font such as Noto Sans CJK
(`fonts-noto-cjk`), since no CJK faces are built into the binary.

## Writing output to RAM

A bare output prefix such as `out` writes to a fresh directory under
`/dev/shm`. The directory is readable by you alone, its path is printed on
stdout, and it is removed when rrocket exits. If `MemAvailable` falls below
1 GiB, the remaining pages go to disk instead. To keep the pages, give a
path such as `./out`, or pass `--no-ram`. See the
[CLI reference](../cli-reference.md#output-destination-ram-backed-by-default).

## Printing and fax

`--print` hands the rendered pages to CUPS through `lp`, which the CUPS client
provides (`cups-client` on Debian and Ubuntu, `cups` on Fedora and Arch);
`--printer NAME` picks a queue, one of those `rrocket --list-printers`
lists, else the default destination (`lpstat -d`). The pages are spooled in an anonymous file under `$TMPDIR`,
else `/var/tmp`. `--fax-to NUMBER` sends a `--fax` document through a CUPS
fax queue (`--fax-queue NAME`) or, without one, HylaFAX's `sendfax`. Printing
is tested through a virtual IPP printer (`ippeveprinter`) only; no physical
printer, CUPS fax queue or HylaFAX server has been tested. See the
[CLI reference](../cli-reference.md#printing).
