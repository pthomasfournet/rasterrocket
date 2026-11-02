# rasterrocket on Linux

Linux is the primary platform: every feature, including every GPU backend,
is developed and tested on it.

- **CI:** each push runs the test suite on x86-64 (`ubuntu-latest`),
  except the GPU tests, which need a device. On arm64 (`ubuntu-24.04-arm`)
  it runs clippy and the tests of the NEON code paths.
- **CPU:** x86-64 uses AVX2, or AVX-512 when built with
  `-C target-cpu=native` on a CPU that has it, with a scalar fallback.
  aarch64 uses NEON, plus SVE2 behind the nightly-only `nightly-sve2`
  feature.

## Install

```bash
sudo apt install build-essential        # the C linker (Debian, Ubuntu)
git clone https://github.com/pthomasfournet/rasterrocket
cd rasterrocket
cargo build --release -p rasterrocket-cli
```

You also need Rust 1.99 or newer, from [rustup.rs](https://rustup.rs). The
binary is `target/release/rrocket`.

## GPU

All GPU features are optional, and all of them fall back to the CPU when no
device is present.

| Backend | Feature flags | Needs |
|---|---|---|
| NVIDIA CUDA 13 | `nvjpeg2k`, `gpu-icc`, `gpu-deskew`, `cache`, … | CUDA 13 toolkit and driver |
| Vulkan compute | `vulkan` | the LunarG SDK (`slangc`) to build; a Vulkan 1.3 driver to run |
| VA-API JPEG decode | `vaapi` | `libva` |

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
