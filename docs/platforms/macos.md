# rasterrocket on macOS

macOS is a supported platform. The whole render path is native code, and on
Apple silicon it uses the NEON instructions of the M-series and A-series
cores. It needs no Homebrew packages and no extra fonts.

- **Tested on:** macOS 27.2 on Apple silicon (A18 Pro, 8 GB).
- **CI:** every push is linted with clippy on `macos-latest`. The tests that
  behave differently on macOS also run there: system font discovery, the
  predefined CJK character maps and the memory probe.
- **Intel Macs:** the x86-64 code paths (AVX2, with a scalar fallback) are
  the ones every Linux PC uses, but they are not tested on macOS.

## Install

1. Install the Xcode Command Line Tools, which provide the C linker:

   ```bash
   xcode-select --install
   ```

2. Install Rust 1.99 or newer, from [rustup.rs](https://rustup.rs). With
   Homebrew instead, run `brew install rustup`, put
   `/opt/homebrew/opt/rustup/bin` on your `PATH` (Homebrew prints the line
   to add to `~/.zshrc`), and run `rustup default stable`.

3. Build the command-line tool:

   ```bash
   git clone https://github.com/pthomasfournet/rasterrocket
   cd rasterrocket
   cargo build --release -p rasterrocket-cli
   ```

   The binary is `target/release/rrocket`. Copy it anywhere on your `PATH`.

To use the library from your own Rust project, add the git dependency shown
in the [README](../../README.md). There are no macOS-specific features to
enable.

## Render something

```bash
rrocket -r 300 --gray --png document.pdf ./page
```

This writes `page-1.png`, `page-2.png`, … to the current directory. The
flags follow `pdftoppm`; see the [CLI reference](../cli-reference.md).

## Fonts

A document that does not embed its fonts is drawn with the fonts macOS
already ships. Nothing needs installing. rrocket searches these directories:

- `/System/Library/Fonts` (including `Supplemental`)
- `/Library/Fonts`
- `~/Library/Fonts`, where fonts you install for yourself go
- any directory named by `RROCKET_FONT_DIR`

The standard PDF fonts are drawn with the metric-compatible faces macOS
ships: Arial for Helvetica, Times New Roman for Times, Courier New for
Courier. Line breaks and spacing therefore match the document.

On a stock macOS 27 install, text in a non-embedded font of each script is
drawn with:

| Script | Face |
|---|---|
| Japanese | Hiragino Sans; Hiragino Mincho ProN for serif text |
| Simplified Chinese | Hiragino Sans GB; Songti SC for serif text |
| Traditional Chinese | Heiti TC |
| Korean | Apple SD Gothic Neo |
| Arabic | Al Bayan |
| Thai | Sukhumvit Set |

Characters that no installed face has come from the faces built into the
binary: Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai and symbols. Fonts
embedded in the PDF are always used as they are.

## Writing output to RAM

On Linux, rrocket writes pages for a bare prefix such as `out` to a
temporary RAM directory under `/dev/shm`. macOS has no `/dev/shm`, so on a
Mac:

- a bare prefix writes to the current directory, like any other prefix;
- `--ram` on its own is refused with a message saying so;
- `--ram-path DIR` writes into a RAM disk you have mounted. rrocket watches
  free memory, using the same figure as macOS's `memory_pressure` tool, and
  moves to disk if it runs low.

You rarely need it: on Apple silicon the internal SSD is as fast as a RAM
disk for page output. To mount a 2 GiB RAM disk at `/Volumes/rrocket`:

```bash
diskutil erasevolume APFS rrocket $(diskutil image attach --noMount ram://4194304)
rrocket --ram-path /Volumes/rrocket/out -r 300 document.pdf page
```

The directory you name must not exist yet. rrocket creates it, makes it
readable by you alone, and removes it with its contents when it exits.

## GPU

rrocket does not use the Mac's GPU. The CPU path is the supported one on
macOS, and the release build needs no GPU features.

- **Metal:** there is no Metal backend.
- **Vulkan through MoltenVK:** the `vulkan` feature builds the cross-vendor
  kernels, but it has not been tested on a Mac. Those kernels are off by
  default on every platform, because the CPU paths are faster.
- **Hardware JPEG decoding (VideoToolbox):** not used. On the scanned
  documents measured, JPEG decoding is about 15 % of render time, too little
  for the media engine to pay off.

## Comparing against pdftoppm

The comparison scripts in `tests/compare/` need a few Homebrew packages:

```bash
brew install poppler imagemagick bash
```

The scripts need bash 4.4 or newer; macOS ships bash 3.2, and Homebrew's
bash takes precedence once installed. See
[Getting Started](../getting-started.md#dependencies) for the full list.

## Known gaps

- No GPU acceleration (see above).
- Intel Macs are not tested.
- The SVE2 code path (`nightly-sve2`) cannot run on Apple silicon, which has
  no SVE. It is for Arm server chips.
