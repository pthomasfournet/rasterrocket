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
binary: Greek, Cyrillic, Hebrew, Arabic, Devanagari, Thai, Bengali, Tamil,
Armenian, Georgian, Ethiopic and symbols. Fonts
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

The CPU path is the supported one on macOS, and the release build needs no
GPU features. The Apple GPU can run rrocket's Vulkan kernels through
MoltenVK, with output byte-identical to the CPU, but it does not make
rendering faster.

- **Metal:** there is no Metal backend.
- **Vulkan through MoltenVK:** see below.
- **Hardware JPEG decoding (VideoToolbox):** not used. On the scanned
  documents measured, JPEG decoding is about 15 % of render time, too little
  for the media engine to pay off.

### Vulkan through MoltenVK

Install the Vulkan loader, MoltenVK and the Slang shader compiler, which the
build needs to compile the kernels:

```bash
brew install molten-vk vulkan-loader vulkan-tools
vulkaninfo --summary          # lists the Apple GPU, driverName = MoltenVK
```

`slangc` comes from the [Slang releases](https://github.com/shader-slang/slang/releases)
(macOS arm64 archive); put its `bin` directory on `PATH`. Then build with the
`vulkan` feature and select the backend:

```bash
cargo build --release -p rasterrocket-cli --features vulkan
target/release/rrocket --backend vulkan document.pdf out
```

rrocket finds Homebrew's loader (`/opt/homebrew/lib/libvulkan.1.dylib`) or
the Vulkan SDK's (`$VULKAN_SDK/lib`, when `$VULKAN_SDK` is an absolute path;
a relative or empty one is ignored with a warning) without
`DYLD_LIBRARY_PATH`. When no loader loads, the error lists each path tried
and why it failed. On the
Apple GPU device buffers live in shared memory, which the CPU reads and
writes in place, so no data is copied to or from the GPU.

What runs on the GPU in this build is the antialiased path fill (solid
colour, rectangular clip), and it is off by default:
`PDF_RASTER_AA_FILL_THRESHOLD=0` sends every eligible fill to the device. A
`vulkan,gpu-jpeg-huffman` build also runs the parallel-Huffman JPEG
decoder, off by default as well (`PDF_RASTER_HUFFMAN_THRESHOLD=0` enables
it). Under the default `--backend auto`, no
Vulkan device is opened while every Vulkan kernel is off.

Measured on an A18 Pro (macOS 27.2, MoltenVK 1.4.2): the fixtures render
byte-identically on the GPU and the CPU. A page of 300 large solid polygons
takes about 1 ms per fill on the GPU against about 0.25 ms on the CPU, because
each fill waits for its result, and that round trip through Metal outweighs the
fill itself. So the kernel stays off.

## Comparing against pdftoppm

The comparison scripts in `tests/compare/` need a few Homebrew packages:

```bash
brew install poppler imagemagick bash
```

`ab_render.sh` and `ab_archive.sh` compare two builds, or the CPU and the
GPU (`AFTER_FLAGS="--backend vulkan"`), byte for byte. The scripts need bash 4.4 or newer; macOS ships bash 3.2, and Homebrew's
bash takes precedence once installed. See
[Getting Started](../getting-started.md#dependencies) for the full list.

## Printing and fax

macOS prints through CUPS, so `--print` works with the built-in `lp`;
`--printer NAME` takes a destination that `rrocket --list-printers` lists,
else the default destination. The pages are spooled in an anonymous file under the per-user
`$TMPDIR`. `--fax-to` needs a CUPS fax queue (`--fax-queue NAME`) or
HylaFAX's `sendfax`; macOS ships neither, so write the document with `--fax`
and send it another way. See the [CLI reference](../cli-reference.md#printing).

## Known gaps

- No GPU acceleration: the Vulkan kernels run on the Apple GPU but are
  slower than the CPU (see above).
- Intel Macs are not tested, nor M-series Macs: every result here comes
  from one A-series machine (NEON, the Apple GPU of an A-series chip).
- The SVE2 code path (`nightly-sve2`) cannot run on Apple silicon, which has
  no SVE. It is for Arm server chips.
