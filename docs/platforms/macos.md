# rasterrocket on macOS

macOS is a supported platform. The whole render path is native code, and on
Apple silicon it uses the NEON instructions of the M-series and A-series
cores. It needs no Homebrew packages and no extra fonts. It has been tested
on one A-series machine only; M-series Macs are untested (see
[Tested hardware](../../README.md#tested-hardware)).

- **Tested on:** macOS 27.2 on Apple silicon (A18 Pro, 8 GB).
- **CI:** every push is linted with clippy on `macos-latest`, GitHub's
  Apple-silicon runner (a virtual machine without GPU tests), which also
  runs the renderer, interpreter and library test suites, the font crate's
  (system font discovery) and the memory probe.
- **M-series Macs:** untested by hand. The NEON code is the same on every
  Apple core, and CI's virtualised runner runs the CPU suites above, but no
  render, benchmark or GPU result here comes from an M-series chip.
- **Intel Macs:** the x86-64 code paths (AVX2, with a scalar fallback) are
  the ones every Linux PC uses, but they are not tested on macOS. An Intel
  build also needs the `nasm` assembler (`brew install nasm`) for the AV1
  decoder's kernels; Apple-silicon builds assemble theirs with the Xcode
  tools.

## Install

1. Install the Xcode Command Line Tools, which provide the C linker:

   ```bash
   xcode-select --install
   ```

2. Install Rust 1.100 or newer, from [rustup.rs](https://rustup.rs). With
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

This writes `page-1.png`, `page-2.png`, … to the current directory. Flags
select the resolution, page range, scaling and output format; see the [CLI reference](../cli-reference.md).

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
GPU features. The A18 Pro's GPU runs rrocket's Vulkan kernels through
MoltenVK, with output byte-identical to the CPU, but it does not make
rendering faster there. M-series GPUs, which have more cores and memory
bandwidth, are untested.

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

A `vulkan,gpu-icc` build converts ICC-based CMYK images of at least
128 000 px through the profile's lookup table on the device, and a
`vulkan,cache` build keeps decoded images in the device image cache.
`--gpu-pages` renders pages whole on the device, beside the CPU workers:
fills, strokes, text, images, forms, annotation appearances, path clips and
soft masks, byte-identical to the CPU at 8 and 16 bits (see the
[CLI reference](../cli-reference.md#backend-selection) for what stays on the
CPU). Under the default `--backend auto`, no Vulkan device is opened unless
`--gpu-pages` is given, and then only once a page that may render on it is
reached; that page and every later one that may render on it wait for the
device and render on it.

Measured on an A18 Pro (macOS 27.2, MoltenVK 1.4.2): the fixtures render
byte-identically on the GPU and the CPU. MoltenVK on an M-series GPU is
untested.

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

macOS prints through CUPS, so `--print` uses the built-in `lp` (the CUPS
path is tested on Linux; printing from macOS is untested);
`--printer NAME` takes a destination that `rrocket --list-printers` lists,
else the default destination. The pages are spooled in an anonymous file under the per-user
`$TMPDIR`. `--fax-to` needs a CUPS fax queue (`--fax-queue NAME`) or
HylaFAX's `sendfax`; macOS ships neither, so write the document with `--fax`
and send it another way. See the [CLI reference](../cli-reference.md#printing).

## Known gaps

- No GPU acceleration: the Vulkan kernels run on the A18 Pro's GPU but are
  slower than the CPU (see above).
- Intel Macs are not tested, nor M-series Macs: every result here comes
  from one A-series machine (NEON, the Apple GPU of an A-series chip).
