# rasterrocket on the BSDs and Windows

Windows is built and tested in CI, on GitHub's Windows runner, and its
CUDA and Vulkan features on one machine of our own, with an RTX 2080 SUPER
and, for Vulkan, an Intel UHD 630 (see
[Windows](#windows)). The BSDs are not tested, and no CI
job covers them. The code has no Linux-only dependency in its render path: it is
Rust, plus two C libraries that `cargo` builds, libdeflate for compression
and mimalloc, the command-line tool's allocator, and, on x86, the AV1
decoder's assembly kernels, which need `nasm`. It is expected to build on
the BSDs wherever Rust and a C compiler are available, but expected is not
tested. Please report what you find.

## FreeBSD, OpenBSD, NetBSD

- **Build:** as on [Linux](linux.md), with the system C compiler and, on
  x86, `nasm` (or `--no-default-features --features bundled-fonts`, which
  drops the assembly kernels).
- **Fonts:** searched in `/usr/local/share/fonts`, where the ports and
  packages install them, as well as `~/.fonts`, `~/.local/share/fonts`, and
  any directory named by `RROCKET_FONT_DIR`.
- **Writing output to RAM:** there is no `/dev/shm`, so a bare prefix writes
  to the current directory. rrocket cannot read the system's free memory
  here. Without it, pages could not be moved to disk before memory runs out,
  so `--ram` and `--ram-path` are refused.
- **GPU:** none tested. The Vulkan feature may work wherever a Vulkan 1.3
  driver exists; untested.
- **Printing and fax:** `--print` and `--fax-to` submit through `lp`, which
  the CUPS port or package provides; see the [Linux guide](linux.md#printing-and-fax).

## Windows

- **CI:** each push runs clippy on `windows-latest`, then the test suites
  of the renderer, interpreter, library, font, parser, colour, encoder,
  comic and command-line crates (the last including end-to-end runs of the
  binary). The runner has no GPU.
- **Build:** with the MSVC toolchain from `rustup`, the Visual Studio
  Build Tools and, on x86-64, `nasm` on `PATH` (`choco install nasm`; the
  installer leaves it in `C:\Program Files\NASM`) for the AV1 decoder's
  kernels; `--no-default-features --features bundled-fonts` builds without
  them.
- **Fonts:** searched in `C:/Windows/Fonts`. Fonts installed for one user
  only (`%LOCALAPPDATA%\Microsoft\Windows\Fonts`) are not searched yet; name
  that directory with `RROCKET_FONT_DIR`.
- **Writing output to RAM:** there is no `/dev/shm`, so a bare prefix writes
  to the current directory and `--ram` is refused. `--ram-path` writes into
  a directory on a RAM disk you have mounted, moving to disk once the
  available physical memory falls below 1 GiB.
- **GPU (CUDA):** the CUDA features (`nvjpeg`, `nvjpeg2k`, `gpu-icc`,
  `gpu-deskew`, `cache`) need CUDA 13.4 or later and the Visual Studio 2026
  Build Tools (C++ workload); CUDA 13.1's and 13.3's nvcc refuse VS 2026's
  MSVC 14.52. A plain `cargo build` finds nvcc in `%CUDA_PATH%\bin`, which
  the CUDA installer sets, the import libraries in `%CUDA_PATH%\lib\x64`,
  and nvcc's host compiler `cl.exe` from the Visual Studio installation, so
  no developer prompt is needed; `NVCC` names another nvcc and `NVCC_CCBIN`
  another `cl.exe`. At run time the CUDA DLLs (`%CUDA_PATH%\bin\x64`, put
  on `PATH` by the installer) must be on `PATH`. `nvjpeg2k` needs NVIDIA's
  nvJPEG2000 Windows redistributable archive for CUDA 13, unpacked, with
  `NVJPEG2K_PATH` naming its root (`include`, `lib`) and its `bin` on
  `PATH`. Built and tested on an RTX 2080 SUPER (`sm_75`, driver 621.24)
  with CUDA 13.5, the VS 2026 Build Tools (MSVC 14.52) and nvJPEG2000
  0.12.0: the GPU crate's suite with every CUDA feature, the deskew and ICC
  suites, and `--backend cuda` against `--backend cpu` on every fixture
  page at 150 dpi, RGB and gray, byte-identical but for nvJPEG2000's JPEG
  2000 decodes (up to 19 levels, as on Linux; those codestreams, under the
  irreversible 9-7 wavelet, now decode on the CPU, verified on the
  RTX 5070 only) and `--deskew`, within 1 level. Other NVIDIA GPUs on Windows are untested.
- **GPU (Vulkan):** the `vulkan` feature needs `slangc` on the build
  host, which the LunarG Vulkan SDK for Windows installs in
  `%VULKAN_SDK%\Bin`; the build finds it there (the installer sets
  `VULKAN_SDK` machine-wide), else on `PATH`, and `SLANGC` names another.
  At run time only the GPU driver's Vulkan loader (`vulkan-1.dll`) is
  needed. Built with SDK 1.4.363.1 (slangc 2026.17) and tested on the
  RTX 2080 SUPER (driver 621.24): the GPU crate's Vulkan tests, the
  GPU-page, image-cache and ICC parity suites, and every fixture page at
  150 dpi, RGB and gray, 8 and 16 bits, under `--backend vulkan` (ICC lookup
  table, image cache), `--backend vulkan --gpu-pages` and `--gpu-pages
  --gpu-devices all`, byte-identical to `--backend cpu`.
  Every page rendered on the device (`DevicePageCounts`: 3 024 on the
  device, 0 on the CPU, per mode); outline pages render there, so the
  device takes the 64-bit (paths) kernel. The largest fixture page
  (`corpus-09` page 415, 14.2 × 25.9 in) and the pages with the most device
  commands, rendered whole (`--band-mib 65536`) at 600 and 1200 dpi under
  `--gpu-pages`, are byte-identical too and lose no device to the Windows
  GPU timeout (TDR): the 1200 dpi `corpus-09` page, 1.6 GB of RGB, takes
  3.7 s for the whole process.
  The Intel UHD 630 beside it, with driver 31.0.101.2141 (Vulkan 1.3.215),
  passes the same runs: the GPU crate's Vulkan tests and the GPU-page,
  image-cache, ICC and teardown suites with `PDF_RASTER_VULKAN_DEVICE=intel`
  (the iGPU tests run on it), and every fixture page under
  `--backend vulkan --vulkan-device intel`, `--backend vulkan --gpu-pages
  --vulkan-device intel` and `--gpu-pages --gpu-devices all`, byte-identical
  to `--backend cpu` (12 096 pages per variant). It has `shaderInt64`, so it
  takes the paths kernel, and renders every page offered under
  `--gpu-pages`; across both devices the 2080 SUPER took 7 442 pages and
  the UHD 630 4 654, none on the CPU. Its `maxStorageBufferRange` is 1 GiB:
  a page whose device pixels exceed it renders on the CPU after a warning
  naming the limit. The device stores each pixel in whole 32-bit words, 4
  bytes at 8 bits and 8 at 16 (gray pages render as RGB), so a page fits
  up to about 268 million pixels at 8 bits and 134 million at 16;
  `corpus-09` page 415 at 1200 dpi (1.6 GB of RGB, 2.1 GB of device
  pixels) does not. The TDR gate on it, the same pages whole at 600 and
  1200 dpi plus `corpus-09` at 950 dpi and `corpus-06` and `corpus-02` at
  1200 dpi in 16 bits, is byte-identical and loses no device; no whole
  process took over 3 s on the device. The gate ran on 2026-11-18, before
  the word layout, when the device stored three samples a pixel: the
  950 dpi page (1.0 GB then, the largest that fitted) and the 16-bit pages
  (0.8 GB then) take 1.33 GB and 1.11 GB of device pixels now, so they
  render on the CPU; the gate has not been run with the word layout. Its
  older driver, 26.20.100.7642, reports
  Vulkan 1.1 and is listed as unusable (`Vulkan API below 1.3`). Other GPUs
  under Windows are untested.
- **Printing:** `--print` sends the pages to the printer's driver through
  GDI, fitting, centring and turning each page to the sheet itself (see
  [Printing on Windows](../cli-reference.md#printing-on-windows));
  `--list-printers` lists the local and connected printers. A printer that
  prints to a file, such as Microsoft Print to PDF, needs `--print-file
  PATH`, which names the file instead of a save dialog:
  `rrocket --printer "Microsoft Print to PDF" --print-file out.pdf in.pdf`.
  The driver takes 8-bit sRGB bitmaps, so `--output-profile` and
  `--depth 16` are refused with `--print`. Ctrl-C, Ctrl-Break or closing
  the console aborts the job and removes the `--print-file` it was
  writing; a second Ctrl-C quits at once.
  Tested on Windows 11 Pro (build 26200) through Microsoft Print to PDF;
  no physical printer has been tested on Windows. The tests that print to
  it skip, saying so, where it is missing; `RR_REQUIRE_PRINT_TO_PDF=1` makes
  that a failure.
- **Fax:** `--fax-to` is refused: it submits through CUPS's `lp` or
  HylaFAX's `sendfax`, which Windows does not have, and Windows documents
  no way to hand an IPP Fax Out printer the number. Write the document with
  `--fax` and send it with a fax service.
