# rasterrocket on the BSDs and Windows

These platforms are not tested, and no CI job covers them. The code has no
Linux-only dependency in its render path: it is Rust, plus two C libraries
that `cargo` builds, libdeflate for compression and mimalloc, the
command-line tool's allocator. It is expected to build anywhere Rust and a C
compiler are available, but expected is not tested. Please report what you
find.

## FreeBSD, OpenBSD, NetBSD

- **Build:** as on [Linux](linux.md), with the system C compiler.
- **Fonts:** searched in `/usr/local/share/fonts`, where the ports and
  packages install them, as well as `~/.fonts`, `~/.local/share/fonts`, and
  any directory named by `RROCKET_FONT_DIR`.
- **Writing output to RAM:** there is no `/dev/shm`, so a bare prefix writes
  to the current directory. rrocket cannot read the system's free memory
  here. Without it, pages could not be moved to disk before memory runs out,
  so `--ram` and `--ram-path` are refused.
- **GPU:** none tested. The Vulkan feature may work wherever a Vulkan 1.3
  driver exists.

## Windows

- **Build:** with the MSVC toolchain from `rustup` and the Visual Studio
  Build Tools.
- **Fonts:** searched in `C:/Windows/Fonts`. Fonts installed for one user
  only (`%LOCALAPPDATA%\Microsoft\Windows\Fonts`) are not searched yet; name
  that directory with `RROCKET_FONT_DIR`.
- **Writing output to RAM:** as on the BSDs, a bare prefix writes to the
  current directory, and `--ram` and `--ram-path` are refused.
- **GPU:** CUDA and Vulkan are untested on Windows.
