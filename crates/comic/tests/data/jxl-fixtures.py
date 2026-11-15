#!/usr/bin/env python3
"""Generate the JPEG XL fixtures of the comic crate's jxl decoder tests.

gray-16x8.jxl: 16 x 8 lossless gray, sample (16x + y) mod 256.
rgb-8x4.jxl:   8 x 4 lossless RGB, pixel (30x, 60y, 255 - 30x).
lossy-32x16.jxl: 32 x 16 lossy (XYB) RGB, pixel (8x, 16y, 128).
gray16-16x8.jxl: 16 x 8 lossless 16-bit gray, sample (4099x + 8191y) mod 65536.

Needs Pillow and pillow-jxl-plugin (pip install pillow pillow-jxl-plugin), and
ffmpeg built with libjxl for the 16-bit fixture.
Usage: python3 -I jxl-fixtures.py [OUTPUT_DIR]  (default: beside this file).
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pillow_jxl  # noqa: F401  (registers the JPEG XL plugin)
from PIL import Image


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    gray = Image.new("L", (16, 8))
    gray.putdata([(x * 16 + y) % 256 for y in range(8) for x in range(16)])
    gray.save(out / "gray-16x8.jxl", lossless=True, effort=7)
    rgb = Image.new("RGB", (8, 4))
    rgb.putdata([(x * 30, y * 60, 255 - x * 30) for y in range(4) for x in range(8)])
    rgb.save(out / "rgb-8x4.jxl", lossless=True, effort=7)
    lossy = Image.new("RGB", (32, 16))
    lossy.putdata([(x * 8, y * 16, 128) for y in range(16) for x in range(32)])
    lossy.save(out / "lossy-32x16.jxl", quality=95, effort=7)
    deep = b"".join(
        struct.pack("<H", (x * 4099 + y * 8191) % 65536) for y in range(8) for x in range(16)
    )
    with tempfile.NamedTemporaryFile(suffix=".raw") as raw:
        raw.write(deep)
        raw.flush()
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray16le",
             "-s", "16x8", "-i", raw.name, "-c:v", "libjxl", "-distance", "0",
             "-effort", "7", "-f", "image2", str(out / "gray16-16x8.jxl")],
            check=True,
        )


if __name__ == "__main__":
    main()
