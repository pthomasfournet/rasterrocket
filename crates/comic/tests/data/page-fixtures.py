#!/usr/bin/env python3
"""Generate the PNG, WebP and TIFF page images and the one-page archives of
the comic crate's tests (and the fuzz seed corpus).

rgb-16x8.png, rgb-16x8-lossless.webp, rgb-16x8-lossy.webp:
                  16 x 8 RGB, pixel (16x mod 256, 32y mod 256, 128).
gray-16x8-packbits.tif:
                  that image in gray, PackBits.
gray-16x8-two-pages.tif:
                  the gray image, then it turned a half turn; Deflate.
one-page.cbt:     1.jpg (an 8 x 8 JPEG of value 90) and ComicInfo.xml,
                  read back from one-page.cbz.
one-page.cbz, one-page.cb7, one-page-rar4.cbr, one-page-rar5.cbr:
                  the same entries, written by the writers the crate's
                  tests use (zip `ZipWriter`, Deflate; sevenz-rust2
                  `ArchiveWriter`, LZMA2; rars `Builder` in the RAR 4 and
                  RAR 5 formats), not by this script.

Needs Pillow.
Usage: python3 -I page-fixtures.py [OUTPUT_DIR]  (default: beside this file).
"""

import io
import pathlib
import sys
import tarfile
import zipfile

from PIL import Image


def main() -> None:
    """Write the fixtures into the output directory."""
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
    im = Image.new("RGB", (16, 8))
    im.putdata([((x * 16) % 256, (y * 32) % 256, 128) for y in range(8) for x in range(16)])
    im.save(out / "rgb-16x8.png", optimize=True)
    im.save(out / "rgb-16x8-lossless.webp", lossless=True)
    im.save(out / "rgb-16x8-lossy.webp", quality=60)
    gray = im.convert("L")
    gray.save(out / "gray-16x8-packbits.tif", compression="packbits")
    gray.save(
        out / "gray-16x8-two-pages.tif",
        save_all=True,
        append_images=[gray.rotate(180)],
        compression="tiff_deflate",
    )

    page = zipfile.ZipFile(out / "one-page.cbz").read("1.jpg")
    entries = [("1.jpg", page), ("ComicInfo.xml", b"<ComicInfo/>")]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    # The archive ends with its two zero blocks, not tarfile's record padding.
    data = buf.getvalue()
    end = -(-len(data.rstrip(b"\0")) // 512) * 512 + 1024
    (out / "one-page.cbt").write_bytes(data[:end])


if __name__ == "__main__":
    main()
