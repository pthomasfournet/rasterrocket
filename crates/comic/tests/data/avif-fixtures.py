#!/usr/bin/env python3
"""Generate the AVIF fixtures of the comic crate's avif decoder tests.

Each picture is encoded to AV1 by ffmpeg (libaom-av1); the HEIF container
around it is written here, so every box layout below is deliberate.

rgb-8x4.avif:     8 x 4 lossless 4:4:4, identity matrix, full range;
                  pixel (30x, 60y, 255 - 30x).  iloc version 0.
gray-16x8.avif:   16 x 8 lossless monochrome, full range; sample
                  (16x + y) mod 256.  iloc version 1, data in idat.
lossy-64x48.avif: 64 x 48 lossy 4:2:0, 8 bits, limited range, matrix
                  unspecified (encoded as BT.601); pixel
                  (4x, 5y, 255 - 2x - 2y).  iloc version 2, two extents on a
                  base offset.
deep-32x16.avif:  32 x 16 lossy 4:2:0, 10 bits, BT.709 full range; the
                  sequence header says "unspecified, limited range" and the
                  colr box alone carries the truth.  Pixel (8x, 16y, 128).
grid-120x100.avif: a 2 x 2 grid of 64 x 64 lossless identity tiles cropped
                  to 120 x 100; pixel (2x, 2y, x + y).  The tiles are hidden
                  items 2 to 5 and the grid is item 1.
turn-8x4.avif:    rgb-8x4 with clap (the middle 6 x 2), irot (one quarter
                  turn anticlockwise) and imir (axis 1) on the item.

Needs ffmpeg with libaom-av1 on PATH.
Usage: python3 -I avif-fixtures.py [OUTPUT_DIR]  (default: beside this file).
"""

import struct
import subprocess
import sys
from pathlib import Path


def encode(pixels, size, pix_fmt, *args):
    """Return the OBUs of one still picture, encoded from rgb24 `pixels`."""
    w, h = size
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
        "-frames:v", "1", "-c:v", "libaom-av1", "-still-picture", "1",
        "-cpu-used", "4", "-pix_fmt", pix_fmt, *args, "-f", "obu", "-",
    ]  # fmt: skip
    obu = subprocess.run(cmd, input=bytes(pixels), capture_output=True, check=True).stdout
    # A stored item carries no temporal delimiter.
    return obu.removeprefix(b"\x12\x00")


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def full(kind, version, flags, payload):
    return box(kind, struct.pack(">I", version << 24 | flags) + payload)


def ispe(w, h):
    return full(b"ispe", 0, 0, struct.pack(">II", w, h))


def av1c(profile, deep, mono, sub_x, sub_y):
    config = deep << 6 | mono << 4 | sub_x << 3 | sub_y << 2
    return box(b"av1C", bytes([0x81, profile << 5, config, 0]))


def nclx(primaries, transfer, matrix, full_range):
    return box(b"colr", b"nclx" + struct.pack(">HHHB", primaries, transfer, matrix, full_range << 7))


def clap(w, h, dx, dy):
    """Clean aperture of `w` x `h`, its centre (`dx`, `dy`) / 2 from the image's."""
    return box(b"clap", struct.pack(">IIIIiIiI", w, 1, h, 1, dx, 2, dy, 2))


class Item:
    def __init__(self, ident, kind, data, props, *, hidden=False, method=0, pieces=1):
        self.ident, self.kind, self.data, self.props = ident, kind, data, props
        self.hidden, self.method, self.pieces = hidden, method, pieces

    def extents(self):
        """The item's data cut into `pieces` runs of (offset in data, length)."""
        step = -(-len(self.data) // self.pieces)
        return [(at, min(step, len(self.data) - at)) for at in range(0, len(self.data), step)]


def iloc(items, version, starts, base):
    """An iloc of 4-byte offsets and lengths; `base` is each item's base offset."""
    out = bytes([0x44, 0x40 if base else 0x00])
    out += struct.pack(">I" if version == 2 else ">H", len(items))
    for item in items:
        out += struct.pack(">I" if version == 2 else ">H", item.ident)
        if version:
            out += struct.pack(">H", item.method)
        out += struct.pack(">H", 0)
        if base:
            out += struct.pack(">I", base)
        extents = item.extents()
        out += struct.pack(">H", len(extents))
        for at, length in extents:
            out += struct.pack(">II", starts[item.ident] + at - base, length)
    return full(b"iloc", version, 0, out)


def container(items, *, brands=(b"avif", b"mif1", b"miaf"), iloc_version=0, base=0, refs=()):
    """A file of ftyp, meta and (unless empty) mdat; item 1 is primary.

    Items of construction method 1 lie in idat, the others in mdat.
    `refs` lists (type, from, [to]) item references.
    """
    ftyp = box(b"ftyp", brands[0] + bytes(4) + b"".join(brands))
    hdlr = full(b"hdlr", 0, 0, bytes(4) + b"pict" + bytes(12) + b"\0")
    pitm = full(b"pitm", 0, 0, struct.pack(">H", 1))
    infes = b"".join(
        full(b"infe", 2, int(i.hidden), struct.pack(">HH4s", i.ident, 0, i.kind) + b"\0")
        for i in items
    )
    iinf = full(b"iinf", 0, 0, struct.pack(">H", len(items)) + infes)
    props, ipma = [], struct.pack(">I", len(items))
    for item in items:
        ipma += struct.pack(">HB", item.ident, len(item.props))
        for prop, essential in item.props:
            if prop not in props:
                props.append(prop)
            ipma += bytes([essential << 7 | props.index(prop) + 1])
    iprp = box(b"iprp", box(b"ipco", b"".join(props)) + full(b"ipma", 0, 0, ipma))
    iref = b""
    if refs:
        iref = full(
            b"iref", 0, 0,
            b"".join(
                box(kind, struct.pack(f">HH{len(to)}H", source, len(to), *to))
                for kind, source, to in refs
            ),
        )  # fmt: skip
    idat_data = b"".join(i.data for i in items if i.method == 1)
    idat = box(b"idat", idat_data) if idat_data else b""
    mdat_data = b"".join(i.data for i in items if i.method == 0)
    mdat = box(b"mdat", mdat_data) if mdat_data else b""

    def meta(starts):
        return full(
            b"meta", 0, 0,
            hdlr + pitm + iloc(items, iloc_version, starts, base) + iinf + iprp + iref + idat,
        )  # fmt: skip

    # File offsets of the mdat items follow the meta box, whose size does
    # not depend on the offsets written into it.
    starts, at_idat = {}, 0
    at_mdat = len(ftyp) + len(meta({i.ident: base for i in items})) + 8
    for item in items:
        if item.method == 1:
            starts[item.ident], at_idat = at_idat + base, at_idat + len(item.data)
        else:
            starts[item.ident], at_mdat = at_mdat, at_mdat + len(item.data)
    return ftyp + meta(starts) + mdat


def picture(w, h, pixel):
    return [v for y in range(h) for x in range(w) for v in pixel(x, y)]


LOSSLESS = ("-aom-params", "lossless=1")
IDENTITY = [(av1c(1, 0, 0, 0, 0), True), (nclx(1, 13, 0, 1), False)]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    rgb = encode(picture(8, 4, lambda x, y: (x * 30, y * 60, 255 - x * 30)), (8, 4), "gbrp", *LOSSLESS)
    props = [(ispe(8, 4), False), *IDENTITY]
    (out / "rgb-8x4.avif").write_bytes(container([Item(1, b"av01", rgb, props)]))

    turned = [*props, (clap(6, 2, 0, 0), True), (box(b"irot", b"\x01"), True)]
    turned.append((box(b"imir", b"\x01"), True))
    (out / "turn-8x4.avif").write_bytes(container([Item(1, b"av01", rgb, turned)]))

    gray = encode(
        picture(16, 8, lambda x, y: ((x * 16 + y) % 256,) * 3), (16, 8), "gray",
        "-vf", "scale=out_range=pc", "-color_range", "pc", *LOSSLESS,
    )  # fmt: skip
    props = [(ispe(16, 8), False), (av1c(0, 0, 1, 1, 1), True)]
    item = Item(1, b"av01", gray, props, method=1)
    (out / "gray-16x8.avif").write_bytes(container([item], iloc_version=1))

    lossy = encode(
        picture(64, 48, lambda x, y: (x * 4, y * 5, 255 - 2 * x - 2 * y)), (64, 48), "yuv420p",
        "-crf", "12",
    )  # fmt: skip
    props = [(ispe(64, 48), False), (av1c(0, 0, 0, 1, 1), True)]
    item = Item(1, b"av01", lossy, props, pieces=2)
    (out / "lossy-64x48.avif").write_bytes(container([item], iloc_version=2, base=40))

    deep = encode(
        picture(32, 16, lambda x, y: (x * 8, y * 16, 128)), (32, 16), "yuv420p10le",
        "-vf", "scale=out_color_matrix=bt709:out_range=pc", "-crf", "8",
        "-bsf:v", "av1_metadata=color_primaries=2:transfer_characteristics=2"
        ":matrix_coefficients=2:color_range=tv",
    )  # fmt: skip
    props = [(ispe(32, 16), False), (av1c(0, 1, 0, 1, 1), True), (nclx(1, 13, 1, 1), False)]
    (out / "deep-32x16.avif").write_bytes(container([Item(1, b"av01", deep, props)]))

    tiles = []
    for index in range(4):
        ox, oy = index % 2 * 64, index // 2 * 64
        pixels = picture(64, 64, lambda x, y: (2 * (x + ox), 2 * (y + oy), x + ox + y + oy))  # noqa: B023
        tile = encode(pixels, (64, 64), "gbrp", *LOSSLESS)
        tiles.append(Item(2 + index, b"av01", tile, [(ispe(64, 64), False), *IDENTITY], hidden=True))
    grid = Item(1, b"grid", struct.pack(">BBBBHH", 0, 0, 1, 1, 120, 100), [(ispe(120, 100), False)])
    (out / "grid-120x100.avif").write_bytes(
        container([grid, *tiles], refs=[(b"dimg", 1, [2, 3, 4, 5])])
    )


if __name__ == "__main__":
    main()
