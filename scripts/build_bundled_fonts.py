#!/usr/bin/env python3
"""Regenerate the bundled fallback fonts in crates/pdf_interp/fonts/noto/.

The renderer draws a character that neither a font's substitute nor any
installed face covers from these faces, compiled into the binary behind the
`bundled-fonts` Cargo feature.  Each is a Noto face (SIL Open Font License
1.1) subset to the code ranges listed below, without hinting, and stored
zlib-compressed at level 9 as `<name>.ttf.z`.  The Hebrew, Arabic,
Devanagari and Thai faces keep their layout tables, which the renderer
shapes with; the others keep none.

Usage:
    python3 scripts/build_bundled_fonts.py [--src DIR]

`--src` is the directory holding the source `.ttf` files (default: the
Debian/Ubuntu `fonts-noto-core` location).  Requires fontTools.
"""

import argparse
import io
import pathlib
import sys
import zlib
from collections.abc import Iterable

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "crates" / "pdf_interp" / "fonts" / "noto"


def ranges(*spans: tuple[int, int]) -> list[int]:
    """Return the code points of the inclusive `(lo, hi)` spans, in order."""
    return [c for lo, hi in spans for c in range(lo, hi + 1)]


GREEK_CYRILLIC = ranges(
    # Greek and Coptic, Greek Extended
    (0x0370, 0x03FF),
    (0x1F00, 0x1FFF),
    # Cyrillic, Supplement, Extended-C, -A, -B
    (0x0400, 0x052F),
    (0x1C80, 0x1C8F),
    (0x2DE0, 0x2DFF),
    (0xA640, 0xA69F),
)
HEBREW = ranges((0x0590, 0x05FF), (0xFB1D, 0xFB4F))
ARABIC = ranges(
    # Arabic, Supplement, Extended-A
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    # presentation forms A and B
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
DEVANAGARI = ranges((0x0900, 0x097F), (0xA8E0, 0xA8FF))
THAI = ranges((0x0E00, 0x0E7F))
SYMBOLS = ranges(
    # letterlike symbols, number forms
    (0x2100, 0x218F),
    # arrows, mathematical operators, miscellaneous technical
    (0x2190, 0x21FF),
    (0x2200, 0x22FF),
    (0x2300, 0x23FF),
    # geometric shapes, miscellaneous symbols, dingbats
    (0x25A0, 0x25FF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    # supplemental mathematical blocks
    (0x27C0, 0x27FF),
    (0x2900, 0x297F),
    (0x2980, 0x29FF),
    (0x2A00, 0x2AFF),
    # miscellaneous symbols and arrows
    (0x2B00, 0x2BFF),
)

# (output name, source file, code points, whether the face is shaped)
FACES = [
    ("NotoSans-GreekCyrillic", "NotoSans-Regular.ttf", GREEK_CYRILLIC, False),
    ("NotoSerif-GreekCyrillic", "NotoSerif-Regular.ttf", GREEK_CYRILLIC, False),
    ("NotoSansHebrew", "NotoSansHebrew-Regular.ttf", HEBREW, True),
    ("NotoSerifHebrew", "NotoSerifHebrew-Regular.ttf", HEBREW, True),
    ("NotoSansArabic", "NotoSansArabic-Regular.ttf", ARABIC, True),
    ("NotoNaskhArabic", "NotoNaskhArabic-Regular.ttf", ARABIC, True),
    ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf", DEVANAGARI, True),
    ("NotoSerifDevanagari", "NotoSerifDevanagari-Regular.ttf", DEVANAGARI, True),
    ("NotoSansThai", "NotoSansThai-Regular.ttf", THAI, True),
    ("NotoSerifThai", "NotoSerifThai-Regular.ttf", THAI, True),
    ("NotoSansMath", "NotoSansMath-Regular.ttf", SYMBOLS, False),
    ("NotoSansSymbols", "NotoSansSymbols-Regular.ttf", SYMBOLS, False),
    ("NotoSansSymbols2", "NotoSansSymbols2-Regular.ttf", SYMBOLS, False),
]


def build(src: pathlib.Path, codes: Iterable[int], shaped: bool) -> bytes:
    """Return the face `src` subset to those of `codes` it maps, with all
    its layout features and the glyphs they reach when `shaped`, else with
    none.

    The source's `head.modified` is kept, so the output depends on the
    source and this script only.
    """
    font = TTFont(src, recalcTimestamp=False)
    cmap = font.getBestCmap()
    opts = subset.Options()
    opts.layout_features = ["*"] if shaped else []
    opts.hinting = False
    opts.notdef_outline = True
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
    opts.drop_tables += ["DSIG"]
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=[c for c in codes if c in cmap])
    sub.subset(font)
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def main() -> None:
    """Rebuild every face of `FACES` into `OUT`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="/usr/share/fonts/truetype/noto", type=pathlib.Path)
    args = parser.parse_args()
    missing = [file for _, file, _, _ in FACES if not (args.src / file).is_file()]
    if missing:
        sys.exit(f"missing under {args.src}: {', '.join(missing)}")
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, file, codes, shaped in FACES:
        data = build(args.src / file, codes, shaped)
        packed = zlib.compress(data, 9)
        (OUT / f"{name}.ttf.z").write_bytes(packed)
        total += len(packed)
        print(f"{name:26} {len(data):8} bytes -> {len(packed):8} zlib")
    print(f"total {total} bytes")


if __name__ == "__main__":
    main()
