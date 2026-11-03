#!/usr/bin/env python3
"""Regenerate the bundled fallback fonts in crates/pdf_interp/fonts/noto/.

The renderer draws a character that neither a font's substitute nor any
installed face covers from these faces, compiled into the binary behind the
`bundled-fonts` Cargo feature.  Each is a Noto face (SIL Open Font License
1.1) subset to the code ranges listed below, without hinting, and stored
zlib-compressed at level 9 as `<name>.ttf.z`.  The Hebrew, Arabic,
Devanagari, Thai, Bengali and Tamil faces keep their layout tables, which
the renderer shapes with; the others keep none.

Usage:
    python3 scripts/build_bundled_fonts.py [--src DIR]

`--src` is the directory holding the source `.ttf` files (default: the
Debian/Ubuntu `fonts-noto-core` location).  Requires fontTools.
"""

import argparse
import io
import os
import pathlib
import sys
import zlib

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
# The shaper's dotted circle, U+25CC, stands for the missing base of a
# broken cluster in the Indic faces.
DEVANAGARI = ranges((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x25CC, 0x25CC))
THAI = ranges((0x0E00, 0x0E7F))
ARMENIAN = ranges((0x0530, 0x058F), (0xFB13, 0xFB17))
# Georgian, Georgian Extended, Georgian Supplement
GEORGIAN = ranges((0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F))
# With the danda and double danda, U+0964 and U+0965, which end sentences
# in Bengali and Tamil text.
BENGALI = ranges((0x0964, 0x0965), (0x0980, 0x09FF), (0x25CC, 0x25CC))
TAMIL = ranges((0x0964, 0x0965), (0x0B80, 0x0BFF), (0x25CC, 0x25CC))
# Ethiopic, Supplement, Extended, Extended-A
ETHIOPIC = ranges((0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F))
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

# (output name, source file, code points, whether the face is shaped, least
# number of the code points the source must map)
FACES = [
    ("NotoSans-GreekCyrillic", "NotoSans-Regular.ttf", GREEK_CYRILLIC, False, 750),
    ("NotoSerif-GreekCyrillic", "NotoSerif-Regular.ttf", GREEK_CYRILLIC, False, 750),
    ("NotoSansHebrew", "NotoSansHebrew-Regular.ttf", HEBREW, True, 120),
    ("NotoSerifHebrew", "NotoSerifHebrew-Regular.ttf", HEBREW, True, 120),
    ("NotoSansArabic", "NotoSansArabic-Regular.ttf", ARABIC, True, 1000),
    ("NotoNaskhArabic", "NotoNaskhArabic-Regular.ttf", ARABIC, True, 1000),
    ("NotoSansDevanagari", "NotoSansDevanagari-Regular.ttf", DEVANAGARI, True, 150),
    ("NotoSerifDevanagari", "NotoSerifDevanagari-Regular.ttf", DEVANAGARI, True, 150),
    ("NotoSansThai", "NotoSansThai-Regular.ttf", THAI, True, 80),
    ("NotoSerifThai", "NotoSerifThai-Regular.ttf", THAI, True, 80),
    ("NotoSansBengali", "NotoSansBengali-Regular.ttf", BENGALI, True, 90),
    ("NotoSerifBengali", "NotoSerifBengali-Regular.ttf", BENGALI, True, 90),
    ("NotoSansTamil", "NotoSansTamil-Regular.ttf", TAMIL, True, 70),
    ("NotoSerifTamil", "NotoSerifTamil-Regular.ttf", TAMIL, True, 70),
    ("NotoSansArmenian", "NotoSansArmenian-Regular.ttf", ARMENIAN, False, 90),
    ("NotoSerifArmenian", "NotoSerifArmenian-Regular.ttf", ARMENIAN, False, 90),
    ("NotoSansGeorgian", "NotoSansGeorgian-Regular.ttf", GEORGIAN, False, 160),
    ("NotoSerifGeorgian", "NotoSerifGeorgian-Regular.ttf", GEORGIAN, False, 160),
    ("NotoSansEthiopic", "NotoSansEthiopic-Regular.ttf", ETHIOPIC, False, 450),
    ("NotoSerifEthiopic", "NotoSerifEthiopic-Regular.ttf", ETHIOPIC, False, 450),
    ("NotoSansMath", "NotoSansMath-Regular.ttf", SYMBOLS, False, 1000),
    ("NotoSansSymbols", "NotoSansSymbols-Regular.ttf", SYMBOLS, False, 300),
    ("NotoSansSymbols2", "NotoSansSymbols2-Regular.ttf", SYMBOLS, False, 600),
]


def build(src: pathlib.Path, codes: list[int], shaped: bool, least: int) -> bytes:
    """Return the face `src` subset to those of `codes` it maps, with all
    its layout features and the glyphs they reach when `shaped`, else with
    none.  Exit with an error when `src` has no Unicode `cmap` or maps
    fewer than `least` of `codes`: a source that is not the expected face.

    The source's `head.modified` is kept, so the output depends on the
    source and this script only.
    """
    font = TTFont(src, recalcTimestamp=False)
    cmap = font.getBestCmap()
    if cmap is None:
        sys.exit(f"{src}: no Unicode cmap")
    kept = [c for c in codes if c in cmap]
    if len(kept) < least:
        sys.exit(
            f"{src}: maps {len(kept)}/{len(codes)} of the requested code points,"
            f" expected at least {least}"
        )
    opts = subset.Options()
    opts.layout_features = ["*"] if shaped else []
    opts.hinting = False
    opts.notdef_outline = True
    opts.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]
    opts.drop_tables += ["DSIG"]
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=kept)
    sub.subset(font)
    buf = io.BytesIO()
    font.save(buf)
    print(f"{src.name:34} {len(kept):5}/{len(codes):5} code points")
    return buf.getvalue()


def main() -> None:
    """Rebuild every face of `FACES` into `OUT`.

    Every face is built before any is written; each output is written to a
    temporary file and renamed over the old one.  `.ttf.z` files in `OUT`
    that `FACES` does not name are reported, and make the exit status
    non-zero.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default="/usr/share/fonts/truetype/noto", type=pathlib.Path)
    args = parser.parse_args()
    missing = [file for _, file, _, _, _ in FACES if not (args.src / file).is_file()]
    if missing:
        sys.exit(f"missing under {args.src}: {', '.join(missing)}")
    built = [
        (name, build(args.src / file, codes, shaped, least))
        for name, file, codes, shaped, least in FACES
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, data in built:
        packed = zlib.compress(data, 9)
        out = OUT / f"{name}.ttf.z"
        tmp = OUT / f"{name}.ttf.z.tmp"
        tmp.write_bytes(packed)
        os.replace(tmp, out)
        total += len(packed)
        print(f"{name:26} {len(data):8} bytes -> {len(packed):8} zlib")
    print(f"total {total} bytes")
    named = {f"{name}.ttf.z" for name, _ in built}
    stale = sorted(p.name for p in OUT.glob("*.ttf.z") if p.name not in named)
    if stale:
        sys.exit(f"not produced by this script, remove or add to FACES: {', '.join(stale)}")


if __name__ == "__main__":
    main()
