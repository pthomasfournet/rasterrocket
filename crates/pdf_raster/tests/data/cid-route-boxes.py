#!/usr/bin/env python3
"""Generate cid-route-boxes.ttf for substitute_fonts.rs.

A face that declares Japanese (OS/2 code page bit 17) and maps U+3042, the
character a face must map to be chosen for Japanese text.  It draws A, B
and U+3042 as solid boxes and has 266 glyphs, of which glyphs 264 and 265
(the Adobe-Japan1 CIDs of A and B) are empty: Latin text in a Japanese CID
font drawn through CID -> Unicode inks the boxes, and CIDs taken as glyph
indices ink nothing.

Usage: python3 -I cid-route-boxes.py [OUTPUT]  (default: beside this file).
The output is byte-reproducible for a given fontTools version (4.61.1 made
the committed file): its timestamps are fixed at 0.
"""

import sys
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

BOX_X0 = 50


def box(x0, y0, x1, y1):
    pen = TTGlyphPen(None)
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()
    return pen.glyph()


def empty():
    return TTGlyphPen(None).glyph()


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".ttf")
    order = [".notdef", "A", "B", "uni3042"] + [f"g{i}" for i in range(4, 266)]
    glyphs = {name: empty() for name in order}
    glyphs["A"] = box(BOX_X0, 0, 450, 700)
    glyphs["B"] = box(BOX_X0, 0, 450, 700)
    glyphs["uni3042"] = box(BOX_X0, 0, 950, 800)
    inked = {"A", "B", "uni3042"}

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x41: "A", 0x42: "B", 0x3042: "uni3042"})
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(
        {
            name: (1000 if name == "uni3042" else 500, BOX_X0 if name in inked else 0)
            for name in order
        }
    )
    fb.setupHorizontalHeader(ascent=880, descent=-120)
    fb.setupNameTable(
        {"familyName": "CID Route Boxes", "styleName": "Regular", "psName": "CIDRouteBoxes"}
    )
    fb.setupOS2(
        sTypoAscender=880,
        sTypoDescender=-120,
        usWinAscent=880,
        usWinDescent=120,
        ulCodePageRange1=(1 << 0) | (1 << 17),
        usWeightClass=400,
    )
    fb.setupPost()
    fb.updateHead(created=0, modified=0)
    fb.font.recalcTimestamp = False
    fb.save(str(out))


if __name__ == "__main__":
    main()
