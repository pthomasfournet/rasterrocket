#!/usr/bin/env python3
# Generates cid-route-boxes.ttf (substitute_fonts.rs): python3 cid-route-boxes.py cid-route-boxes.ttf
# A Japanese-declaring face for the CID → Unicode test: A, B and U+3042 are
# solid boxes; 266 glyphs, of which 264 and 265 (the Adobe-Japan1 CIDs of
# A and B) are empty.
import sys
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

def box(x0, y0, x1, y1):
    pen = TTGlyphPen(None)
    pen.moveTo((x0, y0)); pen.lineTo((x0, y1)); pen.lineTo((x1, y1)); pen.lineTo((x1, y0)); pen.closePath()
    return pen.glyph()

def empty():
    return TTGlyphPen(None).glyph()

order = [".notdef", "A", "B", "uni3042"] + [f"g{i}" for i in range(4, 266)]
glyphs = {n: empty() for n in order}
glyphs["A"] = box(50, 0, 450, 700)
glyphs["B"] = box(50, 0, 450, 700)
glyphs["uni3042"] = box(50, 0, 950, 800)
fb = FontBuilder(1000, isTTF=True)
fb.setupGlyphOrder(order)
fb.setupCharacterMap({0x41: "A", 0x42: "B", 0x3042: "uni3042"})
fb.setupGlyf(glyphs)
fb.setupHorizontalMetrics({n: (1000 if n == "uni3042" else 500, 0) for n in order})
fb.setupHorizontalHeader(ascent=880, descent=-120)
fb.setupNameTable({"familyName": "Noto Sans CJK JP", "styleName": "Regular",
                   "psName": "NotoSansCJKjp-Regular"})
fb.setupOS2(sTypoAscender=880, sTypoDescender=-120, usWinAscent=880, usWinDescent=120,
            ulCodePageRange1=(1 << 0) | (1 << 17), usWeightClass=400)
fb.setupPost()
fb.save(sys.argv[1])
