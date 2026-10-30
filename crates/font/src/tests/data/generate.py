#!/usr/bin/env python3
"""Build the synthetic font fixtures in this directory.

    python3 generate.py            # rewrite the fixtures next to this file

hint-assembled-*.ttf
    TrueType fonts (units per em 1000) whose composite glyph is shaped only
    by its instructions.  GID 1 `square` is a square covering x 100..400,
    y 0..300 and is mapped from U+0042 `B`.  GID 2 `pair` (U+0041 `A`) is
    `square` twice at offset (0, 0); its instructions shift the second
    copy's four points right by half an em (MPPEM * 2048 MUL, i.e.
    ppem / 2 pixels, then SHPIX), so the hinted glyph is two squares,
    x 100..400 and 600..900, while the unhinted glyph is one square drawn
    twice.

    - `-by-name`: family name `PMingLiU`, on the list of fonts whose glyphs
      depend on hinting; no `fpgm` / `prep`.
    - `-by-tables`: family name `Renamed`, no `cvt `, and `fpgm` / `prep`
      of the lengths and checksums recorded for NEC `fadpop7.ttf`
      (fpgm 0xE5 bytes, checksum 0x40C92555; prep 0x117C bytes, checksum
      0xA39B58E3).  Both programs are valid bytecode: NPUSHB blocks whose
      data bytes carry the checksum adjustment, each followed by as many
      POPs.
    - `-plain`: family name `Plain`, no `fpgm` / `prep`: not recognised,
      so it renders unhinted.

symbolic-mac-roman-named.ttf
    The shape of a symbolic subset: GID 1 `therefore` (U+2234 through the
    Adobe Glyph List) is a square covering x 100..400, y 0..300 and is
    mapped from code 0x21 by the only `cmap` subtable, (1,0) Macintosh
    Roman in format 6.  The version 2 `post` table names both glyphs;
    `.notdef` has no contours.

vertical-forms.ttf
    A TrueType font with a `GSUB` `vert` feature: GID 1 `square` (U+0041,
    x 100..400, y 0..300) substitutes to GID 2 `tall` (x 0..300,
    y 100..400); GID 3 `other` (U+0042) has no vertical form.

cid-keyed-type1.cid
    A CID-keyed Type 1 font (CIDFontType 0, binary StartData section)
    assembled by hand: CIDCount 5, FDBytes 1, GDBytes 4, two font dicts.

    - CID 0: `.notdef`, width 1000, no contours (FD 0).
    - CID 1: square x 100..400, y 0..300 (FD 0).
    - CID 2: triangle (50,0) (450,0) (250,400) drawn by subroutine 4 of
      FD 0 (FD 0 has five subroutines; 0..3 are bare `return`s).
    - CID 3: diamond (500,0) (700,200) (500,400) (300,200) in FD 1, whose
      private dict sets lenIV -1 (charstring stored unencrypted).
    - CID 4: no glyph (FD byte 0xFF).

    FD 0 uses the default lenIV 4, so its charstrings and subroutines are
    encrypted with key 4330 after four zero bytes.  The top-level dict
    carries an identity FontMatrix, which applies to no glyph: only the
    font dicts' matrices do.
"""

import os
import struct

from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent
from fontTools.ttLib.tables.ttProgram import Program

HERE = os.path.dirname(os.path.abspath(__file__))

# ── TrueType ────────────────────────────────────────────────────────────────

# PUSHB[4] 4 5 6 7 4; SVTCA[x]; SLOOP; MPPEM; PUSHW[0] 2048; MUL; SHPIX
PAIR_INSTRUCTIONS = bytes(
    [0xB4, 4, 5, 6, 7, 4, 0x01, 0x17, 0x4B, 0xB8, 0x08, 0x00, 0x63, 0x38]
)


def checksum(data):
    data = data + b"\0" * (-len(data) % 4)
    return sum(struct.unpack(f">{len(data) // 4}I", data)) & 0xFFFFFFFF


def forged_program(length, target):
    """Bytecode of `length` bytes summing to `target`: an optional leading
    SVTCA[y] for odd lengths, then NPUSHB n <n bytes> + n POPs blocks."""
    code = bytearray()
    if length % 2:
        code.append(0x00)
    remaining = length - len(code)
    blocks = []
    while remaining:
        n = min(255, (remaining - 2) // 2)
        blocks.append(n)
        remaining -= 2 + 2 * n
    data_start = None
    for n in blocks:
        code += bytes([0x40, n])
        if data_start is None:
            data_start = len(code)
        code += bytes(n)
        code += bytes([0x21]) * n
    assert len(code) == length
    # Adjust the first word-aligned four data bytes of the first block.
    at = data_start + (-data_start % 4)
    assert at + 4 <= data_start + blocks[0]
    word = (target - checksum(bytes(code))) & 0xFFFFFFFF
    code[at : at + 4] = struct.pack(">I", word)
    assert checksum(bytes(code)) == target
    return bytes(code)


def square_glyph():
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((100, 300))
    pen.lineTo((400, 300))
    pen.lineTo((400, 0))
    pen.closePath()
    return pen.glyph()


def pair_glyph():
    glyph = Glyph()
    glyph.numberOfContours = -1
    glyph.components = []
    for _ in range(2):
        component = GlyphComponent()
        component.glyphName = "square"
        component.x, component.y = 0, 0
        component.flags = 0x0004  # ROUND_XY_TO_GRID
        glyph.components.append(component)
    glyph.program = Program()
    glyph.program.fromBytecode(PAIR_INSTRUCTIONS)
    return glyph


def build_truetype(family, fpgm=None, prep=None):
    fb = FontBuilder(1000, isTTF=True)
    order = [".notdef", "square", "pair"]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x41: "pair", 0x42: "square"})
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({".notdef": empty, "square": square_glyph(), "pair": pair_glyph()})
    fb.setupHorizontalMetrics({".notdef": (1000, 0), "square": (1000, 100), "pair": (1000, 100)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    font = fb.font
    font["head"].flags |= 0x0008  # integer ppem
    for tag, code in (("fpgm", fpgm), ("prep", prep)):
        if code is not None:
            table = newTable(tag)
            table.program = Program()
            table.program.fromBytecode(code)
            font[tag] = table
    maxp = font["maxp"]
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 300
    maxp.maxSizeOfInstructions = len(PAIR_INSTRUCTIONS)
    maxp.maxComponentElements = 2
    maxp.maxComponentDepth = 1
    return font


def build_symbolic_mac_roman_named():
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "therefore"])
    fb.setupCharacterMap({})
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({".notdef": empty, "therefore": square_glyph()})
    fb.setupHorizontalMetrics({".notdef": (1000, 0), "therefore": (1000, 100)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "SymbolicMac", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost(keepGlyphNames=True)
    mac = CmapSubtable.newSubtable(6)
    mac.platformID, mac.platEncID, mac.language = 1, 0, 0
    mac.cmap = {0x21: "therefore"}
    fb.font["cmap"].tables = [mac]
    return fb.font


def tall_glyph():
    pen = TTGlyphPen(None)
    pen.moveTo((0, 100))
    pen.lineTo((0, 400))
    pen.lineTo((300, 400))
    pen.lineTo((300, 100))
    pen.closePath()
    return pen.glyph()


def build_vertical_forms():
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "square", "tall", "other"])
    fb.setupCharacterMap({0x41: "square", 0x42: "other"})
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({".notdef": empty, "square": square_glyph(), "tall": tall_glyph(), "other": square_glyph()})
    fb.setupHorizontalMetrics({".notdef": (1000, 0), "square": (1000, 100), "tall": (1000, 0), "other": (1000, 100)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "VerticalForms", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()
    addOpenTypeFeaturesFromString(fb.font, "feature vert { sub square by tall; } vert;")
    return fb.font


def write_truetype(name, font):
    path = os.path.join(HERE, name)
    font.save(path)
    return path


# ── CID-keyed Type 1 ────────────────────────────────────────────────────────


def cs_num(v):
    if -107 <= v <= 107:
        return bytes([v + 139])
    if 108 <= v <= 1131:
        v -= 108
        return bytes([(v >> 8) + 247, v & 0xFF])
    if -1131 <= v <= -108:
        v = -v - 108
        return bytes([(v >> 8) + 251, v & 0xFF])
    return bytes([255]) + struct.pack(">i", v)


HSBW, RMOVETO, RLINETO, CLOSEPATH, ENDCHAR, CALLSUBR, RETURN = 13, 21, 5, 9, 14, 10, 11


def charstring(*items):
    out = bytearray()
    for item in items:
        if isinstance(item, tuple):  # operator
            out += bytes(item)
        else:
            out += cs_num(item)
    return bytes(out)


def op(code):
    return (code,)


def encrypt(plain, key=4330, len_iv=4):
    r = key
    out = bytearray()
    for p in bytes(len_iv) + plain:
        c = p ^ (r >> 8)
        out.append(c)
        r = ((c + r) * 52845 + 22719) & 0xFFFF
    return bytes(out)


def polygon(sbx, points):
    """hsbw, then a closed polygon through absolute `points`."""
    items = [sbx, 1000, op(HSBW)]
    x, y = sbx, 0
    first = True
    for px, py in points:
        items += [px - x, py - y, op(RMOVETO if first else RLINETO)]
        first = False
        x, y = px, py
    items += [op(CLOSEPATH), op(ENDCHAR)]
    return charstring(*items)


def build_cid_type1():
    notdef = charstring(0, 1000, op(HSBW), op(ENDCHAR))
    square = polygon(100, [(100, 0), (400, 0), (400, 300), (100, 300)])
    # The subroutine draws two sides from the current point (50, 0).
    triangle = charstring(
        50, 1000, op(HSBW), 0, 0, op(RMOVETO), 4, op(CALLSUBR), op(CLOSEPATH), op(ENDCHAR)
    )
    subr4 = charstring(400, 0, op(RLINETO), -200, 400, op(RLINETO), op(RETURN))
    stub = charstring(op(RETURN))
    diamond = polygon(500, [(500, 0), (700, 200), (500, 400), (300, 200)])

    fd0_subrs = [encrypt(s) for s in (stub, stub, stub, stub, subr4)]
    glyphs = [
        (0, encrypt(notdef)),
        (0, encrypt(square)),
        (0, encrypt(triangle)),
        (1, diamond),  # FD 1: lenIV -1, unencrypted
        (0xFF, b""),
    ]
    cid_count = len(glyphs)
    fd_bytes, gd_bytes, sd_bytes = 1, 4, 4
    cidmap_len = (cid_count + 1) * (fd_bytes + gd_bytes)
    subrmap_offset = cidmap_len
    subrmap_len = (len(fd0_subrs) + 1) * sd_bytes
    data_offset = subrmap_offset + subrmap_len

    subr_offsets, subr_data = [], bytearray()
    for s in fd0_subrs:
        subr_offsets.append(data_offset + len(subr_data))
        subr_data += s
    subr_offsets.append(data_offset + len(subr_data))
    glyph_base = data_offset + len(subr_data)

    cidmap, glyph_data = bytearray(), bytearray()
    for fd, data in glyphs:
        cidmap += bytes([fd]) + struct.pack(">I", glyph_base + len(glyph_data))
        glyph_data += data
    cidmap += bytes([0]) + struct.pack(">I", glyph_base + len(glyph_data))

    binary = bytes(cidmap) + b"".join(struct.pack(">I", o) for o in subr_offsets)
    binary += bytes(subr_data) + bytes(glyph_data)
    assert len(cidmap) == cidmap_len

    def font_dict(index, private):
        return (
            f"dup {index}\n"
            "%ADOBeginFontDict\n"
            "14 dict begin\n"
            f"/FontName /RRTest-CID-FD{index} def\n"
            "/FontType 1 def\n"
            "/FontMatrix [0.001 0 0 0.001 0 0] def\n"
            "/PaintType 0 def\n"
            "/Private 18 dict dup begin\n"
            "/MinFeature {16 16} def\n"
            "/BlueValues [-10 0 700 710] def\n"
            "/StdHW [50] def\n"
            "/StdVW [50] def\n"
            f"{private}"
            "end def\n"
            "currentdict end\n"
            "%ADOEndFontDict\n"
            "put\n"
        )

    header = (
        "%!PS-Adobe-3.0 Resource-CIDFont\n"
        "%%DocumentNeededResources: ProcSet (CIDInit)\n"
        "%%IncludeResource: ProcSet (CIDInit)\n"
        "%%BeginResource: CIDFont (RRTest-CID)\n"
        "%%Title: (RRTest-CID Adobe Identity 0)\n"
        "%%Version: 1.000\n"
        "/CIDInit /ProcSet findresource begin\n"
        "20 dict begin\n"
        "/CIDFontName /RRTest-CID def\n"
        "/CIDFontVersion 1.000 def\n"
        "/CIDFontType 0 def\n"
        "/CIDSystemInfo 3 dict dup begin\n"
        "  /Registry (Adobe) def\n"
        "  /Ordering (Identity) def\n"
        "  /Supplement 0 def\n"
        "end def\n"
        "/FontBBox [0 -100 1000 900] def\n"
        "/FontMatrix [1 0 0 1 0 0] def\n"
        "/FontInfo 2 dict dup begin\n"
        "  /FamilyName (RRTest) def\n"
        "  /FullName (RRTest CID) def\n"
        "end def\n"
        "/CIDMapOffset 0 def\n"
        f"/FDBytes {fd_bytes} def\n"
        f"/GDBytes {gd_bytes} def\n"
        f"/CIDCount {cid_count} def\n"
        "/FDArray 2 array\n"
        + font_dict(
            0,
            f"/SubrMapOffset {subrmap_offset} def\n/SDBytes {sd_bytes} def\n"
            f"/SubrCount {len(fd0_subrs)} def\n",
        )
        + font_dict(
            1,
            "/lenIV -1 def\n/SubrMapOffset 0 def\n/SDBytes 4 def\n/SubrCount 0 def\n",
        )
        + "def\n"
        f"%%BeginData: {len(binary) + 26} Binary Bytes\n"
        f"(Binary) {len(binary)} StartData "
    )
    trailer = (
        "\n%%EndData\n"
        "%%EndResource\n"
        "%%EOF\n"
    )
    return header.encode("ascii") + binary + trailer.encode("ascii")


def main():
    write_truetype("hint-assembled-by-name.ttf", build_truetype("PMingLiU"))
    write_truetype(
        "hint-assembled-by-tables.ttf",
        build_truetype(
            "Renamed",
            fpgm=forged_program(0xE5, 0x40C92555),
            prep=forged_program(0x117C, 0xA39B58E3),
        ),
    )
    write_truetype("hint-assembled-plain.ttf", build_truetype("Plain"))
    write_truetype("symbolic-mac-roman-named.ttf", build_symbolic_mac_roman_named())
    write_truetype("vertical-forms.ttf", build_vertical_forms())
    with open(os.path.join(HERE, "cid-keyed-type1.cid"), "wb") as f:
        f.write(build_cid_type1())


if __name__ == "__main__":
    main()
