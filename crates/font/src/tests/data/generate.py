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

no-cmap.ttf
    A TrueType subset with no `cmap` table and a format 3 `post` table (no
    glyph names): GID 1 is a square covering x 100..400, y 0..300, reachable
    only as its glyph index.

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

import pathlib
import struct

from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib.tables._g_l_y_f import Glyph, GlyphComponent
from fontTools.ttLib.tables.ttProgram import Program

HERE = pathlib.Path(__file__).resolve().parent

# ── TrueType ────────────────────────────────────────────────────────────────

# PUSHB[4] 4 5 6 7 4; SVTCA[x]; SLOOP; MPPEM; PUSHW[0] 2048; MUL; SHPIX
PAIR_INSTRUCTIONS = bytes([0xB4, 4, 5, 6, 7, 4, 0x01, 0x17, 0x4B, 0xB8, 0x08, 0x00, 0x63, 0x38])


def checksum(data: bytes) -> int:
    """Return the OpenType table checksum of `data` (big-endian u32 sum, zero padded)."""
    data = data + b"\0" * (-len(data) % 4)
    words: tuple[int, ...] = struct.unpack(f">{len(data) // 4}I", data)
    return sum(words) & 0xFFFFFFFF


def forged_program(length: int, target: int) -> bytes:
    """Return bytecode of `length` bytes whose table checksum is `target`.

    The program is an optional leading SVTCA[y] for odd lengths, then blocks
    of NPUSHB n <n bytes> followed by n POPs.
    """
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
    if data_start is None or len(code) != length:
        msg = f"cannot lay out a {length}-byte program"
        raise ValueError(msg)
    # Adjust the first word-aligned four data bytes of the first block.
    at = data_start + (-data_start % 4)
    if at + 4 > data_start + blocks[0]:
        msg = f"the first push block of a {length}-byte program holds no aligned word"
        raise ValueError(msg)
    word = (target - checksum(bytes(code))) & 0xFFFFFFFF
    code[at : at + 4] = struct.pack(">I", word)
    if checksum(bytes(code)) != target:
        msg = "checksum adjustment failed"
        raise ValueError(msg)
    return bytes(code)


def rect_glyph(x0: int, y0: int, x1: int, y1: int) -> Glyph:
    """Return a one-contour glyph covering x `x0..x1`, y `y0..y1`."""
    pen = TTGlyphPen(None)
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()
    return pen.glyph()


def square_glyph() -> Glyph:
    """Return the square shared by every TrueType fixture (x 100..400, y 0..300)."""
    return rect_glyph(100, 0, 400, 300)


def pair_glyph() -> Glyph:
    """Return the composite of `square` twice, shaped only by `PAIR_INSTRUCTIONS`."""
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


def base_font(
    family: str,
    glyphs: dict[str, tuple[Glyph, int, int]],
    cmap: dict[int, str],
    *,
    keep_glyph_names: bool = True,
) -> TTFont:
    """Return a 1000-unit TrueType font of `glyphs` (`name: (glyph, advance, lsb)`).

    `.notdef` comes first, empty, with advance 1000; `cmap` maps code points
    to glyph names.
    """
    fb = FontBuilder(1000, isTTF=True)
    all_glyphs = {".notdef": (TTGlyphPen(None).glyph(), 1000, 0), **glyphs}
    fb.setupGlyphOrder(list(all_glyphs))
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({name: glyph for name, (glyph, _, _) in all_glyphs.items()})
    fb.setupHorizontalMetrics({name: (adv, lsb) for name, (_, adv, lsb) in all_glyphs.items()})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost(keepGlyphNames=keep_glyph_names)
    return fb.font


def build_truetype(family: str, fpgm: bytes | None = None, prep: bytes | None = None) -> TTFont:
    """Return a hint-assembled fixture named `family`, with optional `fpgm` / `prep`."""
    font = base_font(
        family,
        {"square": (square_glyph(), 1000, 100), "pair": (pair_glyph(), 1000, 100)},
        {0x41: "pair", 0x42: "square"},
    )
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


def build_symbolic_mac_roman_named() -> TTFont:
    """Return the symbolic subset reachable only through a (1,0) format 6 cmap."""
    font = base_font("SymbolicMac", {"therefore": (square_glyph(), 1000, 100)}, {})
    mac = CmapSubtable.newSubtable(6)
    mac.platformID, mac.platEncID, mac.language = 1, 0, 0
    mac.cmap = {0x21: "therefore"}
    font["cmap"].tables = [mac]
    return font


def build_vertical_forms() -> TTFont:
    """Return the font whose `vert` feature substitutes `square` by `tall`."""
    font = base_font(
        "VerticalForms",
        {
            "square": (square_glyph(), 1000, 100),
            "tall": (rect_glyph(0, 100, 300, 400), 1000, 0),
            "other": (square_glyph(), 1000, 100),
        },
        {0x41: "square", 0x42: "other"},
    )
    addOpenTypeFeaturesFromString(font, "feature vert { sub square by tall; } vert;")
    return font


def build_no_cmap() -> TTFont:
    """Return the subset with no `cmap` and a format 3 `post` table."""
    font = base_font("NoCmap", {"square": (square_glyph(), 1000, 100)}, {}, keep_glyph_names=False)
    del font["cmap"]
    return font


def write_truetype(name: str, font: TTFont) -> None:
    """Save `font` as `name` next to this script."""
    font.save(HERE / name)


# ── CID-keyed Type 1 ────────────────────────────────────────────────────────


# Type 1 charstring number encoding (Type 1 Font Format, section 6.2): one
# byte for |v| <= 107, two bytes up to 1131, else 255 and a 32-bit integer.
ONE_BYTE_MAX = 107
TWO_BYTE_MAX = 1131


def cs_num(v: int) -> bytes:
    """Return the charstring encoding of the integer `v`."""
    if -ONE_BYTE_MAX <= v <= ONE_BYTE_MAX:
        return bytes([v + 139])
    if ONE_BYTE_MAX < v <= TWO_BYTE_MAX:
        v -= ONE_BYTE_MAX + 1
        return bytes([(v >> 8) + 247, v & 0xFF])
    if -TWO_BYTE_MAX <= v < -ONE_BYTE_MAX:
        v = -v - (ONE_BYTE_MAX + 1)
        return bytes([(v >> 8) + 251, v & 0xFF])
    return bytes([255]) + struct.pack(">i", v)


# One-byte charstring operators; operands are `int`, so the two never mix.
HSBW, RMOVETO, RLINETO, CLOSEPATH, ENDCHAR, CALLSUBR, RETURN = (
    b"\x0d",
    b"\x15",
    b"\x05",
    b"\x09",
    b"\x0e",
    b"\x0a",
    b"\x0b",
)


def charstring(*items: int | bytes) -> bytes:
    """Return the charstring of operands (integers) and operators (bytes), in order."""
    return b"".join(item if isinstance(item, bytes) else cs_num(item) for item in items)


def encrypt(plain: bytes, key: int = 4330, len_iv: int = 4) -> bytes:
    """Return `plain` encrypted with the Type 1 cipher after `len_iv` zero bytes."""
    r = key
    out = bytearray()
    for p in bytes(len_iv) + plain:
        c = p ^ (r >> 8)
        out.append(c)
        r = ((c + r) * 52845 + 22719) & 0xFFFF
    return bytes(out)


def polygon(sbx: int, points: list[tuple[int, int]]) -> bytes:
    """Return hsbw, then a closed polygon through absolute `points`."""
    items: list[int | bytes] = [sbx, 1000, HSBW]
    x, y = sbx, 0
    for i, (px, py) in enumerate(points):
        items += [px - x, py - y, RMOVETO if i == 0 else RLINETO]
        x, y = px, py
    items += [CLOSEPATH, ENDCHAR]
    return charstring(*items)


FD_BYTES, GD_BYTES, SD_BYTES = 1, 4, 4  # widths of the FD index and of every offset


def offset_table(base: int, chunks: list[bytes]) -> tuple[list[int], bytes]:
    """Lay out `chunks` contiguously from offset `base`.

    Returns the start offset of each chunk followed by the end offset, and
    the concatenated data.
    """
    offsets, data = [], bytearray()
    for chunk in chunks:
        offsets.append(base + len(data))
        data += chunk
    offsets.append(base + len(data))
    return offsets, bytes(data)


def cidmap_len(cid_count: int) -> int:
    """Return the CIDMap length: one FD index and offset per CID, plus the end offset."""
    return (cid_count + 1) * (FD_BYTES + GD_BYTES)


def start_data(glyphs: list[tuple[int, bytes]], subrs: list[bytes]) -> bytes:
    """Return the StartData section: CIDMap, SubrMap, subroutines, charstrings.

    `glyphs` holds `(fd_index, charstring)` per CID; the SubrMap belongs to
    FD 0 and follows the CIDMap.
    """
    data_offset = cidmap_len(len(glyphs)) + (len(subrs) + 1) * SD_BYTES
    subr_offsets, subr_data = offset_table(data_offset, subrs)
    glyph_offsets, glyph_data = offset_table(subr_offsets[-1], [g for _, g in glyphs])
    fds = [fd for fd, _ in glyphs] + [0]
    cidmap = b"".join(
        bytes([fd]) + struct.pack(">I", o) for fd, o in zip(fds, glyph_offsets, strict=True)
    )
    subrmap = b"".join(struct.pack(">I", o) for o in subr_offsets)
    return cidmap + subrmap + subr_data + glyph_data


def font_dict(index: int, private: str) -> str:
    """Return FDArray entry `index`, whose private dict ends with `private`."""
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


def build_cid_type1() -> bytes:
    """Return the CID-keyed Type 1 fixture described in the module docstring."""
    notdef = charstring(0, 1000, HSBW, ENDCHAR)
    square = polygon(100, [(100, 0), (400, 0), (400, 300), (100, 300)])
    # The subroutine draws two sides from the current point (50, 0).
    triangle = charstring(50, 1000, HSBW, 0, 0, RMOVETO, 4, CALLSUBR, CLOSEPATH, ENDCHAR)
    subr4 = charstring(400, 0, RLINETO, -200, 400, RLINETO, RETURN)
    stub = charstring(RETURN)
    diamond = polygon(500, [(500, 0), (700, 200), (500, 400), (300, 200)])

    fd0_subrs = [encrypt(s) for s in (stub, stub, stub, stub, subr4)]
    glyphs = [
        (0, encrypt(notdef)),
        (0, encrypt(square)),
        (0, encrypt(triangle)),
        (1, diamond),  # FD 1: lenIV -1, unencrypted
        (0xFF, b""),
    ]
    binary = start_data(glyphs, fd0_subrs)
    subrmap_offset = cidmap_len(len(glyphs))
    cid_count = len(glyphs)

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
        f"/FDBytes {FD_BYTES} def\n"
        f"/GDBytes {GD_BYTES} def\n"
        f"/CIDCount {cid_count} def\n"
        "/FDArray 2 array\n"
        + font_dict(
            0,
            f"/SubrMapOffset {subrmap_offset} def\n/SDBytes {SD_BYTES} def\n"
            f"/SubrCount {len(fd0_subrs)} def\n",
        )
        + font_dict(
            1,
            f"/lenIV -1 def\n/SubrMapOffset 0 def\n/SDBytes {SD_BYTES} def\n/SubrCount 0 def\n",
        )
        + "def\n"
        f"%%BeginData: {len(binary) + 26} Binary Bytes\n"
        f"(Binary) {len(binary)} StartData "
    )
    trailer = "\n%%EndData\n%%EndResource\n%%EOF\n"
    return header.encode("ascii") + binary + trailer.encode("ascii")


def main() -> None:
    """Rewrite every fixture next to this script."""
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
    write_truetype("no-cmap.ttf", build_no_cmap())
    (HERE / "cid-keyed-type1.cid").write_bytes(build_cid_type1())


if __name__ == "__main__":
    main()
