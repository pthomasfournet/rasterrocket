//! Synthetic sfnt fonts and sfnt table surgery for tests.

fn be16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn be32(v: u32) -> [u8; 4] {
    v.to_be_bytes()
}

/// `usize` byte length as a big-endian `u32`.
fn len_be32(n: usize) -> [u8; 4] {
    be32(u32::try_from(n).expect("synthetic table fits in u32"))
}

/// The `glyf` table: GID 0 empty, GID 1 a single-contour box
/// `(0,0)→(500,0)→(500,700)→(0,700)` in word-sized coordinates.
fn glyf() -> Vec<u8> {
    let mut g = Vec::new();
    g.extend_from_slice(&be16(1)); // numberOfContours
    g.extend_from_slice(&be16(0)); // xMin
    g.extend_from_slice(&be16(0)); // yMin
    g.extend_from_slice(&be16(500)); // xMax
    g.extend_from_slice(&be16(700)); // yMax
    g.extend_from_slice(&be16(3)); // endPtsOfContours[0] (4 pts)
    g.extend_from_slice(&be16(0)); // instructionLength
    g.extend_from_slice(&[0x01, 0x01, 0x01, 0x01]); // flags: on-curve, word coords
    for d in [0i16, 500, 0, -500] {
        g.extend_from_slice(&d.to_be_bytes()); // x deltas
    }
    for d in [0i16, 0, 700, 0] {
        g.extend_from_slice(&d.to_be_bytes()); // y deltas
    }
    while g.len() % 2 != 0 {
        g.push(0);
    }
    g
}

/// The `head` table (version 1.0, 1000 units per em, short `loca`).
fn head() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&be32(0x0001_0000)); // version
    h.extend_from_slice(&be32(0x0001_0000)); // fontRevision
    h.extend_from_slice(&be32(0)); // checkSumAdjustment
    h.extend_from_slice(&be32(0x5F0F_3CF5)); // magic
    h.extend_from_slice(&be16(0)); // flags
    h.extend_from_slice(&be16(1000)); // unitsPerEm
    h.extend_from_slice(&[0u8; 16]); // created, modified
    h.extend_from_slice(&be16(0)); // xMin
    h.extend_from_slice(&be16(0)); // yMin
    h.extend_from_slice(&be16(500)); // xMax
    h.extend_from_slice(&be16(700)); // yMax
    h.extend_from_slice(&be16(0)); // macStyle
    h.extend_from_slice(&be16(8)); // lowestRecPPEM
    h.extend_from_slice(&be16(2)); // fontDirectionHint
    h.extend_from_slice(&be16(0)); // indexToLocFormat (0 = short)
    h.extend_from_slice(&be16(0)); // glyphDataFormat
    h
}

/// The `hhea` table (2 hmetrics).
fn hhea() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&be32(0x0001_0000));
    h.extend_from_slice(&be16(700)); // ascender
    h.extend_from_slice(&(-200i16).to_be_bytes()); // descender
    h.extend_from_slice(&be16(0)); // lineGap
    h.extend_from_slice(&be16(600)); // advanceWidthMax
    h.extend_from_slice(&[0u8; 20]); // minLSB … caretOffset, reserved
    h.extend_from_slice(&be16(0)); // metricDataFormat
    h.extend_from_slice(&be16(2)); // numberOfHMetrics
    h
}

/// The (1,0) Macintosh Roman subtable: format 0, code 7 → GID 1.
fn cmap_mac() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&be16(0)); // format
    c.extend_from_slice(&be16(262)); // length
    c.extend_from_slice(&be16(0)); // language
    let mut g = [0u8; 256];
    g[7] = 1;
    c.extend_from_slice(&g);
    c
}

/// The (3,0) Microsoft Symbol subtable: format 4, 0xF007 → GID 1.
fn cmap_symbol() -> Vec<u8> {
    cmap_format4(0xF007)
}

/// A format 4 subtable mapping `code` to GID 1.
pub(super) fn cmap_format4(code: u16) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&be16(4)); // format
    c.extend_from_slice(&be16(32)); // length
    c.extend_from_slice(&be16(0)); // language
    c.extend_from_slice(&be16(4)); // segCountX2 (2 segments)
    c.extend_from_slice(&be16(4)); // searchRange
    c.extend_from_slice(&be16(1)); // entrySelector
    c.extend_from_slice(&be16(0)); // rangeShift
    c.extend_from_slice(&be16(code)); // endCode[0]
    c.extend_from_slice(&be16(0xFFFF)); // endCode[1]
    c.extend_from_slice(&be16(0)); // reservedPad
    c.extend_from_slice(&be16(code)); // startCode[0]
    c.extend_from_slice(&be16(0xFFFF)); // startCode[1]
    // gid = (code + delta) mod 65536.
    let delta =
        u16::try_from(1u32.wrapping_sub(u32::from(code)) & 0xFFFF).expect("masked to 16 bits");
    c.extend_from_slice(&be16(delta));
    c.extend_from_slice(&be16(1)); // idDelta[1]
    c.extend_from_slice(&be16(0)); // idRangeOffset[0]
    c.extend_from_slice(&be16(0)); // idRangeOffset[1]
    c
}

/// A `cmap` table holding `subtables` as `(platform, encoding, data)`.
fn cmap(subtables: &[(u16, u16, Vec<u8>)]) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&be16(0)); // version
    c.extend_from_slice(&be16(
        u16::try_from(subtables.len()).expect("few subtables"),
    ));
    let mut offset = 4 + 8 * subtables.len();
    for (platform, encoding, data) in subtables {
        c.extend_from_slice(&be16(*platform));
        c.extend_from_slice(&be16(*encoding));
        c.extend_from_slice(&len_be32(offset));
        offset += data.len();
    }
    for (_, _, data) in subtables {
        c.extend_from_slice(data);
    }
    c
}

/// Assemble an sfnt directory plus table bodies.
pub(super) fn assemble(mut tables: Vec<([u8; 4], Vec<u8>)>) -> Vec<u8> {
    tables.sort_by_key(|t| t.0);
    let num = u16::try_from(tables.len()).expect("table count fits in u16");
    let mut out = Vec::new();
    out.extend_from_slice(&be32(0x0001_0000)); // sfnt version
    out.extend_from_slice(&be16(num));
    let mut search_range = 1u16;
    let mut entry_selector = 0u16;
    while search_range * 2 <= num {
        search_range *= 2;
        entry_selector += 1;
    }
    out.extend_from_slice(&be16(search_range * 16));
    out.extend_from_slice(&be16(entry_selector));
    out.extend_from_slice(&be16(num * 16 - search_range * 16));
    let dir_len = 12 + 16 * tables.len();
    let mut body = Vec::new();
    for (tag, data) in &tables {
        out.extend_from_slice(tag);
        out.extend_from_slice(&be32(0)); // checksum (unchecked)
        out.extend_from_slice(&len_be32(dir_len + body.len()));
        out.extend_from_slice(&len_be32(data.len()));
        body.extend_from_slice(data);
        while body.len() % 4 != 0 {
            body.push(0);
        }
    }
    out.extend_from_slice(&body);
    out
}

/// A version 3 (nameless) `post` table.
fn post_v3() -> Vec<u8> {
    [be32(0x0003_0000).as_slice(), &[0u8; 28]].concat()
}

/// A version 2 `post` table naming GID 0 `.notdef` and GID 1 `A`, both
/// through the standard Macintosh glyph order (indices 0 and 36).
fn post_v2() -> Vec<u8> {
    [
        be32(0x0002_0000).as_slice(),
        &[0u8; 28],
        &be16(2),
        &be16(0),
        &be16(36),
    ]
    .concat()
}

/// A two-glyph TrueType font (`.notdef` + a 500×700 box at GID 1) whose
/// only charmaps are `cmaps`, with a version 3 (nameless) `post` table.
pub(super) fn box_font(cmaps: &[(u16, u16, Vec<u8>)]) -> Vec<u8> {
    box_font_with_post(cmaps, post_v3())
}

/// [`box_font`] with a `post` table that names GID 1 `A`.
pub(super) fn named_box_font(cmaps: &[(u16, u16, Vec<u8>)]) -> Vec<u8> {
    box_font_with_post(cmaps, post_v2())
}

/// [`box_font`] with a `post` table that names GID 1 `circlesolid`, a name
/// outside the Adobe Glyph List, stored as a custom Pascal string.
pub(super) fn custom_named_box_font(cmaps: &[(u16, u16, Vec<u8>)]) -> Vec<u8> {
    let name = b"circlesolid";
    let post = [
        be32(0x0002_0000).as_slice(),
        &[0u8; 28],
        &be16(2),
        &be16(0),
        &be16(258),
        &[u8::try_from(name.len()).expect("short name")],
        name,
    ]
    .concat();
    box_font_with_post(cmaps, post)
}

fn box_font_with_post(cmaps: &[(u16, u16, Vec<u8>)], post: Vec<u8>) -> Vec<u8> {
    let glyf = glyf();
    let loca = {
        let end = u16::try_from(glyf.len() / 2).expect("glyf length / 2 fits in u16");
        [be16(0), be16(0), be16(end)].concat()
    };
    let maxp = [be32(0x0001_0000).as_slice(), &be16(2), &[0u8; 26]].concat();
    let hmtx = [be16(500), be16(0), be16(600), be16(0)].concat();
    assemble(vec![
        (*b"cmap", cmap(cmaps)),
        (*b"glyf", glyf),
        (*b"head", head()),
        (*b"hhea", hhea()),
        (*b"hmtx", hmtx),
        (*b"loca", loca),
        (*b"maxp", maxp),
        (*b"post", post),
    ])
}

/// The symbolic-subset shape: (1,0) Macintosh Roman and (3,0) Microsoft
/// Symbol charmaps, no Unicode charmap.
pub(super) fn symbolic_box_font() -> Vec<u8> {
    box_font(&[(1, 0, cmap_mac()), (3, 0, cmap_symbol())])
}

/// Only a (3,0) Microsoft Symbol charmap.
pub(super) fn symbol_only_box_font() -> Vec<u8> {
    box_font(&[(3, 0, cmap_symbol())])
}

/// Only a (1,0) Macintosh Roman charmap.
pub(super) fn mac_only_cmap() -> Vec<(u16, u16, Vec<u8>)> {
    vec![(1, 0, cmap_mac())]
}

/// The body of table `tag` in an sfnt.
pub(super) fn table(font: &[u8], tag: [u8; 4]) -> Vec<u8> {
    let num = usize::from(u16::from_be_bytes([font[4], font[5]]));
    (0..num)
        .map(|i| &font[12 + i * 16..28 + i * 16])
        .find(|record| record[..4] == tag)
        .map(|record| {
            let offset = u32::from_be_bytes(record[8..12].try_into().expect("4 bytes")) as usize;
            let len = u32::from_be_bytes(record[12..16].try_into().expect("4 bytes")) as usize;
            font[offset..offset + len].to_vec()
        })
        .expect("table present")
}

/// Hide tables from lookups by renaming their directory tags; the
/// replacement tags keep the directory sorted.
pub(super) fn rename_tables(font: &[u8], renames: &[(&[u8; 4], &[u8; 4])]) -> Vec<u8> {
    let mut out = font.to_vec();
    let num = usize::from(u16::from_be_bytes([font[4], font[5]]));
    for i in 0..num {
        let at = 12 + i * 16;
        if let Some((_, to)) = renames.iter().find(|(from, _)| out[at..at + 4] == from[..]) {
            out[at..at + 4].copy_from_slice(*to);
        }
    }
    out
}
