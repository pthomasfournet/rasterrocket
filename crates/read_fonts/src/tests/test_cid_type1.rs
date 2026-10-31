use super::*;
use crate::ps::cs::test_helpers::{CaptureCommandSink, Command};
use alloc::{format, vec};

fn num(v: i32) -> Vec<u8> {
    match v {
        -107..=107 => vec![(v + 139) as u8],
        108..=1131 => {
            let v = v - 108;
            vec![((v >> 8) + 247) as u8, v as u8]
        }
        -1131..=-108 => {
            let v = -v - 108;
            vec![((v >> 8) + 251) as u8, v as u8]
        }
        _ => [&[255u8][..], &v.to_be_bytes()].concat(),
    }
}

const HSBW: u8 = 13;
const RMOVETO: u8 = 21;
const RLINETO: u8 = 5;
const CLOSEPATH: u8 = 9;
const ENDCHAR: u8 = 14;
const CALLSUBR: u8 = 10;
const RETURN: u8 = 11;
const SEAC: [u8; 2] = [12, 6];

/// Charstring from numbers and single-byte operators (`Err`).
fn charstring(items: &[Result<i32, u8>]) -> Vec<u8> {
    items
        .iter()
        .flat_map(|item| match item {
            Ok(v) => num(*v),
            Err(op) => vec![*op],
        })
        .collect()
}

fn encrypt(plain: &[u8], len_iv: usize) -> Vec<u8> {
    let mut r: u32 = 4330;
    let mut out = Vec::new();
    for &p in core::iter::repeat_n(&0u8, len_iv).chain(plain) {
        let c = p ^ (r >> 8) as u8;
        out.push(c);
        r = ((c as u32 + r) * 52845 + 22719) & 0xFFFF;
    }
    out
}

/// `hsbw`, then a 100 × 100 square with its lower left corner at
/// `(sbx, 0)`.
fn square(sbx: i32) -> Vec<u8> {
    charstring(&[
        Ok(sbx),
        Ok(1000),
        Err(HSBW),
        Ok(0),
        Ok(0),
        Err(RMOVETO),
        Ok(100),
        Ok(0),
        Err(RLINETO),
        Ok(0),
        Ok(100),
        Err(RLINETO),
        Ok(-100),
        Ok(0),
        Err(RLINETO),
        Err(CLOSEPATH),
        Err(ENDCHAR),
    ])
}

struct Dict {
    len_iv: Option<i32>,
    matrix: &'static str,
    subrs: Vec<Vec<u8>>,
}

/// A CID-keyed font with FDBytes 1, GDBytes 2, SDBytes 2; `glyphs` are
/// `(dict, plain charstring)`, dict 0xFF marking a missing CID.
fn build(dicts: &[Dict], glyphs: &[(u8, Vec<u8>)], hex: bool) -> Vec<u8> {
    let cidmap_len = (glyphs.len() + 1) * 3;
    let mut binary_tail = Vec::new();
    let mut subrmaps = Vec::new();
    let subrmap_total: usize = dicts.iter().map(|d| (d.subrs.len() + 1) * 2).sum();
    let base = cidmap_len + subrmap_total;
    let mut subrmap_offsets = Vec::new();
    for dict in dicts {
        subrmap_offsets.push(cidmap_len + subrmaps.len());
        let len_iv = dict.len_iv.unwrap_or(4);
        let mut offsets = Vec::new();
        for subr in &dict.subrs {
            offsets.push(base + binary_tail.len());
            if len_iv >= 0 {
                binary_tail.extend(encrypt(subr, len_iv as usize));
            } else {
                binary_tail.extend_from_slice(subr);
            }
        }
        offsets.push(base + binary_tail.len());
        for offset in offsets {
            subrmaps.extend_from_slice(&(offset as u16).to_be_bytes());
        }
    }
    let mut cidmap = Vec::new();
    for (dict, plain) in glyphs {
        cidmap.push(*dict);
        cidmap.extend_from_slice(&((base + binary_tail.len()) as u16).to_be_bytes());
        let len_iv = dicts
            .get(*dict as usize)
            .map_or(-1, |d| d.len_iv.unwrap_or(4));
        if len_iv >= 0 {
            binary_tail.extend(encrypt(plain, len_iv as usize));
        } else {
            binary_tail.extend_from_slice(plain);
        }
    }
    cidmap.push(0);
    cidmap.extend_from_slice(&((base + binary_tail.len()) as u16).to_be_bytes());
    let binary = [cidmap, subrmaps, binary_tail].concat();
    let mut header = format!(
        "%!PS-Adobe-3.0 Resource-CIDFont\n\
         /CIDInit /ProcSet findresource begin\n20 dict begin\n\
         /CIDFontName /Test def\n/CIDFontType 0 def\n\
         /FontMatrix [1 0 0 1 0 0] def\n\
         /CIDMapOffset 0 def\n/FDBytes 1 def\n/GDBytes 2 def\n\
         /CIDCount {} def\n/FDArray {} array\n",
        glyphs.len(),
        dicts.len()
    );
    for (i, dict) in dicts.iter().enumerate() {
        let len_iv = dict
            .len_iv
            .map_or(String::new(), |v| format!("/lenIV {v} def\n"));
        header += &format!(
            "dup {i}\n%ADOBeginFontDict\n14 dict begin\n/FontName /Test-{i} def\n\
             /FontType 1 def\n/FontMatrix [{}] def\n/PaintType 0 def\n\
             /Private 18 dict dup begin\n/MinFeature {{16 16}} def\n{len_iv}\
             /SubrMapOffset {} def\n/SDBytes 2 def\n/SubrCount {} def\n\
             end def\ncurrentdict end\n%ADOEndFontDict\nput\n",
            dict.matrix,
            subrmap_offsets[i],
            dict.subrs.len()
        );
    }
    header += "def\n";
    let mut out = header.into_bytes();
    if hex {
        out.extend(format!("(Hex) {} StartData ", binary.len()).bytes());
        for (i, byte) in binary.iter().enumerate() {
            out.extend(format!("{byte:02X}").bytes());
            if i % 32 == 31 {
                out.push(b'\n');
            }
        }
        out.extend_from_slice(b">\n");
    } else {
        out.extend(format!("(Binary) {} StartData ", binary.len()).bytes());
        out.extend_from_slice(&binary);
    }
    out.extend_from_slice(b"\n%%EndData\n%%EndResource\n");
    out
}

const IDENTITY: &str = "0.001 0 0 0.001 0 0";

fn sample(hex: bool) -> Vec<u8> {
    let triangle = charstring(&[
        Ok(0),
        Ok(1000),
        Err(HSBW),
        Ok(0),
        Ok(0),
        Err(RMOVETO),
        Ok(1),
        Err(CALLSUBR),
        Err(CLOSEPATH),
        Err(ENDCHAR),
    ]);
    let subr = charstring(&[
        Ok(200),
        Ok(0),
        Err(RLINETO),
        Ok(-100),
        Ok(200),
        Err(RLINETO),
        Err(RETURN),
    ]);
    let dicts = [
        Dict {
            len_iv: None,
            matrix: IDENTITY,
            subrs: vec![charstring(&[Err(RETURN)]), subr],
        },
        Dict {
            len_iv: Some(-1),
            matrix: IDENTITY,
            subrs: vec![],
        },
    ];
    let notdef = charstring(&[Ok(0), Ok(500), Err(HSBW), Err(ENDCHAR)]);
    build(
        &dicts,
        &[
            (0, notdef),
            (0, square(50)),
            (0, triangle),
            (1, square(300)),
            (0xFF, vec![]),
        ],
        hex,
    )
}

fn commands(font: &CidType1Font, cid: u32) -> Vec<Command> {
    let mut sink = CaptureCommandSink::default();
    font.evaluate_charstring(GlyphId::new(cid), &mut sink)
        .unwrap();
    sink.0
}

fn line(x: i32, y: i32) -> Command {
    Command::LineTo(Fixed::from_i32(x), Fixed::from_i32(y))
}

#[test]
fn reads_cidmap_dicts_and_subrs() {
    for hex in [false, true] {
        let data = sample(hex);
        assert!(CidType1Font::is_cid_keyed(&data));
        let font = CidType1Font::new(&data).unwrap();
        assert_eq!(font.name(), Some("Test"));
        assert_eq!(font.num_glyphs(), 5);
        assert_eq!(font.num_font_dicts(), 2);
        assert_eq!(font.upem(), 1000);
        // CID 1: square through the encrypted charstrings of dict 0.
        let square = commands(&font, 1);
        assert_eq!(square[0], Command::MoveTo(Fixed::from_i32(50), Fixed::ZERO));
        assert!(square.contains(&line(150, 100)));
        // CID 2: two sides drawn by subroutine 1 of dict 0.
        let triangle = commands(&font, 2);
        assert!(triangle.contains(&line(200, 0)), "{triangle:?}");
        assert!(triangle.contains(&line(100, 200)), "{triangle:?}");
        // CID 3: dict 1 stores its charstring unencrypted (lenIV -1).
        assert_eq!(font.font_dict_index(GlyphId::new(3)), Some(1));
        assert!(commands(&font, 3).contains(&line(400, 100)));
        // CID 4 has no glyph; CID 5 is beyond CIDCount.
        assert_eq!(font.font_dict_index(GlyphId::new(4)), None);
        let mut sink = CaptureCommandSink::default();
        assert!(font
            .evaluate_charstring(GlyphId::new(4), &mut sink)
            .is_err());
        assert!(font
            .evaluate_charstring(GlyphId::new(5), &mut sink)
            .is_err());
    }
}

#[test]
fn advance_comes_from_hsbw() {
    let font = CidType1Font::new(&sample(false)).unwrap();
    let mut sink = CaptureCommandSink::default();
    let width = font
        .evaluate_charstring(GlyphId::new(0), &mut sink)
        .unwrap();
    assert_eq!(width, Some(Fixed::from_i32(500)));
}

#[test]
fn font_dict_matrix_scales_its_glyphs() {
    // A 2048-unit dictionary: upem follows its matrix, and a dictionary
    // matrix with a skew keeps it after normalization.
    let dicts = [Dict {
        len_iv: Some(0),
        matrix: "0.00048828125 0 0.0001 0.00048828125 0 0",
        subrs: vec![],
    }];
    let data = build(&dicts, &[(0, square(0))], false);
    let font = CidType1Font::new(&data).unwrap();
    assert_eq!(font.upem(), 2048);
    let transform = font.transform(GlyphId::new(0), None).unwrap();
    assert_eq!(transform.matrix.xx, Fixed::ONE);
    assert!(transform.matrix.xy > Fixed::ZERO);
}

#[test]
fn seac_components_are_cids() {
    let dicts = [Dict {
        len_iv: None,
        matrix: IDENTITY,
        subrs: vec![],
    }];
    let mut accented = charstring(&[Ok(0), Ok(0), Ok(0), Ok(1), Ok(2)]);
    accented.extend_from_slice(&SEAC);
    let data = build(
        &dicts,
        &[(0, accented), (0, square(0)), (0, square(500))],
        false,
    );
    let font = CidType1Font::new(&data).unwrap();
    let context = FontDictContext {
        font: &font,
        dict: &font.dicts[0],
    };
    let [base, accent] = context.seac_components(1, 2).unwrap();
    assert_eq!(base, font.charstring(1).unwrap());
    assert_eq!(accent, font.charstring(2).unwrap());
    assert!(context.seac_components(7, 2).is_err());
}

fn position(data: &[u8], needle: &[u8]) -> usize {
    find_all(data, needle)[0]
}

#[test]
fn rejects_malformed_layouts() {
    let good = sample(false);
    assert!(CidType1Font::new(&good).is_ok());
    let replace = |from: &str, to: &str| {
        let at = position(&good, from.as_bytes());
        [&good[..at], to.as_bytes(), &good[at + from.len()..]].concat()
    };
    // Not a PostScript resource, no data section, no FDArray, bad
    // GDBytes, a CIDCount beyond the data.
    assert!(CidType1Font::new(b"\x80\x01junk").is_err());
    let no_data = good
        .split(|&b| b == b'(')
        .next()
        .map(|h| h.to_vec())
        .unwrap();
    let no_data = [&no_data[..], b"def\n"].concat();
    assert!(CidType1Font::new(&no_data).is_err());
    assert!(CidType1Font::new(&replace("/FDArray", "/FDArrayX")).is_err());
    assert!(CidType1Font::new(&replace("/GDBytes 2", "/GDBytes 0")).is_err());
    assert!(CidType1Font::new(&replace("/GDBytes 2", "/GDBytes 9")).is_err());
    assert!(CidType1Font::new(&replace("/CIDCount 5", "/CIDCount 99999")).is_err());
}

#[test]
fn excess_dict_markers_are_ignored() {
    // A marker beyond the declared FDArray size opens no dictionary;
    // keys after it are ignored rather than overwriting the last one.
    let data = sample(false);
    let at = position(&data, b"def\n(Binary)");
    let extra = "%ADOBeginFontDict\n/lenIV 2 def\n";
    let data = [&data[..at], extra.as_bytes(), &data[at..]].concat();
    let font = CidType1Font::new(&data).unwrap();
    assert!(commands(&font, 3).contains(&line(400, 100)));
}

/// A font of unencrypted font dictionaries over the given data section,
/// with FDBytes 0, GDBytes 4 and SDBytes 4; all `dicts` dictionaries name
/// the subroutine map at `subrmap_offset`.
fn raw_font(
    binary: &[u8],
    cid_count: usize,
    dicts: usize,
    subrmap_offset: usize,
    subr_count: usize,
) -> Vec<u8> {
    let mut out = format!(
        "%!PS-Adobe-3.0 Resource-CIDFont\n/CIDFontName /Test def\n\
         /CIDMapOffset 0 def\n/FDBytes 0 def\n/GDBytes 4 def\n\
         /CIDCount {cid_count} def\n/FDArray {dicts} array\n"
    );
    for i in 0..dicts {
        out += &format!(
            "dup {i}\n%ADOBeginFontDict\n/FontMatrix [{IDENTITY}] def\n/lenIV -1 def\n\
             /SubrMapOffset {subrmap_offset} def\n/SDBytes 4 def\n\
             /SubrCount {subr_count} def\n%ADOEndFontDict\nput\n"
        );
    }
    out += &format!("def\n(Binary) {} StartData ", binary.len());
    let mut out = out.into_bytes();
    out.extend_from_slice(binary);
    out
}

/// A CID map of FDBytes 0, GDBytes 4 holding `offset(i)` for entries
/// `0..=cid_count`.
fn cidmap(cid_count: usize, offset: impl Fn(usize) -> usize) -> Vec<u8> {
    (0..=cid_count)
        .flat_map(|i| (offset(i) as u32).to_be_bytes())
        .collect()
}

#[test]
fn overlapping_charstrings_are_bounded_by_the_data_section() {
    // Offsets 0, L, 1, L, 2, L, ...: every even CID's charstring runs from
    // a distinct start to the end of the data section, so the ranges
    // overlap and their total length is quadratic in L.
    let cid_count = 4096;
    let len = (cid_count + 1) * 4;
    let binary = cidmap(cid_count, |i| if i % 2 == 0 { i / 2 } else { len });
    let font = CidType1Font::new(&raw_font(&binary, cid_count, 1, 0, 0)).unwrap();
    assert!(font.charstrings.len() <= len, "{}", font.charstrings.len());
    assert_eq!(font.charstring(0).map(<[u8]>::len), Some(len));
    assert!(font.charstring(2).is_none());
}

#[test]
fn identical_charstring_ranges_share_one_copy() {
    // Offsets 0, L, 0, L, ...: every even CID names the whole section.
    let cid_count = 64;
    let len = (cid_count + 1) * 4;
    let binary = cidmap(cid_count, |i| if i % 2 == 0 { 0 } else { len });
    let font = CidType1Font::new(&raw_font(&binary, cid_count, 1, 0, 0)).unwrap();
    assert_eq!(font.charstrings.len(), len);
    for cid in (0..cid_count as u32).step_by(2) {
        assert_eq!(font.charstring(cid).map(<[u8]>::len), Some(len));
    }
}

#[test]
fn font_dicts_naming_one_subr_map_share_their_subrs() {
    // Every font dictionary names the same map, whose one subroutine is
    // the whole data section.
    let binary = cidmap(1, |i| i * 8);
    let dicts = 200;
    let font = CidType1Font::new(&raw_font(&binary, 0, dicts, 0, 1)).unwrap();
    assert_eq!(font.num_font_dicts(), dicts as u32);
    assert_eq!(font.subrs.len(), 1);
    assert_eq!(font.subrs[0].data.len(), 8);
}
