//! Synthetic Type 1 programs for tests.
//!
//! [`synthetic_pfa`] writes a complete eexec-encrypted Type 1 font whose
//! charstrings exercise the operators embedded PDF fonts depend on:
//!
//! | Glyph | Content |
//! |-------|---------|
//! | `A` | 600×700 box, drawn after a hint-replacement call (`othersubr 3`) |
//! | `acute` | triangle above the x-height |
//! | `Aacute` | `seac` of `A` (code 65) and `acute` (code 194) |
//! | `B` | box whose top edge is a flex (`othersubr 0/1/2`) |
//! | `C` | box with an advance width computed by `div` |

const EEXEC_KEY: u16 = 55665;
const CHARSTRING_KEY: u16 = 4330;

/// Type 1 encryption (Adobe Type 1 Font Format §7.1).
fn encrypt(plain: &[u8], key: u16) -> Vec<u8> {
    let mut r = key;
    plain
        .iter()
        .map(|&p| {
            let c = p ^ (r >> 8) as u8;
            r = (u16::from(c).wrapping_add(r))
                .wrapping_mul(52845)
                .wrapping_add(22719);
            c
        })
        .collect()
}

/// Charstring number encoding (Adobe Type 1 Font Format §6.2).
fn num(out: &mut Vec<u8>, v: i32) {
    match v {
        -107..=107 => out.push(u8::try_from(v + 139).expect("in range")),
        108..=1131 => {
            let v = v - 108;
            out.push(u8::try_from(v / 256 + 247).expect("in range"));
            out.push(u8::try_from(v % 256).expect("in range"));
        }
        -1131..=-108 => {
            let v = -v - 108;
            out.push(u8::try_from(v / 256 + 251).expect("in range"));
            out.push(u8::try_from(v % 256).expect("in range"));
        }
        _ => {
            out.push(255);
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
}

/// A charstring token: a number or an operator byte sequence.
enum T {
    N(i32),
    Op(&'static [u8]),
}

const HSTEM: &[u8] = &[1];
const VSTEM: &[u8] = &[3];
const RLINETO: &[u8] = &[5];
const CLOSEPATH: &[u8] = &[9];
const CALLSUBR: &[u8] = &[10];
const RETURN: &[u8] = &[11];
const HSBW: &[u8] = &[13];
const ENDCHAR: &[u8] = &[14];
const RMOVETO: &[u8] = &[21];
const SEAC: &[u8] = &[12, 6];
const DIV: &[u8] = &[12, 12];
const CALLOTHERSUBR: &[u8] = &[12, 16];
const POP: &[u8] = &[12, 17];
const SETCURRENTPOINT: &[u8] = &[12, 33];

fn charstring(tokens: &[T]) -> Vec<u8> {
    let mut plain = vec![0u8; 4]; // lenIV = 4 leading bytes
    for t in tokens {
        match t {
            T::N(v) => num(&mut plain, *v),
            T::Op(op) => plain.extend_from_slice(op),
        }
    }
    encrypt(&plain, CHARSTRING_KEY)
}

use T::{N, Op};

/// The four standard flex / hint-replacement subroutines plus subroutine 4,
/// the hint set the hint-replacement call in `A` selects.
fn subrs() -> Vec<Vec<u8>> {
    vec![
        charstring(&[
            N(3),
            N(0),
            Op(CALLOTHERSUBR),
            Op(POP),
            Op(POP),
            Op(SETCURRENTPOINT),
            Op(RETURN),
        ]),
        charstring(&[N(0), N(1), Op(CALLOTHERSUBR), Op(RETURN)]),
        charstring(&[N(0), N(2), Op(CALLOTHERSUBR), Op(RETURN)]),
        charstring(&[
            N(3),
            N(1),
            N(3),
            Op(CALLOTHERSUBR),
            Op(POP),
            Op(CALLSUBR),
            Op(RETURN),
        ]),
        charstring(&[
            N(0),
            N(700),
            Op(HSTEM),
            N(50),
            N(500),
            Op(VSTEM),
            Op(RETURN),
        ]),
    ]
}

fn box_path(sbx: i32, w: i32, h: i32) -> Vec<T> {
    vec![
        N(sbx),
        N(0),
        Op(RMOVETO),
        N(w),
        N(0),
        Op(RLINETO),
        N(0),
        N(h),
        Op(RLINETO),
        N(-w),
        N(0),
        Op(RLINETO),
        Op(CLOSEPATH),
    ]
}

fn charstrings() -> Vec<(&'static str, Vec<u8>)> {
    let notdef = charstring(&[N(0), N(250), Op(HSBW), Op(ENDCHAR)]);

    // A: hint replacement selects subr 4 (`4 1 3 callothersubr pop
    // callsubr`), then a 600×700 box starting at the side bearing (50, 0).
    let mut a = vec![
        N(50),
        N(700),
        Op(HSBW),
        N(4),
        N(1),
        N(3),
        Op(CALLOTHERSUBR),
        Op(POP),
        Op(CALLSUBR),
    ];
    a.extend(box_path(0, 600, 700));
    a.push(Op(ENDCHAR));

    // acute: triangle (100,750)-(300,750)-(250,900), side bearing 100.
    let acute = charstring(&[
        N(100),
        N(400),
        Op(HSBW),
        N(0),
        N(750),
        Op(RMOVETO),
        N(200),
        N(0),
        Op(RLINETO),
        N(-50),
        N(150),
        Op(RLINETO),
        Op(CLOSEPATH),
        Op(ENDCHAR),
    ]);

    // Aacute: `asb adx ady bchar achar seac` with the accent placed 150
    // units right of the base.
    let composite = charstring(&[
        N(50),
        N(700),
        Op(HSBW),
        N(100),
        N(150),
        N(0),
        N(65),
        N(194),
        Op(SEAC),
    ]);

    // B: from (50,0) a box whose top edge, drawn right to left from
    // (650,700) to (50,700), is a flex dipping to y = 680 at x = 350.
    // Flex: `1 callsubr`, the reference point (350,700) and the six curve
    // points each as `dx dy rmoveto 2 callsubr`, then `height x y 0
    // callsubr` with the absolute end point.
    let mut b = vec![
        N(50),
        N(700),
        Op(HSBW),
        N(0),
        N(0),
        Op(RMOVETO),
        N(600),
        N(0),
        Op(RLINETO),
        N(0),
        N(700),
        Op(RLINETO),
        N(1),
        Op(CALLSUBR),
    ];
    for (dx, dy) in [
        (-300, 0),
        (200, 0),
        (-100, -20),
        (-100, 0),
        (-100, 0),
        (-100, 20),
        (-100, 0),
    ] {
        b.extend([N(dx), N(dy), Op(RMOVETO), N(2), Op(CALLSUBR)]);
    }
    b.extend([
        N(50),
        N(50),
        N(700),
        N(0),
        Op(CALLSUBR),
        Op(CLOSEPATH),
        Op(ENDCHAR),
    ]);

    // C: advance 1500 / 3 = 500 through `div`.
    let mut c = vec![N(50), N(1500), N(3), Op(DIV), Op(HSBW)];
    c.extend(box_path(0, 400, 400));
    c.push(Op(ENDCHAR));

    vec![
        (".notdef", notdef),
        ("A", charstring(&a)),
        ("acute", acute),
        ("Aacute", composite),
        ("B", charstring(&b)),
        ("C", charstring(&c)),
    ]
}

/// The clear-text font dictionary preceding `eexec`.
const CLEAR_TEXT: &str = "%!PS-AdobeFont-1.0: Synthetic 001.000\n\
11 dict begin\n\
/FontInfo 2 dict dup begin /FullName (Synthetic) readonly def end readonly def\n\
/FontName /Synthetic def\n\
/Encoding StandardEncoding def\n\
/PaintType 0 def\n\
/FontType 1 def\n\
/FontMatrix [0.001 0 0 0.001 0 0] readonly def\n\
/FontBBox {0 0 1000 1000} readonly def\n\
currentdict end\n\
currentfile eexec\n";

/// The decrypted private section: `Private` dictionary, `Subrs`, `CharStrings`.
fn private_section() -> Vec<u8> {
    let mut p = b"XXXX dup /Private 8 dict dup begin\n\
/RD{string currentfile exch readstring pop}executeonly def\n\
/ND{noaccess def}executeonly def\n\
/NP{noaccess put}executeonly def\n\
/lenIV 4 def\n\
/BlueValues [] def\n"
        .to_vec();
    let subrs = subrs();
    p.extend_from_slice(format!("/Subrs {} array\n", subrs.len()).as_bytes());
    for (i, s) in subrs.iter().enumerate() {
        p.extend_from_slice(format!("dup {i} {} RD ", s.len()).as_bytes());
        p.extend_from_slice(s);
        p.extend_from_slice(b" NP\n");
    }
    p.extend_from_slice(b"ND\n");
    let glyphs = charstrings();
    p.extend_from_slice(
        format!("2 index /CharStrings {} dict dup begin\n", glyphs.len()).as_bytes(),
    );
    for (name, cs) in &glyphs {
        p.extend_from_slice(format!("/{name} {} RD ", cs.len()).as_bytes());
        p.extend_from_slice(cs);
        p.extend_from_slice(b" ND\n");
    }
    p.extend_from_slice(b"end\nend\nreadonly put\nnoaccess put\ndup /FontName get exch definefont pop\nmark currentfile closefile\n");
    p
}

fn trailer() -> Vec<u8> {
    let mut t = Vec::new();
    for _ in 0..8 {
        t.extend_from_slice(&[b'0'; 64]);
        t.push(b'\n');
    }
    t.extend_from_slice(b"cleartomark\n");
    t
}

/// The synthetic font as PFA with a hex-encoded eexec section.
pub(super) fn synthetic_pfa() -> Vec<u8> {
    let mut out = CLEAR_TEXT.as_bytes().to_vec();
    for (i, byte) in encrypt(&private_section(), EEXEC_KEY).iter().enumerate() {
        out.extend_from_slice(format!("{byte:02X}").as_bytes());
        if i % 32 == 31 {
            out.push(b'\n');
        }
    }
    out.push(b'\n');
    out.extend_from_slice(&trailer());
    out
}

/// The synthetic font with a binary eexec section, the layout of a PDF
/// `FontFile` stream (`Length1` clear-text bytes, then `Length2` binary).
pub(super) fn synthetic_binary_eexec() -> Vec<u8> {
    let mut out = CLEAR_TEXT.as_bytes().to_vec();
    out.extend_from_slice(&encrypt(&private_section(), EEXEC_KEY));
    out.extend_from_slice(&trailer());
    out
}

/// The synthetic font as PFB: text, binary and text segments.
pub(super) fn synthetic_pfb() -> Vec<u8> {
    let segment = |kind: u8, data: &[u8]| {
        let mut s = vec![0x80, kind];
        s.extend_from_slice(&u32::try_from(data.len()).expect("small").to_le_bytes());
        s.extend_from_slice(data);
        s
    };
    let mut out = segment(1, CLEAR_TEXT.as_bytes());
    out.extend(segment(2, &encrypt(&private_section(), EEXEC_KEY)));
    out.extend(segment(1, &trailer()));
    out.extend_from_slice(&[0x80, 0x03]);
    out
}
