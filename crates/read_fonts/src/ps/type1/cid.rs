//! CID-keyed Type 1 fonts (CIDFontType 0).
//!
//! A CID-keyed font (Adobe Technical Note #5014) is a cleartext PostScript
//! header followed by a binary (or hex) data section introduced by
//! `StartData`. The header gives the layout of the data section
//! (`CIDMapOffset`, `FDBytes`, `GDBytes`, `CIDCount`) and an `FDArray` of
//! Type 1 font dictionaries, each with its own `FontMatrix` and a private
//! dictionary locating its subroutines (`SubrMapOffset`, `SDBytes`,
//! `SubrCount`) and setting its `lenIV`. The data section holds the CIDMap
//! — for each CID, the index of its font dictionary and the offset of its
//! charstring — followed by the subroutines and charstrings, encrypted
//! with the charstring key when the dictionary's `lenIV` is not negative.
//!
//! Parsing follows FreeType's `cid` driver: the font dictionaries are
//! delimited by the `%ADOBeginFontDict` comments, and dictionary keys
//! apply to the most recently opened one.
//! <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidload.c>

use super::{decode_hex, decrypt, Parser, Subrs, Token, CHARSTRING_SEED};
use crate::{
    model::pen::OutlinePen,
    ps::{
        cs::{self, CharstringContext, CharstringKind, CommandSink, NopFilterSink, TransformSink},
        error::Error,
        transform::{FontMatrix, ScaledFontMatrix, Transform},
    },
    types::{Fixed, GlyphId},
    ReadError,
};
use alloc::{borrow::ToOwned, string::String, vec::Vec};
use core::ops::Range;

/// Header that identifies a CID-keyed font resource.
const CID_FONT_HEADER: &[u8] = b"%!PS-Adobe-3.0 Resource-CIDFont";

/// Comment that opens each font dictionary of the `FDArray`.
const FONT_DICT_MARKER: &[u8] = b"%ADOBeginFontDict";

/// A CID-keyed Type 1 font. Glyph identifiers are CIDs.
pub struct CidType1Font {
    name: Option<String>,
    upem: i32,
    dicts: Vec<FontDict>,
    /// Charstring of each CID, `None` for a CID without a glyph.
    glyphs: Vec<Option<GlyphEntry>>,
    /// Decrypted charstrings, with the `lenIV` prefix removed.
    charstrings: Vec<u8>,
}

struct FontDict {
    matrix: FontMatrix,
    subrs: Subrs,
}

struct GlyphEntry {
    dict: u32,
    data: Range<usize>,
}

/// Header values of one font dictionary.
#[derive(Default)]
struct RawFontDict {
    matrix: Option<ScaledFontMatrix>,
    len_iv: Option<i64>,
    subrmap_offset: u64,
    sd_bytes: u64,
    subr_count: u64,
}

/// Header values of the CIDFont.
#[derive(Default)]
struct RawHeader {
    name: Option<String>,
    cidmap_offset: u64,
    fd_bytes: u64,
    gd_bytes: u64,
    cid_count: u64,
    dicts: Vec<RawFontDict>,
}

impl CidType1Font {
    /// Returns true if `data` begins with the header of a CID-keyed font
    /// resource (`%!PS-Adobe-3.0 Resource-CIDFont`).
    pub fn is_cid_keyed(data: &[u8]) -> bool {
        data.starts_with(CID_FONT_HEADER)
    }

    /// Creates a new CID-keyed Type 1 font from the given data.
    ///
    /// Any PostScript header (`%!`) is accepted; the font must provide an
    /// `FDArray` and a `StartData` data section.
    pub fn new(data: &[u8]) -> Result<Self, Error> {
        Self::new_impl(data).ok_or(Error::InvalidFontFormat)
    }

    fn new_impl(data: &[u8]) -> Option<Self> {
        if !data.starts_with(b"%!") {
            return None;
        }
        let (header_end, data_start, hex_len) = find_start_data(data)?;
        let header = parse_header(data.get(..header_end)?)?;
        let hex_binary;
        let binary = match hex_len {
            Some(len) => {
                hex_binary = decode_hex(data.get(data_start..)?.iter().copied())
                    .take(len)
                    .collect::<Vec<_>>();
                &hex_binary[..]
            }
            None => data.get(data_start..)?,
        };
        Self::from_parts(header, binary)
    }

    fn from_parts(header: RawHeader, binary: &[u8]) -> Option<Self> {
        let RawHeader {
            name,
            cidmap_offset,
            fd_bytes,
            gd_bytes,
            cid_count,
            dicts: raw_dicts,
        } = header;
        // <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidload.c#L835>
        if raw_dicts.is_empty() || gd_bytes == 0 || fd_bytes > 4 || gd_bytes > 4 {
            return None;
        }
        let len = binary.len() as u64;
        let entry_len = fd_bytes + gd_bytes;
        if cidmap_offset > len || cid_count.checked_mul(entry_len)? > len - cidmap_offset {
            return None;
        }
        let mut upem = 1000;
        let mut dicts = Vec::with_capacity(raw_dicts.len());
        let mut len_ivs = Vec::with_capacity(raw_dicts.len());
        for raw in &raw_dicts {
            let matrix = match raw.matrix {
                Some(scaled) => {
                    if scaled.scale != 1000 {
                        upem = scaled.scale;
                    }
                    scaled.matrix
                }
                None => FontMatrix::IDENTITY,
            };
            let len_iv = raw.len_iv.unwrap_or(4);
            dicts.push(FontDict {
                matrix,
                subrs: read_subrs(binary, raw, len_iv)?,
            });
            len_ivs.push(len_iv);
        }
        let cid_count = usize::try_from(cid_count).ok()?;
        let read = |pos: u64, size: u64| read_offset(binary, pos, size);
        let mut glyphs = Vec::with_capacity(cid_count);
        let mut charstrings = Vec::new();
        for cid in 0..cid_count as u64 {
            let entry = cidmap_offset + cid * entry_len;
            let glyph = (|| {
                let dict = read(entry, fd_bytes)?;
                let start = read(entry + fd_bytes, gd_bytes)?;
                let end = read(entry + entry_len + fd_bytes, gd_bytes)?;
                let len_iv = *len_ivs.get(usize::try_from(dict).ok()?)?;
                if start >= end || end > len {
                    return None;
                }
                let encoded = binary.get(start as usize..end as usize)?;
                let data_start = charstrings.len();
                if len_iv >= 0 {
                    if len_iv as usize > encoded.len() {
                        return None;
                    }
                    charstrings.extend(
                        decrypt(encoded.iter().copied(), CHARSTRING_SEED).skip(len_iv as usize),
                    );
                } else {
                    charstrings.extend_from_slice(encoded);
                }
                Some(GlyphEntry {
                    dict: dict as u32,
                    data: data_start..charstrings.len(),
                })
            })();
            glyphs.push(glyph);
        }
        charstrings.shrink_to_fit();
        Some(Self {
            name,
            upem,
            dicts,
            glyphs,
            charstrings,
        })
    }

    /// Returns the CIDFont name (`CIDFontName`).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the number of CIDs (`CIDCount`); glyph identifiers range over
    /// `0..num_glyphs()`.
    pub fn num_glyphs(&self) -> u32 {
        self.glyphs.len() as u32
    }

    /// Returns the number of font dictionaries in the `FDArray`.
    pub fn num_font_dicts(&self) -> u32 {
        self.dicts.len() as u32
    }

    /// Returns the units per em: 1000 unless a font dictionary's
    /// `FontMatrix` implies another value.
    pub fn upem(&self) -> i32 {
        self.upem
    }

    /// Returns the index of the font dictionary that holds the given CID's
    /// charstring, or `None` if the CID has no glyph.
    pub fn font_dict_index(&self, cid: GlyphId) -> Option<u32> {
        Some(self.glyph(cid)?.dict)
    }

    /// Returns the transform for the given CID's font dictionary at an
    /// optional size in ppem.
    pub fn transform(&self, cid: GlyphId, ppem: Option<f32>) -> Option<Transform> {
        let dict = self.dicts.get(self.font_dict_index(cid)? as usize)?;
        Some(Transform {
            matrix: dict.matrix,
            scale: ppem.map(|ppem| Transform::compute_scale(ppem, self.upem)),
        })
    }

    /// Evaluates the charstring for the requested CID and sends the results
    /// to the given sink.
    ///
    /// Returns the advance width of the glyph in font units if the
    /// charstring provides one.
    pub fn evaluate_charstring(
        &self,
        cid: GlyphId,
        sink: &mut impl CommandSink,
    ) -> Result<Option<Fixed>, Error> {
        let glyph = self.glyph(cid).ok_or(ReadError::OutOfBounds)?;
        let context = FontDictContext {
            font: self,
            dict: self
                .dicts
                .get(glyph.dict as usize)
                .ok_or(ReadError::OutOfBounds)?,
        };
        let data = self
            .charstrings
            .get(glyph.data.clone())
            .ok_or(ReadError::OutOfBounds)?;
        cs::evaluate(&context, None, data, sink)
    }

    /// Draws the glyph for the given CID with an optional size in ppem to
    /// the given pen.
    ///
    /// Returns the advance width of the glyph if the charstring provides
    /// one.
    pub fn draw(
        &self,
        cid: GlyphId,
        ppem: Option<f32>,
        pen: &mut impl OutlinePen,
    ) -> Result<Option<f32>, Error> {
        let transform = self.transform(cid, ppem).ok_or(ReadError::OutOfBounds)?;
        let mut nop_filter = NopFilterSink::new(pen);
        let mut transformer = TransformSink::new(&mut nop_filter, transform);
        let width = self.evaluate_charstring(cid, &mut transformer)?;
        Ok(width.map(|w| transform.transform_h_metric(w).to_f32().max(0.0)))
    }

    fn glyph(&self, cid: GlyphId) -> Option<&GlyphEntry> {
        self.glyphs.get(cid.to_u32() as usize)?.as_ref()
    }

    fn charstring(&self, cid: u32) -> Option<&[u8]> {
        let glyph = self.glyphs.get(cid as usize)?.as_ref()?;
        self.charstrings.get(glyph.data.clone())
    }
}

/// Charstring context of one font dictionary: its subroutines, and the
/// font's charstrings for `seac`.
struct FontDictContext<'a> {
    font: &'a CidType1Font,
    dict: &'a FontDict,
}

impl CharstringContext for FontDictContext<'_> {
    fn kind(&self) -> CharstringKind {
        CharstringKind::Type1
    }

    /// In a CID-keyed font the `seac` codes are CIDs, not Standard
    /// Encoding codes. Both components run with the calling glyph's
    /// subroutines.
    ///
    /// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/psaux/t1decode.c#L230>
    fn seac_components(&self, base_code: i32, accent_code: i32) -> Result<[&[u8]; 2], Error> {
        let get = |code: i32| {
            u32::try_from(code)
                .ok()
                .and_then(|cid| self.font.charstring(cid))
                .ok_or(Error::InvalidSeacCode(code))
        };
        Ok([get(base_code)?, get(accent_code)?])
    }

    fn subr(&self, index: i32) -> Result<&[u8], Error> {
        Ok(self
            .dict
            .subrs
            .get(index as u32)
            .ok_or(ReadError::OutOfBounds)?)
    }

    fn global_subr(&self, _index: i32) -> Result<&[u8], Error> {
        Err(Error::MissingSubroutines)
    }
}

/// Locates the `StartData` operator that introduces the data section.
///
/// Returns the end of the cleartext header (the start of the `StartData`
/// token), the offset of the data section (one byte past the operator),
/// and, for a `(Hex)` data section, its length in bytes.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidparse.c#L219>
fn find_start_data(data: &[u8]) -> Option<(usize, usize, Option<usize>)> {
    let mut parser = Parser::new(data);
    let mut args = [None, None];
    loop {
        let token = parser.next()?;
        if token == Token::Raw(b"StartData") {
            let header_end = parser.pos - b"StartData".len();
            let hex_len = match args {
                [Some(Token::LitString(b"Hex")), Some(Token::Int(len))] => {
                    Some(usize::try_from(len).ok()?)
                }
                _ => None,
            };
            return Some((header_end, parser.pos + 1, hex_len));
        }
        args = [args[1], Some(token)];
    }
}

/// Reads the CIDFont and font dictionary keys of the cleartext header.
fn parse_header(header: &[u8]) -> Option<RawHeader> {
    let markers = find_all(header, FONT_DICT_MARKER);
    let mut next_marker = 0;
    let mut result = RawHeader::default();
    // Number of dictionaries declared by `FDArray`, once seen.
    let mut declared = None;
    // Index of the dictionary that keys apply to.
    let mut current: Option<usize> = None;
    let mut parser = Parser::new(header);
    loop {
        let before = parser.pos;
        let Some(token) = parser.next() else {
            break;
        };
        // A marker read past while skipping to this token opens the next
        // dictionary, once the FDArray has been declared.
        while next_marker < markers.len() && markers[next_marker] < parser.pos {
            if markers[next_marker] >= before {
                if let Some(count) = declared {
                    let index = current.map_or(0, |i| i + 1);
                    if index < count {
                        result.dicts.push(RawFontDict::default());
                        current = Some(index);
                    } else {
                        current = None;
                    }
                }
            }
            next_marker += 1;
        }
        let Token::Name(key) = token else {
            continue;
        };
        let dict = current.and_then(|i| result.dicts.get_mut(i));
        match (key, dict) {
            (b"CIDFontName", _) => {
                result.name = match parser.next()? {
                    Token::Name(name) | Token::LitString(name) => {
                        core::str::from_utf8(name).ok().map(|s| s.to_owned())
                    }
                    _ => None,
                }
            }
            (b"CIDMapOffset", _) => result.cidmap_offset = read_unsigned(&mut parser)?,
            (b"FDBytes", _) => result.fd_bytes = read_unsigned(&mut parser)?,
            (b"GDBytes", _) => result.gd_bytes = read_unsigned(&mut parser)?,
            (b"CIDCount", _) => result.cid_count = read_unsigned(&mut parser)?,
            (b"FDArray", _) => {
                if declared.is_none() {
                    // A font dictionary takes at least 100 bytes, which
                    // bounds the count by the header size.
                    // <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidload.c#L289>
                    let count = usize::try_from(read_unsigned(&mut parser)?).ok()?;
                    declared = Some(count.min(header.len() / 100));
                }
            }
            // An unreadable or degenerate matrix leaves the default.
            (b"FontMatrix", Some(dict)) => dict.matrix = parser.read_font_matrix(),
            (b"lenIV", Some(dict)) => dict.len_iv = Some(parser.read_int()?),
            (b"SubrMapOffset", Some(dict)) => dict.subrmap_offset = read_unsigned(&mut parser)?,
            (b"SDBytes", Some(dict)) => dict.sd_bytes = read_unsigned(&mut parser)?,
            (b"SubrCount", Some(dict)) => dict.subr_count = read_unsigned(&mut parser)?,
            _ => {}
        }
    }
    Some(result)
}

fn read_unsigned(parser: &mut Parser) -> Option<u64> {
    u64::try_from(parser.read_int()?).ok()
}

/// Reads the subroutines of one font dictionary from the data section.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidload.c#L420>
fn read_subrs(binary: &[u8], dict: &RawFontDict, len_iv: i64) -> Option<Subrs> {
    let mut subrs = Subrs {
        is_dense: true,
        ..Subrs::default()
    };
    let count = dict.subr_count;
    if count == 0 {
        return Some(subrs);
    }
    let len = binary.len() as u64;
    if dict.sd_bytes == 0
        || dict.sd_bytes > 4
        || dict.subrmap_offset > len
        || count.checked_mul(dict.sd_bytes)? > len - dict.subrmap_offset
    {
        return None;
    }
    let mut start = read_offset(binary, dict.subrmap_offset, dict.sd_bytes)?;
    for index in 0..count {
        let pos = dict.subrmap_offset + (index + 1) * dict.sd_bytes;
        let end = read_offset(binary, pos, dict.sd_bytes)?;
        if start > end || end > len {
            return None;
        }
        let encoded = binary.get(start as usize..end as usize)?;
        let data_start = subrs.data.len();
        if len_iv >= 0 {
            subrs
                .data
                .extend(decrypt(encoded.iter().copied(), CHARSTRING_SEED).skip(len_iv as usize));
        } else {
            subrs.data.extend_from_slice(encoded);
        }
        subrs
            .index
            .push((index as u32, data_start..subrs.data.len()));
        start = end;
    }
    subrs.data.shrink_to_fit();
    Some(subrs)
}

/// Reads a big-endian unsigned value of `size` (at most 4) bytes at `pos`.
fn read_offset(data: &[u8], pos: u64, size: u64) -> Option<u64> {
    let start = usize::try_from(pos).ok()?;
    let bytes = data.get(start..start.checked_add(size as usize)?)?;
    Some(
        bytes
            .iter()
            .fold(0u64, |value, &byte| (value << 8) | u64::from(byte)),
    )
}

fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(pos, window)| (window == needle).then_some(pos))
        .collect()
}

#[cfg(test)]
mod tests {
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
}
