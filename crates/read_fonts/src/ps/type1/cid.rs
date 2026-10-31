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
use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec::Vec};
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
    /// Distinct subroutine sets; font dictionaries that locate theirs with
    /// the same map and `lenIV` share one.
    subrs: Vec<Subrs>,
    /// Charstring of each CID, `None` for a CID without a glyph.
    glyphs: Vec<Option<GlyphEntry>>,
    /// Decrypted charstrings, with the `lenIV` prefix removed.
    charstrings: Vec<u8>,
}

struct FontDict {
    matrix: FontMatrix,
    /// Index into `CidType1Font::subrs`.
    subrs: usize,
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
        // Every subroutine and charstring is decoded from its own bytes of
        // the data section, so the decoded total of a well-formed font is
        // at most the section's length. Font dictionaries or CIDs that name
        // the same bytes share one decoded copy; a font whose distinct
        // references still exceed the bound (overlapping ranges, which
        // would otherwise cost memory quadratic in the input) is rejected
        // when its subroutines exceed it and loses the glyphs that do.
        let mut budget = binary.len();
        let mut upem = 1000;
        let mut dicts = Vec::with_capacity(raw_dicts.len());
        let mut len_ivs = Vec::with_capacity(raw_dicts.len());
        let mut subrs = Vec::new();
        let mut subrs_by_map = BTreeMap::new();
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
            let key = (raw.subrmap_offset, raw.sd_bytes, raw.subr_count, len_iv);
            let subrs_index = match subrs_by_map.get(&key) {
                Some(&index) => index,
                None => {
                    subrs.push(read_subrs(binary, raw, len_iv, &mut budget)?);
                    subrs_by_map.insert(key, subrs.len() - 1);
                    subrs.len() - 1
                }
            };
            dicts.push(FontDict {
                matrix,
                subrs: subrs_index,
            });
            len_ivs.push(len_iv);
        }
        let cid_count = usize::try_from(cid_count).ok()?;
        let read = |pos: u64, size: u64| read_offset(binary, pos, size);
        let mut glyphs = Vec::with_capacity(cid_count);
        let mut charstrings = Vec::new();
        let mut decoded: BTreeMap<(u64, u64, i64), Range<usize>> = BTreeMap::new();
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
                if let Some(data) = decoded.get(&(start, end, len_iv)) {
                    return Some(GlyphEntry {
                        dict: dict as u32,
                        data: data.clone(),
                    });
                }
                let encoded = binary.get(start as usize..end as usize)?;
                if len_iv >= 0 && len_iv as usize > encoded.len() {
                    return None;
                }
                budget = budget.checked_sub(encoded.len())?;
                let data_start = charstrings.len();
                if len_iv >= 0 {
                    charstrings.extend(
                        decrypt(encoded.iter().copied(), CHARSTRING_SEED).skip(len_iv as usize),
                    );
                } else {
                    charstrings.extend_from_slice(encoded);
                }
                let data = data_start..charstrings.len();
                decoded.insert((start, end, len_iv), data.clone());
                Some(GlyphEntry {
                    dict: dict as u32,
                    data,
                })
            })();
            glyphs.push(glyph);
        }
        charstrings.shrink_to_fit();
        Some(Self {
            name,
            upem,
            dicts,
            subrs,
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
            .font
            .subrs
            .get(self.dict.subrs)
            .and_then(|subrs| subrs.get(index as u32))
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

/// Reads the subroutines of one font dictionary from the data section,
/// deducting their encoded length from `budget`; `None` if they exceed it.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/cid/cidload.c#L420>
fn read_subrs(binary: &[u8], dict: &RawFontDict, len_iv: i64, budget: &mut usize) -> Option<Subrs> {
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
        *budget = budget.checked_sub(encoded.len())?;
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
#[path = "../../tests/test_cid_type1.rs"]
mod tests;
