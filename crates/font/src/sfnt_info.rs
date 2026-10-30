//! Summary of the faces in an sfnt file (TrueType, OpenType, collection)
//! read from its `OS/2` and `name` tables alone, without loading the file.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use read_fonts::tables::name::Name;
use read_fonts::tables::os2::Os2;
use read_fonts::types::NameId;
use read_fonts::{FontData, FontRead};

/// What the `OS/2` and `name` tables say about one face.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceInfo {
    /// Index of the face in its file (0 outside a collection).
    pub index: u32,
    /// Family name (`name` ID 16, else ID 1), as the file spells it.
    pub family: String,
    /// `OS/2` `ulCodePageRange1` and `ulCodePageRange2`; zero when the
    /// table predates them.
    pub code_page_ranges: [u32; 2],
    /// `OS/2` `usWeightClass` (400 regular, 700 bold).
    pub weight_class: u16,
    /// `OS/2` `fsSelection` italic bit.
    pub italic: bool,
    /// `OS/2` `sFamilyClass`: class in the high byte (1–7 serif designs,
    /// 8 sans serif), subclass in the low byte.
    pub family_class: i16,
}

/// Largest `OS/2` or `name` table read.
const MAX_TABLE_LEN: u32 = 1 << 20;
/// Largest collection walked.
const MAX_FACES: u32 = 64;

/// The faces of the sfnt file at `path`; empty when the file is not an
/// sfnt or cannot be read.
#[must_use]
pub fn face_infos(path: &Path) -> Vec<FaceInfo> {
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    let mut header = [0u8; 12];
    if file.read_exact(&mut header).is_err() {
        return Vec::new();
    }
    let offsets: Vec<u32> = if &header[..4] == b"ttcf" {
        let count =
            u32::from_be_bytes([header[8], header[9], header[10], header[11]]).min(MAX_FACES);
        let mut table = vec![0u8; 4 * count as usize];
        if file.read_exact(&mut table).is_err() {
            return Vec::new();
        }
        table
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_be_bytes(*b))
            .collect()
    } else if matches!(&header[..4], [0, 1, 0, 0] | b"true" | b"OTTO") {
        vec![0]
    } else {
        return Vec::new();
    };
    offsets
        .into_iter()
        .enumerate()
        .filter_map(|(index, offset)| {
            face_info(&mut file, offset).map(|mut info| {
                info.index = u32::try_from(index).unwrap_or(u32::MAX);
                info
            })
        })
        .collect()
}

/// The face whose table directory starts at `offset`.
fn face_info(file: &mut File, offset: u32) -> Option<FaceInfo> {
    let mut dir = [0u8; 12];
    read_at(file, u64::from(offset), &mut dir)?;
    let num_tables = u16::from_be_bytes([dir[4], dir[5]]);
    let mut records = vec![0u8; 16 * usize::from(num_tables)];
    read_at(file, u64::from(offset) + 12, &mut records)?;
    let locate = |tag: &[u8; 4]| {
        records.as_chunks::<16>().0.iter().find_map(|r| {
            (&r[..4] == tag).then(|| {
                let at = u32::from_be_bytes([r[8], r[9], r[10], r[11]]);
                let len = u32::from_be_bytes([r[12], r[13], r[14], r[15]]);
                (at, len)
            })
        })
    };
    let (os2_at, os2_len) = locate(b"OS/2")?;
    let (name_at, name_len) = locate(b"name")?;
    let os2 = read_table(file, os2_at, os2_len)?;
    let name = read_table(file, name_at, name_len)?;
    let os2 = Os2::read(FontData::new(&os2)).ok()?;
    let name = Name::read(FontData::new(&name)).ok()?;
    Some(FaceInfo {
        index: 0,
        family: family_name(&name)?,
        code_page_ranges: [
            os2.ul_code_page_range_1().unwrap_or(0),
            os2.ul_code_page_range_2().unwrap_or(0),
        ],
        weight_class: os2.us_weight_class(),
        italic: os2
            .fs_selection()
            .contains(read_fonts::tables::os2::SelectionFlags::ITALIC),
        family_class: os2.s_family_class(),
    })
}

/// The typographic family name (ID 16) of `name`, else its family name
/// (ID 1), from a Unicode record.
fn family_name(name: &Name<'_>) -> Option<String> {
    let data = name.string_data();
    let record_named = |id: u16| {
        name.name_record()
            .iter()
            .filter(|r| r.name_id() == NameId::new(id) && r.is_unicode())
            .find_map(|r| r.string(data).ok())
            .map(|s| s.chars().collect::<String>())
            .filter(|s| !s.is_empty())
    };
    record_named(16).or_else(|| record_named(1))
}

/// Read `len` bytes at `at` into a new buffer, within [`MAX_TABLE_LEN`].
fn read_table(file: &mut File, at: u32, len: u32) -> Option<Vec<u8>> {
    if len == 0 || len > MAX_TABLE_LEN {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    read_at(file, u64::from(at), &mut buf)?;
    Some(buf)
}

fn read_at(file: &mut File, at: u64, buf: &mut [u8]) -> Option<()> {
    file.seek(SeekFrom::Start(at)).map(drop).ok()?;
    file.read_exact(buf).ok()
}
