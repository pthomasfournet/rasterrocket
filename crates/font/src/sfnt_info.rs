//! What an sfnt file (TrueType, OpenType, collection) says about its faces.
//!
//! The family, weight, slant and code pages of each face are read from its
//! `OS/2` and `name` tables alone, and the Unicode coverage of a face from
//! its `cmap` table alone, without loading the file.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use read_fonts::tables::cmap::{Cmap, CmapIterLimits, CmapSubtable};
use read_fonts::tables::name::Name;
use read_fonts::tables::os2::Os2;
use read_fonts::types::NameId;
use read_fonts::{FontData, FontRead};

use crate::program::unicode_subtable_index;

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
/// Largest `cmap` table read; the `cmap` of a CJK face with variation
/// sequences spans several megabytes.
const MAX_CMAP_LEN: u32 = 16 << 20;
/// Largest collection walked.
const MAX_FACES: u32 = 64;

/// The faces of the sfnt file at `path`; empty when the file is not an
/// sfnt or cannot be read.
#[must_use]
pub fn face_infos(path: &Path) -> Vec<FaceInfo> {
    let Ok(mut file) = File::open(path) else {
        return Vec::new();
    };
    face_offsets(&mut file)
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

/// A set of Unicode scalar values, held as sorted, disjoint, non-adjacent
/// inclusive ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Coverage {
    ranges: Vec<(u32, u32)>,
}

impl Coverage {
    /// The set of the values `points` yields, in any order.
    pub fn from_points(points: impl IntoIterator<Item = u32>) -> Self {
        let mut points: Vec<u32> = points.into_iter().collect();
        points.sort_unstable();
        points.dedup();
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for c in points {
            match ranges.last_mut() {
                Some((_, end)) if *end + 1 == c => *end = c,
                _ => ranges.push((c, c)),
            }
        }
        Self { ranges }
    }

    /// `true` when `c` is in the set.
    #[must_use]
    pub fn contains(&self, c: u32) -> bool {
        let at = self.ranges.partition_point(|&(_, end)| end < c);
        self.ranges.get(at).is_some_and(|&(start, _)| start <= c)
    }

    /// `true` when the set is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

/// The Unicode coverage of face `index` of the sfnt file at `path`.
///
/// The coverage is the set of code points the face's Unicode `cmap`
/// subtable (the one a [`crate::program::FontProgram`] of the face resolves
/// Unicode values through) maps to a glyph other than `.notdef`, read from
/// the table directory and the `cmap` table alone.  `None` when the face has no
/// outline table (`glyf`, `CFF ` or `CFF2`), no Unicode subtable, or cannot
/// be read.
#[must_use]
pub fn unicode_coverage(path: &Path, index: u32) -> Option<Coverage> {
    let mut file = File::open(path).ok()?;
    let offset = *face_offsets(&mut file).get(usize::try_from(index).ok()?)?;
    let records = table_records(&mut file, offset)?;
    let locate = |tag: [u8; 4]| locate_table(&records, tag);
    if locate(*b"glyf").is_none() && locate(*b"CFF ").is_none() && locate(*b"CFF2").is_none() {
        return None;
    }
    let (at, len) = locate(*b"cmap")?;
    let data = read_table(&mut file, at, len, MAX_CMAP_LEN)?;
    let cmap = Cmap::read(FontData::new(&data)).ok()?;
    let limits = CmapIterLimits {
        max_char: 0x10_FFFF,
        glyph_count: u32::from(u16::MAX) + 1,
    };
    let record = cmap
        .encoding_records()
        .get(unicode_subtable_index(&cmap)?)?;
    let subtable = record.subtable(cmap.offset_data()).ok()?;
    let mapped = |c: &u32| {
        subtable
            .map_codepoint(*c)
            .is_some_and(|gid| gid.to_u32() != 0)
    };
    let points: Vec<u32> = if let CmapSubtable::Format0(_) = subtable {
        // The format 0 byte table has no pair iterator; its 256 codes are
        // probed directly.
        (0..=0xFF).filter(mapped).collect()
    } else {
        subtable
            .iter_with_limits(limits)
            .filter(|(_, gid)| gid.to_u32() != 0)
            .map(|(c, _)| c)
            .collect()
    };
    Some(Coverage::from_points(points))
}

/// Offsets of the table directories of the faces in `file`: one for a
/// single-face file, one per member (at most [`MAX_FACES`]) for a
/// collection; empty when the file is not an sfnt.
fn face_offsets(file: &mut File) -> Vec<u32> {
    let mut header = [0u8; 12];
    if read_at(file, 0, &mut header).is_none() {
        return Vec::new();
    }
    if &header[..4] == b"ttcf" {
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
        Vec::new()
    }
}

/// The 16-byte table records of the face whose directory starts at
/// `offset`.
fn table_records(file: &mut File, offset: u32) -> Option<Vec<u8>> {
    let mut dir = [0u8; 12];
    read_at(file, u64::from(offset), &mut dir)?;
    let num_tables = u16::from_be_bytes([dir[4], dir[5]]);
    let mut records = vec![0u8; 16 * usize::from(num_tables)];
    read_at(file, u64::from(offset) + 12, &mut records)?;
    Some(records)
}

/// Offset and length of the table `tag` among the table `records`.
fn locate_table(records: &[u8], tag: [u8; 4]) -> Option<(u32, u32)> {
    records.as_chunks::<16>().0.iter().find_map(|r| {
        (r[..4] == tag).then(|| {
            let at = u32::from_be_bytes([r[8], r[9], r[10], r[11]]);
            let len = u32::from_be_bytes([r[12], r[13], r[14], r[15]]);
            (at, len)
        })
    })
}

/// The face whose table directory starts at `offset`.
fn face_info(file: &mut File, offset: u32) -> Option<FaceInfo> {
    let records = table_records(file, offset)?;
    let (os2_at, os2_len) = locate_table(&records, *b"OS/2")?;
    let (name_at, name_len) = locate_table(&records, *b"name")?;
    let os2 = read_table(file, os2_at, os2_len, MAX_TABLE_LEN)?;
    let name = read_table(file, name_at, name_len, MAX_TABLE_LEN)?;
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

/// Read `len` bytes at `at` into a new buffer; `None` for an empty table
/// or one longer than `max`.
fn read_table(file: &mut File, at: u32, len: u32, max: u32) -> Option<Vec<u8>> {
    if len == 0 || len > max {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_merges_adjacent_points_into_ranges() {
        let set = Coverage::from_points([5, 3, 4, 10, 4, 12, 11]);
        assert_eq!(set.ranges, vec![(3, 5), (10, 12)]);
        for c in [3, 4, 5, 10, 11, 12] {
            assert!(set.contains(c), "{c}");
        }
        for c in [0, 2, 6, 9, 13, u32::MAX] {
            assert!(!set.contains(c), "{c}");
        }
        assert!(Coverage::default().is_empty());
    }

    /// The coverage of a test face is the set of code points its Unicode
    /// subtable maps, `.notdef` excluded.
    #[test]
    fn unicode_coverage_reads_the_cmap() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../font_test_data/test_data/ttf/ahem.ttf"
        ));
        let set = unicode_coverage(path, 0).expect("ahem has a Unicode cmap");
        assert!(set.contains(u32::from('A')));
        assert!(set.contains(u32::from('$')));
        assert!(!set.contains(0x4E2D));
        assert!(unicode_coverage(path, 1).is_none(), "no second face");
    }
}
