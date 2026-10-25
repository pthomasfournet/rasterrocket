//! Name detection for fonts that require hinting to be run for correct
//! contours (FreeType calls these "tricky" fonts).

use crate::{string::StringId, FontRef, Tag};
use alloc::string::String;
use raw::{tables::name::Name, TableProvider};

pub(super) fn require_interpreter(font: &FontRef) -> bool {
    family_name(font).is_some_and(|name| matches_hint_reliant_name_list(&name))
        || matches_hint_reliant_id_list(font)
}

/// The family name FreeType compares against the tricky-font list.
///
/// The name IDs are tried in FreeType's order: WWS family, typographic
/// family, family; or, for a font whose `OS/2` `fsSelection` bit 8 marks it
/// WWS-only, typographic family then family.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/sfnt/sfobjs.c#L1045>
fn family_name(font: &FontRef) -> Option<String> {
    let name = font.name().ok()?;
    let wws_only = font
        .os2()
        .is_ok_and(|os2| os2.fs_selection().bits() & 0x100 != 0);
    let ids: &[StringId] = if wws_only {
        &[StringId::TYPOGRAPHIC_FAMILY_NAME, StringId::FAMILY_NAME]
    } else {
        &[
            StringId::WWS_FAMILY_NAME,
            StringId::TYPOGRAPHIC_FAMILY_NAME,
            StringId::FAMILY_NAME,
        ]
    };
    ids.iter().find_map(|id| ascii_name(&name, *id))
}

/// The ASCII form of name `id`, with the record chosen as FreeType's
/// `tt_face_get_name` chooses it: an English Windows record (encoding 0, 1
/// or 10) first; a Macintosh English or Roman record when the Windows record
/// is not English; a Unicode or ISO record last. Code units outside
/// 32..=127 become `?` and a NUL ends the string, so the `?` in list entries
/// such as `HuaTianKaiTi?` stands for a non-ASCII character.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/sfnt/sfobjs.c#L102>
fn ascii_name(name: &Name, id: StringId) -> Option<String> {
    let records = name.name_record();
    let mut found_unicode = None;
    let mut found_apple_english = None;
    let mut found_apple_roman = None;
    let mut found_win = None;
    let mut is_english = false;
    for (n, record) in records.iter().enumerate() {
        if record.name_id() != id || record.length() == 0 {
            continue;
        }
        match record.platform_id() {
            // Unicode, ISO
            0 | 2 => found_unicode = Some(n),
            // Macintosh
            1 => {
                if record.language_id() == 0 {
                    found_apple_english = Some(n);
                } else if record.encoding_id() == 0 {
                    found_apple_roman = Some(n);
                }
            }
            // Windows; a non-English record only when no English one exists
            3 => {
                let english = record.language_id() & 0x3FF == 0x009;
                if (found_win.is_none() || english) && matches!(record.encoding_id(), 0 | 1 | 10) {
                    is_english = english;
                    found_win = Some(n);
                }
            }
            _ => {}
        }
    }
    let found_apple = found_apple_english.or(found_apple_roman);
    let (index, utf16) = match (found_win, found_apple) {
        (Some(win), apple) if is_english || apple.is_none() => (win, true),
        (_, Some(apple)) => (apple, false),
        _ => (found_unicode?, true),
    };
    let record = records.get(index)?;
    let start = record.string_offset().to_u32() as usize;
    let end = start.checked_add(record.length() as usize)?;
    let bytes = name.string_data().as_bytes().get(start..end)?;
    let to_ascii = |code: u32| match code {
        32..=127 => code as u8 as char,
        _ => '?',
    };
    let ascii = if utf16 {
        bytes
            .chunks_exact(2)
            .map(|pair| u32::from(u16::from_be_bytes([pair[0], pair[1]])))
            .take_while(|&code| code != 0)
            .map(to_ascii)
            .collect()
    } else {
        bytes
            .iter()
            .map(|&byte| u32::from(byte))
            .take_while(|&code| code != 0)
            .map(to_ascii)
            .collect()
    };
    Some(ascii)
}

/// Is this name on the list of fonts that require hinting?
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L174>
fn matches_hint_reliant_name_list(name: &str) -> bool {
    let name = skip_pdf_random_tag(name);
    HINT_RELIANT_NAMES
        .iter()
        // FreeType uses strstr(name, tricky_name) so we use contains() to
        // match behavior.
        .any(|tricky_name| name.contains(*tricky_name))
}

/// Fonts embedded in PDFs add random prefixes. Strip these
/// for tricky font comparison purposes.
///
/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L153>
fn skip_pdf_random_tag(name: &str) -> &str {
    let bytes = name.as_bytes();
    // Random tag is 6 uppercase letters followed by a +
    if bytes.len() < 8 || bytes[6] != b'+' || !bytes.iter().take(6).all(|b| b.is_ascii_uppercase())
    {
        return name;
    }
    core::str::from_utf8(&bytes[7..]).unwrap_or(name)
}

/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L180>
#[rustfmt::skip]
const HINT_RELIANT_NAMES: &[&str] = &[
    "cpop",               /* dftt-p7.ttf; version 1.00, 1992 [DLJGyShoMedium] */
    "DFGirl-W6-WIN-BF",   /* dftt-h6.ttf; version 1.00, 1993 */
    "DFGothic-EB",        /* DynaLab Inc. 1992-1995 */
    "DFGyoSho-Lt",        /* DynaLab Inc. 1992-1995 */
    "DFHei",              /* DynaLab Inc. 1992-1995 [DFHei-Bd-WIN-HK-BF] */
                          /* covers "DFHei-Md-HK-BF", maybe DynaLab Inc. */

    "DFHSGothic-W5",      /* DynaLab Inc. 1992-1995 */
    "DFHSMincho-W3",      /* DynaLab Inc. 1992-1995 */
    "DFHSMincho-W7",      /* DynaLab Inc. 1992-1995 */
    "DFKaiSho-SB",        /* dfkaisb.ttf */
    "DFKaiShu",           /* covers "DFKaiShu-Md-HK-BF", maybe DynaLab Inc. */
    "DFKai-SB",           /* kaiu.ttf; version 3.00, 1998 [DFKaiShu-SB-Estd-BF] */

    "DFMing",             /* DynaLab Inc. 1992-1995 [DFMing-Md-WIN-HK-BF] */
                          /* covers "DFMing-Bd-HK-BF", maybe DynaLab Inc. */

    "DLC",                /* dftt-m7.ttf; version 1.00, 1993 [DLCMingBold] */
                          /* dftt-f5.ttf; version 1.00, 1993 [DLCFongSung] */
                          /* covers following */
                          /* "DLCHayMedium", dftt-b5.ttf; version 1.00, 1993 */
                          /* "DLCHayBold",   dftt-b7.ttf; version 1.00, 1993 */
                          /* "DLCKaiMedium", dftt-k5.ttf; version 1.00, 1992 */
                          /* "DLCLiShu",     dftt-l5.ttf; version 1.00, 1992 */
                          /* "DLCRoundBold", dftt-r7.ttf; version 1.00, 1993 */

    "HuaTianKaiTi?",      /* htkt2.ttf */
    "HuaTianSongTi?",     /* htst3.ttf */
    "Ming(for ISO10646)", /* hkscsiic.ttf; version 0.12, 2007 [Ming] */
                          /* iicore.ttf; version 0.07, 2007 [Ming] */
    "MingLiU",            /* mingliu.ttf */
                          /* mingliu.ttc; version 3.21, 2001 */
    "MingMedium",         /* dftt-m5.ttf; version 1.00, 1993 [DLCMingMedium] */
    "PMingLiU",           /* mingliu.ttc; version 3.21, 2001 */
    "MingLi43",           /* mingli.ttf; version 1.00, 1992 */
];

#[derive(Copy, Clone, PartialEq, Default, Debug)]
struct TableId {
    checksum: u32,
    len: u32,
}

#[derive(Copy, Clone, PartialEq, Default, Debug)]
struct FontId {
    cvt: TableId,
    fpgm: TableId,
    prep: TableId,
}

/// Checks for fonts that require hinting based on the length and checksum of
/// the cvt, fpgm and prep tables; an absent table has length and checksum
/// zero. As in FreeType, a table's checksum is computed only once its length
/// matches an entry, so a font whose table lengths match no entry costs no
/// table scan.
///
/// Roughly equivalent to <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L309>
fn matches_hint_reliant_id_list(font: &FontRef) -> bool {
    let tables = [b"cvt ", b"fpgm", b"prep"].map(|tag| font.table_data(Tag::new(tag)));
    let lens = tables
        .each_ref()
        .map(|data| data.as_ref().map_or(0, |data| data.len() as u32));
    let mut checksums: [Option<u32>; 3] = [None; 3];
    HINT_RELIANT_IDS.iter().any(|id| {
        let entry = [id.cvt, id.fpgm, id.prep];
        if entry.iter().zip(lens).any(|(table, len)| table.len != len) {
            return false;
        }
        entry.iter().enumerate().all(|(i, table)| {
            // Note: FreeType always just computes the checksum
            // <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L281>
            let checksum = *checksums[i].get_or_insert_with(|| {
                tables[i]
                    .as_ref()
                    .map_or(0, |data| raw::tables::compute_checksum(data.as_bytes()))
            });
            checksum == table.checksum
        })
    })
}

/// <https://gitlab.freedesktop.org/freetype/freetype/-/blob/80a507a6b8e3d2906ad2c8ba69329bd2fb2a85ef/src/truetype/ttobjs.c#L314>
#[rustfmt::skip]
const HINT_RELIANT_IDS: &[FontId] = &[
    // MingLiU 1995
    FontId {
        cvt: TableId { checksum: 0x05BCF058, len: 0x000002E4 },
        fpgm: TableId { checksum: 0x28233BF1, len: 0x000087C4 },
        prep: TableId { checksum: 0xA344A1EA, len: 0x000001E1 },
    },
    // MingLiU 1996-
    FontId {
        cvt: TableId { checksum: 0x05BCF058, len: 0x000002E4 },
        fpgm: TableId { checksum: 0x28233BF1, len: 0x000087C4 },
        prep: TableId { checksum: 0xA344A1EB, len: 0x000001E1 },
    },
    // DFGothic-EB
    FontId {
        cvt: TableId { checksum: 0x12C3EBB2, len: 0x00000350 },
        fpgm: TableId { checksum: 0xB680EE64, len: 0x000087A7 },
        prep: TableId { checksum: 0xCE939563, len: 0x00000758 },
    },
    // DFGyoSho-Lt
    FontId {
        cvt: TableId { checksum: 0x11E5EAD4, len: 0x00000350 },
        fpgm: TableId { checksum: 0xCE5956E9, len: 0x0000BC85 },
        prep: TableId { checksum: 0x8272F416, len: 0x00000045 },
    },
    // DFHei-Md-HK-BF
    FontId {
        cvt: TableId { checksum: 0x1257EB46, len: 0x00000350 },
        fpgm: TableId { checksum: 0xF699D160, len: 0x0000715F },
        prep: TableId { checksum: 0xD222F568, len: 0x000003BC },
    },
    // DFHSGothic-W5
    FontId {
        cvt: TableId { checksum: 0x1262EB4E, len: 0x00000350 },
        fpgm: TableId { checksum: 0xE86A5D64, len: 0x00007940 },
        prep: TableId { checksum: 0x7850F729, len: 0x000005FF },
    },
    // DFHSMincho-W3
    FontId {
        cvt: TableId { checksum: 0x122DEB0A, len: 0x00000350 },
        fpgm: TableId { checksum: 0x3D16328A, len: 0x0000859B },
        prep: TableId { checksum: 0xA93FC33B, len: 0x000002CB },
    },
    // DFHSMincho-W7
    FontId {
        cvt: TableId { checksum: 0x125FEB26, len: 0x00000350 },
        fpgm: TableId { checksum: 0xA5ACC982, len: 0x00007EE1 },
        prep: TableId { checksum: 0x90999196, len: 0x0000041F },
    },
    // DFKaiShu
    FontId {
        cvt: TableId { checksum: 0x11E5EAD4, len: 0x00000350 },
        fpgm: TableId { checksum: 0x5A30CA3B, len: 0x00009063 },
        prep: TableId { checksum: 0x13A42602, len: 0x0000007E },
    },
    // DFKaiShu, variant
    FontId {
        cvt: TableId { checksum: 0x11E5EAD4, len: 0x00000350 },
        fpgm: TableId { checksum: 0xA6E78C01, len: 0x00008998 },
        prep: TableId { checksum: 0x13A42602, len: 0x0000007E },
    },
    // DFKaiShu-Md-HK-BF
    FontId {
        cvt: TableId { checksum: 0x11E5EAD4, len: 0x00000360 },
        fpgm: TableId { checksum: 0x9DB282B2, len: 0x0000C06E },
        prep: TableId { checksum: 0x53E6D7CA, len: 0x00000082 },
    },
    // DFMing-Bd-HK-BF
    FontId {
        cvt: TableId { checksum: 0x1243EB18, len: 0x00000350 },
        fpgm: TableId { checksum: 0xBA0A8C30, len: 0x000074AD },
        prep: TableId { checksum: 0xF3D83409, len: 0x0000037B },
    },
    // DLCLiShu
    FontId {
        cvt: TableId { checksum: 0x07DCF546, len: 0x00000308 },
        fpgm: TableId { checksum: 0x40FE7C90, len: 0x00008E2A },
        prep: TableId { checksum: 0x608174B5, len: 0x0000007A },
    },
    // DLCHayBold
    FontId {
        cvt: TableId { checksum: 0xEB891238, len: 0x00000308 },
        fpgm: TableId { checksum: 0xD2E4DCD4, len: 0x0000676F },
        prep: TableId { checksum: 0x8EA5F293, len: 0x000003B8 },
    },
    // HuaTianKaiTi
    FontId {
        cvt: TableId { checksum: 0xFFFBFFFC, len: 0x00000008 },
        fpgm: TableId { checksum: 0x9C9E48B8, len: 0x0000BEA2 },
        prep: TableId { checksum: 0x70020112, len: 0x00000008 },
    },
    // HuaTianSongTi
    FontId {
        cvt: TableId { checksum: 0xFFFBFFFC, len: 0x00000008 },
        fpgm: TableId { checksum: 0x0A5A0483, len: 0x00017C39 },
        prep: TableId { checksum: 0x70020112, len: 0x00000008 },
    },
    // NEC fadpop7.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x40C92555, len: 0x000000E5 },
        prep: TableId { checksum: 0xA39B58E3, len: 0x0000117C },
    },
    // NEC fadrei5.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x33C41652, len: 0x000000E5 },
        prep: TableId { checksum: 0x26D6C52A, len: 0x00000F6A },
    },
    // NEC fangot7.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x6DB1651D, len: 0x0000019D },
        prep: TableId { checksum: 0x6C6E4B03, len: 0x00002492 },
    },
    // NEC fangyo5.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x40C92555, len: 0x000000E5 },
        prep: TableId { checksum: 0xDE51FAD0, len: 0x0000117C },
    },
    // NEC fankyo5.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x85E47664, len: 0x000000E5 },
        prep: TableId { checksum: 0xA6C62831, len: 0x00001CAA },
    },
    // NEC fanrgo5.ttf
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x2D891CFD, len: 0x0000019D },
        prep: TableId { checksum: 0xA0604633, len: 0x00001DE8 },
    },
    // NEC fangot5.ttc
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x40AA774C, len: 0x000001CB },
        prep: TableId { checksum: 0x9B5CAA96, len: 0x00001F9A },
    },
    // NEC fanmin3.ttc
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x0D3DE9CB, len: 0x00000141 },
        prep: TableId { checksum: 0xD4127766, len: 0x00002280 },
    },
    // NEC FA-Gothic, 1996
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x4A692698, len: 0x000001F0 },
        prep: TableId { checksum: 0x340D4346, len: 0x00001FCA },
    },
    // NEC FA-Minchou, 1996
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0xCD34C604, len: 0x00000166 },
        prep: TableId { checksum: 0x6CF31046, len: 0x000022B0 },
    },
    // NEC FA-RoundGothicB, 1996
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0x5DA75315, len: 0x0000019D },
        prep: TableId { checksum: 0x40745A5F, len: 0x000022E0 },
    },
    // NEC FA-RoundGothicM, 1996
    FontId {
        cvt: TableId { checksum: 0x00000000, len: 0x00000000 },
        fpgm: TableId { checksum: 0xF055FC48, len: 0x000001C2 },
        prep: TableId { checksum: 0x3900DED3, len: 0x00001E18 },
    },
    // MINGLI.TTF, 1992
    FontId {
        cvt: TableId { checksum: 0x00170003, len: 0x00000060 },
        fpgm: TableId { checksum: 0xDBB4306E, len: 0x000058AA },
        prep: TableId { checksum: 0xD643482A, len: 0x00000035 },
    },
    // DFHei-Bd-WIN-HK-BF, issue #1087
    FontId {
        cvt: TableId { checksum: 0x1269EB58, len: 0x00000350 },
        fpgm: TableId { checksum: 0x5CD5957A, len: 0x00006A4E },
        prep: TableId { checksum: 0xF758323A, len: 0x00000380 },
    },
    // DFMing-Md-WIN-HK-BF, issue #1087
    FontId {
        cvt: TableId { checksum: 0x122FEB0B, len: 0x00000350 },
        fpgm: TableId { checksum: 0x7F10919A, len: 0x000070A9 },
        prep: TableId { checksum: 0x7CD7E7B7, len: 0x0000025C },
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// An sfnt holding only `tables`, with a zeroed directory checksum.
    fn sfnt(tables: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut tables = tables.to_vec();
        tables.sort_by_key(|(tag, _)| **tag);
        let mut out = Vec::new();
        out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        out.extend_from_slice(&(tables.len() as u16).to_be_bytes());
        out.extend_from_slice(&[0; 6]);
        let mut offset = 12 + 16 * tables.len();
        for (tag, data) in &tables {
            out.extend_from_slice(*tag);
            out.extend_from_slice(&0u32.to_be_bytes());
            out.extend_from_slice(&(offset as u32).to_be_bytes());
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += data.len().next_multiple_of(4);
        }
        for (_, data) in &tables {
            out.extend_from_slice(data);
            out.resize(out.len().next_multiple_of(4), 0);
        }
        out
    }

    /// A `name` table of `(platform, encoding, language, name id, bytes)`
    /// records.
    fn name_table(records: &[(u16, u16, u16, u16, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0u16.to_be_bytes());
        out.extend_from_slice(&(records.len() as u16).to_be_bytes());
        out.extend_from_slice(&(6 + 12 * records.len() as u16).to_be_bytes());
        let mut offset = 0u16;
        for (platform, encoding, language, id, bytes) in records {
            for field in [
                *platform,
                *encoding,
                *language,
                *id,
                bytes.len() as u16,
                offset,
            ] {
                out.extend_from_slice(&field.to_be_bytes());
            }
            offset += bytes.len() as u16;
        }
        for (.., bytes) in records {
            out.extend_from_slice(bytes);
        }
        out
    }

    fn utf16(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(u16::to_be_bytes).collect()
    }

    /// A Windows English Unicode record.
    fn win(id: u16, s: &str) -> (u16, u16, u16, u16, Vec<u8>) {
        (3, 1, 0x409, id, utf16(s))
    }

    fn requires(data: &[u8]) -> bool {
        require_interpreter(&FontRef::new(data).unwrap())
    }

    /// Table data of `len` bytes whose checksum is `checksum`.
    fn forged_table(len: usize, checksum: u32) -> Vec<u8> {
        let mut data = vec![0u8; len];
        data[..4].copy_from_slice(&checksum.to_be_bytes());
        data
    }

    #[test]
    fn family_name_longer_than_any_list_entry_matches() {
        let font = sfnt(&[(b"name", name_table(&[win(1, "ABCDEF+DFKaiShu-SB-Estd-BF")]))]);
        assert!(requires(&font));
    }

    #[test]
    fn non_ascii_name_characters_read_as_question_marks() {
        // U+6977 (楷) reads as `?`, matching the `HuaTianKaiTi?` entry.
        let font = sfnt(&[(b"name", name_table(&[win(1, "HuaTianKaiTi\u{6977}")]))]);
        assert!(requires(&font));
        let name = FontRef::new(&font).unwrap().name().unwrap();
        assert_eq!(
            ascii_name(&name, StringId::FAMILY_NAME).as_deref(),
            Some("HuaTianKaiTi?")
        );
    }

    #[test]
    fn typographic_family_is_preferred_over_family() {
        let tricky_typographic =
            sfnt(&[(b"name", name_table(&[win(1, "Plain"), win(16, "PMingLiU")]))]);
        assert!(requires(&tricky_typographic));
        let tricky_family_only =
            sfnt(&[(b"name", name_table(&[win(1, "PMingLiU"), win(16, "Plain")]))]);
        assert!(!requires(&tricky_family_only));
    }

    #[test]
    fn english_windows_record_is_preferred() {
        let font = sfnt(&[(
            b"name",
            name_table(&[
                (1, 0, 0, 1, b"Plain".to_vec()),
                (3, 1, 0x404, 1, utf16("Plain")),
                win(1, "MingLiU"),
            ]),
        )]);
        assert!(requires(&font));
    }

    #[test]
    fn macintosh_record_is_used_when_windows_record_is_not_english() {
        let font = sfnt(&[(
            b"name",
            name_table(&[
                (3, 1, 0x404, 1, utf16("Plain")),
                (1, 0, 0, 1, b"DFKai-SB".to_vec()),
            ]),
        )]);
        assert!(requires(&font));
    }

    #[test]
    fn table_checksums_identify_a_renamed_font() {
        // NEC fadpop7.ttf: no `cvt `, `fpgm` and `prep` of known length and
        // checksum; the family name is not on the list.
        let fpgm = forged_table(0xE5, 0x40C92555);
        let prep = forged_table(0x117C, 0xA39B58E3);
        let name = name_table(&[win(1, "Renamed")]);
        let font = sfnt(&[
            (b"fpgm", fpgm.clone()),
            (b"prep", prep.clone()),
            (b"name", name.clone()),
        ]);
        assert!(requires(&font));
        // A different checksum, an extra `cvt ` table, or a missing table
        // each break the match.
        let font = sfnt(&[(b"fpgm", fpgm.clone()), (b"prep", forged_table(0x117C, 1))]);
        assert!(!requires(&font));
        let font = sfnt(&[
            (b"cvt ", vec![0; 4]),
            (b"fpgm", fpgm.clone()),
            (b"prep", prep),
        ]);
        assert!(!requires(&font));
        let font = sfnt(&[(b"fpgm", fpgm)]);
        assert!(!requires(&font));
    }

    #[test]
    fn font_without_name_or_matching_tables_is_not_hint_reliant() {
        let font = sfnt(&[(b"name", name_table(&[win(1, "Roboto")]))]);
        assert!(!requires(&font));
        assert!(!requires(&sfnt(&[])));
    }

    #[test]
    fn skip_pdf_tags() {
        // length must be at least 8
        assert_eq!(skip_pdf_random_tag("ABCDEF+"), "ABCDEF+");
        // first six chars must be ascii uppercase
        assert_eq!(skip_pdf_random_tag("AbCdEF+Arial"), "AbCdEF+Arial");
        // no numbers
        assert_eq!(skip_pdf_random_tag("Ab12EF+Arial"), "Ab12EF+Arial");
        // missing +
        assert_eq!(skip_pdf_random_tag("ABCDEFArial"), "ABCDEFArial");
        // too long
        assert_eq!(skip_pdf_random_tag("ABCDEFG+Arial"), "ABCDEFG+Arial");
        // too short
        assert_eq!(skip_pdf_random_tag("ABCDE+Arial"), "ABCDE+Arial");
        // just right
        assert_eq!(skip_pdf_random_tag("ABCDEF+Arial"), "Arial");
    }

    #[test]
    fn all_hint_reliant_names() {
        for name in HINT_RELIANT_NAMES {
            assert!(matches_hint_reliant_name_list(name));
        }
    }

    #[test]
    fn non_hint_reliant_names() {
        for not_tricky in ["Roboto", "Arial", "Helvetica", "Blah", ""] {
            assert!(!matches_hint_reliant_name_list(not_tricky));
        }
    }
}
