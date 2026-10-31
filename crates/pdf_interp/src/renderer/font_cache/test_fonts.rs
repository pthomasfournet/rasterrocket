//! Font descriptors and faces shared by the font cache's tests.

use std::sync::Arc;

use font::program::FontProgram;

use super::{ActiveFont, FontCache, PageFace};
use crate::resources::ResourceKey;
use crate::resources::cmap::CMap;
use crate::resources::font::{BaseEncoding, CidEncoding, CidMetrics, FontDescriptor, PdfFontKind};
use crate::resources::font_substitute::SubstituteStyle;

/// The glyph-name table of a font without `Differences`.
pub fn empty_diffs() -> Box<[Option<Box<str>>; 256]> {
    Box::new(std::array::from_fn(|_| None))
}

/// A `Differences` table assigning each `(code, name)` of `names`.
pub fn diffs(names: &[(usize, &str)]) -> Box<[Option<Box<str>>; 256]> {
    let mut table = empty_diffs();
    for &(code, name) in names {
        table[code] = Some(name.into());
    }
    table
}

/// `ahem.ttf`: a TrueType program with a `cmap` and `post` glyph names.
pub fn ahem() -> Vec<u8> {
    include_bytes!("../../../../font_test_data/test_data/ttf/ahem.ttf").to_vec()
}

/// [`ahem`] with its `cmap` table hidden: the table-directory tag is
/// renamed, so the program keeps its `post` names and no code-keyed
/// subtable.
pub fn ahem_without_cmap() -> Vec<u8> {
    let mut data = ahem();
    let tables = usize::from(u16::from_be_bytes([data[4], data[5]]));
    let at = (0..tables)
        .map(|i| 12 + 16 * i)
        .find(|&at| &data[at..at + 4] == b"cmap")
        .expect("ahem has a cmap table");
    data[at..at + 4].copy_from_slice(b"xmap");
    data
}

/// A simple font of `kind` embedding `program` (none: drawn by a
/// substitute), under `differences` and `base_encoding`.
pub fn simple(
    kind: PdfFontKind,
    program: Option<Vec<u8>>,
    differences: Box<[Option<Box<str>>; 256]>,
    base_encoding: BaseEncoding,
) -> FontDescriptor {
    FontDescriptor {
        kind,
        program: program
            .map(|bytes| Arc::new(FontProgram::parse(bytes, 0).expect("test program parses"))),
        widths: vec![],
        first_char: 0,
        missing_width: 0,
        differences,
        base_encoding,
        cid_encoding: None,
        type3: None,
        substitute: SubstituteStyle::default(),
    }
}

/// An `Identity-H` composite font without a program, of the substitute
/// class `substitute`, with the `ToUnicode` map `to_unicode`.
pub fn composite_substitute(
    to_unicode: Option<CMap>,
    substitute: SubstituteStyle,
) -> FontDescriptor {
    FontDescriptor {
        kind: PdfFontKind::TrueType,
        program: None,
        widths: vec![],
        first_char: 0,
        missing_width: 0,
        differences: empty_diffs(),
        base_encoding: BaseEncoding::Standard,
        cid_encoding: Some(CidEncoding {
            encoding_cmap: Arc::new(CMap::identity()),
            cid_to_gid: None,
            default_width: 1000,
            widths: CidMetrics::new(Vec::new()),
            vertical: false,
            default_vertical: [880, -1000],
            vertical_metrics: CidMetrics::new(Vec::new()),
            to_unicode: to_unicode.map(Arc::new),
        }),
        type3: None,
        substitute,
    }
}

/// `desc` selected as the indirect font dictionary `1 0 R`.
pub fn active(desc: FontDescriptor) -> ActiveFont {
    ActiveFont {
        key: ResourceKey::Object((1, 0)),
        descriptor: Arc::new(desc),
    }
}

/// The face `cache` loads for `desc` at 20 px, prepared for showing
/// `bytes`.
pub fn face_showing<'a>(
    cache: &'a mut FontCache,
    desc: FontDescriptor,
    bytes: &[u8],
) -> &'a PageFace {
    cache
        .get_or_load_showing(&active(desc), b"F1", [20.0, 0.0, 0.0, 20.0], bytes)
        .expect("face loads")
}

/// [`face_showing`] for no string.
pub fn face(cache: &mut FontCache, desc: FontDescriptor) -> &PageFace {
    face_showing(cache, desc, &[])
}

/// A cache drawing substitutes from the standard font directories.
pub fn cache() -> FontCache {
    FontCache::new(Arc::default(), true)
}
