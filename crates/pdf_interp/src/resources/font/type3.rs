//! Type 3 fonts: glyphs drawn by content streams (PDF §9.6.5).

use std::collections::HashMap;
use std::sync::Arc;

use pdf::{Dictionary, Document, Object, ObjectId};

use super::{
    BaseEncoding, FontDescriptor, PdfFontKind, extract_encoding, extract_widths, number,
    width_units,
};
use crate::content::Operator;
use crate::renderer::encodings::glyph_name_for_code;
use crate::resources::font_substitute::SubstituteStyle;

/// Decoded `CharProc` entry for a single Type 3 glyph.
#[derive(Debug, Clone)]
pub struct Type3Glyph {
    /// The glyph's paint procedure, parsed once and shared by every code
    /// whose glyph name selects it.
    pub ops: Arc<[Operator]>,
}

/// All data extracted from a Type 3 font dictionary.
///
/// Unlike fonts with a font program, Type 3 glyphs are PDF content streams that
/// must be executed by a child renderer.  Each `CharProc` stream begins with a
/// `d0` (coloured) or `d1` (shape-only) operator that declares the glyph
/// advance width.
#[derive(Debug, Clone)]
pub struct Type3Data {
    /// Resolved glyph content keyed by char code (0–255).
    /// Missing entries mean `.notdef`; the caller renders nothing for them.
    pub glyphs: Vec<Option<Type3Glyph>>,
    /// Advance width of every char code in glyph space, from the `Widths`
    /// array; 0 for a code outside `[FirstChar, LastChar]`.  A code without
    /// a `CharProc` advances by its width all the same (PDF §9.4.4).  Real
    /// valued: under a unit `FontMatrix` a width is a fraction of a
    /// text-space unit.
    pub widths: Vec<f64>,
    /// `FontMatrix`, mapping glyph space to text space; `[0.001 0 0
    /// 0.001 0 0]` when absent or malformed.
    pub font_matrix: [f64; 6],
    /// The font dictionary, when it is an indirect object carrying its own
    /// `/Resources`: the `CharProcs` resolve their named resources there
    /// instead of in the page's (PDF §9.6.5).
    pub resources: Option<ObjectId>,
}

impl Type3Data {
    /// Return the content for `char_code`, if any `CharProc` exists.
    #[must_use]
    pub fn glyph(&self, char_code: u8) -> Option<&Type3Glyph> {
        self.glyphs.get(usize::from(char_code))?.as_ref()
    }

    /// Advance width of `char_code` in glyph space.
    #[must_use]
    pub fn width(&self, char_code: u8) -> f64 {
        self.widths
            .get(usize::from(char_code))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Greatest decompressed size of one `CharProc` stream.
const MAX_CHARPROC_BYTES: usize = 4 << 20;

/// Greatest decompressed size of the `CharProcs` of one font together:
/// 256 glyphs of 256 KiB each, where a glyph is a few kilobytes of path or
/// inline-image operators.
const MAX_CHARPROCS_BYTES: usize = 64 << 20;

/// Extract [`FontDescriptor`] for a Type 3 font.
///
/// Type 3 fonts have no embedded font program — each glyph is a PDF content
/// stream in the `CharProcs` dictionary.  The `Widths` array and `FirstChar`
/// key give the advance widths.  A code's `CharProc` is the one named by
/// `Differences`, else by the base encoding (`/Encoding /WinAnsiEncoding`,
/// `/BaseEncoding`), else by `StandardEncoding` (PDF §9.6.6.1).
pub(super) fn resolve_type3_font(
    doc: &Document,
    dict: &Dictionary,
    id: Option<ObjectId>,
) -> FontDescriptor {
    let (first_char, real_widths) = extract_widths(doc, dict);
    let font_matrix =
        extract_matrix(doc, dict, b"FontMatrix").unwrap_or([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);
    let (differences, base_encoding) = extract_encoding(doc, dict);
    let glyphs = dict
        .get(b"CharProcs")
        .and_then(|o| crate::resources::resolve_dict(doc, o))
        .map_or_else(
            || {
                log::warn!("font/type3: CharProcs dict missing or unresolvable — no glyphs");
                vec![None; 256]
            },
            |charprocs| charproc_glyphs(doc, &charprocs, &differences, base_encoding),
        );

    let code_widths = (0u32..256)
        .map(|code| {
            code.checked_sub(first_char)
                .and_then(|i| real_widths.get(i as usize).copied())
                .unwrap_or(0.0)
        })
        .collect();

    let resources = match (dict.get(b"Resources"), id) {
        (None, _) => None,
        (Some(_), Some(id)) => Some(id),
        (Some(_), None) => {
            log::warn!(
                "font/type3: /Resources of a direct font dictionary cannot be addressed — \
                 CharProcs resolve in the page's resources"
            );
            None
        }
    };

    FontDescriptor {
        kind: PdfFontKind::Other,
        program: None,
        widths: real_widths.iter().copied().map(width_units).collect(),
        first_char,
        // A Type 3 font advances by `Type3Data::widths` alone (PDF §9.6.5).
        missing_width: 0,
        differences,
        // The encoding selected the procedures of `type3.glyphs`; no
        // program resolves the codes again.
        base_encoding: BaseEncoding::Standard,
        cid_encoding: None,
        type3: Some(Type3Data {
            glyphs,
            widths: code_widths,
            font_matrix,
            resources,
        }),
        substitute: SubstituteStyle::default(),
    }
}

/// The glyph of each code 0–255 from the `CharProcs` dictionary
/// `charprocs`: the procedure its glyph name selects.  Each distinct
/// procedure is parsed once and shared by every code that names it, and
/// the procedures together decompress to at most [`MAX_CHARPROCS_BYTES`].
fn charproc_glyphs(
    doc: &Document,
    charprocs: &Dictionary,
    differences: &[Option<Box<str>>; 256],
    base_encoding: BaseEncoding,
) -> Vec<Option<Type3Glyph>> {
    let mut by_name: HashMap<&str, Option<Type3Glyph>> = HashMap::new();
    let mut budget = MAX_CHARPROCS_BYTES;
    let mut glyphs = Vec::with_capacity(256);
    for code in 0..256 {
        let glyph = glyph_name_for_code(differences, base_encoding, code).and_then(|name| {
            by_name
                .entry(name)
                .or_insert_with(|| {
                    let stream = charprocs.get(name.as_bytes())?;
                    charproc(doc, stream, name, &mut budget)
                })
                .clone()
        });
        glyphs.push(glyph);
    }
    glyphs
}

/// The procedure of glyph `name`, parsed from the stream `obj` (direct or
/// indirect); `None`, logged, when it does not decompress or exceeds
/// [`MAX_CHARPROC_BYTES`] or the font's remaining `budget`, which it
/// draws down.
fn charproc(doc: &Document, obj: &Object, name: &str, budget: &mut usize) -> Option<Type3Glyph> {
    let content = match obj {
        Object::Stream(s) => s.decompressed_content(),
        Object::Reference(id) => doc
            .get_object(*id)
            .ok()?
            .as_stream()?
            .decompressed_content(),
        _ => return None,
    };
    let content = content
        .inspect_err(|e| log::warn!("font/type3: CharProc /{name} does not decompress: {e}"))
        .ok()?;
    let limit = MAX_CHARPROC_BYTES.min(*budget);
    if content.len() > limit {
        log::warn!(
            "font/type3: CharProc /{name} is {} bytes, over the {limit}-byte limit — skipping it",
            content.len()
        );
        return None;
    }
    *budget -= content.len();
    Some(Type3Glyph {
        ops: crate::content::parse(&content).into(),
    })
}

/// The six numbers of the matrix at `dict[key]`, the array or its
/// elements possibly indirect; `None` when absent, shorter than six or
/// holding a non-number.
fn extract_matrix(doc: &Document, dict: &Dictionary, key: &[u8]) -> Option<[f64; 6]> {
    let arr = doc.resolve(dict.get(key)?).ok()?;
    let values: Vec<f64> = arr
        .as_array()?
        .iter()
        .take(6)
        .map(|o| number(doc, o))
        .collect::<Option<_>>()?;
    values.try_into().ok()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::resolve_font;
    use super::super::tests::make_dict;
    use crate::test_helpers::empty_doc;
    use pdf::Object;

    /// Codes whose glyph names select one `CharProc` share one parsed
    /// procedure.
    #[test]
    fn codes_naming_one_charproc_share_its_procedure() {
        let doc = crate::test_helpers::make_doc_with_stream(b"0 0 d0 0 0 1 1 re f", "");
        let mut charprocs = pdf::Dictionary::new();
        charprocs.set(b"g", Object::Reference((2, 0)));
        let encoding = make_dict(&[(
            b"Differences",
            Object::Array(vec![
                Object::Integer(65),
                Object::Name(b"g".to_vec()),
                Object::Name(b"g".to_vec()),
                Object::Name(b"missing".to_vec()),
            ]),
        )]);
        let d = make_dict(&[
            (b"Subtype", Object::Name(b"Type3".to_vec())),
            (b"CharProcs", Object::Dictionary(charprocs)),
            (b"Encoding", Object::Dictionary(encoding)),
        ]);
        let t3 = resolve_font(&doc, &d, None).type3.expect("Type 3 data");
        let (a, b) = (t3.glyph(65).expect("A"), t3.glyph(66).expect("B"));
        assert!(std::sync::Arc::ptr_eq(&a.ops, &b.ops));
        assert!(!a.ops.is_empty());
        assert!(t3.glyph(67).is_none(), "no CharProc of that name");
    }

    /// The fractional widths of a Type 3 font drawn under a unit
    /// `FontMatrix` keep their value: truncated to integers they would
    /// all be 0, stacking every glyph at one position.
    #[test]
    fn type3_widths_are_real_valued() {
        let doc = empty_doc();
        let d = make_dict(&[
            (b"Subtype", Object::Name(b"Type3".to_vec())),
            (b"FirstChar", Object::Integer(65)),
            (
                b"Widths",
                Object::Array(vec![Object::Real(0.5), Object::Real(0.25)]),
            ),
            (
                b"FontMatrix",
                Object::Array(vec![
                    Object::Integer(1),
                    Object::Integer(0),
                    Object::Integer(0),
                    Object::Integer(1),
                    Object::Integer(0),
                    Object::Integer(0),
                ]),
            ),
        ]);
        let t3 = resolve_font(&doc, &d, None).type3.expect("Type 3 data");
        assert!((t3.width(65) - 0.5).abs() < 1e-6);
        assert!((t3.width(66) - 0.25).abs() < 1e-6);
        assert!(t3.width(67).abs() < f64::EPSILON);
    }
}
