//! Type 3 fonts: glyphs drawn by content streams (PDF §9.6.5).

use pdf::{Dictionary, Document, Object, ObjectId};

use super::{
    BaseEncoding, FontDescriptor, PdfFontKind, extract_encoding, extract_widths, width_units,
};
use crate::content::Operator;
use crate::renderer::encodings::glyph_name_for_code;
use crate::resources::font_substitute::SubstituteStyle;

/// Decoded `CharProc` entry for a single Type 3 glyph.
#[derive(Debug, Clone)]
pub struct Type3Glyph {
    /// The glyph's paint procedure, parsed once: every occurrence of the
    /// glyph executes the same operator sequence.
    pub ops: Vec<Operator>,
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
    /// 6-element `FontMatrix` (user-space → glyph-space transform).
    /// Default is `[0.001, 0, 0, 0.001, 0, 0]` per PDF spec §9.6.5.
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

/// Maximum decompressed size for a single Type 3 `CharProc` stream (4 MiB).
///
/// Real Type 3 glyphs are tiny path/image sequences; 4 MiB is a generous cap.
const MAX_CHARPROC_BYTES: usize = 4 * 1024 * 1024;

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

    // FontMatrix transforms glyph space to text space.
    // PDF default is [0.001 0 0 0.001 0 0] (i.e. 1000 units = 1 text-space unit).
    let font_matrix =
        extract_matrix(dict, b"FontMatrix").unwrap_or([0.001, 0.0, 0.0, 0.001, 0.0, 0.0]);

    // CharProcs: glyph name → content stream.
    let charprocs_dict = dict
        .get(b"CharProcs")
        .and_then(|o| crate::resources::resolve_dict(doc, o));

    // Resolve Encoding/Differences to get char_code → glyph_name mapping.
    let (differences, base_encoding) = extract_encoding(doc, dict);

    // Build the glyphs table: 256 entries, one per possible char code.
    let mut glyphs: Vec<Option<Type3Glyph>> = (0..256).map(|_| None).collect();

    if let Some(cp) = charprocs_dict {
        for code in 0u8..=255 {
            let Some(glyph_name) =
                glyph_name_for_code(&differences, base_encoding, usize::from(code))
            else {
                continue; // No name → no glyph for this code.
            };
            let glyph_name = glyph_name.as_bytes();

            // Look up the CharProc stream.
            let Some(stream_obj) = cp.get(glyph_name) else {
                continue;
            };
            let resolved;
            let stream = match stream_obj {
                Object::Stream(s) => s,
                Object::Reference(id) => {
                    let Some(arc) = doc.get_object(*id).ok() else {
                        continue;
                    };
                    resolved = arc;
                    match resolved.as_ref() {
                        Object::Stream(s) => s,
                        _ => continue,
                    }
                }
                _ => continue,
            };

            let content = match stream.decompressed_content() {
                Ok(b) if b.len() <= MAX_CHARPROC_BYTES => b,
                Ok(b) => {
                    log::warn!(
                        "font/type3: CharProc for code {code} is {} bytes (limit {}), skipping",
                        b.len(),
                        MAX_CHARPROC_BYTES
                    );
                    continue;
                }
                Err(e) => {
                    log::warn!("font/type3: failed to decompress CharProc for code {code}: {e}");
                    continue;
                }
            };

            glyphs[usize::from(code)] = Some(Type3Glyph {
                ops: crate::content::parse(&content),
            });
        }
    } else {
        log::warn!("font/type3: CharProcs dict missing or unresolvable — no glyphs");
    }

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
        // Type 3 glyphs carry their own widths via `Type3Glyph::width_units`;
        // the top-level Widths array is the same source the d0/d1 operators
        // store. `MissingWidth` doesn't apply (PDF spec §9.6 — Type 3 uses
        // Widths exclusively), so 0 is the correct never-consulted default.
        missing_width: 0,
        differences,
        // Type 3 glyphs are content-stream procedures, not a font program;
        // name→GID resolution does not apply.
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

/// Read a 6-element matrix from `dict[key]`.  Returns `None` if missing or malformed.
fn extract_matrix(dict: &Dictionary, key: &[u8]) -> Option<[f64; 6]> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < 6 {
        return None;
    }
    let mut m = [0.0f64; 6];
    for (i, obj) in arr.iter().enumerate().take(6) {
        m[i] = match obj {
            Object::Real(r) => f64::from(*r),
            #[expect(
                clippy::cast_precision_loss,
                reason = "PDF matrix values are small integers"
            )]
            Object::Integer(n) => *n as f64,
            _ => return None,
        };
    }
    Some(m)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::resolve_font;
    use super::super::tests::make_dict;
    use crate::test_helpers::empty_doc;
    use pdf::Object;

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
