//! The page group (ISO 32000-2 §11.4.7).
//!
//! A page's content is a transparency group that is composited onto the
//! output medium once it is complete, so a blend mode applied to an object
//! on the page sees only what the page has painted under the object, not
//! the medium.  The page group is therefore an isolated group whose result
//! composites over white; its `/Group` entry may also make it a knockout
//! group.
//!
//! Painting the page as a group costs an alpha plane and a final composite
//! over white.  Where no object can blend, compositing straight onto an
//! opaque white page gives the same pixels, so a page is painted as a group
//! only when it is a knockout group or when a blend mode other than `Normal`
//! is reachable from its resources: through its `ExtGState`s, its form
//! `XObject`s' and Type 3 fonts' resources, its patterns, and the appearance
//! streams of its annotations.

use std::collections::HashSet;

use pdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

use crate::resources::dict_ext::DictExt;
use crate::resources::{TransparencyGroupParams, parse_blend_mode, resolve_dict};

/// Resource dictionaries are followed through forms, patterns and fonts
/// at most this deep.
const MAX_SCAN_DEPTH: u32 = 8;

/// The page group of `page_id` when the page is to be painted as a group.
///
/// Its `/Group` attributes, with `isolated` set, when the page is a knockout
/// group or a blend mode other than `Normal` is reachable from its resources
/// or annotations; `None` when painting onto opaque white gives the same
/// result.
#[must_use]
pub fn page_group(doc: &Document, page_id: ObjectId) -> Option<TransparencyGroupParams> {
    let page = doc.get_dictionary(page_id).ok()?;
    let group = crate::resources::read_transparency_group(doc, &page).unwrap_or_default();
    let mut scan = Scan {
        doc,
        visited: HashSet::new(),
    };
    let blends = group.knockout
        || inherited_resources(doc, page_id).is_some_and(|r| scan.resources_blend(&r, 0))
        || scan.annotations_blend(&page);
    blends.then_some(TransparencyGroupParams {
        isolated: true,
        knockout: group.knockout,
    })
}

/// The `/Resources` of `page_id` or of its nearest ancestor that has one
/// (ISO 32000-2 §7.7.3.4).
fn inherited_resources(doc: &Document, page_id: ObjectId) -> Option<Dictionary> {
    let mut id = Some(page_id);
    let mut seen = HashSet::new();
    while let Some(current) = id.filter(|id| seen.insert(*id)) {
        let node = doc.get_dictionary(current).ok()?;
        if let Some(resources) = node.get(b"Resources") {
            return resolve_dict(doc, resources);
        }
        id = node.get(b"Parent").and_then(Object::as_reference);
    }
    None
}

/// A walk over resource dictionaries that stops at the first blend mode.
struct Scan<'a> {
    doc: &'a Document,
    /// Indirect objects already walked, so shared forms are read once and
    /// a cycle ends.
    visited: HashSet<ObjectId>,
}

impl Scan<'_> {
    /// Whether a blend mode other than `Normal` is set by an `ExtGState`
    /// of `resources` or of the resources reachable from it.
    fn resources_blend(&mut self, resources: &Dictionary, depth: u32) -> bool {
        if depth > MAX_SCAN_DEPTH {
            return false;
        }
        let sub = |key: &[u8]| resources.get(key).and_then(|o| resolve_dict(self.doc, o));
        if let Some(gstates) = sub(b"ExtGState")
            && gstates.iter().any(|(_, o)| {
                resolve_dict(self.doc, o)
                    .and_then(|gs| parse_blend_mode(&gs))
                    .is_some_and(|bm| bm != BlendMode::Normal)
            })
        {
            return true;
        }
        if let Some(xobjects) = sub(b"XObject")
            && xobjects
                .iter()
                .any(|(_, o)| self.stream_resources_blend(o, depth))
        {
            return true;
        }
        if let Some(patterns) = sub(b"Pattern")
            && patterns.iter().any(|(_, o)| self.pattern_blends(o, depth))
        {
            return true;
        }
        sub(b"Font").is_some_and(|fonts| {
            fonts.iter().any(|(_, o)| {
                self.first_visit(o)
                    .and_then(|font| font.get(b"Resources").cloned())
                    .and_then(|r| resolve_dict(self.doc, &r))
                    .is_some_and(|r| self.resources_blend(&r, depth + 1))
            })
        })
    }

    /// Whether the resources of the form `XObject` or appearance stream
    /// `obj` names reach a blend mode.
    fn stream_resources_blend(&mut self, obj: &Object, depth: u32) -> bool {
        self.first_visit(obj)
            .filter(|dict| dict.get_name(b"Subtype") != Some(b"Image"))
            .and_then(|dict| dict.get(b"Resources").cloned())
            .and_then(|r| resolve_dict(self.doc, &r))
            .is_some_and(|r| self.resources_blend(&r, depth + 1))
    }

    /// Whether the pattern `obj` names reaches a blend mode: a shading
    /// pattern through its `ExtGState`, a tiling pattern through its
    /// resources.
    fn pattern_blends(&mut self, obj: &Object, depth: u32) -> bool {
        let Some(pattern) = self.first_visit(obj) else {
            return false;
        };
        if let Some(gs) = pattern
            .get(b"ExtGState")
            .and_then(|o| resolve_dict(self.doc, o))
            && parse_blend_mode(&gs).is_some_and(|bm| bm != BlendMode::Normal)
        {
            return true;
        }
        pattern
            .get(b"Resources")
            .and_then(|r| resolve_dict(self.doc, r))
            .is_some_and(|r| self.resources_blend(&r, depth + 1))
    }

    /// Whether an appearance stream of an annotation of `page` reaches a
    /// blend mode.
    fn annotations_blend(&mut self, page: &Dictionary) -> bool {
        let annots = match page.get(b"Annots") {
            Some(Object::Array(a)) => a.clone(),
            Some(Object::Reference(id)) => self
                .doc
                .get_object(*id)
                .ok()
                .and_then(|o| o.as_array().map(<[Object]>::to_vec))
                .unwrap_or_default(),
            _ => return false,
        };
        annots.iter().any(|annot| {
            let Some(normal) = resolve_dict(self.doc, annot)
                .and_then(|a| resolve_dict(self.doc, a.get(b"AP")?))
                .and_then(|ap| ap.get(b"N").cloned())
            else {
                return false;
            };
            match &normal {
                Object::Reference(_) => {
                    if self.stream_resources_blend(&normal, 0) {
                        return true;
                    }
                    // A dictionary of appearance states.
                    resolve_dict(self.doc, &normal).is_some_and(|states| {
                        states
                            .iter()
                            .any(|(_, s)| self.stream_resources_blend(s, 0))
                    })
                }
                Object::Dictionary(states) => states
                    .iter()
                    .any(|(_, s)| self.stream_resources_blend(s, 0)),
                _ => false,
            }
        })
    }

    /// The dictionary of `obj` — a stream's or a plain one — the first time
    /// the indirect object is met; `None` on a repeat visit or when it is
    /// not a dictionary.
    fn first_visit(&mut self, obj: &Object) -> Option<Dictionary> {
        if let Object::Reference(id) = obj
            && !self.visited.insert(*id)
        {
            return None;
        }
        crate::resources::resolve_stream_dict(self.doc, obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_objects;

    /// Page `(4, 0)` with resources `resources` and page entries `extra`.
    fn doc_with(resources: &str, extra: &str, objects: &[(u32, String)]) -> Document {
        let mut all = vec![(
            4,
            format!("<< /Type /Page /Resources << {resources} >> {extra} >>"),
        )];
        all.extend(objects.iter().cloned());
        make_doc_with_objects(&all)
    }

    #[test]
    fn a_page_without_blend_modes_has_no_group() {
        let doc = doc_with("/ExtGState << /A << /ca 0.5 /BM /Normal >> >>", "", &[]);
        assert_eq!(page_group(&doc, (4, 0)), None);
        let doc = doc_with("", "/Group << /S /Transparency /I true >>", &[]);
        assert_eq!(
            page_group(&doc, (4, 0)),
            None,
            "isolation alone changes nothing"
        );
    }

    #[test]
    fn a_blend_mode_in_the_page_resources_makes_an_isolated_group() {
        let doc = doc_with("/ExtGState << /M << /BM /Multiply >> >>", "", &[]);
        assert_eq!(
            page_group(&doc, (4, 0)),
            Some(TransparencyGroupParams {
                isolated: true,
                knockout: false
            })
        );
    }

    #[test]
    fn a_knockout_page_group_is_kept() {
        let doc = doc_with("", "/Group << /S /Transparency /K true >>", &[]);
        assert_eq!(
            page_group(&doc, (4, 0)),
            Some(TransparencyGroupParams {
                isolated: true,
                knockout: true
            })
        );
    }

    #[test]
    fn a_blend_mode_is_found_through_a_form_and_its_nested_form() {
        let doc = doc_with(
            "/XObject << /F 5 0 R >>",
            "",
            &[
                (
                    5,
                    "<< /Type /XObject /Subtype /Form /Resources << /XObject << /G 6 0 R >> >> \
                     /Length 0 >>\nstream\n\nendstream"
                        .to_owned(),
                ),
                (
                    6,
                    "<< /Type /XObject /Subtype /Form /Resources << /ExtGState << /S << /BM /Screen >> >> >> \
                     /Length 0 >>\nstream\n\nendstream"
                        .to_owned(),
                ),
            ],
        );
        assert!(page_group(&doc, (4, 0)).is_some());
    }

    #[test]
    fn a_cycle_of_forms_without_blend_modes_ends() {
        let doc = doc_with(
            "/XObject << /F 5 0 R >>",
            "",
            &[(
                5,
                "<< /Type /XObject /Subtype /Form /Resources << /XObject << /F 5 0 R >> >> \
                 /Length 0 >>\nstream\n\nendstream"
                    .to_owned(),
            )],
        );
        assert_eq!(page_group(&doc, (4, 0)), None);
    }

    #[test]
    fn a_blend_mode_is_found_in_a_shading_pattern_and_an_appearance_stream() {
        let doc = doc_with(
            "/Pattern << /P << /PatternType 2 /ExtGState << /BM /Darken >> >> >>",
            "",
            &[],
        );
        assert!(page_group(&doc, (4, 0)).is_some(), "shading pattern");
        let doc = doc_with(
            "",
            "/Annots [<< /Subtype /Square /AP << /N 5 0 R >> >>]",
            &[(
                5,
                "<< /Type /XObject /Subtype /Form /Resources << /ExtGState << /S << /BM /Screen >> >> >> \
                 /Length 0 >>\nstream\n\nendstream"
                    .to_owned(),
            )],
        );
        assert!(page_group(&doc, (4, 0)).is_some(), "appearance stream");
    }
}
