//! Annotation rendering helpers for [`PageRenderer`].
//!
//! Free functions taking `renderer: &mut PageRenderer<'_>`, delegated from
//! thin wrapper methods in `mod.rs`.  An annotation without a normal
//! appearance is drawn from an appearance synthesised by [`markup`],
//! [`free_text`], [`popup`] or [`widget`], through the same form path as a
//! stored appearance.  A pop-up is drawn only while open.

mod content;
mod encoder;
mod free_text;
mod line_ending;
mod markup;
mod popup;
mod text;
mod typeset;
mod widget;

use pdf::{Dictionary, Document, Object, ObjectId};

use super::super::gstate::{Ctm, ctm_multiply};
use super::PageRenderer;
use crate::RenderPurpose;
use crate::resources::dict_ext::DictExt;
use raster::types::BlendMode;
use typeset::Typesetter;

/// A synthesised appearance.
struct Synthesized {
    /// The content stream, in form space.
    pub content: Vec<u8>,
    /// The region the content may paint, in form space.
    pub bbox: [f64; 4],
    /// The constant opacity `/CA` applied to stroking and filling.
    pub opacity: f64,
    /// The blend mode the content is painted with.
    pub blend: BlendMode,
    /// The form matrix, from form space to page user space; [`IDENTITY`]
    /// for content written in page user space.
    pub matrix: Ctm,
}

/// The identity matrix.
const IDENTITY: Ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Annotation flag bits (PDF §12.5.3, Table 167).
const HIDDEN: i64 = 1 << 1;
const PRINT: i64 = 1 << 2;
const NO_ROTATE: i64 = 1 << 4;
const NO_VIEW: i64 = 1 << 5;

/// Render all annotations on the page (PDF §12.5).
///
/// Each entry of `/Annots`, an annotation dictionary or a reference to one,
/// that is shown for the renderer's purpose and has a normal appearance
/// (`/AP /N`) is drawn after the page content stream, in array order.
/// Rollover (`/R`) and down (`/D`) appearances are not used.
///
/// Annotations with no appearance stream are skipped; many annotation types
/// (e.g. `Link`) have no visible appearance by default.
///
/// Call this after `execute` and before `finish`.
pub(super) fn render_annotations(renderer: &mut PageRenderer<'_>, page_id: ObjectId) {
    let doc = renderer.resources.doc();
    let Ok(page_dict) = doc.get_dictionary(page_id) else {
        log::warn!(
            "rasterrocket-interp: render_annotations: page object ({}, {}) is not a dictionary — \
             skipping annotations",
            page_id.0,
            page_id.1
        );
        return;
    };

    let entries: Vec<Object> = match page_dict.get(b"Annots") {
        Some(Object::Array(a)) => a.clone(),
        Some(Object::Reference(id)) => {
            let id = *id;
            if let Some(a) = doc
                .get_object(id)
                .ok()
                .and_then(|o| o.as_array().map(<[Object]>::to_vec))
            {
                a
            } else {
                log::warn!(
                    "rasterrocket-interp: render_annotations: /Annots reference ({}, {}) \
                     did not resolve to an array — skipping annotations",
                    id.0,
                    id.1
                );
                return;
            }
        }
        _ => return,
    };

    let form = widget::AcroForm::of(doc);
    for entry in entries {
        match entry {
            Object::Reference(id) => {
                let Ok(annot) = doc.get_dictionary(id) else {
                    log::warn!(
                        "rasterrocket-interp: annotation ({}, {}) object missing or not a \
                         dictionary — skipping",
                        id.0,
                        id.1
                    );
                    continue;
                };
                render_one_annotation(renderer, &annot, Some(id), form.as_ref());
            }
            Object::Dictionary(annot) => {
                render_one_annotation(renderer, &annot, None, form.as_ref());
            }
            _ => {}
        }
    }
}

/// Whether an annotation with flags `flags` is shown for `purpose`
/// (PDF §12.5.3): never when `Hidden`; when viewing, unless `NoView`; when
/// printing, only when `Print`.
const fn shown_for(flags: i64, purpose: RenderPurpose) -> bool {
    if flags & HIDDEN != 0 {
        return false;
    }
    match purpose {
        RenderPurpose::View => flags & NO_VIEW == 0,
        RenderPurpose::Print => flags & PRINT != 0,
    }
}

/// Render the annotation dictionary `annot` (object `id`, or direct when
/// `None`) into `renderer`'s bitmap; `form` is the document's interactive
/// form, whose default resources widget and free text appearances draw
/// with.
fn render_one_annotation(
    renderer: &mut PageRenderer<'_>,
    annot: &Dictionary,
    id: Option<ObjectId>,
    form: Option<&widget::AcroForm>,
) {
    let doc = renderer.resources.doc();
    let flags = annot.get_i64(b"F").unwrap_or(0);
    if !shown_for(flags, renderer.optional_content().purpose()) {
        return;
    }
    if let Some(oc) = annot.get(b"OC")
        && !renderer.oc_visible(oc)
    {
        return;
    }

    let has_normal_appearance = annot
        .get(b"AP")
        .and_then(|o| crate::resources::resolve_dict(doc, o))
        .is_some_and(|ap| ap.get(b"N").is_some());
    if annot.get_name(b"Subtype") == Some(b"Popup") {
        if !popup::is_open(annot) {
            return;
        }
        let parent = popup::parent(doc, annot);
        let purpose = renderer.optional_content().purpose();
        if parent
            .as_ref()
            .is_some_and(|p| !shown_for(p.get_i64(b"F").unwrap_or(0), purpose))
        {
            return;
        }
        if !has_normal_appearance {
            if let Some(rect) = read_rect(annot) {
                let dr = form.and_then(widget::AcroForm::default_resources);
                let resources = text::resources(doc, dr, &[popup::FONT]);
                let no_rotate_rect = (flags & NO_ROTATE != 0).then_some(rect);
                draw_text_appearance(renderer, &resources, id, no_rotate_rect, |ts| {
                    popup::appearance(doc, parent.as_ref(), rect, ts)
                });
            }
            return;
        }
    }
    if annot.get_name(b"Subtype") == Some(b"Widget")
        && let Some(form) = form
        && let Some(field) = widget::Field::to_synthesise(doc, annot, form, has_normal_appearance)
    {
        if let Some(rect) = read_rect(annot) {
            draw_widget(renderer, form, &field, rect, id);
        }
        return;
    }
    if !has_normal_appearance {
        let Some(rect) = read_rect(annot) else {
            return;
        };
        let no_rotate_rect = (flags & NO_ROTATE != 0).then_some(rect);
        if annot.get_name(b"Subtype") == Some(b"FreeText") {
            draw_free_text(renderer, annot, rect, form, id, no_rotate_rect);
        } else if let Some(appearance) = markup::markup(doc, annot, rect) {
            draw_synthesized(renderer, appearance, id, no_rotate_rect);
        }
        return;
    }

    // Check for AP/N before Rect — annotations with no AP stream have no visible
    // appearance (e.g. Link) and are expected to be absent silently.
    let Some(stream_id) = normal_appearance(doc, annot) else {
        return;
    };

    // Annotation rect in page user space: [llx, lly, urx, ury].
    // Warn when AP exists but Rect is absent/malformed — visible annotation dropped.
    let Some(rect) = read_rect(annot) else {
        log::warn!(
            "rasterrocket-interp: annotation {} has AP/N stream but missing or malformed Rect — skipping",
            id.map_or_else(
                || "(direct)".to_owned(),
                |id| format!("({}, {})", id.0, id.1)
            )
        );
        return;
    };

    let Some(mut form) = renderer.resources.form_from_stream_id(stream_id) else {
        return;
    };
    form.matrix = ctm_multiply(
        &form.matrix,
        &fit_to_rect(
            &form.matrix,
            form.bbox.unwrap_or([0.0, 0.0, 1.0, 1.0]),
            rect,
        ),
    );
    if flags & NO_ROTATE != 0 && renderer.rotate_cw != 0 {
        form.matrix = ctm_multiply(&form.matrix, &upright_about(rect, renderer.rotate_cw));
    }
    renderer.do_form_xobject(&form);
}

/// Draw the synthesised appearance `appearance` of the annotation `id`
/// (direct when `None`) with its form matrix, opacity and blend mode;
/// turned upright about the upper-left corner of `no_rotate_rect` when
/// given.
fn draw_synthesized(
    renderer: &mut PageRenderer<'_>,
    appearance: Synthesized,
    id: Option<ObjectId>,
    no_rotate_rect: Option<[f64; 4]>,
) {
    let matrix = match no_rotate_rect {
        Some(rect) if renderer.rotate_cw != 0 => {
            ctm_multiply(&appearance.matrix, &upright_about(rect, renderer.rotate_cw))
        }
        _ => appearance.matrix,
    };
    // The form has no stream object; the annotation (or, for a direct
    // annotation, the page) stands in for it in the active-form stack, and
    // it draws with the resources in effect.
    let form = crate::resources::FormXObject {
        content: appearance.content,
        matrix,
        bbox: Some(appearance.bbox),
        resources_id: id.unwrap_or_else(|| renderer.resources.resource_context_id()),
        has_own_resources: false,
        transparency: None,
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "opacity is clamped to [0, 1], so the product is in [0, 255]"
    )]
    let alpha = (appearance.opacity * 255.0).round() as u8;
    renderer.gstate.save();
    let gs = renderer.gstate.current_mut();
    gs.fill_alpha = alpha;
    gs.stroke_alpha = alpha;
    gs.blend_mode = appearance.blend;
    renderer.do_form_xobject(&form);
    renderer.gstate.restore();
}

/// Draw the synthesised appearance that `build` writes, with `resources`,
/// derived from the form's default resources, in effect; `build` sets its
/// text with a [`encoder::FontTypesetter`] over those resources.  The
/// remaining arguments are those of [`draw_synthesized`].
fn draw_text_appearance(
    renderer: &mut PageRenderer<'_>,
    resources: &Dictionary,
    id: Option<ObjectId>,
    no_rotate_rect: Option<[f64; 4]>,
    build: impl FnOnce(&mut dyn Typesetter) -> Synthesized,
) {
    // The default resources are no object's `/Resources`; object 0, which
    // no document defines, identifies them to the font cache.
    let dr = renderer.resources.with_resources((0, 0), resources);
    let page_resources = std::mem::replace(&mut renderer.resources, dr);
    let appearance = build(&mut encoder::FontTypesetter::new(renderer));
    draw_synthesized(renderer, appearance, id, no_rotate_rect);
    renderer.resources = page_resources;
}

/// Draw the synthesised appearance of the widget of `field` in `rect`,
/// turned by its `/MK /R` rotation.
fn draw_widget(
    renderer: &mut PageRenderer<'_>,
    form: &widget::AcroForm,
    field: &widget::Field,
    rect: [f64; 4],
    id: Option<ObjectId>,
) {
    let doc = renderer.resources.doc();
    let resources = widget::resources(doc, form, field);
    let (local, matrix) = rotated_box(rect, field.rotation());
    draw_text_appearance(renderer, &resources, id, None, |ts| Synthesized {
        content: widget::content(doc, field, local, ts),
        bbox: local,
        opacity: 1.0,
        blend: BlendMode::Normal,
        matrix,
    });
}

/// The box an appearance rotated by `rotation` degrees counter-clockwise
/// (a multiple of 90) is laid out in, with its lower-left corner at the
/// origin and its sides swapped for a quarter turn, and the matrix that
/// turns it and places it on `rect` (ISO 32000-2 §12.5.6.19, `/MK /R`).
fn rotated_box(rect: [f64; 4], rotation: u16) -> ([f64; 4], Ctm) {
    let [x0, y0, x1, y1] = rect;
    let (w, h) = (x1 - x0, y1 - y0);
    match rotation % 360 {
        90 => ([0.0, 0.0, h, w], [0.0, 1.0, -1.0, 0.0, x1, y0]),
        180 => ([0.0, 0.0, w, h], [-1.0, 0.0, 0.0, -1.0, x1, y1]),
        270 => ([0.0, 0.0, h, w], [0.0, -1.0, 1.0, 0.0, x0, y1]),
        _ => ([0.0, 0.0, w, h], [1.0, 0.0, 0.0, 1.0, x0, y0]),
    }
}

/// Draw the synthesised appearance of the free text annotation `annot`
/// with the normalised rectangle `rect`.
fn draw_free_text(
    renderer: &mut PageRenderer<'_>,
    annot: &Dictionary,
    rect: [f64; 4],
    form: Option<&widget::AcroForm>,
    id: Option<ObjectId>,
    no_rotate_rect: Option<[f64; 4]>,
) {
    let doc = renderer.resources.doc();
    let free_text = free_text::FreeText::of(doc, annot, rect, form);
    let resources = text::resources(
        doc,
        form.and_then(widget::AcroForm::default_resources),
        &[free_text.font()],
    );
    draw_text_appearance(renderer, &resources, id, no_rotate_rect, |ts| {
        free_text.appearance(ts)
    });
}

/// The normal appearance stream of `annot` (PDF §12.5.5): `/AP /N` itself,
/// or, when `/N` is a dictionary of appearance states, the entry for the
/// annotation's appearance state.  The state is `/AS`; without it, the
/// field value `/V`, inherited through the `/Parent` fields, which
/// names the appearance state of a check box or radio button (PDF
/// §12.7.5.3); a value naming no state, or no value, selects `Off`.
fn normal_appearance(doc: &Document, annot: &Dictionary) -> Option<ObjectId> {
    let ap = crate::resources::resolve_dict(doc, annot.get(b"AP")?)?;
    let states = match ap.get(b"N")? {
        Object::Reference(id) => match doc.get_object(*id).ok()?.as_ref() {
            Object::Stream(_) => return Some(*id),
            Object::Dictionary(d) => d.clone(),
            _ => return None,
        },
        Object::Dictionary(d) => d.clone(),
        _ => return None,
    };
    let value = || {
        widget::inherited(doc, &widget::field_chain(doc, annot), b"V")
            .and_then(|v| v.as_name().map(<[u8]>::to_vec))
    };
    let state: Vec<u8> = annot.get_name(b"AS").map_or_else(
        || {
            value()
                .filter(|v| states.contains_key(v))
                .unwrap_or_else(|| b"Off".to_vec())
        },
        <[u8]>::to_vec,
    );
    match states.get(&state)? {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

/// The matrix `A` of PDF §12.5.5 Algorithm 8.1 that maps the appearance
/// `bbox`, transformed by its form `matrix`, onto `rect`: the bounding box
/// of the transformed `bbox` is scaled and translated to coincide with
/// `rect`.  A side of zero extent keeps scale 1 and aligns its lower edge.
fn fit_to_rect(matrix: &Ctm, bbox: [f64; 4], rect: [f64; 4]) -> Ctm {
    let [bx0, by0, bx1, by1] = bbox;
    let corners = [(bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)].map(|(x, y)| {
        (
            matrix[0].mul_add(x, matrix[2] * y) + matrix[4],
            matrix[1].mul_add(x, matrix[3] * y) + matrix[5],
        )
    });
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for (x, y) in corners {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let [llx, lly, urx, ury] = rect;
    let scale = |lo: f64, hi: f64, to: f64| {
        if (hi - lo).abs() < f64::EPSILON {
            1.0
        } else {
            to / (hi - lo)
        }
    };
    let sx = scale(x0, x1, urx - llx);
    let sy = scale(y0, y1, ury - lly);
    [sx, 0.0, 0.0, sy, x0.mul_add(-sx, llx), y0.mul_add(-sy, lly)]
}

/// The rotation by `rotate_cw` degrees counter-clockwise about the
/// upper-left corner of `rect`, which cancels the page's `/Rotate` for a
/// `NoRotate` annotation (PDF §12.5.3): the corner keeps its place on the
/// page and the appearance is drawn upright from it.
fn upright_about(rect: [f64; 4], rotate_cw: u16) -> Ctm {
    let (sin, cos) = match rotate_cw % 360 {
        90 => (1.0, 0.0),
        180 => (0.0, -1.0),
        270 => (-1.0, 0.0),
        _ => (0.0, 1.0),
    };
    let (x, y) = (rect[0], rect[3]);
    // T(-x, -y) × R × T(x, y).
    [
        cos,
        sin,
        -sin,
        cos,
        x - cos.mul_add(x, -sin * y),
        y - sin.mul_add(x, cos * y),
    ]
}

/// Read a 4-element rect `[llx, lly, urx, ury]` from a PDF dictionary.
///
/// Returns `None` if the `Rect` key is absent or has fewer than 4 finite
/// numeric entries.
fn read_rect(dict: &pdf::Dictionary) -> Option<[f64; 4]> {
    let mut r = crate::resources::read_f64_n::<4>(dict, b"Rect")
        .filter(|r| r.iter().all(|v| v.is_finite()))?;
    // Normalise so llx ≤ urx and lly ≤ ury.
    if r[0] > r[2] {
        r.swap(0, 2);
    }
    if r[1] > r[3] {
        r.swap(1, 3);
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::{fit_to_rect, normal_appearance, upright_about};
    use crate::test_helpers::make_doc_with_objects;

    #[test]
    fn the_appearance_state_follows_a_value_inherited_from_a_grandparent() {
        let stream = "<< /Length 0 >>\nstream\n\nendstream";
        let doc = make_doc_with_objects(&[
            (
                4,
                "<< /AP << /N << /On 5 0 R /Off 6 0 R >> >> /Parent 7 0 R >>",
            ),
            (5, stream),
            (6, stream),
            (7, "<< /Parent 8 0 R >>"),
            (8, "<< /FT /Btn /V /On >>"),
        ]);
        let annot = doc.get_dictionary((4, 0)).expect("a dictionary");
        assert_eq!(normal_appearance(&doc, &annot), Some((5, 0)));
    }

    fn apply(m: &[f64; 6], (x, y): (f64, f64)) -> (f64, f64) {
        (
            m[0].mul_add(x, m[2] * y) + m[4],
            m[1].mul_add(x, m[3] * y) + m[5],
        )
    }

    #[test]
    fn fitting_maps_the_bbox_corners_onto_the_rect() {
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a = fit_to_rect(&identity, [10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 40.0, 80.0]);
        assert_eq!(apply(&a, (10.0, 20.0)), (0.0, 0.0));
        assert_eq!(apply(&a, (30.0, 60.0)), (40.0, 80.0));
    }

    #[test]
    fn a_degenerate_side_keeps_unit_scale() {
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a = fit_to_rect(&identity, [5.0, 0.0, 5.0, 10.0], [20.0, 0.0, 40.0, 20.0]);
        assert_eq!((a[0], a[3]), (1.0, 2.0));
        assert_eq!(apply(&a, (5.0, 0.0)), (20.0, 0.0));
    }

    #[test]
    fn the_upright_rotation_fixes_the_upper_left_corner() {
        let rect = [10.0, 0.0, 50.0, 20.0];
        for rotate in [90, 180, 270] {
            let r = upright_about(rect, rotate);
            let (x, y) = apply(&r, (10.0, 20.0));
            assert!(
                (x - 10.0).abs() < 1e-9 && (y - 20.0).abs() < 1e-9,
                "{rotate}"
            );
        }
        let r = upright_about(rect, 90);
        let (x, y) = apply(&r, (11.0, 20.0));
        assert!((x - 10.0).abs() < 1e-9 && (y - 21.0).abs() < 1e-9);
    }
}
