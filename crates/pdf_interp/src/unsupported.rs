//! Features a document uses that the renderer does not process: JavaScript
//! actions, 3D and multimedia annotations, XFA forms and portable
//! collections.
//!
//! Detection is structural.  Only dictionary keys and `/S` and `/Subtype`
//! names are read; no script, model or media stream is decoded.

use std::fmt;

use pdf::{Document, Object, ObjectId};

use crate::form_type::{FormType, form_type};
use crate::resources::{self, dict_ext::DictExt};

/// A feature present in a document that the renderer ignores.  Rendering
/// proceeds without it: the page shows its static content only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnsupportedFeature {
    /// A JavaScript action.  Scripts are never executed, so content a
    /// script would draw, hide or fill in is absent.
    JavaScript(ScriptLocation),
    /// An annotation whose content is played or displayed interactively.
    /// That content is not rendered; the annotation's normal appearance,
    /// typically a poster frame, is drawn in its place when it has one
    /// and annotations are rendered.
    MediaAnnotation {
        /// The annotation type.
        kind: MediaKind,
        /// Position of the annotation in the page's `/Annots` array.
        index: usize,
        /// The annotation carries a normal appearance (`/AP /N`).
        has_appearance: bool,
    },
    /// An XFA form (`/AcroForm /XFA`), whose XFA data is not processed.
    Xfa {
        /// A dynamic form (`/NeedsRendering true`): its pages are laid out
        /// by an XFA processor, and the page content rendered is the
        /// placeholder stored in the file.  A static form's pages are
        /// ordinary page content and render as usual.
        dynamic: bool,
    },
    /// A portable collection (catalog `/Collection`): the documents it
    /// carries are embedded files, not pages, and are not rendered; the
    /// pages rendered are the collection's cover pages.
    Portfolio,
}

/// Where a JavaScript action is attached (ISO 32000-2 §12.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScriptLocation {
    /// The catalog's `/OpenAction`, run when the document opens.
    OpenAction,
    /// The catalog's additional actions (`/AA`): the document close, save
    /// and print triggers.
    DocumentActions,
    /// The document-level script name tree (`/Names /JavaScript`).
    NameTree,
    /// The interactive form dictionary's additional actions
    /// (`/AcroForm /AA`).
    FormActions,
    /// The page's additional actions (`/AA`): the page open and close
    /// triggers.
    PageActions,
    /// The activation action (`/A`) of an annotation.
    AnnotationAction {
        /// Position of the annotation in the page's `/Annots` array.
        index: usize,
    },
    /// The additional actions (`/AA`) of an annotation, among them the
    /// keystroke, format, validate and calculate scripts of a form field.
    AnnotationActions {
        /// Position of the annotation in the page's `/Annots` array.
        index: usize,
    },
}

/// An annotation type with interactive content (ISO 32000-2 §12.5.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MediaKind {
    /// `/3D`: a 3D model.
    ThreeD,
    /// `/RichMedia`: video, audio or an interactive application.
    RichMedia,
    /// `/Movie`: a movie (deprecated in PDF 2.0).
    Movie,
    /// `/Sound`: a sound (deprecated in PDF 2.0).
    Sound,
    /// `/Screen`: a region in which media clips play.
    Screen,
}

impl MediaKind {
    /// The kind of an annotation with `/Subtype` `subtype`; `None` for a
    /// subtype without interactive content.
    const fn of(subtype: &[u8]) -> Option<Self> {
        match subtype {
            b"3D" => Some(Self::ThreeD),
            b"RichMedia" => Some(Self::RichMedia),
            b"Movie" => Some(Self::Movie),
            b"Sound" => Some(Self::Sound),
            b"Screen" => Some(Self::Screen),
            _ => None,
        }
    }

    /// The annotation `/Subtype` name.
    #[must_use]
    pub const fn subtype(self) -> &'static str {
        match self {
            Self::ThreeD => "3D",
            Self::RichMedia => "RichMedia",
            Self::Movie => "Movie",
            Self::Sound => "Sound",
            Self::Screen => "Screen",
        }
    }
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.subtype())
    }
}

impl fmt::Display for ScriptLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenAction => f.write_str("/OpenAction"),
            Self::DocumentActions => f.write_str("/AA (catalog additional actions)"),
            Self::NameTree => f.write_str("/Names/JavaScript"),
            Self::FormActions => f.write_str("/AcroForm/AA"),
            Self::PageActions => f.write_str("page /AA"),
            Self::AnnotationAction { index } => write!(f, "annotation {index} /A"),
            Self::AnnotationActions { index } => write!(f, "annotation {index} /AA"),
        }
    }
}

impl fmt::Display for UnsupportedFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JavaScript(location) => write!(f, "JavaScript ({location})"),
            Self::MediaAnnotation {
                kind,
                index,
                has_appearance,
            } => {
                let drawn = if *has_appearance {
                    "normal appearance only"
                } else {
                    "no appearance"
                };
                write!(f, "/{kind} annotation {index} ({drawn})")
            }
            Self::Xfa { dynamic: true } => f.write_str("dynamic XFA form"),
            Self::Xfa { dynamic: false } => f.write_str("static XFA form"),
            Self::Portfolio => f.write_str("portable collection"),
        }
    }
}

impl UnsupportedFeature {
    /// `true` for a [`JavaScript`](Self::JavaScript) action.
    #[must_use]
    pub const fn is_javascript(&self) -> bool {
        matches!(self, Self::JavaScript(_))
    }
}

/// Upper bound on the annotations [`page_unsupported_features`] inspects on
/// one page.
const MAX_SCAN_ANNOTS: usize = 8192;

/// Upper bound on the actions inspected from one action entry, following
/// its `/Next` chain.
const MAX_ACTION_CHAIN: usize = 64;

/// The document-level features of `doc` that the renderer ignores.
///
/// They come in the order: JavaScript at [`ScriptLocation::OpenAction`],
/// [`DocumentActions`](ScriptLocation::DocumentActions),
/// [`NameTree`](ScriptLocation::NameTree) and
/// [`FormActions`](ScriptLocation::FormActions); an XFA form; a portable
/// collection.  Only the catalog and the objects it names are read; empty
/// when the catalog is unreadable.
#[must_use]
pub fn document_unsupported_features(doc: &Document) -> Vec<UnsupportedFeature> {
    let mut found = Vec::new();
    let Ok(catalog) = doc.catalog() else {
        return found;
    };
    let mut script = |location| found.push(UnsupportedFeature::JavaScript(location));

    if catalog
        .get(b"OpenAction")
        .is_some_and(|a| action_has_js(doc, a))
    {
        script(ScriptLocation::OpenAction);
    }
    if catalog
        .get(b"AA")
        .is_some_and(|aa| aa_dict_has_js_action(doc, aa))
    {
        script(ScriptLocation::DocumentActions);
    }
    // `/Names` also holds trees that are not scripts (`/Dests`,
    // `/EmbeddedFiles`); only the `/JavaScript` tree is.
    if catalog
        .get(b"Names")
        .and_then(|n| resources::resolve_dict(doc, n))
        .is_some_and(|names| names.get(b"JavaScript").is_some())
    {
        script(ScriptLocation::NameTree);
    }
    if catalog
        .get(b"AcroForm")
        .and_then(|a| resources::resolve_dict(doc, a))
        .and_then(|form| form.get(b"AA").cloned())
        .is_some_and(|aa| aa_dict_has_js_action(doc, &aa))
    {
        script(ScriptLocation::FormActions);
    }

    match form_type(doc) {
        FormType::XfaFull => found.push(UnsupportedFeature::Xfa { dynamic: true }),
        FormType::XfaForeground => found.push(UnsupportedFeature::Xfa { dynamic: false }),
        _ => {}
    }
    if catalog
        .get(b"Collection")
        .and_then(|c| resources::resolve_dict(doc, c))
        .is_some()
    {
        found.push(UnsupportedFeature::Portfolio);
    }
    found
}

/// The features of page `page_id` that the renderer ignores: JavaScript in
/// the page's `/AA`, then, per annotation in `/Annots` order, a
/// [`MediaAnnotation`](UnsupportedFeature::MediaAnnotation) and JavaScript
/// in its `/A` and in its `/AA`.
///
/// An unreadable page or annotation contributes nothing, and at most
/// 8 192 annotations are inspected.
#[must_use]
pub fn page_unsupported_features(doc: &Document, page_id: ObjectId) -> Vec<UnsupportedFeature> {
    let mut found = Vec::new();
    let Ok(page) = doc.get_dict_arc(page_id) else {
        return found;
    };
    let Some(page) = page.as_dict() else {
        return found;
    };
    if page
        .get(b"AA")
        .is_some_and(|aa| aa_dict_has_js_action(doc, aa))
    {
        found.push(UnsupportedFeature::JavaScript(ScriptLocation::PageActions));
    }
    let annots = resources::resolve_array(doc, page.get(b"Annots"));
    for (index, annot) in annots.iter().enumerate().take(MAX_SCAN_ANNOTS) {
        let Some(annot) = resources::resolve_dict(doc, annot) else {
            continue;
        };
        if let Some(kind) = annot.get_name(b"Subtype").and_then(MediaKind::of) {
            let has_appearance = annot
                .get(b"AP")
                .and_then(|ap| resources::resolve_dict(doc, ap))
                .is_some_and(|ap| ap.get(b"N").is_some());
            found.push(UnsupportedFeature::MediaAnnotation {
                kind,
                index,
                has_appearance,
            });
        }
        if annot.get(b"A").is_some_and(|a| action_has_js(doc, a)) {
            found.push(UnsupportedFeature::JavaScript(
                ScriptLocation::AnnotationAction { index },
            ));
        }
        if annot
            .get(b"AA")
            .is_some_and(|aa| aa_dict_has_js_action(doc, aa))
        {
            found.push(UnsupportedFeature::JavaScript(
                ScriptLocation::AnnotationActions { index },
            ));
        }
    }
    found
}

/// True when page `page_id` carries a JavaScript action: one of the
/// [`ScriptLocation::PageActions`], [`ScriptLocation::AnnotationAction`]
/// or [`ScriptLocation::AnnotationActions`] features of
/// [`page_unsupported_features`].
#[must_use]
pub fn page_has_javascript(doc: &Document, page_id: ObjectId) -> bool {
    page_unsupported_features(doc, page_id)
        .iter()
        .any(UnsupportedFeature::is_javascript)
}

/// Warn, once per feature, about the document-level features of `doc`
/// that change what renders or that carry scripts: JavaScript, a dynamic
/// XFA form and a portable collection.  A static XFA form renders its
/// page content as usual and is not warned about.
pub(crate) fn warn_document_features(doc: &Document) {
    for feature in document_unsupported_features(doc) {
        match feature {
            UnsupportedFeature::JavaScript(location) => log::warn!(
                "document contains JavaScript ({location}); JavaScript is ignored — \
                 rasterrocket does not execute scripts, rendering static content only"
            ),
            UnsupportedFeature::Xfa { dynamic: true } => log::warn!(
                "document is a dynamic XFA form (/NeedsRendering true); XFA is not processed — \
                 the rendered pages are the placeholder content stored in the file"
            ),
            UnsupportedFeature::Portfolio => log::warn!(
                "document is a portable collection (/Collection); the documents it carries \
                 are not rendered — the rendered pages are its cover pages"
            ),
            _ => {}
        }
    }
}

/// Whether the action `obj`, or an action of its `/Next` chain, runs
/// JavaScript: a `/S /JavaScript` action, or a `/S /Rendition` action with
/// a `/JS` script (ISO 32000-2 §12.6.4.14, §12.6.4.17).  At most 64
/// actions are inspected.
fn action_has_js(doc: &Document, obj: &Object) -> bool {
    let mut pending = vec![obj.clone()];
    let mut inspected = 0;
    while let Some(obj) = pending.pop() {
        if inspected == MAX_ACTION_CHAIN {
            break;
        }
        inspected += 1;
        let Some(action) = resources::resolve_dict(doc, &obj) else {
            continue;
        };
        match action.get_name(b"S") {
            Some(b"JavaScript") => return true,
            Some(b"Rendition") if action.get(b"JS").is_some() => return true,
            _ => {}
        }
        match action.get(b"Next") {
            Some(Object::Array(next)) => pending.extend(next.iter().rev().cloned()),
            Some(next) => pending.push(next.clone()),
            None => {}
        }
    }
    false
}

/// Whether `aa_obj` resolves to an additional-actions dictionary in which
/// any trigger (`/O`, `/C`, `/WC`, `/K`, …) holds a JavaScript action.
/// Bare `/AA` presence is not JavaScript: its actions are as often
/// `/GoTo`, `/SubmitForm` or `/Named` actions.
fn aa_dict_has_js_action(doc: &Document, aa_obj: &Object) -> bool {
    let Some(aa) = resources::resolve_dict(doc, aa_obj) else {
        return false;
    };
    aa.iter()
        .any(|(_trigger, action)| action_has_js(doc, action))
}

#[cfg(test)]
mod tests {
    use pdf::{Document, ObjectId};

    use super::{
        MediaKind, ScriptLocation, UnsupportedFeature, document_unsupported_features,
        page_has_javascript, page_unsupported_features, warn_document_features,
    };
    use crate::test_helpers::{make_doc, make_doc_with_objects};

    use ScriptLocation::{
        AnnotationAction, AnnotationActions, DocumentActions, FormActions, NameTree, OpenAction,
        PageActions,
    };
    use UnsupportedFeature::JavaScript;

    /// No features.
    const NONE: [UnsupportedFeature; 0] = [];

    fn doc_features(catalog: &str) -> Vec<UnsupportedFeature> {
        document_unsupported_features(&make_doc(catalog))
    }

    #[test]
    fn open_action_javascript_is_reported_and_warned() {
        let doc = make_doc(" /OpenAction <</S /JavaScript>>");
        assert_eq!(
            document_unsupported_features(&doc),
            [JavaScript(OpenAction)]
        );
        warn_document_features(&doc);
    }

    #[test]
    fn open_action_goto_is_not_javascript() {
        assert_eq!(doc_features(" /OpenAction <</S /GoTo>>"), NONE);
        assert_eq!(doc_features(" /OpenAction [0 /Fit]"), NONE);
    }

    #[test]
    fn javascript_in_a_next_chain_is_found() {
        assert_eq!(
            doc_features(" /OpenAction <</S /GoTo /Next [<</S /Named>> <</S /JavaScript>>]>>"),
            [JavaScript(OpenAction)]
        );
        assert_eq!(
            doc_features(" /OpenAction <</S /GoTo /Next <</S /JavaScript>>>>"),
            [JavaScript(OpenAction)]
        );
    }

    #[test]
    fn a_rendition_action_runs_its_script() {
        assert_eq!(
            doc_features(" /OpenAction <</S /Rendition /JS (x)>>"),
            [JavaScript(OpenAction)]
        );
        assert_eq!(doc_features(" /OpenAction <</S /Rendition /OP 0>>"), NONE);
    }

    #[test]
    fn all_four_document_script_locations_are_reported_in_order() {
        let doc = make_doc(
            " /OpenAction <</S /JavaScript>> /AA <</WC <</S /JavaScript>>>> \
             /Names <</JavaScript <</Names []>>>> /AcroForm <</AA <</K <</S /JavaScript>>>>>>",
        );
        assert_eq!(
            document_unsupported_features(&doc),
            [
                JavaScript(OpenAction),
                JavaScript(DocumentActions),
                JavaScript(NameTree),
                JavaScript(FormActions),
            ]
        );
        warn_document_features(&doc);
    }

    #[test]
    fn additional_actions_without_a_script_are_not_javascript() {
        assert_eq!(doc_features(" /AA <</WC <</S /GoTo /D [0 /Fit]>>>>"), NONE);
        assert_eq!(
            doc_features(" /AcroForm <</AA <</C <</S /SubmitForm /F (https://example/x)>>>>>>"),
            NONE
        );
        assert_eq!(doc_features(" /Names <</Dests <</Names []>>>>"), NONE);
        assert_eq!(doc_features(""), NONE);
    }

    #[test]
    fn additional_actions_with_one_script_among_others_are_javascript() {
        assert_eq!(
            doc_features(" /AA <</WC <</S /GoTo /D [0 /Fit]>> /O <</S /JavaScript>>>>"),
            [JavaScript(DocumentActions)]
        );
    }

    #[test]
    fn xfa_and_collections_are_reported() {
        assert_eq!(
            doc_features(" /AcroForm <</Fields [] /XFA []>> /NeedsRendering true"),
            [UnsupportedFeature::Xfa { dynamic: true }]
        );
        assert_eq!(
            doc_features(" /AcroForm <</Fields [] /XFA []>>"),
            [UnsupportedFeature::Xfa { dynamic: false }]
        );
        assert_eq!(doc_features(" /AcroForm <</Fields []>>"), NONE);
        assert_eq!(
            doc_features(" /Collection <</Type /Collection /View /D>>"),
            [UnsupportedFeature::Portfolio]
        );
    }

    /// The page of [`page_doc`].
    const PAGE: ObjectId = (4, 0);

    /// A document holding a page (object 4, outside the page tree, which
    /// the scan does not consult) that carries the verbatim `page_extra`
    /// and `/Annots` holding `annots` as objects 5, 6, ….
    fn page_doc(page_extra: &str, annots: &[&str]) -> Document {
        let refs: Vec<String> = (5..)
            .take(annots.len())
            .map(|id| format!("{id} 0 R"))
            .collect();
        let page = format!("<</Type /Page /Annots [{}]{page_extra}>>", refs.join(" "));
        let mut objects = vec![(4, page)];
        objects.extend((5..).zip(annots).map(|(id, &a)| (id, a.to_owned())));
        make_doc_with_objects(&objects)
    }

    #[test]
    fn page_actions_javascript_is_reported_on_the_page_only() {
        let doc = page_doc(" /AA <</O <</S /JavaScript>>>>", &[]);
        assert_eq!(
            page_unsupported_features(&doc, PAGE),
            [JavaScript(PageActions)]
        );
        assert!(page_has_javascript(&doc, PAGE));
        assert_eq!(document_unsupported_features(&doc), NONE);
    }

    #[test]
    fn annotation_scripts_are_reported_with_their_index() {
        let doc = page_doc(
            "",
            &[
                "<</Type /Annot /Subtype /Link /A <</S /URI /URI (https://example/x)>>>>",
                "<</Type /Annot /Subtype /Link /A <</S /JavaScript>>>>",
                "<</Type /Annot /Subtype /Widget /AA <</K <</S /JavaScript>>>>>>",
            ],
        );
        assert_eq!(
            page_unsupported_features(&doc, PAGE),
            [
                JavaScript(AnnotationAction { index: 1 }),
                JavaScript(AnnotationActions { index: 2 }),
            ]
        );
    }

    #[test]
    fn non_script_page_and_annotation_actions_are_not_reported() {
        let doc = page_doc(
            " /AA <</O <</S /GoTo /D [0 /Fit]>>>>",
            &["<</Type /Annot /Subtype /Link /A <</S /URI /URI (https://example/x)>>>>"],
        );
        assert_eq!(page_unsupported_features(&doc, PAGE), NONE);
        assert!(!page_has_javascript(&doc, PAGE));
    }

    #[test]
    fn media_annotations_are_reported_with_their_appearance() {
        let doc = page_doc(
            "",
            &[
                "<</Type /Annot /Subtype /3D /Rect [0 0 10 10] /AP <</N 9 0 R>>>>",
                "<</Type /Annot /Subtype /RichMedia /Rect [0 0 10 10]>>",
                "<</Type /Annot /Subtype /Screen /Rect [0 0 10 10] \
                 /AA <</PO <</S /JavaScript>>>>>>",
                "<</Type /Annot /Subtype /Movie /Rect [0 0 10 10]>>",
                "<</Type /Annot /Subtype /Sound /Rect [0 0 10 10]>>",
                "<</Type /Annot /Subtype /Square /Rect [0 0 10 10]>>",
            ],
        );
        let media = |kind, index, has_appearance| UnsupportedFeature::MediaAnnotation {
            kind,
            index,
            has_appearance,
        };
        assert_eq!(
            page_unsupported_features(&doc, PAGE),
            [
                media(MediaKind::ThreeD, 0, true),
                media(MediaKind::RichMedia, 1, false),
                media(MediaKind::Screen, 2, false),
                JavaScript(AnnotationActions { index: 2 }),
                media(MediaKind::Movie, 3, false),
                media(MediaKind::Sound, 4, false),
            ]
        );
    }

    #[test]
    fn features_display_their_location() {
        assert_eq!(
            JavaScript(AnnotationActions { index: 3 }).to_string(),
            "JavaScript (annotation 3 /AA)"
        );
        assert_eq!(
            UnsupportedFeature::MediaAnnotation {
                kind: MediaKind::ThreeD,
                index: 0,
                has_appearance: true,
            }
            .to_string(),
            "/3D annotation 0 (normal appearance only)"
        );
    }
}
