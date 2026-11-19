//! Features a document uses that the renderer does not process: JavaScript
//! actions, 3D and multimedia annotations, XFA forms and portable
//! collections.
//!
//! Detection is structural.  Only dictionary keys and `/S` and `/Subtype`
//! names are read; no script, model or media stream is decoded.

use std::collections::{HashMap, HashSet};
use std::fmt;

use pdf::{Dictionary, Document, Object, ObjectId};

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
    /// The additional actions (`/AA`) of a field in the interactive form's
    /// field tree: an entry of `/AcroForm /Fields` or a descendant reached
    /// through `/Kids`.
    FormFields,
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
    /// The additional actions (`/AA`) of a form field that is an ancestor
    /// of an annotation through its `/Parent` chain: the keystroke,
    /// format, validate and calculate scripts a non-terminal field passes
    /// down to its widgets.
    FieldActions {
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
            Self::FormFields => f.write_str("/AcroForm/Fields field /AA"),
            Self::PageActions => f.write_str("page /AA"),
            Self::AnnotationAction { index } => write!(f, "annotation {index} /A"),
            Self::AnnotationActions { index } => write!(f, "annotation {index} /AA"),
            Self::FieldActions { index } => write!(f, "annotation {index} /Parent field /AA"),
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

/// Upper bound on the depth of the form field hierarchy followed: the
/// `/Parent` links walked up from one annotation, and the `/Kids` levels
/// walked down from `/AcroForm /Fields`.
const MAX_FIELD_DEPTH: usize = 64;

/// Upper bound on the steps of one scan: one per annotation, per action
/// inspected (a reuse of a shared action's verdict included) and per
/// additional-actions dictionary.
const MAX_SCAN_STEPS: usize = 1 << 20;

/// The document-level features of `doc` that the renderer ignores.
///
/// They come in the order: JavaScript at [`ScriptLocation::OpenAction`],
/// [`DocumentActions`](ScriptLocation::DocumentActions),
/// [`NameTree`](ScriptLocation::NameTree),
/// [`FormActions`](ScriptLocation::FormActions) and
/// [`FormFields`](ScriptLocation::FormFields); an XFA form; a portable
/// collection.  Only the catalog and the objects it names are read; empty
/// when the catalog is unreadable.  The action scan performs at most
/// 2^20 steps; past that bound it stops with a warning and the scripts
/// it has not reached are not reported.
#[must_use]
pub fn document_unsupported_features(doc: &Document) -> Vec<UnsupportedFeature> {
    let mut found = Vec::new();
    let Ok(catalog) = doc.catalog() else {
        return found;
    };
    let mut scan = ActionScan::new(doc, None, MAX_SCAN_STEPS);
    let mut script = |location| found.push(UnsupportedFeature::JavaScript(location));

    if catalog
        .get(b"OpenAction")
        .is_some_and(|a| scan.action_has_js(a))
    {
        script(ScriptLocation::OpenAction);
    }
    if catalog
        .get(b"AA")
        .is_some_and(|aa| scan.additional_actions_have_js(aa))
    {
        script(ScriptLocation::DocumentActions);
    }
    // `/Names` also holds trees that are not scripts (`/Dests`,
    // `/EmbeddedFiles`); only the `/JavaScript` tree is.
    if catalog
        .get(b"Names")
        .and_then(|names| {
            with_dict(doc, names, |names| {
                names
                    .get(b"JavaScript")
                    .is_some_and(|tree| name_tree_has_entries(doc, tree))
            })
        })
        .unwrap_or(false)
    {
        script(ScriptLocation::NameTree);
    }
    if catalog.get(b"AcroForm").is_some_and(|form| {
        with_dict(doc, form, |form| {
            form.get(b"AA")
                .is_some_and(|aa| scan.additional_actions_have_js(aa))
        })
        .unwrap_or(false)
    }) {
        script(ScriptLocation::FormActions);
    }
    if catalog.get(b"AcroForm").is_some_and(|form| {
        with_dict(doc, form, |form| {
            form.get(b"Fields")
                .is_some_and(|fields| scan.field_tree_has_js(fields))
        })
        .unwrap_or(false)
    }) {
        script(ScriptLocation::FormFields);
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

/// The features of page `page_id` that the renderer ignores.
///
/// They are JavaScript in the page's `/AA`, then, per annotation in
/// `/Annots` order, a
/// [`MediaAnnotation`](UnsupportedFeature::MediaAnnotation) and JavaScript
/// in its `/A`, in its `/AA` and in the `/AA` of a field on its `/Parent`
/// chain (at most 64 ancestors, each visited once).
///
/// An unreadable page or annotation contributes nothing, and at most
/// 8 192 annotations are inspected.  The scan performs at most 2^20
/// steps; past that bound it stops with a warning and returns the
/// features found so far.
#[must_use]
pub fn page_unsupported_features(doc: &Document, page_id: ObjectId) -> Vec<UnsupportedFeature> {
    page_features_within(doc, page_id, MAX_SCAN_STEPS)
}

/// [`page_unsupported_features`] with a budget of `steps` scan steps.
fn page_features_within(
    doc: &Document,
    page_id: ObjectId,
    steps: usize,
) -> Vec<UnsupportedFeature> {
    let mut found = Vec::new();
    let Ok(page) = doc.get_dict_arc(page_id) else {
        return found;
    };
    let Some(page) = page.as_dict() else {
        return found;
    };
    let mut scan = ActionScan::new(doc, Some(page_id), steps);
    if page
        .get(b"AA")
        .is_some_and(|aa| scan.additional_actions_have_js(aa))
    {
        found.push(UnsupportedFeature::JavaScript(ScriptLocation::PageActions));
    }
    let Some(annots) = page.get(b"Annots").and_then(|a| doc.resolve(a).ok()) else {
        return found;
    };
    let annots = annots.as_array().unwrap_or_default();
    for (index, annot_ref) in annots.iter().enumerate().take(MAX_SCAN_ANNOTS) {
        if !scan.step() {
            break;
        }
        let resolved = doc.resolve(annot_ref).ok();
        let Some(annot) = resolved.as_deref().and_then(Object::as_dict) else {
            continue;
        };
        if let Some(kind) = annot.get_name(b"Subtype").and_then(MediaKind::of) {
            let has_appearance = annot
                .get(b"AP")
                .and_then(|ap| with_dict(doc, ap, |ap| ap.get(b"N").is_some()))
                .unwrap_or(false);
            found.push(UnsupportedFeature::MediaAnnotation {
                kind,
                index,
                has_appearance,
            });
        }
        if annot.get(b"A").is_some_and(|a| scan.action_has_js(a)) {
            found.push(UnsupportedFeature::JavaScript(
                ScriptLocation::AnnotationAction { index },
            ));
        }
        if annot
            .get(b"AA")
            .is_some_and(|aa| scan.additional_actions_have_js(aa))
        {
            found.push(UnsupportedFeature::JavaScript(
                ScriptLocation::AnnotationActions { index },
            ));
        }
        let own = match annot_ref {
            Object::Reference(id) => Some(*id),
            _ => None,
        };
        if scan.ancestors_have_js(annot, own) {
            found.push(UnsupportedFeature::JavaScript(
                ScriptLocation::FieldActions { index },
            ));
        }
    }
    found
}

/// True when page `page_id` carries a JavaScript action.
///
/// That is one of the [`ScriptLocation::PageActions`],
/// [`ScriptLocation::AnnotationAction`],
/// [`ScriptLocation::AnnotationActions`] or
/// [`ScriptLocation::FieldActions`] features of
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

/// `f` applied to the dictionary `obj` resolves to, borrowed from the
/// document's object cache; `None` when `obj` resolves to no dictionary.
fn with_dict<R>(doc: &Document, obj: &Object, f: impl FnOnce(&Dictionary) -> R) -> Option<R> {
    let resolved = doc.resolve(obj).ok()?;
    resolved.as_dict().map(f)
}

/// Whether the name tree whose root is `root` holds an entry: the root has
/// a non-empty `/Names` array (a leaf) or a non-empty `/Kids` array (an
/// intermediate node, whose descendants are not read).
fn name_tree_has_entries(doc: &Document, root: &Object) -> bool {
    with_dict(doc, root, |root| {
        [b"Names".as_slice(), b"Kids"].iter().any(|key| {
            root.get(key)
                .and_then(|a| doc.resolve(a).ok())
                .is_some_and(|a| a.as_array().is_some_and(|a| !a.is_empty()))
        })
    })
    .unwrap_or(false)
}

/// The result of inspecting an action or an additional-actions dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// A JavaScript action was found.
    Script,
    /// Every action reachable from the object was inspected; none runs a
    /// script.
    NoScript,
    /// No script was found, but part of the object was not inspected: a
    /// bound was reached, or a reference led back to an object whose
    /// inspection was still in progress.
    Incomplete,
}

impl Verdict {
    /// The verdict on two parts of one object taken together.
    const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Script, _) | (_, Self::Script) => Self::Script,
            (Self::Incomplete, _) | (_, Self::Incomplete) => Self::Incomplete,
            (Self::NoScript, Self::NoScript) => Self::NoScript,
        }
    }
}

/// What a referenced object is inspected as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Role {
    /// An action dictionary.
    Action,
    /// An additional-actions dictionary.
    AdditionalActions,
}

/// The state of one scan for JavaScript actions: a step budget shared by
/// every action entry of the scan, and the verdicts on the referenced
/// objects already inspected.
///
/// A referenced object is inspected once per scan in each [`Role`]; later
/// references reuse its verdict.  Only a complete verdict is kept, so a
/// reference whose inspection was cut short is inspected again where it
/// recurs.  Every lookup costs one step, which bounds the scan's total
/// work by [`MAX_SCAN_STEPS`] whatever the document's sharing.
struct ActionScan<'d> {
    doc: &'d Document,
    /// The page scanned, `None` for the document catalog.
    page: Option<ObjectId>,
    /// Steps left before the scan stops.
    remaining: usize,
    /// Complete verdicts (`Some(has_script)`), and `None` for an object
    /// whose inspection is in progress.
    verdicts: HashMap<(Role, ObjectId), Option<bool>>,
    /// The exhaustion warning has been logged.
    warned: bool,
}

impl<'d> ActionScan<'d> {
    fn new(doc: &'d Document, page: Option<ObjectId>, steps: usize) -> Self {
        Self {
            doc,
            page,
            remaining: steps,
            verdicts: HashMap::new(),
            warned: false,
        }
    }

    /// The steps spent so far out of a budget of `steps`.
    #[cfg(test)]
    const fn spent(&self, steps: usize) -> usize {
        steps - self.remaining
    }

    /// Spend one step; `false` once the budget is spent.
    fn step(&mut self) -> bool {
        if self.exhausted() {
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Whether the step budget is spent, warning the first time it is
    /// found spent.
    fn exhausted(&mut self) -> bool {
        if self.remaining > 0 {
            return false;
        }
        if !self.warned {
            self.warned = true;
            match self.page {
                Some((num, generation)) => log::warn!(
                    "page {num} {generation} R: the unsupported-feature scan stopped after \
                     {MAX_SCAN_STEPS} steps; scripts and media annotations past that point \
                     are not reported (rendering is unaffected)"
                ),
                None => log::warn!(
                    "document catalog: the unsupported-feature scan stopped after \
                     {MAX_SCAN_STEPS} steps; scripts past that point are not reported \
                     (rendering is unaffected)"
                ),
            }
        }
        true
    }

    /// Whether the action `obj`, or an action of its `/Next` chain, runs
    /// JavaScript: a `/S /JavaScript` action, or a `/S /Rendition` action
    /// with a `/JS` script (ISO 32000-2 §12.6.4.14, §12.6.4.17).  At most
    /// [`MAX_ACTION_CHAIN`] actions are inspected.
    fn action_has_js(&mut self, obj: &Object) -> bool {
        let mut chain = MAX_ACTION_CHAIN;
        self.action(obj, &mut chain) == Verdict::Script
    }

    /// Whether `aa_obj` resolves to an additional-actions dictionary in
    /// which any trigger (`/O`, `/C`, `/WC`, `/K`, …) holds a JavaScript
    /// action.  Bare `/AA` presence is not JavaScript: its actions are as
    /// often `/GoTo`, `/SubmitForm` or `/Named` actions.
    fn additional_actions_have_js(&mut self, aa_obj: &Object) -> bool {
        if !self.step() {
            return false;
        }
        let verdict = self.memoised(Role::AdditionalActions, aa_obj, |scan, aa| {
            let Some(aa) = aa.as_dict() else {
                return Verdict::NoScript;
            };
            let mut verdict = Verdict::NoScript;
            for (_trigger, action) in aa {
                if scan.exhausted() {
                    return Verdict::Incomplete;
                }
                let mut chain = MAX_ACTION_CHAIN;
                verdict = verdict.and(scan.action(action, &mut chain));
                if verdict == Verdict::Script {
                    break;
                }
            }
            verdict
        });
        verdict == Verdict::Script
    }

    /// Whether a field on the `/Parent` chain of the annotation `annot`
    /// (object `own`, when indirect) holds a JavaScript action in its
    /// `/AA`.  At most [`MAX_FIELD_DEPTH`] ancestors are inspected, each
    /// object once, so a chain that leads back to an object already on
    /// it ends there.
    fn ancestors_have_js(&mut self, annot: &Dictionary, own: Option<ObjectId>) -> bool {
        let mut seen: HashSet<ObjectId> = own.into_iter().collect();
        let mut next = annot.get(b"Parent").cloned();
        for _ in 0..MAX_FIELD_DEPTH {
            let Some(parent) = next.take() else {
                return false;
            };
            if let Object::Reference(id) = parent
                && !seen.insert(id)
            {
                return false;
            }
            if !self.step() {
                return false;
            }
            let Ok(resolved) = self.doc.resolve(&parent) else {
                return false;
            };
            let Some(field) = resolved.as_dict() else {
                return false;
            };
            if field
                .get(b"AA")
                .is_some_and(|aa| self.additional_actions_have_js(aa))
            {
                return true;
            }
            next = field.get(b"Parent").cloned();
        }
        false
    }

    /// Whether a field of the field tree whose roots are the array
    /// `fields` (`/AcroForm /Fields`) holds a JavaScript action in its
    /// `/AA`.  The tree is walked depth-first through `/Kids`, at most
    /// [`MAX_FIELD_DEPTH`] levels deep; each indirect node is visited
    /// once, so a `/Kids` entry that leads back to a visited node is
    /// skipped.  Every node taken from the pending stack costs one step,
    /// and the stack never holds more nodes than steps remain.
    fn field_tree_has_js(&mut self, fields: &Object) -> bool {
        let Ok(roots) = self.doc.resolve(fields) else {
            return false;
        };
        let Some(roots) = roots.as_array() else {
            return false;
        };
        let mut seen = HashSet::new();
        let mut stack: Vec<(Object, usize)> = roots.iter().rev().map(|f| (f.clone(), 1)).collect();
        while let Some((node, depth)) = stack.pop() {
            if !self.step() {
                return false;
            }
            if let Object::Reference(id) = node
                && !seen.insert(id)
            {
                continue;
            }
            let Ok(resolved) = self.doc.resolve(&node) else {
                continue;
            };
            let Some(field) = resolved.as_dict() else {
                continue;
            };
            if field
                .get(b"AA")
                .is_some_and(|aa| self.additional_actions_have_js(aa))
            {
                return true;
            }
            if depth >= MAX_FIELD_DEPTH {
                continue;
            }
            let kids = field.get(b"Kids").and_then(|k| self.doc.resolve(k).ok());
            if let Some(kids) = kids.as_deref().and_then(Object::as_array) {
                let room = self.remaining.saturating_sub(stack.len()).min(kids.len());
                stack.extend(kids[..room].iter().rev().map(|k| (k.clone(), depth + 1)));
            }
        }
        false
    }

    /// The verdict on the action `obj` and its `/Next` chain, `chain`
    /// being the number of actions the chain may still inspect.  The
    /// recursion depth is at most [`MAX_ACTION_CHAIN`], since each level
    /// spends one unit of `chain`.
    fn action(&mut self, obj: &Object, chain: &mut usize) -> Verdict {
        if *chain == 0 || !self.step() {
            return Verdict::Incomplete;
        }
        *chain -= 1;
        self.memoised(Role::Action, obj, |scan, action| {
            let Some(action) = action.as_dict() else {
                return Verdict::NoScript;
            };
            match action.get_name(b"S") {
                Some(b"JavaScript") => return Verdict::Script,
                Some(b"Rendition") if action.get(b"JS").is_some() => return Verdict::Script,
                _ => {}
            }
            match action.get(b"Next") {
                Some(Object::Array(next)) => {
                    let mut verdict = Verdict::NoScript;
                    for next in next {
                        if *chain == 0 || scan.exhausted() {
                            return Verdict::Incomplete;
                        }
                        verdict = verdict.and(scan.action(next, chain));
                        if verdict == Verdict::Script {
                            break;
                        }
                    }
                    verdict
                }
                Some(next) => scan.action(next, chain),
                None => Verdict::NoScript,
            }
        })
    }

    /// The verdict `inspect` gives on the object `obj` resolves to.  For a
    /// reference the verdict is looked up first and recorded after, when
    /// complete; a reference to an object under inspection is
    /// [`Verdict::Incomplete`].  An unreadable reference holds no script.
    fn memoised(
        &mut self,
        role: Role,
        obj: &Object,
        inspect: impl FnOnce(&mut Self, &Object) -> Verdict,
    ) -> Verdict {
        let Object::Reference(id) = *obj else {
            return inspect(self, obj);
        };
        match self.verdicts.get(&(role, id)) {
            Some(Some(true)) => return Verdict::Script,
            Some(Some(false)) => return Verdict::NoScript,
            Some(None) => return Verdict::Incomplete,
            None => {}
        }
        let Ok(target) = self.doc.resolve(obj) else {
            let _ = self.verdicts.insert((role, id), Some(false));
            return Verdict::NoScript;
        };
        let _ = self.verdicts.insert((role, id), None);
        let verdict = inspect(self, &target);
        let _ = match verdict {
            Verdict::Script => self.verdicts.insert((role, id), Some(true)),
            Verdict::NoScript => self.verdicts.insert((role, id), Some(false)),
            Verdict::Incomplete => self.verdicts.remove(&(role, id)),
        };
        verdict
    }
}

#[cfg(test)]
mod tests {
    use pdf::{Document, ObjectId};

    use super::{
        ActionScan, MAX_ACTION_CHAIN, MAX_SCAN_STEPS, MediaKind, ScriptLocation,
        UnsupportedFeature, document_unsupported_features, page_features_within,
        page_has_javascript, page_unsupported_features, warn_document_features,
    };
    use crate::test_helpers::{make_doc, make_doc_with_catalog_and_objects, make_doc_with_objects};

    use ScriptLocation::{
        AnnotationAction, AnnotationActions, DocumentActions, FieldActions, FormActions,
        FormFields, NameTree, OpenAction, PageActions,
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
             /Names <</JavaScript <</Names [(init) <</S /JavaScript /JS (x)>>]>>>> \
             /AcroForm <</AA <</K <</S /JavaScript>>>>>>",
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
    fn only_a_script_name_tree_with_entries_is_javascript() {
        assert_eq!(doc_features(" /Names <</JavaScript <</Names []>>>>"), NONE);
        assert_eq!(doc_features(" /Names <</JavaScript <</Kids []>>>>"), NONE);
        assert_eq!(doc_features(" /Names <</JavaScript <<>>>>"), NONE);
        assert_eq!(
            doc_features(" /Names <</JavaScript <</Kids [<</Names [(a) <<>>]>>]>>>>"),
            [JavaScript(NameTree)]
        );
        assert_eq!(
            doc_features(" /Names <</JavaScript <</Names [(a) <</S /JavaScript>>]>>>>"),
            [JavaScript(NameTree)]
        );
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
    fn a_script_on_a_parent_field_is_reported_for_its_widget() {
        let doc = make_doc_with_objects(&[
            (4, "<</Type /Page /Annots [5 0 R 6 0 R]>>"),
            (5, "<</Type /Annot /Subtype /Widget /Parent 7 0 R>>"),
            (6, "<</Type /Annot /Subtype /Link>>"),
            (7, "<</T (sub) /Parent 8 0 R /Kids [5 0 R]>>"),
            (
                8,
                "<</T (top) /FT /Tx /Kids [7 0 R] \
                 /AA <</K <</S /JavaScript /JS (k)>> /F <</S /JavaScript /JS (f)>>>>>>",
            ),
        ]);
        assert_eq!(
            page_unsupported_features(&doc, PAGE),
            [JavaScript(FieldActions { index: 0 })]
        );
        assert!(page_has_javascript(&doc, PAGE));
    }

    #[test]
    fn a_parent_chain_without_a_script_is_not_javascript() {
        let doc = make_doc_with_objects(&[
            (4, "<</Type /Page /Annots [5 0 R]>>"),
            (5, "<</Type /Annot /Subtype /Widget /Parent 6 0 R>>"),
            (6, "<</T (top) /AA <</C <</S /SubmitForm>>>>>>"),
        ]);
        assert_eq!(page_unsupported_features(&doc, PAGE), NONE);
    }

    #[test]
    fn a_cyclic_parent_chain_ends() {
        let doc = make_doc_with_objects(&[
            (4, "<</Type /Page /Annots [5 0 R 6 0 R]>>"),
            (5, "<</Type /Annot /Subtype /Widget /Parent 7 0 R>>"),
            (6, "<</Type /Annot /Subtype /Widget /Parent 6 0 R>>"),
            (7, "<</T (a) /Parent 8 0 R>>"),
            (8, "<</T (b) /Parent 7 0 R>>"),
        ]);
        assert_eq!(page_unsupported_features(&doc, PAGE), NONE);
    }

    #[test]
    fn a_script_in_the_field_tree_is_reported_for_the_document() {
        let doc = make_doc_with_catalog_and_objects(
            " /AcroForm <</Fields [4 0 R]>>",
            &[
                (4, "<</T (top) /Kids [5 0 R]>>"),
                (5, "<</T (mid) /Kids [6 0 R]>>"),
                (
                    6,
                    "<</T (leaf) /Parent 5 0 R /AA <</V <</S /JavaScript>>>>>>",
                ),
            ],
        );
        assert_eq!(
            document_unsupported_features(&doc),
            [JavaScript(FormFields)]
        );
    }

    #[test]
    fn a_cyclic_field_tree_ends() {
        let doc = make_doc_with_catalog_and_objects(
            " /AcroForm <</Fields [4 0 R 4 0 R]>>",
            &[
                (4, "<</T (a) /Kids [5 0 R 4 0 R]>>"),
                (5, "<</T (b) /Kids [4 0 R 5 0 R] /AA <</K <</S /GoTo>>>>>>"),
            ],
        );
        assert_eq!(document_unsupported_features(&doc), NONE);
    }

    /// The reference `n 0 R`.
    const fn reference(n: u32) -> pdf::Object {
        pdf::Object::Reference((n, 0))
    }

    #[test]
    fn a_shared_action_is_inspected_once_per_scan() {
        let next = vec!["<</S /Named /N /NextPage>>"; 50].join(" ");
        let doc = page_doc("", &[&format!("<</S /GoTo /Next [{next}]>>")]);
        let mut scan = ActionScan::new(&doc, None, MAX_SCAN_STEPS);
        assert!(!scan.action_has_js(&reference(5)));
        assert_eq!(scan.spent(MAX_SCAN_STEPS), 51);
        assert!(!scan.action_has_js(&reference(5)));
        assert_eq!(scan.spent(MAX_SCAN_STEPS), 52);
    }

    #[test]
    fn a_shared_script_is_reported_at_every_reference() {
        let doc = page_doc(
            "",
            &[
                "<</Type /Annot /Subtype /Link /A 7 0 R>>",
                "<</Type /Annot /Subtype /Link /A 7 0 R>>",
                "<</S /GoTo /Next [<</S /Named>> <</S /JavaScript>>]>>",
            ],
        );
        assert_eq!(
            page_unsupported_features(&doc, PAGE),
            [
                JavaScript(AnnotationAction { index: 0 }),
                JavaScript(AnnotationAction { index: 1 }),
            ]
        );
    }

    #[test]
    fn a_shared_action_with_a_huge_next_array_costs_one_chain_per_reference() {
        const ANNOTS: u32 = 256;
        const ACTION: u32 = 5 + ANNOTS;
        let next = vec!["<<>>"; 200_000].join(" ");
        let action = format!("<</S /GoTo /Next [{next} <</S /JavaScript>>]>>");
        let annot = format!("<</Type /Annot /Subtype /Link /A {ACTION} 0 R>>");
        let mut objects: Vec<&str> = (0..ANNOTS).map(|_| annot.as_str()).collect();
        objects.push(&action);
        let doc = page_doc("", &objects);
        let mut scan = ActionScan::new(&doc, None, MAX_SCAN_STEPS);
        assert!(!scan.action_has_js(&reference(ACTION)));
        assert_eq!(scan.spent(MAX_SCAN_STEPS), MAX_ACTION_CHAIN);
        assert_eq!(page_unsupported_features(&doc, PAGE), NONE);
    }

    #[test]
    fn an_exhausted_budget_stops_the_scan_and_keeps_what_it_found() {
        let script = "<</Type /Annot /Subtype /Link /A <</S /JavaScript>>>>";
        let doc = page_doc("", &[script, script, script]);
        assert_eq!(
            page_features_within(&doc, PAGE, 4),
            [
                JavaScript(AnnotationAction { index: 0 }),
                JavaScript(AnnotationAction { index: 1 }),
            ]
        );
        assert_eq!(page_features_within(&doc, PAGE, 0), NONE);
    }

    #[test]
    fn features_display_their_location() {
        assert_eq!(
            JavaScript(AnnotationActions { index: 3 }).to_string(),
            "JavaScript (annotation 3 /AA)"
        );
        assert_eq!(
            JavaScript(FieldActions { index: 2 }).to_string(),
            "JavaScript (annotation 2 /Parent field /AA)"
        );
        assert_eq!(
            JavaScript(FormFields).to_string(),
            "JavaScript (/AcroForm/Fields field /AA)"
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
