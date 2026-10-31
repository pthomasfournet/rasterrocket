//! Optional content visibility (ISO 32000-2 §8.11).
//!
//! [`OptionalContent`] holds a document's default optional-content
//! configuration (`/OCProperties /D`) resolved for one [`RenderPurpose`], and
//! decides whether content tagged with an optional content group (OCG) or an
//! optional content membership dictionary (OCMD) is drawn.
//!
//! The state of a group is, in order of application:
//! 1. the configuration's `/BaseState` (`ON` unless `OFF`);
//! 2. `ON` when the group is listed in `/ON`;
//! 3. `OFF` when it is listed in `/OFF`;
//! 4. for each usage application dictionary of `/AS` whose `/Event` is the
//!    purpose's event and whose `/OCGs` lists the group, the group's `/Usage`
//!    state for each `/Category` of `View`, `Print` or `Export` that the group
//!    declares (`ViewState`, `PrintState`, `ExportState`).
//!
//! A group whose `/Intent` shares no name with the configuration's `/Intent`
//! (both default to `View`; `All` matches every intent) does not take part in
//! visibility and counts as `ON`.  Content without a resolvable group or
//! membership dictionary, and every content of a document without
//! `/OCProperties`, is visible.

use std::collections::{HashMap, HashSet};

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;
use crate::resources::{resolve_array, resolve_dict};

/// The use a rendering is made for, which selects the optional-content usage
/// event (§8.11.4.4) and the annotation flags (§12.5.3) that apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RenderPurpose {
    /// On-screen viewing: the `View` usage event applies, and annotations
    /// flagged `NoView` are not drawn.
    View,
    /// Printing: the `Print` usage event applies, and only annotations
    /// flagged `Print` are drawn.
    Print,
}

impl RenderPurpose {
    /// The `/Event` name of a usage application dictionary that applies.
    const fn event(self) -> &'static [u8] {
        match self {
            Self::View => b"View",
            Self::Print => b"Print",
        }
    }
}

/// Maximum nesting depth of a visibility expression (`/VE`).
///
/// Bounds the recursion of a cyclic expression built from indirect arrays.
const MAX_VE_DEPTH: u32 = 32;

/// The values of the indirect operands of one visibility expression, by
/// object.
type Memo = HashMap<ObjectId, Option<bool>>;

/// A usage dictionary category that sets a group's state: the `/Usage`
/// sub-dictionary key and its state entry.
const USAGE_STATES: [(&[u8], &[u8]); 3] = [
    (b"View", b"ViewState"),
    (b"Print", b"PrintState"),
    (b"Export", b"ExportState"),
];

/// The default optional-content configuration of a document, resolved for
/// one [`RenderPurpose`].
#[derive(Debug)]
pub struct OptionalContent {
    purpose: RenderPurpose,
    /// `None` when the document has no usable `/OCProperties /D`.
    config: Option<Config>,
    /// The state of every group `/OCProperties /OCGs` lists, resolved once;
    /// a group the array omits is resolved on each query.
    states: HashMap<ObjectId, bool>,
}

/// The parts of a configuration dictionary (§8.11.4.3) that decide the state
/// of a group.
#[derive(Debug)]
struct Config {
    base_on: bool,
    on: HashSet<ObjectId>,
    off: HashSet<ObjectId>,
    /// The configuration's `/Intent` names.
    intents: Vec<Vec<u8>>,
    /// The usage application dictionaries of `/AS` for the purpose's event:
    /// the groups each lists, and the indices into [`USAGE_STATES`] of the
    /// categories it names.
    usage: Vec<(HashSet<ObjectId>, Vec<usize>)>,
}

impl OptionalContent {
    /// Resolve the default configuration of `doc` for `purpose`.
    #[must_use]
    pub fn new(doc: &Document, purpose: RenderPurpose) -> Self {
        let properties = doc
            .catalog()
            .ok()
            .and_then(|catalog| resolve_dict(doc, catalog.get(b"OCProperties")?));
        let config = properties
            .as_ref()
            .and_then(|props| resolve_dict(doc, props.get(b"D")?))
            .map(|d| Config::new(doc, &d, purpose));
        let mut this = Self {
            purpose,
            config,
            states: HashMap::new(),
        };
        if this.config.is_some() {
            let groups = reference_set(doc, properties.as_ref().and_then(|p| p.get(b"OCGs")));
            this.states = groups
                .into_iter()
                .map(|id| (id, this.resolve_group(doc, id)))
                .collect();
        }
        this
    }

    /// The purpose this configuration was resolved for.
    #[must_use]
    pub const fn purpose(&self) -> RenderPurpose {
        self.purpose
    }

    /// Whether content governed by `oc` is visible: `oc` is the value of an
    /// `/OC` entry or of the `Properties` resource a `BDC /OC` names, an
    /// optional content group or membership dictionary, direct or indirect.
    #[must_use]
    pub fn is_visible(&self, doc: &Document, oc: &Object) -> bool {
        if self.config.is_none() {
            return true;
        }
        match oc {
            Object::Reference(id) => {
                if let Some(&on) = self.states.get(id) {
                    return on;
                }
                match doc.get_object(*id).ok().as_deref() {
                    Some(Object::Dictionary(d)) if d.get_name(b"Type") == Some(b"OCMD") => {
                        self.membership_visible(doc, d)
                    }
                    Some(Object::Dictionary(_)) => self.resolve_group(doc, *id),
                    _ => true,
                }
            }
            Object::Dictionary(d) if d.get_name(b"Type") == Some(b"OCMD") => {
                self.membership_visible(doc, d)
            }
            _ => true,
        }
    }

    /// The state of the group `id`: as resolved at construction when
    /// `/OCGs` lists it, else resolved now.
    fn group_on(&self, doc: &Document, id: ObjectId) -> bool {
        self.states
            .get(&id)
            .copied()
            .unwrap_or_else(|| self.resolve_group(doc, id))
    }

    /// Resolve the state of the group `id` (§8.11.4.3); `true` when it does
    /// not take part in visibility.
    fn resolve_group(&self, doc: &Document, id: ObjectId) -> bool {
        let Some(config) = &self.config else {
            return true;
        };
        let group = doc.get_dictionary(id).ok();
        if let Some(group) = &group
            && !config.considers(doc, group)
        {
            return true;
        }
        let mut on = if config.off.contains(&id) {
            false
        } else if config.on.contains(&id) {
            true
        } else {
            config.base_on
        };
        let Some(usage) = group
            .as_ref()
            .and_then(|g| resolve_dict(doc, g.get(b"Usage")?))
        else {
            return on;
        };
        for (groups, categories) in &config.usage {
            if !groups.contains(&id) {
                continue;
            }
            for &c in categories {
                let (key, state_key) = USAGE_STATES[c];
                if let Some(state) = resolve_dict(doc, usage.get(key).unwrap_or(&Object::Null))
                    .and_then(|d| d.get_name(state_key).map(|s| s != b"OFF"))
                {
                    on = state;
                }
            }
        }
        on
    }

    /// Visibility under an optional content membership dictionary
    /// (§8.11.2.2): its `/VE` expression when usable, otherwise its `/P`
    /// policy over `/OCGs`.  No usable group makes it visible.
    fn membership_visible(&self, doc: &Document, ocmd: &Dictionary) -> bool {
        if let Some(visible) = ocmd
            .get(b"VE")
            .and_then(|ve| self.expression(doc, ve, &mut Memo::new()))
        {
            return visible;
        }
        let ids: Vec<ObjectId> = match ocmd.get(b"OCGs") {
            Some(Object::Reference(id)) => match doc.get_object(*id).ok().as_deref() {
                Some(Object::Array(a)) => references(a),
                Some(Object::Dictionary(_)) => vec![*id],
                _ => Vec::new(),
            },
            Some(Object::Array(a)) => references(a),
            _ => Vec::new(),
        };
        let ids: Vec<ObjectId> = ids
            .into_iter()
            .filter(|id| doc.get_dictionary(*id).is_ok())
            .collect();
        if ids.is_empty() {
            return true;
        }
        let mut states = ids.iter().map(|&id| self.group_on(doc, id));
        match ocmd.get_name(b"P") {
            Some(b"AllOn") => states.all(|on| on),
            Some(b"AnyOff") => states.any(|on| !on),
            Some(b"AllOff") => states.all(|on| !on),
            _ => states.any(|on| on),
        }
    }

    /// Evaluate the visibility expression `ve` (§8.11.2.2), an array or a
    /// reference to one: `[/And e…]`, `[/Or e…]` or `[/Not e]`, each operand
    /// a group or an expression.  `None` when it is malformed or nested
    /// deeper than [`MAX_VE_DEPTH`].
    fn expression(&self, doc: &Document, ve: &Object, memo: &mut Memo) -> Option<bool> {
        match ve {
            Object::Array(items) => self.evaluate(doc, items, 0, memo),
            Object::Reference(id) => match doc.get_object(*id).ok()?.as_ref() {
                Object::Array(items) => self.evaluate(doc, items, 0, memo),
                _ => None,
            },
            _ => None,
        }
    }

    /// The value of the expression whose operator and operands are `items`,
    /// at nesting `depth`.
    fn evaluate(
        &self,
        doc: &Document,
        items: &[Object],
        depth: u32,
        memo: &mut Memo,
    ) -> Option<bool> {
        if depth > MAX_VE_DEPTH {
            return None;
        }
        let (op, operands) = items.split_first()?;
        let mut values = operands
            .iter()
            .map(|operand| self.operand(doc, operand, depth, memo));
        match op.as_name()? {
            b"Not" if operands.len() == 1 => values.next()?.map(|v| !v),
            b"And" if !operands.is_empty() => values.try_fold(true, |acc, v| Some(acc & v?)),
            b"Or" if !operands.is_empty() => values.try_fold(false, |acc, v| Some(acc | v?)),
            _ => None,
        }
    }

    /// The value of one operand of an expression at nesting `depth`: a
    /// group or a nested expression.  The value of an indirect operand is
    /// computed once per evaluation and kept in `memo`, so an expression
    /// that shares subexpressions by reference costs time linear in its
    /// distinct objects.  A `None` stored there is never read back: it
    /// makes the whole expression `None`.
    fn operand(
        &self,
        doc: &Document,
        operand: &Object,
        depth: u32,
        memo: &mut Memo,
    ) -> Option<bool> {
        match operand {
            Object::Reference(id) => {
                if let Some(&value) = memo.get(id) {
                    return value;
                }
                let value = match doc.get_object(*id).ok()?.as_ref() {
                    Object::Dictionary(_) => Some(self.group_on(doc, *id)),
                    Object::Array(items) => self.evaluate(doc, items, depth + 1, memo),
                    _ => None,
                };
                *memo.entry(*id).or_insert(value)
            }
            Object::Array(items) => self.evaluate(doc, items, depth + 1, memo),
            _ => None,
        }
    }
}

impl Config {
    fn new(doc: &Document, d: &Dictionary, purpose: RenderPurpose) -> Self {
        let usage = resolve_array(doc, d.get(b"AS"))
            .iter()
            .filter_map(|a| resolve_dict(doc, a))
            .filter(|a| a.get_name(b"Event") == Some(purpose.event()))
            .map(|a| {
                let groups = reference_set(doc, a.get(b"OCGs"));
                let categories = names(doc, a.get(b"Category"))
                    .iter()
                    .filter_map(|n| USAGE_STATES.iter().position(|(key, _)| key == n))
                    .collect();
                (groups, categories)
            })
            .collect();
        let intents = names(doc, d.get(b"Intent"));
        Self {
            base_on: d.get_name(b"BaseState") != Some(b"OFF"),
            on: reference_set(doc, d.get(b"ON")),
            off: reference_set(doc, d.get(b"OFF")),
            intents: if intents.is_empty() {
                vec![b"View".to_vec()]
            } else {
                intents
            },
            usage,
        }
    }

    /// Whether the group's `/Intent` (default `View`) shares a name with the
    /// configuration's, `All` matching every name (§8.11.2.3).
    fn considers(&self, doc: &Document, group: &Dictionary) -> bool {
        let mut intents = names(doc, group.get(b"Intent"));
        if intents.is_empty() {
            intents.push(b"View".to_vec());
        }
        let all = |v: &[Vec<u8>]| v.iter().any(|n| n == b"All");
        all(&self.intents) || all(&intents) || intents.iter().any(|n| self.intents.contains(n))
    }
}

/// The indirect references among `items`.
fn references(items: &[Object]) -> Vec<ObjectId> {
    items
        .iter()
        .filter_map(|o| match o {
            Object::Reference(id) => Some(*id),
            _ => None,
        })
        .collect()
}

/// The indirect references in the array `obj`.
fn reference_set(doc: &Document, obj: Option<&Object>) -> HashSet<ObjectId> {
    references(&resolve_array(doc, obj)).into_iter().collect()
}

/// The names of `obj`, a name or an array of names.
fn names(doc: &Document, obj: Option<&Object>) -> Vec<Vec<u8>> {
    match obj {
        Some(Object::Name(n)) => vec![n.clone()],
        other => resolve_array(doc, other)
            .iter()
            .filter_map(|o| o.as_name().map(<[u8]>::to_vec))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use pdf::Object;

    use super::{Config, OptionalContent, RenderPurpose};
    use crate::test_helpers::make_doc_with_objects;

    /// A configuration with every group `ON` and no usage rules.
    fn all_on() -> OptionalContent {
        OptionalContent {
            purpose: RenderPurpose::View,
            config: Some(Config {
                base_on: true,
                on: HashSet::new(),
                off: HashSet::new(),
                intents: vec![b"View".to_vec()],
                usage: Vec::new(),
            }),
            states: HashMap::new(),
        }
    }

    #[test]
    fn shared_subexpressions_are_evaluated_once() {
        // Object 4 is a group; object 5 the membership dictionary; each of
        // objects 6..=35 is `[/And next next]`, so the expression unfolds
        // to 2^30 group operands but has only 31 distinct objects.
        const LEVELS: u32 = 30;
        let mut objects: Vec<(u32, String)> = vec![
            (4, "<< /Type /OCG /Name (G) >>".to_owned()),
            (5, "<< /Type /OCMD /VE 6 0 R >>".to_owned()),
        ];
        for id in 6..6 + LEVELS {
            let next = if id + 1 < 6 + LEVELS { id + 1 } else { 4 };
            objects.push((id, format!("[/And {next} 0 R {next} 0 R]")));
        }
        let doc = make_doc_with_objects(&objects);
        assert!(all_on().is_visible(&doc, &Object::Reference((5, 0))));
    }

    #[test]
    fn a_cyclic_expression_falls_back_to_the_policy() {
        // `/VE` refers to itself; the `/P` policy over the OFF group decides.
        let doc = make_doc_with_objects(&[
            (4, "<< /Type /OCG /Name (G) >>"),
            (5, "<< /Type /OCMD /VE 6 0 R /OCGs [4 0 R] /P /AllOff >>"),
            (6, "[/Not 6 0 R]"),
        ]);
        let mut oc = all_on();
        if let Some(config) = &mut oc.config {
            config.off = HashSet::from([(4, 0)]);
        }
        assert!(oc.is_visible(&doc, &Object::Reference((5, 0))));
    }
}
