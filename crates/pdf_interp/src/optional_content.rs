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

use std::collections::HashSet;

use pdf::{Dictionary, Document, Object, ObjectId};

use crate::resources::dict_ext::DictExt;

/// The use a rendering is made for, which selects the optional-content usage
/// event (§8.11.4.4) that applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPurpose {
    /// On-screen viewing: the `View` usage event applies.
    View,
    /// Printing: the `Print` usage event applies.
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
        let config = doc
            .catalog()
            .ok()
            .and_then(|catalog| resolve_dict(doc, catalog.get(b"OCProperties")?))
            .and_then(|props| resolve_dict(doc, props.get(b"D")?))
            .map(|d| Config::new(doc, &d, purpose));
        Self { purpose, config }
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
        let (id, dict) = match oc {
            Object::Reference(id) => match doc.get_object(*id).ok().as_deref() {
                Some(Object::Dictionary(d)) => (Some(*id), d.clone()),
                _ => return true,
            },
            Object::Dictionary(d) => (None, d.clone()),
            _ => return true,
        };
        if dict.get_name(b"Type") == Some(b"OCMD") {
            self.membership_visible(doc, &dict)
        } else {
            id.is_none_or(|id| self.group_on(doc, id))
        }
    }

    /// The state of the group `id` (§8.11.4.3); `true` when it does not take
    /// part in visibility.
    fn group_on(&self, doc: &Document, id: ObjectId) -> bool {
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
        if let Some(visible) = ocmd.get(b"VE").and_then(|ve| self.expression(doc, ve, 0)) {
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

    /// Evaluate the visibility expression `ve` (§8.11.2.2): `[/And e…]`,
    /// `[/Or e…]` or `[/Not e]`, each operand a group or an expression.
    /// `None` when it is malformed or nested deeper than [`MAX_VE_DEPTH`].
    fn expression(&self, doc: &Document, ve: &Object, depth: u32) -> Option<bool> {
        if depth > MAX_VE_DEPTH {
            return None;
        }
        let resolved;
        let items = match ve {
            Object::Array(a) => a.as_slice(),
            Object::Reference(id) => {
                resolved = doc.get_object(*id).ok()?;
                resolved.as_array()?
            }
            _ => return None,
        };
        let (op, operands) = items.split_first()?;
        let mut values = operands
            .iter()
            .map(|operand| self.operand(doc, operand, depth));
        match op.as_name()? {
            b"Not" if operands.len() == 1 => values.next()?.map(|v| !v),
            b"And" if !operands.is_empty() => values.try_fold(true, |acc, v| Some(acc & v?)),
            b"Or" if !operands.is_empty() => values.try_fold(false, |acc, v| Some(acc | v?)),
            _ => None,
        }
    }

    /// The value of one operand of a visibility expression: a group or a
    /// nested expression.
    fn operand(&self, doc: &Document, operand: &Object, depth: u32) -> Option<bool> {
        match operand {
            Object::Reference(id) => match doc.get_object(*id).ok()?.as_ref() {
                Object::Dictionary(_) => Some(self.group_on(doc, *id)),
                Object::Array(_) => self.expression(doc, operand, depth + 1),
                _ => None,
            },
            Object::Array(_) => self.expression(doc, operand, depth + 1),
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

/// `obj` as a dictionary, following one indirect reference.
fn resolve_dict(doc: &Document, obj: &Object) -> Option<Dictionary> {
    crate::resources::resolve_dict(doc, obj)
}

/// The elements of the array `obj`, following one indirect reference; empty
/// when it is absent or not an array.
fn resolve_array(doc: &Document, obj: Option<&Object>) -> Vec<Object> {
    match obj {
        Some(Object::Array(a)) => a.clone(),
        Some(Object::Reference(id)) => doc
            .get_object(*id)
            .ok()
            .and_then(|o| o.as_array().map(<[Object]>::to_vec))
            .unwrap_or_default(),
        _ => Vec::new(),
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
