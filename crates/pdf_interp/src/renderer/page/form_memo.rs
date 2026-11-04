//! Parsed form `XObjects` of one page, so a form drawn by many `Do`
//! operators is decoded and parsed once.
//!
//! A form's operators depend on its stream and on the resources its inline
//! images resolve colour-space names in, which size their data: the form's
//! own resources, or, for a form without any, the resource context it is
//! drawn in.  The key is therefore the stream object and, for a form
//! without resources of its own, that context.  A form is kept from its
//! second use on, so a form drawn once costs no copy of its operators.

use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use pdf::ObjectId;

use crate::content::Operator;
use crate::resources::FormXObject;

/// Host bytes of kept operators the memo holds at most.  A form that would
/// exceed it is parsed on every use.
const BUDGET_BYTES: usize = 64 << 20;

/// The memo shared by a page renderer and the renderers of its pattern
/// cells.
pub(super) type SharedFormMemo = Rc<RefCell<FormMemo>>;

/// A form stream and, for a form without resources of its own, the
/// resource context it is drawn in.
pub(super) type FormKey = (ObjectId, Option<ObjectId>);

/// A form's dictionary entries and its parsed content.
pub(super) struct KeptForm {
    /// The form, its `content` emptied: `ops` holds what it parses to.
    pub(super) form: FormXObject,
    pub(super) ops: Vec<Operator<'static>>,
}

/// Parsed forms keyed by [`FormKey`].
pub(super) struct FormMemo {
    pub(super) kept: HashMap<FormKey, Rc<KeptForm>>,
    /// Forms parsed once without being kept.
    seen: HashSet<FormKey>,
    /// Forms refused for exceeding the budget left, which only shrinks, so
    /// they are never offered again.
    oversized: HashSet<FormKey>,
    /// Sum of the kept forms' [`kept_bytes`].
    bytes: usize,
    budget: usize,
}

/// Host bytes the owned operators `ops` hold: the operators themselves and
/// the heap behind their operands.
fn kept_bytes(ops: &[Operator<'static>]) -> usize {
    ops.iter()
        .map(|op| size_of::<Operator<'static>>().saturating_add(op.heap_bytes()))
        .fold(0, usize::saturating_add)
}

impl FormMemo {
    /// An empty memo holding at most [`BUDGET_BYTES`].
    pub(super) fn shared() -> SharedFormMemo {
        Rc::new(RefCell::new(Self::with_budget(BUDGET_BYTES)))
    }

    fn with_budget(budget: usize) -> Self {
        Self {
            kept: HashMap::new(),
            seen: HashSet::new(),
            oversized: HashSet::new(),
            bytes: 0,
            budget,
        }
    }

    /// The form kept at `key`.
    pub(super) fn get(&self, key: &FormKey) -> Option<Rc<KeptForm>> {
        self.kept.get(key).cloned()
    }

    /// Whether the form at `key` is to be offered to [`Self::insert`]: it
    /// has been parsed before and was not refused for its size.  A first
    /// use is recorded and answered `false`.
    pub(super) fn repeated(&mut self, key: FormKey) -> bool {
        !self.seen.insert(key) && !self.oversized.contains(&key)
    }

    /// Keep `form` at `key` when its owned operators fit the budget left,
    /// and return it.  A form that does not fit is marked so it is not
    /// offered again.
    pub(super) fn insert(&mut self, key: FormKey, form: KeptForm) -> Rc<KeptForm> {
        let form = Rc::new(form);
        let bytes = kept_bytes(&form.ops);
        if bytes > self.budget - self.bytes {
            let _ = self.oversized.insert(key);
        } else if let Entry::Vacant(slot) = self.kept.entry(key) {
            self.bytes += bytes;
            let _ = slot.insert(Rc::clone(&form));
        }
        form
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kept form of `content`'s operators.
    fn kept_form(content: &[u8]) -> KeptForm {
        KeptForm {
            form: FormXObject {
                content: Vec::new(),
                matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                bbox: None,
                resources_id: (1, 0),
                has_own_resources: false,
                transparency: None,
            },
            ops: crate::content::parse(content)
                .into_iter()
                .map(Operator::into_owned)
                .collect(),
        }
    }

    /// The budget charges the heap behind a kept form's operands: a `TJ`
    /// array of 2 000 numbers holds far more than its 4 000 content bytes.
    #[test]
    fn a_form_is_charged_for_its_operand_heap() {
        let mut content = b"[".to_vec();
        content.extend(b"0 ".repeat(2000));
        content.extend_from_slice(b"] TJ");
        let mut memo = FormMemo::with_budget(content.len() + 2 * size_of::<Operator<'static>>());
        let key = ((1, 0), None);
        assert!(!memo.repeated(key));
        assert!(memo.repeated(key));
        let _ = memo.insert(key, kept_form(&content));
        assert!(memo.kept.is_empty(), "the array's heap exceeds the budget");
        assert!(!memo.repeated(key), "a refused form is not offered again");
    }

    /// A form within the budget is kept once and charged its operators'
    /// size and heap.
    #[test]
    fn a_fitting_form_is_kept_and_charged() {
        let mut memo = FormMemo::with_budget(BUDGET_BYTES);
        let key = ((1, 0), None);
        let form = kept_form(b"/a gs (x) Tj 0 0 1 1 re f");
        let expected = 4 * size_of::<Operator<'static>>()
            + 2 * (1 + crate::content::operator::ALLOCATION_OVERHEAD);
        let _ = memo.insert(key, form);
        assert_eq!(memo.kept.len(), 1);
        assert_eq!(memo.bytes, expected);
    }
}
