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
    /// Sum of the kept forms' [`kept_bytes`].
    bytes: usize,
    budget: usize,
}

/// Host bytes a kept form of `ops`, parsed from `content_len` bytes,
/// holds: the operators, and their names and strings, which are at most
/// the content.
const fn kept_bytes(ops: usize, content_len: usize) -> usize {
    ops.saturating_mul(size_of::<Operator<'static>>())
        .saturating_add(content_len)
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
            bytes: 0,
            budget,
        }
    }

    /// The form kept at `key`.
    pub(super) fn get(&self, key: &FormKey) -> Option<Rc<KeptForm>> {
        self.kept.get(key).cloned()
    }

    /// Whether the form at `key`, parsed to `ops` operators from
    /// `content_len` bytes, is to be kept: it has been parsed before and
    /// fits the budget left.  A first use is recorded and answered `false`.
    pub(super) fn repeated(&mut self, key: FormKey, ops: usize, content_len: usize) -> bool {
        if !self.seen.insert(key) {
            return kept_bytes(ops, content_len) <= self.budget - self.bytes;
        }
        false
    }

    /// Keep `form`, parsed from `content_len` bytes, at `key` when it fits
    /// the budget left, and return it.
    pub(super) fn insert(
        &mut self,
        key: FormKey,
        form: KeptForm,
        content_len: usize,
    ) -> Rc<KeptForm> {
        let form = Rc::new(form);
        let bytes = kept_bytes(form.ops.len(), content_len);
        if bytes <= self.budget - self.bytes && !self.kept.contains_key(&key) {
            self.bytes += bytes;
            let _ = self.kept.insert(key, Rc::clone(&form));
        }
        form
    }
}
