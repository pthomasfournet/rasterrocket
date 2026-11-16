//! Object numbers for page-tree nodes written as direct dictionaries.
//!
//! Pages and page-tree nodes are identified by [`ObjectId`].  A node written
//! as a direct dictionary, the catalog's `/Pages` value or an element of a
//! `/Kids` array, has no number in the file; [`DirectNodes`] gives it one
//! above [`MAX_OBJECT_NUMBER`], which no object of the file can have, and
//! holds the node under it for the lifetime of the document.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::{
    dictionary::Dictionary,
    object::{MAX_OBJECT_NUMBER, Object, ObjectId},
};

/// Position given to the catalog's `/Pages` value, which is in no `/Kids`
/// array.  No array holds `u32::MAX` elements, so no `/Kids` element has it.
pub(crate) const PAGES_ROOT_SLOT: u32 = u32::MAX;

/// The direct page-tree nodes of one document, numbered in the order they
/// are first reached.
#[derive(Default)]
pub(crate) struct DirectNodes {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Number of the node at each site `(holder, slot)`: element `slot` of
    /// the `/Kids` array held by object `holder`, or the `/Pages` value of
    /// the catalog `holder` at [`PAGES_ROOT_SLOT`].
    sites: HashMap<(u32, u32), u32>,
    /// The node numbered `MAX_OBJECT_NUMBER + 1 + i` at index `i`.
    nodes: Vec<Arc<Object>>,
}

impl DirectNodes {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        // The state is only ever extended, so a panic while it was held
        // leaves every completed entry valid.
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The node numbered `num`, if one is.
    pub(crate) fn get(&self, num: u32) -> Option<Arc<Object>> {
        let index = num.checked_sub(MAX_OBJECT_NUMBER + 1)? as usize;
        self.lock().nodes.get(index).cloned()
    }

    /// The id of the direct node `dict` at site `(holder, slot)` (see
    /// [`Inner::sites`]), the same id on every call for that site.
    ///
    /// The first call stores a copy of `dict` whose `/Parent` is `parent`
    /// (when given), the node through which the site was reached: a direct
    /// node has no number its own `/Parent` could name.  The dictionaries
    /// of the copy's `/Kids` array are numbered in turn, with the copy as
    /// their `/Parent`, and replaced by references to those numbers, so
    /// every node is stored once and a walk reaches a nested direct node by
    /// reference.  `None` once the numbers above [`MAX_OBJECT_NUMBER`] are
    /// exhausted.
    pub(crate) fn intern(
        &self,
        holder: u32,
        slot: u32,
        dict: &Dictionary,
        parent: Option<ObjectId>,
    ) -> Option<ObjectId> {
        self.lock().intern(holder, slot, dict, parent)
    }
}

impl Inner {
    /// [`DirectNodes::intern`] under the lock.
    fn intern(
        &mut self,
        holder: u32,
        slot: u32,
        dict: &Dictionary,
        parent: Option<ObjectId>,
    ) -> Option<ObjectId> {
        if let Some(&num) = self.sites.get(&(holder, slot)) {
            return Some((num, 0));
        }
        let num = self.insert(dict.clone(), parent)?;
        let _ = self.sites.insert((holder, slot), num);
        Some((num, 0))
    }

    /// Store `dict` as a new node with `/Parent` `parent`, its direct
    /// `/Kids` dictionaries stored first as nodes of their own (see
    /// [`DirectNodes::intern`]); return its number.  Recursion is bounded
    /// by the nesting depth the object parser admits.
    fn insert(&mut self, mut dict: Dictionary, parent: Option<ObjectId>) -> Option<u32> {
        let index = self.nodes.len();
        let num = u32::try_from(index)
            .ok()
            .and_then(|i| i.checked_add(MAX_OBJECT_NUMBER + 1))?;
        self.nodes.push(Arc::new(Object::Null));
        if let Some(parent) = parent {
            dict.set(b"Parent", Object::Reference(parent));
        }
        if let Some(Object::Array(kids)) = dict.get_mut(b"Kids") {
            for kid in kids.iter_mut() {
                if let Object::Dictionary(child) = kid {
                    let child = std::mem::take(child);
                    *kid = self
                        .insert(child, Some((num, 0)))
                        .map_or(Object::Null, |n| Object::Reference((n, 0)));
                }
            }
        }
        self.nodes[index] = Arc::new(Object::Dictionary(dict));
        Some(num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(entries: &[(&str, Object)]) -> Dictionary {
        let mut d = Dictionary::new();
        for (k, v) in entries {
            d.set(k.as_bytes(), v.clone());
        }
        d
    }

    /// A site keeps its number; nested direct kids become references to
    /// nodes of their own whose `/Parent` is the node holding them.
    #[test]
    fn a_site_keeps_its_number_and_nested_kids_become_references() {
        let nodes = DirectNodes::default();
        let page = dict(&[("Type", Object::Name(b"Page".to_vec()))]);
        let root = dict(&[
            ("Type", Object::Name(b"Pages".to_vec())),
            ("Kids", Object::Array(vec![Object::Dictionary(page)])),
        ]);
        let id = nodes
            .intern(1, PAGES_ROOT_SLOT, &root, None)
            .expect("numbered");
        assert!(id.0 > MAX_OBJECT_NUMBER);
        assert_eq!(nodes.intern(1, PAGES_ROOT_SLOT, &root, None), Some(id));
        let stored = nodes.get(id.0).expect("stored");
        let kid = stored
            .as_dict()
            .and_then(|d| d.get(b"Kids"))
            .and_then(Object::as_array);
        let Some([Object::Reference(kid)]) = kid else {
            panic!("kid is not a reference: {kid:?}");
        };
        let kid = nodes.get(kid.0).expect("kid stored");
        assert_eq!(
            kid.as_dict().and_then(|d| d.get(b"Parent")),
            Some(&Object::Reference(id))
        );
        assert!(nodes.get(MAX_OBJECT_NUMBER).is_none());
        assert!(nodes.get(id.0 + 2).is_none());
    }
}
