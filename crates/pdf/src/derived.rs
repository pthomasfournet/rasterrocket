//! Values computed from document objects by higher layers, memoised for the
//! lifetime of the [`Document`](crate::Document).
//!
//! An entry is keyed by the object it derives from and by the type of the
//! value, so independent layers deriving different types from one object do
//! not collide.  Each entry is a [`OnceLock`]: concurrent first requests for
//! one key compute the value once, while requests for other keys proceed
//! without waiting.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use crate::object::ObjectId;

/// A memoised derivation: `None` records that it failed.
type Cell = Arc<OnceLock<Option<Arc<dyn Any + Send + Sync>>>>;

/// Per-document store of derived values.
#[derive(Default)]
pub(crate) struct DerivedCache {
    cells: Mutex<HashMap<(ObjectId, TypeId), Cell>>,
}

impl DerivedCache {
    /// The value of type `T` derived from object `id`, computed by `make` on
    /// the first request.  A `None` from `make` is memoised as well.
    pub(crate) fn get_or_derive<T: Any + Send + Sync>(
        &self,
        id: ObjectId,
        make: impl FnOnce() -> Option<T>,
    ) -> Option<Arc<T>> {
        let cell = {
            let mut cells = self.cells.lock().unwrap_or_else(PoisonError::into_inner);
            Arc::clone(cells.entry((id, TypeId::of::<T>())).or_default())
        };
        let value = cell.get_or_init(|| make().map(|v| Arc::new(v) as Arc<dyn Any + Send + Sync>));
        value.clone().map(|v| {
            v.downcast::<T>()
                .unwrap_or_else(|_| unreachable!("the key's TypeId fixes the stored type"))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell as StdCell;

    use super::*;

    #[test]
    fn derives_once_per_object_and_type() {
        let cache = DerivedCache::default();
        let calls = StdCell::new(0);
        let make = || {
            calls.set(calls.get() + 1);
            Some(7_u32)
        };
        assert_eq!(cache.get_or_derive((1, 0), make).as_deref(), Some(&7));
        assert_eq!(cache.get_or_derive((1, 0), make).as_deref(), Some(&7));
        assert_eq!(calls.get(), 1);

        // Another type derived from the same object is a separate entry.
        assert_eq!(
            cache.get_or_derive((1, 0), || Some("text")).as_deref(),
            Some(&"text")
        );
        // Another object is a separate entry.
        assert_eq!(cache.get_or_derive((2, 0), make).as_deref(), Some(&7));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn memoises_failure() {
        let cache = DerivedCache::default();
        let calls = StdCell::new(0);
        let fail = || {
            calls.set(calls.get() + 1);
            None::<u32>
        };
        assert!(cache.get_or_derive((3, 0), fail).is_none());
        assert!(cache.get_or_derive((3, 0), fail).is_none());
        assert_eq!(calls.get(), 1);
    }
}
