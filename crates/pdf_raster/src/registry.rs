//! Registries of the GPU objects shared by the sessions alive at once.
//!
//! A registry maps a key to the object built for it, held weakly, or to
//! the error its construction returned.  An object is shared by every
//! holder while one exists and dropped with the last of them; a later
//! caller builds it anew.  A failure is retained and returned to every
//! later caller with its key.

use std::sync::{Arc, Mutex, PoisonError, Weak};

/// Objects held weakly, and construction failures, by key.  A linear
/// scan suffices: a process names a handful of keys at most.
pub(crate) type Registry<K, T, E> = Mutex<Vec<(K, Result<Weak<T>, E>)>>;

/// The live object `registry` holds for `key`, or the failure retained
/// for it, or else the object `make` builds now, registered under `key`.
/// The lock is held across `make`, so concurrent first callers build one
/// object; a poisoned lock is taken over, since every entry is pushed
/// whole.
pub(crate) fn shared<K: PartialEq, T, E: Clone>(
    registry: &Registry<K, T, E>,
    key: K,
    make: impl FnOnce() -> Result<Arc<T>, E>,
) -> Result<Arc<T>, E> {
    let mut entries = registry.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(at) = entries.iter().position(|(k, _)| *k == key) {
        match &entries[at].1 {
            Ok(object) => {
                if let Some(object) = object.upgrade() {
                    return Ok(object);
                }
                drop(entries.swap_remove(at));
            }
            Err(e) => return Err(e.clone()),
        }
    }
    let object = make();
    entries.push((
        key,
        object.as_ref().map(Arc::downgrade).map_err(Clone::clone),
    ));
    object
}
