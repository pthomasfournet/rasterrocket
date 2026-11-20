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
#[cfg(any(feature = "gpu-icc", feature = "cache"))]
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

/// The live object `registry` holds for `key`, or the failure retained for
/// it, for objects that stand for a physical resource, `identity` naming
/// it: several keys may name one resource, which gets one object.  The
/// lock is held across `make`, so concurrent first callers build one
/// object.
///
/// Without a live object under `key`, the live object of the resource
/// `known` names, when given, is returned without building one; otherwise
/// `make` builds one, which is dropped in favour of a live object of the
/// same resource should one exist.  Either way `key` is registered for the
/// object returned, so it finds that object while it lives.
#[cfg(device_pages)]
pub(crate) fn shared_by_identity<K: PartialEq, T, E: Clone, I: PartialEq>(
    registry: &Registry<K, T, E>,
    key: K,
    known: Option<I>,
    identity: impl Fn(&T) -> I,
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
    let live = |entries: &[(K, Result<Weak<T>, E>)], id: &I| {
        entries.iter().find_map(|(_, entry)| {
            entry
                .as_ref()
                .ok()
                .and_then(Weak::upgrade)
                .filter(|object| identity(object) == *id)
        })
    };
    let object = known.and_then(|id| live(&entries, &id)).map_or_else(
        || make().map(|made| live(&entries, &identity(&made)).unwrap_or(made)),
        Ok,
    );
    entries.push((
        key,
        object.as_ref().map(Arc::downgrade).map_err(Clone::clone),
    ));
    object
}

#[cfg(all(test, device_pages))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{Registry, shared_by_identity};

    /// An object standing for resource `.0`, the `.1`th built.
    #[derive(Debug, PartialEq, Eq)]
    struct Resource(usize, usize);

    /// The object of `registry` for `key`, `known` naming its resource
    /// when given; a new one stands for `resource`, counted in `built`.
    fn get(
        registry: &Registry<Option<&'static str>, Resource, String>,
        key: Option<&'static str>,
        known: Option<usize>,
        resource: usize,
        built: &mut usize,
    ) -> Arc<Resource> {
        shared_by_identity(
            registry,
            key,
            known,
            |r| r.0,
            || {
                *built += 1;
                Ok(Arc::new(Resource(resource, *built)))
            },
        )
        .expect("built")
    }

    /// A key naming a live resource by another name shares its object
    /// without building one.
    #[test]
    fn a_known_resource_is_shared_without_building() {
        let registry = Mutex::new(Vec::new());
        let mut built = 0;
        let auto = get(&registry, None, None, 0, &mut built);
        let named = get(&registry, Some("0"), Some(0), 0, &mut built);
        assert!(Arc::ptr_eq(&auto, &named));
        let again = get(&registry, Some("0"), Some(0), 0, &mut built);
        assert!(Arc::ptr_eq(&auto, &again));
        assert_eq!(built, 1);
    }

    /// An object built for a resource that is already live is dropped for
    /// the live one; another resource gets an object of its own.
    #[test]
    fn a_duplicate_build_yields_the_live_object() {
        let registry = Mutex::new(Vec::new());
        let mut built = 0;
        let named = get(&registry, Some("gpu"), Some(1), 1, &mut built);
        let auto = get(&registry, None, None, 1, &mut built);
        assert!(Arc::ptr_eq(&auto, &named));
        let other = get(&registry, Some("2"), Some(2), 2, &mut built);
        assert_eq!(*other, Resource(2, 3));
    }

    /// Once every holder has dropped it, the resource is built anew.
    #[test]
    fn a_dropped_object_is_built_again() {
        let registry = Mutex::new(Vec::new());
        let mut built = 0;
        drop(get(&registry, None, None, 0, &mut built));
        let named = get(&registry, Some("0"), Some(0), 0, &mut built);
        assert_eq!(*named, Resource(0, 2));
    }
}
