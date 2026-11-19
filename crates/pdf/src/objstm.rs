//! Object-stream cache.
//!
//! PDF 1.5+ can pack multiple small objects into a single `/Type /ObjStm`
//! stream to save space.  A Type-2 xref entry says "object N is at index K
//! inside container object M."
//!
//! This module decodes a container stream and reads its directory once,
//! then parses each object on demand from its directory offset.  An object
//! is looked up by number: the xref index is a hint that is verified against
//! the directory, and an object whose body fails to parse is missing on its
//! own, without affecting its neighbours.
//!
//! Decoded containers are cached per document under an aggregate byte
//! budget, least recently used first out; an evicted container is decoded
//! again when next needed, until the bytes so decoded again exceed a fixed
//! allowance; past it an evicted container stays unavailable.  A container
//! that fails to open is cached as that failure.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use crate::{
    dictionary::Dictionary,
    error::PdfError,
    lexer::{parse_u64, skip_ws},
    object::{Object, object_number, parse_object},
    stream::decode_stream,
};

/// Most directory entries an object stream may declare through `/N`.
const MAX_OBJECTS: usize = 1_000_000;

/// A decoded object stream: its bytes and its directory of
/// `(object number, absolute byte offset)` pairs, in directory order.
pub(crate) struct ObjStm {
    data: Vec<u8>,
    directory: Vec<(u32, usize)>,
}

impl ObjStm {
    /// Decode `content` with `dict` and read the directory.
    ///
    /// # Errors
    /// The decode failure, or an out-of-range `/N` or `/First`.
    pub(crate) fn open(content: &[u8], dict: &Dictionary) -> Result<Self, String> {
        let decoded = decode_stream(content, dict)?;
        if let Some(reason) = &decoded.truncated {
            log::warn!("object stream: {reason}");
        }
        let mut data = decoded.bytes;
        data.shrink_to_fit();
        let n = bounded_dict_usize(dict, b"N", MAX_OBJECTS)?;
        let first = bounded_dict_usize(dict, b"First", usize::MAX)?;
        if first > data.len() {
            return Err(format!(
                "/First={first} extends past stream length {}",
                data.len()
            ));
        }
        let directory = read_directory(&data, n, first);
        Ok(Self { data, directory })
    }

    /// The directory position of object `num`, taking `index` when it names
    /// that object and searching the directory otherwise.
    fn position_of(&self, num: u32, index: u32) -> Option<usize> {
        let hint = usize::try_from(index).ok()?;
        if self.directory.get(hint).is_some_and(|&(n, _)| n == num) {
            return Some(hint);
        }
        self.directory.iter().position(|&(n, _)| n == num)
    }

    /// Bytes held by this container: its decoded data and its directory.
    const fn retained_bytes(&self) -> usize {
        self.data
            .capacity()
            .saturating_add(self.directory.capacity() * size_of::<(u32, usize)>())
    }

    /// The directory: `(object number, absolute byte offset)` per entry.
    pub(crate) fn directory(&self) -> &[(u32, usize)] {
        &self.directory
    }

    /// Parse the object at directory position `pos`.
    pub(crate) fn parse_at(&self, pos: usize) -> Option<Object> {
        let mut at = self.directory.get(pos)?.1;
        parse_object(&self.data, &mut at)
    }
}

/// Bytes of decoded containers the cache retains, summed over containers,
/// before it evicts the least recently used.  Half the aggregate cap on a
/// page's decoded content; a single container larger than the budget (up to
/// the 1 GiB per-stream decode cap) is retained alone.
const CACHE_BUDGET: usize = 256 * 1024 * 1024;

/// Bytes the cache may decode again for evicted containers, as a multiple
/// of its budget.  Containers whose sizes together exceed the budget, read
/// in alternation, would otherwise be decoded on every read.
const REDECODE_ALLOWANCE_BUDGETS: usize = 4;

/// The outcome of opening one container, computed at most once per slot.
type Slot = Arc<OnceLock<Result<Arc<ObjStm>, String>>>;

/// A cached container: its slot, the bytes charged for it (0 until the
/// thread that opened it has charged them), its last use, and whether it
/// was evicted before this slot was made.
struct Entry {
    slot: Slot,
    bytes: usize,
    last_use: u64,
    reopened: bool,
}

#[derive(Default)]
struct State {
    entries: HashMap<u32, Entry>,
    /// Sum of `bytes` over `entries`.
    charged: usize,
    /// Use counter ordering `last_use`.
    clock: u64,
    /// Containers evicted at least once.
    evicted: HashSet<u32>,
    /// Bytes charged to containers opened again after their eviction.
    redecoded: usize,
    /// Whether the exhausted re-decode allowance has been reported.
    exhausted_warned: bool,
}

/// Cache of decoded object streams, keyed by container object number.
///
/// A container is decoded outside the cache lock, so lookups in other
/// containers proceed meanwhile; concurrent first lookups in one container
/// share a single decode through its slot.
pub(crate) struct ObjStmCache {
    state: Mutex<State>,
    budget: usize,
    redecode_allowance: usize,
}

impl Default for ObjStmCache {
    fn default() -> Self {
        Self::with_budget(CACHE_BUDGET)
    }
}

impl ObjStmCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A cache that retains at most `budget` bytes of containers besides
    /// the most recently opened one, and decodes evicted containers again
    /// for at most [`REDECODE_ALLOWANCE_BUDGETS`] times `budget` bytes.
    fn with_budget(budget: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            budget,
            redecode_allowance: budget.saturating_mul(REDECODE_ALLOWANCE_BUDGETS),
        }
    }

    /// Object `num`, which the xref places at `index` within the object
    /// stream whose encoded form is `stream_content` with dictionary
    /// `stream_dict`.
    ///
    /// # Errors
    /// [`PdfError::BadObject`] naming the container when it cannot be
    /// decoded, and naming `num` when the directory has no entry for it or
    /// its body does not parse.
    pub(crate) fn get_or_parse(
        &self,
        container_id: u32,
        index: u32,
        num: u32,
        stream_content: &[u8],
        stream_dict: &Dictionary,
    ) -> Result<Object, PdfError> {
        let objstm = self.open(container_id, stream_content, stream_dict)?;
        let pos = objstm
            .position_of(num, index)
            .ok_or_else(|| PdfError::BadObject {
                id: num,
                detail: format!("not in object stream {container_id} (xref index {index})"),
            })?;
        objstm.parse_at(pos).ok_or_else(|| PdfError::BadObject {
            id: num,
            detail: format!("unparseable at index {pos} of object stream {container_id}"),
        })
    }

    /// The state, recovered from poisoning: it holds immutable decodes and
    /// byte counts that every update leaves consistent.
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The decoded container `container_id`, opened on first use or after
    /// its eviction while the re-decode allowance lasts; a failure to open,
    /// the exhausted allowance included, is returned for every later use.
    fn open(
        &self,
        container_id: u32,
        stream_content: &[u8],
        stream_dict: &Dictionary,
    ) -> Result<Arc<ObjStm>, PdfError> {
        let slot = {
            let mut state = self.lock();
            state.clock += 1;
            let now = state.clock;
            if !state.entries.contains_key(&container_id) {
                let reopened = state.evicted.contains(&container_id);
                let slot = Slot::default();
                if reopened && state.redecoded >= self.redecode_allowance {
                    if !std::mem::replace(&mut state.exhausted_warned, true) {
                        log::warn!(
                            "object streams: {} bytes decoded again after eviction; \
                             evicted object streams are no longer decoded",
                            state.redecoded
                        );
                    }
                    let _ = slot.set(Err(format!(
                        "object stream evicted and the re-decode allowance of {} bytes exhausted",
                        self.redecode_allowance
                    )));
                }
                let _ = state.entries.insert(
                    container_id,
                    Entry {
                        slot,
                        bytes: 0,
                        last_use: now,
                        reopened,
                    },
                );
            }
            let Some(entry) = state.entries.get_mut(&container_id) else {
                unreachable!("the entry was inserted above");
            };
            entry.last_use = now;
            let slot = Arc::clone(&entry.slot);
            drop(state);
            slot
        };
        let mut opened_here = false;
        let outcome = slot.get_or_init(|| {
            opened_here = true;
            ObjStm::open(stream_content, stream_dict).map(Arc::new)
        });
        if opened_here {
            let bytes = outcome.as_ref().map_or(0, |objstm| objstm.retained_bytes());
            self.charge(container_id, &slot, bytes);
        }
        outcome.clone().map_err(|detail| PdfError::BadObject {
            id: container_id,
            detail,
        })
    }

    /// Charge `bytes` to the entry of `container_id` holding `slot`, and to
    /// the re-decode allowance when the container was evicted before, then
    /// evict the least recently used other containers until the charged
    /// total is within the budget.
    fn charge(&self, container_id: u32, slot: &Slot, bytes: usize) {
        let mut state = self.lock();
        let Some(entry) = state.entries.get_mut(&container_id) else {
            return;
        };
        if !Arc::ptr_eq(&entry.slot, slot) {
            return;
        }
        entry.bytes = bytes;
        let reopened = entry.reopened;
        if bytes == 0 {
            return;
        }
        if reopened {
            state.redecoded = state.redecoded.saturating_add(bytes);
        }
        state.charged = state.charged.saturating_add(bytes);
        while state.charged > self.budget {
            let victim = state
                .entries
                .iter()
                .filter(|&(&id, e)| id != container_id && e.bytes > 0)
                .min_by_key(|(_, e)| e.last_use)
                .map(|(&id, _)| id);
            let Some(victim) = victim else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&victim) {
                state.charged -= evicted.bytes;
                let _ = state.evicted.insert(victim);
            }
        }
        drop(state);
    }

    /// Bytes currently charged to cached containers.
    #[cfg(test)]
    fn charged(&self) -> usize {
        self.lock().charged
    }
}

/// Read an integer dict entry (defaulting to 0 if absent), enforcing
/// `0..=max`. Returns the offending value formatted into the error message.
fn bounded_dict_usize(dict: &Dictionary, key: &[u8], max: usize) -> Result<usize, String> {
    let value = dict.get(key).and_then(Object::as_i64).unwrap_or(0);
    usize::try_from(value)
        .ok()
        .filter(|&v| v <= max)
        .ok_or_else(|| {
            let key_str = String::from_utf8_lossy(key);
            format!("/{key_str}={value} out of range [0, {max}]")
        })
}

/// The directory at the start of a decoded object stream: up to `n` pairs
/// of object number and offset relative to `first`, read until the first
/// pair that is not two integers.  A pair whose offset cannot be addressed
/// is dropped.
fn read_directory(data: &[u8], n: usize, first: usize) -> Vec<(u32, usize)> {
    // Each pair needs at least four bytes of text.
    let mut directory = Vec::with_capacity(n.min(data.len() / 4));
    let mut pos = 0usize;
    for _ in 0..n {
        skip_ws(data, &mut pos);
        let Some(num) = parse_u64(data, &mut pos) else {
            break;
        };
        skip_ws(data, &mut pos);
        let Some(rel) = parse_u64(data, &mut pos) else {
            break;
        };
        let entry = object_number(num).zip(
            usize::try_from(rel)
                .ok()
                .and_then(|rel| first.checked_add(rel))
                .filter(|&abs| abs < data.len()),
        );
        if let Some(entry) = entry {
            directory.push(entry);
        }
    }
    directory
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    /// An object stream whose bodies are `objects`, with `/N` set to
    /// `declared` entries; the directory covers every body.
    fn objstm(objects: &[(u32, &str)], declared: usize) -> (Vec<u8>, Dictionary) {
        let mut directory = String::new();
        let mut bodies = String::new();
        for (num, body) in objects {
            write!(directory, "{num} {} ", bodies.len()).expect("writing to a String");
            bodies.push_str(body);
            bodies.push('\n');
        }
        let first = directory.len();
        let mut dict = Dictionary::new();
        dict.set(
            b"N",
            Object::Integer(i64::try_from(declared).expect("small")),
        );
        dict.set(
            b"First",
            Object::Integer(i64::try_from(first).expect("small")),
        );
        (format!("{directory}{bodies}").into_bytes(), dict)
    }

    fn get(
        cache: &ObjStmCache,
        data: &[u8],
        dict: &Dictionary,
        index: u32,
        num: u32,
    ) -> Result<Object, PdfError> {
        cache.get_or_parse(7, index, num, data, dict)
    }

    #[test]
    fn objects_are_found_by_index() {
        let (data, dict) = objstm(&[(1, "42"), (2, "true")], 2);
        let cache = ObjStmCache::new();
        assert_eq!(
            get(&cache, &data, &dict, 0, 1).unwrap(),
            Object::Integer(42)
        );
        assert_eq!(
            get(&cache, &data, &dict, 1, 2).unwrap(),
            Object::Boolean(true)
        );
    }

    /// One unparseable body leaves the other objects available.
    #[test]
    fn bad_object_does_not_hide_its_neighbours() {
        let (data, dict) = objstm(&[(1, "42"), (2, ")))garbage((("), (3, "true")], 3);
        let cache = ObjStmCache::new();
        assert_eq!(
            get(&cache, &data, &dict, 0, 1).unwrap(),
            Object::Integer(42)
        );
        let err = get(&cache, &data, &dict, 1, 2).unwrap_err();
        assert!(matches!(err, PdfError::BadObject { id: 2, .. }), "{err:?}");
        assert_eq!(
            get(&cache, &data, &dict, 2, 3).unwrap(),
            Object::Boolean(true)
        );
    }

    /// An xref index that names another object is corrected through the
    /// directory.
    #[test]
    fn stale_index_is_corrected_by_object_number() {
        let (data, dict) = objstm(&[(1, "42"), (2, "true")], 2);
        let cache = ObjStmCache::new();
        assert_eq!(
            get(&cache, &data, &dict, 0, 2).unwrap(),
            Object::Boolean(true)
        );
        assert_eq!(
            get(&cache, &data, &dict, 9, 1).unwrap(),
            Object::Integer(42)
        );
    }

    /// An object the directory does not list is missing; the object at the
    /// xref index is never returned in its place.
    #[test]
    fn object_absent_from_directory_is_missing() {
        let (data, dict) = objstm(&[(1, "42"), (2, "true")], 2);
        let cache = ObjStmCache::new();
        let err = get(&cache, &data, &dict, 0, 9).unwrap_err();
        assert!(matches!(err, PdfError::BadObject { id: 9, .. }), "{err:?}");
    }

    /// A directory that breaks off keeps the entries before the break.
    #[test]
    fn truncated_directory_keeps_leading_entries() {
        let (mut data, dict) = objstm(&[(1, "42"), (2, "true")], 3);
        // Prepend nothing; the third declared pair is simply absent, so the
        // directory parse runs into the first body.
        let cache = ObjStmCache::new();
        assert_eq!(
            get(&cache, &data, &dict, 0, 1).unwrap(),
            Object::Integer(42)
        );
        // A pair whose offset is beyond the data is dropped, not addressed.
        drop(data.splice(0..0, b"5 999999 ".iter().copied()));
        let mut dict2 = dict.clone();
        dict2.set(
            b"First",
            Object::Integer(dict.get(b"First").unwrap().as_i64().unwrap() + 9),
        );
        let cache = ObjStmCache::new();
        let err = get(&cache, &data, &dict2, 0, 5).unwrap_err();
        assert!(matches!(err, PdfError::BadObject { id: 5, .. }), "{err:?}");
        assert_eq!(
            get(&cache, &data, &dict2, 1, 1).unwrap(),
            Object::Integer(42)
        );
    }

    /// `/First` past the end of the data is a container error, not a panic.
    #[test]
    fn first_beyond_data_is_rejected() {
        let (data, mut dict) = objstm(&[(1, "42")], 1);
        dict.set(b"First", Object::Integer(10_000));
        let cache = ObjStmCache::new();
        let err = get(&cache, &data, &dict, 0, 1).unwrap_err();
        assert!(matches!(err, PdfError::BadObject { id: 7, .. }), "{err:?}");
    }

    /// A container that fails to open is cached as that failure: a later
    /// lookup returns it without decoding the container again, even when
    /// handed bytes that would now decode.
    #[test]
    fn failed_open_is_cached() {
        let (data, mut bad) = objstm(&[(1, "42")], 1);
        let good = bad.clone();
        bad.set(b"First", Object::Integer(10_000));
        let cache = ObjStmCache::new();
        assert!(get(&cache, &data, &bad, 0, 1).is_err());
        let err = get(&cache, &data, &good, 0, 1).unwrap_err();
        assert!(
            matches!(&err, PdfError::BadObject { id: 7, detail } if detail.contains("/First")),
            "{err:?}"
        );
        assert_eq!(cache.charged(), 0);
    }

    /// Past the budget the least recently used containers are evicted, and
    /// an evicted container is decoded again when next needed.
    #[test]
    fn containers_past_the_budget_are_evicted_and_reopened() {
        let (data, dict) = objstm(&[(1, "42"), (2, "true")], 2);
        let one = ObjStm::open(&data, &dict).unwrap().retained_bytes();
        let cache = ObjStmCache::with_budget(2 * one);
        for container in 1..=5 {
            assert_eq!(
                cache.get_or_parse(container, 0, 1, &data, &dict).unwrap(),
                Object::Integer(42)
            );
            assert!(cache.charged() <= 2 * one, "{}", cache.charged());
        }
        assert_eq!(cache.charged(), 2 * one);
        let cached = |id| cache.lock().entries.contains_key(&id);
        assert!(!cached(1) && !cached(3) && cached(4) && cached(5));
        // Container 4 used last stays when 1 returns; 5 goes.
        let _ = cache.get_or_parse(4, 1, 2, &data, &dict).unwrap();
        assert_eq!(
            cache.get_or_parse(1, 1, 2, &data, &dict).unwrap(),
            Object::Boolean(true)
        );
        assert!(cached(1) && cached(4) && !cached(5));
        assert_eq!(cache.charged(), 2 * one);
    }

    /// Containers that do not fit the budget together, read in alternation,
    /// are decoded again only until the re-decode allowance is spent; past
    /// it an evicted container is an error, while the one cached still
    /// resolves.
    #[test]
    fn re_decodes_stop_past_the_allowance() {
        let (data, dict) = objstm(&[(1, "42")], 1);
        let one = ObjStm::open(&data, &dict).unwrap().retained_bytes();
        let cache = ObjStmCache::with_budget(one);
        let mut errors = 0;
        for round in 0..100_u32 {
            if cache
                .get_or_parse(1 + round % 2, 0, 1, &data, &dict)
                .is_err()
            {
                errors += 1;
            }
        }
        let state = cache.lock();
        assert_eq!(state.redecoded, REDECODE_ALLOWANCE_BUDGETS * one);
        drop(state);
        // Two first opens and four re-opens within the allowance; then
        // container 1, evicted last, is refused on each of its 47 further
        // reads, and container 2 stays cached.
        assert_eq!(errors, 47);
        assert!(cache.get_or_parse(2, 0, 1, &data, &dict).is_ok());
        let err = cache.get_or_parse(1, 0, 1, &data, &dict).unwrap_err();
        assert!(
            matches!(&err, PdfError::BadObject { detail, .. } if detail.contains("re-decode")),
            "{err:?}"
        );
    }

    /// A container that fails to open evicts nothing.
    #[test]
    fn failed_open_evicts_nothing() {
        let (data, dict) = objstm(&[(1, "42")], 1);
        let mut bad = dict.clone();
        bad.set(b"First", Object::Integer(10_000));
        let cache = ObjStmCache::with_budget(1);
        let _ = cache.get_or_parse(1, 0, 1, &data, &dict).unwrap();
        assert!(cache.get_or_parse(2, 0, 1, &data, &bad).is_err());
        assert!(cache.lock().entries.get(&1).is_some_and(|e| e.bytes > 0));
    }

    /// A container larger than the budget is retained alone.
    #[test]
    fn container_over_the_budget_is_retained_alone() {
        let (data, dict) = objstm(&[(1, "42")], 1);
        let cache = ObjStmCache::with_budget(1);
        let _ = cache.get_or_parse(1, 0, 1, &data, &dict).unwrap();
        let _ = cache.get_or_parse(2, 0, 1, &data, &dict).unwrap();
        let one = ObjStm::open(&data, &dict).unwrap().retained_bytes();
        assert_eq!(cache.charged(), one);
        assert_eq!(cache.lock().entries.len(), 1);
    }

    /// Threads that open one container at once share a single decode, which
    /// is charged once.
    #[test]
    fn concurrent_opens_charge_once() {
        let (data, dict) = objstm(&[(1, "42"), (2, "true")], 2);
        let one = ObjStm::open(&data, &dict).unwrap().retained_bytes();
        let cache = ObjStmCache::new();
        std::thread::scope(|scope| {
            for t in 0..8_u32 {
                let (cache, data, dict) = (&cache, &data, &dict);
                let _ = scope.spawn(move || {
                    for _ in 0..100 {
                        let num = 1 + t % 2;
                        assert!(cache.get_or_parse(3, num - 1, num, data, dict).is_ok());
                    }
                });
            }
        });
        assert_eq!(cache.charged(), one);
    }
}
