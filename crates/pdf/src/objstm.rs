//! Object-stream cache.
//!
//! PDF 1.5+ can pack multiple small objects into a single `/Type /ObjStm`
//! stream to save space.  A Type-2 xref entry says "object N is at index K
//! inside container object M."
//!
//! This module decodes a container stream and reads its directory exactly
//! once, then parses each object on demand from its directory offset.  An
//! object is looked up by number: the xref index is a hint that is verified
//! against the directory, and an object whose body fails to parse is missing
//! on its own, without affecting its neighbours.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    dictionary::Dictionary,
    error::PdfError,
    lexer::{parse_u64, skip_ws},
    object::{Object, parse_object},
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
        let data = decode_stream(content, dict)?;
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

/// Cache of decoded object streams, keyed by container object number.
#[derive(Default)]
pub(crate) struct ObjStmCache {
    cache: Mutex<HashMap<u32, Arc<ObjStm>>>,
}

impl ObjStmCache {
    pub(crate) fn new() -> Self {
        Self::default()
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

    /// The decoded container `container_id`, opened on first use.
    fn open(
        &self,
        container_id: u32,
        stream_content: &[u8],
        stream_dict: &Dictionary,
    ) -> Result<Arc<ObjStm>, PdfError> {
        // Poison recovery: the cache stores immutable decodes, safe to reuse.
        let mut guard = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(objstm) = guard.get(&container_id) {
            return Ok(Arc::clone(objstm));
        }
        let objstm = ObjStm::open(stream_content, stream_dict)
            .map(Arc::new)
            .map_err(|detail| PdfError::BadObject {
                id: container_id,
                detail,
            })?;
        let _ = guard.insert(container_id, Arc::clone(&objstm));
        drop(guard);
        Ok(objstm)
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
        let entry = u32::try_from(num).ok().zip(
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
    use super::*;

    /// An object stream whose bodies are `objects`, with `/N` set to
    /// `declared` entries; the directory covers every body.
    fn objstm(objects: &[(u32, &str)], declared: usize) -> (Vec<u8>, Dictionary) {
        let mut directory = String::new();
        let mut bodies = String::new();
        for (num, body) in objects {
            directory.push_str(&format!("{num} {} ", bodies.len()));
            bodies.push_str(body);
            bodies.push('\n');
        }
        let first = directory.len();
        let mut dict = Dictionary::new();
        dict.set(b"N", Object::Integer(declared as i64));
        dict.set(b"First", Object::Integer(first as i64));
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
}
