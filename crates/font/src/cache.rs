//! Glyph bitmap cache.
//!
//! [`GlyphCache`] wraps a [`quick_cache::sync::Cache`] keyed by [`GlyphKey`].
//!
//! # Concurrency
//!
//! `quick_cache` shards its table and records recency atomically, so a hit
//! takes a shard's read lock only: concurrent lookups from the page-render
//! threads do not serialise within a shard.  Two threads missing on one key
//! may both render the glyph; the later insert replaces the earlier, and
//! both bitmaps are identical.
//!
//! # Scope
//!
//! A cache may serve any faces whose [`FaceId`]s come from one
//! [`crate::key::FaceIds`] source, across pages and threads: the identifier
//! names a program at one matrix, which fixes every bitmap keyed by it.
//! Faces from different sources must not share a cache, since their
//! identifiers alias.
//!
//! # Capacity policy
//!
//! The cache holds at most [`CAPACITY_BYTES`] of bitmap data, each entry
//! weighed by its pixel bytes plus [`ENTRY_OVERHEAD`].  Eviction is
//! approximately LRU.  Entries are `Arc<GlyphBitmap>`, so eviction drops the
//! `Arc` without freeing the bitmap data if the renderer still holds a
//! reference.
//!
//! # Key shape
//!
//! [`GlyphKey`] is `(face_id, glyph_id, size_px, x_frac, x_scale, aa)`: it
//! describes outline glyphs of a font program at one matrix.  Glyphs from other
//! sources (Type 3 `CharProcs`, colour glyphs) have no such key and are not
//! cached here; caching them calls for a key derived from the source.

use std::sync::Arc;

use quick_cache::Weighter;
use quick_cache::sync::Cache;

use crate::bitmap::GlyphBitmap;
use crate::key::{FaceId, GlyphKey};

/// Bitmap bytes one cache holds before it evicts.
pub const CAPACITY_BYTES: u64 = 256 << 20;

/// Bytes charged per entry beyond its pixel data: the bitmap header, the
/// `Arc` and the cache slot.
pub const ENTRY_OVERHEAD: u64 = 96;

/// Expected entry count, for the initial table size; the weight limit
/// governs eviction.
const ESTIMATED_ENTRIES: usize = 1 << 16;

/// Weighs an entry by the memory it holds.
#[derive(Clone, Copy, Debug)]
struct BitmapBytes;

impl Weighter<GlyphKey, Arc<GlyphBitmap>> for BitmapBytes {
    fn weight(&self, _key: &GlyphKey, bitmap: &Arc<GlyphBitmap>) -> u64 {
        bitmap.data.len() as u64 + ENTRY_OVERHEAD
    }
}

/// Shared, thread-safe glyph bitmap cache.
///
/// Clone to share across threads — the inner `Arc<Cache>` is cheap to clone
/// and keeps the cache alive as long as any clone exists.
#[derive(Clone, Debug)]
pub struct GlyphCache {
    inner: Arc<Cache<GlyphKey, Arc<GlyphBitmap>, BitmapBytes>>,
}

impl GlyphCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Cache::with_weighter(
                ESTIMATED_ENTRIES,
                CAPACITY_BYTES,
                BitmapBytes,
            )),
        }
    }

    /// Look up a glyph, promoting it to MRU on a hit.
    ///
    /// Returns a shared reference to the bitmap, or `None` on a miss.
    #[must_use]
    pub fn get(&self, key: &GlyphKey) -> Option<Arc<GlyphBitmap>> {
        self.inner.get(key)
    }

    /// Insert a rendered glyph.  Evicts the LRU entry when the cache is full.
    pub fn insert(&self, key: GlyphKey, bitmap: GlyphBitmap) {
        self.inner.insert(key, Arc::new(bitmap));
    }

    /// Remove all cached entries for the given face.
    ///
    /// Call when a [`crate::face::FontFace`] is unloaded to release its
    /// cached bitmaps eagerly rather than waiting for LRU eviction.
    pub fn evict_face(&self, face_id: FaceId) {
        self.inner.retain(|k, _| k.face_id != face_id);
    }

    /// Return the cached bitmap for `key`, rendering and inserting it via
    /// `render` on a miss.
    ///
    /// `render` returning `None` (a glyph that fails to load, or one with no
    /// outline such as a space) is **not** cached: the miss is cheap and
    /// caching it would need a tombstone entry per absent glyph.
    pub fn get_or_render<F>(&self, key: GlyphKey, render: F) -> Option<Arc<GlyphBitmap>>
    where
        F: FnOnce() -> Option<GlyphBitmap>,
    {
        if let Some(hit) = self.inner.get(&key) {
            return Some(hit);
        }
        let bitmap = Arc::new(render()?);
        self.inner.insert(key, Arc::clone(&bitmap));
        Some(bitmap)
    }

    /// Total number of cached glyphs.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.inner.len()
    }
}

impl Default for GlyphCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{FaceId, GlyphKey};

    fn bmp(tag: u8) -> GlyphBitmap {
        GlyphBitmap {
            x_off: 0,
            y_off: 0,
            width: 1,
            height: 1,
            aa: true,
            data: vec![tag],
        }
    }

    fn key(face: u32, glyph: u32) -> GlyphKey {
        GlyphKey::new(FaceId(face), glyph, 12, 0, true)
    }

    #[test]
    fn miss_on_empty() {
        let cache = GlyphCache::new();
        assert!(cache.get(&key(0, 0)).is_none());
    }

    #[test]
    fn insert_then_get() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(0xAA));
        let got = cache.get(&key(0, 1)).expect("should be present");
        assert_eq!(got.data[0], 0xAA);
    }

    #[test]
    fn different_face_ids_are_independent() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(10));
        cache.insert(key(1, 1), bmp(20));
        assert_eq!(cache.get(&key(0, 1)).unwrap().data[0], 10);
        assert_eq!(cache.get(&key(1, 1)).unwrap().data[0], 20);
    }

    #[test]
    fn evict_face_removes_entries() {
        let cache = GlyphCache::new();
        cache.insert(key(5, 1), bmp(1));
        cache.insert(key(5, 2), bmp(2));
        cache.insert(key(6, 1), bmp(3));

        cache.evict_face(FaceId(5));

        assert!(
            cache.get(&key(5, 1)).is_none(),
            "evicted face should be gone"
        );
        assert!(
            cache.get(&key(5, 2)).is_none(),
            "evicted face should be gone"
        );
        assert!(cache.get(&key(6, 1)).is_some(), "other face must survive");
    }

    #[test]
    fn total_len_counts_all_faces() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(1));
        cache.insert(key(0, 2), bmp(2));
        cache.insert(key(1, 1), bmp(3));
        assert_eq!(cache.total_len(), 3);
    }

    #[test]
    fn shared_clone_shares_state() {
        let a = GlyphCache::new();
        let b = a.clone();
        a.insert(key(0, 1), bmp(42));
        assert!(
            b.get(&key(0, 1)).is_some(),
            "clone must share the same backing store"
        );
    }

    #[test]
    fn get_or_render_renders_once_per_key() {
        let cache = GlyphCache::new();
        let calls = std::cell::Cell::new(0);
        let render = || {
            calls.set(calls.get() + 1);
            Some(bmp(7))
        };

        let first = cache.get_or_render(key(0, 1), render).expect("render");
        let second = cache.get_or_render(key(0, 1), render).expect("cached");

        assert_eq!(calls.get(), 1, "second lookup must not re-render");
        assert_eq!(first.data[0], 7);
        assert_eq!(second.data[0], 7);
        assert!(Arc::ptr_eq(&first, &second), "must hand back the same Arc");
    }

    #[test]
    fn get_or_render_distinguishes_keys() {
        let cache = GlyphCache::new();
        let a = cache.get_or_render(key(0, 1), || Some(bmp(1))).expect("a");
        let b = cache.get_or_render(key(0, 2), || Some(bmp(2))).expect("b");
        // Same glyph id but a different face must not alias.
        let c = cache.get_or_render(key(1, 1), || Some(bmp(3))).expect("c");

        assert_eq!((a.data[0], b.data[0], c.data[0]), (1, 2, 3));
        assert_eq!(cache.total_len(), 3);
    }

    #[test]
    fn get_or_render_does_not_cache_failures() {
        let cache = GlyphCache::new();
        assert!(cache.get_or_render(key(0, 1), || None).is_none());
        assert_eq!(cache.total_len(), 0, "a failed render must not be stored");
        // A later successful render for the same key still populates.
        let ok = cache.get_or_render(key(0, 1), || Some(bmp(9))).expect("ok");
        assert_eq!(ok.data[0], 9);
    }

    #[test]
    fn insert_replaces_existing() {
        let cache = GlyphCache::new();
        cache.insert(key(0, 1), bmp(1));
        cache.insert(key(0, 1), bmp(2));
        let got = cache.get(&key(0, 1)).unwrap();
        assert_eq!(got.data[0], 2, "second insert must overwrite first");
    }
}
