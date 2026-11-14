//! Tests of the device image cache.

use super::*;

fn doc(byte: u8) -> DocId {
    DocId([byte; 32])
}

/// Smoke test: basic API shape sanity that does not require a
/// device.  Verifies that newtype keys and the budget struct
/// round-trip and have the documented properties.
#[test]
fn doc_id_and_obj_id_are_hashable() {
    use std::collections::HashSet;
    let mut set: HashSet<(DocId, ObjId)> = HashSet::new();
    let _ = set.insert((doc(0), ObjId(1)));
    let _ = set.insert((doc(0), ObjId(2)));
    let _ = set.insert((doc(0), ObjId(1))); // dup
    assert_eq!(set.len(), 2);
}

#[test]
fn image_layout_bytes_per_pixel() {
    assert_eq!(ImageLayout::Gray.bytes_per_pixel(), 1);
    assert_eq!(ImageLayout::Mask.bytes_per_pixel(), 1);
    assert_eq!(ImageLayout::Rgb.bytes_per_pixel(), 3);
}

#[test]
fn image_layout_blit_codes() {
    assert_eq!(ImageLayout::Rgb.blit_layout(), Some(0));
    assert_eq!(ImageLayout::Gray.blit_layout(), Some(1));
    assert_eq!(ImageLayout::Mask.blit_layout(), None);
}

#[test]
fn blake3_hash_is_deterministic() {
    let bytes = b"hello cache";
    let h1 = ContentHash::of(bytes);
    let h2 = ContentHash::of(bytes);
    assert_eq!(h1, h2);
    // Different input → different hash with overwhelming probability.
    let h3 = ContentHash::of(b"hello cachf");
    assert_ne!(h1, h3);
}

#[test]
fn cache_error_is_std_error() {
    let e = CacheError::EntryExceedsBudget {
        needed: 1 << 30,
        budget: 1 << 20,
    };
    // Display contains the byte counts so an operator can read it.
    let msg = e.to_string();
    assert!(msg.contains("1073741824"));
    assert!(msg.contains("1048576"));
    // The Error::source call below is the load-bearing proof that
    // CacheError satisfies std::error::Error (a separate trait-bound
    // assertion would just be redundant ceremony).
    assert!(std::error::Error::source(&e).is_none());
}

/// The checks below run on every backend a device is present for:
/// CUDA under `gpu-validation`, Vulkan under `vulkan`.
#[cfg(any(feature = "gpu-validation", feature = "vulkan"))]
mod device {
    use super::*;

    /// Build a 16×16 [`InsertRequest`] for a tiny test image.  The
    /// `pixels` slice must outlive the returned request — call sites
    /// keep it in a local binding.
    fn req(obj: u32, h: ContentHash, layout: ImageLayout, pixels: &[u8]) -> InsertRequest<'_> {
        InsertRequest {
            doc: doc(1),
            obj: ObjId(obj),
            hash: h,
            width: 16,
            height: 16,
            layout,
            pixels,
        }
    }

    /// Round-trip a 16×16 RGB image through the cache.  The cache's
    /// budgets must hold the image without eviction.
    pub(super) fn round_trip_rgb_image<B: CacheBackend>(cache: &DeviceImageCache<B>) {
        // 16×16×3 = 768; (i % 256) always fits u8, hence the unwrap.
        let pixels: Vec<u8> = (0_i32..16 * 16 * 3)
            .map(|i| u8::try_from(i % 256).expect("i % 256 fits u8"))
            .collect();
        let h = ContentHash::of(&pixels);
        let entry = cache
            .insert(req(42, h, ImageLayout::Rgb, &pixels))
            .expect("insert");
        assert_eq!(entry.width, 16);
        assert_eq!(entry.height, 16);

        // Lookup by both keys hits.
        let by_id = cache.lookup_by_id(doc(1), ObjId(42)).expect("alias hit");
        let by_hash = cache.lookup_by_hash(&h).expect("primary hit");
        assert!(Arc::ptr_eq(&entry, &by_id));
        assert!(Arc::ptr_eq(&entry, &by_hash));

        // Read back to confirm the upload landed correctly.
        assert_eq!(cache.download(&entry).expect("download"), pixels);

        // Used bytes accounting reflects the entry.
        assert_eq!(cache.used_bytes(), pixels.len() as u64);
    }

    /// LRU eviction of the oldest unpinned entry.  The cache must
    /// budget 600 bytes of VRAM and no host RAM, so that an evicted
    /// entry truly disappears (otherwise the demote-then-promote
    /// machinery would resurrect it on lookup).
    pub(super) fn lru_evicts_oldest_unpinned_entry<B: CacheBackend>(cache: &DeviceImageCache<B>) {
        let make_pixels = |fill: u8| -> Vec<u8> { vec![fill; 256] };

        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);
        let h_c = ContentHash([0xCC; 32]);

        let pixels_a = make_pixels(1);
        let pixels_b = make_pixels(2);
        let entry_a = cache
            .insert(req(1, h_a, ImageLayout::Gray, &pixels_a))
            .expect("a");
        let entry_b = cache
            .insert(req(2, h_b, ImageLayout::Gray, &pixels_b))
            .expect("b");
        // Drop the local Arcs so eviction can reclaim them.
        drop(entry_a);
        drop(entry_b);

        // Touch B so A is the oldest.
        let _ = cache.lookup_by_hash(&h_b);

        // Inserting C must evict A (oldest).
        let pixels_c = make_pixels(3);
        let _entry_c = cache
            .insert(req(3, h_c, ImageLayout::Gray, &pixels_c))
            .expect("c");

        assert!(
            cache.lookup_by_hash(&h_a).is_none(),
            "A should have been evicted"
        );
        assert!(cache.lookup_by_hash(&h_b).is_some(), "B should remain");
        assert!(cache.lookup_by_hash(&h_c).is_some(), "C just inserted");
    }

    /// A pinned entry blocks eviction.  The cache must budget 300
    /// bytes of VRAM: room for one 256-byte entry.
    pub(super) fn pinned_entries_block_eviction<B: CacheBackend>(cache: &DeviceImageCache<B>) {
        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);

        // Hold the Arc — A is now pinned.
        let pixels_a = vec![1u8; 256];
        let _pinned_a = cache
            .insert(req(1, h_a, ImageLayout::Gray, &pixels_a))
            .expect("a");

        // Inserting B must fail because A is pinned and there's no other
        // entry to evict.
        let pixels_b = vec![2u8; 256];
        let err = cache
            .insert(req(2, h_b, ImageLayout::Gray, &pixels_b))
            .unwrap_err();
        match err {
            CacheError::OverBudget {
                needed,
                used,
                budget,
            } => {
                assert_eq!(needed, 256);
                assert_eq!(used, 256);
                assert_eq!(budget, 300);
            }
            other => panic!("expected OverBudget, got {other:?}"),
        }
    }

    /// Bombard the cache with concurrent inserts of overlapping hashes
    /// from many threads.  After the dust settles, the cache should hold
    /// at most one entry per unique hash, and `used_bytes` should equal
    /// the sum of those unique entries' sizes — proving the dedup-race
    /// fix in `insert` (the `Entry::Vacant` atomic transition) accounts
    /// `used_bytes` exactly once per slab.  The cache must budget at
    /// least 4096 bytes of VRAM, so eviction does not enter the picture.
    pub(super) fn concurrent_inserts_dedup_correctly<B: CacheBackend>(cache: &DeviceImageCache<B>) {
        const THREADS: u8 = 16;
        const PER_THREAD: u8 = 64;
        const UNIQUE_HASHES: u8 = 8;
        const PIXEL_BYTES: usize = 256;

        // Each thread inserts PER_THREAD requests cycling through the
        // same UNIQUE_HASHES values, so threads heavily collide on each
        // hash and each request also legitimately overwrites the prior.
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let _ = s.spawn(move || {
                    for i in 0..PER_THREAD {
                        let hi = (t.wrapping_add(i)) % UNIQUE_HASHES;
                        let h = ContentHash([hi + 1; 32]);
                        // Pixel content keyed on hash so all threads
                        // racing on the same hash upload identical bytes.
                        let pixels = vec![hi + 1; PIXEL_BYTES];
                        let _ = cache.insert(InsertRequest {
                            doc: doc(1),
                            obj: ObjId(u32::from(t) * u32::from(PER_THREAD) + u32::from(i)),
                            hash: h,
                            width: 16,
                            height: 16,
                            layout: ImageLayout::Gray,
                            pixels: &pixels,
                        });
                    }
                });
            }
        });

        // Every unique hash should have produced exactly one cache
        // entry; `len()` counts primary entries.
        let len = cache.len();
        assert!(
            len <= usize::from(UNIQUE_HASHES),
            "cache holds {len} entries, expected ≤ {UNIQUE_HASHES}",
        );
        // `used_bytes` must equal exactly `len × PIXEL_BYTES` — one
        // fetch_add per surviving Vacant transition, no double-count.
        assert_eq!(
            cache.used_bytes(),
            (len as u64) * (PIXEL_BYTES as u64),
            "used_bytes drifted: {} B for {len} entries × {PIXEL_BYTES} B",
            cache.used_bytes(),
        );
    }

    /// Concurrent inserts of distinct images, each kept pinned, never
    /// together exceed the VRAM budget: each reserves its bytes before
    /// uploading, so the inserts past the budget fail with `OverBudget`.
    /// The cache must budget exactly four 256-byte entries.
    pub(super) fn concurrent_inserts_stay_within_budget<B: CacheBackend>(
        cache: &DeviceImageCache<B>,
    ) {
        const THREADS: u8 = 32;
        const PIXEL_BYTES: usize = 256;
        let start = std::sync::Barrier::new(usize::from(THREADS));
        let held: Vec<_> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..THREADS)
                .map(|t| {
                    let start = &start;
                    s.spawn(move || {
                        let pixels = vec![t; PIXEL_BYTES];
                        let _ = start.wait();
                        cache
                            .insert(InsertRequest {
                                doc: doc(2),
                                obj: ObjId(u32::from(t)),
                                hash: ContentHash([t.wrapping_add(100); 32]),
                                width: 16,
                                height: 16,
                                layout: ImageLayout::Gray,
                                pixels: &pixels,
                            })
                            .ok()
                    })
                })
                .collect();
            workers
                .into_iter()
                .filter_map(|w| w.join().expect("insert thread"))
                .collect()
        });
        assert_eq!(
            held.len(),
            4,
            "inserts that succeeded with every entry pinned"
        );
        assert_eq!(cache.used_bytes(), 4 * PIXEL_BYTES as u64);
    }

    /// End-to-end host tier round trip: insert entry A, evict it by
    /// inserting B, then look up A again — it must come back via the
    /// host tier's promotion path with bit-identical bytes.  The cache
    /// must budget 1000 bytes of VRAM (exactly one 768-byte RGB entry)
    /// and enough host RAM to hold the demoted copy.
    pub(super) fn evicted_vram_entry_promotes_back_from_host<B: CacheBackend>(
        cache: &DeviceImageCache<B>,
    ) {
        let pixels_a: Vec<u8> = (0_i32..16 * 16 * 3)
            .map(|i| u8::try_from((i * 7) % 256).expect("fits u8"))
            .collect();
        let pixels_b = vec![0xCDu8; 16 * 16 * 3];
        let h_a = ContentHash::of(&pixels_a);
        let h_b = ContentHash::of(&pixels_b);

        // Insert A and drop the local Arc so eviction can reclaim it.
        let entry_a = cache
            .insert(req(1, h_a, ImageLayout::Rgb, &pixels_a))
            .expect("insert a");
        drop(entry_a);

        // Insert B — must evict A and demote it to the host tier.
        let entry_b = cache
            .insert(req(2, h_b, ImageLayout::Rgb, &pixels_b))
            .expect("insert b");

        // Host tier should now have A.
        assert_eq!(
            cache.host.len(),
            1,
            "host tier should hold the demoted A; got {} entries",
            cache.host.len(),
        );

        // Drop B so the upcoming promote-of-A has room in VRAM (the
        // budget is one-entry-wide; promoting A while B is pinned would
        // legitimately fail with OverBudget).
        drop(entry_b);

        // Look up A — promotion from host should succeed and the
        // returned bytes must match the original pixels exactly.  The
        // bit-for-bit readback below is the load-bearing assertion;
        // checking Arc-pointer identity isn't (the heap may reuse a
        // slot under load and produce false negatives).
        let entry_a_promoted = cache.lookup_by_hash(&h_a).expect("promote hit");
        assert_eq!(
            cache.download(&entry_a_promoted).expect("download"),
            pixels_a,
            "promoted bytes must match the originally inserted pixels",
        );
    }
}

/// Defines one test per device check, each on a cache built by
/// `$mk(vram_bytes, host_bytes)`, which returns `None` to skip.
#[cfg(any(feature = "gpu-validation", feature = "vulkan"))]
macro_rules! device_tests {
    ($mk:path) => {
        #[test]
        fn round_trip_rgb_image() {
            if let Some(cache) = $mk(1 << 20, 1 << 24) {
                super::device::round_trip_rgb_image(&cache);
            }
        }

        #[test]
        fn lru_evicts_oldest_unpinned_entry() {
            if let Some(cache) = $mk(600, 0) {
                super::device::lru_evicts_oldest_unpinned_entry(&cache);
            }
        }

        #[test]
        fn pinned_entries_block_eviction() {
            if let Some(cache) = $mk(300, 1 << 24) {
                super::device::pinned_entries_block_eviction(&cache);
            }
        }

        #[test]
        fn concurrent_inserts_dedup_correctly() {
            if let Some(cache) = $mk(8 * 256 * 2, 1 << 24) {
                super::device::concurrent_inserts_dedup_correctly(&cache);
            }
        }

        #[test]
        fn concurrent_inserts_stay_within_budget() {
            if let Some(cache) = $mk(4 * 256, 0) {
                super::device::concurrent_inserts_stay_within_budget(&cache);
            }
        }

        #[test]
        fn evicted_vram_entry_promotes_back_from_host() {
            if let Some(cache) = $mk(1000, 1 << 20) {
                super::device::evicted_vram_entry_promotes_back_from_host(&cache);
            }
        }
    };
}

#[cfg(feature = "gpu-validation")]
mod cuda {
    use super::*;

    /// A fresh cache on the default stream of CUDA device 0.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "shares the skip-on-`None` signature of the Vulkan constructor"
    )]
    fn mk_cache(vram_bytes: u64, host_bytes: u64) -> Option<DeviceImageCache> {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        Some(DeviceImageCache::new(
            ctx.default_stream(),
            VramBudget { vram_bytes },
            HostBudget { host_bytes },
        ))
    }

    device_tests!(mk_cache);
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use crate::backend::vulkan::VulkanBackend;

    /// A fresh cache on [`crate::backend::vulkan::test_backend`]'s device;
    /// `None` (the test is skipped) when no Vulkan device opens.
    fn mk_cache(vram_bytes: u64, host_bytes: u64) -> Option<DeviceImageCache<VulkanBackend>> {
        crate::backend::vulkan::test_backend().map(|backend| {
            DeviceImageCache::new(
                Arc::new(backend),
                VramBudget { vram_bytes },
                HostBudget { host_bytes },
            )
        })
    }

    device_tests!(mk_cache);

    /// The VRAM budget a Vulkan device reports is positive and within
    /// the hard cap.
    #[test]
    fn vram_budget_from_backend_is_capped() {
        let Some(cache) = mk_cache(1, 0) else { return };
        let budget = VramBudget::from_backend(&**cache.device()).expect("budget query");
        assert!(budget.vram_bytes > 0);
        assert!(budget.vram_bytes <= VramBudget::HARD_CAP_BYTES);
    }

    /// A host-tier allocation the host refuses is an error, not an abort.
    #[test]
    fn refused_host_slab_is_an_error() {
        let Some(cache) = mk_cache(1, 0) else { return };
        let err = <VulkanBackend as CacheBackend>::alloc_host_slab(cache.device(), usize::MAX)
            .expect_err("a usize::MAX host slab cannot be allocated");
        assert!(
            matches!(err, CacheError::HostAlloc { bytes: usize::MAX }),
            "{err}"
        );
    }
}
