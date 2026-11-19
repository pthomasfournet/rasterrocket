//! Host RAM tier.
//!
//! The VRAM tier lives in `crates/gpu/src/cache/mod.rs`.  When VRAM
//! eviction reclaims a slab, its bytes are not dropped outright; they are
//! copied into host memory and stashed here so a subsequent lookup can
//! copy them back to VRAM (O(1) host probe + one upload) rather than
//! re-decode the JPEG (~15 ms of CPU work per image).
//!
//! # Invariants
//!
//! - Indexed only by [`super::ContentHash`] (the cache's primary key).
//!   The `(DocId, ObjId)` alias lives in the VRAM tier; an alias is
//!   wired only at `insert` time, so a promote-after-eviction lookup
//!   that reaches us by content hash will not auto-bind a new alias —
//!   that's fine because hash lookups are themselves O(1) once the
//!   caller already has the hash.
//! - Each entry stores its bytes in the backend's
//!   [`super::CacheBackend::HostSlab`].  On CUDA that is page-locked memory
//!   allocated with `cuMemAllocHost(CU_MEMHOSTALLOC_WRITECOMBINED)`:
//!   write-combined memory is fast for upload-on-promote and fast to fill
//!   on demote, but slow for general CPU reads — the cache writes to it
//!   from a `cudaMemcpyAsync(D2H)` and reads from it via
//!   `cudaMemcpyAsync(H2D)`, and the CPU never inspects the bytes.  On
//!   Vulkan it is ordinary host memory, since the backend's transfers
//!   stage through mapped buffers of their own.
//! - The LRU clock is shared with the VRAM tier (callers pass it in)
//!   so cross-tier freshness is comparable.  Used-bytes accounting is
//!   per-tier; each tier evicts on its own pressure.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use super::{CacheBackend, ContentHash, ImageLayout};

/// Budget for the host RAM tier.  Independent of [`super::VramBudget`].
#[derive(Debug, Clone, Copy)]
pub struct HostBudget {
    /// Maximum bytes the tier is allowed to hold in pinned host memory.
    /// When a demotion would push past this, the tier evicts the oldest
    /// entry first.  Set to zero to effectively disable the tier (every
    /// demotion will immediately drop on the eviction scan).
    pub host_bytes: u64,
}

impl HostBudget {
    /// Default: 2 GiB.  On CUDA the tier holds page-locked memory, which
    /// the kernel cannot page out, so the budget bounds how much of the
    /// system's RAM the tier locks.
    pub const DEFAULT: Self = Self {
        host_bytes: 2 * 1024 * 1024 * 1024,
    };
}

/// One host-resident copy of a previously-decoded image.
///
/// Implementation detail of [`HostTier`]; the parent `host_tier` module
/// is private, so `pub` here is already crate-bounded.
pub(crate) struct HostEntry<B: CacheBackend> {
    /// Host memory holding the decoded bytes.
    pub pinned: B::HostSlab,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// LRU timestamp.  Updated on every host-tier hit; load-bearing for
    /// the host tier's own eviction scan.  Atomic store with `Relaxed`
    /// — exact ordering between threads doesn't matter for LRU.
    last_used: AtomicU64,
}

impl<B: CacheBackend> HostEntry<B> {
    /// Build a new entry from a populated host slab.  `last_used` is
    /// initialised to zero and is bumped to the current tick by the
    /// host tier on insert.
    #[must_use]
    pub(crate) const fn new(
        pinned: B::HostSlab,
        width: u32,
        height: u32,
        layout: ImageLayout,
    ) -> Self {
        Self {
            pinned,
            width,
            height,
            layout,
            last_used: AtomicU64::new(0),
        }
    }

    /// Bytes occupied in host memory by this entry.
    #[must_use]
    pub(crate) fn host_bytes(&self) -> u64 {
        B::host_slab_len(&self.pinned) as u64
    }

    fn touch(&self, tick: u64) {
        self.last_used.store(tick, Ordering::Relaxed);
    }

    fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

impl<B: CacheBackend> std::fmt::Debug for HostEntry<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostEntry")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .field("host_bytes", &self.host_bytes())
            .finish_non_exhaustive()
    }
}

/// Concurrent host RAM tier.  All methods take `&self`.
///
/// Implementation detail of [`super::DeviceImageCache`]; the parent
/// `host_tier` module is private, and `mod.rs` re-exports this type
/// as `pub(crate)`, which is where the crate-wide visibility bound
/// is actually enforced.
pub(crate) struct HostTier<B: CacheBackend> {
    entries: DashMap<ContentHash, Arc<HostEntry<B>>>,
    used_bytes: AtomicU64,
    budget: HostBudget,
}

impl<B: CacheBackend> HostTier<B> {
    /// Build an empty host tier with the given budget.
    #[must_use]
    pub(crate) fn new(budget: HostBudget) -> Self {
        Self {
            entries: DashMap::new(),
            used_bytes: AtomicU64::new(0),
            budget,
        }
    }

    /// Live entry count — diagnostics / tests only.  Gated on `cfg(test)`
    /// because production code never inspects the tier directly.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tier currently holds zero entries.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Live host RAM usage in bytes — diagnostics / tests only.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Budget in bytes.
    #[must_use]
    pub(crate) const fn budget_bytes(&self) -> u64 {
        self.budget.host_bytes
    }

    /// Probe by content hash.  Returns the host entry on hit and bumps
    /// its LRU timestamp via `tick`.  Caller is responsible for re-
    /// uploading to VRAM if it needs a device-resident copy; the host
    /// entry stays in place to amortise repeated VRAM evictions.
    #[must_use]
    pub(crate) fn lookup(&self, hash: &ContentHash, tick: u64) -> Option<Arc<HostEntry<B>>> {
        let entry = self.entries.get(hash)?.clone();
        entry.touch(tick);
        Some(entry)
    }

    /// Install a freshly populated host entry.  Evicts in LRU order if
    /// the new entry would exceed the budget.  If the same content
    /// hash already has an entry, the old one is replaced (simpler than
    /// dedup-on-demotion and the bytes are bit-identical anyway).
    ///
    /// Returns the strong `Arc` for symmetry with the VRAM tier; most
    /// callers ignore it.
    ///
    /// # Accounting semantics
    ///
    /// `used_bytes` tracks **bytes managed by this tier's index**, not
    /// bytes physically allocated as host memory.  When this method
    /// removes a prior same-hash entry whose `Arc` has `strong_count >
    /// 1` (a concurrent lookup-holder is still reading it), the
    /// host slab stays allocated until the holder drops, but
    /// `used_bytes` is decremented immediately.  The transient
    /// undercount is bounded by the number of in-flight lookups and
    /// resolves the moment those `Arc`s drop.  This matches the VRAM
    /// tier's contract; both tiers charge bytes to the index.
    ///
    /// Every removal from the index, here and in eviction, subtracts the
    /// bytes of the entry it removed, and every insertion adds its own:
    /// the replaced entry is the one the map's `insert` returns, under the
    /// map's shard lock, so concurrent inserts of one hash subtract each
    /// displaced entry exactly once and `used_bytes` stays the sum over
    /// the index once they return.
    pub(crate) fn insert(
        &self,
        hash: ContentHash,
        entry: HostEntry<B>,
        tick: u64,
    ) -> Arc<HostEntry<B>> {
        entry.touch(tick);
        let bytes = entry.host_bytes();
        let arc = Arc::new(entry);
        // Make room before inserting; subtract the about-to-be-removed
        // entry's size first so a same-hash re-insert reuses the slot.
        if let Some((_, prior)) = self.entries.remove(&hash) {
            let _ = self
                .used_bytes
                .fetch_sub(prior.host_bytes(), Ordering::Relaxed);
        }
        self.evict_to_fit(bytes);
        let _ = self.used_bytes.fetch_add(bytes, Ordering::Relaxed);
        if let Some(displaced) = self.entries.insert(hash, arc.clone()) {
            let _ = self
                .used_bytes
                .fetch_sub(displaced.host_bytes(), Ordering::Relaxed);
        }
        arc
    }

    /// Drop entries in LRU order until the new `needed` bytes fit.
    ///
    /// Unlike the VRAM tier, host entries don't need refcount-based
    /// pinning — device reads of host memory are explicit copies, and no
    /// `Arc<HostEntry>` is kept alive past the upload that reads it.  So a host
    /// entry is always reclaimable as long as its `Arc` `strong_count`
    /// is 1.  We still respect that bound to avoid yanking memory from
    /// under an in-flight DMA.
    fn evict_to_fit(&self, needed: u64) {
        // Bound iteration so a churn pattern (every candidate gets
        // pinned by a parallel promote) doesn't livelock.
        let max_iters = self.entries.len().saturating_mul(2).saturating_add(4);
        for _ in 0..max_iters {
            let used = self.used_bytes.load(Ordering::Relaxed);
            if used.saturating_add(needed) <= self.budget.host_bytes {
                return;
            }
            let oldest = self
                .entries
                .iter()
                .filter_map(|e| {
                    let arc = e.value();
                    (Arc::strong_count(arc) == 1).then(|| (*e.key(), arc.last_used()))
                })
                .min_by_key(|&(_, ts)| ts);
            let Some((key, _)) = oldest else {
                // Nothing reclaimable: the tier is empty, or every entry
                // is held by an in-flight promotion.  The insert then
                // overshoots the budget, by at most the in-flight
                // promotions plus this entry.
                log::warn!(
                    "host tier: cannot evict to fit {needed} B; no entry is reclaimable (the tier is empty or every entry is being promoted; used {used} B, budget {} B)",
                    self.budget.host_bytes,
                );
                return;
            };
            if let Some((_, removed)) = self.entries.remove(&key) {
                let bytes = removed.host_bytes();
                debug_assert!(
                    self.used_bytes.load(Ordering::Relaxed) >= bytes,
                    "host tier used_bytes underflow",
                );
                let _ = self.used_bytes.fetch_sub(bytes, Ordering::Relaxed);
            }
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::CudaBackend;

    /// A pinned slab of `bytes` bytes in device 0's context, through the
    /// CUDA cache's host-slab allocator.
    #[cfg(feature = "gpu-validation")]
    fn alloc_pinned(
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        bytes: usize,
    ) -> cudarc::driver::PinnedHostSlice<u8> {
        <CudaBackend as CacheBackend>::alloc_host_slab(stream, bytes).expect("alloc_pinned")
    }

    #[test]
    fn host_budget_default_is_two_gib() {
        assert_eq!(HostBudget::DEFAULT.host_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn empty_host_tier_reports_zero() {
        let tier = HostTier::<CudaBackend>::new(HostBudget {
            host_bytes: 1 << 20,
        });
        assert_eq!(tier.len(), 0);
        assert!(tier.is_empty());
        assert_eq!(tier.used_bytes(), 0);
        assert_eq!(tier.budget_bytes(), 1 << 20);
    }

    /// Threads inserting the same hashes at once leave `used_bytes` equal
    /// to the bytes of the entries the index holds.  The host slabs of the
    /// Vulkan backend are ordinary host memory, so no device is opened.
    #[cfg(feature = "vulkan")]
    #[test]
    fn concurrent_same_hash_inserts_keep_the_accounting_exact() {
        use crate::backend::vulkan::VulkanBackend;
        const THREADS: usize = 8;
        const ROUNDS: usize = 500;
        let tier = HostTier::<VulkanBackend>::new(HostBudget {
            host_bytes: 1 << 30,
        });
        let hashes = [ContentHash([0x21; 32]), ContentHash([0x42; 32])];
        let barrier = std::sync::Barrier::new(THREADS);
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (tier, barrier) = (&tier, &barrier);
                let _ = s.spawn(move || {
                    for round in 0..ROUNDS {
                        let _ = barrier.wait();
                        let bytes = 64 + 16 * ((t + round) % 5);
                        let slab = vec![0u8; bytes].into_boxed_slice();
                        let entry = HostEntry::new(slab, 8, 8, ImageLayout::Gray);
                        let _ = tier.insert(hashes[round % 2], entry, round as u64);
                    }
                });
            }
        });
        let held: u64 = tier.entries.iter().map(|e| e.value().host_bytes()).sum();
        assert_eq!(tier.len(), hashes.len());
        assert_eq!(tier.used_bytes(), held);
    }

    /// GPU integration test: allocate a pinned slab, write a pattern,
    /// install in the tier, look it up, verify the bytes survived.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn pinned_round_trip_through_host_tier() {
        let stream = cudarc::driver::CudaContext::new(0)
            .expect("CUDA device 0")
            .default_stream();
        let tier = HostTier::<CudaBackend>::new(HostBudget {
            host_bytes: 1 << 20,
        });
        let mut pinned = alloc_pinned(&stream, 256);
        // Fill with a known pattern.  Note: write-combined memory is
        // slow for CPU reads but writes are fine.
        for (i, b) in pinned
            .as_mut_slice()
            .expect("as_mut_slice")
            .iter_mut()
            .enumerate()
        {
            *b = u8::try_from(i & 0xFF).expect("low byte fits u8");
        }
        let hash = ContentHash([0x11; 32]);
        let entry = HostEntry::new(pinned, 16, 16, ImageLayout::Gray);
        let _ = tier.insert(hash, entry, /* tick */ 0);

        let hit = tier.lookup(&hash, /* tick */ 1).expect("hit");
        assert_eq!(hit.width, 16);
        assert_eq!(hit.height, 16);
        assert_eq!(hit.layout, ImageLayout::Gray);
        assert_eq!(hit.host_bytes(), 256);
        assert_eq!(tier.used_bytes(), 256);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn host_tier_evicts_oldest_on_overflow() {
        let stream = cudarc::driver::CudaContext::new(0)
            .expect("CUDA device 0")
            .default_stream();
        // Budget for two 256-byte entries but not three.
        let tier = HostTier::<CudaBackend>::new(HostBudget { host_bytes: 600 });

        let mk_entry = |fill: u8| -> HostEntry<CudaBackend> {
            let mut pinned = alloc_pinned(&stream, 256);
            for b in pinned.as_mut_slice().expect("as_mut_slice").iter_mut() {
                *b = fill;
            }
            HostEntry::new(pinned, 16, 16, ImageLayout::Gray)
        };

        let h_a = ContentHash([0xAA; 32]);
        let h_b = ContentHash([0xBB; 32]);
        let h_c = ContentHash([0xCC; 32]);
        let _ = tier.insert(h_a, mk_entry(1), 1);
        let _ = tier.insert(h_b, mk_entry(2), 2);

        // Touch B to make A the LRU candidate.
        let _ = tier.lookup(&h_b, 3);

        // Inserting C must evict A.
        let _ = tier.insert(h_c, mk_entry(3), 4);

        assert!(tier.lookup(&h_a, 5).is_none(), "A should have been evicted");
        assert!(tier.lookup(&h_b, 6).is_some(), "B should remain");
        assert!(tier.lookup(&h_c, 7).is_some(), "C just inserted");
    }
}
