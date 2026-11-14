//! Device-resident image cache — three tiers.
//!
//! - VRAM tier: primary `DashMap<ContentHash, Arc<...>>` +
//!   `(DocId, ObjId)` alias index, refcount-pinned LRU eviction.
//! - Host RAM tier: demote-on-evict / promote-on-hit; in-process only.
//! - Disk tier: `<root>/<doc>/<hash>.bin` sidecar files for
//!   cross-process persistence; opt-in via [`DeviceImageCache::with_disk`].
//!
//! The cache is generic over the backend its slabs live on
//! ([`CacheBackend`]: CUDA, and Vulkan with the `vulkan` feature); the disk
//! tier is host-only and shared by both.  [`ImageCache`] and [`CachedImage`]
//! hold a cache and an image of either backend.
//!
//! # Module layout (private; named for orientation only)
//!
//! - `any` — the backend-erased [`ImageCache`] and [`CachedImage`].
//! - `backend` — the per-backend transfers ([`CacheBackend`]).
//! - `budget` — VRAM cap struct; auto-detect from the driver.
//! - `eviction` — LRU scan, demote-to-host, monotonic LRU clock.
//! - `promotion` — host-tier and disk-tier promote-on-hit paths.
//! - `host_tier` — host-memory tier.
//! - `disk_tier` — sidecar-file tier (re-exported via [`DiskTier`]).
//! - `page_buffer` — per-page composition target ([`DevicePageBuffer`]).
//!
//! # Concurrency model
//!
//! The cache is `&self` everywhere and safe to call from any number of
//! rayon worker threads.  Internal indices are [`dashmap`]-backed; LRU
//! timestamps are atomic; refcounts come for free from `Arc`.
//!
//! # Lifetime model
//!
//! [`CachedDeviceImage`] is wrapped in `Arc`.  The cache holds one strong
//! reference; every consumer (renderer, blit kernel) that takes a value
//! out of `DeviceImageCache::lookup_by_*` holds another.  Eviction drops the
//! cache's strong reference, leaving the slab alive until all in-flight
//! kernels finish.  This means a slab can stay live in VRAM after eviction
//! until the last consumer drops its `Arc` — that's the desired behaviour:
//! pulling memory from under an in-flight kernel would corrupt the page.

mod any;
mod backend;
mod budget;
mod disk_tier;
mod eviction;
mod host_tier;
mod page_buffer;
mod promotion;

pub use crate::RGBA_BPP;
pub use any::{CachedImage, ImageCache};
pub use backend::CacheBackend;
pub use budget::VramBudget;
pub use disk_tier::{DiskTier, LookupCallbackError};
pub use host_tier::HostBudget;
pub(crate) use host_tier::HostTier;
pub use page_buffer::DevicePageBuffer;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::CudaStream;
use dashmap::DashMap;

use crate::backend::BackendError;
use crate::backend::cuda::CudaBackend;
use dashmap::mapref::entry::Entry;

/// Stable identifier for a PDF document.
///
/// Used as part of the secondary cache key.  The cache treats it as an
/// opaque 32-byte content-addressable identifier; sessions use the
/// BLAKE3 hash of the PDF's bytes ([`ContentHash::of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocId(pub [u8; 32]);

/// Stable identifier for an image within a PDF document.
///
/// Combined with [`DocId`] it forms the secondary alias key for fast
/// same-document lookup that bypasses content hashing.
///
/// **Note:** PDF's authoritative object identifier is `(number, generation)`
/// (see `pdf::ObjectId`).  This newtype intentionally drops the generation
/// because image `XObjects` are written once and never bumped in any PDF
/// produced by tools we encounter.  If a workflow ever rewrites image
/// objects with new generations, callers must construct distinct `ObjId`s
/// (e.g. by hashing `(number, generation)` into the u32) to keep the alias
/// index from returning stale content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjId(pub u32);

/// Component layout of a cached decoded image.  Matches `ImageColorSpace`
/// in `rasterrocket-interp` but kept here to avoid a circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    /// One byte per pixel: gray.
    Gray,
    /// Three bytes per pixel: R, G, B.
    Rgb,
    /// One byte per pixel: 0x00 = paint, 0xFF = transparent.  Byte-expanded
    /// from a logical 1-bit-per-pixel mask.
    Mask,
}

impl ImageLayout {
    /// Bytes per pixel for the byte-expanded layout.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray | Self::Mask => 1,
            Self::Rgb => 3,
        }
    }

    /// The blit kernels' source layout code
    /// ([`crate::backend::params::BlitParams::src_layout`]): `0` for RGB,
    /// `1` for gray; `None` for a mask, which the kernels do not sample.
    #[must_use]
    pub const fn blit_layout(self) -> Option<u32> {
        match self {
            Self::Rgb => Some(0),
            Self::Gray => Some(1),
            Self::Mask => None,
        }
    }
}

/// One cached decoded image living in VRAM.
///
/// `Arc<CachedDeviceImage>` is the only public handle.  Once obtained via
/// [`DeviceImageCache::lookup_by_id`], [`DeviceImageCache::lookup_by_hash`],
/// or [`DeviceImageCache::insert`], the slab is guaranteed alive for the
/// lifetime of the `Arc` — eviction may drop the cache's reference, but
/// the device memory is reclaimed only when the last `Arc` drops.  This
/// guarantee is what makes the cache safe to use across in-flight device
/// work: a blit holds the `Arc` until the work reading the slab completes.
pub struct CachedDeviceImage<B: CacheBackend = CudaBackend> {
    /// Owned device-side allocation.  Dropping `CachedDeviceImage` frees it.
    pub device_ptr: B::DeviceBuffer,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel and renderer interpretation.
    pub layout: ImageLayout,
    /// Byte length of `device_ptr`.
    vram_bytes: u64,
    /// LRU timestamp.  Updated on every cache hit via a single atomic store
    /// (no CAS needed; later writes always win and exact ordering doesn't
    /// matter for LRU correctness).
    pub(super) last_used: AtomicU64,
}

impl<B: CacheBackend> CachedDeviceImage<B> {
    /// An entry owning `slab`, `vram_bytes` long, last used at `tick`.
    pub(super) const fn new(
        slab: B::DeviceBuffer,
        vram_bytes: u64,
        (width, height, layout): (u32, u32, ImageLayout),
        tick: u64,
    ) -> Self {
        Self {
            device_ptr: slab,
            width,
            height,
            layout,
            vram_bytes,
            last_used: AtomicU64::new(tick),
        }
    }

    /// Bytes in VRAM occupied by this entry.  Used for budget accounting.
    #[must_use]
    pub const fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub(super) fn touch(&self, tick: u64) {
        // Relaxed is sufficient: LRU ordering is approximate and doesn't
        // synchronise any other state.
        self.last_used.store(tick, Ordering::Relaxed);
    }

    pub(super) fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }
}

impl<B: CacheBackend> std::fmt::Debug for CachedDeviceImage<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedDeviceImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .field("vram_bytes", &self.vram_bytes())
            .finish_non_exhaustive()
    }
}

/// Errors from cache operations.
#[derive(Debug)]
pub enum CacheError {
    /// A `cudarc` driver call failed (allocation, upload, sync, or budget probe).
    Cuda(cudarc::driver::DriverError),
    /// A [`crate::backend::GpuBackend`] call failed (allocation, transfer,
    /// or budget probe).
    Backend(BackendError),
    /// The cache cannot fit a new entry because every reclaimable slab is
    /// pinned by an outstanding `Arc` (e.g. held by an in-flight blit).
    /// This condition is **transient**: once consumers drop their `Arc`s
    /// — typically once the device work that read the slab has
    /// completed — retry will succeed.
    OverBudget {
        /// Bytes the new entry needs.
        needed: u64,
        /// Total cache occupancy at the time of the failure.  This counts
        /// every entry held by `primary`, including ones whose only
        /// reference is the cache itself; the eviction scan tried and
        /// could not reclaim enough of them.
        used: u64,
        /// Total budget in bytes.
        budget: u64,
    },
    /// A new entry alone exceeds the entire VRAM budget.  Either the
    /// budget is misconfigured or the image is unusably large.
    EntryExceedsBudget {
        /// Bytes the new entry needs.
        needed: u64,
        /// Total budget in bytes.
        budget: u64,
    },
    /// `insert` was called with a zero-length pixel buffer.  Device
    /// allocators reject zero-size allocations, so this is surfaced as a
    /// typed caller error rather than a driver error.
    EmptyPayload,
    /// An image of one backend was passed to a cache of another.
    BackendMismatch,
    /// The host refused an allocation of `bytes` for a host-tier slab.
    HostAlloc {
        /// Bytes requested.
        bytes: usize,
    },
}

impl From<BackendError> for CacheError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

impl CacheError {
    pub(super) const fn cuda(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cuda(e) => write!(f, "cuda: {e}"),
            Self::Backend(e) => write!(f, "gpu backend: {e}"),
            Self::OverBudget {
                needed,
                used,
                budget,
            } => write!(
                f,
                "cache over budget (transient): needed {needed} B, {used} B held by the cache, budget {budget} B; no entry could be evicted, because each is pinned by an outstanding image handle or eviction lost repeated races — retry once outstanding handles drop",
            ),
            Self::EntryExceedsBudget { needed, budget } => write!(
                f,
                "single image {needed} B does not fit in cache budget of {budget} B",
            ),
            Self::EmptyPayload => write!(f, "insert called with zero-length pixel buffer"),
            Self::BackendMismatch => write!(f, "image belongs to a cache of another backend"),
            Self::HostAlloc { bytes } => {
                write!(f, "host allocation of {bytes} B for a cache slab failed")
            }
        }
    }
}

impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            Self::Backend(e) => Some(e),
            Self::OverBudget { .. }
            | Self::EntryExceedsBudget { .. }
            | Self::EmptyPayload
            | Self::BackendMismatch
            | Self::HostAlloc { .. } => None,
        }
    }
}

/// 32-byte BLAKE3 content hash used as the cache's primary key.
///
/// Newtype so it can't be confused with [`DocId`] (also `[u8; 32]`)
/// at a call site — different roles, different keys.  Construct via
/// [`ContentHash::of`] or `ContentHash(bytes)` directly when the caller
/// already has the hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// The BLAKE3 content hash of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
}

/// Three-tier image cache.
///
/// VRAM is the hot tier; host RAM is the demotion target; an optional disk
/// tier ([`Self::with_disk`]) persists decoded images across sessions.
///
/// The VRAM tier is indexed by content hash (primary, for cross-
/// document dedup) and by `(DocId, ObjId)` (secondary, fast same-
/// document lookup).  The host tier is indexed only by content hash
/// and is consulted on a VRAM miss before re-decoding.
///
/// All methods take `&self` and are safe to call from many threads
/// concurrently.  Eviction is read-then-CAS rather than locking the
/// whole tier, so a hot path that touches a present entry never
/// contends with eviction.
///
/// LRU clock (`tick`) is shared across tiers so a host-tier hit's
/// "freshness" is comparable to a VRAM hit's; otherwise a freshly
/// promoted slab would look like the oldest VRAM entry to the next
/// eviction.
pub struct DeviceImageCache<B: CacheBackend = CudaBackend> {
    pub(super) primary: DashMap<ContentHash, Arc<CachedDeviceImage<B>>>,
    /// Maps `(DocId, ObjId)` to the content hash that resolves it.  An
    /// alias entry is cheap (40 bytes) and lets us skip BLAKE3 hashing
    /// on every same-document re-render.
    pub(super) by_doc_obj: DashMap<(DocId, ObjId), ContentHash>,
    /// Host RAM demotion target.  Populated when the eviction path
    /// reclaims a VRAM slab; consulted by [`Self::lookup_by_hash`] on
    /// a primary miss.
    pub(super) host: HostTier<B>,
    /// Disk tier — sidecar cache directory keyed by
    /// `(DocId, ContentHash)`.  `None` disables disk persistence
    /// (fully in-process cache).  Wired in via [`Self::with_disk`].
    pub(super) disk: Option<DiskTier>,
    pub(super) budget: VramBudget,
    /// Monotonic LRU clock.  Incremented on every observable cache
    /// event in either tier.
    pub(super) tick: AtomicU64,
    /// Sum of `vram_bytes()` over every entry currently held by the
    /// primary index.  Maintained on insert / evict.
    pub(super) used_bytes: AtomicU64,
    /// The handle every transfer of the cache goes through
    /// ([`CacheBackend::Device`]).
    pub(super) device: B::Device,
}

/// Bundled arguments for [`DeviceImageCache::insert`].
///
/// Passed by struct rather than by parameter list because the call site
/// is a hot but readable construction; eight-positional `insert(doc,
/// obj, hash, w, h, layout, pixels)` would be easy to misread (which
/// dimension is which?).
#[derive(Debug)]
pub struct InsertRequest<'a> {
    /// Document this image belongs to (for the secondary alias index).
    pub doc: DocId,
    /// PDF object id within the document (for the secondary alias index).
    pub obj: ObjId,
    /// Content hash (BLAKE3) of the encoded source bytes — primary key
    /// for cross-document deduplication.  Caller computes via
    /// [`ContentHash::of`].
    pub hash: ContentHash,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel layout — determines bytes per pixel.
    pub layout: ImageLayout,
    /// Decoded host-side pixels.  Length must equal
    /// `width × height × layout.bytes_per_pixel()`; the cache trusts
    /// the caller and uploads the slice verbatim.
    pub pixels: &'a [u8],
}

impl<B: CacheBackend> DeviceImageCache<B> {
    /// Build an empty cache bound to a backend handle, a VRAM budget, and
    /// a host-RAM demotion budget.  The disk tier is off by default;
    /// enable it via [`Self::with_disk`].
    #[must_use]
    pub fn new(device: B::Device, vram: VramBudget, host: HostBudget) -> Self {
        Self {
            primary: DashMap::new(),
            by_doc_obj: DashMap::new(),
            host: HostTier::new(host),
            disk: None,
            budget: vram,
            tick: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            device,
        }
    }

    /// Attach a disk tier for cross-process persistence.
    ///
    /// On a VRAM and host-RAM miss, a disk hit replaces the decode with a
    /// read of the decoded samples.  Entries are keyed by `DocId`, which
    /// callers derive from the PDF's bytes, not its path, so editing the
    /// source PDF leaves its old entries unreachable.
    #[must_use]
    pub fn with_disk(mut self, disk: DiskTier) -> Self {
        self.disk = Some(disk);
        self
    }

    /// Live entry count — for diagnostics, tests, and benches.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primary.len()
    }

    /// Whether the cache currently holds zero entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primary.is_empty()
    }

    /// Live VRAM usage in bytes.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Budget in bytes.
    #[must_use]
    pub const fn budget_bytes(&self) -> u64 {
        self.budget.vram_bytes
    }

    /// Probe the cache by `(DocId, ObjId)`.  Returns `None` if the
    /// alias is unknown or its content hash is in neither tier; on a
    /// host-tier hit, transparently promotes to VRAM (one
    /// upload).  Updates LRU.
    #[must_use]
    pub fn lookup_by_id(&self, doc: DocId, obj: ObjId) -> Option<Arc<CachedDeviceImage<B>>> {
        let hash = *self.by_doc_obj.get(&(doc, obj))?;
        // Try the in-process tiers first; on miss, fall through to the
        // disk tier which is keyed by `(doc, hash)` (cross-document
        // dedup is in-process only — the disk format scopes entries
        // to the doc-id directory for clean eviction granularity).
        if let Some(entry) = self.lookup_by_hash(&hash) {
            return Some(entry);
        }
        self.promote_from_disk(doc, &hash)
    }

    /// Probe the cache by content hash.  On a VRAM hit, returns the
    /// device-resident handle and bumps LRU.  On a VRAM miss, falls
    /// through to the host tier and promotes back to VRAM (one
    /// upload).  Returns `None` only if the hash is in neither tier or
    /// the promotion upload fails.
    ///
    /// Does NOT probe the disk tier — that's keyed by `(doc, hash)`
    /// and only [`Self::lookup_by_id`] has the doc-id needed for the
    /// lookup.
    #[must_use]
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage<B>>> {
        if let Some(entry) = self.primary.get(hash) {
            let entry = entry.clone();
            entry.touch(self.next_tick());
            return Some(entry);
        }
        self.promote_from_host(hash)
    }

    /// Three-tier lookup keyed on a content hash + the doc-id used
    /// to scope disk-tier entries.  Consults VRAM, host RAM, and
    /// disk in that order.  Used by `decode_dct` after the
    /// `(doc, obj)` alias miss + BLAKE3 hash compute, so a cold
    /// process with a populated disk cache hits the disk tier
    /// instead of paying for a re-decode.
    ///
    /// Re-binds the alias on a hit so subsequent same-document
    /// lookups go through `lookup_by_id` and skip the hash compute.
    #[must_use]
    pub fn lookup_by_hash_for_doc(
        &self,
        doc: DocId,
        obj: ObjId,
        hash: &ContentHash,
    ) -> Option<Arc<CachedDeviceImage<B>>> {
        if let Some(entry) = self.lookup_by_hash(hash) {
            self.alias(doc, obj, *hash);
            return Some(entry);
        }
        let promoted = self.promote_from_disk(doc, hash)?;
        self.alias(doc, obj, *hash);
        Some(promoted)
    }

    /// Bind an existing primary entry to a `(DocId, ObjId)` alias.  Used
    /// after a content-hash hit to make subsequent same-document lookups
    /// O(1).
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if `(doc, obj)` is already bound to
    /// a *different* hash: in a correct PDF workflow object identity is
    /// stable, so a hash change for the same `(DocId, ObjId)` is either
    /// a caller bug (mis-hashed bytes) or a BLAKE3 collision (negligible
    /// probability).  Either way, silently overwriting would corrupt the
    /// alias index.  Release builds tolerate the overwrite to keep
    /// adversarial PDFs from aborting the renderer.
    pub fn alias(&self, doc: DocId, obj: ObjId, hash: ContentHash) {
        if cfg!(debug_assertions)
            && let Some(existing) = self.by_doc_obj.get(&(doc, obj))
            && *existing != hash
        {
            panic!(
                "alias({doc:?}, {obj:?}, …) tried to overwrite {:?} with {hash:?}; \
                 PDF object id should map to a stable content hash",
                *existing,
            );
        }
        let _ = self.by_doc_obj.insert((doc, obj), hash);
    }

    /// Upload host pixels to VRAM, register them under both keys, and
    /// return the cached handle.  Evicts in LRU order if the new entry
    /// would exceed the budget.
    ///
    /// # Ordering contract
    ///
    /// On Vulkan the upload has completed when this returns.  On CUDA it
    /// is enqueued on the cache's internal stream (see
    /// `DeviceImageCache::<CudaBackend>::stream`) and records the slab's
    /// write event.  A kernel launched against
    /// [`CachedDeviceImage::device_ptr`] through cudarc on any other stream
    /// of the same context waits on that event, so it never reads the slab
    /// before the H→D DMA completes.  A consumer that reads the slab
    /// through a raw device pointer must synchronise the cache's stream
    /// first.
    ///
    /// # Errors
    ///
    /// - [`CacheError::EmptyPayload`] — `pixels.is_empty()` (device
    ///   allocators refuse zero-size allocations; this is a typed caller
    ///   error).
    /// - [`CacheError::EntryExceedsBudget`] — the entry alone is larger
    ///   than the configured budget.
    /// - [`CacheError::OverBudget`] — no eviction can free enough space
    ///   (every other entry is pinned by an outstanding `Arc`).  The
    ///   condition is transient.
    /// - [`CacheError::Cuda`] / [`CacheError::Backend`] — the upload itself
    ///   failed.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "all fields are trivially movable (Copy newtypes plus a `&[u8]` slice header); by-value conveys ownership transfer at the call site without changing codegen vs `&InsertRequest`"
    )]
    pub fn insert(&self, req: InsertRequest<'_>) -> Result<Arc<CachedDeviceImage<B>>, CacheError> {
        let InsertRequest {
            doc,
            obj,
            hash,
            width,
            height,
            layout,
            pixels,
        } = req;
        if pixels.is_empty() {
            return Err(CacheError::EmptyPayload);
        }
        // Decoder contract: pixel buffer length matches the dimensions /
        // layout the caller declared.  A mismatched buffer would land
        // wrong-shaped bytes in VRAM and surface as a corrupt blit; loud
        // failure in debug, trust the caller in release.
        debug_assert_eq!(
            pixels.len(),
            (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(layout.bytes_per_pixel()),
            "pixel buffer length does not match width × height × bpp",
        );
        let needed = pixels.len() as u64;
        if needed > self.budget.vram_bytes {
            return Err(CacheError::EntryExceedsBudget {
                needed,
                budget: self.budget.vram_bytes,
            });
        }
        if let Some(existing) = self.lookup_by_hash(&hash) {
            self.alias(doc, obj, hash);
            return Ok(existing);
        }
        // Single Entry transition below so concurrent inserts of the same
        // hash can't both fetch_add `used_bytes` after their Vacant probe.
        self.evict_to_fit(needed)?;
        let slab = B::upload_slab(&self.device, pixels)?;
        let new_entry = Arc::new(CachedDeviceImage::new(
            slab,
            needed,
            (width, height, layout),
            self.next_tick(),
        ));
        match self.primary.entry(hash) {
            Entry::Occupied(occ) => {
                // Lost the dedup race.  `new_entry` drops on function
                // exit, freeing its just-uploaded slab; we never
                // touched `used_bytes`, so accounting stays correct.
                let existing = occ.get().clone();
                drop(occ);
                self.alias(doc, obj, hash);
                existing.touch(self.next_tick());
                Ok(existing)
            }
            Entry::Vacant(vac) => {
                let _ = self
                    .used_bytes
                    .fetch_add(new_entry.vram_bytes(), Ordering::Relaxed);
                let _ = vac.insert(new_entry.clone());
                self.alias(doc, obj, hash);
                // Persist to disk for cross-process / cross-session
                // reuse.  Best-effort: write errors are logged inside
                // `disk.insert` and don't fail the in-memory insert
                // — the renderer always gets a valid `Arc<...>`.
                if let Some(disk) = self.disk.as_ref() {
                    disk.insert(doc, hash, width, height, layout, pixels);
                }
                Ok(new_entry)
            }
        }
    }

    /// The handle the cache issues its transfers through.
    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    /// The pixels of `image` copied back to the host, in the layout
    /// [`CachedDeviceImage::layout`] describes.  The copy is complete when
    /// this returns.
    ///
    /// # Errors
    /// The backend's error of the copy.
    pub fn download(&self, image: &CachedDeviceImage<B>) -> Result<Vec<u8>, CacheError> {
        let len = usize::try_from(image.vram_bytes())
            .map_err(|_| CacheError::Backend(BackendError::msg("slab exceeds usize")))?;
        let mut pixels = vec![0u8; len];
        B::download_slab(&self.device, &image.device_ptr, &mut pixels)?;
        Ok(pixels)
    }
}

impl DeviceImageCache<CudaBackend> {
    /// The CUDA stream on which the cache enqueues uploads.  Exposed so
    /// a caller reading a freshly inserted [`CachedDeviceImage::device_ptr`]
    /// through a raw device pointer can synchronise it first.
    #[must_use]
    pub fn stream(&self) -> &CudaStream {
        &self.device
    }
}

// Compile-time assertion: the cache and its handle are thread-safe.  The
// slabs are the backends' `DeviceBuffer`s and the handles their
// `Device`s, both `Send + Sync` by the trait bounds; the rest of the
// state is atomics and `dashmap`.
const fn _assert_thread_safety()
where
    DeviceImageCache: Send + Sync,
    Arc<CachedDeviceImage>: Send + Sync,
{
}

#[cfg(test)]
mod tests;
