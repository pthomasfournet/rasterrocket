//! Tier-promotion logic for the image cache.
//!
//! Pulled out of `mod.rs` so the host-tier and disk-tier promote-on-hit
//! paths live in one focused module.  These run on a VRAM miss and lift
//! the entry back into VRAM (one upload) so the renderer never
//! sees a re-decode for an image that's still anywhere in the cache.

use std::sync::Arc;

use dashmap::mapref::entry::Entry;

use super::disk_tier::LookupCallbackError;
use super::host_tier::HostEntry;
use super::{CacheBackend, CachedDeviceImage, ContentHash, DeviceImageCache, DocId};

impl<B: CacheBackend> DeviceImageCache<B> {
    /// Lift a host-tier entry back into VRAM and install it in the
    /// primary index.  Returns `None` if the host tier doesn't have it
    /// or the upload fails (treated as a cache miss; caller re-decodes).
    pub(super) fn promote_from_host(
        &self,
        hash: &ContentHash,
    ) -> Option<Arc<CachedDeviceImage<B>>> {
        let host_entry = self.host.lookup(hash, self.next_tick())?;
        let bytes = host_entry.host_bytes();
        // Make room in the VRAM tier and reserve it.  If we can't (every
        // entry pinned by in-flight kernels), fall back to a cache miss;
        // the host entry stays in place for a future retry.
        let reservation = self.evict_to_fit(bytes).ok()?;
        // `promote` keeps the host slab alive until the copy has read
        // it (see `CacheBackend::promote` for the CUDA mechanism).
        let slab = B::promote(&self.device, &host_entry.pinned)
            .map_err(|e| log::warn!("host-tier promotion upload failed: {e}"))
            .ok()?;
        let new_entry = Arc::new(CachedDeviceImage::new(
            slab,
            bytes,
            (host_entry.width, host_entry.height, host_entry.layout),
            self.next_tick(),
        ));
        match self.primary.entry(*hash) {
            Entry::Occupied(occ) => {
                // Another thread won the same race: return its entry,
                // touched as the insert path touches it; the reservation
                // drops, releasing its bytes.
                let existing = occ.get().clone();
                existing.touch(self.next_tick());
                Some(existing)
            }
            Entry::Vacant(vac) => {
                reservation.commit();
                let _ = vac.insert(new_entry.clone());
                Some(new_entry)
            }
        }
    }

    /// Lift a disk-tier entry back into VRAM.  Reads the file via
    /// `posix_fadvise(WILLNEED)` + `read_exact` directly into a
    /// fresh host slab (no transient `Vec`), uploads to
    /// VRAM, and installs in the primary index.  The host tier is
    /// also populated as a side effect so a later same-session
    /// lookup that misses VRAM hits host RAM instead of going back
    /// to disk.
    ///
    /// Returns `None` on disk miss, malformed file, allocation
    /// failure, or upload error — caller treats as a cache miss and
    /// re-decodes the source bytes.
    pub(super) fn promote_from_disk(
        &self,
        doc: DocId,
        hash: &ContentHash,
    ) -> Option<Arc<CachedDeviceImage<B>>> {
        let disk = self.disk.as_ref()?;
        // Slab outlives the callback; the closure fills it via
        // `read_exact` and hands ownership back via `pinned`.
        let mut pinned: Option<B::HostSlab> = None;
        let info = disk.lookup_into(doc, hash, |info, reader| {
            // Host alloc and slab-access failures are caller-side
            // — tag them `Resource` so a transient pinned-pool
            // exhaustion doesn't delete the (perfectly fine) disk
            // entry.
            let mut slab =
                B::alloc_host_slab(&self.device, info.expected_pixel_bytes).map_err(|e| {
                    LookupCallbackError::Resource(std::io::Error::other(format!(
                        "host slab allocation failed: {e}"
                    )))
                })?;
            let dst = B::host_slab_mut(&mut slab).map_err(|e| {
                LookupCallbackError::Resource(std::io::Error::other(format!(
                    "host slab access failed: {e}"
                )))
            })?;
            reader.read_exact(dst).map_err(LookupCallbackError::Read)?;
            pinned = Some(slab);
            Ok(())
        })?;
        let pinned = pinned?;
        log::debug!(
            "disk-tier: hit for ({doc:?}, {hash:?}) {}×{} {:?}",
            info.width,
            info.height,
            info.layout,
        );
        let host_entry = HostEntry::new(pinned, info.width, info.height, info.layout);
        let _arc = self.host.insert(*hash, host_entry, self.next_tick());
        self.promote_from_host(hash)
    }
}
