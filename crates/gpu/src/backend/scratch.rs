//! Reusable buffers of the backend-generic device operations, pooled on the
//! backend that owns their device.

use std::sync::{Mutex, PoisonError};

use super::GpuBackend;
use super::aa_fill::AaFillScratch;
#[cfg(feature = "cache")]
use super::blit_page::BlitScratch;
use super::icc_clut::IccClutScratch;

/// Scratches lent to concurrent operations, one per operation in flight.
///
/// An operation takes an idle scratch, or a new one when every scratch is
/// lent, and returns it when done.  The pool holds as many scratches as
/// operations have run at once, each at the size of the largest operation
/// it served.
pub struct ScratchPool<T> {
    idle: Mutex<Vec<T>>,
}

impl<T> ScratchPool<T> {
    /// An empty pool.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
        }
    }

    /// `f` run on a scratch lent by the pool; the scratch is returned to the
    /// pool afterwards.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R
    where
        T: Default,
    {
        let mut scratch = self.take();
        let result = f(&mut scratch);
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
        result
    }

    fn take(&self) -> T
    where
        T: Default,
    {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default()
    }

    /// Scratches currently idle in the pool.
    #[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
    pub(crate) fn idle_len(&self) -> usize {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// An idle scratch taken out of the pool, or a new one.
    #[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
    pub(crate) fn take_idle(&self) -> T
    where
        T: Default,
    {
        self.take()
    }
}

impl<T> Default for ScratchPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The scratch pools of one backend ([`GpuBackend::scratch`]).
///
/// Their buffers belong to the backend's device and live as long as the
/// backend, so operations on later pages reuse the buffers of earlier ones.
pub struct DeviceScratch<B: GpuBackend> {
    /// Scratches of [`super::aa_fill`].
    pub(crate) aa_fill: ScratchPool<AaFillScratch<B>>,
    /// Scratches of [`super::blit_image`].
    #[cfg(feature = "cache")]
    pub(crate) blit: ScratchPool<BlitScratch<B>>,
    /// Scratches of [`super::icc_clut`].
    pub(crate) icc_clut: ScratchPool<IccClutScratch<B>>,
}

impl<B: GpuBackend> DeviceScratch<B> {
    /// Empty pools.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            aa_fill: ScratchPool::new(),
            #[cfg(feature = "cache")]
            blit: ScratchPool::new(),
            icc_clut: ScratchPool::new(),
        }
    }
}

impl<B: GpuBackend> Default for DeviceScratch<B> {
    fn default() -> Self {
        Self::new()
    }
}
