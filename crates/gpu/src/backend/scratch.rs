//! Reusable buffers of the backend-generic device operations, pooled on the
//! backend that owns their device.

use std::sync::{Condvar, Mutex, PoisonError};

use super::GpuBackend;
#[cfg(feature = "cache")]
use super::blit_page::BlitScratch;
use super::icc_clut::IccClutScratch;

/// Scratches lent to concurrent operations, one per operation in flight.
///
/// An operation takes an idle scratch, or a new one while fewer than the
/// pool's limit exist, and otherwise waits until one is returned; it
/// returns the scratch when done, also when it unwinds.  The pool holds as
/// many scratches as operations have run at once, at most its limit, each
/// at the size of the largest operation it served.
pub struct ScratchPool<T> {
    state: Mutex<Lending<T>>,
    /// Notified when a scratch is returned.
    returned: Condvar,
    /// The most scratches the pool makes.
    limit: usize,
}

/// The scratches of a [`ScratchPool`] not lent, and how many it has made.
struct Lending<T> {
    idle: Vec<T>,
    made: usize,
}

/// A scratch lent by a [`ScratchPool`], returned to it on drop.
pub(crate) struct Lent<'a, T> {
    pool: &'a ScratchPool<T>,
    scratch: Option<T>,
}

impl<T> std::ops::Deref for Lent<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.scratch
            .as_ref()
            .expect("a lent scratch is held until drop")
    }
}

impl<T> std::ops::DerefMut for Lent<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.scratch
            .as_mut()
            .expect("a lent scratch is held until drop")
    }
}

impl<T> Drop for Lent<'_, T> {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.pool
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .idle
                .push(scratch);
            self.pool.returned.notify_one();
        }
    }
}

impl<T> ScratchPool<T> {
    /// An empty pool without a limit.
    #[must_use]
    pub const fn new() -> Self {
        Self::bounded(usize::MAX)
    }

    /// An empty pool that makes at most `limit` scratches (at least one).
    #[must_use]
    pub const fn bounded(limit: usize) -> Self {
        Self {
            state: Mutex::new(Lending {
                idle: Vec::new(),
                made: 0,
            }),
            returned: Condvar::new(),
            limit: if limit == 0 { 1 } else { limit },
        }
    }

    /// `f` run on a scratch lent by the pool; the scratch is returned to the
    /// pool afterwards.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R
    where
        T: Default,
    {
        f(&mut self.lend())
    }

    /// A scratch lent by the pool until the returned guard drops.
    pub(crate) fn lend(&self) -> Lent<'_, T>
    where
        T: Default,
    {
        Lent {
            pool: self,
            scratch: Some(self.take()),
        }
    }

    /// An idle scratch, a new one while fewer than the limit exist, or else
    /// the next one returned.
    fn take(&self) -> T
    where
        T: Default,
    {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(scratch) = state.idle.pop() {
                return scratch;
            }
            if state.made < self.limit {
                state.made += 1;
                return T::default();
            }
            state = self
                .returned
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Drop the idle scratches, freeing what they hold, and return how
    /// many were dropped; the pool makes new ones on demand.  A scratch is
    /// idle only once the operation it was lent to has returned, so no
    /// device work refers to its buffers.
    #[cfg(feature = "vulkan")]
    pub(crate) fn free_idle(&self) -> usize {
        let idle = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let idle = std::mem::take(&mut state.idle);
            state.made -= idle.len();
            idle
        };
        // Waiters may now make new scratches in place of those dropped.
        self.returned.notify_all();
        idle.len()
    }

    /// Scratches currently idle in the pool.
    #[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
    pub(crate) fn idle_len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .idle
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
    /// Scratches of [`super::blit_image`].
    #[cfg(feature = "cache")]
    pub(crate) blit: ScratchPool<BlitScratch<B>>,
    /// Scratches of [`super::icc_clut()`].
    pub(crate) icc_clut: ScratchPool<IccClutScratch<B>>,
}

impl<B: GpuBackend> DeviceScratch<B> {
    /// Empty pools.
    #[must_use]
    pub const fn new() -> Self {
        Self {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::ScratchPool;

    /// A bounded pool lends at most its limit at once: a further borrower
    /// waits for a scratch to be returned and gets that one, so no more
    /// than the limit are ever made.
    #[test]
    fn a_bounded_pool_lends_at_most_its_limit() {
        let pool = ScratchPool::<Vec<u8>>::bounded(2);
        let lent = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for i in 0..8u8 {
                let (pool, lent, most) = (&pool, &lent, &most);
                let _ = scope.spawn(move || {
                    pool.with(|scratch| {
                        let now = lent.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = most.fetch_max(now, Ordering::SeqCst);
                        scratch.push(i);
                        std::thread::sleep(Duration::from_millis(5));
                        let _ = lent.fetch_sub(1, Ordering::SeqCst);
                    });
                });
            }
        });
        assert!(most.load(Ordering::SeqCst) <= 2);
        let state = pool.state.lock().expect("lock");
        assert_eq!((state.made, state.idle.len()), (2, 2));
        let pushes: usize = state.idle.iter().map(Vec::len).sum();
        drop(state);
        assert_eq!(pushes, 8, "every borrower used one of the two scratches");
    }

    /// Freeing the idle scratches drops them and lets the pool make new
    /// ones in their place, up to its limit; a lent scratch is kept.
    #[test]
    #[cfg(feature = "vulkan")]
    fn freeing_the_idle_scratches_lets_the_pool_make_new_ones() {
        let pool = ScratchPool::<Vec<u8>>::bounded(2);
        let mut kept = pool.lend();
        kept.push(1);
        pool.with(|scratch| scratch.push(2));
        assert_eq!(pool.free_idle(), 1);
        assert_eq!(pool.free_idle(), 0);
        pool.with(|scratch| assert!(scratch.is_empty(), "a new scratch"));
        drop(kept);
        let state = pool.state.lock().expect("lock");
        assert_eq!((state.made, state.idle.len()), (2, 2));
    }

    /// A scratch whose borrower panics is returned to the pool.
    #[test]
    fn a_scratch_is_returned_when_its_borrower_unwinds() {
        let pool = ScratchPool::<Vec<u8>>::bounded(1);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.with(|scratch| {
                scratch.push(1);
                panic!("borrower failed");
            });
        }));
        assert!(unwound.is_err());
        assert_eq!(pool.with(|scratch| scratch.clone()), [1]);
    }
}
