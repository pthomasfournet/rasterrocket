//! The CUDA contexts of the [`GpuCtx`]s alive in the process, and the CUDA
//! state a thread keeps for them.
//!
//! A thread keeps per-context state — its stream on each context, its
//! page-locked staging buffers — in thread-local entries, each holding its
//! context.  A context with an entry therefore stays allocated, so its
//! address identifies it for as long as an entry names it.  An entry whose
//! context no live [`GpuCtx`] holds is stale: the thread drops its stale
//! entries whenever it adds one, so a thread holds at most the entries of
//! the contexts that ended since its last addition, and all of them are
//! dropped by [`release_thread_state`].
//!
//! [`GpuCtx`]: crate::GpuCtx

use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::CudaContext;

/// The address of the context of each live [`GpuCtx`](crate::GpuCtx).
static LIVE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn address(ctx: &Arc<CudaContext>) -> usize {
    Arc::as_ptr(ctx).addr()
}

fn live() -> std::sync::MutexGuard<'static, Vec<usize>> {
    LIVE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// List `ctx` as held by a live [`GpuCtx`](crate::GpuCtx).
#[cfg(not(ptx_placeholder))]
pub(crate) fn register(ctx: &Arc<CudaContext>) {
    live().push(address(ctx));
}

/// Remove the listing of `ctx` made by [`register`].
pub(crate) fn unregister(ctx: &Arc<CudaContext>) {
    let at = address(ctx);
    let mut live = live();
    if let Some(i) = live.iter().position(|&a| a == at) {
        let _ = live.swap_remove(i);
    }
}

/// Drop the entries of `entries` whose context, given by `context`, no live
/// [`GpuCtx`](crate::GpuCtx) holds.  The entries are dropped after the
/// listing is unlocked, since dropping one calls into the driver.
pub(crate) fn retain_live<T>(entries: &mut Vec<T>, context: impl Fn(&T) -> &Arc<CudaContext>) {
    let stale: Vec<T> = {
        let live = live();
        let (kept, stale) = std::mem::take(entries)
            .into_iter()
            .partition(|e| live.contains(&address(context(e))));
        *entries = kept;
        stale
    };
    drop(stale);
}

/// The [`GpuCtx`](crate::GpuCtx)s alive in the process.
#[must_use]
pub fn open_contexts() -> usize {
    live().len()
}

/// Drop the CUDA state the calling thread holds.
///
/// That is its stream on every context, its page-locked staging buffers
/// (`cache` feature) and its NPP rotator (`gpu-deskew` feature); each is
/// created again on the thread's next use.
///
/// A thread-local entry otherwise lives until its thread exits, and keeps
/// its context alive until then.  The threads of a pool that is dropped
/// without being joined exit while the process may already be running its
/// exit handlers; calling this on each of them first (e.g. through
/// `rayon::ThreadPool::broadcast`) leaves them no CUDA call to make then.
pub fn release_thread_state() {
    drop(crate::THREAD_STREAMS.take());
    #[cfg(feature = "cache")]
    crate::staging::release_thread_staging();
    #[cfg(feature = "gpu-deskew")]
    crate::npp_rotate::release_thread_rotator();
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use std::sync::Arc;

    use super::release_thread_state;
    use crate::GpuCtx;

    /// A thread's stream on a context whose `GpuCtx` dropped is kept until
    /// the thread creates its next stream, and the current one until
    /// [`release_thread_state`].
    #[test]
    fn a_thread_drops_its_streams_on_ended_contexts() {
        let first = GpuCtx::init().expect("a CUDA context");
        let ended = Arc::downgrade(&first.stream());
        drop(first);
        assert!(ended.upgrade().is_some(), "the thread holds its stream");

        let second = GpuCtx::init().expect("a CUDA context");
        let current = Arc::downgrade(&second.stream());
        assert!(
            ended.upgrade().is_none(),
            "the stream of an ended context outlived the thread's next stream"
        );
        release_thread_state();
        assert!(
            current.upgrade().is_none(),
            "release_thread_state kept the thread's stream"
        );
    }
}
