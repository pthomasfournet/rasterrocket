//! The devices of a GPU-page session, their opening and the pages they
//! hold.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use gpu::backend::BackendError;
use gpu::backend::page_device::PageDevice;

use crate::device_init::SessionDevice;

/// One device of a session's GPU pages.
pub(super) struct Device<B: PageDevice> {
    /// The selector the device is opened with, as given; the automatically
    /// ranked device when `None`.
    selector: Option<String>,
    /// The device once its opening has finished: `None` when it failed.
    slot: Arc<OnceLock<Option<Arc<B>>>>,
    /// The opening of the device.
    opener: Mutex<Opener>,
    /// The pages holding the device: recording for it, waiting for it or
    /// rendering on it.  A page claims its device before recording, since
    /// what it records depends on the device
    /// ([`PageDevice::page_paths`]).
    in_flight: AtomicUsize,
    /// Set once the device has failed; it is offered no page afterwards.
    retired: AtomicBool,
    /// Pages the device rendered.
    pub(super) rendered: AtomicU32,
}

/// The opening of a GPU page device.
enum Opener {
    /// Not started.
    Deferred,
    /// The thread opening the device; joined when the session drops, so
    /// the process never exits while a driver is initialising.
    Running(JoinHandle<()>),
    /// Nothing to open or to join: the device came with the session, or
    /// its opening could not start.
    Settled,
}

/// The wake-up of pages waiting for a session's devices to open.
#[derive(Default)]
pub(super) struct Openings {
    /// Held while a device's outcome is set; `settled` is notified after.
    lock: Mutex<()>,
    settled: Condvar,
}

impl Openings {
    /// Set `slot` to `outcome` (only its first outcome is kept) and wake
    /// the pages waiting.
    fn settle<B>(&self, slot: &OnceLock<Option<Arc<B>>>, outcome: Option<Arc<B>>) {
        let guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        drop(slot.set(outcome));
        drop(guard);
        self.settled.notify_all();
    }

    /// The lock under which a waiter checks the devices before
    /// [`Self::wait`]: no outcome is set while it is held.
    pub(super) fn lock(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Release `guard` until an outcome is set or `wait` has passed.
    pub(super) fn wait<'a>(&self, guard: MutexGuard<'a, ()>, wait: Duration) -> MutexGuard<'a, ()> {
        self.settled
            .wait_timeout(guard, wait)
            .unwrap_or_else(PoisonError::into_inner)
            .0
    }
}

/// What a device can do for a page now.
pub(super) enum State<'a, B> {
    /// Open and in service, with its load.
    Open(&'a B, Load),
    /// Still opening, or not yet asked to.
    Opening,
    /// Did not open.
    NoDevice,
    /// Retired after a failure.
    Retired,
}

impl<B: PageDevice> Device<B> {
    /// A device opened when a page first asks for one, with `selector`.
    pub(super) fn deferred(selector: Option<String>) -> Self {
        Self::new(selector, OnceLock::new(), Opener::Deferred)
    }

    /// The open device `backend`, opened with `selector`.
    pub(super) fn open(selector: Option<String>, backend: Arc<B>) -> Self {
        Self::new(selector, OnceLock::from(Some(backend)), Opener::Settled)
    }

    /// A device whose opening has started elsewhere and never settles.
    #[cfg(test)]
    pub(super) fn never_opening() -> Self {
        Self::new(None, OnceLock::new(), Opener::Settled)
    }

    fn new(selector: Option<String>, slot: OnceLock<Option<Arc<B>>>, opener: Opener) -> Self {
        Self {
            selector,
            slot: Arc::new(slot),
            opener: Mutex::new(opener),
            in_flight: AtomicUsize::new(0),
            retired: AtomicBool::new(false),
            rendered: AtomicU32::new(0),
        }
    }

    /// What the device can do for a page now.
    pub(super) fn state(&self) -> State<'_, B> {
        if self.retired.load(Ordering::Acquire) {
            return State::Retired;
        }
        match self.slot.get() {
            None => State::Opening,
            Some(None) => State::NoDevice,
            Some(Some(backend)) => State::Open(
                backend,
                Load::of(self.in_flight.load(Ordering::Relaxed), &**backend),
            ),
        }
    }

    /// The device, for a page, which holds it until the claim drops.
    pub(super) fn claim<'a>(&'a self, backend: &'a B) -> Claim<'a, B> {
        let _ = self.in_flight.fetch_add(1, Ordering::Relaxed);
        Claim {
            device: self,
            backend,
        }
    }

    /// The device's name once it is open, else the selector it is opened
    /// with and whether its opening has finished.
    pub(super) fn name(&self) -> String {
        let selector = || {
            self.selector.as_ref().map_or_else(
                || "the automatically ranked device".to_owned(),
                |s| format!("device {s:?}"),
            )
        };
        match self.slot.get() {
            Some(Some(backend)) => backend.device_name(),
            Some(None) => format!("{} (did not open)", selector()),
            None => format!("{} (not open)", selector()),
        }
    }

    /// Account for the device failing on page `page_num` with `error`: a
    /// page beyond the device's limits, or beyond the memory it has left,
    /// is logged at `warn` and leaves the device in service; any other
    /// error retires the device, logged at `warn` once.  Whether `error`
    /// retires the device.
    #[must_use]
    pub(super) fn fail(&self, page_num: u32, error: &BackendError) -> bool {
        let retires = B::retires_device(error);
        if !retires {
            log::warn!(
                "page {page_num}: too large for the GPU page device {} ({error}); the page \
                 renders on the CPU and the device stays in service",
                self.name()
            );
        } else if !self.retired.swap(true, Ordering::AcqRel) {
            log::warn!(
                "page {page_num}: the GPU page raster failed on {} ({error}); the device is \
                 retired, this page renders on the CPU, and later pages on the session's \
                 other GPU page devices, or on the CPU when none is in service",
                self.name()
            );
        }
        retires
    }

    /// Join the thread opening the device, if it runs; one still opening
    /// the device is logged at `warn` first, since the join waits for the
    /// driver without a bound.
    pub(super) fn join_opener(&mut self) {
        let opener = std::mem::replace(
            self.opener
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner),
            Opener::Settled,
        );
        let Opener::Running(opener) = opener else {
            return;
        };
        if self.slot.get().is_none() {
            log::warn!(
                "rasterrocket: waiting for the GPU page device {} to finish opening before \
                 the session closes",
                self.name()
            );
        }
        if opener.join().is_err() {
            log::warn!("rasterrocket: the thread opening a GPU page device panicked");
        }
    }

    /// Whether the device's opening has not started.
    #[cfg(all(test, feature = "vulkan"))]
    pub(super) fn deferred_still(&self) -> bool {
        matches!(
            *self.opener.lock().unwrap_or_else(PoisonError::into_inner),
            Opener::Deferred
        )
    }

    /// Whether a thread opens, or opened, the device.
    #[cfg(all(test, feature = "vulkan"))]
    pub(super) fn opening_started(&self) -> bool {
        matches!(
            *self.opener.lock().unwrap_or_else(PoisonError::into_inner),
            Opener::Running(_)
        )
    }

    /// Whether the device has opened.
    #[cfg(all(test, feature = "vulkan"))]
    pub(super) fn is_open(&self) -> bool {
        matches!(self.slot.get(), Some(Some(_)))
    }

    /// Whether the device's opening has finished.
    #[cfg(all(test, feature = "vulkan"))]
    pub(super) fn settled(&self) -> bool {
        self.slot.get().is_some()
    }

    /// Whether a failure retired the device.
    #[cfg(test)]
    pub(super) fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// The pages holding the device.
    #[cfg(test)]
    pub(super) fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }
}

impl<B: SessionDevice> Device<B> {
    /// Start opening the device on a new thread unless it has been
    /// started; `openings` is woken once it settles.  A device that does
    /// not open, or a thread that does not start, is logged at `error`.
    pub(super) fn start_opening(&self, openings: &Arc<Openings>) {
        let mut opener = self.opener.lock().unwrap_or_else(PoisonError::into_inner);
        if !matches!(*opener, Opener::Deferred) {
            return;
        }
        let selector = self.selector.clone();
        let slot = Arc::clone(&self.slot);
        let wake = Arc::clone(openings);
        let spawned = std::thread::Builder::new()
            .name("rrocket-gpu-pages".to_owned())
            .spawn(move || {
                settle_opening(&wake, &slot, || open_device::<B>(selector.as_deref()));
            });
        *opener = match spawned {
            Ok(thread) => Opener::Running(thread),
            Err(e) => {
                log::error!(
                    "rasterrocket: GPU page rendering was requested, but the thread that opens \
                     the {} device {} did not start ({e}); no page renders on it",
                    B::API,
                    self.name()
                );
                openings.settle(&self.slot, None);
                Opener::Settled
            }
        };
    }
}

/// Settle `slot` with the device `open` returns.  A panic in `open` settles
/// it with no device as the stack unwinds, so the pages waiting see the
/// device as failed rather than as opening until their wait runs out.
fn settle_opening<B>(
    openings: &Openings,
    slot: &OnceLock<Option<Arc<B>>>,
    open: impl FnOnce() -> Option<Arc<B>>,
) {
    /// Settles its slot with no device when dropped; a slot already
    /// settled keeps its outcome.
    struct Unsettled<'a, B> {
        openings: &'a Openings,
        slot: &'a OnceLock<Option<Arc<B>>>,
    }
    impl<B> Drop for Unsettled<'_, B> {
        fn drop(&mut self) {
            if self.slot.get().is_none() {
                log::error!(
                    "rasterrocket: GPU page rendering was requested, but opening its device \
                     panicked; no page renders on it"
                );
                self.openings.settle(self.slot, None);
            }
        }
    }
    let guard = Unsettled { openings, slot };
    guard.openings.settle(guard.slot, open());
}

/// The load of an open device: the pages holding it and its mean time
/// per page raster ([`PageDevice::page_mean`]), unknown before one
/// after its first has run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Load {
    pub(super) held: usize,
    pub(super) mean_nanos: Option<u128>,
}

impl Load {
    fn of<B: PageDevice>(held: usize, backend: &B) -> Self {
        Self {
            held,
            mean_nanos: backend.page_mean().map(|mean| mean.as_nanos()),
        }
    }

    /// The expected wait of one more page, in nanoseconds: the held pages
    /// and that page times the mean page raster, `fallback` (at least 1)
    /// when the mean is unknown.
    pub(super) fn wait(self, fallback: u128) -> u128 {
        let mean = self.mean_nanos.unwrap_or(fallback).max(1);
        u128::try_from(self.held)
            .unwrap_or(u128::MAX)
            .saturating_add(1)
            .saturating_mul(mean)
    }
}

/// A device, held by one page until dropped.
pub(crate) struct Claim<'a, B: PageDevice> {
    pub(super) device: &'a Device<B>,
    /// The device.
    pub(crate) backend: &'a B,
}

impl<B: PageDevice> Drop for Claim<'_, B> {
    fn drop(&mut self) {
        let _ = self.device.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The GPU-page device of `selector`, or `None`, logged at `error`, when
/// it does not open.
pub(crate) fn open_device<B: SessionDevice>(selector: Option<&str>) -> Option<Arc<B>> {
    let started = std::time::Instant::now();
    crate::device_init::open_backend::<B>(selector)
        .inspect(|backend| {
            log::debug!(
                "rasterrocket: GPU page device {} open after {:.0} ms",
                backend.device_name(),
                started.elapsed().as_secs_f64() * 1e3
            );
        })
        .inspect_err(|e| {
            log::error!(
                "rasterrocket: GPU page rendering was requested, but the {} \
                 device {} did not open ({e}); no page renders on it",
                B::API,
                selector.map_or_else(|| "picked automatically".to_owned(), |s| format!("{s:?}"))
            );
        })
        .ok()
}

#[cfg(test)]
mod opening_tests {
    use std::sync::{Arc, OnceLock};

    use super::{Openings, settle_opening};

    /// An opening that panics settles its device as failed, so no page
    /// waits on it as still opening.
    #[test]
    fn a_panicking_opening_settles_without_a_device() {
        let openings = Arc::new(Openings::default());
        let slot: Arc<OnceLock<Option<Arc<u8>>>> = Arc::new(OnceLock::new());
        let opener = {
            let (openings, slot) = (Arc::clone(&openings), Arc::clone(&slot));
            std::thread::spawn(move || {
                settle_opening(&openings, &slot, || panic!("the driver panicked"));
            })
        };
        assert!(opener.join().is_err(), "the opening panics");
        assert_eq!(slot.get(), Some(&None));
    }

    /// An opening that returns keeps its outcome.
    #[test]
    fn an_opening_keeps_its_device() {
        let openings = Openings::default();
        let slot = OnceLock::new();
        settle_opening(&openings, &slot, || Some(Arc::new(7_u8)));
        assert_eq!(slot.get(), Some(&Some(Arc::new(7))));
    }
}
