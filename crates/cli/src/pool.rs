//! The page-render thread pool, and the driver that runs a sequence of jobs
//! on it while handing their outcomes back in sequence order.
//!
//! [`run_in_order`] serves every caller that consumes pages in order (comic
//! output, the print spool): a job is begun only within a bounded window
//! past the first outcome not yet reported, so the outcomes held at once,
//! and the pixel or encoded buffers they own, are bounded by the pool's
//! width rather than by the length of the document.

use std::collections::BTreeMap;
use std::ops::{ControlFlow, Deref};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// How many jobs per pool thread may be begun past the first job whose
/// outcome has not been reported.
pub(crate) const JOBS_AHEAD_PER_THREAD: usize = 2;

/// How often [`run_in_order_until`], waiting for an outcome, asks whether
/// the run is cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// The page-render thread pool.  Dropping it destroys every worker's GPU
/// thread state (see [`pdf_raster::release_gpu_thread_state`]); the pool is
/// therefore dropped while the session whose device context that state
/// belongs to is alive.  Its threads are not joined on drop, so their
/// thread-local destructors would otherwise reach the driver concurrently
/// with the process's exit handlers.
pub(crate) struct RenderPool(rayon::ThreadPool);

impl RenderPool {
    /// A pool of `num_threads` workers (0: one per CPU), each with 8 MiB of
    /// stack: path flattening and Bézier subdivision recurse deeply on
    /// degenerate inputs and overflow the 2 MiB default.
    ///
    /// # Errors
    ///
    /// The pool's threads could not be started.
    pub(crate) fn new(num_threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("raster-worker-{i}"))
            .stack_size(8 * 1024 * 1024)
            .build()
            .map(Self)
    }
}

impl Deref for RenderPool {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &rayon::ThreadPool {
        &self.0
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        let _: Vec<()> = self.0.broadcast(|_| pdf_raster::release_gpu_thread_state());
    }
}

/// The outcome a job's slot carries back to the driver.
enum Outcome<T> {
    /// The job ran to completion.
    Done(T),
    /// The job was not begun: the run had stopped before it reached a
    /// worker.
    Skipped,
    /// The job's `work` unwound.
    Unwound,
}

/// Sends its job's outcome to the driver when dropped, so that a `work`
/// that unwinds still reports, as [`Outcome::Unwound`].
struct Slot<T> {
    index: usize,
    outcome: Outcome<T>,
    tx: mpsc::Sender<(usize, Outcome<T>)>,
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        let outcome = std::mem::replace(&mut self.outcome, Outcome::Skipped);
        // The receiver is gone only once the driver has stopped on an
        // unwound job, when no further outcome is wanted.
        let _ = self.tx.send((self.index, outcome));
    }
}

/// Run `work` on each of `jobs` on `pool`, and hand each outcome with its
/// job's index to `report` on the calling thread, in job order.
///
/// A job is taken from `jobs` only while it lies fewer than
/// [`JOBS_AHEAD_PER_THREAD`] × threads jobs past the first one not yet
/// reported, which bounds the outcomes held at once.  Once `report`
/// returns [`ControlFlow::Break`], no further job is taken, jobs taken but
/// not yet begun are skipped, and jobs in progress finish; their outcomes,
/// with those finished but not reported, are returned in job order.  Once
/// a job's `work` unwinds, no further job is taken or reported, and the
/// pool's scope resumes the unwind when the jobs in progress have
/// finished.
pub(crate) fn run_in_order<J: Send, T: Send>(
    pool: &rayon::ThreadPool,
    jobs: impl IntoIterator<Item = J>,
    work: impl Fn(J) -> T + Sync,
    report: impl FnMut(usize, T) -> ControlFlow<()>,
) -> Vec<T> {
    run_in_order_until(pool, jobs, work, report, || false)
}

/// [`run_in_order`], stopped as by a [`ControlFlow::Break`] of `report`
/// once `cancelled` returns `true`.  `cancelled` is asked on the calling
/// thread before jobs are taken and at least every [`CANCEL_POLL`] while
/// the driver waits for an outcome, so a cancellation is seen while jobs
/// run; the caller tells a cancelled run from a finished one by its own
/// state.
pub(crate) fn run_in_order_until<J: Send, T: Send>(
    pool: &rayon::ThreadPool,
    jobs: impl IntoIterator<Item = J>,
    work: impl Fn(J) -> T + Sync,
    mut report: impl FnMut(usize, T) -> ControlFlow<()>,
    cancelled: impl Fn() -> bool,
) -> Vec<T> {
    let ahead = pool.current_num_threads().max(1) * JOBS_AHEAD_PER_THREAD;
    let mut jobs = jobs.into_iter();
    let stop = AtomicBool::new(false);
    let (work, stop) = (&work, &stop);
    let mut unreported = Vec::new();
    pool.in_place_scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let mut ready = BTreeMap::new();
        let (mut started, mut received, mut reported) = (0usize, 0usize, 0usize);
        let mut broken = false;
        loop {
            if !broken && cancelled() {
                broken = true;
                stop.store(true, Ordering::Relaxed);
            }
            while !broken && started < reported + ahead {
                let Some(job) = jobs.next() else { break };
                let tx = tx.clone();
                let index = started;
                scope.spawn(move |_| {
                    let mut slot = Slot {
                        index,
                        outcome: Outcome::Unwound,
                        tx,
                    };
                    slot.outcome = if stop.load(Ordering::Relaxed) {
                        Outcome::Skipped
                    } else {
                        Outcome::Done(work(job))
                    };
                });
                started += 1;
            }
            if received == started {
                unreported = std::mem::take(&mut ready).into_values().collect();
                return;
            }
            // Every spawned slot sends exactly once, and `tx` outlives the
            // loop, so the channel is not disconnected while
            // `received < started`.
            let (index, outcome) = match rx.recv_timeout(CANCEL_POLL) {
                Ok(received) => received,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            };
            received += 1;
            match outcome {
                Outcome::Done(outcome) => {
                    let _ = ready.insert(index, outcome);
                }
                Outcome::Skipped => {}
                Outcome::Unwound => {
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            }
            while !broken && let Some(outcome) = ready.remove(&reported) {
                if report(reported, outcome).is_break() {
                    broken = true;
                    stop.store(true, Ordering::Relaxed);
                }
                reported += 1;
            }
        }
    });
    unreported
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::{JOBS_AHEAD_PER_THREAD, run_in_order, run_in_order_until};

    const THREADS: usize = 2;

    fn pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(THREADS)
            .build()
            .unwrap()
    }

    /// The early jobs are the slowest, so later ones finish first and wait
    /// to be reported.
    fn slow_early(job: usize, n: usize) {
        std::thread::sleep(Duration::from_millis(u64::try_from(n - job).unwrap() * 2));
    }

    #[test]
    fn outcomes_are_reported_in_job_order_with_bounded_holding() {
        let n = 24;
        let (held, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut seen = Vec::new();
        let unreported = run_in_order(
            &pool(),
            0..n,
            |job| {
                let now = held.fetch_add(1, Ordering::SeqCst) + 1;
                let _ = peak.fetch_max(now, Ordering::SeqCst);
                slow_early(job, n);
                job * 10
            },
            |index, outcome| {
                let _ = held.fetch_sub(1, Ordering::SeqCst);
                seen.push((index, outcome));
                ControlFlow::Continue(())
            },
        );
        assert_eq!(unreported, Vec::<usize>::new());
        assert_eq!(seen, (0..n).map(|i| (i, i * 10)).collect::<Vec<_>>());
        let peak = peak.into_inner();
        assert!(
            peak <= THREADS * JOBS_AHEAD_PER_THREAD,
            "{peak} outcomes held at once"
        );
    }

    #[test]
    fn no_jobs_report_nothing() {
        let unreported = run_in_order(
            &pool(),
            std::iter::empty::<usize>(),
            |job| job,
            |_, _| panic!("no outcome to report"),
        );
        assert_eq!(unreported, Vec::<usize>::new());
    }

    /// After `report` breaks on the first outcome, at most one window of
    /// jobs has been taken, and the outcomes of those that ran come back.
    #[test]
    fn a_break_stops_taking_jobs_and_returns_the_rest() {
        let n = 100;
        let (taken, ran) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let jobs = (0..n).inspect(|_| {
            let _ = taken.fetch_add(1, Ordering::SeqCst);
        });
        let mut reported = Vec::new();
        let unreported = run_in_order(
            &pool(),
            jobs,
            |job| {
                let _ = ran.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(2));
                job
            },
            |index, outcome| {
                reported.push((index, outcome));
                ControlFlow::Break(())
            },
        );
        assert_eq!(reported, [(0, 0)]);
        let taken = taken.into_inner();
        assert!(
            taken <= THREADS * JOBS_AHEAD_PER_THREAD,
            "{taken} jobs taken"
        );
        assert!(unreported.is_sorted(), "{unreported:?}");
        assert!(unreported.iter().all(|&job| (1..taken).contains(&job)));
        assert_eq!(unreported.len() + 1, ran.into_inner());
    }

    /// A cancellation while job 2 runs: at most one window of jobs is
    /// taken, nothing is reported after it is seen, and the outcomes of the
    /// jobs that ran and were not reported come back in job order.
    #[test]
    fn a_cancellation_stops_taking_and_reporting_jobs() {
        use std::sync::atomic::AtomicBool;

        let n = 100;
        let (cancel, taken) = (AtomicBool::new(false), AtomicUsize::new(0));
        let jobs = (0..n).inspect(|_| {
            let _ = taken.fetch_add(1, Ordering::SeqCst);
        });
        let mut reported = Vec::new();
        let unreported = run_in_order_until(
            &pool(),
            jobs,
            |job| {
                if job == 2 {
                    cancel.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(250));
                job
            },
            |index, outcome| {
                reported.push((index, outcome));
                ControlFlow::Continue(())
            },
            || cancel.load(Ordering::SeqCst),
        );
        let taken = taken.into_inner();
        assert!(
            taken <= 2 * THREADS * JOBS_AHEAD_PER_THREAD,
            "{taken} jobs taken"
        );
        assert!(reported.len() < 3, "{reported:?}");
        assert!(unreported.is_sorted(), "{unreported:?}");
        assert!(unreported.contains(&2), "{unreported:?}");
    }

    #[test]
    fn an_unwinding_job_propagates_instead_of_hanging() {
        let n = 24;
        let run = std::panic::catch_unwind(|| {
            run_in_order(
                &pool(),
                0..n,
                |job| {
                    slow_early(job, n);
                    assert_ne!(job, 2, "job 2 panics");
                    job
                },
                |_, _| ControlFlow::Continue(()),
            )
        });
        assert!(run.is_err(), "the panic of job 2 reaches the caller");
    }
}
