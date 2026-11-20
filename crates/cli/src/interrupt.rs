//! Interruption of a run: Ctrl-C, and on Linux and macOS also `SIGTERM`
//! and `SIGHUP`; on Windows Ctrl-Break and the console's closing, the
//! user's logging off and the system's shutting down.
//!
//! The first interrupt only marks the run interrupted.  No further page is
//! begun, the pages in progress finish, and `main` returns through its
//! normal exit path, so the RAM output directory and an unfinished print
//! job are removed by their owners' destructors.  A second interrupt ends
//! the process at once: on Linux and macOS the handler is installed
//! one-shot (`SA_RESETHAND`), so the signal's default action applies.
//!
//! The run's state moves from running to interrupted, for good, or to
//! committed while a print job is handed to the spooler.  The job is
//! committed with `commit`: an interrupt that arrives before makes the
//! commit fail, and the job is abandoned; one that arrives while it is
//! committed cannot cancel it, and the user is told so.  No job is
//! submitted after an interrupt was acknowledged as aborting it.
//!
//! The state is armed by [`catch`]: until then no interrupt can be
//! acknowledged, [`interrupted`] is false and every commit succeeds, so a
//! process that never installs the handler (a test binary) shares no
//! interrupt state between its jobs.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// The run is going on.
const RUNNING: u8 = 0;
/// The run is interrupted: no further page is begun.
const INTERRUPTED: u8 = 1;
/// The print job is being handed to the spooler; an interrupt no longer
/// cancels it.
const COMMITTED: u8 = 2;

/// The state of the run: [`RUNNING`], [`INTERRUPTED`] or [`COMMITTED`].
static STATE: AtomicU8 = AtomicU8::new(RUNNING);

/// Whether [`catch`] armed [`STATE`].
static ARMED: AtomicBool = AtomicBool::new(false);

/// Whether the run is interrupted.
pub(crate) fn interrupted() -> bool {
    STATE.load(Ordering::Relaxed) == INTERRUPTED
}

/// What an interrupt did.
#[derive(Debug, PartialEq, Eq)]
enum Interrupt {
    /// Marked the running run interrupted.
    First,
    /// Nothing: the run was already interrupted.
    Again,
    /// Nothing: the print job was already committed.
    TooLate,
}

/// Mark the run whose state is `state` interrupted, unless it is already
/// interrupted or its print job committed.
fn interrupt(state: &AtomicU8) -> Interrupt {
    match state.compare_exchange(RUNNING, INTERRUPTED, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Interrupt::First,
        Err(COMMITTED) => Interrupt::TooLate,
        Err(_) => Interrupt::Again,
    }
}

/// A committed print job: while it lives, an interrupt does not mark the
/// run interrupted; dropped, it returns the run to running.
#[cfg(any(windows, test))]
pub(crate) struct Committed<'a> {
    /// The committed state; `None` when no state is armed.
    state: Option<&'a AtomicU8>,
}

#[cfg(any(windows, test))]
impl Drop for Committed<'_> {
    fn drop(&mut self) {
        if let Some(state) = self.state {
            let _ =
                state.compare_exchange(COMMITTED, RUNNING, Ordering::Relaxed, Ordering::Relaxed);
        }
    }
}

/// Commit the print job of the run whose state is `state`, when armed:
/// `None` when the run is interrupted.
#[cfg(any(windows, test))]
fn commit_state(state: Option<&AtomicU8>) -> Option<Committed<'_>> {
    let Some(state) = state else {
        return Some(Committed { state: None });
    };
    state
        .compare_exchange(RUNNING, COMMITTED, Ordering::Relaxed, Ordering::Relaxed)
        .ok()
        .map(|_| Committed { state: Some(state) })
}

/// Commit the print job, just before it is handed to the spooler, for as
/// long as the returned value lives: `None` when the run is interrupted.
#[cfg(windows)]
pub(crate) fn commit() -> Option<Committed<'static>> {
    commit_state(ARMED.load(Ordering::Relaxed).then_some(&STATE))
}

/// Install the interrupt handler of this platform.  A failure is logged:
/// an interrupt then ends the process at once.
pub(crate) fn catch() {
    ARMED.store(true, Ordering::Relaxed);
    #[cfg(unix)]
    unix::catch();
    #[cfg(windows)]
    windows::catch();
}

#[cfg(unix)]
mod unix {
    /// The signals that interrupt the run.
    const SIGNALS: [(libc::c_int, &str); 3] = [
        (libc::SIGINT, "SIGINT"),
        (libc::SIGTERM, "SIGTERM"),
        (libc::SIGHUP, "SIGHUP"),
    ];

    /// The signal handler: marks the run interrupted.  An atomic
    /// compare-and-exchange on a lock-free `AtomicU8` is all it does, which
    /// is async-signal-safe.
    extern "C" fn on_signal(_: libc::c_int) {
        let _ = super::interrupt(&super::STATE);
    }

    /// Install [`on_signal`], one-shot, for each of [`SIGNALS`] that the
    /// process does not ignore: a signal ignored on entry (`nohup`, a
    /// background job of a non-interactive shell) stays ignored.
    #[expect(
        unsafe_code,
        reason = "sigaction installs the handler of the termination signals"
    )]
    pub(super) fn catch() {
        for (signal, name) in SIGNALS {
            // SAFETY: an all-zero `sigaction` is a valid value of the C
            // structure (no handler, empty mask, no flags).
            let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
            // SAFETY: a null `act` only reads the current action into the
            // writable `old`.
            if unsafe { libc::sigaction(signal, std::ptr::null(), &raw mut old) } != 0
                || old.sa_sigaction == libc::SIG_IGN
            {
                continue;
            }
            // SAFETY: as above.
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
            action.sa_flags = libc::SA_RESETHAND | libc::SA_RESTART;
            // SAFETY: `sa_mask` is a writable signal set of `action`.
            let _ = unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
            // SAFETY: `action` is initialised; its handler is a function
            // that lives as long as the process and is async-signal-safe.
            if unsafe { libc::sigaction(signal, &raw const action, std::ptr::null_mut()) } != 0 {
                log::warn!(
                    "cannot catch {name}: {}; it ends the run at once",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

#[cfg(windows)]
mod windows {
    use std::io::{self, Write as _};

    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    use windows_sys::core::BOOL;

    use super::Interrupt;

    /// Install [`on_console_event`] as a console control handler of the
    /// process.
    #[expect(
        unsafe_code,
        reason = "SetConsoleCtrlHandler installs the console control handler"
    )]
    pub(super) fn catch() {
        // SAFETY: the handler is a function that lives as long as the
        // process and touches only an atomic and standard error.
        if unsafe { SetConsoleCtrlHandler(Some(on_console_event), 1) } == 0 {
            log::warn!(
                "cannot catch Ctrl-C: {}; it ends the run at once",
                io::Error::last_os_error()
            );
        }
    }

    /// The console control handler, which the system runs on a thread of
    /// its own.  The first Ctrl-C or Ctrl-Break, the console's closing, the
    /// user's logging off and the system's shutting down mark the run
    /// interrupted.  A later Ctrl-C or Ctrl-Break goes on to the default
    /// handler, which ends the process at once; so does a second one after
    /// the print job was committed, the first having told the user that
    /// the job is already with the spooler.  On the other events the
    /// system ends the process when the handler returns, so it does not
    /// return: the process ends when `main` does or when the system's
    /// grace period runs out.
    extern "system" fn on_console_event(event: u32) -> BOOL {
        let outcome = super::interrupt(&super::STATE);
        match event {
            CTRL_C_EVENT | CTRL_BREAK_EVENT => match outcome {
                Interrupt::First => {
                    let _ = writeln!(
                        io::stderr(),
                        "rrocket: interrupted: stopping after the pages in progress; \
                         interrupt again to quit at once"
                    );
                    1
                }
                Interrupt::TooLate => BOOL::from(told_too_late()),
                Interrupt::Again => 0,
            },
            _ => loop {
                std::thread::park();
            },
        }
    }

    /// Tell the user, once, that the print job is already with the
    /// spooler; whether this call told them.
    fn told_too_late() -> bool {
        use std::sync::atomic::{AtomicBool, Ordering};
        static TOLD: AtomicBool = AtomicBool::new(false);
        if TOLD.swap(true, Ordering::Relaxed) {
            return false;
        }
        let _ = writeln!(
            io::stderr(),
            "rrocket: interrupted: the print job is already with the spooler; \
             cancel it in the printer's queue"
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU8;

    use super::{Interrupt, RUNNING, commit_state, interrupt};

    #[test]
    fn the_first_interrupt_marks_the_run_and_later_ones_do_nothing() {
        let state = AtomicU8::new(RUNNING);
        assert_eq!(interrupt(&state), Interrupt::First);
        assert_eq!(interrupt(&state), Interrupt::Again);
        assert!(
            commit_state(Some(&state)).is_none(),
            "an interrupted job is not committed"
        );
    }

    #[test]
    fn a_committed_job_is_not_interrupted_and_the_next_one_commits() {
        let state = AtomicU8::new(RUNNING);
        let committed = commit_state(Some(&state)).expect("a running run commits");
        assert_eq!(interrupt(&state), Interrupt::TooLate);
        assert_eq!(interrupt(&state), Interrupt::TooLate);
        drop(committed);
        let next = commit_state(Some(&state)).expect("the next job commits");
        drop(next);
        assert_eq!(interrupt(&state), Interrupt::First);
    }

    #[test]
    fn an_unarmed_run_always_commits() {
        let first = commit_state(None);
        assert!(first.is_some());
        assert!(commit_state(None).is_some(), "jobs at once share no state");
    }
}
