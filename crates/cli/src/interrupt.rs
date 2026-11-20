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

use std::sync::atomic::{AtomicU8, Ordering};

/// The run is going on.
const RUNNING: u8 = 0;
/// The run is interrupted: no further page is begun.
const INTERRUPTED: u8 = 1;

/// The state of the run: [`RUNNING`] or [`INTERRUPTED`].
static STATE: AtomicU8 = AtomicU8::new(RUNNING);

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
}

/// Mark the run whose state is `state` interrupted, unless it is already
/// interrupted.
fn interrupt(state: &AtomicU8) -> Interrupt {
    match state.compare_exchange(RUNNING, INTERRUPTED, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Interrupt::First,
        Err(_) => Interrupt::Again,
    }
}

/// Install the interrupt handler of this platform.  A failure is logged:
/// an interrupt then ends the process at once.
pub(crate) fn catch() {
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
    /// handler, which ends the process at once.  On the other events the system ends the
    /// process when the handler returns, so it does not return: the
    /// process ends when `main` does or when the system's grace period
    /// runs out.
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
                Interrupt::Again => 0,
            },
            _ => loop {
                std::thread::park();
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU8;

    use super::{Interrupt, RUNNING, interrupt};

    #[test]
    fn the_first_interrupt_marks_the_run_and_later_ones_do_nothing() {
        let state = AtomicU8::new(RUNNING);
        assert_eq!(interrupt(&state), Interrupt::First);
        assert_eq!(interrupt(&state), Interrupt::Again);
    }
}
