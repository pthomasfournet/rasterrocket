//! The `warn` records a test thread logs, captured.

use std::cell::RefCell;
use std::sync::Once;

thread_local! {
    /// The records of this thread while a capture runs on it.
    static CAPTURED: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

/// The process's logger: it keeps the `warn` records of a thread whose
/// capture runs and drops every other record.
struct Capture;

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if record.level() != log::Level::Warn {
            return;
        }
        CAPTURED.with(|captured| {
            if let Some(records) = captured.borrow_mut().as_mut() {
                records.push(record.args().to_string());
            }
        });
    }

    fn flush(&self) {}
}

static LOGGER: Capture = Capture;

/// The `warn` records this thread logs while `run` runs.
///
/// # Panics
///
/// When another logger is installed in the process.
pub(super) fn warnings_of(run: impl FnOnce()) -> Vec<String> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&LOGGER).expect("no other logger in the test process");
        log::set_max_level(log::LevelFilter::Warn);
    });
    CAPTURED.with(|captured| *captured.borrow_mut() = Some(Vec::new()));
    run();
    CAPTURED
        .with(|captured| captured.borrow_mut().take())
        .unwrap_or_default()
}
