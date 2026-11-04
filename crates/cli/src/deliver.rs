//! Handing a written document to a system service: the service's
//! submission program runs with its arguments and no shell, and its
//! standard output goes to standard error, keeping standard output for
//! rrocket's own.

use std::process::{Command, Stdio};

/// Run `program` with `argv` to submit the document at `path`, `what`
/// naming the job in messages, and wait for the service to accept it.
/// `missing` tells the user what to do when `program` is not installed.
///
/// # Errors
///
/// The program could not be run or reported a failure; the message names
/// the document, which stays at `path`.
pub(crate) fn submit(
    program: &str,
    argv: &[String],
    what: &str,
    path: &str,
    missing: &str,
) -> Result<(), String> {
    let status = Command::new(program)
        .args(argv)
        .stdin(Stdio::null())
        .stdout(std::io::stderr())
        .status()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("{program} is not installed: {missing}; the document is at {path}")
            } else {
                format!("cannot run {program}: {e}; the document is at {path}")
            }
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} refused {what} ({status}); the document is at {path}"
        ))
    }
}
