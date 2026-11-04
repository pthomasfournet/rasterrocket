//! Handing a document to a system service: the service's submission
//! program runs with its arguments and no shell, and its standard output
//! goes to standard error, keeping standard output for rrocket's own.

use std::process::{Command, Stdio};

/// Parse a CUPS destination name (`lp -d`), a queue optionally followed by
/// `/instance`: non-empty, without white space or control characters, and
/// not beginning with `-`, so no command reads it as an option.
pub(crate) fn parse_queue(s: &str) -> Result<String, String> {
    if s.is_empty() {
        return Err("a queue name is not empty".to_owned());
    }
    if s.starts_with('-') {
        return Err(format!("{s:?} is not a queue name: it begins with -"));
    }
    if let Some(c) = s.chars().find(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!("{s:?} is not a queue name: it holds {c:?}"));
    }
    Ok(s.to_owned())
}

/// The arguments of `lp`: the destination `queue` (else the default one),
/// then `options`, then the document `file`, or none when `lp` reads the
/// document from its standard input.
pub(crate) fn lp_args(
    queue: Option<&str>,
    options: Vec<String>,
    file: Option<&str>,
) -> Vec<String> {
    let mut argv = Vec::new();
    if let Some(queue) = queue {
        argv.extend(["-d".to_owned(), queue.to_owned()]);
    }
    argv.extend(options);
    if let Some(file) = file {
        argv.extend(["--".to_owned(), file.to_owned()]);
    }
    argv
}

/// Run `program` with `argv` and `stdin` to submit a document, `what`
/// naming the job in messages, and wait for the service to accept it.
/// `missing` tells the user what to do when `program` is not installed;
/// `kept` is where the document stays, when it has a path.
///
/// # Errors
///
/// The program could not be run or reported a failure; the message names
/// the kept document, if any.
pub(crate) fn submit(
    program: &str,
    argv: &[String],
    stdin: Stdio,
    what: &str,
    kept: Option<&str>,
    missing: &str,
) -> Result<(), String> {
    let kept = kept.map_or_else(String::new, |path| format!("; the document is at {path}"));
    let status = Command::new(program)
        .args(argv)
        .stdin(stdin)
        .stdout(std::io::stderr())
        .status()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("{program} is not installed: {missing}{kept}")
            } else {
                format!("cannot run {program}: {e}{kept}")
            }
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} refused {what} ({status}){kept}"))
    }
}

/// Run `program` with `argv` in the C locale to read what a system service
/// reports, and return its standard output.  `missing` tells the user what
/// to do when `program` is not installed.
///
/// # Errors
///
/// The program could not be run, reported a failure, or wrote output that
/// is not UTF-8; the message carries what it wrote to standard error.
pub(crate) fn query(program: &str, argv: &[&str], missing: &str) -> Result<String, String> {
    let output = Command::new(program)
        .args(argv)
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("{program} is not installed: {missing}")
            } else {
                format!("cannot run {program}: {e}")
            }
        })?;
    if !output.status.success() {
        let said = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{program} {} failed ({}): {}",
            argv.join(" "),
            output.status,
            said.trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| {
        format!(
            "{program} {} wrote output that is not UTF-8",
            argv.join(" ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{lp_args, parse_queue};

    #[test]
    fn queue_names_cannot_pass_for_options() {
        for name in ["Office", "HP_LaserJet-4", "Office/duplex", "queue@host"] {
            assert_eq!(parse_queue(name).as_deref(), Ok(name));
        }
        for bad in ["", "-o", "Office Jet", "a\tb", "a\nb"] {
            assert!(parse_queue(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lp_takes_the_queue_then_options_then_the_file() {
        let options = vec!["-o".to_owned(), "x=1".to_owned()];
        assert_eq!(
            lp_args(Some("Q"), options.clone(), Some("f")),
            ["-d", "Q", "-o", "x=1", "--", "f"]
        );
        assert_eq!(lp_args(None, options, None), ["-o", "x=1"]);
    }
}
