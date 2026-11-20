/// `eprintln!` whose write may fail: a closed or full standard error ends
/// no run, since every line written there is a diagnostic.
macro_rules! note {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

mod args;
mod bands;
mod cleanup;
mod comic;
mod decrypt_gate;
mod deliver;
mod diagnostics;
mod fax;
mod interrupt;
mod naming;
mod ocr_prep;
mod pool;
mod print;
mod progress;
mod ram;
mod render;
mod tone;

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use args::Args;
use clap::Parser;
use mimalloc::MiMalloc;
use rayon::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Print `rrocket: {message}` to stderr and yield the failure exit status.
fn fail(message: impl std::fmt::Display) -> ExitCode {
    note!("rrocket: {message}");
    ExitCode::FAILURE
}

/// The document on standard input, read to its end.
///
/// # Errors
///
/// When standard input is a terminal, cannot be read, or is empty.
fn read_stdin_pdf() -> Result<Vec<u8>, String> {
    use std::io::{IsTerminal, Read};
    let mut stdin = std::io::stdin();
    if stdin.is_terminal() {
        return Err("INPUT \"-\" reads standard input, which is a terminal; \
                    pipe a PDF in (rrocket - out < doc.pdf) or name the file"
            .to_owned());
    }
    let mut bytes = Vec::new();
    let len = stdin
        .read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read standard input: {e}"))?;
    if len == 0 {
        return Err("standard input is empty; INPUT \"-\" expects a PDF on it".to_owned());
    }
    Ok(bytes)
}

/// Apply the RAM output policy (see `ram`): the guard removes the RAM
/// directory when it drops, and the spill policy chooses each page's
/// directory.  Interrupts are caught first (see `interrupt`), so an
/// interrupt from the creation of the directory on stops the run through
/// `main`'s return, which drops the guard.
///
/// # Errors
///
/// The failure exit status, the error reported.
fn begin_output(args: &mut Args) -> Result<(ram::RamDirGuard, ram::SpillPolicy), ExitCode> {
    interrupt::catch();
    ram::redirect_to_ram(args).map_err(fail)
}

/// Every exit path returns through here rather than `std::process::exit`,
/// so the RAM output guard and the thread pool are dropped before the
/// process ends; an interrupt (see `interrupt`) also ends the run here.
#[expect(
    clippy::too_many_lines,
    reason = "linear startup sequence: arg parse → session open → page list → parallel render"
)]
fn main() -> ExitCode {
    // Default to `warn` so misconfiguration (e.g. PDF_RASTER_BACKEND=cdua,
    // silent GPU init fallback) is visible without RUST_LOG opt-in.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .try_init();

    let mut args = Args::parse();
    if args.list_printers {
        return match print::list_printers() {
            Ok(list) if list.is_empty() => {
                note!("rrocket: {}", print::NO_PRINTERS);
                ExitCode::SUCCESS
            }
            Ok(list) => match print::write_printer_list(&mut std::io::stdout().lock(), &list) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => fail(format_args!("cannot write the printer list: {e}")),
            },
            Err(e) => fail(e),
        };
    }

    if args.odd_only && args.even_only {
        return fail("--odd and --even are mutually exclusive");
    }
    if let Err(e) = args.validate_format_flags() {
        return fail(e);
    }
    args.apply_fax_preset();
    args.apply_print_preset();

    let mut session_config = match args.session_config() {
        Ok(config) => config,
        Err(e) => return fail(e),
    };
    if let Some(profile) = &session_config.output_profile {
        match encode::IccProfile::new(std::sync::Arc::clone(profile.bytes())) {
            Ok(icc) => args.output_icc = Some(icc),
            Err(e) => return fail(format_args!("--output-profile cannot be embedded: {e}")),
        }
    }
    if let Some(warning) = args.untagged_output_warning() {
        note!("rrocket: warning: {warning}");
    }

    // The password is an input to decryption, not an authorisation: it is
    // passed whether or not the gate ran, and does nothing without it.
    session_config.password = args.password.as_deref().map(pdf_raster::Password::from);

    // Dispatch on the input extension. Comic archives render via the comic crate
    // and share the PDF path's output encoder; everything else (incl. `.pdf` and
    // unknown extensions) falls through to the native PDF interpreter below.
    // Comic output takes the same RAM output policy as PDF output; the
    // decryption gate below is skipped for comics, whose embedded PDF opens
    // with the same session settings otherwise.  The output of either begins
    // after any prompt, so an interrupt before it ends the process at once,
    // with nothing to remove.
    if args.is_comic_input() {
        let (_ram_guard, spill_policy) = match begin_output(&mut args) {
            Ok(output) => output,
            Err(status) => return status,
        };
        return match comic::run(&args, &spill_policy, session_config) {
            Ok((0, _)) => fail("no pages rendered"),
            // Per-page errors were already printed to stderr inside `run`; a
            // non-zero exit makes the partial failure observable via `$?`,
            // matching the PDF path's `report_errors`.
            Ok((_, failed)) if failed > 0 => ExitCode::FAILURE,
            Ok((_, _)) => ExitCode::SUCCESS,
            Err(e) => fail(e),
        };
    }

    // `-` names standard input, read to its end and parsed from memory.
    let stdin_pdf = if args.input() == "-" {
        match read_stdin_pdf() {
            Ok(bytes) => Some(bytes),
            Err(e) => return fail(e),
        }
    } else {
        None
    };

    // Encrypted-document liability gate (production/CLI path only). Probe
    // the input cheaply; only an encrypted document triggers the gate, so
    // unencrypted documents pay zero cost and see no prompt.  A document on
    // standard input leaves no terminal to prompt on, so only
    // `--decrypt-owned` or its environment variable authorise it.
    let encrypted = stdin_pdf.as_deref().map_or_else(
        || pdf_raster::is_encrypted(std::path::Path::new(args.input())),
        pdf_raster::is_encrypted_bytes,
    );
    if encrypted {
        // Never silently strip in production: require an explicit decision.
        let authorized = decrypt_gate::prompt_decrypt(args.decrypt_owned);
        if !authorized {
            // Abort with the accurate, actionable error — not the
            // misleading "document has no pages", not a silent strip.
            return fail(pdf_raster::decrypt_gate_declined_message());
        }
        session_config.decrypt_authorized = true;
    }

    let (_ram_guard, spill_policy) = match begin_output(&mut args) {
        Ok(output) => output,
        Err(status) => return status,
    };

    let pool = match pool::RenderPool::new(args.num_threads) {
        Ok(pool) => pool,
        Err(e) => return fail(format_args!("cannot start the render thread pool: {e}")),
    };

    let opened = stdin_pdf.map_or_else(
        || pdf_raster::open_session(std::path::Path::new(args.input()), &session_config),
        |bytes| pdf_raster::open_session_from_bytes(bytes, &session_config),
    );
    let session = match opened {
        Ok(session) => session,
        Err(e) => {
            diagnostics::report_open_error(&e, &args, session_config.policy);
            return ExitCode::FAILURE;
        }
    };

    let n = session.total_pages();
    if n == 0 {
        return fail("document has no pages");
    }
    let Ok(total) = i32::try_from(n) else {
        return fail(format_args!("document has too many pages ({n} > i32::MAX)"));
    };

    let (pages, page_warnings) = match args.build_page_list(total) {
        Ok(selection) => selection,
        Err(e) => return fail(e),
    };
    for w in &page_warnings {
        note!("rrocket: warning: {w}");
    }

    let n_pages = pages.len();
    let log = progress::PageLog::new(args.timings, args.progress, n_pages);

    #[expect(
        clippy::cast_sign_loss,
        reason = "total ≥ 1 guaranteed by the n == 0 exit above; i32::try_from only guards the upper bound"
    )]
    let total_u32 = total as u32;

    if args.fax && u16::try_from(n_pages).is_err() {
        return fail(format_args!(
            "a fax document holds at most 65535 pages; {n_pages} are selected"
        ));
    }
    let fax = args
        .fax
        .then(|| fax::FaxDocument::new(args.tiff_compression(), args.dpi()));
    let mut spool = if args.print {
        match print::begin(&mut args) {
            Ok(spool) => Some(spool),
            Err(e) => return fail(format_args!("nothing printed: {e}")),
        }
    } else {
        None
    };
    let print_pages = print::PageSink::new();
    let document = match (&fax, &spool) {
        (Some(fax), _) => render::Document::Fax(fax),
        (_, Some(_)) => render::Document::Print(&print_pages),
        (None, None) => render::Document::None,
    };
    // A fax document or print job is made whole or not at all: once a
    // page has failed, the pages not yet started are skipped.
    let whole_document = !matches!(document, render::Document::None);
    let document_failed = AtomicBool::new(false);

    let render_one = |&page_num: &i32| {
        #[expect(
            clippy::cast_sign_loss,
            reason = "page_num ≥ 1, enforced by build_page_list"
        )]
        let page_u32 = page_num as u32;
        if document_failed.load(Ordering::Relaxed) || interrupt::interrupted() {
            return None;
        }

        let result = log.page(page_num, || {
            render::render_page(
                &session,
                page_u32,
                total_u32,
                &args,
                &spill_policy,
                document,
            )
        });
        if whole_document && result.is_err() {
            document_failed.store(true, Ordering::Relaxed);
        }
        result.err().map(|e| (page_num, e))
    };

    // Printed pages are appended to the spool in page order as they finish
    // (see `print`); other pages render in one pass.
    let (errors, spool_error) = spool.as_mut().map_or_else(
        || {
            let errors: Vec<_> = pool.install(|| pages.par_iter().filter_map(render_one).collect());
            (errors, None)
        },
        |spool| spool.render_job(&print_pages, &pages, &pool, render_one),
    );

    // The workers' GPU state is destroyed while the session, declared after
    // the pool and so dropped before it, still holds the device context.
    drop(pool);

    if let Some(spool) = spool {
        // Page errors are reported whether or not the spool failed.
        let pages_failed = !errors.is_empty();
        let status = diagnostics::report_errors(errors, &args);
        if interrupt::interrupted() {
            return fail("nothing printed: interrupted; the print job is aborted");
        }
        if let Some(e) = spool_error {
            return fail(format_args!(
                "nothing printed: the print job was not written: {e}"
            ));
        }
        if pages_failed {
            note!("rrocket: nothing printed: pages failed to render");
            return status;
        }
        if let Err(e) = print::complete(spool, &args) {
            return fail(format_args!("nothing printed: {e}"));
        }
        return ExitCode::SUCCESS;
    }
    if interrupt::interrupted() {
        let _ = diagnostics::report_errors(errors, &args);
        return fail("interrupted: the pages not yet begun were not rendered");
    }
    if let Some(document) = fax {
        // A fax missing pages would read as complete: none is written.
        if !errors.is_empty() {
            let status = diagnostics::report_errors(errors, &args);
            note!("rrocket: no fax document written: pages failed to render");
            return status;
        }
        let path = format!("{}.tif", spill_policy.next_prefix());
        if let Err(e) = document.write(&path) {
            return fail(format_args!("fax document not written: {e}"));
        }
        if let Some(number) = &args.fax_to
            && let Err(e) = fax::submit(&path, number, args.fax_queue.as_deref())
        {
            return fail(format_args!("fax to {number} not sent: {e}"));
        }
    }
    diagnostics::report_errors(errors, &args)
}
