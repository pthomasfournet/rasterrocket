mod args;
mod cleanup;
mod comic;
mod decrypt_gate;
mod diagnostics;
mod naming;
mod ram;
mod render;

use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use args::Args;
use clap::Parser;
use mimalloc::MiMalloc;
use rayon::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Print `rrocket: {message}` to stderr and yield the failure exit status.
fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("rrocket: {message}");
    ExitCode::FAILURE
}

/// Every exit path returns through here rather than `std::process::exit`,
/// so the RAM output guard and the thread pool are dropped before the
/// process ends.
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

    if args.odd_only && args.even_only {
        return fail("--odd and --even are mutually exclusive");
    }
    if let Err(e) = args.validate_format_flags() {
        return fail(e);
    }

    // When --ram is set, redirect output_prefix into a fresh tmpfs directory.
    // The guard removes the directory on Drop so a clean exit cleans up; on
    // crash the dir is left behind but /dev/shm is tmpfs and clears on reboot.
    // The SpillPolicy is consulted per-page: when free memory tightens, the
    // writer falls back to the original on-disk prefix automatically.
    let (_ram_guard, spill_policy) = match ram::redirect_to_ram(&mut args) {
        Ok(redirect) => redirect,
        Err(e) => return fail(format_args!("--ram setup failed: {e}")),
    };

    // Dispatch on the input extension. Comic archives render via the comic crate
    // and share the PDF path's output encoder; everything else (incl. `.pdf` and
    // unknown extensions) falls through to the native PDF interpreter below. The
    // `--ram` redirect above already applies, so comic output honours it too;
    // the PDF-only session/decrypt code below is skipped for comics.
    let ext = std::path::Path::new(&args.input)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "cbz" | "cb7" | "cbt" | "cbr") {
        return match comic::run(&args, &spill_policy) {
            Ok((0, _)) => fail("no pages rendered"),
            // Per-page errors were already printed to stderr inside `run`; a
            // non-zero exit makes the partial failure observable via `$?`,
            // matching the PDF path's `report_errors`.
            Ok((_, failed)) if failed > 0 => ExitCode::FAILURE,
            Ok((_, _)) => ExitCode::SUCCESS,
            Err(e) => fail(e),
        };
    }

    let mut session_config = match args.session_config() {
        Ok(config) => config,
        Err(e) => return fail(e),
    };

    // Encrypted-document liability gate (production/CLI path only). Probe
    // the input cheaply; only an encrypted document triggers the gate, so
    // unencrypted documents pay zero cost and see no prompt. `-` (stdin)
    // cannot be probed and is treated as unencrypted — an encrypted stdin
    // stream surfaces the clear EncryptedDocument error from open_session
    // below, never a silent strip.
    if args.input != "-" && pdf_raster::is_encrypted(std::path::Path::new(&args.input)) {
        // Never silently strip in production: require an explicit decision.
        let authorized = decrypt_gate::prompt_decrypt(args.decrypt_owned);
        if !authorized {
            // Abort with the accurate, actionable error — not the
            // misleading "document has no pages", not a silent strip.
            return fail(pdf_raster::decrypt_gate_declined_message());
        }
        session_config.decrypt_authorized = true;
    }
    // The password is an input to decryption, not an authorisation: it is
    // passed whether or not the gate ran, and does nothing without it.
    session_config.password = args.password.as_deref().map(pdf_raster::Password::from);

    let pool = match rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .thread_name(|i| format!("raster-worker-{i}"))
        // 8 MiB per worker: path flattening and Bézier subdivision recurse deeply
        // on degenerate inputs and overflow the 2 MiB rayon default.
        .stack_size(8 * 1024 * 1024)
        .build()
    {
        Ok(pool) => pool,
        Err(e) => return fail(format_args!("cannot start the render thread pool: {e}")),
    };

    let session = match pdf_raster::open_session(std::path::Path::new(&args.input), &session_config)
    {
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
        eprintln!("rrocket: warning: {w}");
    }

    let n_pages = pages.len();
    let done = AtomicU32::new(0);
    let start = Instant::now();

    #[expect(
        clippy::cast_sign_loss,
        reason = "total ≥ 1 guaranteed by the n == 0 exit above; i32::try_from only guards the upper bound"
    )]
    let total_u32 = total as u32;

    let timings = args.timings;

    let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
        pages
            .par_iter()
            .filter_map(|&page_num| {
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "page_num ≥ 1, enforced by build_page_list"
                )]
                let page_u32 = page_num as u32;

                let t0 = timings.then(Instant::now);
                let result =
                    render::render_page(&session, page_u32, total_u32, &args, &spill_policy);
                if let Some(t0) = t0 {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let tid = rayon::current_thread_index().unwrap_or(99);
                    eprintln!("timing: page {page_num:4}  {ms:7.1}ms  thread {tid}");
                }

                if args.progress {
                    let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let elapsed = start.elapsed().as_secs_f64();
                    let completed_usize = usize::try_from(completed).unwrap_or(n_pages);
                    let remaining = n_pages.saturating_sub(completed_usize);
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "ETA display; ±1s accuracy is sufficient"
                    )]
                    let eta_str = if elapsed > 0.5 && completed >= 2 {
                        let rate = f64::from(completed) / elapsed;
                        let eta_s = remaining as f64 / rate;
                        if eta_s.is_finite() {
                            format!("~{eta_s:.1}s remaining")
                        } else {
                            "~?s remaining".to_owned()
                        }
                    } else {
                        "~?s remaining".to_owned()
                    };
                    eprintln!(
                        "rrocket: page {page_num} done  [{completed}/{n_pages}]  \
                         {elapsed:.1}s elapsed  {eta_str}"
                    );
                }

                result.err().map(|e| (page_num, e))
            })
            .collect()
    });

    // Eagerly drop GPU decoders on every worker thread while the CUDA driver is
    // still fully live, before the pool drops.  Avoids the process-exit teardown
    // race where all workers call nvjpegJpegStateDestroy concurrently into a
    // driver that has already started its own atexit shutdown sequence.
    let _ = pool.broadcast(|_| pdf_raster::release_gpu_decoders());

    diagnostics::report_errors(errors)
}
