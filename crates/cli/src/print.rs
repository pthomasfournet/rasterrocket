//! The `--print` output: the rendered pages as one PDF, each page a single
//! image at the render resolution (see [`encode::PdfWriter`]), handed to a
//! CUPS queue with `lp`.  The queue's filters fit the images to the
//! printer: media and orientation, colour, resolution and two-sided
//! sheets; the pages are not rendered again.
//!
//! Each page is compressed on the worker that rendered it into a
//! [`PageSink`] and appended to the [`Spool`] once every page before it is
//! done.  Pages start in page order, at most a fixed number per thread past
//! the first page not yet appended, so at most that many encoded pages are
//! held in memory; a slow page delays the appending of the pages after it
//! and the start of those more than that number past it.  The
//! spool is an anonymous temporary file, private to the process and freed
//! however it ends; `lp` reads it as its standard input.

use std::fs::File;
use std::io::{self, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};

use encode::{PdfPage, PdfPageEncoder, PdfWriter, RowImage, RowLayout};

use crate::cleanup::{Page, SpreadHalf};
use crate::deliver;
use crate::render::{OUTPUT_BUFFER_BYTES, PageCollector, RenderError, lock};

/// The resolution of a printed page when no resolution flag is given, up
/// to [`LARGE_FORMAT_PX`].
pub(crate) const PRINT_DPI: f64 = 600.0;

/// The resolution of a printed page past [`LARGE_FORMAT_PX`] when no
/// resolution flag is given: the native input resolution of large-format
/// photo printers' drivers.
pub(crate) const LARGE_FORMAT_DPI: f64 = 360.0;

/// The pixels at [`PRINT_DPI`] past which a printed page takes
/// [`LARGE_FORMAT_DPI`]: 600 Mpx, about 1 667 square inches, larger than
/// A0 (1 550) and than 36 × 44 in.
pub(crate) const LARGE_FORMAT_PX: f64 = 600e6;

/// The resolution of a printed page of `width_pts × height_pts` when no
/// resolution flag is given.
pub(crate) fn default_dpi((width_pts, height_pts): (f64, f64)) -> f64 {
    let px_per_pt = PRINT_DPI / 72.0;
    if width_pts * px_per_pt * height_pts * px_per_pt > LARGE_FORMAT_PX {
        LARGE_FORMAT_DPI
    } else {
        PRINT_DPI
    }
}

/// Longest job name in bytes: IPP's `name(MAX)`.
const MAX_TITLE_BYTES: usize = 255;

/// `--duplex` argument value: which edge a two-sided sheet turns on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Duplex {
    /// The long edge, as a book.
    Long,
    /// The short edge, as a notepad.
    Short,
}

impl Duplex {
    /// The IPP `sides` keyword.
    const fn sides(self) -> &'static str {
        match self {
            Self::Long => "two-sided-long-edge",
            Self::Short => "two-sided-short-edge",
        }
    }
}

/// What `lp` is told about the job; an option not given leaves the
/// queue's default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Submission<'a> {
    /// The queue, else the default destination.
    pub printer: Option<&'a str>,
    pub copies: Option<u32>,
    pub duplex: Option<Duplex>,
    /// The job's name (see [`job_title`]).
    pub title: String,
}

/// The job name for the input `input`: its file name, `rrocket` for
/// standard input, with control characters replaced by spaces and cut to
/// [`MAX_TITLE_BYTES`] on a character boundary.
pub(crate) fn job_title(input: &str) -> String {
    let name = Path::new(input)
        .file_name()
        .filter(|_| input != "-")
        .map_or_else(|| "rrocket".into(), |name| name.to_string_lossy());
    let mut title: String = name
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let mut cut = title.len().min(MAX_TITLE_BYTES);
    while !title.is_char_boundary(cut) {
        cut -= 1;
    }
    title.truncate(cut);
    title
}

/// The options of `lp` for `job`, the document read from standard input.
fn lp_options(job: &Submission<'_>) -> Vec<String> {
    let mut options = Vec::new();
    if let Some(copies) = job.copies {
        options.extend(["-n".to_owned(), copies.to_string()]);
    }
    options.extend([
        "-t".to_owned(),
        job.title.clone(),
        "-o".to_owned(),
        "document-format=application/pdf".to_owned(),
    ]);
    if let Some(duplex) = job.duplex {
        options.extend(["-o".to_owned(), format!("sides={}", duplex.sides())]);
    }
    options
}

/// The position of a page in the document: its page number, then the left
/// half of a split spread before the right.
type PageKey = (i32, u8);

/// The encoded pages not yet appended to the spool, shared by the workers.
pub(crate) struct PageSink {
    pages: PageCollector<PageKey, PdfPage>,
}

impl PageSink {
    /// An empty sink.
    pub(crate) const fn new() -> Self {
        Self {
            pages: PageCollector::new(),
        }
    }

    /// Compress `page`, rendered at `dpi`, and hold it as document page
    /// `page_num`, half `half` of a split spread: colour, or gray for a
    /// luma page.
    ///
    /// # Errors
    ///
    /// The encoder refused the page.
    pub(crate) fn add(
        &self,
        (page_num, half): (i32, Option<SpreadHalf>),
        page: &Page,
        dpi: (f64, f64),
    ) -> Result<(), RenderError> {
        let coded = match page {
            Page::Rgb(rgb) => PdfPage::encode(RowImage::Rgb(rgb), dpi),
            Page::Gray(luma) | Page::Mono(luma) => PdfPage::encode(RowImage::Gray(luma), dpi),
        }?;
        self.push(page_num, half, coded);
        Ok(())
    }

    /// Hold `coded` as document page `page_num`, half `half` of a split
    /// spread.
    pub(crate) fn push(&self, page_num: i32, half: Option<SpreadHalf>, coded: PdfPage) {
        let half = match half {
            None | Some(SpreadHalf::Left) => 0,
            Some(SpreadHalf::Right) => 1,
        };
        self.pages.push((page_num, half), coded);
    }
}

/// An encoder of a page of `size` pixels at `dpi`, colour when `rgb`
/// and gray otherwise, its compressed image held in a file of its own
/// in the spool's directory when `spill`.
///
/// # Errors
///
/// The file could not be created, or the encoder refused the page.
pub(crate) fn page_encoder(
    (size, dpi): ((u32, u32), (f64, f64)),
    rgb: bool,
    spill: bool,
) -> Result<PdfPageEncoder, RenderError> {
    let file = if spill {
        let dir = spool_dir();
        Some(tempfile::tempfile_in(&dir).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("cannot create a page spill file in {}: {e}", dir.display()),
            )
        })?)
    } else {
        None
    };
    let layout = if rgb { RowLayout::Rgb } else { RowLayout::Gray };
    Ok(PdfPageEncoder::new(size, layout, dpi, file)?)
}

/// The directory of the spool: `$TMPDIR` when set, else `/var/tmp` where it
/// exists (a disk, where `/tmp` may be memory), else the system's temporary
/// directory.
fn spool_dir() -> PathBuf {
    let var_tmp = Path::new("/var/tmp");
    if std::env::var_os("TMPDIR").is_none() && var_tmp.is_dir() {
        var_tmp.to_path_buf()
    } else {
        std::env::temp_dir()
    }
}

/// How many pages per pool thread may be started past the first page not
/// yet appended to the spool.
const PAGES_AHEAD_PER_THREAD: usize = 2;

/// The pages of a print job that are done, by index in the page list, and
/// the errors of those that failed.
struct Progress {
    done: Vec<bool>,
    errors: Vec<(i32, RenderError)>,
}

/// What the workers of one print job share: its progress, the signal of a
/// page done, and whether pages not yet begun are to be skipped.
struct Job {
    progress: Mutex<Progress>,
    finished: Condvar,
    stop: AtomicBool,
}

/// Marks page `index` of a print job done when dropped, with its error.
/// The page counts as done even when its render unwinds; an error or an
/// unwinding render stops the job.
struct PageDone<'a> {
    job: &'a Job,
    index: usize,
    error: Option<(i32, RenderError)>,
}

impl Drop for PageDone<'_> {
    fn drop(&mut self) {
        if self.error.is_some() || std::thread::panicking() {
            self.job.stop.store(true, Ordering::Relaxed);
        }
        let mut state = lock(&self.job.progress);
        state.done[self.index] = true;
        state.errors.extend(self.error.take());
        drop(state);
        self.job.finished.notify_one();
    }
}

/// The print document being written.
pub(crate) struct Spool {
    writer: PdfWriter<BufWriter<File>>,
}

impl Spool {
    /// Open an anonymous temporary file in [`spool_dir`] and begin the PDF.
    ///
    /// # Errors
    ///
    /// The file could not be created or its first bytes written.
    pub(crate) fn create() -> Result<Self, RenderError> {
        let dir = spool_dir();
        let file = tempfile::tempfile_in(&dir).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("cannot create a spool file in {}: {e}", dir.display()),
            )
        })?;
        let writer = PdfWriter::new(BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file))?;
        Ok(Self { writer })
    }

    /// Append the pages `sink` holds for document pages up to `last`
    /// inclusive, in page order.
    ///
    /// # Errors
    ///
    /// The spool could not be written.
    pub(crate) fn write_through(&mut self, sink: &PageSink, last: i32) -> Result<(), RenderError> {
        for page in sink.pages.take_through((last, u8::MAX)) {
            self.writer.write_page(&page)?;
        }
        Ok(())
    }

    /// Render `pages`, strictly ascending page numbers, with `render` on
    /// `pool` into `sink`, appending a page to the spool once it and every
    /// page before it are done.  A page is begun only while it lies fewer
    /// than [`PAGES_AHEAD_PER_THREAD`] × threads pages past the first page
    /// not yet appended, which bounds the encoded pages held in memory.
    /// Once a page fails, a render unwinds or the spool cannot be written,
    /// no further page is begun: pages already queued are skipped, and
    /// pages rendering finish.
    ///
    /// Returns the pages that failed, with their errors, and the spool's
    /// write error.
    pub(crate) fn render_in_order<F>(
        &mut self,
        sink: &PageSink,
        pages: &[i32],
        pool: &rayon::ThreadPool,
        render: F,
    ) -> (Vec<(i32, RenderError)>, Option<RenderError>)
    where
        F: Fn(&i32) -> Option<(i32, RenderError)> + Sync,
    {
        debug_assert!(
            pages.is_sorted_by(|a, b| a < b),
            "print pages must be strictly ascending: the spool takes pages by number"
        );
        let ahead = pool.current_num_threads() * PAGES_AHEAD_PER_THREAD;
        let job = Job {
            progress: Mutex::new(Progress {
                done: vec![false; pages.len()],
                errors: Vec::new(),
            }),
            finished: Condvar::new(),
            stop: AtomicBool::new(false),
        };
        let mut write_error = None;
        let (job, render) = (&job, &render);
        pool.in_place_scope(|scope| {
            let (mut started, mut written) = (0, 0);
            loop {
                while started < pages.len() && started < written + ahead {
                    let (index, page) = (started, &pages[started]);
                    scope.spawn(move |_| {
                        let mut done = PageDone {
                            job,
                            index,
                            error: None,
                        };
                        if !job.stop.load(Ordering::Relaxed) {
                            done.error = render(page);
                        }
                    });
                    started += 1;
                }
                let mut state = lock(&job.progress);
                while written < started && !state.done[written] {
                    state = job
                        .finished
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
                if written == started || job.stop.load(Ordering::Relaxed) {
                    return;
                }
                let ready = state.done[written..started]
                    .iter()
                    .take_while(|&&done| done)
                    .count();
                drop(state);
                written += ready;
                if let Err(e) = self.write_through(sink, pages[written - 1]) {
                    job.stop.store(true, Ordering::Relaxed);
                    write_error = Some(e);
                    return;
                }
            }
        });
        let errors = std::mem::take(&mut lock(&job.progress).errors);
        (errors, write_error)
    }

    /// Complete the document and its file, rewound to its start.
    fn finish(self) -> Result<File, String> {
        let mut file = self
            .writer
            .finish()
            .map_err(|e| e.to_string())?
            .into_inner()
            .map_err(|e| e.error().to_string())?;
        file.rewind().map_err(|e| e.to_string())?;
        Ok(file)
    }

    /// Complete the document and submit it as `job`, `lp` reading it from
    /// its standard input.
    ///
    /// # Errors
    ///
    /// No page was written, the document could not be completed, or `lp`
    /// refused it.
    pub(crate) fn submit(self, job: &Submission<'_>) -> Result<(), String> {
        let file = self
            .finish()
            .map_err(|e| format!("the print document was not written: {e}"))?;
        deliver::submit(
            "lp",
            &deliver::lp_args(job.printer, lp_options(job), None),
            file.into(),
            "the print job",
            None,
            CUPS_CLIENT_HINT,
        )
        .map_err(|e| refusal_hint(e, job.printer.is_some()))
    }
}

/// What to do when `lp` or `lpstat` is not installed.
const CUPS_CLIENT_HINT: &str =
    "install the CUPS client (cups-client on Debian and Ubuntu, cups on Fedora and Arch)";

/// The message for a print job `lp` did not take, with a pointer to the
/// queues when `lp` ran and refused it: `named` is whether `--printer` chose
/// the queue, else the job went to the default destination, which may be
/// unset.
fn refusal_hint(e: deliver::SubmitError, named: bool) -> String {
    match e {
        deliver::SubmitError::NotStarted(msg) => msg,
        deliver::SubmitError::Refused(msg) if named => {
            format!("{msg}; --list-printers lists the queues")
        }
        deliver::SubmitError::Refused(msg) => format!(
            "{msg}; when no default destination is set, name a queue with --printer; \
             --list-printers lists them"
        ),
    }
}

/// What `--list-printers` writes: the CUPS destinations, one per line, the
/// default one followed by a tab and `default`.
///
/// # Errors
///
/// `lpstat` is not installed or cannot reach the scheduler.
pub(crate) fn list_printers() -> Result<String, String> {
    if cfg!(windows) {
        return Err(
            "--list-printers asks CUPS for its queues, which Windows does not have".to_owned(),
        );
    }
    let lpstat_e = deliver::query("lpstat", &["-e"], CUPS_CLIENT_HINT)?;
    let lpstat_d = deliver::query("lpstat", &["-d"], CUPS_CLIENT_HINT)?;
    let queues = queue_names(&lpstat_e);
    let default = default_destination(&lpstat_d, &queues);
    Ok(printer_list(&queues, default))
}

/// Write the `--list-printers` listing to `out` and flush it.  A reader
/// that closed the pipe has taken all it wants, so a broken pipe is not
/// an error.
///
/// # Errors
///
/// Any other write error.
pub(crate) fn write_printer_list(out: &mut impl Write, list: &str) -> io::Result<()> {
    match out.write_all(list.as_bytes()).and_then(|()| out.flush()) {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

/// The destinations `lpstat -e` reports, one per line.
fn queue_names(lpstat_e: &str) -> Vec<&str> {
    lpstat_e
        .lines()
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .collect()
}

/// The queue of `queues` that `lpstat -d` names as the default
/// destination, if any.  The `-d` line ends with `: ` and the queue name in
/// every locale, so the match does not depend on the wording before it; the
/// longest matching name wins, so `Office/duplex` is not taken for a queue
/// `duplex`.  Output that names no queue and is not the C locale's
/// no-default line is logged.
fn default_destination<'q>(lpstat_d: &str, queues: &[&'q str]) -> Option<&'q str> {
    let line = lpstat_d.trim();
    let default = queues
        .iter()
        .filter(|q| {
            line.strip_suffix(**q)
                .is_some_and(|head| head.trim_end().ends_with(':'))
        })
        .max_by_key(|q| q.len())
        .copied();
    if default.is_none() && !line.is_empty() && line != "no system default destination" {
        log::warn!(
            "lpstat -d reported {line:?}, which names none of the queues; \
             no queue is marked default"
        );
    }
    default
}

/// The listing of `queues`, one per line, `default` marked.
fn printer_list(queues: &[&str], default: Option<&str>) -> String {
    let mut list = String::new();
    for &queue in queues {
        list.push_str(queue);
        if Some(queue) == default {
            list.push_str("\tdefault");
        }
        list.push('\n');
    }
    list
}

#[cfg(test)]
mod tests {
    use std::io::{self, Read};

    use color::Gray8;
    use raster::Bitmap;

    use super::{
        Duplex, PAGES_AHEAD_PER_THREAD, PageSink, Spool, Submission, default_destination,
        job_title, lp_options, printer_list, queue_names, refusal_hint, write_printer_list,
    };
    use crate::cleanup::{Page, SpreadHalf};
    use crate::deliver::SubmitError;
    use crate::render::RenderError;

    #[test]
    fn printers_are_listed_one_per_line_with_the_default_marked() {
        let queues = queue_names("Office\nOffice/duplex\n duplex \n\nLab\n");
        assert_eq!(queues, ["Office", "Office/duplex", "duplex", "Lab"]);
        assert_eq!(
            printer_list(&queues, Some("Office/duplex")),
            "Office\nOffice/duplex\tdefault\nduplex\nLab\n"
        );
        assert_eq!(
            printer_list(&queues, None),
            "Office\nOffice/duplex\nduplex\nLab\n"
        );
        assert_eq!(printer_list(&[], None), "");
    }

    /// A writer whose every write fails with `kind`.
    struct Failing(std::io::ErrorKind);

    impl std::io::Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(self.0.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_closed_pipe_ends_the_printer_list_quietly_and_other_errors_surface() {
        let mut out = Vec::new();
        write_printer_list(&mut out, "Office\n").unwrap();
        assert_eq!(out, b"Office\n");
        assert!(
            write_printer_list(&mut Failing(std::io::ErrorKind::BrokenPipe), "Office\n").is_ok()
        );
        let err = write_printer_list(
            &mut Failing(std::io::ErrorKind::PermissionDenied),
            "Office\n",
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn the_default_destination_is_found_in_any_locale() {
        let queues = ["Office", "Office/duplex", "duplex", "Lab"];
        for (lpstat_d, default) in [
            (
                "system default destination: Office/duplex\n",
                Some("Office/duplex"),
            ),
            ("system default destination: duplex\n", Some("duplex")),
            ("destination système par défaut : Lab\n", Some("Lab")),
            ("Standardziel des Systems: Office\n", Some("Office")),
            ("no system default destination\n", None),
            ("system default destination: Gone\n", None),
            ("", None),
        ] {
            assert_eq!(
                default_destination(lpstat_d, &queues),
                default,
                "{lpstat_d:?}"
            );
        }
    }

    #[test]
    fn a_refused_job_points_at_the_queues_and_a_job_lp_never_saw_does_not() {
        let refused = || SubmitError::Refused("lp refused the print job".to_owned());
        assert_eq!(
            refusal_hint(refused(), true),
            "lp refused the print job; --list-printers lists the queues"
        );
        let default = refusal_hint(refused(), false);
        assert!(
            default.contains("--printer") && default.contains("--list-printers"),
            "{default}"
        );
        let missing = "lp is not installed: install the CUPS client";
        assert_eq!(
            refusal_hint(SubmitError::NotStarted(missing.to_owned()), true),
            missing
        );
    }

    #[test]
    fn lp_gets_copies_title_format_and_sides_only_when_given() {
        let job = Submission {
            printer: Some("Office"),
            copies: Some(2),
            duplex: Some(Duplex::Long),
            title: "report.pdf".to_owned(),
        };
        assert_eq!(
            lp_options(&job),
            [
                "-n",
                "2",
                "-t",
                "report.pdf",
                "-o",
                "document-format=application/pdf",
                "-o",
                "sides=two-sided-long-edge",
            ]
        );
        let default = Submission {
            printer: None,
            copies: None,
            duplex: None,
            title: "-o".to_owned(),
        };
        // The title is the value of -t, never an option of its own.
        assert_eq!(
            lp_options(&default),
            ["-t", "-o", "-o", "document-format=application/pdf"]
        );
    }

    #[test]
    fn titles_are_file_names_without_control_characters_within_255_bytes() {
        assert_eq!(job_title("/docs/report.pdf"), "report.pdf");
        assert_eq!(job_title("-"), "rrocket");
        assert_eq!(job_title("a\nb\x1b.pdf"), "a b .pdf");
        let long = "é".repeat(200);
        let title = job_title(&long);
        assert!(title.len() <= 255 && title.chars().all(|c| c == 'é'));
    }

    fn gray(w: u32, h: u32) -> Page {
        Page::Gray(Bitmap::<Gray8>::new(w, h, 1, false))
    }

    /// The `(width, height)` of each image in a PDF, in file order.
    fn image_sizes(pdf: &[u8]) -> Vec<(u32, u32)> {
        let text = String::from_utf8_lossy(pdf);
        text.match_indices("/Width ")
            .map(|(at, _)| {
                let mut fields = text[at..]
                    .split('/')
                    .skip(1)
                    .take(2)
                    .map(|f| f.split(' ').nth(1).unwrap().parse::<u32>().unwrap());
                (fields.next().unwrap(), fields.next().unwrap())
            })
            .collect()
    }

    #[test]
    fn pages_spool_in_page_order() {
        let sink = PageSink::new();
        let mut spool = Spool::create().unwrap();
        sink.add((2, None), &gray(4, 6), (72.0, 72.0)).unwrap();
        sink.add((1, Some(SpreadHalf::Right)), &gray(3, 5), (72.0, 72.0))
            .unwrap();
        sink.add((1, Some(SpreadHalf::Left)), &gray(2, 5), (72.0, 72.0))
            .unwrap();
        sink.add((3, None), &gray(8, 5), (72.0, 72.0)).unwrap();
        spool.write_through(&sink, 2).unwrap();
        spool.write_through(&sink, 3).unwrap();
        let mut pdf = Vec::new();
        let n = spool.finish().unwrap().read_to_end(&mut pdf).unwrap();
        assert_eq!(n, pdf.len());
        assert_eq!(image_sizes(&pdf), [(2, 5), (3, 5), (4, 6), (8, 5)]);
    }

    /// What [`render_pages`] observed.
    struct Rendered {
        /// The spooled image widths, empty when a page failed.
        widths: Vec<u32>,
        /// The pages that failed.
        failed: Vec<i32>,
        /// The most encoded pages ever held at once.
        peak: usize,
        /// How many pages `render` was called for.
        renders: usize,
    }

    /// Render `n` pages on two threads, page `p` a `p × 1` image, the
    /// early pages slowest; page `fail` fails at once and page `panic`
    /// panics.
    fn render_pages(n: i32, fail: Option<i32>, panic: Option<i32>) -> Rendered {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let sink = PageSink::new();
        let mut spool = Spool::create().unwrap();
        let pages: Vec<i32> = (1..=n).collect();
        let (peak, renders) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let render = |&page: &i32| {
            let _ = renders.fetch_add(1, Ordering::Relaxed);
            if Some(page) == fail {
                return Some((page, RenderError::Io(io::Error::other("refused"))));
            }
            std::thread::sleep(std::time::Duration::from_millis(
                u64::try_from(n - page).unwrap() * 3,
            ));
            assert_ne!(Some(page), panic, "page {page} panics");
            sink.add((page, None), &gray(page.unsigned_abs(), 1), (72.0, 72.0))
                .unwrap();
            let _ = peak.fetch_max(sink.pages.len(), Ordering::Relaxed);
            None
        };
        let (errors, write_error) = spool.render_in_order(&sink, &pages, &pool, render);
        assert!(write_error.is_none());
        let failed: Vec<i32> = errors.iter().map(|&(page, _)| page).collect();
        let widths = if failed.is_empty() {
            let mut pdf = Vec::new();
            let n = spool.finish().unwrap().read_to_end(&mut pdf).unwrap();
            assert_eq!(n, pdf.len());
            image_sizes(&pdf).into_iter().map(|(w, _)| w).collect()
        } else {
            Vec::new()
        };
        Rendered {
            widths,
            failed,
            peak: peak.into_inner(),
            renders: renders.into_inner(),
        }
    }

    #[test]
    fn pages_spool_in_page_order_as_they_finish_with_bounded_holding() {
        let run = render_pages(24, None, None);
        assert_eq!(run.widths, (1..=24).collect::<Vec<u32>>());
        assert_eq!(run.failed, Vec::<i32>::new());
        assert_eq!(run.renders, 24);
        assert!(
            run.peak <= 2 * PAGES_AHEAD_PER_THREAD,
            "{} pages held",
            run.peak
        );
    }

    #[test]
    fn a_failed_page_stops_new_pages_and_is_reported() {
        // Pages 1–4 are queued at once; page 1 fails while page 2 renders,
        // so pages 3 and 4 are skipped.
        let run = render_pages(24, Some(1), None);
        assert_eq!(run.failed, [1]);
        assert!(
            run.renders <= 2,
            "{} pages rendered after page 1 failed",
            run.renders
        );
    }

    #[test]
    fn a_panicking_page_propagates_instead_of_hanging() {
        let run = std::panic::catch_unwind(|| render_pages(24, None, Some(2)));
        assert!(run.is_err(), "the panic of page 2 reaches the caller");
    }

    #[test]
    fn an_empty_document_is_refused() {
        assert!(Spool::create().unwrap().finish().is_err());
    }

    /// A page larger than A0 prints at 360 dpi without a resolution flag;
    /// A0 and smaller at 600.
    #[test]
    fn large_pages_print_at_the_large_format_resolution() {
        let inch = 72.0;
        for ((w, h), dpi) in [
            ((8.5, 11.0), 600.0),
            ((33.11, 46.81), 600.0),
            ((36.0, 44.0), 600.0),
            ((36.0, 48.0), 360.0),
            ((44.0, 120.0), 360.0),
        ] {
            assert_eq!(
                super::default_dpi((w * inch, h * inch)),
                dpi,
                "{w} × {h} in"
            );
        }
    }
}
