//! The `--print` output.  On Linux and macOS the pages go to a CUPS queue
//! as one PDF of page images (see [`cups`]), whose filters fit them to the
//! printer.  On Windows they go to the queue's driver through GDI (see
//! `windows`), and rrocket fits them to the sheet itself by the rules of
//! `gdi`.
//!
//! Each page is compressed on the worker that rendered it into a
//! [`PageSink`] and appended to the [`Spool`] once every page before it is
//! done.  Pages start in page order, at most a fixed number per thread past
//! the first page not yet appended, so at most that many encoded pages are
//! held in memory; a slow page delays the appending of the pages after it
//! and the start of those more than that number past it.  Pages are
//! appended on the thread that runs [`Spool::render_job`], which on Windows
//! owns the printer's device context.

#[cfg(not(windows))]
mod cups;
#[cfg(any(windows, test))]
pub(crate) mod gdi;
#[cfg(windows)]
mod windows;

use std::io::{self, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use color::SampleDepth;
use encode::{PdfPage, PdfPageEncoder, RowImage, RowLayout};

use crate::args::Args;
use crate::cleanup::{Page, SpreadHalf};
use crate::render::{PageCollector, RenderError};

/// The resolution of a printed page when no resolution flag is given, up
/// to [`LARGE_FORMAT_PX`].
pub(crate) const PRINT_DPI: f64 = 600.0;

/// The resolution of a printed page past [`LARGE_FORMAT_PX`] when no
/// resolution flag is given: the native input resolution of large-format
/// photo printers' drivers.
pub(crate) const LARGE_FORMAT_DPI: f64 = 360.0;

/// The pixels of 8-bit RGB at [`PRINT_DPI`] past which a printed page
/// takes [`LARGE_FORMAT_DPI`]: 600 Mpx, about 1 667 square inches, larger
/// than A0 (1 550) and than 36 × 44 in.  A canvas of more bytes a pixel
/// takes it past proportionally fewer pixels, so that the bytes of the
/// canvas it renders are bounded alike.
pub(crate) const LARGE_FORMAT_PX: f64 = 600e6;

/// The bytes a pixel of 8-bit RGB holds, in which [`LARGE_FORMAT_PX`] is
/// counted.
const LARGE_FORMAT_BYTES_PER_PX: f64 = 3.0;

/// The resolution of a printed page of `width_pts × height_pts`, rendered
/// on a canvas of `bytes_per_px` bytes a pixel, when no resolution flag is
/// given.
pub(crate) fn default_dpi((width_pts, height_pts): (f64, f64), bytes_per_px: u32) -> f64 {
    let px_per_pt = PRINT_DPI / 72.0;
    let px = width_pts * px_per_pt * height_pts * px_per_pt;
    if px * f64::from(bytes_per_px) > LARGE_FORMAT_PX * LARGE_FORMAT_BYTES_PER_PX {
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
    /// luma page, at the page's depth.
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
            Page::Gray(luma) | Page::Mono(luma) => PdfPage::encode(RowImage::Gray(luma), dpi),
            page => PdfPage::encode(page.row_image(), dpi),
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
/// and gray otherwise, of samples of `depth`, its compressed image held in
/// a file of its own in the spool's directory when `spill`.
///
/// # Errors
///
/// The file could not be created, or the encoder refused the page.
pub(crate) fn page_encoder(
    (size, dpi): ((u32, u32), (f64, f64)),
    (rgb, depth): (bool, SampleDepth),
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
    let layout = if rgb { RowLayout::Rgb } else { RowLayout::Gray }.at_depth(depth);
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

/// Where a [`Spool`] sends its pages, in page order.
pub(crate) trait PageOut {
    /// Send `page`, the next page of the job.
    ///
    /// # Errors
    ///
    /// The page could not be sent; the job is then abandoned.
    fn write_page(&mut self, page: &PdfPage) -> Result<(), RenderError>;

    /// How many times the job's pages are sent: more than once when the
    /// printer cannot make the copies asked for itself.
    fn passes(&self) -> u32 {
        1
    }

    /// Close a pass that another follows, once its every page is sent.
    ///
    /// # Errors
    ///
    /// The pass could not be closed; the job is then abandoned.
    fn end_pass(&mut self) -> Result<(), RenderError> {
        Ok(())
    }
}

/// The print job being written.
pub(crate) struct Spool<W> {
    out: W,
    /// Whether the run is interrupted: [`crate::interrupt::interrupted`].
    interrupted: fn() -> bool,
}

impl<W> Spool<W> {
    /// The job written to `out`, abandoned once the run is interrupted.
    const fn new(out: W) -> Self {
        Self {
            out,
            interrupted: crate::interrupt::interrupted,
        }
    }
}

/// The print job of this platform's route.
#[cfg(not(windows))]
pub(crate) type PrintSpool = Spool<cups::Document>;
/// The print job of this platform's route.
#[cfg(windows)]
pub(crate) type PrintSpool = Spool<windows::Job>;

/// Begin the print job `args` asks for.  On Windows this opens the printer
/// and starts the job, and records the printer's sheet in `args` for the
/// resolution of each page (see [`Args::page_dpi`]).
///
/// # Errors
///
/// The job could not be begun: on Linux and macOS the spool file could not
/// be made; on Windows the printer could not be opened or cannot print the
/// job as asked.
#[cfg_attr(
    not(windows),
    expect(
        clippy::needless_pass_by_ref_mut,
        reason = "on Windows the job records the printer's sheet in `args`"
    )
)]
pub(crate) fn begin(args: &mut Args) -> Result<PrintSpool, String> {
    #[cfg(not(windows))]
    {
        cups::begin(args)
    }
    #[cfg(windows)]
    {
        windows::begin(args, None)
    }
}

/// Complete the print job: on Linux and macOS submit the document with
/// `lp`, on Windows end the job.
///
/// # Errors
///
/// No page was written, or the job was refused.
pub(crate) fn complete(spool: PrintSpool, args: &Args) -> Result<(), String> {
    #[cfg(not(windows))]
    {
        cups::complete(spool, args)
    }
    #[cfg(windows)]
    {
        let _ = args;
        spool.out.finish()
    }
}

impl<W: PageOut> Spool<W> {
    /// Append the pages `sink` holds for document pages up to `last`
    /// inclusive, in page order.
    ///
    /// # Errors
    ///
    /// The page could not be appended.
    pub(crate) fn write_through(&mut self, sink: &PageSink, last: i32) -> Result<(), RenderError> {
        for page in sink.pages.take_through((last, u8::MAX)) {
            self.out.write_page(&page)?;
        }
        Ok(())
    }

    /// [`Self::render_in_order`] once for each of the job's passes (see
    /// [`PageOut::passes`]), until a pass fails, each pass but the last
    /// closed by [`PageOut::end_pass`].
    pub(crate) fn render_job<F>(
        &mut self,
        sink: &PageSink,
        pages: &[i32],
        pool: &rayon::ThreadPool,
        render: F,
    ) -> (Vec<(i32, RenderError)>, Option<RenderError>)
    where
        F: Fn(&i32) -> Option<(i32, RenderError)> + Sync,
    {
        let mut result = (Vec::new(), None);
        for pass in 0..self.out.passes() {
            if pass > 0
                && let Err(e) = self.out.end_pass()
            {
                return (Vec::new(), Some(e));
            }
            result = self.render_in_order(sink, pages, pool, &render);
            if !result.0.is_empty() || result.1.is_some() {
                break;
            }
        }
        result
    }

    /// Render `pages`, strictly ascending page numbers, with `render` on
    /// `pool` into `sink`, appending a page to the spool once it and every
    /// page before it are done (see [`crate::pool::run_in_order`], which
    /// bounds the encoded pages held in memory).  Once a page fails, a
    /// render unwinds or the spool cannot be written, no further page is
    /// begun: pages already queued are skipped, and pages rendering finish.
    /// So too once the run is interrupted (see [`crate::interrupt`],
    /// [`crate::pool::run_in_order_until`]), the spool's error then of kind
    /// [`io::ErrorKind::Interrupted`].
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
        let (mut errors, mut write_error) = (Vec::new(), None);
        let interrupted = self.interrupted;
        let unreported = crate::pool::run_in_order_until(
            pool,
            pages,
            render,
            |index, failed| {
                if let Some(error) = failed {
                    errors.push(error);
                    return ControlFlow::Break(());
                }
                self.write_through(sink, pages[index]).map_or_else(
                    |e| {
                        write_error = Some(e);
                        ControlFlow::Break(())
                    },
                    |()| ControlFlow::Continue(()),
                )
            },
            interrupted,
        );
        errors.extend(unreported.into_iter().flatten());
        if write_error.is_none() && interrupted() {
            write_error = Some(RenderError::Io(io::Error::new(
                io::ErrorKind::Interrupted,
                "the run was interrupted",
            )));
        }
        (errors, write_error)
    }
}

/// What `--list-printers` writes: the printers `--printer` can name, one
/// per line, the default one followed by a tab and `default`.
///
/// # Errors
///
/// The print service could not be asked.
pub(crate) fn list_printers() -> Result<String, String> {
    #[cfg(not(windows))]
    {
        cups::list_printers()
    }
    #[cfg(windows)]
    {
        windows::list_printers()
    }
}

/// The message for a listing that names no printer.
pub(crate) const NO_PRINTERS: &str = if cfg!(windows) {
    "Windows has no printers"
} else {
    "CUPS has no printers"
};

/// Parse a `--printer` value: a CUPS queue name (see
/// [`crate::deliver::parse_queue`]); on Windows a printer name, which may
/// hold spaces but no control character and is not empty.
///
/// # Errors
///
/// The value cannot name a printer.
pub(crate) fn parse_printer(s: &str) -> Result<String, String> {
    if !cfg!(windows) {
        return crate::deliver::parse_queue(s);
    }
    if s.trim().is_empty() {
        return Err("a printer name is not empty".to_owned());
    }
    if let Some(c) = s.chars().find(|c| c.is_control()) {
        return Err(format!("{s:?} is not a printer name: it holds {c:?}"));
    }
    Ok(s.to_owned())
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
    use std::io;

    use color::Gray8;
    use encode::PdfPage;
    use raster::Bitmap;

    use super::{
        PageOut, PageSink, Spool, job_title, parse_printer, printer_list, write_printer_list,
    };
    use crate::cleanup::{Page, SpreadHalf};
    use crate::pool::JOBS_AHEAD_PER_THREAD;
    use crate::render::RenderError;

    /// The image sizes of the pages sent, in order, the job sent `passes`
    /// times; a pass closed between two is recorded as `(0, 0)`.
    struct Sizes {
        sent: Vec<(u32, u32)>,
        passes: u32,
    }

    impl PageOut for Sizes {
        fn write_page(&mut self, page: &PdfPage) -> Result<(), RenderError> {
            self.sent.push(page.size());
            Ok(())
        }

        fn passes(&self) -> u32 {
            self.passes
        }

        fn end_pass(&mut self) -> Result<(), RenderError> {
            self.sent.push((0, 0));
            Ok(())
        }
    }

    fn spool(passes: u32) -> Spool<Sizes> {
        Spool::new(Sizes {
            sent: Vec::new(),
            passes,
        })
    }

    #[test]
    fn printers_are_listed_one_per_line_with_the_default_marked() {
        let queues = ["Office", "Microsoft Print to PDF", "Lab"];
        assert_eq!(
            printer_list(&queues, Some("Microsoft Print to PDF")),
            "Office\nMicrosoft Print to PDF\tdefault\nLab\n"
        );
        assert_eq!(
            printer_list(&queues, None),
            "Office\nMicrosoft Print to PDF\nLab\n"
        );
        assert_eq!(printer_list(&[], None), "");
    }

    /// A printer name is a CUPS queue name, except on Windows, whose
    /// printer names hold spaces.
    #[test]
    fn printer_names_follow_the_platform() {
        for name in ["Office", "Office/duplex"] {
            assert_eq!(parse_printer(name).as_deref(), Ok(name));
        }
        for bad in ["", "a\nb", "a\tb"] {
            assert!(parse_printer(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            parse_printer("Microsoft Print to PDF").is_ok(),
            cfg!(windows)
        );
        assert!(parse_printer("   ").is_err());
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

    #[test]
    fn pages_spool_in_page_order() {
        let sink = PageSink::new();
        let mut spool = spool(1);
        sink.add((2, None), &gray(4, 6), (72.0, 72.0)).unwrap();
        sink.add((1, Some(SpreadHalf::Right)), &gray(3, 5), (72.0, 72.0))
            .unwrap();
        sink.add((1, Some(SpreadHalf::Left)), &gray(2, 5), (72.0, 72.0))
            .unwrap();
        sink.add((3, None), &gray(8, 5), (72.0, 72.0)).unwrap();
        spool.write_through(&sink, 2).unwrap();
        assert_eq!(spool.out.sent, [(2, 5), (3, 5), (4, 6)]);
        spool.write_through(&sink, 3).unwrap();
        assert_eq!(spool.out.sent, [(2, 5), (3, 5), (4, 6), (8, 5)]);
    }

    /// What [`render_pages`] observed.
    struct Rendered {
        /// The sent image widths, empty when a page failed.
        widths: Vec<u32>,
        /// The pages that failed.
        failed: Vec<i32>,
        /// The most encoded pages ever held at once.
        peak: usize,
        /// How many pages `render` was called for.
        renders: usize,
    }

    /// Render `n` pages on two threads in `passes` passes, page `p` a
    /// `p × 1` image, the early pages slowest; page `fail` fails at once
    /// and page `panic` panics.  Once a page has failed, pages not yet
    /// begun return at once, as the CLI's page renderer does for a whole
    /// document.
    fn render_pages(n: i32, passes: u32, fail: Option<i32>, panic: Option<i32>) -> Rendered {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let sink = PageSink::new();
        let mut spool = spool(passes);
        let pages: Vec<i32> = (1..=n).collect();
        let (peak, renders) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let failed = AtomicBool::new(false);
        let render = |&page: &i32| {
            if failed.load(Ordering::Relaxed) {
                return None;
            }
            let _ = renders.fetch_add(1, Ordering::Relaxed);
            if Some(page) == fail {
                failed.store(true, Ordering::Relaxed);
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
        let (errors, write_error) = spool.render_job(&sink, &pages, &pool, render);
        assert!(write_error.is_none());
        let failed: Vec<i32> = errors.iter().map(|&(page, _)| page).collect();
        let widths = if failed.is_empty() {
            spool.out.sent.iter().map(|&(w, _)| w).collect()
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
        let run = render_pages(24, 1, None, None);
        assert_eq!(run.widths, (1..=24).collect::<Vec<u32>>());
        assert_eq!(run.failed, Vec::<i32>::new());
        assert_eq!(run.renders, 24);
        assert!(
            run.peak <= 2 * JOBS_AHEAD_PER_THREAD,
            "{} pages held",
            run.peak
        );
    }

    /// A job of three passes sends its pages three times, each pass in
    /// page order: collated copies, the passes closed between them (a
    /// width of 0 here).
    #[test]
    fn each_pass_sends_every_page_in_order() {
        let run = render_pages(5, 3, None, None);
        assert_eq!(
            run.widths,
            [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        );
        assert_eq!(run.renders, 15);
    }

    #[test]
    fn a_failed_page_stops_new_pages_and_is_reported() {
        // Pages 1–4 are queued at once; page 1 fails while page 2 renders,
        // so pages 3 and 4 are skipped, and no second pass begins.
        let run = render_pages(24, 2, Some(1), None);
        assert_eq!(run.failed, [1]);
        assert!(
            run.renders <= 2,
            "{} pages rendered after page 1 failed",
            run.renders
        );
    }

    /// An interrupt while page 3 renders: no page is begun after one
    /// window, no second pass begins, at most the pages before it are
    /// sent, and the spool's error is of kind `Interrupted`.
    #[test]
    fn an_interrupt_stops_the_job_with_an_interrupted_error() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        static FLAG: AtomicBool = AtomicBool::new(false);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let sink = PageSink::new();
        let mut spool = spool(2);
        spool.interrupted = || FLAG.load(Ordering::Relaxed);
        let pages: Vec<i32> = (1..=24).collect();
        let renders = AtomicUsize::new(0);
        let (errors, write_error) = spool.render_job(&sink, &pages, &pool, |&page: &i32| {
            let _ = renders.fetch_add(1, Ordering::Relaxed);
            if page == 3 {
                FLAG.store(true, Ordering::Relaxed);
            }
            sink.add((page, None), &gray(page.unsigned_abs(), 1), (72.0, 72.0))
                .unwrap();
            None
        });
        assert!(errors.is_empty());
        let Some(RenderError::Io(e)) = write_error else {
            panic!("{write_error:?}");
        };
        assert_eq!(e.kind(), io::ErrorKind::Interrupted);
        let renders = renders.into_inner();
        assert!(
            renders <= 2 * JOBS_AHEAD_PER_THREAD,
            "{renders} pages rendered"
        );
        assert!(spool.out.sent.len() < 3, "{:?}", spool.out.sent);
    }

    #[test]
    fn a_panicking_page_propagates_instead_of_hanging() {
        let run = std::panic::catch_unwind(|| render_pages(24, 1, None, Some(2)));
        assert!(run.is_err(), "the panic of page 2 reaches the caller");
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
                super::default_dpi((w * inch, h * inch), 3),
                dpi,
                "{w} × {h} in"
            );
        }
    }
}
