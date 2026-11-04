//! The `--print` output: the rendered pages as one PDF, each page a single
//! image at the render resolution (see [`encode::PdfWriter`]), handed to a
//! CUPS queue with `lp`.  The queue's filters fit the images to the
//! printer: media and orientation, colour, resolution and two-sided
//! sheets; the pages are not rendered again.
//!
//! Pages render in windows of as many pages as the pool has threads.  Each
//! page is compressed on the worker that rendered it into a [`PageSink`],
//! and a window's pages are appended to the [`Spool`] in page order once the
//! window is done, so at most one window of pages is held in memory.  The
//! spool is an anonymous temporary file, private to the process and freed
//! however it ends; `lp` reads it as its standard input.

use std::fs::File;
use std::io::{BufWriter, Seek};
use std::path::{Path, PathBuf};

use encode::{PdfImage, PdfPage, PdfWriter};

use crate::cleanup::{Page, SpreadHalf};
use crate::deliver;
use crate::render::{OUTPUT_BUFFER_BYTES, PageCollector, RenderError};

/// The resolution of a printed page when no resolution flag is given.
pub(crate) const PRINT_DPI: f64 = 600.0;

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

/// The encoded pages of the current window, shared by the workers.
pub(crate) struct PageSink {
    dpi: (f64, f64),
    pages: PageCollector<PageKey, PdfPage>,
}

impl PageSink {
    /// An empty sink for pages rendered at `dpi`.
    pub(crate) const fn new(dpi: (f64, f64)) -> Self {
        Self {
            dpi,
            pages: PageCollector::new(),
        }
    }

    /// Compress `page` and hold it as document page `page_num`, half
    /// `half` of a split spread: colour, or gray for a luma page.
    ///
    /// # Errors
    ///
    /// The encoder refused the page.
    pub(crate) fn add(
        &self,
        page_num: i32,
        half: Option<SpreadHalf>,
        page: &Page,
    ) -> Result<(), RenderError> {
        let coded = match page {
            Page::Rgb(rgb) => PdfPage::encode(PdfImage::Rgb(rgb), self.dpi),
            Page::Gray(luma) | Page::Mono(luma) => PdfPage::encode(PdfImage::Gray(luma), self.dpi),
        }?;
        let half = match half {
            None | Some(SpreadHalf::Left) => 0,
            Some(SpreadHalf::Right) => 1,
        };
        self.pages.push((page_num, half), coded);
        Ok(())
    }
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

    /// Append the pages `sink` holds, in page order.
    ///
    /// # Errors
    ///
    /// The spool could not be written.
    pub(crate) fn write_from(&mut self, sink: &PageSink) -> Result<(), RenderError> {
        for page in sink.pages.take_sorted() {
            self.writer.write_page(&page)?;
        }
        Ok(())
    }

    /// Render `pages` with `render` on `pool`, a window of as many pages as
    /// the pool has threads at a time into `sink`, appending each window
    /// once it is done.  Rendering stops at the first window with a failed
    /// page, or when the spool cannot be written.
    ///
    /// Returns the pages that failed, with their errors, and the spool's
    /// write error.
    pub(crate) fn render_in_windows<F>(
        &mut self,
        sink: &PageSink,
        pages: &[i32],
        pool: &rayon::ThreadPool,
        render: F,
    ) -> (Vec<(i32, RenderError)>, Option<RenderError>)
    where
        F: Fn(&i32) -> Option<(i32, RenderError)> + Sync + Send + Copy,
    {
        use rayon::prelude::*;

        for window in pages.chunks(pool.current_num_threads().max(1)) {
            let errors: Vec<_> = pool.install(|| window.par_iter().filter_map(render).collect());
            if !errors.is_empty() {
                return (errors, None);
            }
            if let Err(e) = self.write_from(sink) {
                return (Vec::new(), Some(e));
            }
        }
        (Vec::new(), None)
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
            "install the CUPS client (cups-client on Debian and Ubuntu, cups on Fedora and Arch)",
        )
        .map_err(|e| match job.printer {
            Some(_) => format!("{e}; --list-printers lists the queues"),
            None => e,
        })
    }
}

/// What `--list-printers` writes: the CUPS destinations, one per line, the
/// default one followed by a tab and `default`.
///
/// # Errors
///
/// `lpstat` is not installed or cannot reach the scheduler.
pub(crate) fn list_printers() -> Result<String, String> {
    const MISSING: &str = "--list-printers asks CUPS for its queues; install CUPS";
    if cfg!(windows) {
        return Err(
            "--list-printers asks CUPS for its queues, which Windows does not have".to_owned(),
        );
    }
    let queues = deliver::query("lpstat", &["-e"], MISSING)?;
    let default = deliver::query("lpstat", &["-d"], MISSING)?;
    Ok(printer_list(&queues, default_destination(&default)))
}

/// The default destination `lpstat -d` reports in the C locale, if any.
fn default_destination(lpstat_d: &str) -> Option<&str> {
    lpstat_d
        .lines()
        .find_map(|line| line.strip_prefix("system default destination:"))
        .map(str::trim)
        .filter(|name| !name.is_empty())
}

/// The listing of the destinations `lpstat -e` reports, one per line,
/// `default` marked.
fn printer_list(lpstat_e: &str, default: Option<&str>) -> String {
    let mut list = String::new();
    for queue in lpstat_e.lines().map(str::trim).filter(|q| !q.is_empty()) {
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
    use std::io::Read;

    use color::Gray8;
    use raster::Bitmap;

    use super::{
        Duplex, PageSink, Spool, Submission, default_destination, job_title, lp_options,
        printer_list,
    };
    use crate::cleanup::{Page, SpreadHalf};

    #[test]
    fn printers_are_listed_one_per_line_with_the_default_marked() {
        let lpstat_e = "Office\nOffice/duplex\nLab\n";
        assert_eq!(
            default_destination("system default destination: Office/duplex\n"),
            Some("Office/duplex")
        );
        assert_eq!(default_destination("no system default destination\n"), None);
        assert_eq!(
            printer_list(lpstat_e, Some("Office/duplex")),
            "Office\nOffice/duplex\tdefault\nLab\n"
        );
        assert_eq!(printer_list(lpstat_e, None), "Office\nOffice/duplex\nLab\n");
        assert_eq!(printer_list("", None), "");
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
        let sink = PageSink::new((72.0, 72.0));
        let mut spool = Spool::create().unwrap();
        sink.add(2, None, &gray(4, 6)).unwrap();
        sink.add(1, Some(SpreadHalf::Right), &gray(3, 5)).unwrap();
        sink.add(1, Some(SpreadHalf::Left), &gray(2, 5)).unwrap();
        spool.write_from(&sink).unwrap();
        sink.add(3, None, &gray(8, 5)).unwrap();
        spool.write_from(&sink).unwrap();
        let mut pdf = Vec::new();
        let n = spool.finish().unwrap().read_to_end(&mut pdf).unwrap();
        assert_eq!(n, pdf.len());
        assert_eq!(image_sizes(&pdf), [(2, 5), (3, 5), (4, 6), (8, 5)]);
    }

    #[test]
    fn an_empty_document_is_refused() {
        assert!(Spool::create().unwrap().finish().is_err());
    }
}
