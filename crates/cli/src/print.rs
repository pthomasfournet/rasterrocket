//! The `--print` output: the rendered pages as one PDF, each page a single
//! image at the render resolution (see [`encode::PdfWriter`]), spooled to a
//! temporary file and submitted to a CUPS queue with `lp`.  The queue's
//! filters fit the images to the printer: media and orientation, colour,
//! resolution and two-sided sheets; the pages are not rendered again.
//!
//! Pages render in windows of as many pages as the pool has threads.  Each
//! page is compressed on the worker that rendered it, and a window's pages
//! are appended to the spool file in page order once the window is done,
//! so at most one window of pages is held.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use encode::{PdfImage, PdfPage, PdfWriter};

use crate::cleanup::{Page, SpreadHalf};
use crate::deliver;
use crate::render::{OUTPUT_BUFFER_BYTES, RenderError};

/// The resolution of a printed page when no resolution flag is given.
pub(crate) const PRINT_DPI: f64 = 600.0;

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

/// What `lp` is told about the job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Submission<'a> {
    /// The queue, else the default destination.
    pub printer: Option<&'a str>,
    pub copies: u32,
    pub duplex: Option<Duplex>,
    /// The job's name.
    pub title: &'a str,
}

/// Parse a CUPS destination name, a queue optionally followed by
/// `/instance`: non-empty, without spaces or control characters, and not
/// beginning with `-`, so no command reads it as an option.
pub(crate) fn parse_printer(s: &str) -> Result<String, String> {
    if s.is_empty() {
        return Err("a printer name is not empty".to_owned());
    }
    if s.starts_with('-') {
        return Err(format!("{s:?} is not a printer name: it begins with -"));
    }
    if let Some(c) = s.chars().find(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!("{s:?} is not a printer name: it holds {c:?}"));
    }
    Ok(s.to_owned())
}

/// The arguments of `lp` that submit the PDF at `path`.
fn submission_args(path: &str, job: &Submission<'_>) -> Vec<String> {
    let mut argv = Vec::new();
    if let Some(printer) = job.printer {
        argv.extend(["-d".to_owned(), printer.to_owned()]);
    }
    argv.extend([
        "-n".to_owned(),
        job.copies.to_string(),
        "-t".to_owned(),
        job.title.to_owned(),
        "-o".to_owned(),
        "document-format=application/pdf".to_owned(),
    ]);
    if let Some(duplex) = job.duplex {
        argv.extend(["-o".to_owned(), format!("sides={}", duplex.sides())]);
    }
    argv.extend(["--".to_owned(), path.to_owned()]);
    argv
}

/// The position of a page in the document: its page number, then the left
/// half of a split spread before the right.
type PageKey = (i32, u8);

/// A print job being spooled.
pub(crate) struct PrintJob {
    dpi: (f64, f64),
    path: PathBuf,
    writer: Mutex<PdfWriter<BufWriter<File>>>,
    /// The pages of the current window, encoded.
    window: Mutex<Vec<(PageKey, PdfPage)>>,
}

impl PrintJob {
    /// Open a new spool file in the temporary directory for pages at
    /// `dpi`.
    ///
    /// # Errors
    ///
    /// The file could not be created or its first bytes written.
    pub(crate) fn create(dpi: (f64, f64)) -> Result<Self, RenderError> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path =
            std::env::temp_dir().join(format!("rrocket-print-{}-{nanos}.pdf", std::process::id()));
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| {
                std::io::Error::new(e.kind(), format!("cannot create {}: {e}", path.display()))
            })?;
        let writer = PdfWriter::new(BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file));
        match writer {
            Ok(writer) => Ok(Self {
                dpi,
                path,
                writer: Mutex::new(writer),
                window: Mutex::new(Vec::new()),
            }),
            Err(e) => {
                let _ = fs::remove_file(&path);
                Err(e.into())
            }
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
        self.window
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(((page_num, half), coded));
        Ok(())
    }

    /// Append the window's pages to the spool file in page order.
    ///
    /// # Errors
    ///
    /// The spool file could not be written.
    pub(crate) fn write_window(&self) -> Result<(), RenderError> {
        let mut window = std::mem::take(
            &mut *self
                .window
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        window.sort_unstable_by_key(|&(key, _)| key);
        let mut writer = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, page) in window {
            writer.write_page(&page)?;
        }
        drop(writer);
        Ok(())
    }

    /// Render `pages` with `render` on `pool`, a window of as many pages as
    /// the pool has threads at a time, spooling each window once it is
    /// done.  Rendering stops at the first window with a failed page, or
    /// when the spool file cannot be written.
    ///
    /// Returns the pages that failed, with their errors, and the spool
    /// file's write error.
    pub(crate) fn render_in_windows<F>(
        &self,
        pages: &[i32],
        pool: &rayon::ThreadPool,
        render: F,
    ) -> (Vec<(i32, RenderError)>, Option<RenderError>)
    where
        F: Fn(&i32) -> Option<(i32, RenderError)> + Sync + Send + Copy,
    {
        use rayon::prelude::*;

        let mut errors = Vec::new();
        for window in pages.chunks(pool.current_num_threads().max(1)) {
            errors
                .extend(pool.install(|| window.par_iter().filter_map(render).collect::<Vec<_>>()));
            if !errors.is_empty() {
                return (errors, None);
            }
            if let Err(e) = self.write_window() {
                return (errors, Some(e));
            }
        }
        (errors, None)
    }

    /// Complete the spool file and submit it as `job`.  The file is
    /// removed once `lp` has taken it; when the submission fails it stays,
    /// and the message names it.
    ///
    /// # Errors
    ///
    /// No page was written, the file could not be completed, or `lp`
    /// refused it.
    pub(crate) fn submit(self, job: &Submission<'_>) -> Result<(), String> {
        let path = self.path.to_string_lossy().into_owned();
        let completed = self
            .write_window()
            .map_err(|e| e.to_string())
            .and_then(|()| {
                self.writer
                    .into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .finish()
                    .map_err(|e| e.to_string())
            })
            .and_then(|mut out| out.flush().map_err(|e| e.to_string()));
        if let Err(e) = completed {
            let _ = fs::remove_file(&self.path);
            return Err(format!("the print document was not written: {e}"));
        }
        deliver::submit(
            "lp",
            &submission_args(&path, job),
            "the print job",
            &path,
            "install the CUPS client (cups-client, or cups-bsd for lpr)",
        )?;
        if let Err(e) = fs::remove_file(&self.path) {
            eprintln!("rrocket: warning: could not remove the spooled {path}: {e}");
        }
        Ok(())
    }

    /// Remove the spool file without submitting it.
    pub(crate) fn discard(self) {
        drop(self.writer);
        if let Err(e) = fs::remove_file(&self.path) {
            eprintln!(
                "rrocket: warning: could not remove the spooled {}: {e}",
                self.path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use color::Gray8;
    use raster::Bitmap;

    use super::{Duplex, PrintJob, Submission, parse_printer, submission_args};
    use crate::cleanup::{Page, SpreadHalf};

    #[test]
    fn printer_names_cannot_pass_for_options() {
        for name in ["Office", "HP_LaserJet-4", "Office/duplex", "queue@host"] {
            assert_eq!(parse_printer(name).as_deref(), Ok(name));
        }
        for bad in ["", "-o", "Office Jet", "a\tb", "a\nb"] {
            assert!(parse_printer(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lp_gets_the_queue_copies_title_format_and_sides() {
        let job = Submission {
            printer: Some("Office"),
            copies: 2,
            duplex: Some(Duplex::Long),
            title: "report.pdf",
        };
        assert_eq!(
            submission_args("/tmp/j.pdf", &job),
            [
                "-d",
                "Office",
                "-n",
                "2",
                "-t",
                "report.pdf",
                "-o",
                "document-format=application/pdf",
                "-o",
                "sides=two-sided-long-edge",
                "--",
                "/tmp/j.pdf"
            ]
        );
        let default = Submission {
            printer: None,
            copies: 1,
            duplex: None,
            title: "-o",
        };
        let argv = submission_args("/tmp/j.pdf", &default);
        assert_eq!(argv[0], "-n");
        // The title is the value of -t, never an option of its own.
        assert_eq!(&argv[2..4], ["-t", "-o"]);
        assert_eq!(argv.len(), 8);
    }

    fn gray(w: u32, h: u32) -> Page {
        Page::Gray(Bitmap::<Gray8>::new(w, h, 1, false))
    }

    /// The `(width, height)` of each image in a spooled PDF, in file order.
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
        let job = PrintJob::create((72.0, 72.0)).unwrap();
        let path = job.path.clone();
        job.add(2, None, &gray(4, 6)).unwrap();
        job.add(1, Some(SpreadHalf::Right), &gray(3, 5)).unwrap();
        job.add(1, Some(SpreadHalf::Left), &gray(2, 5)).unwrap();
        job.write_window().unwrap();
        job.add(3, None, &gray(8, 5)).unwrap();
        job.write_window().unwrap();
        let writer = job.writer.into_inner().unwrap();
        drop(writer.finish().unwrap().into_inner().unwrap());
        let pdf = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(image_sizes(&pdf), [(2, 5), (3, 5), (4, 6), (8, 5)]);
    }

    #[test]
    fn a_discarded_job_leaves_no_file() {
        let job = PrintJob::create((300.0, 300.0)).unwrap();
        let path = job.path.clone();
        assert!(path.exists());
        job.discard();
        assert!(!path.exists());
    }
}
