//! The CUPS route: the pages as one PDF, each page a single image at the
//! render resolution (see [`encode::PdfWriter`]), handed to a queue with
//! `lp`.  The queue's filters fit the images to the printer: media and
//! orientation, colour, resolution and two-sided sheets; the pages are not
//! rendered again.  The PDF is an anonymous temporary file, private to the
//! process and freed however it ends; `lp` reads it as its standard input.

use std::fs::File;
use std::io::{BufWriter, Seek};

use color::SampleDepth;
use encode::{IccProfile, PdfPage, PdfWriter};

use super::{Duplex, PageOut, Spool, spool_dir};
use crate::args::Args;
use crate::deliver;
use crate::render::{OUTPUT_BUFFER_BYTES, RenderError};

/// The print document: a PDF in an anonymous temporary file.
pub(crate) type Document = PdfWriter<BufWriter<File>>;

impl PageOut for Document {
    fn write_page(&mut self, page: &PdfPage) -> Result<(), RenderError> {
        Ok(Self::write_page(self, page)?)
    }
}

/// What `lp` is told about the job; an option not given leaves the
/// queue's default.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Submission<'a> {
    /// The queue, else the default destination.
    printer: Option<&'a str>,
    copies: Option<u32>,
    duplex: Option<Duplex>,
    /// The job's name (see [`super::job_title`]).
    title: String,
}

impl<'a> Submission<'a> {
    /// The job `args` asks for.
    fn of(args: &'a Args) -> Self {
        Self {
            printer: args.printer.as_deref(),
            copies: args.copies,
            duplex: args.duplex,
            title: super::job_title(args.input()),
        }
    }
}

/// The IPP `sides` keyword of `duplex`.
const fn sides(duplex: Duplex) -> &'static str {
    match duplex {
        Duplex::Long => "two-sided-long-edge",
        Duplex::Short => "two-sided-short-edge",
    }
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
        options.extend(["-o".to_owned(), format!("sides={}", sides(duplex))]);
    }
    options
}

/// Open an anonymous temporary file in [`spool_dir`] and begin the PDF of
/// the job `args` asks for, whose pages hold images of samples of its
/// depth, each tagged with its output profile when given.
///
/// # Errors
///
/// The file could not be created or its first bytes written.
pub(crate) fn begin(args: &Args) -> Result<Spool<Document>, String> {
    create(args.depth(), args.output_icc.as_ref()).map_err(|e| e.to_string())
}

/// [`begin`] for pages of `depth`, tagged with `icc` when given.
fn create(depth: SampleDepth, icc: Option<&IccProfile>) -> Result<Spool<Document>, RenderError> {
    let dir = spool_dir();
    let file = tempfile::tempfile_in(&dir).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("cannot create a spool file in {}: {e}", dir.display()),
        )
    })?;
    let out = PdfWriter::new(
        BufWriter::with_capacity(OUTPUT_BUFFER_BYTES, file),
        depth,
        icc,
    )?;
    Ok(Spool { out })
}

/// Complete the document and its file, rewound to its start.
fn finish(spool: Spool<Document>) -> Result<File, String> {
    let mut file = spool
        .out
        .finish()
        .map_err(|e| e.to_string())?
        .into_inner()
        .map_err(|e| e.error().to_string())?;
    file.rewind().map_err(|e| e.to_string())?;
    Ok(file)
}

/// Complete the document and submit it as the job `args` asks for, `lp`
/// reading it from its standard input.
///
/// # Errors
///
/// No page was written, the document could not be completed, or `lp`
/// refused it.
pub(crate) fn complete(spool: Spool<Document>, args: &Args) -> Result<(), String> {
    let job = Submission::of(args);
    let file = finish(spool).map_err(|e| format!("the print document was not written: {e}"))?;
    deliver::submit(
        "lp",
        &deliver::lp_args(job.printer, lp_options(&job), None),
        file.into(),
        "the print job",
        None,
        CUPS_CLIENT_HINT,
    )
    .map_err(|e| refusal_hint(e, job.printer.is_some()))
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
    let lpstat_e = deliver::query("lpstat", &["-e"], CUPS_CLIENT_HINT)?;
    let lpstat_d = deliver::query("lpstat", &["-d"], CUPS_CLIENT_HINT)?;
    let queues = queue_names(&lpstat_e);
    let default = default_destination(&lpstat_d, &queues);
    Ok(super::printer_list(&queues, default))
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

#[cfg(test)]
mod tests {
    use std::io::Read;

    use color::{Gray8, SampleDepth};
    use raster::Bitmap;

    use super::{
        Submission, create, default_destination, finish, lp_options, queue_names, refusal_hint,
    };
    use crate::cleanup::{Page, SpreadHalf};
    use crate::deliver::SubmitError;
    use crate::print::{Duplex, PageSink, printer_list};

    #[test]
    fn printers_are_listed_one_per_line_with_the_default_marked() {
        let queues = queue_names("Office\nOffice/duplex\n duplex \n\nLab\n");
        assert_eq!(queues, ["Office", "Office/duplex", "duplex", "Lab"]);
        assert_eq!(
            printer_list(&queues, Some("Office/duplex")),
            "Office\nOffice/duplex\tdefault\nduplex\nLab\n"
        );
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
    fn pages_spool_into_the_document_in_page_order() {
        let sink = PageSink::new();
        let mut spool = create(SampleDepth::Eight, None).unwrap();
        sink.add((2, None), &gray(4, 6), (72.0, 72.0)).unwrap();
        sink.add((1, Some(SpreadHalf::Right)), &gray(3, 5), (72.0, 72.0))
            .unwrap();
        sink.add((1, Some(SpreadHalf::Left)), &gray(2, 5), (72.0, 72.0))
            .unwrap();
        spool.write_through(&sink, 2).unwrap();
        let mut pdf = Vec::new();
        let n = finish(spool).unwrap().read_to_end(&mut pdf).unwrap();
        assert_eq!(n, pdf.len());
        assert_eq!(image_sizes(&pdf), [(2, 5), (3, 5), (4, 6)]);
    }

    /// A spool made with an output profile holds it once and tags every
    /// page image with it.
    #[test]
    fn a_spool_tags_its_pages_with_the_output_profile() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/icc/printer-gray-v4.icc"
        );
        let icc = encode::IccProfile::new(std::fs::read(path).unwrap()).unwrap();
        let sink = PageSink::new();
        let mut spool = create(SampleDepth::Eight, Some(&icc)).unwrap();
        sink.add((1, None), &gray(4, 6), (72.0, 72.0)).unwrap();
        sink.add((2, None), &gray(3, 5), (72.0, 72.0)).unwrap();
        spool.write_through(&sink, 2).unwrap();
        let mut pdf = Vec::new();
        let _ = finish(spool).unwrap().read_to_end(&mut pdf).unwrap();
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.starts_with("%PDF-1.7"), "a v4.3 profile needs PDF 1.7");
        assert_eq!(text.matches("/N 1/Alternate/DeviceGray").count(), 1);
        assert_eq!(text.matches("/ColorSpace[/ICCBased 3 0 R]").count(), 2);
    }

    #[test]
    fn an_empty_document_is_refused() {
        assert!(finish(create(SampleDepth::Eight, None).unwrap()).is_err());
    }
}
