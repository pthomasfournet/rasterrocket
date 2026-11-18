//! In-memory library rows: render a page range from bytes held in memory.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;

use pdf_raster::{
    BackendPolicy, RenderParams, SessionConfig, open_session_from_bytes,
    render_page_rgb_with_diagnostics, rgb_into_gray,
};

/// Outcome of one rep: pages rendered, wall microseconds, and how many pages
/// reported diagnostics (a skipped image or content stream, a form past the
/// depth limit, a missing font, an unsupported feature).
#[derive(Debug)]
pub(crate) struct Rep {
    pub pages: u32,
    pub wall_us: u64,
    pub degraded: u32,
}

/// A document and the inclusive page range to render.  A bare path covers
/// `1..=u32::MAX`; the upper bound is clamped to the page count at run time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DocRange {
    pub path: PathBuf,
    pub first: u32,
    pub last: u32,
}

impl DocRange {
    /// Parse `PATH` or `PATH:FIRST-LAST`.  The text after the last `:` is a
    /// range only when what precedes it is not a lone drive letter and it
    /// holds no path separator; otherwise the whole string is the path.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let bare = |t: &str| Self {
            path: PathBuf::from(t),
            first: 1,
            last: u32::MAX,
        };
        let Some((head, tail)) = text.rsplit_once(':') else {
            return Ok(bare(text));
        };
        let drive = head.len() == 1 && head.as_bytes()[0].is_ascii_alphabetic();
        if drive || tail.contains(['/', '\\']) {
            return Ok(bare(text));
        }
        let bad = || format!("{text}: range must be FIRST-LAST with 1 <= FIRST <= LAST");
        let (a, b) = tail.split_once('-').ok_or_else(bad)?;
        let first: u32 = a.parse().map_err(|_| bad())?;
        let last: u32 = b.parse().map_err(|_| bad())?;
        if first == 0 || first > last {
            return Err(bad());
        }
        Ok(Self {
            path: PathBuf::from(head),
            first,
            last,
        })
    }

    /// The file name alone, for reporting.
    pub(crate) fn file_name(&self) -> String {
        self.path.file_name().map_or_else(
            || self.path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        )
    }

    fn read(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))
    }

    /// The effective inclusive range once the page count is known.
    fn clamp(&self, total: u32) -> Result<(u32, u32), String> {
        let last = self.last.min(total);
        if self.first > last {
            return Err(format!(
                "{}: first page {} is past the {total} page(s)",
                self.file_name(),
                self.first
            ));
        }
        Ok((self.first, last))
    }
}

/// One timing sample; `tsv_line` gives the harness's result-row format.
#[derive(Debug, Clone)]
pub(crate) struct Sample {
    pub group: String,
    pub row: String,
    pub doc: String,
    pub pages: u32,
    pub threads: usize,
    pub rep: u32,
    pub wall_us: u64,
}

impl Sample {
    /// `group\trow\tdoc\tpages\tthreads\trep\twall_us`.
    pub(crate) fn tsv_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.group, self.row, self.doc, self.pages, self.threads, self.rep, self.wall_us
        )
    }
}

fn micros(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// Open the document from memory and render its range with `threads` workers.
/// The file is read before the clock starts; the session open is timed.
/// Returns the pages rendered and wall microseconds.  With `page_times`, each
/// page's render time is appended as (page, microseconds).
/// Pages that reported omissions are counted, not failed.
pub(crate) fn library_rep(
    doc: &DocRange,
    dpi: f64,
    threads: usize,
    page_times: Option<&mut Vec<(u32, u64)>>,
) -> Result<Rep, String> {
    let bytes = doc.read()?;
    let name = doc.file_name();
    let params = RenderParams::new(dpi / 72.0);
    let degraded = AtomicU32::new(0);
    let want_times = page_times.is_some();

    let start = Instant::now();
    let session =
        open_session_from_bytes(bytes, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .map_err(|e| format!("{name}: {e}"))?;
    let (first, last) = doc.clamp(session.total_pages())?;
    let next = AtomicU32::new(first);
    let stop = AtomicBool::new(false);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let times: Mutex<Vec<(u32, u64)>> = Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let mut local = Vec::new();
                while !stop.load(Ordering::Relaxed) {
                    let page = next.fetch_add(1, Ordering::Relaxed);
                    if page > last {
                        break;
                    }
                    let t = Instant::now();
                    match render_page_rgb_with_diagnostics(&session, page, &params) {
                        Ok((bitmap, diag)) => {
                            drop(bitmap);
                            if diag.has_omissions() {
                                degraded.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(e) => {
                            stop.store(true, Ordering::Relaxed);
                            let mut slot = failure
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            slot.get_or_insert(format!("{name} page {page}: {e}"));
                            break;
                        }
                    }
                    if want_times {
                        local.push((page, micros(t)));
                    }
                }
                times
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .extend(local);
            });
        }
    });
    let wall_us = micros(start);

    if let Some(e) = failure
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        return Err(e);
    }
    if let Some(out) = page_times {
        out.extend(
            times
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
    Ok(Rep {
        pages: last - first + 1,
        wall_us,
        degraded: degraded.into_inner(),
    })
}

/// Render each page, convert to gray outside the clock, and time only the
/// CPU deskew, summed over pages (`wall_us` is that sum).  The
/// timed span covers skew detection and, where a skew is found, the rotation.
pub(crate) fn deskew_rep(doc: &DocRange, dpi: f64) -> Result<Rep, String> {
    let bytes = doc.read()?;
    let name = doc.file_name();
    let session =
        open_session_from_bytes(bytes, &SessionConfig::with_policy(BackendPolicy::CpuOnly))
            .map_err(|e| format!("{name}: {e}"))?;
    let (first, last) = doc.clamp(session.total_pages())?;
    let params = RenderParams::new(dpi / 72.0);
    let mut total = 0_u64;
    let mut degraded = 0_u32;
    for page in first..=last {
        let (rgb, diag) = render_page_rgb_with_diagnostics(&session, page, &params)
            .map_err(|e| format!("{name} page {page}: {e}"))?;
        degraded += u32::from(diag.has_omissions());
        let mut gray = rgb_into_gray(rgb);
        let start = Instant::now();
        pdf_raster::deskew::apply(&mut gray, BackendPolicy::CpuOnly)
            .map_err(|e| format!("{name} page {page}: {e}"))?;
        total += micros(start);
    }
    Ok(Rep {
        pages: last - first + 1,
        wall_us: total,
        degraded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-page PDF, each page one stroked line, with a correct xref.
    fn two_page_pdf() -> Vec<u8> {
        pdf_with_contents("10 10 m 90 90 l S", "10 90 m 90 10 l S")
    }

    /// A two-page PDF whose pages draw the given content streams, with
    /// `/Length` and the xref computed from the bytes.
    fn pdf_with_contents(first: &str, second: &str) -> Vec<u8> {
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".to_owned(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >>".to_owned(),
            stream(first),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 6 0 R >>".to_owned(),
            stream(second),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{body}\nendobj\n", i + 1).bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        out
    }

    fn stream(content: &str) -> String {
        format!(
            "<< /Length {} >>\nstream\n{content}\nendstream",
            content.len()
        )
    }

    fn fixture(tag: &str) -> PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/os-bench")
            .join(format!("test_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two.pdf");
        std::fs::write(&path, two_page_pdf()).unwrap();
        path
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn doc_range_parses_suffix() {
        let d = DocRange::parse("a/corpus-04.pdf:1-64").unwrap();
        assert_eq!((d.first, d.last), (1, 64));
        assert_eq!(d.path, PathBuf::from("a/corpus-04.pdf"));
        let d = DocRange::parse("a/b.pdf").unwrap();
        assert_eq!((d.first, d.last), (1, u32::MAX));
        for bad in ["x.pdf:0-3", "x.pdf:5-2", "x.pdf:a-b"] {
            assert!(DocRange::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn doc_range_keeps_windows_drive_letters_in_the_path() {
        let d = DocRange::parse(r"C:\x\corpus-04.pdf:1-64").unwrap();
        assert_eq!((d.first, d.last), (1, 64));
        assert_eq!(d.path, PathBuf::from(r"C:\x\corpus-04.pdf"));
        let d = DocRange::parse(r"C:\x\a.pdf").unwrap();
        assert_eq!((d.first, d.last), (1, u32::MAX));
        assert_eq!(d.path, PathBuf::from(r"C:\x\a.pdf"));
    }

    #[test]
    fn tsv_line_has_seven_tab_separated_fields_and_a_bare_file_name() {
        let s = Sample {
            group: "library".into(),
            row: "L12".into(),
            doc: "corpus-02-native-vector-text.pdf".into(),
            pages: 16,
            threads: 12,
            rep: 1,
            wall_us: 123,
        };
        let line = s.tsv_line();
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 7);
        assert!(fields.iter().all(|f| !f.contains(['/', '\\'])));
    }

    #[test]
    fn library_rep_renders_every_page_of_an_in_memory_pdf() {
        let path = fixture("all");
        let doc = DocRange::parse(path.to_str().unwrap()).unwrap();
        let r = library_rep(&doc, 72.0, 2, None).unwrap();
        let (pages, wall_us) = (r.pages, r.wall_us);
        assert_eq!(pages, 2);
        assert!(wall_us > 0);
        let mut v = Vec::new();
        let pages = library_rep(&doc, 72.0, 2, Some(&mut v)).unwrap().pages;
        assert_eq!(pages, 2);
        let mut nums: Vec<u32> = v.iter().map(|&(p, _)| p).collect();
        nums.sort_unstable();
        assert_eq!(nums, [1, 2]);
        cleanup(&path);
    }

    #[test]
    fn a_first_page_past_the_page_count_is_an_error_naming_the_file() {
        let path = fixture("past");
        let doc = DocRange::parse(&format!("{}:3-3", path.display())).unwrap();
        let e = library_rep(&doc, 72.0, 1, None).unwrap_err();
        assert!(e.contains("two.pdf"), "{e}");
        cleanup(&path);
    }

    #[test]
    fn a_missing_file_is_an_error_naming_the_path() {
        let doc = DocRange::parse("no/such/dir/absent.pdf").unwrap();
        let e = library_rep(&doc, 72.0, 1, None).unwrap_err();
        assert!(e.contains("absent.pdf"), "{e}");
        assert!(deskew_rep(&doc, 72.0).is_err());
    }

    #[test]
    fn a_page_that_cannot_be_rendered_fails_the_rep_naming_doc_and_page() {
        // The page tree promises two pages but lists one kid.
        let pdf = String::from_utf8(two_page_pdf())
            .unwrap()
            .replace("[3 0 R 5 0 R]", "[3 0 R 99 0 R]");
        let path = fixture("badpage");
        std::fs::write(&path, pdf).unwrap();
        let doc = DocRange::parse(path.to_str().unwrap()).unwrap();
        let e = library_rep(&doc, 72.0, 2, None).unwrap_err();
        assert!(e.starts_with("two.pdf page 2:"), "{e}");
        cleanup(&path);
    }

    #[test]
    fn a_clean_document_reports_no_degraded_pages() {
        let path = fixture("clean");
        let doc = DocRange::parse(path.to_str().unwrap()).unwrap();
        assert_eq!(library_rep(&doc, 72.0, 2, None).unwrap().degraded, 0);
        cleanup(&path);
    }

    #[test]
    fn a_page_selecting_an_undefined_font_counts_as_degraded() {
        let path = fixture("nofont");
        std::fs::write(
            &path,
            pdf_with_contents("BT /F9 12 Tf 10 50 Td (x) Tj ET", "10 90 m 90 10 l S"),
        )
        .unwrap();
        let doc = DocRange::parse(path.to_str().unwrap()).unwrap();
        let r = library_rep(&doc, 72.0, 1, None).unwrap();
        assert_eq!((r.pages, r.degraded), (2, 1));
        cleanup(&path);
    }

    #[test]
    fn doc_range_last_is_clamped_to_the_page_count() {
        let path = fixture("clamp");
        let doc = DocRange::parse(&format!("{}:1-9", path.display())).unwrap();
        let pages = library_rep(&doc, 72.0, 1, None).unwrap().pages;
        assert_eq!(pages, 2);
        cleanup(&path);
    }
}
