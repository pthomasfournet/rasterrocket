//! The `--fax` output: one TIFF Class F document (RFC 2306) holding every
//! rendered page, each [`FAX_WIDTH`] pixels wide at [`FAX_X_DPI`] pixels
//! per inch and 196 (fine) or 98 (standard) lines per inch, bilevel and
//! fax-coded.
//!
//! A page is laid across the fax width by its shorter side: a landscape
//! page is turned a quarter counterclockwise, its top edge to the left.
//! Pages are coded on the worker that rendered them and written in page
//! order once all are done.  The written document can be handed to the
//! system's fax service through its submission command, `lp` or `sendfax`.

use std::process::{Command, Stdio};
use std::sync::Mutex;

use color::Gray8;
use encode::{TiffCompression, TiffImage, TiffPage, TiffWriter};
use raster::Bitmap;

use crate::cleanup::Page;
use crate::render::{RenderError, write_output};

/// The width of a fax page in pixels: ITU-T T.4's 215 mm scan line.
pub(crate) const FAX_WIDTH: u32 = 1728;

/// The horizontal resolution of a fax page, pixels per inch.
pub(crate) const FAX_X_DPI: f64 = 204.0;

/// How a page is laid on the fax width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FaxLayout {
    /// Device pixels per point along the page's width.
    pub scale_x: f64,
    /// Device pixels per point along the page's height.
    pub scale_y: f64,
    /// Whether the rendered page is turned a quarter counterclockwise.
    pub turn: bool,
}

/// The layout of a page of `width_pts × height_pts` at `y_dpi` lines per
/// inch: its shorter side spans [`FAX_WIDTH`] pixels and its longer side
/// keeps the aspect ratio of the page under the fax's pixel shape.
pub(crate) fn layout((width_pts, height_pts): (f64, f64), y_dpi: f64) -> FaxLayout {
    let turn = width_pts > height_pts;
    let across = f64::from(FAX_WIDTH) / if turn { height_pts } else { width_pts };
    let along = across * y_dpi / FAX_X_DPI;
    if turn {
        FaxLayout {
            scale_x: along,
            scale_y: across,
            turn,
        }
    } else {
        FaxLayout {
            scale_x: across,
            scale_y: along,
            turn,
        }
    }
}

/// The pages of the fax document, coded and keyed by page number.
pub(crate) struct FaxDocument {
    compression: TiffCompression,
    dpi: (f64, f64),
    pages: Mutex<Vec<(i32, TiffPage)>>,
}

impl FaxDocument {
    /// An empty document coded with `compression` at `dpi`.
    pub(crate) const fn new(compression: TiffCompression, dpi: (f64, f64)) -> Self {
        Self {
            compression,
            dpi,
            pages: Mutex::new(Vec::new()),
        }
    }

    /// Turn `page` when `turn`, fit it to the fax width and code it,
    /// bilevel, as document page `page_num`.  The page is the two-level
    /// luma of [`crate::tone::prepare`]; any luma is thresholded.
    ///
    /// # Errors
    ///
    /// The encoder refused the page.
    pub(crate) fn add(&self, page_num: i32, page: Page, turn: bool) -> Result<(), RenderError> {
        let luma = page.into_luma();
        let luma = if turn { quarter_turn(&luma) } else { luma };
        let fitted = fit_width(&luma, FAX_WIDTH);
        let coded = TiffPage::encode(TiffImage::Bilevel(&fitted), self.dpi, self.compression)?;
        self.pages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((page_num, coded));
        Ok(())
    }

    /// Write the document, its pages in order, to `path`.
    ///
    /// # Errors
    ///
    /// No page was added, or the file could not be written.
    pub(crate) fn write(self, path: &str) -> Result<(), RenderError> {
        let mut pages = self
            .pages
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pages.sort_unstable_by_key(|&(page, _)| page);
        let mut writer = TiffWriter::fax_document();
        for (_, page) in pages {
            writer.push_page(page);
        }
        write_output(path, |out| writer.write(out).map_err(RenderError::from))
    }
}

/// Parse a fax number into its dial string: the number is dial digits,
/// `+`, `*`, `#`, the pause `,` and the separators space, `-`, `.`, `(` and
/// `)`, beginning with a dial character; the dial string drops the
/// separators.  It never begins with `-`, so no command reads it as an
/// option, and holds no space, which would split the `name=value` options
/// of `lp`.
pub(crate) fn parse_number(s: &str) -> Result<String, String> {
    let dial = |c: char| c.is_ascii_digit() || matches!(c, '+' | '*' | '#');
    if !s.starts_with(dial) {
        return Err(format!(
            "{s:?} is not a fax number: it begins with a digit, +, * or #"
        ));
    }
    if let Some(c) = s
        .chars()
        .find(|&c| !(dial(c) || matches!(c, ',' | ' ' | '-' | '.' | '(' | ')')))
    {
        return Err(format!("{s:?} is not a fax number: {c:?} is not dialled"));
    }
    Ok(s.chars().filter(|&c| dial(c) || c == ',').collect())
}

/// The program and arguments that submit the fax document at `path` to
/// `number`: `lp` to the print queue `queue` when one is named, its phone
/// option carrying the number, else `sendfax` without a cover page.
fn submission(path: &str, number: &str, queue: Option<&str>) -> (&'static str, Vec<String>) {
    queue.map_or_else(
        || {
            (
                "sendfax",
                ["-n", "-d", number, "--", path].map(str::to_owned).to_vec(),
            )
        },
        |queue| {
            (
                "lp",
                vec![
                    "-d".to_owned(),
                    queue.to_owned(),
                    "-o".to_owned(),
                    format!("phone={number}"),
                    "--".to_owned(),
                    path.to_owned(),
                ],
            )
        },
    )
}

/// Submit the fax document at `path` to the dial string `number` through
/// the system's fax service (see [`submission`]) and wait for the service
/// to accept it.  The command's output goes to standard error, keeping
/// standard output for rrocket's own.
///
/// # Errors
///
/// The command could not be run or reported a failure; the message names
/// the document, which stays at `path`.
pub(crate) fn submit(path: &str, number: &str, queue: Option<&str>) -> Result<(), String> {
    let (program, argv) = submission(path, number, queue);
    let status = Command::new(program)
        .args(&argv)
        .stdin(Stdio::null())
        .stdout(std::io::stderr())
        .status()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "{program} is not installed: name a CUPS fax queue with --fax-queue, or \
                     install HylaFAX for sendfax; the document is at {path}"
                )
            } else {
                format!("cannot run {program}: {e}; the document is at {path}")
            }
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} refused the fax ({status}); the document is at {path}"
        ))
    }
}

/// `src` turned a quarter counterclockwise: its top row becomes the left
/// column, read upward.
fn quarter_turn(src: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let (w, h) = (src.width, src.height);
    let mut dst: Bitmap<Gray8> = Bitmap::new(h, w, 1, false);
    for y in 0..h {
        let row = &src.row_bytes(y)[..w as usize];
        for (x, &luma) in (0..w).zip(row) {
            dst.row_bytes_mut(w - 1 - x)[y as usize] = luma;
        }
    }
    dst
}

/// `src` made exactly `width` pixels wide: centred on white paper when
/// narrower, its middle `width` columns when wider.
fn fit_width(src: &Bitmap<Gray8>, width: u32) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(width, src.height, 1, false);
    let (skip, pad) = if src.width >= width {
        ((src.width - width) / 2, 0)
    } else {
        (0, (width - src.width) / 2)
    };
    let copied = src.width.min(width) as usize;
    for y in 0..src.height {
        let from = &src.row_bytes(y)[skip as usize..skip as usize + copied];
        let to = &mut dst.row_bytes_mut(y)[..width as usize];
        to.fill(255);
        to[pad as usize..pad as usize + copied].copy_from_slice(from);
    }
    dst
}

#[cfg(test)]
mod tests {
    use color::Gray8;
    use raster::Bitmap;

    use super::{FAX_WIDTH, fit_width, layout, parse_number, quarter_turn, submission};

    #[test]
    fn fax_numbers_are_dial_strings_that_cannot_pass_for_options() {
        for (number, dial) in [
            ("+1 (555) 010-9999", "+15550109999"),
            ("0033,,123.45", "0033,,12345"),
            ("*67#5550100", "*67#5550100"),
        ] {
            assert_eq!(parse_number(number).as_deref(), Ok(dial));
        }
        for bad in ["-o", "", " 555", "555; rm", "555x12", "(555) 0100"] {
            assert!(parse_number(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_named_queue_submits_through_lp_else_through_sendfax() {
        let (program, argv) = submission("out.tif", "5550100", Some("Fax"));
        assert_eq!(program, "lp");
        assert_eq!(argv, ["-d", "Fax", "-o", "phone=5550100", "--", "out.tif"]);
        let (program, argv) = submission("out.tif", "5550100", None);
        assert_eq!(program, "sendfax");
        assert_eq!(argv, ["-n", "-d", "5550100", "--", "out.tif"]);
        // A spaced number reaches lp as one option value.
        let dial = parse_number("+1 555 0100").unwrap();
        let (_, argv) = submission("out.tif", &dial, Some("Fax"));
        assert_eq!(argv[3], "phone=+15550100");
    }

    fn bitmap(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Bitmap<Gray8> {
        let mut bmp: Bitmap<Gray8> = Bitmap::new(w, h, 4, false);
        for y in 0..h {
            for (x, px) in (0..w).zip(bmp.row_bytes_mut(y).iter_mut()) {
                *px = f(x, y);
            }
        }
        bmp
    }

    #[test]
    fn a_portrait_page_spans_the_width_and_a_landscape_page_turns() {
        // A4 at fine resolution: 1728 pixels across, 196/204 as many per
        // point down.
        let a4 = layout((595.0, 842.0), 196.0);
        assert!(!a4.turn);
        assert!(a4.scale_x.mul_add(595.0, -f64::from(FAX_WIDTH)).abs() < 1e-9);
        assert!((a4.scale_y / a4.scale_x - 196.0 / 204.0).abs() < 1e-12);
        let landscape = layout((842.0, 595.0), 98.0);
        assert!(landscape.turn);
        assert!(
            landscape
                .scale_y
                .mul_add(595.0, -f64::from(FAX_WIDTH))
                .abs()
                < 1e-9
        );
        assert!((landscape.scale_x / landscape.scale_y - 98.0 / 204.0).abs() < 1e-12);
    }

    #[test]
    fn a_quarter_turn_puts_the_top_edge_on_the_left() {
        // 3 × 2, the top-right pixel marked: it ends top-left.
        let src = bitmap(3, 2, |x, y| u8::from(x == 2 && y == 0));
        let turned = quarter_turn(&src);
        assert_eq!((turned.width, turned.height), (2, 3));
        assert_eq!(&turned.row_bytes(0)[..2], &[1, 0]);
        assert_eq!(&turned.row_bytes(2)[..2], &[0, 0]);
        // The bottom-left pixel ends bottom-right.
        let src = bitmap(3, 2, |x, y| u8::from(x == 0 && y == 1));
        assert_eq!(&quarter_turn(&src).row_bytes(2)[..2], &[0, 1]);
    }

    #[test]
    fn fitting_centres_on_white_or_keeps_the_middle() {
        let narrow = fit_width(&bitmap(2, 1, |_, _| 0), 6);
        assert_eq!(&narrow.row_bytes(0)[..6], &[255, 255, 0, 0, 255, 255]);
        let wide = fit_width(&bitmap(6, 1, |x, _| u8::try_from(x).unwrap()), 2);
        assert_eq!(&wide.row_bytes(0)[..2], &[2, 3]);
    }
}
