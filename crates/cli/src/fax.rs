//! The `--fax` output: one TIFF Class F document (RFC 2306) holding every
//! rendered page, each [`FAX_WIDTH`] pixels wide at [`FAX_X_DPI`] pixels
//! per inch and 196 (fine) or 98 (standard) lines per inch, bilevel and
//! fax-coded.
//!
//! A page is laid across the fax width by its shorter side: a landscape
//! page is turned a quarter counterclockwise, its top edge to the left.
//! A page that then runs longer than [`WHOLE_PAGE_MM`] (ITU-T T.30's B4
//! length) is sent as consecutive fax pages of at most [`PIECE_MM`] (A4
//! length, the length every receiver accepts); A4, Letter, Legal and the
//! other common sizes stay whole.
//! Pages are coded on the worker that rendered them and written in page
//! order once all are done.  The written document can be handed to the
//! system's fax service through its submission command, `lp` or `sendfax`.

use color::Gray8;
use encode::{TiffCompression, TiffImage, TiffPage, TiffWriter};
use raster::Bitmap;

use crate::cleanup::Page;
use crate::deliver;
use crate::render::{PageCollector, RenderError, write_output};

/// The width of a fax page in pixels: ITU-T T.4's 215 mm scan line.
pub(crate) const FAX_WIDTH: u32 = 1728;

/// The horizontal resolution of a fax page, pixels per inch.
pub(crate) const FAX_X_DPI: f64 = 204.0;

/// The longest page sent whole, in millimetres: the B4 length of ITU-T
/// T.30.  An A4 page laid across the 215 mm line is about 304 mm long and a
/// Legal page about 354 mm.
const WHOLE_PAGE_MM: f64 = 364.0;

/// The longest piece a longer page is cut into, in millimetres: the A4
/// length, which T.30 requires every receiver to accept.
const PIECE_MM: f64 = 297.0;

/// The row ranges a fitted page of `height` lines at `y_dpi` lines per inch
/// is sent as: the whole page when it is at most [`WHOLE_PAGE_MM`] long,
/// else consecutive pieces of at most [`PIECE_MM`].
fn pieces(height: u32, y_dpi: f64) -> impl Iterator<Item = std::ops::Range<u32>> {
    let lines = |mm: f64| {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a few thousand lines at fax resolutions; the cast saturates"
        )]
        let n = (mm / 25.4 * y_dpi).floor() as u32;
        n.max(1)
    };
    let piece = if height <= lines(WHOLE_PAGE_MM) {
        height.max(1)
    } else {
        lines(PIECE_MM)
    };
    (0..height.div_ceil(piece)).map(move |i| i * piece..((i + 1) * piece).min(height))
}

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
    /// Keyed by page number and piece index.
    pages: PageCollector<(i32, u32), TiffPage>,
}

impl FaxDocument {
    /// An empty document coded with `compression` at `dpi`.
    pub(crate) const fn new(compression: TiffCompression, dpi: (f64, f64)) -> Self {
        Self {
            compression,
            dpi,
            pages: PageCollector::new(),
        }
    }

    /// Turn `page` when `turn`, fit it to the fax width and code it,
    /// bilevel, as document page `page_num`: one fax page, or several when
    /// it is too long to send whole (see [`pieces`]).  The page is the
    /// two-level luma of [`crate::tone::prepare`]; any luma is thresholded.
    ///
    /// # Errors
    ///
    /// The encoder refused the page.
    pub(crate) fn add(&self, page_num: i32, page: Page, turn: bool) -> Result<(), RenderError> {
        let luma = page.into_luma();
        let luma = if turn { quarter_turn(&luma) } else { luma };
        for (index, rows) in (0u32..).zip(pieces(luma.height, self.dpi.1)) {
            let fitted = fit_width(&luma, FAX_WIDTH, rows);
            let coded = TiffPage::encode(TiffImage::Bilevel(&fitted), self.dpi, self.compression)?;
            self.pages.push((page_num, index), coded);
        }
        Ok(())
    }

    /// Write the document, its pages in order, to `path`.
    ///
    /// # Errors
    ///
    /// No page was added, or the file could not be written.
    pub(crate) fn write(self, path: &str) -> Result<(), RenderError> {
        let mut writer = TiffWriter::fax_document();
        for page in self.pages.take_sorted() {
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
                deliver::lp_args(
                    Some(queue),
                    vec!["-o".to_owned(), format!("phone={number}")],
                    Some(path),
                ),
            )
        },
    )
}

/// Submit the fax document at `path` to the dial string `number` through
/// the system's fax service (see [`submission`]) and wait for the service
/// to accept it.
///
/// # Errors
///
/// See [`deliver::submit`].
pub(crate) fn submit(path: &str, number: &str, queue: Option<&str>) -> Result<(), String> {
    let (program, argv) = submission(path, number, queue);
    deliver::submit(
        program,
        &argv,
        std::process::Stdio::null(),
        "the fax",
        Some(path),
        "name a CUPS fax queue with --fax-queue, or install HylaFAX for sendfax",
    )
    .map_err(|e| e.to_string())
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

/// The `rows` of `src` made exactly `width` pixels wide: centred on white
/// paper when narrower, its middle `width` columns when wider.
fn fit_width(src: &Bitmap<Gray8>, width: u32, rows: std::ops::Range<u32>) -> Bitmap<Gray8> {
    let mut dst: Bitmap<Gray8> = Bitmap::new(width, rows.end - rows.start, 1, false);
    let (skip, pad) = if src.width >= width {
        ((src.width - width) / 2, 0)
    } else {
        (0, (width - src.width) / 2)
    };
    let copied = src.width.min(width) as usize;
    for (y, from_y) in (0..).zip(rows) {
        let from = &src.row_bytes(from_y)[skip as usize..skip as usize + copied];
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

    use super::{FAX_WIDTH, fit_width, layout, parse_number, pieces, quarter_turn, submission};

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
        let narrow = fit_width(&bitmap(2, 1, |_, _| 0), 6, 0..1);
        assert_eq!(&narrow.row_bytes(0)[..6], &[255, 255, 0, 0, 255, 255]);
        let wide = fit_width(&bitmap(6, 1, |x, _| u8::try_from(x).unwrap()), 2, 0..1);
        assert_eq!(&wide.row_bytes(0)[..2], &[2, 3]);
        let tall = fit_width(&bitmap(1, 4, |_, y| u8::try_from(y).unwrap()), 1, 1..3);
        assert_eq!(
            (tall.height, tall.row_bytes(0)[0], tall.row_bytes(1)[0]),
            (2, 1, 2)
        );
    }

    /// The fitted length, in lines, of a `width_pts × height_pts` page.
    fn fitted_lines(width_pts: f64, height_pts: f64, y_dpi: f64) -> u32 {
        let l = layout((width_pts, height_pts), y_dpi);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "page lengths of a few thousand lines"
        )]
        let lines =
            (height_pts.max(width_pts) * if l.turn { l.scale_x } else { l.scale_y }).round() as u32;
        lines
    }

    #[test]
    fn common_page_sizes_stay_whole_and_long_pages_are_cut() {
        for y_dpi in [196.0, 98.0] {
            // A4, US Letter, US Legal, A3, Tabloid; then a landscape A4.
            for (w, h) in [
                (595.0, 842.0),
                (612.0, 792.0),
                (612.0, 1008.0),
                (842.0, 1191.0),
                (792.0, 1224.0),
                (842.0, 595.0),
            ] {
                let lines = fitted_lines(w, h, y_dpi);
                let cut: Vec<_> = pieces(lines, y_dpi).collect();
                assert!(
                    matches!(cut.as_slice(), [whole] if *whole == (0..lines)),
                    "{w}×{h} at {y_dpi}: {cut:?}"
                );
            }
        }
        // A 1 m receipt at fine resolution: 297 mm pieces, the rest last.
        let receipt = fitted_lines(226.8, 2835.0, 196.0);
        let cut: Vec<_> = pieces(receipt, 196.0).collect();
        let piece = 2291;
        assert_eq!(cut.len(), receipt.div_ceil(piece) as usize);
        assert!(cut.iter().all(|r| r.len() <= piece as usize));
        assert_eq!(cut.first().map(|r| r.start), Some(0));
        assert_eq!(cut.last().map(|r| r.end), Some(receipt));
        assert!(cut.windows(2).all(|w| w[0].end == w[1].start));
    }
}
