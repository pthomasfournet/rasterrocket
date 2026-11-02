//! The `--fax` output: one TIFF Class F document (RFC 2306) holding every
//! rendered page, each [`FAX_WIDTH`] pixels wide at [`FAX_X_DPI`] pixels
//! per inch and 196 (fine) or 98 (standard) lines per inch, bilevel and
//! fax-coded.
//!
//! A page is laid across the fax width by its shorter side: a landscape
//! page is turned a quarter counterclockwise, its top edge to the left.
//! Pages are coded on the worker that rendered them and written in page
//! order once all are done.

use std::sync::Mutex;

use color::Gray8;
use encode::{TiffCompression, TiffImage, TiffPage, TiffWriter};
use pdf_raster::rgb_to_gray;
use raster::Bitmap;

use crate::args::Args;
use crate::cleanup::Page;
use crate::render::{RenderError, write_output};
use crate::tone;

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

/// The pages of the fax document, coded and keyed by their order.
pub(crate) struct FaxDocument {
    compression: TiffCompression,
    dpi: (f64, f64),
    /// `(page number, index within the page, coded page)`: a split spread
    /// yields two pages under one number.
    pages: Mutex<Vec<(i32, usize, TiffPage)>>,
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

    /// Turn `page` when `turn`, fit it to the fax width, reduce it to two
    /// levels and code it as page `index` of document page `page_num`.
    ///
    /// # Errors
    ///
    /// The encoder refused the page.
    pub(crate) fn add(
        &self,
        page_num: i32,
        index: usize,
        page: Page,
        turn: bool,
        args: &Args,
    ) -> Result<(), RenderError> {
        let gray = match page {
            Page::Gray(gray) => gray,
            Page::Rgb(rgb) => rgb_to_gray(&rgb),
        };
        let gray = if turn { quarter_turn(&gray) } else { gray };
        let mono = tone::gray_to_mono(&fit_width(&gray, FAX_WIDTH), args.screen());
        let coded = TiffPage::encode(TiffImage::Bilevel(&mono), self.dpi, self.compression)?;
        self.pages
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((page_num, index, coded));
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
        pages.sort_unstable_by_key(|&(page, index, _)| (page, index));
        let mut writer = TiffWriter::new(self.compression);
        for (_, _, page) in pages {
            writer.push_page(page);
        }
        write_output(path, |out| writer.write(out).map_err(RenderError::from))
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

    use super::{FAX_WIDTH, fit_width, layout, quarter_turn};

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
