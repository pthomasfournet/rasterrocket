//! Camera-capture cleanup of rendered pages (`--crop`, `--rectify`,
//! `--split-spreads`, `--dewarp`) through `rasterrocket-camera`.
//!
//! The stages run on the luma of the rendered page.  Grayscale and
//! monochrome output is written from the processed luma directly; RGB
//! output replays the computed geometry on each colour channel
//! ([`camera::Report::replay`]), so colour is preserved and every channel
//! is resampled identically.

#![expect(
    clippy::redundant_pub_crate,
    reason = "bin-only crate with private modules: pub(crate) is the real \
              visibility and cannot escape; widening to pub would misdescribe it"
)]

use camera::{CameraOptions, GrayPage, GrayView, Outcome, Report};
use color::{Gray8, Rgb8};
use pdf_raster::rgb_to_gray;
use raster::Bitmap;

use crate::render::RenderError;

/// Which page of a split double-page spread an output holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpreadHalf {
    /// Columns left of the gutter; written first.
    Left,
    /// Columns from the gutter on.
    Right,
}

/// One output page in the colour model it is written from.
pub(crate) enum Page {
    /// Colour pixels; converted to gray at encode time for `--gray` /
    /// `--mono`.
    Rgb(Bitmap<Rgb8>),
    /// Luma pixels, written only for `--gray` / `--mono`.
    Gray(Bitmap<Gray8>),
}

/// Run the stages in `opts` on `rgb`.  Returns one page, or the left then
/// right page of a split spread.  When no stage changes the page, `rgb` is
/// returned as is.  `gray_output` selects luma output pages.
pub(crate) fn clean(
    rgb: Bitmap<Rgb8>,
    opts: CameraOptions,
    gray_output: bool,
) -> Result<Vec<(Page, Option<SpreadHalf>)>, RenderError> {
    if !opts.any() {
        return Ok(vec![(Page::Rgb(rgb), None)]);
    }
    let luma = rgb_to_gray(&rgb);
    let processed = camera::process(view(&luma), &opts)?;
    if !changed(&processed.report) {
        return Ok(vec![(Page::Rgb(rgb), None)]);
    }
    let pages: Vec<Page> = if gray_output {
        processed
            .pages
            .iter()
            .map(|p| gray_bitmap(p).map(Page::Gray))
            .collect::<Result<_, _>>()?
    } else {
        let [r, g, b] = split_channels(&rgb);
        let replay = |plane: &[u8]| {
            processed.report.replay(GrayView::packed(
                plane,
                luma.width as usize,
                luma.height as usize,
            ))
        };
        let (r, g, b) = (replay(&r)?, replay(&g)?, replay(&b)?);
        r.iter()
            .zip(&g)
            .zip(&b)
            .map(|((r, g), b)| rgb_bitmap(r, g, b).map(Page::Rgb))
            .collect::<Result<_, _>>()?
    };
    let halves = if pages.len() == 2 {
        [Some(SpreadHalf::Left), Some(SpreadHalf::Right)]
    } else {
        [None, None]
    };
    Ok(pages.into_iter().zip(halves).collect())
}

/// Whether any stage changed the image.
fn changed(report: &Report) -> bool {
    report.crop.applied()
        || report.rectify.applied()
        || report.split.applied()
        || report.dewarp.iter().any(Outcome::applied)
}

fn view(bmp: &Bitmap<Gray8>) -> GrayView<'_> {
    GrayView {
        data: bmp.data(),
        width: bmp.width as usize,
        height: bmp.height as usize,
        stride: bmp.stride,
    }
}

/// The R, G and B planes of `rgb`, each tightly packed.
fn split_channels(rgb: &Bitmap<Rgb8>) -> [Vec<u8>; 3] {
    let w = rgb.width as usize;
    let n = w * rgb.height as usize;
    let mut planes = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for y in 0..rgb.height {
        for px in rgb.row_bytes(y)[..3 * w].as_chunks::<3>().0 {
            for (plane, &v) in planes.iter_mut().zip(px) {
                plane.push(v);
            }
        }
    }
    planes
}

/// The dimensions of `page` as bitmap dimensions.
fn dims(page: &GrayPage) -> Result<(u32, u32), RenderError> {
    let too_large = || {
        RenderError::Camera(format!(
            "cleaned page is {}×{}, beyond the u32 bitmap limit",
            page.width, page.height
        ))
    };
    Ok((
        u32::try_from(page.width).map_err(|_| too_large())?,
        u32::try_from(page.height).map_err(|_| too_large())?,
    ))
}

fn gray_bitmap(page: &GrayPage) -> Result<Bitmap<Gray8>, RenderError> {
    let (w, h) = dims(page)?;
    let mut bmp = Bitmap::<Gray8>::new(w, h, 1, false);
    for (y, src) in (0..h).zip(page.data.chunks_exact(page.width)) {
        bmp.row_bytes_mut(y)[..page.width].copy_from_slice(src);
    }
    Ok(bmp)
}

/// Interleave three same-sized planes into an RGB bitmap.
fn rgb_bitmap(
    red: &GrayPage,
    green: &GrayPage,
    blue: &GrayPage,
) -> Result<Bitmap<Rgb8>, RenderError> {
    let (width, height) = dims(red)?;
    let mut bmp = Bitmap::<Rgb8>::new(width, height, 1, false);
    let w = red.width;
    let rows = red
        .data
        .chunks_exact(w)
        .zip(green.data.chunks_exact(w))
        .zip(blue.data.chunks_exact(w));
    for (y, ((rr, gr), br)) in (0..height).zip(rows) {
        let dst = bmp.row_bytes_mut(y)[..3 * w].as_chunks_mut::<3>().0;
        for (((px, &r), &g), &b) in dst.iter_mut().zip(rr).zip(gr).zip(br) {
            *px = [r, g, b];
        }
    }
    Ok(bmp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A light page with dark bars on a brown desk, as RGB.
    fn page_on_desk(w: u32, h: u32) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::<Rgb8>::new(w, h, 1, false);
        for y in 0..h {
            let row = bmp.row_bytes_mut(y);
            for (x, px) in (0..w).zip(row.as_chunks_mut::<3>().0) {
                let on_page = (w / 8..w - w / 8).contains(&x) && (h / 8..h - h / 8).contains(&y);
                let ink = (y / 8) % 3 == 0
                    && (x / 19) % 5 != 4
                    && (w / 4..w - w / 4).contains(&x)
                    && (h / 4..h - h / 4).contains(&y);
                *px = if !on_page {
                    [120, 70, 40]
                } else if ink {
                    [20, 20, 30]
                } else {
                    [245, 240, 225]
                };
            }
        }
        bmp
    }

    #[test]
    fn rgb_crop_keeps_colour() {
        let mut opts = CameraOptions::default();
        opts.crop = true;
        let pages = clean(page_on_desk(800, 1000), opts, false).expect("clean");
        assert_eq!(pages.len(), 1);
        let (Page::Rgb(bmp), None) = &pages[0] else {
            panic!("one RGB page");
        };
        assert!(bmp.width.abs_diff(600) <= 2, "width {}", bmp.width);
        assert!(bmp.height.abs_diff(750) <= 2, "height {}", bmp.height);
        assert_eq!(&bmp.row_bytes(bmp.height / 2 + 2)[..3], &[245, 240, 225]);
    }

    #[test]
    fn gray_output_matches_luma_processing() {
        let mut opts = CameraOptions::default();
        opts.rectify = true;
        let rgb = page_on_desk(800, 1000);
        let luma = rgb_to_gray(&rgb);
        let want = camera::process(view(&luma), &opts).expect("process");
        let pages = clean(rgb, opts, true).expect("clean");
        let (Page::Gray(bmp), None) = &pages[0] else {
            panic!("one gray page");
        };
        let got = &want.pages[0];
        assert_eq!(
            (bmp.width as usize, bmp.height as usize),
            (got.width, got.height)
        );
        assert_eq!(bmp.data(), &got.data[..]);
    }

    #[test]
    fn unchanged_page_is_returned_as_is() {
        let mut flat = Bitmap::<Rgb8>::new(300, 200, 1, false);
        flat.data_mut().fill(240);
        let pages = clean(flat, CameraOptions::all(), false).expect("clean");
        let (Page::Rgb(bmp), None) = &pages[0] else {
            panic!("one RGB page");
        };
        assert!(bmp.data().iter().all(|&v| v == 240));
    }
}
