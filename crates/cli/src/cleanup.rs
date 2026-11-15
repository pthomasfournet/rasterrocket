//! Camera-capture cleanup of rendered pages (`--crop`, `--rectify`,
//! `--split-spreads`, `--dewarp`) through `rasterrocket-camera`, and
//! deskew (`--deskew`) through `rasterrocket`.
//!
//! The stages measure on the 8-bit luma of the rendered page: the page's
//! own at 8 bits, its nearest 8-bit levels at 16.  Grayscale output of an
//! 8-bit page is the processed luma; every other output replays the
//! computed geometry ([`camera::Report::replay`]) on each colour channel,
//! or on the 16-bit luma, at the page's depth, so colour and depth are
//! preserved and every channel is resampled identically.

use camera::{CameraOptions, GrayPage, GrayView, Outcome, Report};
use color::{Gray8, Gray16, Pixel, Rgb8, Rgb16};
use pdf_raster::{
    BackendPolicy, RasterError, rgb_into_gray, rgb_to_gray, rgb16_into_gray16, rgb16_to_gray16,
};
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

/// One page in the colour model and depth it is written from.
pub(crate) enum Page {
    /// Colour pixels.
    Rgb(Bitmap<Rgb8>),
    /// Luma pixels.
    Gray(Bitmap<Gray8>),
    /// Luma pixels of two levels, 0 and 255: bilevel output.
    Mono(Bitmap<Gray8>),
    /// Colour pixels of 16-bit samples.
    Rgb16(Bitmap<Rgb16>),
    /// Luma pixels of 16-bit samples.
    Gray16(Bitmap<Gray16>),
}

impl Page {
    /// The page's 8-bit luma: converted from colour, the page itself, or
    /// the nearest 8-bit levels of a 16-bit page.
    pub(crate) fn into_luma(self) -> Bitmap<Gray8> {
        match self.into_luma_model() {
            Self::Gray16(luma) => luma.converted(),
            Self::Gray(luma) | Self::Mono(luma) => luma,
            Self::Rgb(rgb) => rgb_into_gray(rgb),
            Self::Rgb16(rgb) => rgb16_into_gray16(rgb).converted(),
        }
    }

    /// The page as luma at its depth: converted from colour, or the page
    /// itself.
    pub(crate) fn into_luma_model(self) -> Self {
        match self {
            Self::Rgb(rgb) => Self::Gray(rgb_into_gray(rgb)),
            Self::Mono(luma) => Self::Gray(luma),
            Self::Rgb16(rgb) => Self::Gray16(rgb16_into_gray16(rgb)),
            page @ (Self::Gray(_) | Self::Gray16(_)) => page,
        }
    }

    /// The page's rows for a row-streamed writer: bilevel for
    /// [`Page::Mono`].
    pub(crate) const fn row_image(&self) -> encode::RowImage<'_> {
        match self {
            Self::Rgb(rgb) => encode::RowImage::Rgb(rgb),
            Self::Gray(luma) => encode::RowImage::Gray(luma),
            Self::Mono(luma) => encode::RowImage::Bilevel(luma),
            Self::Rgb16(rgb) => encode::RowImage::Rgb16(rgb),
            Self::Gray16(luma) => encode::RowImage::Gray16(luma),
        }
    }

    /// The rows of a 16-bit page; `None` for an 8-bit one.
    pub(crate) const fn wide_image(&self) -> Option<encode::RowImage<'_>> {
        match self {
            Self::Rgb16(rgb) => Some(encode::RowImage::Rgb16(rgb)),
            Self::Gray16(luma) => Some(encode::RowImage::Gray16(luma)),
            Self::Rgb(_) | Self::Gray(_) | Self::Mono(_) => None,
        }
    }

    /// The page in colour at its depth: a luma page with its value in
    /// every channel.
    pub(crate) fn into_rgb_model(self) -> Self {
        match self {
            Self::Gray(luma) | Self::Mono(luma) => Self::Rgb(spread_luma(&luma)),
            Self::Gray16(luma) => Self::Rgb16(spread_luma(&luma)),
            page @ (Self::Rgb(_) | Self::Rgb16(_)) => page,
        }
    }
}

/// The colour bitmap of `luma`: its value in every channel.
fn spread_luma<G: Pixel, C: Pixel<Sample = G::Sample>>(luma: &Bitmap<G>) -> Bitmap<C> {
    let w = luma.width as usize;
    let mut rgb = Bitmap::<C>::new(luma.width, luma.height, 1, false);
    for y in 0..luma.height {
        let src = &luma.row_samples(y)[..w];
        let dst = rgb.row_samples_mut(y)[..3 * w].as_chunks_mut::<3>().0;
        for (px, &l) in dst.iter_mut().zip(src) {
            *px = [l; 3];
        }
    }
    rgb
}

/// Run the stages in `opts` on `page`.  Returns one page, or the left then
/// right page of a split spread.  When no stage changes the page, it is
/// returned as is.  A luma page, or any page when `luma_output`, yields
/// luma pages; a colour page otherwise yields colour pages, every channel
/// resampled through the geometry computed on its luma.  The pages keep
/// the depth of `page`.
pub(crate) fn clean(
    page: Page,
    opts: CameraOptions,
    luma_output: bool,
) -> Result<Vec<(Page, Option<SpreadHalf>)>, RenderError> {
    if !opts.any() {
        return Ok(vec![(page, None)]);
    }
    let pages = match page {
        Page::Rgb(rgb) if !luma_output => {
            let luma = rgb_to_gray(&rgb);
            let processed = camera::process(view(&luma), &opts)?;
            if !changed(&processed.report) {
                return Ok(vec![(Page::Rgb(rgb), None)]);
            }
            replay_colour(&processed.report, &rgb)?
                .into_iter()
                .map(Page::Rgb)
                .collect()
        }
        Page::Rgb16(rgb) if !luma_output => {
            let processed = camera::process(view(&rgb16_to_gray16(&rgb).converted()), &opts)?;
            if !changed(&processed.report) {
                return Ok(vec![(Page::Rgb16(rgb), None)]);
            }
            replay_colour(&processed.report, &rgb)?
                .into_iter()
                .map(Page::Rgb16)
                .collect()
        }
        page => match page.into_luma_model() {
            Page::Gray16(luma) => {
                let processed = camera::process(view(&luma.converted()), &opts)?;
                if !changed(&processed.report) {
                    return Ok(vec![(Page::Gray16(luma), None)]);
                }
                let plane = packed_samples(&luma);
                processed
                    .report
                    .replay(GrayView::packed(
                        &plane,
                        luma.width as usize,
                        luma.height as usize,
                    ))?
                    .iter()
                    .map(|p| gray_bitmap(p).map(Page::Gray16))
                    .collect::<Result<_, _>>()?
            }
            page => {
                let luma = page.into_luma();
                let processed = camera::process(view(&luma), &opts)?;
                if !changed(&processed.report) {
                    return Ok(vec![(Page::Gray(luma), None)]);
                }
                processed
                    .pages
                    .iter()
                    .map(|p| gray_bitmap(p).map(Page::Gray))
                    .collect::<Result<_, _>>()?
            }
        },
    };
    Ok(halves(pages))
}

/// `page` as luma at its depth, deskewed under `policy`
/// ([`pdf_raster::deskew::apply`]): rotated to correct the skew of its
/// lines, or unchanged when it has no confident angle.
pub(crate) fn deskewed(page: Page, policy: BackendPolicy) -> Result<Page, RenderError> {
    let deskew_error = |e: pdf_raster::deskew::DeskewError| RasterError::Deskew(e.to_string());
    Ok(match page.into_luma_model() {
        Page::Gray(mut luma) => {
            let _skew = pdf_raster::deskew::apply(&mut luma, policy).map_err(deskew_error)?;
            Page::Gray(luma)
        }
        Page::Gray16(mut luma) => {
            let _skew = pdf_raster::deskew::apply(&mut luma, policy).map_err(deskew_error)?;
            Page::Gray16(luma)
        }
        // `into_luma_model` yields gray pages only.
        page @ (Page::Rgb(_) | Page::Rgb16(_) | Page::Mono(_)) => page,
    })
}

/// `pages` with the half of a split spread each holds: none for one page.
fn halves(pages: Vec<Page>) -> Vec<(Page, Option<SpreadHalf>)> {
    let halves = if pages.len() == 2 {
        [Some(SpreadHalf::Left), Some(SpreadHalf::Right)]
    } else {
        [None, None]
    };
    pages.into_iter().zip(halves).collect()
}

/// The pages of `rgb` through the geometry of `report`, each channel
/// replayed at the depth of its samples.
fn replay_colour<C: Pixel>(report: &Report, rgb: &Bitmap<C>) -> Result<Vec<Bitmap<C>>, RenderError>
where
    C::Sample: camera::Sample,
{
    let size = (rgb.width as usize, rgb.height as usize);
    let [r, g, b] = split_channels(rgb);
    let replay = |plane: &[C::Sample]| report.replay(GrayView::packed(plane, size.0, size.1));
    let (r, g, b) = (replay(&r)?, replay(&g)?, replay(&b)?);
    r.iter()
        .zip(&g)
        .zip(&b)
        .map(|((r, g), b)| rgb_bitmap(r, g, b))
        .collect()
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

/// The samples of `luma`'s rows, packed.
fn packed_samples<G: Pixel>(luma: &Bitmap<G>) -> Vec<G::Sample> {
    let w = luma.width as usize * G::NCOMPS;
    let mut samples = Vec::with_capacity(w * luma.height as usize);
    for y in 0..luma.height {
        samples.extend_from_slice(&luma.row_samples(y)[..w]);
    }
    samples
}

/// The R, G and B planes of `rgb`, each tightly packed.
fn split_channels<C: Pixel>(rgb: &Bitmap<C>) -> [Vec<C::Sample>; 3] {
    let w = rgb.width as usize;
    let n = w * rgb.height as usize;
    let mut planes = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for y in 0..rgb.height {
        for px in rgb.row_samples(y)[..3 * w].as_chunks::<3>().0 {
            for (plane, &v) in planes.iter_mut().zip(px) {
                plane.push(v);
            }
        }
    }
    planes
}

/// The dimensions of `page` as bitmap dimensions.
fn dims<S: camera::Sample>(page: &GrayPage<S>) -> Result<(u32, u32), RenderError> {
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

fn gray_bitmap<G: Pixel>(page: &GrayPage<G::Sample>) -> Result<Bitmap<G>, RenderError>
where
    G::Sample: camera::Sample,
{
    let (w, h) = dims(page)?;
    let mut bmp = Bitmap::<G>::new(w, h, 1, false);
    for (y, src) in (0..h).zip(page.data.chunks_exact(page.width)) {
        bmp.row_samples_mut(y)[..page.width].copy_from_slice(src);
    }
    Ok(bmp)
}

/// Interleave three same-sized planes into an RGB bitmap.
fn rgb_bitmap<C: Pixel>(
    red: &GrayPage<C::Sample>,
    green: &GrayPage<C::Sample>,
    blue: &GrayPage<C::Sample>,
) -> Result<Bitmap<C>, RenderError>
where
    C::Sample: camera::Sample,
{
    let (width, height) = dims(red)?;
    let mut bmp = Bitmap::<C>::new(width, height, 1, false);
    let w = red.width;
    let rows = red
        .data
        .chunks_exact(w)
        .zip(green.data.chunks_exact(w))
        .zip(blue.data.chunks_exact(w));
    for (y, ((rr, gr), br)) in (0..height).zip(rows) {
        let dst = bmp.row_samples_mut(y)[..3 * w].as_chunks_mut::<3>().0;
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
        let pages = clean(Page::Rgb(page_on_desk(800, 1000)), opts, false).expect("clean");
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
        let pages = clean(Page::Rgb(rgb), opts, true).expect("clean");
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
        let pages = clean(Page::Rgb(flat), CameraOptions::all(), false).expect("clean");
        let (Page::Rgb(bmp), None) = &pages[0] else {
            panic!("one RGB page");
        };
        assert!(bmp.data().iter().all(|&v| v == 240));
    }

    #[test]
    fn sixteen_bit_pages_are_cleaned_at_their_depth() {
        let mut opts = CameraOptions::default();
        opts.crop = true;
        let eight = page_on_desk(800, 1000);
        let wide: Bitmap<Rgb16> = eight.converted();
        let pages8 = clean(Page::Rgb(eight), opts, false).expect("clean");
        let pages16 = clean(Page::Rgb16(wide.converted()), opts, false).expect("clean");
        let ((Page::Rgb(want), None), (Page::Rgb16(got), None)) = (&pages8[0], &pages16[0]) else {
            panic!("one RGB page at each depth");
        };
        assert_eq!((got.width, got.height), (want.width, want.height));
        let narrowed: Bitmap<Rgb8> = got.converted();
        assert_eq!(
            narrowed.data(),
            want.data(),
            "a crop moves the samples as they are"
        );
        let luma = clean(Page::Rgb16(wide.converted()), opts, true).expect("clean");
        let (Page::Gray16(luma), None) = &luma[0] else {
            panic!("one 16-bit gray page");
        };
        assert_eq!((luma.width, luma.height), (want.width, want.height));
        let flat = Bitmap::<Gray16>::new_filled(300, 200, 1, false, 61_000);
        let pages = clean(Page::Gray16(flat), CameraOptions::all(), false).expect("clean");
        let (Page::Gray16(same), None) = &pages[0] else {
            panic!("the unchanged 16-bit page");
        };
        assert!(same.data().iter().all(|&v| v == 61_000));
    }

    /// FNV-1a 64 of `bytes`, chained from `h`.
    fn fold(h: u64, bytes: &[u8]) -> u64 {
        bytes
            .iter()
            .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    }

    /// The 8-bit pages of fixed inputs through the stages and the tone
    /// operations, digested before the pages gained a 16-bit model.
    #[test]
    fn eight_bit_pages_match_their_golden_digest() {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut opts = CameraOptions::default();
        opts.crop = true;
        opts.rectify = true;
        for luma_output in [false, true] {
            for (page, _) in clean(Page::Rgb(page_on_desk(800, 1000)), opts, luma_output).unwrap() {
                for argv in [&[][..], &["--gray"], &["--invert"], &["--mono"]] {
                    let copy = match &page {
                        Page::Rgb(rgb) => Page::Rgb(rgb_copy(rgb)),
                        Page::Gray(g) | Page::Mono(g) => Page::Gray(gray_copy(g)),
                        _ => unreachable!("8-bit pages"),
                    };
                    let out = crate::tone::prepare(copy, &crate::args::args_with(argv));
                    let bytes: Vec<u8> = match &out {
                        Page::Rgb(rgb) => (0..rgb.height)
                            .flat_map(|y| rgb.row_bytes(y)[..rgb.width as usize * 3].to_vec())
                            .collect(),
                        Page::Gray(g) | Page::Mono(g) => (0..g.height)
                            .flat_map(|y| g.row_bytes(y)[..g.width as usize].to_vec())
                            .collect(),
                        _ => unreachable!("8-bit pages"),
                    };
                    h = fold(h, &bytes);
                }
            }
        }
        assert_eq!(h, GOLDEN, "{h:#x}");
    }

    const GOLDEN: u64 = 0x4362_d67f_748d_fd85;

    fn rgb_copy(rgb: &Bitmap<Rgb8>) -> Bitmap<Rgb8> {
        let mut copy = Bitmap::<Rgb8>::new(rgb.width, rgb.height, 1, false);
        for y in 0..rgb.height {
            copy.row_bytes_mut(y)[..rgb.width as usize * 3]
                .copy_from_slice(&rgb.row_bytes(y)[..rgb.width as usize * 3]);
        }
        copy
    }

    fn gray_copy(g: &Bitmap<Gray8>) -> Bitmap<Gray8> {
        let mut copy = Bitmap::<Gray8>::new(g.width, g.height, 1, false);
        for y in 0..g.height {
            copy.row_bytes_mut(y)[..g.width as usize]
                .copy_from_slice(&g.row_bytes(y)[..g.width as usize]);
        }
        copy
    }
}
