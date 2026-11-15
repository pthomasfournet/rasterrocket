//! OCR preparation of rendered pages (`--clean-borders`,
//! `--clean-background`) through `rasterrocket-camera`.
//!
//! Each operation is measured on the 8-bit luma of the page (its nearest
//! 8-bit levels at 16 bits) and applied to every channel at the page's
//! depth, so colour and depth are preserved.  Under luma output the page is
//! converted to luma first and the luma alone is processed.  A measurement
//! that declines leaves the page unchanged.

use camera::{Background, BorderMask, CameraError, Detected, GrayPage, Sample};
use color::{Gray8, Pixel};
use pdf_raster::{rgb_to_gray, rgb16_to_gray16};
use raster::Bitmap;

use crate::args::Args;
use crate::cleanup::{Page, gray_bitmap, packed_samples, rgb_bitmap, split_channels, view};
use crate::render::RenderError;

/// `page` with its scanner borders whitened under `--clean-borders`, then
/// its background normalised under `--clean-background`.
pub(crate) fn prepare(page: Page, args: &Args) -> Result<Page, RenderError> {
    if !(args.clean_borders || args.clean_background) {
        return Ok(page);
    }
    let mut page = if args.luma_output() {
        page.into_luma_model()
    } else {
        page
    };
    if args.clean_borders
        && let Detected::Found(mask) = camera::find_border_junk(view(&luma8(&page)))?
    {
        page = map_planes(page, &mask)?;
    }
    if args.clean_background {
        let (width, height) = page_size(&page);
        let tile = camera::background_tile_px(args.ocr_dpi(width, height));
        if let Detected::Found(background) = camera::measure_background(view(&luma8(&page)), tile)?
        {
            page = map_planes(page, &background)?;
        }
    }
    Ok(page)
}

/// An operation measured on luma and applied to one plane of samples.
trait PlaneOp {
    fn apply<S: Sample>(&self, plane: GrayPage<S>) -> Result<GrayPage<S>, CameraError>;
}

impl PlaneOp for BorderMask {
    fn apply<S: Sample>(&self, mut plane: GrayPage<S>) -> Result<GrayPage<S>, CameraError> {
        self.whiten(&mut plane)?;
        Ok(plane)
    }
}

impl PlaneOp for Background {
    fn apply<S: Sample>(&self, plane: GrayPage<S>) -> Result<GrayPage<S>, CameraError> {
        self.normalize(plane.view())
    }
}

/// `page` with `op` applied to its luma plane, or to each colour channel.
fn map_planes(page: Page, op: &impl PlaneOp) -> Result<Page, RenderError> {
    Ok(match page {
        Page::Gray(luma) => Page::Gray(map_gray(&luma, op)?),
        Page::Mono(luma) => Page::Mono(map_gray(&luma, op)?),
        Page::Gray16(luma) => Page::Gray16(map_gray(&luma, op)?),
        Page::Rgb(rgb) => Page::Rgb(map_rgb(&rgb, op)?),
        Page::Rgb16(rgb) => Page::Rgb16(map_rgb(&rgb, op)?),
    })
}

fn map_gray<G: Pixel>(luma: &Bitmap<G>, op: &impl PlaneOp) -> Result<Bitmap<G>, RenderError>
where
    G::Sample: Sample,
{
    gray_bitmap(&op.apply(plane(luma, packed_samples(luma)))?)
}

fn map_rgb<C: Pixel>(rgb: &Bitmap<C>, op: &impl PlaneOp) -> Result<Bitmap<C>, RenderError>
where
    C::Sample: Sample,
{
    let [r, g, b] = split_channels(rgb).map(|samples| plane(rgb, samples));
    rgb_bitmap(&op.apply(r)?, &op.apply(g)?, &op.apply(b)?)
}

/// `samples`, one plane of `bmp`, as a packed page of its size.
const fn plane<P: Pixel, S: Sample>(bmp: &Bitmap<P>, samples: Vec<S>) -> GrayPage<S> {
    GrayPage {
        width: bmp.width as usize,
        height: bmp.height as usize,
        data: samples,
    }
}

/// The 8-bit luma the operations are measured on.
fn luma8(page: &Page) -> Bitmap<Gray8> {
    match page {
        Page::Gray(luma) | Page::Mono(luma) => luma.converted(),
        Page::Gray16(luma) => luma.converted(),
        Page::Rgb(rgb) => rgb_to_gray(rgb),
        Page::Rgb16(rgb) => rgb16_to_gray16(rgb).converted(),
    }
}

const fn page_size(page: &Page) -> (u32, u32) {
    match page {
        Page::Gray(b) | Page::Mono(b) => (b.width, b.height),
        Page::Gray16(b) => (b.width, b.height),
        Page::Rgb(b) => (b.width, b.height),
        Page::Rgb16(b) => (b.width, b.height),
    }
}
