//! Cleanup of camera-captured pages: page-boundary crop with perspective
//! rectification, page-curvature dewarp, and double-page spread splitting.
//!
//! # Pipeline
//!
//! ```text
//! rectify → split → per page: dewarp → deskew (RasterOptions::deskew)
//! ```
//!
//! [`process`] runs the camera stages; the
//! [`RasterOptions`](crate::RasterOptions) render path then deskews each
//! resulting page.  Dewarp and deskew run per page, after the split,
//! because both fit one model per page — one surface, one skew angle —
//! and the two pages of a spread are independent sheets: their lines are
//! not aligned and curve in opposite directions toward the gutter.
//!
//! Each stage is a detector followed by a resampler.  A detector that does
//! not reach its acceptance criteria reports nothing and the stage passes
//! the page through unchanged, so a flatbed scan or a born-digital page is
//! returned as rendered.
//!
//! - **Rectify** ([`detect_page_quad`], [`rectify`]): the page boundary is
//!   found as a light quadrilateral on a darker surround and mapped onto an
//!   upright rectangle by a homography.  Everything outside the page
//!   (desk, stand, fingers) is discarded.  The output aspect ratio is the
//!   physical one, recovered from the vanishing geometry of the quad.
//! - **Dewarp** ([`dewarp`]): text lines are tracked across the page and
//!   fitted to a smooth surface model of a page bent about a vertical axis;
//!   each column is resampled so the lines become straight.  Experimental:
//!   vertical displacement only (glyphs foreshortened near a spine stay
//!   narrow), and a page needs at least five tracked text lines.
//! - **Split** ([`find_gutter`], [`split_spread`]): an image wider than tall
//!   whose centre shows a gutter shadow or two facing inner margins is cut
//!   into its left and right pages.
//!
//! All detection runs on luma; resampling preserves the pixel format.
//! Grayscale ([`Gray8`]) and RGB ([`Rgb8`]) bitmaps are supported.

mod dewarp;
mod homography;
mod linalg;
mod plane;
mod quad;
mod spread;
#[cfg(test)]
mod testutil;
mod textlines;

#[cfg(doc)]
use color::{Gray8, Rgb8};
use raster::Bitmap;

pub use plane::CameraPixel;

/// Which camera-capture cleanup stages to run.  Every stage is off by
/// default.
///
/// `#[non_exhaustive]`: construct with [`CameraOptions::default`] or
/// [`CameraOptions::all`] and set fields individually.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CameraOptions {
    /// Detect the page boundary, correct perspective and crop the
    /// surround ([`rectify`]).
    pub rectify: bool,
    /// Straighten curved text lines ([`dewarp`]).  Experimental.
    pub dewarp: bool,
    /// Split a double-page spread into two pages at the gutter
    /// ([`split_spread`]).
    pub split_spreads: bool,
}

impl CameraOptions {
    /// Every stage enabled.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            rectify: true,
            dewarp: true,
            split_spreads: true,
        }
    }

    /// Whether any stage is enabled.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.rectify || self.dewarp || self.split_spreads
    }
}

/// Which page of a split double-page spread a bitmap holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpreadHalf {
    /// The left page of the spread (columns left of the gutter); emitted
    /// first.
    Left,
    /// The right page of the spread (columns from the gutter on).
    Right,
}

/// A page quadrilateral in source pixel coordinates.
///
/// Corners are ordered top-left, top-right, bottom-right, bottom-left and
/// lie on the page edge (continuous coordinates; pixel `(x, y)` covers
/// `[x − 0.5, x + 0.5] × [y − 0.5, y + 0.5]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// `[x, y]` of the top-left, top-right, bottom-right and bottom-left
    /// corners.
    pub corners: [[f64; 2]; 4],
}

// ── Stages ──────────────────────────────────────────────────────────────────

/// Detect the boundary of a light page on a darker surround.
///
/// `None` when no page quad meets the acceptance criteria — in particular
/// when the page fills (nearly) the whole frame, as on a flatbed scan.
#[must_use]
pub fn detect_page_quad<P: CameraPixel>(img: &Bitmap<P>) -> Option<Quad> {
    quad::detect(&P::luma_plane(img))
}

/// Map the page quad of `img` onto an upright rectangle, discarding the
/// surround.  `None` (leave the image as is) when [`detect_page_quad`]
/// finds no page.
#[must_use]
pub fn rectify<P: CameraPixel>(img: &Bitmap<P>) -> Option<Bitmap<P>> {
    let q = detect_page_quad(img)?;
    warp_quad(img, &q)
}

/// Resample the region of `img` inside `quad` onto an upright rectangle
/// sized by [`homography::rectified_size`].
fn warp_quad<P: CameraPixel>(img: &Bitmap<P>, quad: &Quad) -> Option<Bitmap<P>> {
    let (w, h) = homography::rectified_size(quad, img.width as usize, img.height as usize)?;
    let (fw, fh) = (f64::from(w), f64::from(h));
    let rect = [
        [-0.5, -0.5],
        [fw - 0.5, -0.5],
        [fw - 0.5, fh - 0.5],
        [-0.5, fh - 0.5],
    ];
    let hmg = homography::Homography::from_points(&rect, &quad.corners)?;
    log::debug!("camera: rectified page quad {:?} to {w}×{h}", quad.corners);
    Some(plane::resample(img, w, h, |x, y| hmg.apply(x, y)))
}

/// Straighten curved text lines.  Experimental; see the module docs.
///
/// `None` (leave the image as is) when the page has too few text lines,
/// its lines are already straight, or the surface fit is not accepted.
#[must_use]
pub fn dewarp<P: CameraPixel>(img: &Bitmap<P>) -> Option<Bitmap<P>> {
    let luma = P::luma_plane(img);
    let lines = textlines::find(&luma)?;
    let surface = dewarp::fit(&lines, luma.w, luma.h)?;
    Some(plane::resample(img, img.width, img.height, |x, y| {
        (x, surface.source_y(x, y))
    }))
}

/// The column at which `img` splits into two pages: the left page is
/// columns `[0, x)`.  `None` when `img` is not recognisably a spread.
#[must_use]
pub fn find_gutter<P: CameraPixel>(img: &Bitmap<P>) -> Option<u32> {
    let x = spread::find_gutter(&P::luma_plane(img))?;
    let x = u32::try_from(x).ok()?;
    (x > 0 && x < img.width).then_some(x)
}

/// Split a double-page spread at its gutter into `[left, right]`.
/// `None` when [`find_gutter`] finds no gutter.
#[must_use]
pub fn split_spread<P: CameraPixel>(img: &Bitmap<P>) -> Option<[Bitmap<P>; 2]> {
    let x = find_gutter(img)?;
    Some([
        plane::crop_columns(img, 0, x),
        plane::crop_columns(img, x, img.width),
    ])
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/// Run the enabled camera stages on one captured image: [`rectify`], then
/// [`split_spread`], then [`dewarp`] on each resulting page.  Stages whose
/// detector declines leave the image unchanged.
///
/// Returns `(page, None)` for an unsplit page, or `(left, Some(right))`
/// for a split spread ([`SpreadHalf::Left`], [`SpreadHalf::Right`]).
#[must_use]
pub fn process<P: CameraPixel>(
    img: Bitmap<P>,
    opts: &CameraOptions,
) -> (Bitmap<P>, Option<Bitmap<P>>) {
    let img = if opts.rectify {
        rectify(&img).unwrap_or(img)
    } else {
        img
    };
    let (first, second) = match opts.split_spreads.then(|| split_spread(&img)).flatten() {
        Some([left, right]) => (left, Some(right)),
        None => (img, None),
    };
    if opts.dewarp {
        let flat = |page: Bitmap<P>| dewarp(&page).unwrap_or(page);
        (flat(first), second.map(flat))
    } else {
        (first, second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use color::{Gray8, Rgb8};

    fn rgb_page_on_desk(w: u32, h: u32, page: (u32, u32, u32, u32)) -> Bitmap<Rgb8> {
        let mut bmp = Bitmap::<Rgb8>::new(w, h, 1, false);
        for y in 0..h {
            for (x, px) in bmp.row_mut(y).iter_mut().enumerate() {
                let x = u32::try_from(x).unwrap();
                let on_page = (page.0..page.2).contains(&x) && (page.1..page.3).contains(&y);
                *px = if !on_page {
                    Rgb8 {
                        r: 120,
                        g: 70,
                        b: 40,
                    }
                } else if (y / 8) % 3 == 0
                    && (x / 19) % 5 != 4
                    && (page.0 + 40..page.2 - 40).contains(&x)
                    && (page.1 + 40..page.3 - 40).contains(&y)
                {
                    Rgb8 {
                        r: 20,
                        g: 20,
                        b: 30,
                    }
                } else {
                    Rgb8 {
                        r: 245,
                        g: 240,
                        b: 225,
                    }
                };
            }
        }
        bmp
    }

    #[test]
    fn rgb_rectify_crops_to_page_and_keeps_colour() {
        let img = rgb_page_on_desk(900, 1200, (100, 150, 800, 1100));
        let out = rectify(&img).expect("page found");
        assert!(
            (i64::from(out.width) - 700).abs() <= 2,
            "width {}",
            out.width
        );
        assert!(
            (i64::from(out.height) - 950).abs() <= 2,
            "height {}",
            out.height
        );
        let c = out.row(out.height / 2 + 2)[3];
        assert_eq!(
            c,
            Rgb8 {
                r: 245,
                g: 240,
                b: 225
            },
            "paper colour preserved"
        );
    }

    #[test]
    fn disabled_options_are_identity() {
        let img = rgb_page_on_desk(300, 200, (20, 20, 280, 180));
        let before = img.data().to_vec();
        let (page, second) = process(img, &CameraOptions::default());
        assert!(second.is_none());
        assert_eq!(page.data(), &before[..]);
    }

    #[test]
    fn blank_page_passes_every_stage() {
        let mut img = Bitmap::<Gray8>::new(400, 300, 1, false);
        img.data_mut().fill(240);
        let (page, second) = process(img, &CameraOptions::all());
        assert!(second.is_none());
        assert!(page.data().iter().all(|&v| v == 240));
    }
}
