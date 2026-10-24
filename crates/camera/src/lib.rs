//! Cleanup of camera-captured pages on 8-bit grayscale images: page crop,
//! perspective rectification, double-page spread splitting and page
//! curvature dewarp.
//!
//! # Pipeline
//!
//! ```text
//! crop or rectify → split → per page: dewarp
//! ```
//!
//! [`process`] runs the stages enabled in [`CameraOptions`] and returns the
//! resulting pages with a [`Report`] of what each stage did.  Dewarp runs
//! per page, after the split, because it fits one surface per page and the
//! two pages of a spread are independent sheets whose lines curve in
//! opposite directions toward the gutter.
//!
//! Each stage is a detector followed by a resampler.  A detector that does
//! not reach its acceptance criteria declines, and the stage passes the
//! page through unchanged; the report records the reason.  A flatbed scan
//! or a born-digital page is therefore returned as given.  Only malformed
//! input (see [`CameraError`]) is an error.
//!
//! - **Page quad** ([`detect_page_quad`]): the page boundary is found as a
//!   light quadrilateral on a darker surround (desk, stand, hands).
//! - **Crop** ([`crop`]): the image is cut to the axis-aligned bounding box
//!   of the page quad ([`Quad::bounding_rect`]).
//! - **Rectify** ([`rectify`]): the page quad is mapped onto an upright
//!   rectangle by a homography; everything outside the page is discarded.
//!   The output aspect ratio is the physical one, recovered from the
//!   vanishing geometry of the quad.  Supersedes crop when both are enabled.
//! - **Split** ([`find_gutter`], [`split_at`]): an image wider than tall
//!   whose centre shows a gutter shadow or two facing inner margins is cut
//!   into its left and right pages.
//! - **Dewarp** ([`dewarp`]): text lines are tracked across the page and
//!   fitted to a smooth surface model of a page bent about a vertical axis;
//!   each column is resampled so the lines become straight.  Experimental:
//!   vertical displacement only (glyphs foreshortened near a spine stay
//!   narrow), and a page needs at least five tracked text lines.
//!
//! Every function runs on the calling thread, keeps no global state and
//! returns on any input without panicking; the crate has no dependencies
//! beyond `std`.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![expect(
    clippy::redundant_pub_crate,
    reason = "every module is private, so rustc's `unreachable_pub` requires \
              crate-internal items to be `pub(crate)`, which this lint then \
              calls redundant; the two lints cannot both hold"
)]

mod dewarp;
mod error;
mod homography;
mod image;
mod linalg;
mod num;
mod pipeline;
mod quad;
mod spread;
mod stats;
#[cfg(test)]
mod testutil;
mod textlines;

pub use error::{CameraError, MAX_DIMENSION};
pub use image::{GrayPage, GrayView, Rect};
pub use pipeline::{Gutter, Outcome, Processed, Report, Timings, process};

use num::{floor_index, to_f64};

/// Which stages [`process`] runs.  Every stage is off by default.
///
/// `#[non_exhaustive]`: construct with [`CameraOptions::default`] or
/// [`CameraOptions::all`] and set fields individually.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one independent switch per stage"
)]
pub struct CameraOptions {
    /// Crop to the bounding box of the detected page ([`crop`]).
    pub crop: bool,
    /// Correct perspective and crop to the detected page ([`rectify`]).
    /// Supersedes [`crop`](Self::crop) when the rectification applies.
    pub rectify: bool,
    /// Split a double-page spread into two pages at the gutter
    /// ([`split_at`]).
    pub split_spreads: bool,
    /// Straighten curved text lines on each page ([`dewarp`]).
    /// Experimental.
    pub dewarp: bool,
}

impl CameraOptions {
    /// Every stage enabled.
    #[must_use]
    pub const fn all() -> Self {
        Self {
            crop: true,
            rectify: true,
            split_spreads: true,
            dewarp: true,
        }
    }

    /// Whether any stage is enabled.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.crop || self.rectify || self.split_spreads || self.dewarp
    }
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

impl Quad {
    /// The axis-aligned bounding rectangle of the quad in a `width × height`
    /// image: the pixels whose centres lie within the corners' extent on
    /// both axes.  `None` when no pixel centre does or a corner is not
    /// finite.
    #[must_use]
    pub fn bounding_rect(&self, width: usize, height: usize) -> Option<Rect> {
        if !self.corners.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        let fold = |axis: usize| {
            self.corners
                .iter()
                .map(|c| c[axis])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                })
        };
        // Pixel centres i with lo ≤ i ≤ hi.
        let span = |(lo, hi): (f64, f64), len: usize| {
            if hi < 0.0 {
                return None;
            }
            let first = floor_index(lo.ceil()).min(len);
            let end = (floor_index(hi) + 1).min(len);
            (end > first).then(|| (first, end - first))
        };
        let (x, w) = span(fold(0), width)?;
        let (y, h) = span(fold(1), height)?;
        Some(Rect {
            x,
            y,
            width: w,
            height: h,
        })
    }
}

/// The outcome of one detector: what it found, or why it declined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Detected<T> {
    /// The detector's acceptance criteria were met.
    Found(T),
    /// The detector declined; the human-readable reason.
    Declined(String),
}

impl<T> Detected<T> {
    /// The found value, discarding a decline reason.
    #[must_use]
    pub fn found(self) -> Option<T> {
        match self {
            Self::Found(v) => Some(v),
            Self::Declined(_) => None,
        }
    }

    fn from_result(r: Result<T, String>) -> Self {
        match r {
            Ok(v) => Self::Found(v),
            Err(reason) => Self::Declined(reason),
        }
    }
}

// ── Stages ──────────────────────────────────────────────────────────────────

/// Detect the boundary of a light page on a darker surround.
///
/// Declines when no page quad meets the acceptance criteria — in
/// particular when the page fills (nearly) the whole frame, as on a
/// flatbed scan.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed ([`GrayView::validate`]).
pub fn detect_page_quad(img: GrayView<'_>) -> Result<Detected<Quad>, CameraError> {
    img.validate()?;
    Ok(Detected::from_result(quad::detect(&img)))
}

/// Copy the pixels of `img` inside `rect`.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed, or
/// [`CameraError::RectOutOfBounds`] when `rect` is empty or not inside it.
pub fn crop(img: GrayView<'_>, rect: Rect) -> Result<GrayPage, CameraError> {
    img.validate()?;
    if !img.contains(rect) {
        return Err(CameraError::RectOutOfBounds {
            rect,
            width: img.width,
            height: img.height,
        });
    }
    Ok(img.sub(rect).to_page())
}

/// Resample the region of `img` inside `quad` onto an upright rectangle at
/// the page's physical aspect ratio.
///
/// Declines when the rectified page would be smaller than 16 px on a side
/// or larger than twice `img`, or when `quad` is degenerate.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed.
pub fn rectify(img: GrayView<'_>, quad: &Quad) -> Result<Detected<GrayPage>, CameraError> {
    img.validate()?;
    Ok(Detected::from_result(
        pipeline::Warp::plan(quad, img.width, img.height).map(|warp| warp.apply(img)),
    ))
}

/// The column at which `img` splits into two pages: the left page is
/// columns `[0, x)`, the right page `[x, width)`.
///
/// Declines when `img` is not recognisably a double-page spread.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed.
pub fn find_gutter(img: GrayView<'_>) -> Result<Detected<usize>, CameraError> {
    img.validate()?;
    Ok(Detected::from_result(checked_gutter(&img)))
}

/// [`spread::find_gutter`] restricted to columns strictly inside `img`.
fn checked_gutter(img: &GrayView<'_>) -> Result<usize, String> {
    let x = spread::find_gutter(img)?;
    if x > 0 && x < img.width {
        Ok(x)
    } else {
        Err(format!("gutter column {x} is at the image edge"))
    }
}

/// Split `img` into `[left, right]` at column `x`: the left page is
/// columns `[0, x)`.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed, or
/// [`CameraError::SplitOutOfRange`] unless `0 < x < width`.
pub fn split_at(img: GrayView<'_>, x: usize) -> Result<[GrayPage; 2], CameraError> {
    img.validate()?;
    if x == 0 || x >= img.width {
        return Err(CameraError::SplitOutOfRange {
            x,
            width: img.width,
        });
    }
    let [left, right] = pipeline::split_views(img, x);
    Ok([left.to_page(), right.to_page()])
}

/// Straighten curved text lines.  Experimental; see the crate docs.
///
/// Declines when the page has too few text lines, its lines are already
/// straight, or the surface fit is not accepted.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed.
pub fn dewarp(img: GrayView<'_>) -> Result<Detected<GrayPage>, CameraError> {
    img.validate()?;
    Ok(Detected::from_result(
        pipeline::fit_surface(img).map(|surface| pipeline::apply_surface(img, &surface)),
    ))
}

/// Endpoints, in the coordinates of an image of height `height`, of the
/// cut before column `x`.
fn cut_line(x: usize, height: usize) -> [[f64; 2]; 2] {
    let cx = to_f64(x) - 0.5;
    [[cx, -0.5], [cx, to_f64(height) - 0.5]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_rect_covers_touched_pixels() {
        let q = Quad {
            corners: [[10.2, 5.0], [20.7, 6.0], [19.0, 30.4], [9.6, 29.0]],
        };
        assert_eq!(
            q.bounding_rect(100, 100),
            Some(Rect {
                x: 10,
                y: 5,
                width: 11,
                height: 26
            })
        );
        let clipped = Quad {
            corners: [[-5.0, -5.0], [50.0, -5.0], [50.0, 50.0], [-5.0, 50.0]],
        };
        assert_eq!(
            clipped.bounding_rect(40, 30),
            Some(Rect {
                x: 0,
                y: 0,
                width: 40,
                height: 30
            })
        );
        for x in [50.0, -20.0] {
            let outside = Quad {
                corners: [[x, 0.0], [x + 10.0, 0.0], [x + 10.0, 10.0], [x, 10.0]],
            };
            assert_eq!(outside.bounding_rect(40, 30), None, "x {x}");
        }
    }

    #[test]
    fn split_at_rejects_edge_columns() {
        let data = [0u8; 6];
        let img = GrayView::packed(&data, 3, 2);
        assert!(matches!(
            split_at(img, 0),
            Err(CameraError::SplitOutOfRange { x: 0, width: 3 })
        ));
        assert!(split_at(img, 3).is_err());
        let Ok([l, r]) = split_at(img, 1) else {
            panic!("interior split");
        };
        assert_eq!((l.width, r.width), (1, 2));
    }

    #[test]
    fn crop_rejects_rect_outside() {
        let data = [0u8; 6];
        let img = GrayView::packed(&data, 3, 2);
        let rect = Rect {
            x: 2,
            y: 0,
            width: 2,
            height: 1,
        };
        assert!(matches!(
            crop(img, rect),
            Err(CameraError::RectOutOfBounds { .. })
        ));
    }
}
