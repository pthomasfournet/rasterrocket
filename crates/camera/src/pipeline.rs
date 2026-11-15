//! The stage pipeline, its report, and the replayable geometric transform.

use std::time::{Duration, Instant};

use crate::dewarp::Surface;
use crate::homography::{Homography, rectified_size};
use crate::image::{GrayPage, GrayView, Rect, resample};
use crate::num::to_f64;
use crate::{CameraError, CameraOptions, Quad, Sample, cut_line, dewarp, quad, textlines};

/// The pages produced by [`process`] and the report of how.
#[derive(Debug, Clone)]
pub struct Processed {
    /// One page, or the left then the right page of a split spread.
    pub pages: Vec<GrayPage>,
    /// What each stage did.
    pub report: Report,
}

/// What one stage did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Outcome {
    /// The stage was not enabled.
    Off,
    /// The stage changed the image.
    Applied,
    /// The stage was enabled but left the image unchanged; the
    /// human-readable reason.
    Skipped(String),
}

impl Outcome {
    /// Whether the stage changed the image.
    #[must_use]
    pub const fn applied(&self) -> bool {
        matches!(self, Self::Applied)
    }
}

/// Where a spread was split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gutter {
    /// First column of the right page, in the image entering the split
    /// stage (after crop or rectify); the left page is columns `[0, x)`.
    pub x: usize,
    /// First column of the right page in the source image, when the cut is
    /// a source column (no rectification).
    pub source_x: Option<usize>,
    /// Top and bottom ends of the cut in source pixel coordinates
    /// (continuous; see [`Quad`]).
    pub source: [[f64; 2]; 2],
}

/// Wall-clock time spent per stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Timings {
    /// Page-quad detection (shared by crop and rectify).
    pub quad: Duration,
    /// Crop placement.
    pub crop: Duration,
    /// Rectification resampling.
    pub rectify: Duration,
    /// Gutter detection.
    pub split: Duration,
    /// Dewarp detection and resampling, summed over pages.
    pub dewarp: Duration,
    /// Copying unresampled pages into packed output.
    pub output: Duration,
    /// The whole of [`process`].
    pub total: Duration,
}

/// What [`process`] did, stage by stage.
#[derive(Debug, Clone)]
pub struct Report {
    /// The crop stage.
    pub crop: Outcome,
    /// The rectify stage.
    pub rectify: Outcome,
    /// The split stage.
    pub split: Outcome,
    /// The dewarp stage, one entry per output page in page order.
    pub dewarp: Vec<Outcome>,
    /// The page quad in source pixel coordinates, when crop or rectify ran
    /// and the detector found one.
    pub quad: Option<Quad>,
    /// The crop rectangle in source pixels, when the crop applied.
    pub crop_rect: Option<Rect>,
    /// The gutter, when the split applied.
    pub gutter: Option<Gutter>,
    /// The rectifying homography, when rectify applied: a row-major 3 × 3
    /// matrix mapping homogeneous rectified pixel coordinates `(x, y, 1)`
    /// to source pixel coordinates.
    pub homography: Option<[f64; 9]>,
    /// Time spent per stage.
    pub timings: Timings,
    transform: Transform,
}

impl Report {
    /// Apply the geometry computed by [`process`] to another image of the
    /// same size, such as one colour channel of the image whose luma was
    /// processed, or the same image at 16 bits per sample.  Replaying on
    /// the processed image itself reproduces [`Processed::pages`].  The
    /// resampling runs at the depth of `S`.
    ///
    /// # Errors
    ///
    /// [`CameraError`] when `img` is malformed, or
    /// [`CameraError::SizeMismatch`] when its size differs from the
    /// processed image.
    pub fn replay<S: Sample>(&self, img: GrayView<'_, S>) -> Result<Vec<GrayPage<S>>, CameraError> {
        img.validate_size(self.transform.input)?;
        Ok(self.transform.apply(img))
    }
}

// ── Geometry ────────────────────────────────────────────────────────────────

/// A perspective rectification: output size and the homography from
/// output to source pixel coordinates.
#[derive(Debug, Clone)]
pub(crate) struct Warp {
    hmg: Homography,
    width: usize,
    height: usize,
}

impl Warp {
    /// The rectification of `quad` in a `src_w × src_h` image.
    pub(crate) fn plan(quad: &Quad, src_w: usize, src_h: usize) -> Result<Self, String> {
        let (width, height) = rectified_size(quad, src_w, src_h)?;
        let (fw, fh) = (to_f64(width), to_f64(height));
        let rect = [
            [-0.5, -0.5],
            [fw - 0.5, -0.5],
            [fw - 0.5, fh - 0.5],
            [-0.5, fh - 0.5],
        ];
        let hmg = Homography::from_points(&rect, &quad.corners)
            .ok_or_else(|| "page quad is degenerate: three corners are collinear".to_owned())?;
        Ok(Self { hmg, width, height })
    }

    /// Resample `img` through the warp.
    pub(crate) fn apply<S: Sample>(&self, img: GrayView<'_, S>) -> GrayPage<S> {
        resample(img, self.width, self.height, |x, y| self.hmg.apply(x, y))
    }
}

/// Fit a dewarp surface to the text lines of `img`.
pub(crate) fn fit_surface(img: GrayView<'_>) -> Result<Surface, String> {
    let lines = textlines::find(&img)?;
    dewarp::fit(&lines, img.width, img.height)
}

/// Resample `img` through a dewarp surface.
pub(crate) fn apply_surface<S: Sample>(img: GrayView<'_, S>, surface: &Surface) -> GrayPage<S> {
    resample(img, img.width, img.height, |x, y| {
        (x, surface.source_y(x, y))
    })
}

/// Columns `[0, x)` and `[x, width)` of `img`.  Caller guarantees
/// `0 < x < width`.
pub(crate) fn split_views<S: Sample>(img: GrayView<'_, S>, x: usize) -> [GrayView<'_, S>; 2] {
    let h = img.height;
    [
        img.sub(Rect {
            x: 0,
            y: 0,
            width: x,
            height: h,
        }),
        img.sub(Rect {
            x,
            y: 0,
            width: img.width - x,
            height: h,
        }),
    ]
}

/// How the source image is framed before the split.
#[derive(Debug, Clone)]
enum Frame {
    Whole,
    Crop(Rect),
    Warp(Warp),
}

/// A framed image: a view into the source, or a resampled page.
enum Framed<'a, S: Sample = u8> {
    View(GrayView<'a, S>),
    Owned(GrayPage<S>),
}

impl<S: Sample> Framed<'_, S> {
    fn view(&self) -> GrayView<'_, S> {
        match self {
            Self::View(v) => *v,
            Self::Owned(p) => p.view(),
        }
    }
}

impl Frame {
    fn apply<'a, S: Sample>(&self, img: GrayView<'a, S>) -> Framed<'a, S> {
        match self {
            Self::Whole => Framed::View(img),
            Self::Crop(rect) => Framed::View(img.sub(*rect)),
            Self::Warp(warp) => Framed::Owned(warp.apply(img)),
        }
    }

    /// Map a point of the framed image to source pixel coordinates.
    fn to_source(&self, p: [f64; 2]) -> [f64; 2] {
        match self {
            Self::Whole => p,
            Self::Crop(rect) => [p[0] + to_f64(rect.x), p[1] + to_f64(rect.y)],
            Self::Warp(warp) => warp.hmg.apply(p[0], p[1]).into(),
        }
    }
}

/// The complete geometric transform from source image to output pages.
#[derive(Debug, Clone)]
struct Transform {
    input: (usize, usize),
    frame: Frame,
    split: Option<usize>,
    surfaces: Vec<Option<Surface>>,
}

impl Transform {
    /// Caller guarantees a validated `img` of size `self.input`.
    fn apply<S: Sample>(&self, img: GrayView<'_, S>) -> Vec<GrayPage<S>> {
        let framed = self.frame.apply(img);
        let parts = parts(framed.view(), self.split);
        parts
            .into_iter()
            .zip(&self.surfaces)
            .map(|(part, surface)| {
                surface
                    .as_ref()
                    .map_or_else(|| part.to_page(), |s| apply_surface(part, s))
            })
            .collect()
    }
}

/// The page views of `img`: the whole, or both sides of a split.
fn parts<S: Sample>(img: GrayView<'_, S>, split: Option<usize>) -> Vec<GrayView<'_, S>> {
    split.map_or_else(|| vec![img], |x| split_views(img, x).to_vec())
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/// Run the stages enabled in `opts` on one captured image.
///
/// The stages run in the order crop or rectify, split, then dewarp on each
/// resulting page.  A stage whose detector declines leaves the image
/// unchanged and records the reason in the [`Report`].
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed ([`GrayView::validate`]).
pub fn process(img: GrayView<'_>, opts: &CameraOptions) -> Result<Processed, CameraError> {
    img.validate()?;
    let start = Instant::now();
    let mut timings = Timings::default();

    let framing = frame_stage(img, *opts, &mut timings);
    let current = framing.framed.view();
    let (split, split_x) = split_stage(current, *opts, &mut timings);
    let views = parts(current, split_x);
    let dewarped = dewarp_stage(&views, *opts, &mut timings);

    let t = Instant::now();
    let pages: Vec<GrayPage> = views
        .iter()
        .zip(dewarped.pages)
        .map(|(part, page)| page.unwrap_or_else(|| part.to_page()))
        .collect();
    timings.output = t.elapsed();

    let frame = framing.frame;
    let gutter = split_x.map(|x| gutter_report(&frame, x, current.height));
    let crop_rect = match &frame {
        Frame::Crop(rect) => Some(*rect),
        Frame::Whole | Frame::Warp(_) => None,
    };
    let homography = match &frame {
        Frame::Warp(warp) => Some(warp.hmg.matrix()),
        Frame::Whole | Frame::Crop(_) => None,
    };
    timings.total = start.elapsed();
    Ok(Processed {
        pages,
        report: Report {
            crop: framing.crop,
            rectify: framing.rectify,
            split,
            dewarp: dewarped.outcomes,
            quad: framing.quad,
            crop_rect,
            gutter,
            homography,
            timings,
            transform: Transform {
                input: (img.width, img.height),
                frame,
                split: split_x,
                surfaces: dewarped.surfaces,
            },
        },
    })
}

/// The result of the crop / rectify stage.
struct Framing<'a> {
    framed: Framed<'a>,
    frame: Frame,
    quad: Option<Quad>,
    crop: Outcome,
    rectify: Outcome,
}

/// Detect the page quad when crop or rectify is enabled and frame the
/// image by it: rectify when enabled and possible, else crop when enabled.
fn frame_stage<'a>(img: GrayView<'a>, opts: CameraOptions, timings: &mut Timings) -> Framing<'a> {
    let mut out = Framing {
        framed: Framed::View(img),
        frame: Frame::Whole,
        quad: None,
        crop: Outcome::Off,
        rectify: Outcome::Off,
    };
    if !(opts.crop || opts.rectify) {
        return out;
    }
    let t = Instant::now();
    let detected = quad::detect(&img);
    timings.quad = t.elapsed();
    let q = match detected {
        Ok(q) => q,
        Err(reason) => {
            let skipped = Outcome::Skipped(reason);
            if opts.crop {
                out.crop = skipped.clone();
            }
            if opts.rectify {
                out.rectify = skipped;
            }
            return out;
        }
    };
    out.quad = Some(q);

    if opts.rectify {
        let t = Instant::now();
        match Warp::plan(&q, img.width, img.height) {
            Ok(warp) => {
                out.framed = Framed::Owned(warp.apply(img));
                out.frame = Frame::Warp(warp);
                out.rectify = Outcome::Applied;
                if opts.crop {
                    out.crop = Outcome::Skipped("superseded by rectify".to_owned());
                }
            }
            Err(reason) => out.rectify = Outcome::Skipped(reason),
        }
        timings.rectify = t.elapsed();
    }

    if opts.crop && !out.rectify.applied() {
        let t = Instant::now();
        let full = Rect {
            x: 0,
            y: 0,
            width: img.width,
            height: img.height,
        };
        out.crop = match q.bounding_rect(img.width, img.height) {
            Some(rect) if rect != full => {
                out.framed = Framed::View(img.sub(rect));
                out.frame = Frame::Crop(rect);
                Outcome::Applied
            }
            Some(_) => Outcome::Skipped("page bounding box covers the whole image".to_owned()),
            None => Outcome::Skipped("page quad lies outside the image".to_owned()),
        };
        timings.crop = t.elapsed();
    }
    out
}

/// Find the gutter of `img` when the split is enabled.
fn split_stage(
    img: GrayView<'_>,
    opts: CameraOptions,
    timings: &mut Timings,
) -> (Outcome, Option<usize>) {
    if !opts.split_spreads {
        return (Outcome::Off, None);
    }
    let t = Instant::now();
    let found = crate::checked_gutter(&img);
    timings.split = t.elapsed();
    match found {
        Ok(x) => (Outcome::Applied, Some(x)),
        Err(reason) => (Outcome::Skipped(reason), None),
    }
}

/// The result of the dewarp stage, one entry per page.
struct Dewarped {
    /// The dewarped page, or `None` where the page is unchanged.
    pages: Vec<Option<GrayPage>>,
    surfaces: Vec<Option<Surface>>,
    outcomes: Vec<Outcome>,
}

/// Dewarp each page view when the stage is enabled.
fn dewarp_stage(views: &[GrayView<'_>], opts: CameraOptions, timings: &mut Timings) -> Dewarped {
    let mut out = Dewarped {
        pages: Vec::with_capacity(views.len()),
        surfaces: Vec::with_capacity(views.len()),
        outcomes: Vec::with_capacity(views.len()),
    };
    for &part in views {
        if !opts.dewarp {
            out.pages.push(None);
            out.surfaces.push(None);
            out.outcomes.push(Outcome::Off);
            continue;
        }
        let t = Instant::now();
        match fit_surface(part) {
            Ok(surface) => {
                out.pages.push(Some(apply_surface(part, &surface)));
                out.surfaces.push(Some(surface));
                out.outcomes.push(Outcome::Applied);
            }
            Err(reason) => {
                out.pages.push(None);
                out.surfaces.push(None);
                out.outcomes.push(Outcome::Skipped(reason));
            }
        }
        timings.dewarp += t.elapsed();
    }
    out
}

/// The gutter at column `x` of a framed image of height `height`.
fn gutter_report(frame: &Frame, x: usize, height: usize) -> Gutter {
    let [top, bottom] = cut_line(x, height);
    Gutter {
        x,
        source_x: match frame {
            Frame::Whole => Some(x),
            Frame::Crop(rect) => Some(rect.x + x),
            Frame::Warp(_) => None,
        },
        source: [frame.to_source(top), frame.to_source(bottom)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{curved_text, digest, plane, tilted_page};

    /// A light page with text bars on a darker desk; `page` is
    /// `(x0, y0, x1, y1)`, exclusive at the end.
    fn page_on_desk(w: usize, h: usize, page: (usize, usize, usize, usize)) -> GrayPage {
        plane(w, h, |x, y| {
            let on_page = (page.0..page.2).contains(&x) && (page.1..page.3).contains(&y);
            if !on_page {
                80
            } else if (y / 8) % 3 == 0
                && (x / 19) % 5 != 4
                && (page.0 + 40..page.2 - 40).contains(&x)
                && (page.1 + 40..page.3 - 40).contains(&y)
            {
                25
            } else {
                240
            }
        })
    }

    fn opts(f: impl FnOnce(&mut CameraOptions)) -> CameraOptions {
        let mut o = CameraOptions::default();
        f(&mut o);
        o
    }

    #[test]
    fn rectify_crops_to_page_and_keeps_paper() {
        let img = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let out = process(img.view(), &opts(|o| o.rectify = true)).expect("valid input");
        assert_eq!(out.report.rectify, Outcome::Applied);
        assert_eq!(out.pages.len(), 1);
        let page = &out.pages[0];
        assert!(page.width.abs_diff(700) <= 2, "width {}", page.width);
        assert!(page.height.abs_diff(950) <= 2, "height {}", page.height);
        assert_eq!(page.data[(page.height / 2 + 2) * page.width + 3], 240);
        assert!(out.report.homography.is_some());
    }

    #[test]
    fn crop_cuts_to_page_bounding_box() {
        let img = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let out = process(img.view(), &opts(|o| o.crop = true)).expect("valid input");
        assert_eq!(out.report.crop, Outcome::Applied);
        let Some(rect) = out.report.crop_rect else {
            panic!("crop rectangle reported");
        };
        assert!(
            rect.x.abs_diff(100) <= 1 && rect.y.abs_diff(150) <= 1,
            "{rect:?}"
        );
        assert!(
            rect.width.abs_diff(700) <= 2 && rect.height.abs_diff(950) <= 2,
            "{rect:?}"
        );
        let page = &out.pages[0];
        assert_eq!((page.width, page.height), (rect.width, rect.height));
        assert_eq!(out.report.rectify, Outcome::Off);
    }

    #[test]
    fn rectify_supersedes_crop() {
        let img = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let both = opts(|o| {
            o.crop = true;
            o.rectify = true;
        });
        let out = process(img.view(), &both).expect("valid input");
        assert_eq!(out.report.rectify, Outcome::Applied);
        assert!(matches!(out.report.crop, Outcome::Skipped(_)));
        assert!(out.report.crop_rect.is_none());
    }

    #[test]
    fn crop_then_split_reports_source_gutter() {
        // Two text blocks side by side on one sheet, on a desk.
        let img = plane(1700, 1000, |x, y| {
            let on_page = (100..1600).contains(&x) && (80..920).contains(&y);
            let in_text =
                ((180..780).contains(&x) || (920..1520).contains(&x)) && (160..840).contains(&y);
            if !on_page {
                70
            } else if in_text && (y / 8) % 3 == 0 && (x / 17) % 5 != 4 {
                30
            } else {
                235
            }
        });
        let crop_split = opts(|o| {
            o.crop = true;
            o.split_spreads = true;
        });
        let out = process(img.view(), &crop_split).expect("valid input");
        assert_eq!(out.report.crop, Outcome::Applied);
        assert_eq!(out.report.split, Outcome::Applied);
        assert_eq!(out.pages.len(), 2);
        let Some(g) = out.report.gutter else {
            panic!("gutter reported");
        };
        let Some(sx) = g.source_x else {
            panic!("crop keeps the cut a source column");
        };
        assert!(sx.abs_diff(850) <= 12, "gutter at source column {sx}");
        assert_eq!(out.pages[0].width, g.x);
        assert!((g.source[0][0] - (to_f64(sx) - 0.5)).abs() < 1e-9);
    }

    #[test]
    fn replay_reproduces_pages() {
        let img = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let out = process(img.view(), &CameraOptions::all()).expect("valid input");
        let again = out.report.replay(img.view()).expect("same size");
        assert_eq!(again, out.pages);
        let other = plane(10, 10, |_, _| 0);
        assert!(matches!(
            out.report.replay(other.view()),
            Err(CameraError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn disabled_options_are_identity() {
        let img = page_on_desk(300, 200, (20, 20, 280, 180));
        let out = process(img.view(), &CameraOptions::default()).expect("valid input");
        assert_eq!(out.report.crop, Outcome::Off);
        assert_eq!(out.report.dewarp, vec![Outcome::Off]);
        assert_eq!(out.pages, vec![img]);
    }

    #[test]
    fn blank_page_passes_every_stage() {
        let img = plane(400, 300, |_, _| 240);
        let out = process(img.view(), &CameraOptions::all()).expect("valid input");
        for outcome in [&out.report.crop, &out.report.rectify, &out.report.split] {
            assert!(matches!(outcome, Outcome::Skipped(_)), "{outcome:?}");
        }
        assert_eq!(out.pages, vec![img]);
    }

    /// The 8-bit pages of fixed inputs, digested before the resamplers
    /// became generic over the sample.
    #[test]
    fn eight_bit_output_matches_its_golden_digests() {
        let all = CameraOptions::all();
        let desk = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let tilted = tilted_page(1000, 1300);
        let curved = curved_text(1200, 1000, |x, _| 40.0 * (x / 1200.0).powi(3));
        let crop = opts(|o| o.crop = true);
        let rectify = opts(|o| o.rectify = true);
        let dewarp = opts(|o| o.dewarp = true);
        let got: Vec<u64> = [
            (&desk, all),
            (&tilted, rectify),
            (&tilted, crop),
            (&curved, dewarp),
        ]
        .into_iter()
        .map(|(img, o)| {
            let out = process(img.view(), &o).expect("valid input");
            assert!(changed(&out.report), "{:?}", out.report);
            digest(&out.pages)
        })
        .collect();
        assert_eq!(got, GOLDEN, "{got:#x?}");
    }

    const GOLDEN: [u64; 4] = [
        0xac3c_a584_5c68_fbe4,
        0xe5ae_7f73_826a_3125,
        0xf9dd_9700_a596_0696,
        0x5fea_0d1c_825b_cdd6,
    ];

    fn changed(r: &Report) -> bool {
        r.crop.applied() || r.rectify.applied() || r.dewarp.iter().any(Outcome::applied)
    }

    /// `page`'s samples widened to 16 bits (`v · 257`).
    fn widened(page: &GrayPage) -> GrayPage<u16> {
        GrayPage {
            width: page.width,
            height: page.height,
            data: page.data.iter().map(|&v| u16::from(v) * 257).collect(),
        }
    }

    #[test]
    fn sixteen_bit_replay_follows_the_eight_bit_geometry() {
        let desk = page_on_desk(900, 1200, (100, 150, 800, 1100));
        let tilted = tilted_page(1000, 1300);
        let curved = curved_text(1200, 1000, |x, _| 40.0 * (x / 1200.0).powi(3));
        for (img, o) in [
            (&desk, CameraOptions::all()),
            (&tilted, opts(|o| o.rectify = true)),
            (&tilted, opts(|o| o.crop = true)),
            (&curved, opts(|o| o.dewarp = true)),
        ] {
            let out = process(img.view(), &o).expect("valid input");
            let wide = widened(img);
            let pages = out.report.replay(wide.view()).expect("same size");
            assert_eq!(pages.len(), out.pages.len());
            for (deep, narrow) in pages.iter().zip(&out.pages) {
                assert_eq!((deep.width, deep.height), (narrow.width, narrow.height));
                for (&d, &n) in deep.data.iter().zip(&narrow.data) {
                    // round(d / 257) and the 8-bit blend round the same
                    // bilinear value, so they differ by at most one level.
                    let level = (u32::from(d) + 128) / 257;
                    assert!(level.abs_diff(u32::from(n)) <= 1, "{d} vs {n}");
                }
            }
        }
        let small = plane(10, 10, |_, _| 0);
        assert!(matches!(
            process(desk.view(), &CameraOptions::all())
                .expect("valid input")
                .report
                .replay(widened(&small).view()),
            Err(CameraError::SizeMismatch { .. })
        ));
    }
}
