//! Scanner border cleanup: dark regions joined to the image edge and
//! confined to a narrow band along it are whitened.
//!
//! # Model
//!
//! A flatbed or sheet-fed scan of a page smaller than the scan area, or
//! slightly askew, shows dark strips and wedges along its edges.  On a
//! box-downsampled luma plane of at most [`ANALYSIS_SIDE`] pixels a side:
//!
//! 1. Pixels below [`DARK_BELOW`] are dark.  Each dark 4-connected
//!    component that contains an image-edge pixel is a candidate.
//! 2. A candidate is junk when every one of its pixels lies within the
//!    band, [`BAND_PER_CENT`] of the shorter side deep, of an edge the
//!    candidate touches.  Text, rules and figures reaching the edge extend
//!    past the band and are kept whole.
//! 3. The junk mask is dilated by one analysis pixel, which covers the
//!    partially dark pixels along its boundary, and scaled back to source
//!    resolution as runs of columns per row.
//!
//! # Acceptance
//!
//! Declines when more than half the analysis plane is dark (a dark ground:
//! a reversed-out or full-bleed page), when no dark region touches the
//! edge, when every such region extends past the band, or when the junk
//! covers more than a quarter of the page.
//!
//! # Limits
//!
//! Punch holes, which do not reach the edge, and soft gray gutter
//! shadows, which are not dark, are left in place.

use crate::Detected;
use crate::error::CameraError;
use crate::image::{GrayPage, GrayView};
use crate::region::{border_indices, flood};
use crate::sample::Sample;

/// Longest side of the analysis plane, in pixels.
const ANALYSIS_SIDE: usize = 1600;
/// Samples below this luma on the analysis plane are dark.
const DARK_BELOW: u8 = 128;
/// Depth of the border band, per cent of the shorter side.
const BAND_PER_CENT: usize = 6;
/// A dark ground covers more than `num / den` of the analysis plane.
const DARK_GROUND: (usize, usize) = (1, 2);
/// Junk may cover at most `num / den` of the analysis plane.
const MAX_JUNK: (usize, usize) = (1, 4);

/// Dark border regions to whiten, as runs of source pixels.
///
/// Found by [`find_border_junk`]; applied by [`whiten`](Self::whiten).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorderMask {
    width: usize,
    height: usize,
    rows: Vec<RunRow>,
}

/// Source rows `[y0, y1)` share the column runs `[x0, x1)` of `runs`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RunRow {
    y0: usize,
    y1: usize,
    runs: Vec<(usize, usize)>,
}

/// Find the dark regions along the edges of the 8-bit luma `img` that a
/// scanner leaves outside the page.
///
/// Declines on a dark ground, when no dark region touches the edge, when
/// every one reaches past the border band, or when the regions found cover
/// more than a quarter of the page.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed.
pub fn find_border_junk(img: GrayView<'_>) -> Result<Detected<BorderMask>, CameraError> {
    img.validate()?;
    Ok(Detected::from_result(detect(&img)))
}

impl BorderMask {
    /// Set every pixel of `img` inside the mask to white.  `img` is the
    /// measured luma, one channel of the colour image it was taken from,
    /// or either at 16 bits per sample.
    ///
    /// # Errors
    ///
    /// [`CameraError`] when `img` is malformed, or
    /// [`CameraError::SizeMismatch`] when its size differs from the
    /// measured image.
    pub fn whiten<S: Sample>(&self, img: &mut GrayPage<S>) -> Result<(), CameraError> {
        img.view().validate_size((self.width, self.height))?;
        for row in &self.rows {
            for line in
                img.data[row.y0 * self.width..row.y1 * self.width].chunks_exact_mut(self.width)
            {
                for &(x0, x1) in &row.runs {
                    line[x0..x1].fill(S::MAX);
                }
            }
        }
        Ok(())
    }

    /// Whether source pixel `(x, y)` lies inside the mask.
    #[cfg(test)]
    fn covers(&self, x: usize, y: usize) -> bool {
        self.rows.iter().any(|r| {
            (r.y0..r.y1).contains(&y) && r.runs.iter().any(|&(x0, x1)| (x0..x1).contains(&x))
        })
    }
}

fn detect(img: &GrayView<'_>) -> Result<BorderMask, String> {
    let factor = img.factor_for(ANALYSIS_SIDE);
    let small = img.downsample(factor);
    let (w, h) = (small.width, small.height);
    if w == 0 || h == 0 {
        return Err(format!(
            "image is {}×{}; its 1/{factor} analysis plane is empty",
            img.width, img.height
        ));
    }
    let dark: Vec<bool> = small.data.iter().map(|&v| v < DARK_BELOW).collect();
    let dark_count = dark.iter().filter(|&&d| d).count();
    if dark_count * DARK_GROUND.1 > w * h * DARK_GROUND.0 {
        return Err(format!(
            "dark ground: {} % of the page is dark",
            dark_count * 100 / (w * h)
        ));
    }
    let band = (w.min(h) * BAND_PER_CENT).div_ceil(100).max(1);
    let mut seen = vec![false; w * h];
    let mut junk = vec![false; w * h];
    let (mut candidates, mut junk_area) = (0usize, 0usize);
    let (mut stack, mut region) = (Vec::new(), Vec::new());
    for start in border_indices(w, h) {
        if !dark[start] || seen[start] {
            continue;
        }
        region.clear();
        let _ = flood(start, w, h, &mut stack, |i| {
            if dark[i] && !seen[i] {
                seen[i] = true;
                region.push(i);
                true
            } else {
                false
            }
        });
        candidates += 1;
        if within_band(&region, w, h, band) {
            junk_area += region.len();
            for &i in &region {
                junk[i] = true;
            }
        }
    }
    if candidates == 0 {
        return Err("no dark region touches the image edge".to_owned());
    }
    if junk_area == 0 {
        return Err(format!(
            "{candidates} dark region(s) touch the image edge, each reaching past the \
             {} px border band",
            band * factor
        ));
    }
    if junk_area * MAX_JUNK.1 > w * h * MAX_JUNK.0 {
        return Err(format!(
            "dark border regions cover {} % of the page, more than {} %",
            junk_area * 100 / (w * h),
            100 * MAX_JUNK.0 / MAX_JUNK.1
        ));
    }
    Ok(BorderMask {
        width: img.width,
        height: img.height,
        rows: source_runs(
            &dilate(&junk, w, h),
            (w, h),
            factor,
            (img.width, img.height),
        ),
    })
}

/// Whether every pixel of `region` (indices into a `w × h` grid) lies
/// within `band` pixels of an image edge the region touches.
fn within_band(region: &[usize], w: usize, h: usize, band: usize) -> bool {
    let at = |i: usize| (i % w, i / w);
    let [left, right, top, bottom] = region.iter().fold([false; 4], |touched, &i| {
        let (x, y) = at(i);
        [
            touched[0] || x == 0,
            touched[1] || x + 1 == w,
            touched[2] || y == 0,
            touched[3] || y + 1 == h,
        ]
    });
    region.iter().all(|&i| {
        let (x, y) = at(i);
        (left && x < band)
            || (right && x + band >= w)
            || (top && y < band)
            || (bottom && y + band >= h)
    })
}

/// `mask` dilated by one pixel in each of the eight directions.
fn dilate(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                out[yy * w + x.saturating_sub(1)..yy * w + (x + 2).min(w)].fill(true);
            }
        }
    }
    out
}

/// The runs of `mask`, a `w × h` plane downsampled by `factor` from a
/// `src_w × src_h` image, in source pixels.  Analysis pixel `i` covers
/// source pixels `[i · factor, (i + 1) · factor)`; the last column and row
/// also cover the trailing source pixels the downsampling discarded.
fn source_runs(
    mask: &[bool],
    (w, h): (usize, usize),
    factor: usize,
    (src_w, src_h): (usize, usize),
) -> Vec<RunRow> {
    let scale = |i: usize, n: usize, len: usize| if i == n { len } else { i * factor };
    let mut rows = Vec::new();
    for (y, line) in mask.chunks_exact(w).enumerate() {
        let mut runs = Vec::new();
        let mut x = 0;
        while x < w {
            if !line[x] {
                x += 1;
                continue;
            }
            let start = x;
            while x < w && line[x] {
                x += 1;
            }
            runs.push((scale(start, w, src_w), scale(x, w, src_w)));
        }
        if !runs.is_empty() {
            rows.push(RunRow {
                y0: scale(y, h, src_h),
                y1: scale(y + 1, h, src_h),
                runs,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{digest, plane};

    /// Whether `(x, y)` lies in a top wedge 8 px deep at the left end of a
    /// `w`-wide page and 70 px deep at the right.
    const fn in_wedge(x: usize, y: usize, w: usize) -> bool {
        y < 8 + 62 * x / w
    }

    /// A scanned page with text bars, `strip` px of dark scanner bed along
    /// the left edge and, when `wedge`, a dark wedge along the top
    /// ([`in_wedge`]).
    fn flatbed(w: usize, h: usize, strip: usize, wedge: bool) -> GrayPage {
        plane(w, h, |x, y| {
            if x < strip {
                30
            } else if wedge && in_wedge(x, y, w) {
                40
            } else if (150..w - 150).contains(&x)
                && (200..h - 200).contains(&y)
                && (y / 9) % 3 == 0
                && (x / 23) % 5 != 4
            {
                25
            } else {
                235
            }
        })
    }

    fn found(img: &GrayPage) -> BorderMask {
        match find_border_junk(img.view()).expect("valid") {
            Detected::Found(mask) => mask,
            Detected::Declined(reason) => panic!("declined: {reason}"),
        }
    }

    fn declined(img: &GrayPage) -> String {
        match find_border_junk(img.view()).expect("valid") {
            Detected::Found(mask) => panic!("found {mask:?}"),
            Detected::Declined(reason) => reason,
        }
    }

    /// Whiten `img` by `mask`; every pixel the predicate `junk` names
    /// becomes white and every pixel outside the mask keeps its value.
    fn check_whitened(
        img: &GrayPage,
        mask: &BorderMask,
        junk: impl Fn(usize, usize) -> bool,
    ) -> GrayPage {
        let mut out = img.clone();
        mask.whiten(&mut out).expect("same size");
        for y in 0..img.height {
            for x in 0..img.width {
                let (before, after) = (img.data[y * img.width + x], out.data[y * img.width + x]);
                if junk(x, y) {
                    assert_eq!(after, 255, "junk at ({x}, {y})");
                }
                if !mask.covers(x, y) {
                    assert_eq!(after, before, "({x}, {y}) outside the mask");
                }
            }
        }
        out
    }

    #[test]
    fn flatbed_strips_are_whitened() {
        let (w, h) = (1701, 2203);
        let img = flatbed(w, h, 40, true);
        let mask = found(&img);
        let out = check_whitened(&img, &mask, |x, y| x < 40 || in_wedge(x, y, w));
        // The text is untouched.
        for y in 200..h - 200 {
            for x in 150..w - 150 {
                assert_eq!(out.data[y * w + x], img.data[y * w + x]);
            }
        }
        let got = digest(&[out]);
        assert_eq!(got, GOLDEN, "{got:#x}");
    }

    const GOLDEN: u64 = 0x29cc_2eee_3838_4e8a;

    #[test]
    fn bar_running_into_the_page_is_kept() {
        let (w, h) = (1200, 1600);
        let bar = |x: usize, y: usize| x < w * 3 / 10 && (780..820).contains(&y);
        let alone = plane(w, h, |x, y| if bar(x, y) { 20 } else { 235 });
        let reason = declined(&alone);
        assert!(reason.contains("border band"), "{reason}");
        // Beside a top wedge, the wedge goes and the bar stays.
        let both = plane(w, h, |x, y| {
            if bar(x, y) {
                20
            } else if in_wedge(x, y, w) {
                40
            } else {
                235
            }
        });
        let mask = found(&both);
        let out = check_whitened(&both, &mask, |x, y| in_wedge(x, y, w));
        for y in 780..820 {
            for x in 0..w * 3 / 10 {
                assert_eq!(out.data[y * w + x], 20, "bar at ({x}, {y})");
            }
        }
    }

    #[test]
    fn clean_reversed_and_dark_pages_decline() {
        let clean = flatbed(900, 1200, 0, false);
        assert!(declined(&clean).contains("no dark region"));
        let reversed = plane(900, 1200, |x, y| {
            if (100..800).contains(&x) && (y / 9) % 3 == 0 {
                230
            } else {
                20
            }
        });
        assert!(declined(&reversed).contains("dark ground"));
        let dark = plane(900, 1200, |_, _| 15);
        assert!(declined(&dark).contains("dark ground"));
    }

    #[test]
    fn junk_over_a_quarter_of_the_page_declines() {
        // The band holds at most 1 − 0.88² ≈ 23 % of a large page; on a
        // 20 × 20 page it is rounded up to 2 px, and a frame that deep
        // covers 36 %.
        let (w, h) = (20, 20);
        let img = plane(w, h, |x, y| {
            if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
                20
            } else {
                235
            }
        });
        let reason = declined(&img);
        assert!(reason.contains("cover"), "{reason}");
    }

    #[test]
    fn deep_pages_whiten_alike_and_sizes_must_match() {
        let (w, h) = (600, 800);
        let img = flatbed(w, h, 20, false);
        let mask = found(&img);
        let mut narrow = img.clone();
        mask.whiten(&mut narrow).expect("same size");
        let mut deep = GrayPage::<u16> {
            width: w,
            height: h,
            data: img.data.iter().map(|&v| u16::from(v) * 257).collect(),
        };
        mask.whiten(&mut deep).expect("same size");
        let back: Vec<u8> = deep
            .data
            .iter()
            .map(|&v| u8::try_from(v / 257).unwrap_or(0))
            .collect();
        assert_eq!(back, narrow.data);
        let mut other = plane(10, 10, |_, _| 0);
        assert!(matches!(
            mask.whiten(&mut other),
            Err(CameraError::SizeMismatch { .. })
        ));
    }
}
