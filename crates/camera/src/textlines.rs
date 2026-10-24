//! Text-line tracking: sample the vertical position of each line of text
//! across the page width.
//!
//! # Method
//!
//! 1. **Ink map.**  A pixel is ink when it is darker than 0.75× the mean
//!    of a local window (integral image) and at least 24 levels below it;
//!    the local reference makes the test independent of uneven lighting.
//! 2. **Strip profiles.**  The page is cut into vertical strips of
//!    ≈ 1/48 of the width.  In each strip the row-wise ink count is smoothed
//!    and segmented into runs above a density threshold; every run of
//!    plausible line height yields an observation at its baseline (see
//!    [`baseline`]).
//! 3. **Linking.**  Observations are linked left to right into tracks by
//!    predicting each track's position from its recent slope and accepting
//!    the nearest observation within 0.35 line pitches (one-to-one, greedy
//!    by distance), bridging gaps of up to three strips (word spaces,
//!    column gutters).
//! 4. **Cleaning.**  Points farther than a quarter pitch from the line through
//!    their neighbours are dropped; short tracks are
//!    discarded.

use crate::image::{GrayPage, GrayView, upscale_coord};
use crate::num::to_f64;

/// Longest side of the analysis plane, in pixels.
const ANALYSIS_SIDE: usize = 1600;
/// Number of vertical strips across the width.
const STRIPS: usize = 48;
/// Largest strip gap a track may bridge.
const MAX_GAP_STRIPS: usize = 3;

/// One text line: sample points `(x, y)` in source pixel coordinates,
/// ordered by `x`, each with the ink mass that produced it.
#[derive(Debug, Clone)]
pub(crate) struct Track {
    pub(crate) points: Vec<[f64; 2]>,
    pub(crate) weights: Vec<f64>,
}

/// The tracks found on a page and the median line pitch, in source pixels.
#[derive(Debug)]
pub(crate) struct TextLines {
    pub(crate) tracks: Vec<Track>,
    pub(crate) pitch: f64,
}

struct Observation {
    y: f64,
    mass: f64,
}

/// Track the text lines of a full-resolution image.  `Err` holds the
/// reason no line pitch can be established (too little text).
pub(crate) fn find(luma: &GrayView<'_>) -> Result<TextLines, String> {
    let factor = luma.factor_for(ANALYSIS_SIDE);
    let small = luma.downsample(factor);
    let (w, h) = (small.width, small.height);
    if w < STRIPS * 4 || h < 64 {
        return Err(format!(
            "page is {}×{}, {w}×{h} at the 1/{factor} analysis scale; \
             text-line tracking needs at least {}×64 there",
            luma.width,
            luma.height,
            STRIPS * 4
        ));
    }
    let ink = ink_map(&small);
    let strip_w = w / STRIPS;
    let obs: Vec<Vec<Observation>> = (0..STRIPS)
        .map(|s| strip_observations(&ink, w, h, s * strip_w, (s + 1) * strip_w))
        .collect();

    let mut gaps: Vec<f64> = obs
        .iter()
        .flat_map(|o| o.windows(2).map(|p| p[1].y - p[0].y))
        .collect();
    if gaps.len() < 20 {
        return Err(format!(
            "{} line-to-line gaps observed; at least 20 needed to measure the line pitch",
            gaps.len()
        ));
    }
    gaps.sort_by(f64::total_cmp);
    let pitch = gaps[gaps.len() / 2];
    if pitch < 3.0 {
        return Err(format!(
            "line pitch {pitch:.1} px at the analysis scale; at least 3 px needed"
        ));
    }

    let centres: Vec<f64> = (0..STRIPS)
        .map(|s| to_f64(s * strip_w) + to_f64(strip_w) / 2.0)
        .collect();
    let raw = link(&obs, &centres, pitch);
    let min_points = (STRIPS / 6).max(5);
    let min_span = 0.25 * to_f64(w);
    let up = |v: f64| upscale_coord(v, factor);
    let tracks: Vec<Track> = raw
        .into_iter()
        .filter_map(|t| clean(t, pitch))
        .filter(|t| {
            t.points.len() >= min_points
                && t.points.last().map_or(0.0, |p| p[0]) - t.points[0][0] >= min_span
        })
        .map(|t| Track {
            points: t.points.iter().map(|p| [up(p[0]), up(p[1])]).collect(),
            weights: t.weights,
        })
        .collect();
    Ok(TextLines {
        tracks,
        pitch: pitch * to_f64(factor),
    })
}

/// Binary ink map (row-major, `true` = ink) by local-mean thresholding.
fn ink_map(plane: &GrayPage) -> Vec<bool> {
    let (w, h) = (plane.width, plane.height);
    let at = |x: usize, y: usize| plane.data[y * w + x];
    let mut integral = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row = 0u64;
        for x in 0..w {
            row += u64::from(at(x, y));
            integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row;
        }
    }
    let radius = (w / 50).max(6);
    let mut ink = vec![false; w * h];
    for y in 0..h {
        let (y0, y1) = (y.saturating_sub(radius), (y + radius + 1).min(h));
        for x in 0..w {
            let (x0, x1) = (x.saturating_sub(radius), (x + radius + 1).min(w));
            let sum = integral[y1 * (w + 1) + x1] + integral[y0 * (w + 1) + x0]
                - integral[y0 * (w + 1) + x1]
                - integral[y1 * (w + 1) + x0];
            let area = u64::try_from((y1 - y0) * (x1 - x0)).unwrap_or(u64::MAX);
            let v = u64::from(at(x, y));
            // v < 0.75 · mean  and  mean − v ≥ 24, in integers.
            ink[y * w + x] = 4 * v * area < 3 * sum && sum >= (v + 24) * area;
        }
    }
    ink
}

/// Line observations in the column range `[x0, x1)`.
fn strip_observations(ink: &[bool], w: usize, h: usize, x0: usize, x1: usize) -> Vec<Observation> {
    let profile: Vec<f64> = (0..h)
        .map(|y| to_f64(ink[y * w + x0..y * w + x1].iter().filter(|&&i| i).count()))
        .collect();
    let smooth: Vec<f64> = (0..h)
        .map(|y| {
            let (a, b) = (y.saturating_sub(1), (y + 2).min(h));
            profile[a..b].iter().sum::<f64>() / to_f64(b - a)
        })
        .collect();
    let threshold = (0.06 * to_f64(x1 - x0)).max(1.0);
    let max_height = (h / 20).max(3);
    let mut out = Vec::new();
    let mut y = 0;
    while y < h {
        if smooth[y] < threshold {
            y += 1;
            continue;
        }
        let start = y;
        while y < h && smooth[y] >= threshold {
            y += 1;
        }
        let height = y - start;
        if !(2..=max_height).contains(&height) {
            continue;
        }
        let mass: f64 = profile[start..y].iter().sum();
        if mass <= 0.0 {
            continue;
        }
        out.push(Observation {
            y: baseline(&profile[start..y]) + to_f64(start),
            mass,
        });
    }
    out
}

/// Baseline of one line run, relative to the run's first row: the lowest
/// position where the row ink count falls through half its peak, linearly
/// interpolated between rows.
///
/// The body of lowercase text is the densest band of a line and ends at
/// the baseline; descenders below it are sparse.  Punctuation-only runs
/// (dot leaders, rules) sit on the same baseline, whereas their ink
/// centroid would not coincide with that of the text beside them.
fn baseline(run: &[f64]) -> f64 {
    let peak = run.iter().copied().fold(0.0, f64::max);
    let half = 0.5 * peak;
    let r = run.iter().rposition(|&v| v >= half).unwrap_or(0);
    match run.get(r + 1) {
        Some(&below) if run[r] > below => to_f64(r) + (run[r] - half) / (run[r] - below),
        _ => to_f64(r) + 0.5,
    }
}

/// Link per-strip observations into tracks (analysis-plane coordinates).
fn link(obs: &[Vec<Observation>], centres: &[f64], pitch: f64) -> Vec<Track> {
    struct Open {
        track: Track,
        last_strip: usize,
        slope: f64,
    }
    let tol = 0.35 * pitch;
    let mut open: Vec<Open> = Vec::new();
    let mut done: Vec<Track> = Vec::new();
    for (s, strip) in obs.iter().enumerate() {
        let x = centres[s];
        // Retire tracks that can no longer bridge to this strip.
        let (alive, dead): (Vec<Open>, Vec<Open>) = open
            .into_iter()
            .partition(|o| s - o.last_strip <= MAX_GAP_STRIPS + 1);
        done.extend(dead.into_iter().map(|o| o.track));
        open = alive;

        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, o) in open.iter().enumerate() {
            let last = o.track.points[o.track.points.len() - 1];
            let predicted = o.slope.mul_add(x - last[0], last[1]);
            for (oi, ob) in strip.iter().enumerate() {
                let d = (ob.y - predicted).abs();
                if d <= tol {
                    pairs.push((d, ti, oi));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut track_used = vec![false; open.len()];
        let mut obs_used = vec![false; strip.len()];
        for (_, ti, oi) in pairs {
            if track_used[ti] || obs_used[oi] {
                continue;
            }
            track_used[ti] = true;
            obs_used[oi] = true;
            let o = &mut open[ti];
            o.track.points.push([x, strip[oi].y]);
            o.track.weights.push(strip[oi].mass);
            o.last_strip = s;
            o.slope = recent_slope(&o.track.points);
        }
        for (oi, ob) in strip.iter().enumerate() {
            if !obs_used[oi] {
                open.push(Open {
                    track: Track {
                        points: vec![[x, ob.y]],
                        weights: vec![ob.mass],
                    },
                    last_strip: s,
                    slope: 0.0,
                });
            }
        }
    }
    done.extend(open.into_iter().map(|o| o.track));
    done
}

/// Least-squares slope through the last (up to) six points.
fn recent_slope(points: &[[f64; 2]]) -> f64 {
    let tail = &points[points.len().saturating_sub(6)..];
    if tail.len() < 2 {
        return 0.0;
    }
    let n = to_f64(tail.len());
    let mx = tail.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = tail.iter().map(|p| p[1]).sum::<f64>() / n;
    let sxx: f64 = tail.iter().map(|p| (p[0] - mx).powi(2)).sum();
    let sxy: f64 = tail.iter().map(|p| (p[0] - mx) * (p[1] - my)).sum();
    if sxx > 0.0 { sxy / sxx } else { 0.0 }
}

/// Drop points farther than a quarter pitch from the line through their
/// neighbours (up to three on each side).  The criterion is local so that
/// lines bending sharply near a spine keep their ends, while a link that
/// jumped to an adjacent line (a full pitch away) is removed.  `None` when
/// fewer than four points remain.
fn clean(mut t: Track, pitch: f64) -> Option<Track> {
    for _ in 0..2 {
        if t.points.len() < 4 {
            return None;
        }
        let pts = &t.points;
        let keep: Vec<bool> = (0..pts.len())
            .map(|i| {
                let neighbours: Vec<[f64; 2]> = (i.saturating_sub(3)..(i + 4).min(pts.len()))
                    .filter(|&j| j != i)
                    .map(|j| pts[j])
                    .collect();
                local_line(&neighbours).is_none_or(|(c, m)| {
                    (m.mul_add(pts[i][0], c) - pts[i][1]).abs() <= 0.25 * pitch
                })
            })
            .collect();
        let mut i = 0;
        t.points.retain(|_| {
            i += 1;
            keep[i - 1]
        });
        let mut j = 0;
        t.weights.retain(|_| {
            j += 1;
            keep[j - 1]
        });
    }
    (t.points.len() >= 4).then_some(t)
}

/// Least-squares line `y = c + m·x` through `points` as `(c, m)`; `None`
/// for fewer than two distinct abscissae.
fn local_line(points: &[[f64; 2]]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return None;
    }
    let n = to_f64(points.len());
    let mx = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = points.iter().map(|p| p[1]).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p[0] - mx).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let m = points
        .iter()
        .map(|p| (p[0] - mx) * (p[1] - my))
        .sum::<f64>()
        / sxx;
    Some((m.mul_add(-mx, my), m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::curved_text;

    #[test]
    fn tracks_follow_curved_lines() {
        let bow = |x: f64| 25.0 * ((x - 600.0) / 600.0).powi(2);
        let p = curved_text(1200, 900, |x, _| bow(x));
        let lines = find(&p.view()).expect("lines");
        assert!((lines.pitch - 30.0).abs() < 2.0, "pitch {}", lines.pitch);
        assert!(lines.tracks.len() >= 20, "{} tracks", lines.tracks.len());
        for t in &lines.tracks {
            let first = t.points[0];
            let base = first[1] - bow(first[0]);
            for p in &t.points {
                let err = (p[1] - bow(p[0]) - base).abs();
                assert!(err < 2.0, "point {p:?} off its line by {err:.2}");
            }
        }
    }

    #[test]
    fn blank_page_has_no_lines() {
        let p = GrayPage {
            width: 1000,
            height: 800,
            data: vec![240; 800_000],
        };
        assert!(find(&p.view()).is_err());
    }
}
