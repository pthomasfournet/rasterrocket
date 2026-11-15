//! Background normalisation: a tiled estimate of the paper level, inverted
//! into a per-pixel gain that maps paper to white.
//!
//! # Model
//!
//! Uneven illumination multiplies the page by a slowly varying factor.  On
//! the 8-bit luma plane, divided into tiles of `tile_px × tile_px`:
//!
//! 1. A tile's background is the mean of its samples above its floor: the
//!    larger of [`FOREGROUND_FLOOR`] and [`RELATIVE_FLOOR`] of the mean of
//!    the samples above [`FOREGROUND_FLOOR`].  A tile in which fewer than
//!    one sample in [`HOLE_DIVISOR`] lies above its floor is a hole (ink, a
//!    figure) and takes the mean of its measured 8-neighbours, pass by
//!    pass, until no hole remains.
//! 2. The tile map is smoothed by a separable box of half-width
//!    [`SMOOTH_RADIUS`] tiles, truncated symmetrically at the map edges, so
//!    a map linear across the page is reproduced exactly.
//! 3. The map is interpolated bilinearly at every pixel centre (clamped to
//!    the outermost tile centres) and inverted into a gain
//!    `g = W / bg` with `W = 255 · WHITE_DEN / WHITE_NUM`: a sample at or
//!    above `WHITE_NUM / WHITE_DEN` of its local background becomes white,
//!    and darker samples keep their tone relative to it.
//!
//! The gain is measured on 8-bit luma and applied to any image of the same
//! size, at its own depth: every channel of a colour page, or the 16-bit
//! samples of a deep page.  Each output sample is `min(MAX, v · g)`.
//!
//! # Acceptance
//!
//! The measurement declines when the page spans fewer than [`MIN_TILES`]
//! tiles on either axis, when more than half the tiles are holes, or when
//! the background is already white (the 5th percentile of the smoothed map
//! at or above [`WHITE_LEVEL`]).
//!
//! # Limits
//!
//! A light photograph or tint is read as background: its low-frequency
//! shading is lost.  Paper tint in a colour page survives, since one gain
//! scales every channel.  Callers should apply the normalisation only to
//! pages without a text layer of their own.

use crate::Detected;
use crate::error::CameraError;
use crate::image::{GrayPage, GrayView};
use crate::sample::Sample;
use crate::stats::{histogram, percentile_ratio};

// ── Tied constants ──────────────────────────────────────────────────────────

/// Samples at or below this luma are foreground and excluded from tile
/// means.  It also bounds the background from below, hence the gain from
/// above (see [`MAX_GAIN`]).
const FOREGROUND_FLOOR: u8 = 80;
/// A tile's floor is at least this fraction `num / den` of the mean of its
/// samples above [`FOREGROUND_FLOOR`].  Not above `WHITE_NUM / WHITE_DEN`,
/// so every sample that becomes white counts toward the background.
const RELATIVE_FLOOR: (u8, u8) = (7, 8);
/// A tile is a hole when fewer than `1 / HOLE_DIVISOR` of its samples lie
/// above its floor.
const HOLE_DIVISOR: usize = 3;
/// Half-width, in tiles, of the box smoothing the tile map.
const SMOOTH_RADIUS: usize = 2;
/// Samples at or above `WHITE_NUM / WHITE_DEN` of the local background map
/// to white.
const WHITE_NUM: u64 = 15;
/// See [`WHITE_NUM`].
const WHITE_DEN: u64 = 16;
/// Minimum number of tiles along each axis.
const MIN_TILES: usize = 4;
/// The map percentile, as a fraction `num / den`, tested against
/// [`WHITE_LEVEL`].
const WHITE_PERCENTILE: (u64, u64) = (5, 100);
/// A background whose [`WHITE_PERCENTILE`] is at or above this level is
/// already white.
const WHITE_LEVEL: u8 = 250;
/// Fractional bits of background values.
const MAP_BITS: u32 = 4;
/// Fractional bits of gains.
const GAIN_BITS: u32 = 12;
/// Fractional bits of bilinear interpolation weights.
const WEIGHT_BITS: u32 = 8;

/// Largest background value, `255` in [`MAP_BITS`] fixed point.
const MAP_MAX: u32 = 255 << MAP_BITS;
/// Smallest background value: the floor in [`MAP_BITS`] fixed point.
const MAP_MIN: u32 = (FOREGROUND_FLOOR as u32) << MAP_BITS;
/// `2^GAIN_BITS · W` in [`MAP_BITS`] fixed point: the numerator of every
/// gain.
const GAIN_NUMERATOR: u64 = ((1u64 << GAIN_BITS) * 255 * WHITE_DEN) << MAP_BITS;
/// The gain at the smallest background, the largest gain.
const MAX_GAIN: u64 = GAIN_NUMERATOR.div_ceil(WHITE_NUM * MAP_MIN as u64);

const _: () = assert!(
    MAX_GAIN <= u16::MAX as u64,
    "the gain at the foreground floor must fit a u16"
);
const _: () = assert!(WHITE_NUM < WHITE_DEN && WHITE_PERCENTILE.0 <= WHITE_PERCENTILE.1);
const _: () = assert!(
    RELATIVE_FLOOR.0 as u64 * WHITE_DEN <= WHITE_NUM * RELATIVE_FLOOR.1 as u64,
    "the relative floor must not exceed the white fraction"
);

/// The tile size for a page rendered at `dpi`: one eighth of an inch, and
/// at least 8 pixels.
#[must_use]
pub const fn background_tile_px(dpi: u32) -> usize {
    let px = dpi / 8;
    if px < 8 { 8 } else { px as usize }
}

/// A per-pixel gain that maps the measured paper level to white.
///
/// Measured by [`measure_background`]; applied by
/// [`normalize`](Self::normalize).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Background {
    width: usize,
    height: usize,
    /// Gain per pixel in [`GAIN_BITS`] fixed point, row-major.
    gain: Vec<u16>,
}

/// Measure the paper level of the 8-bit luma `img` in tiles of
/// `tile_px × tile_px` (see [`background_tile_px`]).
///
/// Declines when the page spans fewer than 4 tiles on either axis, when
/// more than half the tiles are mostly foreground, or when the background
/// is already white.
///
/// # Errors
///
/// [`CameraError`] when `img` is malformed, or
/// [`CameraError::ZeroSize`] when `tile_px` is 0.
pub fn measure_background(
    img: GrayView<'_>,
    tile_px: usize,
) -> Result<Detected<Background>, CameraError> {
    img.validate()?;
    if tile_px == 0 {
        return Err(CameraError::ZeroSize {
            parameter: "background tile size",
        });
    }
    Ok(Detected::from_result(measure(&img, tile_px)))
}

impl Background {
    /// Apply the gain to `img`: each sample `v` becomes `min(MAX, v · g)`.
    /// `img` is the measured luma itself, one channel of the colour image
    /// it was taken from, or either at 16 bits per sample.
    ///
    /// # Errors
    ///
    /// [`CameraError`] when `img` is malformed, or
    /// [`CameraError::SizeMismatch`] when its size differs from the
    /// measured image.
    pub fn normalize<S: Sample>(&self, img: GrayView<'_, S>) -> Result<GrayPage<S>, CameraError> {
        img.validate_size((self.width, self.height))?;
        let mut data = Vec::with_capacity(self.width * self.height);
        let round = 1u32 << (GAIN_BITS - 1);
        for (y, gains) in self.gain.chunks_exact(self.width).enumerate() {
            // v · g ≤ 65 535 · MAX_GAIN < 2³², so the product fits u32.
            data.extend(
                img.row(y).iter().zip(gains).map(|(&v, &g)| {
                    S::saturate_u32((v.to_u32() * u32::from(g) + round) >> GAIN_BITS)
                }),
            );
        }
        Ok(GrayPage {
            width: self.width,
            height: self.height,
            data,
        })
    }
}

// ── Measurement ─────────────────────────────────────────────────────────────

/// Tile boundaries along an axis of `len` pixels: `n + 1` edges at
/// multiples of `tile`, the last tile absorbing the remainder.  Caller
/// guarantees `n = len / tile ≥ 1`.
fn tile_edges(len: usize, tile: usize, n: usize) -> Vec<usize> {
    let mut edges: Vec<usize> = (0..n).map(|i| i * tile).collect();
    edges.push(len);
    edges
}

fn measure(img: &GrayView<'_>, tile: usize) -> Result<Background, String> {
    let (nx, ny) = (img.width / tile, img.height / tile);
    if nx < MIN_TILES || ny < MIN_TILES {
        return Err(format!(
            "image is {}×{}, {nx}×{ny} tiles of {tile} px; background measurement needs \
             at least {MIN_TILES}×{MIN_TILES}",
            img.width, img.height
        ));
    }
    let (xs, ys) = (
        tile_edges(img.width, tile, nx),
        tile_edges(img.height, tile, ny),
    );
    let mut map = tile_means(img, &xs, &ys);
    let holes = map.iter().filter(|m| m.is_none()).count();
    if holes * 2 > map.len() {
        return Err(format!(
            "{holes} of {} tiles are mostly foreground; at most half may be",
            map.len()
        ));
    }
    fill_holes(&mut map, nx, ny);
    let mut map: Vec<u32> = map.into_iter().map(|m| m.unwrap_or(MAP_MAX)).collect();
    smooth(&mut map, nx, ny);

    let levels = histogram(
        map.iter()
            .map(|&m| u8::try_from((m + (1 << (MAP_BITS - 1))) >> MAP_BITS).unwrap_or(u8::MAX)),
    );
    let low = percentile_ratio(&levels, WHITE_PERCENTILE.0, WHITE_PERCENTILE.1);
    if low >= WHITE_LEVEL {
        return Err(format!(
            "background is already white: its {}th percentile is {low}, at least {WHITE_LEVEL}",
            WHITE_PERCENTILE.0
        ));
    }
    Ok(Background {
        width: img.width,
        height: img.height,
        gain: gain_map(&map, &xs, &ys),
    })
}

/// The background of each tile in [`MAP_BITS`] fixed point, `None` for a
/// hole; row-major over the tiles.
///
/// A tile's floor is the larger of [`FOREGROUND_FLOOR`] and
/// [`RELATIVE_FLOOR`] of the mean of its samples above
/// [`FOREGROUND_FLOOR`]; its background is the mean of its samples above
/// that floor.  The relative floor excludes the partial-coverage edges of
/// ink, which lie between ink and paper.
fn tile_means(img: &GrayView<'_>, xs: &[usize], ys: &[usize]) -> Vec<Option<u32>> {
    let nx = xs.len() - 1;
    let mut map = Vec::with_capacity(nx * (ys.len() - 1));
    let mut floors = vec![FOREGROUND_FLOOR; nx];
    for band in ys.windows(2) {
        floors.fill(FOREGROUND_FLOOR);
        let coarse = band_sums(img, (band[0], band[1]), xs, &floors);
        for (floor, &(sum, count)) in floors.iter_mut().zip(&coarse) {
            if let Some(mean) = mean_of(sum, count, 0) {
                let relative = mean * u128::from(RELATIVE_FLOOR.0) / u128::from(RELATIVE_FLOOR.1);
                *floor = (*floor).max(u8::try_from(relative).unwrap_or(u8::MAX));
            }
        }
        let fine = band_sums(img, (band[0], band[1]), xs, &floors);
        for (span, &(sum, count)) in xs.windows(2).zip(&fine) {
            let area = (span[1] - span[0]) * (band[1] - band[0]);
            map.push(
                (count >= area.div_ceil(HOLE_DIVISOR))
                    .then(|| mean_of(sum, count, MAP_BITS))
                    .flatten()
                    .map(|m| u32::try_from(m).unwrap_or(MAP_MAX)),
            );
        }
    }
    map
}

/// The rounded mean `sum / count` in `bits` fixed point; `None` for an
/// empty count.
fn mean_of(sum: u128, count: usize, bits: u32) -> Option<u128> {
    let n = u128::try_from(count).ok().filter(|&n| n > 0)?;
    Some(((sum << bits) + n / 2) / n)
}

/// The sum and count of the samples above `floors[t]` in each tile `t` of
/// the rows `[y0, y1)`.  A row segment sums to at most `255 · 2³⁰` (`u64`);
/// a tile, to at most 255 times the image area (`u128`).
fn band_sums(
    img: &GrayView<'_>,
    (y0, y1): (usize, usize),
    xs: &[usize],
    floors: &[u8],
) -> Vec<(u128, usize)> {
    let mut acc = vec![(0u128, 0usize); floors.len()];
    for y in y0..y1 {
        let row = img.row(y);
        for ((total, n), (span, &floor)) in acc.iter_mut().zip(xs.windows(2).zip(floors)) {
            let (mut sum, mut count) = (0u64, 0usize);
            for &v in row[span[0]..span[1]].iter().filter(|&&v| v > floor) {
                sum += u64::from(v);
                count += 1;
            }
            *total += u128::from(sum);
            *n += count;
        }
    }
    acc
}

/// Replace every hole by the rounded mean of its measured 8-neighbours, one
/// pass at a time; a tile filled in a pass is measured from the next pass
/// on.  Caller guarantees at least one measured tile.
fn fill_holes(map: &mut [Option<u32>], nx: usize, ny: usize) {
    loop {
        let filled: Vec<(usize, u32)> = (0..map.len())
            .filter(|&i| map[i].is_none())
            .filter_map(|i| {
                let (x, y) = (i % nx, i / nx);
                let (mut sum, mut n) = (0u32, 0u32);
                for yy in y.saturating_sub(1)..(y + 2).min(ny) {
                    for xx in x.saturating_sub(1)..(x + 2).min(nx) {
                        if let Some(v) = map[yy * nx + xx] {
                            sum += v;
                            n += 1;
                        }
                    }
                }
                (n > 0).then(|| (i, (sum + n / 2) / n))
            })
            .collect();
        if filled.is_empty() {
            return;
        }
        for (i, v) in filled {
            map[i] = Some(v);
        }
    }
}

/// Smooth `map` in place by a separable box of half-width
/// [`SMOOTH_RADIUS`]; at the map edges the half-width shrinks to the
/// distance to the edge, so the box stays centred.
fn smooth(map: &mut [u32], nx: usize, ny: usize) {
    let mut line = Vec::with_capacity(nx.max(ny));
    for y in 0..ny {
        line.clear();
        line.extend_from_slice(&map[y * nx..(y + 1) * nx]);
        for (x, out) in map[y * nx..(y + 1) * nx].iter_mut().enumerate() {
            *out = centred_mean(&line, x);
        }
    }
    for x in 0..nx {
        line.clear();
        line.extend((0..ny).map(|y| map[y * nx + x]));
        for y in 0..ny {
            map[y * nx + x] = centred_mean(&line, y);
        }
    }
}

/// The rounded mean of `line[i − r ..= i + r]`, `r` the smaller of
/// [`SMOOTH_RADIUS`] and the distance from `i` to either end.
fn centred_mean(line: &[u32], i: usize) -> u32 {
    let r = SMOOTH_RADIUS.min(i).min(line.len() - 1 - i);
    let window = &line[i - r..=i + r];
    let n = u32::try_from(window.len()).unwrap_or(u32::MAX);
    (window.iter().sum::<u32>() + n / 2) / n
}

/// Bilinear interpolation weights along one axis: for every pixel, the
/// tiles `(i0, i1)` whose centres bracket its centre and the weight of
/// `i1` in [`WEIGHT_BITS`] fixed point.  Beyond the outermost centres the
/// nearest tile takes the whole weight.
fn axis_weights(edges: &[usize]) -> Vec<(usize, usize, u32)> {
    let len = edges[edges.len() - 1];
    // Doubled coordinates: the centre of tile i is (e_i + e_{i+1} − 1) / 2.
    let centres: Vec<usize> = edges.windows(2).map(|e| e[0] + e[1] - 1).collect();
    let last = centres.len() - 1;
    let one = 1u64 << WEIGHT_BITS;
    let mut i = 0;
    (0..len)
        .map(|p| {
            let p2 = 2 * p;
            while i < last && centres[i + 1] <= p2 {
                i += 1;
            }
            if p2 <= centres[0] {
                (0, 0, 0)
            } else if i == last {
                (last, last, 0)
            } else {
                let span = u64::try_from(centres[i + 1] - centres[i]).unwrap_or(u64::MAX);
                let off = u64::try_from(p2 - centres[i]).unwrap_or(0);
                let w = (off * one + span / 2) / span;
                (i, i + 1, u32::try_from(w).unwrap_or(0))
            }
        })
        .collect()
}

/// The per-pixel gain of the smoothed tile map `map`.
fn gain_map(map: &[u32], xs: &[usize], ys: &[usize]) -> Vec<u16> {
    let nx = xs.len() - 1;
    let lut: Vec<u16> = (0..=MAP_MAX)
        .map(|bg| {
            let den = WHITE_NUM * u64::from(bg.max(MAP_MIN));
            u16::try_from(GAIN_NUMERATOR.div_ceil(den)).unwrap_or(u16::MAX)
        })
        .collect();
    let (cols, rows) = (axis_weights(xs), axis_weights(ys));
    let one = 1u32 << WEIGHT_BITS;
    let round = 1u32 << (2 * WEIGHT_BITS - 1);
    let mut gain = Vec::with_capacity(cols.len() * rows.len());
    for &(y0, y1, wy) in &rows {
        let (top, bottom) = (&map[y0 * nx..(y0 + 1) * nx], &map[y1 * nx..(y1 + 1) * nx]);
        gain.extend(cols.iter().map(|&(x0, x1, wx)| {
            // Each term is at most MAP_MAX · 2^(2·WEIGHT_BITS) < 2³².
            let upper = top[x0] * (one - wx) + top[x1] * wx;
            let lower = bottom[x0] * (one - wx) + bottom[x1] * wx;
            let bg = (upper * (one - wy) + lower * wy + round) >> (2 * WEIGHT_BITS);
            lut[usize::try_from(bg.min(MAP_MAX)).unwrap_or(0)]
        }));
    }
    gain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{digest, plane, ramp_lit};

    fn normalized(page: &GrayPage, tile: usize) -> GrayPage {
        let Detected::Found(bg) = measure_background(page.view(), tile).expect("valid") else {
            panic!("ramp-lit page is measured");
        };
        bg.normalize(page.view()).expect("same size")
    }

    #[test]
    fn ramp_lit_page_thresholds_to_its_ink() {
        let (lit, ink) = ramp_lit(1200, 1000);
        let black = |p: &GrayPage| p.data.iter().map(|&v| v < 128).collect::<Vec<_>>();
        assert_ne!(black(&lit), ink, "the raw page does not threshold cleanly");
        let out = normalized(&lit, 32);
        assert_eq!(black(&out), ink);
        for (&v, &is_ink) in out.data.iter().zip(&ink) {
            if !is_ink {
                assert_eq!(v, 255);
            }
        }
        let got = digest(std::slice::from_ref(&out));
        assert_eq!(got, GOLDEN, "{got:#x}");
    }

    const GOLDEN: u64 = 0x42f4_c212_3b02_5b60;

    #[test]
    fn normalized_page_declines() {
        let (lit, _) = ramp_lit(1200, 1000);
        let out = normalized(&lit, 32);
        let again = measure_background(out.view(), 32).expect("valid");
        assert!(
            matches!(&again, Detected::Declined(r) if r.contains("already white")),
            "{again:?}"
        );
    }

    #[test]
    fn declines_small_dark_and_white_pages() {
        let small = plane(100, 100, |_, _| 200);
        assert!(matches!(
            measure_background(small.view(), 32),
            Ok(Detected::Declined(r)) if r.contains("tiles")
        ));
        let dark = plane(256, 256, |x, _| if x < 160 { 20 } else { 200 });
        assert!(matches!(
            measure_background(dark.view(), 32),
            Ok(Detected::Declined(r)) if r.contains("mostly foreground")
        ));
        let white = plane(256, 256, |_, _| 252);
        assert!(matches!(
            measure_background(white.view(), 32),
            Ok(Detected::Declined(r)) if r.contains("already white")
        ));
        assert!(matches!(
            measure_background(white.view(), 0),
            Err(CameraError::ZeroSize { .. })
        ));
    }

    #[test]
    fn holes_take_their_neighbours_level() {
        // A 64 px black square in a 4 × 4-tile region of paper at 200.
        let img = plane(256, 256, |x, y| {
            if (96..160).contains(&x) && (96..160).contains(&y) {
                0
            } else {
                200
            }
        });
        let out = normalized(&img, 32);
        assert!(out.data.iter().all(|&v| v == 0 || v == 255));
        assert_eq!(out.data[128 * 256 + 128], 0);
    }

    #[test]
    fn deep_and_colour_paths_follow_the_luma() {
        let (lit, _) = ramp_lit(600, 500);
        let Detected::Found(bg) = measure_background(lit.view(), 24).expect("valid") else {
            panic!("measured");
        };
        let gray = bg.normalize(lit.view()).expect("same size");
        // A colour page whose channels equal its luma normalises, channel by
        // channel, to the gray result.
        for channel in [&lit, &lit, &lit] {
            assert_eq!(bg.normalize(channel.view()).expect("same size"), gray);
        }
        let wide = GrayPage::<u16> {
            width: lit.width,
            height: lit.height,
            data: lit.data.iter().map(|&v| u16::from(v) * 257).collect(),
        };
        let deep = bg.normalize(wide.view()).expect("same size");
        for (&d, &n) in deep.data.iter().zip(&gray.data) {
            let level = (u32::from(d) + 128) / 257;
            assert!(level.abs_diff(u32::from(n)) <= 1, "{d} vs {n}");
        }
        let other = plane(10, 10, |_, _| 0);
        assert!(matches!(
            bg.normalize(other.view()),
            Err(CameraError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn smoothing_keeps_a_linear_map() {
        let line: Vec<u32> = (0..9).map(|i| 1000 + 37 * i).collect();
        for i in 0..line.len() {
            assert_eq!(centred_mean(&line, i), line[i]);
        }
    }

    #[test]
    fn tile_size_follows_resolution() {
        assert_eq!(background_tile_px(0), 8);
        assert_eq!(background_tile_px(72), 9);
        assert_eq!(background_tile_px(300), 37);
    }
}
