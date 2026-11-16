//! Deterministic inputs shared by the page raster's tests.

use color::SampleDepth;

use super::{
    ImageDraw, ImageLayout, ImageMask, PageCmd, PageCommands, PixelBounds, SampleMap, Samples,
    SamplesId,
};

/// `SplitMix64`.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + i64::try_from(self.next() % u64::try_from(hi - lo).unwrap()).unwrap()
    }

    /// Uniform in `[0, n)`.
    pub(crate) fn below(&mut self, n: u32) -> i32 {
        i32::try_from(self.next() % u64::from(n)).unwrap()
    }

    pub(crate) fn byte(&mut self) -> u8 {
        u8::try_from(self.below(256)).unwrap()
    }

    /// A sample of at most `max`.
    pub(crate) fn sample(&mut self, max: u16) -> u16 {
        u16::try_from(self.below(u32::from(max) + 1)).unwrap()
    }
}

/// The closed polygon through `points`, as segments.
pub(crate) fn polygon(points: &[(i64, i64)]) -> Vec<[i64; 4]> {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            [a.0, a.1, b.0, b.1]
        })
        .collect()
}

/// A rectangle 270 × 173 pixels from (4, 3), fixed point: inside it, its
/// tiles hold a backdrop equal on every row.
pub(crate) fn wide_rectangle() -> Vec<[i64; 4]> {
    polygon(&[
        (1_000, 700),
        (70_000, 700),
        (70_000, 45_000),
        (1_000, 45_000),
    ])
}

/// Twelve steep slivers side by side over a 300 × 200 pixel page, fixed
/// point: most of their tiles list one segment.
pub(crate) fn slivers() -> Vec<[i64; 4]> {
    (0..12)
        .flat_map(|k| {
            let x = 3_000 + k * 5_500;
            polygon(&[(x, 300), (x + 900, 300), (x + 4_000, 50_000)])
        })
        .collect()
}

/// One `[x_hi, x_lo, y_hi, y_lo]` sampling table entry of the source
/// coordinate `(x, y)`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test coordinates are small; the fraction is in [0, 2^32)"
)]
pub(crate) fn table_entry(x: f64, y: f64) -> [u32; 4] {
    let split = |v: f64| {
        let hi = v.floor();
        (
            (hi as i32).cast_unsigned(),
            ((v - hi) * 4_294_967_296.0) as u32,
        )
    };
    let (xh, xl) = split(x);
    let (yh, yl) = split(y);
    [xh, xl, yh, yl]
}

/// Sampling tables over `x0..x1 × y0..y1` of the affine source map
/// `(X, Y) = (ax·u + bx·v + cx, ay·u + by·v + cy)` at pixel centres
/// `(u, v)`: the column terms carry `u`, the row terms `v` and the offset.
pub(crate) fn affine_tables(
    (x0, y0, x1, y1): (i32, i32, i32, i32),
    [ax, bx, cx, ay, by, cy]: [f64; 6],
) -> (Vec<u32>, Vec<u32>) {
    let cols = (x0..x1)
        .flat_map(|x| {
            let u = f64::from(x) + 0.5;
            table_entry(ax * u, ay * u)
        })
        .collect();
    let rows = (y0..y1)
        .flat_map(|y| {
            let v = f64::from(y) + 0.5;
            table_entry(bx.mul_add(v, cx), by.mul_add(v, cy))
        })
        .collect();
    (cols, rows)
}

/// A random affine source map of a `w × h` image over the rectangle
/// `rect`: scaled up or down, flipped, a quarter turn, or rotated and
/// skewed.
pub(crate) fn random_map(rng: &mut Rng, rect: (i32, i32, i32, i32), w: u32, h: u32) -> [f64; 6] {
    let (x0, y0, x1, y1) = rect;
    let (dw, dh) = (f64::from(x1 - x0), f64::from(y1 - y0));
    let (sx, sy) = (f64::from(w) / dw, f64::from(h) / dh);
    let jitter = |rng: &mut Rng| f64::from(rng.below(2001)) / 1000.0 - 1.0;
    let kind = rng.below(5);
    let (ax, bx, ay, by) = match kind {
        0 => (sx, 0.0, 0.0, sy),
        1 => (-sx, 0.0, 0.0, -sy),
        2 => (0.0, f64::from(w) / dh, f64::from(h) / dw, 0.0),
        3 => (sx, 0.3 * sx * jitter(rng), 0.3 * sy * jitter(rng), sy),
        _ => (
            sx * 0.5f64.mul_add(jitter(rng), 1.0),
            0.0,
            0.0,
            sy * (1.0 + jitter(rng)),
        ),
    };
    let (u0, v0) = (f64::from(x0) + 0.5, f64::from(y0) + 0.5);
    let off = |a: f64, b: f64, n: f64, rng: &mut Rng| {
        let start = if a + b < 0.0 { n } else { 0.0 };
        (0.2 * n).mul_add(jitter(rng), b.mul_add(-v0, a.mul_add(-u0, start)))
    };
    let cx = off(ax, bx, f64::from(w), rng);
    let cy = off(ay, by, f64::from(h), rng);
    [ax, bx, cx, ay, by, cy]
}

/// The bytes of the device inputs of `page` as [`PageCommands::bin`] and
/// [`PageCommands::table_parts`] produce them.
pub(crate) fn binned_bytes(page: &PageCommands) -> usize {
    let bins = page.bin().expect("bin");
    let (parts, _) = page.table_parts();
    page.len() * size_of::<PageCmd>()
        + parts.iter().map(|p| p.len()).sum::<usize>()
        + size_of_val(page.row_starts())
        + size_of_val(bins.offsets.as_slice())
        + size_of_val(bins.items.as_slice())
}

/// `n` random `w × h` samples of `channels` channels for a page of
/// `depth`: 16-bit ones on a 16-bit page half the time.
fn random_samples(
    rng: &mut Rng,
    page: &mut PageCommands,
    (w, h, channels): (u32, u32, u32),
) -> SamplesId {
    let n = (w * h * channels) as usize;
    if page.depth() == SampleDepth::Sixteen && rng.below(2) == 0 {
        let s: Vec<u16> = (0..n).map(|_| rng.sample(u16::MAX)).collect();
        page.add_samples(Samples::Sixteen(&s), w, h, channels)
    } else {
        let s: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
        page.add_samples(Samples::Eight(&s), w, h, channels)
    }
    .expect("add samples")
}

/// Push `n` random image draws onto `page`: gray, RGB and stencil images
/// of 1 to 40 samples a side over random rectangles partly off the page,
/// through random maps and filters, without a mask, with a mask read
/// through the image's taps or through tables of its own, composited
/// directly or through the pipe at random alphas.
pub(crate) fn push_random_images(rng: &mut Rng, page: &mut PageCommands, count: usize) {
    let max = page.depth().max_sample();
    let (width, height) = (i64::from(page.width()), i64::from(page.height()));
    let side = |rng: &mut Rng| u32::try_from(rng.range(1, 41)).unwrap();
    let coord = |rng: &mut Rng, extent: i64| i32::try_from(rng.range(-30, extent + 10)).unwrap();
    let rect = |rng: &mut Rng| {
        let (x0, y0) = (coord(rng, width), coord(rng, height));
        let span = |rng: &mut Rng, extent: i64| i32::try_from(rng.range(1, extent + 30)).unwrap();
        (x0, y0, x0 + span(rng, width), y0 + span(rng, height))
    };
    for _ in 0..count {
        let layout = [ImageLayout::Gray, ImageLayout::Rgb, ImageLayout::Stencil]
            [usize::try_from(rng.below(3)).unwrap()];
        let channels = if layout == ImageLayout::Rgb { 3 } else { 1 };
        let (sw, sh) = (side(rng), side(rng));
        let samples = random_samples(rng, page, (sw, sh, channels));
        let area = rect(rng);
        let (cols, rows) = affine_tables(area, random_map(rng, area, sw, sh));
        let filter = |rng: &mut Rng| rng.below(2) == 0;
        let map = SampleMap {
            x0: area.0,
            y0: area.1,
            x1: area.2,
            y1: area.3,
            cols: &cols,
            rows: &rows,
            filter_x: filter(rng),
            filter_y: filter(rng),
        };
        let mr = rect(rng);
        let (mw, mh) = (side(rng), side(rng));
        let (mcols, mrows) = affine_tables(mr, random_map(rng, mr, mw, mh));
        let mask = match (layout, rng.below(3)) {
            (ImageLayout::Stencil, _) | (_, 0) => ImageMask::None,
            (_, 1) => ImageMask::Taps(random_samples(rng, page, (sw, sh, 1))),
            _ => ImageMask::Own(
                random_samples(rng, page, (mw, mh, 1)),
                SampleMap {
                    x0: mr.0,
                    y0: mr.1,
                    x1: mr.2,
                    y1: mr.3,
                    cols: &mcols,
                    rows: &mrows,
                    filter_x: filter(rng),
                    filter_y: filter(rng),
                },
            ),
        };
        let paint = rect(rng);
        let a_input =
            [max, max - 1, 1, max / 2, rng.sample(max)][usize::try_from(rng.below(5)).unwrap()];
        page.push_image(&ImageDraw {
            samples,
            layout,
            map,
            mask,
            bounds: PixelBounds {
                x0: paint.0,
                y0: paint.1,
                x1: paint.2,
                y1: paint.3,
            },
            colour: [0; 3].map(|_| rng.sample(max)),
            a_input,
            direct: rng.below(2) == 0,
        })
        .expect("push image");
    }
}
