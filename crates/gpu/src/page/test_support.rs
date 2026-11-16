//! Deterministic inputs shared by the page raster's tests.

use color::SampleDepth;

use super::{
    AreaPath, ClipId, ClipRegionId, CoverageRows, ImageDraw, ImageLayout, ImageMask, MaskBits,
    PageCmd, PageCommands, PixelBounds, PlaneId, PlaneValues, SampleMap, Samples, SamplesId,
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

/// The identity matrix.
pub(crate) const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// A random polygon of 3 to 11 vertices over a `width × height` page and a
/// margin around it, in device pixels.
pub(crate) fn random_path(rng: &mut Rng, width: u32, height: u32) -> raster::Path {
    let coord = |rng: &mut Rng, extent: u32| {
        f64::from(i32::try_from(rng.range(-40, i64::from(extent) * 8 + 40)).unwrap()) / 8.0
    };
    let mut builder = raster::PathBuilder::new();
    builder
        .move_to(coord(rng, width), coord(rng, height))
        .expect("move_to");
    for _ in 0..rng.range(2, 11) {
        builder
            .line_to(coord(rng, width), coord(rng, height))
            .expect("line_to");
    }
    builder.close(false).expect("close");
    builder.build()
}

/// An antialiased clip of the page `width × height` (its rectangle shrunk
/// by up to 3 pixels a side) intersected with `n` random polygons, each
/// non-zero or even-odd at random; from 8 polygons on, the CPU folds them.
pub(crate) fn random_clip(rng: &mut Rng, width: u32, height: u32, n: usize) -> raster::Clip {
    let inset = |rng: &mut Rng| f64::from(rng.below(4));
    let mut clip = raster::Clip::new(
        inset(rng),
        inset(rng),
        f64::from(width) - inset(rng),
        f64::from(height) - inset(rng),
        true,
    );
    for _ in 0..n {
        let path = random_path(rng, width, height);
        let xpath = raster::xpath::XPath::new(&path, &IDENTITY, 0.1, true);
        clip.clip_to_path(&xpath, rng.below(2) == 0);
    }
    clip
}

/// The path regions of `clip` held by `page` as a clip, `None` for a
/// rectangle-only clip.
pub(crate) fn page_clip(page: &mut PageCommands, clip: &raster::Clip) -> Option<ClipId> {
    let regions: Vec<ClipRegionId> = clip
        .path_regions()
        .map(|(_, table)| {
            let table = table.expect("an antialiased region");
            page.add_clip_region(CoverageRows {
                y_min: table.y_min(),
                row_starts: table.row_starts(),
                spans: table.span_triples(),
                bytes: table.bytes(),
            })
            .expect("add clip region")
        })
        .collect();
    (!regions.is_empty()).then(|| page.add_clip(&regions).expect("add clip"))
}

/// The clip rectangle's pixels as bounds.
pub(crate) const fn clip_bounds(clip: &raster::Clip) -> PixelBounds {
    PixelBounds {
        x0: clip.x_min_i,
        y0: clip.y_min_i,
        x1: clip.x_max_i,
        y1: clip.y_max_i,
    }
}

/// A random mask plane for a page of `depth`: one value outside a random
/// rectangle (0, M or any) and random values inside it, or random values
/// everywhere, or one value everywhere.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test pages are a few hundred pixels wide"
)]
pub(crate) fn random_plane(rng: &mut Rng, width: u32, height: u32, depth: SampleDepth) -> Vec<u16> {
    let max = depth.max_sample();
    let n = (width * height) as usize;
    let pick = |rng: &mut Rng| match rng.below(5) {
        0 => 0,
        1 => max,
        2 => max - 1,
        _ => rng.sample(max),
    };
    match rng.below(4) {
        0 => (0..n).map(|_| pick(rng)).collect(),
        1 => vec![pick(rng); n],
        _ => {
            let outside = pick(rng);
            let (x0, y0) = (rng.below(width), rng.below(height));
            let (x1, y1) = (
                x0 + rng.below(width - x0.unsigned_abs()),
                y0 + rng.below(height - y0.unsigned_abs()),
            );
            (0..n)
                .map(|i| {
                    let (x, y) = ((i % width as usize) as i32, (i / width as usize) as i32);
                    if (x0..=x1).contains(&x) && (y0..=y1).contains(&y) {
                        pick(rng)
                    } else {
                        outside
                    }
                })
                .collect()
        }
    }
}

/// `values` held by `page` as a mask plane of its depth.
pub(crate) fn page_plane(page: &mut PageCommands, values: &[u16]) -> PlaneId {
    match page.depth() {
        SampleDepth::Eight => {
            let v: Vec<u8> = values.iter().map(|&s| u8::try_from(s).unwrap()).collect();
            page.add_mask_plane(PlaneValues::Eight(&v))
        }
        SampleDepth::Sixteen => page.add_mask_plane(PlaneValues::Sixteen(values)),
    }
    .expect("add mask plane")
}

/// Push the antialiased fill of `path` that the CPU composites under
/// `clip`'s rectangle, as outlines, or as their tables when `tables`.
pub(crate) fn push_fill(
    page: &mut PageCommands,
    (path, even_odd): (&raster::Path, bool),
    clip: &raster::Clip,
    (rgb, alpha): ([u16; 3], u16),
    tables: bool,
) {
    let (width, height) = (page.width(), page.height());
    for o in
        raster::fill::aa_fill_outlines(path, &IDENTITY, 0.1, clip, width, height, even_odd, false)
    {
        let (x0, y0, x1, y1) = o.window;
        if tables {
            let table = o.table();
            let Some((x0, y0, x1, y1)) = table.bounds() else {
                continue;
            };
            let rows = CoverageRows {
                y_min: table.y_min(),
                row_starts: table.row_starts(),
                spans: table.span_triples(),
                bytes: table.bytes(),
            };
            page.push_coverage(rows, PixelBounds { x0, y0, x1, y1 }, rgb, alpha)
                .expect("push table");
        } else {
            let area = AreaPath {
                segs: &o.segs,
                window: PixelBounds { x0, y0, x1, y1 },
                even_odd: o.even_odd,
                weight: o.weight,
            };
            page.push_area_path(&area, rgb, alpha)
                .expect("push outline");
        }
    }
}

/// Push `n` random steps onto `page`: clip changes among clips of 1, 3
/// and 9 regions and none, soft-masked objects begun and ended over random
/// planes and rectangles, and fills (outlines unless `tables`), masks,
/// images and opaque runs at random alphas.
pub(crate) fn push_random_modified(rng: &mut Rng, page: &mut PageCommands, n: usize, tables: bool) {
    let (width, height, depth) = (page.width(), page.height(), page.depth());
    let max = depth.max_sample();
    let clips: Vec<Option<ClipId>> = [1, 3, 9]
        .into_iter()
        .map(|k| {
            let clip = random_clip(rng, width, height, k);
            page_clip(page, &clip)
        })
        .chain([None])
        .collect();
    let planes: Vec<PlaneId> = (0..2)
        .map(|_| {
            let values = random_plane(rng, width, height, depth);
            page_plane(page, &values)
        })
        .collect();
    let rect = raster::Clip::new(0.0, 0.0, f64::from(width), f64::from(height), true);
    let w = i32::try_from(width).unwrap();
    let h = i32::try_from(height).unwrap();
    for _ in 0..n {
        let alpha =
            [max, max, max / 2 + 1, 1, rng.sample(max)][usize::try_from(rng.below(5)).unwrap()];
        let rgb = [0; 3].map(|_| rng.sample(max));
        match rng.below(10) {
            0 => page.set_clip(clips[usize::try_from(rng.below(4)).unwrap()]),
            1 => {
                let plane = planes[usize::try_from(rng.below(2)).unwrap()];
                let (x0, y0) = (rng.below(width + 8) - 8, rng.below(height + 8) - 8);
                let bounds = PixelBounds {
                    x0,
                    y0,
                    x1: x0 + rng.below(width + 8),
                    y1: y0 + rng.below(height + 8),
                };
                page.begin_object(plane, bounds).expect("begin object");
            }
            2 => page.end_object(),
            3 => {
                let (mw, mh) = (
                    rng.below(40).unsigned_abs() + 1,
                    rng.below(30).unsigned_abs() + 1,
                );
                let data: Vec<u8> = (0..mw * mh).map(|_| rng.byte()).collect();
                let mask = page
                    .add_mask(&data, mw, mh, MaskBits::Coverage)
                    .expect("add mask");
                let origin = (rng.below(width + 20) - 20, rng.below(height + 20) - 20);
                let bounds = PixelBounds {
                    x0: 0,
                    y0: 0,
                    x1: w - 1,
                    y1: h - 1,
                };
                page.push_mask(mask, origin, bounds, rgb, alpha)
                    .expect("push mask");
            }
            4 => push_random_images(rng, page, 1),
            5 => {
                let y = rng.below(height);
                let x0 = rng.below(width + 5) - 5;
                page.push_opaque_runs(&mut [(y, x0, x0 + rng.below(40))], rgb)
                    .expect("push runs");
            }
            _ => {
                let path = random_path(rng, width, height);
                push_fill(
                    page,
                    (&path, rng.below(2) == 0),
                    &rect,
                    (rgb, alpha),
                    tables,
                );
            }
        }
    }
    page.end_object();
}
