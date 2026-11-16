use color::SampleDepth;
use raster::coverage::AreaOutline;

use super::super::reference::render_page;
use super::super::{PageCommands, PageCommandsError, PixelBounds};
use super::{AreaPath, PATH_COORD_LIMIT, PATH_TILE_SEGMENTS};

/// `SplitMix64`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + i64::try_from(self.next() % u64::try_from(hi - lo).unwrap()).unwrap()
    }
}

/// The closed polygon through `points`, as segments.
fn polygon(points: &[(i64, i64)]) -> Vec<[i64; 4]> {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            [a.0, a.1, b.0, b.1]
        })
        .collect()
}

/// The coverage of every pixel of a `width × height` page that the device
/// raster's host model gives `outline` (black at full alpha over white
/// paper, so a pixel's value is 255 minus its coverage), or why the
/// outline cannot be pushed.
fn device_coverage(
    outline: &AreaOutline,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, PageCommandsError> {
    let (x0, y0, x1, y1) = outline.window;
    let mut page = PageCommands::new(width, height, SampleDepth::Eight, [255; 3]);
    let path = AreaPath {
        segs: &outline.segs,
        window: PixelBounds { x0, y0, x1, y1 },
        even_odd: outline.even_odd,
        weight: outline.weight,
    };
    page.push_area_path(&path, [0; 3], 255)?;
    let pixels = render_page(&page)?;
    Ok(pixels
        .chunks(3)
        .map(|p| 255 - u8::try_from(p[0]).unwrap())
        .collect())
}

/// The coverage of every pixel of the page that the CPU's rasteriser
/// gives `outline`.
fn cpu_coverage(outline: &AreaOutline, width: u32, height: u32) -> Vec<u8> {
    let table = outline.table();
    let (w, h) = (
        i32::try_from(width).unwrap(),
        i32::try_from(height).unwrap(),
    );
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| table.coverage(x, y))
        .collect()
}

fn assert_same(outline: &AreaOutline, width: u32, height: u32, what: &str) {
    let device = device_coverage(outline, width, height).expect("representable");
    let cpu = cpu_coverage(outline, width, height);
    if let Some(i) = (0..cpu.len()).find(|&i| cpu[i] != device[i]) {
        let w = width as usize;
        panic!(
            "{what}: pixel ({}, {}) has coverage {} on the device model, {} on the CPU",
            i % w,
            i / w,
            device[i],
            cpu[i]
        );
    }
}

/// A random window on a `width × height` page.
fn window(rng: &mut Rng, width: u32, height: u32) -> (i32, i32, i32, i32) {
    let (w, h) = (i64::from(width), i64::from(height));
    let (x0, y0) = (rng.range(0, w), rng.range(0, h));
    let (x1, y1) = (rng.range(x0, w), rng.range(y0, h));
    let c = |v: i64| i32::try_from(v).unwrap();
    (c(x0), c(y0), c(x1), c(y1))
}

#[test]
fn random_polygons_match_the_cpu_rasteriser() {
    let mut rng = Rng(0x0A7E_A5EE_D001);
    let (width, height) = (83, 61);
    for case in 0..400 {
        let n = usize::try_from(rng.range(3, 14)).unwrap();
        // Mostly on the page, some vertices far off it.
        let points: Vec<(i64, i64)> = (0..n)
            .map(|_| {
                let far = rng.range(0, 8) == 0;
                let (lo, hi) = if far {
                    (-40_000, 60_000)
                } else {
                    (-2_000, 24_000)
                };
                (rng.range(lo, hi), rng.range(lo, hi))
            })
            .collect();
        let mut segs = polygon(&points);
        if case % 3 == 0 {
            // A second subpath, possibly overlapping the first.
            let m = usize::try_from(rng.range(3, 8)).unwrap();
            let more: Vec<(i64, i64)> = (0..m)
                .map(|_| (rng.range(0, 21_000), rng.range(0, 16_000)))
                .collect();
            segs.extend(polygon(&more));
        }
        let window = if case % 2 == 0 {
            (0, 0, 82, 60)
        } else {
            window(&mut rng, width, height)
        };
        let outline = AreaOutline {
            window,
            segs,
            even_odd: case % 4 >= 2,
            weight: [255, 63, 200][case % 3],
        };
        assert_same(&outline, width, height, &format!("case {case}"));
    }
}

#[test]
fn axis_aligned_and_tile_boundary_edges_match_the_cpu_rasteriser() {
    // Edges on pixel and tile boundaries, vertical edges at a tile's left
    // side, and degenerate (horizontal, zero-length) segments.
    let t = 16 * 256;
    let shapes: [&[(i64, i64)]; 4] = [
        &[(t, 0), (2 * t, 0), (2 * t, 3 * t), (t, 3 * t)],
        &[
            (t - 1, 5),
            (3 * t + 1, 5),
            (3 * t + 1, 2 * t),
            (t - 1, 2 * t),
        ],
        &[(0, 0), (4 * t, t), (0, 2 * t), (0, 2 * t), (2 * t, t)],
        &[(7, 9), (7, 9), (t, 3 * t + 17), (2, 2 * t)],
    ];
    for (i, points) in shapes.iter().enumerate() {
        for even_odd in [false, true] {
            let outline = AreaOutline {
                window: (0, 0, 70, 55),
                segs: polygon(points),
                even_odd,
                weight: 255,
            };
            assert_same(&outline, 71, 56, &format!("shape {i}"));
        }
    }
}

#[test]
fn coordinates_near_the_limit_match_and_past_it_are_refused() {
    let near = PATH_COORD_LIMIT - 1;
    let points = [(-near, 100), (near, 7_000), (3_000, -near), (2_000, near)];
    let outline = AreaOutline {
        window: (0, 0, 47, 39),
        segs: polygon(&points),
        even_odd: false,
        weight: 255,
    };
    assert_same(&outline, 48, 40, "near the limit");
    let mut past = outline;
    past.segs[0][0] = -PATH_COORD_LIMIT;
    assert_eq!(
        device_coverage(&past, 48, 40),
        Err(PageCommandsError::Unrepresentable)
    );
}

/// A closed zig-zag of `n` segments inside the tile at the page's origin.
fn zig_zag(n: usize) -> Vec<[i64; 4]> {
    let mut points: Vec<(i64, i64)> = (0..n - 1)
        .map(|i| {
            let i = i64::try_from(i).unwrap();
            (300 + (i * 37) % 3_500, 200 + (i * 53) % 3_600)
        })
        .collect();
    points.push((3_900, 3_900));
    polygon(&points)
}

#[test]
fn the_most_segments_a_tile_may_list_match_and_more_are_refused() {
    let outline = AreaOutline {
        window: (0, 0, 15, 15),
        segs: zig_zag(PATH_TILE_SEGMENTS),
        even_odd: false,
        weight: 255,
    };
    assert_same(&outline, 16, 16, "a full tile");
    let more = AreaOutline {
        segs: zig_zag(PATH_TILE_SEGMENTS + 1),
        ..outline
    };
    assert_eq!(
        device_coverage(&more, 16, 16),
        Err(PageCommandsError::Unrepresentable)
    );
}

#[test]
fn long_edges_across_a_wide_page_match_the_cpu_rasteriser() {
    // Long diagonals across many tiles: backdrops and per-band column
    // ranges.
    let (w, h) = (403, 97);
    let (fw, fh) = (i64::from(w) * 256, i64::from(h) * 256);
    let points = [
        (-500, 300),
        (fw + 900, fh / 3),
        (fw / 2, fh + 200),
        (40, fh / 2),
    ];
    for even_odd in [false, true] {
        let outline = AreaOutline {
            window: (0, 0, 402, 96),
            segs: polygon(&points),
            even_odd,
            weight: 255,
        };
        assert_same(&outline, w, h, "wide page");
    }
}

#[test]
fn an_opaque_interior_tile_hides_the_items_below_it() {
    let rect =
        |x0: i64, y0: i64, x1: i64, y1: i64| polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]);
    let whole = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 63,
        y1: 63,
    };
    let page_with = |top_alpha: u16, top_weight: u8| {
        let mut page = PageCommands::new(64, 64, SampleDepth::Eight, [255; 3]);
        let below = rect(0, 0, 64 * 256, 64 * 256);
        // Covers tiles fully from column 1, row 1 on.
        let top = rect(16 * 256 - 100, 16 * 256, 64 * 256, 64 * 256);
        for (segs, rgb, alpha, weight) in [
            (&below, [9, 9, 9], 255, 255),
            (&top, [200, 0, 0], top_alpha, top_weight),
        ] {
            let path = AreaPath {
                segs,
                window: whole,
                even_odd: false,
                weight,
            };
            page.push_area_path(&path, rgb, alpha).expect("push");
        }
        page
    };
    let items = |page: &PageCommands, t: usize| {
        let bins = page.bin().expect("bin");
        (bins.offsets[t + 1] - bins.offsets[t]) as usize
    };
    let opaque = page_with(255, 255);
    // Tiles (1, 1) and (2, 2) lie inside the top rectangle, tile (0, 0)
    // outside it, tile (0, 1) on its left edge.
    assert_eq!(items(&opaque, 4 + 1), 1);
    assert_eq!(items(&opaque, 2 * 4 + 2), 1);
    assert_eq!(items(&opaque, 0), 1);
    assert_eq!(items(&opaque, 4), 2);
    assert_eq!(items(&page_with(254, 255), 2 * 4 + 2), 2);
    assert_eq!(items(&page_with(255, 63), 2 * 4 + 2), 2);
    let pixels = render_page(&opaque).expect("render");
    assert_eq!(pixels[(40 * 64 + 40) * 3..][..3], [200, 0, 0]);
    assert_eq!(pixels[(5 * 64 + 5) * 3..][..3], [9, 9, 9]);
}
