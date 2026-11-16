//! The device's outline raster against its host model.

use color::SampleDepth;

use crate::backend::vulkan::VulkanBackend;
use crate::page::reference::render_page;
use crate::page::{AreaPath, PATH_COORD_LIMIT, PATH_TILE_SEGMENTS, PageCommands, PixelBounds};

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

    fn sample(&mut self, max: u16) -> u16 {
        u16::try_from(self.range(0, i64::from(max) + 1)).unwrap()
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

/// `page` on `backend` against its host model, packed rows of samples of
/// its depth widened to 16 bits.
fn assert_device_matches_model(backend: &VulkanBackend, page: &PageCommands, what: &str) {
    let want = render_page(page).expect("host model");
    let n = want.len();
    let got: Vec<u16> = match page.depth() {
        SampleDepth::Eight => {
            let mut out = vec![0u8; n];
            backend
                .render_page(page, &mut out, page.width() as usize * 3)
                .expect("render_page");
            out.into_iter().map(u16::from).collect()
        }
        SampleDepth::Sixteen => {
            let mut out = vec![0u16; n];
            backend
                .render_page(page, &mut out, page.width() as usize * 3)
                .expect("render_page");
            out
        }
    };
    if let Some(i) = (0..n).find(|&i| got[i] != want[i]) {
        let px = i / 3;
        let w = page.width() as usize;
        panic!(
            "{what}: pixel ({}, {}) sample {} is {} on the device, {} in the model",
            px % w,
            px / w,
            i % 3,
            got[i],
            want[i]
        );
    }
}

/// A page of random outlines and opaque runs at `depth`.
fn random_page(
    rng: &mut Rng,
    width: u32,
    height: u32,
    n: usize,
    depth: SampleDepth,
) -> PageCommands {
    let max = depth.max_sample();
    let mut page = PageCommands::new(width, height, depth, [max - 7, max - 20, max - 33]);
    let (fw, fh) = (i64::from(width) * 256, i64::from(height) * 256);
    for i in 0..n {
        if i % 7 == 6 {
            let y = i32::try_from(rng.range(0, i64::from(height))).unwrap();
            let x0 = i32::try_from(rng.range(-5, i64::from(width))).unwrap();
            page.push_opaque_runs(&mut [(y, x0, x0 + 9)], [0; 3].map(|_| rng.sample(max)))
                .expect("push runs");
            continue;
        }
        let k = usize::try_from(rng.range(3, 12)).unwrap();
        let (cx, cy, r) = (
            rng.range(0, fw),
            rng.range(0, fh),
            rng.range(200, (fw.max(fh) / 2).max(201)),
        );
        let points: Vec<(i64, i64)> = (0..k)
            .map(|_| (cx + rng.range(-r, r), cy + rng.range(-r, r)))
            .collect();
        let segs = polygon(&points);
        let x0 = i32::try_from(rng.range(-3, i64::from(width))).unwrap();
        let y0 = i32::try_from(rng.range(-3, i64::from(height))).unwrap();
        let path = AreaPath {
            segs: &segs,
            window: PixelBounds {
                x0,
                y0,
                x1: x0 + i32::try_from(rng.range(0, i64::from(width) + 4)).unwrap(),
                y1: y0 + i32::try_from(rng.range(0, i64::from(height) + 4)).unwrap(),
            },
            even_odd: rng.range(0, 2) == 0,
            weight: [255, 63, 1, 254][usize::try_from(rng.range(0, 4)).unwrap()],
        };
        let alpha =
            [max, max, max / 2 + 1, 1, rng.sample(max)][usize::try_from(rng.range(0, 5)).unwrap()];
        page.push_area_path(&path, [0; 3].map(|_| rng.sample(max)), alpha)
            .expect("push outline");
    }
    page
}

#[test]
fn random_outlines_match_the_host_model() {
    let Some(backend) = super::super::test_backend() else {
        return;
    };
    if !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    let mut rng = Rng(0x0D3E_1CE5_EED5);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        for (width, height, n) in [
            (1, 1, 3),
            (37, 21, 30),
            (130, 70, 200),
            (300, 17, 400),
            (403, 260, 300),
        ] {
            let page = random_page(&mut rng, width, height, n, depth);
            assert_device_matches_model(
                &backend,
                &page,
                &format!("{}-bit {width}×{height}, {n} commands", depth.bits()),
            );
        }
    }
}

#[test]
fn outlines_at_the_integer_bounds_match_the_host_model() {
    let Some(backend) = super::super::test_backend() else {
        return;
    };
    if !backend.page_paths() {
        eprintln!("skipping: the device has no 64-bit integers");
        return;
    }
    let near = PATH_COORD_LIMIT - 1;
    let far = polygon(&[(-near, 100), (near, 7_000), (3_000, -near), (2_000, near)]);
    // A closed zig-zag of the most segments a tile may list.
    let mut points: Vec<(i64, i64)> = (0..PATH_TILE_SEGMENTS - 1)
        .map(|i| {
            let i = i64::try_from(i).unwrap();
            (300 + (i * 37) % 3_500, 200 + (i * 53) % 3_600)
        })
        .collect();
    points.push((3_900, 3_900));
    let dense = polygon(&points);
    let mut page = PageCommands::new(48, 40, SampleDepth::Eight, [255; 3]);
    for (segs, even_odd) in [(&far, false), (&dense, false), (&dense, true)] {
        let path = AreaPath {
            segs,
            window: PixelBounds {
                x0: 0,
                y0: 0,
                x1: 47,
                y1: 39,
            },
            even_odd,
            weight: 255,
        };
        page.push_area_path(&path, [20, 90, 160], 140)
            .expect("push outline");
    }
    assert_device_matches_model(&backend, &page, "integer bounds");
}
