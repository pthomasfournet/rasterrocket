//! Every public entry point returns, without panicking, on degenerate and
//! random input: constant, noisy, gradient, striped and page-like images
//! from 1 × 1 up to tens of megapixels' worth of extent, under every option
//! combination, with packed and padded strides.  Outputs satisfy the
//! documented invariants.

use rasterrocket_camera::{
    CameraError, CameraOptions, GrayPage, GrayView, Outcome, Quad, Rect, crop, detect_page_quad,
    dewarp, find_gutter, process, rectify, split_at,
};

/// 64-bit linear congruential generator (Knuth's MMIX constants).
struct Lcg(u64);

impl Lcg {
    const fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    const fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n.max(1)).unwrap_or(u64::MAX)).unwrap_or(0)
    }

    fn coord(&mut self, extent: usize) -> f64 {
        let span = u32::try_from(extent).unwrap_or(u32::MAX).saturating_mul(3);
        let v = u32::try_from(self.next() % u64::from(span.max(1))).unwrap_or(0);
        f64::from(v) - f64::from(span / 3)
    }
}

#[derive(Debug, Clone, Copy)]
enum Pattern {
    Black,
    White,
    Noise,
    GradientX,
    GradientY,
    StripesX,
    StripesY,
    Checker,
    PageOnDesk,
    Spread,
    CurvedText,
}

const PATTERNS: [Pattern; 11] = [
    Pattern::Black,
    Pattern::White,
    Pattern::Noise,
    Pattern::GradientX,
    Pattern::GradientY,
    Pattern::StripesX,
    Pattern::StripesY,
    Pattern::Checker,
    Pattern::PageOnDesk,
    Pattern::Spread,
    Pattern::CurvedText,
];

fn scaled(v: usize, n: usize) -> u8 {
    u8::try_from(v * 255 / n.max(1)).unwrap_or(u8::MAX)
}

fn pixel(pattern: Pattern, x: usize, y: usize, w: usize, h: usize, rng: &mut Lcg) -> u8 {
    match pattern {
        Pattern::Black => 0,
        Pattern::White => 255,
        Pattern::Noise => rng.byte(),
        Pattern::GradientX => scaled(x, w),
        Pattern::GradientY => scaled(y, h),
        Pattern::StripesX => {
            if (x / 3).is_multiple_of(2) {
                20
            } else {
                230
            }
        }
        Pattern::StripesY => {
            if (y / 5).is_multiple_of(2) {
                20
            } else {
                230
            }
        }
        Pattern::Checker => {
            if (x / 7 + y / 7).is_multiple_of(2) {
                0
            } else {
                255
            }
        }
        Pattern::CurvedText => {
            // Lines at a 24 px pitch bowed by up to 20 px across the width.
            let across = 2.0 * f64::from(u32::try_from(x).unwrap_or(0))
                / f64::from(u32::try_from(w).unwrap_or(1))
                - 1.0;
            let bow = 20.0 * across * across;
            let ys = f64::from(u32::try_from(y).unwrap_or(0)) + bow;
            let line = (ys / 24.0).fract() < 0.35;
            let margin = x < w / 20 || x >= w - w / 20 || y < h / 20 || y >= h - h / 20;
            if line && !margin && (x / 13) % 5 != 4 {
                30
            } else {
                230
            }
        }
        Pattern::PageOnDesk | Pattern::Spread => {
            let on_page = (w / 8..w - w / 8).contains(&x) && (h / 10..h - h / 10).contains(&y);
            let gutter = matches!(pattern, Pattern::Spread) && x.abs_diff(w / 2) < w / 40 + 1;
            if !on_page {
                60 + rng.byte() / 8
            } else if gutter {
                150
            } else if (y / 6).is_multiple_of(3) && (x / 11) % 4 != 3 {
                25
            } else {
                235
            }
        }
    }
}

/// A `w × h` image of `p` in a buffer of row stride `w + pad`, padding
/// filled with noise.
fn image(p: Pattern, w: usize, h: usize, pad: usize, rng: &mut Lcg) -> Vec<u8> {
    let stride = w + pad;
    let mut data = Vec::with_capacity(stride * h);
    for y in 0..h {
        for x in 0..stride {
            data.push(if x < w {
                pixel(p, x, y, w, h, rng)
            } else {
                rng.byte()
            });
        }
    }
    data
}

fn every_option_set() -> Vec<CameraOptions> {
    (0u8..16)
        .map(|bits| {
            let mut o = CameraOptions::default();
            o.crop = bits & 1 != 0;
            o.rectify = bits & 2 != 0;
            o.split_spreads = bits & 4 != 0;
            o.dewarp = bits & 8 != 0;
            o
        })
        .collect()
}

fn assert_packed(page: &GrayPage, ctx: &str) {
    assert!(page.width >= 1 && page.height >= 1, "{ctx}: empty page");
    assert_eq!(
        page.data.len(),
        page.width * page.height,
        "{ctx}: not packed"
    );
}

/// Run `process` and check its invariants; returns which of crop,
/// rectify, split and dewarp applied.
fn check_process(img: GrayView<'_>, opts: CameraOptions, ctx: &str) -> [bool; 4] {
    let out = process(img, &opts).unwrap_or_else(|e| panic!("{ctx}: valid input rejected: {e}"));
    let n = out.pages.len();
    assert!(n == 1 || n == 2, "{ctx}: {n} pages");
    assert_eq!(n == 2, out.report.split.applied(), "{ctx}: split vs pages");
    assert_eq!(out.report.dewarp.len(), n, "{ctx}: dewarp outcomes");
    let area: usize = out.pages.iter().map(|p| p.width * p.height).sum();
    assert!(
        area <= 2 * img.width * img.height,
        "{ctx}: output area {area} exceeds twice the input"
    );
    for page in &out.pages {
        assert_packed(page, ctx);
    }
    if !opts.any() {
        assert!(
            [&out.report.crop, &out.report.rectify, &out.report.split]
                .iter()
                .all(|o| **o == Outcome::Off),
            "{ctx}: stage ran while off"
        );
    }
    let again = out
        .report
        .replay(img)
        .unwrap_or_else(|e| panic!("{ctx}: replay rejected: {e}"));
    assert_eq!(again, out.pages, "{ctx}: replay differs");
    let r = &out.report;
    [
        r.crop.applied(),
        r.rectify.applied(),
        r.split.applied(),
        r.dewarp.iter().any(Outcome::applied),
    ]
}

fn check_stages(img: GrayView<'_>, rng: &mut Lcg, ctx: &str) {
    let detected = detect_page_quad(img).unwrap_or_else(|e| panic!("{ctx}: {e}"));
    let gutter = find_gutter(img).unwrap_or_else(|e| panic!("{ctx}: {e}"));
    if let Some(x) = gutter.found() {
        let [l, r] = split_at(img, x).unwrap_or_else(|e| panic!("{ctx}: {e}"));
        assert_eq!(l.width + r.width, img.width, "{ctx}: split widths");
    }
    let dewarped = dewarp(img).unwrap_or_else(|e| panic!("{ctx}: {e}"));
    if let Some(page) = dewarped.found() {
        assert_eq!((page.width, page.height), (img.width, img.height), "{ctx}");
    }
    // Random, inverted, non-finite and detected quads all rectify or decline.
    let mut quads = vec![Quad {
        corners: [
            [f64::NAN, 0.0],
            [f64::INFINITY, 1.0],
            [0.0, -0.0],
            [1e300, -1e300],
        ],
    }];
    quads.extend(detected.found());
    for _ in 0..3 {
        let mut corners = [[0.0; 2]; 4];
        for c in &mut corners {
            *c = [rng.coord(img.width), rng.coord(img.height)];
        }
        quads.push(Quad { corners });
    }
    for q in &quads {
        if let Some(page) = rectify(img, q)
            .unwrap_or_else(|e| panic!("{ctx}: {e}"))
            .found()
        {
            assert_packed(&page, ctx);
            assert!(
                page.width * page.height <= 2 * img.width * img.height,
                "{ctx}"
            );
        }
        if let Some(rect) = q.bounding_rect(img.width, img.height) {
            let page = crop(img, rect).unwrap_or_else(|e| panic!("{ctx}: {e}"));
            assert_packed(&page, ctx);
        }
    }
    // Arbitrary rectangles and split columns are accepted or rejected as
    // invalid input, never panic.
    for _ in 0..4 {
        let rect = Rect {
            x: rng.below(img.width + 2),
            y: rng.below(img.height + 2),
            width: rng.below(img.width + 2),
            height: rng.below(img.height + 2),
        };
        match crop(img, rect) {
            Ok(page) => assert_eq!((page.width, page.height), (rect.width, rect.height)),
            Err(e) => assert!(matches!(e, CameraError::RectOutOfBounds { .. }), "{e}"),
        }
        let x = rng.below(img.width + 2);
        match split_at(img, x) {
            Ok([l, r]) => assert_eq!((l.width, r.width), (x, img.width - x)),
            Err(e) => assert!(matches!(e, CameraError::SplitOutOfRange { .. }), "{e}"),
        }
    }
}

#[test]
fn small_and_odd_sizes_under_every_option_set() {
    let sizes = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (1, 37),
        (37, 1),
        (3, 5),
        (31, 33),
        (63, 65),
        (64, 32),
        (129, 257),
        (257, 129),
        (641, 479),
        (1301, 701),
    ];
    let options = every_option_set();
    let mut rng = Lcg(0x5EED);
    let mut runs = 0;
    let mut applied = [0usize; 4];
    for (w, h) in sizes {
        for p in PATTERNS {
            for pad in [0, 3] {
                let data = image(p, w, h, pad, &mut rng);
                let img = GrayView {
                    data: &data,
                    width: w,
                    height: h,
                    stride: w + pad,
                };
                let ctx = format!("{p:?} {w}x{h} pad {pad}");
                for opts in &options {
                    let stages = check_process(img, *opts, &format!("{ctx} {opts:?}"));
                    for (n, hit) in applied.iter_mut().zip(stages) {
                        *n += usize::from(hit);
                    }
                    runs += 1;
                }
                check_stages(img, &mut rng, &ctx);
            }
        }
    }
    eprintln!("{runs} process runs; crop, rectify, split, dewarp applied in {applied:?}");
    // The page-like patterns exercise the resampling paths, not only the
    // declines.
    assert!(applied.iter().all(|&n| n > 0), "{applied:?}");
}

#[test]
fn large_and_elongated_sizes() {
    let cases = [
        (Pattern::Noise, 1, 200_000),
        (Pattern::GradientX, 200_000, 1),
        (Pattern::Spread, 100_000, 3),
        (Pattern::StripesY, 3, 100_000),
        (Pattern::Spread, 2400, 1500),
        (Pattern::PageOnDesk, 1500, 2100),
        (Pattern::Noise, 2000, 1600),
    ];
    let mut rng = Lcg(0xB16);
    for (p, w, h) in cases {
        let data = image(p, w, h, 0, &mut rng);
        let img = GrayView::packed(&data, w, h);
        let ctx = format!("{p:?} {w}x{h}");
        let _ = check_process(img, CameraOptions::all(), &ctx);
        let mut crop_split = CameraOptions::default();
        crop_split.crop = true;
        crop_split.split_spreads = true;
        let _ = check_process(img, crop_split, &ctx);
    }
}

#[test]
fn malformed_views_are_errors() {
    let data = [128u8; 64];
    let bad = [
        (0, 4, 4),
        (4, 0, 4),
        (8, 4, 7),
        (8, 9, 8),
        (2, 3, usize::MAX),
        (usize::MAX, 1, usize::MAX),
    ];
    for (width, height, stride) in bad {
        let img = GrayView {
            data: &data,
            width,
            height,
            stride,
        };
        let ctx = format!("{width}x{height} stride {stride}");
        let errors = [
            process(img, &CameraOptions::all()).err(),
            detect_page_quad(img).err(),
            find_gutter(img).err(),
            dewarp(img).err(),
            split_at(img, 1).err(),
            crop(
                img,
                Rect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
            )
            .err(),
            rectify(
                img,
                &Quad {
                    corners: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                },
            )
            .err(),
        ];
        for e in errors {
            let Some(e) = e else {
                panic!("{ctx}: accepted");
            };
            assert!(!e.to_string().is_empty(), "{ctx}");
        }
    }
}
