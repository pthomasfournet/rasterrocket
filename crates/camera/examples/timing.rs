//! Per-stage timings of `process` on a synthetic 16 MP book capture.
//!
//! ```text
//! cargo run --release -p rasterrocket-camera --example timing
//! ```
//!
//! The capture is a 4656 × 3496 image of an open two-page spread lying on
//! a darker, textured desk, with a gutter shadow and lines of word-like
//! bars on both pages that bend toward the gutter.  Each configuration runs several times; the run
//! with the median total is reported.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    reason = "synthetic image arithmetic on dimensions far below 2^31, written \
              as plain formulas"
)]

use std::time::Duration;

use rasterrocket_camera::{CameraOptions, GrayView, Processed, process};

const WIDTH: usize = 4656;
const HEIGHT: usize = 3496;
const RUNS: usize = 7;

fn noise(x: usize, y: usize) -> u8 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (h >> 59) as u8
}

fn capture() -> Vec<u8> {
    let (x0, x1) = (WIDTH * 6 / 100, WIDTH * 94 / 100);
    let (y0, y1) = (HEIGHT * 5 / 100, HEIGHT * 95 / 100);
    let gutter = usize::midpoint(x0, x1);
    let margin = WIDTH / 25;
    let mut data = vec![0u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let on_page = (x0..x1).contains(&x) && (y0..y1).contains(&y);
            let v = if on_page {
                let d = (x as f64 - gutter as f64) / 60.0;
                let shade = 232.0 - 60.0 * (-d * d).exp();
                let in_text = ((x0 + margin..gutter - margin).contains(&x)
                    || (gutter + margin..x1 - margin).contains(&x))
                    && (y0 + margin..y1 - margin).contains(&y);
                // Lines sink by up to 40 px toward the gutter.
                let near = 1.0 - (x as f64 - gutter as f64).abs() / (gutter - x0) as f64;
                let ys = (y as f64 - 40.0 * near.powi(3)).max(0.0) as usize;
                if in_text && (ys / 14).is_multiple_of(3) && (x / 41) % 6 != 5 {
                    30
                } else {
                    shade as u8
                }
            } else {
                70 + noise(x, y)
            };
            data[y * WIDTH + x] = v;
        }
    }
    data
}

fn run(img: GrayView<'_>, opts: CameraOptions) -> Processed {
    let mut runs: Vec<Processed> = (0..RUNS)
        .map(|_| process(img, &opts).expect("valid capture"))
        .collect();
    runs.sort_by_key(|p| p.report.timings.total);
    runs.swap_remove(RUNS / 2)
}

fn ms(d: Duration) -> String {
    format!("{:8.1} ms", d.as_secs_f64() * 1e3)
}

fn print(label: &str, out: &Processed) {
    let r = &out.report;
    let t = &r.timings;
    println!("{label}");
    println!("  crop     {:?}", r.crop);
    println!("  rectify  {:?}", r.rectify);
    println!("  split    {:?}", r.split);
    println!("  dewarp   {:?}", r.dewarp);
    let sizes: Vec<_> = out.pages.iter().map(|p| (p.width, p.height)).collect();
    println!("  pages    {sizes:?}");
    println!("  quad     {}", ms(t.quad));
    println!("  crop     {}", ms(t.crop));
    println!("  rectify  {}", ms(t.rectify));
    println!("  split    {}", ms(t.split));
    println!("  dewarp   {}", ms(t.dewarp));
    println!("  output   {}", ms(t.output));
    println!("  total    {}", ms(t.total));
}

fn main() {
    let data = capture();
    let img = GrayView::packed(&data, WIDTH, HEIGHT);

    let mut crop_split = CameraOptions::default();
    crop_split.crop = true;
    crop_split.split_spreads = true;
    print("crop + split_spreads", &run(img, crop_split));
    print("all stages", &run(img, CameraOptions::all()));
}
