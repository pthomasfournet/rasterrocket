//! Synthetic camera captures built from rendered pages: a page plane
//! photographed by a pinhole camera over a textured desk, a page curled
//! toward its spine, and a two-page spread with a gutter shadow.
//!
//! Every generator is an exact inverse map from capture pixels to page
//! coordinates, so ground truth (corners, flat line positions, gutter) is
//! known analytically.

use camera::{GrayPage, GrayView};
use color::Gray8;
use raster::Bitmap;

/// A tightly packed grayscale image.
#[derive(Clone)]
pub(crate) struct Gray {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

impl Gray {
    pub(crate) fn new(w: usize, h: usize, fill: u8) -> Self {
        Self {
            w,
            h,
            px: vec![fill; w * h],
        }
    }

    pub(crate) fn at(&self, x: usize, y: usize) -> u8 {
        self.px[y * self.w + x]
    }

    /// Bilinear sample at pixel-centre coordinates; `None` outside the
    /// continuous extent `[-0.5, w - 0.5] × [-0.5, h - 0.5]`.
    pub(crate) fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let (wf, hf) = (self.w as f64, self.h as f64);
        if !(x >= -0.5 && y >= -0.5 && x <= wf - 0.5 && y <= hf - 0.5) {
            return None;
        }
        let xc = x.clamp(0.0, wf - 1.0);
        let yc = y.clamp(0.0, hf - 1.0);
        let (x0, y0) = (xc.floor() as usize, yc.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (xc - x0 as f64, yc - y0 as f64);
        let g = |x: usize, y: usize| f64::from(self.at(x, y));
        let top = g(x0, y0) * (1.0 - fx) + g(x1, y0) * fx;
        let bot = g(x0, y1) * (1.0 - fx) + g(x1, y1) * fx;
        Some(top * (1.0 - fy) + bot * fy)
    }

    pub(crate) fn to_bitmap(&self) -> Bitmap<Gray8> {
        let mut b = Bitmap::<Gray8>::new(self.w as u32, self.h as u32, 1, false);
        b.data_mut().copy_from_slice(&self.px);
        b
    }

    pub(crate) fn from_bitmap(b: &Bitmap<Gray8>) -> Self {
        let (w, h) = (b.width as usize, b.height as usize);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..b.height {
            px.extend_from_slice(&b.row_bytes(y)[..w]);
        }
        Self { w, h, px }
    }

    pub(crate) fn view(&self) -> GrayView<'_> {
        GrayView::packed(&self.px, self.w, self.h)
    }

    pub(crate) fn from_page(p: GrayPage) -> Self {
        Self {
            w: p.width,
            h: p.height,
            px: p.data,
        }
    }

    /// Bilinear resize to `w × h`.
    pub(crate) fn resize(&self, w: usize, h: usize) -> Self {
        let (sx, sy) = (self.w as f64 / w as f64, self.h as f64 / h as f64);
        let mut out = Self::new(w, h, 255);
        for y in 0..h {
            for x in 0..w {
                let v = self
                    .sample((x as f64 + 0.5) * sx - 0.5, (y as f64 + 0.5) * sy - 0.5)
                    .unwrap_or(255.0);
                out.px[y * w + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Left-right mirror image.
    pub(crate) fn mirror(&self) -> Self {
        let mut out = self.clone();
        for (dst, src) in out
            .px
            .chunks_exact_mut(self.w)
            .zip(self.px.chunks_exact(self.w))
        {
            for (d, s) in dst.iter_mut().zip(src.iter().rev()) {
                *d = *s;
            }
        }
        out
    }

    /// Write as binary PGM (for offline inspection / OCR).
    pub(crate) fn write_pgm(&self, path: &std::path::Path) {
        let mut bytes = format!("P5\n{} {}\n255\n", self.w, self.h).into_bytes();
        bytes.extend_from_slice(&self.px);
        std::fs::write(path, bytes).expect("write pgm");
    }
}

/// Deterministic per-pixel noise in `[-1, 1]`.
fn hash_noise(x: usize, y: usize, seed: u64) -> f64 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Wood-like desk luma at `(x, y)`: dark base, grain, gradient, noise.
fn desk(x: usize, y: usize, w: usize, h: usize) -> f64 {
    let (xf, yf) = (x as f64 / w as f64, y as f64 / h as f64);
    let grain = 8.0 * ((yf * 90.0 + (xf * 7.0).sin() * 3.0).sin());
    let gradient = 25.0 * xf - 15.0 * yf;
    80.0 + grain + gradient + 10.0 * hash_noise(x, y, 7)
}

/// Pose of the page plane in front of the camera.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pose {
    /// Rotation about the page's horizontal axis (radians): top edge
    /// farther (positive) or nearer.
    pub tilt_x: f64,
    /// Rotation about the page's vertical axis (radians).
    pub tilt_y: f64,
    /// In-plane rotation (radians).
    pub roll: f64,
    /// Canvas size relative to the page bounding box.
    pub canvas_scale: f64,
    /// Offset of the page centre from the canvas centre, in canvas pixels.
    pub shift: (f64, f64),
}

/// A capture and its ground truth.
pub(crate) struct Capture {
    pub image: Gray,
    /// Page corners in capture pixel coordinates (TL, TR, BR, BL), at the
    /// page edge.
    pub corners: [[f64; 2]; 4],
}

/// Photograph `page` (W × H, page pixel units = world units) lying in a
/// plane at distance `1.25 H` from a pinhole camera (focal length equal
/// to the distance, roughly the longer canvas side), over a desk.
///
/// `occluder`: optional finger-like ellipse `(cx, cy, rx, ry)` in capture
/// coordinates painted on top with skin luma.
pub(crate) fn photograph(
    page: &Gray,
    pose: Pose,
    occluder: Option<(f64, f64, f64, f64)>,
) -> Capture {
    let (pw, ph) = (page.w as f64, page.h as f64);
    let dist = 1.25 * ph;
    let focal = dist; // flat, fronto-parallel page images at scale 1
    // Rotation R = Rz(roll) · Ry(tilt_y) · Rx(tilt_x); page axes in camera space.
    let (cx_, sx_) = (pose.tilt_x.cos(), pose.tilt_x.sin());
    let (cy_, sy_) = (pose.tilt_y.cos(), pose.tilt_y.sin());
    let (cz_, sz_) = (pose.roll.cos(), pose.roll.sin());
    let rot = |v: [f64; 3]| {
        let v = [v[0], v[1] * cx_ - v[2] * sx_, v[1] * sx_ + v[2] * cx_];
        let v = [v[0] * cy_ + v[2] * sy_, v[1], -v[0] * sy_ + v[2] * cy_];
        [v[0] * cz_ - v[1] * sz_, v[0] * sz_ + v[1] * cz_, v[2]]
    };
    let ex = rot([1.0, 0.0, 0.0]);
    let ey = rot([0.0, 1.0, 0.0]);
    let n = rot([0.0, 0.0, 1.0]);
    let p0 = [0.0, 0.0, dist];
    // World point of continuous page coordinate (s, r) ∈ [0, W] × [0, H].
    let world = |s: f64, r: f64| {
        let (a, b) = (s - pw / 2.0, r - ph / 2.0);
        [
            p0[0] + a * ex[0] + b * ey[0],
            p0[1] + a * ex[1] + b * ey[1],
            p0[2] + a * ex[2] + b * ey[2],
        ]
    };
    let project_raw = |p: [f64; 3]| [focal * p[0] / p[2], focal * p[1] / p[2]];
    let raw: Vec<[f64; 2]> = [(0.0, 0.0), (pw, 0.0), (pw, ph), (0.0, ph)]
        .iter()
        .map(|&(s, r)| project_raw(world(s, r)))
        .collect();
    let (min_x, max_x) = raw
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
    let (min_y, max_y) = raw
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
    // The principal point sits at the canvas centre offset by `shift` (an
    // off-centre crop of the photograph); the canvas encloses the page.
    let cw = (2.0 * (min_x.abs().max(max_x.abs()) + pose.shift.0.abs()) * pose.canvas_scale).round()
        as usize;
    let ch = (2.0 * (min_y.abs().max(max_y.abs()) + pose.shift.1.abs()) * pose.canvas_scale).round()
        as usize;
    let ppx = cw as f64 / 2.0 + pose.shift.0;
    let ppy = ch as f64 / 2.0 + pose.shift.1;
    // Capture pixel centre (x, y) is at continuous position (x + 0.5, y + 0.5).
    let corners = [0, 1, 2, 3].map(|i| [raw[i][0] + ppx - 0.5, raw[i][1] + ppy - 0.5]);

    let nd = n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2];
    let page_at = |u: f64, v: f64| -> Option<f64> {
        // Ray through continuous canvas position (u, v).
        let d = [(u - ppx) / focal, (v - ppy) / focal, 1.0];
        let t = nd / (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);
        let q = [t * d[0] - p0[0], t * d[1] - p0[1], t * d[2] - p0[2]];
        let s = q[0] * ex[0] + q[1] * ex[1] + q[2] * ex[2] + pw / 2.0;
        let r = q[0] * ey[0] + q[1] * ey[1] + q[2] * ey[2] + ph / 2.0;
        if !(0.0..=pw).contains(&s) || !(0.0..=ph).contains(&r) {
            return None;
        }
        // Mild illumination fall-off across the page.
        let light = 0.88 + 0.12 * (1.0 - s / pw) * (1.0 - 0.5 * r / ph);
        page.sample(s - 0.5, r - 0.5).map(|v| v * light)
    };

    let mut image = Gray::new(cw, ch, 0);
    for y in 0..ch {
        for x in 0..cw {
            // 2 × 2 supersampling.
            let mut acc = 0.0;
            for (oy, ox) in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)] {
                let (u, v) = (x as f64 + ox, y as f64 + oy);
                let mut value = page_at(u, v).unwrap_or_else(|| desk(x, y, cw, ch));
                if let Some((ex_, ey_, rx, ry)) = occluder {
                    let e = ((u - ex_) / rx).powi(2) + ((v - ey_) / ry).powi(2);
                    if e <= 1.0 {
                        value = 150.0 + 15.0 * (1.0 - e);
                    }
                }
                acc += value;
            }
            image.px[y * cw + x] = (acc / 4.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Capture { image, corners }
}

/// Curl the right part of `page` away from the camera, as a book page
/// bending into its spine, and photograph it from above.
///
/// The page surface depth is `d(s) = depth · ((s − s0)/(W − s0))²` for
/// `s > s0 = 0.55 W`.  The camera looks down from distance `2 H` with its
/// principal point over `(0.45 W, 0.4 H)`, so each column's height is
/// scaled about that row by `D / (D + d)`.  The output keeps the page size;
/// pixels beyond the curled edge are paper white.
///
/// Returns the capture and its horizontal-only counterpart: the same
/// column mapping without the vertical displacement, i.e. the result of a
/// perfect vertical dewarp (horizontal foreshortening is not corrected).
pub(crate) fn curl(page: &Gray, depth: f64) -> (Gray, Gray) {
    let (pw, ph) = (page.w as f64, page.h as f64);
    let s0 = 0.55 * pw;
    let dist = 2.0 * ph;
    let (cxw, cyw) = (0.45 * pw, 0.4 * ph);
    let d = |s: f64| {
        if s <= s0 {
            0.0
        } else {
            depth * ((s - s0) / (pw - s0)).powi(2)
        }
    };
    // Arc-length parametrisation: x_world(s) = ∫ cos θ, tan θ = d'(s).
    let step = 0.25;
    let n = (pw / step).ceil() as usize + 1;
    let mut table = Vec::with_capacity(n); // (u, s, d)
    let mut xw = 0.0;
    for k in 0..n {
        let s = k as f64 * step;
        if k > 0 {
            let slope = (d(s) - d(s - step)) / step;
            xw += step / (1.0 + slope * slope).sqrt();
        }
        let depth_s = d(s);
        let u = cxw + dist * (xw - cxw) / (dist + depth_s);
        table.push((u, s, depth_s));
    }
    assert!(
        table.windows(2).all(|w| w[1].0 > w[0].0),
        "curl too strong: projection not monotone"
    );
    let mut out = Gray::new(page.w, page.h, 255);
    let mut reference = Gray::new(page.w, page.h, 255);
    let mut k = 0;
    for x in 0..page.w {
        let u = x as f64 + 0.5;
        while k + 1 < table.len() && table[k + 1].0 < u {
            k += 1;
        }
        if k + 1 >= table.len() || u < table[0].0 {
            continue;
        }
        let (a, b) = (table[k], table[k + 1]);
        let f = (u - a.0) / (b.0 - a.0);
        let s = a.1 + f * (b.1 - a.1);
        let depth_s = a.2 + f * (b.2 - a.2);
        for y in 0..page.h {
            let v = y as f64 + 0.5;
            let r = cyw + (v - cyw) * (dist + depth_s) / dist;
            if let Some(val) = page.sample(s - 0.5, r - 0.5) {
                out.px[y * page.w + x] = val.round() as u8;
            }
            if let Some(val) = page.sample(s - 0.5, v - 0.5) {
                reference.px[y * page.w + x] = val.round() as u8;
            }
        }
    }
    (out, reference)
}

/// Two pages side by side with a gutter shadow at their junction.
pub(crate) fn spread(left: &Gray, right: &Gray) -> Gray {
    let h = left.h.max(right.h);
    let w = left.w + right.w;
    let mut out = Gray::new(w, h, 255);
    let sigma = 0.012 * w as f64;
    for y in 0..h {
        for x in 0..w {
            let v = if x < left.w {
                if y < left.h { left.at(x, y) } else { 255 }
            } else if y < right.h {
                right.at(x - left.w, y)
            } else {
                255
            };
            let dx = (x as f64 + 0.5 - left.w as f64) / sigma;
            let shade = 1.0 - 0.3 * (-dx * dx).exp();
            out.px[y * w + x] = (f64::from(v) * shade).round() as u8;
        }
    }
    out
}
