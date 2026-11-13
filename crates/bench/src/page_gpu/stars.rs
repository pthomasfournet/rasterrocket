//! The star document: pages of random solid star polygons, as a PDF and as
//! device-space paths.
//!
//! Every page is US Letter (612 × 792 pt) and holds `stars` concave star
//! polygons of 5, 7 or 9 points, each filled with `r g b rg … h f`.  The
//! generator is a fixed `SplitMix64` stream, so a seed always yields the same
//! document.  Vertex coordinates are written with two decimals and colour
//! components with four; the device paths are built from the same decimal
//! strings, parsed back, so the PDF and the paths carry identical `f64`s.

use std::fmt::Write as _;

use raster::{Path, PathBuilder};

/// Page width in points.
pub const PAGE_W_PT: f64 = 612.0;
/// Page height in points.
pub const PAGE_H_PT: f64 = 792.0;

/// One star: its colour and its vertices in user space (points, y up).
pub struct Star {
    pub rgb: [u8; 3],
    /// Vertex coordinates exactly as written to the content stream.
    pub pts: Vec<(String, String)>,
}

/// `SplitMix64`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits are exact in f64"
    )]
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }

    fn byte(&mut self) -> u8 {
        self.next_u64().to_le_bytes()[0]
    }
}

/// `pages` pages of `stars` stars each.
pub fn generate(pages: usize, stars: usize, seed: u64) -> Vec<Vec<Star>> {
    let mut rng = Rng(seed);
    (0..pages)
        .map(|_| {
            (0..stars)
                .map(|_| {
                    let points = [5usize, 7, 9][usize::try_from(rng.next_u64() % 3).unwrap_or(0)];
                    let cx = rng.range(0.0, PAGE_W_PT);
                    let cy = rng.range(0.0, PAGE_H_PT);
                    let outer = rng.range(40.0, 160.0);
                    let inner = outer * rng.range(0.38, 0.5);
                    let rot = rng.range(0.0, std::f64::consts::TAU);
                    let rgb = [rng.byte(), rng.byte(), rng.byte()];
                    let n = 2 * points;
                    let pts = (0..n)
                        .map(|i| {
                            let radius = if i % 2 == 0 { outer } else { inner };
                            #[expect(clippy::cast_precision_loss, reason = "i < 18")]
                            let t = rot + std::f64::consts::TAU * i as f64 / n as f64;
                            (
                                format!("{:.2}", radius.mul_add(t.cos(), cx)),
                                format!("{:.2}", radius.mul_add(t.sin(), cy)),
                            )
                        })
                        .collect();
                    Star { rgb, pts }
                })
                .collect()
        })
        .collect()
}

/// The document as an uncompressed PDF.
pub fn write_pdf(doc: &[Vec<Star>]) -> Vec<u8> {
    let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    let obj = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &[u8]| {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", offsets.len()).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    let n = doc.len();
    // Objects: 1 catalog, 2 pages, then (page, content) pairs.
    let mut kids = String::new();
    for i in 0..n {
        let _ = write!(kids, "{} 0 R ", 3 + 2 * i);
    }
    obj(&mut out, &mut offsets, b"<< /Type /Catalog /Pages 2 0 R >>");
    obj(
        &mut out,
        &mut offsets,
        format!("<< /Type /Pages /Kids [{kids}] /Count {n} >>").as_bytes(),
    );
    for (i, page) in doc.iter().enumerate() {
        obj(
            &mut out,
            &mut offsets,
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W_PT} {PAGE_H_PT}] \
                 /Contents {} 0 R >>",
                4 + 2 * i
            )
            .as_bytes(),
        );
        let mut content = String::new();
        for star in page {
            let [r, g, b] = star.rgb.map(|c| f64::from(c) / 255.0);
            let _ = writeln!(content, "{r:.4} {g:.4} {b:.4} rg");
            for (k, (x, y)) in star.pts.iter().enumerate() {
                let _ = writeln!(content, "{x} {y} {}", if k == 0 { "m" } else { "l" });
            }
            content.push_str("h f\n");
        }
        let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
        stream.extend_from_slice(content.as_bytes());
        stream.extend_from_slice(b"\nendstream");
        obj(&mut out, &mut offsets, &stream);
    }
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes(),
    );
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            offsets.len() + 1
        )
        .as_bytes(),
    );
    out
}

/// Device size of a page at `scale` pixels per point.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a Letter page at a sane scale is a few thousand pixels"
)]
pub fn page_px(scale: f64) -> (u32, u32) {
    (
        (PAGE_W_PT * scale).round() as u32,
        (PAGE_H_PT * scale).round() as u32,
    )
}

/// The star's outline in device space (y down) at `scale` pixels per
/// point on a page `height_px` pixels tall: the transform of a page with no
/// rotation and a zero box origin, `x' = s·x`, `y' = height − s·y`, with
/// the same operation order as the interpreter's point transform.
pub fn device_path(star: &Star, scale: f64, height_px: u32) -> Path {
    let ctm = [scale, 0.0, 0.0, scale, 0.0, 0.0];
    let page_h = f64::from(height_px);
    let mut b = PathBuilder::new();
    for (k, (xs, ys)) in star.pts.iter().enumerate() {
        let x: f64 = xs.parse().expect("written as a decimal");
        let y: f64 = ys.parse().expect("written as a decimal");
        let dx = ctm[0].mul_add(x, ctm[2] * y) + ctm[4];
        let dy = page_h - (ctm[1].mul_add(x, ctm[3] * y) + ctm[5]);
        let r = if k == 0 {
            b.move_to(dx, dy)
        } else {
            b.line_to(dx, dy)
        };
        r.expect("a finite star vertex");
    }
    b.close(false).expect("an open subpath");
    b.build()
}
