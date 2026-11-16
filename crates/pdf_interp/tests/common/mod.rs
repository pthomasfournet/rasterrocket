//! The helpers of the tests that measure a conversion against Little
//! CMS's `transicc`: the patch chart, the working space as a profile
//! file, `transicc` itself and CIEDE2000.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rasterrocket_interp::WorkingSpace;

/// The largest ΔE00 accepted between the two conversions.
pub(crate) const MAX_DELTA_E: f64 = 1.0;

pub(crate) fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name)
}

/// The working space `space` as a profile file, described by its tags.
pub(crate) fn working_space_file(dir: &Path, space: WorkingSpace) -> PathBuf {
    let path = dir.join(format!("working-space-{space:?}.icc"));
    std::fs::write(&path, space.profile().encode().expect("encodes")).unwrap();
    path
}

/// A 9-level RGB cube, a 64-step neutral ramp and the corners.
pub(crate) fn patches() -> Vec<[f64; 3]> {
    cube_and_ramp(9)
}

/// A `levels`-level RGB cube and a 64-step neutral ramp.
pub(crate) fn cube_and_ramp(levels: i32) -> Vec<[f64; 3]> {
    let top = f64::from(levels - 1);
    let mut out = Vec::new();
    for r in 0..levels {
        for g in 0..levels {
            for b in 0..levels {
                out.push([r, g, b].map(|v: i32| f64::from(v) / top));
            }
        }
    }
    for i in 0..64 {
        out.push([f64::from(i) / 63.0; 3]);
    }
    out
}

/// `transicc` with `args` on the lines of `input`, its output values.
pub(crate) fn transicc(args: &[&str], input: &str) -> Vec<Vec<f64>> {
    let mut child = Command::new("transicc")
        .arg("-n")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("transicc runs: install liblcms2-utils");
    // The input is written while the output is read: a pipe buffers a
    // bounded amount, and the converter writes each line's answer before
    // it reads the next.
    let mut stdin = child.stdin.take().unwrap();
    let out = std::thread::scope(|s| {
        let writer = s.spawn(move || stdin.write_all(input.as_bytes()).unwrap());
        let out = child.wait_with_output().unwrap();
        writer.join().unwrap();
        out
    });
    assert!(
        out.status.success(),
        "transicc {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect())
        .collect()
}

/// Lines of `values`, each scaled by `full`.
pub(crate) fn lines(values: &[Vec<f64>], full: f64) -> String {
    values
        .iter()
        .map(|v| {
            v.iter()
                .map(|c| format!("{:.4}", c * full))
                .collect::<Vec<_>>()
                .join(" ")
                + "\n"
        })
        .collect()
}

/// CIEDE2000 (Sharma, Wu and Dalal 2005), `kL = kC = kH = 1`.
#[expect(
    clippy::suboptimal_flops,
    reason = "the formula written term by term as published, checked against its test pairs"
)]
pub(crate) fn delta_e00(a: &[f64], b: &[f64]) -> f64 {
    let (l1, a1, b1) = (a[0], a[1], a[2]);
    let (l2, a2, b2) = (b[0], b[1], b[2]);
    let c_bar = f64::midpoint(a1.hypot(b1), a2.hypot(b2));
    let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25f64.powi(7))).sqrt());
    let (a1p, a2p) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (c1p, c2p) = (a1p.hypot(b1), a2p.hypot(b2));
    let hue = |b: f64, a: f64| {
        if b == 0.0 && a == 0.0 {
            0.0
        } else {
            b.atan2(a).to_degrees().rem_euclid(360.0)
        }
    };
    let (h1p, h2p) = (hue(b1, a1p), hue(b2, a2p));
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dhh = 2.0 * (c1p * c2p).sqrt() * (dh / 2.0).to_radians().sin();
    let lbp = f64::midpoint(l1, l2);
    let cbp = f64::midpoint(c1p, c2p);
    let hbp = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        f64::midpoint(h1p, h2p)
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (hbp - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hbp).to_radians().cos()
        + 0.32 * (3.0 * hbp + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hbp - 63.0).to_radians().cos();
    let d_theta = 30.0 * (-((hbp - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (cbp.powi(7) / (cbp.powi(7) + 25f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (lbp - 50.0).powi(2) / (20.0 + (lbp - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * d_theta).to_radians().sin() * rc;
    ((dl / sl).powi(2) + (dc / sc).powi(2) + (dhh / sh).powi(2) + rt * (dc / sc) * (dhh / sh))
        .sqrt()
}

/// A directory of this test's own.
pub(crate) fn tempfile_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rr-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
