//! Helpers shared by the end-to-end render parity tests: synthetic PDF
//! assembly, session construction per backend policy, and byte-exact page
//! comparison.

pub mod icc;

use std::fmt::Write as _;
use std::path::PathBuf;

use rasterrocket::{
    BackendPolicy, RasterSession, SessionConfig, open_session_from_bytes, render_page_rgb,
};

/// 150 DPI: the resolution the parity findings were measured at.
pub const SCALE: f64 = 150.0 / 72.0;

/// Assemble a one-page PDF from `objects` (1-based, in order) with an
/// exact xref table.
pub fn assemble_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.5\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_at = pdf.len();
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for off in offsets {
        let _ = writeln!(xref, "{off:010} 00000 n ");
    }
    let _ = write!(
        xref,
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
        objects.len() + 1
    );
    pdf.extend_from_slice(xref.as_bytes());
    pdf
}

/// A stream object body with `dict` entries and an exact `/Length`.
pub fn stream_object(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut obj = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
    obj.extend_from_slice(data);
    obj.extend_from_slice(b"\nendstream");
    obj
}

/// Path of a corpus fixture under `tests/fixtures`, or `None` (with a
/// note on stderr) when it is absent from this checkout.
pub fn corpus_fixture(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: corpus fixture absent ({})", path.display());
        None
    }
}

/// `(width, height, rgb bytes)` of one rendered page.
pub fn render(session: &RasterSession, page: u32) -> (u32, u32, Vec<u8>) {
    let bmp = render_page_rgb(session, page, SCALE).expect("render page");
    (bmp.width, bmp.height, bmp.data().to_vec())
}

/// Assert two renders are byte-identical, reporting the divergence
/// statistics the way the original findings were measured.
pub fn assert_identical(cpu: &(u32, u32, Vec<u8>), gpu: &(u32, u32, Vec<u8>), what: &str) {
    assert_eq!((cpu.0, cpu.1), (gpu.0, gpu.1), "{what}: dimensions differ");
    assert_eq!(cpu.2.len(), gpu.2.len(), "{what}: byte length differs");
    let differing = cpu.2.iter().zip(&gpu.2).filter(|(a, b)| a != b).count();
    let max_delta = cpu
        .2
        .iter()
        .zip(&gpu.2)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert_eq!(
        differing,
        0,
        "{what}: {differing} of {} bytes differ between the CPU and GPU renders (max delta {max_delta})",
        cpu.2.len()
    );
}

/// Open `bytes` under `policy`.
pub fn session(bytes: Vec<u8>, policy: BackendPolicy) -> RasterSession {
    open_session_from_bytes(bytes, &SessionConfig::with_policy(policy))
        .unwrap_or_else(|e| panic!("{policy:?} session: {e}"))
}

/// Catalog 1, Pages 2, then `page` as object 3 and `rest` from object 4 on.
pub fn one_page(page: &str, rest: Vec<Vec<u8>>) -> Vec<u8> {
    let mut objects = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        page.as_bytes().to_vec(),
    ];
    objects.extend(rest);
    assemble_pdf(&objects)
}

/// Fraction of page 1's pixels that are not white, at 72 dpi on the CPU.
pub fn ink(pdf: Vec<u8>) -> f64 {
    let s = session(pdf, BackendPolicy::CpuOnly);
    let bmp = render_page_rgb(&s, 1, 1.0).expect("page renders");
    let (px, _) = bmp.data().as_chunks::<3>();
    let total = px.len();
    let dark = px.iter().filter(|p| p.iter().any(|&c| c < 128)).count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts of a test page are far below 2^52"
    )]
    let frac = dark as f64 / total as f64;
    frac
}

/// A 200 × 100 pt page whose content fills the 80 × 80 square at (10, 10):
/// 32 % of the page.
pub const SQUARE: &[u8] = b"0 g 10 10 80 80 re f";
