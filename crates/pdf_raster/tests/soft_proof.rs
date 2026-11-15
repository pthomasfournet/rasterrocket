//! Soft proofs against a document's output intent: an RGB page shows the
//! press's gamut and paper, `DeviceCMYK` ink prints as itself, overprint
//! proofs as printed; the refusals of documents and requests a proof
//! cannot serve.

#[expect(
    dead_code,
    reason = "each test binary compiles all shared helpers and uses a subset"
)]
mod common;

use common::{assemble_pdf, stream_object};
use rasterrocket::{
    BackendPolicy, FinishedCanvas, OutputIntentError, ProofIntent, RasterError, RasterSession,
    RenderParams, SampleDepth, SessionConfig, open_session_from_bytes, render_page_canvas,
};

/// The bytes of the synthetic CMYK printer of `tests/fixtures/icc`.
fn icc(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/icc")
        .join(name);
    std::fs::read(path).unwrap()
}

/// A one-page 40 × 10 pt document running `content`, with the resources
/// `resources` and an output intent of subtype `subtype` whose profile is
/// `profile` (of `n` components); without an output intent when `None`.
fn document(content: &str, resources: &str, intent: Option<(&str, &[u8], u8)>) -> Vec<u8> {
    let catalog = match intent {
        Some((subtype, _, _)) => format!(
            "<< /Type /Catalog /Pages 2 0 R /OutputIntents [<< /Type /OutputIntent \
             /S /{subtype} /OutputConditionIdentifier (Test press) /DestOutputProfile 5 0 R >>] >>"
        ),
        None => "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
    };
    let profile = intent.map_or_else(
        || b"null".to_vec(),
        |(_, bytes, n)| stream_object(&format!("/N {n}"), bytes),
    );
    assemble_pdf(&[
        catalog.into_bytes(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 40 10] /Contents 4 0 R \
             /Resources << {resources} >> >>"
        )
        .into_bytes(),
        stream_object("", content.as_bytes()),
        profile,
    ])
}

/// A PDF/X document of `content` on the synthetic CMYK press.
fn pdfx(content: &str, resources: &str) -> Vec<u8> {
    let press = icc("printer-cmyk-v4.icc");
    document(content, resources, Some(("GTS_PDFX", &press, 4)))
}

fn open(
    pdf: Vec<u8>,
    proof: Option<ProofIntent>,
    overprint: bool,
) -> Result<RasterSession, RasterError> {
    let mut config = SessionConfig::with_policy(BackendPolicy::CpuOnly);
    config.proof = proof;
    config.overprint = overprint;
    open_session_from_bytes(pdf, &config)
}

/// The 8-bit RGB pixel at `x` of the middle row of page 1 at 72 dpi.
fn pixel(session: &RasterSession, params: &RenderParams, x: u32) -> [u8; 3] {
    match render_page_canvas(session, 1, params).expect("renders") {
        FinishedCanvas::Rgb8(b) => {
            let row = b.row_samples(b.height / 2);
            let i = x as usize * 3;
            [row[i], row[i + 1], row[i + 2]]
        }
        other => panic!("an 8-bit RGB page, not {other:?}"),
    }
}

/// Paper (x 5), sRGB red (x 15), gray (x 25) and `DeviceCMYK` black (x
/// 35), each 10 pt wide.
const SWATCHES: &str = "1 0 0 rg 10 0 10 10 re f 0.5 g 20 0 10 10 re f 0 0 0 1 k 30 0 10 10 re f";

/// A page of RGB and gray content alone proofs on the press: paper white,
/// red dulled to the press's red, gray kept neutral; `DeviceCMYK` black
/// prints as the press's black ink, not sRGB black.  Absolute colorimetry
/// tints the paper.  The values are `transicc`'s proof within 3 levels.
#[test]
fn an_rgb_page_proofs_on_the_press() {
    let params = RenderParams::new(1.0);
    let plain = open(pdfx(SWATCHES, ""), None, false).unwrap();
    assert_eq!(pixel(&plain, &params, 15), [255, 0, 0]);
    let relative = open(
        pdfx(SWATCHES, ""),
        Some(ProofIntent::RelativeColorimetric),
        false,
    )
    .unwrap();
    // transicc -i '*sRGB' -o '*sRGB' -p printer-cmyk-v4.icc -t1 -m1; black
    // ink from -i printer-cmyk-v4.icc -t1.
    let want: [(u32, [f64; 3]); 4] = [
        (5, [253.6, 254.4, 253.9]),
        (15, [239.6, 77.4, 77.4]),
        (25, [128.0, 128.0, 128.0]),
        (35, [23.5, 23.5, 23.5]),
    ];
    for (x, want) in want {
        let got = pixel(&relative, &params, x);
        for (g, w) in got.iter().zip(want) {
            assert!(
                (f64::from(*g) - w).abs() <= 3.0,
                "x {x}: {got:?} for {want:?}"
            );
        }
    }
    let absolute = open(
        pdfx(SWATCHES, ""),
        Some(ProofIntent::AbsoluteColorimetric),
        false,
    )
    .unwrap();
    let paper = pixel(&absolute, &params, 5);
    assert!(
        paper[2] + 5 <= paper[0],
        "absolute colorimetry shows the paper's tint: {paper:?}"
    );
}

/// Under overprint the proof prints the overlap of cyan and an
/// overprinting magenta as both inks, blue; without it as magenta.
#[test]
fn overprinted_ink_proofs_as_printed() {
    let content = "1 0 0 0 k 0 0 30 10 re f /OP1 gs 0 1 0 0 k 10 0 30 10 re f";
    let resources = "/ExtGState << /OP1 << /Type /ExtGState /OP true /op true /OPM 1 >> >>";
    let params = RenderParams::new(1.0);
    let proof = Some(ProofIntent::RelativeColorimetric);
    let knocked = pixel(
        &open(pdfx(content, resources), proof, false).unwrap(),
        &params,
        20,
    );
    let printed = pixel(
        &open(pdfx(content, resources), proof, true).unwrap(),
        &params,
        20,
    );
    assert!(knocked[0] > 200 && knocked[1] < 140, "magenta: {knocked:?}");
    assert!(
        printed[0] < 120 && printed[1] < 120 && printed[2] > 120,
        "cyan and magenta print blue: {printed:?}"
    );
}

/// A 16-bit proof is the 8-bit one within a level.
#[test]
fn sixteen_bit_proofs_agree_with_eight_bit_ones() {
    let session = open(
        pdfx(SWATCHES, ""),
        Some(ProofIntent::RelativeColorimetric),
        false,
    )
    .unwrap();
    let eight = pixel(&session, &RenderParams::new(1.0), 15);
    let FinishedCanvas::Rgb16(b) = render_page_canvas(
        &session,
        1,
        &RenderParams::new(1.0).with_depth(SampleDepth::Sixteen),
    )
    .unwrap() else {
        panic!("a 16-bit RGB page")
    };
    let row = b.row_samples(b.height / 2);
    for (e, s) in eight.iter().zip(&row[45..48]) {
        assert!(
            (f64::from(*e) - f64::from(*s) / 257.0).abs() <= 1.0,
            "{eight:?} {row:?}"
        );
    }
}

/// A document without an output intent, or with a display's (RGB) one,
/// has no press to proof against; the error says why.
#[test]
fn documents_without_a_press_are_refused() {
    let proof = Some(ProofIntent::RelativeColorimetric);
    let refusal = |pdf| match open(pdf, proof, false) {
        Ok(_) => panic!("the proof opens"),
        Err(e) => e,
    };
    let err = refusal(document(SWATCHES, "", None));
    assert!(
        matches!(err, RasterError::OutputIntent(OutputIntentError::Missing)),
        "{err}"
    );
    let srgb = icc("printer-rgb-v4.icc");
    let err = refusal(document(SWATCHES, "", Some(("GTS_PDFA1", &srgb, 3))));
    let message = err.to_string();
    assert!(
        message.contains("/GTS_PDFA1 names a profile of RGB values")
            && message.contains("nothing to proof against; use --output-profile"),
        "{message}"
    );
}

/// A proof writes neither CMYK pages nor pages over a transparent
/// backdrop.
#[test]
fn cmyk_and_transparent_proofs_are_refused() {
    let session = open(
        pdfx(SWATCHES, ""),
        Some(ProofIntent::RelativeColorimetric),
        false,
    )
    .unwrap();
    for params in [
        RenderParams::new(1.0).with_cmyk(),
        RenderParams::new(1.0).with_transparent_background(),
    ] {
        assert!(matches!(
            render_page_canvas(&session, 1, &params),
            Err(RasterError::InvalidOptions(_))
        ));
    }
}
