//! Type 3 text recorded for the device against the CPU render.

use pdf::Document;

use super::super::DeviceRecording;
use super::{DEPTHS, Rng, backend, cpu_aa, device_aa, record};

/// `body` as a stream object.
fn stream(body: &str) -> Vec<u8> {
    format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()).into_bytes()
}

/// A page whose resources name `/T3`, a Type 3 font of two `d1` glyphs
/// (`a`, a triangle; `b`, a square with a hole, even-odd), `/D0`, a Type 3
/// font whose glyph `a` is a `d0` glyph, and the graphics state `/T`
/// (alpha).
fn type3_doc() -> Document {
    let font = |procs: &str| {
        format!(
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1000 1000] \
             /FontMatrix [0.001 0 0 0.001 0 0] /CharProcs << {procs} >> \
             /Encoding << /Differences [97 /a /b] >> /FirstChar 97 /LastChar 98 \
             /Widths [1000 900] >>"
        )
        .into_bytes()
    };
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            b"<< /Type /Page /MediaBox [0 0 200 200] /Resources << \
              /Font << /T3 5 0 R /D0 8 0 R >> /ExtGState << /T << /ca 0.4 >> >> >> >>"
                .to_vec(),
        ),
        (5, font("/a 6 0 R /b 7 0 R")),
        (
            6,
            stream("1000 0 0 0 800 800 d1 0 0 m 800 0 l 400 800 l h f"),
        ),
        (
            7,
            stream("900 0 0 0 900 900 d1 0 0 900 900 re 200 200 500 500 re f*"),
        ),
        (8, font("/a 9 0 R /b 9 0 R")),
        (9, stream("1000 0 d0 1 0 0 rg 0 0 800 800 re f")),
    ])
}

/// Cached `d1` glyphs at random sizes, positions and colours, partly
/// clipped, recorded and rendered on the device (or the host model) match
/// the CPU render.
#[test]
fn cached_type3_glyphs_match_the_cpu_render() {
    let doc = type3_doc();
    let backend = backend();
    let mut rng = Rng(0x7_3E3);
    let (width, height) = (180, 140);
    for round in 0..4 {
        let mut s = Vec::new();
        for _ in 0..30 {
            let (x, y) = (rng.coord(width), rng.coord(height));
            let size = rng.unit().mul_add(40.0, 2.0);
            let rgb: Vec<f64> = (0..3).map(|_| rng.unit()).collect();
            if rng.below(4) == 0 {
                s.push(format!(
                    "q {} {} 50 40 re W n",
                    rng.coord(width),
                    rng.coord(height)
                ));
            } else {
                s.push("q".into());
            }
            s.push(format!(
                "{} {} {} rg BT /T3 {size} Tf {x} {y} Td (abba ab) Tj ET Q",
                rgb[0], rgb[1], rgb[2]
            ));
        }
        let content = s.join(" ").into_bytes();
        let what = String::from_utf8_lossy(&content);
        match record(&doc, width, height, &content) {
            DeviceRecording::Refused(reason) => panic!("round {round} refused: {reason}\n{what}"),
            DeviceRecording::Recorded(page) => assert!(page.len() > 50, "few commands"),
        }
        for depth in DEPTHS {
            let want = cpu_aa(&doc, (width, height), depth, &content, true);
            let paths = backend
                .as_ref()
                .is_some_and(gpu::backend::vulkan::VulkanBackend::page_paths);
            for b in [None, backend.as_ref()] {
                let got = device_aa(b, &doc, (width, height), depth, &content, (paths, true));
                assert!(
                    got == want,
                    "{} bits, round {round}, device {}:\n{what}",
                    depth.bits(),
                    b.is_some()
                );
            }
        }
    }
}

/// A Type 3 glyph the CPU draws by running its `CharProc` (a `d0` glyph,
/// a `d1` glyph under a translucent fill) refuses the recording.
#[test]
fn type3_glyphs_drawn_by_their_charprocs_refuse_the_recording() {
    let doc = type3_doc();
    for content in [
        &b"BT /D0 20 Tf 5 5 Td (a) Tj ET"[..],
        b"/T gs BT /T3 20 Tf 5 5 Td (a) Tj ET",
    ] {
        match record(&doc, 60, 40, content) {
            DeviceRecording::Refused(reason) => assert_eq!(reason, "Type 3 glyph"),
            DeviceRecording::Recorded(_) => panic!("recorded"),
        }
    }
}
