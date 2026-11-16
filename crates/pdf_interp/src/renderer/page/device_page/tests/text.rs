//! Text recorded for the device against the CPU render.

use super::super::DeviceRecording;
use super::{DEPTHS, Rng, VulkanBackend, backend, cpu_aa, device_aa, doc, record};

/// The strings [`random_text`] shows.
const WORDS: [&str; 8] = [
    "Hello, world!",
    "AVAWfiffl",
    "The quick brown fox",
    "0123456789",
    "jumps\\(over\\)",
    "WAVE Typography",
    "i",
    "%&@#*",
];

/// A random content stream of text: the three fonts at sizes from below a
/// pixel to past the glyph-bitmap limit, filled, stroked (cached and as
/// paths: dashed, translucent, wide), filled and stroked, and invisible,
/// in solid colours at both alphas, under rectangle clips, with character,
/// word and horizontal scaling, rise, rotated and skewed text matrices,
/// kerned arrays and the next-line operators.
#[expect(
    clippy::many_single_char_names,
    reason = "coordinates, matrix entries and the stream under construction"
)]
fn random_text(rng: &mut Rng, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width, height);
    let mut s = vec![String::from("1 0 0 1 0.37 -0.21 cm")];
    for k in 0..24 {
        s.push("q".into());
        if rng.below(4) == 0 {
            let (x, y, dx, dy) = (rng.coord(w), rng.coord(h), rng.coord(w), rng.coord(h));
            s.push(format!("{x} {y} {} {} re W n", dx / 2.0, dy / 2.0));
        }
        let mode = [0, 0, 0, 1, 2, 3][usize::try_from(rng.below(6)).unwrap()];
        let translucent = rng.below(3) == 0;
        s.push(if translucent { "/T gs" } else { "/O gs" }.into());
        let rgb: Vec<f64> = (0..6).map(|_| rng.unit()).collect();
        s.push(format!(
            "{} {} {} rg {} {} {} RG",
            rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        ));
        let line_width = rng.unit().mul_add(2.5, 0.05);
        s.push(format!("{line_width} w {} j", rng.below(3)));
        if rng.below(4) == 0 {
            s.push(format!("[{} 1] 0 d", rng.unit().mul_add(3.0, 0.5)));
        }
        let font = 1 + rng.below(3);
        let size = match rng.below(8) {
            0 => rng.unit() * 0.9,
            1 => rng.unit().mul_add(60.0, 20.0),
            _ => rng.unit().mul_add(14.0, 3.0),
        };
        s.push(format!("BT /F{font} {size} Tf {mode} Tr"));
        s.push(format!(
            "{} Tc {} Tw {} Tz {} Ts {} TL",
            rng.unit() - 0.3,
            rng.unit() * 3.0,
            rng.unit().mul_add(120.0, 40.0),
            rng.unit().mul_add(4.0, -2.0),
            size * 1.2
        ));
        let (x, y) = (rng.coord(w), rng.coord(h));
        if k == 11 {
            // Past the glyph-bitmap limit: the glyphs fill as paths.
            s.push(format!("2600 0 0 2600 {x} {y} Tm /F1 1 Tf"));
        } else if rng.below(3) == 0 {
            let (a, b) = (rng.unit().mul_add(2.0, -1.0), rng.unit().mul_add(2.0, -1.0));
            s.push(format!(
                "{} {b} {} {} {x} {y} Tm",
                a + 1.0,
                -b * 0.5,
                a + 1.0
            ));
        } else {
            s.push(format!("{x} {y} Td"));
        }
        let word = WORDS[usize::try_from(rng.below(8)).unwrap()];
        s.push(match rng.below(4) {
            0 => format!(
                "[({word}) {} (Ty) -250 (W)] TJ",
                i64::from(rng.below(400)) - 200
            ),
            1 => format!("({word}) Tj T* ({word}) '"),
            2 => format!("1 2 ({word}) \""),
            _ => format!("({word}) Tj"),
        });
        s.push("ET Q".into());
    }
    s.join(" ").into_bytes()
}

/// Random text pages recorded and rendered on `backend` (or the host
/// model) match the CPU render at both depths, with antialiased glyphs or
/// not.
fn recorded_text_matches_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let doc = doc();
    let mut rng = Rng(0x7E47);
    for (width, height) in [(1, 1), (151, 97), (200, 200)] {
        for round in 0..5 {
            let content = random_text(&mut rng, width, height);
            let what = String::from_utf8_lossy(&content);
            match record(&doc, width, height, &content) {
                DeviceRecording::Refused(reason) => {
                    panic!("{width}×{height}, round {round} refused: {reason}\n{what}")
                }
                DeviceRecording::Recorded(page) => {
                    assert!(width < 10 || page.len() > 30, "few commands recorded");
                }
            }
            for depth in DEPTHS {
                for text_aa in [true, false] {
                    let want = cpu_aa(&doc, (width, height), depth, &content, text_aa);
                    let paper = depth.max_sample();
                    assert!(
                        width < 10 || want.iter().any(|&v| v != paper),
                        "the page is paper"
                    );
                    let got = device_aa(
                        backend,
                        &doc,
                        (width, height),
                        depth,
                        &content,
                        (paths, text_aa),
                    );
                    assert!(
                        got == want,
                        "{} bits, {width}×{height}, round {round}, outlines {paths}, \
                         text antialiased {text_aa}:\n{what}",
                        depth.bits()
                    );
                }
            }
        }
    }
}

#[test]
fn device_text_matches_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_text_matches_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn device_text_of_tables_and_masks_only_matches_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_text_matches_the_cpu_render(Some(&backend), false);
}

#[test]
fn host_model_of_recorded_text_matches_the_cpu_render() {
    recorded_text_matches_the_cpu_render(None, true);
}

/// Each glyph bitmap of a page is held once, however often it is shown.
#[test]
fn a_repeated_glyph_is_held_once() {
    let doc = doc();
    let bytes = |content: &[u8]| match record(&doc, 1_100, 80, content) {
        DeviceRecording::Recorded(page) => (page.len(), page.bytes()),
        DeviceRecording::Refused(reason) => panic!("refused: {reason}"),
    };
    let (_, empty) = bytes(b"");
    let (n1, one) = bytes(b"BT /F1 60 Tf 5 20 Td (o) Tj ET");
    let (n30, thirty) = bytes(b"BT /F1 60 Tf 5 20 Td (oooooooooooooooooooooooooooooo) Tj ET");
    assert_eq!((n1, n30), (1, 30));
    // One glyph: its mask, its command and its items.  Each further one
    // adds a command and at most nine items, and the glyph's other three
    // quarter-pixel phases at most three masks.
    let glyph = one - empty;
    assert!(
        thirty - one <= 29 * (48 + 9 * 16) + 3 * glyph,
        "30 glyphs take {} bytes, one {glyph}",
        thirty - empty
    );
}

/// A clipping text rendering mode refuses the recording, whatever else
/// it paints; the page then renders on the CPU, never with a clip
/// missing.
#[test]
fn clipping_text_modes_refuse_the_recording() {
    let doc = doc();
    for mode in 4..=7 {
        let content =
            format!("BT /F1 30 Tf {mode} Tr 5 10 Td (Clip) Tj ET 0 0 1 rg 0 0 200 200 re f");
        match record(&doc, 100, 50, content.as_bytes()) {
            DeviceRecording::Refused(reason) => assert_eq!(reason, "text clip mode", "mode {mode}"),
            DeviceRecording::Recorded(_) => panic!("mode {mode} was recorded"),
        }
    }
}

/// Text painted in a way the device raster does not composite refuses the
/// recording with the reason.
#[test]
fn unsupported_text_paints_refuse_the_recording() {
    let doc = doc();
    let refused = |content: &[u8]| match record(&doc, 60, 40, content) {
        DeviceRecording::Refused(reason) => reason,
        DeviceRecording::Recorded(_) => "",
    };
    assert_eq!(
        refused(b"/Pattern cs /P scn BT /F1 12 Tf 5 5 Td (a) Tj ET"),
        "pattern"
    );
    assert_eq!(
        refused(b"/M gs BT /F1 12 Tf 5 5 Td (a) Tj ET"),
        "blend mode"
    );
    assert_eq!(
        refused(b"0 0 m 50 0 l 25 30 l h W n BT /F1 12 Tf 5 5 Td (a) Tj ET"),
        "path clip"
    );
    assert_eq!(refused(b"BT /F1 12 Tf 3 Tr 5 5 Td (a) Tj ET"), "");
}
