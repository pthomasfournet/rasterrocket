//! Path clips and clipping text modes recorded for the device against the
//! CPU render.

use std::fmt::Write as _;

use gpu::backend::page_device::PageDevice;
use pdf::Document;

use super::super::DeviceRecording;
use super::forms::{form_doc, random_forms};
use super::images::{image_doc, random_images};
use super::text::random_text;
use super::{DEPTHS, Rng, VulkanBackend, backend, cpu, device, doc, random_content, record};

#[expect(
    clippy::many_single_char_names,
    reason = "coordinates, matrix entries and the stream under construction"
)]
/// `n` random path clips over a `width × height` page: polygons and
/// curves under both rules, some under a rotated or skewed matrix, and
/// rectangles under a rotation (a rectangle clip otherwise).
fn clip_paths(rng: &mut Rng, width: u32, height: u32, n: usize) -> String {
    let (w, h) = (width, height);
    let mut s = Vec::new();
    for _ in 0..n {
        let rule = if rng.below(3) == 0 { "W*" } else { "W" };
        match rng.below(4) {
            0 => {
                let (a, b) = (rng.unit().mul_add(1.6, -0.8), rng.unit().mul_add(1.6, -0.8));
                let (x, y) = (rng.coord(w), rng.coord(h));
                s.push(format!(
                    "1 {a} {b} 1 {x} {y} cm 0 0 {} {} re {rule} n 1 0 0 1 0 0 cm",
                    rng.coord(w) / 2.0,
                    rng.coord(h) / 2.0
                ));
                // Back to the page's matrix for the paints that follow.
                let det = a.mul_add(-b, 1.0);
                s.push(format!(
                    "{} {} {} {} 0 0 cm {} {} cm",
                    1.0 / det,
                    -a / det,
                    -b / det,
                    1.0 / det,
                    -x,
                    -y
                ));
            }
            1 => {
                let p: Vec<f64> = (0..7).flat_map(|_| [rng.coord(w), rng.coord(h)]).collect();
                s.push(format!(
                    "{} {} m {} {} {} {} {} {} c {} {} {} {} {} {} c h {rule} n",
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    p[5],
                    p[6],
                    p[7],
                    p[8],
                    p[9],
                    p[10],
                    p[11],
                    p[12],
                    p[13]
                ));
            }
            _ => {
                let mut path = format!("{} {} m", rng.coord(w), rng.coord(h));
                for _ in 0..2 + rng.below(7) {
                    let _ = write!(path, " {} {} l", rng.coord(w), rng.coord(h));
                }
                s.push(format!("{path} h {rule} n"));
            }
        }
    }
    s.join(" ")
}

/// `body`, under `n` random path clips in two nested levels, then again
/// unclipped.
fn clipped(rng: &mut Rng, (width, height): (u32, u32), n: usize, body: &[u8]) -> Vec<u8> {
    let inner = n / 2;
    let mut s = format!("q {} ", clip_paths(rng, width, height, n - inner)).into_bytes();
    s.extend(body);
    s.extend(format!(" q {} ", clip_paths(rng, width, height, inner)).as_bytes());
    s.extend(body);
    s.extend(b" Q Q ");
    s.extend(body);
    s
}

/// The content generators the clips are drawn over, with their documents.
type Generator = fn(&mut Rng, u32, u32) -> Vec<u8>;

fn forms_over(rng: &mut Rng, _: u32, _: u32) -> Vec<u8> {
    random_forms(rng)
}

fn kinds() -> [(&'static str, Document, Generator); 4] {
    [
        ("vectors", doc(), random_content),
        ("text", doc(), random_text),
        ("images", image_doc(), random_images),
        ("forms", form_doc(), forms_over),
    ]
}

/// Random clipped pages recorded and rendered on `backend` (or the host
/// model) match the CPU render at 8 and 16 bits: 1, 3 and 9 regions (the
/// CPU folds from 8 on).
fn clipped_pages_match_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let mut rng = Rng(0x0C11_B5E7);
    for (what, doc, generate) in kinds() {
        let (width, height) = if what == "forms" {
            (120, 90)
        } else {
            (113, 87)
        };
        for n in [1, 3, 9] {
            let body = generate(&mut rng, width, height);
            let content = clipped(&mut rng, (width, height), n, &body);
            let text = String::from_utf8_lossy(&content);
            match record(&doc, width, height, &content) {
                DeviceRecording::Refused(reason) => {
                    panic!("{what}, {n} clips refused: {reason}\n{text}")
                }
                DeviceRecording::Recorded(page) => assert!(page.len() > 8, "{what}: few commands"),
            }
            for depth in DEPTHS {
                assert!(
                    device(backend, &doc, (width, height), depth, &content, paths)
                        == cpu(&doc, width, height, depth, &content),
                    "{what}, {n} clips, {} bits, outlines {paths}:\n{text}",
                    depth.bits()
                );
            }
        }
    }
}

#[test]
fn device_clipped_pages_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    clipped_pages_match_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn device_clipped_pages_of_tables_only_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    clipped_pages_match_the_cpu_render(Some(&backend), false);
}

#[test]
fn host_model_of_clipped_pages_matches_the_cpu_render() {
    clipped_pages_match_the_cpu_render(None, true);
}

/// Text shown in a clipping mode (4–7) paints in modes 4–6 and clips the
/// paints after `ET` to its glyphs, on the device as on the CPU.
#[test]
fn clipping_text_modes_match_the_cpu_render() {
    let backend = backend();
    let doc = doc();
    for mode in 4..=7 {
        let content = format!(
            "0 0 1 RG 1 w BT /F1 30 Tf {mode} Tr 5 10 Td (Clip) Tj /F2 24 Tf 0 -2 Td (ping) Tj ET \
             0.2 0.6 0.1 rg 0 0 100 50 re f BT /F3 12 Tf 0 Tr 2 30 Td (inside) Tj ET \
             q 0.5 g 0 0 m 100 50 l 0 50 l h f Q"
        );
        assert!(
            matches!(
                record(&doc, 100, 50, content.as_bytes()),
                DeviceRecording::Recorded(_)
            ),
            "mode {mode} refused"
        );
        for depth in DEPTHS {
            let want = cpu(&doc, 100, 50, depth, content.as_bytes());
            for paths in [true, false] {
                assert_eq!(
                    device(
                        backend.as_ref(),
                        &doc,
                        (100, 50),
                        depth,
                        content.as_bytes(),
                        paths
                    ),
                    want,
                    "mode {mode}, {} bits, outlines {paths}",
                    depth.bits()
                );
            }
        }
    }
}

/// Random text in every mode, the clipping ones included, each text
/// object clipping what follows it in its `q`/`Q` group.
#[test]
fn random_text_in_clipping_modes_matches_the_cpu_render() {
    let backend = backend();
    let doc = doc();
    let mut rng = Rng(0x7E47_C11B);
    let (width, height) = (130, 90);
    for round in 0..6 {
        let body = String::from_utf8(random_text(&mut rng, width, height)).expect("ASCII");
        // Every third text object in a clipping mode, its group painted
        // over after the text.
        let parts: Vec<String> = body
            .split("ET Q")
            .enumerate()
            .map(|(k, part)| match part.find(" Tr") {
                Some(at) if k % 3 == 2 && at > 0 => {
                    let mode = 4 + (k / 3) % 4;
                    format!("{}{mode}{}", &part[..at - 1], &part[at..])
                }
                _ => part.to_string(),
            })
            .collect();
        let body = parts.join("ET 0.3 0.4 0.9 rg 0 0 130 90 re f Q");
        let content = body.into_bytes();
        let text = String::from_utf8_lossy(&content);
        for depth in DEPTHS {
            assert!(
                device(
                    backend.as_ref(),
                    &doc,
                    (width, height),
                    depth,
                    &content,
                    true
                ) == cpu(&doc, width, height, depth, &content),
                "round {round}, {} bits:\n{text}",
                depth.bits()
            );
        }
    }
}

/// A slanted triangle shifted by `(dx, dy)` pixels, as a path.
fn triangle(dx: f64, dy: f64) -> String {
    format!(
        "{} {} m {} {} l {} {} l h",
        3.3 + dx,
        2.1 + dy,
        61.7 + dx,
        17.9 + dy,
        24.2 + dx,
        58.6 + dy
    )
}

/// Four clips of nearly coincident triangles under paints whose own edges
/// lie on theirs, so most edge pixels are partly covered by every clip
/// region and by the paint: there a shape scaled region by region (fills)
/// and one scaled by the clips' product (glyphs, images) differ, and each
/// must match its CPU consumer, at 8 and 16 bits.
#[test]
fn stacked_clip_edges_match_the_cpu_render() {
    let backend = backend();
    let clips: String = (0..4)
        .map(|k| {
            let d = f64::from(k) * 0.23;
            format!("{} W n ", triangle(d, -d * 0.7))
        })
        .collect::<Vec<_>>()
        .join(" ");
    let bodies: [(&str, Document, String); 3] = [
        (
            "fills",
            doc(),
            (0..3)
                .map(|k| {
                    let d = f64::from(k).mul_add(0.17, 0.11);
                    format!("/T gs 0.{k}3 0.7 0.2 rg {} f ", triangle(-d, d))
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        (
            "text",
            doc(),
            "/T gs 0 0 0.6 rg BT /F1 40 Tf 1 8 Td (WMW) Tj 0 Tr /F2 31 Tf 3 14 Td (gl) Tj ET \
             /O gs 0.7 0 0 rg BT /F3 22 Tf 10 30 Td (xx) Tj ET"
                .into(),
        ),
        (
            "images",
            image_doc(),
            ["/M", "/N", "/Q", "/S", "/D"]
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    format!(
                        "q 0.4 0.1 0.8 rg 55 4 -3 58 {} 2 cm {name} Do Q ",
                        3 + 2 * k
                    )
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
    ];
    let mut failed = Vec::new();
    for (what, doc, body) in bodies {
        let content = format!("q {clips} {body} Q");
        for depth in DEPTHS {
            let want = cpu(&doc, 64, 64, depth, content.as_bytes());
            assert!(
                want.iter().any(|&v| v != depth.max_sample()),
                "{what}: nothing painted"
            );
            for paths in [true, false] {
                let got = device(
                    backend.as_ref(),
                    &doc,
                    (64, 64),
                    depth,
                    content.as_bytes(),
                    paths,
                );
                let model = device(None, &doc, (64, 64), depth, content.as_bytes(), paths);
                for (on, pixels) in [("device", got), ("host model", model)] {
                    if pixels != want {
                        failed.push(format!(
                            "{what}, {} bits, outlines {paths}, {on}",
                            depth.bits()
                        ));
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "differ from the CPU: {failed:#?}");
}
