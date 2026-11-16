use color::SampleDepth;
use gpu::backend::vulkan::{DeviceSelector, VulkanBackend};
use pdf::Document;

use super::super::{CanvasBuffers, CanvasRequest};
use super::{DeviceRecording, PageCanvas, PageRenderer, device_page_ineligible};

const DEPTHS: [SampleDepth; 2] = [SampleDepth::Eight, SampleDepth::Sixteen];

/// A page whose resources name the graphics states and fonts the tests
/// use: `/T` (fill and stroke alpha), `/O` (opaque), `/M` (`Multiply`);
/// `/F1`–`/F3`, three of the standard 14.
fn doc() -> Document {
    crate::test_helpers::make_doc_with_objects(&[(
        4,
        "<< /Type /Page /MediaBox [0 0 200 200] /Resources << /ExtGState << \
         /T << /ca 0.37 /CA 0.61 >> /O << /ca 1 /CA 1 >> /M << /BM /Multiply >> \
         >> /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> \
         /F2 << /Type /Font /Subtype /Type1 /BaseFont /Times-BoldItalic >> \
         /F3 << /Type /Font /Subtype /Type1 /BaseFont /Courier >> >> >> >>",
    )])
}

/// The device `PDF_RASTER_VULKAN_DEVICE` names, or the automatically
/// ranked one; `None` (the test is skipped) when no device opens.
fn backend() -> Option<VulkanBackend> {
    let selector = std::env::var("PDF_RASTER_VULKAN_DEVICE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<DeviceSelector>().expect("device selector"));
    VulkanBackend::with_selector(selector.as_ref())
        .inspect_err(|e| eprintln!("skipping: no Vulkan backend: {e}"))
        .ok()
}

/// A renderer of a `width × height` page on an RGB canvas of `depth`.
fn renderer(doc: &Document, width: u32, height: u32, depth: SampleDepth) -> PageRenderer<'_> {
    PageRenderer::try_with_page_group_in(
        &PageRenderer::unit_view(width, height),
        doc,
        (4, 0),
        None,
        &CanvasRequest::Rgb,
        depth,
        CanvasBuffers::default(),
    )
    .expect("the canvas is allocated")
}

/// [`renderer`] on a canvas whose samples are unspecified
/// ([`PageRenderer::try_unpainted_in`]), built in buffers that hold
/// samples no page paints.
fn unpainted(doc: &Document, width: u32, height: u32, depth: SampleDepth) -> PageRenderer<'_> {
    unpainted_aa(doc, width, height, depth, true)
}

/// [`unpainted`], its glyphs antialiased when `text_aa`.
fn unpainted_aa(
    doc: &Document,
    width: u32,
    height: u32,
    depth: SampleDepth,
    text_aa: bool,
) -> PageRenderer<'_> {
    let len = width as usize * height as usize * 3;
    let mut buffers = CanvasBuffers::default();
    buffers.rgb.data = vec![0x5A; len];
    buffers.rgb16.data = vec![0x5A5A; len];
    PageRenderer::try_unpainted_in(
        &PageRenderer::unit_view(width, height),
        doc,
        (4, 0),
        &CanvasRequest::Rgb,
        depth,
        buffers,
    )
    .map(|mut r| {
        r.set_antialias(text_aa, true);
        r
    })
    .expect("the canvas is allocated")
}

/// The canvas samples of `r`, widened to 16 bits.
fn samples(r: &PageRenderer<'_>) -> Vec<u16> {
    match &r.canvas {
        PageCanvas::Rgb(b) => b.data().iter().map(|&v| u16::from(v)).collect(),
        PageCanvas::Rgb16(b) => b.data().to_vec(),
        PageCanvas::Ink(_) | PageCanvas::Ink16(_) => panic!("an RGB canvas"),
    }
}

/// The canvas samples of `content` on a `width × height` page of
/// `depth`.
fn cpu(doc: &Document, width: u32, height: u32, depth: SampleDepth, content: &[u8]) -> Vec<u16> {
    cpu_aa(doc, (width, height), depth, content, true)
}

/// [`cpu`], its glyphs antialiased when `text_aa`.
fn cpu_aa(
    doc: &Document,
    (width, height): (u32, u32),
    depth: SampleDepth,
    content: &[u8],
    text_aa: bool,
) -> Vec<u16> {
    let mut r = renderer(doc, width, height, depth);
    r.set_antialias(text_aa, true);
    r.execute(&crate::content::parse(content));
    samples(&r)
}

/// The recording of `content` on an 8-bit `width × height` page, its
/// outlines recorded as outlines.
fn record(doc: &Document, width: u32, height: u32, content: &[u8]) -> DeviceRecording {
    let mut r = PageRenderer::new(width, height, doc, (4, 0));
    r.record_for_device(true).expect("an RGB page records");
    r.execute(&crate::content::parse(content));
    r.take_device_recording().expect("recorded")
}

/// The canvas samples of `content` on a page of `depth`, recorded (its
/// outlines as outlines when `paths`, else as tables) and rendered on
/// `backend`, or by the device raster's host model without one.
fn device(
    backend: Option<&VulkanBackend>,
    doc: &Document,
    size: (u32, u32),
    depth: SampleDepth,
    content: &[u8],
    paths: bool,
) -> Vec<u16> {
    device_aa(backend, doc, size, depth, content, (paths, true))
}

/// [`device`], its glyphs antialiased when `text_aa`.
fn device_aa(
    backend: Option<&VulkanBackend>,
    doc: &Document,
    (width, height): (u32, u32),
    depth: SampleDepth,
    content: &[u8],
    (paths, text_aa): (bool, bool),
) -> Vec<u16> {
    let mut r = unpainted_aa(doc, width, height, depth, text_aa);
    r.record_for_device(paths).expect("an RGB page records");
    r.execute(&crate::content::parse(content));
    let DeviceRecording::Recorded(page) = r.take_device_recording().expect("recorded") else {
        panic!("the content was refused");
    };
    assert_eq!(page.depth(), depth);
    assert!(paths || !page.has_paths(), "outlines recorded as outlines");
    if let Some(backend) = backend {
        r.paint_device_page(backend, &page).expect("device page");
    } else {
        let model = gpu::page::reference::render_page(&page).expect("host model");
        let row = width as usize * 3;
        match &mut r.canvas {
            PageCanvas::Rgb(b) => {
                let stride = b.stride;
                for (dst, src) in b.data_mut().chunks_mut(stride).zip(model.chunks(row)) {
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = u8::try_from(v).expect("an 8-bit sample");
                    }
                }
            }
            PageCanvas::Rgb16(b) => {
                let stride = b.stride;
                for (dst, src) in b.data_mut().chunks_mut(stride).zip(model.chunks(row)) {
                    dst[..row].copy_from_slice(src);
                }
            }
            PageCanvas::Ink(_) | PageCanvas::Ink16(_) => panic!("an RGB canvas"),
        }
    }
    samples(&r)
}

/// `SplitMix64`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u32) -> u32 {
        u32::try_from(self.next() % u64::from(n)).expect("below n")
    }

    /// A coordinate in `[-20, span + 20)`, in hundredths.
    fn coord(&mut self, span: u32) -> f64 {
        f64::from(self.below((span + 40) * 100)) / 100.0 - 20.0
    }

    fn unit(&mut self) -> f64 {
        f64::from(self.below(1001)) / 1000.0
    }
}

/// A random content stream of fills and strokes of solid colours:
/// polygons, curves and rectangles under the non-zero and even-odd
/// rules, strokes of every cap, join and width (hairlines included) and
/// dashes, translucent paints, rectangle clips and a scaled CTM.
#[expect(
    clippy::many_single_char_names,
    reason = "coordinates, colours and the stream under construction"
)]
fn random_content(rng: &mut Rng, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width, height);
    let mut s = vec![String::from("0.83 0 0 0.91 3.3 -2.7 cm")];
    for _ in 0..60 {
        s.push("q".into());
        if rng.below(4) == 0 {
            let (x, y, dx, dy) = (rng.coord(w), rng.coord(h), rng.coord(w), rng.coord(h));
            s.push(format!("{x} {y} {} {} re W n ", dx / 2.0, dy / 2.0));
        }
        let translucent = rng.below(3) == 0;
        s.push(if translucent { "/T gs" } else { "/O gs" }.into());
        let rgb: Vec<f64> = (0..6).map(|_| rng.unit()).collect();
        s.push(format!(
            "{} {} {} rg {} {} {} RG ",
            rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        ));
        let line_width = match rng.below(4) {
            0 if !translucent => 0.0,
            1 => 0.4,
            _ => rng.unit() * 9.0,
        };
        let (cap, join) = (rng.below(3), rng.below(3));
        s.push(format!("{line_width} w {cap} J {join} j "));
        if rng.below(4) == 0 {
            let (on, off, phase) = (rng.unit().mul_add(6.0, 1.0), rng.unit() * 4.0, rng.unit());
            s.push(format!("[{on} {off}] {phase} d "));
        }
        if rng.below(3) == 0 {
            let (x, y, dx, dy) = (rng.coord(w), rng.coord(h), rng.coord(w), rng.coord(h));
            s.push(format!("{x} {y} {} {} re ", dx / 3.0, dy / 3.0));
        } else {
            let (x, y) = (rng.coord(w), rng.coord(h));
            s.push(format!("{x} {y} m "));
            for _ in 0..2 + rng.below(6) {
                if rng.below(3) == 0 {
                    let p: Vec<f64> = (0..3).flat_map(|_| [rng.coord(w), rng.coord(h)]).collect();
                    s.push(format!(
                        "{} {} {} {} {} {} c ",
                        p[0], p[1], p[2], p[3], p[4], p[5]
                    ));
                } else {
                    let (x, y) = (rng.coord(w), rng.coord(h));
                    s.push(format!("{x} {y} l "));
                }
            }
            if rng.below(2) == 0 {
                s.push("h".into());
            }
        }
        let paint = ["f", "f*", "S", "s", "F", "f"][usize::try_from(rng.below(6)).unwrap()];
        s.push(format!("{paint} Q\n"));
        // `B` with a translucent stroke paints as a knockout group,
        // which the recording refuses; it is drawn opaque only.
        if rng.below(5) == 0 {
            let c: Vec<f64> = (0..3).map(|_| rng.unit()).collect();
            let p: Vec<f64> = (0..3).flat_map(|_| [rng.coord(w), rng.coord(h)]).collect();
            s.push(format!(
                "q /O gs {} {} {} rg {} {} m {} {} l {} {} l h B Q\n",
                c[0], c[1], c[2], p[0], p[1], p[2], p[3], p[4], p[5]
            ));
        }
    }
    s.join(" ").into_bytes()
}

/// Random pages recorded, with outlines or as tables only, and rendered
/// on `backend` (or the host model) match the CPU render.
fn recorded_pages_match_the_cpu_render(backend: Option<&VulkanBackend>, paths: bool) {
    let doc = doc();
    let mut rng = Rng(11);
    for (width, height) in [(1, 1), (97, 83), (200, 131), (33, 257)] {
        for round in 0..4 {
            let content = random_content(&mut rng, width, height);
            match record(&doc, width, height, &content) {
                DeviceRecording::Refused(reason) => {
                    panic!("{width}×{height}, round {round} refused: {reason}")
                }
                DeviceRecording::Recorded(page) => {
                    assert!(width < 10 || page.len() > 20, "few commands recorded");
                }
            }
            for depth in DEPTHS {
                let want = cpu(&doc, width, height, depth, &content);
                let paper = depth.max_sample();
                assert!(
                    width < 10 || want.iter().filter(|&&v| v != paper).count() > want.len() / 4,
                    "the page is mostly paper"
                );
                assert_eq!(
                    device(backend, &doc, (width, height), depth, &content, paths),
                    want,
                    "{} bits, {width}×{height}, round {round}, outlines {paths}:\n{}",
                    depth.bits(),
                    String::from_utf8_lossy(&content)
                );
            }
        }
    }
}

mod text;
mod type3;

#[test]
fn device_page_matches_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_pages_match_the_cpu_render(Some(&backend), backend.page_paths());
}

#[test]
fn device_page_of_tables_only_matches_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    recorded_pages_match_the_cpu_render(Some(&backend), false);
}

#[test]
fn host_model_of_a_recording_matches_the_cpu_render() {
    recorded_pages_match_the_cpu_render(None, true);
}

#[test]
fn opaque_hairlines_and_strokes_match_the_cpu_render() {
    let Some(backend) = backend() else {
        return;
    };
    let doc = doc();
    let content = b"0 0 1 RG 0 w 3.2 4.7 m 90.1 60.3 l 10 70 l 70 2 l S \
                    1 0 0 RG 0.5 w 5 5 m 80 80 l S \
                    0 1 0 RG 6 w 1 J 1 j 10 50 m 50 10 l 85 70 l S \
                    q 20 20 40 40 re W n 0 0 0 RG 0 w 0 0 m 100 90 l S Q";
    for depth in DEPTHS {
        for paths in [false, backend.page_paths()] {
            assert_eq!(
                device(Some(&backend), &doc, (95, 90), depth, content, paths),
                cpu(&doc, 95, 90, depth, content),
                "{} bits, outlines {paths}",
                depth.bits()
            );
        }
    }
}

#[test]
fn unsupported_paints_refuse_the_recording() {
    let doc = doc();
    let refused = |content: &[u8]| match record(&doc, 50, 50, content) {
        DeviceRecording::Refused(reason) => reason,
        DeviceRecording::Recorded(_) => "",
    };
    assert_eq!(refused(b"BT /F1 12 Tf 5 5 Td (a) Tj ET"), "");
    assert_eq!(refused(b"/Im Do"), "XObject");
    assert_eq!(refused(b"/Sh sh"), "shading");
    assert_eq!(refused(b"/M gs 0 0 10 10 re f"), "blend mode");
    assert_eq!(refused(b"/T gs 0 w 0 0 m 9 9 l S"), "translucent hairline");
    assert_eq!(
        refused(b"/T gs 0 0 10 10 re B"),
        "translucent fill and stroke"
    );
    assert_eq!(
        refused(b"0 0 m 9 0 l 5 9 l h W n 0 0 10 10 re f"),
        "path clip"
    );
    assert_eq!(refused(b"/Pattern cs /P scn 0 0 10 10 re f"), "pattern");
    assert_eq!(refused(b"0 0 m 9 0 l 5 9 l h W n"), "");
    assert_eq!(refused(b"q 2 2 6 6 re W n 0 0 10 10 re f Q"), "");
}

#[test]
fn operators_that_paint_otherwise_are_found_before_recording() {
    let ops = crate::content::parse(b"0 0 10 10 re f BT (a) Tj ET /Im Do");
    assert_eq!(
        device_page_ineligible(false, false, true, &ops),
        Some("XObject")
    );
    let ops = crate::content::parse(b"q 0 0 10 10 re f Q");
    assert_eq!(device_page_ineligible(false, false, true, &ops), None);
    assert_eq!(
        device_page_ineligible(true, true, false, &ops),
        Some("overprint simulation")
    );
    assert_eq!(
        device_page_ineligible(false, true, false, &ops),
        Some("page group")
    );
    assert_eq!(
        device_page_ineligible(false, false, false, &ops),
        Some("aliased vector paint")
    );
}

/// A page of full-page fills past the byte limit of a command list is
/// refused when the limit is reached, and renders on the CPU in bounded
/// memory; translucent fills, which hide nothing below them.  Opaque ones
/// hide every earlier item and stay within it.
#[test]
fn a_command_list_past_its_byte_limit_refuses_the_recording() {
    let doc = doc();
    let opaque = b"0 0 2000 2000 re f ".repeat(2_000);
    assert!(matches!(
        record(&doc, 2000, 2000, &opaque),
        DeviceRecording::Recorded(_)
    ));
    let mut fill = b"/T gs ".to_vec();
    fill.extend(b"0 0 2000 2000 re f ".repeat(2_000));
    assert!(matches!(
        record(&doc, 2000, 2000, &fill),
        DeviceRecording::Refused("command list size")
    ));
    let mut hairline = b"0 w 0 0 m ".to_vec();
    hairline.extend(b"2000 2000 l 0 0 l ".repeat(4_000));
    hairline.extend(b"S");
    assert!(matches!(
        record(&doc, 2000, 2000, &hairline),
        DeviceRecording::Refused("command list size")
    ));
}

/// A page refused late in its recording, after more than half of its
/// operator budget, renders on the CPU: the CPU pass has its own
/// operator budget, and keeps the recording's deadline and warned
/// operators.
#[test]
fn the_cpu_pass_after_a_late_refusal_has_its_own_operator_budget() {
    const CONTENT: &[u8] = b"zz 0 0 1 1 re f 0 0 1 1 re f 0 0 m 9 0 l 5 9 l h W n 0 0 10 10 re f";
    let doc = doc();
    let ops = crate::content::parse(CONTENT);
    let n_ops = u64::try_from(ops.len()).expect("few operators");
    let budget = 15;
    assert!(n_ops * 2 > budget && n_ops < budget);
    let mut first = PageRenderer::new(50, 50, &doc, (4, 0));
    first.op_budget = budget;
    first.record_for_device(true).expect("an RGB page records");
    first.execute(&ops);
    assert!(first.budget_status().is_none());
    assert!(matches!(
        first.take_device_recording(),
        Some(DeviceRecording::Refused("path clip"))
    ));
    assert_eq!(first.ops_executed, n_ops, "refused at the last operator");
    let fallback = first.take_device_fallback();
    let mut second = PageRenderer::new(50, 50, &doc, (4, 0));
    second.op_budget = budget;
    second.resume_after_device(fallback);
    assert_eq!(second.deadline, first.deadline);
    assert!(second.warned_unknown_ops.contains(&b"zz"[..]));
    second.execute(&ops);
    assert!(
        second.budget_status().is_none(),
        "the CPU pass ran on the recording's leftover budget: {:?}",
        second.budget_status()
    );
    assert_eq!(
        samples(&second),
        cpu(&doc, 50, 50, SampleDepth::Eight, CONTENT),
        "the CPU pass renders the page"
    );
}

#[test]
fn paper_on_an_unpainted_canvas_is_the_painted_canvas() {
    let doc = doc();
    for depth in DEPTHS {
        let mut r = unpainted(&doc, 23, 11, depth);
        assert!(samples(&r).iter().all(|&v| v == 0x5A || v == 0x5A5A));
        r.paint_paper();
        assert_eq!(samples(&r), samples(&renderer(&doc, 23, 11, depth)));
    }
}
