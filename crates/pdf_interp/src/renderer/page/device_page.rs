//! Recording a page for the whole-page GPU raster.
//!
//! While a page is recorded, a fill or stroke of a solid colour is not
//! painted.  Its coverage is computed on the CPU by the functions the CPU
//! fill and stroke use (`raster::fill::aa_fill_coverage`,
//! `raster::aa_stroke_coverage`, `raster::hairline_runs`), and the result is
//! appended to the page's [`PageCommands`] with the colour, constant alpha
//! and clip rectangle the CPU pipe would composite with.  The device then
//! composites every command with the CPU's arithmetic, so the page it
//! renders is the page the CPU renders.
//!
//! A recorded paint composites as the CPU's `aa` pipe over an opaque RGB
//! canvas: `Normal` blend, no soft mask, no group backdrop, a rectangular
//! clip, identity transfer (a solid colour is mapped through the transfer
//! function before painting).  Anything else refuses the recording with a
//! reason, and the content stream stops there: the caller renders the page
//! on the CPU, whole.

use std::collections::HashSet;

use gpu::backend::BackendError;
use gpu::backend::vulkan::VulkanBackend;
use gpu::page::{
    CoverageRows, PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds,
};
use raster::clip::Clip;
use raster::coverage::CoverageTable;
use raster::types::BlendMode;
use raster::{aa_stroke_coverage, hairline_runs};

use super::canvas::{CanvasKind, PageCanvas};
use super::paint::FillRule;
use super::{DEVICE_MATRIX, PageRenderer, WorkBudget};
use crate::content::Operator;

/// The colour of the paper an opaque RGB canvas starts as.
const PAPER: [u8; 3] = [u8::MAX; 3];

/// A page being recorded.
pub(super) struct Recording {
    commands: PageCommands,
    /// Why the recording stopped, once it has.
    refused: Option<&'static str>,
    /// Scratch: the spans of the table being appended.
    spans: Vec<[i32; 3]>,
    /// Scratch: the runs of the hairline being appended.
    runs: Vec<(i32, i32, i32)>,
}

/// What recording a page produced.
#[derive(Debug)]
pub enum DeviceRecording {
    /// Every paint of the page was recorded.
    Recorded(PageCommands),
    /// The page paints with something the GPU raster does not render.
    Refused(&'static str),
}

/// Why `op` cannot be recorded, if it cannot: the operators that paint
/// other than by filling or stroking a path.
const fn unsupported_operator(op: &Operator<'_>) -> Option<&'static str> {
    match op {
        Operator::ShowText(_)
        | Operator::ShowTextArray(_)
        | Operator::MoveNextLineShow(_)
        | Operator::MoveNextLineShowSpaced { .. } => Some("text"),
        Operator::PaintXObject(_) => Some("XObject"),
        Operator::InlineImage { .. } => Some("inline image"),
        Operator::PaintShading(_) => Some("shading"),
        Operator::Type3GlyphWidth(..) | Operator::Type3GlyphWidthBBox(..) => Some("Type 3 glyph"),
        _ => None,
    }
}

/// Why a page painted on an ink canvas (`ink`), as a page group (`group`)
/// or with vector antialiasing off (`!vector_antialias`) cannot be
/// recorded, if it cannot.
const fn ineligible_canvas(ink: bool, group: bool, vector_antialias: bool) -> Option<&'static str> {
    if ink {
        Some("overprint simulation")
    } else if group {
        Some("page group")
    } else if !vector_antialias {
        Some("aliased vector paint")
    } else {
        None
    }
}

/// Why a page whose content is `ops` cannot be recorded, known before it
/// is interpreted; `None` when nothing rules it out.
///
/// It is ruled out when it paints on an ink canvas (`ink`), as a page
/// group (`group`) or with vector antialiasing off (`!vector_antialias`),
/// or when an operator paints other than by filling or stroking a path.
/// A page not ruled out may still be refused while it is recorded, by the
/// graphics state a paint is made under.
#[must_use]
pub fn device_page_ineligible(
    ink: bool,
    group: bool,
    vector_antialias: bool,
    ops: &[Operator<'_>],
) -> Option<&'static str> {
    ineligible_canvas(ink, group, vector_antialias)
        .or_else(|| ops.iter().find_map(unsupported_operator))
}

/// The refusal reason of a command the page cannot take.
const fn command_refusal(e: &PageCommandsError) -> &'static str {
    match e {
        PageCommandsError::TooLarge => "command list size",
        PageCommandsError::Alloc(_) => "command list allocation",
    }
}

/// What a renderer whose device recording did not render the page hands
/// to the renderer that renders it on the CPU.
///
/// It holds the page's deadline and cancellation flag, which the two
/// passes share, and the unsupported operators already warned about.
pub struct DeviceFallback {
    budget: WorkBudget,
    warned_unknown_ops: HashSet<Vec<u8>>,
    warned_unknown_ops_capped: bool,
}

impl Recording {
    /// Append the coverage `tables` in `rgb` at alpha `alpha`, in order.
    fn push_tables(
        &mut self,
        tables: &[CoverageTable],
        rgb: [u8; 3],
        alpha: u8,
    ) -> Result<(), PageCommandsError> {
        for table in tables {
            let Some((x0, y0, x1, y1)) = table.bounds() else {
                continue;
            };
            self.spans.clear();
            self.spans.extend(table.span_triples());
            let rows = CoverageRows {
                y_min: table.y_min(),
                row_starts: table.row_starts(),
                spans: &self.spans,
                bytes: table.bytes(),
            };
            self.commands
                .push_coverage(rows, PixelBounds { x0, y0, x1, y1 }, rgb, alpha)?;
        }
        Ok(())
    }

    /// Stop the recording with the refusal of `result`'s error, if any,
    /// unless it has already stopped.
    fn refuse_on(&mut self, result: Result<(), PageCommandsError>) {
        if let Err(e) = result {
            self.refuse(command_refusal(&e));
        }
    }

    /// Stop the recording with `reason`, unless it has already stopped.
    fn refuse(&mut self, reason: &'static str) {
        if self.refused.is_none() {
            log::debug!("rasterrocket-interp: page not recordable for the device: {reason}");
            self.refused = Some(reason);
        }
    }
}

/// The paint parameters a recorded paint takes from the graphics state.
struct SolidPaint {
    clip: Clip,
    rgb: [u8; 3],
    alpha: u8,
}

impl PageRenderer<'_> {
    /// Record the page for the whole-page GPU raster instead of painting
    /// it: fills and strokes of a solid colour become paint commands, and
    /// the first paint the raster does not render stops the recording.
    /// Call before [`Self::execute`]; take the outcome with
    /// [`Self::take_device_recording`] after it and
    /// [`Self::render_annotations`].
    ///
    /// # Errors
    ///
    /// Why the page cannot be recorded at all: it is painted at 16 bits,
    /// simulates overprint on an ink canvas, paints as a page group, or
    /// fills without antialiasing.  The renderer is unchanged then.
    pub fn record_for_device(&mut self) -> Result<(), &'static str> {
        if self.canvas.kind().depth() == color::SampleDepth::Sixteen {
            return Err("16-bit samples");
        }
        if let Some(reason) = ineligible_canvas(
            self.canvas.kind() != CanvasKind::Rgb,
            self.canvas.has_alpha() || self.page_group.is_some(),
            self.vector_antialias,
        ) {
            return Err(reason);
        }
        self.device_page = Some(Box::new(Recording {
            commands: PageCommands::new(self.width, self.height, PAPER),
            refused: None,
            spans: Vec::new(),
            runs: Vec::new(),
        }));
        Ok(())
    }

    /// The outcome of [`Self::record_for_device`]; `None` when the page was
    /// not recorded.  The renderer paints normally afterwards.
    pub fn take_device_recording(&mut self) -> Option<DeviceRecording> {
        let recording = self.device_page.take()?;
        Some(
            recording
                .refused
                .map_or(DeviceRecording::Recorded(recording.commands), |reason| {
                    DeviceRecording::Refused(reason)
                }),
        )
    }

    /// Hand the page's remaining work budget and warned operators to the
    /// renderer that renders the page again on the CPU
    /// ([`Self::resume_after_device`]).
    #[must_use]
    pub fn take_device_fallback(&mut self) -> DeviceFallback {
        DeviceFallback {
            budget: self.remaining_budget(),
            warned_unknown_ops: std::mem::take(&mut self.warned_unknown_ops),
            warned_unknown_ops_capped: self.warned_unknown_ops_capped,
        }
    }

    /// Continue from `fallback`, taken from the renderer of this page whose
    /// device recording did not render it: both passes run under the one
    /// page deadline and cancellation flag, this pass with its own operator
    /// budget, so the page executes the operators it would execute on the
    /// CPU alone; an operator warned about in the first pass is not warned
    /// about again.  Call before [`Self::execute`].
    pub fn resume_after_device(&mut self, fallback: DeviceFallback) {
        self.adopt_budget(WorkBudget {
            ops: self.op_budget,
            ..fallback.budget
        });
        self.warned_unknown_ops = fallback.warned_unknown_ops;
        self.warned_unknown_ops_capped = fallback.warned_unknown_ops_capped;
    }

    /// Render `page`, a recording of this page, on `backend` into the
    /// canvas.
    ///
    /// # Errors
    ///
    /// When the canvas is not an RGB canvas, or the device fails; the
    /// canvas is unchanged then.
    pub fn paint_device_page(
        &mut self,
        backend: &VulkanBackend,
        page: &PageCommands,
    ) -> Result<(), BackendError> {
        let PageCanvas::Rgb(bitmap) = &mut self.canvas else {
            return Err(BackendError::msg("a device page paints an RGB canvas"));
        };
        let stride = bitmap.stride;
        backend.render_page(page, bitmap.data_mut(), stride)
    }

    /// Whether the page is being recorded.
    pub(super) const fn device_recording(&self) -> bool {
        self.device_page.is_some()
    }

    /// Whether the page's recording has stopped: the content stream stops
    /// executing.
    pub(super) fn device_refused(&self) -> bool {
        self.device_page
            .as_ref()
            .is_some_and(|r| r.refused.is_some())
    }

    /// Stop the recording with `reason`, unless it has already stopped.
    pub(super) fn refuse_device(&mut self, reason: &'static str) {
        if let Some(recording) = self.device_page.as_mut() {
            recording.refuse(reason);
        }
    }

    /// Refuse the recording when `op` paints other than by filling or
    /// stroking a path; `true` when it does.
    pub(super) fn refuse_device_operator(&mut self, op: &Operator<'_>) -> bool {
        let Some(reason) = unsupported_operator(op) else {
            return false;
        };
        self.refuse_device(reason);
        true
    }

    /// The clip, colour and alpha of a solid paint by the stroking
    /// (`stroke`) or non-stroking operation, or the reason the paint
    /// cannot be recorded.
    fn solid_device_paint(&self, stroke: bool) -> Result<SolidPaint, &'static str> {
        let gs = self.gstate.current();
        let (pattern, colour, alpha) = if stroke {
            (
                gs.stroke_pattern.is_some(),
                gs.painted_stroke_color(),
                gs.stroke_alpha,
            )
        } else {
            (
                gs.fill_pattern.is_some(),
                gs.painted_fill_color(),
                gs.fill_alpha,
            )
        };
        if pattern {
            return Err("pattern");
        }
        if gs.blend_mode != BlendMode::Normal {
            return Err("blend mode");
        }
        if !gs.clip.is_rect() {
            return Err("path clip");
        }
        let rgb = colour
            .bytes_for(color::PixelMode::Rgb8, &self.ink.separator)
            .try_into()
            .map_err(|_| "colour")?;
        Ok(SolidPaint {
            clip: gs.clip.clone_shared(),
            rgb,
            alpha,
        })
    }

    /// Record the fill of `path` under `rule` that [`Self::fill_within`]
    /// would paint.
    pub(super) fn record_device_fill(&mut self, path: &raster::path::Path, rule: FillRule) {
        let paint = match self.solid_device_paint(false) {
            Ok(paint) => paint,
            Err(reason) => return self.refuse_device(reason),
        };
        let flatness = self.gstate.current().flatness.max(0.1);
        let tables = raster::fill::aa_fill_coverage(
            path,
            &DEVICE_MATRIX,
            flatness,
            &paint.clip,
            self.width,
            self.height,
            rule.even_odd,
            rule.zero_area_lines,
        );
        if let Some(recording) = self.device_page.as_mut() {
            let pushed = recording.push_tables(&tables, paint.rgb, paint.alpha);
            recording.refuse_on(pushed);
        }
    }

    /// Record the stroke of `path`, `line_width` wide, that
    /// [`Self::stroke_within`] would paint.  A hairline is recorded when it
    /// is opaque: its runs are composited one by one, and only an opaque
    /// paint gives the same pixels however many runs overlap.
    pub(super) fn record_device_stroke(&mut self, path: &raster::path::Path, line_width: f64) {
        if self.gstate.current().stroke_paints_nothing() {
            return;
        }
        let paint = match self.solid_device_paint(true) {
            Ok(paint) => paint,
            Err(reason) => return self.refuse_device(reason),
        };
        let shape = self.stroke_shape(path, line_width);
        let params = shape.params();
        if params.line_width == 0.0 && paint.alpha != u8::MAX {
            return self.refuse_device("translucent hairline");
        }
        let (width, height) = (self.width, self.height);
        let Some(recording) = self.device_page.as_mut() else {
            return;
        };
        if params.line_width != 0.0 {
            if let Some(table) = aa_stroke_coverage(
                &shape.path,
                &shape.matrix,
                &params,
                &paint.clip,
                width,
                height,
            ) {
                let pushed = recording.push_tables(&[table], paint.rgb, paint.alpha);
                recording.refuse_on(pushed);
            }
            return;
        }
        // The runs are held until the hairline is appended; they are kept
        // within the bytes the page may still take.
        let room = PAGE_COMMANDS_MAX_BYTES.saturating_sub(recording.commands.bytes())
            / size_of::<(i32, i32, i32)>();
        let mut runs = std::mem::take(&mut recording.runs);
        runs.clear();
        let mut overflow = false;
        hairline_runs(
            &shape.path,
            &shape.matrix,
            &params,
            &paint.clip,
            width,
            height,
            |x0, x1, y| {
                if runs.len() < room {
                    runs.push((y, x0, x1));
                } else {
                    overflow = true;
                }
            },
        );
        if overflow {
            recording.refuse(command_refusal(&PageCommandsError::TooLarge));
        } else {
            let pushed = recording.commands.push_opaque_runs(&mut runs, paint.rgb);
            recording.refuse_on(pushed);
        }
        recording.runs = runs;
    }
}

#[cfg(test)]
mod tests {
    use gpu::backend::vulkan::{DeviceSelector, VulkanBackend};
    use pdf::Document;

    use super::{DeviceRecording, PageRenderer, device_page_ineligible};

    /// A page whose resources name the graphics states the tests use:
    /// `/T` (fill and stroke alpha), `/O` (opaque), `/M` (`Multiply`).
    fn doc() -> Document {
        crate::test_helpers::make_doc_with_objects(&[(
            4,
            "<< /Type /Page /MediaBox [0 0 200 200] /Resources << /ExtGState << \
             /T << /ca 0.37 /CA 0.61 >> /O << /ca 1 /CA 1 >> /M << /BM /Multiply >> \
             >> >> >>",
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

    /// The canvas bytes of `content` on a `width × height` page.
    fn cpu(doc: &Document, width: u32, height: u32, content: &[u8]) -> Vec<u8> {
        let mut r = PageRenderer::new(width, height, doc, (4, 0));
        r.execute(&crate::content::parse(content));
        r.rgb_canvas().data().to_vec()
    }

    /// The recording of `content` on a `width × height` page.
    fn record(doc: &Document, width: u32, height: u32, content: &[u8]) -> DeviceRecording {
        let mut r = PageRenderer::new(width, height, doc, (4, 0));
        r.record_for_device().expect("an RGB page records");
        r.execute(&crate::content::parse(content));
        r.take_device_recording().expect("recorded")
    }

    /// The canvas bytes of `content` recorded and rendered on `backend`.
    fn device(
        backend: &VulkanBackend,
        doc: &Document,
        width: u32,
        height: u32,
        content: &[u8],
    ) -> Vec<u8> {
        let mut r = PageRenderer::new(width, height, doc, (4, 0));
        r.record_for_device().expect("an RGB page records");
        r.execute(&crate::content::parse(content));
        let DeviceRecording::Recorded(page) = r.take_device_recording().expect("recorded") else {
            panic!("the content was refused");
        };
        r.paint_device_page(backend, &page).expect("device page");
        r.rgb_canvas().data().to_vec()
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
                        let p: Vec<f64> =
                            (0..3).flat_map(|_| [rng.coord(w), rng.coord(h)]).collect();
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

    #[test]
    fn device_page_matches_the_cpu_render() {
        let Some(backend) = backend() else {
            return;
        };
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
                let want = cpu(&doc, width, height, &content);
                assert!(
                    width < 10 || want.iter().filter(|&&b| b != u8::MAX).count() > want.len() / 4,
                    "the page is mostly paper"
                );
                assert_eq!(
                    device(&backend, &doc, width, height, &content),
                    want,
                    "{width}×{height}, round {round}:\n{}",
                    String::from_utf8_lossy(&content)
                );
            }
        }
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
        assert_eq!(
            device(&backend, &doc, 95, 90, content),
            cpu(&doc, 95, 90, content)
        );
    }

    #[test]
    fn unsupported_paints_refuse_the_recording() {
        let doc = doc();
        let refused = |content: &[u8]| match record(&doc, 50, 50, content) {
            DeviceRecording::Refused(reason) => reason,
            DeviceRecording::Recorded(_) => "",
        };
        assert_eq!(refused(b"BT /F1 12 Tf (a) Tj ET"), "text");
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
        let ops = crate::content::parse(b"0 0 10 10 re f BT (a) Tj ET");
        assert_eq!(
            device_page_ineligible(false, false, true, &ops),
            Some("text")
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
    /// memory.
    #[test]
    fn a_command_list_past_its_byte_limit_refuses_the_recording() {
        let doc = doc();
        let fill = b"0 0 2000 2000 re f ".repeat(2_000);
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
        const CONTENT: &[u8] =
            b"zz 0 0 1 1 re f 0 0 1 1 re f 0 0 m 9 0 l 5 9 l h W n 0 0 10 10 re f";
        let doc = doc();
        let ops = crate::content::parse(CONTENT);
        let n_ops = u64::try_from(ops.len()).expect("few operators");
        let budget = 15;
        assert!(n_ops * 2 > budget && n_ops < budget);
        let mut first = PageRenderer::new(50, 50, &doc, (4, 0));
        first.op_budget = budget;
        first.record_for_device().expect("an RGB page records");
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
            second.rgb_canvas().data(),
            cpu(&doc, 50, 50, CONTENT),
            "the CPU pass renders the page"
        );
    }
}
