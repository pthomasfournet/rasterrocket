//! Recording a page for the whole-page GPU raster.
//!
//! While a page is recorded, a fill or stroke of a solid colour is not
//! painted.  It is appended to the page's [`PageCommands`] with the colour,
//! constant alpha and clip rectangle the CPU pipe would composite with, as
//! the outlines the CPU's exact-area rasteriser would rasterise
//! (`raster::fill::aa_fill_outlines`, `raster::aa_stroke_outline`), whose
//! coverage the device computes with the CPU's integer arithmetic; or, on a
//! device that does not rasterise outlines and for an outline past the
//! device's integer range, as the coverage tables the CPU computes from
//! them.  A hairline is appended as the pixel runs the CPU paints
//! (`raster::hairline_runs`).  A glyph composited from a coverage bitmap
//! is appended as a mask command reading that bitmap (see [`glyphs`]).
//! An image is appended as an image command sampling the samples the CPU
//! would sample, through the CPU's sampling tables (see [`images`]).  A
//! form `XObject` records the paints of its content; one that is a
//! transparency group refuses the recording, and a soft mask's group
//! paints its own canvas with the recording set aside.  A path clip is
//! recorded as the coverage tables of its regions (see [`clips`]), and the
//! paints of an object under a soft mask as one device object blended back
//! by the mask's plane (see [`soft_masks`]).
//! The device then composites every command
//! with the CPU's arithmetic, so the page it renders is the page the CPU
//! renders.
//!
//! A recorded paint composites as the CPU's `aa` pipe over an opaque RGB
//! canvas of 8 or 16 bits per sample (an image at full constant alpha:
//! as the CPU's direct image composite): `Normal` blend, no soft mask in
//! the graphics state but a soft mask, no group backdrop, a clip of
//! antialiased path regions, identity transfer (a solid
//! colour is mapped through the transfer function before painting; an
//! image's samples are not).  Its colour and
//! alpha are recorded as samples of the canvas's depth.  Anything else
//! refuses the recording with a reason, and the content stream stops
//! there: the caller renders the page on the CPU, whole.

use std::collections::HashSet;
use std::rc::Rc;

use color::{Sample, SampleDepth};
use gpu::backend::BackendError;
use gpu::backend::page_device::PageDevice;
use gpu::backend::vulkan::VulkanBackend;
use gpu::page::{
    AreaPath, CoverageRows, PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds,
};
use raster::clip::Clip;
use raster::coverage::{AreaOutline, CoverageTable};
use raster::types::BlendMode;
use raster::{aa_stroke_outline, hairline_runs};

use super::canvas::{CanvasKind, PageCanvas};
use super::image_memo::SharedImageMemo;
use super::paint::FillRule;
use super::{DEVICE_MATRIX, PageRenderer, WorkBudget};
use crate::content::Operator;
use clips::HeldClips;
use glyphs::HeldMasks;
pub(super) use glyphs::PlacedGlyph;
use soft_masks::HeldPlanes;

mod clips;
mod glyphs;
mod images;
mod soft_masks;

/// A page being recorded.
pub(super) struct Recording {
    commands: PageCommands,
    /// Whether outlines are recorded as outlines, else as tables.
    paths: bool,
    /// Why the recording stopped, once it has.
    refused: Option<&'static str>,
    /// Scratch: the runs of the hairline being appended.
    runs: Vec<(i32, i32, i32)>,
    /// The glyph bitmaps the page holds as masks.
    masks: HeldMasks,
    /// The path clips the page holds.
    clips: HeldClips,
    /// The soft masks the page holds as planes.
    planes: HeldPlanes,
}

/// What recording a page produced.
#[derive(Debug)]
pub enum DeviceRecording {
    /// Every paint of the page was recorded.
    Recorded(Box<PageCommands>),
    /// The page paints with something the GPU raster does not render.
    Refused(&'static str),
}

/// Why `op` cannot be recorded, if it cannot: the operators that paint
/// other than by filling or stroking a path, showing text or drawing an
/// `XObject` or inline image.
const fn unsupported_operator(op: &Operator<'_>) -> Option<&'static str> {
    match op {
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
/// or when an operator paints other than by filling or stroking a path,
/// showing text or drawing an `XObject` or inline image.  A page not ruled
/// out may still be refused while it is recorded, by the graphics state a
/// paint is made under, the way a glyph is drawn or the way an image is
/// sampled.
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
        PageCommandsError::Unrepresentable => "outline range",
    }
}

/// What a renderer whose device recording did not render the page hands
/// to the renderer that renders it on the CPU.
///
/// It holds the page's deadline and cancellation flag, which the two
/// passes share, the unsupported operators already warned about, and the
/// page's image decodes, which the second pass draws without decoding
/// again.
pub struct DeviceFallback {
    budget: WorkBudget,
    warned_unknown_ops: HashSet<Vec<u8>>,
    warned_unknown_ops_capped: bool,
    /// The images the first pass decoded, and the device colour
    /// conversions their decodes counted.
    image_memo: SharedImageMemo,
    device_icc: (u32, u32),
}

impl Recording {
    /// Append the coverage `tables` in `rgb` at alpha `alpha`, samples of
    /// the page's depth, in order.
    fn push_tables(
        &mut self,
        tables: &[CoverageTable],
        rgb: [u16; 3],
        alpha: u16,
    ) -> Result<(), PageCommandsError> {
        for table in tables {
            let Some((x0, y0, x1, y1)) = table.bounds() else {
                continue;
            };
            let rows = CoverageRows {
                y_min: table.y_min(),
                row_starts: table.row_starts(),
                spans: table.span_triples(),
                bytes: table.bytes(),
            };
            self.commands
                .push_coverage(rows, PixelBounds { x0, y0, x1, y1 }, rgb, alpha)?;
        }
        Ok(())
    }

    /// Append `outlines` in `rgb` at alpha `alpha`, samples of the page's
    /// depth, in order: as outlines when the recording takes them and the
    /// device's integer range holds them, else as their coverage tables.
    fn push_outlines(
        &mut self,
        outlines: &[AreaOutline],
        rgb: [u16; 3],
        alpha: u16,
    ) -> Result<(), PageCommandsError> {
        for outline in outlines {
            let (x0, y0, x1, y1) = outline.window;
            let path = AreaPath {
                segs: &outline.segs,
                window: PixelBounds { x0, y0, x1, y1 },
                even_odd: outline.even_odd,
                weight: outline.weight,
            };
            let pushed = if self.paths {
                self.commands.push_area_path(&path, rgb, alpha)
            } else {
                Err(PageCommandsError::Unrepresentable)
            };
            match pushed {
                Err(PageCommandsError::Unrepresentable) => {
                    self.push_tables(&[outline.table()], rgb, alpha)?;
                }
                other => other?,
            }
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

/// The paint parameters a recorded paint takes from the graphics state;
/// the colour and alpha are samples of the canvas's depth, as the CPU
/// pipe composites them.
struct SolidPaint {
    clip: Clip,
    rgb: [u16; 3],
    alpha: u16,
}

impl PageRenderer<'_> {
    /// Record the page for the whole-page GPU raster instead of painting
    /// it: fills and strokes of a solid colour become paint commands, and
    /// the first paint the raster does not render stops the recording.
    /// Outlines are recorded as outlines when `paths`, for a device that
    /// rasterises them, else as coverage tables.  Call before
    /// [`Self::execute`]; take the outcome with
    /// [`Self::take_device_recording`] after it and
    /// [`Self::render_annotations`].
    ///
    /// # Errors
    ///
    /// Why the page cannot be recorded at all: its paint trace, which only
    /// the CPU records, is armed, or it simulates overprint on an ink
    /// canvas, paints as a page group, or fills without antialiasing.  The
    /// renderer is unchanged then.
    pub fn record_for_device(&mut self, paths: bool) -> Result<(), &'static str> {
        if self.trace.is_some() {
            return Err("paint trace");
        }
        let kind = self.canvas.kind();
        if let Some(reason) = ineligible_canvas(
            !matches!(kind, CanvasKind::Rgb | CanvasKind::Rgb16),
            self.canvas.has_alpha() || self.page_group.is_some(),
            self.vector_antialias,
        ) {
            return Err(reason);
        }
        // An opaque RGB canvas starts as white paper.
        self.device_page = Some(Box::new(Recording {
            commands: PageCommands::new(
                self.width,
                self.height,
                kind.depth(),
                [kind.depth().max_sample(); 3],
            ),
            paths,
            refused: None,
            runs: Vec::new(),
            masks: HeldMasks::default(),
            clips: HeldClips::default(),
            planes: HeldPlanes::default(),
        }));
        Ok(())
    }

    /// The outcome of [`Self::record_for_device`]; `None` when the page was
    /// not recorded.  The renderer paints normally afterwards.
    pub fn take_device_recording(&mut self) -> Option<DeviceRecording> {
        let recording = self.device_page.take()?;
        Some(recording.refused.map_or_else(
            || DeviceRecording::Recorded(Box::new(recording.commands)),
            DeviceRecording::Refused,
        ))
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
            image_memo: Rc::clone(&self.image_memo),
            device_icc: (
                self.diag.device_icc_conversions,
                self.diag.device_icc_failures,
            ),
        }
    }

    /// Continue from `fallback`, taken from the renderer of this page whose
    /// device recording did not render it: both passes run under the one
    /// page deadline and cancellation flag, this pass with its own operator
    /// budget, so the page executes the operators it would execute on the
    /// CPU alone; an operator warned about in the first pass is not warned
    /// about again; an image the first pass decoded is drawn from its
    /// decode, whose device colour conversions this pass counts.  Call
    /// before [`Self::execute`].
    pub fn resume_after_device(&mut self, fallback: DeviceFallback) {
        self.adopt_budget(WorkBudget {
            ops: self.op_budget,
            ..fallback.budget
        });
        self.warned_unknown_ops = fallback.warned_unknown_ops;
        self.warned_unknown_ops_capped = fallback.warned_unknown_ops_capped;
        self.image_memo = fallback.image_memo;
        let (conversions, failures) = fallback.device_icc;
        self.diag.device_icc_conversions = conversions;
        self.diag.device_icc_failures = failures;
    }

    /// Render `page`, a recording of this page, on `backend` into the
    /// canvas.
    ///
    /// # Errors
    ///
    /// When the canvas is not an RGB canvas of the page's depth, or the
    /// device fails; the canvas is unchanged then.
    pub fn paint_device_page(
        &mut self,
        backend: &VulkanBackend,
        page: &PageCommands,
    ) -> Result<(), BackendError> {
        match &mut self.canvas {
            PageCanvas::Rgb(bitmap) => {
                let stride = bitmap.stride;
                backend.render_page(page, bitmap.data_mut(), stride)
            }
            PageCanvas::Rgb16(bitmap) => {
                let stride = bitmap.stride;
                backend.render_page(page, bitmap.data_mut(), stride)
            }
            PageCanvas::Ink(_) | PageCanvas::Ink16(_) => {
                Err(BackendError::msg("a device page paints an RGB canvas"))
            }
        }
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
        let alpha = if stroke {
            gs.stroke_alpha
        } else {
            gs.fill_alpha
        };
        self.solid_device_paint_at(stroke, alpha)
    }

    /// [`Self::solid_device_paint`] at the constant alpha `alpha`.
    fn solid_device_paint_at(&self, stroke: bool, alpha: u8) -> Result<SolidPaint, &'static str> {
        let gs = self.gstate.current();
        let (pattern, colour) = if stroke {
            (gs.stroke_pattern.is_some(), gs.painted_stroke_color())
        } else {
            (gs.fill_pattern.is_some(), gs.painted_fill_color())
        };
        if pattern {
            return Err("pattern");
        }
        if gs.blend_mode != BlendMode::Normal {
            return Err("blend mode");
        }
        let (rgb, alpha) = match self.canvas.kind().depth() {
            SampleDepth::Eight => (colour.bytes().map(u16::from), u16::from(alpha)),
            SampleDepth::Sixteen => (*colour.rgb16(), u16::from_u8(alpha)),
        };
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
        let outlines = raster::fill::aa_fill_outlines(
            path,
            &DEVICE_MATRIX,
            flatness,
            &paint.clip,
            self.width,
            self.height,
            rule.even_odd,
            rule.zero_area_lines,
        );
        if let Some(recording) = self.device_page.as_mut()
            && recording.clip_to(&paint.clip)
        {
            let pushed = recording.push_outlines(&outlines, paint.rgb, paint.alpha);
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
        if params.line_width == 0.0 && paint.alpha != self.canvas.kind().depth().max_sample() {
            return self.refuse_device("translucent hairline");
        }
        let (width, height) = (self.width, self.height);
        let Some(recording) = self.device_page.as_mut() else {
            return;
        };
        // Hairline runs pass the clip already, and are never clipped again.
        if !recording.clip_to(&paint.clip) {
            return;
        }
        if params.line_width != 0.0 {
            if let Some(outline) = aa_stroke_outline(
                &shape.path,
                &shape.matrix,
                &params,
                &paint.clip,
                width,
                height,
            ) {
                let pushed = recording.push_outlines(&[outline], paint.rgb, paint.alpha);
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
mod tests;
