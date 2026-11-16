//! PDF operator dispatch — `execute_one`.

use std::sync::Arc;

use super::super::gstate::ctm_multiply;
use super::paint::{int_to_cap, int_to_join};
use super::text_context::{Following, keeps_text_context};
use super::text_ops::apply_kern;
use super::{MAX_WARNED_UNKNOWN_OPS, PageRenderer};
use crate::content::{Operator, TextArrayElement};
use crate::resources::ExtGStateParams;
use crate::resources::colorspace::ColorSpace;
use crate::resources::transfer::TransferSetting;

/// The open marked-content sequences of a content stream, innermost last.
///
/// `BMC` and `BDC` open one sequence each and `EMC` closes the innermost:
/// hidden for an optional-content sequence whose group is hidden, visible
/// otherwise.  Content is skipped while any open sequence is hidden; the
/// count of hidden ones answers that in constant time per operator.
#[derive(Debug, Default)]
pub(super) struct MarkedContent {
    /// Visibility of each open sequence, innermost last.
    open: Vec<bool>,
    /// The number of `false` entries of `open`.
    hidden: usize,
}

impl MarkedContent {
    /// Open a sequence, `visible` or hidden.
    fn open(&mut self, visible: bool) {
        self.open.push(visible);
        self.hidden += usize::from(!visible);
    }

    /// Close the innermost sequence; an unmatched `EMC` closes nothing.
    fn close(&mut self) {
        if self.open.pop() == Some(false) {
            self.hidden -= 1;
        }
    }

    /// Whether content is inside a hidden sequence.
    const fn hidden(&self) -> bool {
        self.hidden > 0
    }
}

impl PageRenderer<'_> {
    /// Execute `op`, followed in its content stream by `rest`.
    pub(super) fn execute_one(&mut self, op: &Operator, rest: &[Operator]) {
        if !keeps_text_context(op) {
            self.text_tail = None;
        }
        // ── Marked content and optional content visibility ─────────────────────
        // Marked-content operators are tracked regardless of visibility; every
        // other operator is skipped inside a hidden optional-content sequence.
        match op {
            Operator::BeginOptionalContent(key) => {
                let visible = self
                    .resources
                    .property(key)
                    .is_none_or(|oc| self.oc_visible(&oc));
                if !visible && !self.marked_content.hidden() {
                    self.trace_hidden(key);
                }
                self.marked_content.open(visible);
                return;
            }
            Operator::BeginMarkedContent => {
                self.marked_content.open(true);
                return;
            }
            Operator::EndMarkedContent => {
                self.marked_content.close();
                return;
            }
            _ => {}
        }
        if self.marked_content.hidden() {
            return;
        }
        #[cfg(feature = "vulkan")]
        if self.device_recording() && self.refuse_device_operator(op) {
            return;
        }
        // A `d1` glyph description is a stencil in the text fill colour.
        if op.sets_colour() && self.gstate.current().colour_locked {
            return;
        }
        if marks_in_gstate_intent(op) {
            self.record_intent(None);
        }

        self.dispatch(op, rest);
        if op.sets_colour() {
            self.trace_colour(op);
        }
    }

    /// Record that a marking operator paints under `own`, an image's own
    /// intent, else the graphics state's.  Recorded at dispatch, before
    /// any clipping to the window drawn, so every band of a page records
    /// the same set.
    pub(super) fn record_intent(&mut self, own: Option<crate::RenderingIntent>) {
        let intent = self.painting_target(own).intent.intent;
        self.diag.content_intents.insert(intent);
    }

    /// The target an object's colours convert to: the working space, under
    /// `own`, an image's own intent, else the graphics state's, with the
    /// graphics state's black-point compensation.
    pub(super) fn painting_target(
        &self,
        own: Option<crate::RenderingIntent>,
    ) -> crate::ColourTarget {
        let gs = self.gstate.current().colour_intent;
        crate::ColourTarget::new(
            crate::ColourIntent::new(own.unwrap_or(gs.intent), gs.bpc),
            self.space,
        )
    }

    /// The target the graphics state's colours convert to.
    pub(super) fn colour_target(&self) -> crate::ColourTarget {
        self.painting_target(None)
    }

    #[expect(clippy::too_many_lines, reason = "operator dispatch table")]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF matrix components and PDF spec variable names"
    )]
    fn dispatch(&mut self, op: &Operator, rest: &[Operator]) {
        match op {
            // ── Graphics state ────────────────────────────────────────────────
            Operator::Save => self.gstate.save_op(),
            Operator::Restore => self.gstate.restore_op(),

            Operator::ConcatMatrix(m) => {
                let old = self.gstate.current().ctm;
                // PDF §8.3.4: `cm M` sets new_CTM = M × old_CTM.  The argument
                // matrix M maps the new user space to the old user space; the old
                // CTM then maps old user space to device space.
                self.gstate.current_mut().ctm = ctm_multiply(m, &old);
            }

            Operator::SetLineWidth(w) => self.gstate.current_mut().set_line_width(*w),
            Operator::SetLineCap(c) => self.gstate.current_mut().line_cap = int_to_cap(*c),
            Operator::SetLineJoin(j) => self.gstate.current_mut().line_join = int_to_join(*j),
            Operator::SetMiterLimit(m) => self.gstate.current_mut().miter_limit = *m,
            Operator::SetFlatness(f) => self.gstate.current_mut().flatness = *f,
            Operator::SetDash { dashes, phase } => {
                self.gstate.current_mut().dash = (dashes.clone(), *phase);
            }
            Operator::SetExtGState(name) => {
                if let Some(params) = self.resources.ext_gstate(name) {
                    self.apply_ext_gstate(&params);
                }
            }

            // ── Colour ────────────────────────────────────────────────────────
            Operator::SetFillGray(g) => {
                self.set_fill_device(ColorSpace::DeviceGray, &[*g]);
            }
            Operator::SetFillRgb(r, g, b) => {
                self.set_fill_device(ColorSpace::DeviceRgb, &[*r, *g, *b]);
            }
            Operator::SetFillCmyk(c, m, y, k) => {
                self.set_fill_device(ColorSpace::DeviceCmyk, &[*c, *m, *y, *k]);
            }
            Operator::SetFillColor(comps) => {
                let cs = self.gstate.current().fill_color_space.clone();
                self.set_fill_colour(cs, comps);
            }
            Operator::SetFillColorSpace(name) => {
                // PDF §8.6.8: selecting a colour space sets the current
                // colour to the space's initial colour and clears any
                // pattern.  The declared space interprets later `sc`/`scn`
                // operands and an uncoloured pattern's tint (§8.7.3.3).
                let cs = self.resources.color_space(name);
                let initial = cs.initial_components();
                self.set_fill_colour(cs, &initial);
            }
            Operator::SetFillPattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.fill_pattern = Some(Arc::from(&**name));
                gs.fill_pattern_components.clone_from(components);
            }

            Operator::SetStrokeGray(g) => {
                self.set_stroke_device(ColorSpace::DeviceGray, &[*g]);
            }
            Operator::SetStrokeRgb(r, g, b) => {
                self.set_stroke_device(ColorSpace::DeviceRgb, &[*r, *g, *b]);
            }
            Operator::SetStrokeCmyk(c, m, y, k) => {
                self.set_stroke_device(ColorSpace::DeviceCmyk, &[*c, *m, *y, *k]);
            }
            Operator::SetStrokeColor(comps) => {
                let cs = self.gstate.current().stroke_color_space.clone();
                self.set_stroke_colour(cs, comps);
            }
            Operator::SetStrokeColorSpace(name) => {
                // Mirror of SetFillColorSpace; see comment above.
                let cs = self.resources.color_space(name);
                let initial = cs.initial_components();
                self.set_stroke_colour(cs, &initial);
            }
            Operator::SetStrokePattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.stroke_pattern = Some(Arc::from(&**name));
                gs.stroke_pattern_components.clone_from(components);
            }

            // ── Path construction ─────────────────────────────────────────────
            Operator::MoveTo(x, y) => {
                let (dx, dy) = self.to_device(*x, *y);
                if let Err(e) = self.path_builder().move_to(dx, dy) {
                    log::debug!("rasterrocket-interp: MoveTo failed: {e}");
                }
            }
            Operator::LineTo(x, y) => {
                let (dx, dy) = self.to_device(*x, *y);
                if let Err(e) = self.path_builder().line_to(dx, dy) {
                    log::debug!("rasterrocket-interp: LineTo failed: {e}");
                }
            }
            Operator::CurveTo(x1, y1, x2, y2, x3, y3) => {
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(dx1, dy1, dx2, dy2, dx3, dy3) {
                    log::debug!("rasterrocket-interp: CurveTo failed: {e}");
                }
            }
            Operator::CurveToV(x2, y2, x3, y3) => {
                // `v`: first control point = current point.
                let Some(cp) = self.path_builder().cur_pt() else {
                    log::debug!(
                        "rasterrocket-interp: CurveToV with no current point — operator ignored"
                    );
                    return;
                };
                let (dx2, dy2) = self.to_device(*x2, *y2);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(cp.x, cp.y, dx2, dy2, dx3, dy3) {
                    log::debug!("rasterrocket-interp: CurveToV failed: {e}");
                }
            }
            Operator::CurveToY(x1, y1, x3, y3) => {
                // `y`: second control point = endpoint.
                let (dx1, dy1) = self.to_device(*x1, *y1);
                let (dx3, dy3) = self.to_device(*x3, *y3);
                if let Err(e) = self.path_builder().curve_to(dx1, dy1, dx3, dy3, dx3, dy3) {
                    log::debug!("rasterrocket-interp: CurveToY failed: {e}");
                }
            }
            Operator::ClosePath => {
                if let Some(b) = self.path.as_mut()
                    && let Err(e) = b.close(false)
                {
                    log::debug!("rasterrocket-interp: ClosePath failed: {e}");
                }
            }
            Operator::Rectangle(x, y, w, h) => {
                // PDF §8.5.2.1: `re` defines four corners in user space.
                // All four must be independently transformed — mixing device-space
                // coordinates from two corners produces the wrong shape under a
                // sheared CTM.
                let (x0, y0) = self.to_device(*x, *y);
                let (x1, y1) = self.to_device(*x + *w, *y);
                let (x2, y2) = self.to_device(*x + *w, *y + *h);
                let (x3, y3) = self.to_device(*x, *y + *h);
                let b = self.path_builder();
                // `re` is self-contained; individual segment failures are benign —
                // a partial rectangle is better than a panic on a malformed PDF.
                let _ = b.move_to(x0, y0);
                let _ = b.line_to(x1, y1);
                let _ = b.line_to(x2, y2);
                let _ = b.line_to(x3, y3);
                let _ = b.close(true);
            }

            // ── Path painting ─────────────────────────────────────────────────
            Operator::Fill => self.do_fill(false),
            Operator::FillEvenOdd => self.do_fill(true),
            Operator::Stroke => self.do_stroke(),
            Operator::CloseStroke => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_stroke();
            }
            Operator::CloseFillStroke => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_fill_then_stroke(false);
            }
            Operator::CloseFillStrokeEvenOdd => {
                if let Some(b) = self.path.as_mut() {
                    let _ = b.close(false);
                }
                self.do_fill_then_stroke(true);
            }
            Operator::FillStroke => self.do_fill_then_stroke(false),
            Operator::FillStrokeEvenOdd => self.do_fill_then_stroke(true),
            Operator::EndPath => {
                self.apply_pending_clip();
                self.path = None;
            }

            // ── Clipping ──────────────────────────────────────────────────────
            // W/W* set a pending flag; the path-terminating operator that
            // follows paints the path, then intersects it into the clip
            // (ISO 32000-1 §8.5.4).
            Operator::Clip => self.pending_clip = Some(false),
            Operator::ClipEvenOdd => self.pending_clip = Some(true),

            // ── Text objects ──────────────────────────────────────────────────
            Operator::BeginText => self.gstate.current_mut().text.begin_text(),
            Operator::EndText => self.end_text(),

            // ── Text state and positioning ────────────────────────────────────
            Operator::SetFont { .. }
            | Operator::SetCharSpacing(_)
            | Operator::SetWordSpacing(_)
            | Operator::SetHorizScaling(_)
            | Operator::SetLeading(_)
            | Operator::SetTextRise(_)
            | Operator::SetTextRenderMode(_)
            | Operator::TextMove(..)
            | Operator::TextMoveSetLeading(..)
            | Operator::SetTextMatrix(_)
            | Operator::NextLine => self.set_text_state(op),

            // ── Text showing ──────────────────────────────────────────────────
            Operator::ShowText(_)
            | Operator::MoveNextLineShow(_)
            | Operator::MoveNextLineShowSpaced { .. } => {
                if let Some(bytes) = self.gstate.current_mut().text.begin_show(op) {
                    self.show_text(bytes, Following::ops(rest));
                }
            }
            Operator::ShowTextArray(elems) => {
                for (i, elem) in elems.iter().enumerate() {
                    match elem {
                        TextArrayElement::Text(bytes) => self.show_text(
                            bytes,
                            Following {
                                elements: &elems[i + 1..],
                                ops: rest,
                            },
                        ),
                        TextArrayElement::Offset(kern) => {
                            apply_kern(&mut self.gstate.current_mut().text, *kern);
                        }
                    }
                }
            }

            // ── XObjects / images / shading ────────────────────────────────────
            Operator::PaintXObject(name) => {
                self.do_xobject(name);
            }
            Operator::InlineImage { params, data } => {
                self.record_intent(inline_intent(params));
                self.trace_inline_image(params, data);
                // The same block in a form or pattern cell drawn repeatedly
                // decodes once per page.
                let target = self.painting_target(inline_intent(params));
                let key = super::image_memo::ImageMemo::inline_key(
                    self.canvas.kind().depth(),
                    target,
                    self.resources.color_space_source(),
                    params,
                    data,
                );
                let memoized = self
                    .image_memo
                    .borrow()
                    .get_inline(&key, self.keeps_tints());
                if let Some(decoded) = memoized {
                    self.blit_image(&decoded, None);
                    return;
                }
                let color_spaces = self.resources.color_space_dict();
                #[cfg(feature = "gpu-icc")]
                let icc_counts = crate::resources::image::DeviceIccCounts::default();
                let keep_tints = self.keeps_tints();
                let mut ctx = crate::resources::image::ImageDecodeCtx {
                    doc: self.resources.doc(),
                    color_spaces,
                    tint_budget: Some(&self.tint_budget),
                    #[cfg(feature = "nvjpeg")]
                    nvjpeg: self.nvjpeg.as_mut(),
                    #[cfg(feature = "vaapi")]
                    vaapi: self.vaapi_jpeg_queue.as_ref(),
                    #[cfg(feature = "nvjpeg2k")]
                    nvjpeg2k: crate::resources::image::DeviceJpx::new(&mut self.nvjpeg2k, None),
                    #[cfg(feature = "gpu-icc")]
                    icc: crate::resources::image::IccHandles {
                        gpu_ctx: self.gpu_ctx.as_ref(),
                        #[cfg(feature = "vulkan")]
                        vk_backend: self.vk_backend.as_deref(),
                        counts: Some(&icc_counts),
                    },
                    #[cfg(feature = "cache")]
                    image_cache: None,
                    #[cfg(feature = "cache")]
                    doc_id: None,
                    keep_tints,
                    target,
                    depth: self.canvas.kind().depth(),
                };
                // Route to whichever GPU JPEG decoder is active (only one is Some per render).
                #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
                let decoded = if self.jpeg_vk.is_some() {
                    crate::resources::image::decode_inline_image(
                        &mut ctx,
                        params,
                        data,
                        self.jpeg_vk.as_mut(),
                    )
                } else {
                    crate::resources::image::decode_inline_image(
                        &mut ctx,
                        params,
                        data,
                        self.jpeg_gpu.as_mut(),
                    )
                };
                #[cfg(not(all(feature = "gpu-jpeg-huffman", feature = "vulkan")))]
                let decoded = crate::resources::image::decode_inline_image(
                    &mut ctx,
                    params,
                    data,
                    #[cfg(feature = "gpu-jpeg-huffman")]
                    self.jpeg_gpu.as_mut(),
                );
                #[cfg(feature = "gpu-icc")]
                self.diag.add_device_icc(&icc_counts);
                match decoded {
                    crate::resources::ImageResolution::Ok(decoded) => {
                        if let Some(reason) = &decoded.partial {
                            self.record_partial_image(None, reason.clone());
                        }
                        let decoded = self.image_memo.borrow_mut().insert_inline(key, decoded);
                        self.blit_image(&decoded, None);
                    }
                    crate::resources::ImageResolution::DecodeFailed(msg) => {
                        self.record_image_failure(None, msg);
                    }
                    crate::resources::ImageResolution::Absent => {}
                }
            }
            Operator::PaintShading(name) => {
                self.do_shading(name);
            }

            // ── Type 3 glyph metrics ──────────────────────────────────────────
            // A `d1` description is a stencil: it paints, fills and strokes
            // alike, in the text fill colour in effect at the text-showing
            // operator (PDF §9.6.5), so the stroke colour takes the fill
            // colour and the colour operators that follow are ignored.
            Operator::Type3GlyphWidthBBox(..) => {
                let gs = self.gstate.current_mut();
                gs.stroke_color.clone_from(&gs.fill_color);
                gs.stroke_pattern.clone_from(&gs.fill_pattern);
                gs.stroke_pattern_components
                    .clone_from(&gs.fill_pattern_components);
                gs.stroke_color_space.clone_from(&gs.fill_color_space);
                gs.stroke_components.clone_from(&gs.fill_components);
                gs.stroke_alpha = gs.fill_alpha;
                gs.colour_locked = true;
            }

            Operator::Unknown(kw) => self.warn_unknown_operator(kw),

            Operator::SetRenderingIntent(name) => {
                let bpc = self.gstate.current().colour_intent.bpc;
                self.set_colour_intent(crate::ColourIntent::new(
                    crate::RenderingIntent::from_pdf_name(name),
                    bpc,
                ));
            }

            // Operators without effect on the pixels.  `d0` declares an advance, taken from the font's
            // `Widths`; `MP`/`DP` mark a point and `BX`/`EX` a compatibility
            // section.
            // The marked-content sequence operators are handled by
            // `execute_one` before dispatch.
            Operator::Type3GlyphWidth(..)
            | Operator::MarkedContentPoint
            | Operator::CompatibilitySection
            | Operator::BeginOptionalContent(_)
            | Operator::BeginMarkedContent
            | Operator::EndMarkedContent => {}
        }
    }

    /// Apply the text-state or text-positioning operator `op` to the
    /// current text state.  A `Tf` naming no font of the resources is
    /// recorded as missing.
    fn set_text_state(&mut self, op: &Operator) {
        if let Operator::SetFont { name, .. } = op
            && self.resources.font_entry(name).is_none()
        {
            self.record_missing_font(name);
        }
        let (font_cache, resources) = (&mut self.font_cache, &self.resources);
        let applied = self
            .gstate
            .current_mut()
            .text
            .apply_state_op(op, |name| font_cache.select(resources, name));
        debug_assert!(applied, "{op:?} is a text-state operator");
    }

    /// Put the parameters of an `ExtGState` dictionary into effect in the
    /// current graphics state.
    pub(super) fn apply_ext_gstate(&mut self, params: &ExtGStateParams) {
        let gs = self.gstate.current_mut();
        if let Some(a) = params.fill_alpha {
            gs.fill_alpha = a;
        }
        if let Some(a) = params.stroke_alpha {
            gs.stroke_alpha = a;
        }
        if let Some(w) = params.line_width {
            gs.set_line_width(w);
        }
        if let Some(c) = params.line_cap {
            gs.line_cap = int_to_cap(c);
        }
        if let Some(j) = params.line_join {
            gs.line_join = int_to_join(j);
        }
        if let Some(m) = params.miter_limit {
            gs.miter_limit = m;
        }
        if let Some(f) = params.flatness {
            gs.flatness = f;
        }
        if let Some(bm) = params.blend_mode {
            gs.blend_mode = bm;
        }
        if let Some(o) = params.overprint_stroke {
            gs.overprint_stroke = o;
        }
        if let Some(o) = params.overprint_fill {
            gs.overprint_fill = o;
        }
        if let Some(m) = params.overprint_mode {
            gs.overprint_mode = m;
        }
        match &params.transfer {
            Some(TransferSetting::Identity) => gs.transfer = None,
            Some(TransferSetting::Functions(t)) => gs.transfer = Some(Arc::clone(t)),
            None => {}
        }
        self.set_device_black(
            params.black_generation.as_ref(),
            params.undercolour_removal.as_ref(),
        );
        if params.rendering_intent.is_some() || params.black_point_compensation.is_some() {
            let current = self.gstate.current().colour_intent;
            self.set_colour_intent(crate::ColourIntent::new(
                params.rendering_intent.unwrap_or(current.intent),
                params.black_point_compensation.unwrap_or(current.bpc),
            ));
        }
        if let Some(sm) = &params.soft_mask {
            self.set_soft_mask(sm);
        }
    }

    /// Warn (once per distinct keyword per page) that `kw` is an unsupported
    /// content operator, so a repeating junk operator doesn't flood the log
    /// and bury real image-decode error lines.
    ///
    /// The first occurrence of every distinct keyword still warns — the page
    /// may be rendering incompletely and that signal must surface; only exact
    /// repeats are suppressed.
    ///
    /// The distinct-keyword set is hard-capped at [`MAX_WARNED_UNKNOWN_OPS`].
    /// An adversarial content stream can emit up to `MAX_PAGE_OPS` operators,
    /// each a distinct (and arbitrarily long) junk keyword; without the cap the
    /// set would grow to multiple gigabytes of attacker-controlled bytes before
    /// the operator-budget watchdog aborts the page, defeating the page memory
    /// budget. On reaching the cap we emit one final summary warning and stop
    /// tracking new keywords: the "page has unsupported content" signal is
    /// preserved while memory stays bounded.
    fn warn_unknown_operator(&mut self, kw: &[u8]) {
        if self.warned_unknown_ops.len() >= MAX_WARNED_UNKNOWN_OPS {
            // `contains` (no allocation) for already-tracked keywords; only the
            // first keyword that overflows the cap takes the summary branch.
            if !self.warned_unknown_ops.contains(kw) && !self.warned_unknown_ops_capped {
                self.warned_unknown_ops_capped = true;
                log::warn!(
                    "rasterrocket-interp: more than {MAX_WARNED_UNKNOWN_OPS} distinct \
                     unsupported content operators on this page; further distinct \
                     operators are no longer individually reported (page content \
                     may be incomplete)"
                );
            }
            return;
        }
        if self.warned_unknown_ops.insert(kw.to_vec()) {
            log::warn!(
                "rasterrocket-interp: unsupported content operator {:?} \
                 (further occurrences suppressed for this page)",
                String::from_utf8_lossy(kw)
            );
        }
    }
}

/// Whether `op` paints under the graphics state's rendering intent: the
/// path-painting, text-showing and shading operators.  Images, which may
/// carry an intent of their own, record theirs where they are drawn.
const fn marks_in_gstate_intent(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Fill
            | Operator::FillEvenOdd
            | Operator::Stroke
            | Operator::CloseStroke
            | Operator::CloseFillStroke
            | Operator::CloseFillStrokeEvenOdd
            | Operator::FillStroke
            | Operator::FillStrokeEvenOdd
            | Operator::ShowText(_)
            | Operator::ShowTextArray(_)
            | Operator::MoveNextLineShow(_)
            | Operator::MoveNextLineShowSpaced { .. }
            | Operator::PaintShading(_)
    )
}

/// The `Intent` of the inline image whose parameters are `params`; `None`
/// when absent.
fn inline_intent(params: &[u8]) -> Option<crate::RenderingIntent> {
    if !params.windows(6).any(|w| w == b"Intent") {
        return None;
    }
    let dict = crate::resources::image::inline::parse_inline_params(params);
    let name = dict.get(b"Intent")?.as_name()?;
    Some(crate::RenderingIntent::from_pdf_name(name))
}

#[cfg(test)]
mod tests {
    use super::MarkedContent;

    #[test]
    fn marked_content_is_hidden_while_a_hidden_sequence_is_open() {
        let mut mc = MarkedContent::default();
        mc.open(true);
        mc.open(false);
        mc.open(true);
        assert!(mc.hidden());
        mc.close();
        assert!(mc.hidden());
        mc.close();
        assert!(!mc.hidden());
        mc.close();
        mc.close();
        assert!(!mc.hidden(), "an unmatched EMC closes nothing");
        mc.open(false);
        assert!(mc.hidden());
    }
}
