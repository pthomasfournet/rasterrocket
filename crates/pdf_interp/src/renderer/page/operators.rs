//! PDF operator dispatch — `execute_one`.

use std::sync::Arc;

use super::super::gstate::ctm_multiply;
use super::text_ops::{translate_text_x, translate_text_y};
use super::{MAX_WARNED_UNKNOWN_OPS, PageRenderer, int_to_cap, int_to_join};
use crate::content::{Operator, TextArrayElement};
use crate::resources::ExtGStateParams;
use crate::resources::colorspace::ColorSpace;
use crate::resources::transfer::TransferSetting;

impl PageRenderer<'_> {
    pub(super) fn execute_one(&mut self, op: &Operator) {
        // ── Marked content and optional content visibility ─────────────────────
        // Marked-content operators are tracked regardless of visibility; every
        // other operator is skipped inside a hidden optional-content sequence.
        match op {
            Operator::BeginOptionalContent(key) => {
                let visible = self
                    .resources
                    .property(key)
                    .is_none_or(|oc| self.oc_visible(&oc));
                self.ocg_stack.push(visible);
                return;
            }
            Operator::BeginMarkedContent => {
                self.ocg_stack.push(true);
                return;
            }
            Operator::EndMarkedContent => {
                // An unmatched `EMC` closes nothing.
                let _ = self.ocg_stack.pop();
                return;
            }
            _ => {}
        }
        if self.ocg_stack.iter().any(|&v| !v) {
            return;
        }
        // A `d1` glyph description is a stencil in the text fill colour.
        if op.sets_colour() && self.gstate.current().colour_locked {
            return;
        }

        self.dispatch(op);
    }

    #[expect(clippy::too_many_lines, reason = "operator dispatch table")]
    #[expect(
        clippy::match_same_arms,
        reason = "intentional stubs for unimplemented operators"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "PDF matrix components and PDF spec variable names"
    )]
    fn dispatch(&mut self, op: &Operator) {
        match op {
            // ── Graphics state ────────────────────────────────────────────────
            Operator::Save => self.gstate.save(),
            Operator::Restore => self.gstate.restore(),

            Operator::ConcatMatrix(m) => {
                let old = self.gstate.current().ctm;
                // PDF §8.3.4: `cm M` sets new_CTM = M × old_CTM.  The argument
                // matrix M maps the new user space to the old user space; the old
                // CTM then maps old user space to device space.
                self.gstate.current_mut().ctm = ctm_multiply(m, &old);
            }

            Operator::SetLineWidth(w) => self.gstate.current_mut().line_width = *w,
            Operator::SetLineCap(c) => self.gstate.current_mut().line_cap = int_to_cap(*c),
            Operator::SetLineJoin(j) => self.gstate.current_mut().line_join = int_to_join(*j),
            Operator::SetMiterLimit(m) => self.gstate.current_mut().miter_limit = *m,
            Operator::SetFlatness(f) => self.gstate.current_mut().flatness = *f,
            Operator::SetDash { dashes, phase } => {
                self.gstate.current_mut().dash = (dashes.clone(), *phase);
            }
            // Rendering intent is an output-intent hint only; no change to rendering.
            Operator::SetRenderingIntent(_) => {}

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
                let c = self.color_for_components(&cs, comps);
                self.set_fill(c);
            }
            Operator::SetFillColorSpace(name) => {
                // PDF §8.6.8: selecting a colour space sets the current
                // colour to the space's initial colour and clears any
                // pattern.  The declared space interprets later `sc`/`scn`
                // operands and an uncoloured pattern's tint (§8.7.3.3).
                let cs = self.resources.color_space(name);
                let c = self.color_for_components(&cs, &cs.initial_components());
                let gs = self.gstate.current_mut();
                gs.fill_color = c;
                gs.fill_pattern = None;
                gs.fill_pattern_components.clear();
                gs.fill_color_space = cs;
            }
            Operator::SetFillPattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.fill_pattern = Some(name.clone());
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
                let c = self.color_for_components(&cs, comps);
                self.set_stroke(c);
            }
            Operator::SetStrokeColorSpace(name) => {
                // Mirror of SetFillColorSpace; see comment above.
                let cs = self.resources.color_space(name);
                let c = self.color_for_components(&cs, &cs.initial_components());
                let gs = self.gstate.current_mut();
                gs.stroke_color = c;
                gs.stroke_pattern = None;
                gs.stroke_pattern_components.clear();
                gs.stroke_color_space = cs;
            }
            Operator::SetStrokePattern { name, components } => {
                let gs = self.gstate.current_mut();
                gs.stroke_pattern = Some(name.clone());
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
            // W/W* set a pending flag; the clip is applied by the *next*
            // path-terminating operator (PDF §8.5.4).  The path is NOT
            // consumed here — it may still be painted by e.g. "W f".
            Operator::Clip => self.pending_clip = Some(false),
            Operator::ClipEvenOdd => self.pending_clip = Some(true),

            // ── Text objects ──────────────────────────────────────────────────
            Operator::BeginText => self.gstate.current_mut().text.begin_text(),
            Operator::EndText => {}

            // ── Text state ────────────────────────────────────────────────────
            Operator::SetFont { name, size } => {
                if self.resources.font_entry(name).is_none() {
                    self.record_missing_font(name);
                }
                let font = self.font_cache.select(&self.resources, name);
                let ts = &mut self.gstate.current_mut().text;
                ts.font_name.clone_from(name);
                ts.font = Some(font);
                ts.font_size = *size;
            }
            Operator::SetCharSpacing(v) => self.gstate.current_mut().text.char_spacing = *v,
            Operator::SetWordSpacing(v) => self.gstate.current_mut().text.word_spacing = *v,
            Operator::SetHorizScaling(v) => self.gstate.current_mut().text.horiz_scaling = *v,
            Operator::SetLeading(v) => self.gstate.current_mut().text.leading = *v,
            Operator::SetTextRise(v) => self.gstate.current_mut().text.rise = *v,
            Operator::SetTextRenderMode(v) => self.gstate.current_mut().text.render_mode = *v,

            // ── Text positioning ──────────────────────────────────────────────
            Operator::TextMove(tx, ty) => {
                self.gstate.current_mut().text.move_by(*tx, *ty);
            }
            Operator::TextMoveSetLeading(tx, ty) => {
                self.gstate
                    .current_mut()
                    .text
                    .move_by_and_set_leading(*tx, *ty);
            }
            Operator::SetTextMatrix(m) => self.gstate.current_mut().text.set_matrix(*m),
            Operator::NextLine => self.gstate.current_mut().text.next_line(),

            // ── Text showing ──────────────────────────────────────────────────
            Operator::ShowText(bytes) => {
                self.show_text(bytes);
            }
            Operator::ShowTextArray(elems) => {
                for elem in elems {
                    match elem {
                        TextArrayElement::Text(bytes) => self.show_text(bytes),
                        TextArrayElement::Offset(kern) => {
                            // PDF §9.4.3: kern is in thousandths of a text-space unit.
                            // Displacement = -(w/1000) * Th * Tfs, where Th is horizontal
                            // scaling (Tz as a fraction, default 1.0) and Tfs is font size;
                            // in vertical writing mode -(w/1000) * Tfs along y, without Th.
                            // Tz=0 is degenerate (invisible text); treat as 1.0 like show_text.
                            let ts = &mut self.gstate.current_mut().text;
                            let vertical = ts
                                .font
                                .as_ref()
                                .and_then(|f| f.descriptor.cid_encoding.as_ref())
                                .is_some_and(|enc| enc.vertical);
                            if vertical {
                                let shift = -kern / 1000.0 * ts.font_size;
                                ts.text_matrix = translate_text_y(ts.text_matrix, shift);
                            } else {
                                let hz = if ts.horiz_scaling.abs() < f64::EPSILON {
                                    1.0
                                } else {
                                    ts.horiz_scaling / 100.0
                                };
                                let shift = -kern / 1000.0 * ts.font_size * hz;
                                ts.text_matrix = translate_text_x(ts.text_matrix, shift);
                            }
                        }
                    }
                }
            }
            Operator::MoveNextLineShow(bytes) => {
                self.gstate.current_mut().text.next_line();
                self.show_text(bytes);
            }
            Operator::MoveNextLineShowSpaced { aw, ac, text, .. } => {
                let gs = self.gstate.current_mut();
                gs.text.word_spacing = *aw;
                gs.text.char_spacing = *ac;
                gs.text.next_line();
                self.show_text(text);
            }

            // ── XObjects / images / shading ────────────────────────────────────
            Operator::PaintXObject(name) => {
                self.do_xobject(name);
            }
            Operator::InlineImage { params, data } => {
                // The same block in a form or pattern cell drawn repeatedly
                // decodes once per page.
                let key = super::image_memo::ImageMemo::inline_key(
                    self.resources.resource_context_id(),
                    params,
                    data,
                );
                let memoized = self.image_memo.borrow().get_inline(&key);
                if let Some(decoded) = memoized {
                    self.blit_image(&decoded);
                    return;
                }
                let color_spaces = self.resources.color_space_dict();
                let mut ctx = crate::resources::image::ImageDecodeCtx {
                    doc: self.resources.doc(),
                    color_spaces,
                    #[cfg(feature = "nvjpeg")]
                    nvjpeg: self.nvjpeg.as_mut(),
                    #[cfg(feature = "vaapi")]
                    vaapi: self.vaapi_jpeg_queue.as_ref(),
                    #[cfg(feature = "nvjpeg2k")]
                    nvjpeg2k: self.nvjpeg2k.as_mut(),
                    #[cfg(feature = "gpu-icc")]
                    icc: crate::resources::image::IccHandles {
                        gpu_ctx: self.gpu_ctx.as_deref(),
                    },
                    #[cfg(feature = "cache")]
                    image_cache: None,
                    #[cfg(feature = "cache")]
                    doc_id: None,
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
                match decoded {
                    crate::resources::ImageResolution::Ok(decoded) => {
                        if let Some(reason) = &decoded.partial {
                            self.record_partial_image(None, reason.clone());
                        }
                        let decoded = self.image_memo.borrow_mut().insert_inline(key, decoded);
                        self.blit_image(&decoded);
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
            // `d0`/`d1` declare the glyph advance width and (for `d1`) a bounding
            // box.  Width is sourced from the font's `Widths` array at call time;
            // the BBox is informational.  A `d1` description is a stencil: it
            // paints, fills and strokes alike, in the text fill colour in
            // effect at the text-showing operator (PDF §9.6.5), so the stroke
            // colour takes the fill colour and the colour operators that follow
            // are ignored.
            Operator::Type3GlyphWidth(..) => {}
            Operator::Type3GlyphWidthBBox(..) => {
                let gs = self.gstate.current_mut();
                gs.stroke_color = gs.fill_color.clone();
                gs.stroke_pattern = gs.fill_pattern.clone();
                gs.stroke_pattern_components = gs.fill_pattern_components.clone();
                gs.stroke_color_space = gs.fill_color_space.clone();
                gs.stroke_alpha = gs.fill_alpha;
                gs.colour_locked = true;
            }

            // ── No-ops ────────────────────────────────────────────────────────
            Operator::MarkedContentPoint | Operator::CompatibilitySection => {}

            // Handled in `execute_one` above; unreachable here.
            Operator::BeginOptionalContent(_)
            | Operator::BeginMarkedContent
            | Operator::EndMarkedContent => {}

            Operator::Unknown(kw) => self.warn_unknown_operator(kw),

            // Compile-time exhaustiveness guard.
            #[expect(
                unreachable_patterns,
                reason = "future Operator variants are caught here"
            )]
            _ => {}
        }
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
            gs.line_width = w;
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
        match &params.transfer {
            Some(TransferSetting::Identity) => gs.transfer = None,
            Some(TransferSetting::Functions(t)) => gs.transfer = Some(Arc::clone(t)),
            None => {}
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
