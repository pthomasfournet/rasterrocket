//! The paint-trace hooks of [`PageRenderer`]: arming and taking the trace,
//! and the lines of the operations its painting entry points record.

use std::fmt::Write as _;

use pdf::{Dictionary, Object};
use raster::fnv::Fnv1a;
use raster::path::Path;
use raster::types::BlendMode;

use super::PaintTrace;
use super::format::{
    chained_clip, colour_text, push_coord, push_key, push_list, push_name, push_path, push_unit,
};
use crate::RenderingIntent;
use crate::content::Operator;
use crate::renderer::page::PageRenderer;
use crate::renderer::page::text_glyphs::TextModeOps;
use crate::resources::{FormXObject, ResourceKey, SoftMaskParams};

/// The colour a graphics state starts with (ISO 32000-1 §8.6.4.1).
const INITIAL_COLOUR: &str = "DeviceGray(0)";

impl PageRenderer<'_> {
    /// Record the page's paint trace while it executes; the trace is taken
    /// with [`Self::take_paint_trace`] or, for a page drawn in windows,
    /// [`super::super::PageState::take_paint_trace`].  Call before
    /// [`Self::execute`].  Only the renderer of a page's first window
    /// records: [`Self::adopt_page_state`] stops a later window's trace.
    pub fn set_paint_trace(&mut self) {
        let m = &self.initial_ctm;
        let scale = m[0].mul_add(m[3], -(m[1] * m[2])).abs().sqrt();
        self.trace = Some(Box::new(PaintTrace::new(
            self.page_width,
            self.page_height,
            scale,
        )));
    }

    /// The paint trace recorded since [`Self::set_paint_trace`], if any.
    pub fn take_paint_trace(&mut self) -> Option<PaintTrace> {
        self.trace.take().map(|t| *t)
    }

    /// Whether the page's paint trace is being recorded.
    #[inline]
    pub(in crate::renderer::page) fn tracing(&self) -> bool {
        self.trace.as_deref().is_some_and(PaintTrace::records)
    }

    /// Run `f` with no operation recorded: content whose effect on the
    /// page is recorded as one operation, a Type 3 glyph's description.
    pub(in crate::renderer::page) fn without_trace<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.quiet += 1;
        }
        let out = f(self);
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.quiet -= 1;
        }
        out
    }

    /// Append the line `body` to the trace.
    #[inline]
    pub(in crate::renderer::page) fn trace_line(&mut self, body: &str) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(body);
        }
    }

    /// Append `body` to the trace and indent what follows, up to
    /// [`Self::trace_close`].
    pub(in crate::renderer::page) fn trace_open(&mut self, body: &str) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.open(body);
        }
    }

    /// End the indentation of the latest [`Self::trace_open`], then append
    /// `body`, if any.
    pub(in crate::renderer::page) fn trace_close(&mut self, body: Option<&str>) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.close(body);
        }
    }

    /// Append the paint attributes of the stroking (`stroke`) or
    /// non-stroking operation under the current graphics state: colour or
    /// pattern, constant alpha, blend mode, overprint, soft mask, transfer,
    /// rendering intent and clip.
    pub(in crate::renderer::page) fn push_paint(&self, out: &mut String, stroke: bool) {
        let gs = self.gstate.current();
        let (pattern, components, colour, alpha, overprint, nothing) = if stroke {
            (
                gs.stroke_pattern.as_deref(),
                &gs.stroke_pattern_components,
                gs.trace_stroke.as_deref(),
                gs.stroke_alpha,
                gs.overprint_stroke,
                gs.stroke_paints_nothing(),
            )
        } else {
            (
                gs.fill_pattern.as_deref(),
                &gs.fill_pattern_components,
                gs.trace_fill.as_deref(),
                gs.fill_alpha,
                gs.overprint_fill,
                gs.fill_paints_nothing(),
            )
        };
        out.push_str(" paint=");
        if let Some(name) = pattern {
            out.push_str("pattern");
            push_name(out, name);
            if let Some(key) = self.resources.pattern_key(name) {
                push_key(out, &key);
            }
            if !components.is_empty() {
                out.push('(');
                push_list(out, components, push_unit);
                out.push(')');
            }
        } else {
            out.push_str(colour.unwrap_or(INITIAL_COLOUR));
        }
        if nothing {
            out.push_str(" nothing");
        }
        if overprint {
            let _ = write!(out, " overprint opm={:?}", gs.overprint_mode);
        }
        self.push_compositing(out, if stroke { "CA" } else { "ca" }, alpha);
    }

    /// Append the compositing attributes of a paint at constant alpha
    /// `alpha`, printed as `key`, under the current graphics state: blend
    /// mode and rendering intent unless `Normal` and relative
    /// colorimetric, soft mask and transfer when present, and clip.
    pub(in crate::renderer::page) fn push_compositing(
        &self,
        out: &mut String,
        key: &str,
        alpha: u8,
    ) {
        let gs = self.gstate.current();
        let _ = write!(out, " {key}={alpha}");
        if gs.blend_mode != BlendMode::Normal {
            let _ = write!(out, " bm={:?}", gs.blend_mode);
        }
        if gs.trace_smask != 0 {
            let _ = write!(out, " smask={:016x}", gs.trace_smask);
        }
        if gs.transfer.is_some() {
            out.push_str(" transfer");
        }
        if gs.colour_intent.intent != RenderingIntent::RelativeColorimetric {
            let _ = write!(out, " ri={:?}", gs.colour_intent.intent);
        }
        if !gs.colour_intent.bpc {
            out.push_str(" bpc=off");
        }
        let _ = write!(out, " clip={:016x}", gs.trace_clip);
    }

    /// Append the device width of a stroke `line_width` user-space units
    /// wide under the current CTM, scaled by its area scale (a width under
    /// a singular CTM is in device pixels).
    fn push_device_width(&self, out: &mut String, line_width: f64) {
        out.push_str(" dlw=");
        let scale = super::super::paint::area_scale(&self.device_matrix()).unwrap_or(1.0);
        push_coord(out, line_width * scale);
    }

    /// Record the fill (`stroke == false`), stroke, or fill and stroke
    /// (`fill_rule` and `stroke`) of `path` by a path-painting operator.
    pub(in crate::renderer::page) fn trace_path_paint(
        &mut self,
        path: &Path,
        fill_rule: Option<bool>,
        stroke: bool,
    ) {
        if !self.tracing() {
            return;
        }
        let line = self.path_paint_line(path, fill_rule, stroke);
        self.trace_line(&line);
    }

    #[cold]
    #[inline(never)]
    fn path_paint_line(&self, path: &Path, fill_rule: Option<bool>, stroke: bool) -> String {
        let mut out = String::new();
        out.push_str(match (fill_rule, stroke) {
            (Some(_), true) => "fillstroke",
            (Some(_), false) => "fill",
            (None, _) => "stroke",
        });
        if let Some(even_odd) = fill_rule {
            out.push_str(if even_odd { " eo " } else { " nz " });
        } else {
            out.push(' ');
        }
        push_path(&mut out, path, self.window_origin);
        if fill_rule.is_some() {
            self.push_paint(&mut out, false);
        }
        if stroke {
            let gs = self.gstate.current();
            out.push_str(" lw=");
            push_coord(&mut out, gs.line_width);
            self.push_device_width(&mut out, self.min_stroke_width());
            let _ = write!(out, " cap={:?} join={:?} ml=", gs.line_cap, gs.line_join);
            push_coord(&mut out, gs.miter_limit);
            if !gs.dash.0.is_empty() {
                out.push_str(" dash=[");
                push_list(&mut out, &gs.dash.0, push_coord);
                out.push_str("] phase=");
                push_coord(&mut out, gs.dash.1);
            }
            out.push_str(" |");
            self.push_paint(&mut out, true);
        }
        out
    }

    /// Record the narrowing of the clip by `body`, and name the clip in
    /// the graphics state.
    pub(in crate::renderer::page) fn trace_clip(&mut self, body: &str) {
        let parent = self.gstate.current().trace_clip;
        let clip = chained_clip(parent, body);
        self.gstate.current_mut().trace_clip = clip;
        self.trace_line(&format!("clip {clip:016x}<{parent:016x} {body}"));
    }

    /// Record the clip by `path` under the even-odd rule when `even_odd`.
    pub(in crate::renderer::page) fn trace_clip_path(&mut self, path: &Path, even_odd: bool) {
        if !self.tracing() {
            return;
        }
        let mut body = String::from(if even_odd { "eo " } else { "nz " });
        push_path(&mut body, path, self.window_origin);
        self.trace_clip(&body);
    }

    /// The line of a string shown with the font `font` (`None` without
    /// one) in the mode of operations `mode`, its fill through outlines
    /// when `fill_as_path`, under the current graphics and text state;
    /// `None` when no trace is recorded.  Built before the string is
    /// shown, which advances the text matrix.
    pub(in crate::renderer::page) fn text_trace_line(
        &self,
        font: Option<&ResourceKey>,
        mode: TextModeOps,
        fill_as_path: bool,
    ) -> Option<String> {
        if !self.tracing() {
            return None;
        }
        let gs = self.gstate.current();
        let ts = &gs.text;
        let mut out = String::from("text ");
        push_name(&mut out, &ts.font_name);
        if let Some(key) = font {
            push_key(&mut out, key);
        }
        out.push_str(" size=");
        push_coord(&mut out, ts.font_size);
        let _ = write!(out, " Tr={}", ts.render_mode);
        for (key, v) in [
            ("Tc", ts.char_spacing),
            ("Tw", ts.word_spacing),
            ("Tz", ts.horiz_scaling),
            ("Ts", ts.rise),
        ] {
            let _ = write!(out, " {key}=");
            push_coord(&mut out, v);
        }
        out.push_str(" tm=[");
        push_list(&mut out, &ts.text_matrix, push_unit);
        out.push_str("] ctm=[");
        push_list(&mut out, &gs.ctm, push_unit);
        out.push(']');
        if mode.fill {
            if fill_as_path {
                out.push_str(" outlines");
            }
            self.push_paint(&mut out, false);
        }
        if mode.stroke {
            out.push_str(" | lw=");
            push_coord(&mut out, gs.line_width);
            self.push_device_width(&mut out, gs.line_width);
            self.push_paint(&mut out, true);
        }
        if mode.clip {
            out.push_str(" | clips");
        }
        Some(out)
    }

    /// Record the string of line `text` and the lines of its glyphs
    /// `glyphs`, one per line, nested under it.
    pub(in crate::renderer::page) fn trace_text(&mut self, text: &str, glyphs: &str) {
        let Some(trace) = self.trace.as_deref_mut() else {
            return;
        };
        trace.open(text);
        for line in glyphs.lines() {
            trace.line(line);
        }
        trace.close(None);
    }

    /// Record the drawing of the image `XObject` `name` under the current
    /// graphics state.
    pub(in crate::renderer::page) fn trace_image_xobject(&mut self, name: &[u8]) {
        if !self.tracing() {
            return;
        }
        let mut out = String::from("image ");
        push_name(&mut out, name);
        let dict = self.resources.xobject_id(name).and_then(|id| {
            let _ = write!(out, "({} {} R)", id.0, id.1);
            crate::resources::resolve_stream_dict(self.resources.doc(), &Object::Reference(id))
        });
        self.push_image(&mut out, dict.as_ref());
        self.trace_line(&out);
    }

    /// Record the drawing of an inline image of parameters `params` and
    /// data `data` under the current graphics state.
    pub(in crate::renderer::page) fn trace_inline_image(&mut self, params: &[u8], data: &[u8]) {
        if !self.tracing() {
            return;
        }
        let dict = crate::resources::image::inline::parse_inline_params(params);
        let mut out = format!(
            "image inline len={} data={:016x}",
            data.len(),
            Fnv1a::of(data)
        );
        self.push_image(&mut out, Some(&dict));
        self.trace_line(&out);
    }

    /// Append the attributes of an image of stream dictionary `dict`
    /// (`None` when unreadable) drawn under the current graphics state:
    /// its size, depth, colour space and masks, the page-space bounding
    /// box of the unit square it fills, and, for a stencil mask, the fill.
    fn push_image(&self, out: &mut String, dict: Option<&Dictionary>) {
        let int = |key: &[u8]| {
            dict.and_then(|d| d.get(key))
                .and_then(Object::as_i64)
                .map_or_else(|| "-".to_owned(), |v| v.to_string())
        };
        let flag = |key: &[u8]| {
            dict.and_then(|d| d.get(key))
                .and_then(Object::as_bool)
                .unwrap_or(false)
        };
        let stencil = flag(b"ImageMask");
        let _ = write!(
            out,
            " size={}x{} bpc={}",
            int(b"Width"),
            int(b"Height"),
            int(b"BitsPerComponent")
        );
        out.push_str(" cs=");
        match dict.and_then(|d| d.get(b"ColorSpace")) {
            Some(Object::Name(name)) => push_name(out, name),
            Some(Object::Reference((n, g))) => {
                let _ = write!(out, "({n} {g} R)");
            }
            Some(Object::Array(_)) => out.push_str("array"),
            _ => out.push('-'),
        }
        if stencil {
            out.push_str(" stencil");
        }
        match dict.and_then(|d| d.get(b"Mask")) {
            Some(Object::Array(_)) => out.push_str(" colourkey"),
            Some(_) => out.push_str(" mask"),
            None => {}
        }
        if dict.and_then(|d| d.get(b"SMask")).is_some() {
            out.push_str(" smaskimage");
        }
        if flag(b"Interpolate") {
            out.push_str(" interpolate");
        }
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut unit = raster::path::PathBuilder::new();
        for (i, &(x, y)) in corners.iter().enumerate() {
            let (dx, dy) = self.to_device(x, y);
            if i == 0 {
                unit.move_to(dx, dy);
            } else {
                let _ = unit.line_to(dx, dy);
            }
        }
        let _ = unit.close(true);
        out.push(' ');
        push_path(out, &unit.build(), self.window_origin);
        if stencil {
            self.push_paint(out, false);
        } else {
            self.push_compositing(out, "ca", self.gstate.current().fill_alpha);
        }
    }

    /// Record the painting of the shading `name` by `sh`.
    pub(in crate::renderer::page) fn trace_shading(&mut self, name: &[u8]) {
        if !self.tracing() {
            return;
        }
        let mut out = String::from("sh ");
        push_name(&mut out, name);
        if let Some(key) = self.resources.shading_key(name) {
            push_key(&mut out, &key);
        }
        out.push_str(" ctm=[");
        push_list(&mut out, &self.gstate.current().ctm, push_unit);
        out.push(']');
        self.push_compositing(&mut out, "ca", self.gstate.current().fill_alpha);
        self.trace_line(&out);
    }

    /// Record the start of the form `form`'s execution; its content is
    /// recorded nested, up to [`Self::trace_close`].
    pub(in crate::renderer::page) fn trace_form(&mut self, form: &FormXObject) {
        if !self.tracing() {
            return;
        }
        let (n, g) = form.resources_id;
        let mut out = format!("form ({n} {g} R) matrix=[");
        push_list(&mut out, &form.matrix, push_unit);
        out.push_str("] bbox=");
        match &form.bbox {
            Some(bbox) => push_list(&mut out, bbox, push_unit),
            None => out.push('-'),
        }
        if let Some(group) = form.transparency {
            let _ = write!(
                out,
                " group isolated={} knockout={}",
                u8::from(group.isolated),
                u8::from(group.knockout)
            );
        }
        self.push_compositing(&mut out, "ca", self.gstate.current().fill_alpha);
        self.trace_open(&out);
    }

    /// Record the soft mask `params` installs and name it in the graphics
    /// state; the mask group's content is recorded nested, by `render`.
    pub(in crate::renderer::page) fn trace_soft_mask<T>(
        &mut self,
        params: &SoftMaskParams,
        render: impl FnOnce(&mut Self) -> T,
    ) -> T {
        if !self.tracing() {
            return render(self);
        }
        let SoftMaskParams::Group {
            luminosity,
            group: (n, g),
            backdrop,
            transfer,
        } = params
        else {
            self.trace_line("smask none");
            let out = render(self);
            self.gstate.current_mut().trace_smask = 0;
            return out;
        };
        let mut body = format!(
            "({n} {g} R) {} bc=[",
            if *luminosity { "luminosity" } else { "alpha" }
        );
        push_list(&mut body, backdrop, push_unit);
        body.push(']');
        if transfer.is_some() {
            body.push_str(" transfer");
        }
        body.push_str(" ctm=[");
        push_list(&mut body, &self.gstate.current().ctm, push_unit);
        body.push(']');
        let name = Fnv1a::of(body.as_bytes());
        self.trace_open(&format!("smask {name:016x} {body}"));
        let out = render(self);
        self.trace_close(Some("smask end"));
        self.gstate.current_mut().trace_smask = name;
        out
    }

    /// Record that the optional content `name` (an `XObject`, or the
    /// properties of a marked-content sequence) is hidden.
    pub(in crate::renderer::page) fn trace_hidden(&mut self, name: &[u8]) {
        if !self.tracing() {
            return;
        }
        let mut out = String::from("hidden ");
        push_name(&mut out, name);
        self.trace_line(&out);
    }

    /// Record the clip by the quadrilateral of window-space `corners`, the
    /// device image of a user-space rectangle (a form's or a cell's
    /// `BBox`).
    pub(in crate::renderer::page) fn trace_clip_rect(&mut self, corners: &[(f64, f64); 4]) {
        if !self.tracing() {
            return;
        }
        let (ox, oy) = (
            f64::from(self.window_origin.0),
            f64::from(self.window_origin.1),
        );
        let mut body = String::from("rect");
        for (x, y) in corners {
            body.push(' ');
            push_coord(&mut body, x + ox);
            body.push(',');
            push_coord(&mut body, y + oy);
        }
        self.trace_clip(&body);
    }

    /// Record the colour `op`, a colour operator just executed, has set.
    pub(in crate::renderer::page) fn trace_colour(&mut self, op: &Operator<'_>) {
        if !self.tracing() {
            return;
        }
        self.record_colour(op);
    }

    #[cold]
    #[inline(never)]
    fn record_colour(&mut self, op: &Operator<'_>) {
        let gs = self.gstate.current();
        let (stroke, comps): (bool, Vec<f64>) = match op {
            Operator::SetFillGray(g) => (false, vec![*g]),
            Operator::SetFillRgb(r, g, b) => (false, vec![*r, *g, *b]),
            Operator::SetFillCmyk(c, m, y, k) => (false, vec![*c, *m, *y, *k]),
            Operator::SetFillColor(comps) => (false, comps.clone()),
            Operator::SetFillColorSpace(_) => (false, gs.fill_color_space.initial_components()),
            Operator::SetStrokeGray(g) => (true, vec![*g]),
            Operator::SetStrokeRgb(r, g, b) => (true, vec![*r, *g, *b]),
            Operator::SetStrokeCmyk(c, m, y, k) => (true, vec![*c, *m, *y, *k]),
            Operator::SetStrokeColor(comps) => (true, comps.clone()),
            Operator::SetStrokeColorSpace(_) => (true, gs.stroke_color_space.initial_components()),
            // A pattern is printed from the graphics state's pattern name.
            _ => return,
        };
        let space = if stroke {
            &gs.stroke_color_space
        } else {
            &gs.fill_color_space
        };
        let text = Some(colour_text(space, &comps));
        let gs = self.gstate.current_mut();
        if stroke {
            gs.trace_stroke = text;
        } else {
            gs.trace_fill = text;
        }
    }
}
