//! The paint trace of a page: one line of text per painting operation the
//! content performs, in the whole page's device space.
//!
//! A trace records operations as the content states them, under the
//! graphics state in effect: paths, fill rules, stroke parameters, clips,
//! source colours, images, shadings, glyphs, groups and soft masks.  It
//! records nothing that depends on the window a renderer draws, on a memo
//! or cache hit, or on the canvas: every window of a page executes the
//! whole content, and the trace of its first window is the page's.
//!
//! # Numbers
//!
//! Device coordinates, line widths and font sizes are rounded to the
//! nearest 1/256 pixel, the subpixel grid of the coverage rasteriser, and
//! printed as the exact decimal of the rounded value; colour components
//! and matrix entries are rounded to six decimal places.  Rounding to nearest splits at odd
//! multiples of half a quantum, which the integer and half-integer
//! coordinates of real content do not meet, so a last-bit difference in
//! the arithmetic that produced a value leaves the printed value
//! unchanged.
//!
//! # Paths
//!
//! A path prints its point, subpath and curve-segment counts, the bounding
//! box of its points (control points included) and the FNV-1a hash of its
//! rounded points and flags; a path of at most [`INLINE_POINTS`] points
//! also prints its points.
//!
//! # Clips and colours
//!
//! Each clip prints a line naming the clip it produces by a hash chained
//! from the clip it narrows, and each paint cites the hash of the clip in
//! effect: equal clips have equal names however many clips precede them.
//! A paint prints the source colour in effect, as the colour operator gave
//! it, in the colour space then selected.

use std::fmt::{self, Write as _};
use std::sync::Arc;

use raster::fnv::Fnv1a;
use raster::path::Path;

use pdf::{Dictionary, Object};

use super::PageRenderer;
use super::text_glyphs::TextModeOps;
use crate::RenderingIntent;
use crate::content::Operator;
use crate::resources::colorspace::{ColorSpace, Colorant};
use crate::resources::{FormXObject, ResourceKey, SoftMaskParams};
use raster::types::BlendMode;

/// The first line of every trace: the format and its version.
const HEADER: &str = "rasterrocket-paint-trace 1";

/// The most points a path prints in full.
pub(super) const INLINE_POINTS: usize = 8;

/// Fraction bits of a printed device coordinate, line width or font size.
const COORD_BITS: u32 = 8;

/// Decimal places of a printed colour component or matrix entry.
const UNIT_DIGITS: u32 = 6;

/// The colour a graphics state starts with (ISO 32000-1 §8.6.4.1).
const INITIAL_COLOUR: &str = "DeviceGray(0)";

/// The paint trace of a page.
#[derive(Clone, Debug)]
pub struct PaintTrace {
    text: String,
    /// Nesting depth of the open groups, masks and text objects; each
    /// indents its lines by two spaces.
    depth: usize,
    /// Nesting count of the regions whose operations are not recorded.
    quiet: u32,
    /// Whether the trace is complete: it is carried by the renderer of a
    /// later window of its page, which records nothing.
    sealed: bool,
}

impl PaintTrace {
    /// An empty trace of a page `width` × `height` device pixels.
    fn new(width: u32, height: u32) -> Self {
        Self {
            text: format!("{HEADER}\npage px={width}x{height}\n"),
            depth: 0,
            quiet: 0,
            sealed: false,
        }
    }

    /// The trace's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The trace's text, taken.
    #[must_use]
    pub fn into_string(self) -> String {
        self.text
    }

    /// Whether operations are recorded.
    const fn records(&self) -> bool {
        self.quiet == 0 && !self.sealed
    }

    /// Stop recording: the trace is complete.
    pub(super) const fn seal(&mut self) {
        self.sealed = true;
    }

    /// Append the line `body` at the current depth.
    fn line(&mut self, body: &str) {
        if !self.records() {
            return;
        }
        for _ in 0..self.depth {
            self.text.push_str("  ");
        }
        self.text.push_str(body);
        self.text.push('\n');
    }

    /// Append `body` and indent the lines after it, up to [`Self::close`].
    pub(super) fn open(&mut self, body: &str) {
        if !self.records() {
            return;
        }
        self.line(body);
        self.depth += 1;
    }

    /// End the indentation of the latest [`Self::open`], then append
    /// `body`, if any.
    pub(super) fn close(&mut self, body: Option<&str>) {
        if !self.records() {
            return;
        }
        self.depth = self.depth.saturating_sub(1);
        if let Some(body) = body {
            self.line(body);
        }
    }
}

impl fmt::Display for PaintTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// `v` rounded to the nearest integer, half away from zero; `None` when
/// `v` is not finite.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the float-to-int conversion saturates, and a value beyond 2^63 \
              quanta has no device meaning"
)]
fn finite_round(v: f64) -> Option<i64> {
    v.is_finite().then(|| v.round() as i64)
}

/// `v` rounded to the nearest multiple of `2^-bits`, in units of that
/// multiple; `None` when `v` is not finite.
fn quantise(v: f64, bits: u32) -> Option<i64> {
    finite_round(v * f64::from(1_u32 << bits))
}

/// Append the non-finite `v` as `nan`, `inf` or `-inf`.
fn push_non_finite(out: &mut String, v: f64) {
    out.push_str(if v.is_nan() {
        "nan"
    } else if v > 0.0 {
        "inf"
    } else {
        "-inf"
    });
}

/// Append `v`, rounded to the nearest multiple of `2^-bits`, as the exact
/// decimal of the rounded value; `nan`, `inf` or `-inf` when not finite.
fn push_quantised(out: &mut String, v: f64, bits: u32) {
    let Some(q) = quantise(v, bits) else {
        push_non_finite(out, v);
        return;
    };
    push_fixed(out, q, bits);
}

/// Append the fixed-point value `q`, of `bits` fraction bits, as its exact
/// decimal: `2^-bits = 5^bits · 10^-bits`, so the fraction has `bits`
/// decimal digits, trailing zeros dropped.
fn push_fixed(out: &mut String, q: i64, bits: u32) {
    let a = i128::from(q).abs();
    let one = 1_i128 << bits;
    if q < 0 {
        out.push('-');
    }
    let _ = write!(out, "{}", a / one);
    let frac = a % one;
    if frac != 0 {
        let digits = frac * 5_i128.pow(bits);
        let text = format!("{digits:0width$}", width = bits as usize);
        out.push('.');
        out.push_str(text.trim_end_matches('0'));
    }
}

/// Append the device coordinate `v`.
pub(super) fn push_coord(out: &mut String, v: f64) {
    push_quantised(out, v, COORD_BITS);
}

/// Append the colour component or matrix entry `v`.
pub(super) fn push_unit(out: &mut String, v: f64) {
    let scale = 10_i64.pow(UNIT_DIGITS);
    #[expect(
        clippy::cast_precision_loss,
        reason = "10^6 is exactly representable in f64"
    )]
    let Some(q) = finite_round(v * scale as f64) else {
        push_non_finite(out, v);
        return;
    };
    let a = i128::from(q).abs();
    let one = i128::from(scale);
    if q < 0 {
        out.push('-');
    }
    let _ = write!(out, "{}", a / one);
    let frac = a % one;
    if frac != 0 {
        let text = format!("{frac:0width$}", width = UNIT_DIGITS as usize);
        out.push('.');
        out.push_str(text.trim_end_matches('0'));
    }
}

/// Append the values `vs`, comma-separated, each by `push`.
pub(super) fn push_list(out: &mut String, vs: &[f64], push: fn(&mut String, f64)) {
    for (i, &v) in vs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push(out, v);
    }
}

/// Append `name` as a PDF name: `/`, then each byte outside `!`..=`~`, and
/// `#` itself, as `#` and two hexadecimal digits.
pub(super) fn push_name(out: &mut String, name: &[u8]) {
    out.push('/');
    for &b in name {
        if (b'!'..=b'~').contains(&b) && b != b'#' {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "#{b:02x}");
        }
    }
}

/// Append the identity of the object a resource refers to: `(n g R)`, or
/// for a direct object, the name it is held under in the resources of the
/// object `(n g)`.
pub(super) fn push_key(out: &mut String, key: &ResourceKey) {
    match key {
        ResourceKey::Object((n, g)) => {
            let _ = write!(out, "({n} {g} R)");
        }
        ResourceKey::Direct {
            context: (n, g),
            name,
        } => {
            let _ = write!(out, "(direct {n} {g} ");
            push_name(out, name);
            out.push(')');
        }
    }
}

/// Append the line of the glyph of `code`, drawn as the glyph `selector`
/// (`(selector, by_gid)`; `None` when the font draws nothing for it), at
/// the pen `pen` in the page's device space.
pub(super) fn push_glyph(
    out: &mut String,
    code: u32,
    selector: Option<(u32, bool)>,
    pen: (f64, f64),
) {
    let _ = write!(out, "glyph c={code:x}");
    match selector {
        Some((selector, by_gid)) => {
            let _ = write!(out, " sel={selector}{}", if by_gid { "g" } else { "" });
        }
        None => out.push_str(" none"),
    }
    out.push_str(" pen=");
    push_coord(out, pen.0);
    out.push(',');
    push_coord(out, pen.1);
    out.push('\n');
}

/// Append the summary of `path`, whose points lie in a window whose
/// top-left pixel is the page's pixel `origin`: its points' counts,
/// bounding box and hash in the page's device space, and the points of a
/// path of at most [`INLINE_POINTS`] points.
pub(super) fn push_path(out: &mut String, path: &Path, origin: (u32, u32)) {
    let (ox, oy) = (f64::from(origin.0), f64::from(origin.1));
    let mut hash = Fnv1a::new();
    let mut bbox: Option<[i64; 4]> = None;
    let (mut subpaths, mut curve_points) = (0_usize, 0_usize);
    for (pt, flags) in path.pts.iter().zip(&path.flags) {
        let (x, y) = (
            quantise(pt.x + ox, COORD_BITS),
            quantise(pt.y + oy, COORD_BITS),
        );
        hash.update(&x.unwrap_or(i64::MIN).to_le_bytes());
        hash.update(&y.unwrap_or(i64::MIN).to_le_bytes());
        hash.update(&[flags.bits()]);
        subpaths += usize::from(flags.is_first());
        curve_points += usize::from(flags.is_curve());
        if let (Some(x), Some(y)) = (x, y) {
            bbox = Some(bbox.map_or([x, y, x, y], |[x0, y0, x1, y1]| {
                [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
            }));
        }
    }
    let _ = write!(
        out,
        "n={} sub={subpaths} cub={}",
        path.pts.len(),
        curve_points / 2
    );
    out.push_str(" bbox=");
    match bbox {
        Some(b) => {
            for (i, &v) in b.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_fixed(out, v, COORD_BITS);
            }
        }
        None => out.push('-'),
    }
    let _ = write!(out, " h={:016x}", hash.finish());
    if path.pts.len() <= INLINE_POINTS && !path.pts.is_empty() {
        out.push_str(" pts=");
        for (i, (pt, flags)) in path.pts.iter().zip(&path.flags).enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push(if flags.is_first() {
                'm'
            } else if flags.is_curve() {
                'c'
            } else {
                'l'
            });
            push_coord(out, pt.x + ox);
            out.push(',');
            push_coord(out, pt.y + oy);
            if flags.is_last() && flags.is_closed() {
                out.push_str(" z");
            }
        }
    }
}

/// The name of colour space `cs` in a trace: its family, and the
/// parameters that identify it.
pub(super) fn space_name(cs: &ColorSpace) -> String {
    match cs {
        ColorSpace::DeviceGray => "DeviceGray".to_owned(),
        ColorSpace::DeviceRgb => "DeviceRGB".to_owned(),
        ColorSpace::DeviceCmyk => "DeviceCMYK".to_owned(),
        ColorSpace::Cal(_) => "Cal".to_owned(),
        ColorSpace::Lab(_) => "Lab".to_owned(),
        ColorSpace::IccBased { n, stream_id } => match stream_id {
            Some((id, g)) => format!("ICCBased[{n} ({id} {g} R)]"),
            None => format!("ICCBased[{n}]"),
        },
        ColorSpace::Indexed { base, hival, .. } => {
            format!("Indexed[{} {hival}]", space_name(base))
        }
        ColorSpace::Pattern { base } => base.as_ref().map_or_else(
            || "Pattern".to_owned(),
            |base| format!("Pattern[{}]", space_name(base)),
        ),
        ColorSpace::Separation {
            alternate,
            colorant,
            ..
        } => {
            let mut s = "Separation[".to_owned();
            push_colorant(&mut s, colorant);
            let _ = write!(s, " {}]", space_name(alternate));
            s
        }
        ColorSpace::DeviceN {
            colorants,
            alternate,
            ..
        } => {
            let mut s = "DeviceN[".to_owned();
            for colorant in colorants.iter() {
                push_colorant(&mut s, colorant);
                s.push(' ');
            }
            let _ = write!(s, "{}]", space_name(alternate));
            s
        }
    }
}

/// Append the name of `colorant`.
fn push_colorant(out: &mut String, colorant: &Colorant) {
    match colorant {
        Colorant::None => out.push_str("/None"),
        Colorant::All => out.push_str("/All"),
        Colorant::Process(ink) => {
            let _ = write!(out, "/{ink:?}");
        }
        Colorant::Spot(name) => push_name(out, name),
    }
}

/// The colour `comps` in the space `cs`, as a paint prints it.
fn colour_text(cs: &ColorSpace, comps: &[f64]) -> Arc<str> {
    let mut s = space_name(cs);
    s.push('(');
    push_list(&mut s, comps, push_unit);
    s.push(')');
    Arc::from(s)
}

/// The name of the clip that narrowing the clip named `parent` by the
/// operation printed as `body` produces.
fn chained_clip(parent: u64, body: &str) -> u64 {
    let mut hash = Fnv1a::new();
    hash.update(&parent.to_le_bytes());
    hash.update(body.as_bytes());
    hash.finish()
}

impl PageRenderer<'_> {
    /// Record the page's paint trace while it executes; the trace is taken
    /// with [`Self::take_paint_trace`] or, for a page drawn in windows,
    /// [`super::PageState::take_paint_trace`].  Call before
    /// [`Self::execute`].  Only the renderer of a page's first window
    /// records: [`Self::adopt_page_state`] stops a later window's trace.
    pub fn set_paint_trace(&mut self) {
        self.trace = Some(Box::new(PaintTrace::new(self.page_width, self.page_height)));
    }

    /// The paint trace recorded since [`Self::set_paint_trace`], if any.
    pub fn take_paint_trace(&mut self) -> Option<PaintTrace> {
        self.trace.take().map(|t| *t)
    }

    /// Whether the page's paint trace is being recorded.
    #[inline]
    pub(super) fn tracing(&self) -> bool {
        self.trace.as_deref().is_some_and(PaintTrace::records)
    }

    /// Run `f` with no operation recorded: content whose effect on the
    /// page is recorded as one operation, a Type 3 glyph's description.
    pub(super) fn without_trace<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
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
    pub(super) fn trace_line(&mut self, body: &str) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.line(body);
        }
    }

    /// Append `body` to the trace and indent what follows, up to
    /// [`Self::trace_close`].
    pub(super) fn trace_open(&mut self, body: &str) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.open(body);
        }
    }

    /// End the indentation of the latest [`Self::trace_open`], then append
    /// `body`, if any.
    pub(super) fn trace_close(&mut self, body: Option<&str>) {
        if let Some(trace) = self.trace.as_deref_mut() {
            trace.close(body);
        }
    }

    /// Append the paint attributes of the stroking (`stroke`) or
    /// non-stroking operation under the current graphics state: colour or
    /// pattern, constant alpha, blend mode, overprint, soft mask, transfer,
    /// rendering intent and clip.
    pub(super) fn push_paint(&self, out: &mut String, stroke: bool) {
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
    pub(super) fn push_compositing(&self, out: &mut String, key: &str, alpha: u8) {
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
        if gs.rendering_intent != RenderingIntent::RelativeColorimetric {
            let _ = write!(out, " ri={:?}", gs.rendering_intent);
        }
        let _ = write!(out, " clip={:016x}", gs.trace_clip);
    }

    /// Record the fill (`stroke == false`), stroke, or fill and stroke
    /// (`fill_rule` and `stroke`) of `path` by a path-painting operator.
    pub(super) fn trace_path_paint(&mut self, path: &Path, fill_rule: Option<bool>, stroke: bool) {
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
    pub(super) fn trace_clip(&mut self, body: &str) {
        let parent = self.gstate.current().trace_clip;
        let clip = chained_clip(parent, body);
        self.gstate.current_mut().trace_clip = clip;
        self.trace_line(&format!("clip {clip:016x}<{parent:016x} {body}"));
    }

    /// Record the clip by `path` under the even-odd rule when `even_odd`.
    pub(super) fn trace_clip_path(&mut self, path: &Path, even_odd: bool) {
        if !self.tracing() {
            return;
        }
        let mut body = String::from(if even_odd { "eo " } else { "nz " });
        push_path(&mut body, path, self.window_origin);
        self.trace_clip(&body);
    }

    /// Record the text object a string shown with the font `font` (`None`
    /// without one) in the mode of operations `mode` paints, its fill
    /// through outlines when `fill_as_path`, and its glyphs' lines
    /// `glyphs`.
    pub(super) fn trace_text(
        &mut self,
        font: Option<&ResourceKey>,
        mode: TextModeOps,
        fill_as_path: bool,
        glyphs: &str,
    ) {
        if !self.tracing() {
            return;
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
            self.push_paint(&mut out, true);
        }
        if mode.clip {
            out.push_str(" | clips");
        }
        let Some(trace) = self.trace.as_deref_mut() else {
            return;
        };
        trace.open(&out);
        for line in glyphs.lines() {
            trace.line(line);
        }
        trace.close(None);
    }

    /// Record the drawing of the image `XObject` `name` under the current
    /// graphics state.
    pub(super) fn trace_image_xobject(&mut self, name: &[u8]) {
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
    pub(super) fn trace_inline_image(&mut self, params: &[u8], data: &[u8]) {
        if !self.tracing() {
            return;
        }
        let dict = crate::resources::image::inline::parse_inline_params(params);
        let mut out = format!("image inline len={} h={:016x}", data.len(), Fnv1a::of(data));
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
            " w={} h={} bpc={}",
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
            let _ = if i == 0 {
                unit.move_to(dx, dy)
            } else {
                unit.line_to(dx, dy)
            };
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
    pub(super) fn trace_shading(&mut self, name: &[u8]) {
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
    pub(super) fn trace_form(&mut self, form: &FormXObject) {
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
    pub(super) fn trace_soft_mask<T>(
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
    pub(super) fn trace_hidden(&mut self, name: &[u8]) {
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
    pub(super) fn trace_clip_rect(&mut self, corners: &[(f64, f64); 4]) {
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
    pub(super) fn trace_colour(&mut self, op: &Operator<'_>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(q: i64, bits: u32) -> String {
        let mut s = String::new();
        push_fixed(&mut s, q, bits);
        s
    }

    fn coord(v: f64) -> String {
        let mut s = String::new();
        push_coord(&mut s, v);
        s
    }

    #[test]
    fn fixed_point_values_print_as_exact_decimals() {
        assert_eq!(fixed(0, 8), "0");
        assert_eq!(fixed(256 * 12 + 128, 8), "12.5");
        assert_eq!(fixed(256 * 12 + 1, 8), "12.00390625");
        assert_eq!(fixed(-1, 8), "-0.00390625");
        assert_eq!(fixed(1, 16), "0.0000152587890625");
        let mut unit = String::new();
        push_list(&mut unit, &[0.2, -0.5, 1.0, 1e-7, 0.123_456_5], push_unit);
        assert_eq!(unit, "0.2,-0.5,1,0,0.123457");
    }

    #[test]
    fn coordinates_round_to_the_nearest_quantum() {
        assert_eq!(coord(10.0), "10");
        assert_eq!(coord(10.0 - 1e-9), "10");
        assert_eq!(coord(10.0 + 1e-9), "10");
        assert_eq!(coord(0.5 / 256.0 - 1e-9), "0");
        assert_eq!(coord(f64::NAN), "nan");
        assert_eq!(coord(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn names_escape_delimiters_and_controls() {
        let mut s = String::new();
        push_name(&mut s, b"F1 #a\x01");
        assert_eq!(s, "/F1#20#23a#01");
    }

    #[test]
    fn a_small_path_prints_its_points_in_the_page_space() {
        let mut pb = raster::path::PathBuilder::new();
        let _ = pb.move_to(1.0, 2.0);
        let _ = pb.line_to(3.5, 2.0);
        let _ = pb.line_to(3.5, 4.0);
        let _ = pb.close(true);
        let mut s = String::new();
        push_path(&mut s, &pb.build(), (10, 20));
        assert!(
            s.starts_with("n=4 sub=1 cub=0 bbox=11,22,13.5,24 h="),
            "{s}"
        );
        assert!(s.ends_with(" pts=m11,22 l13.5,22 l13.5,24 l11,22 z"), "{s}");
    }

    #[test]
    fn a_path_hash_ignores_last_bit_differences() {
        let path = |dx: f64| {
            let mut pb = raster::path::PathBuilder::new();
            let _ = pb.move_to(1.0 + dx, 2.0);
            let _ = pb.line_to(30.0, 2.0 - dx);
            pb.build()
        };
        let summary = |p: &Path| {
            let mut s = String::new();
            push_path(&mut s, p, (0, 0));
            s
        };
        assert_eq!(summary(&path(0.0)), summary(&path(1e-12)));
        assert_ne!(summary(&path(0.0)), summary(&path(1.0 / 256.0)));
    }

    #[test]
    fn a_clip_name_depends_on_its_parent_and_its_body() {
        assert_eq!(chained_clip(0, "nz a"), chained_clip(0, "nz a"));
        assert_ne!(chained_clip(0, "nz a"), chained_clip(1, "nz a"));
        assert_ne!(chained_clip(0, "nz a"), chained_clip(0, "eo a"));
    }
}
