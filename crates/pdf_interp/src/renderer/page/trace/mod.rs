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
//! rounded points and flags; a path of at most
//! [`format::INLINE_POINTS`] points also prints its points.
//!
//! # Clips and colours
//!
//! Each clip prints a line naming the clip it produces by a hash chained
//! from the clip it narrows, and each paint cites the hash of the clip in
//! effect: equal clips have equal names however many clips precede them.
//! A paint prints the source colour in effect, as the colour operator gave
//! it, in the colour space then selected.

mod format;
mod hooks;

use std::fmt;

pub(super) use format::push_glyph;
use format::push_unit;

/// The first line of every trace: the format and its version.
const HEADER: &str = "rasterrocket-paint-trace 1";

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
    /// An empty trace of a page `width` × `height` device pixels, drawn
    /// at `scale` device pixels per unit of default user space.
    fn new(width: u32, height: u32, scale: f64) -> Self {
        let mut text = format!("{HEADER}\npage px={width}x{height} scale=");
        push_unit(&mut text, scale);
        text.push('\n');
        Self {
            text,
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
