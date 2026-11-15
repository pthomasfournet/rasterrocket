//! The sample depth of a page canvas, and the sources and buffers of a
//! paint as a canvas of either depth takes them.
//!
//! A page is painted on 8-bit or 16-bit samples, chosen per render.  The
//! paint code is written once over [`CanvasSample`], the sample type of the
//! canvas it paints: the depth-tagged values the renderer holds — a group
//! backdrop, a snapshot's samples, a shading's pattern — are projected onto
//! that type, and a source of 8-bit samples reaches a 16-bit canvas through
//! [`Widened`], which widens each sample exactly ([`Sample::from_u8`]).

use std::cell::RefCell;

use color::{PixelMode, Sample};
use raster::pipe::{ChannelMask, Pattern};
use raster::transparency::Backdrop;

use super::super::color::RasterColor;
use crate::resources::transfer::Transfer;

/// Samples of either depth, held by a value the renderer keeps across
/// paints of one canvas.
#[derive(Debug)]
pub(crate) enum SampleVec {
    /// 8-bit samples.
    U8(Vec<u8>),
    /// 16-bit samples.
    U16(Vec<u16>),
}

/// The backdrop of a non-isolated group, of the depth of its canvas.
pub(super) enum CanvasBackdrop {
    /// A backdrop of 8-bit samples.
    U8(Backdrop<u8>),
    /// A backdrop of 16-bit samples.
    U16(Backdrop<u16>),
}

/// The sample type of a page canvas: the projections of the renderer's
/// depth-tagged values onto it.  Each projection gives `None` for a value
/// of the other depth, which a canvas never receives: the renderer makes
/// every value at the depth of the canvas it paints.
pub(super) trait CanvasSample: Sample {
    /// `pattern` when it yields samples of this type.
    fn pattern_u8(pattern: &dyn Pattern<u8>) -> Option<&dyn Pattern<Self>>;

    /// `pattern` when it yields samples of this type.
    fn pattern_u16(pattern: &dyn Pattern<u16>) -> Option<&dyn Pattern<Self>>;

    /// The samples of `v` when they are of this type.
    fn samples(v: &SampleVec) -> Option<&[Self]>;

    /// `v` as depth-tagged samples.
    fn into_samples(v: Vec<Self>) -> SampleVec;

    /// The backdrop `b` when its samples are of this type.
    fn backdrop(b: &CanvasBackdrop) -> Option<&Backdrop<Self>>;

    /// `b` as a depth-tagged backdrop.
    fn into_backdrop(b: Backdrop<Self>) -> CanvasBackdrop;

    /// The samples of the solid colour `colour` for a pixel of `mode`:
    /// its bytes for an 8-bit pixel ([`RasterColor::bytes_for`]); for a
    /// 16-bit one its 16-bit ink on an ink pixel, its 16-bit sRGB
    /// otherwise.
    fn solid(colour: &RasterColor, mode: PixelMode) -> &[Self];

    /// The red, green and blue tables of `transfer` at this depth.
    fn rgb_tables(transfer: &Transfer) -> [&Self::Table; 3];

    /// The channels `channels` of the ink `ink` mapped through the ink
    /// tables of `transfer` at this depth.
    fn map_ink(transfer: &Transfer, ink: [Self; 8], channels: ChannelMask) -> [Self; 8];

    /// Run `f` on a span of `len` samples of this type held by the thread
    /// for an sRGB span converted to ink; its contents are unspecified.
    ///
    /// # Panics
    ///
    /// When called again from within `f`.
    fn with_rgb_span<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R;
}

thread_local! {
    /// The 8-bit sRGB span of [`CanvasSample::with_rgb_span`].  Separate
    /// from the pipe's own span buffers, which are held while a pattern
    /// fills a span.  Grow-never-shrink.
    static RGB_SPAN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// The 16-bit sRGB span of [`CanvasSample::with_rgb_span`].
    static RGB_SPAN16: RefCell<Vec<u16>> = const { RefCell::new(Vec::new()) };
}

impl CanvasSample for u8 {
    fn pattern_u8(pattern: &dyn Pattern<u8>) -> Option<&dyn Pattern<Self>> {
        Some(pattern)
    }

    fn pattern_u16(_: &dyn Pattern<u16>) -> Option<&dyn Pattern<Self>> {
        None
    }

    fn samples(v: &SampleVec) -> Option<&[Self]> {
        match v {
            SampleVec::U8(v) => Some(v),
            SampleVec::U16(_) => None,
        }
    }

    fn into_samples(v: Vec<Self>) -> SampleVec {
        SampleVec::U8(v)
    }

    fn backdrop(b: &CanvasBackdrop) -> Option<&Backdrop<Self>> {
        match b {
            CanvasBackdrop::U8(b) => Some(b),
            CanvasBackdrop::U16(_) => None,
        }
    }

    fn into_backdrop(b: Backdrop<Self>) -> CanvasBackdrop {
        CanvasBackdrop::U8(b)
    }

    fn solid(colour: &RasterColor, mode: PixelMode) -> &[Self] {
        colour.bytes_for(mode)
    }

    fn rgb_tables(transfer: &Transfer) -> [&Self::Table; 3] {
        transfer.tables()
    }

    fn map_ink(transfer: &Transfer, ink: [Self; 8], channels: ChannelMask) -> [Self; 8] {
        transfer.apply_ink(ink, channels)
    }

    fn with_rgb_span<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        RGB_SPAN.with_borrow_mut(|span| {
            span.resize(len, 0);
            f(span)
        })
    }
}

impl CanvasSample for u16 {
    fn pattern_u8(_: &dyn Pattern<u8>) -> Option<&dyn Pattern<Self>> {
        None
    }

    fn pattern_u16(pattern: &dyn Pattern<u16>) -> Option<&dyn Pattern<Self>> {
        Some(pattern)
    }

    fn samples(v: &SampleVec) -> Option<&[Self]> {
        match v {
            SampleVec::U16(v) => Some(v),
            SampleVec::U8(_) => None,
        }
    }

    fn into_samples(v: Vec<Self>) -> SampleVec {
        SampleVec::U16(v)
    }

    fn backdrop(b: &CanvasBackdrop) -> Option<&Backdrop<Self>> {
        match b {
            CanvasBackdrop::U16(b) => Some(b),
            CanvasBackdrop::U8(_) => None,
        }
    }

    fn into_backdrop(b: Backdrop<Self>) -> CanvasBackdrop {
        CanvasBackdrop::U16(b)
    }

    fn solid(colour: &RasterColor, mode: PixelMode) -> &[Self] {
        if mode == PixelMode::DeviceN16 {
            colour.ink16()
        } else {
            colour.rgb16()
        }
    }

    fn rgb_tables(transfer: &Transfer) -> [&Self::Table; 3] {
        transfer.tables16()
    }

    fn map_ink(transfer: &Transfer, ink: [Self; 8], channels: ChannelMask) -> [Self; 8] {
        let tables = transfer.ink_tables16();
        std::array::from_fn(|i| {
            if channels.paints(i) {
                tables[i][usize::from(ink[i])]
            } else {
                ink[i]
            }
        })
    }

    fn with_rgb_span<R>(len: usize, f: impl FnOnce(&mut [Self]) -> R) -> R {
        RGB_SPAN16.with_borrow_mut(|span| {
            span.resize(len, 0);
            f(span)
        })
    }
}

thread_local! {
    /// The 8-bit colour and coverage a [`Widened`] pattern widens.  The
    /// inner pattern yields 8-bit samples, so it is never a `Widened`
    /// itself and the buffers are never borrowed twice.  Grow-never-shrink.
    static NARROW_SPAN: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// A [`Pattern`] of 8-bit samples served at the depth of `S`, each sample
/// widened exactly by [`Sample::from_u8`]; at 8 bits the inner pattern
/// fills the span itself.  Coverage and the static-colour hint are the
/// inner pattern's.
pub(super) struct Widened<'a, S> {
    inner: &'a dyn Pattern<u8>,
    _sample: std::marker::PhantomData<fn() -> S>,
}

impl<'a, S: Sample> Widened<'a, S> {
    /// `inner` served at the depth of `S`.
    pub(super) const fn new(inner: &'a dyn Pattern<u8>) -> Self {
        Self {
            inner,
            _sample: std::marker::PhantomData,
        }
    }
}

/// Write the exact widening of each byte of `narrow` to `wide`.
fn widen<S: Sample>(narrow: &[u8], wide: &mut [S]) {
    for (w, &n) in wide.iter_mut().zip(narrow) {
        *w = S::from_u8(n);
    }
}

impl<S: Sample> Pattern<S> for Widened<'_, S> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]) {
        if let Some(out) = S::as_u8_mut(out) {
            self.inner.fill_span(y, x0, x1, out);
            return;
        }
        NARROW_SPAN.with_borrow_mut(|(colour, _)| {
            colour.resize(out.len(), 0);
            self.inner.fill_span(y, x0, x1, colour);
            widen(colour, out);
        });
    }

    fn is_static_color(&self) -> bool {
        self.inner.is_static_color()
    }

    fn may_leave_unpainted(&self) -> bool {
        self.inner.may_leave_unpainted()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        if let (Some(out), Some(coverage)) = (S::as_u8_mut(out), S::as_u8_mut(coverage)) {
            self.inner.fill_span_coverage(y, x0, x1, out, coverage);
            return;
        }
        NARROW_SPAN.with_borrow_mut(|(colour, shape)| {
            colour.resize(out.len(), 0);
            shape.resize(coverage.len(), 0);
            self.inner.fill_span_coverage(y, x0, x1, colour, shape);
            widen(colour, out);
            widen(shape, coverage);
        });
    }
}

/// A source pattern as a canvas of samples of type `S` takes it: one of
/// that type, or one of 8-bit samples widened.
pub(super) enum SourcePattern<'a, S> {
    /// A pattern of the canvas's samples.
    Native(&'a dyn Pattern<S>),
    /// A pattern of 8-bit samples, widened.
    Widened(Widened<'a, S>),
}

impl<'a, S: CanvasSample> SourcePattern<'a, S> {
    /// `pattern`, of 8-bit samples, for a canvas of samples of type `S`.
    pub(super) fn of_u8(pattern: &'a dyn Pattern<u8>) -> Self {
        S::pattern_u8(pattern).map_or_else(|| Self::Widened(Widened::new(pattern)), Self::Native)
    }

    /// The pattern to hand to the pipe.
    pub(super) fn as_dyn(&self) -> &dyn Pattern<S> {
        match self {
            Self::Native(p) => *p,
            Self::Widened(p) => p,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::pipe::RowPattern;

    #[test]
    fn a_widened_pattern_is_exact_and_passes_through_at_8_bits() {
        let data = [0u8, 1, 128, 255, 7, 9];
        let row = RowPattern { data: &data };
        let mut narrow = [0u8; 6];
        SourcePattern::<u8>::of_u8(&row)
            .as_dyn()
            .fill_span(0, 0, 1, &mut narrow);
        assert_eq!(narrow, data);
        let mut wide = [0u16; 6];
        let mut coverage = [0u16; 2];
        SourcePattern::<u16>::of_u8(&row)
            .as_dyn()
            .fill_span_coverage(0, 0, 1, &mut wide, &mut coverage);
        assert_eq!(wide, data.map(u16::from_u8));
        assert_eq!(coverage, [u16::MAX; 2]);
    }
}
