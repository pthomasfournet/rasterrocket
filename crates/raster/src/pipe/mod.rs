//! Compositing pipeline.
//!
//! Each paint operation composites spans through [`render_span`], which is
//! monomorphized over `P: Pixel` and selects one of three variants per span.
//!
//! # Three pipeline variants
//!
//! | Variant | When selected |
//! |---------|--------------|
//! | `simple` | `a_input == 255`, `BlendMode::Normal`, no shape, no backdrop |
//! | `aa`     | shape present, `BlendMode::Normal`, no backdrop |
//! | `general`| everything else — blend modes, non-isolated groups |

pub(crate) mod aa;
pub(crate) mod blend;
pub(crate) mod general;
mod scratch;
pub(crate) mod simple;
pub(crate) mod transfer;

use std::ops::RangeInclusive;

use crate::bitmap::Bitmap;
use crate::types::BlendMode;
use color::{Pixel, Sample};

pub(crate) use scratch::Scratch;
pub use transfer::TransferSet;

/// Source colour for a single paint operation, of samples of type `S`.
///
/// The source is resolved once per span and may be a flat colour (most common) or
/// a dynamic [`Pattern`] queried per-pixel.
pub enum PipeSrc<'a, S: Sample = u8> {
    /// Flat colour: `ncomps` samples, already in device space.
    Solid(&'a [S]),
    /// Dynamic pattern source — `fill_span` is called per output row.
    Pattern(&'a dyn Pattern<S>),
}

/// A source of per-pixel colour for use with the compositing pipeline.
///
/// Implementors write device-space samples of type `S` into the provided
/// buffer.  The buffer length is `(x1 - x0 + 1) * ncomps` where `ncomps` is
/// the channel count of the target bitmap's pixel type.
pub trait Pattern<S: Sample = u8>: Send + Sync {
    /// Fill `out` with colour samples for pixels `x0..=x1` on scanline `y`.
    ///
    /// `out.len()` is guaranteed to be `(x1 - x0 + 1) * ncomps`.
    ///
    /// Must not composite: `fill_span` runs while the pipe holds its
    /// thread-local span buffers, so a nested composite would re-enter them.
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]);

    /// Return `true` if this pattern yields the same colour at every coordinate.
    /// When `true`, `fill_span` will be called once and the result reused across
    /// the whole span (optimisation hint only — correctness is not affected).
    fn is_static_color(&self) -> bool {
        false
    }

    /// Return `true` if this pattern may leave some pixels unpainted or only
    /// partially covered.
    ///
    /// A pixel of coverage `c` composites the pattern colour with its source
    /// alpha scaled by `c / S::MAX`; an unpainted pixel (coverage 0) keeps its
    /// destination colour and alpha, exactly as a pixel outside the filled
    /// area does.  [`render_span`] evaluates such a pattern through
    /// [`Self::fill_span_coverage`] and composites only its covered runs; a
    /// pattern returning `false` covers every pixel of every span fully.
    fn may_leave_unpainted(&self) -> bool {
        false
    }

    /// Fill `out` as [`Self::fill_span`] does and set `coverage[i]` to the
    /// coverage of pixel `x0 + i`, from 0 (unpainted) to `S::MAX` (fully
    /// painted).
    ///
    /// `coverage.len()` is `x1 - x0 + 1`.  The colour bytes of an unpainted
    /// pixel are unspecified.  The default covers every pixel fully.
    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        self.fill_span(y, x0, x1, out);
        coverage.fill(S::MAX);
    }

    /// Set `channels[i]` to the channels the pattern's own content wrote
    /// at pixel `x0 + i` of scanline `y`, the bits of a [`ChannelMask`],
    /// and 0 where that is not recorded, and return `true`; or return
    /// `false`, `channels` unspecified, for a pattern without a per-pixel
    /// record.  `channels.len()` is `x1 - x0 + 1`.
    ///
    /// Under a pipe with an [`overprint`](PipeState::overprint) mask,
    /// [`render_span`] paints each pixel of a pattern with a record
    /// through the recorded channels within that mask, and a pixel
    /// recorded 0 through the mask itself.  The default has no record.
    fn fill_span_channels(&self, y: i32, x0: i32, x1: i32, channels: &mut [u8]) -> bool {
        let _ = (y, x0, x1, channels);
        false
    }
}

/// A [`Pattern`] of another pixel grid, read on a window of it.
///
/// The window's pixel `(0, 0)` is the grid's pixel `origin`: its pixel
/// `(x, y)` takes `inner`'s colour and coverage at `(x + origin.0, y +
/// origin.1)`.  The offset is an integer, so `inner` sees the coordinates
/// it would see on its own grid.
pub struct Shifted<'a, S: Sample = u8> {
    /// The pattern over the other grid.
    pub inner: &'a dyn Pattern<S>,
    /// The other grid's pixel at the bitmap's pixel `(0, 0)`.
    pub origin: (i32, i32),
}

impl<S: Sample> Pattern<S> for Shifted<'_, S> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]) {
        let (dx, dy) = self.origin;
        self.inner.fill_span(y + dy, x0 + dx, x1 + dx, out);
    }

    fn is_static_color(&self) -> bool {
        self.inner.is_static_color()
    }

    fn may_leave_unpainted(&self) -> bool {
        self.inner.may_leave_unpainted()
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        let (dx, dy) = self.origin;
        self.inner
            .fill_span_coverage(y + dy, x0 + dx, x1 + dx, out, coverage);
    }

    fn fill_span_channels(&self, y: i32, x0: i32, x1: i32, channels: &mut [u8]) -> bool {
        let (dx, dy) = self.origin;
        self.inner
            .fill_span_channels(y + dy, x0 + dx, x1 + dx, channels)
    }
}

/// A [`Pattern`] serving one precomputed span of colour samples.
///
/// `fill_span` copies `data` into `out`, whose length must equal
/// `data.len()`: a `RowPattern` is valid only for the span it holds.
pub struct RowPattern<'a, S: Sample = u8> {
    /// Colour samples of the span, `P::NCOMPS` per pixel.
    pub data: &'a [S],
}

impl<S: Sample> Pattern<S> for RowPattern<'_, S> {
    fn fill_span(&self, _y: i32, x0: i32, x1: i32, out: &mut [S]) {
        assert_eq!(
            out.len(),
            self.data.len(),
            "RowPattern::fill_span: out.len()={} != data.len()={} (x0={x0} x1={x1})",
            out.len(),
            self.data.len(),
        );
        out.copy_from_slice(self.data);
    }
}

thread_local! {
    /// Destination colour and alpha saved across a masked composite.
    /// Separate from the coverage scratch because the masked composite
    /// encloses its use.
    static MASKED_SPAN: Scratch = const { Scratch::new() };
    /// The colour the blend function reads under a masked paint of a blend
    /// mode other than `Normal`.
    static UNPAINTED_SOURCE: Scratch = const { Scratch::new() };
    /// Colour of the span of a pattern that may leave pixels unpainted.
    static COVERAGE_COLOUR: Scratch = const { Scratch::new() };
    /// Coverage of that span.
    static COVERAGE_VALUES: Scratch = const { Scratch::new() };
    /// Coverage of that span combined with the span's shape.
    static COVERAGE_SHAPE: Scratch = const { Scratch::new() };
    /// The per-pixel channels a pattern recorded over a span.
    static SPAN_CHANNELS: Scratch = const { Scratch::new() };
    /// A pattern's colour over a span whose paint is recorded.
    static MARK_COLOUR: Scratch = const { Scratch::new() };
    /// A pattern's coverage over a span whose paint is recorded.
    static MARK_COVERAGE: Scratch = const { Scratch::new() };
}

/// Immutable parameters for one paint operation, built once per fill/stroke/glyph call.
///
/// `'bmp` borrows slices out of the destination bitmap's alpha plane and the
/// current graphics state's transfer tables.  `S` is the sample type of the
/// destination's pixels.
#[derive(Copy, Clone, Debug)]
pub struct PipeState<'bmp, S: Sample = u8> {
    /// Compositing blend mode.
    pub blend_mode: BlendMode,

    /// Source opacity, pre-scaled: `state.fill_alpha · S::MAX` rounded.
    /// For stroke operations the caller passes the stroke alpha.
    pub a_input: S,

    /// Transfer function tables for the current pixel mode.  Applied to the
    /// composited result before writing.
    pub transfer: TransferSet<'bmp, S>,

    /// The backdrop of the non-isolated group being painted into, consulted
    /// by the blend function; `None` outside such a group, and for
    /// `BlendMode::Normal`, which does not consult it -- except that a paint
    /// with an [`overprint`](Self::overprint) mask inside such a group must
    /// supply it whatever the blend mode, since the unpainted channels are
    /// kept relative to it.
    pub backdrop: Option<GroupBackdrop<'bmp, S>>,

    /// The colour channels this paint writes; `None` writes all of them.
    /// A mask covering every channel of the pixel is equivalent to `None`.
    ///
    /// Inside a non-isolated group the group bitmap holds only the group's own
    /// colour, so the caller must also set [`backdrop`](Self::backdrop); the
    /// unpainted channels then keep the premultiplied colour the group
    /// composites to over that backdrop.
    pub overprint: Option<ChannelMask>,

    /// A blending space of one gray component for the blend function, when
    /// the group painted into composites in gray on a canvas of more
    /// components; `None` blends the destination's own components.
    pub gray: Option<&'bmp dyn GrayBlend<S>>,
}

/// A blending space of one gray component on a destination of more.
///
/// The blend function acts on the gray of the source and of the pixel it
/// blends with, and its result is written as a pixel of that gray (ISO
/// 32000-2 §11.3.4).  The compositing formula is linear in the colours it
/// mixes, so a destination whose colours are reduced to their gray
/// afterwards holds the result of compositing in gray, provided
/// [`gray`](Self::gray) is linear and the gray of the pixel
/// [`neutral`](Self::neutral) writes for `g` is `g`.
pub trait GrayBlend<S: Sample>: std::fmt::Debug {
    /// The gray of the pixel `px`.
    fn gray(&self, px: &[S]) -> S;
    /// Write into `px` a pixel whose gray is `gray`.
    fn neutral(&self, gray: S, px: &mut [S]);
}

/// The set of colour channels a paint operation writes.
///
/// Bit `i` is set when channel `i` (the `i`-th sample of a pixel) is painted.
/// The destination alpha is composited regardless, since the object's shape
/// is painted.  Under `Normal` a channel outside the mask receives no ink
/// from the source and keeps its premultiplied value: over an opaque
/// destination it keeps its value exactly, and over a translucent one its
/// value scales by the ratio of the old alpha to the new (see
/// `restore_unpainted`).  Under another blend mode it is blended with
/// itself (see [`render_span`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChannelMask(u8);

impl ChannelMask {
    /// Every channel is painted.
    pub const ALL: Self = Self(u8::MAX);

    /// The mask whose painted channels are the set bits of `bits`.
    #[must_use]
    pub const fn new(bits: u8) -> Self {
        Self(bits)
    }

    /// The mask as a bit set; bit `i` is channel `i`.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether `channel` is painted; `false` for `channel >= 8`.
    #[must_use]
    pub const fn paints(self, channel: usize) -> bool {
        channel < 8 && (self.0 >> channel) & 1 == 1
    }
}

/// The backdrop of a non-isolated transparency group (ISO 32000-2 §11.4.7):
/// the pixels under the group when it was opened.
///
/// The group's bitmap holds the group's own colour and alpha.  A blend mode
/// applied inside the group blends the source with the group's content
/// composited over this backdrop, weighted by that union's alpha, and the
/// result is stored as the group's own colour again, so the backdrop shapes
/// the blend but never enters the stored pixels; compositing the group onto
/// its parent then needs no removal of the backdrop.
#[derive(Copy, Clone, Debug)]
pub struct GroupBackdrop<'a, S: Sample = u8> {
    /// Colour samples of the copied rectangle, `ncomps` per pixel, row-major.
    pub colour: &'a [S],
    /// Alpha of each copied pixel; `None` for an opaque backdrop.
    pub alpha: Option<&'a [S]>,
    /// Device column of the copied rectangle's top-left pixel.
    pub x0: i32,
    /// Device row of the copied rectangle's top-left pixel.
    pub y0: i32,
    /// Pixels per row of the copied rectangle.
    pub width: usize,
    /// Rows of the copied rectangle.
    pub height: usize,
}

impl<S: Sample> GroupBackdrop<'_, S> {
    /// The colour samples and alpha of the backdrop at device pixel
    /// `(x, y)`; `None` outside the copied rectangle, where the backdrop is
    /// transparent.
    #[must_use]
    pub fn pixel(&self, x: i32, y: i32, ncomps: usize) -> Option<(&[S], S)> {
        let col = usize::try_from(x.checked_sub(self.x0)?).ok()?;
        let row = usize::try_from(y.checked_sub(self.y0)?).ok()?;
        if col >= self.width || row >= self.height {
            return None;
        }
        let i = row * self.width + col;
        let alpha = self.alpha.map_or(S::MAX, |a| a[i]);
        Some((&self.colour[i * ncomps..(i + 1) * ncomps], alpha))
    }
}

impl<S: Sample> PipeState<'_, S> {
    /// Returns `true` when the simple (no-transparency) fast path is applicable.
    ///
    /// Applicable when `a_input` is opaque, there is no shape and no backdrop.
    #[must_use]
    pub fn no_transparency(&self, uses_shape: bool) -> bool {
        self.a_input == S::MAX && !uses_shape && self.backdrop.is_none()
    }

    /// Returns `true` when the AA (shape-only) fast path is applicable.
    ///
    /// Applicable when there is no backdrop and the blend mode is
    /// `BlendMode::Normal`.
    #[must_use]
    pub fn use_aa_path(&self) -> bool {
        self.backdrop.is_none() && self.blend_mode == BlendMode::Normal
    }
}

/// Run `f` on the shape of the 8-bit `coverage` at the sample depth `S`:
/// `coverage` itself for 8-bit samples, each value widened exactly
/// ([`Sample::from_u8`]) for wider ones.
///
/// Coverage (antialiasing, glyph and hairline masks) is 8-bit at every
/// sample depth; a shape is a sample, since a group's alpha plane is the
/// shape of its composite.
pub fn with_shape<S: Sample, R>(coverage: &[u8], f: impl FnOnce(&[S]) -> R) -> R {
    thread_local! {
        static WIDENED: Scratch = const { Scratch::new() };
    }
    if let Some(shape) = S::from_u8_slice(coverage) {
        return f(shape);
    }
    WIDENED.with(|scratch| {
        scratch.with::<S, R>(coverage.len(), |shape| {
            for (s, &c) in shape.iter_mut().zip(coverage) {
                *s = S::from_u8(c);
            }
            f(shape)
        })
    })
}

/// Select and run the appropriate pipeline variant for a horizontal span.
///
/// - `pipe`: compositing parameters for this paint operation.
/// - `src`: source colour (solid or pattern).
/// - `dst_pixels`: the destination row's samples for `x0..=x1`.
/// - `dst_alpha`: the destination row's alpha for `x0..=x1`, or `None`
///   for a bitmap without an alpha plane, which is opaque.
/// - `shape`: per-pixel shape of `x0..=x1`, or `None` for full coverage.
/// - `x0`, `x1`: inclusive device column range of the span.
/// - `y`: device row, at which a pattern source is evaluated.
///
/// When `pipe.overprint` masks out channels, under `Normal` those channels
/// of every pixel keep their premultiplied destination value (their value
/// exactly where the destination is opaque) and the others are composited
/// as usual; under another blend mode each unpainted channel composites
/// with the colour the blend function reads there as its source (see
/// `render_blended_unpainted`).
///
/// A pattern source that [may leave pixels unpainted](Pattern::may_leave_unpainted)
/// is composited over its painted pixels only.  Under an overprint mask, a
/// pattern source with a [per-pixel record of its channels](Pattern::fill_span_channels)
/// paints each pixel through the channels recorded there within the mask.
///
/// # Panics
///
/// Panics if `x0 > x1`, if `P::NCOMPS == 0` (`Mono1` is not composited by
/// the pipe), or if `dst_pixels`, `dst_alpha` or `shape` does not hold
/// exactly the span's `x1 − x0 + 1` pixels.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination row, its alpha, the shape and the span position are independent inputs"
)]
pub fn render_span<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    mut dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    assert!(x0 <= x1, "render_span: x0={x0} > x1={x1}");
    assert!(
        P::NCOMPS > 0,
        "render_span: Mono1 is not composited by the pipe"
    );
    let count = usize::try_from(i64::from(x1) - i64::from(x0) + 1)
        .expect("render_span: span length fits usize");
    assert_eq!(
        dst_pixels.len(),
        count * P::NCOMPS,
        "render_span: dst_pixels holds {} samples for a {count}-pixel span",
        dst_pixels.len()
    );
    assert!(
        dst_alpha.as_ref().is_none_or(|a| a.len() == count)
            && shape.is_none_or(|s| s.len() == count),
        "render_span: alpha or shape length differs from the {count}-pixel span"
    );

    debug_assert!(
        pipe.backdrop.is_none() || pipe.overprint.is_none() || dst_alpha.is_some(),
        "render_span: a masked paint inside a non-isolated group needs the group's alpha plane"
    );
    if let (Some(mask), PipeSrc::Pattern(pattern)) = (pipe.overprint, src) {
        let split = SPAN_CHANNELS.with(|scratch| {
            scratch.with::<u8, bool>(count, |channels| {
                if !pattern.fill_span_channels(y, x0, x1, channels) {
                    return false;
                }
                render_channel_runs::<P>(
                    pipe,
                    (mask, channels),
                    src,
                    Span {
                        dst: dst_pixels,
                        alpha: dst_alpha.as_deref_mut(),
                        shape,
                    },
                    (x0, y),
                );
                true
            })
        });
        if split {
            return;
        }
    }
    render_masked::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y);
}

/// The channels painted at a pixel by a paint through `mask` of a pattern
/// that recorded `recorded` there (see [`Pattern::fill_span_channels`]).
const fn pixel_channels(mask: u8, recorded: u8) -> u8 {
    if recorded == 0 { mask } else { recorded & mask }
}

/// Composite a span of a pattern with a per-pixel record of its channels,
/// `channels`, through `mask`: each run of pixels painting the same
/// channels (see [`pixel_channels`]) through a pipe of that mask, and a
/// run painting every channel of `P` as an unmasked paint, whose `Normal`
/// composite consults no backdrop.
fn render_channel_runs<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    (mask, channels): (ChannelMask, &[u8]),
    src: &PipeSrc<'_, P::Sample>,
    span: Span<'_, P::Sample>,
    (x0, y): (i32, i32),
) {
    let Span {
        dst,
        alpha: mut dst_alpha,
        shape,
    } = span;
    let ncomps = P::NCOMPS;
    let full = u8::MAX >> (8 - ncomps);
    let count = channels.len();
    let mut a = 0;
    while a < count {
        let run = pixel_channels(mask.bits(), channels[a]);
        let b = channels[a..]
            .iter()
            .position(|&c| pixel_channels(mask.bits(), c) != run)
            .map_or(count, |n| a + n);
        let whole = run & full == full;
        let run_pipe = PipeState {
            overprint: (!whole).then_some(ChannelMask::new(run)),
            backdrop: if whole && pipe.blend_mode == BlendMode::Normal {
                None
            } else {
                pipe.backdrop
            },
            ..*pipe
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "a < b ≤ count = x1 − x0 + 1, which fits in i32"
        )]
        let (run_x0, run_x1) = (x0 + a as i32, x0 + b as i32 - 1);
        render_masked::<P>(
            &run_pipe,
            src,
            &mut dst[a * ncomps..b * ncomps],
            dst_alpha.as_deref_mut().map(|al| &mut al[a..b]),
            shape.map(|s| &s[a..b]),
            run_x0,
            run_x1,
            y,
        );
        a = b;
    }
}

/// Composite a validated span through the pipe's overprint mask (see
/// [`render_span`]).
#[expect(
    clippy::too_many_arguments,
    reason = "forwards the render_span arguments"
)]
fn render_masked<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    mut dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    // A mask covering every channel of `P` writes the whole pixel.
    let full = u8::MAX >> (8 - P::NCOMPS);
    match pipe.overprint {
        Some(mask) if mask.bits() & full != full && pipe.blend_mode != BlendMode::Normal => {
            render_blended_unpainted::<P>(
                pipe,
                mask,
                src,
                Span {
                    dst: dst_pixels,
                    alpha: dst_alpha,
                    shape,
                },
                (x0, x1, y),
            );
        }
        Some(mask) if mask.bits() & full != full => {
            let saved_len = dst_pixels.len() + dst_alpha.as_ref().map_or(0, |a| a.len());
            MASKED_SPAN.with(|scratch| {
                scratch.with::<P::Sample, ()>(saved_len, |saved| {
                    let (old_colour, old_alpha) = saved.split_at_mut(dst_pixels.len());
                    old_colour.copy_from_slice(dst_pixels);
                    if let Some(a) = dst_alpha.as_deref() {
                        old_alpha.copy_from_slice(a);
                    }
                    render_unmasked::<P>(
                        pipe,
                        src,
                        dst_pixels,
                        dst_alpha.as_deref_mut(),
                        shape,
                        x0,
                        x1,
                        y,
                    );
                    restore_unpainted::<P>(
                        pipe,
                        mask,
                        &SpanBefore {
                            colour: old_colour,
                            alpha: dst_alpha.as_deref().map(|new| (&*old_alpha, new)),
                        },
                        dst_pixels,
                        (x0..=x1, y),
                    );
                });
            });
        }
        _ => render_unmasked::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y),
    }
}

/// Composite a span under a blend mode other than `Normal` whose paint
/// writes only the channels of `mask` (ISO 32000-1 §11.7.4.3).
///
/// The object is painted with `CompatibleOverprint` in an implicit
/// non-isolated, non-knockout group, whose result is composited under the
/// blend mode.  The object alone in such a group leaves the group's colour
/// equal to the source in the channels it paints and to the backdrop the
/// blend function sees in the others; the group's alpha is the object's.
/// The span is therefore composited from a source whose unpainted channels
/// hold, at each pixel, the colour the blend function reads there: the
/// destination, or inside a non-isolated group the group's content over its
/// backdrop.  A white-preserving mode leaves an unpainted channel of tint 0
/// at tint 0.
///
/// The pipe's transfer applies to the substituted channels too; the ink
/// canvas, the only one painted through a mask, composites under the
/// identity transfer.
fn render_blended_unpainted<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    mask: ChannelMask,
    src: &PipeSrc<'_, P::Sample>,
    span: Span<'_, P::Sample>,
    (x0, x1, y): (i32, i32, i32),
) {
    let Span { dst, alpha, shape } = span;
    UNPAINTED_SOURCE.with(|scratch| {
        scratch.with::<P::Sample, ()>(dst.len(), |seen| {
            blended_backdrop::<P>(pipe, dst, alpha.as_deref(), seen, (x0, y));
            let source = Unpainted {
                source: src,
                mask,
                seen,
                x0,
                ncomps: P::NCOMPS,
            };
            render_unmasked::<P>(
                pipe,
                &PipeSrc::Pattern(&source),
                dst,
                alpha,
                shape,
                x0,
                x1,
                y,
            );
        });
    });
}

/// Write into `out` the colour the blend function reads at each pixel of
/// the span `dst` (alpha `alpha`, `None` for opaque) starting at device
/// pixel `(x0, y)`: the destination's, or with a group backdrop the
/// destination composited over it, the backdrop being transparent outside
/// its rectangle.
fn blended_backdrop<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    dst: &[P::Sample],
    alpha: Option<&[P::Sample]>,
    out: &mut [P::Sample],
    (x0, y): (i32, i32),
) {
    let ncomps = P::NCOMPS;
    let Some(backdrop) = pipe.backdrop.as_ref() else {
        out.copy_from_slice(dst);
        return;
    };
    for (i, ((x, px), seen)) in (x0..)
        .zip(dst.chunks_exact(ncomps))
        .zip(out.chunks_exact_mut(ncomps))
        .enumerate()
    {
        let a_dst = alpha.map_or(P::Sample::MAX, |a| a[i]);
        match backdrop.pixel(x, y, ncomps) {
            Some((c_0, a_0)) => {
                let _ = general::union_colour(px, a_dst.wide(), c_0, a_0.wide(), seen);
            }
            // Over a transparent backdrop the union is the destination.
            None => seen.copy_from_slice(px),
        }
    }
}

/// The source of a masked paint under a blend mode other than `Normal`:
/// `source` in the channels of `mask`, and in the others the colour
/// `seen` the blend function reads at the pixel, `ncomps` samples a pixel
/// over the span starting at device column `x0`.
struct Unpainted<'a, S: Sample> {
    source: &'a PipeSrc<'a, S>,
    mask: ChannelMask,
    seen: &'a [S],
    x0: i32,
    ncomps: usize,
}

impl<S: Sample> Unpainted<'_, S> {
    /// Overwrite the unpainted channels of `out`, the pixels from device
    /// column `x0`, with the colour the blend function reads there.
    fn substitute(&self, x0: i32, out: &mut [S]) {
        let skip = usize::try_from(x0 - self.x0)
            .expect("Unpainted: a span starting left of the painted span");
        let seen = &self.seen[skip * self.ncomps..];
        assert!(
            out.len() <= seen.len(),
            "Unpainted: a span reaching right of the painted span"
        );
        for (px, seen) in out
            .chunks_exact_mut(self.ncomps)
            .zip(seen.chunks_exact(self.ncomps))
        {
            for (channel, (c, &b)) in px.iter_mut().zip(seen).enumerate() {
                if !self.mask.paints(channel) {
                    *c = b;
                }
            }
        }
    }
}

impl<S: Sample> Pattern<S> for Unpainted<'_, S> {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [S]) {
        match self.source {
            PipeSrc::Solid(colour) => {
                for px in out.chunks_exact_mut(self.ncomps) {
                    px.copy_from_slice(colour);
                }
            }
            PipeSrc::Pattern(pattern) => pattern.fill_span(y, x0, x1, out),
        }
        self.substitute(x0, out);
    }

    fn may_leave_unpainted(&self) -> bool {
        matches!(self.source, PipeSrc::Pattern(p) if p.may_leave_unpainted())
    }

    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [S], coverage: &mut [S]) {
        match self.source {
            PipeSrc::Solid(_) => {
                self.fill_span(y, x0, x1, out);
                coverage.fill(S::MAX);
            }
            PipeSrc::Pattern(pattern) => {
                pattern.fill_span_coverage(y, x0, x1, out, coverage);
                self.substitute(x0, out);
            }
        }
    }
}

/// A span's state before a masked paint.
struct SpanBefore<'a, S> {
    /// Colour samples before the paint.
    colour: &'a [S],
    /// With an alpha plane: its samples before and after the paint.
    alpha: Option<(&'a [S], &'a [S])>,
}

/// Write the channels `mask` does not paint as the `CompatibleOverprint`
/// blend function composites them under `Normal` (ISO 32000-1 §11.7.4.3).
///
/// For an unpainted channel the blend function is `B(c_b, c_s) = c_b`, and
/// the source carries no ink in it (`c_s = 0`): the channel is either a
/// process component that nonzero overprint mode leaves unpainted because
/// it is zero (§8.6.7), or a colorant absent from the source colour space
/// (§11.7.4).  The general compositing formula (§11.3.6),
/// `c_r = (1 − α_s/α_r)·c_b + (α_s/α_r)·((1 − α_b)·c_s + α_b·B(c_b, c_s))`
/// with `α_r = α_s + α_b − α_s·α_b`, then reduces to `α_r·c_r = α_b·c_b`:
/// the paint leaves the channel's premultiplied value unchanged, whatever
/// the source alpha.  The stored value is [`solve_own_colour`]'s `c'`, with
/// the backdrop of `pipe.backdrop` at the pixel, or a transparent one
/// outside a non-isolated group and outside the backdrop's rectangle.
///
/// `old` is the span's state before the paint and `xs` its device columns.
/// The old samples stand where the paint leaves the alpha unchanged, which
/// includes every pixel of a bitmap without an alpha plane (opaque before
/// and after).
fn restore_unpainted<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    mask: ChannelMask,
    old: &SpanBefore<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    (xs, y): (RangeInclusive<i32>, i32),
) {
    let ncomps = P::NCOMPS;
    for (i, (x, (px, old_px))) in xs
        .zip(
            dst_pixels
                .chunks_exact_mut(ncomps)
                .zip(old.colour.chunks_exact(ncomps)),
        )
        .enumerate()
    {
        let solved = old.alpha.and_then(|(a_old, a_new)| {
            let backdrop = pipe.backdrop.as_ref().and_then(|bd| bd.pixel(x, y, ncomps));
            solve_own_colour(old_px, (a_old[i], a_new[i]), backdrop)
        });
        for (channel, (new, &old)) in px.iter_mut().zip(old_px).enumerate() {
            if !mask.paints(channel) {
                *new = solved.map_or(old, |s| s[channel]);
            }
        }
    }
}

/// The colour `c'` an unpainted channel stores at the pixel's new alpha
/// `a'`, given its old colour `c` at alpha `a`, inside a group over the
/// backdrop `(c_bd, a_bd)` (`None` for a transparent backdrop, `a_bd = 0`).
///
/// The premultiplied colour of the group over its backdrop is unchanged by
/// the paint (see [`restore_unpainted`]):
/// `a'·c' + (1 − a')·a_bd·c_bd = a·c + (1 − a)·a_bd·c_bd`, so, with alphas
/// in `0..=M` (`M = S::MAX`), `c' = (M·a·c + (a' − a)·a_bd·c_bd) / (M·a')`,
/// rounded to nearest once.  Compositing never lowers alpha (`a' ≥ a`), so
/// `c'` is a weighted mean of `c` and `c_bd` and lies in `0..=M`; the clamp
/// enforces that range for any input.  Outside a group (`a_bd = 0`) the
/// formula is `c' = a·c / a'`.
///
/// `None` where `a' = a`, for which `c' = c` exactly, and where `a' = 0`,
/// for which the pixel is transparent and its colour has no effect: the old
/// samples stand.
fn solve_own_colour<S: Sample>(
    old: &[S],
    (a_old, a_new): (S, S),
    backdrop: Option<(&[S], S)>,
) -> Option<[S; 8]> {
    if a_new == a_old || a_new == S::default() {
        return None;
    }
    let transparent = [S::default(); 8];
    let (c_bd, a_bd) = backdrop.unwrap_or_else(|| (&transparent, S::default()));
    let max = S::MAX.signed();
    let (a_old, a_new, a_bd) = (a_old.signed(), a_new.signed(), a_bd.signed());
    let den = max * a_new;
    let mut out = [S::default(); 8];
    for ((o, &c), &c_b) in out.iter_mut().zip(old).zip(c_bd) {
        let num = max * a_old * c.signed() + (a_new - a_old) * a_bd * c_b.signed();
        *o = S::from_signed((num + den / 2).div_euclid(den));
    }
    Some(out)
}

/// Composite a validated span, writing every colour channel.
#[expect(
    clippy::too_many_arguments,
    reason = "forwards the render_span arguments"
)]
fn render_unmasked<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    if let PipeSrc::Pattern(pat) = src
        && pat.may_leave_unpainted()
    {
        render_painted_runs::<P>(pipe, *pat, dst_pixels, dst_alpha, shape, x0, x1, y);
        return;
    }
    dispatch_span::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y);
}

/// Composite the pixels `x0..=x1` of row `y` of `bitmap` through
/// [`render_span`].
///
/// The span takes the row's alpha when the bitmap has an alpha plane, and
/// the paint is recorded in its painted channels when it tracks them
/// ([`mark_painted`]).
///
/// # Panics
///
/// Panics if `x0 > x1`, if a coordinate is negative, or if the span leaves
/// the bitmap.
pub fn render_bitmap_span<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    let (Ok(col0), Ok(col1), Ok(row)) =
        (usize::try_from(x0), usize::try_from(x1), u32::try_from(y))
    else {
        panic!("render_bitmap_span: span {x0}..={x1} on row {y} has a negative coordinate");
    };
    assert!(
        col0 <= col1 && col1 < bitmap.width as usize,
        "render_bitmap_span: span {x0}..={x1} outside a bitmap {} pixels wide",
        bitmap.width
    );
    let (pixels, alpha, painted) = bitmap.row_planes_mut(row);
    render_span::<P>(
        pipe,
        src,
        &mut pixels[col0 * P::NCOMPS..(col1 + 1) * P::NCOMPS],
        alpha.map(|a| &mut a[col0..=col1]),
        shape,
        x0,
        x1,
        y,
    );
    if let Some(painted) = painted {
        mark_painted::<P>(&mut painted[col0..=col1], pipe, src, shape, (x0, x1, y));
    }
}

/// Record a paint of `src` through `pipe` with `shape` in `painted`.
///
/// `painted` holds the painted-channels bytes of the pixels `x0..=x1` of
/// row `y` (see [`Bitmap::track_painted`]).  At each pixel the source
/// reaches with nonzero alpha (constant alpha, shape and the pattern's
/// coverage all nonzero), the channels it writes there are added: the
/// pipe's [`overprint`](PipeState::overprint) mask, narrowed by a
/// pattern's [per-pixel record](Pattern::fill_span_channels) as
/// [`render_span`] narrows it, and every channel for `None`.
///
/// # Panics
///
/// Panics if `x0 > x1`, or if `painted` or `shape` does not hold exactly
/// the span's `x1 − x0 + 1` pixels.
pub fn mark_painted<P: Pixel>(
    painted: &mut [u8],
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    shape: Option<&[P::Sample]>,
    (x0, x1, y): (i32, i32, i32),
) {
    assert!(x0 <= x1, "mark_painted: x0={x0} > x1={x1}");
    let count = painted.len();
    assert!(
        i64::try_from(count).is_ok_and(|n| n == i64::from(x1) - i64::from(x0) + 1)
            && shape.is_none_or(|s| s.len() == count),
        "mark_painted: the painted row or shape differs from the span"
    );
    let zero = P::Sample::default();
    if pipe.a_input == zero {
        return;
    }
    let mask = pipe.overprint.unwrap_or(ChannelMask::ALL).bits();
    let pattern = match src {
        PipeSrc::Pattern(pattern) => Some(*pattern),
        PipeSrc::Solid(_) => None,
    };
    let reached = |i: usize, coverage: Option<&[P::Sample]>| {
        shape.is_none_or(|s| s[i] != zero) && coverage.is_none_or(|c| c[i] != zero)
    };
    let mut record = |coverage: Option<&[P::Sample]>| {
        SPAN_CHANNELS.with(|scratch| {
            scratch.with::<u8, ()>(count, |channels| {
                let recorded = pipe.overprint.is_some()
                    && pattern.is_some_and(|p| p.fill_span_channels(y, x0, x1, channels));
                for (i, p) in painted.iter_mut().enumerate() {
                    if reached(i, coverage) {
                        *p |= if recorded {
                            pixel_channels(mask, channels[i])
                        } else {
                            mask
                        };
                    }
                }
            });
        });
    };
    match pattern {
        Some(p) if p.may_leave_unpainted() => {
            MARK_COLOUR.with(|colour| {
                colour.with::<P::Sample, ()>(count * P::NCOMPS, |colour| {
                    MARK_COVERAGE.with(|coverage| {
                        coverage.with::<P::Sample, ()>(count, |coverage| {
                            p.fill_span_coverage(y, x0, x1, colour, coverage);
                            record(Some(coverage));
                        });
                    });
                });
            });
        }
        _ => record(None),
    }
}

/// [`render_bitmap_span`] with the shape given as 8-bit coverage (see
/// [`with_shape`]).
///
/// # Panics
///
/// As [`render_bitmap_span`].
pub fn render_bitmap_span_coverage<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    coverage: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    match coverage {
        Some(c) => with_shape::<P::Sample, ()>(c, |shape| {
            render_bitmap_span::<P>(bitmap, pipe, src, Some(shape), x0, x1, y);
        }),
        None => render_bitmap_span::<P>(bitmap, pipe, src, None, x0, x1, y),
    }
}

/// Composite the covered runs of a span whose pattern may leave pixels
/// unpainted.
///
/// The pattern is evaluated once over the whole span.  Each maximal run of
/// pixels with non-zero coverage is then composited by [`dispatch_span`]
/// from the precomputed colours.  A fully covered run keeps the span's shape; a partially
/// covered run takes the product of shape and coverage as its shape.
/// Unpainted pixels are not touched.
#[expect(
    clippy::too_many_arguments,
    reason = "forwards the render_span arguments"
)]
fn render_painted_runs<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    pat: &dyn Pattern<P::Sample>,
    dst_pixels: &mut [P::Sample],
    dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    let ncomps = P::NCOMPS;
    let count = dst_pixels.len() / ncomps;
    COVERAGE_COLOUR.with(|colour| {
        colour.with::<P::Sample, ()>(count * ncomps, |colour| {
            COVERAGE_VALUES.with(|coverage| {
                coverage.with::<P::Sample, ()>(count, |coverage| {
                    COVERAGE_SHAPE.with(|combined| {
                        combined.with::<P::Sample, ()>(count, |combined| {
                            pat.fill_span_coverage(y, x0, x1, colour, coverage);
                            composite_painted_runs::<P>(
                                pipe,
                                &EvaluatedSpan { colour, coverage },
                                combined,
                                Span {
                                    dst: dst_pixels,
                                    alpha: dst_alpha,
                                    shape,
                                },
                                (x0, y),
                            );
                        });
                    });
                });
            });
        });
    });
}

/// A pattern's colour and coverage over a span.
struct EvaluatedSpan<'a, S> {
    colour: &'a [S],
    coverage: &'a [S],
}

/// The destination of a span: its colour, its alpha when the bitmap has an
/// alpha plane, and the shape painted on it.
struct Span<'a, S> {
    dst: &'a mut [S],
    alpha: Option<&'a mut [S]>,
    shape: Option<&'a [S]>,
}

/// The run loop of [`render_painted_runs`] over the evaluated span `src`
/// starting at device pixel `(x0, y)`, `combined` its scratch for a
/// partially covered run's shape.
fn composite_painted_runs<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &EvaluatedSpan<'_, P::Sample>,
    combined: &mut [P::Sample],
    span: Span<'_, P::Sample>,
    (x0, y): (i32, i32),
) {
    let EvaluatedSpan { colour, coverage } = *src;
    let Span {
        dst: dst_pixels,
        alpha: mut dst_alpha,
        shape,
    } = span;
    let ncomps = P::NCOMPS;
    let count = coverage.len();
    let mut start = 0;
    let zero = P::Sample::default();
    while let Some(a) = coverage[start..]
        .iter()
        .position(|&c| c != zero)
        .map(|i| start + i)
    {
        let b = coverage[a..]
            .iter()
            .position(|&c| c == zero)
            .map_or(count, |i| a + i);
        let cov = &coverage[a..b];
        let run_shape = if cov.iter().all(|&c| c == P::Sample::MAX) {
            shape.map(|s| &s[a..b])
        } else {
            let run = &mut combined[a..b];
            match shape {
                Some(s) => {
                    for ((o, &s), &c) in run.iter_mut().zip(&s[a..b]).zip(cov) {
                        *o = P::Sample::div_max(s.wide() * c.wide());
                    }
                }
                None => run.copy_from_slice(cov),
            }
            Some(&*run)
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "a < b ≤ count = x1 − x0 + 1, which fits in i32"
        )]
        let (run_x0, run_x1) = (x0 + a as i32, x0 + b as i32 - 1);
        dispatch_span::<P>(
            pipe,
            &PipeSrc::Pattern(&RowPattern {
                data: &colour[a * ncomps..b * ncomps],
            }),
            &mut dst_pixels[a * ncomps..b * ncomps],
            dst_alpha.as_deref_mut().map(|al| &mut al[a..b]),
            run_shape,
            run_x0,
            run_x1,
            y,
        );
        start = b;
    }
}

/// Run the pipeline variant that `pipe` and `shape` select for a span whose
/// every pixel is painted.
#[expect(
    clippy::too_many_arguments,
    reason = "forwards the render_span arguments"
)]
fn dispatch_span<P: Pixel>(
    pipe: &PipeState<'_, P::Sample>,
    src: &PipeSrc<'_, P::Sample>,
    dst_pixels: &mut [P::Sample],
    dst_alpha: Option<&mut [P::Sample]>,
    shape: Option<&[P::Sample]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    let uses_shape = shape.is_some();

    if pipe.no_transparency(uses_shape) && pipe.blend_mode == BlendMode::Normal {
        simple::render_span_simple::<P>(pipe, src, dst_pixels, dst_alpha, x0, x1, y);
    } else if uses_shape && pipe.use_aa_path() {
        aa::render_span_aa::<P>(
            pipe,
            src,
            dst_pixels,
            dst_alpha,
            shape.expect("use_aa_path requires shape"),
            x0,
            x1,
            y,
        );
    } else {
        general::render_span_general::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y);
    }
}

// ── Shared transfer helpers ───────────────────────────────────────────────────

/// Apply the per-channel transfer tables to `src` and write the result into `dst`.
///
/// Both slices must have the same length and match the pixel mode:
/// 1 sample → gray, 3 → RGB, 4 → CMYK/XBGR, 8 → `DeviceN`.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; 10-15 instructions, must inline across crate boundary"
)]
#[inline(always)]
pub(crate) fn apply_transfer_pixel<S: Sample>(pipe: &PipeState<'_, S>, src: &[S], dst: &mut [S]) {
    debug_assert_eq!(
        src.len(),
        dst.len(),
        "apply_transfer_pixel: length mismatch"
    );
    let t = &pipe.transfer;
    match src.len() {
        1 => dst[0] = S::lookup(t.gray, src[0]),
        3 => {
            dst[0] = S::lookup(t.rgb[0], src[0]);
            dst[1] = S::lookup(t.rgb[1], src[1]);
            dst[2] = S::lookup(t.rgb[2], src[2]);
        }
        4 => {
            dst[0] = S::lookup(t.cmyk[0], src[0]);
            dst[1] = S::lookup(t.cmyk[1], src[1]);
            dst[2] = S::lookup(t.cmyk[2], src[2]);
            dst[3] = S::lookup(t.cmyk[3], src[3]);
        }
        8 => {
            for ((&s, d), table) in src.iter().zip(dst.iter_mut()).zip(t.device_n) {
                *d = S::lookup(table, s);
            }
        }
        n => {
            debug_assert!(false, "apply_transfer_pixel: unexpected ncomps={n}");
            dst.copy_from_slice(src);
        }
    }
}

/// Apply transfer tables in-place to a single pixel slice.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; delegates to apply_transfer_pixel, must inline"
)]
#[inline(always)]
pub(crate) fn apply_transfer_in_place<S: Sample>(pipe: &PipeState<'_, S>, px: &mut [S]) {
    // Avoid allocating: for ncomps ≤ 8 copy into a stack buffer, apply, copy back.
    let n = px.len();
    debug_assert!(n <= 8, "apply_transfer_in_place: ncomps={n} > 8");
    let mut tmp = [S::default(); 8];
    tmp[..n].copy_from_slice(px);
    apply_transfer_pixel(pipe, &tmp[..n], px);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::make_pipe;
    use crate::types::BlendMode;
    use color::{DeviceN8, Rgb8};

    /// Paints `[x, 200, 50]` at every even `x` and nothing at odd `x`.
    struct EvenColumns;

    impl Pattern for EvenColumns {
        fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
            let mut coverage = vec![0; out.len() / 3];
            self.fill_span_coverage(y, x0, x1, out, &mut coverage);
        }

        fn may_leave_unpainted(&self) -> bool {
            true
        }

        fn fill_span_coverage(
            &self,
            _y: i32,
            x0: i32,
            x1: i32,
            out: &mut [u8],
            coverage: &mut [u8],
        ) {
            for ((x, px), c) in (x0..=x1).zip(out.as_chunks_mut::<3>().0).zip(coverage) {
                *c = if x % 2 == 0 { 255 } else { 0 };
                *px = [u8::try_from(x).expect("small x"), 200, 50];
            }
        }
    }

    /// Render `EvenColumns` over a backdrop of `[7, 7, 7]` / alpha 99 on
    /// pixels `3..=9` and return the pixels and alpha.
    fn render_even_columns(pipe: &PipeState<'_>, shape: Option<&[u8]>) -> (Vec<u8>, Vec<u8>) {
        let mut dst = vec![7u8; 7 * 3];
        let mut alpha = vec![99u8; 7];
        render_span::<Rgb8>(
            pipe,
            &PipeSrc::Pattern(&EvenColumns),
            &mut dst,
            Some(&mut alpha),
            shape,
            3,
            9,
            0,
        );
        (dst, alpha)
    }

    fn assert_odd_columns_untouched(dst: &[u8], alpha: &[u8]) {
        for (i, x) in (3..=9).enumerate() {
            if x % 2 == 1 {
                assert_eq!(&dst[i * 3..i * 3 + 3], &[7, 7, 7], "x={x} colour");
                assert_eq!(alpha[i], 99, "x={x} alpha");
            } else {
                assert_ne!(&dst[i * 3..i * 3 + 3], &[7, 7, 7], "x={x} painted");
            }
        }
    }

    #[test]
    fn unpainted_pixels_keep_the_backdrop_on_the_simple_path() {
        let (dst, alpha) = render_even_columns(&make_pipe(255, BlendMode::Normal), None);
        assert_odd_columns_untouched(&dst, &alpha);
        for (i, x) in (4u8..=8).step_by(2).enumerate() {
            let at = (2 * i + 1) * 3;
            assert_eq!(
                &dst[at..at + 3],
                &[x, 200, 50],
                "x={x} carries its own colour"
            );
        }
    }

    #[test]
    fn unpainted_pixels_keep_the_backdrop_on_the_aa_path() {
        let shape = [255u8, 128, 255, 128, 255, 128, 255];
        let (dst, alpha) = render_even_columns(&make_pipe(200, BlendMode::Normal), Some(&shape));
        assert_odd_columns_untouched(&dst, &alpha);
    }

    #[test]
    fn unpainted_pixels_keep_the_backdrop_on_the_general_path() {
        let (dst, alpha) = render_even_columns(&make_pipe(255, BlendMode::Difference), None);
        assert_odd_columns_untouched(&dst, &alpha);
    }

    /// White at coverage `[0, 64, 128, 255]`, repeating with `x`.
    struct RampCoverage;

    impl Pattern for RampCoverage {
        fn fill_span(&self, _y: i32, _x0: i32, _x1: i32, out: &mut [u8]) {
            out.fill(255);
        }

        fn may_leave_unpainted(&self) -> bool {
            true
        }

        fn fill_span_coverage(
            &self,
            _y: i32,
            x0: i32,
            x1: i32,
            out: &mut [u8],
            coverage: &mut [u8],
        ) {
            out.fill(255);
            for (x, c) in (x0..=x1).zip(coverage) {
                *c = [0, 64, 128, 255][usize::try_from(x % 4).expect("x ≥ 0")];
            }
        }
    }

    /// Render `RampCoverage` over opaque black on pixels `0..=7` and return
    /// the red channel.
    fn render_ramp_coverage(pipe: &PipeState<'_>, shape: Option<&[u8]>) -> Vec<u8> {
        let mut dst = vec![0u8; 8 * 3];
        let mut alpha = vec![255u8; 8];
        render_span::<Rgb8>(
            pipe,
            &PipeSrc::Pattern(&RampCoverage),
            &mut dst,
            Some(&mut alpha),
            shape,
            0,
            7,
            0,
        );
        dst.iter().step_by(3).copied().collect()
    }

    #[test]
    fn partial_coverage_scales_the_source_alpha() {
        let red = render_ramp_coverage(&make_pipe(255, BlendMode::Normal), None);
        assert_eq!(red, [0, 64, 128, 255, 0, 64, 128, 255]);
    }

    #[test]
    fn partial_coverage_multiplies_the_shape() {
        let shape = [128u8; 8];
        let red = render_ramp_coverage(&make_pipe(255, BlendMode::Normal), Some(&shape));
        for (x, (&r, want)) in red.iter().zip([0u8, 32, 64, 128].repeat(2)).enumerate() {
            assert!(r.abs_diff(want) <= 1, "x={x}: {r} vs {want}");
        }
    }

    #[test]
    fn partial_coverage_composes_with_the_general_path() {
        // White screened over black is white, weighted by the coverage.
        let red = render_ramp_coverage(&make_pipe(255, BlendMode::Screen), None);
        for (x, (&r, want)) in red.iter().zip([0u8, 64, 128, 255].repeat(2)).enumerate() {
            assert!(r.abs_diff(want) <= 1, "x={x}: {r} vs {want}");
        }
    }

    #[test]
    fn no_transparency_opaque_normal() {
        let pipe = make_pipe(255, BlendMode::Normal);
        assert!(pipe.no_transparency(false));
        assert!(!pipe.no_transparency(true)); // shape disables it
    }

    #[test]
    fn no_transparency_alpha_less_than_255() {
        let pipe = make_pipe(200, BlendMode::Normal);
        assert!(!pipe.no_transparency(false));
    }

    #[test]
    fn use_aa_path_for_translucent_normal() {
        let pipe = make_pipe(200, BlendMode::Normal);
        assert!(pipe.use_aa_path());
    }

    #[test]
    fn use_aa_path_false_for_blend_mode() {
        let pipe = make_pipe(200, BlendMode::Multiply);
        assert!(!pipe.use_aa_path());
    }

    #[test]
    fn render_span_simple_solid_opaque() {
        // Simple path: a_input=255, Normal, no shape.
        let pipe = make_pipe(255, BlendMode::Normal);
        let src_color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&src_color);

        let mut dst = vec![0u8; 3 * 4]; // 4 pixels
        let mut alpha = vec![0u8; 4];
        render_span::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), None, 0, 3, 0);

        // All pixels should be set to src color.
        for i in 0..4 {
            assert_eq!(dst[i * 3], 200, "pixel {i} R");
            assert_eq!(dst[i * 3 + 1], 100, "pixel {i} G");
            assert_eq!(dst[i * 3 + 2], 50, "pixel {i} B");
            assert_eq!(alpha[i], 255, "pixel {i} alpha");
        }
    }

    #[test]
    fn render_span_aa_blends_with_dst() {
        // AA path: shape=128 (~50% coverage), src=255 white, dst=0 black (fully opaque).
        let pipe = make_pipe(255, BlendMode::Normal);
        let src_color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&src_color);
        let shape = vec![128u8; 4];

        let mut dst = vec![0u8; 3 * 4];
        let mut alpha = vec![255u8; 4]; // fully opaque dst so blending is visible
        render_span::<Rgb8>(
            &pipe,
            &src,
            &mut dst,
            Some(&mut alpha),
            Some(&shape),
            0,
            3,
            0,
        );

        // a_src = div255(255*128) ≈ 128; a_dst=255; blends white over black at ~50%.
        for i in 0..4 {
            let v = dst[i * 3];
            assert!(v > 100 && v < 160, "pixel {i} R={v} expected ~128");
        }
    }

    const OLD: [u8; 8] = [200, 0, 0, 0, 0, 0, 0, 0];
    const INK_K: [u8; 8] = [0, 0, 0, 255, 0, 0, 0, 0];

    fn overprint_pipe(a_input: u8, overprint: Option<ChannelMask>) -> PipeState<'static> {
        PipeState {
            overprint,
            ..make_pipe(a_input, BlendMode::Normal)
        }
    }

    /// Paint `INK_K` over four `OLD` pixels, returning the pixels.
    fn paint_devicen(pipe: &PipeState<'_>, shape: Option<&[u8]>) -> Vec<u8> {
        let mut dst = OLD.repeat(4);
        render_span::<DeviceN8>(
            pipe,
            &PipeSrc::Solid(&INK_K),
            &mut dst,
            None,
            shape,
            0,
            3,
            0,
        );
        dst
    }

    /// Solid `INK_K` with the channels `channels` recorded at the pixels
    /// from column 0.
    struct RecordedK {
        channels: [u8; 4],
    }

    impl Pattern for RecordedK {
        fn fill_span(&self, _y: i32, _x0: i32, _x1: i32, out: &mut [u8]) {
            for px in out.as_chunks_mut::<8>().0 {
                *px = INK_K;
            }
        }

        fn fill_span_channels(&self, _y: i32, x0: i32, x1: i32, channels: &mut [u8]) -> bool {
            let (a, b) = (usize::try_from(x0).unwrap(), usize::try_from(x1).unwrap());
            channels.copy_from_slice(&self.channels[a..=b]);
            true
        }
    }

    /// A pattern's per-pixel record narrows the pipe's mask at each pixel,
    /// a pixel recorded 0 takes the mask itself, and the record reaches a
    /// bitmap's painted channels; without a mask the record is not read.
    #[test]
    fn recorded_channels_narrow_the_mask_per_pixel() {
        let pattern = RecordedK {
            channels: [0b1000, 0b1001, 0, 0b1000],
        };
        let paint = |overprint| {
            let mut bitmap = Bitmap::<DeviceN8>::new(4, 1, 1, false);
            bitmap.row_bytes_mut(0).copy_from_slice(&OLD.repeat(4));
            bitmap.track_painted().unwrap();
            let pipe = overprint_pipe(255, overprint);
            render_bitmap_span::<DeviceN8>(
                &mut bitmap,
                &pipe,
                &PipeSrc::Pattern(&pattern),
                None,
                0,
                3,
                0,
            );
            (
                bitmap.row_bytes(0).to_vec(),
                bitmap.painted_row(0).unwrap().to_vec(),
            )
        };
        let (dst, painted) = paint(Some(ChannelMask::new(0b1001)));
        let px: Vec<_> = dst
            .as_chunks::<8>()
            .0
            .iter()
            .map(|p| p[..4].to_vec())
            .collect();
        assert_eq!(
            px,
            [
                [200, 0, 0, 255],
                [0, 0, 0, 255],
                [0, 0, 0, 255],
                [200, 0, 0, 255]
            ]
        );
        assert_eq!(painted, [0b1000, 0b1001, 0b1001, 0b1000]);
        let (dst, painted) = paint(None);
        assert!(dst.as_chunks::<8>().0.iter().all(|p| *p == INK_K));
        assert_eq!(painted, [u8::MAX; 4]);
    }

    #[test]
    fn overprint_mask_keeps_unpainted_channels() {
        let pipe = overprint_pipe(255, Some(ChannelMask::new(0b1000)));
        let dst = paint_devicen(&pipe, None);
        for px in dst.as_chunks::<8>().0 {
            assert_eq!(*px, [200, 0, 0, 255, 0, 0, 0, 0]);
        }
    }

    #[test]
    fn overprint_mask_with_alpha_blends_only_painted() {
        let pipe = overprint_pipe(128, Some(ChannelMask::new(0b1000)));
        let dst = paint_devicen(&pipe, None);
        for px in dst.as_chunks::<8>().0 {
            assert_eq!(px[0], 200, "unpainted C");
            assert!(px[3].abs_diff(128) <= 1, "K = {}", px[3]);
        }
    }

    #[test]
    fn overprint_mask_with_aa_shape() {
        let pipe = overprint_pipe(255, Some(ChannelMask::new(0b1000)));
        let dst = paint_devicen(&pipe, Some(&[255, 128, 0, 255]));
        assert_eq!(dst[16..24], OLD, "uncovered pixel unchanged");
        assert_eq!(dst[8], 200, "half-covered pixel keeps C");
        assert!(dst[8 + 3] > 0 && dst[8 + 3] < 255, "K = {}", dst[8 + 3]);
    }

    #[test]
    fn overprint_none_equals_all() {
        let shape = [255, 128, 0, 77];
        let cases: [(u8, BlendMode, Option<&[u8]>); 3] = [
            (255, BlendMode::Normal, None),
            (200, BlendMode::Normal, Some(&shape)),
            (200, BlendMode::Multiply, Some(&shape)),
        ];
        for (a, mode, shape) in cases {
            let render = |overprint| {
                let pipe = PipeState {
                    overprint,
                    ..make_pipe(a, mode)
                };
                paint_devicen(&pipe, shape)
            };
            assert_eq!(render(None), render(Some(ChannelMask::ALL)), "{a} {mode:?}");
        }
    }

    #[test]
    fn channel_mask_paints_reports_bits() {
        let m = ChannelMask::new(0b1010);
        assert!(!m.paints(0) && m.paints(1) && !m.paints(2) && m.paints(3));
        assert!(!m.paints(8));
        assert_eq!(m.bits(), 0b1010);
        assert!((0..8).all(|c| ChannelMask::ALL.paints(c)));
    }

    /// Paint K through mask `0b1000` into one `DeviceN8` pixel of group-own colour
    /// `own` / alpha `a_dst`, over an opaque cyan backdrop when `with_backdrop`.
    fn group_paint(a_dst: u8, a_input: u8, with_backdrop: bool) -> ([u8; 8], u8) {
        let colour = [200u8, 0, 0, 0, 0, 0, 0, 0];
        let backdrop = GroupBackdrop {
            colour: &colour,
            alpha: None,
            x0: 0,
            y0: 0,
            width: 1,
            height: 1,
        };
        let mut dst = [100u8, 0, 0, 0, 0, 0, 0, 0];
        let mut alpha = [a_dst];
        masked_paint(
            a_input,
            with_backdrop.then_some(backdrop),
            None,
            &mut dst,
            &mut alpha,
        );
        (dst, alpha[0])
    }

    #[test]
    fn overprint_in_an_empty_group_keeps_the_backdrop_colour() {
        let (dst, alpha) = group_paint(0, 255, true);
        assert_eq!(dst[0], 200, "C composites to the backdrop's cyan");
        assert_eq!(dst[3], 255);
        assert_eq!(alpha, 255);
    }

    #[test]
    fn overprint_in_a_partial_group_preserves_the_composited_colour() {
        // Own C = 100 at alpha 128 over C = 200: union 149.  After a paint
        // that lifts alpha to 192, own C = (149*255 - 63*200) / 192 = 132.
        let (dst, alpha) = group_paint(128, 128, true);
        assert!(alpha.abs_diff(192) <= 1, "alpha = {alpha}");
        assert!(dst[0].abs_diff(132) <= 1, "own C = {}", dst[0]);
        let a = u32::from(alpha);
        let composited = (a * u32::from(dst[0]) + (255 - a) * 200) / 255;
        assert!(composited.abs_diff(149) <= 1, "composited C = {composited}");
    }

    /// Paint `INK_K` through mask `0b1000` with source alpha `a_input` and
    /// `shape` over a span of `DeviceN8` pixels of colour `dst` and alpha
    /// `alpha`, inside a group over `backdrop` when given.
    fn masked_paint(
        a_input: u8,
        backdrop: Option<GroupBackdrop<'_>>,
        shape: Option<&[u8]>,
        dst: &mut [u8],
        alpha: &mut [u8],
    ) {
        let pipe = PipeState {
            backdrop,
            overprint: Some(ChannelMask::new(0b1000)),
            ..make_pipe(a_input, BlendMode::Normal)
        };
        let x1 = i32::try_from(alpha.len()).expect("test span fits i32") - 1;
        render_span::<DeviceN8>(
            &pipe,
            &PipeSrc::Solid(&INK_K),
            dst,
            Some(alpha),
            shape,
            0,
            x1,
            0,
        );
    }

    #[test]
    fn overprint_over_a_translucent_backdrop_preserves_the_premultiplied_colour() {
        // Own C = 100 at alpha 128 over C = 200 at alpha 128: union alpha
        // 192, premultiplied union C = (128*255*100 + 127*128*200) / 255^2
        // = 100.2.  The paint lifts the own alpha to 192 and the union alpha
        // to 224; own C = (255*128*100 + 64*128*200) / (255*192) = 100.2.
        let bd_colour = [200u8, 0, 0, 0, 0, 0, 0, 0];
        let backdrop = GroupBackdrop {
            colour: &bd_colour,
            alpha: Some(&[128]),
            x0: 0,
            y0: 0,
            width: 1,
            height: 1,
        };
        let mut dst = [100u8, 0, 0, 0, 0, 0, 0, 0];
        let mut alpha = [128u8];
        masked_paint(128, Some(backdrop), None, &mut dst, &mut alpha);
        assert_eq!(alpha[0], 192);
        assert_eq!(dst[0], 100, "own C");
        assert_eq!(dst[3], 170, "K = 128*255 / 192");
    }

    #[test]
    fn untouched_translucent_pixels_keep_their_colour_across_masked_spans() {
        // Pixel 1 has shape 0 in every span: its alpha never changes, so
        // its own colour must not drift however many spans pass over it.
        let bd_colour = [200u8, 0, 0, 0, 0, 0, 0, 0].repeat(2);
        let backdrop = GroupBackdrop {
            colour: &bd_colour,
            alpha: None,
            x0: 0,
            y0: 0,
            width: 2,
            height: 1,
        };
        let mut dst = [[0u8; 8], [100, 0, 0, 0, 0, 0, 0, 0]].concat();
        let mut alpha = [0u8, 128];
        for _ in 0..200 {
            masked_paint(255, Some(backdrop), Some(&[255, 0]), &mut dst, &mut alpha);
        }
        assert_eq!(dst[8..], [100, 0, 0, 0, 0, 0, 0, 0], "untouched pixel");
        assert_eq!(alpha[1], 128);
        assert_eq!(dst[..8], [200, 0, 0, 255, 0, 0, 0, 0], "painted pixel");
    }

    #[test]
    fn overprint_without_a_backdrop_adds_no_ink_to_a_transparent_pixel() {
        let (dst, alpha) = group_paint(0, 255, false);
        assert_eq!(dst[0], 0, "C takes the source's absent ink");
        assert_eq!(dst[3], 255);
        assert_eq!(alpha, 255, "alpha takes the painted shape");
    }

    #[test]
    fn overprint_without_a_backdrop_keeps_the_premultiplied_colour() {
        // Own C = 200 at alpha 128 under an opaque K paint: C·alpha is
        // kept, so C = 200*128/255 = 100.4 at alpha 255.
        let mut dst = [200u8, 0, 0, 0, 0, 0, 0, 0];
        let mut alpha = [128u8];
        masked_paint(255, None, None, &mut dst, &mut alpha);
        assert_eq!(alpha[0], 255);
        assert_eq!(dst, [100, 0, 0, 255, 0, 0, 0, 0]);
    }

    #[test]
    fn a_mask_covering_the_pixel_width_writes_every_channel() {
        let colour = [9u8, 9, 9];
        let paint = |overprint| {
            let pipe = PipeState {
                overprint,
                ..make_pipe(255, BlendMode::Normal)
            };
            let mut dst = [1u8, 2, 3];
            render_span::<Rgb8>(
                &pipe,
                &PipeSrc::Solid(&colour),
                &mut dst,
                None,
                None,
                0,
                0,
                0,
            );
            dst
        };
        assert_eq!(paint(Some(ChannelMask::new(0b111))), [9, 9, 9]);
        assert_eq!(paint(Some(ChannelMask::new(0b101))), [9, 2, 9]);
    }

    /// Paint `src` through `overprint` under `mode` at constant alpha
    /// `a_input` over the `DeviceN8` pixels `dst` of alpha `alpha` (`None`
    /// for opaque), with the group backdrop `backdrop`.
    fn blended_paint(
        mode: BlendMode,
        (a_input, overprint): (u8, Option<ChannelMask>),
        src: &PipeSrc<'_>,
        backdrop: Option<GroupBackdrop<'_>>,
        dst: &mut [u8],
        alpha: Option<&mut [u8]>,
    ) {
        let pipe = PipeState {
            backdrop,
            overprint,
            ..make_pipe(a_input, mode)
        };
        let x1 = i32::try_from(dst.len() / 8).expect("test span fits i32") - 1;
        render_span::<DeviceN8>(&pipe, src, dst, alpha, None, 0, x1, 0);
    }

    /// A masked paint under a blend mode other than `Normal` composites
    /// its unpainted channels as an unmasked paint whose source holds
    /// the destination's colour there: C and the spot of tint 0 blend with
    /// themselves under `Screen`, at full and at half alpha.
    #[test]
    fn masked_blend_modes_blend_unpainted_channels_with_themselves() {
        let old = [128u8, 0, 0, 0, 0, 0, 0, 0];
        for a_input in [255, 128] {
            let mut masked = old;
            blended_paint(
                BlendMode::Screen,
                (a_input, Some(ChannelMask::new(0b1000))),
                &PipeSrc::Solid(&INK_K),
                None,
                &mut masked,
                None,
            );
            let mut unmasked = old;
            blended_paint(
                BlendMode::Screen,
                (a_input, None),
                &PipeSrc::Solid(&[128, 0, 0, 255, 0, 0, 0, 0]),
                None,
                &mut unmasked,
                None,
            );
            assert_eq!(masked, unmasked, "alpha {a_input}");
            assert!(
                masked[0] < 128,
                "alpha {a_input}: C screened to {}",
                masked[0]
            );
            assert_eq!(masked[4], 0, "alpha {a_input}: the spot stays blank");
        }
    }

    /// Inside a non-isolated group the unpainted channels of a masked
    /// blended paint take the group's content over its backdrop as their
    /// source; the spot pattern source is substituted per pixel too.
    #[test]
    fn masked_blend_modes_read_the_group_backdrop() {
        let colour = [200u8, 0, 0, 0, 0, 0, 0, 0].repeat(2);
        let backdrop = GroupBackdrop {
            colour: &colour,
            alpha: None,
            x0: 0,
            y0: 0,
            width: 2,
            height: 1,
        };
        let row = [INK_K, [0, 0, 0, 128, 0, 0, 0, 0]].concat();
        let pattern = RowPattern { data: &row };
        let mut masked = [0u8; 16];
        let mut masked_alpha = [0u8; 2];
        blended_paint(
            BlendMode::Multiply,
            (255, Some(ChannelMask::new(0b1000))),
            &PipeSrc::Pattern(&pattern),
            Some(backdrop),
            &mut masked,
            Some(&mut masked_alpha),
        );
        let substituted = [[200u8, 0, 0, 255, 0, 0, 0, 0], [200, 0, 0, 128, 0, 0, 0, 0]].concat();
        let pattern = RowPattern { data: &substituted };
        let mut unmasked = [0u8; 16];
        let mut unmasked_alpha = [0u8; 2];
        blended_paint(
            BlendMode::Multiply,
            (255, None),
            &PipeSrc::Pattern(&pattern),
            Some(backdrop),
            &mut unmasked,
            Some(&mut unmasked_alpha),
        );
        assert_eq!((masked, masked_alpha), (unmasked, unmasked_alpha));
        // Multiply of the backdrop's cyan with itself: 1 − (55/255)² of ink.
        assert!(masked[0] > 240, "C multiplied to {}", masked[0]);
    }

    /// A spot channel is blended by a white-preserving separable mode and
    /// painted as under `Normal` by `Difference`, `Exclusion` and the
    /// non-separable modes.
    #[test]
    fn spot_channels_blend_only_under_white_preserving_separable_modes() {
        let spot = |mode, tint| {
            let mut dst = [0u8, 0, 0, 0, 128, 0, 0, 0];
            let src = [0u8, 0, 0, 0, tint, 0, 0, 0];
            blended_paint(
                mode,
                (255, None),
                &PipeSrc::Solid(&src),
                None,
                &mut dst,
                None,
            );
            dst[4]
        };
        // 1 − (127/255)² of ink.
        assert!(spot(BlendMode::Multiply, 128).abs_diff(191) <= 1);
        assert_eq!(spot(BlendMode::Darken, 255), 255);
        for mode in [
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
        ] {
            assert_eq!(spot(mode, 255), 255, "{mode:?}");
            assert_eq!(spot(mode, 0), 0, "{mode:?}");
        }
    }
}
