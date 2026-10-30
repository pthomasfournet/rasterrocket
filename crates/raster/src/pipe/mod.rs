//! Compositing pipeline — replaces `SplashPipe` and its `pipeRun*` family.
//!
//! The C++ design uses a struct with a function-pointer field (`pipe->run`) selected at
//! `pipeInit` time.  Here each paint operation calls a span-level function that is
//! monomorphized over `P: Pixel` at compile time, eliminating per-pixel vtable overhead.
//!
//! # Three pipeline variants
//!
//! | Variant | When selected | C++ equivalent |
//! |---------|--------------|----------------|
//! | `simple` | `a_input == 255`, `BlendMode::Normal` | `pipeRunSimple*` |
//! | `aa`     | shape byte present, `BlendMode::Normal`, no group correction | `pipeRunAA*` |
//! | `general`| everything else — blend mode, non-isolated groups, overprint | `pipeRun` |

pub mod aa;
pub mod blend;
pub mod general;
pub mod simple;
pub mod transfer;

use std::cell::RefCell;

use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

pub use transfer::TransferSet;

/// Source colour for a single paint operation.
///
/// The source is resolved once per span and may be a flat colour (most common) or
/// a dynamic [`Pattern`] queried per-pixel.
pub enum PipeSrc<'a> {
    /// Flat colour: `ncomps` bytes, already in device space.
    Solid(&'a [u8]),
    /// Dynamic pattern source — `fill_span` is called per output row.
    Pattern(&'a dyn Pattern),
}

/// A source of per-pixel colour for use with the compositing pipeline.
///
/// Implementors return raw device-space byte sequences into the provided buffer.
/// The buffer length is `(x1 - x0 + 1) * ncomps` where `ncomps` is the pixel size
/// for the target bitmap mode.
pub trait Pattern: Send + Sync {
    /// Fill `out` with colour bytes for pixels `x0..=x1` on scanline `y`.
    ///
    /// `out.len()` is guaranteed to be `(x1 - x0 + 1) * ncomps`.
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]);

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
    /// alpha scaled by `c / 255`; an unpainted pixel (coverage 0) keeps its
    /// destination colour and alpha, exactly as a pixel outside the filled
    /// area does.  [`render_span`] evaluates such a pattern through
    /// [`Self::fill_span_coverage`] and composites only its covered runs; a
    /// pattern returning `false` covers every pixel of every span fully.
    fn may_leave_unpainted(&self) -> bool {
        false
    }

    /// Fill `out` as [`Self::fill_span`] does and set `coverage[i]` to the
    /// coverage of pixel `x0 + i`, from 0 (unpainted) to 255 (fully painted).
    ///
    /// `coverage.len()` is `x1 - x0 + 1`.  The colour bytes of an unpainted
    /// pixel are unspecified.  The default covers every pixel fully.
    fn fill_span_coverage(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], coverage: &mut [u8]) {
        self.fill_span(y, x0, x1, out);
        coverage.fill(255);
    }
}

/// A [`Pattern`] serving one precomputed span of colour bytes.
///
/// `fill_span` copies `data` into `out`, whose length must equal
/// `data.len()`: a `RowPattern` is valid only for the span it holds.
pub(crate) struct RowPattern<'a> {
    /// Colour bytes of the span, `P::BYTES` per pixel.
    pub(crate) data: &'a [u8],
}

impl Pattern for RowPattern<'_> {
    fn fill_span(&self, _y: i32, x0: i32, x1: i32, out: &mut [u8]) {
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

/// Per-thread scratch for the spans of a pattern that may leave pixels
/// unpainted: colour, coverage, and coverage combined with the span's shape.
/// Grow-never-shrink.
struct CoverageScratch {
    colour: Vec<u8>,
    coverage: Vec<u8>,
    shape: Vec<u8>,
}

thread_local! {
    static COVERAGE_SPAN: RefCell<CoverageScratch> = const {
        RefCell::new(CoverageScratch {
            colour: Vec::new(),
            coverage: Vec::new(),
            shape: Vec::new(),
        })
    };
}

/// Immutable parameters for one paint operation, built once per fill/stroke/glyph call.
///
/// `'bmp` borrows slices out of the destination bitmap's alpha plane and the
/// current graphics state's transfer tables.
#[derive(Copy, Clone, Debug)]
pub struct PipeState<'bmp> {
    /// Compositing blend mode.
    pub blend_mode: BlendMode,

    /// Source opacity, pre-scaled: `state.fill_alpha * 255.0` rounded to `u8`.
    /// For stroke operations the caller passes `stroke_alpha * 255.0`.
    pub a_input: u8,

    /// Overprint mask: bit `k` set means channel `k` is painted.
    /// `0xFFFF_FFFF` means all channels are painted (the default).
    pub overprint_mask: u32,

    /// If `true`, overprinted channels are additively blended (ink accumulation)
    /// rather than replaced.  Corresponds to `state.overprint_additive`.
    pub overprint_additive: bool,

    /// Transfer function tables for the current pixel mode.  Applied to the
    /// composited result before writing.
    pub transfer: TransferSet<'bmp>,

    /// The backdrop of the non-isolated group being painted into, consulted
    /// by the blend function; `None` outside such a group, and for
    /// `BlendMode::Normal`, which does not consult it.
    pub backdrop: Option<GroupBackdrop<'bmp>>,
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
pub struct GroupBackdrop<'a> {
    /// Colour bytes of the copied rectangle, `ncomps` per pixel, row-major.
    pub colour: &'a [u8],
    /// Alpha of each copied pixel; `None` for an opaque backdrop.
    pub alpha: Option<&'a [u8]>,
    /// Device column of the copied rectangle's top-left pixel.
    pub x0: i32,
    /// Device row of the copied rectangle's top-left pixel.
    pub y0: i32,
    /// Pixels per row of the copied rectangle.
    pub width: usize,
    /// Rows of the copied rectangle.
    pub height: usize,
}

impl GroupBackdrop<'_> {
    /// The colour bytes and alpha of the backdrop at device pixel `(x, y)`;
    /// `None` outside the copied rectangle, where the backdrop is
    /// transparent.
    #[must_use]
    pub fn pixel(&self, x: i32, y: i32, ncomps: usize) -> Option<(&[u8], u8)> {
        let col = usize::try_from(x.checked_sub(self.x0)?).ok()?;
        let row = usize::try_from(y.checked_sub(self.y0)?).ok()?;
        if col >= self.width || row >= self.height {
            return None;
        }
        let i = row * self.width + col;
        let alpha = self.alpha.map_or(255, |a| a[i]);
        Some((&self.colour[i * ncomps..(i + 1) * ncomps], alpha))
    }
}

impl PipeState<'_> {
    /// Returns `true` when the simple (no-transparency) fast path is applicable.
    ///
    /// Applicable when `a_input == 255`, there is no shape and no backdrop.
    ///
    /// Overprint must be excluded: the simple path overwrites `dst_pixels` before
    /// reading the original destination, making channel-selective restore impossible.
    #[must_use]
    pub const fn no_transparency(&self, uses_shape: bool) -> bool {
        self.a_input == 255
            && !uses_shape
            && self.backdrop.is_none()
            && self.overprint_mask == 0xFFFF_FFFF
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

/// Select and run the appropriate pipeline variant for a horizontal span.
///
/// - `pipe`: compositing parameters for this paint operation.
/// - `src`: source colour (solid or pattern).
/// - `dst_pixels`: raw pixel bytes for the destination row, starting at `x0`.
/// - `dst_alpha`: alpha plane bytes for the destination row, starting at `x0`.
///   `None` means the destination has no separate alpha (Mono1, or solid-color bitmaps
///   without transparency).
/// - `shape`: per-pixel AA shape byte, or `None` for the simple path.
/// - `screen`: optional halftone screen for Mono1 dithering.
/// - `x0`, `x1`: inclusive pixel coordinate range within the row.
/// - `y`: scanline y coordinate (used by pattern and halftone screen).
/// - `ncomps`: bytes per pixel (must match `P::BYTES`).
///
/// A pattern source that [may leave pixels unpainted](Pattern::may_leave_unpainted)
/// is composited over its painted pixels only.
///
/// # Panics
///
/// Panics in debug mode if `x0 > x1` or `P::BYTES == 0` (Mono1 must be handled
/// by the caller).  Also panics if `dst_pixels.len()` does not equal
/// `(x1 - x0 + 1) * P::BYTES`.
#[expect(
    clippy::too_many_arguments,
    reason = "render_span mirrors the C++ SplashPipe API; all arguments are necessary"
)]
pub fn render_span<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert!(x0 <= x1, "render_span: x0={x0} > x1={x1}");
    debug_assert!(P::BYTES > 0, "render_span: Mono1 must be handled by caller");

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    debug_assert_eq!(
        dst_pixels.len(),
        count * P::BYTES,
        "render_span: dst_pixels length mismatch"
    );

    if let PipeSrc::Pattern(pat) = src
        && pat.may_leave_unpainted()
    {
        render_painted_runs::<P>(pipe, *pat, dst_pixels, dst_alpha, shape, x0, x1, y);
        return;
    }
    dispatch_span::<P>(pipe, src, dst_pixels, dst_alpha, shape, x0, x1, y);
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
    pipe: &PipeState<'_>,
    pat: &dyn Pattern,
    dst_pixels: &mut [u8],
    mut dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
    x0: i32,
    x1: i32,
    y: i32,
) {
    let ncomps = P::BYTES;
    let count = dst_pixels.len() / ncomps;
    COVERAGE_SPAN.with(|cell| {
        let CoverageScratch {
            colour,
            coverage,
            shape: combined,
        } = &mut *cell.borrow_mut();
        colour.resize(count * ncomps, 0);
        coverage.resize(count, 0);
        pat.fill_span_coverage(y, x0, x1, colour, coverage);

        let mut start = 0;
        while let Some(a) = coverage[start..]
            .iter()
            .position(|&c| c != 0)
            .map(|i| start + i)
        {
            let b = coverage[a..]
                .iter()
                .position(|&c| c == 0)
                .map_or(count, |i| a + i);
            let cov = &coverage[a..b];
            let run_shape = if cov.iter().all(|&c| c == 255) {
                shape.map(|s| &s[a..b])
            } else {
                combined.clear();
                match shape {
                    Some(s) => combined.extend(
                        s[a..b]
                            .iter()
                            .zip(cov)
                            .map(|(&s, &c)| div255(u32::from(s) * u32::from(c))),
                    ),
                    None => combined.extend_from_slice(cov),
                }
                Some(combined.as_slice())
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
    });
}

/// Run the pipeline variant that `pipe` and `shape` select for a span whose
/// every pixel is painted.
#[expect(
    clippy::too_many_arguments,
    reason = "forwards the render_span arguments"
)]
fn dispatch_span<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: Option<&[u8]>,
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
//
// Identical transfer logic was duplicated across `aa`, `simple`, and `general`.
// These `pub(crate)` functions are the single canonical implementations.

/// Apply the per-channel transfer LUTs to `src` and write the result into `dst`.
///
/// Both slices must have the same length and match the pixel mode:
/// 1 byte → gray, 3 → RGB, 4 → CMYK/XBGR, 8 → `DeviceN`.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; 10-15 instructions, must inline across crate boundary"
)]
#[inline(always)]
pub(crate) fn apply_transfer_pixel(pipe: &PipeState<'_>, src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(
        src.len(),
        dst.len(),
        "apply_transfer_pixel: length mismatch"
    );
    let t = &pipe.transfer;
    match src.len() {
        1 => dst[0] = t.gray[src[0] as usize],
        3 => {
            dst[0] = t.rgb[0][src[0] as usize];
            dst[1] = t.rgb[1][src[1] as usize];
            dst[2] = t.rgb[2][src[2] as usize];
        }
        4 => {
            dst[0] = t.cmyk[0][src[0] as usize];
            dst[1] = t.cmyk[1][src[1] as usize];
            dst[2] = t.cmyk[2][src[2] as usize];
            dst[3] = t.cmyk[3][src[3] as usize];
        }
        8 => {
            for (i, (&s, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
                *d = t.device_n[i][s as usize];
            }
        }
        n => {
            debug_assert!(false, "apply_transfer_pixel: unexpected ncomps={n}");
            dst.copy_from_slice(src);
        }
    }
}

/// Apply transfer LUTs in-place to a single pixel slice.
#[expect(
    clippy::inline_always,
    reason = "called per-pixel in the innermost compositing loop; delegates to apply_transfer_pixel, must inline"
)]
#[inline(always)]
pub(crate) fn apply_transfer_in_place(pipe: &PipeState<'_>, px: &mut [u8]) {
    // Avoid allocating: for ncomps ≤ 8 copy into a stack buffer, apply, copy back.
    let n = px.len();
    debug_assert!(n <= 8, "apply_transfer_in_place: ncomps={n} > 8");
    let mut tmp = [0u8; 8];
    tmp[..n].copy_from_slice(px);
    apply_transfer_pixel(pipe, &tmp[..n], px);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::make_pipe;
    use crate::types::BlendMode;
    use color::Rgb8;

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

    #[test]
    fn no_transparency_false_with_overprint() {
        // Overprint must not use the simple path (dst channels already overwritten).
        let mut pipe = make_pipe(255, BlendMode::Normal);
        pipe.overprint_mask = 0x0000_0001; // only channel 0 painted
        assert!(
            !pipe.no_transparency(false),
            "overprint must route to general pipe"
        );
    }
}
