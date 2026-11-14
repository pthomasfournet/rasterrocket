//! The compositor at 16 bits per sample, checked against itself at 8.
//!
//! Each test runs one operation on 8-bit inputs and on the same inputs
//! widened to 16 bits ([`Sample::from_u8`]), and requires the 16-bit result
//! narrowed to 8 bits ([`Sample::to_u8`]) to lie within a stated tolerance
//! of the 8-bit result: within one level, the rounding of either result,
//! unless a test states otherwise.  A larger tolerance is the error of the
//! 8-bit evaluation, whose integer quotients truncate at a coarser step;
//! the 16-bit evaluation of the same formula is the more exact.

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::fill::fill;
use crate::pipe::{GroupBackdrop, PipeSrc, PipeState, TransferSet, blend, render_span};
use crate::shading::ColorRamp;
use crate::shading::axial::AxialPattern;
use crate::testutil::{identity_matrix, rect_path};
use crate::transparency::{GroupParams, begin_group, paint_group};
use crate::types::BlendMode;
use color::{Gray8, Gray16, Pixel, Rgb8, Rgb16, Sample};

/// A 64-bit linear congruential sequence (Knuth's MMIX constants).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }

    fn below(&mut self, n: u64) -> usize {
        usize::try_from(self.next() % n).expect("small")
    }
}

fn widen(v: &[u8]) -> Vec<u16> {
    v.iter().map(|&b| <u16 as Sample>::from_u8(b)).collect()
}

/// The largest difference between `narrow` narrowed to 8 bits and `wide8`.
fn worst(wide: &[u16], narrow: &[u8]) -> u8 {
    wide.iter()
        .zip(narrow)
        .map(|(&w, &n)| w.to_u8().abs_diff(n))
        .max()
        .unwrap_or(0)
}

fn pipe<S: Sample>(mode: BlendMode, a_input: S) -> PipeState<'static, S> {
    PipeState {
        blend_mode: mode,
        a_input,
        transfer: TransferSet::identity_rgb(),
        backdrop: None,
        overprint: None,
    }
}

/// One random span composited at both depths through [`render_span`]:
/// `(16-bit result, 8-bit result)` of colour, then of alpha.
/// A 16-bit result beside the 8-bit result of the same operation.
type AtBothDepths = (Vec<u16>, Vec<u8>);

fn span_at_both_depths<P8, P16>(
    rng: &mut Lcg,
    mode: BlendMode,
    with_backdrop: bool,
) -> (AtBothDepths, AtBothDepths)
where
    P8: Pixel<Sample = u8>,
    P16: Pixel<Sample = u16>,
{
    let n = P8::NCOMPS;
    let count = 1 + rng.below(24);
    let x1 = i32::try_from(count).expect("small") - 1;
    let dst: Vec<u8> = (0..count * n).map(|_| rng.byte()).collect();
    let alpha: Vec<u8> = (0..count).map(|_| rng.byte()).collect();
    let shape: Vec<u8> = (0..count).map(|_| rng.byte()).collect();
    let colour: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
    let bd_colour: Vec<u8> = (0..count * n).map(|_| rng.byte()).collect();
    let bd_alpha: Vec<u8> = (0..count).map(|_| rng.byte()).collect();
    let a_input = rng.byte();

    let (mut d8, mut a8) = (dst.clone(), alpha.clone());
    let mut p8 = pipe::<u8>(mode, a_input);
    let mut p16 = pipe::<u16>(mode, <u16 as Sample>::from_u8(a_input));
    let (colour16, alpha16) = (widen(&bd_colour), widen(&bd_alpha));
    if with_backdrop {
        p8.backdrop = Some(GroupBackdrop {
            colour: &bd_colour,
            alpha: Some(&bd_alpha),
            x0: 0,
            y0: 0,
            width: count,
            height: 1,
        });
        p16.backdrop = Some(GroupBackdrop {
            colour: &colour16,
            alpha: Some(&alpha16),
            x0: 0,
            y0: 0,
            width: count,
            height: 1,
        });
    }
    render_span::<P8>(
        &p8,
        &PipeSrc::Solid(&colour),
        &mut d8,
        Some(&mut a8),
        Some(&shape),
        0,
        x1,
        0,
    );
    let (mut d16, mut a16) = (widen(&dst), widen(&alpha));
    let (c16, s16) = (widen(&colour), widen(&shape));
    render_span::<P16>(
        &p16,
        &PipeSrc::Solid(&c16),
        &mut d16,
        Some(&mut a16),
        Some(&s16),
        0,
        x1,
        0,
    );
    ((d16, d8), (a16, a8))
}

/// Source over at 16 bits narrows to within two levels of source over at 8
/// bits through the AA pipe over an alpha plane, and to within one level
/// on an opaque destination, the 8-bit SIMD tier against the 16-bit loop.
#[test]
fn source_over_agrees_with_8_bits() {
    let mut rng = Lcg(3);
    for _ in 0..20_000 {
        let ((c16, c8), (a16, a8)) =
            span_at_both_depths::<Rgb8, Rgb16>(&mut rng, BlendMode::Normal, false);
        // The 8-bit over divides by the result alpha with truncation, up
        // to one level low, and the quotient by a small result alpha
        // amplifies the rounding of the inputs; the bound is two levels
        // where the result alpha is at least 128.
        for ((w, n), &a) in c16.chunks(3).zip(c8.chunks(3)).zip(&a8) {
            if a >= 128 {
                assert!(worst(w, n) <= 2, "colour {w:?} vs {n:?} at alpha {a}");
            }
        }
        assert!(worst(&a16, &a8) <= 1, "alpha {a16:?} vs {a8:?}");
    }
    // Opaque destination, solid RGB: the 8-bit SIMD tier against the
    // 16-bit scalar loop.
    for _ in 0..20_000 {
        let count = 1 + rng.below(40);
        let dst: Vec<u8> = (0..count * 3).map(|_| rng.byte()).collect();
        let shape: Vec<u8> = (0..count).map(|_| rng.byte()).collect();
        let colour = [rng.byte(), rng.byte(), rng.byte()];
        let a = rng.byte();
        let x1 = i32::try_from(count).expect("small") - 1;
        let mut d8 = dst.clone();
        render_span::<Rgb8>(
            &pipe(BlendMode::Normal, a),
            &PipeSrc::Solid(&colour),
            &mut d8,
            None,
            Some(&shape),
            0,
            x1,
            0,
        );
        let mut d16 = widen(&dst);
        let (c16, s16) = (widen(&colour), widen(&shape));
        render_span::<Rgb16>(
            &pipe(BlendMode::Normal, <u16 as Sample>::from_u8(a)),
            &PipeSrc::Solid(&c16),
            &mut d16,
            None,
            Some(&s16),
            0,
            x1,
            0,
        );
        assert!(worst(&d16, &d8) <= 1, "{d16:?} vs {d8:?}");
    }
}

const SEPARABLE: [BlendMode; 11] = [
    BlendMode::Multiply,
    BlendMode::Screen,
    BlendMode::Overlay,
    BlendMode::Darken,
    BlendMode::Lighten,
    BlendMode::ColorDodge,
    BlendMode::ColorBurn,
    BlendMode::HardLight,
    BlendMode::SoftLight,
    BlendMode::Difference,
    BlendMode::Exclusion,
];

/// Each separable blend function at 16 bits narrows to the 8-bit function
/// over every pair of 8-bit inputs, exactly for the products, minima and
/// differences, within one level for colour dodge and colour burn (a
/// truncated quotient), and within two for soft light, whose 8-bit
/// evaluation truncates two nested quotients.
#[test]
fn separable_blend_functions_agree_with_8_bits() {
    let src: Vec<u8> = (0..=255).collect();
    let src16 = widen(&src);
    let mut out8 = [0u8; 256];
    let mut out16 = [0u16; 256];
    for mode in SEPARABLE {
        for d in 0..=255u8 {
            blend::apply_separable::<u8>(mode, &src, &[d; 256], &mut out8);
            let d16 = <u16 as Sample>::from_u8(d);
            blend::apply_separable::<u16>(mode, &src16, &[d16; 256], &mut out16);
            for (s, (&w, &n)) in out16.iter().zip(&out8).enumerate() {
                let diff = w.to_u8().abs_diff(n);
                let tolerance = match mode {
                    BlendMode::ColorDodge | BlendMode::ColorBurn => 1,
                    BlendMode::SoftLight => 2,
                    _ => 0,
                };
                assert!(
                    diff <= tolerance,
                    "{mode:?}({s}, {d}): 16-bit {w} ({}) vs 8-bit {n}",
                    w.to_u8()
                );
            }
        }
    }
}

const NONSEPARABLE: [BlendMode; 4] = [
    BlendMode::Hue,
    BlendMode::Saturation,
    BlendMode::Color,
    BlendMode::Luminosity,
];

/// The non-separable blend functions at 16 bits narrow to within ten
/// levels of the 8-bit ones.  The 8-bit evaluation truncates the quotient
/// of `SetSat` and `ClipColor` by small channel differences, and is up to
/// ten levels from the exact result; the 16-bit evaluation is within a
/// tenth of a level of it.
#[test]
fn nonseparable_blend_functions_agree_with_8_bits() {
    let mut rng = Lcg(5);
    for mode in NONSEPARABLE {
        for _ in 0..50_000 {
            let s = [rng.byte(), rng.byte(), rng.byte()];
            let d = [rng.byte(), rng.byte(), rng.byte()];
            let r8 = blend::apply_nonseparable_rgb::<u8>(mode, s, d);
            let r16 = blend::apply_nonseparable_rgb::<u16>(
                mode,
                s.map(<u16 as Sample>::from_u8),
                d.map(<u16 as Sample>::from_u8),
            );
            assert!(
                worst(&r16, &r8) <= 10,
                "{mode:?}({s:?}, {d:?}): {r16:?} vs {r8:?}"
            );
        }
    }
}

/// The general pipe (blend modes, with and without a non-isolated group's
/// backdrop) at 16 bits narrows to within a few levels of 8 bits wherever
/// the result alpha is at least 64.  The over truncates (two levels, as in
/// [`source_over_agrees_with_8_bits`]); over a backdrop the blend function
/// reads the union colour, rounded once more at 8 bits, which overlay and
/// soft light amplify to five levels.  Colour dodge and colour burn over a
/// backdrop are not bounded: their quotient by `M − s` or by `s` grows
/// without bound near its singularity, so one level of the union colour
/// can move the result arbitrarily far ([`separable_blend_functions_agree_with_8_bits`]
/// bounds the functions themselves).
#[test]
fn general_pipe_agrees_with_8_bits() {
    let mut rng = Lcg(9);
    for mode in std::iter::once(BlendMode::Normal).chain(SEPARABLE) {
        for with_backdrop in [false, true] {
            for _ in 0..2_000 {
                let ((c16, c8), (a16, a8)) =
                    span_at_both_depths::<Gray8, Gray16>(&mut rng, mode, with_backdrop);
                assert!(worst(&a16, &a8) <= 1, "{mode:?} alpha");
                for ((&w, &n), &a) in c16.iter().zip(&c8).zip(&a8) {
                    let tolerance = match (mode, with_backdrop) {
                        (BlendMode::ColorDodge | BlendMode::ColorBurn, true) => continue,
                        (BlendMode::Overlay | BlendMode::SoftLight, true) => 5,
                        _ => 2,
                    };
                    if a >= 64 {
                        assert!(
                            w.to_u8().abs_diff(n) <= tolerance,
                            "{mode:?} backdrop={with_backdrop}: {w} vs {n} at alpha {a}"
                        );
                    }
                }
            }
        }
    }
}

/// A two-sample ramp written at 4 096 positions gives 4 096 distinct
/// 16-bit values, against 256 at 8 bits.
#[test]
fn a_ramp_has_more_than_256_levels_at_16_bits() {
    let ramp16 = ColorRamp::<1, u16>::two([0], [u16::MAX]);
    let ramp8 = ColorRamp::<1, u8>::two([0], [u8::MAX]);
    let (mut levels16, mut levels8) = (Vec::new(), Vec::new());
    for k in 0..4_096u32 {
        let s = f64::from(k) / 4_095.0;
        let (mut o16, mut o8) = ([0u16], [0u8]);
        ramp16.write(s, &mut o16);
        ramp8.write(s, &mut o8);
        levels16.push(o16[0]);
        levels8.push(o8[0]);
        assert!(o16[0].to_u8().abs_diff(o8[0]) <= 1, "s={s}");
    }
    levels16.dedup();
    levels8.dedup();
    assert_eq!(levels16.len(), 4_096);
    assert_eq!(levels8.len(), 256);
}

/// An antialiased axial fill on a 16-bit bitmap narrows to within one
/// level of the same fill on an 8-bit bitmap.
#[test]
fn an_antialiased_shaded_fill_agrees_with_8_bits() {
    let (w, h) = (64u32, 16u32);
    let clip = Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, true);
    let path = rect_path(2.3, 1.7, 61.6, 14.2);
    let mut b8: Bitmap<Rgb8> = Bitmap::new_filled(w, h, 1, false, 255);
    let mut b16: Bitmap<Rgb16> = Bitmap::new_filled(w, h, 1, false, u16::MAX);
    let ramp8 = ColorRamp::two([10u8, 200, 0], [250, 20, 128]);
    let ramp16 = ColorRamp::two(
        [10u8, 200, 0].map(<u16 as Sample>::from_u8),
        [250u8, 20, 128].map(<u16 as Sample>::from_u8),
    );
    let a8 = AxialPattern::new(ramp8, 0.0, 0.0, 64.0, 0.0, true, true);
    let a16 = AxialPattern::new(ramp16, 0.0, 0.0, 64.0, 0.0, true, true);
    let m = identity_matrix();
    fill::<Rgb8>(
        &mut b8,
        &clip,
        &path,
        &pipe(BlendMode::Normal, 200),
        &PipeSrc::Pattern(&a8),
        &m,
        1.0,
        true,
    );
    fill::<Rgb16>(
        &mut b16,
        &clip,
        &path,
        &pipe(BlendMode::Normal, <u16 as Sample>::from_u8(200)),
        &PipeSrc::Pattern(&a16),
        &m,
        1.0,
        true,
    );
    for y in 0..h {
        assert!(worst(b16.row_samples(y), b8.row_samples(y)) <= 1, "row {y}");
    }
}

/// A group painted at 16 bits through its alpha plane narrows to within
/// one level of the same group at 8 bits.
#[test]
fn a_transparency_group_agrees_with_8_bits() {
    fn render<P: Pixel>(paint: P::Sample, opacity: P::Sample) -> Bitmap<P> {
        let (w, h) = (16u32, 8u32);
        let clip = Clip::new(0.0, 0.0, f64::from(w) - 0.001, f64::from(h) - 0.001, true);
        let mut parent: Bitmap<P> = Bitmap::new_filled(w, h, 1, false, P::Sample::MAX);
        let params = GroupParams {
            x_min: 0,
            y_min: 0,
            x_max: 15,
            y_max: 7,
            isolated: true,
        };
        let mut group = begin_group(&parent, None, &clip, params, Bitmap::new(w, h, 1, true));
        let colour = vec![paint; P::NCOMPS];
        fill::<P>(
            &mut group.bitmap,
            &clip,
            &rect_path(1.4, 1.2, 14.7, 6.6),
            &pipe(BlendMode::Normal, P::Sample::MAX),
            &PipeSrc::Solid(&colour),
            &identity_matrix(),
            1.0,
            true,
        );
        let _ = paint_group(&mut parent, group, &pipe(BlendMode::Multiply, opacity));
        parent
    }
    let b8 = render::<Rgb8>(90, 170);
    let b16 = render::<Rgb16>(<u16 as Sample>::from_u8(90), <u16 as Sample>::from_u8(170));
    for y in 0..b8.height {
        assert!(worst(b16.row_samples(y), b8.row_samples(y)) <= 1, "row {y}");
    }
}
