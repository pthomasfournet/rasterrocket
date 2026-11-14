//! Digests of the 8-bit compositing results.
//!
//! Each test hashes the output of one family of 8-bit compositing
//! functions over a fixed set of inputs (exhaustive where the domain
//! allows, a fixed pseudo-random sequence elsewhere) and compares the hash
//! with a constant.  The constants were taken from the 8-bit-only
//! compositor; any change to the arithmetic an 8-bit render performs
//! changes a digest.

use crate::pipe::{
    ChannelMask, GroupBackdrop, PipeSrc, PipeState, RowPattern, TransferSet, blend, render_span,
};
use crate::shading::ColorRamp;
use crate::types::BlendMode;
use color::{DeviceN8, Gray8, Pixel, Rgb8, TransferLut};

/// FNV-1a over a byte stream.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

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

    /// A byte biased towards the ends of the range, where the formulas
    /// branch.
    fn level(&mut self) -> u8 {
        match self.next() % 4 {
            0 => 0,
            1 => 255,
            _ => self.byte(),
        }
    }

    /// `true` with probability `1 / n`.
    fn one_in(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }

    fn below(&mut self, n: u64) -> usize {
        usize::try_from(self.next() % n).expect("small")
    }
}

const SEPARABLE: [BlendMode; 12] = [
    BlendMode::Normal,
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

const NONSEPARABLE: [BlendMode; 4] = [
    BlendMode::Hue,
    BlendMode::Saturation,
    BlendMode::Color,
    BlendMode::Luminosity,
];

#[test]
fn separable_blend_modes_over_every_pair() {
    let mut h = Fnv::new();
    let src: Vec<u8> = (0..=255u8).collect();
    let mut out = [0u8; 256];
    for mode in SEPARABLE {
        for d in 0..=255u8 {
            blend::apply_separable(mode, &src, &[d; 256], &mut out);
            h.bytes(&out);
        }
    }
    assert_eq!(h.0, SEPARABLE_DIGEST, "separable digest {:#x}", h.0);
}

#[test]
fn nonseparable_blend_modes_over_a_sequence() {
    let mut h = Fnv::new();
    let mut rng = Lcg(1);
    for mode in NONSEPARABLE {
        for _ in 0..200_000 {
            let s = [rng.level(), rng.level(), rng.level()];
            let d = [rng.level(), rng.level(), rng.level()];
            h.bytes(&blend::apply_nonseparable_rgb(mode, s, d));
        }
    }
    assert_eq!(h.0, NONSEPARABLE_DIGEST, "non-separable digest {:#x}", h.0);
}

const ALL_MODES: [BlendMode; 16] = [
    BlendMode::Normal,
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
    BlendMode::Hue,
    BlendMode::Saturation,
    BlendMode::Color,
    BlendMode::Luminosity,
];

/// Composite `spans` random spans of pixels of type `P` through
/// [`render_span`] and hash every result: every blend mode, constant
/// alpha, with and without an alpha plane, shape, backdrop, overprint
/// mask, a pattern source and a non-identity transfer.
fn span_digest<P: Pixel>(spans: usize, seed: u64) -> u64 {
    static DN: [[u8; 256]; 8] = [TransferLut::INVERTED.0; 8];
    let id = TransferLut::IDENTITY.as_array();
    let inv = TransferLut::INVERTED.as_array();
    let n = P::BYTES;
    let mut h = Fnv::new();
    let mut rng = Lcg(seed);
    for _ in 0..spans {
        let count = 1 + rng.below(40);
        let x0 = i32::try_from(rng.below(8)).expect("small");
        let x1 = x0 + i32::try_from(count).expect("small") - 1;
        let mut dst: Vec<u8> = (0..count * n).map(|_| rng.level()).collect();
        let with_alpha = rng.one_in(2);
        let mut alpha: Vec<u8> = (0..count).map(|_| rng.level()).collect();
        let shape: Vec<u8> = (0..count).map(|_| rng.level()).collect();
        let colour: Vec<u8> = (0..n).map(|_| rng.level()).collect();
        let row: Vec<u8> = (0..count * n).map(|_| rng.level()).collect();
        let bd_colour: Vec<u8> = (0..48 * n).map(|_| rng.level()).collect();
        let bd_alpha: Vec<u8> = (0..48).map(|_| rng.level()).collect();
        let transfer = if rng.one_in(4) {
            TransferSet::new([inv, id, inv], inv, [id, inv, id, inv], &DN)
        } else {
            TransferSet::identity_rgb()
        };
        let backdrop = (with_alpha && rng.one_in(3)).then(|| GroupBackdrop {
            colour: &bd_colour,
            alpha: (rng.one_in(2)).then_some(bd_alpha.as_slice()),
            x0: 0,
            y0: 0,
            width: 48,
            height: 1,
        });
        let overprint = (with_alpha && rng.one_in(3)).then(|| ChannelMask::new(rng.byte()));
        let pipe = PipeState {
            blend_mode: ALL_MODES[rng.below(16)],
            a_input: rng.level(),
            transfer,
            backdrop,
            overprint,
        };
        let pattern = RowPattern { data: &row };
        let src = if rng.one_in(2) {
            PipeSrc::Solid(&colour)
        } else {
            PipeSrc::Pattern(&pattern)
        };
        let shape = (!rng.one_in(3)).then_some(shape.as_slice());
        render_span::<P>(
            &pipe,
            &src,
            &mut dst,
            with_alpha.then_some(alpha.as_mut_slice()),
            shape,
            x0,
            x1,
            0,
        );
        h.bytes(&dst);
        if with_alpha {
            h.bytes(&alpha);
        }
    }
    h.0
}

#[test]
fn rgb_spans() {
    let d = span_digest::<Rgb8>(60_000, 7);
    assert_eq!(d, RGB_SPAN_DIGEST, "RGB span digest {d:#x}");
}

#[test]
fn gray_spans() {
    let d = span_digest::<Gray8>(60_000, 11);
    assert_eq!(d, GRAY_SPAN_DIGEST, "gray span digest {d:#x}");
}

#[test]
fn ink_spans() {
    let d = span_digest::<DeviceN8>(60_000, 13);
    assert_eq!(d, INK_SPAN_DIGEST, "ink span digest {d:#x}");
}

#[test]
fn colour_ramp_positions() {
    let mut h = Fnv::new();
    let two = ColorRamp::two([0u8, 128, 255], [255, 7, 0]);
    let many = ColorRamp::from_samples(
        (0..=255u8)
            .map(|v| [v, v.wrapping_mul(7), 255 - v])
            .collect(),
    )
    .expect("ramp");
    let mut out = [0u8; 3];
    for ramp in [&two, &many] {
        for k in 0..=70_000u32 {
            ramp.write(f64::from(k) / 70_000.0, &mut out);
            h.bytes(&out);
        }
    }
    assert_eq!(h.0, RAMP_DIGEST, "ramp digest {:#x}", h.0);
}

const SEPARABLE_DIGEST: u64 = 0x997b_adbe_23b5_0fa5;
const NONSEPARABLE_DIGEST: u64 = 0x54ca_930d_3671_d848;
const RGB_SPAN_DIGEST: u64 = 0x7a9d_a368_6f82_a82a;
const GRAY_SPAN_DIGEST: u64 = 0xd835_ca3b_915e_462d;
const INK_SPAN_DIGEST: u64 = 0xcacd_4f19_3aa8_d646;
const RAMP_DIGEST: u64 = 0xd4b4_e6e3_cc00_d16a;
