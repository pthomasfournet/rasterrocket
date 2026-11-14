//! Property tests of the colour conversions and rounding helpers.
//!
//! Each property runs the case count given to `config` unless
//! `PROPTEST_CASES` sets another.

use proptest::prelude::*;
use proptest::test_runner::Config;
use rasterrocket_color::cmyk::{SpotLut, cmyk_to_rgb, cmyk_to_rgb_row, ink_to_rgb_row};
use rasterrocket_color::convert::{
    ceil_i32, div255, floor_i32, lerp_u8, luminosity, round_half_up_i32,
};

/// The runner configuration: `cases` cases, or `PROPTEST_CASES` when set.
fn config(cases: u32) -> Config {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cases);
    Config {
        cases,
        ..Config::default()
    }
}

/// Rec. 601 luma of an sRGB triple, unrounded.
fn luma(rgb: [u8; 3]) -> f64 {
    0.114f64.mul_add(
        f64::from(rgb[2]),
        0.299f64.mul_add(f64::from(rgb[0]), 0.587 * f64::from(rgb[1])),
    )
}

/// A float within four ulps of `n + offset`, `n` small or up to millions.
fn near(offset: f64) -> impl Strategy<Value = f64> {
    let n = prop_oneof![-3i32..3, -3_000_000i32..3_000_000];
    (n, -4i32..=4).prop_map(move |(n, ulps)| {
        let step = if ulps < 0 {
            f64::next_down
        } else {
            f64::next_up
        };
        (0..ulps.unsigned_abs()).fold(f64::from(n) + offset, |x, _| step(x))
    })
}

/// A float in the `i32` range, often within a few ulps of an integer or a
/// half-integer.
fn finite() -> impl Strategy<Value = f64> {
    prop_oneof![near(0.5), near(0.0), -1e9f64..1e9]
}

/// A float near a half-integer, an integer, or anywhere, including
/// values past the `i32` range, infinities and NaN.
fn float() -> impl Strategy<Value = f64> {
    prop_oneof![
        finite(),
        any::<f64>(),
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
    ]
}

/// Whether `x` lies in the `i32` range.
fn in_i32(x: f64) -> bool {
    x >= f64::from(i32::MIN) && x <= f64::from(i32::MAX)
}

proptest! {
    #![proptest_config(config(2048))]

    /// The row converter, whichever SIMD tier runs, agrees with the
    /// per-pixel conversion on every pixel, at any length and any start
    /// alignment of the input and output slices.
    #[test]
    fn row_conversion_matches_per_pixel(
        cmyk in prop::collection::vec(any::<u8>(), 0..1024),
        offset in 0usize..64,
        out_offset in 0usize..64,
    ) {
        let offset = offset.min(cmyk.len());
        let src = &cmyk[offset..offset + (cmyk.len() - offset) / 4 * 4];
        let mut out = vec![0u8; out_offset + src.len() / 4 * 3];
        cmyk_to_rgb_row(src, &mut out[out_offset..]);
        for (i, (px, rgb)) in src
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out[out_offset..].as_chunks::<3>().0)
            .enumerate()
        {
            prop_assert_eq!(*rgb, cmyk_to_rgb(px[0], px[1], px[2], px[3]), "pixel {}", i);
        }
    }

    /// Adding ink never lightens a colour by more than the rounding of the
    /// interpolation stages.
    #[test]
    fn more_ink_is_not_lighter(
        ink in any::<[u8; 4]>(),
        channel in 0usize..4,
        extra in 1u8..=255,
    ) {
        let mut more = ink;
        more[channel] = ink[channel].saturating_add(extra);
        let before = luma(cmyk_to_rgb(ink[0], ink[1], ink[2], ink[3]));
        let after = luma(cmyk_to_rgb(more[0], more[1], more[2], more[3]));
        prop_assert!(after <= before + 1.5, "{:?} {} -> {:?} {}", ink, before, more, after);
    }

    /// An ink row converts as its pixels' process inks, each spot's process
    /// equivalent added as a probabilistic union, through the CMYK
    /// conversion; rows span several of the converter's internal chunks.
    #[test]
    fn ink_rows_match_the_per_pixel_union(
        ink in prop::collection::vec(any::<u8>(), 0..300).prop_map(|v| {
            let mut v = v;
            v.truncate(v.len() / 8 * 8);
            v
        }),
        luts in prop::collection::vec(
            prop::option::of(prop::collection::vec(any::<[u8; 4]>(), 256)),
            4,
        ),
        repeat in 1usize..40,
    ) {
        let ink: Vec<u8> = std::iter::repeat_n(&ink, repeat).flatten().copied().collect();
        let tables: Vec<Option<SpotLut>> = luts
            .iter()
            .map(|l| l.as_ref().map(|v| v.as_slice().try_into().expect("256 entries")))
            .collect();
        let spots: [Option<&SpotLut>; 4] = std::array::from_fn(|i| tables[i].as_ref());
        let mut out = vec![0u8; ink.len() / 8 * 3];
        ink_to_rgb_row(&ink, &spots, &mut out);
        // The probabilistic union `a + b - a·b/255`, rounded to nearest.
        let union = |a: u8, b: u8| {
            let (a, b) = (u32::from(a), u32::from(b));
            u8::try_from((255 * (a + b) - a * b + 127) / 255).expect("at most 255")
        };
        for (px, rgb) in ink.as_chunks::<8>().0.iter().zip(out.as_chunks::<3>().0) {
            let mut cmyk = [px[0], px[1], px[2], px[3]];
            for (lut, &tint) in spots.iter().zip(&px[4..]) {
                if let Some(lut) = lut {
                    for (c, &q) in cmyk.iter_mut().zip(&lut[usize::from(tint)]) {
                        *c = union(*c, q);
                    }
                }
            }
            prop_assert_eq!(*rgb, cmyk_to_rgb(cmyk[0], cmyk[1], cmyk[2], cmyk[3]));
        }
    }

    /// Floor and ceiling agree with `f64`'s inside the `i32` range and
    /// saturate outside it; NaN maps to `i32::MIN`.
    #[test]
    fn floor_and_ceil_match_f64(x in float()) {
        for (got, exact) in [(floor_i32(x), x.floor()), (ceil_i32(x), x.ceil())] {
            let want = if x.is_nan() {
                i32::MIN
            } else if in_i32(exact) {
                #[expect(clippy::cast_possible_truncation, reason = "integral and in range")]
                let v = exact as i32;
                v
            } else if exact > 0.0 {
                i32::MAX
            } else {
                i32::MIN
            };
            prop_assert_eq!(got, want, "x = {:?}", x);
        }
    }

    /// Rounding returns the nearest integer, a tie going toward +∞.
    #[test]
    fn round_half_up_is_nearest(x in finite()) {
        let r = round_half_up_i32(x);
        let (lo, hi) = (x.floor(), x.ceil());
        #[expect(clippy::cast_possible_truncation, reason = "integral and in range")]
        let want = if x - lo >= hi - x { hi as i32 } else { lo as i32 };
        prop_assert_eq!(r, want, "x = {:?}", x);
    }

    /// `div255` of a product of two bytes is the product over 255 rounded
    /// to nearest.
    #[test]
    fn div255_rounds_products(a in any::<u8>(), b in any::<u8>()) {
        let p = u32::from(a) * u32::from(b);
        prop_assert_eq!(u32::from(div255(p)), (2 * p + 255) / 510);
    }

    /// Interpolation stays between its endpoints and reaches them.
    #[test]
    fn lerp_stays_between_its_endpoints(a in any::<u8>(), b in any::<u8>(), t in 0u32..=256) {
        let v = lerp_u8(a, b, t);
        prop_assert!(a.min(b) <= v && v <= a.max(b));
        prop_assert_eq!(lerp_u8(a, b, 0), a);
        prop_assert_eq!(lerp_u8(a, b, 256), b);
    }

    /// Luminosity lies between the darkest and lightest channel.
    #[test]
    fn luminosity_is_a_weighted_mean(r in any::<u8>(), g in any::<u8>(), b in any::<u8>()) {
        let l = luminosity(r, g, b);
        prop_assert!(r.min(g).min(b) <= l && l <= r.max(g).max(b));
    }
}
