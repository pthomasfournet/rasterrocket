//! Colour samples of 8 and 16 bits and their compositing arithmetic.
//!
//! [`Sample`] is the channel type of a pixel: `u8` for the 8-bit pixel
//! types, `u16` for the 16-bit ones.  It carries the integer arithmetic the
//! compositor is written in, once for both depths: products of samples are
//! formed in [`Sample::Wide`], an unsigned type that holds a product of
//! three samples (`u32` for `u8`, `u64` for `u16`), and reduced by
//! [`Sample::div_max`].  The `u8` instance of every operation is the 8-bit
//! primitive of [`crate::convert`] itself.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Shr, Sub};

use bytemuck::{Pod, Zeroable};

use crate::convert::{div255, gray_to_u8, lerp_u8};

/// The bits of a sample: the depth of a pixel type or of a render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SampleDepth {
    /// 8 bits, [`u8`] samples.
    #[default]
    Eight,
    /// 16 bits, [`u16`] samples.
    Sixteen,
}

impl SampleDepth {
    /// Bits per sample.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }

    /// Bytes per sample.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Eight => 1,
            Self::Sixteen => 2,
        }
    }
}

/// The unsigned integer type products of samples are formed in.
pub trait Wide:
    Copy
    + Ord
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Shr<u32, Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Zero.
    const ZERO: Self;
    /// The value `v`.
    fn of(v: u32) -> Self;
    /// The value `v`, which the type holds: at most the square of the
    /// largest sample.
    fn of_u64(v: u64) -> Self;
}

impl Wide for u32 {
    const ZERO: Self = 0;
    #[inline]
    fn of(v: u32) -> Self {
        v
    }
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the caller passes a value of at most 255², which u32 holds"
    )]
    fn of_u64(v: u64) -> Self {
        debug_assert!(v <= u64::from(Self::MAX), "u32::of_u64: {v}");
        v as Self
    }
}

impl Wide for u64 {
    const ZERO: Self = 0;
    #[inline]
    fn of(v: u32) -> Self {
        Self::from(v)
    }
    #[inline]
    fn of_u64(v: u64) -> Self {
        v
    }
}

/// One colour or alpha channel value: `0` is no intensity (or no ink, or
/// transparent), [`Self::MAX`] full intensity.
pub trait Sample:
    Copy + Pod + Zeroable + Ord + Default + Debug + Hash + Send + Sync + 'static
{
    /// Zero: no intensity, no ink, transparent.
    const ZERO: Self;
    /// The largest value: full intensity, full ink, opaque.
    const MAX: Self;
    /// Bits per sample.
    const BITS: u32;
    /// The depth of this sample type.
    const DEPTH: SampleDepth;
    /// The scale of the interpolation weight of [`Self::lerp`]:
    /// `2^BITS`.
    const LERP_ONE: u32;
    /// The type a product of up to three samples is formed in.
    type Wide: Wide;
    /// A lookup table with one entry for every sample value.
    type Table: AsRef<[Self]> + Copy + Debug + Send + Sync + 'static;

    /// The entry of `table` at `v`.
    fn lookup(table: &Self::Table, v: Self) -> Self;

    /// The table mapping every value to itself.
    fn identity_table() -> &'static Self::Table;

    /// The sample as a [`Self::Wide`].
    fn wide(self) -> Self::Wide;

    /// The sample of value `w`, which is at most [`Self::MAX`].
    fn from_wide(w: Self::Wide) -> Self;

    /// The sample as an `i64`.
    fn signed(self) -> i64;

    /// The sample of value `v` clamped to `0..=MAX`.
    fn from_signed(v: i64) -> Self;

    /// [`Self::MAX`] as a [`Self::Wide`].
    #[inline]
    #[must_use]
    fn max_wide() -> Self::Wide {
        Self::MAX.wide()
    }

    /// `x / MAX` rounded to nearest, for `x ≤ MAX²`; at most `MAX`.
    fn div_max(x: Self::Wide) -> Self;

    /// `a + (b − a)·t / LERP_ONE` rounded half up, for `t ≤ LERP_ONE`; it
    /// lies between `a` and `b` and is exact at both ends.
    fn lerp(a: Self, b: Self, t: u32) -> Self;

    /// `round(clamp(v, 0, 1) · MAX)`; NaN is `0`.
    fn from_unit(v: f64) -> Self;

    /// The sample as a fraction of full intensity, `v / MAX`.
    fn to_unit(self) -> f64;

    /// The 8-bit value `v` on this sample's scale, `v · MAX / 255`, which
    /// is exact: `v` for `u8`, `257 · v` for `u16`.
    fn from_u8(v: u8) -> Self;

    /// The nearest 8-bit value, `round(v · 255 / MAX)`: the inverse of
    /// [`Self::from_u8`].
    fn to_u8(self) -> u8;

    /// The samples of `s` as bytes when the sample is `u8`; `None` for any
    /// wider sample.  Selects the 8-bit SIMD kernels.
    fn as_u8(s: &[Self]) -> Option<&[u8]>;

    /// [`Self::as_u8`] for a mutable slice.
    fn as_u8_mut(s: &mut [Self]) -> Option<&mut [u8]>;

    /// The bytes `s` as samples when the sample is `u8`; `None` for any
    /// wider sample, whose values need [`Self::from_u8`].
    fn from_u8_slice(s: &[u8]) -> Option<&[Self]>;

    /// The samples of `s` as `u16` when the sample is `u16`; `None`
    /// otherwise.
    fn as_u16(s: &[Self]) -> Option<&[u16]>;

    /// The 16-bit samples `s` as samples when the sample is `u16`; `None`
    /// otherwise.
    fn from_u16_slice(s: &[u16]) -> Option<&[Self]>;

    /// The samples `v` mapped by the function of this sample's depth:
    /// `narrow` for `u8`, `wide` for `u16`.  A conversion defined at each
    /// depth by its own function is written once over the sample through
    /// this.
    fn by_depth<const N: usize, const M: usize>(
        v: [Self; N],
        narrow: impl FnOnce([u8; N]) -> [u8; M],
        wide: impl FnOnce([u16; N]) -> [u16; M],
    ) -> [Self; M];

    /// The sample of type `T` on this sample's scale: the value itself at
    /// the same depth, else its nearest 8-bit value ([`Self::to_u8`]) on
    /// this scale, which is exact when widening.
    #[inline]
    #[must_use]
    fn from_sample<T: Sample>(v: T) -> Self {
        if T::DEPTH == Self::DEPTH {
            Self::from_signed(v.signed())
        } else {
            Self::from_u8(v.to_u8())
        }
    }

    /// `s` as samples of this type when they are of the same depth.
    #[inline]
    #[must_use]
    fn same_depth<T: Sample>(s: &[T]) -> Option<&[Self]> {
        T::as_u8(s)
            .and_then(Self::from_u8_slice)
            .or_else(|| T::as_u16(s).and_then(Self::from_u16_slice))
    }
}

/// Every byte, in order.
static IDENTITY_U8: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut v = u8::MAX;
    while v > 0 {
        t[v as usize] = v;
        v -= 1;
    }
    t
};

/// Every 16-bit value, in order.
static IDENTITY_U16: [u16; 65_536] = {
    let mut t = [0u16; 65_536];
    let mut v = u16::MAX;
    while v > 0 {
        t[v as usize] = v;
        v -= 1;
    }
    t
};

impl Sample for u8 {
    const ZERO: Self = 0;
    const MAX: Self = Self::MAX;
    const BITS: u32 = 8;
    const DEPTH: SampleDepth = SampleDepth::Eight;
    const LERP_ONE: u32 = 256;
    type Wide = u32;
    type Table = [Self; 256];

    #[inline]
    fn lookup(table: &[Self; 256], v: Self) -> Self {
        table[usize::from(v)]
    }

    #[inline]
    fn identity_table() -> &'static [Self; 256] {
        &IDENTITY_U8
    }

    #[inline]
    fn wide(self) -> u32 {
        u32::from(self)
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the caller passes a value of at most 255"
    )]
    fn from_wide(w: u32) -> Self {
        debug_assert!(w <= 255, "u8::from_wide: {w} > 255");
        w as Self
    }

    #[inline]
    fn signed(self) -> i64 {
        i64::from(self)
    }

    #[inline]
    #[expect(clippy::cast_sign_loss, reason = "clamped to 0..=255 first")]
    fn from_signed(v: i64) -> Self {
        v.clamp(0, 255) as Self
    }

    #[inline]
    fn div_max(x: u32) -> Self {
        div255(x)
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: u32) -> Self {
        lerp_u8(a, b, t)
    }

    #[inline]
    fn from_unit(v: f64) -> Self {
        gray_to_u8(v)
    }

    #[inline]
    fn to_unit(self) -> f64 {
        f64::from(self) / 255.0
    }

    #[inline]
    fn from_u8(v: u8) -> Self {
        v
    }

    #[inline]
    fn to_u8(self) -> u8 {
        self
    }

    #[inline]
    fn as_u8(s: &[Self]) -> Option<&[u8]> {
        Some(s)
    }

    #[inline]
    fn as_u8_mut(s: &mut [Self]) -> Option<&mut [u8]> {
        Some(s)
    }

    #[inline]
    fn from_u8_slice(s: &[u8]) -> Option<&[Self]> {
        Some(s)
    }

    #[inline]
    fn as_u16(_: &[Self]) -> Option<&[u16]> {
        None
    }

    #[inline]
    fn by_depth<const N: usize, const M: usize>(
        v: [Self; N],
        narrow: impl FnOnce([Self; N]) -> [Self; M],
        _wide: impl FnOnce([u16; N]) -> [u16; M],
    ) -> [Self; M] {
        narrow(v)
    }

    #[inline]
    fn from_u16_slice(_: &[u16]) -> Option<&[Self]> {
        None
    }
}

impl Sample for u16 {
    const ZERO: Self = 0;
    const MAX: Self = Self::MAX;
    const BITS: u32 = 16;
    const DEPTH: SampleDepth = SampleDepth::Sixteen;
    const LERP_ONE: u32 = 1 << 16;
    type Wide = u64;
    type Table = [Self; 65_536];

    #[inline]
    fn lookup(table: &[Self; 65_536], v: Self) -> Self {
        table[usize::from(v)]
    }

    #[inline]
    fn identity_table() -> &'static [Self; 65_536] {
        &IDENTITY_U16
    }

    #[inline]
    fn wide(self) -> u64 {
        u64::from(self)
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the caller passes a value of at most 65535"
    )]
    fn from_wide(w: u64) -> Self {
        debug_assert!(w <= 65_535, "u16::from_wide: {w} > 65535");
        w as Self
    }

    #[inline]
    fn signed(self) -> i64 {
        i64::from(self)
    }

    #[inline]
    #[expect(clippy::cast_sign_loss, reason = "clamped to 0..=65535 first")]
    fn from_signed(v: i64) -> Self {
        v.clamp(0, 65_535) as Self
    }

    /// `(t + (t >> 16)) >> 16` with `t = x + 2^15`: the 16-bit analogue of
    /// [`div255`], `x / 65535` rounded to nearest for every `x ≤ 65535²`
    /// (an exact half never occurs, 65535 being odd).
    #[inline]
    fn div_max(x: u64) -> Self {
        let t = x + 0x8000;
        Self::from_wide(((t + (t >> 16)) >> 16).min(65_535))
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: u32) -> Self {
        debug_assert!(t <= 1 << 16, "u16::lerp: t={t} out of range [0, 65536]");
        let t = u64::from(t);
        Self::from_wide((u64::from(a) * ((1 << 16) - t) + u64::from(b) * t + 0x8000) >> 16)
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, 1] and scaled to [0, 65535]; NaN saturates to 0"
    )]
    fn from_unit(v: f64) -> Self {
        (v.clamp(0.0, 1.0) * 65_535.0).round() as Self
    }

    #[inline]
    fn to_unit(self) -> f64 {
        f64::from(self) / 65_535.0
    }

    #[inline]
    fn from_u8(v: u8) -> Self {
        Self::from(v) * 257
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a quotient of at most 65535·255/65535 = 255"
    )]
    fn to_u8(self) -> u8 {
        ((u32::from(self) * 255 + 32_767) / 65_535) as u8
    }

    #[inline]
    fn as_u8(_: &[Self]) -> Option<&[u8]> {
        None
    }

    #[inline]
    fn as_u8_mut(_: &mut [Self]) -> Option<&mut [u8]> {
        None
    }

    #[inline]
    fn from_u8_slice(_: &[u8]) -> Option<&[Self]> {
        None
    }

    #[inline]
    fn as_u16(s: &[Self]) -> Option<&[u16]> {
        Some(s)
    }

    #[inline]
    fn by_depth<const N: usize, const M: usize>(
        v: [Self; N],
        _narrow: impl FnOnce([u8; N]) -> [u8; M],
        wide: impl FnOnce([Self; N]) -> [Self; M],
    ) -> [Self; M] {
        wide(v)
    }

    #[inline]
    fn from_u16_slice(s: &[u16]) -> Option<&[Self]> {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn u8_instances_are_the_8_bit_primitives() {
        for x in 0..=65_535u32 {
            assert_eq!(<u8 as Sample>::div_max(x), div255(x));
        }
        for v in 0..=255u8 {
            assert_eq!(<u8 as Sample>::from_u8(v), v);
            assert_eq!(v.to_u8(), v);
        }
    }

    #[test]
    fn u16_div_max_rounds_to_nearest_at_the_edges() {
        let max = 65_535u64;
        for x in (0..=200_000u64).chain(max * max - 200_000..=max * max) {
            let exact = (2 * x + max) / (2 * max);
            assert_eq!(u64::from(<u16 as Sample>::div_max(x)), exact, "x={x}");
        }
    }

    /// Each depth takes its own function.
    #[test]
    fn by_depth_takes_the_function_of_the_depth() {
        let narrow = |[v]: [u8; 1]| [v, v / 2];
        let wide = |[v]: [u16; 1]| [v, v / 4];
        assert_eq!(u8::by_depth([200], narrow, wide), [200, 100]);
        assert_eq!(u16::by_depth([200], narrow, wide), [200, 50]);
    }

    #[test]
    fn a_sample_of_another_depth_rescales() {
        for v in 0..=255u8 {
            assert_eq!(u16::from_sample(v), u16::from_u8(v));
            assert_eq!(u8::from_sample(v), v);
        }
        for v in [0u16, 1, 128, 32_767, 65_535] {
            assert_eq!(u8::from_sample(v), v.to_u8());
            assert_eq!(u16::from_sample(v), v);
        }
        assert_eq!(u16::same_depth(&[3u16, 4]), Some(&[3u16, 4][..]));
        assert_eq!(u16::same_depth(&[3u8]), None);
        assert_eq!(u8::same_depth(&[3u8]), Some(&[3u8][..]));
    }

    #[test]
    fn widening_a_byte_is_exact_and_narrowing_inverts_it() {
        for v in 0..=255u8 {
            let w = <u16 as Sample>::from_u8(v);
            assert_eq!(u32::from(w), u32::from(v) * 65_535 / 255);
            assert_eq!(w.to_u8(), v);
        }
    }

    #[test]
    fn narrowing_rounds_to_nearest() {
        for v in 0..=65_535u16 {
            let exact = (f64::from(v) * 255.0 / 65_535.0).round();
            assert!(
                (f64::from(v.to_u8()) - exact).abs() < 1e-9,
                "v={v}: {} vs {exact}",
                v.to_u8()
            );
        }
    }

    #[test]
    fn from_unit_matches_the_8_bit_quantiser_at_8_bits() {
        for k in 0..=10_000u32 {
            let v = f64::from(k) / 10_000.0;
            assert_eq!(<u8 as Sample>::from_unit(v), gray_to_u8(v));
            let w = <u16 as Sample>::from_unit(v);
            assert!(w.to_u8().abs_diff(gray_to_u8(v)) <= 1, "v={v}");
        }
        assert_eq!(<u16 as Sample>::from_unit(f64::NAN), 0);
        assert_eq!(<u16 as Sample>::from_unit(2.0), 65_535);
    }

    proptest! {
        #[test]
        fn u16_div_max_is_exact(x in 0..=65_535u64 * 65_535) {
            let exact = (2 * x + 65_535) / (2 * 65_535);
            prop_assert_eq!(u64::from(<u16 as Sample>::div_max(x)), exact);
        }

        #[test]
        fn u16_lerp_is_bounded_and_exact_at_the_ends(a: u16, b: u16, t in 0..=65_536u32) {
            let v = <u16 as Sample>::lerp(a, b, t);
            prop_assert!(a.min(b) <= v && v <= a.max(b));
            prop_assert_eq!(<u16 as Sample>::lerp(a, b, 0), a);
            prop_assert_eq!(<u16 as Sample>::lerp(a, b, 65_536), b);
        }

        #[test]
        fn u16_over_narrows_to_within_one_of_the_8_bit_over(cs: u8, cd: u8, a: u8) {
            let over8 = div255((255 - u32::from(a)) * u32::from(cd) + u32::from(a) * u32::from(cs));
            let (cs16, cd16, a16) = (
                u64::from(<u16 as Sample>::from_u8(cs)),
                u64::from(<u16 as Sample>::from_u8(cd)),
                u64::from(<u16 as Sample>::from_u8(a)),
            );
            let over16 = <u16 as Sample>::div_max((65_535 - a16) * cd16 + a16 * cs16);
            prop_assert!(over16.to_u8().abs_diff(over8) <= 1);
        }
    }
}
