//! The sample types an image may hold.

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

/// An unsigned integer gray sample of `BITS` bits: `u8` or `u16`.  The
/// value [`MAX`](Self::MAX) is white.
///
/// Detection runs on 8-bit samples; [`Report::replay`](crate::Report::replay)
/// resamples an image of either type.  Sealed: implemented for `u8` and
/// `u16` only.
pub trait Sample: sealed::Sealed + Copy + Default + Eq + std::fmt::Debug + 'static {
    /// White, the largest value.
    const MAX: Self;

    /// The value as `f64` (exact).
    fn to_f64(self) -> f64;

    /// `v` rounded half up and saturated to `0..=MAX`; NaN maps to 0.
    fn from_f64(v: f64) -> Self;
}

impl Sample for u8 {
    const MAX: Self = Self::MAX;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 255] before the cast"
    )]
    fn from_f64(v: f64) -> Self {
        (v + 0.5).clamp(0.0, 255.0) as Self
    }
}

impl Sample for u16 {
    const MAX: Self = Self::MAX;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 65535] before the cast"
    )]
    fn from_f64(v: f64) -> Self {
        (v + 0.5).clamp(0.0, 65_535.0) as Self
    }
}

#[cfg(test)]
mod tests {
    use super::Sample;

    #[test]
    fn rounding_saturates_at_either_width() {
        assert_eq!(u8::from_f64(-3.0), 0);
        assert_eq!(u8::from_f64(127.5), 128);
        assert_eq!(u8::from_f64(300.0), 255);
        assert_eq!(u8::from_f64(f64::NAN), 0);
        assert_eq!(u16::from_f64(32_767.5), 32_768);
        assert_eq!(u16::from_f64(70_000.0), 65_535);
        assert_eq!(u16::from_f64(f64::NAN), 0);
    }
}
