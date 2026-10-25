//! Function-based shading pattern.
//!
//! A PDF Type 1 shading applies an arbitrary function `f(x, y) → colour`
//! over a rectangular domain.  The function is supplied as a closure; this
//! module wraps it in the [`Pattern`] interface.
//!
//! The closure receives device-space pixel coordinates and returns an RGB
//! triple already in device colour space, or `None` for a pixel outside the
//! shading's domain, which is then not painted.  Any colour-space conversion
//! is the caller's responsibility.
//!
//! # Panics
//!
//! The closure itself must not panic.  Panics inside the closure propagate
//! through `fill_span` to the rasterizer caller.

use crate::pipe::Pattern;

/// The closure type of a [`FunctionPattern`].
type ShadingFn = dyn Fn(f64, f64) -> Option<[u8; 3]> + Send + Sync;

/// A shading pattern driven by an arbitrary `f(x, y) → Option<[R, G, B]>`
/// closure.
///
/// The closure is boxed so it can capture ICC profiles, LUTs, or any other
/// per-page state.  It must be `Send + Sync` because [`Pattern`] is used
/// across rayon threads.
pub struct FunctionPattern {
    func: Box<ShadingFn>,
}

impl FunctionPattern {
    /// Create a function-based pattern from a closure `f(x, y)`.
    ///
    /// `x` and `y` are device-space pixel coordinates; `None` leaves the
    /// pixel unpainted.
    #[must_use]
    pub fn new<F>(func: F) -> Self
    where
        F: Fn(f64, f64) -> Option<[u8; 3]> + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
        }
    }
}

impl Pattern for FunctionPattern {
    fn fill_span(&self, y: i32, x0: i32, x1: i32, out: &mut [u8]) {
        // out.len() == (x1-x0+1)*3 is an invariant guaranteed by render_span.
        for (x, px) in (x0..=x1).zip(out.as_chunks_mut::<3>().0) {
            *px = (self.func)(f64::from(x), f64::from(y)).unwrap_or_default();
        }
    }

    fn may_leave_unpainted(&self) -> bool {
        true
    }

    fn fill_span_painted(&self, y: i32, x0: i32, x1: i32, out: &mut [u8], painted: &mut [bool]) {
        for ((x, px), p) in (x0..=x1).zip(out.as_chunks_mut::<3>().0).zip(painted) {
            let rgb = (self.func)(f64::from(x), f64::from(y));
            *p = rgb.is_some();
            *px = rgb.unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_x_coordinate_in_output() {
        // Each pixel's R/G/B equals its x coordinate (mod 256). fill_span is
        // called with integer x ∈ [10, 13], so f64-as-u8 is exact here.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test passes integer x ∈ [10,13]; cast is exact"
        )]
        let p = FunctionPattern::new(|x, _y| {
            let v = x as u8;
            Some([v, v, v])
        });
        let mut out = vec![0u8; 4 * 3];
        p.fill_span(0, 10, 13, &mut out);
        for i in 0..4usize {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i ∈ [0,4); 10 + i ≤ 13 fits in u8"
            )]
            let expected = (10 + i) as u8;
            assert_eq!(out[i * 3], expected, "pixel {i} R");
            assert_eq!(out[i * 3 + 1], expected, "pixel {i} G");
            assert_eq!(out[i * 3 + 2], expected, "pixel {i} B");
        }
    }

    #[test]
    fn constant_function_fills_uniform_color() {
        let p = FunctionPattern::new(|_, _| Some([128, 64, 32]));
        let mut out = vec![0u8; 3 * 3];
        p.fill_span(5, 0, 2, &mut out);
        for i in 0..3usize {
            assert_eq!(out[i * 3], 128, "pixel {i} R");
            assert_eq!(out[i * 3 + 1], 64, "pixel {i} G");
            assert_eq!(out[i * 3 + 2], 32, "pixel {i} B");
        }
    }

    #[test]
    fn encodes_y_coordinate_in_output() {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test passes integer y = 42; cast is exact"
        )]
        let p = FunctionPattern::new(|_x, y| Some([0, 0, y as u8]));
        let mut out = [0u8; 3];
        p.fill_span(42, 0, 0, &mut out);
        assert_eq!(out[2], 42, "blue channel should encode y");
    }

    #[test]
    fn single_pixel_span_writes_three_bytes() {
        let p = FunctionPattern::new(|_, _| Some([1, 2, 3]));
        let mut out = [0u8; 3];
        p.fill_span(0, 7, 7, &mut out);
        assert_eq!(out, [1, 2, 3]);
    }

    #[test]
    fn pixels_outside_the_domain_are_not_painted() {
        let p = FunctionPattern::new(|x, _| (x < 2.0).then_some([9, 9, 9]));
        let mut out = [0u8; 4 * 3];
        let mut painted = [false; 4];
        p.fill_span_painted(0, 0, 3, &mut out, &mut painted);
        assert_eq!(painted, [true, true, false, false]);
        assert_eq!(&out[..6], &[9; 6]);
    }
}
