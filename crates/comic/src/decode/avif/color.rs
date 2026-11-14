//! Decoded AV1 planes to 8-bit gray or RGB.
//!
//! The conversion is integer arithmetic throughout.  With `d` the bit depth
//! and `S = 6 + d`, a sample triple (Y, U, V) maps to
//!
//! ```text
//! C = clamp((cy·(Y − y0) + cu·(U − 2^(d−1)) + cv·(V − 2^(d−1)) + 2^(S−1)) >> S, 0, 255)
//! ```
//!
//! per output channel `C`, where `cy`, `cu`, `cv` are the matrix
//! coefficients scaled by `255 · 2^S` over the luma or chroma range, rounded
//! to nearest.  The reduction from 10 or 12 bits to 8 is part of the same
//! rounding, never a separate truncation.  Full-range 8-bit luma has
//! `cy = 2^14` exactly, so a pixel without chroma keeps its value.
//!
//! Subsampled chroma is brought to luma resolution by bilinear
//! interpolation with chroma sited between luma samples: weights 3:1 along
//! each halved axis, hence 9:3:3:1 for 4:2:0, edges replicated.

use std::borrow::Cow;

use rav1d::StillLayout;

use super::DecodeError;

/// A plane sample: 8 bits, or 10 and 12 bits in 16.
pub(super) trait Sample: Copy + Default + Into<i32> + Into<u16> {
    /// The samples as bytes when they are bytes.
    fn as_bytes(samples: &[Self]) -> Option<&[u8]>;
}

impl Sample for u8 {
    fn as_bytes(samples: &[Self]) -> Option<&[u8]> {
        Some(samples)
    }
}

impl Sample for u16 {
    fn as_bytes(_: &[Self]) -> Option<&[u8]> {
        None
    }
}

/// One plane: borrowed from the decoder's picture, or owned.  Row `y` is
/// the `width` samples from index `y · stride`, and `data` holds at least
/// `(height − 1) · stride + width` samples when the plane is not empty.
#[derive(Debug)]
pub(super) struct Plane<'a, T: Clone> {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) stride: usize,
    pub(super) data: Cow<'a, [T]>,
}

impl<T: Sample> Plane<'_, T> {
    /// A zeroed plane without row padding.
    pub(super) fn zeroed(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            stride: width,
            data: Cow::Owned(vec![T::default(); width * height]),
        }
    }

    pub(super) fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.stride..][..self.width]
    }
}

/// A picture in the decoder's colour model.  Invariant: plane 0 is
/// `width × height`; planes 1 and 2 are empty for [`StillLayout::Mono`] and
/// otherwise `chroma_size(width, height)`.
#[derive(Debug)]
pub(super) struct Frame<'a, T: Clone> {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) depth: u8,
    pub(super) layout: StillLayout,
    pub(super) planes: [Plane<'a, T>; 3],
}

/// The size of a chroma plane under `layout` for `width × height` luma.
pub(super) const fn chroma_size(
    layout: StillLayout,
    width: usize,
    height: usize,
) -> (usize, usize) {
    if matches!(layout, StillLayout::Mono) {
        return (0, 0);
    }
    let (sx, sy) = layout.chroma_shift();
    (
        (width + (1 << sx) - 1) >> sx,
        (height + (1 << sy) - 1) >> sy,
    )
}

/// How samples map to colour: ITU-T H.273 code points and the range flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ColorSpec {
    pub(super) primaries: u16,
    pub(super) matrix: u16,
    pub(super) full_range: bool,
}

/// Tight 8-bit output.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Pixels {
    Gray(Vec<u8>),
    Rgb(Vec<u8>),
}

/// The luma weights (Kr, Kb) of matrix code point `matrix`, `None` for the
/// code points that are not of the Y′CbCr form.
///
/// Code point 2 (unspecified) and those without an implementation here read
/// as BT.601, the convention for untagged pictures.  Code point 12 derives
/// the weights from the primaries.
const fn luma_weights(matrix: u16, primaries: u16) -> (f64, f64) {
    match matrix {
        1 => (0.2126, 0.0722),
        4 => (0.30, 0.11),
        7 => (0.212, 0.087),
        9 => (0.2627, 0.0593),
        12 => match primaries {
            1 => (0.2126, 0.0722),
            5 => (0.2220, 0.0713),
            6 | 7 => (0.2124, 0.0866),
            9 => (0.2627, 0.0593),
            12 => (0.2290, 0.0793),
            _ => (0.299, 0.114),
        },
        _ => (0.299, 0.114),
    }
}

/// Fixed-point conversion constants for one bit depth, range and matrix.
#[derive(Clone, Copy, Debug)]
struct Matrix {
    /// Luma black level.
    y0: i32,
    /// Chroma zero level, `2^(d−1)`.
    c0: i32,
    /// Luma gain.
    cy: i32,
    /// Per output channel (R, G, B): the gains of U and V.
    c: [[i32; 2]; 3],
    /// `2^(S−1)`.
    round: i32,
    /// `S`.
    shift: u32,
}

impl Matrix {
    fn new(depth: u8, spec: ColorSpec) -> Self {
        let up = u32::from(depth) - 8;
        let shift = 14 + up;
        let max = (1i32 << depth) - 1;
        let (y0, y_range, c_range) = if spec.full_range {
            (0, max, max)
        } else {
            (16 << up, 219 << up, 224 << up)
        };
        let one = f64::from(1u32 << shift) * 255.0;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "gains are below 2^17: |k| ≤ 2, 255 / range ≤ 2^(8−d) · 1.17, scale 2^(6+d)"
        )]
        let fixed = |k: f64, range: i32| (k * one / f64::from(range)).round() as i32;
        let chroma = if spec.matrix == 8 {
            // YCgCo: U carries Cg and V carries Co.
            [[-1.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]
        } else {
            let (kr, kb) = luma_weights(spec.matrix, spec.primaries);
            let kg = 1.0 - kr - kb;
            [
                [0.0, 2.0 * (1.0 - kr)],
                [-2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg],
                [2.0 * (1.0 - kb), 0.0],
            ]
        };
        Self {
            y0,
            c0: 1 << (depth - 1),
            cy: fixed(1.0, y_range),
            c: chroma.map(|row| row.map(|k| fixed(k, c_range))),
            round: 1 << (shift - 1),
            shift,
        }
    }

    /// A luma-scaled sample alone: gray, or one channel of an identity
    /// matrix picture.
    #[inline]
    fn level<T: Sample>(&self, v: T) -> u8 {
        clamp((self.cy * (Into::<i32>::into(v) - self.y0) + self.round) >> self.shift)
    }

    /// Whether [`Self::level`] is the identity on 8-bit samples.
    const fn level_is_identity(&self) -> bool {
        self.shift == 14 && self.y0 == 0
    }

    /// One row of Y′CbCr to interleaved RGB; `u` and `v` are at luma
    /// resolution.
    fn rgb_row<Y: Sample, C: Sample>(&self, y: &[Y], u: &[C], v: &[C], out: &mut [u8]) {
        let (out, _) = out.as_chunks_mut::<3>();
        for (((px, &y), &u), &v) in out.iter_mut().zip(y).zip(u).zip(v) {
            let luma = self.cy * (Into::<i32>::into(y) - self.y0) + self.round;
            let cb = Into::<i32>::into(u) - self.c0;
            let cr = Into::<i32>::into(v) - self.c0;
            for (channel, gains) in px.iter_mut().zip(&self.c) {
                *channel = clamp((luma + gains[0] * cb + gains[1] * cr) >> self.shift);
            }
        }
    }
}

/// `v` saturated to `0..=255`.
#[inline]
fn clamp(v: i32) -> u8 {
    u8::try_from(v.clamp(0, 255)).unwrap_or(u8::MAX)
}

/// Interpolate one chroma row to `out.len()` luma columns.
///
/// `near` is the chroma row covering the output row and `far` the next
/// nearest one (`None` when chroma is not halved vertically).  `mix` is
/// scratch of the chroma width, `ceil(out.len() / 2)`.  Vertically
/// `mix = 3·near + far` (or `4·near`); horizontally each output is
/// `(3·mix[i] + mix[i ± 1] + 8) >> 4` with the neighbour on the output's
/// side, replicated at the edges.  Every intermediate fits 16 bits:
/// `mix ≤ 4·4095` and `4·mix + 8 ≤ 65 528`.
fn upsample_row<T: Sample>(near: &[T], far: Option<&[T]>, mix: &mut [u16], out: &mut [u16]) {
    match far {
        Some(far) => {
            for ((m, &n), &f) in mix.iter_mut().zip(near).zip(far) {
                *m = 3 * Into::<u16>::into(n) + Into::<u16>::into(f);
            }
        }
        None => {
            for (m, &n) in mix.iter_mut().zip(near) {
                *m = 4 * Into::<u16>::into(n);
            }
        }
    }
    let Some(last) = mix.len().checked_sub(1) else {
        return;
    };
    let blend = |centre: u16, side: u16| (3 * centre + side + 8) >> 4;
    // Outputs 2i and 2i + 1 lie in chroma sample i, towards i − 1 and
    // i + 1.  The interior pairs read a window of three and carry no
    // edge test.
    let (pairs, tail) = out.as_chunks_mut::<2>();
    if let Some(first) = pairs.first_mut() {
        *first = [blend(mix[0], mix[0]), blend(mix[0], mix[last.min(1)])];
    }
    for (pair, window) in pairs.iter_mut().skip(1).zip(mix.windows(3)) {
        *pair = [blend(window[1], window[0]), blend(window[1], window[2])];
    }
    if last > 0 {
        let before = mix[last - 1];
        if let Some(pair) = pairs.get_mut(last) {
            *pair = [blend(mix[last], before), blend(mix[last], mix[last])];
        }
        if let [odd] = tail {
            *odd = blend(mix[last], before);
        }
    } else if let [odd] = tail {
        *odd = blend(mix[0], mix[0]);
    }
}

/// Convert `frame` to 8-bit pixels: gray for a monochrome picture, RGB
/// otherwise.
///
/// # Errors
///
/// [`DecodeError::Codec`] for an identity (GBR) matrix on subsampled
/// chroma, and for the reversible `YCgCo` matrices (code points 16 and 17),
/// which are not implemented.
pub(super) fn to_pixels<T: Sample>(
    frame: &Frame<'_, T>,
    spec: ColorSpec,
) -> Result<Pixels, DecodeError> {
    let matrix = Matrix::new(frame.depth, spec);
    let (w, h) = (frame.width, frame.height);
    let [luma, u, v] = &frame.planes;
    if frame.layout == StillLayout::Mono {
        let mut gray = Vec::with_capacity(w * h);
        for y in 0..h {
            let row = luma.row(y);
            match T::as_bytes(row) {
                Some(bytes) if matrix.level_is_identity() => gray.extend_from_slice(bytes),
                _ => gray.extend(row.iter().map(|&s| matrix.level(s))),
            }
        }
        return Ok(Pixels::Gray(gray));
    }
    if matches!(spec.matrix, 16 | 17) {
        return Err(DecodeError::Codec(format!(
            "avif: unsupported matrix coefficients {}",
            spec.matrix
        )));
    }
    let mut out = vec![0u8; w * h * 3];
    let rows = out.chunks_exact_mut((w * 3).max(1));
    if spec.matrix == 0 {
        if frame.layout != StillLayout::I444 {
            return Err(DecodeError::Codec(
                "avif: identity matrix on subsampled chroma".to_owned(),
            ));
        }
        // G, B, R are carried as Y, U, V.
        for (y, row) in rows.enumerate() {
            let (row, _) = row.as_chunks_mut::<3>();
            let planes = luma.row(y).iter().zip(u.row(y)).zip(v.row(y));
            for (px, ((&g, &b), &r)) in row.iter_mut().zip(planes) {
                *px = [matrix.level(r), matrix.level(g), matrix.level(b)];
            }
        }
        return Ok(Pixels::Rgb(out));
    }
    if frame.layout == StillLayout::I444 {
        for (y, row) in rows.enumerate() {
            matrix.rgb_row(luma.row(y), u.row(y), v.row(y), row);
        }
        return Ok(Pixels::Rgb(out));
    }
    let halved_rows = frame.layout == StillLayout::I420;
    let mut mix = vec![0u16; u.width];
    let (mut wide_u, mut wide_v) = (vec![0u16; w], vec![0u16; w]);
    for (y, row) in rows.enumerate() {
        // Output row `y` lies in chroma row `y / 2`, nearer the row above
        // when `y` is even and the row below when odd.
        let (near, far) = if halved_rows {
            let near = y / 2;
            let far = if y % 2 == 0 {
                near.saturating_sub(1)
            } else {
                (near + 1).min(u.height - 1)
            };
            (near, Some(far))
        } else {
            (y, None)
        };
        upsample_row(u.row(near), far.map(|f| u.row(f)), &mut mix, &mut wide_u);
        upsample_row(v.row(near), far.map(|f| v.row(f)), &mut mix, &mut wide_v);
        matrix.rgb_row(luma.row(y), &wide_u, &wide_v, row);
    }
    Ok(Pixels::Rgb(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(matrix: u16, full_range: bool) -> ColorSpec {
        ColorSpec {
            primaries: 2,
            matrix,
            full_range,
        }
    }

    fn frame<T: Sample>(
        layout: StillLayout,
        depth: u8,
        (width, height): (usize, usize),
        planes: [Vec<T>; 3],
    ) -> Frame<'static, T> {
        let chroma = chroma_size(layout, width, height);
        let mut sizes = [(width, height), chroma, chroma].into_iter();
        Frame {
            width,
            height,
            depth,
            layout,
            planes: planes.map(|data| {
                let (width, height) = sizes.next().unwrap();
                Plane {
                    width,
                    height,
                    stride: width,
                    data: Cow::Owned(data),
                }
            }),
        }
    }

    fn rgb<T: Sample>(f: &Frame<'_, T>, spec: ColorSpec) -> Vec<u8> {
        match to_pixels(f, spec).unwrap() {
            Pixels::Rgb(v) => v,
            Pixels::Gray(_) => panic!("a colour frame converts to RGB"),
        }
    }

    #[test]
    fn neutral_chroma_keeps_full_range_luma_and_scales_limited_range() {
        let f = |range| {
            frame::<u8>(
                StillLayout::I444,
                8,
                (4, 1),
                [vec![0, 16, 235, 255], vec![128; 4], vec![128; 4]],
            )
            .pipe(|f| rgb(&f, spec(1, range)))
        };
        assert_eq!(f(true), [0, 0, 0, 16, 16, 16, 235, 235, 235, 255, 255, 255]);
        assert_eq!(f(false), [0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]);
    }

    trait Pipe: Sized {
        fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
            f(self)
        }
    }
    impl<T> Pipe for T {}

    #[test]
    fn matrices_reproduce_their_primaries() {
        // Full-range Y′CbCr of pure red under each matrix:
        // Y = Kr, Cb = −Kr / (2 (1 − Kb)), Cr = 1/2.
        for (matrix, kr, kb) in [
            (6u16, 0.299, 0.114),
            (2, 0.299, 0.114),
            (1, 0.2126, 0.0722),
            (9, 0.2627, 0.0593),
        ] {
            let y = (255.0f64 * kr).round();
            let cb = (128.0 - 255.0 * kr / (2.0 * (1.0 - kb))).round();
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "0..=255"
            )]
            let f = frame::<u8>(
                StillLayout::I444,
                8,
                (1, 1),
                [vec![y as u8], vec![cb as u8], vec![255]],
            );
            let px = rgb(&f, spec(matrix, true));
            assert!(
                px[0] >= 253 && px[1] <= 2 && px[2] <= 2,
                "matrix {matrix}: {px:?}"
            );
        }
    }

    #[test]
    fn deep_samples_round_to_eight_bits() {
        // 10-bit full range: 1023 → 255, 512 → 128 (127.6), 2 → 0 (0.499), 3 → 1.
        let f = frame::<u16>(
            StillLayout::Mono,
            10,
            (4, 1),
            [vec![1023, 512, 2, 3], vec![], vec![]],
        );
        assert_eq!(
            to_pixels(&f, spec(2, true)).unwrap(),
            Pixels::Gray(vec![255, 128, 0, 1])
        );
        // 12-bit limited range: black 256, white 3760.
        let f = frame::<u16>(
            StillLayout::Mono,
            12,
            (3, 1),
            [vec![256, 3760, 2010], vec![], vec![]],
        );
        assert_eq!(
            to_pixels(&f, spec(2, false)).unwrap(),
            Pixels::Gray(vec![0, 255, 128])
        );
    }

    #[test]
    fn identity_matrix_carries_gbr_and_needs_full_chroma() {
        let f = frame::<u8>(
            StillLayout::I444,
            8,
            (2, 1),
            [vec![10, 20], vec![30, 40], vec![50, 60]],
        );
        assert_eq!(rgb(&f, spec(0, true)), [50, 10, 30, 60, 20, 40]);
        let f = frame::<u8>(StillLayout::I420, 8, (2, 2), [vec![0; 4], vec![0], vec![0]]);
        assert!(to_pixels(&f, spec(0, true)).is_err());
    }

    #[test]
    fn ycgco_inverts() {
        // R, G, B = 200, 100, 40: Y = 110, Cg = −10, Co = 80 (offset 128).
        let f = frame::<u8>(
            StillLayout::I444,
            8,
            (1, 1),
            [vec![110], vec![118], vec![208]],
        );
        assert_eq!(rgb(&f, spec(8, true)), [200, 100, 40]);
    }

    #[test]
    fn chroma_interpolates_with_three_to_one_weights() {
        let mut mix = [0u16; 2];
        let mut out = [0u16; 4];
        // One chroma row, halved horizontally only: 4·c, then 3:1.
        upsample_row::<u8>(&[0, 160], None, &mut mix, &mut out);
        assert_eq!(out, [0, 40, 120, 160]);
        // Halved both ways: near 3 parts, far 1 part.
        upsample_row::<u8>(&[0, 160], Some(&[160, 0]), &mut mix, &mut out);
        assert_eq!(out, [40, 60, 100, 120]);
        // An odd luma width ends on a left sample.
        let mut odd = [0u16; 3];
        upsample_row::<u8>(&[0, 160], None, &mut mix, &mut odd);
        assert_eq!(odd, [0, 40, 120]);
    }

    /// A 64-bit linear congruential generator (Knuth's MMIX constants);
    /// the high bits are the output.
    fn noise(state: &mut u64) -> u16 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u16::try_from(*state >> 48).unwrap()
    }

    #[test]
    fn interpolation_matches_its_definition_on_random_rows() {
        // The definition, one output at a time with clamped neighbours.
        let reference = |near: &[u16], far: Option<&[u16]>, width: usize| -> Vec<u16> {
            let mix = |i: usize| {
                let i = i.min(near.len() - 1);
                let far = far.map_or(near[i], |far| far[i]);
                3 * u32::from(near[i]) + u32::from(far)
            };
            (0..width)
                .map(|x| {
                    let side = if x % 2 == 0 {
                        (x / 2).saturating_sub(1)
                    } else {
                        x / 2 + 1
                    };
                    u16::try_from((3 * mix(x / 2) + mix(side) + 8) >> 4).unwrap()
                })
                .collect()
        };
        let mut state = 1;
        for width in 1..=67usize {
            for depth in [8u32, 10, 12] {
                let chroma = width.div_ceil(2);
                let mut row = || -> Vec<u16> {
                    (0..chroma)
                        .map(|_| noise(&mut state) >> (16 - depth))
                        .collect()
                };
                let (near, far) = (row(), row());
                let mut mix = vec![0u16; chroma];
                let mut out = vec![0u16; width];
                for far in [None, Some(far.as_slice())] {
                    upsample_row(&near, far, &mut mix, &mut out);
                    assert_eq!(
                        out,
                        reference(&near, far, width),
                        "{width} wide, {depth} bits"
                    );
                }
                if depth == 8 {
                    let narrow = |row: &[u16]| -> Vec<u8> {
                        row.iter().map(|&v| u8::try_from(v).unwrap()).collect()
                    };
                    let (near8, far8) = (narrow(&near), narrow(&far));
                    upsample_row(&near8, Some(&far8), &mut mix, &mut out);
                    assert_eq!(out, reference(&near, Some(&far), width), "{width} wide");
                }
            }
        }
    }

    #[test]
    fn subsampled_frames_convert_at_every_small_size() {
        for layout in [StillLayout::I420, StillLayout::I422] {
            for w in 1..=5 {
                for h in 1..=5 {
                    let (cw, ch) = chroma_size(layout, w, h);
                    let f = frame::<u8>(
                        layout,
                        8,
                        (w, h),
                        [vec![90; w * h], vec![128; cw * ch], vec![128; cw * ch]],
                    );
                    assert_eq!(
                        rgb(&f, spec(1, true)),
                        vec![90; w * h * 3],
                        "{layout:?} {w}x{h}"
                    );
                }
            }
        }
    }
}
