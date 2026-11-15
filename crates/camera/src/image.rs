//! Grayscale image types, box downsampling and inverse-mapped resampling.
//!
//! The images hold samples of a [`Sample`] type: 8-bit for detection, 8-
//! or 16-bit for the replayed resampling.  Lengths and strides count
//! samples.
//!
//! Coordinate convention: integer coordinates address pixel centres; the
//! continuous extent of a `w`-pixel row is `[-0.5, w - 0.5]`.

use crate::error::{CameraError, MAX_DIMENSION};
use crate::num::{floor_index, to_f64};
use crate::sample::Sample;

/// A borrowed grayscale image of samples `S` (8-bit by default): row `y`
/// is `data[y · stride .. y · stride + width]`, top-down.
///
/// Every function taking a view validates it first (see
/// [`validate`](Self::validate)).
#[derive(Debug, Clone, Copy)]
pub struct GrayView<'a, S: Sample = u8> {
    /// Samples; at least `stride · (height − 1) + width` long.
    pub data: &'a [S],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Distance in samples between the starts of consecutive rows
    /// (≥ `width`).
    pub stride: usize,
}

/// An owned, tightly packed grayscale image of samples `S` (8-bit by
/// default; `stride == width`), row-major, top-down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayPage<S: Sample = u8> {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width · height` samples.
    pub data: Vec<S>,
}

/// An axis-aligned pixel rectangle: columns `x .. x + width`, rows
/// `y .. y + height`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    /// First column.
    pub x: usize,
    /// First row.
    pub y: usize,
    /// Number of columns.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
}

impl<'a, S: Sample> GrayView<'a, S> {
    /// A view of a tightly packed buffer (`stride == width`).
    #[must_use]
    pub const fn packed(data: &'a [S], width: usize, height: usize) -> Self {
        Self {
            data,
            width,
            height,
            stride: width,
        }
    }

    /// Check the view's invariants: non-zero dimensions not above
    /// [`MAX_DIMENSION`], `stride ≥ width`, a buffer of at least
    /// `stride · (height − 1) + width` samples, and a pixel area whose double
    /// fits in `usize`.
    ///
    /// # Errors
    ///
    /// The first violated invariant, as a [`CameraError`].
    pub const fn validate(&self) -> Result<(), CameraError> {
        let (width, height, stride) = (self.width, self.height, self.stride);
        if width == 0 || height == 0 {
            return Err(CameraError::ZeroDimension { width, height });
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(CameraError::TooLarge { width, height });
        }
        if stride < width {
            return Err(CameraError::StrideTooSmall { stride, width });
        }
        let overflow = CameraError::Overflow {
            width,
            height,
            stride,
        };
        let Some(rows) = stride.checked_mul(height - 1) else {
            return Err(overflow);
        };
        let Some(required) = rows.checked_add(width) else {
            return Err(overflow);
        };
        // Rectified output may reach twice the source area.
        match width.checked_mul(height) {
            Some(area) if area.checked_mul(2).is_some() => {}
            _ => return Err(overflow),
        }
        if self.data.len() < required {
            return Err(CameraError::BufferTooShort {
                len: self.data.len(),
                required,
            });
        }
        Ok(())
    }

    /// [`validate`](Self::validate), then require the size `expected`.
    ///
    /// # Errors
    ///
    /// The first violated invariant, or [`CameraError::SizeMismatch`].
    pub(crate) fn validate_size(&self, expected: (usize, usize)) -> Result<(), CameraError> {
        self.validate()?;
        let actual = (self.width, self.height);
        if actual == expected {
            Ok(())
        } else {
            Err(CameraError::SizeMismatch { expected, actual })
        }
    }

    /// Row `y` as `width` samples.  Caller guarantees `y < height` on a
    /// validated view.
    pub(crate) fn row(&self, y: usize) -> &'a [S] {
        let start = y * self.stride;
        &self.data[start..start + self.width]
    }

    /// Value at `(x, y)`.  Caller guarantees `x < width`, `y < height`.
    pub(crate) const fn at(&self, x: usize, y: usize) -> S {
        self.data[y * self.stride + x]
    }

    /// The sub-view covering `rect`.  Caller guarantees `rect` lies inside
    /// the view and is non-empty.
    pub(crate) fn sub(&self, rect: Rect) -> Self {
        let start = rect.y * self.stride + rect.x;
        Self {
            data: &self.data[start..],
            width: rect.width,
            height: rect.height,
            stride: self.stride,
        }
    }

    /// Whether `rect` is non-empty and lies inside the view.
    pub(crate) const fn contains(&self, rect: Rect) -> bool {
        rect.width > 0
            && rect.height > 0
            && rect.x <= self.width
            && rect.y <= self.height
            && rect.width <= self.width - rect.x
            && rect.height <= self.height - rect.y
    }

    /// A packed copy of the view.
    pub(crate) fn to_page(self) -> GrayPage<S> {
        let mut data = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            data.extend_from_slice(self.row(y));
        }
        GrayPage {
            width: self.width,
            height: self.height,
            data,
        }
    }
}

impl GrayView<'_> {
    /// Smallest integer factor `f ≥ 1` with `max(w, h) / f ≤ max_side`.
    pub(crate) fn factor_for(&self, max_side: usize) -> usize {
        self.width.max(self.height).div_ceil(max_side.max(1)).max(1)
    }

    /// Box-filter downsample by an integer `factor`; trailing pixels that do
    /// not fill a whole block are discarded.  Downsampled pixel `i` covers
    /// source pixels `[i·f, (i+1)·f)`, so its centre is at `(i + 0.5)·f − 0.5`
    /// in source coordinates (see [`upscale_coord`]).
    pub(crate) fn downsample(&self, factor: usize) -> GrayPage {
        if factor <= 1 {
            return self.to_page();
        }
        let (w, h) = (self.width / factor, self.height / factor);
        let area = u64::try_from(factor * factor).unwrap_or(u64::MAX);
        let mut data = vec![0u8; w * h];
        let mut acc = vec![0u64; w];
        for (y, out) in data.chunks_exact_mut(w.max(1)).enumerate() {
            acc.fill(0);
            for sy in y * factor..(y + 1) * factor {
                let src = self.row(sy);
                for (a, block) in acc.iter_mut().zip(src.chunks_exact(factor)) {
                    *a += block.iter().map(|&v| u64::from(v)).sum::<u64>();
                }
            }
            for (o, &a) in out.iter_mut().zip(&acc) {
                *o = u8::try_from((a + area / 2) / area).unwrap_or(u8::MAX);
            }
        }
        GrayPage {
            width: w,
            height: h,
            data,
        }
    }

    /// Bilinear sample at continuous `(x, y)`, clamped to the image.
    pub(crate) fn sample(&self, x: f64, y: f64) -> f64 {
        let xc = x.clamp(0.0, to_f64(self.width - 1));
        let yc = y.clamp(0.0, to_f64(self.height - 1));
        let (x0, y0) = (floor_index(xc), floor_index(yc));
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (xc - to_f64(x0), yc - to_f64(y0));
        let top = f64::from(self.at(x0, y0)).mul_add(1.0 - fx, f64::from(self.at(x1, y0)) * fx);
        let bot = f64::from(self.at(x0, y1)).mul_add(1.0 - fx, f64::from(self.at(x1, y1)) * fx);
        top.mul_add(1.0 - fy, bot * fy)
    }
}

impl<S: Sample> GrayPage<S> {
    /// A view of the page.
    #[must_use]
    pub fn view(&self) -> GrayView<'_, S> {
        GrayView::packed(&self.data, self.width, self.height)
    }
}

/// Map a coordinate of an image downsampled by `factor` back to the source
/// pixel grid.
pub(crate) fn upscale_coord(v: f64, factor: usize) -> f64 {
    to_f64(factor).mul_add(v + 0.5, -0.5)
}

/// Build an `out_w × out_h` page whose pixel `(x, y)` is the bilinear
/// sample of `src` at `map(x, y)`.
///
/// Samples outside the continuous extent of `src` are white
/// ([`Sample::MAX`]).  Inside it, the 2×2 footprint is clamped to the edge
/// pixels and the blend is rounded half up ([`Sample::from_f64`]).  Caller
/// guarantees `out_w, out_h ≥ 1` and a validated `src`.
pub(crate) fn resample<S, F>(
    src: GrayView<'_, S>,
    out_w: usize,
    out_h: usize,
    map: F,
) -> GrayPage<S>
where
    S: Sample,
    F: Fn(f64, f64) -> (f64, f64),
{
    let mut data = vec![S::MAX; out_w * out_h];
    let (sw, sh) = (src.width, src.height);
    let (max_x, max_y) = (to_f64(sw) - 0.5, to_f64(sh) - 0.5);
    for (oy, row) in data.chunks_exact_mut(out_w.max(1)).enumerate() {
        for (ox, px) in row.iter_mut().enumerate() {
            let (sx, sy) = map(to_f64(ox), to_f64(oy));
            if !(sx >= -0.5 && sy >= -0.5 && sx <= max_x && sy <= max_y) {
                continue;
            }
            let xc = sx.clamp(0.0, to_f64(sw - 1));
            let yc = sy.clamp(0.0, to_f64(sh - 1));
            let (x0, y0) = (floor_index(xc), floor_index(yc));
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (fx, fy) = (xc - to_f64(x0), yc - to_f64(y0));
            let (r0, r1) = (src.row(y0), src.row(y1));
            let top = r0[x0].to_f64().mul_add(1.0 - fx, r0[x1].to_f64() * fx);
            let bot = r1[x0].to_f64().mul_add(1.0 - fx, r1[x1].to_f64() * fx);
            *px = S::from_f64(top.mul_add(1.0 - fy, bot * fy));
        }
    }
    GrayPage {
        width: out_w,
        height: out_h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::plane;

    #[test]
    fn downsample_averages_blocks() {
        let p = plane(4, 2, |x, _| if x < 2 { 0 } else { 200 });
        let d = p.view().downsample(2);
        assert_eq!((d.width, d.height), (2, 1));
        assert_eq!(d.data, vec![0, 200]);
    }

    #[test]
    fn downsample_honours_stride() {
        // 2×2 image in a stride-3 buffer; the padding column is never read.
        let data = [10u8, 30, 255, 50, 70];
        let v = GrayView {
            data: &data,
            width: 2,
            height: 2,
            stride: 3,
        };
        assert_eq!(v.downsample(2).data, vec![40]);
    }

    #[test]
    fn resample_identity_preserves_pixels() {
        let src = plane(3, 2, |x, y| u8::try_from(x * 13 + y * 50).unwrap_or(0));
        let dst = resample(src.view(), 3, 2, |x, y| (x, y));
        assert_eq!(dst, src);
    }

    #[test]
    fn resample_outside_is_white() {
        let src = plane(2, 2, |_, _| 0);
        let dst = resample(src.view(), 2, 1, |x, y| (x + 10.0, y));
        assert_eq!(dst.data, vec![255, 255]);
        let deep = GrayPage::<u16> {
            width: 2,
            height: 2,
            data: vec![0, 1000, 2000, 3000],
        };
        let out = resample(deep.view(), 3, 1, |x, y| (x * 0.5, y + 0.5));
        // x = 0, 0.5 at row 0.5: the blends 1000 and 1500; x = 1 → 2000.
        assert_eq!(out.data, vec![1000, 1500, 2000]);
        let outside = resample(deep.view(), 1, 1, |x, y| (x + 10.0, y));
        assert_eq!(outside.data, vec![u16::MAX]);
    }

    #[test]
    fn sub_view_and_copy() {
        let src = plane(4, 3, |x, y| u8::try_from(10 * y + x).unwrap_or(0));
        let rect = Rect {
            x: 1,
            y: 1,
            width: 2,
            height: 2,
        };
        assert!(src.view().contains(rect));
        assert_eq!(src.view().sub(rect).to_page().data, vec![11, 12, 21, 22]);
        let past = Rect { x: 3, ..rect };
        assert!(!src.view().contains(past));
    }

    #[test]
    fn validation_rejects_malformed_views() {
        let data = [0u8; 10];
        let v = |width, height, stride| GrayView {
            data: &data,
            width,
            height,
            stride,
        };
        assert!(v(5, 2, 5).validate().is_ok());
        assert!(matches!(
            v(0, 2, 5).validate(),
            Err(CameraError::ZeroDimension { .. })
        ));
        assert!(matches!(
            v(5, 2, 4).validate(),
            Err(CameraError::StrideTooSmall { .. })
        ));
        assert!(matches!(
            v(5, 2, 6).validate(),
            Err(CameraError::BufferTooShort {
                len: 10,
                required: 11
            })
        ));
        assert!(matches!(
            v(2, 3, usize::MAX).validate(),
            Err(CameraError::Overflow { .. })
        ));
        assert!(matches!(
            v(MAX_DIMENSION + 1, 1, MAX_DIMENSION + 1).validate(),
            Err(CameraError::TooLarge { .. })
        ));
        // Lengths count samples, not bytes.
        let deep = [0u16; 10];
        assert!(GrayView::packed(&deep, 5, 2).validate().is_ok());
        assert!(matches!(
            GrayView {
                data: &deep,
                width: 5,
                height: 2,
                stride: 6
            }
            .validate(),
            Err(CameraError::BufferTooShort {
                len: 10,
                required: 11
            })
        ));
    }
}
