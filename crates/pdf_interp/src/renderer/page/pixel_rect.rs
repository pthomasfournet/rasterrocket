//! Rectangles of device pixels.

use raster::Clip;

use super::PageRenderer;

/// A non-empty rectangle of device pixels: columns `x0..x1`, rows `y0..y1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PixelRect {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl PixelRect {
    /// The rectangle `x0..x1 × y0..y1`, or `None` when it is empty.
    pub fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Self> {
        (x0 < x1 && y0 < y1).then_some(Self { x0, y0, x1, y1 })
    }

    /// The pixels `clip` admits within a `width × height` bitmap.
    pub fn of_clip(clip: &Clip, width: u32, height: u32) -> Option<Self> {
        let bound = |v: i64, len: u32| u32::try_from(v.clamp(0, i64::from(len))).ok();
        Self::new(
            bound(i64::from(clip.x_min_i), width)?,
            bound(i64::from(clip.y_min_i), height)?,
            bound(i64::from(clip.x_max_i) + 1, width)?,
            bound(i64::from(clip.y_max_i) + 1, height)?,
        )
    }

    /// The pixels of a `width × height` bitmap within `margin` of the
    /// bounding box of the finite `points`; `None` when there are none.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the bounds are clamped to [0, dimension] before the casts"
    )]
    pub fn around(
        points: impl IntoIterator<Item = (f64, f64)>,
        margin: f64,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in points {
            if x.is_finite() && y.is_finite() {
                (min_x, max_x) = (min_x.min(x), max_x.max(x));
                (min_y, max_y) = (min_y.min(y), max_y.max(y));
            }
        }
        if min_x > max_x {
            return None;
        }
        let lo = |v: f64, len: u32| (v - margin).floor().clamp(0.0, f64::from(len)) as u32;
        let hi = |v: f64, len: u32| (v + margin).ceil().clamp(0.0, f64::from(len)) as u32;
        Self::new(
            lo(min_x, width),
            lo(min_y, height),
            hi(max_x, width),
            hi(max_y, height),
        )
    }

    /// The pixels in both rectangles.
    pub fn intersect(self, other: Self) -> Option<Self> {
        Self::new(
            self.x0.max(other.x0),
            self.y0.max(other.y0),
            self.x1.min(other.x1),
            self.y1.min(other.y1),
        )
    }

    pub const fn width(self) -> u32 {
        self.x1 - self.x0
    }

    pub const fn height(self) -> u32 {
        self.y1 - self.y0
    }

    pub const fn area(self) -> u64 {
        self.width() as u64 * self.height() as u64
    }

    /// The rectangle as a closed device-space path.
    pub fn to_path(self) -> raster::path::Path {
        let (x0, y0) = (f64::from(self.x0), f64::from(self.y0));
        let (x1, y1) = (f64::from(self.x1), f64::from(self.y1));
        super::shading_paint::polygon(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }
}

impl PageRenderer<'_> {
    /// The pixels a painting operator can touch: the current clip's
    /// rectangle, restricted to within `margin` of the device-space
    /// `points` of the object.
    pub(super) fn paint_region(
        &self,
        points: impl IntoIterator<Item = (f64, f64)>,
        margin: f64,
    ) -> Option<PixelRect> {
        self.within_clip(PixelRect::around(points, margin, self.width, self.height)?)
    }

    /// The current clip's rectangle within the bitmap.
    pub(super) fn clip_rect(&self) -> Option<PixelRect> {
        PixelRect::of_clip(&self.gstate.current().clip, self.width, self.height)
    }

    /// The pixels of `rect` the current clip's rectangle admits.
    pub(super) fn within_clip(&self, rect: PixelRect) -> Option<PixelRect> {
        rect.intersect(self.clip_rect()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn around_bounds_the_points_with_a_margin_within_the_bitmap() {
        let r = PixelRect::around([(2.5, 3.0), (7.2, 1.5), (f64::NAN, 0.0)], 1.0, 8, 8).unwrap();
        assert_eq!((r.x0, r.y0, r.x1, r.y1), (1, 0, 8, 4));
        assert_eq!((r.width(), r.height(), r.area()), (7, 4, 28));
        assert!(PixelRect::around([(9.0, 9.0)], 0.0, 8, 8).is_none());
        assert!(PixelRect::around([(f64::NAN, 0.0)], 1.0, 8, 8).is_none());
    }

    #[test]
    fn of_clip_takes_the_clip_rectangle_within_the_bitmap() {
        let clip = Clip::new(-3.0, 2.0, 5.5, 100.0, true);
        let r = PixelRect::of_clip(&clip, 10, 10).unwrap();
        assert_eq!((r.x0, r.y0, r.x1, r.y1), (0, 2, 6, 10));
        let off = Clip::new(20.0, 0.0, 30.0, 5.0, true);
        assert!(PixelRect::of_clip(&off, 10, 10).is_none());
    }

    #[test]
    fn intersect_is_empty_for_disjoint_rectangles() {
        let a = PixelRect::new(0, 0, 4, 4).unwrap();
        let b = PixelRect::new(2, 2, 6, 6).unwrap();
        assert_eq!(a.intersect(b), PixelRect::new(2, 2, 4, 4));
        assert!(a.intersect(PixelRect::new(4, 0, 5, 5).unwrap()).is_none());
    }
}
