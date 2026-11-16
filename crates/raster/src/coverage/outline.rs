//! An outline as the exact-area rasteriser takes it: fixed-point segments
//! and the pixel window they are rasterised over.

use super::{AreaRaster, CoverageTable, fixed};
use crate::xpath::{XPath, XPathFlags};

/// The input of one exact-area rasterisation.
///
/// It holds the segments of a closed outline in the rasteriser's fixed
/// point, the inclusive pixel window whose coverage it yields, the even-odd
/// rule or non-zero, and a weight out of 255 the coverage is scaled by.
///
/// A segment `[x0, y0, x1, y1]` holds `floor(256 · v)` of its device
/// coordinates `v`, clamped as the rasteriser clamps them, in the
/// direction the outline runs.  Segments with a NaN coordinate are left
/// out, as the rasteriser skips them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AreaOutline {
    /// The window `(x0, y0, x1, y1)`, inclusive, non-empty.
    pub window: (i32, i32, i32, i32),
    /// The segments, fixed point.
    pub segs: Vec<[i64; 4]>,
    /// Whether the even-odd rule applies, else non-zero.
    pub even_odd: bool,
    /// The coverage of every pixel is scaled by `weight / 255`.
    pub weight: u8,
}

impl AreaOutline {
    /// The outline of the device-space `xpath` over `window`.
    #[must_use]
    pub fn of(xpath: &XPath, window: (i32, i32, i32, i32), even_odd: bool, weight: u8) -> Self {
        let segs = xpath
            .segs
            .iter()
            .filter(|s| ![s.x0, s.y0, s.x1, s.y1].iter().any(|v| v.is_nan()))
            .map(|s| {
                let (a, b) = ((s.x0, s.y0), (s.x1, s.y1));
                let (from, to) = if s.flags.contains(XPathFlags::FLIPPED) {
                    (b, a)
                } else {
                    (a, b)
                };
                [fixed(from.0), fixed(from.1), fixed(to.0), fixed(to.1)]
            })
            .collect();
        Self {
            window,
            segs,
            even_odd,
            weight,
        }
    }

    /// The rasteriser over the window with every segment added.
    #[must_use]
    pub fn raster(&self) -> AreaRaster {
        let (x0, y0, x1, y1) = self.window;
        let mut raster = AreaRaster::new(x0, y0, x1, y1);
        for &[ax, ay, bx, by] in &self.segs {
            raster.add_fixed(ax, ay, bx, by);
        }
        raster
    }

    /// The coverage the outline yields, scaled by its weight.
    #[must_use]
    pub fn table(&self) -> CoverageTable {
        let mut table = CoverageTable::empty();
        let weight = self.weight;
        self.raster().rows_mut(self.even_odd, |y, x, coverage| {
            scale(coverage, weight);
            table.push_row(y, x, coverage);
        });
        table
    }
}

/// `coverage` scaled by `weight / 255`, rounded as the compositing pipe
/// rounds.
pub(crate) fn scale(coverage: &mut [u8], weight: u8) {
    if weight != u8::MAX {
        for c in coverage.iter_mut() {
            *c = color::convert::div255(u32::from(*c) * u32::from(weight));
        }
    }
}
