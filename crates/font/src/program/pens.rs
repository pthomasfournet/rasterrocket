//! Outline pens internal to the program queries.

use skrifa::outline::OutlinePen;

/// Pen accumulating the box of every point it receives.
#[derive(Default)]
pub(super) struct BoundsPen {
    /// `[x_min, y_min, x_max, y_max]`; `None` before the first point.
    pub(super) bounds: Option<[f32; 4]>,
}

impl BoundsPen {
    fn add(&mut self, x: f32, y: f32) {
        let [x0, y0, x1, y1] = self.bounds.get_or_insert([x, y, x, y]);
        *x0 = x0.min(x);
        *y0 = y0.min(y);
        *x1 = x1.max(x);
        *y1 = y1.max(y);
    }
}

impl OutlinePen for BoundsPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.add(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.add(x, y);
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.add(cx0, cy0);
        self.add(x, y);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.add(cx0, cy0);
        self.add(cx1, cy1);
        self.add(x, y);
    }
    fn close(&mut self) {}
}

/// Pen that discards every command; used to evaluate charstring widths.
pub(super) struct NullPen;

impl OutlinePen for NullPen {
    fn move_to(&mut self, _x: f32, _y: f32) {}
    fn line_to(&mut self, _x: f32, _y: f32) {}
    fn quad_to(&mut self, _cx0: f32, _cy0: f32, _x: f32, _y: f32) {}
    fn curve_to(&mut self, _cx0: f32, _cy0: f32, _cx1: f32, _cy1: f32, _x: f32, _y: f32) {}
    fn close(&mut self) {}
}
