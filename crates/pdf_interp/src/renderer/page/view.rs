//! The device-space placement of a page render and its initial CTM.

use super::super::gstate::{Ctm, GStateStack};
use crate::PageGeometry;

/// Build the initial page CTM that maps PDF user space to device pixels.
///
/// `scale_x` and `scale_y` are device pixels per user-space unit along the
/// device axes (`x_dpi / 72 × UserUnit`, `y_dpi / 72 × UserUnit`), applied
/// after rotation.  `rotate_cw`
/// is the page `/Rotate`, which callers must
/// pass already normalised to one of {0, 90, 180, 270} (the only values ISO
/// 32000-2 §14.11.6 defines; `page_size_pts_by_id` snaps `/Rotate` to this set
/// upstream).  Any other value is a caller contract violation: it trips a
/// `debug_assert` and, in release, renders unrotated rather than silently
/// emitting a wrong-but-plausible rotation.  `origin_x`/`origin_y` are the
/// selected page box's lower-left corner in PDF user space (ISO 32000-2
/// §14.11.2); zero for the common box-at-origin case.
///
/// The `to_device` helper applies an additional Y-flip (`height_px - dy`), so
/// these matrices only carry scale + rotation + the box-origin pre-translation;
/// the flip is not folded in here.  `w`/`h` are the output bitmap dimensions in
/// points (`width_px / scale_x`, `height_px / scale_y`); for 90°/270° the
/// caller has already swapped the pixel dimensions, so here `w` is the
/// post-rotation width.
///
/// PDF user-space content coordinates are absolute: a content point at PDF
/// `(X, Y)` with `X ∈ [llx, llx+boxW]`, `Y ∈ [lly, lly+boxH]`.  We want the box
/// lower-left `(llx, lly)` to land at device origin, so each branch is the
/// box-at-origin matrix with the substitution `X → X-llx`, `Y → Y-lly` — i.e. an
/// innermost pre-translation `T(-llx, -lly)` composed under rotate and then
/// the device scale.  The device scale multiplies the matrix entries that
/// produce device x (`a`, `c`, `e`) by `sx = scale_x` and those that produce
/// device y (`b`, `d`, `f`) by `sy = scale_y`.  Per branch:
///   Rotate 0   → `[ sx,   0,   0,  sy,     -llx·sx,     -lly·sy]`
///   Rotate 90  → `[  0, -sy,  sx,   0,     -lly·sx,  (h+llx)·sy]`
///   Rotate 180 → `[-sx,   0,   0, -sy,  (w+llx)·sx,  (h+lly)·sy]`
///   Rotate 270 → `[  0,  sy, -sx,   0,  (w+lly)·sx,     -llx·sy]`
/// When `llx = lly = 0` every branch reduces to the box-at-origin value, so
/// origin-at-zero PDFs (the vast majority) are unchanged in scale.
///
/// The `initial_ctm_rotate{0,90,180,270}_maps_box_corners_to_device_pixels`
/// tests pin each box corner to its required device pixel for every rotation,
/// so a sign error in any branch fails a test rather than shipping a mirrored
/// or upside-down render.
///
/// Degenerate inputs cannot reach here as silent garbage: both scales are
/// asserted finite-positive by the sole caller, and `origin_x`/`origin_y` originate from
/// `page_size_pts_by_id`, which rejects non-finite / zero-area boxes upstream
/// and logs malformed-box recovery.  This function therefore performs only the
/// arithmetic; it does not re-validate.
pub(super) fn build_initial_ctm(
    width: u32,
    height: u32,
    (sx, sy): (f64, f64),
    rotate_cw: u16,
    origin_x: f64,
    origin_y: f64,
) -> Ctm {
    let llx = origin_x;
    let lly = origin_y;
    let w = f64::from(width) / sx;
    let h = f64::from(height) / sy;
    match rotate_cw % 360 {
        // Rotate 0: standard scale + y-flip handled by to_device.
        0 => [sx, 0.0, 0.0, sy, -llx * sx, -lly * sy],
        // Rotate 90 CW: swap axes. Bitmap width = original H, height = original W.
        90 => [0.0, -sy, sx, 0.0, -lly * sx, (h + llx) * sy],
        // Rotate 180: flip both axes.
        180 => [-sx, 0.0, 0.0, -sy, (w + llx) * sx, (h + lly) * sy],
        // Rotate 270 CW (= 90 CCW): swap axes, opposite orientation.
        270 => [0.0, sy, -sx, 0.0, (w + lly) * sx, -llx * sy],
        // Non-spec value: a caller contract violation (callers normalise to
        // {0,90,180,270} upstream).  Fail loudly in debug; in release fall back
        // to the unrotated matrix so a stray value renders upright rather than
        // silently rotated-and-wrong.
        other => {
            debug_assert!(
                false,
                "build_initial_ctm: rotate_cw must be normalised to {{0,90,180,270}}, got {other}"
            );
            [sx, 0.0, 0.0, sy, -llx * sx, -lly * sy]
        }
    }
}

/// The initial graphics state of a `width × height` window drawn under
/// `ctm`, clipping in the coverage mode `antialias`.
pub(super) fn initial_gstate(width: u32, height: u32, ctm: Ctm, antialias: bool) -> GStateStack {
    let mut gstate = GStateStack::new(width, height, antialias);
    gstate.current_mut().ctm = ctm;
    gstate
}

/// The device-space placement of a page render: the page's pixel extent and
/// initial-CTM parameters, and the window of that extent the renderer draws.
///
/// Device pixels are counted from the top-left corner of the whole page.  The
/// renderer's bitmap covers the window `[x, x + width) × [y, y + height)`; its
/// initial CTM is the whole-page CTM translated by the window's offset, so a
/// window draws exactly the device pixels the whole-page render places there.
///
/// The page's pixel extent is its geometry's [`PageGeometry::size_px`] at
/// the scales, which counts `user_unit` (ISO 32000-2 §14.11.2: one user-space
/// unit is `user_unit / 72` inch), and the initial CTM maps one user-space
/// unit to `scale × user_unit` device pixels, so the content fills the extent.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PageView {
    /// Width of the whole page in device pixels, after rotation.
    pub page_width: u32,
    /// Height of the whole page in device pixels, after rotation.
    pub page_height: u32,
    /// Device pixels per point along the page's width (`x_dpi / 72` for the
    /// renderer's callers).  Positive and finite.
    pub scale_x: f64,
    /// Device pixels per point along the page's height (`y_dpi / 72` for the
    /// renderer's callers).  Positive and finite.
    pub scale_y: f64,
    /// Points per user-space unit: the page's `/UserUnit`.
    pub user_unit: f64,
    /// The page `/Rotate`, clockwise degrees: one of 0, 90, 180, 270.
    pub rotate_cw: u16,
    /// X of the selected page box's lower-left corner in PDF user space.
    pub origin_x: f64,
    /// Y of the selected page box's lower-left corner in PDF user space.
    pub origin_y: f64,
    /// Left edge of the drawn window, in device pixels from the page's left edge.
    pub x: u32,
    /// Top edge of the drawn window, in device pixels from the page's top edge.
    pub y: u32,
    /// Width of the drawn window in device pixels.
    pub width: u32,
    /// Height of the drawn window in device pixels.
    pub height: u32,
}

impl PageView {
    /// The view of the whole page `geom` drawn at `(scale_x, scale_y)`
    /// device pixels per point: `geom.size_px(scale_x, scale_y)` pixels,
    /// with the geometry's rotation, box origin and `user_unit`.
    #[must_use]
    pub fn whole(geom: &PageGeometry, (scale_x, scale_y): (f64, f64)) -> Self {
        let (page_width, page_height) = geom.size_px(scale_x, scale_y);
        Self {
            page_width,
            page_height,
            scale_x,
            scale_y,
            user_unit: geom.user_unit,
            rotate_cw: geom.rotate_cw,
            origin_x: geom.origin_x,
            origin_y: geom.origin_y,
            x: 0,
            y: 0,
            width: page_width,
            height: page_height,
        }
    }

    /// This view restricted to the window at `(x, y)` of `width × height`
    /// device pixels.
    #[must_use]
    pub const fn window(self, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            ..self
        }
    }

    /// The initial CTM: the whole page's, with the window's top-left corner
    /// moved to the device origin.
    ///
    /// One user-space unit is `user_unit` points, so the CTM's scale is
    /// `scale × user_unit` device pixels per unit.  Device y grows downward
    /// from the window's top edge (`device_y = height − ctm_y`), so a window
    /// `y` pixels below the page top and `height` tall shifts the CTM's y
    /// translation by `page_height − y − height`.
    pub(super) fn initial_ctm(&self) -> Ctm {
        let mut ctm = build_initial_ctm(
            self.page_width,
            self.page_height,
            (self.scale_x * self.user_unit, self.scale_y * self.user_unit),
            self.rotate_cw,
            self.origin_x,
            self.origin_y,
        );
        let below = i64::from(self.page_height) - i64::from(self.y) - i64::from(self.height);
        ctm[4] -= f64::from(self.x);
        #[expect(
            clippy::cast_precision_loss,
            reason = "|below| < 2^33, exactly representable in f64"
        )]
        {
            ctm[5] -= below as f64;
        }
        ctm
    }
}
