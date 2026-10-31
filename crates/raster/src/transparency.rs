//! Transparency group compositing: opening a group over a parent bitmap,
//! compositing it back, and discarding it.
//!
//! # PDF transparency model (§11.3–11.4)
//!
//! A **transparency group** is an intermediate compositing surface.  The caller:
//!
//! 1. Calls [`begin_group`] with a transparent bitmap of the parent's size,
//!    which becomes the group bitmap.  All subsequent paint operations
//!    target that group.
//! 2. Renders into the group normally (fill, stroke, image, shading, glyph calls
//!    on the group bitmap), within the group's region.
//! 3. Calls [`paint_group`] (or [`discard_group`] on error) to composite the
//!    group back into the underlying bitmap.  Both return the group bitmap
//!    with its region cleared, transparent again for the next group.
//!
//! # Isolated vs. non-isolated groups
//!
//! Every group bitmap starts transparent and accumulates the group's own
//! colour and alpha.  A non-isolated group also keeps a copy of the backdrop
//! under it ([`GroupBitmap::backdrop`]), which blend modes applied inside
//! the group consult through [`crate::pipe::GroupBackdrop`]; compositing
//! the group back is the same for both kinds (ISO 32000-2 §11.4.7).  When the
//! parent is itself a non-isolated group, the backdrop is the parent's
//! content composited over the parent's own backdrop, so it reaches back to
//! the nearest isolated ancestor or the page (§11.4.8).

use crate::bitmap::Bitmap;
use crate::clip::Clip;
use crate::pipe::general::{MAX_COMPS, union_colour};
use crate::pipe::{self, GroupBackdrop, PipeSrc, PipeState};
use color::Pixel;
use color::convert::div255;

// ── Public types ──────────────────────────────────────────────────────────────
/// Parameters for one transparency group, collected before [`begin_group`].
#[derive(Clone, Debug)]
pub struct GroupParams {
    /// Left edge of the group bounding box in device pixels (inclusive).
    pub x_min: i32,
    /// Top edge of the group bounding box in device pixels (inclusive).
    pub y_min: i32,
    /// Right edge of the group bounding box in device pixels (inclusive).
    pub x_max: i32,
    /// Bottom edge of the group bounding box in device pixels (inclusive).
    pub y_max: i32,
    /// `false` → the backdrop under the group is kept for blend modes
    /// applied inside it.
    pub isolated: bool,
}

/// A group bitmap together with its compositing metadata.
///
/// The group bitmap has the parent's dimensions and shares its device
/// coordinate system, so content renders into it unchanged.  Only the
/// region `params.x_min..=x_max × y_min..=y_max` — the bounding box
/// intersected with the clip's rectangle and the parent by
/// [`begin_group`] — is composited back, and the content is to be clipped
/// to it.
///
/// Returned by [`begin_group`]; passed to [`paint_group`] or [`discard_group`].
pub struct GroupBitmap<P: Pixel> {
    /// The rendered group content.  Its alpha plane is the group's own
    /// shape × opacity: zero wherever the group has painted nothing.
    pub bitmap: Bitmap<P>,
    /// Clip region at the time the group was opened (restored on pop).
    pub saved_clip: Clip,
    /// Compositing parameters; the bounding box is clamped to the parent.
    pub params: GroupParams,
    /// For a non-isolated group: the parent's pixels under the region when
    /// the group was opened, composited over the parent's own backdrop.
    pub backdrop: Option<Backdrop>,
}

/// The pixels of a parent bitmap under a non-isolated group's region.
pub struct Backdrop {
    /// Colour bytes, `P::BYTES` per pixel, row-major over the region.
    colour: Vec<u8>,
    /// Alpha per pixel, or `None` for a parent without an alpha plane.
    alpha: Option<Vec<u8>>,
    /// Device column and row of the region's top-left pixel.
    x0: i32,
    y0: i32,
    /// Pixels per row.
    width: usize,
    /// Rows.
    height: usize,
}

impl Backdrop {
    /// Copy the region `x0..x1 × y0..y1` (half-open) of `parent`, each
    /// pixel composited over `under`, the parent's own backdrop, where that
    /// covers it.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "the region lies within the parent, whose dimensions are below 2^31"
    )]
    fn copy<P: Pixel>(
        parent: &Bitmap<P>,
        under: Option<&Self>,
        (x0, y0, x1, y1): (u32, u32, u32, u32),
    ) -> Self {
        let ncomps = P::BYTES;
        let (width, height) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut colour = Vec::with_capacity(width * height * ncomps);
        let mut alpha = parent
            .has_alpha()
            .then(|| Vec::with_capacity(width * height));
        let under = under.map(Self::as_pipe_backdrop);
        for y in y0..y1 {
            let (cx0, cx1) = (x0 as usize * ncomps, x1 as usize * ncomps);
            let row_start = colour.len();
            colour.extend_from_slice(&parent.row_bytes(y)[cx0..cx1]);
            let (Some(a), Some(row)) = (alpha.as_mut(), parent.alpha_row(y)) else {
                continue;
            };
            let alpha_start = a.len();
            a.extend_from_slice(&row[x0 as usize..x1 as usize]);
            let Some(under) = under else {
                continue;
            };
            let colours = colour[row_start..].chunks_exact_mut(ncomps);
            for ((x, c), a) in (x0..x1).zip(colours).zip(&mut a[alpha_start..]) {
                let Some((c_0, a_0)) = under.pixel(x as i32, y as i32, ncomps) else {
                    continue;
                };
                let mut out = [0u8; MAX_COMPS];
                let out = &mut out[..ncomps];
                let a_union = union_colour(c, u32::from(*a), c_0, u32::from(a_0), out);
                c.copy_from_slice(out);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "union_colour returns an alpha of at most 255"
                )]
                {
                    *a = a_union as u8;
                }
            }
        }
        let (x0, y0) = (x0 as i32, y0 as i32);
        Self {
            colour,
            alpha,
            x0,
            y0,
            width,
            height,
        }
    }

    /// The backdrop as the pipe consults it.
    #[must_use]
    pub fn as_pipe_backdrop(&self) -> GroupBackdrop<'_> {
        GroupBackdrop {
            colour: &self.colour,
            alpha: self.alpha.as_deref(),
            x0: self.x0,
            y0: self.y0,
            width: self.width,
            height: self.height,
        }
    }
}

impl<P: Pixel> GroupBitmap<P> {
    /// Returns the `(width, height)` of the group bitmap in pixels.
    #[must_use]
    pub const fn dims(&self) -> (u32, u32) {
        (self.bitmap.width, self.bitmap.height)
    }

    /// The group region as half-open pixel ranges `(x0, y0, x1, y1)`.
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        reason = "begin_group clamped the bounds to [0, parent dimension)"
    )]
    pub const fn region(&self) -> (u32, u32, u32, u32) {
        let p = &self.params;
        (
            p.x_min as u32,
            p.y_min as u32,
            p.x_max as u32 + 1,
            p.y_max as u32 + 1,
        )
    }
}

// ── Group lifecycle ───────────────────────────────────────────────────────────

/// Open a new transparency group on `bitmap`, a transparent bitmap of the
/// parent's dimensions with an alpha plane — a fresh one, or one returned
/// by [`paint_group`] or [`discard_group`].
///
/// - `parent` is the current destination bitmap.  For a non-isolated group its
///   pixels inside the region are kept as the group's backdrop, composited
///   over `parent_backdrop`, the backdrop of `parent` when it is itself a
///   non-isolated group.
/// - The returned [`GroupBitmap`] becomes the new render target until
///   [`paint_group`] or [`discard_group`] is called.  It has the parent's
///   dimensions, so painting into it uses the parent's device coordinates.
///
/// The region is the bounding box intersected with `clip`'s rectangle and
/// clamped to `parent`; an empty region is promoted to the nearest single
/// pixel.
///
/// # Panics
///
/// Panics if `bitmap` is not the parent's size or has no alpha plane, and in
/// debug builds if the bounding box is inverted (`x_min` > `x_max` or
/// `y_min` > `y_max`); a release build promotes an inverted box to one pixel.
#[must_use]
pub fn begin_group<P: Pixel>(
    parent: &Bitmap<P>,
    parent_backdrop: Option<&Backdrop>,
    clip: &Clip,
    mut params: GroupParams,
    bitmap: Bitmap<P>,
) -> GroupBitmap<P> {
    debug_assert!(
        params.x_min <= params.x_max,
        "begin_group: inverted x range [{}, {}]",
        params.x_min,
        params.x_max
    );
    debug_assert!(
        params.y_min <= params.y_max,
        "begin_group: inverted y range [{}, {}]",
        params.y_min,
        params.y_max
    );
    assert!(
        bitmap.width == parent.width && bitmap.height == parent.height && bitmap.has_alpha(),
        "begin_group: the group bitmap must be the parent's size, with an alpha plane"
    );

    let clamp_to = |v: i32, len: u32| v.clamp(0, i32::try_from(len).unwrap_or(i32::MAX) - 1);
    params.x_min = clamp_to(params.x_min.max(clip.x_min_i), parent.width);
    params.y_min = clamp_to(params.y_min.max(clip.y_min_i), parent.height);
    params.x_max = clamp_to(params.x_max.min(clip.x_max_i), parent.width).max(params.x_min);
    params.y_max = clamp_to(params.y_max.min(clip.y_max_i), parent.height).max(params.y_min);

    let mut group = GroupBitmap {
        bitmap,
        saved_clip: clip.clone_shared(),
        params,
        backdrop: None,
    };
    if !group.params.isolated {
        group.backdrop = Some(Backdrop::copy(parent, parent_backdrop, group.region()));
    }
    group
}

/// Composite a finished group back into the parent bitmap and return the
/// saved clip and the group bitmap, its region cleared.
///
/// `pipe` controls blend mode, opacity, and transfer for the compositing step.
/// The group's own alpha plane is folded into the effective source alpha as
/// `eff_a = div255(group_alpha × pipe.a_input)`.
///
/// Pixels with `group_alpha == 0` are skipped entirely (no-op fast path).
///
/// # Panics
///
/// Panics if `group.bitmap` has no alpha plane; a group from [`begin_group`]
/// always has one.
pub fn paint_group<P: Pixel>(
    parent: &mut Bitmap<P>,
    group: GroupBitmap<P>,
    pipe: &PipeState<'_>,
) -> (Clip, Bitmap<P>) {
    let (x0, y0, x1, y1) = group.region();
    let ncomps = P::BYTES;
    let width = group.bitmap.width as usize;
    let alpha = group
        .bitmap
        .alpha_plane()
        .expect("group bitmap is allocated with an alpha plane");

    let x_end = (x1 as usize).min(parent.width as usize);

    for y in y0..y1.min(parent.height) {
        let g_row = group.bitmap.row_bytes(y);
        let g_alpha = &alpha[y as usize * width..(y as usize + 1) * width];
        let (p_row, mut p_alpha) = parent.row_and_alpha_mut(y);

        for x in x0 as usize..x_end {
            let g_src_a = g_alpha[x];
            if g_src_a == 0 {
                continue; // fully transparent — leave parent unchanged
            }

            let off = x * ncomps;

            // Effective source alpha = group alpha × overall pipe opacity.
            let eff_a = div255(u32::from(g_src_a) * u32::from(pipe.a_input));

            let pixel_pipe = PipeState {
                a_input: eff_a,
                ..*pipe
            };
            let src = PipeSrc::Solid(&g_row[off..off + ncomps]);
            let dst_pix = &mut p_row[off..off + ncomps];
            let dst_alpha: Option<&mut [u8]> = p_alpha.as_mut().map(|a| &mut a[x..=x]);

            #[expect(
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "x and y are below the parent's u32 dimensions, which \
                          device bitmaps keep under 2^31"
            )]
            pipe::render_span::<P>(
                &pixel_pipe,
                &src,
                dst_pix,
                dst_alpha,
                None,
                x as i32,
                x as i32,
                y as i32,
            );
        }
    }

    discard_group(group)
}

/// Discard a group without compositing it (used on error paths).
///
/// Returns the clip that was saved when the group was opened so the caller
/// can restore graphics state cleanly, and the group bitmap with its region
/// cleared: transparent again, as [`begin_group`] requires.
#[must_use]
pub fn discard_group<P: Pixel>(group: GroupBitmap<P>) -> (Clip, Bitmap<P>) {
    let (x0, y0, x1, y1) = group.region();
    let GroupBitmap {
        mut bitmap,
        saved_clip,
        ..
    } = group;
    let ncomps = P::BYTES;
    for y in y0..y1.min(bitmap.height) {
        let (row, alpha) = bitmap.row_and_alpha_mut(y);
        row[x0 as usize * ncomps..x1 as usize * ncomps].fill(0);
        if let Some(alpha) = alpha {
            alpha[x0 as usize..x1 as usize].fill(0);
        }
    }
    (saved_clip, bitmap)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::testutil::{make_clip, simple_pipe};
    use color::Rgb8;

    fn default_params(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> GroupParams {
        GroupParams {
            x_min,
            y_min,
            x_max,
            y_max,
            isolated: true,
        }
    }

    /// An isolated group starts transparent.
    #[test]
    fn isolated_group_starts_transparent() {
        let parent: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, true);
        let clip = make_clip(8, 8);
        let params = default_params(0, 0, 7, 7);
        let group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );
        assert!(
            group
                .bitmap
                .alpha_plane()
                .expect("alpha plane")
                .iter()
                .all(|&a| a == 0),
            "isolated group must start fully transparent"
        );
    }

    /// A non-isolated group starts transparent too, keeping the parent's
    /// pixels under its region as its backdrop.
    #[test]
    fn non_isolated_group_keeps_the_backdrop_aside() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        if let Some(a) = parent.alpha_plane_mut() {
            a.fill(128);
        }
        parent.row_bytes_mut(2)[3..6].copy_from_slice(&[10, 20, 30]);
        let clip = make_clip(4, 4);
        let mut params = default_params(1, 1, 2, 2);
        params.isolated = false;
        let group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );
        assert!(
            group
                .bitmap
                .alpha_plane()
                .expect("alpha plane")
                .iter()
                .all(|&a| a == 0),
            "the group's own alpha starts at zero"
        );
        let backdrop = group.backdrop.as_ref().expect("backdrop kept");
        let pipe_backdrop = backdrop.as_pipe_backdrop();
        assert_eq!(
            pipe_backdrop.pixel(1, 2, 3),
            Some((&[10u8, 20, 30][..], 128))
        );
        assert_eq!(pipe_backdrop.pixel(2, 2, 3), Some((&[0u8, 0, 0][..], 128)));
        assert_eq!(pipe_backdrop.pixel(0, 0, 3), None, "outside the region");
        assert!(
            begin_group::<Rgb8>(
                &parent,
                None,
                &clip,
                default_params(0, 0, 3, 3),
                Bitmap::new(parent.width, parent.height, 1, true)
            )
            .backdrop
            .is_none(),
            "an isolated group has no backdrop"
        );
    }

    /// A non-isolated group nested in another takes as backdrop its
    /// parent's content composited over the parent's own backdrop.
    #[test]
    fn nested_non_isolated_backdrop_reaches_the_outer_backdrop() {
        let mut page: Bitmap<Rgb8> = Bitmap::new(4, 1, 4, true);
        for (px, c) in page
            .row_bytes_mut(0)
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(0u8..)
        {
            px.copy_from_slice(&[c * 10, 100, 200]);
        }
        if let Some(a) = page.alpha_plane_mut() {
            a.fill(255);
        }
        let clip = make_clip(4, 1);
        let mut params = default_params(0, 0, 3, 0);
        params.isolated = false;
        let outer = begin_group::<Rgb8>(
            &page,
            None,
            &clip,
            params.clone(),
            Bitmap::new(4, 1, 1, true),
        );
        let mut outer_bitmap = outer.bitmap;
        // The outer group paints opaque white at x = 1 and half-alpha
        // black at x = 2.
        outer_bitmap.row_bytes_mut(0)[3..6].fill(255);
        let alpha = outer_bitmap.alpha_plane_mut().expect("alpha plane");
        alpha[1] = 255;
        alpha[2] = 128;
        let inner = begin_group::<Rgb8>(
            &outer_bitmap,
            outer.backdrop.as_ref(),
            &clip,
            params,
            Bitmap::new(4, 1, 1, true),
        );
        let backdrop = inner.backdrop.expect("backdrop kept");
        let backdrop = backdrop.as_pipe_backdrop();
        assert_eq!(
            backdrop.pixel(0, 0, 3),
            Some((&[0u8, 100, 200][..], 255)),
            "nothing painted: the page"
        );
        assert_eq!(
            backdrop.pixel(1, 0, 3),
            Some((&[255u8, 255, 255][..], 255)),
            "opaque content hides the page"
        );
        assert_eq!(
            backdrop.pixel(2, 0, 3),
            Some((&[9u8, 49, 99][..], 255)),
            "half-alpha black over the page"
        );
    }

    /// Painting a solid-white opaque group over a black parent yields white.
    #[test]
    fn paint_group_opaque_white_over_black() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        let pipe = simple_pipe();

        let params = default_params(1, 1, 2, 2); // 2×2 group at (1,1)
        let mut group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );

        // Fill the group with white pixels and full alpha.
        for y in 0..group.bitmap.height {
            let row = group.bitmap.row_bytes_mut(y);
            for chunk in row.as_chunks_mut::<3>().0 {
                *chunk = [255, 255, 255];
            }
        }
        group
            .bitmap
            .alpha_plane_mut()
            .expect("alpha plane")
            .fill(255);

        let (_clip, _) = paint_group::<Rgb8>(&mut parent, group, &pipe);

        assert_eq!(parent.row(1)[1].r, 255, "pixel (1,1) R should be white");
        assert_eq!(parent.row(1)[2].r, 255, "pixel (1,2) R should be white");
        assert_eq!(
            parent.row(0)[0].r,
            0,
            "pixels outside the region are not composited"
        );
        assert_eq!(
            parent.row(3)[3].r,
            0,
            "pixels outside the region are not composited"
        );
    }

    /// Content painted into a group whose region is away from the origin is
    /// composited at the same device coordinates.
    #[test]
    fn offset_group_composites_in_place() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, true);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let mut group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            default_params(5, 6, 7, 7),
            Bitmap::new(parent.width, parent.height, 1, true),
        );
        assert_eq!(
            group.dims(),
            (8, 8),
            "group shares the parent's coordinates"
        );

        group.bitmap.row_bytes_mut(6)[5 * 3..6 * 3].copy_from_slice(&[0, 0, 255]);
        group.bitmap.alpha_plane_mut().expect("alpha plane")[6 * 8 + 5] = 255;

        let (_clip, _) = paint_group::<Rgb8>(&mut parent, group, &pipe);
        assert_eq!(
            parent.row(6)[5].b,
            255,
            "pixel (5,6) carries the group content"
        );
        assert_eq!(parent.row(0)[0].b, 0, "nothing lands at the origin");
    }

    /// Discarding a group returns the saved clip without painting.
    #[test]
    fn discard_group_does_not_paint() {
        let parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        let params = default_params(0, 0, 3, 3);
        let mut group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );

        // Fill group with red.
        for y in 0..group.bitmap.height {
            let row = group.bitmap.row_bytes_mut(y);
            for chunk in row.as_chunks_mut::<3>().0 {
                *chunk = [255, 0, 0];
            }
        }
        group
            .bitmap
            .alpha_plane_mut()
            .expect("alpha plane")
            .fill(255);

        let (_saved, bitmap) = discard_group(group);

        assert_eq!(parent.row(0)[0].r, 0, "discard must not paint");
        assert!(
            bitmap.data().iter().all(|&b| b == 0)
                && bitmap.alpha_plane().unwrap().iter().all(|&a| a == 0),
            "the returned bitmap is transparent again"
        );
    }

    /// The region is the bounding box within the clip's rectangle.
    #[test]
    fn region_is_the_bbox_within_the_clip() {
        let parent: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, true);
        let clip = Clip::new(2.0, 1.0, 6.0, 5.0, false);
        let group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            default_params(0, 0, 4, 7),
            Bitmap::new(8, 8, 1, true),
        );
        assert_eq!(group.region(), (2, 1, 5, 5));
    }

    /// A transparent group pixel (alpha=0) leaves the parent unchanged.
    #[test]
    fn transparent_group_pixel_is_skipped() {
        let mut parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        // Set parent pixel (0,0) to blue.
        parent.row_bytes_mut(0)[..3].copy_from_slice(&[0, 0, 255]);
        let clip = make_clip(4, 4);
        let pipe = simple_pipe();
        let params = default_params(0, 0, 0, 0); // 1×1 group
        let group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );
        // The group alpha is all 0 (isolated, not painted into).
        let (_saved, _) = paint_group::<Rgb8>(&mut parent, group, &pipe);
        assert_eq!(
            parent.row(0)[0].b,
            255,
            "transparent group pixel must not paint"
        );
    }

    /// `x_max == i32::MAX` must not overflow during `begin_group`.
    #[test]
    fn begin_group_x_max_i32_max_does_not_overflow() {
        let parent: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, true);
        let clip = make_clip(4, 4);
        // Very large bounding box; should be clamped silently to parent bounds.
        let params = default_params(0, 0, i32::MAX, i32::MAX);
        let group = begin_group::<Rgb8>(
            &parent,
            None,
            &clip,
            params,
            Bitmap::new(parent.width, parent.height, 1, true),
        );
        // The region is clamped to the parent: 4×4.
        assert_eq!(group.region(), (0, 0, 4, 4));
    }
}
