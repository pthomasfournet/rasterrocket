//! Shading painting: the `sh` operator and shading patterns (PDF §8.7.4).
//!
//! A shading paints only the points within its extent: an axial or radial
//! shading's parameter range (as extended by `Extend`), a function-based
//! shading's `Domain`, a mesh's patches.  The raster patterns report the
//! points they leave unpainted, so the backdrop survives there.  `BBox`, when
//! present, further clips the shading.  `Background` fills the painted area
//! before the shading when, and only when, the shading is used through a
//! shading pattern (§8.7.4.3).

use std::sync::Arc;

use raster::clip::Clip;
use raster::path::{Path, PathBuilder};
use raster::pipe::{Pattern, Shifted};
use raster::shading::shaded_fill;
use raster::stroke::{StrokeParams, stroke, stroke_outline};
use raster::xpath::XPath;

use super::canvas::{CanvasKind, CanvasPattern, CanvasPipe, CanvasSource, with_canvas};
use super::paint::{FillRule, PaintParams, fill_device_path};
use super::pixel_rect::PixelRect;
use super::{DEVICE_MATRIX, PageRenderer};
use crate::renderer::color::RasterColor;
use crate::resources::ShadingPattern;
use crate::resources::shading::{ResolvedShading, ShadingResult};
use crate::resources::transfer::Transfer;
use crate::separation::Separator;

/// Return `clip` intersected with the device-space quadrilateral `bbox`.
fn clip_to_bbox(mut clip: Clip, bbox: Option<[(f64, f64); 4]>, flatness: f64) -> Clip {
    if let Some(corners) = bbox {
        let path = polygon(&corners);
        clip.clip_to_path(&XPath::new(&path, &DEVICE_MATRIX, flatness, true), false);
    }
    clip
}

/// The closed polygon through `corners`.
pub(super) fn polygon(corners: &[(f64, f64)]) -> Path {
    let mut pb = PathBuilder::new();
    for (i, &(x, y)) in corners.iter().enumerate() {
        if i == 0 {
            pb.move_to(x, y);
        } else {
            let added = pb.line_to(x, y);
            debug_assert!(added.is_ok(), "polygon: path construction failed");
        }
    }
    let closed = pb.close(true);
    debug_assert!(closed.is_ok(), "polygon: close failed");
    pb.build()
}

/// The `Background` of `shading` as painted on a canvas of `kind`, if it
/// has one: on an ink canvas its ink, given for a shading of ink, mapped
/// through `transfer`, which the pipe applies on an RGB canvas (see
/// `ShadingPaint::pipe`).
fn background(
    kind: CanvasKind,
    shading: &ResolvedShading,
    transfer: Option<&Arc<Transfer>>,
) -> Option<RasterColor> {
    let bg = shading.background?;
    Some(match (kind, shading.background_ink) {
        (CanvasKind::Rgb | CanvasKind::Rgb16, _) => RasterColor::from_srgb(bg),
        (CanvasKind::Ink | CanvasKind::Ink16, None) => {
            RasterColor::from_srgb(bg).transferred(transfer)
        }
        (CanvasKind::Ink | CanvasKind::Ink16, Some(ink)) => {
            RasterColor::with_ink(bg, ink, shading.paints).transferred(transfer)
        }
    })
}

/// The pattern of a smooth shading's colours, of the depth it was resolved
/// at, read on a window of the page.
enum WindowPattern<'a> {
    /// 8-bit sRGB or ink.
    U8(Shifted<'a, u8>),
    /// 16-bit sRGB.
    U16(Shifted<'a, u16>),
}

/// The pattern of a smooth shading's colours, resolved on the page's
/// pixel grid, read on the window whose pixel `(0, 0)` is the page's pixel
/// `origin`; `None` for a mesh.
fn window_pattern(shading: &ResolvedShading, origin: (i32, i32)) -> Option<WindowPattern<'_>> {
    match &shading.result {
        ShadingResult::Pattern(p) | ShadingResult::InkPattern(p) => {
            let inner: &dyn Pattern = p.as_ref();
            Some(WindowPattern::U8(Shifted { inner, origin }))
        }
        ShadingResult::Pattern16(p) | ShadingResult::InkPattern16(p) => {
            let inner: &dyn Pattern<u16> = p.as_ref();
            Some(WindowPattern::U16(Shifted { inner, origin }))
        }
        ShadingResult::Mesh(_)
        | ShadingResult::Mesh16(_)
        | ShadingResult::InkMesh(_)
        | ShadingResult::InkMesh16(_) => None,
    }
}

/// `window`, the [`window_pattern`] of `shading`, for a canvas of `kind`,
/// its colours converted to ink by `separator` and mapped through
/// `transfer` on an ink canvas; `None` for
/// colours of ink on an RGB canvas, which takes none, and for 16-bit
/// colours on an 8-bit canvas, which is never given them.
fn smooth_pattern<'a>(
    kind: CanvasKind,
    shading: &ResolvedShading,
    window: &'a WindowPattern<'a>,
    (transfer, separator): (Option<&'a Transfer>, &'a Separator),
) -> Option<CanvasPattern<'a>> {
    match (window, &shading.result, kind) {
        (WindowPattern::U8(window), ShadingResult::Pattern(_), _) => {
            Some(CanvasPattern::new(kind, window, (transfer, separator)))
        }
        (WindowPattern::U8(window), ShadingResult::InkPattern(_), CanvasKind::Ink) => {
            Some(CanvasPattern::ink(window, transfer, shading.carries()))
        }
        (
            WindowPattern::U16(window),
            ShadingResult::Pattern16(_),
            CanvasKind::Rgb16 | CanvasKind::Ink16,
        ) => Some(CanvasPattern::wide(kind, window, (transfer, separator))),
        (WindowPattern::U16(window), ShadingResult::InkPattern16(_), CanvasKind::Ink16) => {
            Some(CanvasPattern::ink16(window, transfer, shading.carries()))
        }
        _ => None,
    }
}

/// The compositing parameters of one shading paint, read from the graphics
/// state in effect; the pipe borrows them.
struct ShadingPaint {
    /// The alpha, blend mode, channels and backdrop of the paint.
    paint: PaintParams,
    /// The transfer function in effect.
    transfer: Option<Arc<Transfer>>,
}

impl ShadingPaint {
    /// The pipe for a canvas of `kind`: the paint's pipe, applying the transfer
    /// function to the shading's colours on an RGB canvas.  On an ink
    /// canvas the shading's sources apply it before their conversion to
    /// ink.
    fn pipe(&self, kind: CanvasKind) -> CanvasPipe<'_> {
        CanvasPipe {
            paint: &self.paint,
            transfer: CanvasPattern::pipe_transfer(kind, self.transfer.as_deref()),
        }
    }
}

impl PageRenderer<'_> {
    /// The window's image of `corners`, a shading's `BBox` corners in the
    /// page's device space.
    fn window_corners(&self, corners: [(f64, f64); 4]) -> [(f64, f64); 4] {
        corners.map(|p| self.page_to_window(p))
    }

    /// The compositing parameters of a paint of `shading` by the stroking
    /// (`stroke`) or non-stroking operation under the graphics state in
    /// effect.  Under overprint the shading writes the channels of its
    /// colour space; overprint mode does not apply to shadings.
    fn shading_paint(&self, stroke: bool, shading: &ResolvedShading) -> ShadingPaint {
        let gs = self.gstate.current();
        let alpha = if stroke {
            gs.stroke_alpha
        } else {
            gs.fill_alpha
        };
        let overprint = self.overprint_channels(stroke, shading.paints);
        ShadingPaint {
            paint: self.paint_params(alpha, gs.blend_mode, overprint, shading.carries()),
            transfer: gs.transfer.clone(),
        }
    }

    /// Run `paint`, which paints the shading of `pattern`, with the
    /// pattern's `ExtGState` in effect (PDF §8.7.3.3).
    ///
    /// `paint` receives the compositing state for the non-stroking
    /// (`stroke == false`) or stroking constant alpha and the blend mode,
    /// and the shading.  The `ExtGState` applies on top of the current
    /// graphics state for the duration of `paint` only; a soft mask it sets
    /// is resolved in pattern space and masks what `paint` paints.
    pub(super) fn paint_shading_pattern(
        &mut self,
        pattern: &ShadingPattern,
        stroke: bool,
        region: Option<PixelRect>,
        paint: impl FnOnce(&mut Self, &CanvasPipe<'_>, &ResolvedShading),
    ) {
        let ShadingPattern {
            shading,
            ext_gstate,
            matrix,
        } = pattern;
        let Some(ext) = ext_gstate else {
            let state = self.shading_paint(stroke, shading);
            let pipe = state.pipe(self.canvas.kind());
            paint(self, &pipe, shading);
            return;
        };
        self.gstate.save();
        let ctm = std::mem::replace(&mut self.gstate.current_mut().ctm, *matrix);
        self.apply_ext_gstate(ext);
        self.gstate.current_mut().ctm = ctm;
        let state = self.shading_paint(stroke, shading);
        let pipe = state.pipe(self.canvas.kind());
        match self.gstate.current_mut().soft_mask.take() {
            Some(mask) => self.paint_through_soft_mask(&mask, region, |r| paint(r, &pipe, shading)),
            None => paint(self, &pipe, shading),
        }
        self.gstate.restore();
    }

    /// Execute an `sh` shading operator: paint the named shading over the
    /// current clip region.  The shading's `Background` does not apply.
    pub(super) fn do_shading(&mut self, name: &[u8]) {
        self.trace_shading(name);
        let gs = self.gstate.current();
        let ctm = gs.ctm;
        let flatness = gs.flatness.max(0.1);

        let Some(shading) = self.memo_shading(name, &ctm) else {
            log::warn!(
                "rasterrocket-interp: sh /{} — shading not available (unsupported type or missing resource)",
                String::from_utf8_lossy(name)
            );
            return;
        };
        let state = self.shading_paint(false, &shading);
        let pipe = state.pipe(self.canvas.kind());
        let transfer = state.transfer.as_deref();

        let clip = clip_to_bbox(
            self.gstate.current().clip.clone_shared(),
            shading.bbox.map(|c| self.window_corners(c)),
            flatness,
        );
        let origin = self.page_origin();
        // The shading is one object under the soft mask in effect; it can
        // paint the whole clip.
        let region = PixelRect::of_clip(&clip, self.width, self.height);
        let kind = self.canvas.kind();
        self.paint_object(region, state.paint.alpha, |r| match &shading.result {
            ShadingResult::Pattern(_)
            | ShadingResult::Pattern16(_)
            | ShadingResult::InkPattern(_)
            | ShadingResult::InkPattern16(_) => {
                let area = polygon(&[
                    (clip.x_min, clip.y_min),
                    (clip.x_max, clip.y_min),
                    (clip.x_max, clip.y_max),
                    (clip.x_min, clip.y_max),
                ]);
                let Some(window) = window_pattern(&shading, origin) else {
                    return;
                };
                let separator = r.separator(shading.gray);
                let Some(pattern) = smooth_pattern(kind, &shading, &window, (transfer, &separator))
                else {
                    return;
                };
                with_canvas!(&mut r.canvas, |b| {
                    let pipe = pipe.at();
                    let Some(pattern) = pattern.at() else {
                        return;
                    };
                    shaded_fill(
                        b,
                        &clip,
                        &area,
                        &pipe,
                        pattern.as_dyn(),
                        &DEVICE_MATRIX,
                        flatness,
                        true,
                        false,
                    );
                });
            }
            ShadingResult::Mesh(_)
            | ShadingResult::Mesh16(_)
            | ShadingResult::InkMesh(_)
            | ShadingResult::InkMesh16(_) => {
                r.paint_mesh(&shading, &clip, &pipe, transfer);
            }
        });
    }

    /// Fill `path` with a shading pattern under `clip`: the `Background`
    /// first, over the whole path, then the shading clipped by its `BBox`.
    pub(super) fn fill_with_shading(
        &mut self,
        path: &Path,
        rule: FillRule,
        clip: Clip,
        pipe: &CanvasPipe<'_>,
        flatness: f64,
        shading: &ResolvedShading,
    ) {
        let antialias = self.vector_antialias;
        let kind = self.canvas.kind();
        let transfer = self.gstate.current().transfer.clone();
        let separator = self.separator(shading.gray);
        let paint = |r: &mut Self, clip: &Clip, source: CanvasSource<'_>| {
            with_canvas!(&mut r.canvas, |b| {
                let pipe = pipe.at();
                source.paint(b, |b, src| {
                    fill_device_path(b, clip, path, &pipe, src, flatness, antialias, rule);
                });
            });
        };
        if let Some(bg) = background(kind, shading, transfer.as_ref()) {
            paint(self, &clip, CanvasSource::Solid(&bg, &separator));
        }
        let clip = clip_to_bbox(clip, shading.bbox.map(|c| self.window_corners(c)), flatness);
        if let Some(window) = window_pattern(shading, self.page_origin()) {
            if let Some(pattern) =
                smooth_pattern(kind, shading, &window, (transfer.as_deref(), &separator))
            {
                paint(self, &clip, CanvasSource::Pattern(&pattern));
            }
            return;
        }
        let mut clip = clip;
        clip.clip_to_path(
            &XPath::new(path, &DEVICE_MATRIX, flatness, true),
            rule.even_odd,
        );
        self.paint_mesh(shading, &clip, pipe, transfer.as_deref());
    }

    /// Paint the triangles of the mesh of `shading` that meet `clip`
    /// through `pipe`, mapped through `transfer` on an ink canvas; a smooth
    /// shading paints nothing here.
    fn paint_mesh(
        &mut self,
        shading: &ResolvedShading,
        clip: &Clip,
        pipe: &CanvasPipe<'_>,
        transfer: Option<&Transfer>,
    ) {
        let origin = self.page_origin();
        let separator = self.separator(shading.gray);
        match &shading.result {
            ShadingResult::Mesh(mesh) => {
                for tri in mesh.meeting(clip, origin) {
                    self.canvas
                        .gouraud_triangle(origin, clip, pipe, (transfer, &separator), tri);
                }
            }
            ShadingResult::Mesh16(mesh) => {
                for tri in mesh.meeting(clip, origin) {
                    self.canvas
                        .gouraud_triangle16(origin, clip, pipe, (transfer, &separator), tri);
                }
            }
            ShadingResult::InkMesh(mesh) => {
                for tri in mesh.meeting(clip, origin) {
                    self.canvas.gouraud_ink_triangle(
                        origin,
                        clip,
                        pipe,
                        transfer,
                        shading.carries(),
                        tri,
                    );
                }
            }
            ShadingResult::InkMesh16(mesh) => {
                for tri in mesh.meeting(clip, origin) {
                    self.canvas.gouraud_ink_triangle16(
                        origin,
                        clip,
                        pipe,
                        transfer,
                        shading.carries(),
                        tri,
                    );
                }
            }
            ShadingResult::Pattern(_)
            | ShadingResult::Pattern16(_)
            | ShadingResult::InkPattern(_)
            | ShadingResult::InkPattern16(_) => {}
        }
    }

    /// Stroke `path` (in the space `matrix` maps to the device) with a
    /// shading pattern under `clip`: the `Background` first, then the
    /// shading clipped by its `BBox`.  A mesh shading paints its triangles
    /// clipped to the stroke's outline; a hairline's outline is one device
    /// pixel wide.
    #[expect(
        clippy::too_many_arguments,
        reason = "the stroke's geometry, compositing state and shading"
    )]
    pub(super) fn stroke_with_shading(
        &mut self,
        path: &Path,
        matrix: &[f64; 6],
        params: &StrokeParams<'_>,
        clip: Clip,
        pipe: &CanvasPipe<'_>,
        flatness: f64,
        shading: &ResolvedShading,
    ) {
        let kind = self.canvas.kind();
        let transfer = self.gstate.current().transfer.clone();
        let separator = self.separator(shading.gray);
        let paint = |r: &mut Self, clip: &Clip, source: CanvasSource<'_>| {
            with_canvas!(&mut r.canvas, |b| {
                let pipe = pipe.at();
                source.paint(b, |b, src| {
                    stroke(b, clip, path, &pipe, src, matrix, params);
                });
            });
        };
        if let Some(bg) = background(kind, shading, transfer.as_ref()) {
            paint(self, &clip, CanvasSource::Solid(&bg, &separator));
        }
        let clip = clip_to_bbox(clip, shading.bbox.map(|c| self.window_corners(c)), flatness);
        if let Some(window) = window_pattern(shading, self.page_origin()) {
            if let Some(pattern) =
                smooth_pattern(kind, shading, &window, (transfer.as_deref(), &separator))
            {
                paint(self, &clip, CanvasSource::Pattern(&pattern));
            }
            return;
        }
        let device_scale = matrix[0]
            .mul_add(matrix[3], -(matrix[1] * matrix[2]))
            .abs()
            .sqrt();
        let hairline = StrokeParams {
            line_width: 1.0 / device_scale,
            ..*params
        };
        let params = if params.line_width == 0.0 && device_scale > 0.0 {
            &hairline
        } else {
            params
        };
        let Some(outline) = stroke_outline(path, matrix, params) else {
            return;
        };
        let mut clip = clip;
        clip.clip_to_path(&outline, false);
        self.paint_mesh(shading, &clip, pipe, transfer.as_deref());
    }
}
