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
use raster::pipe::{PipeSrc, PipeState};
use raster::shading::shaded_fill;
use raster::stroke::{StrokeParams, stroke, stroke_outline};
use raster::xpath::XPath;
use raster::{eo_fill, fill};

use super::canvas::{CanvasKind, CanvasPattern, with_canvas};
use super::paint::PaintParams;
use super::pixel_rect::PixelRect;
use super::{DEVICE_MATRIX, PageRenderer};
use crate::renderer::color::RasterColor;
use crate::resources::ShadingPattern;
use crate::resources::shading::{ResolvedShading, ShadingResult};
use crate::resources::transfer::Transfer;

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
        let added = if i == 0 {
            pb.move_to(x, y)
        } else {
            pb.line_to(x, y)
        };
        debug_assert!(added.is_ok(), "polygon: path construction failed");
    }
    let closed = pb.close(true);
    debug_assert!(closed.is_ok(), "polygon: close failed");
    pb.build()
}

/// A shading's `Background` `bg` as painted on a canvas of `kind`: on an
/// ink canvas mapped through `transfer`, which the pipe applies on an RGB
/// canvas (see `ShadingPaint::pipe`).
fn background(kind: CanvasKind, bg: [u8; 3], transfer: Option<&Transfer>) -> RasterColor {
    let bg = RasterColor::from_bytes(bg);
    match kind {
        CanvasKind::Rgb => bg,
        CanvasKind::Ink => bg.transferred(transfer),
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
    fn pipe(&self, kind: CanvasKind) -> PipeState<'_> {
        let mut pipe = self.paint.pipe();
        if let Some(set) = CanvasPattern::pipe_transfer(kind, self.transfer.as_deref()) {
            pipe.transfer = set;
        }
        pipe
    }
}

impl PageRenderer<'_> {
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
            paint: self.paint_params(alpha, gs.blend_mode, overprint),
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
        paint: impl FnOnce(&mut Self, &PipeState<'_>, &ResolvedShading),
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
            shading.bbox,
            flatness,
        );
        // The shading is one object under the soft mask in effect; it can
        // paint the whole clip.
        let region = PixelRect::of_clip(&clip, self.width, self.height);
        self.paint_object(region, state.paint.alpha, |r| match &shading.result {
            ShadingResult::Pattern(pattern) => {
                let area = polygon(&[
                    (clip.x_min, clip.y_min),
                    (clip.x_max, clip.y_min),
                    (clip.x_max, clip.y_max),
                    (clip.x_min, clip.y_max),
                ]);
                let pattern = CanvasPattern::new(r.canvas.kind(), pattern.as_ref(), transfer);
                with_canvas!(&mut r.canvas, |b| shaded_fill(
                    b,
                    &clip,
                    &area,
                    &pipe,
                    pattern.as_dyn(),
                    &DEVICE_MATRIX,
                    flatness,
                    true,
                    false,
                ));
            }
            ShadingResult::Mesh(mesh) => {
                for tri in mesh.meeting(&clip) {
                    r.canvas.gouraud_triangle(&clip, &pipe, transfer, tri);
                }
            }
        });
    }

    /// Fill `path` with a shading pattern under `clip`: the `Background`
    /// first, over the whole path, then the shading clipped by its `BBox`.
    pub(super) fn fill_with_shading(
        &mut self,
        path: &Path,
        even_odd: bool,
        clip: Clip,
        pipe: &PipeState<'_>,
        flatness: f64,
        shading: &ResolvedShading,
    ) {
        let antialias = self.vector_antialias;
        let kind = self.canvas.kind();
        let transfer = self.gstate.current().transfer.clone();
        let paint = |r: &mut Self, clip: &Clip, src: &PipeSrc<'_>| {
            with_canvas!(&mut r.canvas, |b| {
                let fill_fn = if even_odd { eo_fill } else { fill };
                fill_fn(
                    b,
                    clip,
                    path,
                    pipe,
                    src,
                    &DEVICE_MATRIX,
                    flatness,
                    antialias,
                );
            });
        };
        if let Some(bg) = shading.background {
            let bg = background(kind, bg, transfer.as_deref());
            paint(self, &clip, &PipeSrc::Solid(bg.bytes_for(kind.mode())));
        }
        let clip = clip_to_bbox(clip, shading.bbox, flatness);
        match &shading.result {
            ShadingResult::Pattern(pattern) => {
                let pattern = CanvasPattern::new(kind, pattern.as_ref(), transfer.as_deref());
                paint(self, &clip, &PipeSrc::Pattern(pattern.as_dyn()));
            }
            ShadingResult::Mesh(mesh) => {
                let mut clip = clip;
                clip.clip_to_path(&XPath::new(path, &DEVICE_MATRIX, flatness, true), even_odd);
                for tri in mesh.meeting(&clip) {
                    self.canvas
                        .gouraud_triangle(&clip, pipe, transfer.as_deref(), tri);
                }
            }
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
        pipe: &PipeState<'_>,
        flatness: f64,
        shading: &ResolvedShading,
    ) {
        let kind = self.canvas.kind();
        let transfer = self.gstate.current().transfer.clone();
        if let Some(bg) = shading.background {
            let bg = background(kind, bg, transfer.as_deref());
            let src = PipeSrc::Solid(bg.bytes_for(kind.mode()));
            with_canvas!(&mut self.canvas, |b| stroke(
                b, &clip, path, pipe, &src, matrix, params
            ));
        }
        let clip = clip_to_bbox(clip, shading.bbox, flatness);
        match &shading.result {
            ShadingResult::Pattern(pattern) => {
                let pattern = CanvasPattern::new(kind, pattern.as_ref(), transfer.as_deref());
                let src = PipeSrc::Pattern(pattern.as_dyn());
                with_canvas!(&mut self.canvas, |b| stroke(
                    b, &clip, path, pipe, &src, matrix, params
                ));
            }
            ShadingResult::Mesh(mesh) => {
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
                for tri in mesh.meeting(&clip) {
                    self.canvas
                        .gouraud_triangle(&clip, pipe, transfer.as_deref(), tri);
                }
            }
        }
    }
}
