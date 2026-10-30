//! Shading painting: the `sh` operator and shading patterns (PDF §8.7.4).
//!
//! A shading paints only the points within its extent: an axial or radial
//! shading's parameter range (as extended by `Extend`), a function-based
//! shading's `Domain`, a mesh's patches.  The raster patterns report the
//! points they leave unpainted, so the backdrop survives there.  `BBox`, when
//! present, further clips the shading.  `Background` fills the painted area
//! before the shading when, and only when, the shading is used through a
//! shading pattern (§8.7.4.3).

use color::Rgb8;
use raster::clip::Clip;
use raster::path::{Path, PathBuilder};
use raster::pipe::{PipeSrc, PipeState};
use raster::shading::{gouraud::gouraud_triangle_fill, shaded_fill};
use raster::stroke::{StrokeParams, stroke};
use raster::xpath::XPath;
use raster::{Bitmap, eo_fill, fill};

use super::{DEVICE_MATRIX, PageRenderer};
use crate::resources::ShadingPattern;
use crate::resources::shading::{ResolvedShading, ShadingResult};

/// Return `clip` intersected with the device-space quadrilateral `bbox`.
fn clip_to_bbox(mut clip: Clip, bbox: Option<[(f64, f64); 4]>, flatness: f64) -> Clip {
    if let Some(corners) = bbox {
        let path = polygon(&corners);
        clip.clip_to_path(&XPath::new(&path, &DEVICE_MATRIX, flatness, true), false);
    }
    clip
}

/// The closed polygon through `corners`.
fn polygon(corners: &[(f64, f64)]) -> Path {
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

impl PageRenderer<'_> {
    /// Run `paint`, which paints the shading of `pattern`, with the
    /// pattern's `ExtGState` in effect (PDF §8.7.3.3).
    ///
    /// `paint` receives the compositing state for the non-stroking
    /// (`stroke == false`) or stroking constant alpha and the blend mode,
    /// and the shading.  The `ExtGState` applies on top of the current
    /// graphics state for the duration of `paint` only; a soft mask it sets
    /// is resolved in pattern space and masks what `paint` paints.
    pub(super) fn paint_shading_pattern<R>(
        &mut self,
        pattern: ShadingPattern,
        stroke: bool,
        paint: impl FnOnce(&mut Self, &PipeState<'static>, ResolvedShading) -> R,
    ) -> R {
        let ShadingPattern {
            shading,
            ext_gstate,
            matrix,
        } = pattern;
        let pipe_of = |r: &Self| {
            let gs = r.gstate.current();
            let alpha = if stroke {
                gs.stroke_alpha
            } else {
                gs.fill_alpha
            };
            Self::make_pipe(alpha, gs.blend_mode)
        };
        let Some(ext) = ext_gstate else {
            let pipe = pipe_of(self);
            return paint(self, &pipe, shading);
        };
        self.gstate.save();
        let ctm = std::mem::replace(&mut self.gstate.current_mut().ctm, matrix);
        self.apply_ext_gstate(&ext);
        self.gstate.current_mut().ctm = ctm;
        let pipe = pipe_of(self);
        let result = match self.gstate.current_mut().soft_mask.take() {
            Some(mask) => self.paint_through_soft_mask(&mask, |r| paint(r, &pipe, shading)),
            None => paint(self, &pipe, shading),
        };
        self.gstate.restore();
        result
    }

    /// Execute an `sh` shading operator: paint the named shading over the
    /// current clip region.  The shading's `Background` does not apply.
    pub(super) fn do_shading(&mut self, name: &[u8]) {
        let gs = self.gstate.current();
        let ctm = gs.ctm;
        let flatness = gs.flatness.max(0.1);
        let pipe = Self::make_pipe(gs.fill_alpha, gs.blend_mode);
        let page_h = f64::from(self.height);

        let Some(shading) = self.resources.shading(name, &ctm, page_h) else {
            log::warn!(
                "rasterrocket-interp: sh /{} — shading not available (unsupported type or missing resource)",
                String::from_utf8_lossy(name)
            );
            return;
        };

        let clip = clip_to_bbox(
            self.gstate.current().clip.clone_shared(),
            shading.bbox,
            flatness,
        );
        match shading.result {
            ShadingResult::Pattern(pattern) => {
                let area = polygon(&[
                    (clip.x_min, clip.y_min),
                    (clip.x_max, clip.y_min),
                    (clip.x_max, clip.y_max),
                    (clip.x_min, clip.y_max),
                ]);
                shaded_fill::<Rgb8>(
                    &mut self.bitmap,
                    &clip,
                    &area,
                    &pipe,
                    pattern.as_ref(),
                    &DEVICE_MATRIX,
                    flatness,
                    true,
                    false,
                );
            }
            ShadingResult::Mesh(triangles) => {
                for tri in triangles {
                    gouraud_triangle_fill::<Rgb8>(&mut self.bitmap, &clip, &pipe, tri);
                }
            }
        }
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
        shading: ResolvedShading,
    ) {
        let paint = |bitmap: &mut Bitmap<Rgb8>, clip: &Clip, src: &PipeSrc<'_>| {
            let fill_fn = if even_odd {
                eo_fill::<Rgb8>
            } else {
                fill::<Rgb8>
            };
            fill_fn(
                bitmap,
                clip,
                path,
                pipe,
                src,
                &DEVICE_MATRIX,
                flatness,
                true,
            );
        };
        if let Some(bg) = shading.background {
            paint(&mut self.bitmap, &clip, &PipeSrc::Solid(&bg));
        }
        let clip = clip_to_bbox(clip, shading.bbox, flatness);
        match shading.result {
            ShadingResult::Pattern(pattern) => {
                paint(&mut self.bitmap, &clip, &PipeSrc::Pattern(pattern.as_ref()));
            }
            ShadingResult::Mesh(triangles) => {
                let mut clip = clip;
                clip.clip_to_path(&XPath::new(path, &DEVICE_MATRIX, flatness, true), even_odd);
                for tri in triangles {
                    gouraud_triangle_fill::<Rgb8>(&mut self.bitmap, &clip, pipe, tri);
                }
            }
        }
    }

    /// Stroke `path` (in the space `matrix` maps to the device) with a
    /// shading pattern under `clip`: the `Background` first, then the
    /// shading clipped by its `BBox`.
    ///
    /// Returns `false`, painting nothing, for a mesh shading, whose
    /// triangles cannot serve as a stroke's colour source.
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
        shading: ResolvedShading,
    ) -> bool {
        let ShadingResult::Pattern(pattern) = shading.result else {
            return false;
        };
        if let Some(bg) = shading.background {
            stroke(
                &mut self.bitmap,
                &clip,
                path,
                pipe,
                &PipeSrc::Solid(&bg),
                matrix,
                params,
            );
        }
        let clip = clip_to_bbox(clip, shading.bbox, flatness);
        stroke(
            &mut self.bitmap,
            &clip,
            path,
            pipe,
            &PipeSrc::Pattern(pattern.as_ref()),
            matrix,
            params,
        );
        true
    }
}
