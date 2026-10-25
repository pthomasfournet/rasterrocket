//! Shading painting: the `sh` operator (PDF §8.7.4).
//!
//! A shading paints only the points within its extent: an axial or radial
//! shading's parameter range (as extended by `Extend`), a function-based
//! shading's `Domain`, a mesh's patches.  The raster patterns report the
//! points they leave unpainted, so the backdrop survives there.  `BBox`, when
//! present, further clips the shading.  `Background` has no effect through
//! `sh` (§8.7.4.3).

use color::Rgb8;
use raster::clip::Clip;
use raster::path::{Path, PathBuilder};
use raster::shading::{gouraud::gouraud_triangle_fill, shaded_fill};
use raster::xpath::XPath;

use super::{DEVICE_MATRIX, PageRenderer};
use crate::resources::shading::ShadingResult;

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
}
