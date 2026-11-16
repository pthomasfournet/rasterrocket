//! Pattern and shading resources (ISO 32000-1 §8.7) of the resource
//! dictionaries in effect.

use super::{Category, PageResources};
use crate::resources::dict_ext::DictExt;
use crate::resources::{
    ExtGStateParams, read_matrix, resolve_dict, resolve_stream_dict, shading, tiling,
};

/// A shading pattern (`PatternType` 2, PDF §8.7.3.3) resolved for painting.
#[non_exhaustive]
pub struct ShadingPattern {
    /// The pattern's shading, placed by `matrix`.
    pub shading: shading::ResolvedShading,
    /// The pattern's `ExtGState`: graphics state parameters in effect while
    /// the shading is painted.
    pub ext_gstate: Option<ExtGStateParams>,
    /// The pattern matrix concatenated onto the default coordinate space of
    /// the pattern's parent content stream: pattern space to device space,
    /// before the device y-flip.
    pub matrix: [f64; 6],
}

impl PageResources<'_> {
    /// Resolve the named `Pattern` resource as a tiling descriptor.
    ///
    /// Returns `None` if the name is absent, the pattern is not `PatternType 1`
    /// (tiling), or any required key is missing.  A debug message is logged for
    /// `PatternType` 2 (shading patterns referenced via `scn`).
    #[must_use]
    pub fn tiling_pattern(&self, name: &[u8]) -> Option<tiling::TilingDescriptor> {
        let patterns = self.resource_subdict(Category::Pattern)?;
        tiling::resolve_tiling(self.doc, patterns, name)
    }

    /// Resolve the named `Shading` resource, for an ink canvas when `ink`
    /// plans its colorants (see [`shading::resolve_shading_object`]).
    ///
    /// Returns `None` if the name is absent, the type is unsupported, or any
    /// required key is missing.
    #[must_use]
    pub(crate) fn shading(
        &self,
        name: &[u8],
        ctm: &[f64; 6],
        canvas: shading::Canvas,
        ink: Option<shading::PlanColorants<'_>>,
    ) -> Option<shading::ResolvedShading> {
        let shadings = self.resource_subdict(Category::Shading)?;
        shading::resolve_shading(self, shadings, name, ctm, canvas, ink)
    }

    /// Resolve the named `Pattern` resource as a shading pattern
    /// (`PatternType` 2, PDF §8.7.4.2).
    ///
    /// `pattern_ctm` maps the default coordinate space of the pattern's
    /// parent content stream to the device; the pattern's `Matrix` is
    /// concatenated onto it.  The shading is resolved for an ink canvas
    /// when `ink` plans its colorants.  Returns `None` if the name is
    /// absent, the pattern is not a shading pattern, or its shading does
    /// not resolve.
    #[must_use]
    pub(crate) fn shading_pattern(
        &self,
        name: &[u8],
        pattern_ctm: &[f64; 6],
        canvas: shading::Canvas,
        ink: Option<shading::PlanColorants<'_>>,
    ) -> Option<ShadingPattern> {
        let patterns = self.resource_subdict(Category::Pattern)?;
        let pattern = resolve_stream_dict(self.doc, patterns.get(name)?)?;
        if pattern.get_i64(b"PatternType") != Some(2) {
            return None;
        }
        let matrix = read_matrix(&pattern).map_or(*pattern_ctm, |m| {
            crate::renderer::gstate::ctm_multiply(&m, pattern_ctm)
        });
        let Some(shading) =
            shading::resolve_shading_object(self, pattern.get(b"Shading")?, &matrix, canvas, ink)
        else {
            log::warn!(
                "rasterrocket-interp: shading pattern /{} has no usable Shading — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        };
        let ext_gstate = pattern
            .get(b"ExtGState")
            .and_then(|obj| resolve_dict(self.doc, obj))
            .map(|d| ExtGStateParams::from_dict(self.doc, &d));
        Some(ShadingPattern {
            shading,
            ext_gstate,
            matrix,
        })
    }
}
