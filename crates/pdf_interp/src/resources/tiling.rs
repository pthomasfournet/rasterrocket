//! PDF tiling pattern resource resolution (`PatternType` 1).
//!
//! A tiling pattern is a content stream that is repeatedly stamped across the
//! painting area at fixed `XStep` × `YStep` intervals (PDF §8.7.3).
//!
//! # Paint types
//!
//! A coloured pattern (`PaintType` 1) paints its cell in its own colours.
//! An uncoloured one (`PaintType` 2) paints a stencil in the tint the `scn`
//! operands give in the pattern space's underlying colour space.
//!
//! # Coordinate system
//!
//! The pattern's `Matrix` transforms from pattern space into user space; this is
//! then combined with the caller's CTM to reach device space.  The tile is
//! rasterised at the scale implied by `ctm * matrix` so that one pattern-space
//! unit maps to the correct number of device pixels.

use pdf::{Dictionary, Document, Object, ObjectId};

use super::{read_bbox, read_f64_1, read_matrix};

/// Parameters extracted from a PDF Type 1 (tiling) pattern stream.
///
/// Callers use these to rasterise a tile bitmap and assemble a
/// `TiledPattern` (private to `crate::renderer::page::patterns`).
#[non_exhaustive]
pub struct TilingDescriptor {
    /// Content stream of the pattern tile (already decompressed).
    pub content: Vec<u8>,
    /// Object ID of the pattern stream — used as the resource context for fonts
    /// and sub-objects referenced from the tile content.
    pub stream_id: ObjectId,
    /// True if the pattern stream carries its own `Resources` dict; false means
    /// the tile inherits from the parent context (rare but allowed by the spec).
    pub has_own_resources: bool,
    /// Pattern bounding box `[xmin ymin xmax ymax]` in pattern space.
    pub bbox: [f64; 4],
    /// X spacing between tile origins in pattern space.
    pub x_step: f64,
    /// Y spacing between tile origins in pattern space.
    pub y_step: f64,
    /// Pattern-to-user-space matrix (default: identity).
    pub matrix: [f64; 6],
    /// `PaintType`.
    pub paint_type: PaintType,
}

/// The `PaintType` of a tiling pattern (PDF §8.7.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PaintType {
    /// `1`: the cell's content stream sets its own colours.
    Coloured,
    /// `2`: the cell is a stencil painted in the tint given with the
    /// pattern.
    Uncoloured,
}

/// Look up a named `Pattern` resource and extract its tiling descriptor.
///
/// `patterns` is the `/Pattern` resource sub-dictionary in effect.  Returns
/// `None` if the name is absent, the object is not a
/// tiling pattern, or any required key is missing.
#[must_use]
pub fn resolve_tiling(
    doc: &Document,
    patterns: &Dictionary,
    name: &[u8],
) -> Option<TilingDescriptor> {
    let pat_obj = patterns.get(name)?;

    let Object::Reference(id) = pat_obj else {
        log::debug!(
            "rasterrocket-interp: Pattern /{} is not a reference — skipping",
            String::from_utf8_lossy(name)
        );
        return None;
    };
    let stream_id = *id;

    let obj = doc.get_object(stream_id).ok()?;
    let stream = obj.as_ref().as_stream()?;

    // Must be `PatternType` 1 (tiling).
    let pattern_type = stream.dict.get(b"PatternType")?.as_i64()?;
    if pattern_type != 1 {
        log::debug!(
            "rasterrocket-interp: Pattern /{} has PatternType {pattern_type} (only 1 supported for tiling)",
            String::from_utf8_lossy(name)
        );
        return None;
    }

    let missing = |key: &str| -> Option<TilingDescriptor> {
        log::warn!(
            "rasterrocket-interp: tiling pattern /{} has no usable /{key} — skipping",
            String::from_utf8_lossy(name)
        );
        None
    };
    let paint_type = match stream.dict.get(b"PaintType").and_then(Object::as_i64) {
        Some(1) => PaintType::Coloured,
        Some(2) => PaintType::Uncoloured,
        other => {
            log::warn!(
                "rasterrocket-interp: tiling pattern /{} has PaintType {other:?}, not 1 or 2 — \
                 painting it as coloured",
                String::from_utf8_lossy(name)
            );
            PaintType::Coloured
        }
    };
    let Some(bbox) = read_bbox(&stream.dict) else {
        return missing("BBox");
    };
    let (Some(x_step), Some(y_step)) = (
        read_f64_1(&stream.dict, b"XStep"),
        read_f64_1(&stream.dict, b"YStep"),
    ) else {
        return missing("XStep or YStep");
    };

    // Zero or NaN step would cause divide-by-zero or infinite tile counts downstream.
    if !x_step.is_finite() || !y_step.is_finite() || x_step == 0.0 || y_step == 0.0 {
        log::warn!(
            "rasterrocket-interp: Pattern /{} has invalid XStep ({x_step}) or YStep ({y_step}) — skipping",
            String::from_utf8_lossy(name)
        );
        return None;
    }

    let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let content = stream
        .decompressed_content()
        .map_err(|e| {
            log::warn!(
                "rasterrocket-interp: Pattern /{} — failed to decompress content stream: {e}",
                String::from_utf8_lossy(name)
            );
        })
        .ok()?;
    let has_own_resources = stream.dict.get(b"Resources").is_some();

    Some(TilingDescriptor {
        content,
        stream_id,
        has_own_resources,
        bbox,
        x_step,
        y_step,
        matrix,
        paint_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_objects;

    fn tiling(dict: &str) -> Option<TilingDescriptor> {
        let doc = make_doc_with_objects(&[(
            4,
            format!("<< /PatternType 1 {dict} /Length 0 >>\nstream\n\nendstream"),
        )]);
        let mut patterns = Dictionary::new();
        patterns.set("P", Object::Reference((4, 0)));
        resolve_tiling(&doc, &patterns, b"P")
    }

    #[test]
    fn the_paint_type_selects_coloured_or_uncoloured() {
        let cell = "/BBox [0 0 8 8] /XStep 8 /YStep 8";
        let paint = |p: &str| tiling(&format!("{cell} {p}")).map(|d| d.paint_type);
        assert_eq!(paint("/PaintType 1"), Some(PaintType::Coloured));
        assert_eq!(paint("/PaintType 2"), Some(PaintType::Uncoloured));
        assert_eq!(
            paint(""),
            Some(PaintType::Coloured),
            "absent reads as coloured"
        );
        assert_eq!(paint("/PaintType 7"), Some(PaintType::Coloured));
    }

    #[test]
    fn a_pattern_without_a_cell_or_step_is_refused() {
        assert!(tiling("/PaintType 1 /XStep 8 /YStep 8").is_none());
        assert!(tiling("/PaintType 1 /BBox [0 0 8 8] /XStep 8").is_none());
        assert!(tiling("/PaintType 1 /BBox [0 0 8 8] /XStep 0 /YStep 8").is_none());
    }
}
