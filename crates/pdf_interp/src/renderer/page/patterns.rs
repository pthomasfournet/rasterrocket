//! Tiling pattern resolution and tile rendering.

use super::super::color::RasterColor;
use super::super::gstate::ctm_multiply;
use super::pattern_memo::{CellRaster, RasterKey};
use super::pixel_rect::PixelRect;
use super::{MAX_PATTERN_DEPTH, MAX_TILE_PX, PageRenderer, components_to_color};
use crate::resources::{ColorSpace, PageResources, ShadingPattern};
use std::rc::Rc;

use color::Rgb8;
use pdf::Document;
use raster::path::PathBuilder;
use raster::{Bitmap, CELL_PHASES, CellPhase, Lattice, TiledPattern};

/// Resolve an uncoloured-tiling-pattern tint to a [`RasterColor`].
///
/// Per PDF §8.7.3.3, the tint components on `scn /Name ...` are interpreted
/// in the *base* colour space declared by the preceding `cs /Pattern <base>`.
/// When the gstate's fill colour space is `Pattern { base: Some(b) }` we
/// convert through `b`; otherwise we fall back to the legacy component-count
/// heuristic (1 → Gray, 3 → RGB, 4 → CMYK) — handles malformed PDFs that
/// skip the base or set `cs /Pattern` bare.
fn resolve_pattern_tint(doc: &Document, fill_cs: &ColorSpace, components: &[f64]) -> RasterColor {
    if let ColorSpace::Pattern { base: Some(base) } = fill_cs {
        if components.len() == base.ncomponents() {
            return RasterColor::from_bytes(base.convert_to_rgb(doc, components));
        }
        log::debug!(
            "rasterrocket-interp: pattern tint comp count {} mismatches base CS expected {} — using heuristic",
            components.len(),
            base.ncomponents(),
        );
    }
    components_to_color(components)
}

/// Side, in pixels, at which a sub-pixel cell is sampled to take its mean.
const MEAN_SAMPLE_PX: f64 = 8.0;

/// Upper bound on the cell bitmap's area as a multiple of the lattice
/// cell's: cells stacked deeper than this are not painted.
const MAX_CELL_OVERDRAW: f64 = 256.0;

/// Upper bound on the cells rendered in place when the cell exceeds the
/// painted region.
const MAX_DIRECT_CELLS: usize = 4096;

/// The number of whole pixels spanning a length of `v` pixels: its ceiling,
/// taken below a tolerance that absorbs the rounding of the products that
/// produce it (`100 × 0.1` is `10.000000000000002`), and at least 1.
fn pixel_ceil(v: f64) -> f64 {
    (v - 1e-6).ceil().max(1.0)
}

/// The mean of `sample` over a period of `period_px` pixels: its
/// alpha-weighted mean colour and its coverage, `Σ alpha / period_px`
/// clamped to 255.  The colour is black where nothing is painted.
fn mean_over_period(sample: &Bitmap<Rgb8>, period_px: f64) -> ([u8; 3], u8) {
    let mut sum_a = 0u64;
    let mut sum_c = [0u64; 3];
    for y in 0..sample.height {
        let row = sample.row_bytes(y);
        let alpha = sample.alpha_row(y);
        for (x, px) in row.as_chunks::<3>().0.iter().enumerate() {
            let a = u64::from(alpha.map_or(255, |r| r[x]));
            sum_a += a;
            for (s, &c) in sum_c.iter_mut().zip(px) {
                *s += a * u64::from(c);
            }
        }
    }
    if sum_a == 0 {
        return ([0; 3], 0);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the quotients are clamped to [0, 255] before the cast"
    )]
    let coverage = (sum_a as f64 / period_px).round().clamp(0.0, 255.0) as u8;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an alpha-weighted mean of u8 values is at most 255"
    )]
    let colour = sum_c.map(|s| ((s + sum_a / 2) / sum_a) as u8);
    (colour, coverage)
}

/// A resolved `Pattern` resource.
pub(super) enum PatternPaint {
    /// A tiling pattern (`PatternType` 1), rasterised to a tile.
    Tiled(TiledPattern),
    /// A shading pattern (`PatternType` 2).
    Shading(Rc<ShadingPattern>),
}

/// The content of a tiling pattern's cell, ready to render.
struct CellContent<'a> {
    desc: &'a crate::resources::tiling::TilingDescriptor,
    ops: &'a [crate::content::Operator],
    /// The colour of a `PaintType` 2 (uncoloured) pattern; `None` for
    /// `PaintType` 1 (coloured), whose content supplies its own colours.
    tint: Option<&'a RasterColor>,
}

/// The bounding box of a pattern cell in device pixels, relative to the
/// pattern origin.
struct CellBounds {
    min_x: f64,
    min_y: f64,
    width: f64,
    height: f64,
}

/// The bounds of `bbox` under the linear part of `device`; `None` when the
/// box is empty or non-finite.
fn cell_bounds(device: &[f64; 6], bbox: &[f64; 4]) -> Option<CellBounds> {
    let [x0, y0, x1, y1] = *bbox;
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        let dx = device[0].mul_add(x, device[2] * y);
        let dy = device[1].mul_add(x, device[3] * y);
        (min_x, max_x) = (min_x.min(dx), max_x.max(dx));
        (min_y, max_y) = (min_y.min(dy), max_y.max(dy));
    }
    let (width, height) = (max_x - min_x, max_y - min_y);
    (width > 0.0 && height > 0.0 && width.is_finite() && height.is_finite()).then_some(CellBounds {
        min_x,
        min_y,
        width,
        height,
    })
}

/// The geometry of a tiling pattern's cell in device pixels.
struct CellGeometry {
    /// The linear part of the map from pattern space to device pixels.
    linear: [f64; 4],
    /// The cell positions: the top-left pixel of each cell's bounding box.
    lattice: Lattice,
    /// The area of one lattice cell, in pixels.
    area: f64,
    /// The cell's bounding box, relative to the pattern origin.
    bounds: CellBounds,
}

impl CellGeometry {
    /// The geometry of `desc` under `device`, the map from pattern space to
    /// device pixels; `Err` names what makes the pattern unpaintable.
    fn new(
        device: &[f64; 6],
        desc: &crate::resources::tiling::TilingDescriptor,
    ) -> Result<Self, &'static str> {
        if !device.iter().all(|v| v.is_finite()) {
            return Err("matrix is non-finite");
        }
        let bounds = cell_bounds(device, &desc.bbox).ok_or("cell is empty")?;
        // Cell (col, row) is the image of (col · XStep, row · YStep).
        let lattice = Lattice {
            origin: (device[4] + bounds.min_x, device[5] + bounds.min_y),
            col_step: (device[0] * desc.x_step, device[1] * desc.x_step),
            row_step: (device[2] * desc.y_step, device[3] * desc.y_step),
        };
        let area = lattice.determinant().abs();
        if area.is_nan() || area <= 0.0 {
            return Err("lattice is degenerate");
        }
        Ok(Self {
            linear: [device[0], device[1], device[2], device[3]],
            lattice,
            area,
            bounds,
        })
    }

    /// The whole-pixel sides of the cell's bitmap.
    fn cell_size(&self) -> (f64, f64) {
        (
            pixel_ceil(self.bounds.width),
            pixel_ceil(self.bounds.height),
        )
    }

    /// The map from pattern space to the pixels of a bitmap on which the
    /// cell's bounding box has its top-left corner at `(x, y)`, scaled by
    /// `k`.
    fn placement(&self, k: f64, x: f64, y: f64) -> [f64; 6] {
        let linear = self.linear.map(|v| v * k);
        [
            linear[0],
            linear[1],
            linear[2],
            linear[3],
            self.bounds.min_x.mul_add(-k, x),
            self.bounds.min_y.mul_add(-k, y),
        ]
    }
}

impl PageRenderer<'_> {
    /// Resolve the named `Pattern` resource as a shading or tiling pattern.
    /// A tiling pattern is composited over `region`, the device pixels the
    /// painting operator can touch.
    pub(super) fn resolve_pattern(
        &mut self,
        name: &[u8],
        region: PixelRect,
    ) -> Option<PatternPaint> {
        if let Some(shading) = self.memo_shading_pattern(name) {
            return Some(PatternPaint::Shading(shading));
        }
        self.resolve_fill_pattern(name, region)
            .map(PatternPaint::Tiled)
    }

    /// Resolve a tiling pattern over `region`.
    ///
    /// The cell is rasterised once per quarter-pixel phase and composited
    /// at every lattice point whose cell meets the region when the cell is
    /// no larger than the region, or when the same raster was asked for
    /// before and fits the page's pattern memo; otherwise each such cell's
    /// content is rendered directly onto a region-sized bitmap at its exact
    /// position.  A lattice finer than a pixel is reduced to its mean.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the region's area is page-bounded; the cell's byte count is positive and saturates"
    )]
    pub(super) fn resolve_fill_pattern(
        &mut self,
        name: &[u8],
        region: PixelRect,
    ) -> Option<TiledPattern> {
        // Break self-referential / cyclic pattern chains.  Each cell spawns a
        // fresh PageRenderer with its own pattern_depth seeded one above the
        // parent (see render_cells below), so reaching the cap means a
        // pattern's content stream re-invokes a pattern and we'd otherwise
        // recurse to stack overflow on adversarial input.
        if self.pattern_depth >= MAX_PATTERN_DEPTH {
            log::warn!(
                "rasterrocket-interp: Pattern /{} nesting depth {MAX_PATTERN_DEPTH} exceeded — skipping",
                String::from_utf8_lossy(name),
            );
            return None;
        }

        let tiling = self.memo_tiling_cell(name)?;
        let desc = &tiling.desc;

        let gs = self.gstate.current();
        // PaintType 2 (uncoloured) patterns paint shape-only content streams;
        // the tint comes from the `scn /Name c1 [c2 c3 [c4]]` arguments captured
        // here as `fill_pattern_components`.  PaintType 1 (coloured) ignores it.
        //
        // PDF §8.7.3.3: the tint is interpreted in whichever colour space was
        // declared by the preceding `cs /Pattern <base_cs>` operator.  When
        // we have that base recorded on the gstate, route through it;
        // otherwise fall back to the legacy component-count heuristic
        // (1 → Gray, 3 → RGB, 4 → CMYK) for malformed PDFs that skip the
        // base or use a bare `/Pattern` cs.
        let tint = if desc.paint_type == 2 && !gs.fill_pattern_components.is_empty() {
            Some(resolve_pattern_tint(
                self.resources.doc(),
                &gs.fill_color_space,
                &gs.fill_pattern_components,
            ))
        } else {
            None
        };
        // PDF §8.7.3.1: the pattern matrix maps pattern space to the default
        // coordinate space of the parent content stream, not to the CTM in
        // effect at the fill.  `device` carries on to device pixels.
        let pat_ctm = ctm_multiply(&desc.matrix, &self.pattern_ctm);
        let device = flip_y(&pat_ctm, f64::from(self.height));
        let geometry = match CellGeometry::new(&device, desc) {
            Ok(geometry) => geometry,
            Err(why) => {
                log::debug!(
                    "rasterrocket-interp: Pattern /{} {why} — skipping",
                    String::from_utf8_lossy(name)
                );
                return None;
            }
        };
        let cell = CellContent {
            desc,
            ops: &tiling.ops,
            tint: tint.as_ref(),
        };
        let key = RasterKey::new(
            desc.stream_id,
            &device,
            tint.as_ref(),
            (!desc.has_own_resources).then(|| self.resources.resource_context_id()),
        );

        // A lattice cell below one pixel in area repeats more than once per
        // pixel; box-filtered over a pixel, the pattern is its mean.
        if geometry.area < 1.0 {
            return Some(self.mean_pattern(&cell, &geometry, key));
        }
        let (cell_w, cell_h) = geometry.cell_size();
        if cell_w * cell_h > MAX_CELL_OVERDRAW * geometry.area {
            log::warn!(
                "rasterrocket-interp: Pattern /{} cell ({cell_w}×{cell_h} px) overlaps its lattice ({:.1} px² per cell) more than {MAX_CELL_OVERDRAW}× — skipping",
                String::from_utf8_lossy(name),
                geometry.area,
            );
            return None;
        }
        // A cell larger than the region is rendered in place, unless the
        // same cell raster serves repeated fills.
        let cell_bytes = (cell_w + 1.0) * (cell_h + 1.0) * 4.0;
        if cell_w * cell_h <= region.area() as f64
            || self
                .pattern_memo
                .borrow_mut()
                .repeated(&key, cell_bytes as usize)
        {
            self.composed_pattern(&cell, &geometry, region, &key)
        } else {
            self.direct_pattern(name, &cell, &geometry, region)
        }
    }

    /// The mean colour and coverage of a cell over one lattice period,
    /// sampled at about `MEAN_SAMPLE_PX` across.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the sample sides are whole pixels from pixel_ceil, at most MAX_TILE_PX"
    )]
    fn mean_pattern(
        &mut self,
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        key: RasterKey,
    ) -> TiledPattern {
        let raster = self.memo_raster(key, |r| {
            let (w, h) = (geometry.bounds.width, geometry.bounds.height);
            let k = (MEAN_SAMPLE_PX / w.min(h)).min(MAX_TILE_PX / w.max(h));
            let size = (pixel_ceil(w * k) as u32, pixel_ceil(h * k) as u32);
            let sample = r.render_cells(cell, size, &[geometry.placement(k, 0.0, 0.0)]);
            let (colour, coverage) = mean_over_period(&sample, geometry.area * k * k);
            CellRaster::Mean(colour, coverage)
        });
        match raster {
            CellRaster::Mean(colour, coverage) => TiledPattern::uniform(colour, coverage),
            CellRaster::Cell(_) => unreachable!("a mean is kept under its key"),
        }
    }

    /// The cell rasterised once per quarter-pixel phase and composited at
    /// every lattice point whose cell meets `region`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "the cell sides are whole pixels bounded by the region or the memo budget; region coordinates are page-bounded"
    )]
    fn composed_pattern(
        &mut self,
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        region: PixelRect,
        key: &RasterKey,
    ) -> Option<TiledPattern> {
        // Every phase's bitmap has room for the cell offset by up to one
        // pixel.
        let size = (
            pixel_ceil(geometry.bounds.width + 1.0) as u32,
            pixel_ceil(geometry.bounds.height + 1.0) as u32,
        );
        let scale = f64::from(CELL_PHASES).recip();
        let raster_at = |r: &mut Self, phase: CellPhase| {
            let raster = r.memo_raster(key.at_phase(phase), |r| {
                let (x, y) = (f64::from(phase.0) * scale, f64::from(phase.1) * scale);
                CellRaster::Cell(Rc::new(r.render_cells(
                    cell,
                    size,
                    &[geometry.placement(1.0, x, y)],
                )))
            });
            let CellRaster::Cell(bitmap) = raster else {
                unreachable!("a cell bitmap is kept under its key")
            };
            bitmap
        };
        TiledPattern::compose(
            |phase| raster_at(self, phase),
            size,
            &geometry.lattice,
            region.x0 as i32,
            region.y0 as i32,
            region.width(),
            region.height(),
        )
    }

    /// The cells meeting `region` rendered in place on a region-sized
    /// bitmap, for a cell larger than the region.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "the cell sides are whole pixels from pixel_ceil; region coordinates are page-bounded"
    )]
    fn direct_pattern(
        &mut self,
        name: &[u8],
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        region: PixelRect,
    ) -> Option<TiledPattern> {
        let (cell_w, cell_h) = geometry.cell_size();
        let (rx0, ry0) = (region.x0 as i32, region.y0 as i32);
        let rect = (rx0, ry0, region.x1 as i32, region.y1 as i32);
        let (cols, rows) = geometry
            .lattice
            .cells_meeting(rect, cell_w as u32, cell_h as u32)?;
        let placements: Vec<[f64; 6]> = rows
            .flat_map(|row| cols.clone().map(move |col| (col, row)))
            .take(MAX_DIRECT_CELLS + 1)
            .map(|(col, row)| {
                let (px, py) = geometry.lattice.position(col, row);
                geometry.placement(1.0, px - f64::from(rx0), py - f64::from(ry0))
            })
            .collect();
        if placements.len() > MAX_DIRECT_CELLS {
            log::warn!(
                "rasterrocket-interp: Pattern /{} needs more than {MAX_DIRECT_CELLS} cells over a {}×{} px area — skipping",
                String::from_utf8_lossy(name),
                region.width(),
                region.height(),
            );
            return None;
        }
        let size = (region.width(), region.height());
        let screen = self.render_cells(cell, size, &placements);
        Some(TiledPattern::screen(screen, rx0, ry0))
    }

    /// Render the pattern cell once per matrix of `placements` onto one
    /// `width × height` bitmap, initially transparent, whose alpha plane
    /// records the coverage of the content.  Each placement maps pattern
    /// space to the bitmap's pixels (y down); the content is clipped to the
    /// cell's `BBox` (PDF §8.7.3.1).
    ///
    /// A child `PageRenderer` scoped to the pattern's resource context runs
    /// the content.  An uncoloured pattern's tint is set as the fill and
    /// stroke colour before the content runs.
    fn render_cells(
        &mut self,
        cell: &CellContent<'_>,
        (width, height): (u32, u32),
        placements: &[[f64; 6]],
    ) -> Bitmap<Rgb8> {
        let desc = cell.desc;
        // Build a temporary child renderer sharing the same document / font engine.
        let doc = self.resources.doc();
        let mut child = PageRenderer::new(width, height, doc, desc.stream_id);
        child.set_document_fonts(std::sync::Arc::clone(self.font_cache.document_fonts()));
        child.set_antialias(self.text_antialias, self.vector_antialias);
        child.bitmap = Bitmap::new(width, height, 1, true);
        // Propagate pattern_depth so a pattern stream that recursively
        // invokes another pattern (or itself) hits MAX_PATTERN_DEPTH instead
        // of unbounded stack recursion.  saturating_add guards against the
        // (impossible) overflow.
        child.pattern_depth = self.pattern_depth.saturating_add(1);
        // Share the per-page work watchdog.  A fresh PageRenderer would
        // otherwise get its own 50 M-op budget and a fresh 60 s deadline,
        // letting a pathological pattern content stream (nested up to
        // MAX_PATTERN_DEPTH) escape the page watchdog entirely — the parent's
        // execute() loop is blocked synchronously here and never gets to tick
        // its own budget or check its own deadline.
        child.adopt_parent_budget(self);
        child.share_memos(self);
        child.set_optional_content(std::sync::Arc::clone(self.optional_content()));

        // If the pattern has no own resources, point it at the parent context.
        if !desc.has_own_resources {
            child.resources = PageResources::new(doc, self.resources.resource_context_id());
        }

        for placement in placements {
            let ctm = flip_y(placement, f64::from(height));
            child.pattern_ctm = ctm;
            let depth = child.gstate.depth();
            child.gstate.save();
            let gs = child.gstate.current_mut();
            gs.ctm = ctm;
            if let Some(c) = cell.tint {
                gs.fill_color = c.clone();
                gs.stroke_color = c.clone();
            }
            child.clip_to_user_rect(&desc.bbox);
            child.execute(cell.ops);
            child.gstate.restore_to(depth);
            if child.budget_exceeded.is_some() {
                break;
            }
        }
        // Account the cells' work against the shared page budget and propagate
        // any breach so the parent's execute() loop bails loudly on its next
        // iteration instead of rendering more pattern fills unbounded.
        child.fold_child_into(self);
        child.finish().0
    }

    /// Intersect the clip with `bbox`, a rectangle `[x0, y0, x1, y1]` in
    /// user space, under the current CTM.  A non-finite device image of the
    /// rectangle leaves the clip unchanged.
    #[expect(
        clippy::float_cmp,
        reason = "only a quad whose corners align exactly takes the rectangle clip"
    )]
    pub(super) fn clip_to_user_rect(&mut self, bbox: &[f64; 4]) {
        let [x0, y0, x1, y1] = *bbox;
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| self.to_device(x, y));
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return;
        }
        let axis_aligned = corners[0].1 == corners[1].1
            && corners[1].0 == corners[2].0
            && corners[2].1 == corners[3].1
            && corners[3].0 == corners[0].0;
        if axis_aligned {
            let (xa, xb) = (
                corners[0].0.min(corners[1].0),
                corners[0].0.max(corners[1].0),
            );
            let (ya, yb) = (
                corners[0].1.min(corners[2].1),
                corners[0].1.max(corners[2].1),
            );
            self.gstate.current_mut().clip.clip_to_rect(xa, ya, xb, yb);
            return;
        }
        let mut pb = PathBuilder::new();
        let _ = pb.move_to(corners[0].0, corners[0].1);
        for (x, y) in &corners[1..] {
            let _ = pb.line_to(*x, *y);
        }
        let _ = pb.close(true);
        self.clip_path_into_gstate(&pb.build(), false);
    }
}

/// `m` followed by the flip between y-up coordinates and the y-down rows of
/// a bitmap `height` pixels tall.  The flip is an involution, so this maps
/// a CTM to the device matrix of a renderer and back.
fn flip_y(m: &[f64; 6], height: f64) -> [f64; 6] {
    [m[0], -m[1], m[2], -m[3], m[4], height - m[5]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::empty_doc;

    #[test]
    fn pixel_ceil_absorbs_product_rounding() {
        assert_eq!(pixel_ceil(100.0 * 0.1), 10.0);
        assert_eq!(pixel_ceil(10.5), 11.0);
        assert_eq!(pixel_ceil(0.3), 1.0);
    }

    #[test]
    fn mean_over_period_weights_colour_by_alpha_and_coverage_by_period() {
        // 2 × 1 sample: opaque red, half-transparent blue; period of 4 px.
        let mut bmp = Bitmap::<Rgb8>::new(2, 1, 1, true);
        bmp.row_bytes_mut(0)
            .copy_from_slice(&[255, 0, 0, 0, 0, 255]);
        bmp.alpha_plane_mut().unwrap().copy_from_slice(&[255, 128]);
        let (colour, coverage) = mean_over_period(&bmp, 4.0);
        assert_eq!(colour, [170, 0, 85]);
        assert_eq!(coverage, 96);

        // Nothing painted: no colour, no coverage.
        let empty = Bitmap::<Rgb8>::new(2, 1, 1, true);
        assert_eq!(mean_over_period(&empty, 1.0), ([0; 3], 0));

        // Overlapping content past the period saturates the coverage.
        let (_, full) = mean_over_period(&bmp, 1.0);
        assert_eq!(full, 255);
    }

    #[test]
    fn pattern_tint_devicergb_base_passes_through() {
        // Verifies the DeviceRGB base path doesn't regress vs the heuristic.
        // The Lab test below proves the new path is actually taken (since
        // the heuristic would mis-interpret a Lab tint as RGB).
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceRgb)),
        };
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[1.0, 0.0, 0.0]);
        assert_eq!(tint.as_slice(), &[255, 0, 0]);
    }

    #[test]
    fn pattern_tint_lab_base_routes_through_lab_conversion() {
        // Three-component tint that the heuristic would interpret as RGB,
        // but Lab interprets very differently.  L=50,a=0,b=0 ≈ mid-grey.
        let doc = empty_doc();
        let lab = crate::resources::colorspace::resolve(
            &doc,
            &pdf::parse_object(b"[/Lab << /WhitePoint [0.9505 1 1.089] >>]", &mut 0).unwrap(),
        );
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(lab)),
        };
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[50.0, 0.0, 0.0]);
        // sRGB-encoded mid-grey from L*=50 is ~119 (gamma-correct).
        let &[r, g, b] = tint.as_slice() else {
            panic!("RasterColor must produce a 3-byte slice");
        };
        assert!(
            (115..=125).contains(&r) && r == g && g == b,
            "expected mid-grey ~119 from Lab L=50, got [{r},{g},{b}]"
        );
    }

    #[test]
    fn pattern_tint_falls_back_when_no_base() {
        // No base on the Pattern (malformed PDF) → heuristic path.
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern { base: None };
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[0.5, 0.5, 0.5]);
        // components_to_color heuristic: 3 components → RGB(0.5, 0.5, 0.5).
        assert_eq!(tint.as_slice(), &[128, 128, 128]);
    }

    #[test]
    fn pattern_tint_falls_back_on_count_mismatch() {
        // Base says RGB (3 components) but caller supplies 1 → heuristic.
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceRgb)),
        };
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[0.5]);
        // Heuristic: 1 component → Gray(0.5) = mid-grey.
        assert_eq!(tint.as_slice(), &[128, 128, 128]);
    }

    #[test]
    fn pattern_tint_devicegray_base_one_component() {
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceGray)),
        };
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[1.0]);
        assert_eq!(tint.as_slice(), &[255, 255, 255]);
    }

    #[test]
    fn pattern_tint_devicecmyk_base_four_components() {
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceCmyk)),
        };
        // (0,0,0,1) = solid K: all four components reach the CMYK conversion.
        let tint = resolve_pattern_tint(&doc, &fill_cs, &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(tint.as_slice(), &color::cmyk::cmyk_to_rgb(0, 0, 0, 255));
    }
}
