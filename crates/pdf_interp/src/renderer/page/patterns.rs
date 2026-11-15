//! Tiling pattern resolution and tile rendering.

use super::super::color::{InkChannels, RasterColor};
use super::super::gstate::ctm_multiply;
use super::canvas::{CanvasKind, PageCanvas};
use super::pattern_memo::{CellRaster, RasterKey, cell_raster_bytes};
use super::pixel_rect::PixelRect;
use super::{DEVICE_MATRIX, MAX_PATTERN_DEPTH, MAX_TILE_PX, PageRenderer, components_to_color};
use crate::ColourIntent;
use crate::resources::{ColorSpace, ShadingPattern};
use std::rc::Rc;

use color::{DeviceN8, DeviceN16, Pixel, Rgb8, Rgb16, Sample, SampleDepth};
use pdf::Document;
use raster::path::PathBuilder;
use raster::pipe::ChannelMask;
use raster::{Bitmap, CELL_PHASES, CellPhase, Lattice, TiledPattern, XPath};

/// Resolve an uncoloured-tiling-pattern tint to a [`RasterColor`].
///
/// The tint components of `scn /Name c1 … cn` are colour values in the
/// underlying colour space of the fill's `[/Pattern base]` space
/// (PDF §8.7.3.3), and convert through `base` under `intent`.  A bare `/Pattern` space,
/// or a component count `base` does not take, is read by its count:
/// 1 grey, 3 RGB, 4 CMYK.
fn resolve_pattern_tint(
    doc: &Document,
    intent: ColourIntent,
    fill_cs: &ColorSpace,
    components: &[f64],
    depth: SampleDepth,
) -> RasterColor {
    if let ColorSpace::Pattern { base: Some(base) } = fill_cs {
        if components.len() == base.ncomponents() {
            return RasterColor::from_srgb(base.convert_to_srgb(doc, intent, components, depth));
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

/// Largest raster, in bytes, of a cell composited at lattice points; a
/// larger cell is rendered in place.
const MAX_COMPOSED_CELL_BYTES: usize = 64 << 20;

/// The number of whole pixels spanning a length of `v` pixels: its ceiling,
/// taken below a tolerance that absorbs the rounding of the products that
/// produce it (`100 × 0.1` is `10.000000000000002`), and at least 1.
fn pixel_ceil(v: f64) -> f64 {
    (v - 1e-6).ceil().max(1.0)
}

/// The mean of `sample` over a period of `period_px` pixels: its
/// alpha-weighted mean colour, one pixel's samples, and its coverage,
/// `Σ alpha / period_px` clamped to the sample's maximum.  The colour is
/// all zeros where nothing is painted.
fn mean_over_period<P: Pixel>(sample: &Bitmap<P>, period_px: f64) -> (Vec<P::Sample>, P::Sample) {
    let wide = |v: P::Sample| u64::try_from(v.signed()).unwrap_or(0);
    let mut sum_a = 0u64;
    let mut sum_c = vec![0u64; P::NCOMPS];
    for y in 0..sample.height {
        let row = sample.row_samples(y);
        let alpha = sample.alpha_row(y);
        for (x, px) in row.chunks_exact(P::NCOMPS).enumerate() {
            let a = wide(alpha.map_or(P::Sample::MAX, |r| r[x]));
            sum_a += a;
            for (s, &c) in sum_c.iter_mut().zip(px) {
                *s += a * wide(c);
            }
        }
    }
    if sum_a == 0 {
        return (vec![P::Sample::default(); P::NCOMPS], P::Sample::default());
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "the quotient is rounded, and clamped to the sample range by from_signed"
    )]
    let coverage = P::Sample::from_signed((sum_a as f64 / period_px).round() as i64);
    // An alpha-weighted mean of samples is at most the sample's maximum.
    let colour = sum_c
        .iter()
        .map(|s| P::Sample::from_signed(i64::try_from((s + sum_a / 2) / sum_a).unwrap_or(0)))
        .collect();
    (colour, coverage)
}

/// A resolved `Pattern` resource.
pub(super) enum PatternPaint {
    /// A tiling pattern (`PatternType` 1), rasterised to a tile.
    Tiled(Tile),
    /// A shading pattern (`PatternType` 2).
    Shading(Rc<ShadingPattern>),
}

/// A tiling pattern rasterised to a tile of the canvas's pixel type.
pub(super) enum Tile {
    /// A tile of sRGB, for an RGB canvas.
    Rgb(TiledPattern),
    /// A tile of 16-bit sRGB, for a 16-bit RGB canvas.
    Rgb16(TiledPattern<Rgb16>),
    /// A tile of ink, for an ink canvas, with the channels the cell's
    /// content painted and carried: the union over its paints of the
    /// channels each wrote and those its colour carried.
    Ink {
        pattern: TiledPattern<DeviceN8>,
        channels: InkChannels,
    },
    /// A tile of 16-bit ink, for a 16-bit ink canvas, with the channels
    /// as for [`Self::Ink`].
    Ink16 {
        pattern: TiledPattern<DeviceN16>,
        channels: InkChannels,
    },
}

impl Tile {
    /// The channels the tile paints on an ink canvas: those its cell's
    /// content painted for a tile of ink, every channel for one of sRGB.
    pub(super) const fn paints(&self) -> ChannelMask {
        match self {
            Self::Rgb(_) | Self::Rgb16(_) => ChannelMask::ALL,
            Self::Ink { channels, .. } | Self::Ink16 { channels, .. } => channels.paints,
        }
    }

    /// The channels that carry the tile's colours on an ink canvas: those
    /// its cell's content carried for a tile of ink, every channel for one
    /// of sRGB.
    pub(super) const fn carries(&self) -> ChannelMask {
        match self {
            Self::Rgb(_) | Self::Rgb16(_) => ChannelMask::ALL,
            Self::Ink { channels, .. } | Self::Ink16 { channels, .. } => channels.carries,
        }
    }
}

/// The mean colour, coverage and channels of a cell as the pattern memo
/// keeps them.
type KeptMean<'a, S> = (&'a [S], S, InkChannels);

/// The pixel type a tiling cell is rasterised in: the pixel type of the
/// canvas the pattern paints, so that a cell on an ink canvas keeps its
/// spot colorants and the channels its content wrote.
trait CellPixel: Pixel {
    /// The canvas of this pixel type.
    const KIND: CanvasKind;

    /// The bitmap of `canvas`, when it is of this pixel type.
    fn bitmap(canvas: PageCanvas) -> Option<Bitmap<Self>>;

    /// A cell bitmap and the channels its content painted and carried, as
    /// the pattern memo keeps them.
    fn cell(bitmap: Rc<Bitmap<Self>>, channels: InkChannels) -> CellRaster;

    /// The mean colour, coverage and channels of a cell, as the pattern
    /// memo keeps them.
    fn mean(colour: &[Self::Sample], coverage: Self::Sample, channels: InkChannels) -> CellRaster;

    /// The cell bitmap and channels of a kept raster of this pixel type.
    fn as_cell(raster: CellRaster) -> Option<(Rc<Bitmap<Self>>, InkChannels)>;

    /// The mean colour, coverage and channels of a kept raster of this
    /// pixel type.
    fn as_mean(raster: &CellRaster) -> Option<KeptMean<'_, Self::Sample>>;
}

impl CellPixel for Rgb8 {
    const KIND: CanvasKind = CanvasKind::Rgb;

    fn bitmap(canvas: PageCanvas) -> Option<Bitmap<Self>> {
        match canvas {
            PageCanvas::Rgb(b) => Some(b),
            PageCanvas::Rgb16(_) | PageCanvas::Ink(_) | PageCanvas::Ink16(_) => None,
        }
    }

    fn cell(bitmap: Rc<Bitmap<Self>>, _channels: InkChannels) -> CellRaster {
        CellRaster::Cell(bitmap)
    }

    fn mean(colour: &[u8], coverage: u8, _channels: InkChannels) -> CellRaster {
        let mut rgb = [0; 3];
        rgb.copy_from_slice(colour);
        CellRaster::Mean(rgb, coverage)
    }

    fn as_cell(raster: CellRaster) -> Option<(Rc<Bitmap<Self>>, InkChannels)> {
        match raster {
            CellRaster::Cell(bitmap) => Some((bitmap, InkChannels::ALL)),
            _ => None,
        }
    }

    fn as_mean(raster: &CellRaster) -> Option<(&[u8], u8, InkChannels)> {
        match raster {
            CellRaster::Mean(colour, coverage) => Some((colour, *coverage, InkChannels::ALL)),
            _ => None,
        }
    }
}

impl CellPixel for Rgb16 {
    const KIND: CanvasKind = CanvasKind::Rgb16;

    fn bitmap(canvas: PageCanvas) -> Option<Bitmap<Self>> {
        match canvas {
            PageCanvas::Rgb16(b) => Some(b),
            PageCanvas::Rgb(_) | PageCanvas::Ink(_) | PageCanvas::Ink16(_) => None,
        }
    }

    fn cell(bitmap: Rc<Bitmap<Self>>, _channels: InkChannels) -> CellRaster {
        CellRaster::Cell16(bitmap)
    }

    fn mean(colour: &[u16], coverage: u16, _channels: InkChannels) -> CellRaster {
        let mut rgb = [0; 3];
        rgb.copy_from_slice(colour);
        CellRaster::Mean16(rgb, coverage)
    }

    fn as_cell(raster: CellRaster) -> Option<(Rc<Bitmap<Self>>, InkChannels)> {
        match raster {
            CellRaster::Cell16(bitmap) => Some((bitmap, InkChannels::ALL)),
            _ => None,
        }
    }

    fn as_mean(raster: &CellRaster) -> Option<(&[u16], u16, InkChannels)> {
        match raster {
            CellRaster::Mean16(colour, coverage) => Some((colour, *coverage, InkChannels::ALL)),
            _ => None,
        }
    }
}

impl CellPixel for DeviceN8 {
    const KIND: CanvasKind = CanvasKind::Ink;

    fn bitmap(canvas: PageCanvas) -> Option<Bitmap<Self>> {
        match canvas {
            PageCanvas::Ink(b) => Some(b),
            PageCanvas::Rgb(_) | PageCanvas::Rgb16(_) | PageCanvas::Ink16(_) => None,
        }
    }

    fn cell(bitmap: Rc<Bitmap<Self>>, channels: InkChannels) -> CellRaster {
        CellRaster::InkCell(bitmap, channels)
    }

    fn mean(colour: &[u8], coverage: u8, channels: InkChannels) -> CellRaster {
        let mut ink = [0; 8];
        ink.copy_from_slice(colour);
        CellRaster::InkMean(ink, coverage, channels)
    }

    fn as_cell(raster: CellRaster) -> Option<(Rc<Bitmap<Self>>, InkChannels)> {
        match raster {
            CellRaster::InkCell(bitmap, channels) => Some((bitmap, channels)),
            _ => None,
        }
    }

    fn as_mean(raster: &CellRaster) -> Option<(&[u8], u8, InkChannels)> {
        match raster {
            CellRaster::InkMean(colour, coverage, channels) => Some((colour, *coverage, *channels)),
            _ => None,
        }
    }
}

impl CellPixel for DeviceN16 {
    const KIND: CanvasKind = CanvasKind::Ink16;

    fn bitmap(canvas: PageCanvas) -> Option<Bitmap<Self>> {
        match canvas {
            PageCanvas::Ink16(b) => Some(b),
            PageCanvas::Rgb(_) | PageCanvas::Rgb16(_) | PageCanvas::Ink(_) => None,
        }
    }

    fn cell(bitmap: Rc<Bitmap<Self>>, channels: InkChannels) -> CellRaster {
        CellRaster::InkCell16(bitmap, channels)
    }

    fn mean(colour: &[u16], coverage: u16, channels: InkChannels) -> CellRaster {
        let mut ink = [0; 8];
        ink.copy_from_slice(colour);
        CellRaster::InkMean16(ink, coverage, channels)
    }

    fn as_cell(raster: CellRaster) -> Option<(Rc<Bitmap<Self>>, InkChannels)> {
        match raster {
            CellRaster::InkCell16(bitmap, channels) => Some((bitmap, channels)),
            _ => None,
        }
    }

    fn as_mean(raster: &CellRaster) -> Option<(&[u16], u16, InkChannels)> {
        match raster {
            CellRaster::InkMean16(colour, coverage, channels) => {
                Some((colour, *coverage, *channels))
            }
            _ => None,
        }
    }
}

/// The content of a tiling pattern's cell, ready to render.
struct CellContent<'a> {
    desc: &'a crate::resources::tiling::TilingDescriptor,
    ops: &'a [crate::content::Operator<'static>],
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
    /// painting operator can touch; `extent` is the pixel area of the
    /// operator's object on the whole page (see [`Self::page_extent`]).
    pub(super) fn resolve_pattern(
        &mut self,
        name: &[u8],
        region: PixelRect,
        extent: u64,
    ) -> Option<PatternPaint> {
        if let Some(shading) = self.memo_shading_pattern(name) {
            return Some(PatternPaint::Shading(shading));
        }
        self.resolve_fill_pattern(name, region, extent)
            .map(PatternPaint::Tiled)
    }

    /// Resolve a tiling pattern over `region`, as a tile of the canvas's
    /// pixel type.
    ///
    /// The cell is rasterised once per quarter-pixel phase and composited
    /// at every lattice point whose cell meets the region when the cell is
    /// no larger than `extent`, the object's area on the whole page, and
    /// its raster within [`MAX_COMPOSED_CELL_BYTES`]; otherwise each such
    /// cell's content is rendered directly onto a region-sized bitmap at
    /// its exact position.  A lattice finer than a pixel is reduced to its
    /// mean.
    pub(super) fn resolve_fill_pattern(
        &mut self,
        name: &[u8],
        region: PixelRect,
        extent: u64,
    ) -> Option<Tile> {
        // Each cell renders in a child renderer whose `pattern_depth` is one
        // above its parent's (see `render_cells`), so the cap bounds the
        // recursion of a pattern whose content paints with a pattern,
        // itself included.
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
        // A `PaintType` 2 (uncoloured) pattern's content paints shapes only,
        // in the tint the `scn` operands give (`fill_pattern_components`);
        // a `PaintType` 1 (coloured) pattern's content sets its own colours.
        let tint = if desc.paint_type == crate::resources::tiling::PaintType::Uncoloured
            && !gs.fill_pattern_components.is_empty()
        {
            Some(resolve_pattern_tint(
                self.resources.doc(),
                gs.colour_intent,
                &gs.fill_color_space,
                &gs.fill_pattern_components,
                self.canvas.kind().depth(),
            ))
        } else {
            None
        };
        // PDF §8.7.3.1: the pattern matrix maps pattern space to the default
        // coordinate space of the parent content stream, not to the CTM in
        // effect at the fill.  `device` carries on to the whole page's
        // device pixels: the lattice is placed there, and moved onto the
        // window by whole pixels last, so a window stamps the cells where
        // the page does.
        let pat_ctm = ctm_multiply(&desc.matrix, &self.pattern_ctm);
        let device = flip_y(&pat_ctm, f64::from(self.page_height));
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
        let kind = self.canvas.kind();
        let key = RasterKey::new(
            desc.stream_id,
            &device,
            tint.as_ref(),
            (!desc.has_own_resources).then(|| self.resources.identity()),
            kind,
        );
        match kind {
            CanvasKind::Rgb => self
                .tile::<Rgb8>(name, &cell, &geometry, (region, extent), key)
                .map(|(pattern, _)| Tile::Rgb(pattern)),
            CanvasKind::Rgb16 => self
                .tile::<Rgb16>(name, &cell, &geometry, (region, extent), key)
                .map(|(pattern, _)| Tile::Rgb16(pattern)),
            CanvasKind::Ink => self
                .tile::<DeviceN8>(name, &cell, &geometry, (region, extent), key)
                .map(|(pattern, channels)| Tile::Ink { pattern, channels }),
            CanvasKind::Ink16 => self
                .tile::<DeviceN16>(name, &cell, &geometry, (region, extent), key)
                .map(|(pattern, channels)| Tile::Ink16 { pattern, channels }),
        }
    }

    /// The tile of `cell` over `region` in pixels of type `P`, with the
    /// channels its content painted (see [`Self::resolve_fill_pattern`]),
    /// for an object of `extent` pixels on the whole page.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the extent is page-bounded, far below 2^52"
    )]
    fn tile<P: CellPixel>(
        &mut self,
        name: &[u8],
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        (region, extent): (PixelRect, u64),
        key: RasterKey,
    ) -> Option<(TiledPattern<P>, InkChannels)> {
        // A lattice cell below one pixel in area repeats more than once per
        // pixel; box-filtered over a pixel, the pattern is its mean.
        if geometry.area < 1.0 {
            return self.mean_pattern::<P>(cell, geometry, key);
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
        // A cell larger than the object, or whose raster is larger than
        // MAX_COMPOSED_CELL_BYTES, is rendered in place.  Both are measured
        // on the whole page, so every window of it chooses as the page does.
        if cell_w * cell_h <= extent as f64
            && cell_raster_bytes::<P>(cell_w, cell_h) <= MAX_COMPOSED_CELL_BYTES
        {
            self.composed_pattern::<P>(cell, geometry, region, &key)
        } else {
            self.direct_pattern::<P>(name, cell, geometry, region)
        }
    }

    /// The mean colour and coverage of a cell over one lattice period,
    /// sampled at about `MEAN_SAMPLE_PX` across.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the sample sides are whole pixels from pixel_ceil, at most MAX_TILE_PX"
    )]
    fn mean_pattern<P: CellPixel>(
        &mut self,
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        key: RasterKey,
    ) -> Option<(TiledPattern<P>, InkChannels)> {
        let raster = self.memo_raster(key, |r| {
            let (w, h) = (geometry.bounds.width, geometry.bounds.height);
            let k = (MEAN_SAMPLE_PX / w.min(h)).min(MAX_TILE_PX / w.max(h));
            let size = (pixel_ceil(w * k) as u32, pixel_ceil(h * k) as u32);
            let (sample, channels) =
                r.render_cells::<P>(cell, size, &[geometry.placement(k, 0.0, 0.0)])?;
            let (colour, coverage) = mean_over_period(&sample, geometry.area * k * k);
            Some(P::mean(&colour, coverage, channels))
        })?;
        let (colour, coverage, channels) =
            P::as_mean(&raster).expect("a mean of the canvas's pixel type is kept under its key");
        Some((TiledPattern::uniform(colour, coverage)?, channels))
    }

    /// The cell rasterised once per quarter-pixel phase and composited at
    /// every lattice point whose cell meets `region`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "the cell sides are whole pixels bounded by the region or the memo budget; region coordinates are page-bounded"
    )]
    fn composed_pattern<P: CellPixel>(
        &mut self,
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        region: PixelRect,
        key: &RasterKey,
    ) -> Option<(TiledPattern<P>, InkChannels)> {
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
                let (bitmap, channels) =
                    r.render_cells::<P>(cell, size, &[geometry.placement(1.0, x, y)])?;
                Some(P::cell(Rc::new(bitmap), channels))
            })?;
            Some(
                P::as_cell(raster)
                    .expect("a cell of the canvas's pixel type is kept under its key"),
            )
        };
        #[cfg(test)]
        if self.refuses_test_canvas(region.width(), region.height()) {
            return None;
        }
        let mut channels = InkChannels::NONE;
        let (ox, oy) = self.page_origin();
        TiledPattern::compose(
            |phase| {
                raster_at(self, phase).map(|(bitmap, painted)| {
                    channels = channels.union(painted);
                    bitmap
                })
            },
            size,
            &geometry.lattice,
            region.x0 as i32 + ox,
            region.y0 as i32 + oy,
            region.width(),
            region.height(),
        )
        .unwrap_or_else(|e| {
            self.abort_allocation(e);
            None
        })
        .map(|pattern| (pattern.moved(-ox, -oy), channels))
    }

    /// The cells meeting `region` rendered in place on a region-sized
    /// bitmap, for a cell larger than the region.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "the cell sides are whole pixels from pixel_ceil; region coordinates are page-bounded"
    )]
    fn direct_pattern<P: CellPixel>(
        &mut self,
        name: &[u8],
        cell: &CellContent<'_>,
        geometry: &CellGeometry,
        region: PixelRect,
    ) -> Option<(TiledPattern<P>, InkChannels)> {
        let (cell_w, cell_h) = geometry.cell_size();
        let (ox, oy) = self.page_origin();
        let (rx0, ry0) = (region.x0 as i32 + ox, region.y0 as i32 + oy);
        let rect = (rx0, ry0, region.x1 as i32 + ox, region.y1 as i32 + oy);
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
        let (screen, channels) = self.render_cells::<P>(cell, size, &placements)?;
        Some((TiledPattern::screen(screen, rx0 - ox, ry0 - oy), channels))
    }

    /// Render the pattern cell once per matrix of `placements` onto one
    /// `width × height` bitmap, initially transparent, whose alpha plane
    /// records the coverage of the content.  Each placement maps pattern
    /// space to the bitmap's pixels (y down); the content is clipped to the
    /// cell's `BBox` (PDF §8.7.3.1).
    ///
    /// A child `PageRenderer` scoped to the pattern's resource context runs
    /// the content, on a canvas of pixels of type `P` sharing the page's
    /// spot slots.  An uncoloured pattern's tint is set as the fill and
    /// stroke colour before the content runs.  The bitmap comes with the
    /// channels the content painted.  `None`, with the render stopped, when
    /// the allocator refuses the bitmap.
    fn render_cells<P: CellPixel>(
        &mut self,
        cell: &CellContent<'_>,
        (width, height): (u32, u32),
        placements: &[[f64; 6]],
    ) -> Option<(Bitmap<P>, InkChannels)> {
        let desc = cell.desc;
        // The cell's operators count against the page's operation budget
        // and deadline, which the parent cannot check while the cell runs.
        let mut child = self.try_child(P::KIND, width, height, desc.stream_id)?;
        // One level deeper, bounded by MAX_PATTERN_DEPTH.
        child.pattern_depth = self.pattern_depth.saturating_add(1);

        // A pattern without resources of its own inherits the resources in
        // effect, whichever dictionary they came from.
        if !desc.has_own_resources {
            child.resources = self.resources.inherited();
        }

        for placement in placements {
            let ctm = flip_y(placement, f64::from(height));
            child.pattern_ctm = ctm;
            child.gstate.save();
            let gs = child.gstate.current_mut();
            gs.ctm = ctm;
            if let Some(c) = cell.tint {
                gs.fill_color = c.clone();
                gs.stroke_color = c.clone();
            }
            // Cells abut: an aliased edge leaves no pixel on a shared edge
            // partly covered by each of two cells.
            child.clip_to_user_rect(&desc.bbox, false);
            child.execute_content_stream(cell.ops);
            child.gstate.restore();
            if child.budget_exceeded.is_some() {
                break;
            }
        }
        // The cells' operations and any budget breach pass to the parent,
        // whose execution stops on a breach.
        child.fold_child_into(self);
        let channels = child.painted_channels();
        P::bitmap(child.finish_canvas()).map(|bitmap| (bitmap, channels))
    }

    /// Intersect the clip with `bbox`, a rectangle `[x0, y0, x1, y1]` in
    /// user space, under the current CTM.  A non-finite device image of the
    /// rectangle leaves the clip unchanged.  A rectangle not aligned with
    /// the device axes clips with antialiased edges when `antialias` and the
    /// clip is antialiased, otherwise with aliased ones.
    #[expect(
        clippy::float_cmp,
        reason = "only a quad whose corners align exactly takes the rectangle clip"
    )]
    pub(super) fn clip_to_user_rect(&mut self, bbox: &[f64; 4], antialias: bool) {
        let [x0, y0, x1, y1] = *bbox;
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|(x, y)| self.to_device(x, y));
        self.trace_clip_rect(&corners);
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
        if antialias {
            self.intersect_clip_path(&pb.build(), false);
        } else {
            let flatness = self.gstate.current().flatness.max(0.1);
            let xpath = XPath::new(&pb.build(), &DEVICE_MATRIX, flatness, true);
            self.gstate
                .current_mut()
                .clip
                .clip_to_path_aliased(&xpath, false);
        }
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
        assert_eq!(mean_over_period(&empty, 1.0), (vec![0; 3], 0));

        // Overlapping content past the period saturates the coverage.
        let (_, full) = mean_over_period(&bmp, 1.0);
        assert_eq!(full, 255);
    }

    /// A pattern without resources of its own resolves names in the
    /// resources in effect, even when those are a dictionary given directly
    /// (an interactive form's `/DR`) rather than a page's `/Resources`.
    #[test]
    fn a_pattern_without_resources_inherits_a_direct_resource_dictionary() {
        let cell = b"q 8 0 0 8 0 0 cm /Im Do Q";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, b"<< /Type /Page >>".to_vec()),
            (
                5,
                format!(
                    "<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 8 8] \
                     /XStep 8 /YStep 8 /Length {} >>\nstream\n{}\nendstream",
                    cell.len(),
                    String::from_utf8_lossy(cell)
                )
                .into_bytes(),
            ),
            (
                6,
                b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray \
                  /BitsPerComponent 8 /Length 1 >>\nstream\n\x00\nendstream"
                    .to_vec(),
            ),
        ]);
        let resources = pdf::parse_object(
            b"<< /Pattern << /P 5 0 R >> /XObject << /Im 6 0 R >> >>",
            &mut 0,
        )
        .unwrap();
        let mut renderer = PageRenderer::new(8, 8, &doc, (4, 0));
        renderer.resources = renderer
            .resources
            .with_resources((0, 0), resources.as_dict().unwrap());
        renderer.execute(&crate::content::parse(b"/Pattern cs /P scn 0 0 8 8 re f"));
        let px: Vec<u8> = (0..8)
            .map(|y| renderer.rgb_canvas().row_bytes(y)[12])
            .collect();
        assert!(px.iter().all(|&v| v == 0), "cell image painted: {px:?}");
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
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[1.0, 0.0, 0.0],
            SampleDepth::Eight,
        );
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
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[50.0, 0.0, 0.0],
            SampleDepth::Eight,
        );
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
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[0.5, 0.5, 0.5],
            SampleDepth::Eight,
        );
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
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[0.5],
            SampleDepth::Eight,
        );
        // Heuristic: 1 component → Gray(0.5) = mid-grey.
        assert_eq!(tint.as_slice(), &[128, 128, 128]);
    }

    #[test]
    fn pattern_tint_devicegray_base_one_component() {
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceGray)),
        };
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[1.0],
            SampleDepth::Eight,
        );
        assert_eq!(tint.as_slice(), &[255, 255, 255]);
    }

    #[test]
    fn pattern_tint_devicecmyk_base_four_components() {
        let doc = empty_doc();
        let fill_cs = ColorSpace::Pattern {
            base: Some(Box::new(ColorSpace::DeviceCmyk)),
        };
        // (0,0,0,1) = solid K: all four components reach the CMYK conversion.
        let tint = resolve_pattern_tint(
            &doc,
            ColourIntent::INITIAL,
            &fill_cs,
            &[0.0, 0.0, 0.0, 1.0],
            SampleDepth::Eight,
        );
        assert_eq!(tint.as_slice(), &color::cmyk::cmyk_to_rgb(0, 0, 0, 255));
    }
}
