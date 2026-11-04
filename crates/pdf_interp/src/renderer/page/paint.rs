//! Path painting: colour selection, clipping and the fill and stroke of a
//! path through the raster pipe.

use std::sync::Arc;

use color::TransferLut;
use raster::{
    TiledPattern, eo_fill, fill,
    path::PathBuilder,
    pipe::{ChannelMask, Pattern, PipeSrc, PipeState, TransferSet},
    stroke::{StrokeParams, stroke},
    transparency::Backdrop,
    types::{BlendMode, LineCap, LineJoin},
    xpath::XPath,
};

use super::super::color::RasterColor;
use super::super::gstate::{Ctm, ctm_transform, invert_affine};
use super::canvas::{CanvasKind, CanvasPattern, with_canvas};
use super::patterns::PatternPaint;
use super::pixel_rect::PixelRect;
use super::{DEVICE_MATRIX, PageRenderer};
use crate::resources::ColorSpace;
use crate::resources::transfer::Transfer;

/// `path` with every point mapped through `m`.  An affine map carries Bézier
/// control points to the control points of the mapped curve, so flags and
/// stroke-adjust hints carry over unchanged.
fn transform_path(path: &raster::path::Path, m: &[f64; 6]) -> raster::path::Path {
    raster::path::Path {
        pts: path
            .pts
            .iter()
            .map(|p| raster::xpath::transform(m, p.x, p.y))
            .collect(),
        ..path.clone()
    }
}

/// The compositing parameters of one paint: its constant alpha and blend
/// mode, the channels it writes, and the backdrop of the enclosing
/// non-isolated group that a paint consults (see
/// [`PageRenderer::paint_backdrop`]).  The pipe borrows them.
pub(super) struct PaintParams {
    /// The constant alpha of the painting operation.
    pub(super) alpha: u8,
    /// The blend mode in effect.
    pub(super) blend_mode: BlendMode,
    /// The channels the paint writes; `None` for all.
    pub(super) overprint: Option<ChannelMask>,
    /// The group backdrop the paint consults, if any.
    pub(super) backdrop: Option<Arc<Backdrop>>,
}

impl PaintParams {
    /// The pipe of the paint, with the identity transfer.
    pub(super) fn pipe(&self) -> PipeState<'_> {
        PipeState {
            blend_mode: self.blend_mode,
            a_input: self.alpha,
            transfer: TransferSet::identity_rgb(),
            backdrop: self.backdrop.as_deref().map(Backdrop::as_pipe_backdrop),
            overprint: self.overprint,
        }
    }
}

/// The area scale `√|det|` of the linear part of `m`: the factor by which
/// it scales lengths on average.  `None` for a singular or non-finite `m`,
/// whose inverse [`invert_affine`] does not give.
pub(super) fn area_scale(m: &[f64; 6]) -> Option<f64> {
    invert_affine(m).map(|_| m[0].mul_add(m[3], -(m[1] * m[2])).abs().sqrt())
}

/// The farthest a stroke `line_width` wide reaches past its path in device
/// pixels under `device`: half the width, stretched at most by the matrix's
/// largest singular value (bounded above by the Frobenius norm of its 2×2
/// part), times the miter limit, or √2 for a square cap.  Under a singular
/// matrix the width is taken in device pixels.
pub(super) fn stroke_reach(device: &[f64; 6], line_width: f64, miter_limit: f64) -> f64 {
    let [a, b, c, d, _, _] = *device;
    let stretch = invert_affine(device).map_or(1.0, |_| {
        a.mul_add(a, b.mul_add(b, c.mul_add(c, d * d))).sqrt()
    });
    line_width / 2.0 * stretch * miter_limit.max(std::f64::consts::SQRT_2)
}

impl PageRenderer<'_> {
    // ── Internal helpers ──────────────────────────────────────────────────────

    pub(super) fn set_fill(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.fill_color = c;
        gs.fill_pattern = None;
    }

    pub(super) fn set_stroke(&mut self, c: RasterColor) {
        let gs = self.gstate.current_mut();
        gs.stroke_color = c;
        gs.stroke_pattern = None;
    }

    /// `g`, `rg`, `k`: select the device space `device`, or the default
    /// space that replaces it (PDF §8.6.5.6), for fill operations and set
    /// the fill colour to `comps` in that space (PDF §8.6.8).
    pub(super) fn set_fill_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        let c = self.color_for_components(&space, comps);
        self.set_fill(c);
        self.gstate.current_mut().fill_color_space = space;
    }

    /// `G`, `RG`, `K`: the stroking counterpart of [`Self::set_fill_device`].
    pub(super) fn set_stroke_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        let c = self.color_for_components(&space, comps);
        self.set_stroke(c);
        self.gstate.current_mut().stroke_color_space = space;
    }

    /// Resolve `sc`/`scn` (or `SC`/`SCN`) component operands against an
    /// already-selected colour space.
    ///
    /// For plain device spaces the raw components ARE the colour, so the
    /// fast [`components_to_color`] path is kept (and stays bit-identical
    /// for RGB/CMYK/Gray).  Parameterised spaces — `Separation`, `DeviceN`,
    /// `ICCBased`, `Indexed`, `Lab` — must run the components through the
    /// space's tint/profile transform (PDF §8.6.6): treating a Separation
    /// tint of `1.0` as a grey of `1.0` paints white and silently drops all
    /// content on spot-colour pages.
    ///
    /// On an ink canvas a `Separation` or `DeviceN` colour is given as its
    /// colorants' tints, each in its own channel (see `InkMap`); on an RGB
    /// canvas it is its sRGB conversion, like every other colour.  An
    /// `Indexed` colour resolves as its palette entry in the base space.
    pub(super) fn color_for_components(&mut self, cs: &ColorSpace, comps: &[f64]) -> RasterColor {
        let doc = self.resources.doc();
        match cs {
            ColorSpace::DeviceGray | ColorSpace::DeviceRgb | ColorSpace::DeviceCmyk => {
                components_to_color(comps)
            }
            // Pattern operands are handled by the pattern dispatch, not here;
            // fall back to the legacy heuristic for any stray component.
            ColorSpace::Pattern { .. } => components_to_color(comps),
            // A four-component ICC space is a process CMYK space: its
            // components are the ink.
            ColorSpace::IccBased { n: 4, .. } => match comps {
                &[c, m, y, k] => {
                    RasterColor::with_process_inks(cs.convert_to_rgb(doc, comps), [c, m, y, k])
                }
                _ => RasterColor::from_bytes(cs.convert_to_rgb(doc, comps)),
            },
            ColorSpace::Separation { .. } | ColorSpace::DeviceN { .. }
                if self.canvas.kind() == CanvasKind::Ink =>
            {
                let (ink, paints) = self.ink_map.colorant_ink(doc, cs, comps);
                RasterColor::with_ink(cs.convert_to_rgb(doc, comps), ink, paints)
            }
            // An indexed colour is its palette entry's colour in the base
            // space.
            ColorSpace::Indexed { .. } => match cs.indexed_base_components(comps) {
                Some((base, entry)) => self.color_for_components(base, &entry),
                None => RasterColor::from_bytes(cs.convert_to_rgb(doc, comps)),
            },
            _ => RasterColor::from_bytes(cs.convert_to_rgb(doc, comps)),
        }
    }

    pub(super) fn path_builder(&mut self) -> &mut PathBuilder {
        self.path.get_or_insert_with(PathBuilder::new)
    }

    /// Transform a user-space point to device space using the current CTM.
    /// Applies a y-flip because PDF uses bottom-left origin but bitmaps use top-left.
    pub(super) fn to_device(&self, x: f64, y: f64) -> (f64, f64) {
        let ctm = &self.gstate.current().ctm;
        let (dx, dy) = ctm_transform(ctm, x, y);
        self.page_to_window((dx, f64::from(self.page_height) - dy))
    }

    /// Move a point in the whole page's device space (y down) into the
    /// drawn window (see [`super::view::page_to_window`]).
    pub(super) fn page_to_window(&self, point: (f64, f64)) -> (f64, f64) {
        super::view::page_to_window(self.window_origin, point)
    }

    /// `ctm`, a matrix into the whole page's space, re-based to map into
    /// the drawn window (see [`super::view::window_ctm`]); unchanged when the
    /// renderer draws the whole page.
    pub(super) fn window_ctm(&self, ctm: &Ctm) -> Ctm {
        super::view::window_ctm(ctm, self.window_origin, self.height, self.page_height)
    }

    /// The user-space → device-space matrix: the CTM, re-based to the
    /// window, followed by the y-flip.
    pub(super) fn device_matrix(&self) -> [f64; 6] {
        let m = self.window_ctm(&self.gstate.current().ctm);
        [
            m[0],
            -m[1],
            m[2],
            -m[3],
            m[4],
            f64::from(self.height) - m[5],
        ]
    }

    /// Apply a pending `W`/`W*` clip using the current path, then discard the path.
    ///
    /// Called by `EndPath` (`n`) when no painting follows.
    pub(super) fn apply_pending_clip(&mut self) {
        if let Some(even_odd) = self.pending_clip.take() {
            if let Some(builder) = self.path.take() {
                self.clip_path_into_gstate(&builder.build(), even_odd);
            } else {
                log::debug!("rasterrocket-interp: W/W* with no current path — clip unchanged");
            }
        }
    }

    /// Apply a pending `W`/`W*` clip using an already-built path.
    ///
    /// Called by paint helpers (`do_fill`, `do_stroke`, `do_fill_then_stroke`)
    /// when a `W`/`W*` preceded the painting operator.  The same path is used
    /// for both clipping and painting (PDF §8.5.4).
    pub(super) fn apply_pending_clip_with(&mut self, path: &raster::path::Path) {
        if let Some(even_odd) = self.pending_clip.take() {
            self.clip_path_into_gstate(path, even_odd);
        }
    }

    /// Intersect `path` into the current graphics-state clip region.
    pub(super) fn clip_path_into_gstate(&mut self, path: &raster::path::Path, even_odd: bool) {
        let flatness = self.gstate.current().flatness.max(0.1);
        let xpath = XPath::new(path, &DEVICE_MATRIX, flatness, true);
        self.gstate
            .current_mut()
            .clip
            .clip_to_path(&xpath, even_odd);
    }

    /// The tables of `transfer` as the pipe applies them to a source that
    /// is not mapped beforehand: a pattern's colours, a shading and its
    /// background.  A solid fill or stroke colour is mapped before painting
    /// (see `InterpGState::painted_fill_color`) and paints through the
    /// identity.
    pub(super) fn transfer_set(transfer: &Transfer) -> TransferSet<'_> {
        static DN: [[u8; 256]; 8] = [TransferLut::IDENTITY.0; 8];
        let id = TransferLut::IDENTITY.as_array();
        TransferSet::new(transfer.tables(), id, [id; 4], &DN)
    }

    /// The backdrop a paint under `blend_mode` writing the channels
    /// `overprint` consults: the enclosing non-isolated group's, unless the
    /// mode is `Normal` and the paint writes every channel.  A `Normal`
    /// paint of every channel does not depend on the backdrop; a masked
    /// paint keeps its unpainted channels relative to it whatever the mode.
    pub(super) fn paint_backdrop(
        &self,
        blend_mode: BlendMode,
        overprint: Option<ChannelMask>,
    ) -> Option<Arc<Backdrop>> {
        (blend_mode != BlendMode::Normal || overprint.is_some())
            .then(|| self.group.backdrop.clone())
            .flatten()
    }

    /// The parameters of a paint at constant alpha `alpha` under
    /// `blend_mode` that writes the channels `overprint` (`None` for
    /// all), with the backdrop it consults.
    pub(super) fn paint_params(
        &self,
        alpha: u8,
        blend_mode: BlendMode,
        overprint: Option<ChannelMask>,
    ) -> PaintParams {
        PaintParams {
            alpha,
            blend_mode,
            overprint,
            backdrop: self.paint_backdrop(blend_mode, overprint),
        }
    }

    /// The parameters of a paint of the solid colour `colour` by the
    /// stroking (`stroke`) or non-stroking operation at constant alpha
    /// `alpha` under `blend_mode`: it writes the channels of
    /// [`Self::overprint_mask`].
    pub(super) fn solid_paint(
        &self,
        stroke: bool,
        colour: &RasterColor,
        alpha: u8,
        blend_mode: BlendMode,
    ) -> PaintParams {
        self.paint_params(alpha, blend_mode, self.overprint_mask(stroke, colour))
    }

    /// The channels a paint of the stroking (`stroke`) or non-stroking
    /// operation writes when its source paints the channels `paints`
    /// (ISO 32000-1 §8.6.7): `paints` with overprint set for the operation on an
    /// ink canvas; otherwise `None`, every channel — on an RGB canvas,
    /// with overprint off, and when `paints` covers every channel.
    pub(super) fn overprint_channels(
        &self,
        stroke: bool,
        paints: ChannelMask,
    ) -> Option<ChannelMask> {
        if self.canvas.kind() != CanvasKind::Ink {
            return None;
        }
        let gs = self.gstate.current();
        let overprint = if stroke {
            gs.overprint_stroke
        } else {
            gs.overprint_fill
        };
        (overprint && paints != ChannelMask::ALL).then_some(paints)
    }

    /// The channels a fill or stroke of the solid colour `colour` writes
    /// under the overprint state and mode in effect (see
    /// [`Self::overprint_channels`] and [`RasterColor::paints_with_opm`]).
    pub(super) fn overprint_mask(&self, stroke: bool, colour: &RasterColor) -> Option<ChannelMask> {
        let mode = self.gstate.current().overprint_mode;
        self.overprint_channels(stroke, colour.paints_with_opm(mode))
    }

    pub(super) fn do_fill(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_path(&path, even_odd);
    }

    pub(super) fn do_stroke(&mut self) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.stroke_path(&path);
    }

    pub(super) fn do_fill_then_stroke(&mut self, even_odd: bool) {
        let Some(builder) = self.path.take() else {
            self.pending_clip = None;
            return;
        };
        let path = builder.build();
        self.apply_pending_clip_with(&path);
        self.fill_and_stroke(&path, even_odd);
    }

    /// Fill `path` using the current fill colour or pattern.  Shared by `do_fill`
    /// and `do_fill_then_stroke`.
    pub(super) fn fill_path(&mut self, path: &raster::path::Path, even_odd: bool) {
        // Stroke adjustment is always on: a lone axis-aligned quadrilateral
        // fills whole device pixels.
        if self.gstate.current().fill_paints_nothing() {
            return;
        }
        let hinted = path.with_fill_adjust_hints();
        let path = hinted.as_ref().unwrap_or(path);
        let region = self.paint_region(path.pts.iter().map(|p| (p.x, p.y)), 1.0);
        let alpha = self.gstate.current().fill_alpha;
        self.paint_object(region, alpha, |r| r.fill_within(path, even_odd, region));
    }

    /// Fill `path`, whose pixels lie within `region`.
    pub(super) fn fill_within(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        region: Option<PixelRect>,
    ) {
        let gs = self.gstate.current();
        let clip = gs.clip.clone_shared();
        let flatness = gs.flatness.max(0.1);
        let (fill_alpha, blend_mode) = (gs.fill_alpha, gs.blend_mode);
        let transfer = gs.transfer.clone();

        // Resolve the fill source — pattern or solid colour.  The solid
        // colour is already mapped through the transfer function; a
        // pattern's colours map in the pipe.
        let pat_name = gs.fill_pattern.clone();
        let solid_color = gs.painted_fill_color();

        // A pattern colour that does not resolve, or whose object lies
        // outside the clip, paints nothing: the fill colour is not a
        // fallback for it.
        let tiled = match pat_name
            .as_deref()
            .map(|name| region.and_then(|region| self.resolve_pattern(name, region)))
        {
            Some(Some(PatternPaint::Shading(pattern))) => {
                self.paint_shading_pattern(&pattern, false, region, |r, pipe, shading| {
                    r.fill_with_shading(path, even_odd, clip, pipe, flatness, shading);
                });
                return;
            }
            Some(Some(PatternPaint::Tiled(tiled))) => Some(tiled),
            Some(None) => return,
            None => None,
        };
        // A tiling pattern writes every channel; a solid colour the
        // channels of its space under the overprint state in effect.
        let paint = if tiled.is_some() {
            self.paint_params(fill_alpha, blend_mode, None)
        } else {
            self.solid_paint(false, &solid_color, fill_alpha, blend_mode)
        };
        let mut pipe = paint.pipe();
        let source = PaintSource::new(
            self.canvas.kind(),
            tiled.as_ref(),
            &solid_color,
            transfer.as_deref(),
            &mut pipe,
        );
        let src = source.src();

        // Solid-colour fills only. The device evaluates the CPU scanner's
        // intersection table and the counts are composited through the CPU
        // pipeline, so either backend yields the bytes the CPU fill below
        // would. A session attaches at most one of the two backends.
        #[cfg(feature = "gpu-aa")]
        if source.pattern.is_none()
            && self.vector_antialias
            && self.try_device_aa_fill(path, even_odd, flatness, &pipe, &src)
        {
            return;
        }

        let antialias = self.vector_antialias;
        with_canvas!(&mut self.canvas, |b| {
            let fill_fn = if even_odd { eo_fill } else { fill };
            fill_fn(
                b,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                antialias,
            );
        });
    }

    /// Fill `path` on the attached GPU backend, if any; `true` when the
    /// device painted it. See `gpu_ops::try_backend_aa_fill`.
    #[cfg(feature = "gpu-aa")]
    fn try_device_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        flatness: f64,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
    ) -> bool {
        use super::gpu_ops::try_backend_aa_fill;
        #[cfg(feature = "vulkan")]
        if let Some(vk) = self.vk_backend.clone() {
            return try_backend_aa_fill(self, &*vk, path, even_odd, flatness, pipe, src);
        }
        let Some(ctx) = self.gpu_ctx.clone() else {
            return false;
        };
        let cuda = gpu::backend::cuda::CudaBackend::from_ctx(ctx);
        try_backend_aa_fill(self, &cuda, path, even_odd, flatness, pipe, src)
    }

    /// Stroke `path` with the current line width, widened to at least one
    /// device pixel.
    pub(super) fn stroke_path(&mut self, path: &raster::path::Path) {
        let line_width = self.min_stroke_width();
        self.stroke_path_at(path, line_width);
    }

    /// Stroke a text outline (render modes 1, 2, 5, 6) with the current line
    /// width as given: the outline's own fill carries the glyph shape, and a
    /// widened outline stroke thickens every stem.
    pub(super) fn stroke_text_outline(&mut self, path: &raster::path::Path) {
        let line_width = self.gstate.current().line_width;
        self.stroke_path_at(path, line_width);
    }

    pub(super) fn stroke_path_at(&mut self, path: &raster::path::Path, line_width: f64) {
        let region = self.stroke_bounds(path, line_width);
        let alpha = self.gstate.current().stroke_alpha;
        self.paint_object(region, alpha, |r| r.stroke_within(path, line_width, region));
    }

    /// The current line width, raised so the stroke is at least one device
    /// pixel thick, the device thickness being the user width times the
    /// CTM's area scale `√|det|`.  Width 0 keeps its one-pixel hairline;
    /// under a singular CTM the width is already in device pixels.
    pub(super) fn min_stroke_width(&self) -> f64 {
        let line_width = self.gstate.current().line_width;
        if line_width <= 0.0 {
            return line_width;
        }
        let scale = area_scale(&self.device_matrix()).unwrap_or(1.0);
        line_width.max(1.0 / scale)
    }

    /// The pixels a stroke of `path` can touch: the path's bounds grown by
    /// the stroke's reach — half the line width times the miter limit (or
    /// √2, for a square cap), in device pixels — within the clip.  Under a
    /// singular CTM the stroker works in device space with the line width
    /// unscaled.
    pub(super) fn stroke_bounds(
        &self,
        path: &raster::path::Path,
        line_width: f64,
    ) -> Option<PixelRect> {
        let gs = self.gstate.current();
        let reach = stroke_reach(&self.device_matrix(), line_width, gs.miter_limit);
        self.paint_region(path.pts.iter().map(|p| (p.x, p.y)), reach + 1.0)
    }

    /// The geometry of a stroke of `path`, in device space, `line_width`
    /// wide under the current graphics state.
    ///
    /// PDF §8.4.3.2 / §8.4.3.6: the line width and dash lengths are
    /// user-space distances under the CTM at stroke time.  The path is held
    /// in device space, so it is mapped back through the inverse device
    /// matrix and the stroker transforms the outline it builds; flatness
    /// stays a device-space tolerance.  A singular CTM collapses the path
    /// to a line or point, so it is stroked in device space unchanged.
    pub(super) fn stroke_shape(&self, path: &raster::path::Path, line_width: f64) -> StrokeShape {
        let gs = self.gstate.current();
        let device = self.device_matrix();
        let device_flatness = gs.flatness.max(0.1);
        let (path, matrix, flatness) = match (invert_affine(&device), area_scale(&device)) {
            (Some(inv), Some(scale)) => {
                (transform_path(path, &inv), device, device_flatness / scale)
            }
            _ => (path.clone(), DEVICE_MATRIX, device_flatness),
        };
        StrokeShape {
            path,
            matrix,
            device_flatness,
            line_width,
            line_cap: gs.line_cap,
            line_join: gs.line_join,
            miter_limit: gs.miter_limit,
            flatness,
            line_dash: gs.dash.0.clone(),
            line_dash_phase: gs.dash.1,
            vector_antialias: self.vector_antialias,
        }
    }

    /// Stroke `path` `line_width` wide, whose stroke lies within `region`.
    pub(super) fn stroke_within(
        &mut self,
        path: &raster::path::Path,
        line_width: f64,
        region: Option<PixelRect>,
    ) {
        let gs = self.gstate.current();
        if gs.stroke_paints_nothing() {
            return;
        }
        let clip = gs.clip.clone_shared();
        let (stroke_alpha, blend_mode) = (gs.stroke_alpha, gs.blend_mode);
        let transfer = gs.transfer.clone();
        let pat_name = gs.stroke_pattern.clone();
        let solid_color = gs.painted_stroke_color();
        let shape = self.stroke_shape(path, line_width);
        let (path, matrix, device_flatness) = (&shape.path, &shape.matrix, shape.device_flatness);
        let params = shape.params();

        // Resolve the stroke source — pattern or solid colour.  A pattern
        // colour that does not resolve paints nothing.
        let tiled = match pat_name
            .as_deref()
            .map(|name| region.and_then(|region| self.resolve_pattern(name, region)))
        {
            Some(Some(PatternPaint::Shading(pattern))) => {
                self.paint_shading_pattern(&pattern, true, region, |r, pipe, shading| {
                    r.stroke_with_shading(
                        path,
                        matrix,
                        &params,
                        clip,
                        pipe,
                        device_flatness,
                        shading,
                    );
                });
                return;
            }
            Some(Some(PatternPaint::Tiled(tiled))) => Some(tiled),
            Some(None) => return,
            None => None,
        };
        let paint = if tiled.is_some() {
            self.paint_params(stroke_alpha, blend_mode, None)
        } else {
            self.solid_paint(true, &solid_color, stroke_alpha, blend_mode)
        };
        let mut pipe = paint.pipe();
        let source = PaintSource::new(
            self.canvas.kind(),
            tiled.as_ref(),
            &solid_color,
            transfer.as_deref(),
            &mut pipe,
        );
        let src = source.src();

        with_canvas!(&mut self.canvas, |b| stroke(
            b, &clip, path, &pipe, &src, matrix, &params
        ));
    }
}

/// The geometry of a stroke as the stroker takes it (see
/// [`PageRenderer::stroke_shape`]).
pub(super) struct StrokeShape {
    /// The path, in the space `matrix` maps to device pixels.
    pub(super) path: raster::path::Path,
    /// The map from the path's space to device pixels.
    pub(super) matrix: [f64; 6],
    /// The flatness tolerance in device pixels.
    pub(super) device_flatness: f64,
    line_width: f64,
    line_cap: LineCap,
    line_join: LineJoin,
    miter_limit: f64,
    /// The flatness tolerance in the path's space.
    flatness: f64,
    line_dash: Vec<f64>,
    line_dash_phase: f64,
    vector_antialias: bool,
}

impl StrokeShape {
    /// The stroker's parameters.
    pub(super) fn params(&self) -> StrokeParams<'_> {
        StrokeParams {
            line_width: self.line_width,
            line_cap: self.line_cap,
            line_join: self.line_join,
            miter_limit: self.miter_limit,
            flatness: self.flatness,
            line_dash: &self.line_dash,
            line_dash_phase: self.line_dash_phase,
            stroke_adjust: false,
            vector_antialias: self.vector_antialias,
        }
    }
}

/// What a fill or stroke paints, as the canvas takes it: a tiling pattern
/// or a solid colour.
struct PaintSource<'a> {
    /// The tiling pattern, if the paint has one.
    pattern: Option<CanvasPattern<'a>>,
    /// The solid colour's bytes for the canvas.
    solid: &'a [u8],
}

impl<'a> PaintSource<'a> {
    /// The source of a paint of `tiled`, or else of `solid`, on a canvas of
    /// `kind`.  `solid` is already mapped through the transfer function on
    /// either canvas.  `transfer` applies to the pattern's colours only:
    /// `pipe` takes its tables on an RGB canvas, the pattern applies it
    /// before conversion on an ink canvas.
    fn new(
        kind: CanvasKind,
        tiled: Option<&'a TiledPattern>,
        solid: &'a RasterColor,
        transfer: Option<&'a Transfer>,
        pipe: &mut PipeState<'a>,
    ) -> Self {
        if let Some(set) = tiled.and_then(|_| CanvasPattern::pipe_transfer(kind, transfer)) {
            pipe.transfer = set;
        }
        Self {
            pattern: tiled.map(|p| CanvasPattern::new(kind, p as &dyn Pattern, transfer)),
            solid: solid.bytes_for(kind.mode()),
        }
    }

    /// The source as the pipe takes it.
    fn src(&self) -> PipeSrc<'_> {
        self.pattern.as_ref().map_or_else(
            || PipeSrc::Solid(self.solid),
            |p| PipeSrc::Pattern(p.as_dyn()),
        )
    }
}

// ── Component → colour helpers ────────────────────────────────────────────────

/// Map a raw component slice to a [`RasterColor`] by channel count.
pub(super) fn components_to_color(comps: &[f64]) -> RasterColor {
    match comps {
        [g] => RasterColor::gray(*g),
        [r, g, b] => RasterColor::rgb(*r, *g, *b),
        [c, m, y, k] => RasterColor::cmyk(*c, *m, *y, *k),
        _ => RasterColor::default(),
    }
}

/// Map a PDF line cap integer (0–2) to [`LineCap`]; defaults to `Butt`.
pub(super) const fn int_to_cap(v: i32) -> LineCap {
    match v {
        1 => LineCap::Round,
        2 => LineCap::Projecting,
        _ => LineCap::Butt,
    }
}

/// Map a PDF line join integer (0–2) to [`LineJoin`]; defaults to `Miter`.
pub(super) const fn int_to_join(v: i32) -> LineJoin {
    match v {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        _ => LineJoin::Miter,
    }
}
