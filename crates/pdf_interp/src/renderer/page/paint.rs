//! Path painting: colour selection, clipping and the fill and stroke of a
//! path through the raster pipe.

use std::sync::Arc;

use color::TransferLut;
use raster::{
    eo_fill, fill,
    path::PathBuilder,
    pipe::{Pattern, PipeSrc, PipeState, TransferSet},
    stroke::{StrokeParams, stroke},
    transparency::Backdrop,
    types::{BlendMode, LineCap, LineJoin},
    xpath::XPath,
};

use super::super::color::RasterColor;
use super::super::gstate::{ctm_transform, invert_affine};
use super::patterns::PatternPaint;
use super::pixel_rect::PixelRect;
use super::{DEVICE_MATRIX, PageRenderer};
use crate::resources::ColorSpace;
use crate::resources::transfer::Transfer;
#[cfg(feature = "vulkan")]
use gpu::backend::vulkan::VulkanBackend;

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

/// The area scale `√|det|` of the linear part of `m`: the factor by which
/// it scales lengths on average.  `None` for a singular or non-finite `m`,
/// whose inverse [`invert_affine`] does not give.
fn area_scale(m: &[f64; 6]) -> Option<f64> {
    invert_affine(m).map(|_| m[0].mul_add(m[3], -(m[1] * m[2])).abs().sqrt())
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
    pub(super) fn color_for_components(&self, cs: &ColorSpace, comps: &[f64]) -> RasterColor {
        match cs {
            ColorSpace::DeviceGray | ColorSpace::DeviceRgb | ColorSpace::DeviceCmyk => {
                components_to_color(comps)
            }
            // Pattern operands are handled by the pattern dispatch, not here;
            // fall back to the legacy heuristic for any stray component.
            ColorSpace::Pattern { .. } => components_to_color(comps),
            _ => RasterColor::from_bytes(cs.convert_to_rgb(self.resources.doc(), comps)),
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
        (dx, f64::from(self.height) - dy)
    }

    /// The user-space → device-space matrix: the CTM followed by the y-flip
    /// that [`Self::to_device`] applies.
    pub(super) fn device_matrix(&self) -> [f64; 6] {
        let m = self.gstate.current().ctm;
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

    /// Build a [`PipeState`] with the given opacity and blend mode;
    /// `backdrop` is the enclosing non-isolated group's, for a blend mode
    /// that consults it (see [`Self::blend_backdrop`]).
    pub(super) fn make_pipe(
        a_input: u8,
        blend_mode: BlendMode,
        backdrop: Option<&Backdrop>,
    ) -> PipeState<'_> {
        PipeState {
            blend_mode,
            a_input,
            transfer: TransferSet::identity_rgb(),
            backdrop: backdrop.map(Backdrop::as_pipe_backdrop),
        }
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

    /// The backdrop a paint under `blend_mode` blends with: the enclosing
    /// non-isolated group's, unless the mode is `Normal`, whose result does
    /// not depend on it.
    pub(super) fn blend_backdrop(&self, blend_mode: BlendMode) -> Option<Arc<Backdrop>> {
        (blend_mode != BlendMode::Normal)
            .then(|| self.group.backdrop.clone())
            .flatten()
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
        let backdrop = self.blend_backdrop(gs.blend_mode);
        let transfer = gs.transfer.clone();
        let mut pipe = Self::make_pipe(gs.fill_alpha, gs.blend_mode, backdrop.as_deref());

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
        if let (Some(_), Some(transfer)) = (&tiled, &transfer) {
            pipe.transfer = Self::transfer_set(transfer);
        }
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        // GPU fill dispatch — solid-colour fills only (patterns not yet
        // GPU-accelerated).  Cloning the `Arc` releases the immutable borrow
        // before the mutable `self` borrow inside the helpers.
        //
        // The device evaluates the CPU scanner's intersection table with
        // the CPU's span walk and composites through the CPU pipeline, so
        // either branch yields the bytes the CPU fill below would.  The
        // Vulkan branch fires before the CUDA branch when both are present,
        // but in practice `rasterrocket::render::open_session` wires only
        // one of them based on `BackendPolicy`, so the second branch only
        // runs under `Auto` / `ForceCuda`.
        #[cfg(feature = "vulkan")]
        if tiled.is_none()
            && self.vector_antialias
            && let Some(vk) = self.vk_backend.clone()
            && self.try_vk_aa_fill(path, even_odd, flatness, &pipe, &src, &vk)
        {
            return;
        }
        #[cfg(feature = "gpu-aa")]
        if tiled.is_none()
            && self.vector_antialias
            && let Some(ctx) = self.gpu_ctx.clone()
            && self.try_gpu_aa_fill(path, even_odd, flatness, &pipe, &src, &ctx)
        {
            return;
        }

        if even_odd {
            eo_fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                self.vector_antialias,
            );
        } else {
            fill(
                &mut self.bitmap,
                &clip,
                path,
                &pipe,
                &src,
                &DEVICE_MATRIX,
                flatness,
                self.vector_antialias,
            );
        }
    }

    /// Attempt to rasterise `path` with the GPU scanline-AA kernel.
    ///
    /// Returns `true` if the GPU path was taken (caller skips the CPU fill).
    /// Returns `false` if the fill is not device-eligible (see
    /// `gpu_ops::prepare_aa_fill`) or the GPU call fails (warning logged;
    /// CPU fill used as fallback).
    #[cfg(feature = "gpu-aa")]
    pub(super) fn try_gpu_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        flatness: f64,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        ctx: &gpu::GpuCtx,
    ) -> bool {
        super::gpu_ops::try_gpu_aa_fill(self, path, even_odd, flatness, pipe, src, ctx)
    }

    /// Vulkan twin of [`Self::try_gpu_aa_fill`].  See [`vk_ops::try_vk_aa_fill`].
    #[cfg(feature = "vulkan")]
    pub(super) fn try_vk_aa_fill(
        &mut self,
        path: &raster::path::Path,
        even_odd: bool,
        flatness: f64,
        pipe: &raster::pipe::PipeState<'_>,
        src: &raster::pipe::PipeSrc<'_>,
        backend: &Arc<VulkanBackend>,
    ) -> bool {
        super::vk_ops::try_vk_aa_fill(self, backend, path, even_odd, flatness, pipe, src)
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
        let device = self.device_matrix();
        let scale = invert_affine(&device).map_or(1.0, |_| {
            device[0].hypot(device[1]).max(device[2].hypot(device[3]))
        });
        let reach = line_width / 2.0 * scale * gs.miter_limit.max(std::f64::consts::SQRT_2);
        self.paint_region(path.pts.iter().map(|p| (p.x, p.y)), reach + 1.0)
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
        let backdrop = self.blend_backdrop(gs.blend_mode);
        let transfer = gs.transfer.clone();
        let mut pipe = Self::make_pipe(gs.stroke_alpha, gs.blend_mode, backdrop.as_deref());
        let line_cap = gs.line_cap;
        let line_join = gs.line_join;
        let miter_limit = gs.miter_limit;
        let flatness = gs.flatness.max(0.1);
        let line_dash = gs.dash.0.clone();
        let line_dash_phase = gs.dash.1;
        let pat_name = gs.stroke_pattern.clone();
        let solid_color = gs.painted_stroke_color();

        // PDF §8.4.3.2 / §8.4.3.6: the line width and dash lengths are
        // user-space distances under the CTM at stroke time.  The path is held
        // in device space, so it is mapped back through the inverse device
        // matrix and the stroker transforms the outline it builds; flatness
        // stays a device-space tolerance.  A singular CTM collapses the path
        // to a line or point, so it is stroked in device space unchanged.
        let device = self.device_matrix();
        let device_flatness = flatness;
        let (path, matrix, flatness) = match (invert_affine(&device), area_scale(&device)) {
            (Some(inv), Some(scale)) => (transform_path(path, &inv), device, flatness / scale),
            _ => (path.clone(), DEVICE_MATRIX, flatness),
        };

        let params = StrokeParams {
            line_width,
            line_cap,
            line_join,
            miter_limit,
            flatness,
            line_dash: &line_dash,
            line_dash_phase,
            stroke_adjust: false,
            vector_antialias: self.vector_antialias,
        };

        // Resolve the stroke source — pattern or solid colour.  A pattern
        // colour that does not resolve paints nothing.
        let tiled = match pat_name
            .as_deref()
            .map(|name| region.and_then(|region| self.resolve_pattern(name, region)))
        {
            Some(Some(PatternPaint::Shading(pattern))) => {
                self.paint_shading_pattern(&pattern, true, region, |r, pipe, shading| {
                    r.stroke_with_shading(
                        &path,
                        &matrix,
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
        if let (Some(_), Some(transfer)) = (&tiled, &transfer) {
            pipe.transfer = Self::transfer_set(transfer);
        }
        let src = tiled.as_ref().map_or(PipeSrc::Solid(&solid_color), |p| {
            PipeSrc::Pattern(p as &dyn Pattern)
        });

        stroke(
            &mut self.bitmap,
            &clip,
            &path,
            &pipe,
            &src,
            &matrix,
            &params,
        );
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
