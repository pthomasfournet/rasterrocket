//! Resolved shadings and tiling-pattern cells of one page, so a pattern or
//! shading painted by many operators is decoded and rasterised once.
//!
//! A resolved shading is in device space: it is keyed by its resource (the
//! object, or the resource context and name of a direct dictionary), the
//! exact bits of the matrix that places it and of the page height, the
//! working space and rendering intent its colours convert to, and, when
//! the resources in effect replace a device colour space with a default
//! space, the resource context, so every use sees what a fresh resolve under
//! its own matrix and resources returns.  A tiling cell is keyed by its
//! pattern stream and, for a pattern without resources of its own, by the
//! resource context its content resolves names in, which sizes its inline
//! images.  A cell's raster is keyed additionally by the exact bits of the
//! pattern-to-device matrix, the tint of an uncoloured pattern and the
//! sub-pixel phase it is rasterised at.  On an ink canvas a shading is keyed
//! as well by the separator its colours convert by, and a cell's raster by
//! the canvas's ink setup, which decide the ink they hold.  A raster
//! rendered while the page's work budget ran out is not kept.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use color::{DeviceN8, DeviceN16, Pixel, Rgb8, Rgb16};
use pdf::ObjectId;
use raster::{Bitmap, CellPhase};

use super::super::color::{InkChannels, RasterColor};
use super::PageRenderer;
use super::canvas::CanvasKind;
use crate::content::Operator;
use crate::resources::ColorSpace;
use crate::resources::shading::{Canvas, PlanColorants, ResolvedShading};
use crate::resources::tiling::TilingDescriptor;
use crate::resources::{ResourceIdentity, ResourceKey, ShadingPattern};
use crate::separation::{InkSetup, Separator};

/// Host bytes of meshes and cell rasters the memo holds at most.  An entry
/// that would exceed it is used without being kept.
const BUDGET_BYTES: usize = 64 << 20;

/// The memo shared by a page renderer and the renderers of its pattern
/// cells.
pub(super) type SharedPatternMemo = Rc<RefCell<PatternMemo>>;

/// A part of a key that takes part in its equality and not in its hash:
/// keys equal in it hash alike, whatever it holds.
#[derive(Clone, PartialEq, Eq)]
struct Unhashed<T>(T);

impl<T> Hash for Unhashed<T> {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

/// A shading or shading pattern placed in device space.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct PlacedKey {
    id: ResourceKey,
    matrix: [u64; 6],
    canvas: [u64; 2],
    /// The working space and rendering intent the colours convert to.
    target: crate::ColourTarget,
    /// The resource context, when its default colour spaces take part.
    defaults: Option<ObjectId>,
    /// The canvas the object is resolved for.
    kind: CanvasKind,
    /// The separator of the colours on an ink canvas, `None` on an RGB
    /// one.
    separator: Unhashed<Option<Separator>>,
}

impl PlacedKey {
    /// The key of object `id` under `matrix` on `canvas`, of the kind
    /// `kind`, whose colours convert by `separator` on an ink canvas,
    /// resolved with the default colour spaces of `defaults`.
    pub(super) fn new(
        id: ResourceKey,
        matrix: &[f64; 6],
        canvas: Canvas,
        (kind, separator): (CanvasKind, &Separator),
        defaults: Option<ObjectId>,
    ) -> Self {
        Self {
            id,
            matrix: matrix.map(f64::to_bits),
            canvas: [canvas.width.to_bits(), canvas.height.to_bits()],
            target: canvas.target,
            defaults,
            kind,
            separator: Unhashed(kind.is_ink().then(|| separator.clone())),
        }
    }
}

/// A tiling cell rasterised under one pattern-to-device matrix.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct RasterKey {
    id: ObjectId,
    device: [u64; 6],
    tint: Option<RasterColor>,
    context: Option<ResourceIdentity>,
    phase: CellPhase,
    /// The canvas the cell is rasterised for, and its working space.
    kind: (CanvasKind, crate::WorkingSpace),
    /// The ink setup of an ink canvas, `None` for an RGB one.
    ink: Unhashed<Option<InkSetup>>,
}

impl RasterKey {
    /// The key of pattern stream `id` under `device` with `tint`,
    /// rasterised for a canvas of `kind`, working space `space` and, on an
    /// ink canvas, ink setup `ink`; `context` is the identity of the
    /// resources a pattern without its own inherits.
    pub(super) fn new(
        id: ObjectId,
        device: &[f64; 6],
        tint: Option<&RasterColor>,
        context: Option<ResourceIdentity>,
        (kind, space, ink): (CanvasKind, crate::WorkingSpace, &InkSetup),
    ) -> Self {
        Self {
            id,
            device: device.map(f64::to_bits),
            tint: tint.cloned(),
            context,
            phase: (0, 0),
            kind: (kind, space),
            ink: Unhashed(kind.is_ink().then(|| ink.clone())),
        }
    }

    /// The key of the same cell rasterised at sub-pixel `phase`.
    pub(super) fn at_phase(&self, phase: CellPhase) -> Self {
        Self {
            phase,
            ..self.clone()
        }
    }
}

/// A tiling pattern's descriptor and parsed content.
pub(super) struct TilingCell {
    pub(super) desc: TilingDescriptor,
    pub(super) ops: Vec<Operator<'static>>,
}

/// A rasterised tiling cell.
#[derive(Clone)]
pub(super) enum CellRaster {
    /// The mean colour and coverage of a cell below one pixel in area.
    Mean([u8; 3], u8),
    /// The cell bitmap, composited at every lattice point.
    Cell(Rc<Bitmap<Rgb8>>),
    /// The mean 16-bit colour and coverage of a cell below one pixel in
    /// area.
    Mean16([u16; 3], u16),
    /// The cell bitmap of 16-bit samples.
    Cell16(Rc<Bitmap<Rgb16>>),
    /// The mean ink and coverage of a cell below one pixel in area, with
    /// the channels the cell's content painted and carried.
    InkMean([u8; 8], u8, InkChannels),
    /// The cell bitmap of ink, with the channels its content painted and
    /// carried.
    InkCell(Rc<Bitmap<DeviceN8>>, InkChannels),
    /// The mean 16-bit ink and coverage of a cell below one pixel in area,
    /// with the channels the cell's content painted and carried.
    InkMean16([u16; 8], u16, InkChannels),
    /// The cell bitmap of 16-bit ink, with the channels its content
    /// painted and carried.
    InkCell16(Rc<Bitmap<DeviceN16>>, InkChannels),
}

/// Resolved shadings and tiling cells keyed by object.  A `None` value
/// records an object that does not resolve.
pub(super) struct PatternMemo {
    shading_patterns: HashMap<PlacedKey, Option<Rc<ShadingPattern>>>,
    shadings: HashMap<PlacedKey, Option<Rc<ResolvedShading>>>,
    /// Keyed by pattern stream and, for a pattern without resources of its
    /// own, by the resource context the content was parsed in.
    cells: HashMap<(ObjectId, Option<ResourceIdentity>), Option<Rc<TilingCell>>>,
    rasters: HashMap<RasterKey, CellRaster>,
    budget: Budget,
}

/// Host bytes held by the kept meshes and rasters, against a limit.
struct Budget {
    used: usize,
    limit: usize,
}

impl Budget {
    /// Reserve `bytes`; `false` when they do not fit.
    const fn reserve(&mut self, bytes: usize) -> bool {
        let fits = bytes <= self.limit - self.used;
        if fits {
            self.used += bytes;
        }
        fits
    }
}

/// The value kept at `key` in `map`, else `resolve()`, kept when its
/// `bytes` fit `budget`.  `None`, an object that does not resolve, is kept
/// as well.
fn kept_or<K: Eq + Hash, V>(
    map: &mut HashMap<K, Option<Rc<V>>>,
    budget: &mut Budget,
    key: K,
    bytes: impl FnOnce(&V) -> usize,
    resolve: impl FnOnce() -> Option<Rc<V>>,
) -> Option<Rc<V>> {
    if let Some(kept) = map.get(&key) {
        return kept.clone();
    }
    let value = resolve();
    if budget.reserve(value.as_deref().map_or(0, bytes)) {
        let _ = map.insert(key, value.clone());
    }
    value
}

impl PatternMemo {
    /// An empty memo holding at most [`BUDGET_BYTES`].
    pub(super) fn shared() -> SharedPatternMemo {
        Rc::new(RefCell::new(Self::with_budget(BUDGET_BYTES)))
    }

    fn with_budget(budget: usize) -> Self {
        Self {
            shading_patterns: HashMap::new(),
            shadings: HashMap::new(),
            cells: HashMap::new(),
            rasters: HashMap::new(),
            budget: Budget {
                used: 0,
                limit: budget,
            },
        }
    }

    /// The shading pattern at `key`, kept or from `resolve`.
    pub(super) fn shading_pattern(
        &mut self,
        key: PlacedKey,
        resolve: impl FnOnce() -> Option<Rc<ShadingPattern>>,
    ) -> Option<Rc<ShadingPattern>> {
        kept_or(
            &mut self.shading_patterns,
            &mut self.budget,
            key,
            |p| shading_bytes(&p.shading),
            resolve,
        )
    }

    /// The shading at `key`, kept or from `resolve`.
    pub(super) fn shading(
        &mut self,
        key: PlacedKey,
        resolve: impl FnOnce() -> Option<Rc<ResolvedShading>>,
    ) -> Option<Rc<ResolvedShading>> {
        kept_or(
            &mut self.shadings,
            &mut self.budget,
            key,
            shading_bytes,
            resolve,
        )
    }

    /// The descriptor and content of tiling pattern `id` used under the
    /// resources `context` identifies, kept or from `resolve`.  A pattern
    /// with resources of its own, or one that does not resolve, is kept
    /// for every context.
    pub(super) fn cell(
        &mut self,
        id: ObjectId,
        context: ResourceIdentity,
        resolve: impl FnOnce() -> Option<Rc<TilingCell>>,
    ) -> Option<Rc<TilingCell>> {
        if let Some(kept) = [(id, None), (id, Some(context))]
            .iter()
            .find_map(|key| self.cells.get(key))
        {
            return kept.clone();
        }
        let cell = resolve();
        let inherits = cell.as_ref().is_some_and(|c| !c.desc.has_own_resources);
        let _ = self
            .cells
            .insert((id, inherits.then_some(context)), cell.clone());
        cell
    }

    /// The kept raster at `key`.
    pub(super) fn raster(&self, key: &RasterKey) -> Option<CellRaster> {
        self.rasters.get(key).cloned()
    }

    /// Keep `raster` at `key` when it fits the budget.  An entry already
    /// kept stays.
    pub(super) fn insert_raster(&mut self, key: RasterKey, raster: CellRaster) {
        let bytes = match &raster {
            CellRaster::Mean(..)
            | CellRaster::Mean16(..)
            | CellRaster::InkMean(..)
            | CellRaster::InkMean16(..) => 0,
            CellRaster::Cell(bitmap) => bitmap_bytes(bitmap),
            CellRaster::Cell16(bitmap) => bitmap_bytes(bitmap),
            CellRaster::InkCell(bitmap, _) => bitmap_bytes(bitmap),
            CellRaster::InkCell16(bitmap, _) => bitmap_bytes(bitmap),
        };
        if !self.rasters.contains_key(&key) && self.budget.reserve(bytes) {
            let _ = self.rasters.insert(key, raster);
        }
    }
}

/// The bytes `bitmap` holds: its pixels and its alpha plane, if any.
fn bitmap_bytes<P: Pixel>(bitmap: &Bitmap<P>) -> usize {
    let samples = bitmap.data().len() + bitmap.alpha_plane().map_or(0, <[P::Sample]>::len);
    samples * size_of::<P::Sample>()
}

/// A bound on the bytes of the raster of a cell spanning `cell_w × cell_h`
/// device pixels in `P`: one pixel more on each axis for the cell's
/// placement within its pixels, each pixel `P::BYTES` and one byte of
/// alpha.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the cell sides are finite and non-negative and P::BYTES is far below 2^52; \
              the float-to-integer cast saturates"
)]
pub(super) fn cell_raster_bytes<P: Pixel>(cell_w: f64, cell_h: f64) -> usize {
    ((cell_w + 1.0) * (cell_h + 1.0) * (P::BYTES + 1) as f64) as usize
}

impl PageRenderer<'_> {
    /// The named `Pattern` resource resolved as a shading pattern under the
    /// pattern matrix in effect, for the canvas, from the memo when kept.
    pub(super) fn memo_shading_pattern(&self, name: &[u8]) -> Option<Rc<ShadingPattern>> {
        let canvas = self.shading_canvas();
        let kind = self.canvas.kind();
        let pattern_ctm = self.pattern_ctm;
        let separator = self.separator(false);
        let key = PlacedKey::new(
            self.resources.pattern_key(name)?,
            &pattern_ctm,
            canvas,
            (kind, &separator),
            self.default_spaces_context(),
        );
        let (resources, ink_map) = (&self.resources, &self.ink_map);
        self.pattern_memo.borrow_mut().shading_pattern(key, || {
            let doc = resources.doc();
            let mut plan = |space: &ColorSpace| ink_map.borrow_mut().plan(doc, space, &separator);
            let ink: Option<PlanColorants<'_>> = kind.is_ink().then_some(&mut plan);
            resources
                .shading_pattern(name, &pattern_ctm, canvas, ink)
                .map(Rc::new)
        })
    }

    /// The named `Shading` resource resolved under `ctm`, a matrix into the
    /// whole page's space, for the canvas, from the memo when kept.
    pub(super) fn memo_shading(&self, name: &[u8], ctm: &[f64; 6]) -> Option<Rc<ResolvedShading>> {
        let canvas = self.shading_canvas();
        let kind = self.canvas.kind();
        let separator = self.separator(false);
        let key = PlacedKey::new(
            self.resources.shading_key(name)?,
            ctm,
            canvas,
            (kind, &separator),
            self.default_spaces_context(),
        );
        let (resources, ink_map) = (&self.resources, &self.ink_map);
        self.pattern_memo.borrow_mut().shading(key, || {
            let doc = resources.doc();
            let mut plan = |space: &ColorSpace| ink_map.borrow_mut().plan(doc, space, &separator);
            let ink: Option<PlanColorants<'_>> = kind.is_ink().then_some(&mut plan);
            resources.shading(name, ctm, canvas, ink).map(Rc::new)
        })
    }

    /// The canvas shadings are resolved for: the whole page, whose device
    /// space every CTM maps to.  A window paints a shading resolved there,
    /// moved by the window's origin last (see `Self::page_origin`), so it
    /// receives the colours the whole page computes for its pixels.
    fn shading_canvas(&self) -> Canvas {
        Canvas {
            width: f64::from(self.page_width),
            height: f64::from(self.page_height),
            depth: self.canvas.kind().depth(),
            target: self.colour_target(),
        }
    }

    /// The resource context whose default colour spaces a shading or an
    /// image resolved here takes; `None` when the resources in effect have
    /// none.
    pub(super) fn default_spaces_context(&self) -> Option<ObjectId> {
        self.resources.has_default_spaces().then(|| {
            self.resources
                .color_space_source()
                .unwrap_or_else(|| self.resources.resource_context_id())
        })
    }

    /// The named `Pattern` resource resolved as a tiling pattern, with its
    /// content parsed, from the memo when kept.  A tiling pattern is a
    /// stream, so only an indirect one resolves.
    pub(super) fn memo_tiling_cell(&self, name: &[u8]) -> Option<Rc<TilingCell>> {
        let Some(ResourceKey::Object(id)) = self.resources.pattern_key(name) else {
            return None;
        };
        let context = self.resources.identity();
        self.pattern_memo.borrow_mut().cell(id, context, || {
            self.resources.tiling_pattern(name).map(|desc| {
                // The cell's inline images name colour spaces from the
                // pattern's own resources, or the parent context's when it
                // has none.
                let cell_resources = if desc.has_own_resources {
                    self.resources.for_context(desc.stream_id)
                } else {
                    self.resources.inherited()
                };
                let components = |name: &[u8]| cell_resources.inline_image_components(name);
                let ops = crate::content::parse_with(&desc.content, Some(&components))
                    .into_iter()
                    .map(Operator::into_owned)
                    .collect();
                Rc::new(TilingCell { desc, ops })
            })
        })
    }

    /// The raster at `key`, from the memo when kept, else from `render`, or
    /// `None` when `render` gives none.  A raster rendered after the page's
    /// work budget ran out is not kept.
    pub(super) fn memo_raster(
        &mut self,
        key: RasterKey,
        render: impl FnOnce(&mut Self) -> Option<CellRaster>,
    ) -> Option<CellRaster> {
        if let Some(kept) = self.pattern_memo.borrow().raster(&key) {
            return Some(kept);
        }
        let raster = render(self)?;
        if self.budget_exceeded.is_none() {
            self.pattern_memo
                .borrow_mut()
                .insert_raster(key, raster.clone());
        }
        Some(raster)
    }
}

/// Host bytes a resolved shading holds (see [`ResolvedShading::held_bytes`]).
const fn shading_bytes(shading: &ResolvedShading) -> usize {
    shading.held_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::shading::gouraud::GouraudVertex;

    fn mesh(triangles: usize) -> Rc<ResolvedShading> {
        let v = GouraudVertex {
            x: 0.0,
            y: 0.0,
            color: [0; 3],
        };
        let mesh = crate::resources::shading::Mesh::new(vec![[v; 3]; triangles]);
        let held_bytes = std::mem::size_of_val(mesh.triangles());
        Rc::new(ResolvedShading {
            result: crate::resources::shading::ShadingResult::Mesh(mesh),
            bbox: None,
            background: None,
            background_ink: None,
            paints: raster::pipe::ChannelMask::ALL,
            held_bytes,
            gray: false,
        })
    }

    const ID: ObjectId = (5, 0);
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    /// The key of `id` under `matrix` on a canvas 100 pixels wide and
    /// `page_h` high.
    fn placed(id: ObjectId, matrix: &[f64; 6], page_h: f64) -> PlacedKey {
        let canvas = Canvas {
            width: 100.0,
            height: page_h,
            depth: color::SampleDepth::Eight,
            target: crate::ColourTarget::SRGB,
        };
        PlacedKey::new(
            ResourceKey::Object(id),
            matrix,
            canvas,
            (CanvasKind::Rgb, &Separator::Model),
            None,
        )
    }

    /// Whether `memo` holds the shading at `key`: a lookup that would
    /// otherwise resolve to `None`.
    fn holds(memo: &mut PatternMemo, key: PlacedKey) -> bool {
        memo.shading(key, || None).is_some()
    }

    /// A cell raster's bound counts each pixel's bytes in the pixel type
    /// and its alpha: four for sRGB, nine for ink.
    #[test]
    fn cell_raster_bytes_count_the_pixel_type() {
        assert_eq!(cell_raster_bytes::<Rgb8>(9.0, 4.0), 10 * 5 * 4);
        assert_eq!(cell_raster_bytes::<DeviceN8>(9.0, 4.0), 10 * 5 * 9);
    }

    /// The content of a pattern without resources of its own is parsed in
    /// each resource context it is used in: an inline image naming a colour
    /// space takes its data length from the context's definition.
    #[test]
    fn an_inheriting_cell_is_parsed_per_resource_context() {
        // One RGB sample whose bytes read `EI`: a one-component space ends
        // the image data after the first byte.
        let cell = b"BI /W 1 /H 1 /CS /C /BPC 8 ID  EI\nEI";
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (4, b"<< /Type /Page >>".to_vec()),
            (
                5,
                format!(
                    "<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 1 1] \
                     /XStep 1 /YStep 1 /Length {} >>\nstream\n{}\nendstream",
                    cell.len(),
                    String::from_utf8_lossy(cell)
                )
                .into_bytes(),
            ),
        ]);
        let resources = |space: &str| {
            pdf::parse_object(
                format!("<< /Pattern << /P 5 0 R >> /ColorSpace << /C /{space} >> >>").as_bytes(),
                &mut 0,
            )
            .unwrap()
        };
        let (gray, rgb) = (resources("DeviceGray"), resources("DeviceRGB"));
        let mut renderer = PageRenderer::new(1, 1, &doc, (4, 0));
        let page = renderer.resources.inherited();
        let image_len = |renderer: &PageRenderer<'_>| {
            let cell = renderer.memo_tiling_cell(b"P").expect("cell");
            match cell.ops.first() {
                Some(Operator::InlineImage { data, .. }) => data.len(),
                other => panic!("{other:?}"),
            }
        };
        renderer.resources = page.with_resources((10, 0), gray.as_dict().unwrap());
        assert_eq!(image_len(&renderer), 1);
        renderer.resources = page.with_resources((11, 0), rgb.as_dict().unwrap());
        assert_eq!(image_len(&renderer), 3);

        // Sets built over dictionaries share their font-cache context, not
        // their cells.
        renderer.resources = page.with_resources((0, 0), gray.as_dict().unwrap());
        assert_eq!(image_len(&renderer), 1);
        renderer.resources = page.with_resources((0, 0), rgb.as_dict().unwrap());
        assert_eq!(image_len(&renderer), 3);
    }

    #[test]
    fn a_shading_is_kept_per_matrix_and_page_height() {
        let mut memo = PatternMemo::with_budget(BUDGET_BYTES);
        assert!(
            memo.shading(placed(ID, &IDENTITY, 100.0), || Some(mesh(1)))
                .is_some()
        );
        assert!(holds(&mut memo, placed(ID, &IDENTITY, 100.0)));
        let moved = [1.0, 0.0, 0.0, 1.0, 0.5, 0.0];
        assert!(!holds(&mut memo, placed(ID, &moved, 100.0)));
        assert!(!holds(&mut memo, placed(ID, &IDENTITY, 200.0)));
        assert!(!holds(&mut memo, placed((6, 0), &IDENTITY, 100.0)));
    }

    #[test]
    fn an_unresolvable_shading_is_kept_as_none() {
        let mut memo = PatternMemo::with_budget(0);
        assert!(
            memo.shading(placed(ID, &IDENTITY, 100.0), || None)
                .is_none()
        );
        assert!(
            memo.shading(placed(ID, &IDENTITY, 100.0), || Some(mesh(0)))
                .is_none(),
            "the kept None answers"
        );
    }

    #[test]
    fn a_mesh_over_the_budget_is_not_kept() {
        let size = std::mem::size_of::<[GouraudVertex; 3]>();
        let mut memo = PatternMemo::with_budget(size * 3);
        let _ = memo.shading(placed(ID, &IDENTITY, 100.0), || Some(mesh(2)));
        let _ = memo.shading(placed((6, 0), &IDENTITY, 100.0), || Some(mesh(2)));
        assert!(holds(&mut memo, placed(ID, &IDENTITY, 100.0)));
        assert!(
            !holds(&mut memo, placed((6, 0), &IDENTITY, 100.0)),
            "2 + 2 triangles exceed 3"
        );
    }

    #[test]
    fn a_raster_is_kept_per_tint_context_and_phase() {
        let mut memo = PatternMemo::with_budget(BUDGET_BYTES);
        let red = RasterColor::rgb(1.0, 0.0, 0.0);
        let key = RasterKey::new(
            ID,
            &IDENTITY,
            Some(&red),
            None,
            (
                CanvasKind::Rgb,
                crate::WorkingSpace::Srgb,
                &InkSetup::PREVIEW,
            ),
        );
        memo.insert_raster(key.clone(), CellRaster::Mean([255, 0, 0], 255));
        assert!(memo.raster(&key).is_some());
        let green = RasterColor::rgb(0.0, 1.0, 0.0);
        assert!(
            memo.raster(&RasterKey::new(
                ID,
                &IDENTITY,
                Some(&green),
                None,
                (
                    CanvasKind::Rgb,
                    crate::WorkingSpace::Srgb,
                    &InkSetup::PREVIEW
                )
            ))
            .is_none()
        );
        assert!(
            memo.raster(&RasterKey::new(
                ID,
                &IDENTITY,
                Some(&red),
                Some(ResourceIdentity::Object((1, 0))),
                (
                    CanvasKind::Rgb,
                    crate::WorkingSpace::Srgb,
                    &InkSetup::PREVIEW
                )
            ))
            .is_none()
        );
        assert!(memo.raster(&key.at_phase((1, 0))).is_none());
    }
}
