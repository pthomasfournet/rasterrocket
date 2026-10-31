//! Resolved shadings and tiling-pattern cells of one page, so a pattern or
//! shading painted by many operators is decoded and rasterised once.
//!
//! A resolved shading is in device space: it is keyed by its resource (the
//! object, or the resource context and name of a direct dictionary), the
//! exact bits of the matrix that places it and of the page height, and, when
//! the resources in effect replace a device colour space with a default
//! space, the resource context, so every use sees what a fresh resolve under
//! its own matrix and resources returns.  A tiling cell is keyed by its
//! pattern stream and, for a pattern without resources of its own, by the
//! resource context its content resolves names in, which sizes its inline
//! images.  A cell's raster is keyed additionally by the exact bits of the
//! pattern-to-device matrix, the tint of an uncoloured pattern and the
//! sub-pixel phase it is rasterised at.  A raster rendered while the page's
//! work budget ran out is not kept.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use color::Rgb8;
use pdf::ObjectId;
use raster::{Bitmap, CellPhase};

use super::super::color::RasterColor;
use super::PageRenderer;
use crate::content::Operator;
use crate::resources::shading::{ResolvedShading, ShadingResult};
use crate::resources::tiling::TilingDescriptor;
use crate::resources::{ResourceKey, ShadingPattern};

/// Host bytes of meshes and cell rasters the memo holds at most.  An entry
/// that would exceed it is used without being kept.
const BUDGET_BYTES: usize = 64 << 20;

/// The memo shared by a page renderer and the renderers of its pattern
/// cells.
pub(super) type SharedPatternMemo = Rc<RefCell<PatternMemo>>;

/// A shading or shading pattern placed in device space.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct PlacedKey {
    id: ResourceKey,
    matrix: [u64; 6],
    page_h: u64,
    /// The resource context, when its default colour spaces take part.
    defaults: Option<ObjectId>,
}

impl PlacedKey {
    /// The key of object `id` under `matrix` on a page `page_h` pixels
    /// high, resolved with the default colour spaces of `defaults`.
    pub(super) fn new(
        id: ResourceKey,
        matrix: &[f64; 6],
        page_h: f64,
        defaults: Option<ObjectId>,
    ) -> Self {
        Self {
            id,
            matrix: matrix.map(f64::to_bits),
            page_h: page_h.to_bits(),
            defaults,
        }
    }
}

/// A tiling cell rasterised under one pattern-to-device matrix.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct RasterKey {
    id: ObjectId,
    device: [u64; 6],
    tint: Option<RasterColor>,
    context: Option<ObjectId>,
    phase: CellPhase,
}

impl RasterKey {
    /// The key of pattern stream `id` under `device` with `tint`;
    /// `context` is the resource context of a pattern without its own
    /// resources.
    pub(super) fn new(
        id: ObjectId,
        device: &[f64; 6],
        tint: Option<&RasterColor>,
        context: Option<ObjectId>,
    ) -> Self {
        Self {
            id,
            device: device.map(f64::to_bits),
            tint: tint.cloned(),
            context,
            phase: (0, 0),
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
}

/// Resolved shadings and tiling cells keyed by object.  A `None` value
/// records an object that does not resolve.
pub(super) struct PatternMemo {
    shading_patterns: HashMap<PlacedKey, Option<Rc<ShadingPattern>>>,
    shadings: HashMap<PlacedKey, Option<Rc<ResolvedShading>>>,
    /// Keyed by pattern stream and, for a pattern without resources of its
    /// own, by the resource context the content was parsed in.
    cells: HashMap<(ObjectId, Option<ObjectId>), Option<Rc<TilingCell>>>,
    rasters: HashMap<RasterKey, CellRaster>,
    /// Rasters asked for once without being rendered whole.
    deferred: HashSet<RasterKey>,
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

    /// Whether `bytes` fit what is left.
    const fn fits(&self, bytes: usize) -> bool {
        bytes <= self.limit - self.used
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
            deferred: HashSet::new(),
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

    /// The descriptor and content of tiling pattern `id` used in resource
    /// context `context`, kept or from `resolve`.  A pattern with resources
    /// of its own, or one that does not resolve, is kept for every context.
    pub(super) fn cell(
        &mut self,
        id: ObjectId,
        context: ObjectId,
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

    /// Whether the cell raster at `key`, of at most `bytes`, is kept, or
    /// has been asked for before and fits the budget left.  A first request
    /// is recorded and answered `false`.
    pub(super) fn repeated(&mut self, key: &RasterKey, bytes: usize) -> bool {
        if self.rasters.contains_key(key) {
            return true;
        }
        if self.deferred.contains(key) {
            return self.budget.fits(bytes);
        }
        let _ = self.deferred.insert(key.clone());
        false
    }

    /// Keep `raster` at `key` when it fits the budget.  An entry already
    /// kept stays.
    pub(super) fn insert_raster(&mut self, key: RasterKey, raster: CellRaster) {
        let bytes = match &raster {
            CellRaster::Mean(..) => 0,
            CellRaster::Cell(bitmap) => {
                bitmap.data().len() + bitmap.alpha_plane().map_or(0, <[u8]>::len)
            }
        };
        if !self.rasters.contains_key(&key) && self.budget.reserve(bytes) {
            let _ = self.rasters.insert(key, raster);
        }
    }
}

impl PageRenderer<'_> {
    /// The named `Pattern` resource resolved as a shading pattern under the
    /// pattern matrix in effect, from the memo when kept.
    pub(super) fn memo_shading_pattern(&self, name: &[u8]) -> Option<Rc<ShadingPattern>> {
        let page_h = f64::from(self.height);
        let key = PlacedKey::new(
            self.resources.pattern_key(name)?,
            &self.pattern_ctm,
            page_h,
            self.default_spaces_context(),
        );
        self.pattern_memo.borrow_mut().shading_pattern(key, || {
            self.resources
                .shading_pattern(name, &self.pattern_ctm, page_h)
                .map(Rc::new)
        })
    }

    /// The named `Shading` resource resolved under `ctm`, from the memo when
    /// kept.
    pub(super) fn memo_shading(&self, name: &[u8], ctm: &[f64; 6]) -> Option<Rc<ResolvedShading>> {
        let page_h = f64::from(self.height);
        let key = PlacedKey::new(
            self.resources.shading_key(name)?,
            ctm,
            page_h,
            self.default_spaces_context(),
        );
        self.pattern_memo.borrow_mut().shading(key, || {
            self.resources.shading(name, ctm, page_h).map(Rc::new)
        })
    }

    /// The resource context whose default colour spaces a shading resolved
    /// here takes; `None` when the resources in effect have none.
    fn default_spaces_context(&self) -> Option<ObjectId> {
        self.resources
            .has_default_spaces()
            .then(|| self.resources.resource_context_id())
    }

    /// The named `Pattern` resource resolved as a tiling pattern, with its
    /// content parsed, from the memo when kept.  A tiling pattern is a
    /// stream, so only an indirect one resolves.
    pub(super) fn memo_tiling_cell(&self, name: &[u8]) -> Option<Rc<TilingCell>> {
        let Some(ResourceKey::Object(id)) = self.resources.pattern_key(name) else {
            return None;
        };
        let context = self.resources.resource_context_id();
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

    /// The raster at `key`, from the memo when kept, else from `render`.  A
    /// raster rendered after the page's work budget ran out is not kept.
    pub(super) fn memo_raster(
        &mut self,
        key: RasterKey,
        render: impl FnOnce(&mut Self) -> CellRaster,
    ) -> CellRaster {
        if let Some(kept) = self.pattern_memo.borrow().raster(&key) {
            return kept;
        }
        let raster = render(self);
        if self.budget_exceeded.is_none() {
            self.pattern_memo
                .borrow_mut()
                .insert_raster(key, raster.clone());
        }
        raster
    }
}

/// Host bytes of a resolved shading's mesh; a gradient counts nothing.
fn shading_bytes(shading: &ResolvedShading) -> usize {
    match &shading.result {
        ShadingResult::Mesh(mesh) => std::mem::size_of_val(mesh.triangles()),
        ShadingResult::Pattern(_) => 0,
    }
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
        Rc::new(ResolvedShading {
            result: ShadingResult::Mesh(crate::resources::shading::Mesh::new(vec![
                [v; 3];
                triangles
            ])),
            bbox: None,
            background: None,
        })
    }

    const ID: ObjectId = (5, 0);
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    fn placed(id: ObjectId, matrix: &[f64; 6], page_h: f64) -> PlacedKey {
        PlacedKey::new(ResourceKey::Object(id), matrix, page_h, None)
    }

    /// Whether `memo` holds the shading at `key`: a lookup that would
    /// otherwise resolve to `None`.
    fn holds(memo: &mut PatternMemo, key: PlacedKey) -> bool {
        memo.shading(key, || None).is_some()
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
        let key = RasterKey::new(ID, &IDENTITY, Some(&red), None);
        memo.insert_raster(key.clone(), CellRaster::Mean([255, 0, 0], 255));
        assert!(memo.raster(&key).is_some());
        let green = RasterColor::rgb(0.0, 1.0, 0.0);
        assert!(
            memo.raster(&RasterKey::new(ID, &IDENTITY, Some(&green), None))
                .is_none()
        );
        assert!(
            memo.raster(&RasterKey::new(ID, &IDENTITY, Some(&red), Some((1, 0))))
                .is_none()
        );
        assert!(memo.raster(&key.at_phase((1, 0))).is_none());
    }

    #[test]
    fn a_raster_is_repeated_from_its_second_request_within_the_budget() {
        let mut memo = PatternMemo::with_budget(100);
        let key = RasterKey::new(ID, &IDENTITY, None, None);
        assert!(!memo.repeated(&key, 50));
        assert!(memo.repeated(&key, 50));
        assert!(!memo.repeated(&key, 101), "over the budget left");
        let other = RasterKey::new((6, 0), &IDENTITY, None, None);
        memo.insert_raster(other.clone(), CellRaster::Mean([0; 3], 0));
        assert!(memo.repeated(&other, 1000), "a kept raster is repeated");
    }
}
