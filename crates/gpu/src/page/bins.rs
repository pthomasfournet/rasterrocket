//! Binning a page's commands into [`PAGE_TILE`]-pixel tiles.

use super::path::NO_BACKDROP;
use super::{KIND_PATH, PAGE_TILE, PageCommands, PageCommandsError, PixelBounds, index};

/// The tile lists of a page: tile `t` (row-major over `tiles_x` columns)
/// paints `items[offsets[t]..offsets[t + 1]]` in order.
///
/// An item is `[command, list, len, backdrop]`: the command's index, and
/// for an outline's command the tile's segments and backdrop, as
/// [`PageCommands::table_parts`] lays them out.  The low 14 bits of `len`
/// count the segments: the one segment of index `list` when it is 1, else
/// `segment_lists[list..][..len]`.  With bit 31 of `len` set, `backdrop`
/// is the two's-complement cover of every row of the tile; otherwise it
/// indexes the backdrops, `u32::MAX` standing for zero.  A table's command
/// has `[command, 0, 0, u32::MAX]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileBins {
    /// Tile columns.
    pub tiles_x: u32,
    /// Tile rows.
    pub tiles_y: u32,
    /// `tiles_x × tiles_y + 1` offsets into `items`.
    pub offsets: Vec<u32>,
    /// The items, grouped by tile, in paint order within a tile.
    pub items: Vec<[u32; 4]>,
}

impl PageCommands {
    /// Count one more item in every tile `bounds` touches, `bounds` on the
    /// page, its per-tile counts allocated.
    pub(super) fn count_rect(&mut self, bounds: PixelBounds) {
        self.tile_refs += bounds.tiles();
        let tiles_x = self.width.div_ceil(PAGE_TILE) as usize;
        let tile = |v: i32| v.unsigned_abs() as usize / PAGE_TILE as usize;
        for ty in tile(bounds.y0)..=tile(bounds.y1) {
            for live in &mut self.live[ty * tiles_x..][tile(bounds.x0)..=tile(bounds.x1)] {
                *live += 1;
            }
        }
    }

    /// Allocate the per-tile item counts on the first push.
    pub(super) fn reserve_live(&mut self) -> Result<(), PageCommandsError> {
        if self.live.is_empty() {
            let n = self.tile_count();
            self.live.try_reserve_exact(n)?;
            self.live.resize(n, 0);
        }
        Ok(())
    }

    /// Call `f(tile, item, occludes)` for every tile each command paints,
    /// in paint order, with `item` as in [`TileBins::items`] and
    /// `occludes` whether it hides the items before it in the tile.
    pub(super) fn for_each_item(&self, mut f: impl FnMut(usize, [u32; 4], bool)) {
        let tiles_x = self.width.div_ceil(PAGE_TILE) as usize;
        let tile = |v: i32| usize::try_from(v).unwrap_or(0) / PAGE_TILE as usize;
        for (c, i) in self.cmds.iter().zip(0u32..) {
            if c.kind == KIND_PATH {
                for t in &self.path_tiles[c.row_base as usize..][..c.n_rows as usize] {
                    let (flag, backdrop) = t.backdrop.item_words();
                    let item = [i, t.list, t.list_len | flag, backdrop];
                    f(t.tile as usize, item, t.occludes);
                }
                continue;
            }
            for ty in tile(c.y0)..=tile(c.y1) {
                for tx in tile(c.x0)..=tile(c.x1) {
                    f(ty * tiles_x + tx, [i, 0, 0, NO_BACKDROP], false);
                }
            }
        }
    }

    /// The commands' tile lists, built on the host: each command is listed,
    /// in paint order, in every tile its bounds touch, and an outline's
    /// command in every tile it may cover; the items a later item of the
    /// tile hides are left out: an outline at full alpha and weight that
    /// lists no segment in the tile and covers every pixel of it on the
    /// page fully paints them in its colour, whatever lay below.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the lists' memory cannot be reserved, or
    /// they outnumber `u32` indices (which the push limit precludes).
    pub fn bin(&self) -> Result<TileBins, PageCommandsError> {
        let tiles_x = self.width.div_ceil(PAGE_TILE);
        let tiles_y = self.height.div_ceil(PAGE_TILE);
        let n_tiles = self.tile_count();
        index(self.tile_refs).map(drop)?;
        // Per tile: the items seen so far, and the index of the last one
        // that hides those before it.
        let mut seen = Vec::new();
        seen.try_reserve_exact(n_tiles)?;
        seen.resize(n_tiles, 0u32);
        let mut first = seen.clone();
        self.for_each_item(|t, _, occludes| {
            if occludes {
                first[t] = seen[t];
            }
            seen[t] += 1;
        });
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(n_tiles + 1)?;
        offsets.resize(n_tiles + 1, 0u32);
        for t in 0..n_tiles {
            offsets[t + 1] = offsets[t] + seen[t] - first[t];
        }
        let n_items = offsets[n_tiles] as usize;
        let mut cursor = offsets.clone();
        let mut items = Vec::new();
        items.try_reserve_exact(n_items)?;
        items.resize(n_items, [0u32; 4]);
        seen.fill(0);
        self.for_each_item(|t, item, _| {
            if seen[t] >= first[t] {
                items[cursor[t] as usize] = item;
                cursor[t] += 1;
            }
            seen[t] += 1;
        });
        Ok(TileBins {
            tiles_x,
            tiles_y,
            offsets,
            items,
        })
    }
}
