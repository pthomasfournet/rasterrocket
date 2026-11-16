//! The bytes of a page's device inputs, counted against
//! [`PAGE_COMMANDS_MAX_BYTES`](super::PAGE_COMMANDS_MAX_BYTES).

use super::{PageCmd, PageCommands, PageCommandsError};

/// `v` as a `u32` element index.
pub(super) fn index(v: usize) -> Result<u32, PageCommandsError> {
    u32::try_from(v).map_err(|_| PageCommandsError::TooLarge)
}

/// Counts of the elements of a page's device inputs.
#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub(super) cmds: usize,
    pub(super) spans: usize,
    pub(super) coverage: usize,
    pub(super) row_starts: usize,
    /// Tile references, four words each.
    pub(super) tile_refs: usize,
    /// Of a change only: the tile references it hides, which
    /// [`PageCommands::bin`] leaves out.
    pub(super) hidden_refs: usize,
    pub(super) segs: usize,
    pub(super) seg_lists: usize,
    pub(super) backdrops: usize,
}

impl Counts {
    /// The bytes of device inputs holding these elements and `tiles + 1`
    /// tile offsets, saturating.  The coverage bytes are packed four to a
    /// word.
    const fn bytes(self, tiles: usize) -> usize {
        let words = (self.coverage.saturating_add(3) / 4)
            .saturating_add(self.row_starts)
            .saturating_add(tiles.saturating_add(1))
            .saturating_add(self.tile_refs.saturating_mul(4))
            .saturating_add(self.segs.saturating_mul(4))
            .saturating_add(self.seg_lists)
            .saturating_add(self.backdrops);
        size_of::<PageCmd>()
            .saturating_mul(self.cmds)
            .saturating_add(size_of::<[i32; 3]>().saturating_mul(self.spans))
            .saturating_add(size_of::<u32>().saturating_mul(words))
    }
}

impl PageCommands {
    /// Bytes of the page's device inputs: the commands, their tables and
    /// their tile lists.  At most [`PAGE_COMMANDS_MAX_BYTES`].
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes_with(Counts {
            cmds: 0,
            spans: 0,
            coverage: 0,
            row_starts: 0,
            tile_refs: 0,
            hidden_refs: 0,
            segs: 0,
            seg_lists: 0,
            backdrops: 0,
        })
    }

    /// Bytes the page's device inputs would take with `more` more elements;
    /// `more.hidden_refs` of the tile references are taken away.
    pub(super) const fn bytes_with(&self, more: Counts) -> usize {
        Counts {
            cmds: self.cmds.len().saturating_add(more.cmds),
            spans: self.spans.len().saturating_add(more.spans),
            coverage: self.bytes.len().saturating_add(more.coverage),
            row_starts: self.row_starts.len().saturating_add(more.row_starts),
            tile_refs: self
                .tile_refs
                .saturating_add(more.tile_refs)
                .saturating_sub(more.hidden_refs),
            hidden_refs: 0,
            segs: self.segs.len().saturating_add(more.segs),
            seg_lists: self.seg_lists.len().saturating_add(more.seg_lists),
            backdrops: self.backdrops.len().saturating_add(more.backdrops),
        }
        .bytes(self.tile_count())
    }
}
