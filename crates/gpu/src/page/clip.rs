//! Path clips: coverage tables that scale the coverage of the commands
//! pushed while they are in effect.
//!
//! A clip region is a coverage table the CPU computed for a path clip,
//! held once by the page ([`PageCommands::add_clip_region`]).  A clip is a
//! sequence of regions, newest first ([`PageCommands::add_clip`]); the
//! clip in effect ([`PageCommands::set_clip`]) scales each command pushed
//! after it as the CPU clip does: with `c_i` the coverage of region `i`
//! and `div255(v) = v / 255` rounded to nearest, a value `v` is scaled to
//! `div255(… div255(div255(v · c_1) · c_2) … · c_n)`.  An outline's or a
//! table's coverage is scaled itself; a mask's coverage `m` becomes
//! `div255(m · s)` and an image's shape `a` becomes `a · s / 255` at the
//! page's depth (`div_max(a · s · M / 255)`), `s` being the clip's scaling
//! of 255.  The two differ from two regions on, and each matches the CPU
//! consumer of its kind.  A command's bounds are narrowed to the pixels
//! every region covers, and a clipped outline hides no item below it.
//!
//! Records, in the modifier words: a region `[y_min, n_rows, row_base]`,
//! its rows in the row offsets as a table command's; a clip `[n, region
//! word …]`.

use super::table::table_counts;
use super::{
    Counts, CoverageRows, PAGE_COMMANDS_MAX_BYTES, PageCmd, PageCommands, PageCommandsError,
    PixelBounds, index,
};

/// Words of a clip region's record.
const REGION_WORDS: usize = 3;

/// A clip region a page holds ([`PageCommands::add_clip_region`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRegionId {
    word: u32,
    /// The pixels the region covers, `None` for none.
    bounds: Option<PixelBounds>,
}

/// A clip a page holds ([`PageCommands::add_clip`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipId {
    word: u32,
    /// The pixels every region covers, `None` for none.
    bounds: Option<PixelBounds>,
}

/// The clip and soft-masked object of the commands pushed now.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Modifiers {
    pub(super) clip: Option<ClipId>,
    /// The word of the object's record.
    pub(super) object: Option<u32>,
}

impl Modifiers {
    /// Whether a command pushed now is neither clipped nor part of an
    /// object: only such an outline may hide the items below it.
    pub(super) const fn is_plain(self) -> bool {
        self.clip.is_none() && self.object.is_none()
    }

    /// A command carrying these modifiers, its other fields zero.
    pub(super) fn stamp(self) -> PageCmd {
        PageCmd {
            clip: self.clip.map_or(0, |c| c.word + 1),
            object: self.object.map_or(0, |w| w + 1),
            ..PageCmd::default()
        }
    }
}

/// The pixels `table` covers, `None` for none.
fn table_bounds(table: &CoverageRows<'_>) -> Option<PixelBounds> {
    let rows = table.row_starts.windows(2).zip(table.y_min..);
    let mut covered = rows.filter(|(w, _)| w[0] < w[1]).map(|(_, y)| y);
    let y0 = covered.next()?;
    let y1 = covered.last().unwrap_or(y0);
    let x0 = table.spans.iter().map(|s| s[0]).min()?;
    let x1 = table.spans.iter().map(|s| s[1]).max()?;
    Some(PixelBounds { x0, y0, x1, y1 })
}

impl PageCommands {
    /// Hold `table`, the coverage of a path clip, as a clip region.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When the table is malformed ([`CoverageRows::is_well_formed`]).
    pub fn add_clip_region(
        &mut self,
        table: CoverageRows<'_>,
    ) -> Result<ClipRegionId, PageCommandsError> {
        assert!(
            table.is_well_formed(),
            "add_clip_region: malformed coverage table"
        );
        let more = Counts {
            mod_words: REGION_WORDS,
            ..table_counts(&table)
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let word = index(self.mod_words.len())?;
        index(self.mod_words.len() + REGION_WORDS).map(drop)?;
        let at = self.reserve_rows(&table)?;
        self.mod_words.try_reserve(REGION_WORDS)?;
        let (row_base, n_rows) = self.extend_rows(&table, at);
        self.mod_words
            .extend([table.y_min.cast_unsigned(), n_rows, row_base]);
        Ok(ClipRegionId {
            word,
            bounds: table_bounds(&table),
        })
    }

    /// Hold the clip of `regions`, newest first: the order in which they
    /// scale a coverage.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When `regions` is empty.
    pub fn add_clip(&mut self, regions: &[ClipRegionId]) -> Result<ClipId, PageCommandsError> {
        assert!(!regions.is_empty(), "add_clip: a clip without regions");
        let n = regions.len() + 1;
        let more = Counts {
            mod_words: n,
            ..Counts::ZERO
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let word = index(self.mod_words.len())?;
        index(self.mod_words.len() + n).map(drop)?;
        self.mod_words.try_reserve(n)?;
        self.mod_words.push(index(regions.len())?);
        self.mod_words.extend(regions.iter().map(|r| r.word));
        let bounds = regions[1..].iter().fold(regions[0].bounds, |acc, r| {
            acc.zip(r.bounds).and_then(|(a, b)| a.intersect(b))
        });
        Ok(ClipId { word, bounds })
    }

    /// Clip the commands pushed from now on by `clip`, or by none.
    /// Opaque runs ([`Self::push_opaque_runs`]) are never clipped.
    pub const fn set_clip(&mut self, clip: Option<ClipId>) {
        self.modifiers.clip = clip;
    }

    /// The pixels a command pushed now may paint: those of the page the
    /// clip's regions all cover; `None` when there is none.
    pub(super) fn paint_bounds(&self) -> Result<Option<PixelBounds>, PageCommandsError> {
        let page = self.page_bounds()?;
        Ok(self.modifiers.clip.map_or(page, |clip| {
            page.zip(clip.bounds).and_then(|(p, c)| p.intersect(c))
        }))
    }

    /// `v` scaled by the regions of the clip whose record is at modifier
    /// word `word`, at pixel `(x, y)`.
    pub(super) fn clip_scale(&self, word: u32, x: i32, y: i32, v: u8) -> u8 {
        let at = word as usize;
        let n = self.mod_words[at] as usize;
        self.mod_words[at + 1..][..n].iter().fold(v, |v, &region| {
            color::convert::div255(u32::from(v) * u32::from(self.region_coverage(region, x, y)))
        })
    }

    /// The coverage of pixel `(x, y)` by the clip region whose record is at
    /// modifier word `word`.
    fn region_coverage(&self, word: u32, x: i32, y: i32) -> u8 {
        let [y_min, n_rows, row_base] = *self.mod_words[word as usize..]
            .first_chunk::<REGION_WORDS>()
            .expect("a region record");
        let Ok(row) = u32::try_from(i64::from(y) - i64::from(y_min.cast_signed())) else {
            return 0;
        };
        if row >= n_rows {
            return 0;
        }
        let r = (row_base + row) as usize;
        let spans = &self.spans[self.row_starts[r] as usize..self.row_starts[r + 1] as usize];
        super::reference::row_coverage(spans, &self.bytes, x)
    }
}

#[cfg(test)]
mod tests;
