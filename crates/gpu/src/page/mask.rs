//! Coverage masks: rectangles of coverage bytes a page holds once and any
//! number of commands paint through, each at its own position.
//!
//! A mask's bytes are appended to the page's coverage bytes, row-major,
//! `width` bytes per row; a command painting it at `(left, top)` takes the
//! coverage of pixel `(x, y)` from byte `(y − top) · width + (x − left)` of
//! the mask, and paints no pixel outside the mask's rectangle there.

use super::{
    Counts, KIND_MASK, PAGE_COMMANDS_MAX_BYTES, PageCmd, PageCommands, PageCommandsError,
    PixelBounds, index, pack_rgb,
};

/// A mask held by a page ([`PageCommands::add_mask`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskId {
    /// Offset of the mask's first byte in the page's coverage bytes.
    offset: u32,
    width: u32,
    height: u32,
}

impl MaskId {
    /// Width of the mask in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height of the mask in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// The layout of a mask's source bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskBits {
    /// One coverage byte per pixel, `width` bytes per row.
    Coverage,
    /// One bit per pixel, most significant first, `⌈width / 8⌉` bytes per
    /// row; a set bit is full coverage, a clear one none.
    Mono,
}

impl PageCommands {
    /// Hold the `width × height` mask `data`, laid out as `bits`, for
    /// [`Self::push_mask`].  Bytes missing at the end of `data` read as no
    /// coverage, bytes past the mask are ignored.
    ///
    /// # Errors
    ///
    /// [`PageCommandsError`] when the mask would take the page past
    /// [`PAGE_COMMANDS_MAX_BYTES`] or its memory cannot be reserved; the
    /// page is unchanged then.
    pub fn add_mask(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        bits: MaskBits,
    ) -> Result<MaskId, PageCommandsError> {
        let n = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| PageCommandsError::TooLarge)?;
        let more = Counts {
            coverage: n,
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let offset = index(self.bytes.len())?;
        i32::try_from(self.bytes.len() + n)
            .map(drop)
            .map_err(|_| PageCommandsError::TooLarge)?;
        self.bytes.try_reserve(n)?;
        let w = width as usize;
        match bits {
            MaskBits::Coverage => {
                let kept = &data[..n.min(data.len())];
                self.bytes.extend_from_slice(kept);
                self.bytes.resize(self.bytes.len() + n - kept.len(), 0);
            }
            MaskBits::Mono => {
                let row_bytes = w.div_ceil(8);
                for y in 0..height as usize {
                    self.bytes.extend((0..w).map(|x| {
                        let set = data
                            .get(y * row_bytes + x / 8)
                            .is_some_and(|&b| (b >> (7 - x % 8)) & 1 != 0);
                        if set { u8::MAX } else { 0 }
                    }));
                }
            }
        }
        Ok(MaskId {
            offset,
            width,
            height,
        })
    }

    /// Paint `rgb` at constant alpha `a_input` with the coverage of `mask`
    /// placed with its top-left pixel at `(left, top)`, as the shape, over
    /// the pixels of `bounds` that lie in the mask's rectangle and on the
    /// page.  Nothing is pushed when none does.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When a sample of `rgb` or `a_input` exceeds the page's depth.
    pub fn push_mask(
        &mut self,
        mask: MaskId,
        (left, top): (i32, i32),
        bounds: PixelBounds,
        rgb: [u16; 3],
        a_input: u16,
    ) -> Result<(), PageCommandsError> {
        let max = self.depth.max_sample();
        assert!(
            rgb.iter().all(|&v| v <= max) && a_input <= max,
            "push_mask: colour {rgb:?} or alpha {a_input} exceeds {} bits",
            self.depth.bits()
        );
        if mask.width == 0 || mask.height == 0 {
            return Ok(());
        }
        let last = |origin: i32, extent: u32| {
            i32::try_from(i64::from(origin) + i64::from(extent) - 1).unwrap_or(i32::MAX)
        };
        let rect = PixelBounds {
            x0: left,
            y0: top,
            x1: last(left, mask.width),
            y1: last(top, mask.height),
        };
        let Some(bounds) = self
            .page_bounds()?
            .and_then(|p| p.intersect(bounds))
            .and_then(|b| b.intersect(rect))
        else {
            return Ok(());
        };
        let more = Counts {
            cmds: 1,
            tile_refs: bounds.tiles(),
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        self.cmds.try_reserve(1)?;
        self.reserve_live()?;
        self.count_rect(bounds);
        let [rg, b] = pack_rgb(rgb);
        self.cmds.push(PageCmd {
            kind: KIND_MASK,
            flags: left.cast_unsigned(),
            rg,
            b,
            a_input: u32::from(a_input),
            table_y0: top,
            n_rows: mask.width,
            row_base: mask.offset,
            x0: bounds.x0,
            y0: bounds.y0,
            x1: bounds.x1,
            y1: bounds.y1,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests;
