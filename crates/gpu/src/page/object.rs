//! Objects painted under a soft mask.
//!
//! A soft mask is a plane of mask values, one sample of the page's depth
//! per page pixel, which the page holds once
//! ([`PageCommands::add_mask_plane`]).  The commands pushed between
//! [`PageCommands::begin_object`] and [`PageCommands::end_object`] are one
//! object: each pixel of the object's rectangle is blended back, once the
//! object's last command has painted it, toward its value `p` before the
//! object's first command: with `c` its painted value and `m` the mask's
//! value, `div_max(c · m + p · (M − m))` per channel, unchanged where
//! `m = M`, as the CPU's soft mask.  An object's commands hide no item
//! below them.
//!
//! The page holds a plane as the rectangle of the samples that differ from
//! its first, all the others being equal to it.  Records, in the modifier
//! words: a plane `[offset, deep, x0, y0, x1, y1, outside]`, its samples
//! from byte `offset` of the coverage bytes over the inclusive rectangle
//! `(x0, y0)–(x1, y1)` row by row, two bytes each, low byte first, when
//! `deep`, and `outside` the value of every pixel outside it (an empty
//! rectangle has `x1 < x0`); an object `[plane word, x0, y0, x1, y1]`.

use super::{Counts, PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds, index};

/// Words of a plane's record.
const PLANE_WORDS: usize = 7;

/// Words of an object's record.
const OBJECT_WORDS: usize = 5;

/// A soft mask's values: one sample per page pixel, row-major.
#[derive(Clone, Copy, Debug)]
pub enum PlaneValues<'a> {
    /// The values of a mask on an 8-bit page.
    Eight(&'a [u8]),
    /// The values of a mask on a 16-bit page.
    Sixteen(&'a [u16]),
}

/// A mask plane a page holds ([`PageCommands::add_mask_plane`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneId {
    word: u32,
}

/// The rectangle of the samples of a `width`-wide plane that differ from
/// `values[0]`, inclusive, `None` when none does.
fn differing<T: Copy + PartialEq>(
    values: &[T],
    width: usize,
) -> Option<(usize, usize, usize, usize)> {
    let first = *values.first()?;
    let mut rect: Option<(usize, usize, usize, usize)> = None;
    for (y, row) in values.chunks_exact(width.max(1)).enumerate() {
        let Some(a) = row.iter().position(|&v| v != first) else {
            continue;
        };
        let b = row.iter().rposition(|&v| v != first).unwrap_or(a);
        rect = Some(rect.map_or((a, y, b, y), |(x0, y0, x1, _)| {
            (x0.min(a), y0, x1.max(b), y)
        }));
    }
    rect
}

impl PageCommands {
    /// Hold a soft mask's `values` as a plane.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When `values` is not of the page's depth or does not hold one value
    /// per page pixel.
    pub fn add_mask_plane(
        &mut self,
        values: PlaneValues<'_>,
    ) -> Result<PlaneId, PageCommandsError> {
        let (width, height) = (self.width as usize, self.height as usize);
        let (len, deep) = match values {
            PlaneValues::Eight(v) => (v.len(), false),
            PlaneValues::Sixteen(v) => (v.len(), true),
        };
        assert!(
            len == width * height && deep == (self.depth.max_sample() > 255),
            "add_mask_plane: {len} values of {} bits for a {width}×{height} page of {} bits",
            if deep { 16 } else { 8 },
            self.depth.bits()
        );
        let (rect, outside) = match values {
            PlaneValues::Eight(v) => (differing(v, width), v.first().map_or(0, |&s| u16::from(s))),
            PlaneValues::Sixteen(v) => (differing(v, width), v.first().copied().unwrap_or(0)),
        };
        let sample_bytes = if deep { 2 } else { 1 };
        let n_bytes = rect.map_or(0, |(x0, y0, x1, y1)| {
            (x1 - x0 + 1) * (y1 - y0 + 1) * sample_bytes
        });
        let pad = self.bytes.len() % sample_bytes;
        let more = Counts {
            coverage: pad + n_bytes,
            mod_words: PLANE_WORDS,
            ..Counts::ZERO
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let word = index(self.mod_words.len())?;
        index(self.mod_words.len() + PLANE_WORDS).map(drop)?;
        let offset = index(self.bytes.len() + pad)?;
        index(self.bytes.len() + pad + n_bytes).map(drop)?;
        self.bytes.try_reserve(pad + n_bytes)?;
        self.mod_words.try_reserve(PLANE_WORDS)?;
        self.bytes.resize(self.bytes.len() + pad, 0);
        if let Some((x0, y0, x1, y1)) = rect {
            for y in y0..=y1 {
                let row = y * width;
                match values {
                    PlaneValues::Eight(v) => self.bytes.extend_from_slice(&v[row + x0..=row + x1]),
                    PlaneValues::Sixteen(v) => self
                        .bytes
                        .extend(v[row + x0..=row + x1].iter().flat_map(|s| s.to_le_bytes())),
                }
            }
        }
        // The page's extent fits i32, so the rectangle's words do.
        let word_of = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let (x0, y0, x1, y1) = rect.map_or((1, 0, 0, 0), |(x0, y0, x1, y1)| {
            (word_of(x0), word_of(y0), word_of(x1), word_of(y1))
        });
        self.mod_words
            .extend([offset, u32::from(deep), x0, y0, x1, y1, u32::from(outside)]);
        Ok(PlaneId { word })
    }

    /// Start an object painted under the mask `plane` within the inclusive
    /// rectangle `rect`: the commands pushed until [`Self::end_object`]
    /// are blended back by the mask within `rect`.  A started object ends
    /// first.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    pub fn begin_object(
        &mut self,
        plane: PlaneId,
        rect: PixelBounds,
    ) -> Result<(), PageCommandsError> {
        self.end_object();
        let more = Counts {
            mod_words: OBJECT_WORDS,
            ..Counts::ZERO
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let word = index(self.mod_words.len())?;
        index(self.mod_words.len() + OBJECT_WORDS).map(drop)?;
        self.mod_words.try_reserve(OBJECT_WORDS)?;
        self.mod_words.extend([
            plane.word,
            rect.x0.cast_unsigned(),
            rect.y0.cast_unsigned(),
            rect.x1.cast_unsigned(),
            rect.y1.cast_unsigned(),
        ]);
        self.modifiers.object = Some(word);
        Ok(())
    }

    /// End the object started by [`Self::begin_object`], if any.
    pub const fn end_object(&mut self) {
        self.modifiers.object = None;
    }

    /// The mask value at pixel `(x, y)` of the object whose record is at
    /// modifier word `object`, or `None` outside the object's rectangle.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "plane rectangles lie on the page, whose extent fits i32"
    )]
    pub(super) fn object_mask(&self, object: u32, x: i32, y: i32) -> Option<u16> {
        let rec = &self.mod_words[object as usize..][..OBJECT_WORDS];
        let [rx0, ry0, rx1, ry1] = [rec[1], rec[2], rec[3], rec[4]].map(u32::cast_signed);
        if !(rx0..=rx1).contains(&x) || !(ry0..=ry1).contains(&y) {
            return None;
        }
        let p = &self.mod_words[rec[0] as usize..][..PLANE_WORDS];
        let [x0, y0, x1, y1] = [p[2], p[3], p[4], p[5]].map(|v| v as i32);
        if !(x0..=x1).contains(&x) || !(y0..=y1).contains(&y) {
            return Some(u16::try_from(p[6]).unwrap_or(u16::MAX));
        }
        let i = ((y - y0) * (x1 - x0 + 1) + (x - x0)).unsigned_abs() as usize;
        let at = p[0] as usize;
        Some(if p[1] == 0 {
            u16::from(self.bytes[at + i])
        } else {
            u16::from_le_bytes([self.bytes[at + 2 * i], self.bytes[at + 2 * i + 1]])
        })
    }
}

#[cfg(test)]
mod tests;
