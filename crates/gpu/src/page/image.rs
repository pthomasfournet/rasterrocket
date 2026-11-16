//! Images: commands that sample an image's samples per pixel through the
//! separable fixed-point tables the CPU's image sampler builds, and
//! composite the sampled colour over the page.
//!
//! A pixel `(x, y)` of a draw's bounds takes its source coordinate from
//! the column entry of `x` plus the row entry of `y` of its tables, each
//! entry `[x_hi, x_lo, y_hi, y_lo]`: an integer part and a 2^-32 fraction
//! per axis, summed with the carry of the fractions.  A pixel whose
//! coordinate lies off the image paints nothing.  Along an interpolated
//! axis the taps are the two samples whose centres bracket the coordinate
//! (the coordinate less half a sample, floored, and its successor, each
//! clamped onto the axis), weighted by the top 8 bits of that coordinate's
//! fraction; along any other axis the one sample.  The taps blend each
//! channel as `((p00·(256−wx) + p10·wx)·(256−wy) + (p01·(256−wx) +
//! p11·wx)·wy + 2^15) >> 16`, at the depth of the samples, and an 8-bit
//! result on a 16-bit page is widened by 257.
//!
//! A draw's alpha (its shape) is, at the page's depth `M = 2^bits − 1`:
//! `M` for an image without a mask; its mask's sampled value, the mask
//! read through the image's own taps or through tables of its own, whose
//! pixel is first clamped onto its tables' rectangle and whose taps are
//! clamped onto the mask; or, for a stencil, `M` less its sampled value.
//! A shape of 0 leaves the pixel unchanged.  The colour is the sampled RGB,
//! the sampled gray in every channel, or a stencil's colour.  A draw
//! composites either directly (`direct`: a shape of `M` writes the colour,
//! any other `s` gives `(c·s + d·(M − s)) / M`, truncated) or as the
//! antialiasing pipe (`a = div_max(a_input·s)`, `div_max((M − a)·d +
//! a·c)`).

use super::{
    Counts, KIND_IMAGE, PAGE_COMMANDS_MAX_BYTES, PageCmd, PageCommands, PageCommandsError,
    PixelBounds, index, pack_rgb,
};

/// `u32` words per sampling table entry.
pub const IMAGE_TABLE_ENTRY_WORDS: usize = 4;

/// Words of one sampler in an image record: offset, width, height,
/// flags, the tables' rectangle `x0, y0, x1, y1`, and the word index of
/// the column and of the row table.
pub(crate) const SAMPLER_WORDS: usize = 10;

/// Words of an image record: the image's sampler and its mask's.
pub(crate) const IMAGE_RECORD_WORDS: usize = 2 * SAMPLER_WORDS;

/// Sampler flag: samples of 16 bits.
pub(crate) const SAMPLES_DEEP: u32 = 1;
/// Sampler flag: interpolate between columns.
pub(crate) const FILTER_X: u32 = 2;
/// Sampler flag: interpolate between rows.
pub(crate) const FILTER_Y: u32 = 4;

/// Command flags: the colour layout (bits 0–1).
pub(crate) const LAYOUT_MASK: u32 = 3;
/// Layout: one gray sample per pixel.
pub(crate) const LAYOUT_GRAY: u32 = 0;
/// Layout: three samples per pixel, RGB.
pub(crate) const LAYOUT_RGB: u32 = 1;
/// Layout: a stencil's one sample per pixel.
pub(crate) const LAYOUT_STENCIL: u32 = 2;
/// Command flag: the direct composite.
pub(crate) const DIRECT: u32 = 4;
/// Command flags: the mask (bits 4–5): none, through the image's taps, or
/// through its own tables.
pub(crate) const MASK_SHIFT: u32 = 4;
pub(crate) const MASK_NONE: u32 = 0;
pub(crate) const MASK_TAPS: u32 = 1;
pub(crate) const MASK_OWN: u32 = 2;

/// Samples a page holds ([`PageCommands::add_samples`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplesId {
    /// Offset of the first byte in the page's coverage bytes.
    offset: u32,
    width: u32,
    height: u32,
    channels: u32,
    deep: bool,
}

impl SamplesId {
    /// Width in samples.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height in samples.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }
}

/// An image's samples, row-major, `channels` per pixel.
#[derive(Clone, Copy, Debug)]
pub enum Samples<'a> {
    /// 8-bit samples.
    Eight(&'a [u8]),
    /// 16-bit samples.
    Sixteen(&'a [u16]),
}

/// How an image's samples colour its pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// One gray sample per pixel, painted in every channel.
    Gray,
    /// Three samples per pixel.
    Rgb,
    /// One sample per pixel whose complement is the shape of the draw's
    /// colour.
    Stencil,
}

/// The sampling tables of a draw over the rectangle `x0..x1 × y0..y1`
/// (exclusive ends) of page pixels: one entry of
/// [`IMAGE_TABLE_ENTRY_WORDS`] words per column in `cols` and per row in
/// `rows`.
#[derive(Clone, Copy, Debug)]
pub struct SampleMap<'a> {
    /// Leftmost column.
    pub x0: i32,
    /// Top row.
    pub y0: i32,
    /// Column past the right edge.
    pub x1: i32,
    /// Row past the bottom edge.
    pub y1: i32,
    /// Column entries.
    pub cols: &'a [u32],
    /// Row entries.
    pub rows: &'a [u32],
    /// Interpolate between columns.
    pub filter_x: bool,
    /// Interpolate between rows.
    pub filter_y: bool,
}

impl SampleMap<'_> {
    /// Whether the tables hold one entry per column and row of a non-empty
    /// rectangle.
    fn is_well_formed(&self) -> bool {
        let extent = |a: i32, b: i32| usize::try_from(i64::from(b) - i64::from(a)).ok();
        self.x0 < self.x1
            && self.y0 < self.y1
            && extent(self.x0, self.x1)
                .is_some_and(|n| self.cols.len() == n * IMAGE_TABLE_ENTRY_WORDS)
            && extent(self.y0, self.y1)
                .is_some_and(|n| self.rows.len() == n * IMAGE_TABLE_ENTRY_WORDS)
    }
}

/// The mask of an image draw.
#[derive(Clone, Copy, Debug)]
pub enum ImageMask<'a> {
    /// No mask: the shape is full.
    None,
    /// One sample per pixel of the image's dimensions, read through the
    /// image's taps.
    Taps(SamplesId),
    /// One sample per pixel read through tables of its own.
    Own(SamplesId, SampleMap<'a>),
}

/// One draw of an image ([`PageCommands::push_image`]).
#[derive(Clone, Copy, Debug)]
pub struct ImageDraw<'a> {
    /// The image's samples.
    pub samples: SamplesId,
    /// Their layout.
    pub layout: ImageLayout,
    /// Their sampling tables.
    pub map: SampleMap<'a>,
    /// The draw's mask.
    pub mask: ImageMask<'a>,
    /// The pixels the draw may paint; it paints none outside its tables'
    /// rectangle or the page.
    pub bounds: PixelBounds,
    /// A stencil's colour.
    pub colour: [u16; 3],
    /// The constant alpha of the pipe composite.
    pub a_input: u16,
    /// Composite directly instead of through the pipe.
    pub direct: bool,
}

impl PageCommands {
    /// Hold `samples`, `width × height` pixels of `channels` samples each,
    /// for [`Self::push_image`]; 16-bit samples are held as their bytes,
    /// low byte first.
    ///
    /// # Errors
    ///
    /// As [`Self::add_mask`].
    ///
    /// # Panics
    ///
    /// When `samples` holds fewer than `width × height × channels`
    /// samples, or `channels` is neither 1 nor 3.
    pub fn add_samples(
        &mut self,
        samples: Samples<'_>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<SamplesId, PageCommandsError> {
        assert!(
            channels == 1 || channels == 3,
            "add_samples: {channels} channels"
        );
        let n = usize::try_from(u64::from(width) * u64::from(height) * u64::from(channels))
            .map_err(|_| PageCommandsError::TooLarge)?;
        let (len, deep) = match samples {
            Samples::Eight(s) => (s.len(), false),
            Samples::Sixteen(s) => (s.len(), true),
        };
        assert!(len >= n, "add_samples: {len} samples for {n}");
        let n_bytes = if deep { n * 2 } else { n };
        let more = Counts {
            coverage: n_bytes,
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let offset = index(self.bytes.len())?;
        i32::try_from(self.bytes.len() + n_bytes)
            .map(drop)
            .map_err(|_| PageCommandsError::TooLarge)?;
        self.bytes.try_reserve(n_bytes)?;
        match samples {
            Samples::Eight(s) => self.bytes.extend_from_slice(&s[..n]),
            Samples::Sixteen(s) => self
                .bytes
                .extend(s[..n].iter().flat_map(|v| v.to_le_bytes())),
        }
        Ok(SamplesId {
            offset,
            width,
            height,
            channels,
            deep,
        })
    }

    /// Paint `draw` over the pixels of its bounds that lie in its tables'
    /// rectangle and on the page.  Nothing is pushed when none does.
    ///
    /// # Errors
    ///
    /// As [`Self::push_coverage`].
    ///
    /// # Panics
    ///
    /// When a table does not hold one entry per column and row of its
    /// rectangle, the samples' channels do not match the layout (a mask:
    /// one), 16-bit samples are drawn on an 8-bit page, or a sample of
    /// `colour` or `a_input` exceeds the page's depth.
    pub fn push_image(&mut self, draw: &ImageDraw<'_>) -> Result<(), PageCommandsError> {
        let max = self.depth.max_sample();
        assert!(
            draw.colour.iter().all(|&v| v <= max) && draw.a_input <= max,
            "push_image: colour {:?} or alpha {} exceeds {} bits",
            draw.colour,
            draw.a_input,
            self.depth.bits()
        );
        let wide_page = max > 255;
        let channels = match draw.layout {
            ImageLayout::Rgb => 3,
            ImageLayout::Gray | ImageLayout::Stencil => 1,
        };
        assert!(
            draw.samples.channels == channels && draw.map.is_well_formed(),
            "push_image: malformed image"
        );
        assert!(
            wide_page || !draw.samples.deep,
            "push_image: 16-bit samples on an 8-bit page"
        );
        let (mask_mode, mask) = mask_parts(draw, wide_page);
        let rect = PixelBounds {
            x0: draw.map.x0,
            y0: draw.map.y0,
            x1: draw.map.x1 - 1,
            y1: draw.map.y1 - 1,
        };
        let Some(bounds) = self
            .paint_bounds()?
            .and_then(|p| p.intersect(draw.bounds))
            .and_then(|b| b.intersect(rect))
        else {
            return Ok(());
        };
        let tables = |map: &SampleMap<'_>| map.cols.len() + map.rows.len();
        let n_words = IMAGE_RECORD_WORDS
            + tables(&draw.map)
            + mask.and_then(|(_, map)| map).map_or(0, |map| tables(&map));
        let more = Counts {
            cmds: 1,
            tile_refs: bounds.tiles(),
            image_words: n_words,
            ..Counts::default()
        };
        if self.bytes_with(more) > PAGE_COMMANDS_MAX_BYTES {
            return Err(PageCommandsError::TooLarge);
        }
        let record = index(self.image_words.len())?;
        index(self.image_words.len() + n_words).map(drop)?;
        self.cmds.try_reserve(1)?;
        self.image_words.try_reserve(n_words)?;
        self.reserve_live()?;
        let base = self.image_words.len();
        self.image_words.resize(base + IMAGE_RECORD_WORDS, 0);
        let image_sampler = self.push_tables(draw.samples, &draw.map);
        self.image_words[base..base + SAMPLER_WORDS].copy_from_slice(&image_sampler);
        if let Some((m, map)) = mask {
            let sampler = map.map_or_else(
                || sampler_words(m, None, [0, 0]),
                |map| self.push_tables(m, &map),
            );
            self.image_words[base + SAMPLER_WORDS..base + IMAGE_RECORD_WORDS]
                .copy_from_slice(&sampler);
        }
        self.count_rect(bounds);
        let layout = match draw.layout {
            ImageLayout::Gray => LAYOUT_GRAY,
            ImageLayout::Rgb => LAYOUT_RGB,
            ImageLayout::Stencil => LAYOUT_STENCIL,
        };
        let [rg, b] = pack_rgb(draw.colour);
        self.cmds.push(PageCmd {
            kind: KIND_IMAGE,
            flags: layout | if draw.direct { DIRECT } else { 0 } | mask_mode << MASK_SHIFT,
            rg,
            b,
            a_input: u32::from(draw.a_input),
            row_base: record,
            x0: bounds.x0,
            y0: bounds.y0,
            x1: bounds.x1,
            y1: bounds.y1,
            ..self.modifiers.stamp()
        });
        Ok(())
    }

    /// Append `map`'s tables to the image words, reserved beforehand, and
    /// return the sampler words of `samples` read through them.
    fn push_tables(&mut self, samples: SamplesId, map: &SampleMap<'_>) -> [u32; SAMPLER_WORDS] {
        let at = |v: usize| u32::try_from(v).expect("the push limit bounds the words");
        let cols = at(self.image_words.len());
        self.image_words.extend_from_slice(map.cols);
        let rows = at(self.image_words.len());
        self.image_words.extend_from_slice(map.rows);
        sampler_words(samples, Some(map), [cols, rows])
    }
}

/// The mask mode of `draw` and its mask's samples with their own tables,
/// if any, its mask checked as [`PageCommands::push_image`] documents for
/// a page of 16 bits when `wide_page`.
fn mask_parts<'a>(
    draw: &ImageDraw<'a>,
    wide_page: bool,
) -> (u32, Option<(SamplesId, Option<SampleMap<'a>>)>) {
    let (mode, mask) = match draw.mask {
        ImageMask::None => return (MASK_NONE, None),
        ImageMask::Taps(m) => {
            assert!(
                m.width == draw.samples.width && m.height == draw.samples.height,
                "push_image: a mask read through the image's taps has its dimensions"
            );
            (MASK_TAPS, (m, None))
        }
        ImageMask::Own(m, map) => {
            assert!(map.is_well_formed(), "push_image: malformed mask tables");
            (MASK_OWN, (m, Some(map)))
        }
    };
    assert!(
        mask.0.channels == 1 && (wide_page || !mask.0.deep),
        "push_image: malformed mask"
    );
    (mode, Some(mask))
}

/// The sampler words of `samples` read through `map`, whose tables start at
/// the image words `tables`.
fn sampler_words(
    samples: SamplesId,
    map: Option<&SampleMap<'_>>,
    tables: [u32; 2],
) -> [u32; SAMPLER_WORDS] {
    let filters = map.map_or(0, |m| {
        (u32::from(m.filter_x) * FILTER_X) | (u32::from(m.filter_y) * FILTER_Y)
    });
    let flags = (u32::from(samples.deep) * SAMPLES_DEEP) | filters;
    let rect = map.map_or([0; 4], |m| [m.x0, m.y0, m.x1, m.y1].map(i32::cast_unsigned));
    [
        samples.offset,
        samples.width,
        samples.height,
        flags,
        rect[0],
        rect[1],
        rect[2],
        rect[3],
        tables[0],
        tables[1],
    ]
}

/// The taps of one pixel: columns, rows, and the weights of the second of
/// each, out of 256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Taps {
    x: [u32; 2],
    y: [u32; 2],
    wx: u32,
    wy: u32,
}

/// Taps along an axis of `n` samples at integer part `i` and fraction
/// `frac`, `i` clamped onto the axis.
#[expect(
    clippy::cast_sign_loss,
    reason = "every index is clamped to be non-negative first"
)]
const fn axis_taps(i: i32, frac: u32, n: u32, blend: bool) -> ([u32; 2], u32) {
    const HALF: u32 = 1 << 31;
    let last = n - 1;
    if i < 0 {
        return ([0, 0], 0);
    }
    if i as u32 > last {
        return ([last, last], 0);
    }
    if !blend {
        return ([i as u32, i as u32], 0);
    }
    let lower = i - if frac < HALF { 1 } else { 0 };
    let weight = frac.wrapping_sub(HALF) >> 24;
    let lo = if lower < 0 { 0 } else { lower as u32 };
    let hi = if (lower + 1) as u32 > last {
        last
    } else {
        (lower + 1) as u32
    };
    ([lo, hi], if lo == hi { 0 } else { weight })
}

/// Host model of the kernel's image sampling: the words of one page's
/// image records and tables, and its coverage bytes.
pub(crate) struct ImageModel<'a> {
    pub(crate) words: &'a [u32],
    pub(crate) bytes: &'a [u8],
    /// Bits per page sample.
    pub(crate) bits: u32,
}

impl ImageModel<'_> {
    /// The taps of pixel `(x, y)` through the sampler at word `s`; `None`
    /// when its coordinate lies off the samples and `clamp` is not set.
    /// With `clamp`, the pixel is first clamped onto the tables'
    /// rectangle and the taps onto the samples.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "two's-complement words by construction; the pixel lies in the rectangle"
    )]
    fn taps(&self, sampler: usize, x: i32, y: i32, clamp: bool) -> Option<Taps> {
        let word = |i: usize| self.words[sampler + i];
        let (x0, y0, x1, y1) = (
            word(4) as i32,
            word(5) as i32,
            word(6) as i32,
            word(7) as i32,
        );
        let (x, y) = if clamp {
            (x.clamp(x0, x1 - 1), y.clamp(y0, y1 - 1))
        } else {
            (x, y)
        };
        let col = word(8) as usize + (x - x0) as usize * IMAGE_TABLE_ENTRY_WORDS;
        let row = word(9) as usize + (y - y0) as usize * IMAGE_TABLE_ENTRY_WORDS;
        let at = |i: usize| self.words[i];
        let (fx, cx) = at(col + 1).overflowing_add(at(row + 1));
        let ix = (at(col) as i32)
            .wrapping_add(at(row) as i32)
            .wrapping_add(i32::from(cx));
        let (fy, cy) = at(col + 3).overflowing_add(at(row + 3));
        let iy = (at(col + 2) as i32)
            .wrapping_add(at(row + 2) as i32)
            .wrapping_add(i32::from(cy));
        let (width, height) = (word(1), word(2));
        let on = |i: i32, n: u32| u32::try_from(i).is_ok_and(|i| i < n);
        if !clamp && (!on(ix, width) || !on(iy, height)) {
            return None;
        }
        let (xs, wx) = axis_taps(ix, fx, width, word(3) & FILTER_X != 0);
        let (ys, wy) = axis_taps(iy, fy, height, word(3) & FILTER_Y != 0);
        Some(Taps {
            x: xs,
            y: ys,
            wx,
            wy,
        })
    }

    /// Channel `ch` of the sampler at word `s`, `channels` per pixel,
    /// blended over `taps` and taken to the page's depth.
    fn blend(&self, s: usize, taps: Taps, ch: u32, channels: u32) -> u32 {
        let w = |i: usize| self.words[s + i];
        let deep = w(3) & SAMPLES_DEEP != 0;
        let fetch = |x: u32, y: u32| {
            let i = ((y * w(1) + x) * channels + ch) as usize;
            let at = w(0) as usize;
            if deep {
                u32::from(self.bytes[at + 2 * i]) | u32::from(self.bytes[at + 2 * i + 1]) << 8
            } else {
                u32::from(self.bytes[at + i])
            }
        };
        let v = if taps.wx == 0 && taps.wy == 0 {
            fetch(taps.x[0], taps.y[0])
        } else {
            let row = |y| fetch(taps.x[0], y) * (256 - taps.wx) + fetch(taps.x[1], y) * taps.wx;
            (row(taps.y[0]) * (256 - taps.wy) + row(taps.y[1]) * taps.wy + 0x8000) >> 16
        };
        if self.bits == 16 && !deep { v * 257 } else { v }
    }

    /// The colour and shape of pixel `(x, y)`, inside its bounds, painted
    /// by the image command `c`; `None` where it paints nothing.
    pub(crate) fn pixel(&self, cmd: &PageCmd, x: i32, y: i32) -> Option<([u32; 3], u32)> {
        let max = (1u32 << self.bits) - 1;
        let image = cmd.row_base as usize;
        let mask = image + SAMPLER_WORDS;
        let taps = self.taps(image, x, y, false)?;
        let layout = cmd.flags & LAYOUT_MASK;
        let shape = match (layout, (cmd.flags >> MASK_SHIFT) & 3) {
            (LAYOUT_STENCIL, _) => max - self.blend(image, taps, 0, 1),
            (_, MASK_TAPS) => self.blend(mask, taps, 0, 1),
            (_, MASK_OWN) => self.blend(mask, self.taps(mask, x, y, true)?, 0, 1),
            _ => max,
        };
        if shape == 0 {
            return None;
        }
        let colour = match layout {
            LAYOUT_RGB => [0, 1, 2].map(|ch| self.blend(image, taps, ch, 3)),
            LAYOUT_GRAY => [self.blend(image, taps, 0, 1); 3],
            _ => [cmd.rg & 0xFFFF, cmd.rg >> 16, cmd.b],
        };
        Some((colour, shape))
    }
}

#[cfg(test)]
mod tests;
