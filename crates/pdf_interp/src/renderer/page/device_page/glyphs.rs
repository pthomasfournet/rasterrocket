//! Glyphs recorded for the whole-page GPU raster.
//!
//! A glyph the CPU composites from a coverage bitmap (a cached glyph, a
//! cached glyph stroke, a cached Type 3 glyph) is recorded as a mask
//! command: the bitmap is held once per page as a [`gpu::page`] mask, its
//! bytes the bitmap's own, and each glyph places it where
//! `raster::glyph::fill_glyph` places it, within the clip rectangle's
//! pixels, in the colour and constant alpha the CPU pipe composites with.
//! The device reads the coverage `fill_glyph` reads and composites it with
//! the CPU's arithmetic.
//!
//! A clip that is a rectangle passes, in `fill_glyph`, every pixel of its
//! integer bounds fully and no other: its row coverage is 255 or 0 by
//! column, and a glyph byte scaled by 255 is itself.  A clip with path
//! regions scales the glyph's coverage on the device by the clip's
//! coverage, as `fill_glyph` does.

use std::collections::HashMap;
use std::sync::Arc;

use font::bitmap::GlyphBitmap;
use gpu::page::{MaskBits, MaskId, PageCommandsError, PixelBounds};

use super::super::PageRenderer;
use super::Recording;

/// The masks a page's recording holds, by the bitmap each copies.  The
/// bitmaps are kept alive with the recording, so no address names two
/// bitmaps.
#[derive(Default)]
pub(super) struct HeldMasks {
    by_bitmap: HashMap<*const GlyphBitmap, MaskId>,
    bitmaps: Vec<Arc<GlyphBitmap>>,
}

/// One glyph as `fill_glyph` blits it: its pen pixel and its bitmap.
pub(in crate::renderer::page) struct PlacedGlyph<'a> {
    pub pen_x: i32,
    pub pen_y: i32,
    pub bitmap: &'a Arc<GlyphBitmap>,
}

impl Recording {
    /// The mask copying `bitmap`, held on first use.
    fn mask_of(&mut self, bitmap: &Arc<GlyphBitmap>) -> Result<MaskId, PageCommandsError> {
        if let Some(&mask) = self.masks.by_bitmap.get(&Arc::as_ptr(bitmap)) {
            return Ok(mask);
        }
        let bits = if bitmap.aa {
            MaskBits::Coverage
        } else {
            MaskBits::Mono
        };
        let mask = self
            .commands
            .add_mask(&bitmap.data, bitmap.width, bitmap.height, bits)?;
        self.masks.bitmaps.try_reserve(1)?;
        self.masks.by_bitmap.try_reserve(1)?;
        self.masks.bitmaps.push(Arc::clone(bitmap));
        let _ = self.masks.by_bitmap.insert(Arc::as_ptr(bitmap), mask);
        Ok(mask)
    }

    /// Append `glyphs` in `rgb` at alpha `alpha`, samples of the page's
    /// depth, within `clip`'s pixels, in order.
    fn push_glyphs<'a>(
        &mut self,
        glyphs: impl IntoIterator<Item = PlacedGlyph<'a>>,
        clip: PixelBounds,
        rgb: [u16; 3],
        alpha: u16,
    ) -> Result<(), PageCommandsError> {
        for glyph in glyphs {
            let mask = self.mask_of(glyph.bitmap)?;
            let origin = (
                glyph.pen_x.saturating_sub(glyph.bitmap.x_off),
                glyph.pen_y.saturating_sub(glyph.bitmap.y_off),
            );
            self.commands.push_mask(mask, origin, clip, rgb, alpha)?;
        }
        Ok(())
    }
}

impl PageRenderer<'_> {
    /// Record the glyphs that `fill_glyph` would composite in the solid
    /// stroking (`stroke`) or non-stroking colour at constant alpha
    /// `alpha` through the clip in effect.
    pub(in crate::renderer::page) fn record_device_glyphs<'a>(
        &mut self,
        glyphs: impl IntoIterator<Item = PlacedGlyph<'a>>,
        stroke: bool,
        alpha: u8,
    ) {
        let paint = match self.solid_device_paint_at(stroke, alpha) {
            Ok(paint) => paint,
            Err(reason) => return self.refuse_device(reason),
        };
        let clip = PixelBounds {
            x0: paint.clip.x_min_i,
            y0: paint.clip.y_min_i,
            x1: paint.clip.x_max_i,
            y1: paint.clip.y_max_i,
        };
        if let Some(recording) = self.device_page.as_mut()
            && recording.clip_to(&paint.clip)
        {
            let pushed = recording.push_glyphs(glyphs, clip, paint.rgb, paint.alpha);
            recording.refuse_on(pushed);
        }
    }
}
