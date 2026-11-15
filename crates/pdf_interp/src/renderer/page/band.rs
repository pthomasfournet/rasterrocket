//! A page rendered in windows (bands) by successive renderers that share
//! the page's decoded and resolved resources.
//!
//! Each window is drawn by a renderer of its own over the page's whole
//! content stream; a window receives the device pixels the whole-page
//! render places there (see [`super::view`]).  [`PageState`] carries from
//! one renderer to the next what the page has decoded, parsed and resolved
//! (images, forms, shadings, tiling cells), the tint-transform budget, the
//! spot colorants' slots of an ink canvas, and the diagnostics, so the
//! renderers of a page decode each image once, assign spot slots in the
//! order the whole page does, and report one set of diagnostics.

use std::rc::Rc;

use super::{
    CanvasBuffers, FinishedCanvas, PageDiagnostics, PageRenderer, SharedInkMap, form_memo,
    image_memo, pattern_memo,
};
use crate::resources::IMAGE_FILTER_COUNT;
use crate::resources::image::colorspace::TintBudget;

/// The state the renderers of one page's windows share.
///
/// Taken from a renderer by [`PageRenderer::try_finish_band`] and given to
/// the renderer of the next window by [`PageRenderer::adopt_page_state`].
/// Its memos are reference-counted for one thread: the windows of a page
/// are rendered one after another.
pub struct PageState {
    image_memo: image_memo::SharedImageMemo,
    tint_budget: Rc<TintBudget>,
    pattern_memo: pattern_memo::SharedPatternMemo,
    form_memo: form_memo::SharedFormMemo,
    ink_map: SharedInkMap,
    /// The diagnostics of the windows drawn so far.
    diag: PageDiagnostics,
    /// The image draws of the windows drawn so far, by filter.
    filter_counts: [u32; IMAGE_FILTER_COUNT],
}

impl PageState {
    /// The diagnostics of every window drawn with this state: an image
    /// failure or missing font reported once, the flags of any window, the
    /// largest source-resolution hint, device counters summed, the glyph
    /// counts of the first window, and the dominant filter over the image
    /// draws of all windows.
    #[must_use]
    pub fn diagnostics(&self) -> PageDiagnostics {
        let mut diag = self.diag.clone();
        diag.dominant_filter = crate::resources::image::dominant_filter(&self.filter_counts);
        diag.unslotted_spots = self.ink_map.borrow().unslotted().to_vec();
        diag
    }
    /// The rendering intents of the marking operators executed so far
    /// (see [`PageDiagnostics::content_intents`]): every window executes
    /// the whole content stream, so the first window's set is the page's.
    #[must_use]
    pub const fn content_intents(&self) -> crate::IntentSet {
        self.diag.content_intents
    }
}

impl PageRenderer<'_> {
    /// Continue the page whose earlier window `state` comes from: draw from
    /// its memos, spend its tint budget, slot spot colorants in its ink
    /// map and add to its diagnostics, whose glyph counts, the first
    /// window's, are the page's.  Call before execute, after
    /// [`set_document_images`](Self::set_document_images), whose store the
    /// state's image memo already holds.
    pub fn adopt_page_state(&mut self, state: PageState) {
        let PageState {
            image_memo,
            tint_budget,
            pattern_memo,
            form_memo,
            ink_map,
            diag,
            filter_counts,
        } = state;
        self.image_memo = image_memo;
        self.tint_budget = tint_budget;
        self.pattern_memo = pattern_memo;
        self.form_memo = form_memo;
        self.ink_map = ink_map;
        self.diag = diag;
        self.filter_counts = filter_counts;
        self.census = super::text_census::Census::Uncounted;
    }

    /// Finish the window this renderer drew, as
    /// [`try_finish_keeping`](Self::try_finish_keeping) does, and return
    /// the page's state for the renderer of the next window.  The page's
    /// diagnostics are the state's ([`PageState::diagnostics`]) once the
    /// last window is finished.
    ///
    /// # Errors
    ///
    /// The allocator refused the window's RGB bitmap.
    pub fn try_finish_band(
        mut self,
        transparent: bool,
    ) -> Result<(FinishedCanvas, PageState, CanvasBuffers), raster::BitmapAllocError> {
        let state = PageState {
            image_memo: Rc::clone(&self.image_memo),
            tint_budget: Rc::clone(&self.tint_budget),
            pattern_memo: Rc::clone(&self.pattern_memo),
            form_memo: Rc::clone(&self.form_memo),
            ink_map: Rc::clone(&self.ink_map),
            diag: std::mem::take(&mut self.diag),
            filter_counts: self.filter_counts,
        };
        let cmyk = self.ink.cmyk;
        let (canvas, mut buffers, ink_map, _) = self.into_parts(!transparent);
        let mut finished =
            canvas.try_into_finished_in(&ink_map.borrow().luts(), cmyk, &mut buffers)?;
        if transparent {
            finished.zero_transparent_colour();
        }
        Ok((finished, state, buffers))
    }
}
