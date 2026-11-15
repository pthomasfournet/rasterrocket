//! Setting the fill and stroke colours of the graphics state, and the
//! rendering intent they are converted under.
//!
//! A colour is converted when it is set (see
//! [`PageRenderer::color_for_components`]), under the intent in effect.
//! A colour whose conversion depends on the intent keeps its components,
//! and an intent change that selects another conversion of its space
//! converts it again, so a painting operator always paints its colour as
//! converted under the intent in effect at that operator (PDF §8.6.5.8).

use std::sync::Arc;

use super::PageRenderer;
use crate::ColourIntent;
use crate::renderer::gstate::{ColourMut, InterpGState};
use crate::resources::ColorSpace;

/// The fill or the stroke colour of a graphics state.
type Select = for<'a> fn(&'a mut InterpGState) -> ColourMut<'a>;

impl PageRenderer<'_> {
    /// `g`, `rg`, `k`: select the device space `device`, or the default
    /// space that replaces it (PDF §8.6.5.6), for fill operations and set
    /// the fill colour to `comps` in that space (PDF §8.6.8).
    pub(super) fn set_fill_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        self.set_fill_colour(space, comps);
    }

    /// `G`, `RG`, `K`: the stroking counterpart of [`Self::set_fill_device`].
    pub(super) fn set_stroke_device(&mut self, device: ColorSpace, comps: &[f64]) {
        let space = self.resources.device_space(device);
        self.set_stroke_colour(space, comps);
    }

    /// Set the fill colour space to `space` and the fill colour to `comps`
    /// in it, clearing any fill pattern.
    pub(super) fn set_fill_colour(&mut self, space: ColorSpace, comps: &[f64]) {
        self.set_colour(InterpGState::fill_mut, space, comps);
    }

    /// The stroking counterpart of [`Self::set_fill_colour`].
    pub(super) fn set_stroke_colour(&mut self, space: ColorSpace, comps: &[f64]) {
        self.set_colour(InterpGState::stroke_mut, space, comps);
    }

    fn set_colour(&mut self, select: Select, space: ColorSpace, comps: &[f64]) {
        let colour = self.color_for_components(&space, comps);
        let components = space.depends_on_intent().then(|| Arc::from(comps));
        let slot = select(self.gstate.current_mut());
        *slot.colour = colour;
        *slot.pattern = None;
        slot.pattern_components.clear();
        *slot.space = space;
        *slot.components = components;
    }

    /// `ri`, `gs /RI /UseBlackPtComp`: put `intent` into effect.  The fill
    /// and stroke colours whose spaces convert differently under `intent`
    /// than under the intent replaced convert again from their components,
    /// unless a `d1` glyph description has locked them.
    pub(super) fn set_colour_intent(&mut self, intent: ColourIntent) {
        let gs = self.gstate.current_mut();
        let replaced = std::mem::replace(&mut gs.colour_intent, intent);
        if replaced == intent || gs.colour_locked {
            return;
        }
        let doc = self.resources.doc();
        for select in [InterpGState::fill_mut as Select, InterpGState::stroke_mut] {
            let slot = select(self.gstate.current_mut());
            let Some(comps) = slot.components.clone() else {
                continue;
            };
            if slot.space.converts_alike(doc, replaced, intent) {
                continue;
            }
            let space = slot.space.clone();
            let colour = self.color_for_components(&space, &comps);
            *select(self.gstate.current_mut()).colour = colour;
        }
    }
}
