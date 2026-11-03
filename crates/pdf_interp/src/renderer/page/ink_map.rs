//! The spot colorants of a page assigned to the spot channels of an ink
//! canvas.
//!
//! An ink pixel is `[C, M, Y, K, S0, S1, S2, S3]`: the four process
//! channels and four spot channels, or slots.  A `Separation` or `DeviceN`
//! colour paints each of its colorants in that colorant's channel — a
//! process name in its process channel, a spot name in the slot the page
//! assigns it.  The first four distinct spot names met on the page take
//! slots 0–3 in that order; a colour whose spots do not all fit the free
//! slots takes none and paints the process ink of its alternate colour
//! instead.  Each slot
//! carries a [`SpotLut`], the process equivalent of every tint of its spot,
//! through which the finished canvas converts to sRGB.

use std::sync::Arc;

use color::cmyk::SpotLut;
use color::convert::gray_to_u8;
use pdf::Document;
use raster::pipe::ChannelMask;

use super::super::color::PROCESS_CHANNELS;
use crate::resources::ColorSpace;
use crate::resources::colorspace::Colorant;

/// The number of spot slots of an ink canvas.
const SLOTS: usize = 4;

/// The channel of spot slot 0; slot `s` is channel `SPOT_CHANNEL + s`.
const SPOT_CHANNEL: usize = 4;

/// The spot colorants of a page, each with its slot and its process
/// equivalent.
#[derive(Default)]
pub(super) struct InkMap {
    /// The assigned slots, slot `s` at index `s`; at most [`SLOTS`].
    slots: Vec<Slot>,
    /// The distinct spot colorants that found no slot, in order of first
    /// sight.
    unslotted: Vec<String>,
}

/// A spot colorant holding a slot.
struct Slot {
    /// The colorant name.
    name: Arc<[u8]>,
    /// The process ink of each tint of the colorant.
    lut: Box<SpotLut>,
}

impl InkMap {
    /// The slot of the spot colorant `name` of `space`, assigning the next
    /// free slot and building its table on first sight; `None` when the
    /// page's slots are all held by other colorants.  The same name in
    /// several spaces shares one slot and the table of the space it was
    /// first met in.
    fn slot(&mut self, doc: &Document, name: &[u8], space: &ColorSpace) -> Option<usize> {
        if let Some(slot) = self.slots.iter().position(|s| *s.name == *name) {
            return Some(slot);
        }
        if self.slots.len() == SLOTS {
            return None;
        }
        self.slots.push(Slot {
            name: Arc::from(name),
            lut: Box::new(spot_lut(doc, name, space)),
        });
        Some(self.slots.len() - 1)
    }

    /// Whether every spot colorant of `space` holds a slot or can take a
    /// free one: the distinct spot names of `space` without a slot number
    /// at most the free slots.
    fn fits(&self, space: &ColorSpace) -> bool {
        let mut unseen: Vec<&[u8]> = Vec::new();
        for colorant in space.colorants() {
            if let Colorant::Spot(name) = colorant
                && !self.slots.iter().any(|s| *s.name == **name)
                && !unseen.contains(&&**name)
            {
                unseen.push(name);
            }
        }
        unseen.len() <= SLOTS - self.slots.len()
    }

    /// Record the spot colorants of `space` that hold no slot, which paint
    /// their alternate colour in process ink; the first is logged as a
    /// warning once per page.
    fn report_overflow(&mut self, space: &ColorSpace) {
        for colorant in space.colorants() {
            let Colorant::Spot(name) = colorant else {
                continue;
            };
            if self.slots.iter().any(|s| s.name == *name) {
                continue;
            }
            let name = String::from_utf8_lossy(name);
            if self.unslotted.iter().any(|n| *n == name) {
                continue;
            }
            if self.unslotted.is_empty() {
                log::warn!(
                    "rasterrocket-interp: more than {SLOTS} spot colorants on the page; \
                     {name:?} and later spots paint their alternate colour in process ink"
                );
            }
            self.unslotted.push(name.into_owned());
        }
    }

    /// The spot colorants that found no slot and painted their alternate
    /// colour in process ink, in order of first sight.
    pub(super) fn into_unslotted(self) -> Vec<String> {
        self.unslotted
    }

    /// The process-equivalent table of each slot, `None` for a free slot.
    pub(super) fn luts(&self) -> [Option<&SpotLut>; SLOTS] {
        std::array::from_fn(|s| self.slots.get(s).map(|slot| &*slot.lut))
    }

    /// The ink and the painted channels of the colour `comps` in the
    /// `Separation` or `DeviceN` space `space`: each colorant's tint in
    /// the colorant's channel — a repeated channel keeps the larger tint —
    /// `/All` in every channel, `/None` in none.  When the colour's spot
    /// colorants do not all fit the slots, none of them takes one: the
    /// colour is instead the process ink of its alternate colour and
    /// paints the process channels.
    pub(super) fn colorant_ink(
        &mut self,
        doc: &Document,
        space: &ColorSpace,
        comps: &[f64],
    ) -> ([u8; 8], ChannelMask) {
        let process_fallback = |map: &mut Self| {
            map.report_overflow(space);
            let [c, m, y, k] = space.process_ink(doc, comps);
            ([c, m, y, k, 0, 0, 0, 0], PROCESS_CHANNELS)
        };
        if !self.fits(space) {
            return process_fallback(self);
        }
        let mut ink = [0u8; 8];
        let mut bits = 0u8;
        for (colorant, &tint) in space.colorants().iter().zip(comps) {
            let tint = gray_to_u8(tint);
            let channel = match colorant {
                Colorant::None => continue,
                Colorant::All => {
                    for v in &mut ink {
                        *v = (*v).max(tint);
                    }
                    bits = ChannelMask::ALL.bits();
                    continue;
                }
                Colorant::Process(process) => process.channel(),
                Colorant::Spot(name) => {
                    // Every spot fits, so the slot is given.
                    let Some(slot) = self.slot(doc, name, space) else {
                        return process_fallback(self);
                    };
                    SPOT_CHANNEL + slot
                }
            };
            ink[channel] = ink[channel].max(tint);
            bits |= 1 << channel;
        }
        (ink, ChannelMask::new(bits))
    }
}

/// The process ink of each tint of the spot colorant `name` of `space`,
/// alone — every other component of `space` at 0 — through the space's
/// tint transform and alternate space ([`ColorSpace::process_ink`]).
/// `name` is a spot colorant of `space`; the first component converts
/// otherwise.
fn spot_lut(doc: &Document, name: &[u8], space: &ColorSpace) -> SpotLut {
    let component = space
        .colorants()
        .iter()
        .position(|c| matches!(c, Colorant::Spot(spot) if **spot == *name));
    debug_assert!(
        component.is_some(),
        "{:?} is not a spot colorant of {space:?}",
        String::from_utf8_lossy(name)
    );
    let component = component.unwrap_or(0);
    let mut comps = vec![0.0; space.ncomponents()];
    let mut lut = [[0; 4]; 256];
    for (tint, entry) in (0..=u8::MAX).zip(lut.iter_mut()) {
        if let Some(c) = comps.get_mut(component) {
            *c = f64::from(tint) / 255.0;
        }
        *entry = space.process_ink(doc, &comps);
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::resolve;
    use crate::test_helpers::empty_doc;

    /// The space declared by the PDF object `src`.
    fn space(doc: &Document, src: &str) -> ColorSpace {
        let obj = pdf::parse_object(src.as_bytes(), &mut 0).expect("colour-space object parses");
        resolve(doc, &obj)
    }

    /// `[/Separation /name /DeviceCMYK f]` with `f` linear from no ink to
    /// the inks `c1`.
    fn separation(doc: &Document, name: &str, c1: &str) -> ColorSpace {
        space(
            doc,
            &format!(
                "[/Separation /{name} /DeviceCMYK \
                 << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [{c1}] /N 1 >>]"
            ),
        )
    }

    #[test]
    fn ink_map_assigns_slots_in_order_of_first_use_and_shares_names() {
        let doc = empty_doc();
        let a = separation(&doc, "A", "1 0 0 0");
        let b = separation(&doc, "B", "0 1 0 0");
        let mut map = InkMap::default();
        assert_eq!(map.slot(&doc, b"A", &a), Some(0));
        assert_eq!(map.slot(&doc, b"B", &b), Some(1));
        assert_eq!(map.slot(&doc, b"A", &a), Some(0));
        // The same name in another space shares the slot.
        let dn = space(
            &doc,
            "[/DeviceN [/B /A] /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 0 1 0] /N 1 >>]",
        );
        assert_eq!(map.slot(&doc, b"A", &dn), Some(0));
        assert_eq!(map.slot(&doc, b"B", &dn), Some(1));
        let luts = map.luts();
        assert!(luts[0].is_some() && luts[1].is_some());
        assert!(luts[2].is_none() && luts[3].is_none());
    }

    #[test]
    fn ink_map_holds_four_slots() {
        let doc = empty_doc();
        let mut map = InkMap::default();
        for (i, name) in [b"S0", b"S1", b"S2", b"S3"].into_iter().enumerate() {
            let s = separation(&doc, std::str::from_utf8(name).unwrap(), "0 0 0 1");
            assert_eq!(map.slot(&doc, name, &s), Some(i));
        }
        let fifth = separation(&doc, "S4", "0 0 1 0");
        assert_eq!(map.slot(&doc, b"S4", &fifth), None);
        assert_eq!(map.slot(&doc, b"S4", &fifth), None);
        assert_eq!(map.slot(&doc, b"S2", &fifth), Some(2));
        assert!(map.luts().iter().all(Option::is_some));
    }

    #[test]
    fn spot_lut_is_the_process_ink_of_each_tint() {
        let doc = empty_doc();
        let lut = spot_lut(&doc, b"Teal", &separation(&doc, "Teal", "1 0 0.5 0"));
        assert_eq!(lut[0], [0, 0, 0, 0]);
        assert_eq!(lut[255], [255, 0, 128, 0]);
        assert_eq!(lut[51], [51, 0, 26, 0]);
        // A spot of a `DeviceN` space converts alone: the Type 4 transform
        // takes (k, s) to the inks (0, s, s, k), so black stays at 0.
        let function = "{ dup 3 -1 roll 0 4 1 roll }";
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            format!(
                "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
                 stream\n{function}\nendstream",
                function.len()
            ),
        )]);
        let dn = space(&doc, "[/DeviceN [/Black /Teal] /DeviceCMYK 4 0 R]");
        let lut = spot_lut(&doc, b"Teal", &dn);
        assert_eq!(lut[255], [0, 255, 255, 0]);
        assert_eq!(lut[0], [0, 0, 0, 0]);
        assert_eq!(lut[102], [0, 102, 102, 0]);
    }

    #[test]
    fn colorant_ink_places_each_tint_in_its_channel() {
        let doc = empty_doc();
        let mut map = InkMap::default();
        let cyan = separation(&doc, "Cyan", "1 0 0 0");
        assert_eq!(
            map.colorant_ink(&doc, &cyan, &[0.6]),
            ([153, 0, 0, 0, 0, 0, 0, 0], ChannelMask::new(0b0001))
        );
        let all = separation(&doc, "All", "1 1 1 1");
        assert_eq!(
            map.colorant_ink(&doc, &all, &[0.6]),
            ([153; 8], ChannelMask::ALL)
        );
        let none = separation(&doc, "None", "0 0 0 0");
        assert_eq!(
            map.colorant_ink(&doc, &none, &[0.6]),
            ([0; 8], ChannelMask::new(0))
        );
        let spot = separation(&doc, "Spot", "0 1 0 0");
        assert_eq!(
            map.colorant_ink(&doc, &spot, &[1.0]),
            ([0, 0, 0, 0, 255, 0, 0, 0], ChannelMask::new(0b1_0000))
        );
    }

    #[test]
    fn colorant_ink_without_a_slot_is_the_alternate_process_ink() {
        let doc = empty_doc();
        let mut map = InkMap::default();
        for (slot, name) in ["S0", "S1", "S2", "S3"].into_iter().enumerate() {
            let (_, paints) = map.colorant_ink(&doc, &separation(&doc, name, "0 0 0 1"), &[1.0]);
            assert_eq!(paints, ChannelMask::new(1 << (SPOT_CHANNEL + slot)));
        }
        let fifth = separation(&doc, "S4", "0 0.5 1 0");
        assert_eq!(
            map.colorant_ink(&doc, &fifth, &[1.0]),
            ([0, 128, 255, 0, 0, 0, 0, 0], PROCESS_CHANNELS)
        );
    }
    /// A `DeviceN` colour whose new spots outnumber the free slots takes
    /// none of them, so a later spot can still have the slot.
    #[test]
    fn colorant_ink_takes_no_slot_for_a_colour_that_does_not_fit() {
        let doc = empty_doc();
        let mut map = InkMap::default();
        for (slot, name) in ["S0", "S1", "S2"].into_iter().enumerate() {
            let taken = map.slot(&doc, name.as_bytes(), &separation(&doc, name, "0 0 0 1"));
            assert_eq!(taken, Some(slot));
        }
        let dn = space(
            &doc,
            "[/DeviceN [/New1 /New2] /DeviceCMYK \
             << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>]",
        );
        let (ink, paints) = map.colorant_ink(&doc, &dn, &[1.0, 1.0]);
        assert_eq!(paints, PROCESS_CHANNELS);
        assert_eq!(ink[4..], [0; 4]);
        assert!(map.luts()[3].is_none(), "the last slot stays free");
        let fourth = separation(&doc, "S3", "1 0 0 0");
        assert_eq!(map.slot(&doc, b"S3", &fourth), Some(3));
        // A colour whose spots are all seen fits whatever the free slots.
        let (ink, paints) = map.colorant_ink(&doc, &separation(&doc, "S1", "0 0 0 1"), &[0.5]);
        assert_eq!((ink[5], paints), (128, ChannelMask::new(0b10_0000)));
    }

    /// A tint transform that does not evaluate still gives a defined table.
    #[test]
    fn spot_lut_degrades_when_the_tint_transform_fails() {
        let doc = empty_doc();
        let broken = space(&doc, "[/Separation /Spot /DeviceCMYK 999 0 R]");
        let lut = spot_lut(&doc, b"Spot", &broken);
        assert_eq!(lut[0], [0; 4]);
        assert_eq!(lut[255][3], 255);
        assert!(
            lut.windows(2).all(|w| w[0][3] <= w[1][3]),
            "black ink grows with the tint"
        );
    }
}
