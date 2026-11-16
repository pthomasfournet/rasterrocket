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
//! through which the finished canvas converts to sRGB, and for a page of
//! 16-bit samples a [`SpotLut16`] of the same at 16 bits.

use std::collections::HashMap;
use std::sync::Arc;

use color::cmyk::{SpotLut, SpotLut16};
use color::{Sample, SampleDepth};
use pdf::Document;
use raster::pipe::ChannelMask;

use super::super::color::{ColorantPlan, PROCESS_CHANNELS, no_spots, paint_tint};
use crate::resources::ColorSpace;
use crate::resources::colorspace::Colorant;
use crate::separation::Separator;
use crate::{ColourIntent, ColourTarget};

/// The number of spot slots of an ink canvas.
const SLOTS: usize = 4;

/// The channel of spot slot 0; slot `s` is channel `SPOT_CHANNEL + s`.
const SPOT_CHANNEL: usize = 4;

/// The reproducing inks a page keeps at most; further colours are solved
/// each time they are met.
const REPRODUCED_MAX: usize = 4096;

/// The spot colorants of a page, each with its slot and its process
/// equivalent.
#[derive(Default)]
pub(super) struct InkMap {
    /// The depth of the page's samples: a slot's 16-bit table is built for
    /// a page of 16 bits.
    depth: SampleDepth,
    /// The assigned slots, slot `s` at index `s`; at most [`SLOTS`].
    slots: Vec<Slot>,
    /// The distinct spot colorants that found no slot, in order of first
    /// sight.
    unslotted: Vec<String>,
    /// The inks reproducing each sRGB colour from the given inks already
    /// solved on the page (see [`Self::ink_reproducing`]).
    reproduced: Reproduced,
    /// The same for the process equivalents of the slots' tints, which
    /// would otherwise fill the table of the page's colours.
    spot_reproduced: Reproduced,
    /// The conversion of the colours not given as ink: the process
    /// equivalent of a spot whose alternate colour is not process ink.
    separator: Separator,
}

/// The results of [`color::cmyk::ink_reproducing_at`], keyed by its
/// arguments: the depth and the sRGB colour, held at 16 bits, which holds
/// an 8-bit colour and its inks exactly, and the bits of the given inks.
/// At most [`REPRODUCED_MAX`] entries.
#[derive(Default)]
struct Reproduced(HashMap<(u32, [u16; 3], [u64; 4]), [u16; 4]>);

impl Reproduced {
    /// [`color::cmyk::ink_reproducing_at`] of `rgb` from `given`, solved
    /// once per distinct pair while the table has room.
    fn get<S: Sample>(&mut self, rgb: [S; 3], given: [f64; 4]) -> [S; 4] {
        let key = (S::BITS, rgb.map(u16::from_sample), given.map(f64::to_bits));
        if let Some(ink) = self.0.get(&key) {
            return ink.map(S::from_sample);
        }
        let ink = color::cmyk::ink_reproducing_at(rgb, given);
        if self.0.len() < REPRODUCED_MAX {
            let _ = self.0.insert(key, ink.map(u16::from_sample));
        }
        ink
    }
}

/// A spot colorant holding a slot.
struct Slot {
    /// The colorant name.
    name: Arc<[u8]>,
    /// The process ink of each tint of the colorant.
    lut: Box<SpotLut>,
    /// The same at 16 bits, for a page of 16-bit samples.
    lut16: Option<SpotLut16>,
}

impl InkMap {
    /// The map of a page of samples of `depth`, whose colours not given
    /// as ink convert by `separator`.
    pub(super) fn new(depth: SampleDepth, separator: Separator) -> Self {
        Self {
            depth,
            separator,
            ..Self::default()
        }
    }

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
        let (separator, reproduced) = (&self.separator, &mut self.spot_reproduced);
        let lut = Box::new(spot_lut(doc, name, space, separator, reproduced));
        let lut16 = (self.depth == SampleDepth::Sixteen)
            .then(|| spot_lut16(doc, name, space, separator, reproduced));
        self.slots.push(Slot {
            name: Arc::from(name),
            lut,
            lut16,
        });
        Some(self.slots.len() - 1)
    }

    /// Whether every spot colorant of `space` holds a slot or can take a
    /// free one: the distinct spot names of `space` without a slot number
    /// at most the free slots.
    fn fits(&self, space: &ColorSpace) -> bool {
        let free = SLOTS - self.slots.len();
        let mut unseen: [&[u8]; SLOTS] = [&[]; SLOTS];
        let mut count = 0;
        for colorant in space.colorants() {
            if let Colorant::Spot(name) = colorant
                && !self.slots.iter().any(|s| *s.name == **name)
                && !unseen[..count].contains(&&**name)
            {
                if count == free {
                    return false;
                }
                unseen[count] = name;
                count += 1;
            }
        }
        true
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
    pub(super) fn unslotted(&self) -> &[String] {
        &self.unslotted
    }

    /// [`Self::unslotted`], taken.
    pub(super) fn into_unslotted(self) -> Vec<String> {
        self.unslotted
    }

    /// Record `names`, spot colorants that found no slot on a canvas of
    /// slots of its own, after those already recorded, each once.
    pub(super) fn note_unslotted(&mut self, names: Vec<String>) {
        for name in names {
            if !self.unslotted.contains(&name) {
                self.unslotted.push(name);
            }
        }
    }

    /// The process inks printing the sRGB colour `rgb` of the
    /// four-component ICC colour `given` at the depth of `S` (see
    /// [`color::cmyk::ink_reproducing_at`]), solved once per page for each
    /// distinct colour.
    pub(super) fn ink_reproducing<S: Sample>(&mut self, rgb: [S; 3], given: [f64; 4]) -> [S; 4] {
        self.reproduced.get(rgb, given)
    }

    /// The process-equivalent tables of the slots, `None` for a free slot.
    pub(super) fn luts(&self) -> SpotTables<'_> {
        SpotTables {
            narrow: std::array::from_fn(|s| self.slots.get(s).map(|slot| &*slot.lut)),
            wide: std::array::from_fn(|s| self.slots.get(s).and_then(|slot| slot.lut16.as_ref())),
        }
    }

    /// The ink and the painted channels of the colour `comps` in the
    /// `Separation` or `DeviceN` space `space`: each colorant's tint in
    /// the colorant's channel — a repeated channel keeps the larger tint —
    /// `/All` in every channel, `/None` in none.  When the colour's spot
    /// colorants do not all fit the slots, none of them takes one: the
    /// colour is instead the process ink of its alternate colour,
    /// converted to `target` and separated by `separator`, the paint's, and
    /// paints the process channels; so is a colour with a process colorant
    /// under a separator of every process colour
    /// ([`Separator::separates_process_colours`]), whose process channels
    /// hold no ink.
    pub(super) fn colorant_ink<S: Sample>(
        &mut self,
        doc: &Document,
        (target, separator): (ColourTarget, &Separator),
        space: &ColorSpace,
        comps: &[f64],
    ) -> ([S; 8], ChannelMask) {
        let process_fallback = |map: &mut Self, overflow: bool| {
            if overflow {
                map.report_overflow(space);
            }
            let reproduced = &mut map.reproduced;
            let cmyk = space.process_ink_with(doc, target, comps, separator, &mut |rgb, given| {
                reproduced.get(rgb, given)
            });
            (no_spots(cmyk), PROCESS_CHANNELS)
        };
        if separator.separates_process_colours() && has_process_colorant(space) {
            return process_fallback(self, false);
        }
        if !self.fits(space) {
            return process_fallback(self, true);
        }
        let mut ink = [S::ZERO; 8];
        let mut bits = 0u8;
        for (colorant, &tint) in space.colorants().iter().zip(comps) {
            let tint = S::from_unit(tint);
            let Some(channels) = self.channels(doc, space, colorant) else {
                return process_fallback(self, true);
            };
            paint_tint(&mut ink, channels, tint);
            bits |= channels;
        }
        (ink, ChannelMask::new(bits))
    }

    /// The plan by which every colour of `space` paints on the page as
    /// ink, `None` for a space whose colours paint their sRGB conversion.
    ///
    /// The colours of a `Separation` or `DeviceN` space paint their
    /// colorants (see [`Self::colorant_ink`]), the space's spot colorants
    /// taking their slots.  Those of `DeviceCMYK` paint their components as
    /// process ink unless `separator` separates their colour
    /// ([`Separator::paints_device_cmyk`]), and so do those of a
    /// four-component ICC space unless
    /// the separator reproduces such a colour from its profile
    /// ([`Separator::reproduces_process_colour`]); `separator`, the paint's,
    /// converts them to the ink of an output profile that is not theirs
    /// ([`Separator::cmyk_link`]).
    pub(super) fn plan(
        &mut self,
        doc: &Document,
        space: &ColorSpace,
        separator: &Separator,
    ) -> Option<ColorantPlan> {
        match space {
            ColorSpace::Separation { .. } | ColorSpace::DeviceN { .. } => {
                Some(self.colorant_plan(doc, space, separator))
            }
            ColorSpace::DeviceCmyk => separator.paints_device_cmyk().then(process_plan),
            ColorSpace::IccBased { n: 4, .. } => (!separator.reproduces_process_colour()
                && !separator.separates_process_colours())
            .then(|| {
                separator
                    .cmyk_link(doc, space)
                    .map_or_else(process_plan, ColorantPlan::Linked)
            }),
            _ => None,
        }
    }

    /// [`Self::plan`] of a `Separation` or `DeviceN` space, whose colours
    /// fall back to `separator`'s ink of their alternate colour as
    /// [`Self::colorant_ink`]'s do.
    fn colorant_plan(
        &mut self,
        doc: &Document,
        space: &ColorSpace,
        separator: &Separator,
    ) -> ColorantPlan {
        if separator.separates_process_colours() && has_process_colorant(space) {
            return ColorantPlan::Process(separator.clone());
        }
        if !self.fits(space) {
            self.report_overflow(space);
            return ColorantPlan::Process(separator.clone());
        }
        let channels: Option<Box<[u8]>> = space
            .colorants()
            .iter()
            .map(|colorant| self.channels(doc, space, colorant))
            .collect();
        channels.map_or_else(
            || {
                self.report_overflow(space);
                ColorantPlan::Process(separator.clone())
            },
            ColorantPlan::Channels,
        )
    }

    /// The bit mask of the channels `colorant` of `space` paints: its
    /// process channel or its spot's slot, every channel for `/All`, none
    /// for `/None`; `None` when its spot finds no slot.
    fn channels(&mut self, doc: &Document, space: &ColorSpace, colorant: &Colorant) -> Option<u8> {
        Some(match colorant {
            Colorant::None => 0,
            Colorant::All => ChannelMask::ALL.bits(),
            Colorant::Process(process) => 1 << process.channel(),
            Colorant::Spot(name) => 1 << (SPOT_CHANNEL + self.slot(doc, name, space)?),
        })
    }
}

/// Whether a colorant of `space` is a process colorant.
fn has_process_colorant(space: &ColorSpace) -> bool {
    space
        .colorants()
        .iter()
        .any(|c| matches!(c, Colorant::Process(_)))
}

/// The plan of a process colour: its four components paint the process
/// channels C, M, Y and K in turn.
fn process_plan() -> ColorantPlan {
    ColorantPlan::Channels(Box::new(std::array::from_fn::<u8, 4, _>(|i| 1 << i)))
}

/// The spot-colorant tables of a page's slots, by depth.
pub(super) struct SpotTables<'a> {
    /// The 8-bit table of each slot.
    pub(super) narrow: [Option<&'a SpotLut>; SLOTS],
    /// The 16-bit table of each slot, for a page of 16-bit samples.
    pub(super) wide: [Option<&'a SpotLut16>; SLOTS],
}

/// The process ink of the tints of one spot colorant, alone — every other
/// component of its space at 0 — through the space's tint transform and
/// alternate space ([`ColorSpace::process_ink_with`]), the inks
/// reproducing an ICC colour taken from a page's [`Reproduced`].  A
/// colorant's table belongs to its slot for the whole page, whatever the
/// intent of the operators painting it, and converts under the initial
/// intent, relative colorimetric, into the separator's working space.
///
/// The table is folded into every pixel of the page, painted or not, so
/// tint 0 is no ink: the ink of the alternate colour at tint 0, the paper,
/// which a profile may separate to a little ink, is removed from the ink
/// of every tint as the probabilistic union adds it, `(u − p) / (1 − p)`
/// for a channel of ink `u` over paper `p`.
struct SpotTint<'a> {
    doc: &'a Document,
    space: &'a ColorSpace,
    component: usize,
    comps: Vec<f64>,
    separator: &'a Separator,
    reproduced: &'a mut Reproduced,
    /// The ink of tint 0, each channel of `[0, 1]`.
    paper: [f64; 4],
}

impl SpotTint<'_> {
    /// The process ink of tint `t` ([0, 1]) at the depth of `S`, the paper's
    /// removed.
    fn ink<S: Sample>(&mut self, t: f64) -> [S; 4] {
        let ink = self.raw::<u16>(t);
        std::array::from_fn(|i| {
            let (u, p) = (ink[i].to_unit(), self.paper[i]);
            if p <= 0.0 || p >= 1.0 {
                S::from_unit(u)
            } else {
                S::from_unit(((u - p) / (1.0 - p)).clamp(0.0, 1.0))
            }
        })
    }

    /// The process ink of tint `t` as the alternate colour separates.
    fn raw<S: Sample>(&mut self, t: f64) -> [S; 4] {
        if let Some(c) = self.comps.get_mut(self.component) {
            *c = t;
        }
        let reproduced = &mut *self.reproduced;
        self.space.process_ink_with(
            self.doc,
            ColourTarget::new(ColourIntent::INITIAL, self.separator.space()),
            &self.comps,
            self.separator,
            &mut |rgb, given| reproduced.get(rgb, given),
        )
    }
}

/// The process ink of each tint of the spot colorant `name` of `space`
/// (see [`SpotTint`]).
fn spot_lut(
    doc: &Document,
    name: &[u8],
    space: &ColorSpace,
    separator: &Separator,
    reproduced: &mut Reproduced,
) -> SpotLut {
    let mut tint = spot_tint(doc, name, space, separator, reproduced);
    std::array::from_fn(|t| tint.ink(f64::from(u8::try_from(t).unwrap_or(u8::MAX)) / 255.0))
}

/// [`spot_lut`] at 16 bits, at the tints of a [`SpotLut16`].
fn spot_lut16(
    doc: &Document,
    name: &[u8],
    space: &ColorSpace,
    separator: &Separator,
    reproduced: &mut Reproduced,
) -> SpotLut16 {
    let mut tint = spot_tint(doc, name, space, separator, reproduced);
    SpotLut16::from_fn(|t| tint.ink(t))
}

/// The tints of the spot colorant `name` of `space` (see [`SpotTint`]).
/// `name` is a spot colorant of `space`; the first component converts
/// otherwise.
fn spot_tint<'a>(
    doc: &'a Document,
    name: &[u8],
    space: &'a ColorSpace,
    separator: &'a Separator,
    reproduced: &'a mut Reproduced,
) -> SpotTint<'a> {
    let component = space
        .colorants()
        .iter()
        .position(|c| matches!(c, Colorant::Spot(spot) if **spot == *name));
    debug_assert!(
        component.is_some(),
        "{:?} is not a spot colorant of {space:?}",
        String::from_utf8_lossy(name)
    );
    let mut tint = SpotTint {
        doc,
        space,
        component: component.unwrap_or(0),
        comps: vec![0.0; space.ncomponents()],
        separator,
        reproduced,
        paper: [0.0; 4],
    };
    tint.paper = tint.raw::<u16>(0.0).map(Sample::to_unit);
    tint
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

    /// The page's reproducing inks are those the solver gives, kept once
    /// per distinct colour and given inks.
    #[test]
    fn ink_reproducing_is_solved_once_per_colour() {
        let mut map = InkMap::default();
        let given = [0.2, 0.0, 0.7, 0.1];
        let want = color::cmyk::ink_reproducing([90, 140, 60], given);
        assert_eq!(map.ink_reproducing([90u8, 140, 60], given), want);
        assert_eq!(map.ink_reproducing([90u8, 140, 60], given), want);
        assert_eq!(map.reproduced.0.len(), 1);
        let _ = map.ink_reproducing([90u8, 140, 61], given);
        assert_eq!(map.reproduced.0.len(), 2);
    }

    /// The solves of a slot's table are kept apart from the page's
    /// colours: building one leaves the page's table as it was.
    #[test]
    fn spot_tables_keep_their_solves_apart() {
        let doc = crate::test_helpers::make_doc_with_stream(
            &moxcms::ColorProfile::new_lab().encode().unwrap(),
            " /N 4",
        );
        let spot = space(
            &doc,
            "[/Separation /Spot [/ICCBased 2 0 R] \
             << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0.2 0.7 0.1 0] /N 1 >>]",
        );
        let mut map = InkMap::default();
        assert_eq!(map.slot(&doc, b"Spot", &spot), Some(0));
        assert!(
            !map.spot_reproduced.0.is_empty(),
            "the table solves through the ICC space"
        );
        assert!(map.reproduced.0.is_empty());
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
        let luts = map.luts().narrow;
        assert!(luts[0].is_some() && luts[1].is_some());
        assert!(luts[2].is_none() && luts[3].is_none());
        assert!(map.luts().wide.iter().all(Option::is_none), "an 8-bit page");
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
        assert!(map.luts().narrow.iter().all(Option::is_some));
    }

    #[test]
    fn spot_lut_is_the_process_ink_of_each_tint() {
        let doc = empty_doc();
        let lut = spot_lut(
            &doc,
            b"Teal",
            &separation(&doc, "Teal", "1 0 0.5 0"),
            &Separator::Model,
            &mut Reproduced::default(),
        );
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
        let lut = spot_lut(
            &doc,
            b"Teal",
            &dn,
            &Separator::Model,
            &mut Reproduced::default(),
        );
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
            map.colorant_ink::<u8>(&doc, (ColourTarget::SRGB, &Separator::Model), &cyan, &[0.6]),
            ([153, 0, 0, 0, 0, 0, 0, 0], ChannelMask::new(0b0001))
        );
        let all = separation(&doc, "All", "1 1 1 1");
        assert_eq!(
            map.colorant_ink::<u8>(&doc, (ColourTarget::SRGB, &Separator::Model), &all, &[0.6]),
            ([153; 8], ChannelMask::ALL)
        );
        let none = separation(&doc, "None", "0 0 0 0");
        assert_eq!(
            map.colorant_ink::<u8>(&doc, (ColourTarget::SRGB, &Separator::Model), &none, &[0.6]),
            ([0; 8], ChannelMask::new(0))
        );
        let spot = separation(&doc, "Spot", "0 1 0 0");
        assert_eq!(
            map.colorant_ink::<u8>(&doc, (ColourTarget::SRGB, &Separator::Model), &spot, &[1.0]),
            ([0, 0, 0, 0, 255, 0, 0, 0], ChannelMask::new(0b1_0000))
        );
    }

    #[test]
    fn colorant_ink_without_a_slot_is_the_alternate_process_ink() {
        let doc = empty_doc();
        let mut map = InkMap::default();
        for (slot, name) in ["S0", "S1", "S2", "S3"].into_iter().enumerate() {
            let (_, paints) = map.colorant_ink::<u8>(
                &doc,
                (ColourTarget::SRGB, &Separator::Model),
                &separation(&doc, name, "0 0 0 1"),
                &[1.0],
            );
            assert_eq!(paints, ChannelMask::new(1 << (SPOT_CHANNEL + slot)));
        }
        let fifth = separation(&doc, "S4", "0 0.5 1 0");
        assert_eq!(
            map.colorant_ink::<u8>(
                &doc,
                (ColourTarget::SRGB, &Separator::Model),
                &fifth,
                &[1.0]
            ),
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
        let (ink, paints) = map.colorant_ink::<u8>(
            &doc,
            (ColourTarget::SRGB, &Separator::Model),
            &dn,
            &[1.0, 1.0],
        );
        assert_eq!(paints, PROCESS_CHANNELS);
        assert_eq!(ink[4..], [0; 4]);
        assert!(map.luts().narrow[3].is_none(), "the last slot stays free");
        let fourth = separation(&doc, "S3", "1 0 0 0");
        assert_eq!(map.slot(&doc, b"S3", &fourth), Some(3));
        // A colour whose spots are all seen fits whatever the free slots.
        let (ink, paints) = map.colorant_ink::<u8>(
            &doc,
            (ColourTarget::SRGB, &Separator::Model),
            &separation(&doc, "S1", "0 0 0 1"),
            &[0.5],
        );
        assert_eq!((ink[5], paints), (128, ChannelMask::new(0b10_0000)));
    }

    /// A space fits when its distinct new spots, a repeated name counted
    /// once, number at most the free slots.
    #[test]
    fn fits_counts_distinct_new_spots_against_free_slots() {
        let doc = empty_doc();
        let dn = |names: &str| {
            space(
                &doc,
                &format!(
                    "[/DeviceN [{names}] /DeviceCMYK \
                     << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [0 1 0 0] /N 1 >>]"
                ),
            )
        };
        let mut map = InkMap::default();
        assert!(map.fits(&dn("/A /B /C /D")));
        assert!(map.fits(&dn("/A /A /B /B /C /D /Cyan /None")));
        assert!(!map.fits(&dn("/A /B /C /D /E")));
        for name in ["A", "B", "C"] {
            let _ = map.slot(&doc, name.as_bytes(), &separation(&doc, name, "0 0 0 1"));
        }
        assert!(map.fits(&dn("/A /B /C /D")));
        assert!(map.fits(&dn("/C /E /E")));
        assert!(!map.fits(&dn("/D /E")));
    }

    /// A tint transform that does not evaluate still gives a defined table.
    #[test]
    fn spot_lut_degrades_when_the_tint_transform_fails() {
        let doc = empty_doc();
        let broken = space(&doc, "[/Separation /Spot /DeviceCMYK 999 0 R]");
        let lut = spot_lut(
            &doc,
            b"Spot",
            &broken,
            &Separator::Model,
            &mut Reproduced::default(),
        );
        assert_eq!(lut[0], [0; 4]);
        assert_eq!(lut[255][3], 255);
        assert!(
            lut.windows(2).all(|w| w[0][3] <= w[1][3]),
            "black ink grows with the tint"
        );
    }

    /// On a page of 16-bit samples a slot's 16-bit table follows its 8-bit
    /// one, and a colorant's ink is its tint at 16 bits.
    #[test]
    fn a_16_bit_page_slots_its_spots_at_16_bits() {
        let doc = empty_doc();
        let teal = separation(&doc, "Teal", "1 0 0.5 0");
        let mut map = InkMap::new(SampleDepth::Sixteen, Separator::Model);
        let (ink, paints) = map.colorant_ink::<u16>(
            &doc,
            (ColourTarget::SRGB, &Separator::Model),
            &teal,
            &[0.25],
        );
        assert_eq!(
            (ink, paints),
            ([0, 0, 0, 0, 16_384, 0, 0, 0], ChannelMask::new(0b1_0000))
        );
        let tables = map.luts();
        let (Some(narrow), Some(wide)) = (tables.narrow[0], tables.wide[0]) else {
            panic!("the slot holds both tables");
        };
        for tint in 0..=u8::MAX {
            let at16 = wide.at(u16::from_u8(tint));
            for (w, n) in at16.iter().zip(narrow[usize::from(tint)]) {
                assert!(w.to_u8().abs_diff(n) <= 1, "tint {tint}: {at16:?}");
            }
        }
        assert_eq!(wide.at(u16::MAX), [u16::MAX, 0, 32_768, 0]);
    }
}
