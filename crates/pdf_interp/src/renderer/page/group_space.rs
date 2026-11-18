//! Transparency groups composited in their own colour space (ISO 32000-2
//! §11.3.4, §11.6.6).
//!
//! A group's colours convert into the space its `/CS` names, its objects
//! composite there, and its result converts into its parent's space when
//! the group composites.  A group without a space, or whose space is the
//! parent canvas's, is painted on a canvas of the parent's kind as any
//! group is.  Otherwise the canvas of the group's space is:
//!
//! - for `DeviceCMYK` over an RGB canvas, an ink canvas whose colours not
//!   given as ink separate by the inverse of the process-ink model, which
//!   reproduces them through the model;
//! - for an ICC CMYK space over an RGB canvas, an ink canvas of the
//!   profile's ink (see [`crate::separation::GroupInk`]) with spot slots of
//!   its own, whose result converts back through the profile; a profile
//!   that cannot separate composites as `DeviceCMYK` does;
//! - for an sRGB space over a wide canvas, an sRGB canvas;
//! - for a `CalRGB` or non-sRGB ICC RGB space over any canvas, an
//!   additive ink canvas of the space's own components (see
//!   [`crate::separation::GroupRgb`]): a colour of the space paints its
//!   components, any other converts from the parent's working space into
//!   the space, and the result converts back as a colour of the space does
//!   on the page; a space whose conversions do not build composites in
//!   the parent's working space, which the page's diagnostics report;
//! - for an sRGB space over a canvas of ink, an additive ink canvas
//!   ([`CanvasSpace::AdditiveInk`]) of sRGB.
//!
//! The process channels of an additive ink canvas hold the complements of
//! its RGB, which blend as RGB does, and its spot channels the page's
//! spots, so a spot colour keeps its channel through the group (ISO
//! 32000-2 §11.7.3) and its overprint; a process colour paints no spot ink
//! and a spot colour white RGB.  A CMYK group within one composites through
//! the process-ink model.
//!
//! A gray group is painted on its parent's canvas and its colour reduced
//! to gray as it composites (see [`super::space_convert`]): the reduction
//! is linear in the samples, so it equals compositing in gray for every
//! object of the `Normal` blend mode.  An object of another blend mode
//! blends the gray of its colour with the gray of what it is painted over
//! ([`GrayReduction`]), so the group's reduced result is that of
//! compositing in gray.  A group without a space inherits its parent's,
//! gray as well.
//!
//! The fill and stroke colours the group inherits convert again for its
//! canvas, and the graphics state saved around the form restores them.  A
//! non-isolated group's backdrop converts into the group's space for the
//! blend modes that read it; the parent's pixels are composited from the
//! group's converted result alone, so a pixel the group leaves unpainted
//! keeps its value.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use raster::GrayBlend;
use raster::clip::Clip;
use raster::pipe::ChannelMask;
use raster::transparency::GroupParams;

use pdf::ObjectId;

use super::canvas::{CanvasGroup, CanvasKind, PageCanvas};
use super::depth::CanvasSample;
use super::pixel_rect::PixelRect;
use super::space_convert::{self, CanvasSpace, InkColour, SpaceConversion};
use super::{PageRenderer, SharedInkMap, ink_map, page_ink_map};
use crate::renderer::color::PROCESS_CHANNELS;
use crate::resources::{ColorSpace, GroupSpace, RgbGroupSpace};
use crate::separation::{AdditiveRgb, GroupInk, GroupRgb, InkSetup, Separator};
use crate::{WorkingSpace, working_space};

/// The blending space of a group composited in gray on a canvas of RGB or
/// ink: the gray of a pixel is that of its reduction to gray
/// ([`space_convert::gray_of`]), and a gray is written as the pixel of the
/// canvas whose reduction it is: a neutral RGB pixel, or black ink alone,
/// whose conventional gray is exact.
pub(super) struct GrayReduction {
    /// The space of the canvas the group is painted on.
    space: CanvasSpace,
    /// The page's spot colorants, folded into an ink pixel's process ink.
    ink_map: SharedInkMap,
}

impl std::fmt::Debug for GrayReduction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrayReduction")
            .field("space", &self.space)
            .finish_non_exhaustive()
    }
}

impl<S: CanvasSample> GrayBlend<S> for GrayReduction {
    fn gray(&self, px: &[S]) -> S {
        let map = self.ink_map.borrow();
        space_convert::gray_of(&self.space, px, &map.luts())
    }

    fn neutral(&self, gray: S, px: &mut [S]) {
        match &self.space {
            CanvasSpace::Ink => {
                px.fill(S::default());
                px[3] = S::from_wide(S::max_wide() - gray.wide());
            }
            CanvasSpace::Rgb(space) => {
                px.copy_from_slice(&working_space::from_srgb::<S, S>(*space, [gray; 3]));
            }
            CanvasSpace::AdditiveInk(space) => {
                let rgb = working_space::from_srgb::<S, S>(space.working(), [gray; 3]);
                let rgb = space.of_working(rgb);
                px.fill(S::default());
                for (p, v) in px.iter_mut().zip(rgb) {
                    *p = S::from_wide(S::max_wide() - v.wide());
                }
            }
        }
    }
}

/// The conversions of a `CalRGB` or ICC group space to and from a working
/// space, `None` when they do not build.
pub(super) type GroupRgbEntry = ((RgbGroupSpace, WorkingSpace), Option<Arc<GroupRgb>>);

/// How a transparency group is painted relative to its parent's canvas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct GroupTarget {
    /// The space of the group's canvas.
    pub(super) space: CanvasSpace,
    /// Whether the group's colour reduces to gray as it composites.
    pub(super) gray: bool,
    /// The profile stream of an ICC CMYK space whose ink an ink canvas
    /// over an RGB one holds.
    profile: Option<ObjectId>,
}

/// The parent's state a group painted on a canvas of another space keeps
/// aside while its content executes.
pub(super) struct SwitchedParent {
    /// The parent's canvas.
    canvas: PageCanvas,
    /// The parent's working space.
    space: WorkingSpace,
    /// The parent's ink setup.
    ink: InkSetup,
    /// The parent's ink map, when the group's canvas holds spot slots of
    /// its own.
    ink_map: Option<SharedInkMap>,
}

impl<'doc> PageRenderer<'doc> {
    /// The space the current canvas holds its colours in.
    pub(super) fn canvas_space(&self) -> CanvasSpace {
        match (self.canvas.kind().is_ink(), &self.ink.separator) {
            (true, Separator::Additive(space)) => CanvasSpace::AdditiveInk(space.clone()),
            (true, _) => CanvasSpace::Ink,
            (false, _) => CanvasSpace::Rgb(self.space),
        }
    }

    /// Open the transparency group of a form whose group attributes are
    /// `tg` and whose box is `bbox`, on a canvas of its space, and how it
    /// is painted; no group, with the render stopped, when the allocator
    /// refuses its canvas, and none when its extent is not finite.
    pub(super) fn open_form_group(
        &mut self,
        tg: crate::resources::TransparencyGroupParams,
        bbox: Option<[f64; 4]>,
    ) -> (Option<CanvasGroup>, GroupTarget) {
        let target = self.group_target(tg.space);
        let group = self.group_params(bbox, tg).and_then(|params| {
            let clip = self.gstate.current().clip.clone_shared();
            if target.space != self.canvas_space() {
                return self.begin_group_in(&target, &clip, params);
            }
            let bitmap = self.group_bitmap()?;
            Some(
                self.canvas
                    .begin_group(self.group.backdrop.as_deref(), &clip, params, bitmap),
            )
        });
        (group, target)
    }

    /// Paint on the canvas of `group`, opened by [`Self::open_form_group`]
    /// for `target`: the parent's canvas set aside when the group is in
    /// another space, exchanged with the group's bitmap otherwise.
    pub(super) fn enter_form_group(
        &mut self,
        group: &mut CanvasGroup,
        target: &GroupTarget,
    ) -> Option<SwitchedParent> {
        self.canvas_swaps += 1;
        if target.space == self.canvas_space() {
            self.canvas.swap_group(group);
            None
        } else {
            Some(self.enter_group_in(group, target))
        }
    }

    /// Return to the parent's canvas after the content of `group`, entered
    /// by [`Self::enter_form_group`] for `target`: the group holding its
    /// result in the parent's space, reduced to gray for a gray group;
    /// `None`, with the render stopped, when a canvas for the conversion is
    /// refused.
    pub(super) fn leave_form_group(
        &mut self,
        mut group: CanvasGroup,
        target: &GroupTarget,
        switched: Option<SwitchedParent>,
    ) -> Option<CanvasGroup> {
        self.canvas_swaps -= 1;
        if let Some(parent) = switched {
            return self.leave_group_in(group, parent);
        }
        self.canvas.swap_group(&mut group);
        if target.gray {
            self.reduce_group_to_gray(&mut group);
        }
        Some(group)
    }

    /// The gray blending space of a group painted for `target` and of
    /// colour space `space`, entered on the current canvas: that of the
    /// canvas for a gray group, and the parent's for a group without a
    /// space, which inherits it; `None` for a group blending in the
    /// components of its canvas.
    pub(super) fn group_gray(
        &self,
        target: &GroupTarget,
        space: Option<GroupSpace>,
        parent: Option<&Rc<GrayReduction>>,
    ) -> Option<Rc<GrayReduction>> {
        if target.gray {
            Some(self.gray_reduction())
        } else if space.is_none() {
            parent.cloned()
        } else {
            None
        }
    }

    /// The gray blending space of the current canvas.
    pub(super) fn gray_reduction(&self) -> Rc<GrayReduction> {
        Rc::new(GrayReduction {
            space: self.canvas_space(),
            ink_map: Rc::clone(&self.ink_map),
        })
    }

    /// How a group of colour space `space` is painted over the current
    /// canvas.  A device space resolves through the default colour spaces
    /// of the resources in effect, the group's own.
    fn group_target(&mut self, space: Option<GroupSpace>) -> GroupTarget {
        let parent = self.canvas_space();
        let space = space.map(|s| self.resolve_group_space(s));
        let target = match space {
            None | Some(GroupSpace::Gray) => parent.clone(),
            Some(GroupSpace::Cmyk { .. }) => CanvasSpace::Ink,
            Some(s @ GroupSpace::Rgb(rgb @ (RgbGroupSpace::Icc(_) | RgbGroupSpace::Cal(_)))) => {
                if let Some(group) = self.group_rgb(rgb) {
                    CanvasSpace::AdditiveInk(AdditiveRgb::Group(group))
                } else {
                    self.diag.approximated_group_spaces.record(s);
                    match parent {
                        CanvasSpace::Ink => CanvasSpace::Rgb(self.space),
                        _ => parent.clone(),
                    }
                }
            }
            Some(GroupSpace::Rgb(_)) => CanvasSpace::Rgb(WorkingSpace::Srgb),
        };
        let target = match target {
            CanvasSpace::Rgb(space) if parent.has_spots() => {
                CanvasSpace::AdditiveInk(AdditiveRgb::Working(space))
            }
            other => other,
        };
        let profile = match (space, &parent) {
            (Some(s @ GroupSpace::Cmyk { profile: Some(id) }), CanvasSpace::Rgb(working)) => {
                let separates =
                    GroupInk::load(self.resources.doc(), id, self.group_press(*working)).is_some();
                if !separates {
                    self.diag.approximated_group_spaces.record(s);
                }
                separates.then_some(id)
            }
            (Some(s @ GroupSpace::Cmyk { profile: Some(_) }), CanvasSpace::AdditiveInk(_)) => {
                self.diag.approximated_group_spaces.record(s);
                None
            }
            _ => None,
        };
        GroupTarget {
            space: target,
            gray: space == Some(GroupSpace::Gray),
            profile,
        }
    }

    /// The conversions of the group space `space` to and from the working
    /// space in effect, built once per page.
    fn group_rgb(&mut self, space: RgbGroupSpace) -> Option<Arc<GroupRgb>> {
        let key = (space, self.space);
        if let Some((_, group)) = self.group_rgb.iter().find(|(k, _)| *k == key) {
            return group.clone();
        }
        let group = GroupRgb::load(self.resources.doc(), space, self.space);
        self.group_rgb.push((key, group.clone()));
        group
    }

    /// `space` with a device RGB space replaced by the default RGB space
    /// of the resources in effect, when they name one.
    fn resolve_group_space(&self, space: GroupSpace) -> GroupSpace {
        space.with_default_rgb(self.resources.doc(), || {
            self.resources.device_space(ColorSpace::DeviceRgb)
        })
    }

    /// A transparent canvas of `kind` and the current canvas's size for a
    /// transparency group, from the pool or fresh; `None`, with the render
    /// stopped, when the allocator refuses a fresh one.
    pub(super) fn group_bitmap_of(&mut self, kind: CanvasKind) -> Option<PageCanvas> {
        let dims = self.canvas.dims();
        self.group_pool
            .iter()
            .rposition(|b| b.kind() == kind && b.dims() == dims)
            .map(|i| self.group_pool.swap_remove(i))
            .or_else(|| self.try_canvas(kind, dims.0, dims.1))
    }

    /// The canvas kind of `space` at the current canvas's depth.
    const fn kind_of(&self, space: &CanvasSpace) -> CanvasKind {
        let depth = self.canvas.kind().depth();
        match space {
            CanvasSpace::Ink | CanvasSpace::AdditiveInk(_) => CanvasKind::ink(depth),
            CanvasSpace::Rgb(_) => CanvasKind::rgb(depth),
        }
    }

    /// The ink setup and working space of a canvas opened for `target`
    /// over the current one: an ink canvas over an RGB one, without
    /// simulating overprint, separates into the ink of the target's
    /// profile from the parent's working space, else by the process-ink
    /// model from sRGB; any other keeps the parent's setup.
    fn setup_of(&self, target: &GroupTarget) -> (InkSetup, WorkingSpace) {
        match (&target.space, self.canvas_space()) {
            (CanvasSpace::Ink, CanvasSpace::Rgb(working)) => {
                let group = target.profile.and_then(|id| {
                    GroupInk::load(self.resources.doc(), id, self.group_press(working))
                });
                let (separator, space) = group
                    .map_or((Separator::Model, WorkingSpace::Srgb), |g| {
                        (Separator::Group(g), working)
                    });
                let ink = InkSetup {
                    overprint: false,
                    cmyk: false,
                    separator,
                };
                (ink, space)
            }
            (CanvasSpace::Ink, CanvasSpace::AdditiveInk(_)) => (
                InkSetup {
                    overprint: false,
                    cmyk: false,
                    separator: Separator::Model,
                },
                WorkingSpace::Srgb,
            ),
            (CanvasSpace::Ink, CanvasSpace::Ink) => (self.ink.clone(), self.space),
            (CanvasSpace::AdditiveInk(space), CanvasSpace::Rgb(_)) => (
                InkSetup {
                    overprint: false,
                    cmyk: false,
                    separator: Separator::Additive(space.clone()),
                },
                space.working(),
            ),
            (CanvasSpace::AdditiveInk(space), _) => {
                let ink = InkSetup {
                    separator: Separator::Additive(space.clone()),
                    ..self.ink.clone()
                };
                (ink, space.working())
            }
            (CanvasSpace::Rgb(space), _) => (self.ink.clone(), *space),
        }
    }

    /// The working space `working` and the black-point compensation in
    /// effect, which a group's ink separates under: the group's result
    /// converts back under the graphics state the group is painted in.
    fn group_press(&self, working: WorkingSpace) -> (WorkingSpace, bool) {
        (working, self.gstate.current().colour_intent.bpc)
    }

    /// The colour of the ink the current canvas holds, converted to the
    /// working space and intent in effect.
    fn ink_colour(&self) -> InkColour<'doc> {
        InkColour::of(
            &self.ink.separator,
            self.resources.doc(),
            self.colour_target(),
        )
    }

    /// Open a transparency group of `params` painted in `target.space`, a
    /// space other than the current canvas's: its canvas, a non-isolated
    /// group's backdrop converted into its space.  `None`, with the render
    /// stopped, when the allocator refuses the canvas.
    fn begin_group_in(
        &mut self,
        target: &GroupTarget,
        clip: &Clip,
        params: GroupParams,
    ) -> Option<CanvasGroup> {
        let bitmap = self.group_bitmap_of(self.kind_of(&target.space))?;
        let (ink, _) = self.setup_of(target);
        let colour = self.ink_colour();
        let map = Rc::clone(&self.ink_map);
        let map = map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from: self.canvas_space(),
            to: target.space.clone(),
            gray: false,
            separator: &ink.separator,
            spots: &spots,
            ink: &colour,
        };
        Some(self.canvas.begin_converted_group(
            self.group.backdrop.as_deref(),
            clip,
            params,
            bitmap,
            &conversion,
        ))
    }

    /// Paint on the canvas of `group`, opened for `target`: the parent's
    /// canvas, working space and ink setup are set aside, and its ink map
    /// for a canvas of a group's ink, which takes a map of its own; the
    /// inherited colours convert for the group's canvas.
    fn enter_group_in(&mut self, group: &mut CanvasGroup, target: &GroupTarget) -> SwitchedParent {
        let (ink, working) = self.setup_of(target);
        let ink_map = ink.separator.group_ink().is_some().then(|| {
            let depth = self.canvas.kind().depth();
            let map = ink_map::InkMap::new(depth, ink.separator.clone());
            std::mem::replace(&mut self.ink_map, Rc::new(RefCell::new(map)))
        });
        let canvas = std::mem::replace(&mut self.canvas, group.take_bitmap());
        let parent = SwitchedParent {
            canvas,
            space: std::mem::replace(&mut self.space, working),
            ink: std::mem::replace(&mut self.ink, ink),
            ink_map,
        };
        self.reconvert_colours(|_| true);
        parent
    }

    /// Return to the parent's canvas set aside by [`Self::enter_group_in`]
    /// and give `group` its canvas back, converted into the parent's space
    /// on a canvas of the parent's kind; the canvas the group was painted
    /// on, cleared, returns to the pool.  `None`, with the render stopped,
    /// when the allocator refuses the converted canvas.
    fn leave_group_in(
        &mut self,
        group: CanvasGroup,
        parent: SwitchedParent,
    ) -> Option<CanvasGroup> {
        let from = self.canvas_space();
        let painted = std::mem::replace(&mut self.canvas, parent.canvas);
        self.space = parent.space;
        let source = std::mem::replace(&mut self.ink, parent.ink);
        let own_map = parent
            .ink_map
            .map(|map| std::mem::replace(&mut self.ink_map, map));
        let spots = own_map.clone().unwrap_or_else(|| Rc::clone(&self.ink_map));
        let mut group = group;
        let mut converted = self.group_bitmap_of(self.canvas.kind());
        if let Some(converted) = &mut converted
            && let Some(region) = group_rect(&group)
        {
            self.convert_canvas(&painted, converted, region, (from.clone(), &source, &spots));
        }
        drop(spots);
        if let Some(map) = own_map {
            let unslotted = page_ink_map(map).into_unslotted();
            self.ink_map.borrow_mut().note_unslotted(unslotted);
        }
        let Some(converted) = converted else {
            group.put_bitmap(painted);
            self.group_pool.push(group.discard());
            return None;
        };
        group.put_bitmap(painted);
        let (rehomed, cleared) = group.rehome(converted);
        self.group_pool.push(cleared);
        if self.canvas.kind().is_ink() {
            let channels = if from.has_spots() {
                ChannelMask::ALL
            } else {
                PROCESS_CHANNELS
            };
            self.record_painted(None, channels);
        }
        Some(rehomed)
    }

    /// Reduce the colour of `group`, painted on a canvas of the current
    /// canvas's kind, to gray.
    fn reduce_group_to_gray(&self, group: &mut CanvasGroup) {
        let space = self.canvas_space();
        let mut bitmap = group.take_bitmap();
        let region = group_rect(group);
        let separator = self.separator(true);
        let colour = self.ink_colour();
        let map = Rc::clone(&self.ink_map);
        let map = map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from: space.clone(),
            to: space.clone(),
            gray: true,
            separator: &separator,
            spots: &spots,
            ink: &colour,
        };
        if let Some(region) = region {
            bitmap.convert_region(region, &conversion);
        }
        group.put_bitmap(bitmap);
        if space == CanvasSpace::Ink {
            self.record_painted(None, PROCESS_CHANNELS);
        }
    }

    /// Convert `region` of `src`, a group's canvas painted in `from` and
    /// set up as `source`, its spots slotted in `map`, into `dst`, a
    /// transparent canvas of the current canvas's kind.
    fn convert_canvas(
        &self,
        src: &PageCanvas,
        dst: &mut PageCanvas,
        region: PixelRect,
        (from, source, map): (CanvasSpace, &InkSetup, &SharedInkMap),
    ) {
        let separator = self.separator(false);
        let colour = InkColour::of(
            &source.separator,
            self.resources.doc(),
            self.colour_target(),
        );
        let map = map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from,
            to: self.canvas_space(),
            gray: false,
            separator: &separator,
            spots: &spots,
            ink: &colour,
        };
        src.convert_region_into(dst, region, &conversion);
    }
}

/// The region of `group` as a pixel rectangle; `None` when empty.
fn group_rect(group: &CanvasGroup) -> Option<PixelRect> {
    let (x0, y0, x1, y1) = group.region();
    PixelRect::new(x0, y0, x1, y1)
}
