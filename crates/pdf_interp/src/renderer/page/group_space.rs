//! Transparency groups composited in their own colour space (ISO 32000-2
//! §11.3.4, §11.6.6).
//!
//! A group's colours convert into the space its `/CS` names, its objects
//! composite there, and its result converts into its parent's space when
//! the group composites.  A group without a space, or whose space is the
//! parent canvas's, is painted on a canvas of the parent's kind as any
//! group is.  Otherwise the canvas of the group's space is:
//!
//! - for a CMYK space over an RGB canvas, an ink canvas whose colours not
//!   given as ink separate by the inverse of the process-ink model, which
//!   reproduces them through the model (an ICC CMYK space composites as
//!   `DeviceCMYK` does);
//! - for an sRGB space over a wide or an ink canvas, an sRGB canvas;
//! - for any other RGB space over an ink canvas, an RGB canvas of the
//!   separation's working space; over an RGB canvas the parent's own.
//!
//! A gray group is painted on its parent's canvas and its colour reduced
//! to gray as it composites (see [`super::space_convert`]): the reduction
//! is linear in the samples, so it equals compositing in gray for every
//! object of the `Normal` blend mode.
//!
//! The fill and stroke colours the group inherits convert again for its
//! canvas, and the graphics state saved around the form restores them.  A
//! non-isolated group's backdrop converts into the group's space for the
//! blend modes that read it; the parent's pixels are composited from the
//! group's converted result alone, so a pixel the group leaves unpainted
//! keeps its value.

use std::rc::Rc;

use raster::clip::Clip;
use raster::transparency::GroupParams;

use super::PageRenderer;
use super::canvas::{CanvasGroup, CanvasKind, PageCanvas};
use super::pixel_rect::PixelRect;
use super::space_convert::{CanvasSpace, SpaceConversion};
use crate::WorkingSpace;
use crate::renderer::color::PROCESS_CHANNELS;
use crate::resources::{ColorSpace, GroupSpace, RgbGroupSpace};
use crate::separation::{InkSetup, Separator};

/// How a transparency group is painted relative to its parent's canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GroupTarget {
    /// The space of the group's canvas.
    pub(super) space: CanvasSpace,
    /// Whether the group's colour reduces to gray as it composites.
    pub(super) gray: bool,
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
}

impl PageRenderer<'_> {
    /// The space the current canvas holds its colours in.
    pub(super) const fn canvas_space(&self) -> CanvasSpace {
        if self.canvas.kind().is_ink() {
            CanvasSpace::Ink
        } else {
            CanvasSpace::Rgb(self.space)
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
                return self.begin_group_in(target, &clip, params);
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
        target: GroupTarget,
    ) -> Option<SwitchedParent> {
        if target.space == self.canvas_space() {
            self.canvas.swap_group(group);
            None
        } else {
            Some(self.enter_group_in(group, target.space))
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
        target: GroupTarget,
        switched: Option<SwitchedParent>,
    ) -> Option<CanvasGroup> {
        if let Some(parent) = switched {
            return self.leave_group_in(group, parent);
        }
        self.canvas.swap_group(&mut group);
        if target.gray {
            self.reduce_group_to_gray(&mut group);
        }
        Some(group)
    }

    /// How a group of colour space `space` is painted over the current
    /// canvas.  A device space resolves through the default colour spaces
    /// of the resources in effect, the group's own.
    fn group_target(&mut self, space: Option<GroupSpace>) -> GroupTarget {
        let parent = self.canvas_space();
        let space = space.map(|s| self.resolve_group_space(s));
        let target = match (space, parent) {
            (Some(GroupSpace::Rgb(RgbGroupSpace::Profiled)), CanvasSpace::Ink) => {
                CanvasSpace::Rgb(self.space)
            }
            (None | Some(GroupSpace::Gray | GroupSpace::Rgb(RgbGroupSpace::Profiled)), _) => parent,
            (Some(GroupSpace::Cmyk { .. }), _) => CanvasSpace::Ink,
            (Some(GroupSpace::Rgb(_)), _) => CanvasSpace::Rgb(WorkingSpace::Srgb),
        };
        if let Some(space) = space
            && space.approximated()
            && (target != parent || matches!(space, GroupSpace::Rgb(_)))
        {
            self.diag.approximated_group_spaces.record(space);
        }
        GroupTarget {
            space: target,
            gray: space == Some(GroupSpace::Gray),
        }
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
    const fn kind_of(&self, space: CanvasSpace) -> CanvasKind {
        let depth = self.canvas.kind().depth();
        match space {
            CanvasSpace::Ink => CanvasKind::ink(depth),
            CanvasSpace::Rgb(_) => CanvasKind::rgb(depth),
        }
    }

    /// The ink setup and working space of a canvas of `space` opened over
    /// the current one: an ink canvas over an RGB one separates by the
    /// process-ink model, without simulating overprint; any other keeps
    /// the parent's setup.
    fn setup_of(&self, space: CanvasSpace) -> (InkSetup, WorkingSpace) {
        match (space, self.canvas_space()) {
            (CanvasSpace::Ink, CanvasSpace::Rgb(_)) => (
                InkSetup {
                    overprint: false,
                    cmyk: false,
                    separator: Separator::Model,
                },
                WorkingSpace::Srgb,
            ),
            (CanvasSpace::Ink, CanvasSpace::Ink) => (self.ink.clone(), self.space),
            (CanvasSpace::Rgb(space), _) => (self.ink.clone(), space),
        }
    }

    /// Open a transparency group of `params` painted in `target.space`, a
    /// space other than the current canvas's: its canvas, a non-isolated
    /// group's backdrop converted into its space.  `None`, with the render
    /// stopped, when the allocator refuses the canvas.
    fn begin_group_in(
        &mut self,
        target: GroupTarget,
        clip: &Clip,
        params: GroupParams,
    ) -> Option<CanvasGroup> {
        let bitmap = self.group_bitmap_of(self.kind_of(target.space))?;
        let (ink, _) = self.setup_of(target.space);
        let map = Rc::clone(&self.ink_map);
        let map = map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from: self.canvas_space(),
            to: target.space,
            gray: false,
            separator: &ink.separator,
            spots: &spots,
        };
        Some(self.canvas.begin_converted_group(
            self.group.backdrop.as_deref(),
            clip,
            params,
            bitmap,
            &conversion,
        ))
    }

    /// Paint on the canvas of `group`, opened in `space`: the parent's
    /// canvas, working space and ink setup are set aside and the inherited
    /// colours convert for the group's canvas.
    fn enter_group_in(&mut self, group: &mut CanvasGroup, space: CanvasSpace) -> SwitchedParent {
        let (ink, working) = self.setup_of(space);
        let canvas = std::mem::replace(&mut self.canvas, group.take_bitmap());
        let parent = SwitchedParent {
            canvas,
            space: std::mem::replace(&mut self.space, working),
            ink: std::mem::replace(&mut self.ink, ink),
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
        self.ink = parent.ink;
        let mut group = group;
        let Some(mut converted) = self.group_bitmap_of(self.canvas.kind()) else {
            group.put_bitmap(painted);
            self.group_pool.push(group.discard());
            return None;
        };
        if let Some(region) = group_rect(&group) {
            self.convert_canvas(&painted, &mut converted, region, from);
        }
        group.put_bitmap(painted);
        let (rehomed, cleared) = group.rehome(converted);
        self.group_pool.push(cleared);
        if self.canvas.kind().is_ink() {
            self.record_painted(None, PROCESS_CHANNELS);
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
        let map = Rc::clone(&self.ink_map);
        let map = map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from: space,
            to: space,
            gray: true,
            separator: &separator,
            spots: &spots,
        };
        if let Some(region) = region {
            bitmap.convert_region(region, &conversion);
        }
        group.put_bitmap(bitmap);
        if space == CanvasSpace::Ink {
            self.record_painted(None, PROCESS_CHANNELS);
        }
    }

    /// Convert `region` of `src`, a group's canvas painted in `from`, into
    /// `dst`, a transparent canvas of the current canvas's kind.
    fn convert_canvas(
        &self,
        src: &PageCanvas,
        dst: &mut PageCanvas,
        region: PixelRect,
        from: CanvasSpace,
    ) {
        let separator = self.separator(false);
        let map = self.ink_map.borrow();
        let spots = map.luts();
        let conversion = SpaceConversion {
            from,
            to: self.canvas_space(),
            gray: false,
            separator: &separator,
            spots: &spots,
        };
        src.convert_region_into(dst, region, &conversion);
    }
}

/// The region of `group` as a pixel rectangle; `None` when empty.
fn group_rect(group: &CanvasGroup) -> Option<PixelRect> {
    let (x0, y0, x1, y1) = group.region();
    PixelRect::new(x0, y0, x1, y1)
}
