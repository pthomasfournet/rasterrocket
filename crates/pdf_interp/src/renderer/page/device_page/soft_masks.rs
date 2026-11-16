//! Soft masks recorded for the whole-page GPU raster.
//!
//! The mask's plane is the one `gs` computed on the host, by the CPU's
//! code: held once per page, it masks the commands of each object painted
//! under it.  An object is what the CPU's `paint_through_soft_mask`
//! blends back as one: its paints are recorded as one device object over
//! the rectangle the CPU blends, the object's region within the clip
//! rectangle; a paint the CPU leaves unmasked (no region, or a mask of
//! another size) is recorded unmasked.  The planes are kept alive with
//! the recording, so no address names two planes.

use std::collections::HashMap;
use std::sync::Arc;

use gpu::page::{PixelBounds, PlaneId, PlaneValues};

use super::super::PageRenderer;
use super::super::depth::SampleVec;
use super::super::pixel_rect::PixelRect;
use super::{Recording, command_refusal};
use crate::renderer::gstate::SoftMaskPlane;

/// The mask planes a page's recording holds, by the plane each copies.
#[derive(Default)]
pub(super) struct HeldPlanes {
    by_plane: HashMap<*const SoftMaskPlane, PlaneId>,
    planes: Vec<Arc<SoftMaskPlane>>,
}

impl Recording {
    /// The plane copying `mask`, held on first use.
    fn plane_of(&mut self, mask: &Arc<SoftMaskPlane>) -> Result<PlaneId, &'static str> {
        if let Some(&plane) = self.planes.by_plane.get(&Arc::as_ptr(mask)) {
            return Ok(plane);
        }
        let deep = self.commands.depth().max_sample() > u16::from(u8::MAX);
        let values = match &mask.values {
            SampleVec::U8(v) if !deep => PlaneValues::Eight(v),
            SampleVec::U16(v) if deep => PlaneValues::Sixteen(v),
            _ => return Err("soft mask depth"),
        };
        let plane = self
            .commands
            .add_mask_plane(values)
            .map_err(|e| command_refusal(&e))?;
        self.planes.planes.push(Arc::clone(mask));
        let _ = self.planes.by_plane.insert(Arc::as_ptr(mask), plane);
        Ok(plane)
    }
}

impl PageRenderer<'_> {
    /// Record `paint` — the painting of one object within `region`, the
    /// pixels it can touch — under `mask`, as `paint_through_soft_mask`
    /// paints it.
    pub(in crate::renderer::page) fn record_through_soft_mask(
        &mut self,
        mask: &Arc<SoftMaskPlane>,
        region: Option<PixelRect>,
        paint: impl FnOnce(&mut Self),
    ) {
        let rect = region
            .filter(|_| mask.width == self.width && mask.height == self.height)
            .and_then(|r| self.within_clip(r));
        let Some(rect) = rect else {
            paint(self);
            return;
        };
        let Some(recording) = self.device_page.as_mut() else {
            return;
        };
        let begun = recording.plane_of(mask).and_then(|plane| {
            let bounds = pixel_bounds(rect).ok_or("soft mask range")?;
            recording
                .commands
                .begin_object(plane, bounds)
                .map_err(|e| command_refusal(&e))
        });
        if let Err(reason) = begun {
            return self.refuse_device(reason);
        }
        paint(self);
        if let Some(recording) = self.device_page.as_mut() {
            recording.commands.end_object();
        }
    }
}

/// `rect`'s pixels as inclusive bounds; `None` past the `i32` range.
fn pixel_bounds(rect: PixelRect) -> Option<PixelBounds> {
    let at = |v: u32| i32::try_from(v).ok();
    Some(PixelBounds {
        x0: at(rect.x0)?,
        y0: at(rect.y0)?,
        x1: at(rect.x1)? - 1,
        y1: at(rect.y1)? - 1,
    })
}
