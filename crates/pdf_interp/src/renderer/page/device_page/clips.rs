//! Path clips recorded for the whole-page GPU raster.
//!
//! A clip with path regions is recorded as the regions' coverage tables,
//! the tables `raster::Clip::mul_row` reads, held once per page, and the
//! clip as their sequence newest first; each paint is pushed with the clip
//! in effect, which scales it on the device as the CPU clip scales it.
//! The clips are kept alive with the recording, so no region identity
//! names two regions.

use std::collections::HashMap;

use gpu::page::{ClipId, ClipRegionId, CoverageRows};
use raster::RegionKey;
use raster::clip::Clip;

use super::{Recording, command_refusal};

/// The clip regions and clips a page's recording holds, by identity.
#[derive(Default)]
pub(super) struct HeldClips {
    regions: HashMap<RegionKey, ClipRegionId>,
    clips: HashMap<RegionKey, ClipId>,
    held: Vec<Clip>,
}

impl Recording {
    /// Push the paints that follow under `clip`; the reason when its
    /// regions cannot be recorded.
    pub(super) fn use_clip(&mut self, clip: &Clip) -> Result<(), &'static str> {
        let Some(key) = clip.path_key() else {
            self.commands.set_clip(None);
            return Ok(());
        };
        if let Some(&id) = self.clips.clips.get(&key) {
            self.commands.set_clip(Some(id));
            return Ok(());
        }
        let mut regions = Vec::new();
        for (region, table) in clip.path_regions() {
            let table = table.ok_or("aliased path clip")?;
            let id = if let Some(&id) = self.clips.regions.get(&region) {
                id
            } else {
                let id = self
                    .commands
                    .add_clip_region(CoverageRows {
                        y_min: table.y_min(),
                        row_starts: table.row_starts(),
                        spans: table.span_triples(),
                        bytes: table.bytes(),
                    })
                    .map_err(|e| command_refusal(&e))?;
                let _ = self.clips.regions.insert(region, id);
                id
            };
            regions.push(id);
        }
        let id = self
            .commands
            .add_clip(&regions)
            .map_err(|e| command_refusal(&e))?;
        let _ = self.clips.clips.insert(key, id);
        self.clips.held.push(clip.clone_shared());
        self.commands.set_clip(Some(id));
        Ok(())
    }

    /// [`Self::use_clip`], stopping the recording when it fails; whether
    /// the paints may be pushed.
    pub(super) fn clip_to(&mut self, clip: &Clip) -> bool {
        match self.use_clip(clip) {
            Ok(()) => true,
            Err(reason) => {
                self.refuse(reason);
                false
            }
        }
    }
}
