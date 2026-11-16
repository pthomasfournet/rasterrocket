//! The process ink of a transparency group whose colour space is an ICC
//! CMYK profile (ISO 32000-2 §11.3.4, §11.6.6), composited over a canvas
//! of RGB.
//!
//! The group's canvas holds the profile's ink.  A colour of the profile's
//! space paints its components; one of another CMYK profile converts to
//! this profile's ink through the PCS; every other colour, `DeviceCMYK`
//! included, is the colour it shows on an RGB page, separated by the
//! profile's PCS → device table from the working space.  The result
//! converts back by the profile's device → PCS table, the conversion a
//! colour of the space takes on an RGB page.
//!
//! The profile is read as an [`OutputProfile`] and its separation built,
//! once per document, profile stream and working space; a profile that
//! is not an output profile or cannot separate yields none, and the group
//! composites through the process-ink model instead.

use std::sync::{Arc, OnceLock};

use pdf::{Document, ObjectId};

use super::{PressSeparation, ProfileInk};
use crate::resources::icc_shaper::fetch_profile_bytes;
use crate::{OutputProfile, RenderingIntent, WorkingSpace};

/// The separation through a group's ICC CMYK profile, by working space
/// and black-point compensation, each built on first use.
#[derive(Default)]
struct GroupPresses([[OnceLock<Option<Arc<PressSeparation>>>; 2]; WorkingSpace::ALL.len()]);

/// The process ink of the ICC CMYK space of profile stream `id`, separated
/// under one rendering intent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GroupInk {
    /// The profile stream.
    id: ObjectId,
    /// The separation through the profile under the paint's intent.
    pub(super) ink: ProfileInk,
}

impl GroupInk {
    /// The ink of profile stream `id` of `doc`, separated from `space`
    /// under relative colorimetry, with black-point compensation when
    /// `bpc`, so that it inverts the profile's conversion of a colour of
    /// its space under the same compensation; `None` when the stream is not
    /// a CMYK output profile whose PCS → device table separates, which is
    /// logged once per stream, space and compensation.
    pub(crate) fn load(
        doc: &Document,
        id: ObjectId,
        (space, bpc): (WorkingSpace, bool),
    ) -> Option<Self> {
        let presses = doc.derived(id, || Some(GroupPresses::default()))?;
        let press = presses.0[space.index()][usize::from(bpc)]
            .get_or_init(|| {
                separation(doc, id, space, bpc)
                    .map_err(|e| {
                        log::warn!(
                            "rasterrocket-interp: the ICC CMYK profile of a transparency group \
                             does not separate ({e}); the group composites through the \
                             process-ink model"
                        );
                    })
                    .ok()
            })
            .clone()?;
        Some(Self {
            id,
            ink: ProfileInk::new(&press, RenderingIntent::RelativeColorimetric),
        })
    }

    /// The profile stream.
    pub(crate) const fn id(&self) -> ObjectId {
        self.id
    }

    /// The ink separated under `intent`.
    pub(super) fn under(&self, intent: RenderingIntent) -> Self {
        if self.ink.press.intent(intent) == self.ink.intent {
            return self.clone();
        }
        Self {
            id: self.id,
            ink: ProfileInk::new(&self.ink.press, intent),
        }
    }
}

/// The separation from `space` through profile stream `id`, each paint
/// under its own intent, with black-point compensation when `bpc`.
fn separation(
    doc: &Document,
    id: ObjectId,
    space: WorkingSpace,
    bpc: bool,
) -> Result<Arc<PressSeparation>, String> {
    let bytes = fetch_profile_bytes(doc, id).ok_or("the profile stream does not decode")?;
    let profile = OutputProfile::from_bytes(bytes).map_err(|e| e.to_string())?;
    if profile.space() != crate::OutputSpace::Cmyk {
        return Err("the profile is not CMYK".to_owned());
    }
    PressSeparation::new(Arc::new(profile), space, None, bpc, None)
        .map(Arc::new)
        .map_err(|e| e.to_string())
}
