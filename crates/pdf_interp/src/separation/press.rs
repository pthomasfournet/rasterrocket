//! The separation of a page's colours through a CMYK output profile, one
//! table per rendering intent.
//!
//! Each paint separates under its own intent (ISO 32000-2 §8.6.5.8): the
//! graphics state's, or an image's `/Intent`, unless the session fixes one
//! intent for every paint.  The table of an intent is tabulated by the
//! profile on first use and shared by every page of the profile.  The table
//! of the session's intent, relative colorimetry when it fixes none, is
//! built when the separation is made, so a profile that cannot separate is
//! a loud error before the page is painted; a later intent whose table
//! cannot be built separates through that table, with a warning.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::ProfileSeparation;
use crate::{OutputProfile, OutputProfileError, PreserveBlack, RenderingIntent, WorkingSpace};

/// The separation of colours of a working space through a CMYK output
/// profile under each rendering intent.
///
/// It compensates the black point or not, and preserves black, or not, in
/// colours given as the ink of another CMYK profile.
pub struct PressSeparation {
    profile: Arc<OutputProfile>,
    space: WorkingSpace,
    bpc: bool,
    preserve: Option<PreserveBlack>,
    /// The intent every paint separates under, when the session fixes one.
    fixed: Option<RenderingIntent>,
    /// The table of `fixed`, else of relative colorimetry.
    first: Arc<ProfileSeparation>,
    /// Whether a table that could not be built has been reported.
    warned: AtomicBool,
}

impl std::fmt::Debug for PressSeparation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressSeparation")
            .field("space", &self.space)
            .field("bpc", &self.bpc)
            .field("preserve", &self.preserve)
            .field("fixed", &self.fixed)
            .finish_non_exhaustive()
    }
}

impl PartialEq for PressSeparation {
    /// Two separations are equal when they separate through the same
    /// profile alike.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.profile, &other.profile)
            && self.space == other.space
            && self.bpc == other.bpc
            && self.preserve == other.preserve
            && self.fixed == other.fixed
    }
}

impl Eq for PressSeparation {}

impl PressSeparation {
    /// The separation through `profile` of colours of `space`, under
    /// `fixed` for every paint when given, else under each paint's intent,
    /// with black-point compensation when `bpc`, preserving the black of
    /// another CMYK profile's ink by `preserve`; the table of `fixed`, else
    /// of relative colorimetry, is tabulated now.
    ///
    /// # Errors
    ///
    /// As [`OutputProfile::separation_table`].
    pub(crate) fn new(
        profile: Arc<OutputProfile>,
        space: WorkingSpace,
        fixed: Option<RenderingIntent>,
        bpc: bool,
        preserve: Option<PreserveBlack>,
    ) -> Result<Self, OutputProfileError> {
        let first = profile.separation_table(
            space,
            fixed.unwrap_or(RenderingIntent::RelativeColorimetric),
            bpc,
        )?;
        Ok(Self {
            profile,
            space,
            bpc,
            preserve,
            fixed,
            first,
            warned: AtomicBool::new(false),
        })
    }

    /// The working space of the colours separated.
    #[must_use]
    pub const fn space(&self) -> WorkingSpace {
        self.space
    }

    /// The output profile.
    #[must_use]
    pub const fn profile(&self) -> &Arc<OutputProfile> {
        &self.profile
    }

    /// Whether the separation compensates the black point.
    #[must_use]
    pub const fn bpc(&self) -> bool {
        self.bpc
    }

    /// The preservation of black in the conversion of another CMYK
    /// profile's ink to the output profile's.
    #[must_use]
    pub const fn preserve(&self) -> Option<PreserveBlack> {
        self.preserve
    }

    /// The intent a paint under `intent` separates under: the session's,
    /// when it fixes one, else `intent`.
    #[must_use]
    pub fn intent(&self, intent: RenderingIntent) -> RenderingIntent {
        self.fixed.unwrap_or(intent)
    }

    /// The table a paint under `intent` separates through: that of
    /// [`Self::intent`], tabulated on first use; the table built with the
    /// separation when that one cannot be built, which is logged once.
    #[must_use]
    pub fn table(&self, intent: RenderingIntent) -> Arc<ProfileSeparation> {
        let intent = self.intent(intent);
        if self.fixed.is_some() || intent == RenderingIntent::RelativeColorimetric {
            return Arc::clone(&self.first);
        }
        self.profile
            .separation_table(self.space, intent, self.bpc)
            .unwrap_or_else(|e| {
                if !self.warned.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "rasterrocket-interp: the output profile has no {intent} separation \
                         ({e}); colours painted under it separate under relative colorimetry"
                    );
                }
                Arc::clone(&self.first)
            })
    }
}
