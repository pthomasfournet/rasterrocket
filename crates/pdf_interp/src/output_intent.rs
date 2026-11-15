//! Output intents (ISO 32000-2 §14.11.5): the printing condition a
//! document, or one page of it, was prepared for.
//!
//! An output intent dictionary names its subtype (`/S`: `GTS_PDFX`,
//! `GTS_PDFA1`, `ISO_PDFE1`, …), the condition (`/OutputConditionIdentifier`)
//! and, as `/DestOutputProfile`, the ICC profile of the intended output
//! device.  A page's own `/OutputIntents` array replaces the document's
//! (the catalog's) for that page.  Of an array, the intent chosen is the
//! first of subtype `GTS_PDFX` that embeds a profile, else the first of
//! any subtype that does.
//!
//! A soft proof needs a press: [`OutputIntent::press_profile`] accepts
//! only a CMYK profile.

use std::sync::Arc;

use pdf::{Document, Object, ObjectId};

use crate::output_profile::{OutputProfile, OutputProfileError, OutputSpace};
use crate::resources::dict_ext::DictExt;
use crate::resources::{resolve_array, resolve_dict};

/// The output intent of a document or page, as chosen from its
/// `/OutputIntents` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputIntent {
    /// The subtype `/S`, without the `/`.
    pub subtype: String,
    /// The `/OutputConditionIdentifier`, when given.
    pub condition: Option<String>,
    /// The object of the `/DestOutputProfile` stream.
    pub profile: ObjectId,
}

/// Why a document or page has no output intent a soft proof can use.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputIntentError {
    /// Neither the page nor the document carries `/OutputIntents`.
    Missing,
    /// No intent of the array embeds a profile: `subtype` and `condition`
    /// are those of the intent that would have been chosen, and
    /// `external` whether it refers to its profile by
    /// `/DestOutputProfileRef` instead.
    NoProfile {
        /// The subtype of the intent.
        subtype: String,
        /// Its condition identifier.
        condition: Option<String>,
        /// Whether it gives a `/DestOutputProfileRef`.
        external: bool,
    },
    /// The profile is not an ICC output profile the converter reads.
    Unreadable {
        /// The subtype of the intent.
        subtype: String,
        /// Why the profile was refused.
        reason: String,
    },
    /// The profile is not of CMYK device values: it describes a display
    /// or another non-press output.
    NotCmyk {
        /// The subtype of the intent.
        subtype: String,
        /// The profile's colour space.
        space: String,
    },
}

impl std::fmt::Display for OutputIntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => f.write_str(
                "the document carries no /OutputIntents: a soft proof needs the press \
                 condition the file was prepared for (a PDF/X output intent)",
            ),
            Self::NoProfile {
                subtype,
                condition,
                external,
            } => {
                write!(f, "the output intent /{subtype}")?;
                if let Some(condition) = condition {
                    write!(f, " ({condition})")?;
                }
                if *external {
                    f.write_str(
                        " refers to an external profile (/DestOutputProfileRef), which is not \
                         read: nothing to proof against",
                    )
                } else {
                    f.write_str(" embeds no /DestOutputProfile: nothing to proof against")
                }
            }
            Self::Unreadable { subtype, reason } => {
                write!(
                    f,
                    "the profile of the output intent /{subtype} is unusable: {reason}"
                )
            }
            Self::NotCmyk { subtype, space } => write!(
                f,
                "the output intent /{subtype} names a profile of {space} values, which \
                 describes a display, not a press: nothing to proof against; use --output-profile \
                 (SessionConfig::output_profile) for a display profile"
            ),
        }
    }
}

impl std::error::Error for OutputIntentError {}

/// The output intent of the document's catalog.
///
/// # Errors
///
/// [`OutputIntentError::Missing`] when the catalog has no
/// `/OutputIntents`; [`OutputIntentError::NoProfile`] when no intent of
/// it embeds a profile.
pub fn document_output_intent(doc: &Document) -> Result<OutputIntent, OutputIntentError> {
    let catalog = doc.catalog().map_err(|_| OutputIntentError::Missing)?;
    choose(doc, catalog.get(b"OutputIntents")).ok_or(OutputIntentError::Missing)?
}

/// The output intent of page `page_id`'s own `/OutputIntents`; `None`
/// when the page has none, and the document's applies.
///
/// # Errors
///
/// [`OutputIntentError::NoProfile`] when no intent of the page's array
/// embeds a profile.
pub fn page_output_intent(
    doc: &Document,
    page_id: ObjectId,
) -> Option<Result<OutputIntent, OutputIntentError>> {
    let page = doc.get_dictionary(page_id).ok()?;
    choose(doc, page.get(b"OutputIntents"))
}

/// The intent chosen from the `/OutputIntents` value `array`; `None`
/// when it is absent or holds no intent dictionary.
fn choose(
    doc: &Document,
    array: Option<&Object>,
) -> Option<Result<OutputIntent, OutputIntentError>> {
    let intents: Vec<_> = resolve_array(doc, array)
        .iter()
        .filter_map(|entry| resolve_dict(doc, entry))
        .collect();
    let first = intents
        .iter()
        .find(|d| d.get_name(b"S") == Some(b"GTS_PDFX"))
        .or_else(|| intents.first())?;
    let embedded = |d: &&pdf::Dictionary| d.get(b"DestOutputProfile").is_some();
    let chosen = intents
        .iter()
        .filter(|d| d.get_name(b"S") == Some(b"GTS_PDFX"))
        .find(embedded)
        .or_else(|| intents.iter().find(embedded));
    let subtype = |d: &pdf::Dictionary| {
        d.get_name(b"S").map_or_else(
            || "(none)".to_owned(),
            |s| String::from_utf8_lossy(s).into_owned(),
        )
    };
    let condition = |d: &pdf::Dictionary| {
        d.get(b"OutputConditionIdentifier")
            .and_then(Object::as_str)
            .map(|s| String::from_utf8_lossy(s).into_owned())
    };
    let Some(chosen) = chosen else {
        return Some(Err(OutputIntentError::NoProfile {
            subtype: subtype(first),
            condition: condition(first),
            external: first.get(b"DestOutputProfileRef").is_some(),
        }));
    };
    let Some(profile) = chosen
        .get(b"DestOutputProfile")
        .and_then(Object::as_reference)
    else {
        return Some(Err(OutputIntentError::Unreadable {
            subtype: subtype(chosen),
            reason: "/DestOutputProfile is not a reference to a stream".to_owned(),
        }));
    };
    Some(Ok(OutputIntent {
        subtype: subtype(chosen),
        condition: condition(chosen),
        profile,
    }))
}

impl OutputIntent {
    /// The intent's profile as the CMYK press of a soft proof.
    ///
    /// # Errors
    ///
    /// [`OutputIntentError::Unreadable`] when the stream does not decode
    /// or is not an output profile; [`OutputIntentError::NotCmyk`] for a
    /// profile of RGB, gray or other device values.
    pub fn press_profile(&self, doc: &Document) -> Result<Arc<OutputProfile>, OutputIntentError> {
        let unreadable = |reason: String| OutputIntentError::Unreadable {
            subtype: self.subtype.clone(),
            reason,
        };
        let object = doc
            .get_object(self.profile)
            .map_err(|e| unreadable(e.to_string()))?;
        let stream = object
            .as_stream()
            .ok_or_else(|| unreadable("/DestOutputProfile is not a stream".to_owned()))?;
        let bytes = stream.decompressed_content().map_err(unreadable)?;
        let not_cmyk = |space: String| OutputIntentError::NotCmyk {
            subtype: self.subtype.clone(),
            space,
        };
        let profile = OutputProfile::from_bytes(bytes).map_err(|e| match e {
            OutputProfileError::ColorSpace(space) => not_cmyk(space),
            other => unreadable(other.to_string()),
        })?;
        match profile.space() {
            OutputSpace::Cmyk => Ok(Arc::new(profile)),
            OutputSpace::Rgb => Err(not_cmyk("RGB".to_owned())),
            OutputSpace::Gray => Err(not_cmyk("gray".to_owned())),
        }
    }
}

#[cfg(test)]
mod tests;
