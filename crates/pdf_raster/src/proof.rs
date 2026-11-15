//! Soft proofing a session's pages against their output intents (see
//! [`SessionConfig::proof`](crate::SessionConfig::proof)).
//!
//! A page's press is the profile of its own output intent, else the
//! document's (see [`pdf_interp::output_intent`]).  Each press profile is
//! read once per session and its proof tabulated on the first page that
//! names it, shared by every page that does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use pdf::{Document, ObjectId};
use pdf_interp::{OutputIntent, OutputIntentError, OutputProfile, ProofIntent, SoftProof};

use crate::RasterError;

/// The soft proof of a session.
pub(crate) struct SessionProof {
    /// The intent of the conversion from the press to the display.
    pub(crate) intent: ProofIntent,
    /// The display's output profile; `None` for the working space.
    display: Option<Arc<OutputProfile>>,
    /// The output intent of the document, or why it has none.
    document: Result<OutputIntent, OutputIntentError>,
    /// The proof of each press profile, by its stream.
    proofs: Mutex<HashMap<ObjectId, Arc<SoftProof>>>,
}

impl SessionProof {
    /// The proof of `doc` under `intent` on `display`; the document's
    /// press is read here, and each press's table on the first page it
    /// proofs.
    ///
    /// # Errors
    ///
    /// [`RasterError::OutputIntent`] when the document has no output
    /// intent and no page has one of its own, or its intent's profile is
    /// not a CMYK press; [`RasterError::OutputProfile`] when the display
    /// is CMYK.
    pub(crate) fn open(
        doc: &Document,
        intent: ProofIntent,
        display: Option<Arc<OutputProfile>>,
    ) -> Result<Self, RasterError> {
        let document = pdf_interp::document_output_intent(doc);
        let this = Self {
            intent,
            display,
            document,
            proofs: Mutex::new(HashMap::new()),
        };
        match &this.document {
            Ok(oi) => {
                this.proof_of(doc, oi).map(drop)?;
                let condition = oi
                    .condition
                    .as_deref()
                    .map_or_else(String::new, |c| format!(" ({c})"));
                log::info!(
                    "soft proof against the output intent /{}{condition}, {intent}",
                    oi.subtype
                );
            }
            Err(OutputIntentError::Missing) if any_page_intent(doc) => {}
            Err(e) => return Err(RasterError::OutputIntent(e.clone())),
        }
        Ok(this)
    }

    /// The proof of page `page_id`: of its own output intent, else the
    /// document's.
    ///
    /// # Errors
    ///
    /// As [`Self::open`], for the page's intent.
    pub(crate) fn for_page(
        &self,
        doc: &Document,
        page_id: ObjectId,
    ) -> Result<Arc<SoftProof>, RasterError> {
        let own = pdf_interp::page_output_intent(doc, page_id);
        let intent = own
            .as_ref()
            .map_or(self.document.as_ref(), Result::as_ref)
            .map_err(|e| RasterError::OutputIntent(e.clone()))?;
        self.proof_of(doc, intent)
    }

    /// The proof of the press of `intent`, built on first use.
    fn proof_of(
        &self,
        doc: &Document,
        intent: &OutputIntent,
    ) -> Result<Arc<SoftProof>, RasterError> {
        if let Some(proof) = self
            .proofs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&intent.profile)
        {
            return Ok(Arc::clone(proof));
        }
        let press = intent
            .press_profile(doc)
            .map_err(RasterError::OutputIntent)?;
        let proof = Arc::new(
            SoftProof::new(press, self.display.clone()).map_err(RasterError::OutputProfile)?,
        );
        Ok(Arc::clone(
            self.proofs
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(intent.profile)
                .or_insert(proof),
        ))
    }
}

/// Whether some page of `doc` carries `/OutputIntents` of its own.
fn any_page_intent(doc: &Document) -> bool {
    doc.get_pages()
        .any(|(_, id)| pdf_interp::page_output_intent(doc, id).is_some())
}
