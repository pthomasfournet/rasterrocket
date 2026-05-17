//! Google Cloud Vision input optimization.
//!
//! Produces an upload-ready grayscale JPEG payload guaranteed to fit GCV's
//! binding limit — 10 MB of base64 inside the `images:annotate` JSON request
//! (NOT the 20 MB raw-file limit) — decided deterministically and locally.
//! No HTTP, auth, or batching: this crate renders pixels, it is not a GCV
//! API client.

use base64::Engine as _;

/// Budget for a single GCV `images:annotate` request.
#[derive(Debug, Clone)]
pub struct GcvBudget {
    /// Max length of the base64 string. GCV rejects JSON requests whose
    /// inline `content` exceeds 10 MB of base64.
    pub max_base64_bytes: usize,
    /// Lowest JPEG quality the step-down may use before falling back to
    /// downscaling. Below this, OCR accuracy degrades unacceptably.
    pub min_quality: u8,
    /// Starting JPEG quality.
    pub start_quality: u8,
}

impl Default for GcvBudget {
    fn default() -> Self {
        Self {
            max_base64_bytes: 10 * 1024 * 1024,
            min_quality: 60,
            start_quality: 90,
        }
    }
}

/// A GCV-ready encoded page.
#[derive(Debug, Clone)]
pub struct GcvImage {
    /// Fitted baseline grayscale JPEG bytes — guaranteed within budget.
    /// The universal artifact: write to disk, upload to GCS for
    /// `files:asyncBatchAnnotate`, or feed any other consumer.
    pub jpeg: Vec<u8>,
    /// Final JPEG quality used by the fitting algorithm.
    pub quality: u8,
    /// Final pixel dimensions (may be downscaled from the page if the
    /// quality floor was hit).
    pub width: u32,
    /// Final pixel height.
    pub height: u32,
}

impl GcvImage {
    /// base64 of `jpeg`, ready to drop into the GCV `images:annotate`
    /// request body. The fitting algorithm measured this exact length
    /// against the budget, so the result is within GCV's JSON ceiling.
    #[must_use]
    pub fn to_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.jpeg)
    }
}

/// Why a page could not be fitted into a [`GcvBudget`].
#[derive(Debug)]
pub enum GcvError {
    /// The page cannot fit the budget even at `min_quality` and the
    /// resolution floor, or it would exceed GCV's 75 MP OCR limit even
    /// after maximum downscale. The caller decides whether to split or skip
    /// — never an over-budget payload.
    Unfittable {
        /// Smallest base64 length achieved before giving up.
        smallest_base64: usize,
        /// The budget that could not be met.
        budget: usize,
    },
    /// The underlying JPEG codec failed.
    Encode(encode::EncodeError),
}

impl std::fmt::Display for GcvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unfittable {
                smallest_base64,
                budget,
            } => write!(
                f,
                "page does not fit GCV budget: smallest {smallest_base64} B base64 > {budget} B"
            ),
            Self::Encode(e) => write!(f, "JPEG encode failed: {e}"),
        }
    }
}

impl std::error::Error for GcvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(e) => Some(e),
            Self::Unfittable { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_match_gcv_limits() {
        let b = GcvBudget::default();
        assert_eq!(b.max_base64_bytes, 10 * 1024 * 1024, "GCV JSON ceiling");
        assert_eq!(b.min_quality, 60);
        assert_eq!(b.start_quality, 90);
    }

    #[test]
    fn to_base64_roundtrips() {
        let img = GcvImage {
            jpeg: vec![0xFF, 0xD8, 0xFF, 0xD9],
            quality: 90,
            width: 2,
            height: 2,
        };
        let b64 = img.to_base64();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .unwrap();
        assert_eq!(decoded, img.jpeg);
    }
}
