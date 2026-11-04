//! The document a per-document store serves.
//!
//! A store keyed by object number holds the objects of one document only:
//! object `12 0 R` of another document is a different object.  A
//! [`DocumentBinding`] records the first document a store is given to and
//! answers, for every later one, whether it is that document.

use std::sync::OnceLock;

/// The [`pdf::Document::instance_id`] of the document a store serves, set
/// by the first [`Self::bind`].
#[derive(Debug, Default)]
pub(crate) struct DocumentBinding(OnceLock<u64>);

impl DocumentBinding {
    /// Bind to `doc` if no document is bound yet; whether `doc` is the
    /// bound document.
    pub(crate) fn bind(&self, doc: &pdf::Document) -> bool {
        let id = doc.instance_id();
        *self.0.get_or_init(|| id) == id
    }
}
