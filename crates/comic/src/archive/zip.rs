//! `.cbz` (ZIP) container.

use std::io::Cursor;

use super::{Archive, ArchiveBudget};
use crate::ComicError;

/// A ZIP archive held in memory; its entries are inflated as they are taken.
pub(super) struct ZipArchive {
    inner: zip::ZipArchive<Cursor<Vec<u8>>>,
    /// Charged by every entry taken.
    budget: ArchiveBudget,
}

impl ZipArchive {
    /// Parse ZIP bytes; the entries taken later are charged to `budget`.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the central directory cannot be read.
    pub(super) fn open(bytes: Vec<u8>, budget: ArchiveBudget) -> Result<Self, ComicError> {
        let inner = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| ComicError::BadArchive(format!("zip: {e}")))?;
        Ok(Self { inner, budget })
    }
}

impl Archive for ZipArchive {
    fn entry_names(&mut self) -> Vec<String> {
        (0..self.inner.len())
            .filter_map(|i| {
                let entry = self
                    .inner
                    .by_index(i)
                    .map_err(|e| log::warn!("comic: zip entry #{i} unreadable, skipped: {e}"))
                    .ok()?;
                entry
                    .name()
                    .map(std::borrow::Cow::into_owned)
                    .map_err(|e| log::warn!("comic: zip entry #{i} name undecodable, skipped: {e}"))
                    .ok()
            })
            .collect()
    }

    fn take_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        let f = self
            .inner
            .by_name(name)
            .map_err(|e| ComicError::BadArchive(format!("zip entry {name}: {e}")))?;
        // `f.size()` is the attacker-controlled uncompressed size from the
        // central directory; the helper clamps it and bounds the inflated read.
        let size_hint = f.size();
        super::read_entry_capped(f, name, size_hint, &mut self.budget)
    }
}
