//! `.cbz` (ZIP) container.

use std::io::Cursor;

use super::Archive;
use crate::ComicError;

/// A ZIP archive held in memory.
pub(super) struct ZipArchive {
    inner: zip::ZipArchive<Cursor<Vec<u8>>>,
}

impl ZipArchive {
    /// Parse ZIP bytes.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the central directory cannot be read.
    pub(super) fn open(bytes: Vec<u8>) -> Result<Self, ComicError> {
        let inner = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| ComicError::BadArchive(format!("zip: {e}")))?;
        Ok(Self { inner })
    }
}

impl Archive for ZipArchive {
    fn entry_names(&mut self) -> Vec<String> {
        (0..self.inner.len())
            .filter_map(|i| self.inner.by_index(i).ok().map(|f| f.name().to_owned()))
            .collect()
    }

    fn read_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        let f = self
            .inner
            .by_name(name)
            .map_err(|e| ComicError::BadArchive(format!("zip entry {name}: {e}")))?;
        // `f.size()` is the attacker-controlled uncompressed size from the
        // central directory; the helper clamps it and bounds the inflated read.
        let size_hint = f.size();
        super::read_entry_capped(f, name, size_hint)
    }
}
