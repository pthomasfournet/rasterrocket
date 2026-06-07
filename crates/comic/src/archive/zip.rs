//! `.cbz` (ZIP) container.

use std::io::{Cursor, Read};

use super::Archive;
use crate::ComicError;

/// A ZIP archive held in memory.
pub struct ZipArchive {
    inner: zip::ZipArchive<Cursor<Vec<u8>>>,
}

impl ZipArchive {
    /// Parse ZIP bytes.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the central directory cannot be read.
    pub fn open(bytes: Vec<u8>) -> Result<Self, ComicError> {
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
        let mut f = self
            .inner
            .by_name(name)
            .map_err(|e| ComicError::BadArchive(format!("zip entry {name}: {e}")))?;
        let mut out = Vec::with_capacity(usize::try_from(f.size()).unwrap_or(0));
        let _ = f
            .read_to_end(&mut out)
            .map_err(|e| ComicError::BadArchive(format!("zip read {name}: {e}")))?;
        Ok(out)
    }
}
