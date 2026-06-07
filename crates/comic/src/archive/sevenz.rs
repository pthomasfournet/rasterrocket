//! `.cb7` (7-Zip) container.

use std::io::Cursor;

use sevenz_rust2::{ArchiveReader, Password};

use super::Archive;
use crate::ComicError;

/// A 7-Zip archive decoded eagerly into an in-memory name→bytes table.
pub struct SevenzArchive {
    entries: Vec<(String, Vec<u8>)>,
}

impl SevenzArchive {
    /// Parse 7z bytes into an in-memory table.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the archive cannot be opened/decoded.
    pub fn open(bytes: Vec<u8>) -> Result<Self, ComicError> {
        let mut reader = ArchiveReader::new(Cursor::new(bytes), Password::empty())
            .map_err(|e| ComicError::BadArchive(format!("7z: {e}")))?;
        let mut entries = Vec::new();
        reader
            .for_each_entries(|entry, rdr| {
                if entry.is_directory() {
                    return Ok(true);
                }
                let mut data = Vec::with_capacity(usize::try_from(entry.size()).unwrap_or(0));
                let _ = rdr.read_to_end(&mut data)?;
                entries.push((entry.name().to_owned(), data));
                Ok(true)
            })
            .map_err(|e| ComicError::BadArchive(format!("7z: {e}")))?;
        Ok(Self { entries })
    }
}

impl Archive for SevenzArchive {
    fn entry_names(&mut self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    fn read_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| ComicError::BadArchive(format!("7z: entry {name} not found")))
    }
}

#[cfg(test)]
mod tests {
    use sevenz_rust2::{ArchiveEntry, ArchiveWriter};

    use super::*;
    use crate::archive::Archive;

    /// Build a real .cb7 (7z) container in memory with the given entries.
    fn make_7z(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = ArchiveWriter::new(Cursor::new(Vec::new())).unwrap();
        w.set_encrypt_header(false);
        for (name, data) in entries {
            let _ = w
                .push_archive_entry(
                    ArchiveEntry::new_file(name),
                    Some(Cursor::new(data.to_vec())),
                )
                .unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn sevenz_round_trip() {
        let bytes = make_7z(&[("002.png", b"two"), ("001.png", b"one")]);
        let mut ar = SevenzArchive::open(bytes).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["001.png", "002.png"]);
        assert_eq!(ar.read_entry("001.png").unwrap(), b"one");
        assert_eq!(ar.read_entry("002.png").unwrap(), b"two");
    }

    #[test]
    fn sevenz_missing_entry_is_bad_archive() {
        let bytes = make_7z(&[("a.png", b"a")]);
        let mut ar = SevenzArchive::open(bytes).unwrap();
        assert!(matches!(
            ar.read_entry("nope.png"),
            Err(ComicError::BadArchive(_))
        ));
    }

    #[test]
    fn sevenz_garbage_is_bad_archive() {
        assert!(matches!(
            SevenzArchive::open(b"not a 7z archive".to_vec()),
            Err(ComicError::BadArchive(_))
        ));
    }
}
