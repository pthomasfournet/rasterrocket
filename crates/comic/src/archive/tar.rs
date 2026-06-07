//! `.cbt` (TAR) container.

use std::io::{Cursor, Read};

use super::Archive;
use crate::ComicError;

/// A TAR archive held in memory. Entries are indexed eagerly on open because
/// `tar::Archive` is a forward-only reader.
pub struct TarArchive {
    entries: Vec<(String, Vec<u8>)>,
}

impl TarArchive {
    /// Parse TAR bytes into an in-memory name→bytes table.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the stream is not a valid tar.
    pub fn open(bytes: Vec<u8>) -> Result<Self, ComicError> {
        let mut ar = tar::Archive::new(Cursor::new(bytes));
        let mut entries = Vec::new();
        let iter = ar
            .entries()
            .map_err(|e| ComicError::BadArchive(format!("tar: {e}")))?;
        for entry in iter {
            let mut e = entry.map_err(|e| ComicError::BadArchive(format!("tar: {e}")))?;
            // Skip directory entries so the name list and read_entry only ever
            // expose files (matching the 7z impl).
            if e.header().entry_type().is_dir() {
                continue;
            }
            let path = e
                .path()
                .map_err(|e| ComicError::BadArchive(format!("tar path: {e}")))?
                .to_string_lossy()
                .into_owned();
            let mut data = Vec::new();
            let _ = e
                .read_to_end(&mut data)
                .map_err(|e| ComicError::BadArchive(format!("tar read {path}: {e}")))?;
            entries.push((path, data));
        }
        Ok(Self { entries })
    }
}

impl Archive for TarArchive {
    fn entry_names(&mut self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    fn read_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.clone())
            .ok_or_else(|| ComicError::BadArchive(format!("tar: entry {name} not found")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::Archive;

    /// Build a real .cbt (tar) container in memory with the given entries.
    fn make_tar(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(u64::try_from(data.len()).expect("test fixture fits u64"));
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        builder.into_inner().unwrap()
    }

    #[test]
    fn tar_round_trip() {
        let bytes = make_tar(&[("002.png", b"two"), ("001.png", b"one")]);
        let mut ar = TarArchive::open(bytes).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["001.png", "002.png"]);
        assert_eq!(ar.read_entry("001.png").unwrap(), b"one");
        assert_eq!(ar.read_entry("002.png").unwrap(), b"two");
    }

    #[test]
    fn tar_skips_directory_entries() {
        // A tar with a directory entry plus a file: only the file is exposed.
        let mut builder = tar::Builder::new(Vec::new());
        let mut dir = tar::Header::new_gnu();
        dir.set_entry_type(tar::EntryType::Directory);
        dir.set_size(0);
        dir.set_cksum();
        builder.append_data(&mut dir, "pages/", &[][..]).unwrap();
        let mut file = tar::Header::new_gnu();
        file.set_size(3);
        file.set_cksum();
        builder
            .append_data(&mut file, "pages/01.png", &b"img"[..])
            .unwrap();
        let bytes = builder.into_inner().unwrap();

        let mut ar = TarArchive::open(bytes).unwrap();
        assert_eq!(
            ar.entry_names(),
            vec!["pages/01.png"],
            "directory entry must not appear in the name list"
        );
    }

    #[test]
    fn tar_missing_entry_is_bad_archive() {
        let bytes = make_tar(&[("a.png", b"a")]);
        let mut ar = TarArchive::open(bytes).unwrap();
        assert!(matches!(
            ar.read_entry("nope.png"),
            Err(ComicError::BadArchive(_))
        ));
    }
}
