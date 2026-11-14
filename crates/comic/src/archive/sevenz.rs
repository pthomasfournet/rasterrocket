//! `.cb7` (7-Zip) container.

use std::io::Cursor;

use sevenz_rust2::{ArchiveReader, EntryDecision, Password};

use super::{Archive, ArchiveBudget, DecodedEntries};
use crate::ComicError;

/// The largest allocation, in KiB, the archive's header and decoders (LZMA
/// dictionary, the `PPMd` model) may make: 1 GiB.  An archive declaring more is
/// refused when it is opened or its block decoded, before the allocation.
const MAX_DECODER_KIB: usize = 1 << 20;

/// A 7-Zip archive decoded eagerly into an in-memory name→bytes table.
pub(super) struct SevenzArchive {
    entries: DecodedEntries,
}

impl SevenzArchive {
    /// Parse 7z bytes into an in-memory table, every entry charged to
    /// `budget`.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the archive cannot be opened/decoded,
    /// including when it needs more than [`MAX_DECODER_KIB`] of decoder
    /// memory; [`ComicError::ArchiveTooLarge`] if its entries exceed
    /// `budget`.
    pub(super) fn open(bytes: Vec<u8>, mut budget: ArchiveBudget) -> Result<Self, ComicError> {
        let mut reader = ArchiveReader::new_with_memory_limit(
            Cursor::new(bytes),
            Password::empty(),
            MAX_DECODER_KIB,
        )
        .map_err(|e| ComicError::BadArchive(format!("7z: {e}")))?;
        let mut entries = Vec::new();
        // The read helper returns `ComicError`, which cannot be converted
        // into the archive crate's error type, so an entry's read error (over
        // a size limit, or corrupt data) is held here, iteration stopped, and
        // the error returned after the walk.
        let mut entry_error: Option<ComicError> = None;
        reader
            .for_each_entries_with_decision(|entry, rdr| {
                if entry.is_directory() {
                    return Ok(EntryDecision::Continue);
                }
                let name = entry.name().to_owned();
                // `entry.size()` is the attacker-controlled uncompressed size;
                // the helper clamps it and bounds the inflated read.
                match super::read_entry_capped(&mut *rdr, &name, entry.size(), &mut budget) {
                    Ok(data) => {
                        entries.push((name, Ok(data)));
                        Ok(EntryDecision::Continue)
                    }
                    Err(e) => {
                        entry_error = Some(e);
                        Ok(EntryDecision::Stop)
                    }
                }
            })
            .map_err(|e| ComicError::BadArchive(format!("7z: {e}")))?;
        if let Some(e) = entry_error {
            return Err(e);
        }
        Ok(Self {
            entries: DecodedEntries::new("7z", entries),
        })
    }
}

impl Archive for SevenzArchive {
    fn entry_names(&mut self) -> Vec<String> {
        self.entries.names()
    }

    fn take_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        self.entries.take(name)
    }
}

#[cfg(test)]
mod tests {
    use sevenz_rust2::{ArchiveEntry, ArchiveWriter};

    use super::*;
    use crate::archive::Archive;

    /// A budget no test reaches.
    fn ample() -> ArchiveBudget {
        ArchiveBudget::new(u64::MAX)
    }

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
        let mut ar = SevenzArchive::open(bytes, ample()).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["001.png", "002.png"]);
        assert_eq!(ar.take_entry("001.png").unwrap(), b"one");
        assert_eq!(ar.take_entry("002.png").unwrap(), b"two");
    }

    #[test]
    fn sevenz_missing_entry_is_bad_archive() {
        let bytes = make_7z(&[("a.png", b"a")]);
        let mut ar = SevenzArchive::open(bytes, ample()).unwrap();
        assert!(matches!(
            ar.take_entry("nope.png"),
            Err(ComicError::BadArchive(_))
        ));
    }

    #[test]
    fn sevenz_garbage_is_bad_archive() {
        assert!(matches!(
            SevenzArchive::open(b"not a 7z archive".to_vec(), ample()),
            Err(ComicError::BadArchive(_))
        ));
    }

    #[test]
    fn non_solid_entries_share_the_archive_budget() {
        // Each entry is its own LZMA2 folder (non-solid), so every entry is
        // a fresh 1000:1 inflation and only the sum bounds the archive: 200
        // zero entries at the 512 MiB per-entry cap fit in about 15 MB and
        // ask for 100 GiB.  Scaled down: 200 entries of 64 KiB against a
        // 1 MiB budget stop at the seventeenth entry.
        let zeros = vec![0u8; 64 * 1024];
        let names: Vec<String> = (0..200).map(|i| format!("{i:03}.png")).collect();
        let entries: Vec<(&str, &[u8])> = names.iter().map(|n| (n.as_str(), &zeros[..])).collect();
        let bytes = make_7z(&entries);
        assert!(bytes.len() < zeros.len(), "{} bytes", bytes.len());
        assert!(matches!(
            SevenzArchive::open(bytes.clone(), ArchiveBudget::new(1 << 20)),
            Err(ComicError::ArchiveTooLarge { limit: 1_048_576 })
        ));
        // The whole archive fits a budget of its total size.
        let total = 200 * 64 * 1024;
        assert!(SevenzArchive::open(bytes, ArchiveBudget::new(total)).is_ok());
    }
}
