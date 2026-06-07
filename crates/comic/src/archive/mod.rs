//! Archive containers behind a common trait.

mod sevenz;
mod tar;
mod zip;

use std::path::Path;

use crate::ComicError;

/// A read-only comic-archive container. Implementations hold the whole archive
/// in memory (comics are image-sized, not video-sized) and expose entries by
/// name. Ordering/filtering is the caller's job (see `crate::order`).
pub trait Archive {
    /// All entry names in the archive, in stored order (unsorted, unfiltered).
    fn entry_names(&mut self) -> Vec<String>;
    /// Read one entry's raw bytes by name.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the entry is missing or unreadable.
    fn read_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError>;
}

/// Choose an impl by file extension, taking the already-read archive `bytes`.
///
/// # Errors
/// [`ComicError::RarUnsupported`] for `.cbr`; [`ComicError::BadArchive`] for an
/// unknown extension or a container that fails to parse.
pub fn open_archive_from_ext(name: &str, bytes: Vec<u8>) -> Result<Box<dyn Archive>, ComicError> {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "cbz" => Ok(Box::new(zip::ZipArchive::open(bytes)?)),
        "cb7" => Ok(Box::new(sevenz::SevenzArchive::open(bytes)?)),
        "cbt" => Ok(Box::new(tar::TarArchive::open(bytes)?)),
        "cbr" => Err(ComicError::RarUnsupported),
        other => Err(ComicError::BadArchive(format!(
            "unrecognised comic extension {other:?} (expected cbz/cb7/cbt; cbr is unsupported)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    /// Build a .cbz (ZIP) in memory with the given (name, bytes) entries.
    ///
    /// `::zip` (crate-root path) refers to the external crate, not the sibling
    /// `zip` submodule that `super::*` would otherwise resolve to here.
    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = ::zip::ZipWriter::new(&mut buf);
            let opts: ::zip::write::FileOptions<'_, ()> = ::zip::write::FileOptions::default();
            for (name, data) in entries {
                w.start_file(*name, opts).unwrap();
                w.write_all(data).unwrap();
            }
            let _ = w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn cbr_path_is_rar_unsupported() {
        let res = open_archive_from_ext("foo.cbr", Vec::new());
        assert!(matches!(res, Err(crate::ComicError::RarUnsupported)));
    }

    #[test]
    fn zip_entries_round_trip() {
        let bytes = make_zip(&[("b.png", b"B"), ("a.png", b"A"), ("ComicInfo.xml", b"<x/>")]);
        let mut ar = open_archive_from_ext("x.cbz", bytes).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["ComicInfo.xml", "a.png", "b.png"]);
        assert_eq!(ar.read_entry("a.png").unwrap(), b"A");
    }

    #[test]
    fn unknown_extension_is_bad_archive() {
        let res = open_archive_from_ext("foo.txt", Vec::new());
        assert!(matches!(res, Err(crate::ComicError::BadArchive(_))));
    }
}
