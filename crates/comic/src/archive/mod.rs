//! Archive containers behind a common trait.

mod sevenz;
mod tar;
mod zip;

use std::io::Read;
use std::path::Path;

use crate::ComicError;

/// Maximum decompressed size accepted for a single archive entry.
///
/// A decompression bomb declares a tiny compressed entry that inflates to many
/// gigabytes; without a cap, decoding it exhausts memory. 512 MiB comfortably
/// admits any legitimate comic page image (a 600-megapixel page — the render
/// size ceiling — encodes to well under this) or a reasonable embedded PDF,
/// while stopping a multi-gigabyte bomb.
pub(crate) const MAX_ENTRY_BYTES: u64 = 512 * 1024 * 1024;

/// Read `reader` to a `Vec`, refusing to buffer more than [`MAX_ENTRY_BYTES`].
///
/// Bounds the decompressed size of a single entry so a decompression bomb
/// cannot exhaust memory. The capacity hint is clamped to the cap so a lying
/// size field cannot force a giant up-front allocation. `entry` names the entry
/// for the error message.
///
/// # Errors
/// [`ComicError::BadArchive`] if the entry yields more than the cap, or if the
/// underlying read fails.
pub(crate) fn read_entry_capped<R: Read>(
    reader: R,
    entry: &str,
    size_hint: u64,
) -> Result<Vec<u8>, ComicError> {
    read_capped(reader, entry, size_hint, MAX_ENTRY_BYTES)
}

/// Limit-parameterised core of [`read_entry_capped`].
///
/// `limit` is injected so tests can exercise the bounding logic with a tiny cap
/// instead of moving the production 512 MiB. The public path always passes
/// [`MAX_ENTRY_BYTES`].
fn read_capped<R: Read>(
    mut reader: R,
    entry: &str,
    size_hint: u64,
    limit: u64,
) -> Result<Vec<u8>, ComicError> {
    // Cap the pre-allocation: trust the hint only up to the limit.
    let cap = size_hint.min(limit);
    let mut out = Vec::with_capacity(usize::try_from(cap).unwrap_or(0));
    // Read one byte past the limit so we can detect an over-limit entry.
    let read = Read::take(&mut reader, limit.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|e| ComicError::BadArchive(format!("read {entry}: {e}")))?;
    if read as u64 > limit {
        return Err(ComicError::BadArchive(format!(
            "entry {entry} exceeds the {limit}-byte per-entry size limit \
             (possible decompression bomb)"
        )));
    }
    Ok(out)
}

/// A read-only comic-archive container. Implementations hold the whole archive
/// in memory (comics are image-sized, not video-sized) and expose entries by
/// name. Ordering/filtering is the caller's job (see `crate::order`).
pub(crate) trait Archive {
    /// All entry names in the archive, in stored order (unsorted, unfiltered).
    fn entry_names(&mut self) -> Vec<String>;
    /// Take one entry's raw bytes by name.  An implementation that holds the
    /// decoded entries moves the bytes out, so each entry is taken at most
    /// once.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the entry is missing, unreadable or
    /// already taken.
    fn take_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError>;
}

/// The entries of a container decoded in full on open, in stored order, for
/// the formats whose readers cannot seek to an entry by name.
pub(super) struct DecodedEntries {
    /// Container format, for error messages.
    format: &'static str,
    /// Each entry's name and bytes; `None` once the bytes have been taken.
    entries: Vec<(String, Option<Vec<u8>>)>,
}

impl DecodedEntries {
    /// The decoded `entries` of a `format` container.
    pub(super) fn new(format: &'static str, entries: Vec<(String, Vec<u8>)>) -> Self {
        Self {
            format,
            entries: entries.into_iter().map(|(n, d)| (n, Some(d))).collect(),
        }
    }

    /// All entry names, in stored order.
    pub(super) fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Move the bytes of entry `name` out.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if no entry has that name or its bytes
    /// were already taken.
    pub(super) fn take(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        let format = self.format;
        let (_, data) = self
            .entries
            .iter_mut()
            .find(|(n, _)| n == name)
            .ok_or_else(|| ComicError::BadArchive(format!("{format}: entry {name} not found")))?;
        data.take()
            .ok_or_else(|| ComicError::BadArchive(format!("{format}: entry {name} already read")))
    }
}

/// Choose an impl by file extension, taking the already-read archive `bytes`.
///
/// # Errors
/// [`ComicError::RarUnsupported`] for `.cbr`; [`ComicError::BadArchive`] for an
/// unknown extension or a container that fails to parse.
pub(crate) fn open_archive_from_ext(
    name: &str,
    bytes: Vec<u8>,
) -> Result<Box<dyn Archive>, ComicError> {
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
    use super::*;
    use crate::test_helpers::make_cbz;

    #[test]
    fn cbr_path_is_rar_unsupported() {
        let res = open_archive_from_ext("foo.cbr", Vec::new());
        assert!(matches!(res, Err(crate::ComicError::RarUnsupported)));
    }

    #[test]
    fn zip_entries_round_trip() {
        let bytes = make_cbz(&[("b.png", b"B"), ("a.png", b"A"), ("ComicInfo.xml", b"<x/>")]);
        let mut ar = open_archive_from_ext("x.cbz", bytes).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["ComicInfo.xml", "a.png", "b.png"]);
        assert_eq!(ar.take_entry("a.png").unwrap(), b"A");
    }

    #[test]
    fn unknown_extension_is_bad_archive() {
        let res = open_archive_from_ext("foo.txt", Vec::new());
        assert!(matches!(res, Err(crate::ComicError::BadArchive(_))));
    }

    #[test]
    fn over_limit_rejected() {
        // An infinite reader against a tiny injected limit: the cap fires after
        // reading just past the limit, so no large allocation happens.
        let r = std::io::repeat(0u8);
        assert!(matches!(
            super::read_capped(r, "bomb.jpg", u64::MAX, 16),
            Err(crate::ComicError::BadArchive(_))
        ));
    }

    #[test]
    fn under_limit_ok() {
        let r = &b"hello"[..];
        let out = super::read_capped(r, "ok.jpg", 5, 1024).unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn at_limit_ok() {
        // Exactly `limit` bytes must pass; only strictly-more is rejected.
        let data = [0u8; 16];
        let out = super::read_capped(&data[..], "edge.jpg", 16, 16).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn capped_clamps_lying_size_hint() {
        // A size hint far above the limit must not blow up the up-front alloc;
        // the read still succeeds for an under-limit body.
        let r = &b"hi"[..];
        let out = super::read_capped(r, "liar.jpg", u64::MAX, 1024).unwrap();
        assert_eq!(out, b"hi");
    }
}
