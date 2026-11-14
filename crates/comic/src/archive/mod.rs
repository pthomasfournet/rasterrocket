//! Archive containers behind a common trait.

mod rar;
mod sevenz;
mod tar;
mod zip;

use std::collections::HashMap;
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

/// Default of [`ComicOptions::max_archive_bytes`](crate::ComicOptions::max_archive_bytes):
/// 4 GiB, the most decompressed bytes one archive may yield across all of
/// its entries.
///
/// Every page-image entry is held decompressed from the opening of the
/// archive until its pages are rendered, so the sum over entries, not the
/// size of one entry, bounds the memory an archive costs.  Eight entries at
/// the per-entry cap fill it; a comic of a thousand 4 MB pages uses a
/// thousandth of it.
pub const DEFAULT_MAX_ARCHIVE_BYTES: u64 = 4 << 30;

/// The decompressed bytes an archive may still yield: one budget charged by
/// every entry read from the archive, whatever its container format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArchiveBudget {
    /// The bound on the sum of all entries' decompressed sizes.
    limit: u64,
    /// The bytes yielded so far; never above `limit`.
    used: u64,
}

impl ArchiveBudget {
    /// A budget of `limit` decompressed bytes, none used.
    pub(crate) const fn new(limit: u64) -> Self {
        Self { limit, used: 0 }
    }

    /// The bound on the sum of all entries' decompressed sizes.
    pub(crate) const fn limit(&self) -> u64 {
        self.limit
    }

    /// The error of an archive that inflates past this budget.
    pub(crate) const fn exceeded(&self) -> ComicError {
        ComicError::ArchiveTooLarge { limit: self.limit }
    }
}

/// Read `reader` to a `Vec`, refusing to buffer more than [`MAX_ENTRY_BYTES`]
/// or more than remains of `budget`, which the entry's size is charged to.
///
/// The capacity hint is clamped to the cap so a lying size field cannot
/// force a giant up-front allocation, and the read stops one byte past the
/// smaller bound, so neither a bomb entry nor a run of entries each under
/// the per-entry cap is ever buffered past its bound.  `entry` names the
/// entry for the error message.
///
/// # Errors
/// [`ComicError::BadArchive`] if the entry yields more than the per-entry
/// cap or the underlying read fails; [`ComicError::ArchiveTooLarge`] if it
/// yields more than remains of the archive budget.
pub(crate) fn read_entry_capped<R: Read>(
    reader: R,
    entry: &str,
    size_hint: u64,
    budget: &mut ArchiveBudget,
) -> Result<Vec<u8>, ComicError> {
    read_capped(reader, entry, size_hint, MAX_ENTRY_BYTES, budget)
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
    budget: &mut ArchiveBudget,
) -> Result<Vec<u8>, ComicError> {
    let remaining = budget.limit - budget.used;
    let bound = limit.min(remaining);
    // Cap the pre-allocation: trust the hint only up to the bound.
    let cap = size_hint.min(bound);
    let mut out = Vec::with_capacity(usize::try_from(cap).unwrap_or(0));
    // Read one byte past the bound so an over-bound entry is detected.
    let read = Read::take(&mut reader, bound.saturating_add(1))
        .read_to_end(&mut out)
        .map_err(|e| ComicError::BadArchive(format!("read {entry}: {e}")))? as u64;
    if read > limit {
        return Err(ComicError::BadArchive(format!(
            "entry {entry} exceeds the {limit}-byte per-entry size limit \
             (possible decompression bomb)"
        )));
    }
    if read > remaining {
        return Err(budget.exceeded());
    }
    budget.used += read;
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
    /// already taken; [`ComicError::ArchiveTooLarge`] if reading it takes
    /// the archive past its decompressed-size budget.
    fn take_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError>;
}

/// An entry's bytes, or the error of an entry that failed to decode on its
/// own.
pub(super) type EntryBytes = Result<Vec<u8>, ComicError>;

/// The entries of a container decoded in full on open, in stored order, for
/// the formats whose readers cannot seek to an entry by name.
pub(super) struct DecodedEntries {
    /// Container format, for error messages.
    format: &'static str,
    /// Each entry's name and its bytes or the error that replaced them;
    /// `None` once taken.
    entries: Vec<(String, Option<EntryBytes>)>,
    /// The position in `entries` of the first entry of each name.
    index: HashMap<String, usize>,
}

impl DecodedEntries {
    /// The decoded `entries` of a `format` container: each entry's bytes,
    /// or the error of an entry that failed to decode on its own.
    pub(super) fn new(format: &'static str, entries: Vec<(String, EntryBytes)>) -> Self {
        // Collected last to first, so the first entry of a name is the one
        // inserted last and kept.
        let index = entries
            .iter()
            .enumerate()
            .rev()
            .map(|(at, (name, _))| (name.clone(), at))
            .collect();
        Self {
            format,
            entries: entries.into_iter().map(|(n, d)| (n, Some(d))).collect(),
            index,
        }
    }

    /// All entry names, in stored order.
    pub(super) fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Move the bytes of entry `name` out: of the first entry of that name
    /// when several share it.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if no entry has that name or its bytes
    /// were already taken; the entry's own error if it failed to decode.
    pub(super) fn take(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        let format = self.format;
        let at = *self
            .index
            .get(name)
            .ok_or_else(|| ComicError::BadArchive(format!("{format}: entry {name} not found")))?;
        self.entries[at]
            .1
            .take()
            .ok_or_else(|| ComicError::BadArchive(format!("{format}: entry {name} already read")))?
    }
}

/// Choose an impl by file extension, taking the already-read archive `bytes`;
/// every entry the archive yields is charged to `budget`.
///
/// # Errors
/// [`ComicError::BadArchive`] for an unknown extension or a container that
/// fails to parse.
pub(crate) fn open_archive_from_ext(
    name: &str,
    bytes: Vec<u8>,
    budget: ArchiveBudget,
) -> Result<Box<dyn Archive>, ComicError> {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "cbz" => Ok(Box::new(zip::ZipArchive::open(bytes, budget)?)),
        "cb7" => Ok(Box::new(sevenz::SevenzArchive::open(bytes, budget)?)),
        "cbt" => Ok(Box::new(tar::TarArchive::open(bytes, budget)?)),
        "cbr" => Ok(Box::new(rar::RarArchive::open(bytes, budget)?)),
        other => Err(ComicError::BadArchive(format!(
            "unrecognised comic extension {other:?} (expected cbz, cb7, cbt or cbr)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_cbz;

    /// A budget no test reaches.
    fn ample() -> ArchiveBudget {
        ArchiveBudget::new(u64::MAX)
    }

    #[test]
    fn an_unknown_extension_is_a_bad_archive() {
        for name in ["foo.cbx", "foo.txt"] {
            let res = open_archive_from_ext(name, Vec::new(), ample());
            assert!(
                matches!(res, Err(crate::ComicError::BadArchive(_))),
                "{name}"
            );
        }
    }

    #[test]
    fn zip_entries_round_trip() {
        let bytes = make_cbz(&[("b.png", b"B"), ("a.png", b"A"), ("ComicInfo.xml", b"<x/>")]);
        let mut ar = open_archive_from_ext("x.cbz", bytes, ample()).unwrap();
        let mut names = ar.entry_names();
        names.sort();
        assert_eq!(names, vec!["ComicInfo.xml", "a.png", "b.png"]);
        assert_eq!(ar.take_entry("a.png").unwrap(), b"A");
    }

    #[test]
    fn zip_entries_share_the_archive_budget() {
        // Three 4 KiB entries, each well under the per-entry cap, against a
        // 10 000-byte archive budget: the third take overdraws it.
        let page = [0u8; 4096];
        let bytes = make_cbz(&[("1.png", &page), ("2.png", &page), ("3.png", &page)]);
        let mut ar = open_archive_from_ext("x.cbz", bytes, ArchiveBudget::new(10_000)).unwrap();
        assert!(ar.take_entry("1.png").is_ok());
        assert!(ar.take_entry("2.png").is_ok());
        assert!(matches!(
            ar.take_entry("3.png"),
            Err(ComicError::ArchiveTooLarge { limit: 10_000 })
        ));
    }

    #[test]
    fn over_limit_rejected() {
        // An infinite reader against a tiny injected limit: the cap fires after
        // reading just past the limit, so no large allocation happens.
        let r = std::io::repeat(0u8);
        assert!(matches!(
            super::read_capped(r, "bomb.jpg", u64::MAX, 16, &mut ample()),
            Err(crate::ComicError::BadArchive(_))
        ));
    }

    #[test]
    fn over_budget_rejected_and_bounded() {
        // An infinite reader under a generous per-entry limit stops one byte
        // past the remaining budget, not at the per-entry limit.
        let mut budget = ArchiveBudget::new(32);
        let out = super::read_capped(&[1u8; 20][..], "a.jpg", 20, 1024, &mut budget).unwrap();
        assert_eq!(out.len(), 20);
        assert!(matches!(
            super::read_capped(std::io::repeat(0u8), "b.jpg", 0, u64::MAX, &mut budget),
            Err(ComicError::ArchiveTooLarge { limit: 32 })
        ));
        // Exactly the remainder still fits.
        let out = super::read_capped(&[1u8; 12][..], "c.jpg", 12, 1024, &mut budget).unwrap();
        assert_eq!(out.len(), 12);
        assert!(super::read_capped(&[][..], "d.jpg", 0, 1024, &mut budget).is_ok());
        assert!(super::read_capped(&[1u8][..], "e.jpg", 1, 1024, &mut budget).is_err());
    }

    #[test]
    fn under_limit_ok() {
        let r = &b"hello"[..];
        let out = super::read_capped(r, "ok.jpg", 5, 1024, &mut ample()).unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn at_limit_ok() {
        // Exactly `limit` bytes must pass; only strictly-more is rejected.
        let data = [0u8; 16];
        let out = super::read_capped(&data[..], "edge.jpg", 16, 16, &mut ample()).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn capped_clamps_lying_size_hint() {
        // A size hint far above the limit must not blow up the up-front alloc;
        // the read still succeeds for an under-limit body.
        let r = &b"hi"[..];
        let out = super::read_capped(r, "liar.jpg", u64::MAX, 1024, &mut ample()).unwrap();
        assert_eq!(out, b"hi");
    }

    #[test]
    fn decoded_entries_take_the_first_of_a_name() {
        let mut entries = DecodedEntries::new(
            "test",
            vec![
                ("a".to_owned(), Ok(b"first".to_vec())),
                (
                    "b".to_owned(),
                    Err(ComicError::BadArchive("damaged".to_owned())),
                ),
                ("a".to_owned(), Ok(b"second".to_vec())),
            ],
        );
        assert_eq!(entries.take("a").unwrap(), b"first");
        assert!(
            matches!(entries.take("a"), Err(ComicError::BadArchive(m)) if m.contains("already read"))
        );
        assert!(matches!(entries.take("b"), Err(ComicError::BadArchive(m)) if m == "damaged"));
        assert!(
            matches!(entries.take("c"), Err(ComicError::BadArchive(m)) if m.contains("not found"))
        );
    }

    #[test]
    fn taking_every_entry_of_a_large_table_is_linear() {
        // 50 000 entries taken in reverse: a linear search per take makes
        // 1.25·10^9 string comparisons, seconds even optimised; the index
        // makes it milliseconds.
        let n = 50_000;
        let names: Vec<String> = (0..n).map(|i| format!("{i:07}.png")).collect();
        let mut entries = DecodedEntries::new(
            "test",
            names
                .iter()
                .map(|name| (name.clone(), Ok(Vec::new())))
                .collect(),
        );
        let start = std::time::Instant::now();
        for name in names.iter().rev() {
            assert!(entries.take(name).is_ok());
        }
        let elapsed = start.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(2), "{elapsed:?}");
    }
}
