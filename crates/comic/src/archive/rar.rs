//! `.cbr` (RAR) container, RAR 1.3 through RAR 7.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use rars::{ArchiveReadOptions, ArchiveReader};

use super::{DecodedEntries, MAX_ENTRY_BYTES};
use crate::ComicError;

/// The largest dictionary, in bytes, a RAR 5 or later member may declare:
/// 1 GiB, the decoder allocation bound of the `.cb7` reader.  A member
/// declaring more is refused before its dictionary is allocated.
const MAX_DICTIONARY_BYTES: u64 = 1 << 30;

/// A RAR archive decoded eagerly into an in-memory name→bytes table.
///
/// Every member is decoded in one pass on open: in a solid archive each
/// member continues the decoded history of those before it, so reading
/// members one at a time would decode the archive once per page.
pub(super) struct RarArchive {
    entries: DecodedEntries,
}

/// One member's decoded bytes, written by the extractor and shared with the
/// entry table that outlives the writer.
#[derive(Clone, Default)]
struct Sink(Rc<RefCell<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl RarArchive {
    /// Parse and decode RAR bytes.  Each member is bounded by
    /// [`MAX_ENTRY_BYTES`] and each declared dictionary by
    /// [`MAX_DICTIONARY_BYTES`]; a member over either, an encrypted member
    /// (there is no password to give) or a failed checksum refuses the archive.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the archive cannot be parsed or decoded.
    pub(super) fn open(bytes: Vec<u8>) -> Result<Self, ComicError> {
        let bad = |e: rars::Error| ComicError::BadArchive(format!("rar: {e}"));
        let options = || {
            ArchiveReadOptions::new()
                .with_max_member_output_bytes(MAX_ENTRY_BYTES)
                .with_rar50_dictionary_size_limit(MAX_DICTIONARY_BYTES)
        };
        let archive = ArchiveReader::read_owned_with_options(bytes, options()).map_err(bad)?;
        let mut members: Vec<(String, Sink)> = Vec::new();
        archive
            .extract_to_with_options(options(), |meta| {
                let sink = Sink::default();
                if !meta.is_directory {
                    members.push((
                        String::from_utf8_lossy(&meta.name).into_owned(),
                        sink.clone(),
                    ));
                }
                Ok(Box::new(sink))
            })
            .map_err(bad)?;
        let entries = members
            .into_iter()
            .map(|(name, sink)| (name, sink.0.take()))
            .collect();
        Ok(Self {
            entries: DecodedEntries::new("rar", entries),
        })
    }
}

impl super::Archive for RarArchive {
    fn entry_names(&mut self) -> Vec<String> {
        self.entries.names()
    }

    fn take_entry(&mut self, name: &str) -> Result<Vec<u8>, ComicError> {
        self.entries.take(name)
    }
}

#[cfg(test)]
mod tests {
    use rars::{ArchiveVersion, Builder};

    use super::RarArchive;
    use crate::archive::Archive;

    /// A `format` archive of `entries` and one directory, compressed and,
    /// when `solid`, solid.
    fn make_rar(format: ArchiveVersion, solid: bool, entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = Builder::new(format).solid(solid);
        builder
            .add_directory(b"pages".to_vec(), None, None)
            .expect("directory");
        for (name, data) in entries {
            builder
                .add_bytes(name.as_bytes().to_vec(), data.to_vec(), None, None)
                .expect("entry");
        }
        builder.to_bytes().expect("archive")
    }

    #[test]
    fn rar_members_round_trip_in_every_family() {
        let page: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        for format in [ArchiveVersion::Rar29, ArchiveVersion::Rar50] {
            for solid in [false, true] {
                let bytes = make_rar(
                    format,
                    solid,
                    &[("pages/2.png", b"two"), ("pages/1.png", &page)],
                );
                let mut rar = RarArchive::open(bytes).expect("open");
                assert_eq!(
                    rar.entry_names(),
                    ["pages/2.png", "pages/1.png"],
                    "{format:?}"
                );
                assert_eq!(rar.take_entry("pages/1.png").expect("page 1"), page);
                assert_eq!(rar.take_entry("pages/2.png").expect("page 2"), b"two");
                assert!(rar.take_entry("pages/2.png").is_err(), "taken once");
            }
        }
    }

    #[test]
    fn a_damaged_rar_is_a_bad_archive() {
        let mut bytes = make_rar(ArchiveVersion::Rar50, false, &[("1.png", &[7; 4096])]);
        let last = bytes.len() - 40;
        bytes[last] ^= 0xFF;
        assert!(RarArchive::open(bytes).is_err());
        assert!(RarArchive::open(b"Rar!\x1a\x07\x00".to_vec()).is_err());
    }
}
