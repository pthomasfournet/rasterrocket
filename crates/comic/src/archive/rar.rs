//! `.cbr` (RAR) container, RAR 1.3 through RAR 7.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use rars::{
    ArchiveReadOptions, ArchiveReader, ExtractionDecision, ExtractionErrorAction, ExtractionOutcome,
};

use super::{ArchiveBudget, DecodedEntries, MAX_ENTRY_BYTES};
use crate::ComicError;

/// The largest dictionary, in bytes, a RAR 5 or later member may declare:
/// 1 GiB, the decoder allocation bound of the `.cb7` reader.  A member
/// declaring more is refused before its dictionary is allocated.
const MAX_DICTIONARY_BYTES: u64 = 1 << 30;

/// The most reader workspace, in bytes, one extraction may hold at once
/// (decoder and model state, filter scratch, packed staging, in every RAR
/// family): 2 GiB, the largest admitted dictionary and as much again for the
/// state beside it.
const MAX_WORKSPACE_BYTES: u64 = 2 * MAX_DICTIONARY_BYTES;

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

/// A member being extracted: its name, the sink its bytes go to, and the
/// error of a member that failed on its own.
struct Member {
    name: String,
    sink: Sink,
    error: Option<ComicError>,
}

impl RarArchive {
    /// Parse and decode RAR bytes.  Each member is bounded by
    /// [`MAX_ENTRY_BYTES`], each declared dictionary by
    /// [`MAX_DICTIONARY_BYTES`], the reader's workspace by
    /// [`MAX_WORKSPACE_BYTES`] and the sum of all members' sizes by `budget`.
    ///
    /// A member of a non-solid archive that fails on its own (a failed
    /// checksum, corrupt data, encryption: there is no password to give)
    /// becomes that entry's error and the other members are kept.  In a
    /// solid archive every later member depends on the failed one's
    /// history, so the failure refuses the archive.
    ///
    /// # Errors
    /// [`ComicError::BadArchive`] if the archive cannot be parsed, a member
    /// is over a per-member limit, or a member of a solid archive fails;
    /// [`ComicError::ArchiveTooLarge`] if the members exceed `budget`.
    pub(super) fn open(bytes: Vec<u8>, budget: ArchiveBudget) -> Result<Self, ComicError> {
        let bad = |e: rars::Error| match e.root_cause() {
            rars::Error::TotalOutputLimitExceeded { .. } => budget.exceeded(),
            _ => ComicError::BadArchive(format!("rar: {e}")),
        };
        let options = || {
            ArchiveReadOptions::new()
                .with_max_member_output_bytes(MAX_ENTRY_BYTES)
                .with_max_total_output_bytes(budget.limit())
                .with_rar50_dictionary_size_limit(MAX_DICTIONARY_BYTES)
                .with_max_reader_workspace_bytes(MAX_WORKSPACE_BYTES)
        };
        let archive = ArchiveReader::read_owned_with_options(bytes, options()).map_err(bad)?;
        let members = RefCell::new(Vec::<Member>::new());
        let outcome = archive
            .extract_with_control_and_errors(
                options(),
                |member| {
                    if member.meta.is_directory || member.meta.is_redirection {
                        return Ok(ExtractionDecision::Skip);
                    }
                    let sink = Sink::default();
                    members.borrow_mut().push(Member {
                        name: member.meta.name_lossy(),
                        sink: sink.clone(),
                        error: None,
                    });
                    Ok(ExtractionDecision::Extract(Box::new(sink)))
                },
                // Called only for a failure the reader can continue past: a
                // member of a non-solid archive, never a resource limit.  The
                // failed member is the one whose writer was opened last.
                |member, error| {
                    let name = member.meta.name_lossy();
                    log::warn!("comic: rar member {name} is unreadable, its page fails: {error}");
                    let error = ComicError::BadArchive(format!("rar member {name}: {error}"));
                    let mut members = members.borrow_mut();
                    match members.last_mut() {
                        Some(last) if last.name == name => last.error = Some(error),
                        _ => members.push(Member {
                            name,
                            sink: Sink::default(),
                            error: Some(error),
                        }),
                    }
                    Ok(ExtractionErrorAction::Continue)
                },
            )
            .map_err(bad)?;
        // The selector never stops the walk.
        debug_assert_eq!(outcome, ExtractionOutcome::Complete);
        let entries = members
            .into_inner()
            .into_iter()
            .map(|Member { name, sink, error }| {
                let data = error.map_or_else(|| Ok(sink.0.take()), Err);
                (name, data)
            })
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
    use crate::ComicError;
    use crate::archive::{Archive, ArchiveBudget};

    /// A budget no test reaches.
    fn ample() -> ArchiveBudget {
        ArchiveBudget::new(u64::MAX)
    }

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
                let mut rar = RarArchive::open(bytes, ample()).expect("open");
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
        assert!(RarArchive::open(bytes, ample()).is_err());
        assert!(RarArchive::open(b"Rar!\x1a\x07\x00".to_vec(), ample()).is_err());
    }

    #[test]
    fn members_share_the_archive_budget() {
        // Zeros compress far below their size, so only the total bounds the
        // archive; directories cost nothing.
        let zeros = vec![0u8; 64 * 1024];
        for format in [ArchiveVersion::Rar29, ArchiveVersion::Rar50] {
            for solid in [false, true] {
                let entries: Vec<(&str, &[u8])> =
                    [("1.png", &zeros[..]), ("2.png", &zeros), ("3.png", &zeros)].into();
                let bytes = make_rar(format, solid, &entries);
                assert!(
                    bytes.len() < zeros.len(),
                    "{format:?}: {} bytes",
                    bytes.len()
                );
                assert!(
                    matches!(
                        RarArchive::open(bytes.clone(), ArchiveBudget::new(150_000)),
                        Err(ComicError::ArchiveTooLarge { limit: 150_000 })
                    ),
                    "{format:?} solid {solid}"
                );
                assert!(RarArchive::open(bytes, ArchiveBudget::new(3 * 65_536)).is_ok());
            }
        }
    }

    /// `bytes` with one byte flipped inside the packed data of the member
    /// that holds `marker`, a run of bytes no header contains.
    fn damage(mut bytes: Vec<u8>, marker: &[u8]) -> Vec<u8> {
        let at = bytes
            .windows(marker.len())
            .position(|w| w == marker)
            .expect("the member is stored");
        bytes[at + marker.len() / 2] ^= 0xFF;
        bytes
    }

    #[test]
    fn a_damaged_member_of_a_non_solid_rar_fails_alone() {
        // Stored members, so the marker appears in the archive as is and
        // the flip lands in the payload, failing its checksum.
        let marker: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) ^ 0x5A).collect();
        for format in [ArchiveVersion::Rar29, ArchiveVersion::Rar50] {
            let mut builder = Builder::new(format).store(true);
            for (name, data) in [
                ("1.png", &b"one"[..]),
                ("2.png", &marker),
                ("3.png", b"three"),
            ] {
                builder
                    .add_bytes(name.as_bytes().to_vec(), data.to_vec(), None, None)
                    .expect("entry");
            }
            let bytes = damage(builder.to_bytes().expect("archive"), &marker);
            let mut rar = RarArchive::open(bytes, ample()).expect("the other members are kept");
            assert_eq!(rar.entry_names(), ["1.png", "2.png", "3.png"], "{format:?}");
            assert_eq!(rar.take_entry("1.png").expect("page 1"), b"one");
            assert!(
                matches!(rar.take_entry("2.png"), Err(ComicError::BadArchive(m)) if m.contains("2.png")),
                "{format:?}"
            );
            assert_eq!(rar.take_entry("3.png").expect("page 3"), b"three");
        }
    }
}
