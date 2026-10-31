//! Kernel access-pattern advice for the `Document` mapping.
//!
//! All helpers are no-ops on platforms without `posix_fadvise` (Windows and
//! the Apple systems among them).  The syscalls go through
//! `rustix`'s safe wrappers, so this module contains no `unsafe` code (the
//! crate denies `unsafe_code`; the one exception is the file mapping in
//! `document.rs`).
//!
//! The hints are advisory.  A wrong hint never causes a correctness issue;
//! the worst outcome is suboptimal kernel page-cache behaviour.

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use std::fs::File;

#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
use rustix::fs::{Advice, fadvise};

/// Advise the kernel that `file` is read at random positions, so
/// readahead is disabled for the whole file (`len` `None` is "to EOF").
///
/// Called once when a document is mapped, before any byte is read.  A
/// failure is ignored: the advice is optional and its absence leaves the
/// default readahead in place.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
pub(crate) fn advise_random(file: &File) {
    let _ = fadvise(file, 0, None, Advice::Random);
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
pub(crate) fn advise_random(_: &std::fs::File) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a `File` whose contents are `bytes`.  The on-disk path is
    /// unlinked when the inner `NamedTempFile` drops at end of this
    /// function, but the returned fd keeps the inode alive on Linux for
    /// as long as the test holds it (POSIX unlink-while-open semantics).
    fn temp_file_with(bytes: &[u8]) -> std::fs::File {
        let mut tmp = tempfile::NamedTempFile::new().expect("create tmp");
        tmp.write_all(bytes).expect("write");
        tmp.reopen().expect("reopen")
    }

    #[test]
    fn advise_random_does_not_panic() {
        advise_random(&temp_file_with(b"hello world"));
    }
}
