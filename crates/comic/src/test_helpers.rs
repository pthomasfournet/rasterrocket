//! Shared test fixtures, available to every module's `#[cfg(test)]` blocks.

use std::io::Write as _;

/// Build a `.cbz` (ZIP) in memory from the given `(name, bytes)` entries.
///
/// `::zip` (crate-root path) is the external crate, not the sibling `zip`
/// archive submodule that a `use super::*` would otherwise resolve to.
pub(crate) fn make_cbz(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut w = ::zip::ZipWriter::new(&mut buf);
        let opts: ::zip::write::FileOptions<'_, ()> = ::zip::write::FileOptions::default();
        for (name, data) in entries {
            w.start_file(*name, opts).unwrap();
            w.write_all(data).unwrap();
        }
        // `finish` returns the inner writer; `unused_results` requires binding it.
        let _ = w.finish().unwrap();
    }
    buf.into_inner()
}
