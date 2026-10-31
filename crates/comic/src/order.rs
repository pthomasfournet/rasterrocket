//! Page ordering: natural-sort comparison and image-file filtering.

use std::cmp::Ordering;

/// True if `name` is a directory or a macOS resource-fork (`__MACOSX`) member,
/// considering both `/` and `\` separators (some zips use Windows paths).
fn is_dir_or_resource_fork(name: &str) -> bool {
    name.ends_with('/')
        || name.ends_with('\\')
        || name.starts_with("__MACOSX/")
        || name.starts_with("__MACOSX\\")
        || name.contains("/__MACOSX/")
        || name.contains("\\__MACOSX\\")
        || name.contains("/__MACOSX\\")
        || name.contains("\\__MACOSX/")
}

/// Basename after the last `/` OR `\` separator, so Windows-style zip names
/// resolve to the same basename as POSIX ones.
fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

/// True if `name` is a decodable image entry we should treat as a page.
///
/// Rejects directories (trailing `/` or `\`), the macOS resource-fork prefix
/// `__MACOSX`, dotfiles like `Thumbs.db`, and anything whose final extension
/// is not a supported image type. The check is on the *last* extension, so
/// `cover.jpg.bak` is rejected. Both `/` and `\` are honoured as separators.
#[must_use]
pub(crate) fn is_image_file(name: &str) -> bool {
    if is_dir_or_resource_fork(name) {
        return false;
    }
    let base = basename(name);
    if base.eq_ignore_ascii_case("Thumbs.db") {
        return false;
    }
    let Some((_, ext)) = base.rsplit_once('.') else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "jpg" | "jpeg" | "png" | "webp" | "tiff" | "tif"
    )
}

/// True if `name` is a PDF entry (last extension `.pdf`, not a directory or
/// `__MACOSX` resource fork). Mirrors [`is_image_file`]'s path/junk guards.
#[must_use]
pub(crate) fn is_pdf_file(name: &str) -> bool {
    if is_dir_or_resource_fork(name) {
        return false;
    }
    basename(name)
        .rsplit_once('.')
        .is_some_and(|(_, ext)| ext.eq_ignore_ascii_case("pdf"))
}

/// Numeric-aware ("natural") comparison of two entry names, so `page2` orders
/// before `page10`. Runs of ASCII digits compare by numeric value; other runs
/// compare case-insensitively, then case-sensitively as a tie-break for
/// determinism.
#[must_use]
pub(crate) fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut ai = a.bytes().peekable();
    let mut bi = b.bytes().peekable();
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    // Compare full digit runs by value, ignoring leading zeros.
                    let na = take_number(&mut ai);
                    let nb = take_number(&mut bi);
                    match na.cmp(&nb) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                } else {
                    // Compare case-insensitively, then by raw byte as a
                    // deterministic tie-break for letters differing only in case.
                    let ord = ca
                        .to_ascii_lowercase()
                        .cmp(&cb.to_ascii_lowercase())
                        .then(ca.cmp(&cb));
                    if ord != Ordering::Equal {
                        return ord;
                    }
                    // Equal here — advance both and continue scanning.
                    let _ = ai.next();
                    let _ = bi.next();
                }
            }
        }
    }
}

/// Consume a run of ASCII digits from `it` and return its numeric value.
/// Saturates at `u64::MAX` for pathologically long runs (still deterministic).
fn take_number(it: &mut std::iter::Peekable<std::str::Bytes<'_>>) -> u64 {
    let mut n: u64 = 0;
    while let Some(&c) = it.peek() {
        if !c.is_ascii_digit() {
            break;
        }
        n = n.saturating_mul(10).saturating_add(u64::from(c - b'0'));
        let _ = it.next();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_image_file_accepts_known_extensions() {
        for n in [
            "a.jpg",
            "a.JPEG",
            "p.png",
            "x.webp",
            "s.tiff",
            "s.TIF",
            "sub/dir/03.Jpg",
        ] {
            assert!(is_image_file(n), "{n} should be an image");
        }
    }

    #[test]
    fn is_image_file_rejects_metadata_and_junk() {
        for n in [
            "ComicInfo.xml",
            "__MACOSX/._cover.jpg",
            "Thumbs.db",
            "folder/",
            "notes.txt",
            ".hidden/x.png/",
            "cover.jpg.bak",
        ] {
            assert!(!is_image_file(n), "{n} should be rejected");
        }
    }

    #[test]
    fn is_pdf_file_matches_only_pdfs() {
        assert!(is_pdf_file("book.pdf"));
        assert!(is_pdf_file("sub/Book.PDF"));
        assert!(!is_pdf_file("p.jpg"));
        assert!(!is_pdf_file("__MACOSX/._x.pdf"));
        assert!(!is_pdf_file("dir/"));
    }

    #[test]
    fn backslash_paths_are_filtered_like_posix() {
        // Windows-style separators must be honoured for the __MACOSX guard, the
        // directory guard, and basename extraction.
        assert!(!is_image_file("__MACOSX\\._cover.jpg"));
        assert!(!is_image_file("a\\__MACOSX\\x.jpg"));
        assert!(is_image_file("sub\\page.jpg")); // real image in a windows-path subdir
        assert!(!is_image_file("winsub\\")); // backslash directory
        assert!(!is_pdf_file("__MACOSX\\._book.pdf"));
        assert!(is_pdf_file("scans\\book.pdf"));
    }

    #[test]
    fn natural_sort_orders_numerically() {
        let mut v = vec!["page10.jpg", "page2.jpg", "page1.jpg", "page20.jpg"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            v,
            vec!["page1.jpg", "page2.jpg", "page10.jpg", "page20.jpg"]
        );
    }

    #[test]
    fn natural_sort_handles_zero_padding_and_paths() {
        let mut v = vec!["ch1/009.png", "ch1/010.png", "ch1/8.png"];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["ch1/8.png", "ch1/009.png", "ch1/010.png"]);
    }

    #[test]
    fn natural_cmp_is_case_insensitive_on_letters() {
        assert_eq!(
            natural_cmp("Page2.jpg", "page10.jpg"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn natural_cmp_handles_digit_vs_non_digit_boundary() {
        // One side on a digit run, the other not: falls through to byte compare.
        // '1' (0x31) < '_' (0x5F).
        assert_eq!(natural_cmp("page1", "page_1"), Ordering::Less);
        // Equal numeric run, then letters decide.
        assert_eq!(natural_cmp("p1a", "p1b"), Ordering::Less);
    }
}
