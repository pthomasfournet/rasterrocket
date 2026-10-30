//! The predefined `CMaps` of ISO 32000-2 Table 116 (`cmaps/`), parsed on
//! first use and shared process-wide.
//!
//! The Identity `CMaps` are not files: [`CMap::identity`] builds them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use super::cmap::{CMap, parse_cmap_nested};

/// Name and zlib-compressed source of every vendored `CMap`.
static CMAPS: &[(&str, &[u8])] = &[
    ("83pv-RKSJ-H", include_bytes!("../../cmaps/83pv-RKSJ-H.z")),
    ("90ms-RKSJ-H", include_bytes!("../../cmaps/90ms-RKSJ-H.z")),
    ("90ms-RKSJ-V", include_bytes!("../../cmaps/90ms-RKSJ-V.z")),
    ("90msp-RKSJ-H", include_bytes!("../../cmaps/90msp-RKSJ-H.z")),
    ("90msp-RKSJ-V", include_bytes!("../../cmaps/90msp-RKSJ-V.z")),
    ("90pv-RKSJ-H", include_bytes!("../../cmaps/90pv-RKSJ-H.z")),
    ("Add-RKSJ-H", include_bytes!("../../cmaps/Add-RKSJ-H.z")),
    ("Add-RKSJ-V", include_bytes!("../../cmaps/Add-RKSJ-V.z")),
    ("B5pc-H", include_bytes!("../../cmaps/B5pc-H.z")),
    ("B5pc-V", include_bytes!("../../cmaps/B5pc-V.z")),
    ("CNS-EUC-H", include_bytes!("../../cmaps/CNS-EUC-H.z")),
    ("CNS-EUC-V", include_bytes!("../../cmaps/CNS-EUC-V.z")),
    ("ETen-B5-H", include_bytes!("../../cmaps/ETen-B5-H.z")),
    ("ETen-B5-V", include_bytes!("../../cmaps/ETen-B5-V.z")),
    ("ETenms-B5-H", include_bytes!("../../cmaps/ETenms-B5-H.z")),
    ("ETenms-B5-V", include_bytes!("../../cmaps/ETenms-B5-V.z")),
    ("EUC-H", include_bytes!("../../cmaps/EUC-H.z")),
    ("EUC-V", include_bytes!("../../cmaps/EUC-V.z")),
    ("Ext-RKSJ-H", include_bytes!("../../cmaps/Ext-RKSJ-H.z")),
    ("Ext-RKSJ-V", include_bytes!("../../cmaps/Ext-RKSJ-V.z")),
    ("GB-EUC-H", include_bytes!("../../cmaps/GB-EUC-H.z")),
    ("GB-EUC-V", include_bytes!("../../cmaps/GB-EUC-V.z")),
    ("GBK-EUC-H", include_bytes!("../../cmaps/GBK-EUC-H.z")),
    ("GBK-EUC-V", include_bytes!("../../cmaps/GBK-EUC-V.z")),
    ("GBK2K-H", include_bytes!("../../cmaps/GBK2K-H.z")),
    ("GBK2K-V", include_bytes!("../../cmaps/GBK2K-V.z")),
    ("GBKp-EUC-H", include_bytes!("../../cmaps/GBKp-EUC-H.z")),
    ("GBKp-EUC-V", include_bytes!("../../cmaps/GBKp-EUC-V.z")),
    ("GBpc-EUC-H", include_bytes!("../../cmaps/GBpc-EUC-H.z")),
    ("GBpc-EUC-V", include_bytes!("../../cmaps/GBpc-EUC-V.z")),
    ("H", include_bytes!("../../cmaps/H.z")),
    ("HKscs-B5-H", include_bytes!("../../cmaps/HKscs-B5-H.z")),
    ("HKscs-B5-V", include_bytes!("../../cmaps/HKscs-B5-V.z")),
    ("KSC-EUC-H", include_bytes!("../../cmaps/KSC-EUC-H.z")),
    ("KSC-EUC-V", include_bytes!("../../cmaps/KSC-EUC-V.z")),
    ("KSCms-UHC-H", include_bytes!("../../cmaps/KSCms-UHC-H.z")),
    (
        "KSCms-UHC-HW-H",
        include_bytes!("../../cmaps/KSCms-UHC-HW-H.z"),
    ),
    (
        "KSCms-UHC-HW-V",
        include_bytes!("../../cmaps/KSCms-UHC-HW-V.z"),
    ),
    ("KSCms-UHC-V", include_bytes!("../../cmaps/KSCms-UHC-V.z")),
    ("KSCpc-EUC-H", include_bytes!("../../cmaps/KSCpc-EUC-H.z")),
    (
        "UniCNS-UCS2-H",
        include_bytes!("../../cmaps/UniCNS-UCS2-H.z"),
    ),
    (
        "UniCNS-UCS2-V",
        include_bytes!("../../cmaps/UniCNS-UCS2-V.z"),
    ),
    (
        "UniCNS-UTF16-H",
        include_bytes!("../../cmaps/UniCNS-UTF16-H.z"),
    ),
    (
        "UniCNS-UTF16-V",
        include_bytes!("../../cmaps/UniCNS-UTF16-V.z"),
    ),
    ("UniGB-UCS2-H", include_bytes!("../../cmaps/UniGB-UCS2-H.z")),
    ("UniGB-UCS2-V", include_bytes!("../../cmaps/UniGB-UCS2-V.z")),
    (
        "UniGB-UTF16-H",
        include_bytes!("../../cmaps/UniGB-UTF16-H.z"),
    ),
    (
        "UniGB-UTF16-V",
        include_bytes!("../../cmaps/UniGB-UTF16-V.z"),
    ),
    (
        "UniJIS-UCS2-H",
        include_bytes!("../../cmaps/UniJIS-UCS2-H.z"),
    ),
    (
        "UniJIS-UCS2-HW-H",
        include_bytes!("../../cmaps/UniJIS-UCS2-HW-H.z"),
    ),
    (
        "UniJIS-UCS2-HW-V",
        include_bytes!("../../cmaps/UniJIS-UCS2-HW-V.z"),
    ),
    (
        "UniJIS-UCS2-V",
        include_bytes!("../../cmaps/UniJIS-UCS2-V.z"),
    ),
    (
        "UniJIS-UTF16-H",
        include_bytes!("../../cmaps/UniJIS-UTF16-H.z"),
    ),
    (
        "UniJIS-UTF16-V",
        include_bytes!("../../cmaps/UniJIS-UTF16-V.z"),
    ),
    ("UniKS-UCS2-H", include_bytes!("../../cmaps/UniKS-UCS2-H.z")),
    ("UniKS-UCS2-V", include_bytes!("../../cmaps/UniKS-UCS2-V.z")),
    (
        "UniKS-UTF16-H",
        include_bytes!("../../cmaps/UniKS-UTF16-H.z"),
    ),
    (
        "UniKS-UTF16-V",
        include_bytes!("../../cmaps/UniKS-UTF16-V.z"),
    ),
    ("V", include_bytes!("../../cmaps/V.z")),
];

/// The predefined `CMap` `name`, or `None` when no such `CMap` exists.
/// `Identity-H` and `Identity-V` are not predefined files; see
/// [`CMap::identity`].
#[must_use]
pub fn predefined_cmap(name: &[u8]) -> Option<Arc<CMap>> {
    predefined_cmap_nested(name, 0)
}

/// [`predefined_cmap`] requested from a `usecmap` at nesting `depth`.
pub(super) fn predefined_cmap_nested(name: &[u8], depth: u32) -> Option<Arc<CMap>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, Arc<CMap>>>> = OnceLock::new();
    let &(key, compressed) = CMAPS.iter().find(|(n, _)| n.as_bytes() == name)?;
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(cmap) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(key)
    {
        return Some(Arc::clone(cmap));
    }
    // Parsed outside the lock: a `usecmap` parent is resolved through this
    // same cache.  Two threads parsing one `CMap` at once keep one copy.
    let source = pdf::inflate(compressed)
        .inspect_err(|e| log::error!("cmap: bundled {key}: {e}"))
        .ok()?;
    let parsed = Arc::new(parse_cmap_nested(&source, depth)?);
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    Some(Arc::clone(cache.entry(key).or_insert(parsed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every vendored `CMap` inflates and parses to a usable mapping.
    #[test]
    fn every_vendored_cmap_parses() {
        for &(name, _) in CMAPS {
            let cmap = predefined_cmap(name.as_bytes()).unwrap_or_else(|| panic!("{name}"));
            assert!(!cmap.codespaces.is_empty(), "{name} has codespace ranges");
            assert!(!cmap.map.is_empty(), "{name} maps codes");
            assert!(!cmap.identity, "{name} is not an identity CMap");
        }
        assert!(predefined_cmap(b"Identity-H").is_none());
        assert!(predefined_cmap(b"No-Such-CMap").is_none());
    }

    /// `90ms-RKSJ-H` is Shift-JIS: one-byte ASCII to the half-width Roman
    /// CIDs, two-byte kanji to their Adobe-Japan1 CIDs.
    #[test]
    fn rksj_h_maps_shift_jis() {
        let cmap = predefined_cmap(b"90ms-RKSJ-H").expect("90ms-RKSJ-H");
        assert_eq!(cmap.codespaces.len(), 4);
        let codes: Vec<(u32, u32, u8)> = cmap
            .iter_codes(&[0x41, 0x93, 0xFA, 0x96, 0x7B])
            .map(|c| (c.code, c.value, c.len))
            .collect();
        assert_eq!(
            codes,
            vec![(0x41, 264, 1), (0x93FA, 3284, 2), (0x967B, 3722, 2)]
        );
    }

    /// `UniJIS-UCS2-H` is two-byte Unicode; its `HW` variant overrides the
    /// ASCII range through `usecmap` and inherits the rest.
    #[test]
    fn unijis_ucs2_variants_share_mappings_through_usecmap() {
        let ucs2 = predefined_cmap(b"UniJIS-UCS2-H").expect("UniJIS-UCS2-H");
        assert_eq!(ucs2.lookup(0x0041), 34);
        assert_eq!(ucs2.lookup(0x65E5), 3284);
        let hw = predefined_cmap(b"UniJIS-UCS2-HW-H").expect("UniJIS-UCS2-HW-H");
        assert_eq!(hw.lookup(0x0041), 264);
        assert_eq!(hw.lookup(0x65E5), 3284);
    }

    /// The same `CMap` is parsed once per process.
    #[test]
    fn predefined_cmaps_are_shared() {
        let a = predefined_cmap(b"GBK-EUC-H").expect("GBK-EUC-H");
        let b = predefined_cmap(b"GBK-EUC-H").expect("GBK-EUC-H");
        assert!(Arc::ptr_eq(&a, &b));
    }
}
