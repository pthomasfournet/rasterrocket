//! The predefined `CMaps` of ISO 32000-2 Table 116 (`cmaps/`), parsed on
//! first use and shared process-wide.
//!
//! The Identity `CMaps` are not files: [`CMap::identity`] builds them.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use super::cmap::{CMap, parse_cmap_nested};
use super::font_substitute::Script;

/// One vendored `CMap`: its name, character collection and the
/// zlib-compressed resource file `cmaps/<name>.z`.
macro_rules! cmap {
    ($name:literal, $script:ident) => {
        (
            $name,
            Script::$script,
            include_bytes!(concat!("../../cmaps/", $name, ".z")),
        )
    };
}

/// Name, character collection and zlib-compressed source of every
/// vendored `CMap`.
static CMAPS: &[(&str, Script, &[u8])] = &[
    cmap!("83pv-RKSJ-H", Japanese),
    cmap!("90ms-RKSJ-H", Japanese),
    cmap!("90ms-RKSJ-V", Japanese),
    cmap!("90msp-RKSJ-H", Japanese),
    cmap!("90msp-RKSJ-V", Japanese),
    cmap!("90pv-RKSJ-H", Japanese),
    cmap!("Add-RKSJ-H", Japanese),
    cmap!("Add-RKSJ-V", Japanese),
    cmap!("B5pc-H", TraditionalChinese),
    cmap!("B5pc-V", TraditionalChinese),
    cmap!("CNS-EUC-H", TraditionalChinese),
    cmap!("CNS-EUC-V", TraditionalChinese),
    cmap!("ETen-B5-H", TraditionalChinese),
    cmap!("ETen-B5-V", TraditionalChinese),
    cmap!("ETenms-B5-H", TraditionalChinese),
    cmap!("ETenms-B5-V", TraditionalChinese),
    cmap!("EUC-H", Japanese),
    cmap!("EUC-V", Japanese),
    cmap!("Ext-RKSJ-H", Japanese),
    cmap!("Ext-RKSJ-V", Japanese),
    cmap!("GB-EUC-H", SimplifiedChinese),
    cmap!("GB-EUC-V", SimplifiedChinese),
    cmap!("GBK-EUC-H", SimplifiedChinese),
    cmap!("GBK-EUC-V", SimplifiedChinese),
    cmap!("GBK2K-H", SimplifiedChinese),
    cmap!("GBK2K-V", SimplifiedChinese),
    cmap!("GBKp-EUC-H", SimplifiedChinese),
    cmap!("GBKp-EUC-V", SimplifiedChinese),
    cmap!("GBpc-EUC-H", SimplifiedChinese),
    cmap!("GBpc-EUC-V", SimplifiedChinese),
    cmap!("H", Japanese),
    cmap!("HKscs-B5-H", TraditionalChinese),
    cmap!("HKscs-B5-V", TraditionalChinese),
    cmap!("KSC-EUC-H", Korean),
    cmap!("KSC-EUC-V", Korean),
    cmap!("KSCms-UHC-H", Korean),
    cmap!("KSCms-UHC-HW-H", Korean),
    cmap!("KSCms-UHC-HW-V", Korean),
    cmap!("KSCms-UHC-V", Korean),
    cmap!("KSCpc-EUC-H", Korean),
    cmap!("UniCNS-UCS2-H", TraditionalChinese),
    cmap!("UniCNS-UCS2-V", TraditionalChinese),
    cmap!("UniCNS-UTF16-H", TraditionalChinese),
    cmap!("UniCNS-UTF16-V", TraditionalChinese),
    cmap!("UniGB-UCS2-H", SimplifiedChinese),
    cmap!("UniGB-UCS2-V", SimplifiedChinese),
    cmap!("UniGB-UTF16-H", SimplifiedChinese),
    cmap!("UniGB-UTF16-V", SimplifiedChinese),
    cmap!("UniJIS-UCS2-H", Japanese),
    cmap!("UniJIS-UCS2-HW-H", Japanese),
    cmap!("UniJIS-UCS2-HW-V", Japanese),
    cmap!("UniJIS-UCS2-V", Japanese),
    cmap!("UniJIS-UTF16-H", Japanese),
    cmap!("UniJIS-UTF16-V", Japanese),
    cmap!("UniKS-UCS2-H", Korean),
    cmap!("UniKS-UCS2-V", Korean),
    cmap!("UniKS-UTF16-H", Korean),
    cmap!("UniKS-UTF16-V", Korean),
    cmap!("V", Japanese),
];

/// The predefined `CMap` `name`, or `None` when no such `CMap` exists.
/// `Identity-H` and `Identity-V` are not predefined files; see
/// [`CMap::identity`].
#[must_use]
pub(crate) fn predefined_cmap(name: &[u8]) -> Option<Arc<CMap>> {
    predefined_cmap_nested(name, 0)
}

/// [`predefined_cmap`] requested from a `usecmap` at nesting `depth`.
pub(super) fn predefined_cmap_nested(name: &[u8], depth: u32) -> Option<Arc<CMap>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, Arc<CMap>>>> = OnceLock::new();
    let &(key, _, compressed) = CMAPS.iter().find(|(n, ..)| n.as_bytes() == name)?;
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

/// The CID → Unicode table of a character collection.
///
/// Several code points can share a CID (a unified character and its
/// compatibility or full-width form; WAVE DASH, FULLWIDTH TILDE and TILDE
/// OPERATOR): each CID keeps all of them, in preference order, so that a
/// face lacking the first still draws the CID through another.
#[derive(Debug, Default)]
pub(crate) struct CollectionUnicode {
    /// Code points of each CID that has any: those outside the
    /// compatibility blocks first, then by value.
    unicode: HashMap<u32, Box<[u32]>>,
    /// CIDs reached only through the vertical `UCS2-V` `CMap`: the
    /// vertical forms (rotated or repositioned glyphs) of their code
    /// points.
    vertical: HashSet<u32>,
}

impl CollectionUnicode {
    /// The code points of `cid`, in preference order; empty when it has
    /// none.
    #[must_use]
    pub(crate) fn get(&self, cid: u32) -> &[u32] {
        self.unicode.get(&cid).map_or(&[], |codes| codes)
    }

    /// Whether `cid` is the vertical form of its code point.
    #[must_use]
    pub(crate) fn is_vertical(&self, cid: u32) -> bool {
        self.vertical.contains(&cid)
    }

    /// Every CID with its code points, in no fixed order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &[u32])> + '_ {
        self.unicode.iter().map(|(&cid, codes)| (cid, &**codes))
    }
}

/// A [`CollectionUnicode`] shared process-wide.
pub(crate) type CidToUnicode = Arc<CollectionUnicode>;

/// CID → Unicode for the character collection of `script`.
///
/// The table inverts the collection's `UCS2-H` `CMap`, then its `UCS2-V`
/// `CMap` for the CIDs only the vertical one reaches.  `None` for a
/// script without a collection.  Built once per process.
#[must_use]
pub(crate) fn cid_to_unicode(script: Script) -> Option<CidToUnicode> {
    static CACHE: OnceLock<Mutex<HashMap<Script, CidToUnicode>>> = OnceLock::new();
    let (name, vertical_name): (&[u8], &[u8]) = match script {
        Script::Japanese => (b"UniJIS-UCS2-H", b"UniJIS-UCS2-V"),
        Script::SimplifiedChinese => (b"UniGB-UCS2-H", b"UniGB-UCS2-V"),
        Script::TraditionalChinese => (b"UniCNS-UCS2-H", b"UniCNS-UCS2-V"),
        Script::Korean => (b"UniKS-UCS2-H", b"UniKS-UCS2-V"),
        Script::Arabic | Script::Thai => return None,
    };
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(table) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&script)
    {
        return Some(Arc::clone(table));
    }
    let cmap = predefined_cmap(name)?;
    let mut pairs: Vec<(u32, bool, u32)> = cmap
        .map
        .iter()
        .map(|(&code, &cid)| (cid, is_compatibility_block(code), code))
        .collect();
    // The collection's legacy encodings map single-byte ASCII to half-width
    // Roman CIDs the Unicode `CMap` does not reach; those codes are their
    // own code points.
    for &(legacy, _, _) in CMAPS.iter().filter(|&&(_, s, _)| s == script) {
        if let Some(legacy) = predefined_cmap(legacy.as_bytes()) {
            pairs.extend(
                legacy
                    .map
                    .iter()
                    .filter(|&(&code, _)| (0x20..=0x7E).contains(&code))
                    .map(|(&code, &cid)| (cid, false, code)),
            );
        }
    }
    let mut unicode = invert(pairs);
    let vertical_pairs: Vec<(u32, bool, u32)> = predefined_cmap(vertical_name)
        .map(|cmap| {
            cmap.map
                .iter()
                .filter(|&(_, cid)| !unicode.contains_key(cid))
                .map(|(&code, &cid)| (cid, is_compatibility_block(code), code))
                .collect()
        })
        .unwrap_or_default();
    let vertical_only = invert(vertical_pairs);
    let vertical = vertical_only.keys().copied().collect();
    unicode.extend(vertical_only);
    let table = Arc::new(CollectionUnicode { unicode, vertical });
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    Some(Arc::clone(cache.entry(script).or_insert(table)))
}

/// The CID → code points map of `(cid, compatibility, code)` triples: per
/// CID, the codes outside the compatibility blocks before those inside
/// them, each group by value.
fn invert(mut pairs: Vec<(u32, bool, u32)>) -> HashMap<u32, Box<[u32]>> {
    pairs.sort_unstable();
    pairs.dedup();
    let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
    for (cid, _, code) in pairs {
        let codes = map.entry(cid).or_default();
        if !codes.contains(&code) {
            codes.push(code);
        }
    }
    map.into_iter()
        .map(|(cid, codes)| (cid, codes.into_boxed_slice()))
        .collect()
}

/// Whether `code` lies in a block of radicals, compatibility ideographs or
/// full-width forms, whose characters duplicate unified ones.
const fn is_compatibility_block(code: u32) -> bool {
    matches!(
        code,
        0x2E80..=0x2FDF | 0x3190..=0x319F | 0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFFEF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverted `UCS2-H` `CMap` maps a CID back to a code point.
    #[test]
    fn cid_to_unicode_inverts_the_ucs2_cmap() {
        let japan1 = cid_to_unicode(Script::Japanese).expect("Japan1");
        assert_eq!(japan1.get(3284).first(), Some(&0x65E5));
        assert_eq!(japan1.get(34).first(), Some(&0x41));
        // Half-width `A`, reached only through the legacy encodings.
        assert_eq!(japan1.get(264).first(), Some(&0x41));
        assert!(!japan1.is_vertical(3284));
        // WAVE DASH keeps every code point of its CID, unified first.
        assert_eq!(japan1.get(665), &[0x223C, 0x301C, 0xFF5E]);
        let gb1 = cid_to_unicode(Script::SimplifiedChinese).expect("GB1");
        assert_eq!(gb1.get(34).first(), Some(&0x41));
        assert!(cid_to_unicode(Script::Thai).is_none());
    }

    /// A CID only the `UCS2-V` `CMap` reaches maps to its code point and is
    /// marked vertical: the vertical long-vowel mark, ideographic comma and
    /// full stop of Adobe-Japan1.
    #[test]
    fn vertical_only_cids_map_to_their_code_points() {
        let japan1 = cid_to_unicode(Script::Japanese).expect("Japan1");
        let v = predefined_cmap(b"UniJIS-UCS2-V").expect("UniJIS-UCS2-V");
        let h = predefined_cmap(b"UniJIS-UCS2-H").expect("UniJIS-UCS2-H");
        for unicode in [0x30FC, 0x3001, 0x3002] {
            let cid = v.lookup(unicode);
            assert_ne!(cid, h.lookup(unicode), "U+{unicode:04X} has a vertical CID");
            assert!(japan1.get(cid).contains(&unicode), "CID {cid}");
            assert!(japan1.is_vertical(cid), "CID {cid}");
        }
    }

    /// Every vendored `CMap` inflates and parses to a usable mapping.
    #[test]
    fn every_vendored_cmap_parses() {
        for &(name, ..) in CMAPS {
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
