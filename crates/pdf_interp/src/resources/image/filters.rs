//! The `/Filter` and `/DecodeParms` entries of an image stream: the filter
//! chain, its terminal codec, and the parameters that codec receives.

use std::borrow::Cow;

use pdf::{Dictionary, Object};

/// `true` for a filter that transforms bytes without being an image codec
/// (PDF §7.4): an image ending in one decodes as raw samples after the whole
/// filter chain is applied.  The set is the stream decoder's own.
pub(super) fn is_transport_filter(name: &str) -> bool {
    pdf::is_decoded_filter(name.as_bytes())
}

/// Collect every filter name from a `/Filter` entry, in application order,
/// abbreviations expanded.
///
/// Accepts a bare `Name` (one filter) or a `Name` array (a filter chain, PDF
/// §7.4: filters are listed in the order they were applied during encoding, so
/// decoding applies them left to right).  Non-name array entries are skipped.
/// An empty or absent array yields an empty vector (no filter).
pub(crate) fn image_filter_chain(obj: &Object) -> Vec<Cow<'_, str>> {
    match obj {
        Object::Name(n) => vec![full_filter_name(n)],
        Object::Array(arr) => arr
            .iter()
            .filter_map(|o| o.as_name().map(full_filter_name))
            .collect(),
        _ => Vec::new(),
    }
}

/// Extract the *terminal* image filter from a `/Filter` entry, its
/// abbreviation expanded.
///
/// Accepts a bare `Name`, a single-element `Name` array, or a multi-element
/// filter chain (e.g. `[/ASCII85Decode /CCITTFaxDecode]`).  For a chain, the
/// returned name is the LAST filter — the terminal image codec.  Earlier
/// filters in the chain are transport/prefilters (ASCII85, Flate, …) whose
/// bytes are produced by [`pdf::Stream::decompressed_content`] before the
/// terminal codec runs; see `dispatch::decode_image_stream`.
///
/// An empty array (`Filter = []`) is treated as no filter (returns `None`)
/// and a debug message is emitted, matching the behaviour of an absent key.
pub(crate) fn filter_name(obj: &Object) -> Option<Cow<'_, str>> {
    match obj {
        Object::Name(n) => Some(full_filter_name(n)),
        Object::Array(arr) => {
            if arr.is_empty() {
                log::debug!("image: Filter is an empty array — treating as no filter");
                return None;
            }
            arr.iter()
                .rev()
                .find_map(|o| o.as_name())
                .map(full_filter_name)
        }
        _ => None,
    }
}

/// The full name of the filter `name`, abbreviations expanded.
fn full_filter_name(name: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(pdf::filter_full_name(name))
}

/// Resolve the `/DecodeParms` entry that belongs to the *terminal* filter.
///
/// `/DecodeParms` is index-aligned to `/Filter` (PDF §7.4.1): entry `i` is the
/// parameters for filter `i`; a `null` entry or an absent key means "no
/// parameters".  When `/Filter` is a chain, the terminal codec
/// (`CCITTFaxDecode`, `JBIG2Decode`, …) must receive ITS entry — the last one —
/// not the whole array, and not a prefilter's entry.
///
/// Forms handled:
/// - `/DecodeParms` is a dict, `/Filter` is a single name → that dict.
/// - `/DecodeParms` is an array → the last array element (if a dict / non-null).
/// - absent / `null` / mismatched → `None`.
pub(super) fn terminal_decode_parms(dict: &Dictionary, filter_count: usize) -> Option<&Object> {
    match dict.get(b"DecodeParms")? {
        Object::Array(arr) => {
            // Prefer the entry index-aligned to the terminal (last) filter;
            // fall back to the last array element if lengths disagree.
            let idx = filter_count.checked_sub(1).filter(|&i| i < arr.len());
            let entry = idx.map_or_else(|| arr.last(), |i| arr.get(i))?;
            match entry {
                Object::Null => None,
                other => Some(other),
            }
        }
        Object::Null => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_name_single_name() {
        let obj = Object::Name(b"FlateDecode".to_vec());
        assert_eq!(filter_name(&obj).as_deref(), Some("FlateDecode"));
    }

    #[test]
    fn filter_names_expand_abbreviations() {
        let obj = Object::Array(vec![
            Object::Name(b"A85".to_vec()),
            Object::Name(b"DCT".to_vec()),
        ]);
        assert_eq!(filter_name(&obj).as_deref(), Some("DCTDecode"));
        assert_eq!(image_filter_chain(&obj), ["ASCII85Decode", "DCTDecode"]);
    }

    #[test]
    fn filter_name_single_element_array() {
        let obj = Object::Array(vec![Object::Name(b"CCITTFaxDecode".to_vec())]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_chain_returns_terminal_codec() {
        // [/ASCII85Decode /CCITTFaxDecode]: the terminal (last) filter is
        // the image codec.
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"CCITTFaxDecode".to_vec()),
        ]);
        assert_eq!(filter_name(&obj).as_deref(), Some("CCITTFaxDecode"));
    }

    #[test]
    fn filter_name_empty_array_is_none() {
        assert!(filter_name(&Object::Array(vec![])).is_none());
    }

    #[test]
    fn image_filter_chain_orders_left_to_right() {
        let obj = Object::Array(vec![
            Object::Name(b"ASCII85Decode".to_vec()),
            Object::Name(b"FlateDecode".to_vec()),
        ]);
        let chain = image_filter_chain(&obj);
        assert_eq!(chain, vec!["ASCII85Decode", "FlateDecode"]);
    }

    #[test]
    fn image_filter_chain_single_name() {
        let obj = Object::Name(b"DCTDecode".to_vec());
        assert_eq!(image_filter_chain(&obj), vec!["DCTDecode"]);
    }

    #[test]
    fn terminal_decode_parms_picks_last_array_entry() {
        // [/ASCII85Decode /CCITTFaxDecode] with DecodeParms [null <</K -1>>]:
        // the CCITT codec must receive its OWN params (index 1), not null and
        // not the whole array.
        let mut ccitt = Dictionary::new();
        ccitt.set(b"K", Object::Integer(-1));
        let mut dict = Dictionary::new();
        dict.set(
            b"DecodeParms",
            Object::Array(vec![Object::Null, Object::Dictionary(ccitt)]),
        );
        let parms = terminal_decode_parms(&dict, 2).expect("terminal parms present");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"K"))
                .and_then(Object::as_i64),
            Some(-1)
        );
    }

    #[test]
    fn terminal_decode_parms_single_dict() {
        let mut ccitt = Dictionary::new();
        ccitt.set(b"Columns", Object::Integer(2480));
        let mut dict = Dictionary::new();
        dict.set(b"DecodeParms", Object::Dictionary(ccitt));
        let parms = terminal_decode_parms(&dict, 1).expect("single dict");
        assert_eq!(
            parms
                .as_dict()
                .and_then(|d| d.get(b"Columns"))
                .and_then(Object::as_i64),
            Some(2480)
        );
    }

    #[test]
    fn terminal_decode_parms_null_and_absent_are_none() {
        let dict = Dictionary::new();
        assert!(terminal_decode_parms(&dict, 1).is_none());

        let mut with_null = Dictionary::new();
        with_null.set(
            b"DecodeParms",
            Object::Array(vec![Object::Null, Object::Null]),
        );
        assert!(terminal_decode_parms(&with_null, 2).is_none());
    }

    #[test]
    fn chained_ascii85_then_flate_roundtrip() {
        // A `[/ASCII85Decode /FlateDecode]` stream the way a PDF stores it:
        // the payload is flate-compressed, then ASCII85-encoded.  The fixture
        // bytes below are `"chained filter round-trip payload \x00\x01\x02\xff"`
        // run through zlib then Adobe ASCII85 (terminated with `~>`).
        // `decompressed_content` must undo both filters left to right and
        // recover the original payload — this is exactly the prefilter step
        // the scanned-book corpus case relies on (only the terminal codec
        // differs: CCITTFax there, treated as raw here).
        let original: &[u8] = b"chained filter round-trip payload \x00\x01\x02\xff";
        let ascii85: &[u8] = b"GaqFP8Bf:N9i4I)bUH+8;CK[@btHG9.EX2<-qLG`a\\PT-?smOA%fdE-%FY~>";

        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(vec![
                Object::Name(b"ASCII85Decode".to_vec()),
                Object::Name(b"FlateDecode".to_vec()),
            ]),
        );
        let stream = pdf::Stream::new(dict, ascii85.to_vec());
        let decoded = stream
            .decompressed_content()
            .expect("ASCII85 then Flate must round-trip");
        assert_eq!(decoded, original);
    }
}
