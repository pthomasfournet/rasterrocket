//! Page striping (7.4.8.6, 7.4.10).

use crate::check;

#[test]
fn striped() {
    check("page_striped", "truchet");
}

#[test]
fn striped_unknown_height() {
    check("page_striped_unknown_height", "truchet");
}

#[test]
fn striped_last_stripe_implicit() {
    check("page_striped_last_stripe_implicit", "truchet");
}

#[test]
fn striped_single_stripe() {
    check("page_striped_single_stripe", "truchet");
}

#[test]
fn striped_no_end_of_stripe() {
    check("page_striped_no_end_of_stripe", "truchet");
}

/// An embedded page information segment (7.4.8) declaring `width` × `height`.
fn page_information(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0, 0, 0, 0, 48, 0, 1, 0, 0, 0, 19];
    data.extend_from_slice(&width.to_be_bytes());
    data.extend_from_slice(&height.to_be_bytes());
    data.extend_from_slice(&[0; 8]);
    data.extend_from_slice(&[0, 0, 0]);
    data
}

/// A page wider or taller than the largest bitmap is rejected when the page
/// information is parsed, before any region is decoded.
#[test]
fn page_larger_than_bitmap_limit_is_rejected() {
    use hayro_jbig2::{DecodeError, DecodeSettings, OverflowError, Page};

    let settings = DecodeSettings::default();
    let open =
        |w, h| Page::new_embedded(&page_information(w, h), None, &settings).map(|p| p.width());
    assert_eq!(open(65_535, 65_535).ok(), Some(65_535));
    for (w, h) in [(65_536, 1), (1, 65_536), (u32::MAX, 0x1000_0000)] {
        assert!(
            matches!(
                open(w, h),
                Err(DecodeError::Overflow(OverflowError::BitmapDimension))
            ),
            "{w}x{h}: {:?}",
            open(w, h)
        );
    }
}
