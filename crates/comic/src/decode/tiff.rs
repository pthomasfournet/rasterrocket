//! TIFF decode for comic/scan pages, via the pure-Rust `tiff` crate.
//!
//! A TIFF file can hold multiple pages (IFDs). Comics/scans in archives are
//! one-image-per-entry, so only the first IFD is decoded; a warning fires if
//! the file carries additional pages.

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};

use super::{
    CodecPixels, DecodeError, drop_alpha_in_place, gray_from_tight, guard_dimensions,
    rgb_from_tight,
};

/// Decode the first page of a TIFF: a gray page yields a `Gray8` bitmap, an
/// RGB or RGBA page an `Rgb8` bitmap with alpha dropped.
///
/// # Errors
///
/// [`DecodeError::Codec`] on malformed/unsupported streams.
pub(super) fn decode(bytes: &[u8]) -> Result<CodecPixels, DecodeError> {
    let mut dec = Decoder::new(std::io::Cursor::new(bytes))
        .map_err(|e| DecodeError::Codec(format!("tiff: {e}")))?;
    if dec.more_images() {
        log::warn!("tiff: multi-page TIFF; only the first page is decoded");
    }
    let (w, h) = dec
        .dimensions()
        .map_err(|e| DecodeError::Codec(format!("tiff: {e}")))?;
    if w == 0 || h == 0 {
        return Err(DecodeError::Codec(format!("tiff: zero dimensions {w}x{h}")));
    }
    // Bound the allocation by the project's own page-size limits before
    // read_image(), matching the PNG/WebP guards (the tiff crate has its own
    // internal cap, but this keeps every codec on the same MAX_PX_* policy).
    guard_dimensions("tiff", w, h)?;
    let color = dec
        .colortype()
        .map_err(|e| DecodeError::Codec(format!("tiff: {e}")))?;
    let img = dec
        .read_image()
        .map_err(|e| DecodeError::Codec(format!("tiff: {e}")))?;
    let mut u8s: Vec<u8> = match img {
        DecodingResult::U8(v) => v,
        // 16-bit → 8-bit: keep the high byte (the standard sample reduction).
        DecodingResult::U16(v) => v.iter().map(|&x| (x >> 8) as u8).collect(),
        _ => {
            return Err(DecodeError::Codec(
                "tiff: unsupported sample format (expected 8/16-bit integer)".to_owned(),
            ));
        }
    };
    match color {
        ColorType::RGB(_) => rgb_from_tight(w, h, u8s),
        ColorType::RGBA(_) => {
            drop_alpha_in_place(&mut u8s, 4);
            rgb_from_tight(w, h, u8s)
        }
        ColorType::Gray(_) => gray_from_tight(w, h, u8s),
        other => Err(DecodeError::Codec(format!(
            "tiff: unsupported color type {other:?}"
        ))),
    }
}
