//! Loose-image decode front door.
//!
//! One place that sniffs an image's format by magic bytes, dispatches to the
//! per-codec wrapper, and normalises the result to grayscale at the depth
//! of the [`Target`] — the single decode→bitmap→gray sequence shared by
//! every format.  A codec keeps 16-bit samples only for a 16-bit target and
//! a source of more than 8 bits; an 8-bit source on a 16-bit target is its
//! 8-bit page widened exactly (`v · 257`).  Every codec checks the header
//! dimensions against the rendered-page limits before it allocates pixels
//! ([`guard_dimensions`]); the caller names the archive entry in the
//! resulting [`DecodeError::TooLarge`].

mod avif;
mod jpeg;
mod jxl;
mod png;
mod tiff;
mod webp;

use color::{Gray8, Gray16, Pixel, Rgb8, Rgb16, Sample, SampleDepth};
use raster::Bitmap;

/// The most samples (pixels × channels, every plane and frame counted) a
/// codec may allocate for one image: 1 Gi.  At 2 bytes per sample, the width
/// of the deep formats, the planes of an image stay within 2 GiB.
const MAX_SAMPLES: usize = 1 << 30;

/// The page a decode produces: the byte budget of its pixels and the
/// depth of its samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Target {
    /// The page byte budget, counted as [`guard_dimensions`] does.
    pub(crate) max_bytes: u64,
    /// The depth of the page's samples.
    pub(crate) depth: SampleDepth,
}

impl Target {
    /// Pages of samples of `depth` within `max_bytes`.
    pub(crate) const fn new(max_bytes: u64, depth: SampleDepth) -> Self {
        Self { max_bytes, depth }
    }

    /// Whether a codec keeps the samples of a deep source at 16 bits.
    const fn deep(self) -> bool {
        matches!(self.depth, SampleDepth::Sixteen)
    }
}

/// [`Target`] of the default budget at 8 bits.
#[cfg(test)]
pub(crate) const EIGHT: Target =
    Target::new(pdf_raster::DEFAULT_MAX_PAGE_BYTES, SampleDepth::Eight);

/// [`Target`] of the default budget at 16 bits.
#[cfg(test)]
pub(crate) const SIXTEEN: Target =
    Target::new(pdf_raster::DEFAULT_MAX_PAGE_BYTES, SampleDepth::Sixteen);

/// A codec's output in the colour model the stream carries.
enum CodecPixels {
    /// Single-channel luma.
    Gray(Bitmap<Gray8>),
    /// Interleaved RGB.
    Rgb(Bitmap<Rgb8>),
    /// Single-channel 16-bit luma, from a deep source on a 16-bit target.
    Gray16(Bitmap<Gray16>),
    /// Interleaved 16-bit RGB, from a deep source on a 16-bit target.
    Rgb16(Bitmap<Rgb16>),
}

/// A decoded page: luma at the depth of its [`Target`].
pub(crate) enum PageImage {
    /// 8-bit luma.
    Eight(Bitmap<Gray8>),
    /// 16-bit luma.
    Sixteen(Bitmap<Gray16>),
}

impl PageImage {
    /// The page at `depth`: as it is, widened exactly, or narrowed to the
    /// nearest 8-bit level.
    fn at(self, depth: SampleDepth) -> Self {
        match (self, depth) {
            (Self::Eight(g), SampleDepth::Sixteen) => Self::Sixteen(resampled(&g)),
            (Self::Sixteen(g), SampleDepth::Eight) => Self::Eight(resampled(&g)),
            (page, _) => page,
        }
    }
}

/// The packed bitmap of `src`'s samples on the scale of `Q`'s samples
/// ([`Sample::from_sample`]).
fn resampled<P: Pixel, Q: Pixel>(src: &Bitmap<P>) -> Bitmap<Q> {
    let live = src.width as usize * P::NCOMPS;
    let mut data = Vec::with_capacity(live * src.height as usize);
    for y in 0..src.height {
        data.extend(
            src.row_samples(y)[..live]
                .iter()
                .map(|&v| Q::Sample::from_sample(v)),
        );
    }
    Bitmap::from_packed(src.width, src.height, data)
}

/// Recognised container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageFormat {
    /// JPEG (`FF D8 FF`).
    Jpeg,
    /// PNG (`89 50 4E 47`).
    Png,
    /// WebP (`RIFF....WEBP`).
    WebP,
    /// TIFF (`II*\0` little-endian or `MM\0*` big-endian).
    Tiff,
    /// JPEG XL: a bare codestream (`FF 0A`) or an ISO BMFF container
    /// (`00 00 00 0C 'JXL ' 0D 0A 87 0A`).
    Jxl,
    /// AVIF: an ISO BMFF file whose `ftyp` box names the brand `avif`,
    /// `avis` or `mif1`.
    Avif,
}

/// Why decoding a single entry failed.
#[derive(Debug)]
pub(crate) enum DecodeError {
    /// No recognised image magic bytes.
    Unsupported,
    /// The header dimensions exceed the rendered-page limits.
    TooLarge { width: u32, height: u32 },
    /// The codec rejected the byte stream.
    Codec(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "unrecognised image format"),
            Self::TooLarge { width, height } => {
                write!(f, "image too large ({width}x{height})")
            }
            Self::Codec(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Detect the image format from leading magic bytes. `None` if unrecognised.
#[must_use]
pub(crate) fn sniff(bytes: &[u8]) -> Option<ImageFormat> {
    if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
        return Some(ImageFormat::Jpeg);
    }
    if bytes.len() >= 8 && bytes[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(ImageFormat::Png);
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some(ImageFormat::WebP);
    }
    if bytes.len() >= 4
        && (bytes[..4] == [0x49, 0x49, 0x2A, 0x00] || bytes[..4] == [0x4D, 0x4D, 0x00, 0x2A])
    {
        return Some(ImageFormat::Tiff);
    }
    if bytes.starts_with(&[0xFF, 0x0A])
        || bytes.starts_with(b"\x00\x00\x00\x0CJXL \x0D\x0A\x87\x0A")
    {
        return Some(ImageFormat::Jxl);
    }
    if avif::is_avif(bytes) {
        return Some(ImageFormat::Avif);
    }
    None
}

/// The pages of an image file, each named by the key [`decode_image`]
/// takes: for a TIFF, the directories that are pages (see [`tiff::pages`]);
/// for any other format, its single image as key 0.  Key 0 always names the
/// first image.
///
/// # Errors
///
/// [`DecodeError::Codec`] when a TIFF's header or first directory is
/// unreadable.
pub(crate) fn image_pages(bytes: &[u8]) -> Result<Vec<usize>, DecodeError> {
    match sniff(bytes) {
        Some(ImageFormat::Tiff) => tiff::pages(bytes),
        _ => Ok(vec![0]),
    }
}

/// Decode page `page` (a key [`image_pages`] lists) of arbitrary image
/// bytes to a packed grayscale bitmap at the depth of `target`, sniffing the
/// format first. Gray streams are returned as decoded; RGB streams are
/// converted in their own allocation by [`pdf_raster::rgb_into_gray`] or
/// [`pdf_raster::rgb16_into_gray16`]; an 8-bit page on a 16-bit target is
/// then widened exactly.
///
/// # Errors
///
/// [`DecodeError::Unsupported`] if no codec matches the magic bytes;
/// [`DecodeError::Codec`] if the matched codec rejects the stream.
pub(crate) fn decode_image(
    bytes: &[u8],
    page: usize,
    target: Target,
) -> Result<PageImage, DecodeError> {
    let pixels = match sniff(bytes).ok_or(DecodeError::Unsupported)? {
        ImageFormat::Jpeg => jpeg::decode(bytes, target)?,
        ImageFormat::Png => png::decode(bytes, target)?,
        ImageFormat::WebP => webp::decode(bytes, target)?,
        ImageFormat::Tiff => tiff::decode(bytes, page, target)?,
        ImageFormat::Jxl => jxl::decode(bytes, target)?,
        ImageFormat::Avif => avif::decode(bytes, target)?,
    };
    let page = match pixels {
        CodecPixels::Gray(gray) => PageImage::Eight(gray),
        CodecPixels::Rgb(rgb) => PageImage::Eight(pdf_raster::rgb_into_gray(rgb)),
        CodecPixels::Gray16(gray) => PageImage::Sixteen(gray),
        CodecPixels::Rgb16(rgb) => PageImage::Sixteen(pdf_raster::rgb16_into_gray16(rgb)),
    };
    Ok(page.at(target.depth))
}

/// Reject an image whose header dimensions exceed the page byte budget of
/// `target` at 3 samples per pixel (the RGB buffer a colour decode
/// allocates) of the target's depth, or the side ceiling, before the codec
/// allocates its pixel buffer.
fn guard_dimensions(w: u32, h: u32, target: Target) -> Result<(), DecodeError> {
    pdf_raster::validate_page_bytes(w, h, 3 * target.depth.bytes(), target.max_bytes).map_err(
        |_| DecodeError::TooLarge {
            width: w,
            height: h,
        },
    )
}

/// Unpack gray samples of `bits` (1, 2 or 4) bits, most significant first
/// with each of the `h` rows padded to a byte, into 8-bit luma: sample `v`
/// becomes `v · 255 / (2^bits − 1)`.
fn unpack_gray_bits(packed: &[u8], w: u32, h: u32, bits: u8) -> Result<Vec<u8>, DecodeError> {
    debug_assert!(
        matches!(bits, 1 | 2 | 4),
        "{bits} bits do not pack a byte evenly"
    );
    let per_byte = 8 / usize::from(bits);
    let row_bytes = (w as usize).div_ceil(per_byte);
    let rows = packed.len() / row_bytes.max(1);
    if rows < h as usize {
        return Err(DecodeError::Codec(format!(
            "{} bytes hold {rows} of {h} rows of {w} {bits}-bit samples",
            packed.len()
        )));
    }
    let max = (1u16 << bits) - 1;
    let mask = u8::try_from(max).unwrap_or(u8::MAX);
    let mut out = Vec::with_capacity(w as usize * h as usize);
    for row in packed.chunks_exact(row_bytes).take(h as usize) {
        let samples = row.iter().flat_map(|&byte| {
            (0..per_byte).map(move |i| (byte >> (8 - usize::from(bits) * (i + 1))) & mask)
        });
        out.extend(
            samples
                .take(w as usize)
                .map(|v| u8::try_from(u16::from(v) * 255 / max).unwrap_or(u8::MAX)),
        );
    }
    Ok(out)
}

/// Wrap a tight interleaved RGB buffer as an `Rgb8` bitmap without copying.
fn rgb_from_tight(w: u32, h: u32, rgb: Vec<u8>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, rgb).map(CodecPixels::Rgb)
}

/// Wrap a tight 8-bit luma buffer as a `Gray8` bitmap without copying.
fn gray_from_tight(w: u32, h: u32, gray: Vec<u8>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, gray).map(CodecPixels::Gray)
}

/// [`rgb_from_tight`] of 16-bit samples.
fn rgb16_from_tight(w: u32, h: u32, rgb: Vec<u16>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, rgb).map(CodecPixels::Rgb16)
}

/// [`gray_from_tight`] of 16-bit samples.
fn gray16_from_tight(w: u32, h: u32, gray: Vec<u16>) -> Result<CodecPixels, DecodeError> {
    packed_from_tight(w, h, gray).map(CodecPixels::Gray16)
}

/// Take over a tight `w·h·P::NCOMPS`-sample buffer as a packed bitmap,
/// rejecting a buffer whose length disagrees with the dimensions.
fn packed_from_tight<P: Pixel>(
    w: u32,
    h: u32,
    data: Vec<P::Sample>,
) -> Result<Bitmap<P>, DecodeError> {
    let expected = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(P::NCOMPS));
    if expected != Some(data.len()) {
        return Err(DecodeError::Codec(format!(
            "pixel buffer of {} samples inconsistent with {w}x{h}x{}",
            data.len(),
            P::NCOMPS
        )));
    }
    Ok(Bitmap::from_packed(w, h, data))
}

/// Big-endian 16-bit samples of `bytes`, two bytes each; a trailing odd
/// byte is dropped.
fn be_samples(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_be_bytes(pair))
        .collect()
}

/// Keep the first `keep` of every `channels`-sample pixel in place, dropping
/// the rest (an alpha channel, or a gray image's extra samples).  Sample `c`
/// of pixel `i` moves from `i·channels + c` to `i·keep + c`, never past its
/// source, so a forward pass reads each sample before it is overwritten.
fn keep_leading_samples<T: Copy>(buf: &mut Vec<T>, channels: usize, keep: usize) {
    debug_assert!(
        0 < keep && keep < channels,
        "keep {keep} of {channels} samples"
    );
    let n = buf.len() / channels;
    for i in 0..n {
        for c in 0..keep {
            buf[i * keep + c] = buf[i * channels + c];
        }
    }
    buf.truncate(n * keep);
}

#[cfg(test)]
pub(crate) mod tests;
