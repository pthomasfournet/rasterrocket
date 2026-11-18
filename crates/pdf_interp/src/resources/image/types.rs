//! The types of an image decode: its request, its context, the decoded
//! samples and their description, and the outcome of a resolution.

use pdf::{Dictionary, Document};

#[cfg(feature = "nvjpeg2k")]
use super::DeviceJpx;
#[cfg(feature = "gpu-icc")]
use super::IccHandles;
use super::SoftMask;
use super::colorspace::TintBudget;
use crate::resources::ColorSpace;
#[cfg(feature = "vaapi")]
use gpu::JpegQueueHandle;
#[cfg(feature = "nvjpeg")]
use gpu::nvjpeg::NvJpegDecoder;

/// Outcome of [`resolve_image`](super::resolve_image): distinguishes a
/// genuine decode failure from a legitimately-absent resource so callers can
/// surface errors rather than silently rendering a blank region.
pub enum ImageResolution {
    /// Image decoded successfully.
    Ok(DecodedImage),
    /// The codec returned `None` or the stream was structurally invalid.  The
    /// page is incomplete; the message should be surfaced to the caller.  It
    /// states the reason only: the caller knows the image's name.
    DecodeFailed(String),
    /// The image has no raster appearance: the named `XObject` is of a
    /// non-image subtype (e.g. `/PS`), or the image's colour space paints
    /// nothing (every colorant `/None`, PDF §8.6.6.4).  Nothing is lost by
    /// skipping it.
    Absent,
}

/// A decoded image with the dictionary entries that govern its drawing.
pub struct DecodedImage {
    /// The decoded image.  A decoder that reconstructs at reduced resolution
    /// (see [`ImageRequest::draw_size`]) yields fewer pixels than the stream
    /// holds.
    pub image: ImageDescriptor,
    /// The stream's `/Width`: the source pixel count across, whatever
    /// resolution `image` was decoded at.
    pub source_width: u32,
    /// `/Interpolate`: the image asks to be interpolated when magnified
    /// (PDF 32000-1 §8.9.5.3).
    pub interpolate: bool,
    /// The image dictionary's `/SMask` refers to a soft-mask stream, which
    /// replaces the soft mask of the graphics state for this image (PDF
    /// 32000-1 Table 89).
    pub own_soft_mask: bool,
    /// What the stream lacked when `image` holds padding in place of
    /// samples the stream does not have (a stream shorter than its
    /// dictionary declares, or a codec that stopped short): the image is
    /// drawn, and the page is incomplete.
    pub partial: Option<String>,
    /// The ink amounts of an image in a `Separation`, `DeviceN` or process
    /// CMYK space, or the components of one in a `CalRGB` or ICC RGB
    /// space, kept when the decode was asked to keep them.
    pub tints: Option<ImageTints>,
    /// The tints the decode was asked to keep
    /// (`ImageDecodeCtx::keep_tints`): `tints` is then present for every
    /// image they apply to.
    pub kept_tints: KeptTints,
    /// The rendering intent the image's colours were converted under,
    /// which an ink canvas separates them under too.
    pub intent: crate::RenderingIntent,
}

/// The components of an image an ink canvas paints directly, on the grid
/// of its decoded samples.
///
/// They are the colorant tints of a `Separation` or `DeviceN` space, the
/// components of a process CMYK space, or the components of a `CalRGB` or
/// ICC RGB space.
#[derive(Debug)]
pub struct ImageTints {
    /// The image's colour space.
    pub space: ColorSpace,
    /// One sample per component of `space` per pixel, `/Decode` applied:
    /// 0 is no colorant, or the component's 0, the sample's maximum the
    /// full tint, or the component's 1.
    pub data: TintData,
}

/// The images whose components a decode keeps beside their colours
/// ([`DecodedImage::tints`]).  Each level keeps those of the levels below
/// it, so a decode serves every request of its level or a lower one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum KeptTints {
    /// No image's.
    #[default]
    None,
    /// Those of an image in a `Separation`, `DeviceN` or process CMYK
    /// space, the ink amounts an ink canvas paints.
    Ink,
    /// Those, and the components of an image in a `CalRGB` or ICC RGB
    /// space, which the canvas of a transparency group of that space
    /// paints.
    InkAndRgb,
}

/// The tints of an image at the depth they were kept at: 16 bits for the
/// samples of more than 8 bits of a 16-bit render, 8 otherwise.
#[derive(Debug, Clone)]
pub enum TintData {
    /// 8-bit tints.
    U8(Vec<u8>),
    /// 16-bit tints.
    U16(Vec<u16>),
}

impl TintData {
    /// The bytes the tints hold.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        match self {
            Self::U8(t) => t.len(),
            Self::U16(t) => t.len() * 2,
        }
    }
}

/// Colour space of the decoded image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageColorSpace {
    /// One byte per pixel, 0 = black, 255 = white.
    Gray,
    /// Three bytes per pixel: R, G, B.
    Rgb,
    /// 1-bit mask: pixel = 0 means "paint with current colour", 1 = transparent.
    /// Decoded as one byte per pixel, `0x00` paint and `0xFF` transparent; an
    /// intermediate byte `v` (from area-averaging reduction) paints with
    /// coverage `255 − v`.
    Mask,
}

impl ImageColorSpace {
    /// Decoded bytes per pixel after the codec produces an [`ImageDescriptor`].
    /// `Mask` is byte-expanded by the decoders (one byte per pixel, not bit-packed).
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray | Self::Mask => 1,
            Self::Rgb => 3,
        }
    }
}

/// The compression filter used to store an image in the PDF stream.
///
/// Used by [`PageDiagnostics`](crate::renderer::page::PageDiagnostics) to
/// classify image content without re-reading the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
#[non_exhaustive]
pub enum ImageFilter {
    /// `DCTDecode` — JPEG baseline or progressive.
    Dct = 0,
    /// `JPXDecode` — JPEG 2000.
    Jpx = 1,
    /// `CCITTFaxDecode` — Group 3 (T.4) or Group 4 (T.6) fax compression.
    CcittFax = 2,
    /// `JBIG2Decode` — JBIG2 bilevel compression.
    Jbig2 = 3,
    /// `FlateDecode` — zlib/deflate.
    Flate = 4,
    /// No filter — raw uncompressed pixels.
    Raw = 5,
}

impl ImageFilter {
    /// Map a PDF filter name string to the matching [`ImageFilter`] variant.
    ///
    /// Unrecognised or absent names map to [`ImageFilter::Raw`].  This is the
    /// single canonical mapping used by the renderer, the inline-image decoder,
    /// and the pre-scan pass.
    #[must_use]
    pub(crate) fn from_filter_str(name: Option<&str>) -> Self {
        match name {
            Some("DCTDecode") => Self::Dct,
            Some("JPXDecode") => Self::Jpx,
            Some("CCITTFaxDecode") => Self::CcittFax,
            Some("JBIG2Decode") => Self::Jbig2,
            Some("FlateDecode") => Self::Flate,
            _ => Self::Raw,
        }
    }

    /// Every variant, indexed by its discriminant.
    pub const ALL: [Self; IMAGE_FILTER_COUNT] = [
        Self::Dct,
        Self::Jpx,
        Self::CcittFax,
        Self::Jbig2,
        Self::Flate,
        Self::Raw,
    ];
}

const _: () = {
    let mut i = 0;
    while i < IMAGE_FILTER_COUNT {
        assert!(
            ImageFilter::ALL[i] as usize == i,
            "ImageFilter::ALL is indexed by discriminant"
        );
        i += 1;
    }
};

/// Number of [`ImageFilter`] variants, the length of a per-filter count
/// array indexed by discriminant.
pub const IMAGE_FILTER_COUNT: usize = 6;

/// The filter with the highest non-zero count in `counts` (indexed by
/// discriminant), the later variant on a tie; `None` when every count is 0.
#[must_use]
pub(crate) fn dominant_filter(counts: &[u32; IMAGE_FILTER_COUNT]) -> Option<ImageFilter> {
    counts
        .iter()
        .zip(ImageFilter::ALL)
        .filter(|&(&count, _)| count > 0)
        .max_by_key(|&(&count, _)| count)
        .map(|(_, filter)| filter)
}

/// Backing storage for decoded image pixels.
///
/// Two variants depending on whether the `cache` feature is on:
///
/// - [`ImageData::Cpu`] always available; produced by every CPU
///   decode path (CCITT, JBIG2, raw, JPEG fallback).
/// - `ImageData::Gpu` (feature `cache`) — a `gpu::cache::CachedImage`
///   handed back from the device image cache of the session's backend.
///   Produced by `decode_dct` when a cache is wired in; consumed by the
///   GPU blit kernel, which samples the slab in device memory without
///   uploading the pixels again.
///
/// The enum is [`#[non_exhaustive]`] so future variants (e.g. an
/// `Arc<[u8]>` shared-host variant) can be added without breaking
/// downstream `match` consumers.
///
/// Hot-path access goes through [`Self::as_cpu`] which returns
/// `Option<&[u8]>` — `Some(&[..])` for the CPU variant, `None` for the
/// GPU variant.  GPU consumers must match on the variant explicitly to
/// reach the device pointer.
#[derive(Debug)]
#[non_exhaustive]
pub enum ImageData {
    /// Host-resident pixel bytes — layout defined by [`ImageColorSpace`].
    Cpu(Vec<u8>),
    /// Host-resident 16-bit samples, in the layout of [`ImageColorSpace`]:
    /// the samples of a stream of more than 8 bits a sample, decoded for a
    /// 16-bit render ([`ImageDecodeCtx::depth`]).
    Cpu16(Vec<u16>),
    /// Host-resident samples of two values, one bit each, of a
    /// one-byte-per-pixel [`ImageColorSpace`]; [`Self::as_cpu`] expands
    /// them to bytes.
    Bilevel(Box<super::Bilevel>),
    /// Device-resident image, owned by the device image cache.  The handle
    /// keeps the slab alive past in-flight kernel reads even if the
    /// cache evicts; see `gpu::cache::CachedDeviceImage` docs.
    #[cfg(feature = "cache")]
    Gpu(gpu::cache::CachedImage),
}

/// The reason bilevel samples have no bytes: their expansion, one byte a
/// sample, cannot be allocated.
pub(crate) const BILEVEL_EXPANSION_REFUSED: &str =
    "bilevel samples cannot be expanded to bytes: allocation refused";

impl ImageData {
    /// Borrow the host-resident pixel bytes if this is a CPU-backed image.
    /// Returns `None` for variants whose pixels live in device memory, and
    /// for bilevel samples whose bytes cannot be allocated.
    #[must_use]
    pub fn as_cpu(&self) -> Option<&[u8]> {
        match self {
            Self::Cpu(bytes) => Some(bytes.as_slice()),
            Self::Bilevel(bits) => bits.samples(),
            Self::Cpu16(_) => None,
            #[cfg(feature = "cache")]
            Self::Gpu(_) => None,
        }
    }

    /// The host-resident bytes ([`Self::as_cpu`]), or the reason there are
    /// none.
    ///
    /// # Errors
    /// The reason: 16-bit samples, device-resident samples, or bilevel
    /// samples whose bytes cannot be allocated.
    pub(crate) fn cpu_bytes(&self) -> Result<&[u8], &'static str> {
        match self {
            Self::Cpu(bytes) => Ok(bytes.as_slice()),
            Self::Bilevel(bits) => bits.samples().ok_or(BILEVEL_EXPANSION_REFUSED),
            Self::Cpu16(_) => Err("16-bit samples where 8-bit samples are needed"),
            #[cfg(feature = "cache")]
            Self::Gpu(_) => Err("pixels device-resident but GPU dispatch unavailable"),
        }
    }

    /// Whether the samples are host-resident, of either depth.
    #[must_use]
    pub const fn is_host(&self) -> bool {
        matches!(self, Self::Cpu(_) | Self::Cpu16(_) | Self::Bilevel(_))
    }

    /// The host-resident 16-bit samples of a [`Self::Cpu16`] image
    /// narrowed to bytes ([`color::Sample::to_u8`]); any other variant is
    /// left as it is.
    pub(crate) fn narrow(&mut self) {
        if let Self::Cpu16(wide) = self {
            *self = Self::Cpu(wide.iter().map(|&v| color::Sample::to_u8(v)).collect());
        }
    }

    /// The host-resident bytes of a [`Self::Cpu`] image to modify, those
    /// of a [`Self::Bilevel`] image expanded first; `None` for any other
    /// variant, or when the expanded bytes cannot be allocated, which
    /// leaves the bilevel image as it is.
    pub(crate) fn as_cpu_mut(&mut self) -> Option<&mut Vec<u8>> {
        if let Self::Bilevel(bits) = self {
            let bytes = bits.take_samples()?;
            *self = Self::Cpu(bytes);
        }
        match self {
            Self::Cpu(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// Borrow the host-resident 16-bit samples of a [`Self::Cpu16`] image.
    #[must_use]
    pub const fn as_cpu16(&self) -> Option<&[u16]> {
        match self {
            Self::Cpu16(samples) => Some(samples.as_slice()),
            _ => None,
        }
    }

    /// The bytes of memory the samples occupy once drawn, host or device:
    /// those of a bilevel image are its bits and their expansion to bytes,
    /// which the first [`Self::as_cpu`] makes and keeps
    /// ([`super::Bilevel::heap_bytes`]).  No host memory is read for a
    /// device-resident image.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Cpu(bytes) => bytes.len(),
            Self::Cpu16(samples) => size_of_val(samples.as_slice()),
            Self::Bilevel(bits) => bits.heap_bytes(),
            #[cfg(feature = "cache")]
            Self::Gpu(cached) => usize::try_from(cached.vram_bytes()).unwrap_or(usize::MAX),
        }
    }
}

/// An image `XObject` to resolve, by resource name, and the size it will be
/// drawn at.
#[derive(Clone, Copy, Debug)]
pub struct ImageRequest<'a> {
    /// The `XObject` resource name.
    pub name: &'a [u8],
    /// Device-pixel extent `(width, height)` the image's unit square maps to,
    /// when known.  A decoder that can reconstruct at reduced resolution
    /// (JPEG 2000) decodes at no less than this along either axis; `None`
    /// decodes at full resolution.  Every other decoder ignores it, so their
    /// decodes serve draws of any size.
    pub draw_size: Option<(u32, u32)>,
}

impl<'a> ImageRequest<'a> {
    /// Request `name` at full resolution.
    #[must_use]
    pub const fn full(name: &'a [u8]) -> Self {
        Self {
            name,
            draw_size: None,
        }
    }
}

/// Decoded image data, ready for blitting onto the page bitmap.
///
/// See the table of the `image` module for the exact byte layout
/// per [`ImageColorSpace`].
#[derive(Debug)]
pub struct ImageDescriptor {
    /// Pixel width of the decoded image.
    pub width: u32,
    /// Pixel height of the decoded image.
    pub height: u32,
    /// Colour interpretation of `data`.
    pub color_space: ImageColorSpace,
    /// Pixel storage — host or (in future) device-resident.  Layout defined
    /// by `color_space` (see the `image` module).
    pub data: ImageData,
    /// Optional mask (`/SMask`, `/Mask`, or an embedded alpha channel) over
    /// the same unit square, at its own resolution, which need not be the
    /// image's.  `None` means every pixel is fully opaque.
    pub smask: Option<SoftMask>,
    /// The compression filter that was applied to this image in the PDF stream.
    pub filter: ImageFilter,
}

/// Everything an image decode reads besides the image's own bytes.
///
/// Holds the document, the `ColorSpace` resources an inline image's `/CS`
/// name resolves through, the GPU decoders available to the image's filter
/// and the device image cache.  Each handle exists only with its feature;
/// `None` selects the CPU path or skips the cache.
pub struct ImageDecodeCtx<'a> {
    /// Document the image's indirect objects resolve in.
    pub doc: &'a Document,
    /// The `/ColorSpace` subdictionary of the resources in effect, through
    /// which an inline image's `/CS` name other than a device space resolves
    /// (PDF §8.9.7).
    pub color_spaces: Option<&'a Dictionary>,
    /// The tint-transform work left to the page's images; `None` bounds
    /// each image by its own budget alone.
    pub tint_budget: Option<&'a TintBudget>,
    /// nvJPEG decoder for large `DCTDecode` images.
    #[cfg(feature = "nvjpeg")]
    pub nvjpeg: Option<&'a mut NvJpegDecoder>,
    /// VA-API JPEG queue for large `DCTDecode` images.
    #[cfg(feature = "vaapi")]
    pub vaapi: Option<&'a JpegQueueHandle>,
    /// The nvJPEG2000 decoder large `JPXDecode` images may use, obtained
    /// on first use when it is supplied on demand.
    #[cfg(feature = "nvjpeg2k")]
    pub nvjpeg2k: Option<DeviceJpx<'a>>,
    /// ICC conversion handles.
    #[cfg(feature = "gpu-icc")]
    pub icc: IccHandles<'a>,
    /// Device-resident image cache.
    #[cfg(feature = "cache")]
    pub image_cache: Option<&'a gpu::cache::ImageCache>,
    /// Identity of the document within the cache.
    #[cfg(feature = "cache")]
    pub doc_id: Option<gpu::cache::DocId>,
    /// The images whose components are kept beside their colours
    /// ([`DecodedImage::tints`]).
    pub keep_tints: KeptTints,
    /// The working space the image's colours convert into, and the
    /// rendering intent they convert under: the image's own `/Intent`,
    /// else the graphics state's (PDF §8.6.5.8).
    pub target: crate::ColourTarget,
    /// The depth of the render the image is drawn in.  At 16 bits a stream
    /// of more than 8 bits a sample keeps its samples at 16 bits
    /// ([`ImageData::Cpu16`]); every other stream decodes to 8-bit samples
    /// at either depth.
    pub depth: color::SampleDepth,
}

impl<'a> ImageDecodeCtx<'a> {
    /// A context for `doc` with no colour-space resources and every GPU
    /// handle off.
    #[must_use]
    pub const fn cpu(doc: &'a Document) -> Self {
        Self {
            doc,
            color_spaces: None,
            tint_budget: None,
            #[cfg(feature = "nvjpeg")]
            nvjpeg: None,
            #[cfg(feature = "vaapi")]
            vaapi: None,
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: None,
            #[cfg(feature = "gpu-icc")]
            icc: IccHandles::none(),
            #[cfg(feature = "cache")]
            image_cache: None,
            #[cfg(feature = "cache")]
            doc_id: None,
            keep_tints: KeptTints::None,
            target: crate::ColourTarget::SRGB,
            depth: color::SampleDepth::Eight,
        }
    }
}
