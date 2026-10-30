//! Soft-mask (`SMask`) decoding for PDF images.
//!
//! A PDF `SMask` is a grayscale image whose pixel values become the alpha channel
//! of the parent image.  This module decodes `SMask` streams, explicit `/Mask`
//! stencils and colour-key masks into a [`SoftMask`].  A mask decoded from its
//! own stream keeps that stream's resolution: the blit samples it through its
//! own grid, so its resolution is independent of the resolution the parent
//! image was decoded at.

use pdf::{Document, Object, ObjectId, Stream};

use super::colorspace::ImageColour;
use super::dispatch::{ImageStream, decode_image_stream};
use super::{ImageColorSpace, ImageData, ImageDecodeCtx, ImageDescriptor};
use super::{decode_budget, try_with_capacity, validated_dims};
use crate::resources::colorspace::ColorSpace;

// ── Public API ────────────────────────────────────────────────────────────────

/// Per-pixel alpha covering an image's unit square.
///
/// `alpha` holds exactly `width × height` bytes in row-major order, top row
/// first: `0x00` = fully transparent, `0xFF` = fully opaque.  The grid may
/// differ from the parent image's; both span the same unit square, so the
/// blit maps each device pixel into each grid separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftMask {
    width: u32,
    height: u32,
    alpha: Vec<u8>,
    /// The mask asks to be interpolated when magnified (its own
    /// `/Interpolate`, PDF §8.9.5.3).
    interpolate: bool,
}

impl SoftMask {
    /// Wrap `alpha` as a `width × height` mask that is not interpolated.
    /// Returns `None` when either dimension is zero or `alpha` does not
    /// hold exactly `width × height` bytes.
    #[must_use]
    pub fn new(width: u32, height: u32, alpha: Vec<u8>) -> Option<Self> {
        let expected = (width as usize).checked_mul(height as usize)?;
        (expected != 0 && alpha.len() == expected).then_some(Self {
            width,
            height,
            alpha,
            interpolate: false,
        })
    }

    /// The same mask, interpolated when magnified if `interpolate`.
    #[must_use]
    pub const fn interpolated(mut self, interpolate: bool) -> Self {
        self.interpolate = interpolate;
        self
    }

    /// Whether the mask asks to be interpolated when magnified.
    #[must_use]
    pub const fn interpolate(&self) -> bool {
        self.interpolate
    }

    /// Mask width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Mask height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Alpha bytes, `width × height`, row-major.
    #[must_use]
    pub fn alpha(&self) -> &[u8] {
        &self.alpha
    }
}

/// Decode a soft-mask (`SMask`) image stream at its own resolution.
///
/// The stream decodes through the same filter dispatch as a base image, so
/// any filter chain an image may use is accepted.  The mask's dimensions
/// are those the codec reports, which for `CCITTFaxDecode`, `JBIG2Decode`,
/// `DCTDecode` and `JPXDecode` take precedence over the dictionary's
/// `/Width` and `/Height`.  The decoded gray samples, after the mask's own
/// `/Decode`, are the alpha directly (§11.6.5.2); a stream that decodes to
/// three components (a producer's RGB JPEG) yields its luminosity
/// (§11.3.5.3).  The mask's own `/Interpolate` is kept with it.
///
/// Returns `None` when the stream is unresolvable, fails to decode, decodes
/// to a stencil, or its dimensions are degenerate; the caller then blits
/// the image unmasked.
pub(super) fn decode_smask(ctx: &mut ImageDecodeCtx<'_>, id: ObjectId) -> Option<SoftMask> {
    use crate::resources::dict_ext::DictExt as _;

    // Bind the Arc to a local so the borrow into the stream below stays alive.
    let obj_arc = ctx.doc.get_object(id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;
    let desc = decode_mask_stream(ctx, stream, false, "SMask")?;
    let alpha = match desc.color_space {
        ImageColorSpace::Gray => alpha_from_decoded("SMask", &desc)?,
        ImageColorSpace::Rgb => {
            log::debug!(
                "image: SMask (object {id:?}) holds three components; using its luminosity"
            );
            let bytes = desc.data.as_cpu()?;
            let expected = (desc.width as usize).checked_mul(desc.height as usize)?;
            if bytes.len() < expected * 3 {
                return None;
            }
            bytes[..expected * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|&[r, g, b]| color::convert::luminosity(r, g, b))
                .collect()
        }
        ImageColorSpace::Mask => {
            log::warn!("image: SMask (object {id:?}) decoded as a stencil — skipping mask");
            return None;
        }
    };
    let interpolate = stream.dict.get_bool(b"Interpolate").unwrap_or(false);
    Some(SoftMask::new(desc.width, desc.height, alpha)?.interpolated(interpolate))
}

/// Decode an explicit `/Mask N 0 R` stencil-mask image into an alpha buffer.
///
/// This is the PDF §8.9.6.3 *explicit mask* (an image `XObject` with
/// `/ImageMask true`), distinct from `/SMask` (§11.6.5.2 soft mask) and from
/// the `/Mask [colour-key]` array form (§8.9.6.4).  The mask designates which
/// parts of the **base image** are painted: per §8.9.6.3 with the default
/// `Decode [0 1]`, mask sample 0 → the base image is painted (opaque),
/// mask sample 1 → masked out (the background shows through).
///
/// The stencil decodes through the shared image dispatch to
/// [`ImageColorSpace::Mask`] bytes using the §8.9.6.1 fill-stencil
/// convention: `0x00` = "paint", `0xFF` = "transparent".  For an explicit
/// mask those same markers map directly onto base-image alpha — a "paint"
/// point shows the base image (alpha `0xFF`), a "transparent" point shows
/// the background (alpha `0x00`).  Hence `alpha = !mask_byte`.  `Decode
/// [1 0]` on the mask is already folded into the decoder output, so no extra
/// inversion is applied here.
///
/// The mask keeps the stencil's decoded resolution.  Returns `None` if the
/// referenced object is not a usable stencil image so the caller can blit the
/// base image unmasked rather than discard it.
pub(super) fn decode_explicit_mask(ctx: &mut ImageDecodeCtx<'_>, id: ObjectId) -> Option<SoftMask> {
    use crate::resources::dict_ext::DictExt as _;

    let obj_arc = ctx.doc.get_object(id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;

    // §8.9.6.3: an explicit mask must be an image with ImageMask true.
    if !stream.dict.get_bool(b"ImageMask").unwrap_or(false) {
        log::warn!("image: /Mask object {id:?} is not an ImageMask stencil — skipping mask");
        return None;
    }

    let desc = decode_mask_stream(ctx, stream, true, "/Mask")?;

    // A stencil decoded with is_mask=true must yield ImageColorSpace::Mask.
    // If it does not, the bytes are *not* the §8.9.6.1 paint/transparent
    // markers, so `!byte` would invert arbitrary samples into garbage alpha —
    // exactly the silent-wrong mask-polarity class.  Reject loudly instead of
    // a debug-only assert that vanishes in release builds.
    if desc.color_space != ImageColorSpace::Mask {
        log::warn!(
            "image: /Mask object {id:?} decoded as {:?}, expected Mask stencil — skipping mask",
            desc.color_space
        );
        return None;
    }

    // §8.9.6.3: paint marker (0x00) → base image opaque (alpha 0xFF);
    // transparent marker (0xFF) → background shows (alpha 0x00).
    let mut alpha = alpha_from_decoded("/Mask", &desc)?;
    for b in &mut alpha {
        *b = !*b;
    }
    SoftMask::new(desc.width, desc.height, alpha)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Decode a mask stream (an `SMask` when `is_mask` is false, an explicit
/// stencil `/Mask` when true) through the shared image dispatch.  `what`
/// names the mask kind in log messages.
fn decode_mask_stream(
    ctx: &mut ImageDecodeCtx<'_>,
    stream: &Stream,
    is_mask: bool,
    what: &str,
) -> Option<ImageDescriptor> {
    use crate::resources::dict_ext::DictExt as _;

    let w_raw = stream.dict.get_i64(b"Width")?;
    let h_raw = stream.dict.get_i64(b"Height")?;
    let Some((width, height)) = validated_dims(w_raw, h_raw) else {
        log::warn!("image: {what} degenerate dimensions {w_raw}×{h_raw}, skipping mask");
        return None;
    };
    if let Err(msg) = decode_budget(width, height, 1) {
        log::warn!("image: {what}: {msg} — skipping mask");
        return None;
    }
    // A mask's samples are its own alpha or stencil: the `/Default*`
    // spaces of the page do not apply to them.
    let src = ImageStream {
        stream,
        width,
        height,
        is_mask,
        colour: ImageColour::resolve(ctx.doc, &stream.dict, None),
    };
    decode_image_stream(
        ctx,
        &src,
        None,
        #[cfg(feature = "cache")]
        None,
        #[cfg(feature = "gpu-jpeg-huffman")]
        None::<&mut gpu::jpeg_decoder::JpegGpuDecoder<gpu::backend::cuda::CudaBackend>>,
    )
    .map_err(|e| log::warn!("image: {what}: {e} — skipping mask"))
    .ok()
    .map(|decoded| decoded.image)
}

/// Extract exactly `width × height` host-resident bytes from a decoded mask
/// descriptor; `what` names the mask kind when the buffer is short.
///
/// A short buffer is a decode failure, never silently-wrong alpha.
fn alpha_from_decoded(what: &str, desc: &ImageDescriptor) -> Option<Vec<u8>> {
    let bytes = desc.data.as_cpu()?;
    let expected = (desc.width as usize).checked_mul(desc.height as usize)?;
    if bytes.len() < expected {
        log::warn!(
            "image: {what} decoded {} bytes, expected {expected} for {}×{} — skipping mask",
            bytes.len(),
            desc.width,
            desc.height
        );
        return None;
    }
    let mut alpha = try_with_capacity(expected)?;
    alpha.extend_from_slice(&bytes[..expected]);
    Some(alpha)
}

/// Build an alpha buffer from a `/Mask [colour-key]` array (§8.9.6.4) over
/// 8-bit decoded samples: `base` holds `components` bytes per pixel.
///
/// See [`colour_key_mask_with`].  Returns `None` when the array does not
/// match the component count or `base` is shorter than the image.
pub(super) fn colour_key_mask(
    ranges: &[Object],
    base: &[u8],
    components: usize,
    img_w: u32,
    img_h: u32,
) -> Option<SoftMask> {
    let npix = (img_w as usize).checked_mul(img_h as usize)?;
    if base.len() < npix.checked_mul(components)? {
        return None;
    }
    colour_key_mask_with(ranges, components, img_w, img_h, |p, c| {
        u32::from(base[p * components + c])
    })
}

/// Build an alpha buffer from a `/Mask [colour-key]` array (§8.9.6.4).
///
/// The array holds `2 × n` integers `[min₁ max₁ … minₙ maxₙ]`, one range
/// per colour component.  `sample(p, c)` is component `c` of pixel `p`
/// (row-major) as the ranges are expressed.  A pixel whose every component
/// falls within its `[min, max]` range is masked (alpha `0x00`); every
/// other pixel is opaque (alpha `0xFF`).
///
/// The mask shares the image's grid.  Returns `None` when the array length
/// does not match `components` or a range is inverted, so the caller blits
/// unmasked.
pub(super) fn colour_key_mask_with(
    ranges: &[Object],
    components: usize,
    img_w: u32,
    img_h: u32,
    sample: impl Fn(usize, usize) -> u32,
) -> Option<SoftMask> {
    if components == 0 || ranges.len() != components * 2 {
        return None;
    }
    let bounds: Vec<i64> = ranges.iter().filter_map(pdf::Object::as_i64).collect();
    if bounds.len() != components * 2 {
        return None;
    }
    // §8.9.6.4: each component range is `min max` with min ≤ max.  An inverted
    // pair would make `v >= min && v <= max` unsatisfiable for that component,
    // silently producing a fully-opaque image (no masking) — the silent-wrong
    // class.  Reject so the caller logs and blits unmasked instead.
    if (0..components).any(|c| bounds[c * 2] > bounds[c * 2 + 1]) {
        log::warn!("image: colour-key /Mask has an inverted min>max range — skipping mask");
        return None;
    }
    let npix = (img_w as usize).checked_mul(img_h as usize)?;
    let alpha: Vec<u8> = (0..npix)
        .map(|p| {
            let in_range = (0..components).all(|c| {
                let v = i64::from(sample(p, c));
                v >= bounds[c * 2] && v <= bounds[c * 2 + 1]
            });
            if in_range { 0x00 } else { 0xFF }
        })
        .collect();
    SoftMask::new(img_w, img_h, alpha)
}

/// The `/Matte` array of soft-mask stream `id`: the matte colour, with one
/// component per component of the parent image's colour space
/// (§11.6.5.3).  `None` when the mask has no `/Matte` or an entry is not a
/// number.
pub(super) fn matte(doc: &Document, id: ObjectId) -> Option<Vec<f64>> {
    let obj_arc = doc.get_object(id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;
    let Object::Array(arr) = stream.dict.get(b"Matte")? else {
        return None;
    };
    arr.iter().map(Object::as_f64).collect()
}

/// Recover the colours of an image pre-blended with the matte colour
/// `matte` (§11.6.5.3), given in the image's colour space `cs`
/// (`DeviceGray` when the image names none).
///
/// Each stored component is `c' = m + α·(c − m)`; the inverse
/// `c = m + (c' − m)/α`, clamped to the component range, is applied to the
/// decoded device samples with the matte converted to the same device
/// space.  Pixels of alpha 0 are left as stored.  The mask must share the
/// image's grid, as §11.6.5.3 requires of a mask carrying `/Matte`;
/// otherwise, or when the matte does not match the colour space's
/// component count, the image is left as stored.
pub(super) fn remove_matte(
    doc: &Document,
    cs: Option<&ColorSpace>,
    matte: &[f64],
    img: &mut ImageDescriptor,
) {
    let cs = cs.unwrap_or(&ColorSpace::DeviceGray);
    if matte.len() != cs.ncomponents() {
        log::warn!(
            "image: /Matte has {} components, the image's colour space {}; drawing as stored",
            matte.len(),
            cs.ncomponents()
        );
        return;
    }
    let rgb = cs.convert_to_rgb(doc, matte);
    let device: &[u8] = match img.color_space {
        ImageColorSpace::Rgb => &rgb,
        ImageColorSpace::Gray => &rgb[..1],
        ImageColorSpace::Mask => return,
    };
    let Some(mask) = &img.smask else { return };
    if (mask.width(), mask.height()) != (img.width, img.height) {
        log::warn!(
            "image: /Matte mask is {}×{}, the image {}×{}; drawing as stored",
            mask.width(),
            mask.height(),
            img.width,
            img.height
        );
        return;
    }
    #[cfg_attr(
        not(feature = "cache"),
        expect(
            clippy::infallible_destructuring_match,
            reason = "the cache-feature build adds a device-resident arm; keep one shape"
        )
    )]
    let data = match &mut img.data {
        ImageData::Cpu(bytes) => bytes,
        #[cfg(feature = "cache")]
        ImageData::Gpu(_) => {
            log::warn!("image: /Matte needs host-resident samples; drawing as stored");
            return;
        }
    };
    unmatte(data, mask.alpha(), device);
}

/// Invert `c' = m + α·(c − m)` (§11.6.5.3) over interleaved 8-bit
/// `samples`, `matte.len()` components per pixel, with `alpha` holding one
/// byte per pixel: `c = m + (c' − m)/α`, rounded and clamped to 0..=255.
/// Pixels of alpha 0 are left as stored.
pub(super) fn unmatte(samples: &mut [u8], alpha: &[u8], matte: &[u8]) {
    if matte.is_empty() {
        return;
    }
    for (px, &a) in samples.chunks_exact_mut(matte.len()).zip(alpha) {
        if a == 0 {
            continue;
        }
        for (c, &m) in px.iter_mut().zip(matte) {
            let v = f32::from(m) + (f32::from(*c) - f32::from(m)) * 255.0 / f32::from(a);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=255 before the cast"
            )]
            {
                *c = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_helpers::make_doc_with_stream;

    /// Decode object (2, 0) of `doc` as an `SMask`.
    fn smask_of(doc: &pdf::Document) -> Option<SoftMask> {
        decode_smask(&mut ImageDecodeCtx::cpu(doc), (2, 0))
    }

    /// Decode object (2, 0) of `doc` as an explicit stencil `/Mask`.
    fn explicit_of(doc: &pdf::Document) -> Option<SoftMask> {
        decode_explicit_mask(&mut ImageDecodeCtx::cpu(doc), (2, 0))
    }

    /// `/Image` dictionary entries of a `w × h` 8-bit gray `SMask`.
    fn smask_dict(w: u32, h: u32, filter: &str) -> String {
        format!(
            " /Subtype /Image /Width {w} /Height {h} /BitsPerComponent 8 \
              /ColorSpace /DeviceGray {filter}"
        )
    }

    #[test]
    fn soft_mask_requires_exact_length() {
        assert!(SoftMask::new(2, 2, vec![0; 3]).is_none());
        assert!(SoftMask::new(2, 2, vec![0; 5]).is_none());
        assert!(SoftMask::new(0, 2, Vec::new()).is_none());
        let m = SoftMask::new(2, 1, vec![7, 9]).expect("exact length");
        assert_eq!((m.width(), m.height(), m.alpha()), (2, 1, &[7u8, 9][..]));
    }

    /// An `SMask` keeps its stream's resolution whatever the parent image's.
    #[test]
    fn smask_keeps_native_resolution() {
        let doc = crate::test_helpers::make_doc_with_stream(
            &[1u8, 2, 3, 4, 5, 6],
            " /Subtype /Image /Width 3 /Height 2 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray",
        );
        let m = smask_of(&doc).expect("raw 8-bpc SMask must decode");
        assert_eq!((m.width(), m.height()), (3, 2));
        assert_eq!(m.alpha(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn alpha_from_decoded_short_buffer_returns_none() {
        // A bitonal codec that reports 4×4 but produced only 3 bytes must be
        // skipped loudly with the codec named.
        let desc = ImageDescriptor {
            width: 4,
            height: 4,
            color_space: ImageColorSpace::Gray,
            data: crate::resources::image::ImageData::Cpu(vec![0u8, 0xFF, 0x00]),
            smask: None,
            filter: crate::resources::image::ImageFilter::Raw,
        };
        assert!(alpha_from_decoded("JBIG2Decode", &desc).is_none());
    }

    #[test]
    fn alpha_from_decoded_exact_buffer_passes_through() {
        // Exact-length buffer is consumed as alpha directly (§11.6.5.2: the
        // decoded sample value IS the alpha — no inversion).
        let desc = ImageDescriptor {
            width: 2,
            height: 2,
            color_space: ImageColorSpace::Gray,
            data: crate::resources::image::ImageData::Cpu(vec![0x00, 0xFF, 0x10, 0xF0]),
            smask: None,
            filter: crate::resources::image::ImageFilter::Raw,
        };
        assert_eq!(
            alpha_from_decoded("CCITTFaxDecode", &desc).unwrap(),
            vec![0x00, 0xFF, 0x10, 0xF0]
        );
    }

    #[test]
    fn decode_smask_raw_bpc8_too_short_is_transparent_where_missing() {
        // 2×2 SMask needs 4 bytes; the two missing samples are alpha 0.
        let doc = make_doc_with_stream(&[0u8, 255], &smask_dict(2, 2, ""));
        assert_eq!(
            smask_of(&doc).expect("short SMask decodes").alpha(),
            &[0, 255, 0, 0]
        );
    }

    /// Transport filters other than Flate, alone or in a chain, reach the
    /// raw-sample path.
    #[test]
    fn smask_accepts_every_transport_filter() {
        let cases: [(&str, &[u8]); 5] = [
            ("/Filter /ASCIIHexDecode", b"0a141e28>"),
            ("/Filter /RunLengthDecode", &[3, 10, 20, 30, 40, 128]),
            (
                "/Filter [/ASCIIHexDecode /RunLengthDecode]",
                b"030a141e2880>",
            ),
            // One uncompressed Brotli meta-block holding the four samples.
            ("/Filter /BrotliDecode", &[139, 1, 128, 10, 20, 30, 40, 3]),
            (
                "/Filter [/ASCIIHexDecode /BrotliDecode]",
                b"8b01800a141e2803>",
            ),
        ];
        for (filter, data) in cases {
            let doc = make_doc_with_stream(data, &smask_dict(2, 2, filter));
            let m = smask_of(&doc).unwrap_or_else(|| panic!("{filter}: SMask must decode"));
            assert_eq!(m.alpha(), &[10, 20, 30, 40], "{filter}");
        }
    }

    /// A bilevel codec in a stencil `/Mask` receives its own entry of an
    /// array-form `/DecodeParms`: here Group 4 (`K -1`), where one `1` bit
    /// codes an all-white row (vertical mode, offset 0, against the
    /// imaginary white reference row).
    #[test]
    fn explicit_mask_ccitt_reads_array_decode_parms() {
        let doc = make_doc_with_stream(
            &[0xC0],
            " /Subtype /Image /Width 8 /Height 2 /ImageMask true \
              /Filter [/CCITTFaxDecode] /DecodeParms [<< /K -1 /Columns 8 /Rows 2 >>]",
        );
        let mask = explicit_of(&doc).expect("G4 stencil must decode");
        // White is sample 1: transparent, so the base image is masked out.
        assert_eq!((mask.width(), mask.height()), (8, 2));
        assert!(mask.alpha().iter().all(|&a| a == 0x00));
    }

    /// A codec behind a transport filter decodes the transport output.
    #[test]
    fn smask_dct_behind_ascii_hex() {
        let hex: String = JPEG_1X1_GRAY128
            .iter()
            .map(|b| format!("{b:02x}"))
            .chain(std::iter::once(">".to_owned()))
            .collect();
        let doc = make_doc_with_stream(
            hex.as_bytes(),
            &smask_dict(1, 1, "/Filter [/ASCIIHexDecode /DCTDecode]"),
        );
        let m = smask_of(&doc).expect("AHx → DCT SMask must decode");
        assert!(m.alpha()[0] > 64 && m.alpha()[0] < 192);
    }

    /// A 1×1 grayscale JPEG, value 128, quality 90, generated by:
    ///   from PIL import Image, io
    ///   img = Image.new('L', (1, 1), 128)
    ///   `buf = io.BytesIO(); img.save(buf, 'JPEG', quality=90)`
    const JPEG_1X1_GRAY128: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00,
        0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02,
        0x02, 0x03, 0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04,
        0x05, 0x0A, 0x07, 0x07, 0x06, 0x08, 0x0C, 0x0A, 0x0C, 0x0C, 0x0B, 0x0A, 0x0B, 0x0B, 0x0D,
        0x0E, 0x12, 0x10, 0x0D, 0x0E, 0x11, 0x0E, 0x0B, 0x0B, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14,
        0x15, 0x15, 0x15, 0x0C, 0x0F, 0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xFF,
        0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00,
        0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05,
        0x04, 0x04, 0x00, 0x00, 0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A,
        0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
        0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
        0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93,
        0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9,
        0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6,
        0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
        0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x2B, 0xFF,
        0xD9,
    ];

    /// A `DCTDecode` `SMask` takes the JPEG's dimensions and its luma as alpha.
    #[test]
    fn decode_smask_dct_1x1_gray() {
        let doc = make_doc_with_stream(JPEG_1X1_GRAY128, &smask_dict(1, 1, "/Filter /DCTDecode"));
        let m = smask_of(&doc).expect("PIL-produced 1×1 JPEG must decode successfully");
        assert_eq!((m.width(), m.height()), (1, 1));
        // PIL encoded value 128 at quality=90; after JPEG quantisation the
        // decoded value is in the broad mid-grey range.
        assert!(
            m.alpha()[0] > 64 && m.alpha()[0] < 192,
            "decoded alpha {:#04x} outside plausible mid-grey range [65, 191]",
            m.alpha()[0]
        );
    }

    /// A truncated (corrupt) `DCTDecode` `SMask` is rejected, not a panic.
    #[test]
    fn decode_smask_dct_corrupt_returns_none() {
        let doc =
            make_doc_with_stream(&[0xFF, 0xD8, 0xFF], &smask_dict(1, 1, "/Filter /DCTDecode"));
        assert!(smask_of(&doc).is_none());
    }

    // ── Explicit /Mask (§8.9.6.3) regression ─────────────────────────────────

    /// PDF §8.9.6.3 explicit `/Mask` polarity.
    ///
    /// A `/Mask N 0 R` referencing an `/ImageMask true` stencil must produce a
    /// base-image alpha buffer where the stencil's *paint* points are opaque
    /// (`0xFF`) and its *masked-out* points are transparent (`0x00`).  The bug
    /// was that `/Mask` was never resolved at all, leaving `JPXDecode` pages
    /// (e.g. 1853-ragon-orthodoxie-maconnique p62) blitted fully opaque over
    /// the whole page and rendering near-all-dark instead of mostly-white.
    ///
    /// Stencil bytes (raw 1-bpp, default `Decode [0 1]`): a `0` bit paints
    /// (decoder → `0x00`), a `1` bit is transparent (decoder → `0xFF`).  Row
    /// of 8 px `0b0101_0101`: bits 0,1,0,1,0,1,0,1 ⇒ stencil bytes
    /// `[00 FF 00 FF 00 FF 00 FF]` ⇒ alpha `!byte` = `[FF 00 FF 00 …]`.
    #[test]
    fn explicit_mask_polarity_paint_is_opaque() {
        // 1 row × 8 px, 1 bpp ⇒ one byte 0b0101_0101.
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0b0101_0101u8],
            " /Subtype /Image /Width 8 /Height 1 /ImageMask true \
              /BitsPerComponent 1",
        );
        let mask = explicit_of(&doc).expect("explicit /Mask stencil must decode");
        // Paint bit (0) → opaque base image (0xFF); 1 bit → background (0x00).
        assert_eq!((mask.width(), mask.height()), (8, 1));
        assert_eq!(
            mask.alpha(),
            &[0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00]
        );
    }

    /// A `/Mask` object that is *not* an `ImageMask` stencil must be rejected
    /// (returns `None`) so the caller blits the base image unmasked rather
    /// than mis-applying a non-stencil image as alpha.
    /// An `SMask` stream's `/Decode [1 0]` inverts its samples before they
    /// are read as alpha.
    #[test]
    fn smask_decode_array_inverts_alpha() {
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0x00u8, 0xFF, 0x40],
            " /Subtype /Image /Width 3 /Height 1 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray /Decode [1 0]",
        );
        let mask = smask_of(&doc).expect("raw 8-bpc SMask must decode");
        assert_eq!(mask.alpha(), &[0xFF, 0x00, 0xBF]);
    }

    #[test]
    fn explicit_mask_rejects_non_imagemask() {
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0u8, 0, 0, 0],
            " /Subtype /Image /Width 2 /Height 2 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray",
        );
        assert!(explicit_of(&doc).is_none());
    }

    // ── Colour-key /Mask (§8.9.6.4) ──────────────────────────────────────────

    #[test]
    fn colour_key_mask_gray_in_range_is_transparent() {
        // Range [10 20]: pixels 15 (in) → masked (0x00); 5 and 25 (out) → 0xFF.
        let base = [15u8, 5, 25, 12];
        let ranges = [Object::Integer(10), Object::Integer(20)];
        let mask = colour_key_mask(&ranges, &base, 1, 4, 1).expect("valid colour-key");
        assert_eq!(mask.alpha(), &[0x00, 0xFF, 0xFF, 0x00]);
    }

    #[test]
    fn colour_key_mask_rgb_all_components_must_match() {
        // Range R[0 10] G[0 10] B[0 10]: px0 (5,5,5) in → 0x00;
        // px1 (5,5,99) one comp out → 0xFF.
        let base = [5u8, 5, 5, 5, 5, 99];
        let ranges = [
            Object::Integer(0),
            Object::Integer(10),
            Object::Integer(0),
            Object::Integer(10),
            Object::Integer(0),
            Object::Integer(10),
        ];
        let mask = colour_key_mask(&ranges, &base, 3, 2, 1).expect("valid colour-key");
        assert_eq!(mask.alpha(), &[0x00, 0xFF]);
    }

    #[test]
    fn colour_key_mask_arity_mismatch_returns_none() {
        // 3 ints for a 1-component image (needs exactly 2) ⇒ None.
        let ranges = [Object::Integer(0), Object::Integer(1), Object::Integer(2)];
        assert!(colour_key_mask(&ranges, &[0u8], 1, 1, 1).is_none());
    }

    /// §8.9.6.4 requires `min ≤ max`.  An inverted range must fail loudly-
    /// graceful (`None` → caller blits unmasked) rather than silently leaving
    /// the whole image opaque, which the v1 sin class would have done.
    #[test]
    fn colour_key_mask_inverted_range_returns_none() {
        // min=20 > max=10 for a 1-component image ⇒ None, not all-opaque.
        let ranges = [Object::Integer(20), Object::Integer(10)];
        assert!(colour_key_mask(&ranges, &[15u8, 5], 1, 2, 1).is_none());
    }

    /// A `/Mask N 0 R` whose referenced stream is *not* a stencil (decodes as
    /// Gray/Rgb, not [`ImageColorSpace::Mask`]) must be rejected.  Feeding
    /// non-marker bytes through `!byte` would invert arbitrary samples into
    /// garbage alpha — the silent-wrong mask-polarity class this guard exists
    /// to stop.  The `/ImageMask true` dict claim is honoured by the decoder,
    /// so this is exercised via the release-safe colour-space check rather
    /// than a debug-only assert.
    #[test]
    fn explicit_mask_imagemask_false_is_rejected_before_invert() {
        // No /ImageMask true ⇒ early ImageMask-flag guard rejects it.
        let doc = crate::test_helpers::make_doc_with_stream(
            &[0x12u8, 0x34],
            " /Subtype /Image /Width 2 /Height 1 /BitsPerComponent 8 \
              /ColorSpace /DeviceGray",
        );
        assert!(explicit_of(&doc).is_none());
    }
}
