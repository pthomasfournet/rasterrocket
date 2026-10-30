//! PDF colour space resolution for image decoding.
//!
//! Maps the `ColorSpace` entry of an image stream dictionary to one of the
//! three device spaces the blit pipeline understands (`Gray`, `Rgb`, `Cmyk`),
//! extracts palette data for `Indexed` spaces, and builds the per-sample
//! lookup that turns spot-colour tints into device colour.
//!
//! The `Cmyk` variant is always converted to `Rgb` before the descriptor
//! leaves this module; callers never see raw CMYK pixels.
//!
//! A `Separation` (or single-colorant `DeviceN`) image decodes to one tint
//! byte per pixel with the sample layout of `Gray`, but the byte is a tint —
//! `0` is no colorant, `255` is full colorant (PDF §8.6.6.4) — not a gray
//! level, whose polarity is the reverse.  [`tint_lut`] evaluates the space's
//! tint transform for every tint value and [`apply_tint_lut`] rewrites the
//! decoded bytes through it, so a codec never has to know about spot colour.
//!
//! An `ICCBased` N=1 or N=3 image decodes to device samples in the same way;
//! [`sample_rewrite`] selects the profile's matrix/TRC conversion to sRGB,
//! and [`apply_sample_rewrite`] runs it, or the tint lookup, on the final
//! samples.

use pdf::{Dictionary, Document, Object};

use std::sync::Arc;

use super::{ImageColorSpace, ImageData, ImageDescriptor};
use crate::resources::colorspace::{ColorSpace, needs_sample_conversion, resolve, samples_to_rgb};
use crate::resources::dict_ext::DictExt as _;
use crate::resources::icc_shaper::{
    GrayShaper, ProfileConversion, RgbShaper, conversion_for_stream,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Internal resolved colour space — what the decode path will actually produce.
///
/// `Cmyk` is a transient state: any function that returns an `ImageDescriptor`
/// converts CMYK bytes to RGB before the descriptor leaves the image module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResolvedCs {
    Gray,
    Rgb,
    /// Raw CMYK (4 bytes/pixel, 255 = full ink).  Converted to RGB before returning.
    Cmyk,
}

impl ResolvedCs {
    pub(super) const fn components(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }

    pub(super) const fn to_image_cs(self) -> ImageColorSpace {
        match self {
            Self::Gray => ImageColorSpace::Gray,
            Self::Rgb | Self::Cmyk => ImageColorSpace::Rgb,
        }
    }
}

// ── Colour space resolution ───────────────────────────────────────────────────

/// Maximum recursion depth for `Indexed` base-space chains.
///
/// PDF spec §8.6.6.3 forbids nested `Indexed` spaces, but adversarial PDFs are
/// not required to be spec-compliant.  We stop after this many levels to prevent
/// stack overflow.
const MAX_CS_DEPTH: u8 = 8;

/// Resolve the `ColorSpace` entry of an image stream dictionary to one of the
/// three device spaces the blit path understands.
///
/// Handles:
/// - `DeviceGray` / `CalGray`        → Gray
/// - `DeviceRGB`  / `CalRGB`         → Rgb
/// - `DeviceCMYK`                    → Cmyk (caller converts to RGB)
/// - `ICCBased`   → inspect `N` in the ICC stream dict (1→Gray, 3→Rgb, 4→Cmyk)
/// - `Indexed`    → resolve the base space; Indexed expansion handled separately
/// - `Separation` / `DeviceN`        → Gray sample layout: one tint byte per
///   pixel.  The tint → colour mapping is applied afterwards by
///   [`apply_tint_lut`].  Raw samples and `Indexed` palettes in a
///   multi-colorant `DeviceN` or Lab space are converted per sample to RGB
///   before this resolution applies (see `needs_sample_conversion`).
/// - unknown / absent                → Gray (safe fallback)
pub(super) fn resolve_cs(doc: &Document, cs_obj: &Object) -> ResolvedCs {
    resolve_cs_depth(doc, cs_obj, 0)
}

fn resolve_cs_depth(doc: &Document, cs_obj: &Object, depth: u8) -> ResolvedCs {
    match cs_obj {
        Object::Name(n) => device_cs_name(n),
        Object::Reference(id) => {
            // Indirect colour space reference — dereference and resolve.
            doc.get_object(*id).ok().map_or(ResolvedCs::Gray, |obj| {
                resolve_cs_depth(doc, obj.as_ref(), depth)
            })
        }
        Object::Array(arr) => {
            let name = arr.first().and_then(Object::as_name).unwrap_or(b"");
            match name {
                b"DeviceRGB" | b"CalRGB" => ResolvedCs::Rgb,
                b"DeviceCMYK" => ResolvedCs::Cmyk,
                b"ICCBased" => {
                    // Second element is a reference to the ICC stream.
                    let stream_dict = arr
                        .get(1)
                        .and_then(|o| super::super::resolve_stream_dict(doc, o));
                    icc_based_cs(stream_dict.as_ref())
                }
                b"Indexed" => {
                    // [/Indexed base hival lookup] — base is element 1.
                    // Guard against adversarial nested Indexed chains.
                    if depth >= MAX_CS_DEPTH {
                        log::debug!(
                            "image: Indexed colour space nested too deeply (depth {depth}) — treating as Gray"
                        );
                        return ResolvedCs::Gray;
                    }
                    arr.get(1)
                        .map_or(ResolvedCs::Gray, |o| resolve_cs_depth(doc, o, depth + 1))
                }
                // DeviceGray, CalGray, Separation, DeviceN, and unknown → Gray.
                _ => {
                    if !matches!(
                        name,
                        b"DeviceGray" | b"CalGray" | b"Separation" | b"DeviceN"
                    ) {
                        log::debug!(
                            "image: unknown array colour space {:?} — treating as Gray",
                            String::from_utf8_lossy(name)
                        );
                    }
                    ResolvedCs::Gray
                }
            }
        }
        _ => ResolvedCs::Gray,
    }
}

/// Map a `ColorSpace` device name to a [`ResolvedCs`].
pub(super) const fn device_cs_name(name: &[u8]) -> ResolvedCs {
    match name {
        b"DeviceRGB" | b"CalRGB" => ResolvedCs::Rgb,
        b"DeviceCMYK" => ResolvedCs::Cmyk,
        _ => ResolvedCs::Gray,
    }
}

/// Inspect the `N` key in an `ICCBased` stream dict to determine the colour space.
fn icc_based_cs(stream_dict: Option<&Dictionary>) -> ResolvedCs {
    match stream_dict.and_then(|d| d.get_i64(b"N")) {
        Some(1) => ResolvedCs::Gray,
        Some(3) => ResolvedCs::Rgb,
        Some(4) => ResolvedCs::Cmyk,
        Some(n) => {
            log::debug!("image: ICCBased with N={n}, treating as Gray");
            ResolvedCs::Gray
        }
        None => {
            log::debug!("image: ICCBased missing N, treating as Gray");
            ResolvedCs::Gray
        }
    }
}

/// Extract the raw ICC profile bytes from an `ICCBased` colour space object.
///
/// Returns `None` if the object is not `[/ICCBased <ref>]`, the stream cannot
/// be dereferenced, or decompression fails.  Only compiled under `gpu-icc`.
#[cfg(feature = "gpu-icc")]
pub(super) fn extract_icc_bytes(doc: &Document, cs_obj: &Object) -> Option<Vec<u8>> {
    let arr = cs_obj.as_array()?;
    if arr.first().and_then(Object::as_name) != Some(b"ICCBased") {
        return None;
    }
    let Object::Reference(id) = arr.get(1)? else {
        return None;
    };
    let obj_arc = doc.get_object(*id).ok()?;
    let stream = obj_arc.as_ref().as_stream()?;
    stream
        .decompressed_content()
        .map_err(|e| log::debug!("image: ICCBased stream decompression failed: {e}"))
        .ok()
}

// ── Spot-colour tint lookup ───────────────────────────────────────────────────

/// Colour of every 8-bit tint value of a single-colorant image colour space.
///
/// Entry `i` is tint `i / 255` passed through the space's tint transform and
/// converted from the alternate space to sRGB.  `Gray` is chosen when every
/// entry is neutral so the image keeps its one-byte layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TintLut {
    Gray(Box<[u8; 256]>),
    Rgb(Box<[[u8; 3]; 256]>),
}

/// Build the tint lookup for an image whose `ColorSpace` is a `Separation`,
/// a single-colorant `DeviceN`, or an `Indexed` space over either.
///
/// Returns `None` for every other colour space.  An `Indexed` space
/// qualifies because palette expansion yields one base-space tint byte per
/// pixel, which is exactly what the lookup consumes.
fn tint_lut(doc: &Document, cs_obj: &Object) -> Option<TintLut> {
    let cs = resolve(doc, cs_obj);
    let tint_cs = match &cs {
        ColorSpace::Indexed { base, .. } => base.as_ref(),
        other => other,
    };
    if !matches!(
        tint_cs,
        ColorSpace::Separation { .. } | ColorSpace::DeviceN { ncomps: 1, .. }
    ) {
        return None;
    }

    let mut rgb = Box::new([[0u8; 3]; 256]);
    for (sample, entry) in (0u8..=255).zip(rgb.iter_mut()) {
        *entry = tint_cs.convert_to_rgb(doc, &[f64::from(sample) / 255.0]);
    }

    if rgb.iter().all(|[r, g, b]| r == g && g == b) {
        let mut gray = Box::new([0u8; 256]);
        for (out, [v, _, _]) in gray.iter_mut().zip(rgb.iter()) {
            *out = *v;
        }
        Some(TintLut::Gray(gray))
    } else {
        Some(TintLut::Rgb(rgb))
    }
}

/// Rewrite a decoded one-tint-byte-per-pixel image through `lut`.
///
/// Only a host-resident `Gray`-layout descriptor carries tint bytes; any
/// other layout is a producer error (e.g. a three-component JPEG declared as
/// `Separation`) or device-resident data, and is left untouched.
fn apply_tint_lut(img: &mut ImageDescriptor, lut: &TintLut) {
    if img.color_space != ImageColorSpace::Gray {
        log::debug!(
            "image: {:?} sample layout is not one tint per pixel — leaving samples as decoded",
            img.color_space
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
    let bytes = match &mut img.data {
        ImageData::Cpu(bytes) => bytes,
        #[cfg(feature = "cache")]
        ImageData::Gpu(_) => {
            log::debug!(
                "image: tint lookup needs host-resident samples — leaving samples as decoded"
            );
            return;
        }
    };
    match lut {
        TintLut::Gray(table) => {
            for b in bytes.iter_mut() {
                *b = table[usize::from(*b)];
            }
        }
        TintLut::Rgb(table) => {
            let mut rgb = Vec::with_capacity(bytes.len() * 3);
            for &b in bytes.iter() {
                rgb.extend_from_slice(&table[usize::from(b)]);
            }
            *bytes = rgb;
            img.color_space = ImageColorSpace::Rgb;
        }
    }
}

// ── Post-decode sample rewrite ────────────────────────────────────────────────

/// A colour mapping applied to an image's samples once they are final.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SampleRewrite {
    /// Spot-colour tint bytes to colour; see [`tint_lut`].
    Tint(TintLut),
    /// Gray samples through an `ICCBased` N=1 matrix/TRC profile.
    IccGray(Arc<GrayShaper>),
    /// RGB samples through an `ICCBased` N=3 matrix/TRC profile.
    IccRgb(Arc<RgbShaper>),
}

/// The rewrite an image in colour space `cs_obj` needs, if any.
///
/// An `Indexed` space over `ICCBased` qualifies: palette expansion yields
/// base-space samples.  A profile that is sRGB at 8-bit precision, or is
/// not a matrix/TRC profile, needs none and the samples pass through.
pub(super) fn sample_rewrite(doc: &Document, cs_obj: &Object) -> Option<SampleRewrite> {
    if let Some(lut) = tint_lut(doc, cs_obj) {
        return Some(SampleRewrite::Tint(lut));
    }
    let cs = resolve(doc, cs_obj);
    let icc_cs = match &cs {
        ColorSpace::Indexed { base, .. } => base.as_ref(),
        other => other,
    };
    let ColorSpace::IccBased {
        n,
        stream_id: Some(id),
    } = icc_cs
    else {
        return None;
    };
    match (n, conversion_for_stream(doc, *id)) {
        (1, ProfileConversion::Gray(Some(shaper))) => Some(SampleRewrite::IccGray(shaper)),
        (3, ProfileConversion::Rgb(Some(shaper))) => Some(SampleRewrite::IccRgb(shaper)),
        _ => None,
    }
}

/// Apply `rewrite` to the host-resident samples of `img`.
///
/// A sample layout the rewrite does not expect (a producer declaring a
/// space whose channel count the codec output contradicts) or
/// device-resident data is left untouched.
pub(super) fn apply_sample_rewrite(img: &mut ImageDescriptor, rewrite: &SampleRewrite) {
    let expected = match rewrite {
        SampleRewrite::Tint(lut) => return apply_tint_lut(img, lut),
        SampleRewrite::IccGray(_) => ImageColorSpace::Gray,
        SampleRewrite::IccRgb(_) => ImageColorSpace::Rgb,
    };
    if img.color_space != expected {
        log::debug!(
            "image: {:?} sample layout does not match the ICC profile — leaving samples as decoded",
            img.color_space
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
    let bytes = match &mut img.data {
        ImageData::Cpu(bytes) => bytes,
        #[cfg(feature = "cache")]
        ImageData::Gpu(_) => {
            log::debug!(
                "image: ICC conversion needs host-resident samples — leaving samples as decoded"
            );
            return;
        }
    };
    match rewrite {
        SampleRewrite::IccGray(shaper) => shaper.convert_in_place(bytes),
        SampleRewrite::IccRgb(shaper) => shaper.convert_in_place(bytes),
        SampleRewrite::Tint(_) => {}
    }
}

// ── Indexed palette ───────────────────────────────────────────────────────────

/// The `[/Indexed base hival lookup]` array `cs_obj` holds, if it is one.
pub(super) fn indexed_array(cs_obj: &Object) -> Option<&[Object]> {
    match cs_obj {
        Object::Array(arr) if arr.first().and_then(Object::as_name) == Some(b"Indexed") => {
            Some(arr)
        }
        _ => None,
    }
}

/// Decode an `Indexed` colour space lookup table into a flat RGB/Gray palette.
///
/// `cs_arr` is the full `[/Indexed base hival lookup]` array.  Returns
/// `(palette, base_cs)` where `palette[i * stride .. i * stride + stride]`
/// is the colour for index `i` and `stride` is `base_cs.components()`.
///
/// `Cmyk` base spaces are converted to `Rgb` before return so callers always
/// receive either 1-byte (Gray) or 3-byte (Rgb) palette entries.
///
/// Returns `None` if the array is malformed or the lookup stream cannot be read.
pub(super) fn indexed_palette(doc: &Document, cs_arr: &[Object]) -> Option<(Vec<u8>, ResolvedCs)> {
    // [/Indexed base hival lookup]
    if cs_arr.len() < 4 {
        return None;
    }
    let base = resolve_cs(doc, &cs_arr[1]);

    // hival is clamped to [0, 255] before converting — cast is lossless.
    #[expect(
        clippy::cast_sign_loss,
        reason = "clamped to [0, 255] — never negative"
    )]
    let hival = cs_arr[2].as_i64()?.clamp(0, 255) as usize;
    let n_entries = hival + 1;

    // Lookup is either a string (literal bytes) or a reference to a stream.
    let lookup_bytes: Vec<u8> = match &cs_arr[3] {
        Object::String(bytes, _) => bytes.clone(),
        Object::Reference(id) => {
            let obj_arc = doc.get_object(*id).ok()?;
            match obj_arc.as_ref() {
                Object::Stream(s) => s.decompressed_content().ok()?,
                Object::String(bytes, _) => bytes.clone(),
                _ => return None,
            }
        }
        _ => return None,
    };

    // A base the device layouts cannot carry converts every entry to RGB.
    let base_space = resolve(doc, &cs_arr[1]);
    if needs_sample_conversion(&base_space) {
        let stride = base_space.ncomponents();
        let entries = lookup_bytes.get(..n_entries.checked_mul(stride)?)?;
        let palette = entries
            .chunks_exact(stride)
            .flat_map(|entry| samples_to_rgb(doc, &base_space, entry))
            .collect();
        return Some((palette, ResolvedCs::Rgb));
    }

    let base_stride = base.components();
    let needed = n_entries.checked_mul(base_stride)?;
    if lookup_bytes.len() < needed {
        log::debug!(
            "image: Indexed palette too short ({} bytes, need {needed})",
            lookup_bytes.len()
        );
        return None;
    }

    // Build the output palette, converting CMYK entries to RGB inline.
    let palette: Vec<u8> = if base == ResolvedCs::Cmyk {
        color::cmyk::cmyk_to_rgb_vec(&lookup_bytes[..needed])
    } else {
        lookup_bytes[..needed].to_vec()
    };

    // CMYK base is treated as Rgb after palette construction above.
    let out_cs = if base == ResolvedCs::Gray {
        ResolvedCs::Gray
    } else {
        ResolvedCs::Rgb
    };
    Some((palette, out_cs))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_cs_name_gray() {
        assert_eq!(device_cs_name(b"DeviceGray"), ResolvedCs::Gray);
        assert_eq!(device_cs_name(b"CalGray"), ResolvedCs::Gray);
    }

    #[test]
    fn device_cs_name_rgb() {
        assert_eq!(device_cs_name(b"DeviceRGB"), ResolvedCs::Rgb);
        assert_eq!(device_cs_name(b"CalRGB"), ResolvedCs::Rgb);
    }

    #[test]
    fn device_cs_name_cmyk() {
        assert_eq!(device_cs_name(b"DeviceCMYK"), ResolvedCs::Cmyk);
    }

    #[test]
    fn device_cs_name_unknown_is_gray() {
        assert_eq!(device_cs_name(b"Unknown"), ResolvedCs::Gray);
    }

    #[test]
    fn resolve_cs_indirect_reference_falls_back_to_gray() {
        // An unresolvable Reference should not panic — it falls back to Gray.
        // Reference (999, 0) is absent from the empty doc's xref so resolution
        // fails and we get the fallback.
        let doc = crate::test_helpers::empty_doc();
        let obj = Object::Reference((999, 0));
        assert_eq!(resolve_cs(&doc, &obj), ResolvedCs::Gray);
    }

    // ── Tint lookup ───────────────────────────────────────────────────────────

    use super::super::ImageFilter;
    use crate::test_helpers::make_doc_with_objects;

    fn name_obj(n: &[u8]) -> Object {
        Object::Name(n.to_vec())
    }

    /// `[/Separation /Black /DeviceCMYK 4 0 R]` where object 4 is the Type 4
    /// tint transform `t ↦ (0, 0, 0, t)` as a real scanner producer writes it.
    fn separation_black_doc() -> (Document, Object) {
        const PROGRAM: &str = "{dup 0 mul exch dup 0 mul exch dup 0 mul exch 1 mul }";
        let body = format!(
            "<< /FunctionType 4 /Domain [0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
             stream\n{PROGRAM}\nendstream",
            PROGRAM.len()
        );
        let doc = make_doc_with_objects(&[(4, &body)]);
        let cs = Object::Array(vec![
            name_obj(b"Separation"),
            name_obj(b"Black"),
            name_obj(b"DeviceCMYK"),
            Object::Reference((4, 0)),
        ]);
        (doc, cs)
    }

    #[test]
    fn tint_lut_separation_black_maps_zero_tint_to_white() {
        // PDF §8.6.6.4: tint 0 is no colorant.  Through t ↦ (0,0,0,t) and the
        // CMYK alternate that is paper white; tint 1 is solid K.  This is the
        // reverse of the Gray sample polarity the bytes are decoded in.
        let (doc, cs) = separation_black_doc();
        let Some(TintLut::Gray(lut)) = tint_lut(&doc, &cs) else {
            panic!("a neutral alternate colour must yield a Gray lookup");
        };
        let solid_k = color::cmyk::CORNERS[8][0];
        assert_eq!(lut[0], 255, "tint 0 = no ink = white");
        assert_eq!(lut[255], solid_k, "tint 1 = solid K");
        assert!(
            lut.windows(2).all(|w| w[0] >= w[1]),
            "more ink must never be lighter"
        );
        assert_eq!(
            lut[128],
            color::cmyk::cmyk_to_rgb(0, 0, 0, 128)[0],
            "half tint is the half-K grey"
        );
    }

    #[test]
    fn tint_lut_indexed_over_separation_uses_the_base_tint() {
        let (doc, sep) = separation_black_doc();
        let cs = Object::Array(vec![
            name_obj(b"Indexed"),
            sep,
            Object::Integer(1),
            Object::String(vec![0x00, 0xFF], pdf::StringFormat::Hexadecimal),
        ]);
        let Some(TintLut::Gray(lut)) = tint_lut(&doc, &cs) else {
            panic!("Indexed over Separation carries base-space tints per pixel");
        };
        assert_eq!((lut[0], lut[255]), (255, color::cmyk::CORNERS[8][0]));
    }

    #[test]
    fn tint_lut_coloured_alternate_yields_rgb() {
        // Type 2 tint transform into DeviceRGB: white at tint 0, red at tint 1.
        let doc = make_doc_with_objects(&[(
            4,
            "<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>",
        )]);
        let cs = Object::Array(vec![
            name_obj(b"Separation"),
            name_obj(b"Spot"),
            name_obj(b"DeviceRGB"),
            Object::Reference((4, 0)),
        ]);
        let Some(TintLut::Rgb(lut)) = tint_lut(&doc, &cs) else {
            panic!("a chromatic alternate colour must yield an Rgb lookup");
        };
        assert_eq!(lut[0], [255, 255, 255]);
        assert_eq!(lut[255], [255, 0, 0]);
    }

    #[test]
    fn tint_lut_none_for_non_tint_spaces() {
        let doc = crate::test_helpers::empty_doc();
        assert_eq!(tint_lut(&doc, &name_obj(b"DeviceGray")), None);
        assert_eq!(tint_lut(&doc, &name_obj(b"DeviceRGB")), None);
        assert_eq!(
            tint_lut(
                &doc,
                &Object::Array(vec![name_obj(b"ICCBased"), Object::Reference((9, 0))])
            ),
            None
        );
        // Multi-colorant DeviceN samples are converted per tuple, not
        // through a one-tint lookup.
        let dn = Object::Array(vec![
            name_obj(b"DeviceN"),
            Object::Array(vec![name_obj(b"Cyan"), name_obj(b"Black")]),
            name_obj(b"DeviceCMYK"),
            Object::Reference((9, 0)),
        ]);
        assert_eq!(tint_lut(&doc, &dn), None);
    }

    #[test]
    fn tint_lut_single_colorant_devicen_is_treated_like_separation() {
        let (doc, _) = separation_black_doc();
        let dn = Object::Array(vec![
            name_obj(b"DeviceN"),
            Object::Array(vec![name_obj(b"Black")]),
            name_obj(b"DeviceCMYK"),
            Object::Reference((4, 0)),
        ]);
        let Some(TintLut::Gray(lut)) = tint_lut(&doc, &dn) else {
            panic!("one-colorant DeviceN is a Separation with another name");
        };
        assert_eq!((lut[0], lut[255]), (255, color::cmyk::CORNERS[8][0]));
    }

    fn gray_desc(bytes: &[u8]) -> ImageDescriptor {
        ImageDescriptor {
            width: u32::try_from(bytes.len()).unwrap(),
            height: 1,
            color_space: ImageColorSpace::Gray,
            data: ImageData::Cpu(bytes.to_vec()),
            smask: None,
            filter: ImageFilter::Raw,
        }
    }

    #[test]
    fn apply_tint_lut_gray_rewrites_in_place() {
        let mut lut = Box::new([0u8; 256]);
        for (i, v) in lut.iter_mut().enumerate() {
            *v = 255 - u8::try_from(i).unwrap();
        }
        let mut img = gray_desc(&[0, 128, 255]);
        apply_tint_lut(&mut img, &TintLut::Gray(lut));
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 127, 0]);
    }

    #[test]
    fn apply_tint_lut_rgb_widens_to_three_bytes_per_pixel() {
        let mut lut = Box::new([[0u8; 3]; 256]);
        lut[0] = [255, 255, 255];
        lut[255] = [255, 0, 0];
        let mut img = gray_desc(&[0, 255]);
        apply_tint_lut(&mut img, &TintLut::Rgb(lut));
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 255, 255, 255, 0, 0]);
    }

    #[test]
    fn apply_tint_lut_leaves_non_gray_layouts_alone() {
        let lut = TintLut::Gray(Box::new([7u8; 256]));
        let mut img = gray_desc(&[1, 2, 3]);
        img.color_space = ImageColorSpace::Rgb;
        apply_tint_lut(&mut img, &lut);
        assert_eq!(img.data.as_cpu().unwrap(), &[1, 2, 3]);
        let mut mask = gray_desc(&[0]);
        mask.color_space = ImageColorSpace::Mask;
        apply_tint_lut(&mut mask, &lut);
        assert_eq!(mask.color_space, ImageColorSpace::Mask);
        assert_eq!(mask.data.as_cpu().unwrap(), &[0]);
    }

    // ── ICC sample rewrite ────────────────────────────────────────────────────

    /// A document whose object 2 is an ICC profile stream with `/N n`,
    /// gamma 1.8 (`u8Fixed8` 461) on sRGB primaries.
    fn gamma_1_8_doc(n: u8) -> Document {
        let gamma = moxcms::ToneReprCurve::Lut(vec![461]);
        let profile = if n == 1 {
            let mut p = moxcms::ColorProfile::new_gray_with_gamma(1.0);
            p.gray_trc = Some(gamma);
            p
        } else {
            let mut p = moxcms::ColorProfile::new_srgb();
            p.cicp = None;
            p.red_trc = Some(gamma.clone());
            p.green_trc = Some(gamma.clone());
            p.blue_trc = Some(gamma);
            p
        };
        crate::test_helpers::make_doc_with_stream(&profile.encode().unwrap(), &format!(" /N {n}"))
    }

    fn icc_cs() -> Object {
        Object::Array(vec![name_obj(b"ICCBased"), Object::Reference((2, 0))])
    }

    #[test]
    fn icc_rgb_image_samples_go_through_the_profile() {
        let doc = gamma_1_8_doc(3);
        let rewrite = sample_rewrite(&doc, &icc_cs()).expect("gamma 1.8 is not sRGB");
        let mut img = gray_desc(&[0, 128, 255, 150, 118, 80]);
        img.width = 2;
        img.color_space = ImageColorSpace::Rgb;
        apply_sample_rewrite(&mut img, &rewrite);
        // encode((v/255)^1.8): (0, 128, 255) → (0, 146, 255);
        // (150, 118, 80) → (167, 137, 99).
        assert_eq!(img.data.as_cpu().unwrap(), &[0, 146, 255, 167, 137, 99]);
    }

    #[test]
    fn icc_gray_image_samples_go_through_the_profile() {
        let doc = gamma_1_8_doc(1);
        let rewrite = sample_rewrite(&doc, &icc_cs()).expect("gamma 1.8 is not sRGB");
        let mut img = gray_desc(&[0, 128, 255]);
        apply_sample_rewrite(&mut img, &rewrite);
        assert_eq!(img.data.as_cpu().unwrap(), &[0, 146, 255]);
    }

    #[test]
    fn indexed_over_icc_uses_the_base_profile() {
        let doc = gamma_1_8_doc(3);
        let cs = Object::Array(vec![
            name_obj(b"Indexed"),
            icc_cs(),
            Object::Integer(0),
            Object::String(vec![128, 128, 128], pdf::StringFormat::Hexadecimal),
        ]);
        assert!(matches!(
            sample_rewrite(&doc, &cs),
            Some(SampleRewrite::IccRgb(_))
        ));
    }

    #[test]
    fn srgb_and_mismatched_profiles_need_no_rewrite() {
        let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
        let doc = crate::test_helpers::make_doc_with_stream(&srgb, " /N 3");
        assert_eq!(
            sample_rewrite(&doc, &icc_cs()),
            None,
            "sRGB is the identity"
        );
        // A gray profile declared N=3 is not applied to RGB samples.
        let doc = crate::test_helpers::make_doc_with_stream(
            &{
                let mut p = moxcms::ColorProfile::new_gray_with_gamma(1.0);
                p.gray_trc = Some(moxcms::ToneReprCurve::Lut(vec![461]));
                p.encode().unwrap()
            },
            " /N 3",
        );
        assert_eq!(sample_rewrite(&doc, &icc_cs()), None);
    }

    #[test]
    fn icc_rewrite_leaves_a_contradicting_layout_alone() {
        let doc = gamma_1_8_doc(3);
        let rewrite = sample_rewrite(&doc, &icc_cs()).unwrap();
        let mut img = gray_desc(&[128, 128]);
        apply_sample_rewrite(&mut img, &rewrite);
        assert_eq!(img.data.as_cpu().unwrap(), &[128, 128]);
    }
}
