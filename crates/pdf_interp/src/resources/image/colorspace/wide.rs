//! The conversion of samples at 16 bits.

use color::Sample;

use super::decode_array::decode_wide;
use super::decoded::Decoded;
use crate::WorkingSpace;
use crate::resources::colorspace::ColorSpace;
use crate::resources::image::ImageColorSpace;

/// Convert `samples`, 16-bit samples of `n` components, through
/// `decoded`'s space, profile and `/Decode` at 16 bits: each component's
/// value is `dmin + s · (dmax − dmin) / 65535`, and the colour that of
/// [`ColorSpace::convert`] at 16 bits.  Device gray and RGB map their
/// components alone; an ICC transform converts the whole image at once.
/// `None` when the converted samples cannot be allocated.
pub(super) fn convert16(
    decoded: &Decoded<'_>,
    n: usize,
    samples: Vec<u16>,
) -> Option<(ImageColorSpace, Vec<u16>)> {
    let identity = decoded.identity_decode();
    let device = decoded.profile.is_none();
    let wide = decoded.target.space == WorkingSpace::Wide;
    match (decoded.space, n) {
        (ColorSpace::DeviceGray | ColorSpace::DeviceRgb, 1 | 3) if device => {
            let mut samples = samples;
            if !identity {
                for px in samples.chunks_exact_mut(n) {
                    for (s, &pair) in px.iter_mut().zip(decoded.decode) {
                        *s = u16::from_unit(decode_wide(*s, pair));
                    }
                }
            }
            let cs = if n == 1 {
                ImageColorSpace::Gray
            } else {
                if wide {
                    crate::working_space::from_srgb16_in_place(&mut samples);
                }
                ImageColorSpace::Rgb
            };
            return Some((cs, samples));
        }
        _ => {}
    }
    // The components of a pixel are held on the stack: the spaces that
    // convert at 16 bits have at most an ICC profile's fifteen.
    if n == 0 || n > MAX_WIDE_COMPONENTS {
        return None;
    }
    let mut out: Vec<u16> = Vec::new();
    out.try_reserve_exact(samples.len() / n * 3).ok()?;
    let icc = decoded.icc_conversion();
    let to_inputs = |px: &[u16], comps: &mut [f64; MAX_WIDE_COMPONENTS]| {
        for ((c, &s), &pair) in comps.iter_mut().zip(px).zip(decoded.decode) {
            *c = decode_wide(s, pair);
        }
    };
    let mut comps = [0.0; MAX_WIDE_COMPONENTS];
    // A transform converts every pixel at once.
    if let Some(conv) = icc.as_deref() {
        let mut inputs = Vec::new();
        inputs.try_reserve_exact(samples.len()).ok()?;
        for px in samples.chunks_exact(n) {
            to_inputs(px, &mut comps);
            inputs.extend(comps[..n].iter().map(|&v| u16::from_unit(v)));
        }
        out.resize(samples.len() / n * 3, 0);
        if conv.transform_pixels16(&inputs, &mut out).is_some() {
            return Some((ImageColorSpace::Rgb, out));
        }
        out.clear();
    }
    // Each pixel converts alone, a run of equal pixels once; device CMYK
    // through the process-ink model directly.
    let cmyk = matches!(decoded.space, ColorSpace::DeviceCmyk) && n == 4 && icc.is_none();
    let mut last: Option<([u16; MAX_WIDE_COMPONENTS], [u16; 3])> = None;
    for px in samples.chunks_exact(n) {
        let rgb = match &last {
            Some((seen, rgb)) if seen[..n] == *px => *rgb,
            _ => {
                to_inputs(px, &mut comps);
                let rgb = if cmyk {
                    let srgb =
                        color::cmyk::cmyk_f64_to_rgb_at(comps[0], comps[1], comps[2], comps[3]);
                    crate::working_space::from_srgb::<u16, u16>(decoded.target.space, srgb)
                } else {
                    let comps = &comps[..n];
                    icc.as_deref()
                        .and_then(|conv| conv.convert::<u16>(decoded.doc, comps))
                        .unwrap_or_else(|| {
                            decoded
                                .space
                                .convert::<u16>(decoded.doc, decoded.target, comps)
                        })
                };
                let mut seen = [0; MAX_WIDE_COMPONENTS];
                seen[..n].copy_from_slice(px);
                last = Some((seen, rgb));
                rgb
            }
        };
        out.extend_from_slice(&rgb);
    }
    Some((ImageColorSpace::Rgb, out))
}

/// `bytes` widened exactly to 16-bit samples; `None` when they cannot be
/// allocated.
fn widen(bytes: &[u8]) -> Option<Vec<u16>> {
    let mut wide = Vec::new();
    wide.try_reserve_exact(bytes.len()).ok()?;
    wide.extend(bytes.iter().map(|&b| u16::from_u8(b)));
    Some(wide)
}

/// The most components a pixel converted at 16 bits has: those of an ICC
/// profile's colour space.
const MAX_WIDE_COMPONENTS: usize = 15;

impl Decoded<'_> {
    /// [`convert16`] of the 8-bit samples `bytes` of `n` components
    /// widened exactly.  CMYK samples under the `/Decode` `[0 1]` of every
    /// component through a tabulated ICC conversion convert on the device
    /// when it takes them ([`cmyk_clut_to_rgb`]): that `/Decode` maps each
    /// widened sample `257 · v` to itself, so the 16-bit transform reads
    /// the cells of the 8-bit sample `v`, which the device grid holds.
    /// `None` as [`convert16`].
    ///
    /// [`cmyk_clut_to_rgb`]: crate::resources::image::icc_device::cmyk_clut_to_rgb
    pub(super) fn convert_bytes16(
        &self,
        n: usize,
        bytes: &[u8],
        #[cfg(feature = "gpu-icc")] icc: &crate::resources::image::IccHandles<'_>,
    ) -> Option<(ImageColorSpace, Vec<u16>)> {
        #[cfg(feature = "gpu-icc")]
        if n == 4
            && self.decode.iter().all(|&pair| pair == (0.0, 1.0))
            && let Some(conv) = self.icc_conversion()
            && let Some(rgb) =
                crate::resources::image::icc_device::cmyk_clut_to_rgb::<u16>(icc, bytes, &conv)
        {
            return Some((ImageColorSpace::Rgb, rgb));
        }
        convert16(self, n, widen(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use pdf::{Dictionary, Document, Object};

    use super::*;
    use crate::resources::image::colorspace::Layout;
    use crate::resources::image::colorspace::test_support::{
        convert, convert16_of, decode_arr, gamma_1_8_doc, icc_cs, name_obj,
    };
    use crate::test_helpers::{empty_doc, make_doc_with_stream};

    /// The 16-bit conversion of 8-bit samples widened, narrowed to bytes,
    /// is the 8-bit conversion of those samples to within `bound`.
    fn assert_follows_8_bits(doc: &Document, cs: &Object, layout: Layout, bytes: &[u8], bound: u8) {
        let wide: Vec<u16> = bytes.iter().map(|&v| u16::from_u8(v)).collect();
        let at16 = convert16_of(doc, cs.clone(), &[], layout, &wide);
        let at8 = convert(doc, cs.clone(), &[], layout, bytes);
        assert_eq!(at16.color_space, at8.color_space);
        let (w, n) = (at16.data.as_cpu16().unwrap(), at8.data.as_cpu().unwrap());
        assert_eq!(w.len(), n.len());
        for (a, b) in w.iter().zip(n) {
            assert!(a.to_u8().abs_diff(*b) <= bound, "{w:?} vs {n:?}");
        }
    }

    /// An inverted `/Decode` maps each 16-bit sample to its complement at
    /// 16 bits.
    #[test]
    fn a_16_bit_inverted_decode_is_the_complement() {
        let doc = empty_doc();
        let img = convert16_of(
            &doc,
            name_obj(b"DeviceGray"),
            &[("Decode", decode_arr(&[1, 0]))],
            Layout::Color(1),
            &[0, 1000, 65_535],
        );
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu16().unwrap(), &[65_535, 64_535, 0]);
    }

    /// 16-bit device CMYK converts through the process-ink model at 16
    /// bits, as a colour of the space does, and follows the 8-bit
    /// conversion within a level.
    #[test]
    fn a_16_bit_cmyk_image_converts_through_the_model_at_16_bits() {
        let doc = empty_doc();
        let samples = [0, 30_000, 65_535, 1_000, 40_000, 40_000, 2_000, 0];
        let img = convert16_of(
            &doc,
            name_obj(b"DeviceCMYK"),
            &[],
            Layout::Color(4),
            &samples,
        );
        let want: Vec<u16> = samples
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| {
                ColorSpace::DeviceCmyk.convert::<u16>(
                    &doc,
                    crate::ColourTarget::SRGB,
                    &px.map(u16::to_unit),
                )
            })
            .collect();
        assert_eq!(img.data.as_cpu16().unwrap(), want.as_slice());
        let bytes = [0, 120, 255, 4, 156, 156, 8, 0, 17, 17, 17, 17];
        assert_follows_8_bits(&doc, &name_obj(b"DeviceCMYK"), Layout::Color(4), &bytes, 1);
    }

    /// A 16-bit Lab image converts at 16 bits and follows the 8-bit
    /// conversion within a level.
    #[test]
    fn a_16_bit_lab_image_follows_its_8_bit_conversion() {
        let doc = empty_doc();
        let lab = Object::Array(vec![
            name_obj(b"Lab"),
            Object::Dictionary({
                let mut d = Dictionary::new();
                d.set(
                    "WhitePoint",
                    Object::Array(vec![
                        Object::Real(0.9505),
                        Object::Integer(1),
                        Object::Real(1.089),
                    ]),
                );
                d
            }),
        ]);
        let bytes = [128, 128, 128, 200, 90, 160, 40, 170, 60];
        assert_follows_8_bits(&doc, &lab, Layout::Color(3), &bytes, 1);
    }

    /// A profile compiled to shaper tables has no 16-bit transform: its
    /// 16-bit samples convert one pixel at a time through the tables, and
    /// follow the 8-bit conversion within a level.
    #[test]
    fn a_16_bit_shaper_profile_converts_each_pixel() {
        let doc = gamma_1_8_doc(3);
        let bytes = [0, 128, 255, 150, 118, 80];
        assert_follows_8_bits(&doc, &icc_cs(), Layout::Color(3), &bytes, 1);
    }

    /// A profile compiled to a transform converts the 16-bit samples
    /// through its 16-bit transform, which follows the 8-bit one within
    /// two levels.
    #[test]
    fn a_16_bit_transform_profile_converts_the_image_at_once() {
        let profile = moxcms::ColorProfile::new_lab().encode().unwrap();
        let doc = make_doc_with_stream(&profile, " /N 3");
        assert!(matches!(
            crate::resources::colorspace::icc::IccConversion::load(
                &doc,
                (2, 0),
                3,
                crate::ColourTarget::SRGB
            )
            .as_deref(),
            Some(crate::resources::colorspace::icc::IccConversion::Transform { .. })
        ));
        let bytes = [128, 128, 128, 200, 90, 160, 40, 170, 60];
        assert_follows_8_bits(&doc, &icc_cs(), Layout::Color(3), &bytes, 2);
    }
}
