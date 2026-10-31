//! Colour-value conversion through an ICC profile stream (PDF §8.6.5.5).
//!
//! A profile stream is compiled once per document into an
//! [`IccConversion`]: a gray or RGB matrix/TRC profile into the shared
//! shaper tables of `super::super::icc_shaper`, any other 3- or 4-channel
//! profile into a `moxcms` 8-bit transform to sRGB.  A profile that admits
//! neither compiles to its `/Alternate` space when that space takes the
//! same number of components (§8.6.5.5, Table 66).  Converting a colour
//! value is then a table lookup, one transform call, or a conversion in the
//! alternate space.

use std::sync::Arc;
#[cfg(feature = "gpu-icc")]
use std::sync::OnceLock;

use moxcms::{ColorProfile, Layout, Transform8BitExecutor, TransformOptions};
use pdf::{Document, ObjectId};

use super::ColorSpace;
use crate::resources::icc_shaper::{ProfileConversion, fetch_profile_bytes};

#[cfg(feature = "gpu-icc")]
pub(crate) mod clut;

/// The compiled sRGB conversion of one ICC profile stream for colour values
/// of its declared component count.
pub(crate) enum IccConversion {
    /// A gray (`n = 1`) or RGB (`n = 3`) matrix/TRC profile.
    Shaper(ProfileConversion),
    /// Any other profile with `n` of 1, 3 or 4: an 8-bit transform whose
    /// source pixel holds `n` channels.
    Transform {
        n: u8,
        xform: Arc<Transform8BitExecutor>,
        /// The CMYK → sRGB lookup table of a four-channel transform for the
        /// device conversion, baked on first use (see [`Self::cmyk_clut`]).
        #[cfg(feature = "gpu-icc")]
        clut: OnceLock<Option<Arc<[u8]>>>,
    },
    /// A profile with no conversion of its own: the `/Alternate` space of
    /// the stream, a device or CIE-based space of the same component count.
    Alternate(ColorSpace),
}

impl IccConversion {
    /// The conversion of profile stream `id` for `n`-component values,
    /// compiled on first use and shared for the lifetime of `doc`.  `None`
    /// when neither the profile nor the `/Alternate` space converts `n`
    /// components.
    pub(crate) fn load(doc: &Document, id: ObjectId, n: u8) -> Option<Arc<Self>> {
        doc.derived(id, || {
            fetch_profile_bytes(doc, id)
                .and_then(|bytes| Self::compile(&bytes, n))
                .or_else(|| Self::alternate(doc, id, n).map(Self::Alternate))
        })
    }

    /// The `/Alternate` space of profile stream `id` when it is a device,
    /// `CalGray`, `CalRGB` or Lab space of `n` components.  Spaces that
    /// convert through another space (ICC-based, Indexed, Separation,
    /// `DeviceN`) are excluded, so an alternate cannot lead back to this
    /// profile.
    fn alternate(doc: &Document, id: ObjectId, n: u8) -> Option<ColorSpace> {
        let obj = doc.get_object(id).ok()?;
        let alt = obj.as_ref().as_stream()?.dict.get(b"Alternate")?;
        let space = super::resolve(doc, alt);
        let usable = space.ncomponents() == usize::from(n)
            && matches!(
                space,
                ColorSpace::DeviceGray
                    | ColorSpace::DeviceRgb
                    | ColorSpace::DeviceCmyk
                    | ColorSpace::Cal(_)
                    | ColorSpace::Lab(_)
            );
        if !usable {
            log::debug!("colorspace: ICCBased /Alternate {space:?} unusable for /N {n}");
        }
        usable.then_some(space)
    }

    /// Compile profile `bytes` for `n`-component values without consulting
    /// a cache.
    pub(crate) fn compile(bytes: &[u8], n: u8) -> Option<Self> {
        let shaper = ProfileConversion::from_profile_bytes(bytes);
        if matches!(
            (&shaper, n),
            (ProfileConversion::Gray(_), 1) | (ProfileConversion::Rgb(_), 3)
        ) {
            return Some(Self::Shaper(shaper));
        }
        let layout = match n {
            1 => Layout::Gray,
            3 => Layout::Rgb,
            4 => Layout::Rgba,
            _ => return None,
        };
        let src = ColorProfile::new_from_slice(bytes)
            .map_err(|e| log::debug!("colorspace: ICC profile parse failed: {e}"))
            .ok()?;
        let xform = src
            .create_transform_8bit(
                layout,
                &ColorProfile::new_srgb(),
                Layout::Rgb,
                TransformOptions::default(),
            )
            .map_err(|e| log::debug!("colorspace: ICC transform creation failed: {e}"))
            .ok()?;
        Some(Self::Transform {
            n,
            xform,
            #[cfg(feature = "gpu-icc")]
            clut: OnceLock::new(),
        })
    }

    /// The CMYK → sRGB table of a four-channel transform on the
    /// `DEFAULT_GRID_N` lattice, baked once per conversion and shared;
    /// `None` for any other conversion or when the bake fails.
    #[cfg(feature = "gpu-icc")]
    pub(crate) fn cmyk_clut(&self) -> Option<Arc<[u8]>> {
        use clut::{DEFAULT_GRID_N, bake_cmyk_clut_with};

        let Self::Transform { n: 4, xform, clut } = self else {
            return None;
        };
        clut.get_or_init(|| {
            bake_cmyk_clut_with(&**xform, DEFAULT_GRID_N)
                .map_err(|e| {
                    log::warn!(
                        "colorspace: ICC CLUT bake failed, using the profile-free CMYK model: {e}"
                    );
                })
                .ok()
                .map(Arc::from)
        })
        .clone()
    }

    /// Whether this conversion yields the device conversion of the same
    /// component count: a shaper profile equal to sRGB at 8 bits, or a
    /// device `/Alternate` space.
    pub(super) const fn is_device_equivalent(&self) -> bool {
        matches!(
            self,
            Self::Shaper(ProfileConversion::Gray(None) | ProfileConversion::Rgb(None))
                | Self::Alternate(
                    ColorSpace::DeviceGray | ColorSpace::DeviceRgb | ColorSpace::DeviceCmyk
                )
        )
    }

    /// Convert interleaved 8-bit pixels of `n` channels in `src` to the
    /// sRGB bytes of `dst`, which holds three bytes per pixel.  `None` for
    /// a conversion that is not a transform, a length mismatch, or a
    /// transform failure.
    pub(crate) fn transform_pixels(&self, src: &[u8], dst: &mut [u8]) -> Option<()> {
        let Self::Transform { n, xform, .. } = self else {
            return None;
        };
        let n = usize::from(*n);
        if !src.len().is_multiple_of(n) || dst.len() != src.len() / n * 3 {
            return None;
        }
        xform
            .transform(src, dst)
            .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
            .ok()
    }

    /// Convert normalised components to sRGB bytes.  Components are
    /// quantised to 8 bits for the shaper and the transform.  `None` when
    /// `comps` does not hold the compiled component count or the transform
    /// fails.
    pub(crate) fn convert(&self, doc: &Document, comps: &[f64]) -> Option<[u8; 3]> {
        match self {
            Self::Alternate(space) => {
                (comps.len() == space.ncomponents()).then(|| space.convert_to_rgb(doc, comps))
            }
            Self::Shaper(shaper) => shaper.convert_components(comps),
            Self::Transform { n, xform, .. } => {
                if comps.len() != usize::from(*n) {
                    return None;
                }
                let mut src = [0u8; 4];
                for (dst, &c) in src.iter_mut().zip(comps) {
                    *dst = color::convert::gray_to_u8(c);
                }
                let mut rgb = [0u8; 3];
                xform
                    .transform(&src[..comps.len()], &mut rgb)
                    .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
                    .ok()?;
                Some(rgb)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::ColorSpace;
    use crate::test_helpers::make_doc_with_stream;

    fn srgb_doc() -> Document {
        let bytes = ColorProfile::new_srgb()
            .encode()
            .expect("moxcms encodes its own sRGB profile");
        make_doc_with_stream(&bytes, " /N 3")
    }

    /// An sRGB `/DefaultRGB` converts exactly as `DeviceRGB` does, so the
    /// device space stays selected.
    #[test]
    fn srgb_default_space_keeps_the_device_space() {
        let doc = srgb_doc();
        let conv = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        assert!(conv.is_device_equivalent());
        let mut dict = pdf::Dictionary::new();
        dict.set(
            "DefaultRGB",
            pdf::Object::Array(vec![
                pdf::Object::Name(b"ICCBased".to_vec()),
                pdf::Object::Reference((2, 0)),
            ]),
        );
        assert_eq!(
            super::super::default_space(&doc, &dict, &ColorSpace::DeviceRgb),
            None
        );
    }

    #[test]
    fn profile_stream_is_compiled_once_per_document() {
        let doc = srgb_doc();
        let first = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        let second = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn matrix_trc_profile_compiles_to_the_shaper() {
        let doc = srgb_doc();
        let conv = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        assert!(matches!(*conv, IccConversion::Shaper(_)));
        assert_eq!(conv.convert(&doc, &[1.0, 0.0, 0.0]), Some([255, 0, 0]));
        assert_eq!(conv.convert(&doc, &[1.0, 0.0]), None);
    }

    #[test]
    fn unusable_profiles_do_not_compile() {
        let bytes = ColorProfile::new_srgb().encode().expect("encodes");
        assert!(IccConversion::compile(&bytes, 1).is_none());
        assert!(IccConversion::compile(b"not a profile", 3).is_none());
    }

    /// `[/ICCBased <unparseable profile, /N n, /Alternate alt>]`.
    fn garbage_doc(n: u8, alternate: &str) -> Document {
        make_doc_with_stream(b"not a profile", &format!(" /N {n} /Alternate {alternate}"))
    }

    fn icc(n: u8) -> ColorSpace {
        ColorSpace::IccBased {
            n,
            stream_id: Some((2, 0)),
        }
    }

    #[test]
    fn unusable_profile_converts_through_its_alternate() {
        let doc = garbage_doc(3, "[/Lab << /WhitePoint [0.9505 1 1.089] >>]");
        let gray = icc(3).convert_to_rgb(&doc, &[50.0, 0.0, 0.0]);
        assert!(gray.iter().all(|&c| c.abs_diff(119) <= 1), "got {gray:?}");
    }

    #[test]
    fn alternate_with_another_component_count_is_ignored() {
        let doc = garbage_doc(3, "/DeviceGray");
        assert_eq!(icc(3).convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
    }

    #[test]
    fn derived_alternates_are_ignored() {
        let doc = garbage_doc(3, "[/ICCBased 2 0 R]");
        assert_eq!(icc(3).convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
        // A Separation alternate whose own alternate is this profile.
        let doc = garbage_doc(
            1,
            "[/Separation /Spot [/ICCBased 2 0 R] << /FunctionType 2 /Domain [0 1] /N 1 >>]",
        );
        assert_eq!(icc(1).convert_to_rgb(&doc, &[1.0]), [255; 3]);
    }

    #[test]
    fn cal_alternate_converts_through_its_parameters() {
        // CalGray with the default Gamma 1: 0.5 is linear light, sRGB 188.
        let doc = garbage_doc(1, "[/CalGray << /WhitePoint [0.9505 1 1.089] >>]");
        let gray = icc(1).convert_to_rgb(&doc, &[0.5]);
        assert!(gray.iter().all(|&c| c.abs_diff(188) <= 1), "got {gray:?}");
    }
}
