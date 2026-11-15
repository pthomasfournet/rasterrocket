//! Colour-value conversion through an ICC profile stream (PDF §8.6.5.5).
//!
//! A profile stream is parsed once per document into an [`IccProfile`],
//! which compiles an [`IccConversion`] per device-to-PCS model its
//! rendering intents select (ISO 32000-1 §8.6.5.8): the `AToB0`, `AToB1`
//! or `AToB2` table of the intent (ICC.1:2022 §8.10.2, `AToB0` standing in
//! for a table the profile lacks), or, in a profile without such tables,
//! its matrix/TRC model.  Each model compiles at most twice, media-relative
//! and ICC-absolute; intents that select the same model share one
//! conversion.
//!
//! A gray or RGB matrix/TRC model compiles into the shared shaper tables of
//! `super::super::icc_shaper`, any other 3- or 4-channel profile into a
//! `moxcms` 8-bit transform to sRGB, beside which a 16-bit transform is
//! created on first use by a 16-bit render.  ICC-absolute colorimetry
//! scales the PCS by the source's media white (see [`crate::pcs`]), folded
//! into the sRGB destination's colorants.  A profile that admits neither
//! compiles to its `/Alternate` space when that space takes the same
//! number of components (§8.6.5.5, Table 66).  Converting a colour value
//! is then a table lookup, one transform call, or a conversion in the
//! alternate space.

use std::sync::{Arc, OnceLock};

use color::Sample;
use moxcms::{
    ColorProfile, Layout, Transform8BitExecutor, Transform16BitExecutor, TransformOptions,
};
use pdf::{Document, ObjectId};

use super::ColorSpace;
use crate::RenderingIntent;
use crate::resources::icc_shaper::{ProfileConversion, fetch_profile_bytes};

#[cfg(feature = "gpu-icc")]
pub(crate) mod clut;

/// The device-to-PCS model of a profile: its matrix/TRC model, or one of
/// its `AToB` tables.  The discriminant indexes [`IccProfile::conversions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Model {
    /// The matrix/TRC model of a profile without `AToB` tables.
    Matrix = 0,
    /// `AToB0`, the perceptual table.
    AToB0 = 1,
    /// `AToB1`, the colorimetric table.
    AToB1 = 2,
    /// `AToB2`, the saturation table.
    AToB2 = 3,
}

impl Model {
    /// Every model, by discriminant.
    const ALL: [Self; 4] = [Self::Matrix, Self::AToB0, Self::AToB1, Self::AToB2];

    /// The model `intent` selects in `profile`: the table of the intent
    /// (`AToB1` for both colorimetric intents), `AToB0` when the profile
    /// lacks it, the matrix/TRC model when the profile has no table.
    const fn of(profile: &ColorProfile, intent: RenderingIntent) -> Self {
        let (own, model) = match intent {
            RenderingIntent::Perceptual => (&profile.lut_a_to_b_perceptual, Self::AToB0),
            RenderingIntent::RelativeColorimetric | RenderingIntent::AbsoluteColorimetric => {
                (&profile.lut_a_to_b_colorimetric, Self::AToB1)
            }
            RenderingIntent::Saturation => (&profile.lut_a_to_b_saturation, Self::AToB2),
        };
        if own.is_some() {
            model
        } else if profile.lut_a_to_b_perceptual.is_some() {
            Self::AToB0
        } else if profile.lut_a_to_b_colorimetric.is_some()
            || profile.lut_a_to_b_saturation.is_some()
        {
            // No table of the intent nor `AToB0`: the converter reads the
            // model it selects for the intent's own tag.
            model
        } else {
            Self::Matrix
        }
    }

    /// The converter intent that evaluates the model.  The matrix/TRC
    /// model is intent-independent and evaluates under the converter's
    /// default intent.
    const fn cms(self) -> moxcms::RenderingIntent {
        match self {
            Self::Matrix | Self::AToB0 => moxcms::RenderingIntent::Perceptual,
            Self::AToB1 => moxcms::RenderingIntent::RelativeColorimetric,
            Self::AToB2 => moxcms::RenderingIntent::Saturation,
        }
    }
}

/// Number of conversions an [`IccProfile`] holds: every [`Model`],
/// media-relative and ICC-absolute.
const SLOTS: usize = 8;

/// The slot of the ICC-absolute conversion of a model: the model's own
/// slot offset by this.
const ABSOLUTE: usize = 4;

/// One ICC profile stream, parsed, and the conversions of its colour
/// values compiled for the rendering intents that use it.
pub(crate) struct IccProfile {
    /// The component count of the colour values converted.
    n: u8,
    /// The parsed profile; `None` when the stream is missing or is not a
    /// profile.
    profile: Option<Arc<ColorProfile>>,
    /// The PCS scale of ICC-absolute colorimetry (see
    /// [`crate::pcs::scale_from_white`]); `None` when the media white is
    /// D50, under which ICC-absolute is media-relative colorimetry.
    absolute: Option<[f64; 3]>,
    /// The slot in [`Self::conversions`] of each intent, by
    /// [`RenderingIntent::index`].
    slots: [usize; 4],
    /// The conversion of each slot, compiled on first use; `None` when
    /// neither the profile nor the `/Alternate` space converts.
    conversions: [OnceLock<Option<Arc<IccConversion>>>; SLOTS],
}

impl IccProfile {
    /// The profile of `bytes` for `n`-component values; a profile that
    /// does not parse compiles no conversion of its own.
    fn new(bytes: Option<&[u8]>, n: u8) -> Self {
        let profile = bytes.and_then(|bytes| {
            ColorProfile::new_from_slice(bytes)
                .map_err(|e| log::debug!("colorspace: ICC profile parse failed: {e}"))
                .ok()
        });
        let absolute = profile
            .as_ref()
            .and_then(|p| crate::pcs::scale_from_white(crate::pcs::media_white(p)));
        let slots = RenderingIntent::ALL.map(|intent| {
            let Some(profile) = &profile else {
                return 0;
            };
            let model = Model::of(profile, intent) as usize;
            if intent == RenderingIntent::AbsoluteColorimetric && absolute.is_some() {
                model + ABSOLUTE
            } else {
                model
            }
        });
        Self {
            n,
            profile: profile.map(Arc::new),
            absolute,
            slots,
            conversions: Default::default(),
        }
    }

    /// The conversion of the profile's colour values under `intent`,
    /// compiled on first use; `alternate` supplies the `/Alternate` space
    /// a profile without a conversion of its own converts through.
    fn conversion(
        &self,
        intent: RenderingIntent,
        alternate: impl FnOnce() -> Option<ColorSpace>,
    ) -> Option<Arc<IccConversion>> {
        let slot = self.slots[intent.index()];
        self.conversions[slot]
            .get_or_init(|| {
                self.compile(slot)
                    .or_else(|| alternate().map(IccConversion::Alternate))
                    .map(Arc::new)
            })
            .clone()
    }

    /// Compile the conversion of `slot` from the profile.  `None` when
    /// there is no profile, `n` has no layout, or the converter refuses the
    /// transform.
    fn compile(&self, slot: usize) -> Option<IccConversion> {
        let profile = self.profile.as_ref()?;
        let absolute = slot >= ABSOLUTE;
        let model = slot % ABSOLUTE;
        if model == Model::Matrix as usize && !absolute {
            let shaper = ProfileConversion::from_profile(profile);
            if matches!(
                (&shaper, self.n),
                (ProfileConversion::Gray(_), 1) | (ProfileConversion::Rgb(_), 3)
            ) {
                return Some(IccConversion::Shaper(shaper));
            }
        }
        let layout = layout(self.n)?;
        let dest = self
            .absolute
            .filter(|_| absolute)
            .map_or_else(ColorProfile::new_srgb, crate::pcs::scaled_srgb);
        let options = TransformOptions {
            rendering_intent: Model::ALL[model].cms(),
            ..TransformOptions::default()
        };
        let xform = profile
            .create_transform_8bit(layout, &dest, Layout::Rgb, options)
            .map_err(|e| log::debug!("colorspace: ICC transform creation failed: {e}"))
            .ok()?;
        Some(IccConversion::Transform {
            n: self.n,
            xform,
            profile: Arc::clone(profile),
            dest: Box::new(dest),
            options,
            xform16: OnceLock::new(),
            #[cfg(feature = "gpu-icc")]
            clut: OnceLock::new(),
        })
    }

    /// Whether the profile's colour values convert as those of the device
    /// space of the same component count under every intent: one
    /// conversion serves every intent, and it is device-equivalent (see
    /// [`IccConversion::is_device_equivalent`]).
    fn is_device_equivalent(&self, alternate: impl FnOnce() -> Option<ColorSpace>) -> bool {
        self.slots.iter().all(|&s| s == self.slots[0])
            && self
                .conversion(RenderingIntent::RelativeColorimetric, alternate)
                .is_some_and(|conv| conv.is_device_equivalent())
    }
}

/// The source layout of `n`-channel colour values.
const fn layout(n: u8) -> Option<Layout> {
    match n {
        1 => Some(Layout::Gray),
        3 => Some(Layout::Rgb),
        4 => Some(Layout::Rgba),
        _ => None,
    }
}

/// The compiled sRGB conversion of one ICC profile stream for colour values
/// of its declared component count under one device-to-PCS model.
pub(crate) enum IccConversion {
    /// A gray (`n = 1`) or RGB (`n = 3`) matrix/TRC profile.
    Shaper(ProfileConversion),
    /// Any other profile with `n` of 1, 3 or 4: an 8-bit transform whose
    /// source pixel holds `n` channels.
    Transform {
        n: u8,
        xform: Arc<Transform8BitExecutor>,
        /// The source profile, from which the 16-bit transform is created.
        profile: Arc<ColorProfile>,
        /// The destination profile, sRGB, of both transforms.
        dest: Box<ColorProfile>,
        /// The options of both transforms.
        options: TransformOptions,
        /// The 16-bit transform, created on first use; `None` when moxcms
        /// refuses it.
        xform16: OnceLock<Option<Arc<Transform16BitExecutor>>>,
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
    /// The conversion of profile stream `id` for `n`-component values
    /// under `intent`, compiled on first use and shared for the lifetime of
    /// `doc`.  `None` when neither the profile nor the `/Alternate` space
    /// converts `n` components.
    pub(crate) fn load(
        doc: &Document,
        id: ObjectId,
        n: u8,
        intent: RenderingIntent,
    ) -> Option<Arc<Self>> {
        Self::profile(doc, id, n)?.conversion(intent, || Self::alternate(doc, id, n))
    }

    /// Whether profile stream `id` converts `n`-component values as the
    /// device space of `n` components does under every intent.
    pub(super) fn is_device_equivalent_stream(doc: &Document, id: ObjectId, n: u8) -> bool {
        Self::profile(doc, id, n)
            .is_some_and(|p| p.is_device_equivalent(|| Self::alternate(doc, id, n)))
    }

    /// The parsed profile stream `id`, shared for the lifetime of `doc`.
    fn profile(doc: &Document, id: ObjectId, n: u8) -> Option<Arc<IccProfile>> {
        doc.derived(id, || {
            Some(IccProfile::new(fetch_profile_bytes(doc, id).as_deref(), n))
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

    /// Compile profile `bytes` for `n`-component values under `intent`
    /// without consulting a cache.
    pub(crate) fn compile(bytes: &[u8], n: u8, intent: RenderingIntent) -> Option<Arc<Self>> {
        IccProfile::new(Some(bytes), n).conversion(intent, || None)
    }

    /// The CMYK → sRGB table of a four-channel transform on the
    /// `DEFAULT_GRID_N` lattice, baked once per conversion and shared;
    /// `None` for any other conversion or when the bake fails.
    #[cfg(feature = "gpu-icc")]
    pub(crate) fn cmyk_clut(&self) -> Option<Arc<[u8]>> {
        use clut::{DEFAULT_GRID_N, bake_cmyk_clut_with};

        let Self::Transform {
            n: 4, xform, clut, ..
        } = self
        else {
            return None;
        };
        clut.get_or_init(|| {
            bake_cmyk_clut_with(&**xform, DEFAULT_GRID_N)
                .map_err(|e| {
                    log::warn!(
                        "colorspace: ICC CLUT bake failed, converting through the transform: {e}"
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

    /// Convert normalised components to sRGB at the depth of `S`.
    /// Components are quantised to that depth for the transform, which is
    /// the 8-bit or the 16-bit one; the shaper evaluates at that depth (see
    /// `ProfileConversion::convert_components_at`).  `None` when `comps`
    /// does not hold the compiled component count or the transform fails.
    pub(crate) fn convert<S: Sample>(&self, doc: &Document, comps: &[f64]) -> Option<[S; 3]> {
        match self {
            Self::Alternate(space) => {
                (comps.len() == space.ncomponents()).then(|| space.convert(doc, comps))
            }
            Self::Shaper(shaper) => shaper.convert_components_at(comps),
            Self::Transform { n, xform, .. } => {
                if comps.len() != usize::from(*n) {
                    return None;
                }
                if S::DEPTH == color::SampleDepth::Eight {
                    return run(&**xform, comps).map(|rgb| rgb.map(S::from_u8));
                }
                self.xform16().map_or_else(
                    || run(&**xform, comps).map(|rgb| rgb.map(S::from_u8)),
                    |x| run(&**x, comps).map(|rgb| rgb.map(|v| S::from_unit(v.to_unit()))),
                )
            }
        }
    }

    /// The 16-bit transform of a [`Self::Transform`], created on first
    /// use; `None` for another conversion, or when moxcms refuses it, which
    /// is logged once.
    fn xform16(&self) -> Option<&Arc<Transform16BitExecutor>> {
        let Self::Transform {
            n,
            profile,
            dest,
            options,
            xform16,
            ..
        } = self
        else {
            return None;
        };
        xform16
            .get_or_init(|| {
                profile
                    .create_transform_16bit(layout(*n)?, dest, Layout::Rgb, *options)
                    .map_err(|e| {
                        log::warn!(
                            "colorspace: 16-bit ICC transform creation failed, \
                             converting at 8 bits: {e}"
                        );
                    })
                    .ok()
            })
            .as_ref()
    }

    /// [`Self::transform_pixels`] at 16 bits, through the 16-bit transform.
    /// `None` for a conversion that is not a transform, a length mismatch,
    /// or a transform that is refused or fails.
    pub(crate) fn transform_pixels16(&self, src: &[u16], dst: &mut [u16]) -> Option<()> {
        let Self::Transform { n, .. } = self else {
            return None;
        };
        let n = usize::from(*n);
        if !src.len().is_multiple_of(n) || dst.len() != src.len() / n * 3 {
            return None;
        }
        self.xform16()?
            .transform(src, dst)
            .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
            .ok()
    }
}

/// Run `xform` on the normalised components `comps`, quantised to its
/// sample type: one pixel of `comps.len()` channels to one sRGB pixel.
/// `None` when the transform fails.
fn run<S: Sample>(
    xform: &(dyn moxcms::TransformExecutor<S> + Send + Sync),
    comps: &[f64],
) -> Option<[S; 3]> {
    let mut src = [S::default(); 4];
    for (dst, &c) in src.iter_mut().zip(comps) {
        *dst = S::from_unit(c);
    }
    let mut rgb = [S::default(); 3];
    xform
        .transform(&src[..comps.len()], &mut rgb)
        .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
        .ok()?;
    Some(rgb)
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
        let conv = IccConversion::load(&doc, (2, 0), 3, RenderingIntent::default())
            .expect("sRGB profile compiles");
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
        let first = IccConversion::load(&doc, (2, 0), 3, RenderingIntent::default())
            .expect("sRGB profile compiles");
        let second = IccConversion::load(&doc, (2, 0), 3, RenderingIntent::default())
            .expect("sRGB profile compiles");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn matrix_trc_profile_compiles_to_the_shaper() {
        let doc = srgb_doc();
        let conv = IccConversion::load(&doc, (2, 0), 3, RenderingIntent::default())
            .expect("sRGB profile compiles");
        assert!(matches!(*conv, IccConversion::Shaper(_)));
        assert_eq!(
            conv.convert::<u8>(&doc, &[1.0, 0.0, 0.0]),
            Some([255, 0, 0])
        );
        assert_eq!(conv.convert::<u8>(&doc, &[1.0, 0.0]), None);
    }

    #[test]
    fn unusable_profiles_do_not_compile() {
        let bytes = ColorProfile::new_srgb().encode().expect("encodes");
        let intent = RenderingIntent::default();
        assert!(IccConversion::compile(&bytes, 1, intent).is_none());
        assert!(IccConversion::compile(b"not a profile", 3, intent).is_none());
    }

    /// The bytes of the test profile `name` of `tests/fixtures/icc`, a
    /// synthetic printer whose `AToB0`, `AToB1` and `AToB2` tables differ
    /// and whose paper white is tinted.
    fn fixture(name: &str) -> Vec<u8> {
        let path = format!(
            "{}/../../tests/fixtures/icc/{name}.icc",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
    }

    /// `[/ICCBased <printer-cmyk-v4>]` and the conversion of every intent.
    fn cmyk_printer() -> (Document, [Arc<IccConversion>; 4]) {
        let doc = make_doc_with_stream(&fixture("printer-cmyk-v4"), " /N 4");
        let conv = RenderingIntent::ALL
            .map(|intent| IccConversion::load(&doc, (2, 0), 4, intent).expect("compiles"));
        (doc, conv)
    }

    #[test]
    fn each_intent_converts_through_its_own_table() {
        use RenderingIntent::{AbsoluteColorimetric, Perceptual, RelativeColorimetric, Saturation};
        let (doc, conv) = cmyk_printer();
        let at = |intent: RenderingIntent| &conv[intent.index()];
        for (a, b) in [
            (RelativeColorimetric, Perceptual),
            (RelativeColorimetric, Saturation),
            (Perceptual, Saturation),
            (RelativeColorimetric, AbsoluteColorimetric),
        ] {
            assert!(!Arc::ptr_eq(at(a), at(b)), "{a} and {b} share a conversion");
        }
        // A saturated ink: the perceptual table maps it back out of the
        // compressed gamut, so the two differ.
        let ink = [0.0, 1.0, 1.0, 0.0];
        let relative = at(RelativeColorimetric).convert::<u8>(&doc, &ink).unwrap();
        let perceptual = at(Perceptual).convert::<u8>(&doc, &ink).unwrap();
        assert_ne!(relative, perceptual);
        // Repeated loads return the compiled conversion.
        let again = IccConversion::load(&doc, (2, 0), 4, RelativeColorimetric).unwrap();
        assert!(Arc::ptr_eq(&again, at(RelativeColorimetric)));
    }

    #[test]
    fn absolute_colorimetry_reproduces_the_paper_white() {
        use RenderingIntent::{AbsoluteColorimetric, RelativeColorimetric};
        let (doc, conv) = cmyk_printer();
        let paper = [0.0; 4];
        let relative = conv[RelativeColorimetric.index()].convert::<u8>(&doc, &paper);
        assert!(
            relative.unwrap().iter().all(|&c| c >= 254),
            "relative paper {relative:?}"
        );
        // The paper white (0.93, 0.965, 0.76) relative to D50: darker, and
        // yellow, its blue the lowest channel; an independent converter
        // gives (252.9, 250.9, 245.2).
        let [r, g, b] = conv[AbsoluteColorimetric.index()]
            .convert::<u8>(&doc, &paper)
            .unwrap();
        assert!(
            r < 254 && g < 254 && b + 5 < r.min(g),
            "absolute paper {:?}",
            [r, g, b]
        );
        let wide = conv[AbsoluteColorimetric.index()]
            .convert::<u16>(&doc, &paper)
            .unwrap();
        assert_eq!(wide.map(|v| u8::from_unit(v.to_unit())), [r, g, b]);
    }

    /// A matrix/TRC profile has one model: every intent but an
    /// ICC-absolute one off a D50 media white shares its conversion.
    #[test]
    fn a_matrix_profile_shares_one_conversion() {
        let doc = srgb_doc();
        let conv = RenderingIntent::ALL
            .map(|intent| IccConversion::load(&doc, (2, 0), 3, intent).expect("compiles"));
        assert!(conv.iter().all(|c| Arc::ptr_eq(c, &conv[0])));
        assert!(matches!(*conv[0], IccConversion::Shaper(_)));
    }

    /// A profile with both a matrix/TRC model and an `AToB0` table converts
    /// through the table.
    #[test]
    fn a_device_to_pcs_table_takes_precedence_over_the_matrix() {
        let printer = ColorProfile::new_from_slice(&fixture("printer-rgb-v4")).expect("parses");
        let mut both = ColorProfile::new_srgb();
        both.lut_a_to_b_perceptual = printer.lut_a_to_b_perceptual;
        let bytes = both.encode().expect("encodes");
        let doc = make_doc_with_stream(&bytes, " /N 3");
        let conv = IccConversion::load(&doc, (2, 0), 3, RenderingIntent::default()).unwrap();
        assert!(matches!(*conv, IccConversion::Transform { .. }));
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
