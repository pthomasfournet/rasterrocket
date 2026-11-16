//! The conversion of a colour in a space to the working space and to
//! process ink.

use color::Sample;
use pdf::Document;

use super::icc::IccConversion;
use super::{ColorSpace, TintTransform};
use crate::renderer::color::Srgb;
use crate::separation::Separator;
use crate::{ColourTarget, WorkingSpace};

impl ColorSpace {
    /// The process ink `[C, M, Y, K]` (0 = no ink) of the colour `comps`
    /// in this space, as a CMYK device would print it: the components
    /// themselves in `DeviceCMYK`, unless `separator` separates its colour
    /// ([`Separator::paints_device_cmyk`]); in a four-component ICC space,
    /// unless `separator` separates every process colour
    /// ([`Separator::separates_process_colours`]), under
    /// the preview's separator ([`Separator::Model`]) the inks nearest the
    /// components, with the same zero components, printing the profile's
    /// sRGB colour in the process-ink model
    /// ([`color::cmyk::ink_reproducing`]), and the components themselves
    /// under any other, converted to the ink of a CMYK output profile that
    /// is not the space's ([`Separator::cmyk_link`]); the ink of the
    /// alternate colour for a `Separation`
    /// or `DeviceN` space whose tint transform evaluates; otherwise
    /// `separator`'s ink of the sRGB conversion.  A component count other
    /// than [`ncomponents`](Self::ncomponents) converts as
    /// [`convert_to_rgb`](Self::convert_to_rgb) does, to black.  The ink
    /// and the colours it is found from are at the depth of `S`
    /// ([`Self::convert`]).
    #[must_use]
    pub(crate) fn process_ink<S: Sample>(
        &self,
        doc: &Document,
        target: ColourTarget,
        comps: &[f64],
        separator: &Separator,
    ) -> [S; 4] {
        self.process_ink_with(
            doc,
            target,
            comps,
            separator,
            &mut color::cmyk::ink_reproducing_at,
        )
    }

    /// [`process_ink`](Self::process_ink) with the inks reproducing the
    /// sRGB colour of a four-component ICC colour from its components
    /// found by `reproduce`, which returns what
    /// [`color::cmyk::ink_reproducing_at`] does for the same arguments.
    pub(crate) fn process_ink_with<S: Sample>(
        &self,
        doc: &Document,
        target: ColourTarget,
        comps: &[f64],
        separator: &Separator,
        reproduce: &mut dyn FnMut([S; 3], [f64; 4]) -> [S; 4],
    ) -> [S; 4] {
        match self {
            Self::DeviceCmyk if separator.paints_device_cmyk() => {
                if let &[c, m, y, k] = comps {
                    return [c, m, y, k].map(S::from_unit);
                }
            }
            Self::IccBased { n: 4, .. } if !separator.separates_process_colours() => {
                if let &[c, m, y, k] = comps {
                    if !separator.reproduces_process_colour() {
                        let cmyk = [c, m, y, k].map(S::from_unit);
                        return separator
                            .cmyk_link(doc, self)
                            .map_or(cmyk, |link| link.ink(cmyk));
                    }
                    return reproduce(self.convert::<S>(doc, target, comps), [c, m, y, k]);
                }
            }
            Self::Separation {
                alternate,
                tint: Some(tint),
                ..
            }
            | Self::DeviceN {
                alternate,
                tint: Some(tint),
                ..
            } if comps.len() == self.ncomponents() => {
                if let Some(alt) = tint.alternate_components(alternate, comps) {
                    return alternate.process_ink_with(doc, target, &alt, separator, reproduce);
                }
            }
            _ => {}
        }
        separator.ink(self.convert::<S>(doc, target, comps))
    }

    /// Convert a tint-component slice to a byte triple of the working space
    /// per PDF §8.6.
    ///
    /// `comps` must have exactly [`ncomponents`](Self::ncomponents)
    /// elements; any mismatch falls back to opaque black.
    ///
    /// `doc` is required so the conversion can dereference lazy
    /// `ObjectId` fields on parameterised variants (ICC profile streams,
    /// Separation/DeviceN tint functions, Indexed palette lookups).
    /// Pass the page's document; the borrow is read-only.
    ///
    /// `target` holds the working space the colour converts into and the
    /// rendering intent of the conversion (PDF §8.6.5.8): the intent
    /// selects the profile table of an ICC-based space, also one reached as
    /// an `Indexed` base or a `Separation`/`DeviceN` alternate; every other
    /// conversion is independent of it.  A device colour keeps its sRGB
    /// meaning: in the wide space its sRGB value converts through
    /// [`crate::working_space::from_srgb16`] at 16 bits.
    ///
    /// # Variant behaviour
    ///
    /// - `DeviceGray/Rgb`: direct conversion via the `color::convert`
    ///   helpers; `DeviceCmyk` via the process-ink model of `color::cmyk`.
    /// - `Cal`: gamma and matrix to XYZ, Bradford adaptation from the
    ///   space's white point to the working white, the working space's
    ///   encoding (see `CalSpace`).
    /// - `Lab`: `a*` and `b*` clipped to the `Range`, CIE Lab → XYZ relative
    ///   to the `WhitePoint`, Bradford adaptation to the working white, the
    ///   working space's encoding (see `LabSpace`).
    /// - `IccBased`: converts through the `icc::IccConversion` compiled
    ///   from the profile stream `stream_id`: shaper tables for a gray or
    ///   RGB matrix/TRC profile, otherwise a `moxcms` transform of a
    ///   3- or 4-channel profile to the working space, otherwise the stream's device or
    ///   Lab `/Alternate` space of the same component count.  When none
    ///   applies (missing stream, unusable profile and alternate, transform
    ///   failure) the device space with `n` components converts.
    /// - `Indexed`: the index (rounded, clamped to `hival`) selects a
    ///   palette entry from `lookup`; its bytes
    ///   map linearly onto the base space's component ranges and convert
    ///   through the base space.  A missing or short palette yields black.
    /// - `Pattern`: returns black — patterns are not directly convertible
    ///   to RGB; tile rasterisation is invoked elsewhere.
    /// - `Separation`: passes the tint through the tint transform `tint`
    ///   and converts the result via the alternate colour space.  A missing
    ///   function or a failed evaluation drops back to `1 - tint` as gray.
    /// - `DeviceN`: passes one tint per colorant through the tint-transform
    ///   function and converts the result via the alternate colour space.
    ///   Any failure drops back to `1 - comps[0]` as gray.
    #[must_use]
    pub fn convert_to_rgb(&self, doc: &Document, target: ColourTarget, comps: &[f64]) -> [u8; 3] {
        self.convert::<u8>(doc, target, comps)
    }

    /// [`Self::convert_to_rgb`] at both depths for a render at `depth`, as
    /// the working-space value a [`Srgb`] holds:
    /// the 8-bit bytes, and the 16-bit value converted at 16 bits for a
    /// 16-bit render, the bytes widened otherwise.
    #[must_use]
    pub(crate) fn convert_to_srgb(
        &self,
        doc: &Document,
        target: ColourTarget,
        comps: &[f64],
        depth: color::SampleDepth,
    ) -> Srgb {
        let bytes = self.convert::<u8>(doc, target, comps);
        match depth {
            color::SampleDepth::Eight => Srgb::of_bytes(bytes),
            color::SampleDepth::Sixteen => Srgb {
                bytes,
                wide: self.convert::<u16>(doc, target, comps),
            },
        }
    }

    /// [`Self::convert_to_rgb`] at the depth of `S`: every conversion
    /// evaluates at that depth and quantises its result once, so the 8-bit
    /// instance is the 8-bit conversion.
    #[must_use]
    pub(crate) fn convert<S: Sample>(
        &self,
        doc: &Document,
        target: ColourTarget,
        comps: &[f64],
    ) -> [S; 3] {
        if comps.len() != self.ncomponents() {
            log::debug!(
                "colorspace: {:?} expects {} components, got {}",
                self,
                self.ncomponents(),
                comps.len(),
            );
            return [S::default(); 3];
        }
        let wide = target.space == WorkingSpace::Wide;
        match self {
            Self::DeviceGray => [S::from_unit(comps[0]); 3],
            Self::DeviceRgb if wide => {
                from_srgb16([comps[0], comps[1], comps[2]].map(u16::from_unit))
            }
            Self::DeviceRgb => [comps[0], comps[1], comps[2]].map(S::from_unit),
            Self::DeviceCmyk if wide => from_srgb16(color::cmyk::cmyk_f64_to_rgb_at(
                comps[0], comps[1], comps[2], comps[3],
            )),
            Self::DeviceCmyk => {
                color::cmyk::cmyk_f64_to_rgb_at(comps[0], comps[1], comps[2], comps[3])
            }
            Self::Cal(cal) => cal.convert(comps, target.space),
            Self::Lab(lab) => lab.convert(comps, target.space),
            Self::IccBased { n, stream_id } => stream_id
                .and_then(|id| IccConversion::load(doc, id, *n, target))
                .and_then(|conv| conv.convert(doc, comps))
                .unwrap_or_else(|| {
                    if wide {
                        from_srgb16(icc_fallback_by_n(*n, comps))
                    } else {
                        icc_fallback_by_n(*n, comps)
                    }
                }),
            Self::Indexed { .. } => self.indexed_base_components(comps).map_or_else(
                || [S::default(); 3],
                |(base, entry)| base.convert(doc, target, &entry),
            ),
            Self::Pattern { .. } => [S::default(); 3],
            Self::Separation {
                alternate, tint, ..
            }
            | Self::DeviceN {
                alternate, tint, ..
            } => {
                // PDF §8.6.6.4, §8.6.6.5: pass the tint components, one per
                // colorant, through the tint-transform function to produce
                // values in the alternate colour space, then convert via the
                // alternate.  On any failure (no function, or evaluation
                // fails), use the `1 - tint` gray approximation of the first
                // component.
                if let Some(tint) = tint
                    && let Some(rgb) = tint.to_rgb(doc, target, alternate, comps)
                {
                    return rgb;
                }
                separation_fallback_gray(comps[0])
            }
        }
    }
}

impl TintTransform {
    /// Pass `tints` through the function and convert the result in
    /// `alternate`.  `None` when evaluation fails.
    fn to_rgb<S: Sample>(
        &self,
        doc: &Document,
        target: ColourTarget,
        alternate: &ColorSpace,
        tints: &[f64],
    ) -> Option<[S; 3]> {
        let outputs = self.alternate_components(alternate, tints)?;
        Some(alternate.convert(doc, target, &outputs))
    }
}

/// A spot tint with no resolvable transform approximates as gray with
/// `1 - tint`: tint 0 is no colorant (white), tint 1 is full colorant
/// (black), per PDF §8.6.6.4.
fn separation_fallback_gray<S: Sample>(tint: f64) -> [S; 3] {
    [S::from_unit(1.0 - tint); 3]
}

/// The wide value at the depth of `S` of the 16-bit sRGB value `srgb`
/// ([`crate::working_space::from_srgb16`]).
fn from_srgb16<S: Sample>(srgb: [u16; 3]) -> [S; 3] {
    crate::working_space::from_srgb16(srgb).map(S::from_sample)
}

/// Fall back to the device space matching the ICC profile's channel count
/// when the moxcms transform isn't available. Mirrors the pre-CLUT behaviour.
fn icc_fallback_by_n<S: Sample>(n: u8, comps: &[f64]) -> [S; 3] {
    match n {
        1 => [S::from_unit(comps[0]); 3],
        3 => [comps[0], comps[1], comps[2]].map(S::from_unit),
        4 => color::cmyk::cmyk_f64_to_rgb_at(comps[0], comps[1], comps[2], comps[3]),
        _ => [S::default(); 3],
    }
}

#[cfg(test)]
mod tests {
    use pdf::{Document, Object};

    use super::*;
    use crate::resources::colorspace::test_support::{RI, devicen_teal_doc, lab_d65, spot, spots};
    use crate::test_helpers::empty_doc;

    #[test]
    fn convert_devicegray_full_range() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, RI, &[0.0]),
            [0, 0, 0]
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, RI, &[1.0]),
            [255, 255, 255],
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, RI, &[0.5]),
            [128, 128, 128], // gray_to_u8 rounds 0.5*255 = 127.5 → 128
        );
    }

    #[test]
    fn convert_devicergb_passes_through() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, RI, &[1.0, 0.0, 0.0]),
            [255, 0, 0],
        );
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, RI, &[0.0, 1.0, 0.0]),
            [0, 255, 0],
        );
    }

    #[test]
    fn convert_devicecmyk_uses_process_ink_model() {
        let doc = empty_doc();
        // (0, 0, 0, 0) = paper white.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[0.0, 0.0, 0.0, 0.0]),
            [255, 255, 255],
        );
        // (1, 1, 1, 1) = four-colour solid = black.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[1.0, 1.0, 1.0, 1.0]),
            [0, 0, 0],
        );
        // Solid cyan is the printed process cyan, not the ideal (0, 255, 255).
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[1.0, 0.0, 0.0, 0.0]),
            color::cmyk::CORNERS[4],
        );
    }

    #[test]
    fn convert_lab_black_is_black() {
        let doc = empty_doc();
        // L*=0 ≡ black; a*/b* irrelevant.
        let black = lab_d65().convert_to_rgb(&doc, RI, &[0.0, 0.0, 0.0]);
        assert_eq!(black, [0, 0, 0]);
    }

    #[test]
    fn convert_lab_white_is_white() {
        let doc = empty_doc();
        // L*=100, a*=b*=0 ≡ D65 white.  sRGB encoding may not be exactly 255
        // due to floating-point rounding; allow ±2 LSB.
        let [r, g, b] = lab_d65().convert_to_rgb(&doc, RI, &[100.0, 0.0, 0.0]);
        assert!(
            r >= 253 && g >= 253 && b >= 253,
            "expected ~white, got [{r},{g},{b}]"
        );
    }

    #[test]
    fn convert_iccbased_no_stream_falls_back_by_n() {
        let doc = empty_doc();
        let icc1 = ColorSpace::IccBased {
            n: 1,
            stream_id: None,
        };
        let icc3 = ColorSpace::IccBased {
            n: 3,
            stream_id: None,
        };
        let icc4 = ColorSpace::IccBased {
            n: 4,
            stream_id: None,
        };
        // No stream_id → skip moxcms, use device-by-N fallback directly.
        assert_eq!(icc1.convert_to_rgb(&doc, RI, &[1.0]), [255, 255, 255]);
        assert_eq!(icc3.convert_to_rgb(&doc, RI, &[1.0, 0.0, 0.0]), [255, 0, 0]);
        assert_eq!(
            icc4.convert_to_rgb(&doc, RI, &[0.0, 0.0, 0.0, 1.0]),
            color::cmyk::CORNERS[8]
        );
    }

    #[test]
    fn convert_iccbased_unresolvable_stream_falls_back() {
        // stream_id points to an object not in the doc: no profile bytes,
        // so the device-by-N fallback runs.
        let doc = empty_doc();
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((999, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, RI, &[1.0, 0.0, 0.0]), [255, 0, 0]);
    }

    #[test]
    fn convert_iccbased_n1_without_a_profile_stream_falls_back() {
        // Object 1 of the empty document is not a stream: no profile bytes.
        let doc = empty_doc();
        let icc = ColorSpace::IccBased {
            n: 1,
            stream_id: Some((1, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, RI, &[0.5]), [128, 128, 128]);
    }

    /// Every conversion at 16 bits, narrowed to 8, lies within one level of
    /// the 8-bit conversion: device, CIE-based and ICC-based spaces alike.
    #[test]
    fn sixteen_bit_conversions_narrow_to_the_8_bit_ones() {
        let gamma = moxcms::ToneReprCurve::Lut(vec![461]);
        let mut rgb = moxcms::ColorProfile::new_srgb();
        rgb.cicp = None;
        rgb.red_trc = Some(gamma.clone());
        rgb.green_trc = Some(gamma.clone());
        rgb.blue_trc = Some(gamma);
        let shaper = crate::test_helpers::make_doc_with_stream(&rgb.encode().unwrap(), " /N 3");
        let mut adobe = moxcms::ColorProfile::new_adobe_rgb();
        adobe.cicp = None;
        let matrix = crate::test_helpers::make_doc_with_stream(&adobe.encode().unwrap(), " /N 3");
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((2, 0)),
        };
        let empty = empty_doc();
        let cases: [(&Document, ColorSpace, Vec<f64>); 6] = [
            (&empty, ColorSpace::DeviceGray, vec![0.37]),
            (&empty, ColorSpace::DeviceCmyk, vec![0.1, 0.6, 0.3, 0.2]),
            (&empty, lab_d65(), vec![61.0, 12.0, -40.0]),
            (&shaper, icc.clone(), vec![0.2, 0.5, 0.8]),
            (&matrix, icc.clone(), vec![0.9, 0.4, 0.1]),
            (&empty, icc, vec![0.3, 0.3, 0.3]),
        ];
        for (doc, space, comps) in cases {
            let narrow = space.convert::<u8>(doc, RI, &comps);
            let wide = space.convert::<u16>(doc, RI, &comps);
            for (w, n) in wide.iter().zip(narrow) {
                assert!(
                    w.to_u8().abs_diff(n) <= 1,
                    "{space:?} {comps:?}: {wide:?} {narrow:?}"
                );
            }
            assert_eq!(
                space
                    .convert_to_srgb(doc, RI, &comps, color::SampleDepth::Sixteen)
                    .wide,
                wide
            );
        }
    }

    #[test]
    fn convert_iccbased_gamma_1_8_profiles_apply_their_curves() {
        // u8Fixed8 461 = gamma 1.8: encode((128/255)^1.8) = 0.5740 → 146.
        let gamma = moxcms::ToneReprCurve::Lut(vec![461]);
        let mut gray = moxcms::ColorProfile::new_gray_with_gamma(1.0);
        gray.gray_trc = Some(gamma.clone());
        let doc = crate::test_helpers::make_doc_with_stream(&gray.encode().unwrap(), " /N 1");
        let icc = ColorSpace::IccBased {
            n: 1,
            stream_id: Some((2, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, RI, &[128.0 / 255.0]), [146; 3]);

        let mut rgb = moxcms::ColorProfile::new_srgb();
        rgb.cicp = None;
        rgb.red_trc = Some(gamma.clone());
        rgb.green_trc = Some(gamma.clone());
        rgb.blue_trc = Some(gamma);
        let doc = crate::test_helpers::make_doc_with_stream(&rgb.encode().unwrap(), " /N 3");
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((2, 0)),
        };
        let half = 128.0 / 255.0;
        assert_eq!(icc.convert_to_rgb(&doc, RI, &[half, half, half]), [146; 3]);
    }

    #[test]
    fn convert_iccbased_srgb_profile_round_trips_through_moxcms() {
        // Embed the canonical sRGB profile as a stream, reference it from
        // an IccBased CS with n=3, and verify that primary colours
        // round-trip through the moxcms sRGB→sRGB transform.  Identity-up-
        // to-LSB confirms the actual code path (not the fallback) runs.
        //
        // Tolerance: ±2 LSB covers the worst case of (a) 8-bit input
        // quantisation, (b) profile encode/decode rounding through the
        // moxcms parametric curve, and (c) the final 8-bit output cast.
        // Tighter than ±2 risks intermittent failures across moxcms
        // releases; looser invites false positives.
        const TOL: u8 = 2;
        fn channels_match(got: [u8; 3], expected: [u8; 3], label: &str) {
            for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                let diff = g.abs_diff(e);
                assert!(
                    diff <= TOL,
                    "{label}: channel {i} got {g}, expected {e}±{TOL} (diff {diff})"
                );
            }
        }

        let icc_bytes = moxcms::ColorProfile::new_srgb()
            .encode()
            .expect("moxcms must encode its own sRGB profile");
        let doc = crate::test_helpers::make_doc_with_stream(&icc_bytes, " /N 3");
        let icc = ColorSpace::IccBased {
            n: 3,
            stream_id: Some((2, 0)),
        };

        // Primary colours round-trip through sRGB→sRGB as near-identity.
        channels_match(
            icc.convert_to_rgb(&doc, RI, &[1.0, 0.0, 0.0]),
            [255, 0, 0],
            "pure red",
        );
        channels_match(
            icc.convert_to_rgb(&doc, RI, &[0.0, 1.0, 0.0]),
            [0, 255, 0],
            "pure green",
        );
        channels_match(
            icc.convert_to_rgb(&doc, RI, &[0.0, 0.0, 1.0]),
            [0, 0, 255],
            "pure blue",
        );
        // Mid grey (0.5, 0.5, 0.5).  sRGB encodes 0.5-linear ≈ 188 but the
        // input is *already* sRGB-encoded; identity should keep it at
        // ~128 modulo transform rounding.  Additionally assert neutrality
        // — moxcms shouldn't introduce a colour cast on grey inputs.
        let grey = icc.convert_to_rgb(&doc, RI, &[0.5, 0.5, 0.5]);
        channels_match(grey, [128, 128, 128], "mid grey");
        assert!(
            grey[0] == grey[1] && grey[1] == grey[2],
            "sRGB→sRGB grey must stay neutral, got {grey:?}",
        );
    }

    #[test]
    fn convert_separation_without_tint_fn_falls_back_to_gray() {
        // No tint transform → the 1 - tint heuristic.
        let doc = empty_doc();
        let sep = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: None,
            colorant: spot(),
        };
        assert_eq!(sep.convert_to_rgb(&doc, RI, &[0.0]), [255, 255, 255]); // no ink
        assert_eq!(sep.convert_to_rgb(&doc, RI, &[1.0]), [0, 0, 0]); // full ink
    }

    #[test]
    fn convert_separation_unresolvable_tint_fn_falls_back() {
        // The tint transform references a missing object → evaluation
        // fails → the gray heuristic.
        let doc = empty_doc();
        let sep = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((999, 0)))),
            colorant: spot(),
        };
        assert_eq!(sep.convert_to_rgb(&doc, RI, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn convert_devicen_multi_input_evaluates_the_tint_transform() {
        let doc = devicen_teal_doc();
        let dn = ColorSpace::DeviceN {
            colorants: spots(2),
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((4, 0)))),
        };
        // Full spot, no black: the teal CMYK, not the gray of comps[0].
        let teal = dn.convert_to_rgb(&doc, RI, &[0.0, 1.0]);
        let want = ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[1.0, 0.08, 0.26, 0.38]);
        assert_eq!(teal, want);
        assert!(teal[2] > teal[0] + 40, "teal, got {teal:?}");
        // Full black on top saturates K.
        assert_eq!(
            dn.convert_to_rgb(&doc, RI, &[1.0, 1.0]),
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[1.0, 0.08, 0.26, 1.0])
        );
        // No ink at all is paper white.
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[0.0, 0.0]), [255, 255, 255]);
    }

    #[test]
    fn convert_devicen_multi_input_without_tint_fn_falls_back_to_gray() {
        // No tint transform → gray heuristic on comps[0].
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            colorants: spots(3),
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: None,
        };
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[1.0, 0.5, 0.5]), [0, 0, 0]);
    }

    #[test]
    fn convert_devicen_single_input_no_tint_fn_falls_back_to_gray() {
        // One colorant with no resolvable tint function → gray heuristic on the
        // single tint component (1.0 → "full ink" → black).
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            colorants: spots(1),
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: None,
        };
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[1.0]), [0, 0, 0]);
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[0.0]), [255, 255, 255]);
    }

    #[test]
    fn convert_devicen_single_input_unresolvable_tint_fn_falls_back() {
        // One colorant with a tint reference that doesn't resolve → evaluation
        // returns None → fall back to the gray heuristic.
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            colorants: spots(1),
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((999, 0)))),
        };
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn convert_separation_with_separation_alternate_terminates() {
        // Adversarial: a Separation whose alternate is another Separation
        // (PDF spec forbids but our parser would accept).  Conversion must
        // not infinite-loop; the parser's MAX_CS_DEPTH bounds the
        // structure, and convert_to_rgb's recursion is bounded by the
        // same depth.  Verifies the recursion terminates with *some* output.
        let doc = empty_doc();
        let inner = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceGray),
            tint: None,
            colorant: spot(),
        };
        let outer = ColorSpace::Separation {
            alternate: Box::new(inner),
            tint: None,
            colorant: spot(),
        };
        // No infinite loop, deterministic byte triple.
        let result = outer.convert_to_rgb(&doc, RI, &[0.5]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn convert_wrong_component_count_falls_back_to_black() {
        let doc = empty_doc();
        // DeviceRgb wants 3 components; passing 1 must not panic.
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, RI, &[0.5]),
            [0, 0, 0]
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, RI, &[]),
            [0, 0, 0]
        );
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, RI, &[0.5, 0.5]),
            [0, 0, 0],
        );
    }

    #[test]
    fn convert_pattern_returns_black() {
        let doc = empty_doc();
        // Pattern is never directly convertible to RGB.
        assert_eq!(
            ColorSpace::Pattern { base: None }.convert_to_rgb(&doc, RI, &[]),
            [0, 0, 0],
        );
    }
}
