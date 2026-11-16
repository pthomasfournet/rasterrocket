//! The 8-bit conversion of samples in one space under one `/Decode`.

use std::cell::Cell;
use std::sync::Arc;

use pdf::Document;

use super::budget::TintBudget;
use super::decode_array::{decode_luts, is_identity_pair};
use crate::ColourTarget;
use crate::resources::colorspace::icc::IccConversion;
use crate::resources::colorspace::{CalSpace, ColorSpace, EXACT_WORK_BUDGET, decoded_sample};
use crate::resources::icc_shaper::{ProfileConversion, RgbShaper};
use crate::resources::image::{ImageColorSpace, try_filled, try_with_capacity};

/// The conversion of samples in one space under one `/Decode`.
pub(super) struct Decoded<'a> {
    pub(super) doc: &'a Document,
    /// The working space and intent the samples convert to.
    pub(super) target: ColourTarget,
    pub(super) space: &'a ColorSpace,
    pub(super) decode: &'a [(f64, f64)],
    max_sample: f64,
    /// The codestream's own profile, compiled, which converts in place of
    /// `space` (then the device space of the same channel count) when
    /// present.
    pub(super) profile: Option<Arc<IccConversion>>,
    /// The work of one exact conversion: one through the profile, else
    /// [`ColorSpace::conversion_cost`].
    pub(super) cost: u64,
    /// The work the image's conversions may spend: [`EXACT_WORK_BUDGET`],
    /// or the page's [`TintBudget`] when that holds less and a conversion
    /// costs more than one operation.
    pub(super) allowance: u64,
    /// The work spent by exact conversions so far, at [`Self::cost`] each,
    /// when a conversion costs more than one operation; zero otherwise.
    pub(super) spent: Cell<u64>,
    /// The page's budget, when [`Self::allowance`] was drawn from it and
    /// is below [`EXACT_WORK_BUDGET`]: a lattice or a drop is then a
    /// curtailment of the page (see [`TintBudget::curtailments`]).
    pub(super) limiting: Option<&'a TintBudget>,
}

/// The colour of every value of a one-component sample.
pub(super) enum Lut {
    /// Every value maps to the neutral gray of the same level.
    Identity,
    /// Every value maps to a neutral gray: one byte per pixel.
    Gray(Box<[u8; 256]>),
    Rgb(Box<[[u8; 3]; 256]>),
}

/// The conversion of three-component samples after `/Decode`.
pub(super) enum RgbConvert {
    Identity,
    /// `CalRGB`, or a matrix/TRC ICC profile.
    Shaper(Arc<RgbShaper>),
    /// Any other ICC profile.
    Transform(Arc<IccConversion>),
    /// Per-pixel conversion through the space, with `/Decode` folded in.
    Memo,
}

impl RgbConvert {
    /// The conversion through `conv` when it is a shaper or a transform.
    fn of_profile(conv: &Arc<IccConversion>) -> Option<Self> {
        match &**conv {
            IccConversion::Shaper(ProfileConversion::Rgb(Some(shaper))) => {
                Some(Self::Shaper(Arc::clone(&shaper.shaper)))
            }
            IccConversion::Transform { .. } => Some(Self::Transform(Arc::clone(conv))),
            _ => None,
        }
    }

    /// The conversion of a `CalRGB` space: its shaper, else the identity.
    fn of_cal(cal: &CalSpace) -> Self {
        cal.rgb_shaper().map_or(Self::Identity, Self::Shaper)
    }
}

/// The conversion of four-component samples after `/Decode`.
enum CmykConvert {
    /// The device CMYK model.
    Device,
    /// An ICC CMYK profile.
    Icc(Arc<IccConversion>),
    /// Per-pixel conversion through the space, with `/Decode` folded in.
    Memo,
}

impl CmykConvert {
    /// The conversion through `conv`: its transform, else the device model.
    fn of_profile(conv: Arc<IccConversion>) -> Self {
        if matches!(*conv, IccConversion::Transform { .. }) {
            Self::Icc(conv)
        } else {
            Self::Device
        }
    }
}

impl<'a> Decoded<'a> {
    /// The conversion of samples in `space` under `decode`, through
    /// `profile` when there is one, drawing on `budget` (see
    /// [`TintBudget`]).
    pub(super) fn new(
        doc: &'a Document,
        target: ColourTarget,
        space: &'a ColorSpace,
        decode: &'a [(f64, f64)],
        max_sample: f64,
        profile: Option<Arc<IccConversion>>,
        budget: Option<&'a TintBudget>,
    ) -> Self {
        let cost = if profile.is_some() {
            1
        } else {
            space.conversion_cost().max(1)
        };
        let limiting = budget.filter(|b| cost > 1 && b.remaining() < EXACT_WORK_BUDGET);
        let allowance = limiting.map_or(EXACT_WORK_BUDGET, TintBudget::remaining);
        Self {
            doc,
            target,
            space,
            decode,
            max_sample,
            profile,
            cost,
            allowance,
            spent: Cell::new(0),
            limiting,
        }
    }

    /// The ICC conversion of the samples: the codestream's own profile,
    /// else that of an ICC-based space.
    pub(super) fn icc_conversion(&self) -> Option<Arc<IccConversion>> {
        self.profile.clone().or_else(|| match self.space {
            ColorSpace::IccBased {
                n: k,
                stream_id: Some(id),
            } => IccConversion::load(self.doc, *id, *k, self.target),
            _ => None,
        })
    }

    /// Whether `/Decode` leaves every sample unchanged.
    pub(super) fn identity_decode(&self) -> bool {
        self.decode.iter().all(|&(d0, d1)| is_identity_pair(d0, d1))
    }

    /// The colour of the sample tuple `px` through the profile when there
    /// is one, otherwise through the space.
    fn to_rgb(&self, px: &[u8]) -> [u8; 3] {
        self.to_rgb_with(px, &mut Vec::new())
    }

    /// [`Self::to_rgb`], with the component values built in `comps`.
    pub(super) fn to_rgb_with(&self, px: &[u8], comps: &mut Vec<f64>) -> [u8; 3] {
        if self.cost > 1 {
            self.spent.set(self.spent.get().saturating_add(self.cost));
        }
        comps.clear();
        comps.extend(
            px.iter()
                .zip(self.decode)
                .map(|(&s, &pair)| decoded_sample(s, pair, self.max_sample)),
        );
        self.profile
            .as_ref()
            .and_then(|p| p.convert(self.doc, comps))
            .unwrap_or_else(|| self.space.convert_to_rgb(self.doc, self.target, comps))
    }

    /// The table of the space's conversion of every one-component sample:
    /// exact when 256 conversions fit [`Self::allowance`], otherwise
    /// interpolated in a lattice (see [`Self::lattice`]).  A table whose
    /// entries are each neutral to within one level (the rounding of a
    /// CIE-based gray conversion) keeps one byte per pixel, taking the
    /// green channel.  `None` when no lattice fits the allowance.
    pub(super) fn lut(&self) -> Option<Lut> {
        let mut rgb = Box::new([[0u8; 3]; 256]);
        if 256 * self.cost <= self.allowance {
            for (s, entry) in (0u8..=255).zip(rgb.iter_mut()) {
                *entry = self.to_rgb(&[s]);
            }
        } else {
            let lattice = self.lattice(1)?;
            let mut axes = Vec::with_capacity(1);
            for (s, entry) in (0u8..=255).zip(rgb.iter_mut()) {
                *entry = lattice.interpolate(&[s], &mut axes);
            }
        }
        if rgb
            .iter()
            .any(|[r, g, b]| r.abs_diff(*g) > 1 || b.abs_diff(*g) > 1)
        {
            return Some(Lut::Rgb(rgb));
        }
        let mut gray = Box::new([0u8; 256]);
        for (out, [_, g, _]) in gray.iter_mut().zip(rgb.iter()) {
            *out = *g;
        }
        Some(if (0u8..=255).zip(gray.iter()).all(|(s, &v)| s == v) {
            Lut::Identity
        } else {
            Lut::Gray(gray)
        })
    }

    /// One-component samples through the table; `None` when the table
    /// cannot be made (see [`Self::lut`]) or the widened output cannot be
    /// allocated.
    pub(super) fn convert_one(&self, mut bytes: Vec<u8>) -> Option<(ImageColorSpace, Vec<u8>)> {
        Some(match self.lut()? {
            Lut::Identity => (ImageColorSpace::Gray, bytes),
            Lut::Gray(table) => {
                for b in &mut bytes {
                    *b = table[usize::from(*b)];
                }
                (ImageColorSpace::Gray, bytes)
            }
            Lut::Rgb(table) => {
                let mut rgb = try_with_capacity(bytes.len() * 3)?;
                for &b in &bytes {
                    rgb.extend_from_slice(&table[usize::from(b)]);
                }
                (ImageColorSpace::Rgb, rgb)
            }
        })
    }

    /// The conversion of the space's three-component samples; the
    /// identity for `DeviceRGB` and for an ICC profile that neither
    /// converts nor has a CIE-based alternate.
    pub(super) fn rgb_convert(&self) -> RgbConvert {
        if let Some(profile) = &self.profile {
            return RgbConvert::of_profile(profile).unwrap_or(RgbConvert::Identity);
        }
        match self.space {
            ColorSpace::Cal(cal) => RgbConvert::of_cal(cal),
            ColorSpace::IccBased {
                stream_id: Some(id),
                ..
            } => {
                let Some(conv) = IccConversion::load(self.doc, *id, 3, self.target) else {
                    return RgbConvert::Identity;
                };
                RgbConvert::of_profile(&conv).unwrap_or_else(|| match &*conv {
                    IccConversion::Alternate(ColorSpace::Cal(cal), _) => RgbConvert::of_cal(cal),
                    IccConversion::Alternate(ColorSpace::Lab(_), _) => RgbConvert::Memo,
                    _ => RgbConvert::Identity,
                })
            }
            ColorSpace::Lab(_) | ColorSpace::DeviceN { .. } => RgbConvert::Memo,
            _ => RgbConvert::Identity,
        }
    }

    /// Three-component samples: `/Decode` as a remap, then the conversion.
    /// `None` when the output cannot be allocated.
    pub(super) fn convert_rgb(&self, bytes: Vec<u8>) -> Option<(ImageColorSpace, Vec<u8>)> {
        let convert = self.rgb_convert();
        if matches!(convert, RgbConvert::Memo) {
            return Some((ImageColorSpace::Rgb, self.convert_memo(&bytes, 3)?));
        }
        let mut bytes = self.remap(bytes);
        match convert {
            RgbConvert::Identity | RgbConvert::Memo => {}
            RgbConvert::Shaper(shaper) => shaper.convert_in_place(&mut bytes),
            RgbConvert::Transform(conv) => {
                let mut out = try_filled(bytes.len(), 0)?;
                if conv.transform_pixels(&bytes, &mut out).is_some() {
                    bytes = out;
                }
            }
        }
        Some((ImageColorSpace::Rgb, bytes))
    }

    /// The conversion of the space's four-component samples; the device
    /// model for `DeviceCMYK` and for an ICC profile without a transform.
    fn cmyk_convert(&self) -> CmykConvert {
        if let Some(profile) = &self.profile {
            return CmykConvert::of_profile(Arc::clone(profile));
        }
        match self.space {
            ColorSpace::IccBased {
                stream_id: Some(id),
                ..
            } => IccConversion::load(self.doc, *id, 4, self.target)
                .map_or(CmykConvert::Device, CmykConvert::of_profile),
            ColorSpace::DeviceN { .. } => CmykConvert::Memo,
            _ => CmykConvert::Device,
        }
    }

    /// Four-component samples: `/Decode` as a remap, then the conversion.
    /// `None` when the byte count is not a multiple of four.
    pub(super) fn convert_cmyk(
        &self,
        bytes: Vec<u8>,
        #[cfg(feature = "gpu-icc")] icc: &crate::resources::image::IccHandles<'_>,
    ) -> Option<(ImageColorSpace, Vec<u8>)> {
        let convert = self.cmyk_convert();
        if matches!(convert, CmykConvert::Memo) {
            return Some((ImageColorSpace::Rgb, self.convert_memo(&bytes, 4)?));
        }
        let bytes = self.remap(bytes);
        if !bytes.len().is_multiple_of(4) {
            log::warn!(
                "image: CMYK sample buffer length {} is not a multiple of 4",
                bytes.len()
            );
            return None;
        }
        #[cfg(feature = "gpu-icc")]
        if let CmykConvert::Icc(conv) = &convert
            && let Some(rgb) =
                crate::resources::image::icc_device::cmyk_clut_to_rgb(icc, &bytes, conv)
        {
            return Some((ImageColorSpace::Rgb, rgb));
        }
        let mut rgb = try_filled(bytes.len() / 4 * 3, 0)?;
        let converted = match &convert {
            CmykConvert::Icc(conv) => conv.transform_pixels(&bytes, &mut rgb).is_some(),
            CmykConvert::Device | CmykConvert::Memo => false,
        };
        if !converted {
            color::cmyk::cmyk_to_rgb_row(&bytes, &mut rgb);
        }
        Some((ImageColorSpace::Rgb, rgb))
    }

    /// `/Decode` applied to samples whose default decode is `[0, 1]`:
    /// each sample `s` becomes `round(dmin · 255 + s · (dmax − dmin))`,
    /// clamped to the byte range.
    fn remap(&self, mut bytes: Vec<u8>) -> Vec<u8> {
        if self.identity_decode() {
            return bytes;
        }
        let luts = decode_luts(self.decode);
        for px in bytes.chunks_mut(luts.len()) {
            for (s, lut) in px.iter_mut().zip(&luts) {
                *s = lut[usize::from(*s)];
            }
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use pdf::{Document, Object};

    use crate::resources::colorspace::resolve;
    use crate::resources::image::ImageColorSpace;
    use crate::resources::image::colorspace::Layout;
    use crate::resources::image::colorspace::test_support::{
        cal, convert, decode_arr, devicen_teal, gamma_1_8_doc, icc_cs, name_obj, teal_rgb,
    };
    use crate::test_helpers::{empty_doc, make_doc_with_objects, make_doc_with_stream};

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
    fn separation_black_maps_zero_tint_to_white() {
        // PDF §8.6.6.4: tint 0 is no colorant.  Through t ↦ (0,0,0,t) and the
        // CMYK alternate that is paper white; tint 1 is solid K.  This is the
        // reverse of the Gray sample polarity the bytes are decoded in.
        let (doc, cs) = separation_black_doc();
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 128, 255]);
        assert_eq!(
            img.color_space,
            ImageColorSpace::Gray,
            "neutral entries keep one byte"
        );
        let px = img.data.as_cpu().unwrap();
        let solid_k = color::cmyk::CORNERS[8][0];
        assert_eq!(px[0], 255, "tint 0 = no ink = white");
        assert_eq!(px[2], solid_k, "tint 1 = solid K");
        assert_eq!(
            px[1],
            color::cmyk::cmyk_to_rgb(0, 0, 0, 128)[0],
            "half tint is the half-K grey"
        );
    }

    #[test]
    fn indexed_over_separation_uses_the_base_tint() {
        let (doc, sep) = separation_black_doc();
        let cs = Object::Array(vec![
            name_obj(b"Indexed"),
            sep,
            Object::Integer(1),
            Object::String(vec![0x00, 0xFF], pdf::StringFormat::Hexadecimal),
        ]);
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 1]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(
            img.data.as_cpu().unwrap(),
            &[255, color::cmyk::CORNERS[8][0]]
        );
    }

    #[test]
    fn coloured_tint_widens_to_three_bytes_per_pixel() {
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
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 255]);
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 255, 255, 255, 0, 0]);
    }

    #[test]
    fn gray_decode_is_folded_into_the_table() {
        // §8.9.5.2: output = dmin·255 + s·(dmax − dmin) = 255 − s under [1 0].
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceGray"),
            &[("Decode", decode_arr(&[1, 0]))],
            Layout::Color(1),
            &[0, 255, 64],
        );
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 0, 191]);
        let identity = convert(
            &doc,
            name_obj(b"DeviceGray"),
            &[("Decode", decode_arr(&[0, 1]))],
            Layout::Color(1),
            &[128],
        );
        assert_eq!(identity.data.as_cpu().unwrap(), &[128]);
    }

    #[test]
    fn separation_decode_applies_before_the_tint_transform() {
        // §8.9.5.2: Decode maps samples into the tint domain first, so
        // [1 0] turns sample 0 into full colorant.
        let (doc, cs) = separation_black_doc();
        let img = convert(
            &doc,
            cs,
            &[("Decode", decode_arr(&[1, 0]))],
            Layout::Color(1),
            &[0, 255],
        );
        assert_eq!(
            img.data.as_cpu().unwrap(),
            &[color::cmyk::CORNERS[8][0], 255]
        );
    }

    #[test]
    fn rgb_decode_remaps_each_component() {
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceRGB"),
            &[("Decode", decode_arr(&[1, 0, 0, 1, 1, 0]))],
            Layout::Color(3),
            &[0, 0, 0, 255, 255, 255],
        );
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 0, 255, 0, 255, 0]);
    }

    #[test]
    fn device_cmyk_converts_through_the_process_ink_model() {
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceCMYK"),
            &[],
            Layout::Color(4),
            &[0, 0, 0, 0, 255, 0, 0, 0],
        );
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(&px[..3], &[255, 255, 255]);
        assert_eq!(&px[3..], &color::cmyk::CORNERS[4]);
    }

    #[test]
    fn cal_images_convert_as_the_fill_path_does() {
        let doc = empty_doc();
        let rgb_space = resolve(&doc, &cal("CalRGB", "[1.8 1.8 1.8]"));
        let want = rgb_space.convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[128.0 / 255.0; 3]);
        let img = convert(
            &doc,
            cal("CalRGB", "[1.8 1.8 1.8]"),
            &[],
            Layout::Color(3),
            &[128, 128, 128],
        );
        let got = img.data.as_cpu().unwrap();
        assert!(
            got.iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1),
            "got {got:?}, fill path {want:?}"
        );
        let gray_space = resolve(&doc, &cal("CalGray", "1.8"));
        let want = gray_space.convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[128.0 / 255.0]);
        let img = convert(&doc, cal("CalGray", "1.8"), &[], Layout::Color(1), &[128]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &want[..1]);
    }

    #[test]
    fn indexed_looks_up_the_palette_and_clamps_to_hival() {
        let doc = empty_doc();
        let cs = pdf::parse_object(b"[/Indexed /DeviceRGB 1 <ff000000ff00>]", &mut 0).unwrap();
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 1, 7]);
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(
            img.data.as_cpu().unwrap(),
            &[255, 0, 0, 0, 255, 0, 0, 255, 0]
        );
    }

    #[test]
    fn indexed_over_cmyk_converts_the_palette() {
        let doc = empty_doc();
        let cs = pdf::parse_object(b"[/Indexed /DeviceCMYK 0 <ff000000>]", &mut 0).unwrap();
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0]);
        assert_eq!(img.data.as_cpu().unwrap(), &color::cmyk::CORNERS[4]);
    }

    #[test]
    fn indexed_short_palette_is_black() {
        let doc = empty_doc();
        let cs = pdf::parse_object(b"[/Indexed /DeviceRGB 3 <ff0000>]", &mut 0).unwrap();
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 1]);
        assert_eq!(img.data.as_cpu().unwrap(), &[255, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn indexed_over_multi_colorant_devicen_expands_the_palette() {
        let (doc, devicen) = devicen_teal();
        let cs = Object::Array(vec![
            name_obj(b"Indexed"),
            devicen,
            Object::Integer(1),
            // Entry 0: full spot; entry 1: no ink.
            Object::String(vec![0, 255, 0, 0], pdf::StringFormat::Hexadecimal),
        ]);
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0, 1]);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(px[0..3], teal_rgb());
        assert_eq!(px[3..6], [255, 255, 255]);
    }

    #[test]
    fn icc_rgb_image_samples_go_through_the_profile() {
        let doc = gamma_1_8_doc(3);
        let img = convert(
            &doc,
            icc_cs(),
            &[],
            Layout::Color(3),
            &[0, 128, 255, 150, 118, 80],
        );
        // encode((v/255)^1.8): (0, 128, 255) → (0, 146, 255);
        // (150, 118, 80) → (167, 137, 99).
        assert_eq!(img.data.as_cpu().unwrap(), &[0, 146, 255, 167, 137, 99]);
    }

    #[test]
    fn icc_gray_image_samples_go_through_the_profile() {
        let doc = gamma_1_8_doc(1);
        let img = convert(&doc, icc_cs(), &[], Layout::Color(1), &[0, 128, 255]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
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
        let img = convert(&doc, cs, &[], Layout::Color(1), &[0]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &[146]);
    }

    #[test]
    fn srgb_profile_is_the_identity() {
        let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
        let doc = make_doc_with_stream(&srgb, " /N 3");
        let img = convert(&doc, icc_cs(), &[], Layout::Color(3), &[10, 20, 30]);
        assert_eq!(img.data.as_cpu().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn unusable_profile_converts_through_its_alternate() {
        let doc = make_doc_with_stream(
            b"not a profile",
            " /N 3 /Alternate [/Lab << /WhitePoint [0.9505 1 1.089] >>]",
        );
        // (128, 128, 128) in the default Lab range is L* 50.2, a* = b* ≈ 0.
        let img = convert(&doc, icc_cs(), &[], Layout::Color(3), &[128, 128, 128]);
        let px = img.data.as_cpu().unwrap();
        assert!(px.iter().all(|&c| c.abs_diff(119) <= 2), "got {px:?}");
        let doc = make_doc_with_stream(
            b"not a profile",
            " /N 1 /Alternate [/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 1.8 >>]",
        );
        let img = convert(&doc, icc_cs(), &[], Layout::Color(1), &[128]);
        let want = resolve(&doc, &cal("CalGray", "1.8")).convert_to_rgb(
            &doc,
            crate::ColourTarget::SRGB,
            &[128.0 / 255.0],
        );
        assert_eq!(img.data.as_cpu().unwrap(), &want[..1]);
    }

    /// A `lut8Type` (`mft1`) RGB profile with a Lab connection space whose
    /// 17³ table holds the Lab of each grid colour decoded with gamma 1.8
    /// on sRGB primaries.
    fn lut_rgb_profile() -> Vec<u8> {
        const GRID: u8 = 17;
        /// `s15Fixed16Number` of `v`.
        fn s15f16(v: f64) -> [u8; 4] {
            #[expect(clippy::cast_possible_truncation, reason = "|v| < 2; fits s15Fixed16")]
            let fixed = (v * 65536.0).round() as i32;
            fixed.to_be_bytes()
        }
        /// `v` rounded to the nearest 8-bit level.
        fn level(v: f64) -> u8 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to [0, 255] before the cast"
            )]
            let out = v.round().clamp(0.0, 255.0) as u8;
            out
        }
        fn lab_bytes(rgb: [f64; 3]) -> [u8; 3] {
            // sRGB linear → XYZ (D50 via the ICC colorants) → Lab, 8-bit ICC encoding.
            let colorants: [[f64; 3]; 3] = [
                [0.436_065_7, 0.385_147_1, 0.143_066_4],
                [0.222_488_4, 0.716_873_2, 0.060_607_9],
                [0.013_916_0, 0.097_076_4, 0.714_096_1],
            ];
            let xyz = colorants
                .map(|row| row[2].mul_add(rgb[2], row[0].mul_add(rgb[0], row[1] * rgb[1])));
            let white = [0.9642, 1.0, 0.8249];
            let cie_f = |t: f64| {
                if t > 216.0 / 24389.0 {
                    t.cbrt()
                } else {
                    (841.0 / 108.0f64).mul_add(t, 4.0 / 29.0)
                }
            };
            let [fx, fy, fz] = [
                cie_f(xyz[0] / white[0]),
                cie_f(xyz[1] / white[1]),
                cie_f(xyz[2] / white[2]),
            ];
            let l_star = 116.0f64.mul_add(fy, -16.0);
            let a_star = 500.0 * (fx - fy);
            let b_star = 200.0 * (fy - fz);
            [
                level(l_star * 255.0 / 100.0),
                level(a_star + 128.0),
                level(b_star + 128.0),
            ]
        }
        let mut tag = b"mft1\0\0\0\0".to_vec();
        tag.extend_from_slice(&[3, 3, GRID, 0]);
        for v in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0f64] {
            tag.extend_from_slice(&s15f16(v));
        }
        // Input and output curves: identity.
        for _ in 0..3 {
            tag.extend(0..=255u8);
        }
        let linear = |i: u8| (f64::from(i) / f64::from(GRID - 1)).powf(1.8);
        for r in 0..GRID {
            for g in 0..GRID {
                for b in 0..GRID {
                    tag.extend_from_slice(&lab_bytes([linear(r), linear(g), linear(b)]));
                }
            }
        }
        for _ in 0..3 {
            tag.extend(0..=255u8);
        }
        let mut wtpt = b"XYZ \0\0\0\0".to_vec();
        for v in [0.9642, 1.0, 0.8249f64] {
            wtpt.extend_from_slice(&s15f16(v));
        }
        let tags: [(&[u8; 4], Vec<u8>); 2] = [(b"A2B0", tag), (b"wtpt", wtpt)];
        let table_len = 4 + 12 * tags.len();
        let mut table = u32::try_from(tags.len()).unwrap().to_be_bytes().to_vec();
        let mut data = Vec::new();
        for (sig, body) in &tags {
            let offset = 128 + table_len + data.len();
            table.extend_from_slice(*sig);
            table.extend_from_slice(&u32::try_from(offset).unwrap().to_be_bytes());
            table.extend_from_slice(&u32::try_from(body.len()).unwrap().to_be_bytes());
            data.extend_from_slice(body);
            while data.len() % 4 != 0 {
                data.push(0);
            }
        }
        let mut header = vec![0u8; 128];
        header[..4].copy_from_slice(
            &u32::try_from(128 + table.len() + data.len())
                .unwrap()
                .to_be_bytes(),
        );
        header[8..12].copy_from_slice(&0x0210_0000u32.to_be_bytes());
        header[12..16].copy_from_slice(b"mntr");
        header[16..20].copy_from_slice(b"RGB ");
        header[20..24].copy_from_slice(b"Lab ");
        header[36..40].copy_from_slice(b"acsp");
        for (i, v) in [0.9642, 1.0, 0.8249f64].into_iter().enumerate() {
            header[68 + 4 * i..72 + 4 * i].copy_from_slice(&s15f16(v));
        }
        [header, table, data].concat()
    }

    #[test]
    fn lut_based_profile_converts_through_moxcms() {
        let doc = make_doc_with_stream(&lut_rgb_profile(), " /N 3");
        let space = resolve(&doc, &icc_cs());
        let want = space.convert_to_rgb(&doc, crate::ColourTarget::SRGB, &[128.0 / 255.0; 3]);
        assert!(
            want.iter().all(|&c| c.abs_diff(146) <= 3),
            "the fill path lightens mid-gray through the table, got {want:?}"
        );
        let img = convert(
            &doc,
            icc_cs(),
            &[],
            Layout::Color(3),
            &[128, 128, 128, 255, 255, 255],
        );
        let px = img.data.as_cpu().unwrap();
        assert!(
            px[..3].iter().zip(want).all(|(&g, w)| g.abs_diff(w) <= 1),
            "image {px:?} vs fill {want:?}"
        );
        assert!(
            px[3..].iter().all(|&c| c >= 253),
            "white stays white: {px:?}"
        );
    }
}
