//! Image colour: the colour space of an image's samples, resolved once per
//! image, and the conversion of a codec's samples into the layouts the
//! blit path draws.
//!
//! A codec yields [`Samples`]: stencil bytes for an image mask, otherwise
//! `n` interleaved 8-bit components per pixel in the image's colour space,
//! before `/Decode`.  [`ImageColour::convert`] turns them into an
//! [`ImageDescriptor`] of [`ImageColorSpace::Gray`] or
//! [`ImageColorSpace::Rgb`] samples through the same [`ColorSpace`]
//! conversions the fill operators use, so an image in any colour space the
//! fill path converts is drawn the same way:
//!
//! - one component per pixel: a 256-entry table of the space's conversion
//!   of every sample, with `/Decode` folded in, kept one byte per pixel
//!   when every entry is neutral;
//! - three components: `/Decode` as an 8-bit remap, then the space's
//!   conversion: a shaper for `CalRGB` and matrix/TRC ICC profiles, a
//!   `moxcms` transform for other ICC profiles, and a memoised per-pixel
//!   conversion for Lab and `DeviceN`;
//! - four components: `/Decode` as an 8-bit remap, then the device CMYK
//!   model, an ICC transform, or the memoised conversion for `DeviceN`;
//! - any other count (`DeviceN`): the memoised conversion.
//!
//! The samples of an `Indexed` image are palette indices and convert
//! through the palette as one-component samples.  A device space takes the
//! `/DefaultGray`, `/DefaultRGB` or `/DefaultCMYK` entry of the colour-space
//! resources in effect (§8.6.5.6).  When the dictionary names no colour
//! space, an ICC profile the codestream carries (JPEG 2000 `colr` box)
//! converts the samples as an `ICCBased` space of the same channel count
//! would; otherwise the channel count selects the device space.

mod budget;
mod decode_array;
mod decoded;
mod memo;
mod palette;
mod samples;
#[cfg(test)]
mod test_support;
mod wide;

use color::Sample;
use pdf::{Dictionary, Document, Object};

pub(crate) use budget::TintBudget;
use decode_array::{decode_luts, decode_wide, pairs_equal, parse_decode};
use decoded::Decoded;
#[cfg(feature = "cache")]
use decoded::{Lut, RgbConvert};
pub(super) use samples::{KeySamples, Layout, Samples};
use wide::convert16;

use super::{
    ImageColorSpace, ImageData, ImageDescriptor, ImageFilter, ImageTints, SoftMask, TintData,
};
use crate::resources::colorspace::icc::IccConversion;
use crate::resources::colorspace::{ColorSpace, default_space, resolve};
use crate::resources::dict_ext::DictExt as _;
use crate::{ColourTarget, WorkingSpace};

/// Whether the colours of `space` are amounts of ink an ink canvas
/// paints: colorant tints in a `Separation` or `DeviceN` space, process
/// inks in `DeviceCMYK` or a four-component ICC space.
const fn is_ink(space: &ColorSpace) -> bool {
    matches!(
        space,
        ColorSpace::Separation { .. }
            | ColorSpace::DeviceN { .. }
            | ColorSpace::DeviceCmyk
            | ColorSpace::IccBased { n: 4, .. }
    )
}

/// Largest `BitsPerComponent` of an `Indexed` image (PDF §8.6.6.3).
const MAX_INDEX_BITS: u32 = 8;

/// The colour of an image's samples, resolved once from its dictionary and
/// the colour-space resources in effect.
#[derive(Debug)]
pub(super) struct ImageColour {
    /// The samples' colour space; `None` when the dictionary names none, in
    /// which case the codec's channel count selects the device space.
    space: Option<ColorSpace>,
    /// The `/Decode` pairs, one per component of `space`: the dictionary's
    /// array when it is well formed, otherwise the space's default.
    decode: Vec<(f64, f64)>,
    /// Whether `decode` is the space's default.
    #[cfg_attr(
        not(any(test, feature = "cache")),
        expect(
            dead_code,
            reason = "read by `identity`, which gates the device image cache"
        )
    )]
    default_decode: bool,
    /// The largest value a sample takes: `2^bpc − 1` for the palette
    /// indices of an `Indexed` image, which are not scaled when expanded
    /// to a byte, and 255 for every other space.
    max_sample: f64,
}

impl ImageColour {
    /// Resolve the colour of the image `dict` describes; `color_spaces` is
    /// the `/ColorSpace` resource dictionary whose `/Default*` entries
    /// replace a device space.
    pub(super) fn resolve(
        doc: &Document,
        dict: &Dictionary,
        color_spaces: Option<&Dictionary>,
    ) -> Self {
        let space = dict.get(b"ColorSpace").map(|cs| {
            let space = resolve(doc, cs);
            color_spaces
                .and_then(|res| default_space(doc, res, &space))
                .unwrap_or(space)
        });
        let bpc = dict
            .get_i64(b"BitsPerComponent")
            .and_then(|b| u32::try_from(b).ok())
            .unwrap_or(8);
        let (default, max_sample) = match &space {
            Some(indexed @ ColorSpace::Indexed { .. }) => {
                let bits = bpc.clamp(1, MAX_INDEX_BITS);
                (indexed.default_decode(bits), f64::from((1u32 << bits) - 1))
            }
            Some(space) => (default_decode(doc, space), 255.0),
            None => (Vec::new(), 255.0),
        };
        let decode = parse_decode(dict, default.len()).filter(|d| !pairs_equal(d, &default));
        let default_decode = decode.is_none();
        Self {
            space,
            decode: decode.unwrap_or(default),
            default_decode,
            max_sample,
        }
    }

    /// The colour of samples in `space` under its default `/Decode`.
    #[cfg(test)]
    pub(super) fn of(space: ColorSpace) -> Self {
        let decode = space.default_decode(8);
        Self {
            space: Some(space),
            decode,
            default_decode: true,
            max_sample: 255.0,
        }
    }

    /// Components per pixel a raw (unfiltered) image stores: those of its
    /// colour space, one when the dictionary names none.
    pub(super) fn components(&self) -> usize {
        self.space.as_ref().map_or(1, ColorSpace::ncomponents)
    }

    /// Bytes per pixel the decode budget charges: the larger of the stored
    /// components and the converted bytes per pixel, an upper bound on the
    /// buffer [`convert`](Self::convert) produces.  One-component samples
    /// of a palette or a spot colour may widen to RGB (three bytes); every
    /// count of two or more converts to RGB; gray-family samples stay one
    /// byte.
    pub(super) fn budget_components(&self) -> usize {
        let n = self.components();
        let widens = n > 1
            || matches!(
                self.space,
                Some(
                    ColorSpace::Indexed { .. }
                        | ColorSpace::Separation { .. }
                        | ColorSpace::DeviceN { .. }
                )
            );
        if widens { n.max(3) } else { n }
    }

    /// Whether the samples are palette indices (`Indexed`).
    pub(super) const fn indexed(&self) -> bool {
        matches!(self.space, Some(ColorSpace::Indexed { .. }))
    }

    /// The samples' colour space, when the dictionary names one.
    pub(super) const fn space(&self) -> Option<&ColorSpace> {
        self.space.as_ref()
    }

    /// Whether painting the image produces no output: every colorant of
    /// its space is `/None` (PDF §8.6.6.4, §8.6.6.5).
    pub(super) fn paints_nothing(&self) -> bool {
        self.space.as_ref().is_some_and(ColorSpace::paints_nothing)
    }

    /// Whether the decoded samples are drawn as they are: a `DeviceGray`
    /// or `DeviceRGB` image under the default `/Decode`, or one whose
    /// colour space the codec's channel count decides.  Such an image may
    /// be kept device-resident.
    #[cfg_attr(
        not(any(test, feature = "cache")),
        expect(dead_code, reason = "gates the device image cache")
    )]
    pub(super) const fn identity(&self) -> bool {
        self.default_decode
            && matches!(
                self.space,
                None | Some(ColorSpace::DeviceGray | ColorSpace::DeviceRgb)
            )
    }

    /// The space `n`-component samples convert through: the image's own
    /// space when it has `n` components, otherwise the device space of `n`
    /// components, as for a producer whose codestream contradicts the
    /// dictionary.  `None` for a count no device space has.
    fn space_for(&self, n: usize) -> Option<(&ColorSpace, &[(f64, f64)])> {
        if let Some(space) = &self.space {
            if space.ncomponents() == n {
                return Some((space, &self.decode));
            }
            log::debug!(
                "image: {n}-component samples declared {space:?}; converting as the device space"
            );
        }
        let device: &'static ColorSpace = match n {
            1 => &ColorSpace::DeviceGray,
            3 => &ColorSpace::DeviceRgb,
            4 => &ColorSpace::DeviceCmyk,
            _ => return None,
        };
        Some((device, DEVICE_DECODE[..n].as_ref()))
    }

    /// Whether 16-bit samples of the image convert at 16 bits (see
    /// [`convert16`]): those of a device, CIE-based or ICC-based space, and
    /// of an image naming no space.  Samples in any other space convert at
    /// 8 bits at every depth.
    pub(super) fn converts_wide(&self) -> bool {
        self.space.as_ref().is_none_or(|space| {
            matches!(
                space,
                ColorSpace::DeviceGray
                    | ColorSpace::DeviceRgb
                    | ColorSpace::DeviceCmyk
                    | ColorSpace::Cal(_)
                    | ColorSpace::Lab(_)
                    | ColorSpace::IccBased { .. }
            )
        })
    }

    /// Whether the image's samples are amounts of ink an ink canvas can
    /// paint ([`Self::tints`]): those of a space of ink ([`is_ink`]).
    pub(super) const fn has_tints(&self) -> bool {
        match &self.space {
            Some(space) => is_ink(space),
            None => false,
        }
    }

    /// Convert `samples` to the gray or RGB descriptor the blit draws.
    ///
    /// A conversion through tint transforms spends from `budget`, the
    /// work left to the page's images, when there is one (see
    /// [`TintBudget`]).  `None` when the samples' layout cannot be converted
    /// (a component count no space describes), the converted samples cannot
    /// be allocated, or the work left cannot convert them even
    /// approximately.  Device-resident samples are drawn as they are; they
    /// are only produced for an [`identity`](Self::identity) colour.
    /// Colours convert to `target` (see [`ColorSpace::convert_to_rgb`]),
    /// for a canvas of samples of `depth`.
    pub(super) fn convert(
        &self,
        doc: &Document,
        (target, depth): (ColourTarget, color::SampleDepth),
        samples: Samples,
        budget: Option<&TintBudget>,
        #[cfg(feature = "gpu-icc")] icc: &super::IccHandles<'_>,
    ) -> Option<ImageDescriptor> {
        let Samples {
            width,
            height,
            layout,
            data,
            smask,
            embedded_profile,
            partial: _,
            key_source: _,
        } = samples;
        let n = match layout {
            Layout::Mask => {
                return Some(descriptor(
                    width,
                    height,
                    ImageColorSpace::Mask,
                    data,
                    smask,
                ));
            }
            Layout::Color(n) => usize::from(n),
        };
        let (space, decode) = self.space_for(n)?;
        let profile = embedded_profile
            .as_deref()
            .filter(|_| self.space.is_none())
            .and_then(|bytes| IccConversion::compile(bytes, u8::try_from(n).ok()?, target));
        let max_sample = if n == self.components() {
            self.max_sample
        } else {
            255.0
        };
        // On a 16-bit canvas a colour of three or four components that
        // converts converts at 16 bits, from its samples widened exactly, as
        // a fill of the same colour does; `DeviceRGB`, which converts as it
        // is in sRGB, keeps its 8-bit samples there.
        let device_rgb = matches!(space, ColorSpace::DeviceRgb) && profile.is_none();
        let at16 = matches!(n, 3 | 4)
            && self.converts_wide()
            && (target.space == WorkingSpace::Wide
                || (depth == color::SampleDepth::Sixteen && !device_rgb));
        let decoded = Decoded::new(doc, target, space, decode, max_sample, profile, budget);
        let bytes = match data {
            ImageData::Cpu(bytes) if at16 => {
                let (cs, wide) = decoded.convert_bytes16(
                    n,
                    &bytes,
                    #[cfg(feature = "gpu-icc")]
                    icc,
                )?;
                return Some(descriptor(width, height, cs, ImageData::Cpu16(wide), smask));
            }
            ImageData::Cpu(bytes) => bytes,
            ImageData::Cpu16(wide) => {
                let (cs, wide) = convert16(&decoded, n, wide)?;
                return Some(descriptor(width, height, cs, ImageData::Cpu16(wide), smask));
            }
            #[cfg(feature = "cache")]
            ImageData::Gpu(cached) => {
                let identity = match n {
                    1 => matches!(decoded.lut(), Some(Lut::Identity)),
                    3 => {
                        decoded.identity_decode()
                            && matches!(decoded.rgb_convert(), RgbConvert::Identity)
                    }
                    _ => false,
                };
                debug_assert!(
                    identity,
                    "device-resident samples are only cached for an identity colour"
                );
                if !identity {
                    log::warn!(
                        "image: device-resident samples in {space:?} drawn without colour conversion"
                    );
                }
                let cs = if n == 1 {
                    ImageColorSpace::Gray
                } else {
                    ImageColorSpace::Rgb
                };
                return Some(descriptor(width, height, cs, ImageData::Gpu(cached), smask));
            }
        };
        let converted = match n {
            1 => decoded.convert_one(bytes),
            3 => decoded.convert_rgb(bytes),
            4 => decoded.convert_cmyk(
                bytes,
                #[cfg(feature = "gpu-icc")]
                icc,
            ),
            _ => decoded
                .convert_memo(&bytes, n)
                .map(|rgb| (ImageColorSpace::Rgb, rgb)),
        };
        if let Some(budget) = budget {
            budget.spend(decoded.spent.get());
        }
        let (cs, bytes) = converted?;
        Some(descriptor(width, height, cs, ImageData::Cpu(bytes), smask))
    }
}

impl ImageColour {
    /// The ink amounts of `samples` in the image's `Separation`, `DeviceN`
    /// or process CMYK space ([`Self::has_tints`]): each component mapped
    /// through its `/Decode` pair to a sample of the samples' depth, 0 no
    /// ink and the maximum the full tint; for an `Indexed` image over a
    /// space of ink, the tints of its palette entries
    /// ([`Self::palette_tints`]).  `None` for any other space, for samples
    /// whose channel count is not the space's, and for samples that are
    /// not host-resident.
    pub(super) fn tints(&self, samples: &Samples) -> Option<ImageTints> {
        if self.indexed() {
            return self.palette_tints(samples);
        }
        let space = self.space.as_ref().filter(|_| self.has_tints())?;
        let n = space.ncomponents();
        if samples.layout != Layout::Color(u8::try_from(n).ok()?) || self.decode.len() != n {
            return None;
        }
        if let Some(wide) = samples.data.as_cpu16() {
            let mut data = Vec::new();
            data.try_reserve_exact(wide.len()).ok()?;
            data.extend(wide.chunks_exact(n).flat_map(|px| {
                px.iter()
                    .zip(&self.decode)
                    .map(|(&s, &pair)| u16::from_unit(decode_wide(s, pair)))
            }));
            return Some(ImageTints {
                space: space.clone(),
                data: TintData::U16(data),
            });
        }
        let bytes = samples.data.as_cpu()?;
        let luts = decode_luts(&self.decode);
        if luts.len() != n {
            return None;
        }
        let mut data = Vec::new();
        data.try_reserve_exact(bytes.len()).ok()?;
        data.extend(
            bytes
                .chunks_exact(n)
                .flat_map(|px| px.iter().zip(&luts).map(|(&s, lut)| lut[usize::from(s)])),
        );
        Some(ImageTints {
            space: space.clone(),
            data: TintData::U8(data),
        })
    }
}

/// `(0, 1)` for each of up to four components: the default `/Decode` of a
/// device space.
const DEVICE_DECODE: [(f64, f64); 4] = [(0.0, 1.0); 4];

/// The default `/Decode` of an 8-bit image in `space` (PDF §8.9.5.2,
/// Table 90).  An `ICCBased` space takes the `/Range` of its profile
/// stream, or, when it converts through its `/Alternate` space, that
/// space's default; every other space takes
/// [`ColorSpace::default_decode`].
fn default_decode(doc: &Document, space: &ColorSpace) -> Vec<(f64, f64)> {
    let ColorSpace::IccBased {
        n,
        stream_id: Some(id),
    } = space
    else {
        return space.default_decode(8);
    };
    let range = doc.get_object(*id).ok().and_then(|obj| {
        let dict = &obj.as_ref().as_stream()?.dict;
        let Object::Array(arr) = dict.get(b"Range")? else {
            return None;
        };
        let vals: Vec<f64> = arr.iter().filter_map(Object::as_f64).collect();
        (vals.len() == usize::from(*n) * 2 && vals.iter().all(|v| v.is_finite()))
            .then(|| vals.as_chunks::<2>().0.iter().map(|&p| p.into()).collect())
    });
    // The `/Alternate` space stands in for a profile that converts under no
    // intent; the initial intent's conversion answers for every intent.
    range.unwrap_or_else(|| {
        match IccConversion::load(doc, *id, *n, ColourTarget::SRGB).as_deref() {
            Some(IccConversion::Alternate(alt, _)) => alt.default_decode(8),
            _ => space.default_decode(8),
        }
    })
}

const fn descriptor(
    width: u32,
    height: u32,
    color_space: ImageColorSpace,
    data: ImageData,
    smask: Option<SoftMask>,
) -> ImageDescriptor {
    ImageDescriptor {
        width,
        height,
        color_space,
        data,
        smask,
        filter: ImageFilter::Raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::colorspace::test_support::{
        cal, convert, gamma_1_8_doc, icc_cs, name_obj,
    };
    use crate::test_helpers::empty_doc;

    #[test]
    fn device_gray_is_the_identity() {
        let doc = empty_doc();
        let colour = ImageColour::of(ColorSpace::DeviceGray);
        assert!(colour.identity());
        let img = convert(
            &doc,
            name_obj(b"DeviceGray"),
            &[],
            Layout::Color(1),
            &[0, 128, 255],
        );
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &[0, 128, 255]);
    }

    #[test]
    fn non_finite_decode_falls_back_to_the_default() {
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceGray"),
            &[(
                "Decode",
                Object::Array(vec![Object::Real(f32::NAN), Object::Real(1.0)]),
            )],
            Layout::Color(1),
            &[128],
        );
        assert_eq!(img.data.as_cpu().unwrap(), &[128]);
    }

    #[test]
    fn samples_contradicting_the_space_convert_as_the_device_space() {
        // A three-channel codestream declared DeviceGray draws as RGB.
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceGray"),
            &[],
            Layout::Color(3),
            &[10, 20, 30],
        );
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        assert_eq!(img.data.as_cpu().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn unspecified_space_takes_the_layout() {
        let doc = empty_doc();
        let colour = ImageColour::resolve(&doc, &Dictionary::new(), None);
        assert!(colour.identity());
        assert_eq!(colour.components(), 1);
        let img = colour
            .convert(
                &doc,
                (ColourTarget::SRGB, color::SampleDepth::Eight),
                Samples::cpu(1, 1, Layout::Color(3), vec![1, 2, 3]),
                None,
                #[cfg(feature = "gpu-icc")]
                &super::super::IccHandles::none(),
            )
            .unwrap();
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
    }

    #[test]
    fn stencil_samples_pass_through() {
        let doc = empty_doc();
        let img = convert(
            &doc,
            name_obj(b"DeviceRGB"),
            &[],
            Layout::Mask,
            &[0x00, 0xFF],
        );
        assert_eq!(img.color_space, ImageColorSpace::Mask);
        assert_eq!(img.data.as_cpu().unwrap(), &[0x00, 0xFF]);
    }

    #[test]
    fn device_space_takes_the_default_entry() {
        let doc = empty_doc();
        let mut resources = Dictionary::new();
        resources.set("DefaultGray", cal("CalGray", "1.8"));
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", name_obj(b"DeviceGray"));
        let colour = ImageColour::resolve(&doc, &dict, Some(&resources));
        assert!(!colour.identity());
        assert!(matches!(colour.space(), Some(ColorSpace::Cal(_))));
        // Without the resources the device space stays.
        assert!(ImageColour::resolve(&doc, &dict, None).identity());
    }

    #[test]
    fn none_colorants_paint_nothing() {
        let doc = empty_doc();
        let fn_obj = "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>";
        for src in [
            format!("[/Separation /None /DeviceGray {fn_obj}]"),
            format!("[/DeviceN [/None /None] /DeviceGray {fn_obj}]"),
        ] {
            let cs = pdf::parse_object(src.as_bytes(), &mut 0).unwrap();
            let mut dict = Dictionary::new();
            dict.set("ColorSpace", cs);
            assert!(
                ImageColour::resolve(&doc, &dict, None).paints_nothing(),
                "{src}"
            );
        }
        let cs = pdf::parse_object(
            format!("[/Separation /Spot /DeviceGray {fn_obj}]").as_bytes(),
            &mut 0,
        )
        .unwrap();
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", cs);
        assert!(!ImageColour::resolve(&doc, &dict, None).paints_nothing());
    }

    #[test]
    fn samples_contradicting_the_profile_convert_as_the_device_space() {
        let doc = gamma_1_8_doc(3);
        let img = convert(&doc, icc_cs(), &[], Layout::Color(1), &[128, 128]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &[128, 128]);
    }
}
