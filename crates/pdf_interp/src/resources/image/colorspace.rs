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

use std::collections::HashMap;
use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use super::{ImageColorSpace, ImageData, ImageDescriptor, ImageFilter, SoftMask};
use crate::resources::colorspace::icc::IccConversion;
use crate::resources::colorspace::{ColorSpace, decoded_sample, default_space, resolve};
use crate::resources::dict_ext::DictExt as _;
use crate::resources::icc_shaper::{ProfileConversion, RgbShaper};

#[cfg(feature = "gpu-icc")]
use crate::resources::icc_shaper::fetch_profile_bytes;

// ── Codec output ──────────────────────────────────────────────────────────────

/// The sample layout a codec yields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// Stencil bytes: `0x00` paints, `0xFF` is transparent.
    Mask,
    /// `n` interleaved 8-bit components per pixel in the image's colour
    /// space, before `/Decode`.
    Color(u8),
}

/// A codec's output: an image's samples before colour conversion.
#[derive(Debug)]
pub(super) struct Samples {
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    pub data: ImageData,
    /// Alpha the codestream itself carries (JPEG 2000 under `/SMaskInData`),
    /// on the image's own grid.
    pub smask: Option<SoftMask>,
    /// The ICC profile of the codestream's own colour specification, for
    /// an image dictionary that names no colour space.
    pub embedded_profile: Option<Vec<u8>>,
    /// What the stream lacked when the samples are only partly its own:
    /// a stream shorter than its dictionary declares, or a codec that
    /// stopped short, leaves the missing samples padded.
    pub partial: Option<String>,
}

impl Samples {
    /// Host-resident samples of `layout`.
    pub(super) const fn cpu(width: u32, height: u32, layout: Layout, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            layout,
            data: ImageData::Cpu(data),
            smask: None,
            embedded_profile: None,
            partial: None,
        }
    }

    /// Stencil bytes.
    pub(super) const fn mask(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self::cpu(width, height, Layout::Mask, data)
    }

    /// One component per pixel.
    pub(super) const fn gray(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self::cpu(width, height, Layout::Color(1), data)
    }
}

// ── Resolution ────────────────────────────────────────────────────────────────

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

    /// Convert `samples` to the gray or RGB descriptor the blit draws.
    ///
    /// `None` when the samples' layout cannot be converted (a component
    /// count no space describes) or the converted samples cannot be
    /// allocated.  Device-resident samples are drawn as they are; they are
    /// only produced for an [`identity`](Self::identity) colour.
    pub(super) fn convert(
        &self,
        doc: &Document,
        samples: Samples,
        #[cfg(feature = "gpu-icc")] icc: &mut super::IccHandles<'_>,
    ) -> Option<ImageDescriptor> {
        let Samples {
            width,
            height,
            layout,
            data,
            smask,
            embedded_profile,
            partial: _,
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
            .and_then(|bytes| {
                let conv = IccConversion::compile(bytes, u8::try_from(n).ok()?)?;
                Some(EmbeddedProfile {
                    conv: Arc::new(conv),
                    bytes,
                })
            });
        let decoded = Decoded {
            doc,
            space,
            decode,
            max_sample: if n == self.components() {
                self.max_sample
            } else {
                255.0
            },
            profile,
        };
        #[cfg_attr(
            not(feature = "cache"),
            expect(
                clippy::infallible_destructuring_match,
                reason = "the cache-feature build adds a device-resident arm; keep one shape"
            )
        )]
        let bytes = match data {
            ImageData::Cpu(bytes) => bytes,
            #[cfg(feature = "cache")]
            ImageData::Gpu(cached) => {
                let identity = match n {
                    1 => matches!(decoded.lut(), Lut::Identity),
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
        let (cs, bytes) = match n {
            1 => decoded.convert_one(bytes)?,
            3 => decoded.convert_rgb(bytes)?,
            4 => decoded.convert_cmyk(
                bytes,
                #[cfg(feature = "gpu-icc")]
                icc,
            )?,
            _ => (ImageColorSpace::Rgb, decoded.convert_memo(&bytes, n)?),
        };
        Some(descriptor(width, height, cs, ImageData::Cpu(bytes), smask))
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
    range.unwrap_or_else(|| match IccConversion::load(doc, *id, *n).as_deref() {
        Some(IccConversion::Alternate(alt)) => alt.default_decode(8),
        _ => space.default_decode(8),
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

// ── Conversion ────────────────────────────────────────────────────────────────

/// Distinct sample tuples memoised before the table is reset; bounds the
/// memory of a continuous-tone image's per-pixel conversion.
const MEMO_LIMIT: usize = 1 << 16;

/// The conversion of samples in one space under one `/Decode`.
struct Decoded<'a> {
    doc: &'a Document,
    space: &'a ColorSpace,
    decode: &'a [(f64, f64)],
    max_sample: f64,
    /// The codestream's own profile, which converts in place of `space`
    /// (then the device space of the same channel count) when present.
    profile: Option<EmbeddedProfile<'a>>,
}

/// An ICC profile compiled from a codestream's colour specification.
struct EmbeddedProfile<'a> {
    conv: Arc<IccConversion>,
    /// The profile as stored, for the device CLUT bake.
    #[cfg_attr(
        not(feature = "gpu-icc"),
        expect(dead_code, reason = "read by the gpu-icc CLUT bake")
    )]
    bytes: &'a [u8],
}

/// The colour of every value of a one-component sample.
enum Lut {
    /// Every value maps to the neutral gray of the same level.
    Identity,
    /// Every value maps to a neutral gray: one byte per pixel.
    Gray(Box<[u8; 256]>),
    Rgb(Box<[[u8; 3]; 256]>),
}

/// The conversion of three-component samples after `/Decode`.
enum RgbConvert {
    Identity,
    /// `CalRGB`, or a matrix/TRC ICC profile.
    Shaper(Arc<RgbShaper>),
    /// Any other ICC profile.
    Transform(Arc<IccConversion>),
    /// Per-pixel conversion through the space, with `/Decode` folded in.
    Memo,
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

impl Decoded<'_> {
    /// Whether `/Decode` leaves every sample unchanged.
    fn identity_decode(&self) -> bool {
        self.decode.iter().all(|&(d0, d1)| is_identity_pair(d0, d1))
    }

    /// The component values of the sample tuple `px`.
    fn components(&self, px: &[u8]) -> Vec<f64> {
        px.iter()
            .zip(self.decode)
            .map(|(&s, &pair)| decoded_sample(s, pair, self.max_sample))
            .collect()
    }

    /// The colour of the sample tuple `px` through the profile when there
    /// is one, otherwise through the space.
    fn to_rgb(&self, px: &[u8]) -> [u8; 3] {
        let comps = self.components(px);
        self.profile
            .as_ref()
            .and_then(|p| p.conv.convert(self.doc, &comps))
            .unwrap_or_else(|| self.space.convert_to_rgb(self.doc, &comps))
    }

    /// The table of the space's conversion of every one-component sample.
    /// A table whose entries are each neutral to within one level (the
    /// rounding of a CIE-based gray conversion) keeps one byte per pixel,
    /// taking the green channel.
    fn lut(&self) -> Lut {
        let mut rgb = Box::new([[0u8; 3]; 256]);
        for (s, entry) in (0u8..=255).zip(rgb.iter_mut()) {
            *entry = self.to_rgb(&[s]);
        }
        if rgb
            .iter()
            .any(|[r, g, b]| r.abs_diff(*g) > 1 || b.abs_diff(*g) > 1)
        {
            return Lut::Rgb(rgb);
        }
        let mut gray = Box::new([0u8; 256]);
        for (out, [_, g, _]) in gray.iter_mut().zip(rgb.iter()) {
            *out = *g;
        }
        if (0u8..=255).zip(gray.iter()).all(|(s, &v)| s == v) {
            Lut::Identity
        } else {
            Lut::Gray(gray)
        }
    }

    /// One-component samples through the table; `None` when the widened
    /// output cannot be allocated.
    fn convert_one(&self, mut bytes: Vec<u8>) -> Option<(ImageColorSpace, Vec<u8>)> {
        Some(match self.lut() {
            Lut::Identity => (ImageColorSpace::Gray, bytes),
            Lut::Gray(table) => {
                for b in &mut bytes {
                    *b = table[usize::from(*b)];
                }
                (ImageColorSpace::Gray, bytes)
            }
            Lut::Rgb(table) => {
                let mut rgb = super::try_with_capacity(bytes.len() * 3)?;
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
    fn rgb_convert(&self) -> RgbConvert {
        if let Some(profile) = &self.profile {
            return match &*profile.conv {
                IccConversion::Shaper(ProfileConversion::Rgb(Some(shaper))) => {
                    RgbConvert::Shaper(Arc::clone(shaper))
                }
                IccConversion::Transform { .. } => RgbConvert::Transform(Arc::clone(&profile.conv)),
                _ => RgbConvert::Identity,
            };
        }
        match self.space {
            ColorSpace::Cal(cal) => cal
                .rgb_shaper()
                .map_or(RgbConvert::Identity, RgbConvert::Shaper),
            ColorSpace::IccBased {
                stream_id: Some(id),
                ..
            } => {
                let Some(conv) = IccConversion::load(self.doc, *id, 3) else {
                    return RgbConvert::Identity;
                };
                match &*conv {
                    IccConversion::Shaper(ProfileConversion::Rgb(Some(shaper))) => {
                        RgbConvert::Shaper(Arc::clone(shaper))
                    }
                    IccConversion::Transform { .. } => RgbConvert::Transform(Arc::clone(&conv)),
                    IccConversion::Alternate(ColorSpace::Cal(cal)) => cal
                        .rgb_shaper()
                        .map_or(RgbConvert::Identity, RgbConvert::Shaper),
                    IccConversion::Alternate(ColorSpace::Lab(_)) => RgbConvert::Memo,
                    _ => RgbConvert::Identity,
                }
            }
            ColorSpace::Lab(_) | ColorSpace::DeviceN { .. } => RgbConvert::Memo,
            _ => RgbConvert::Identity,
        }
    }

    /// Three-component samples: `/Decode` as a remap, then the conversion.
    /// `None` when the output cannot be allocated.
    fn convert_rgb(&self, bytes: Vec<u8>) -> Option<(ImageColorSpace, Vec<u8>)> {
        let convert = self.rgb_convert();
        if matches!(convert, RgbConvert::Memo) {
            return Some((ImageColorSpace::Rgb, self.convert_memo(&bytes, 3)?));
        }
        let mut bytes = self.remap(bytes);
        match convert {
            RgbConvert::Identity | RgbConvert::Memo => {}
            RgbConvert::Shaper(shaper) => shaper.convert_in_place(&mut bytes),
            RgbConvert::Transform(conv) => {
                let mut out = super::try_filled(bytes.len(), 0)?;
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
            return if matches!(*profile.conv, IccConversion::Transform { .. }) {
                CmykConvert::Icc(Arc::clone(&profile.conv))
            } else {
                CmykConvert::Device
            };
        }
        match self.space {
            ColorSpace::IccBased {
                stream_id: Some(id),
                ..
            } => match IccConversion::load(self.doc, *id, 4) {
                Some(conv) if matches!(*conv, IccConversion::Transform { .. }) => {
                    CmykConvert::Icc(conv)
                }
                _ => CmykConvert::Device,
            },
            ColorSpace::DeviceN { .. } => CmykConvert::Memo,
            _ => CmykConvert::Device,
        }
    }

    /// Four-component samples: `/Decode` as a remap, then the conversion.
    /// `None` when the byte count is not a multiple of four.
    fn convert_cmyk(
        &self,
        bytes: Vec<u8>,
        #[cfg(feature = "gpu-icc")] icc: &mut super::IccHandles<'_>,
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
        if let Some(ctx) = icc.gpu_ctx {
            let profile_bytes = match (&convert, &self.profile, &self.space) {
                (CmykConvert::Icc(_), Some(profile), _) => Some(profile.bytes.to_vec()),
                (
                    CmykConvert::Icc(_),
                    None,
                    ColorSpace::IccBased {
                        stream_id: Some(id),
                        ..
                    },
                ) => fetch_profile_bytes(self.doc, *id),
                _ => None,
            };
            let clut = profile_bytes.and_then(|profile| {
                super::icc::bake_cmyk_clut_cached(
                    &profile,
                    super::icc::DEFAULT_GRID_N,
                    icc.clut_cache.as_deref_mut()?,
                )
                .map_err(|e| {
                    log::warn!(
                        "image: ICC CLUT bake failed, using the profile-free CMYK model: {e}"
                    );
                })
                .ok()
            });
            match ctx.icc_cmyk_to_rgb(
                &bytes,
                clut.as_deref().map(|t| (t, super::icc::DEFAULT_GRID_N)),
            ) {
                Ok(rgb) => return Some((ImageColorSpace::Rgb, rgb)),
                Err(e) => log::warn!("image: GPU CMYK→RGB failed, falling back to CPU: {e}"),
            }
        }
        let mut rgb = super::try_filled(bytes.len() / 4 * 3, 0)?;
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
        let luts: Vec<[u8; 256]> = self
            .decode
            .iter()
            .map(|&(dmin, dmax)| decode_lut(dmin, dmax))
            .collect();
        for px in bytes.chunks_mut(luts.len()) {
            for (s, lut) in px.iter_mut().zip(&luts) {
                *s = lut[usize::from(*s)];
            }
        }
        bytes
    }

    /// `n`-component samples through the space, each distinct tuple
    /// converted once; `None` when the output cannot be allocated.
    fn convert_memo(&self, bytes: &[u8], n: usize) -> Option<Vec<u8>> {
        let mut memo: HashMap<Box<[u8]>, [u8; 3]> = HashMap::new();
        let mut out = super::try_with_capacity(bytes.len() / n * 3)?;
        for px in bytes.chunks_exact(n) {
            let rgb = if let Some(&rgb) = memo.get(px) {
                rgb
            } else {
                if memo.len() >= MEMO_LIMIT {
                    memo.clear();
                }
                let rgb = self.to_rgb(px);
                let _ = memo.insert(px.into(), rgb);
                rgb
            };
            out.extend_from_slice(&rgb);
        }
        Some(out)
    }
}

// ── `/Decode` ─────────────────────────────────────────────────────────────────

/// The `/Decode` array of `dict` as `(dmin, dmax)` pairs for `components`
/// components.  `None` when the array is absent, holds fewer than
/// `2 × components` numbers, or holds a non-finite value (a malformed array
/// would otherwise map every sample to black through `NaN as u8 == 0`).
pub(super) fn parse_decode(dict: &Dictionary, components: usize) -> Option<Vec<(f64, f64)>> {
    let Some(Object::Array(arr)) = dict.get(b"Decode") else {
        return None;
    };
    let vals: Vec<f64> = arr.iter().filter_map(pdf::Object::as_f64).collect();
    if vals.len() < components * 2 {
        if !arr.is_empty() {
            log::debug!(
                "image: Decode array has {} entries, need {}×2={} — ignoring (identity assumed)",
                arr.len(),
                components,
                components * 2,
            );
        }
        return None;
    }
    let pairs: Vec<(f64, f64)> = vals
        .as_chunks::<2>()
        .0
        .iter()
        .take(components)
        .map(|&pair| pair.into())
        .collect();
    if pairs
        .iter()
        .any(|&(d0, d1)| !d0.is_finite() || !d1.is_finite())
    {
        log::warn!(
            "image: Decode array contains non-finite value(s) — ignoring (identity assumed)"
        );
        return None;
    }
    Some(pairs)
}

/// Table of the `/Decode` pair `(dmin, dmax)` over every 8-bit sample `s`:
/// `round(dmin·255 + s·(dmax − dmin))` clamped to `[0, 255]`, giving values
/// in `[dmin·255, dmax·255]` (or swapped when `dmax < dmin`).
fn decode_lut(dmin: f64, dmax: f64) -> [u8; 256] {
    std::array::from_fn(|i| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "i indexes a 256-entry table"
        )]
        let s = i as u8;
        let remapped = dmin.mul_add(255.0, f64::from(s) * (dmax - dmin));
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "remapped is clamped to [0.0, 255.0] before cast"
        )]
        let out = remapped.round().clamp(0.0, 255.0) as u8;
        out
    })
}

/// Tolerance within which two `Decode` values are equal.  Decode values
/// are parsed PDF reals, so exact values are typical.
const DECODE_EPS: f64 = 1e-9;

/// `true` when `(dmin, dmax)` is `(0, 1)`.
fn is_identity_pair(dmin: f64, dmax: f64) -> bool {
    dmin.abs() < DECODE_EPS && (dmax - 1.0).abs() < DECODE_EPS
}

fn pairs_equal(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&(a0, a1), &(b0, b1))| {
            (a0 - b0).abs() < DECODE_EPS && (a1 - b1).abs() < DECODE_EPS
        })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{empty_doc, make_doc_with_objects, make_doc_with_stream};

    fn name_obj(n: &[u8]) -> Object {
        Object::Name(n.to_vec())
    }

    /// Convert `data` of `layout` as an image whose dictionary holds
    /// `/ColorSpace cs` and `extras`.
    fn convert(
        doc: &Document,
        cs: Object,
        extras: &[(&str, Object)],
        layout: Layout,
        data: &[u8],
    ) -> ImageDescriptor {
        let mut dict = Dictionary::new();
        dict.set("ColorSpace", cs);
        for (k, v) in extras {
            dict.set(*k, v.clone());
        }
        let colour = ImageColour::resolve(doc, &dict, None);
        let width = u32::try_from(data.len() / layout_components(layout)).unwrap();
        colour
            .convert(
                doc,
                Samples::cpu(width, 1, layout, data.to_vec()),
                #[cfg(feature = "gpu-icc")]
                &mut super::super::IccHandles::none(),
            )
            .expect("converts")
    }

    fn layout_components(layout: Layout) -> usize {
        match layout {
            Layout::Mask => 1,
            Layout::Color(n) => usize::from(n),
        }
    }

    fn decode_arr(values: &[i64]) -> Object {
        Object::Array(values.iter().map(|&v| Object::Integer(v)).collect())
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
                Samples::cpu(1, 1, Layout::Color(3), vec![1, 2, 3]),
                #[cfg(feature = "gpu-icc")]
                &mut super::super::IccHandles::none(),
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

    // ── Cal*, Default*, /None ─────────────────────────────────────────────────

    fn cal(family: &str, gamma: &str) -> Object {
        pdf::parse_object(
            format!("[/{family} << /WhitePoint [0.9505 1.0 1.089] /Gamma {gamma} >>]").as_bytes(),
            &mut 0,
        )
        .unwrap()
    }

    #[test]
    fn cal_images_convert_as_the_fill_path_does() {
        let doc = empty_doc();
        let rgb_space = resolve(&doc, &cal("CalRGB", "[1.8 1.8 1.8]"));
        let want = rgb_space.convert_to_rgb(&doc, &[128.0 / 255.0; 3]);
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
        let want = gray_space.convert_to_rgb(&doc, &[128.0 / 255.0]);
        let img = convert(&doc, cal("CalGray", "1.8"), &[], Layout::Color(1), &[128]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &want[..1]);
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

    // ── Indexed ───────────────────────────────────────────────────────────────

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

    /// Two-colorant `[/Black /Spot]` `DeviceN` space whose Type 4 tint
    /// transform adds Black to K and maps the spot to the teal CMYK
    /// `(1, 0.08, 0.26, 0.38)` per unit tint.
    fn devicen_teal() -> (Document, Object) {
        const PROGRAM: &str = "{ exch 1 index 0.08 mul 2 index 0.26 mul 3 2 roll \
                               3 index 0.38 mul add dup 1 gt { pop 1 } if }";
        let body = format!(
            "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
             stream\n{PROGRAM}\nendstream",
            PROGRAM.len()
        );
        let doc = make_doc_with_objects(&[(4, &body)]);
        let cs = Object::Array(vec![
            name_obj(b"DeviceN"),
            Object::Array(vec![name_obj(b"Black"), name_obj(b"Spot")]),
            name_obj(b"DeviceCMYK"),
            Object::Reference((4, 0)),
        ]);
        (doc, cs)
    }

    fn teal_rgb() -> [u8; 3] {
        color::cmyk::cmyk_f64_to_rgb(1.0, 0.08, 0.26, 0.38)
    }

    #[test]
    fn multi_colorant_devicen_converts_each_pixel() {
        let (doc, cs) = devicen_teal();
        // Pixels (Black, Spot): spot only, no ink, black only.
        let img = convert(&doc, cs, &[], Layout::Color(2), &[0, 255, 0, 0, 255, 0]);
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(px[0..3], teal_rgb());
        assert_eq!(px[3..6], [255, 255, 255]);
        assert_eq!(px[6..9], color::cmyk::cmyk_f64_to_rgb(0.0, 0.0, 0.0, 1.0));
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

    // ── ICC ───────────────────────────────────────────────────────────────────

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
        make_doc_with_stream(&profile.encode().unwrap(), &format!(" /N {n}"))
    }

    fn icc_cs() -> Object {
        Object::Array(vec![name_obj(b"ICCBased"), Object::Reference((2, 0))])
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
        let want = resolve(&doc, &cal("CalGray", "1.8")).convert_to_rgb(&doc, &[128.0 / 255.0]);
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
        let want = space.convert_to_rgb(&doc, &[128.0 / 255.0; 3]);
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

    #[test]
    fn samples_contradicting_the_profile_convert_as_the_device_space() {
        let doc = gamma_1_8_doc(3);
        let img = convert(&doc, icc_cs(), &[], Layout::Color(1), &[128, 128]);
        assert_eq!(img.color_space, ImageColorSpace::Gray);
        assert_eq!(img.data.as_cpu().unwrap(), &[128, 128]);
    }

    // ── `/Decode` parsing ─────────────────────────────────────────────────────

    #[test]
    fn parse_decode_requires_a_full_array() {
        let mut dict = Dictionary::new();
        assert_eq!(parse_decode(&dict, 1), None);
        dict.set("Decode", decode_arr(&[1]));
        assert_eq!(parse_decode(&dict, 1), None);
        dict.set("Decode", decode_arr(&[1, 0, 0, 1]));
        assert_eq!(parse_decode(&dict, 1), Some(vec![(1.0, 0.0)]));
        assert_eq!(parse_decode(&dict, 2), Some(vec![(1.0, 0.0), (0.0, 1.0)]));
    }

    #[test]
    fn decode_lut_maps_the_endpoints() {
        let lut = decode_lut(1.0, 0.0);
        assert_eq!((lut[0], lut[255], lut[64]), (255, 0, 191));
        let lut = decode_lut(0.0, 0.5);
        assert_eq!((lut[0], lut[255]), (0, 128));
    }
}
