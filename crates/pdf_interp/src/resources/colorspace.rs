//! PDF colour-space taxonomy (§8.6) for tracking the current `cs`/`CS`
//! selection on the graphics state.
//!
//! This is the *input* representation — the colour space declared by the
//! content stream — distinct from the [`super::image::ImageColorSpace`]
//! output category used by the image blit path. The taxonomy is needed by
//! the operators that resolve uncoloured-pattern tints (§8.7.3.3) and the
//! gstate-level conversion to sRGB for stroke/fill operators.
//!
//! # Scope
//!
//! ICC profiles are referenced by [`ObjectId`] and read when
//! [`ColorSpace::convert_to_rgb`] runs; an ICC profile stream is compiled
//! once per document into an `icc::IccConversion`.  `CalGray`, `CalRGB`
//! and Lab parameters are resolved into [`CalSpace`] and [`LabSpace`].
//! Separation and `DeviceN` spaces hold their compiled tint transform
//! ([`TintTransform`]) and `Indexed` spaces their palette bytes, whether
//! given inline or by reference; indirect ones are loaded once per
//! document.
//!
//! # PDF §8.6 mapping
//!
//! | PDF variant   | [`ColorSpace`] variant |
//! |---|---|
//! | `/DeviceGray`, `/G` (Table 89 abbreviation), `/CalGray` | [`ColorSpace::DeviceGray`] |
//! | `/DeviceRGB`, `/RGB` (Table 89), `/CalRGB` | [`ColorSpace::DeviceRgb`] |
//! | `/DeviceCMYK`, `/CMYK` (Table 89) | [`ColorSpace::DeviceCmyk`] |
//! | `[/CalGray ...]`, `[/CalRGB ...]` | [`ColorSpace::Cal`]; the device space when the parameters are invalid |
//! | `[/Lab ...]`         | [`ColorSpace::Lab`]; D65 and `[-100 100 -100 100]` when the parameters are invalid |
//! | `[/ICCBased <ref>]`  | [`ColorSpace::IccBased`] |
//! | `[/Indexed ...]`, `[/I ...]` (Table 93) | [`ColorSpace::Indexed`] |
//! | `/Pattern` or `[/Pattern <base>]` | [`ColorSpace::Pattern`] |
//! | `[/Separation ...]`  | [`ColorSpace::Separation`] |
//! | `[/DeviceN ...]`     | [`ColorSpace::DeviceN`] |
//! | unknown / malformed  | [`ColorSpace::DeviceGray`] (safe fallback) |
//!
//! # Why the lazy `ObjectId` shape
//!
//! Resolving an ICC profile up front would force the graphics state to
//! carry several kilobytes of profile data per push/pop.  Storing the
//! `ObjectId` keeps the gstate small and lets the conversion path
//! dereference only when it actually needs the data.  Tint transforms and
//! palettes are shared through `Arc`, so they cost one pointer.

use std::sync::Arc;

use pdf::{Dictionary, Document, Object, ObjectId};

use super::shading::function::Function;
use crate::resources::dict_ext::DictExt as _;

mod cal;
pub(crate) mod icc;
mod lab;

pub use cal::CalSpace;
use icc::IccConversion;
pub use lab::LabSpace;

/// Maximum recursion depth for nested colour-space chains
/// (Indexed/Pattern/Separation/DeviceN's `alternate` base). PDF spec
/// forbids cycles but adversarial PDFs are not required to be
/// spec-compliant; this bound prevents stack overflow on crafted input.
pub(crate) const MAX_CS_DEPTH: u8 = 8;

/// A resolved PDF colour-space declaration tracked on the graphics state.
///
/// See module docs for the §8.6 mapping. Each variant carries enough data
/// to (a) report its component count for span / tint validation and
/// (b) reach the underlying conversion data when an RGB output is needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColorSpace {
    /// Single-channel grey. PDF §8.6.4.2.
    DeviceGray,
    /// Three-channel RGB. PDF §8.6.4.3.
    DeviceRgb,
    /// Four-channel CMYK. PDF §8.6.4.4.
    DeviceCmyk,
    /// `CalGray` or `CalRGB` with valid parameters. PDF §8.6.5.2–3.
    Cal(Box<CalSpace>),
    /// CIE Lab with its `WhitePoint` and `Range`. PDF §8.6.5.4.
    Lab(Box<LabSpace>),
    /// ICC-profile-based space. PDF §8.6.5.5. `n` is the channel count
    /// (1, 3, or 4) read from the profile stream's `/N` entry.
    IccBased {
        /// Component count from the profile's `/N` entry: 1, 3, or 4.
        n: u8,
        /// `Reference` to the ICC profile stream; `None` if the array's
        /// second element wasn't an indirect reference.
        stream_id: Option<ObjectId>,
    },
    /// Indexed-palette lookup. PDF §8.6.6.3. `base` is the underlying
    /// colour space the palette indexes into; `hival` is the max valid
    /// index (palette entries = `hival + 1`).
    Indexed {
        /// Underlying colour space the palette indexes into.
        base: Box<Self>,
        /// Maximum valid index — palette has `hival + 1` entries.
        hival: u8,
        /// The palette bytes, from an inline string or an indirect string
        /// or stream; `None` when the lookup is missing or unreadable.
        lookup: Option<Arc<[u8]>>,
    },
    /// Pattern space. PDF §8.7.2. The optional `base` is the underlying
    /// colour space for uncoloured patterns (`PaintType` 2); `None` for
    /// coloured patterns.
    Pattern {
        /// Underlying colour space for `PaintType` 2 (uncoloured) patterns;
        /// `None` for `PaintType` 1 (coloured) and bare `/Pattern`.
        base: Option<Box<Self>>,
    },
    /// Single spot-colour. PDF §8.6.6.4. `alternate` is the device space
    /// the tint converts to when the spot colorant isn't separately
    /// available; `tint` maps the tint into it.
    Separation {
        /// Device space the tint converts into when the spot colorant
        /// isn't available.
        alternate: Box<Self>,
        /// The compiled tint transform; `None` when the array has none.
        tint: Option<TintTransform>,
        /// The colorant is `/None`: painting produces no output.
        none: bool,
    },
    /// Multi-channel spot-colour set. PDF §8.6.6.5. `ncomps` is the
    /// number of tint inputs; `alternate` and `tint` mirror
    /// `Separation`.
    DeviceN {
        /// Number of tint components — from the `Names` array length.
        ncomps: u8,
        /// Device space the tint converts into when spot colorants
        /// aren't available.
        alternate: Box<Self>,
        /// The compiled tint transform; `None` when the array has none.
        tint: Option<TintTransform>,
        /// Every colorant is `/None`: painting produces no output.
        none: bool,
    },
}

impl ColorSpace {
    /// The number of tint components a `scn`/`SCN` operator must supply
    /// for this colour space. Used to validate operand counts and to
    /// dispatch the legacy component-count heuristic for uncoloured
    /// patterns.
    #[must_use]
    pub fn ncomponents(&self) -> usize {
        match self {
            // Single-component spaces: scalar grey, palette index, or single
            // spot-colour tint.  All take exactly one operand on `scn`.
            Self::DeviceGray | Self::Indexed { .. } | Self::Separation { .. } => 1,
            Self::DeviceRgb | Self::Lab(_) => 3,
            Self::DeviceCmyk => 4,
            Self::Cal(cal) => cal.ncomponents(),
            Self::IccBased { n, .. } => usize::from(*n),
            Self::Pattern { .. } => 0,
            Self::DeviceN { ncomps, .. } => usize::from(*ncomps),
        }
    }

    /// Whether painting in this space produces no output: a `Separation`
    /// space or a `DeviceN` space whose colorants are all `/None`
    /// (PDF §8.6.6.4, §8.6.6.5), or an uncoloured-pattern space over one.
    #[must_use]
    pub fn paints_nothing(&self) -> bool {
        match self {
            Self::Separation { none, .. } | Self::DeviceN { none, .. } => *none,
            Self::Pattern { base: Some(base) } => base.paints_nothing(),
            _ => false,
        }
    }

    /// Convert a tint-component slice to an sRGB byte triple per PDF §8.6.
    ///
    /// `comps` must have exactly [`ncomponents`](Self::ncomponents)
    /// elements; any mismatch falls back to opaque black.
    ///
    /// `doc` is required so the conversion can dereference lazy
    /// `ObjectId` fields on parameterised variants (ICC profile streams,
    /// Separation/DeviceN tint functions, Indexed palette lookups).
    /// Pass the page's document; the borrow is read-only.
    ///
    /// # Variant behaviour
    ///
    /// - `DeviceGray/Rgb`: direct conversion via the `color::convert`
    ///   helpers; `DeviceCmyk` via the process-ink model of `color::cmyk`.
    /// - `Cal`: gamma and matrix to XYZ, Bradford adaptation from the
    ///   space's white point to D65, sRGB encoding (see [`CalSpace`]).
    /// - `Lab`: `a*` and `b*` clipped to the `Range`, CIE Lab → XYZ relative
    ///   to the `WhitePoint`, Bradford adaptation to D65, sRGB encoding (see
    ///   [`LabSpace`]).
    /// - `IccBased`: converts through the `icc::IccConversion` compiled
    ///   from the profile stream `stream_id`: shaper tables for a gray or
    ///   RGB matrix/TRC profile, otherwise a `moxcms` 8-bit transform of a
    ///   3- or 4-channel profile to sRGB, otherwise the stream's device or
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
    /// - `DeviceN`: passes all `ncomps` tints through the tint-transform
    ///   function and converts the result via the alternate colour space.
    ///   Any failure drops back to `1 - comps[0]` as gray.
    #[must_use]
    pub fn convert_to_rgb(&self, doc: &Document, comps: &[f64]) -> [u8; 3] {
        if comps.len() != self.ncomponents() {
            log::debug!(
                "colorspace: {:?} expects {} components, got {}",
                self,
                self.ncomponents(),
                comps.len(),
            );
            return [0, 0, 0];
        }
        match self {
            Self::DeviceGray => gray_byte_to_rgb(color::convert::gray_to_u8(comps[0])),
            Self::DeviceRgb => color::convert::rgb_to_bytes(comps[0], comps[1], comps[2]),
            Self::DeviceCmyk => {
                color::cmyk::cmyk_f64_to_rgb(comps[0], comps[1], comps[2], comps[3])
            }
            Self::Cal(cal) => cal.convert(comps),
            Self::Lab(lab) => lab.convert(comps),
            Self::IccBased { n, stream_id } => stream_id
                .and_then(|id| IccConversion::load(doc, id, *n))
                .and_then(|conv| conv.convert(doc, comps))
                .unwrap_or_else(|| icc_fallback_by_n(*n, comps)),
            Self::Indexed {
                base,
                hival,
                lookup,
            } => lookup
                .as_deref()
                .and_then(|bytes| indexed_to_rgb(doc, bytes, base, *hival, comps[0]))
                .unwrap_or([0, 0, 0]),
            Self::Pattern { .. } => [0, 0, 0],
            Self::Separation {
                alternate, tint, ..
            } => {
                // PDF §8.6.6.4: pass the single tint component through the
                // tint-transform function to produce values in the alternate
                // colour space, then convert via the alternate.  On any
                // failure (no function, or evaluation fails), use the
                // `1 - tint` gray approximation.
                if let Some(tint) = tint
                    && let Some(rgb) = tint.to_rgb(doc, alternate, comps)
                {
                    return rgb;
                }
                separation_fallback_gray(comps[0])
            }
            Self::DeviceN {
                alternate, tint, ..
            } => {
                // PDF §8.6.6.5: like Separation, with one tint input per
                // colorant.
                if let Some(tint) = tint
                    && let Some(rgb) = tint.to_rgb(doc, alternate, comps)
                {
                    return rgb;
                }
                separation_fallback_gray(comps[0])
            }
        }
    }
}

/// Expand a single grey byte into an opaque RGB triple `[v, v, v]`.
const fn gray_byte_to_rgb(v: u8) -> [u8; 3] {
    [v, v, v]
}

/// A compiled `Separation` / `DeviceN` tint transform (PDF §8.6.6.4),
/// given directly or by reference.  Two values are equal when they share
/// one compiled function.
#[derive(Debug, Clone)]
pub struct TintTransform(Arc<Function>);

impl TintTransform {
    /// Compile the tint-transform object `obj`.
    #[must_use]
    pub(crate) fn load(doc: &Document, obj: &Object) -> Self {
        Self(Function::load(doc, obj))
    }

    /// Pass `tints` through the function and convert the result in
    /// `alternate`.  `None` when evaluation fails.
    fn to_rgb(&self, doc: &Document, alternate: &ColorSpace, tints: &[f64]) -> Option<[u8; 3]> {
        let outputs = self.0.eval_n(tints, alternate.ncomponents())?;
        Some(alternate.convert_to_rgb(doc, &outputs))
    }
}

impl PartialEq for TintTransform {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TintTransform {}

impl ColorSpace {
    /// The default `Decode` array of an image in this space (PDF §8.9.5.2,
    /// Table 90): one `(dmin, dmax)` pair per component.  `[0, 1]` for
    /// device, CIE-based, ICC-based and tint spaces; `L*` in `[0, 100]`
    /// and `a*`, `b*` over the Lab `Range`; and `[0, 2^bpc − 1]` for
    /// `Indexed`, whose samples are palette indices at `bpc` bits.
    #[must_use]
    pub fn default_decode(&self, bpc: u32) -> Vec<(f64, f64)> {
        match self {
            Self::Indexed { .. } => vec![(0.0, f64::from((1u32 << bpc.min(16)) - 1))],
            Self::Lab(lab) => {
                let [amin, amax, bmin, bmax] = lab.range();
                vec![(0.0, 100.0), (amin, amax), (bmin, bmax)]
            }
            _ => vec![(0.0, 1.0); self.ncomponents()],
        }
    }
}

/// Convert palette index `index` of an `Indexed` space to sRGB.
///
/// PDF §8.6.6.3: the lookup holds `hival + 1` entries of
/// `base.ncomponents()` bytes; byte `b` of an entry maps linearly onto the
/// base component's range (`[0, 1]` for device and tint spaces, `L*` in
/// `[0, 100]` and `a*`, `b*` over the Lab `Range`).  `None` when `lookup`
/// is shorter than the entry.
fn indexed_to_rgb(
    doc: &Document,
    lookup: &[u8],
    base: &ColorSpace,
    hival: u8,
    index: f64,
) -> Option<[u8; 3]> {
    let n = base.ncomponents();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [0, hival] ⊂ [0, 255] first"
    )]
    let entry = index.round().clamp(0.0, f64::from(hival)) as usize;
    let raw = lookup.get(entry * n..(entry + 1) * n)?;
    Some(samples_to_rgb(doc, base, raw))
}

/// Convert one 8-bit sample tuple of `space` to sRGB.
///
/// Sample byte `b` of component `i` maps linearly onto that component's
/// default range ([`ColorSpace::default_decode`] at 8 bits).
/// `samples.len()` must equal `space.ncomponents()`; a mismatch converts to
/// black, as [`ColorSpace::convert_to_rgb`] does.
pub(crate) fn samples_to_rgb(doc: &Document, space: &ColorSpace, samples: &[u8]) -> [u8; 3] {
    let comps: Vec<f64> = samples
        .iter()
        .zip(space.default_decode(8))
        .map(|(&b, (dmin, dmax))| decoded_sample(b, (dmin, dmax), 255.0))
        .collect();
    space.convert_to_rgb(doc, &comps)
}

/// The component value of sample `s` under the `Decode` pair
/// `(dmin, dmax)` when samples run from 0 to `max_sample` (PDF §8.9.5.2):
/// `dmin + s · (dmax − dmin) / max_sample`.
pub(crate) fn decoded_sample(s: u8, (dmin, dmax): (f64, f64), max_sample: f64) -> f64 {
    (f64::from(s) / max_sample).mul_add(dmax - dmin, dmin)
}

/// A spot tint with no resolvable transform approximates as gray with
/// `1 - tint`: tint 0 is no colorant (white), tint 1 is full colorant
/// (black), per PDF §8.6.6.4.
fn separation_fallback_gray(tint: f64) -> [u8; 3] {
    let v = (1.0 - tint).clamp(0.0, 1.0);
    gray_byte_to_rgb(color::convert::gray_to_u8(v))
}

/// Fall back to the device space matching the ICC profile's channel count
/// when the moxcms transform isn't available. Mirrors the pre-CLUT behaviour.
fn icc_fallback_by_n(n: u8, comps: &[f64]) -> [u8; 3] {
    match n {
        1 => gray_byte_to_rgb(color::convert::gray_to_u8(comps[0])),
        3 => color::convert::rgb_to_bytes(comps[0], comps[1], comps[2]),
        4 => color::cmyk::cmyk_f64_to_rgb(comps[0], comps[1], comps[2], comps[3]),
        _ => [0, 0, 0],
    }
}

/// Resolve a PDF `ColorSpace` object into a [`ColorSpace`] tracking value.
///
/// Returns [`ColorSpace::DeviceGray`] on unknown, malformed, or
/// recursion-bounded inputs — the same conservative fallback the image
/// pipeline uses (see `image::colorspace::resolve_cs`).
///
/// `cs_obj` may be:
/// - a `Name` for the device families (e.g. `/DeviceRGB`, `/CMYK`),
/// - an `Array` whose first element names a parameterised family
///   (`[/ICCBased <ref>]`, `[/Indexed base hival lookup]`, etc.), or
/// - a `Reference` to either of the above.
#[must_use]
pub fn resolve(doc: &Document, cs_obj: &Object) -> ColorSpace {
    resolve_depth(doc, cs_obj, 0)
}

fn resolve_depth(doc: &Document, cs_obj: &Object, depth: u8) -> ColorSpace {
    if depth >= MAX_CS_DEPTH {
        log::debug!("colorspace: nested too deep (depth {depth}); falling back to DeviceGray");
        return ColorSpace::DeviceGray;
    }
    match cs_obj {
        Object::Name(n) => resolve_device_name(n),
        Object::Reference(id) => doc
            .get_object(*id)
            .ok()
            .map_or(ColorSpace::DeviceGray, |obj| {
                resolve_depth(doc, obj.as_ref(), depth + 1)
            }),
        Object::Array(arr) => resolve_array(doc, arr, depth),
        _ => {
            log::debug!("colorspace: unexpected object type — fallback DeviceGray");
            ColorSpace::DeviceGray
        }
    }
}

/// Map a PDF device-space name (including the `cs`-operator abbreviations
/// from PDF §8.9.7 Table 89) to a [`ColorSpace`]. Unknown names fall back
/// to `DeviceGray`.
///
/// Public so callers that hold a `&[u8]` name straight from the operator
/// stream can short-circuit the four device-space cases without allocating
/// an `Object::Name`. Internal callers in this module use it directly from
/// the `resolve_*` dispatch.
#[must_use]
pub fn resolve_device_name(name: &[u8]) -> ColorSpace {
    match name {
        b"DeviceGray" | b"G" | b"CalGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" | b"RGB" | b"CalRGB" => ColorSpace::DeviceRgb,
        b"DeviceCMYK" | b"CMYK" => ColorSpace::DeviceCmyk,
        b"Pattern" => ColorSpace::Pattern { base: None },
        _ => {
            log::debug!(
                "colorspace: unknown device name {:?} — fallback DeviceGray",
                String::from_utf8_lossy(name)
            );
            ColorSpace::DeviceGray
        }
    }
}

/// The default colour space that the `/ColorSpace` resource dictionary
/// `cs_dict` declares for the device space `device` (PDF §8.6.5.6):
/// `/DefaultGray`, `/DefaultRGB` or `/DefaultCMYK`.
///
/// `None` when `device` is not a device space, the entry is absent, or the
/// entry is unusable: a space of a different component count, a device
/// space, or an `Indexed` or `Pattern` space.  `None` also when the entry
/// is an ICC-based space whose conversion equals the device conversion, so
/// the device path stays in effect for sRGB-equivalent profiles.
#[must_use]
pub fn default_space(
    doc: &Document,
    cs_dict: &Dictionary,
    device: &ColorSpace,
) -> Option<ColorSpace> {
    let key: &[u8] = match device {
        ColorSpace::DeviceGray => b"DefaultGray",
        ColorSpace::DeviceRgb => b"DefaultRGB",
        ColorSpace::DeviceCmyk => b"DefaultCMYK",
        _ => return None,
    };
    let space = resolve(doc, cs_dict.get(key)?);
    let usable = space.ncomponents() == device.ncomponents()
        && !matches!(
            space,
            ColorSpace::DeviceGray
                | ColorSpace::DeviceRgb
                | ColorSpace::DeviceCmyk
                | ColorSpace::Indexed { .. }
                | ColorSpace::Pattern { .. }
        );
    if !usable {
        log::debug!(
            "colorspace: /{} {space:?} unusable for {device:?}",
            String::from_utf8_lossy(key)
        );
        return None;
    }
    if let ColorSpace::IccBased {
        n,
        stream_id: Some(id),
    } = space
        && IccConversion::load(doc, id, n).is_some_and(|conv| conv.is_device_equivalent())
    {
        return None;
    }
    Some(space)
}

fn resolve_array(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    let Some(family) = arr.first().and_then(Object::as_name) else {
        log::debug!("colorspace: array missing leading family name — fallback");
        return ColorSpace::DeviceGray;
    };
    match family {
        b"DeviceGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" => ColorSpace::DeviceRgb,
        b"CalGray" => resolve_cal(doc, arr, 1, ColorSpace::DeviceGray),
        b"CalRGB" => resolve_cal(doc, arr, 3, ColorSpace::DeviceRgb),
        b"DeviceCMYK" => ColorSpace::DeviceCmyk,
        b"Lab" => ColorSpace::Lab(Box::new(LabSpace::resolve(doc, arr.get(1)))),
        b"ICCBased" => resolve_icc_based(doc, arr.get(1), depth),
        b"Indexed" | b"I" => resolve_indexed(doc, arr, depth),
        b"Pattern" => ColorSpace::Pattern {
            base: arr
                .get(1)
                .map(|o| Box::new(resolve_depth(doc, o, depth + 1))),
        },
        b"Separation" => resolve_separation(doc, arr, depth),
        b"DeviceN" => resolve_device_n(doc, arr, depth),
        _ => {
            log::debug!(
                "colorspace: unknown array family {:?} — fallback DeviceGray",
                String::from_utf8_lossy(family)
            );
            ColorSpace::DeviceGray
        }
    }
}

/// `[/CalGray dict]` (`n = 1`) or `[/CalRGB dict]` (`n = 3`); `device`
/// when the parameter dictionary is missing or invalid.
fn resolve_cal(doc: &Document, arr: &[Object], n: u8, device: ColorSpace) -> ColorSpace {
    arr.get(1)
        .and_then(|params| CalSpace::resolve(doc, params, n))
        .map_or(device, |cal| ColorSpace::Cal(Box::new(cal)))
}

fn resolve_icc_based(doc: &Document, ref_obj: Option<&Object>, _depth: u8) -> ColorSpace {
    let Some(ref_obj) = ref_obj else {
        return ColorSpace::DeviceGray;
    };
    let stream_id = reference_id(ref_obj);
    // Read N from the stream dict if reachable.
    let n = super::resolve_stream_dict(doc, ref_obj)
        .and_then(|d| d.get_i64(b"N"))
        .and_then(|n| u8::try_from(n).ok())
        .filter(|&n| matches!(n, 1 | 3 | 4))
        .unwrap_or(0);
    if n == 0 {
        log::debug!("colorspace: ICCBased stream has missing or invalid /N");
        return ColorSpace::DeviceGray;
    }
    ColorSpace::IccBased { n, stream_id }
}

fn resolve_indexed(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Indexed base hival lookup]
    let Some(base_obj) = arr.get(1) else {
        return ColorSpace::DeviceGray;
    };
    let base = Box::new(resolve_depth(doc, base_obj, depth + 1));
    let hival = arr
        .get(2)
        .and_then(Object::as_i64)
        .map(|n| n.clamp(0, 255))
        .and_then(|n| u8::try_from(n).ok())
        .unwrap_or(0);
    ColorSpace::Indexed {
        base,
        hival,
        lookup: arr.get(3).and_then(|o| indexed_lookup(doc, o)),
    }
}

/// The palette bytes of an `Indexed` lookup `obj` (PDF §8.6.6.3): a string,
/// or a reference to a string or stream, read once per document.
fn indexed_lookup(doc: &Document, obj: &Object) -> Option<Arc<[u8]>> {
    /// Document-cache entry for an indirect lookup.
    struct Lookup(Arc<[u8]>);

    let bytes = |o: &Object| match o {
        Object::String(b, _) => Some(Arc::<[u8]>::from(b.as_slice())),
        Object::Stream(s) => s
            .decompressed_content()
            .map_err(|e| log::debug!("colorspace: Indexed lookup stream: {e}"))
            .ok()
            .map(Arc::from),
        _ => None,
    };
    match obj {
        Object::Reference(id) => doc
            .derived(*id, || {
                let o = doc.get_object(*id).ok()?;
                bytes(o.as_ref()).map(Lookup)
            })
            .map(|l| Arc::clone(&l.0)),
        direct => bytes(direct),
    }
}

fn resolve_separation(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Separation name alternate tint_transform]
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    ColorSpace::Separation {
        alternate,
        tint: arr.get(3).map(|o| TintTransform::load(doc, o)),
        none: arr.get(1).is_some_and(is_none_colorant),
    }
}

fn resolve_device_n(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/DeviceN names alternate tint_transform attributes?]
    let Some(names) = arr.get(1).and_then(Object::as_array) else {
        return ColorSpace::DeviceGray;
    };
    let ncomps = u8::try_from(names.len()).unwrap_or(0);
    if ncomps == 0 {
        log::debug!("colorspace: DeviceN names array empty — fallback DeviceGray");
        return ColorSpace::DeviceGray;
    }
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    ColorSpace::DeviceN {
        ncomps,
        alternate,
        tint: arr.get(3).map(|o| TintTransform::load(doc, o)),
        none: names.iter().all(is_none_colorant),
    }
}

/// Whether `name` is the colorant `/None`, which never produces visible
/// output (PDF §8.6.6.4, §8.6.6.5).
fn is_none_colorant(name: &Object) -> bool {
    name.as_name() == Some(b"None".as_slice())
}

const fn reference_id(obj: &Object) -> Option<ObjectId> {
    match obj {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::empty_doc;

    fn name_obj(s: &[u8]) -> Object {
        Object::Name(s.to_vec())
    }

    #[test]
    fn device_names_resolve_to_device_variants() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceGray")),
            ColorSpace::DeviceGray
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceRGB")),
            ColorSpace::DeviceRgb
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceCMYK")),
            ColorSpace::DeviceCmyk
        );
    }

    #[test]
    fn abbreviated_names_resolve_per_table_89() {
        let doc = empty_doc();
        // PDF §8.9.7 Table 89 abbreviations (used in inline images but also
        // accepted by `cs`/`CS` per most decoders).
        assert_eq!(resolve(&doc, &name_obj(b"G")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"RGB")), ColorSpace::DeviceRgb);
        assert_eq!(resolve(&doc, &name_obj(b"CMYK")), ColorSpace::DeviceCmyk);
    }

    #[test]
    fn cal_spaces_approximate_to_device() {
        let doc = empty_doc();
        // The bare names carry no parameters; they map to the device spaces.
        assert_eq!(resolve(&doc, &name_obj(b"CalGray")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"CalRGB")), ColorSpace::DeviceRgb);
    }

    #[test]
    fn pattern_name_yields_pattern_no_base() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"Pattern")),
            ColorSpace::Pattern { base: None },
        );
    }

    #[test]
    fn unknown_name_falls_back_to_devicegray() {
        let doc = empty_doc();
        assert_eq!(resolve(&doc, &name_obj(b"Bogus")), ColorSpace::DeviceGray);
    }

    /// `[/Lab << /WhitePoint [0.9505 1 1.089] >>]`.
    fn lab_d65() -> ColorSpace {
        parse_space(&empty_doc(), "[/Lab << /WhitePoint [0.9505 1 1.089] >>]")
    }

    #[test]
    fn lab_array_resolves_its_parameters_by_value_or_reference() {
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            "<< /WhitePoint [0.9505 1 1.089] /Range [-50 50 -50 50] >>",
        )]);
        let by_ref = resolve(
            &doc,
            &Object::Array(vec![name_obj(b"Lab"), Object::Reference((4, 0))]),
        );
        let inline = parse_space(
            &doc,
            "[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-50 50 -50 50] >>]",
        );
        assert_eq!(by_ref, inline);
        let ColorSpace::Lab(space) = &by_ref else {
            panic!("Lab expected");
        };
        assert_eq!(space.range(), [-50.0, 50.0, -50.0, 50.0]);
        assert_ne!(by_ref, lab_d65());
    }

    #[test]
    fn pattern_array_with_base_recurses_into_base() {
        let doc = empty_doc();
        let arr = vec![name_obj(b"Pattern"), name_obj(b"DeviceRGB")];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Pattern {
                base: Some(Box::new(ColorSpace::DeviceRgb)),
            },
        );
    }

    #[test]
    fn indexed_array_resolves_base_and_clamps_hival() {
        let doc = empty_doc();
        // [/Indexed /DeviceRGB 255 <lookup>]
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(255),
            Object::String(b"\x00\x01\x02".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 255,
                lookup: Some(Arc::from(&b"\x00\x01\x02"[..])),
            },
        );
    }

    #[test]
    fn indexed_hival_out_of_range_clamps_to_255() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceGray"),
            Object::Integer(99_999), // adversarial
            Object::String(b"".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceGray),
                hival: 255,
                lookup: Some(Arc::from(&[][..])),
            },
        );
    }

    #[test]
    fn indexed_lookup_reference_to_a_missing_object_has_no_palette() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(15),
            Object::Reference((42, 0)),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 15,
                lookup: None,
            },
        );
    }

    #[test]
    fn separation_extracts_alternate_and_tint() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Separation"),
            name_obj(b"PANTONE_185_C"),
            name_obj(b"DeviceCMYK"),
            Object::Reference((42, 0)),
        ];
        let ColorSpace::Separation {
            alternate,
            tint,
            none,
        } = resolve(&doc, &Object::Array(arr))
        else {
            panic!("expected Separation");
        };
        assert_eq!(*alternate, ColorSpace::DeviceCmyk);
        assert!(tint.is_some());
        assert!(!none);
    }

    #[test]
    fn device_n_extracts_ncomps_from_names_array() {
        let doc = empty_doc();
        let names = Object::Array(vec![
            name_obj(b"Cyan"),
            name_obj(b"Magenta"),
            name_obj(b"Spot1"),
        ]);
        let arr = vec![
            name_obj(b"DeviceN"),
            names,
            name_obj(b"DeviceCMYK"),
            Object::Reference((7, 0)),
        ];
        let ColorSpace::DeviceN {
            ncomps,
            alternate,
            tint,
            none,
        } = resolve(&doc, &Object::Array(arr))
        else {
            panic!("expected DeviceN");
        };
        assert_eq!((ncomps, *alternate), (3, ColorSpace::DeviceCmyk));
        assert!(tint.is_some());
        assert!(!none);
    }

    #[test]
    fn malformed_array_falls_back_to_devicegray() {
        let doc = empty_doc();
        // Empty array, no family name.
        assert_eq!(
            resolve(&doc, &Object::Array(vec![])),
            ColorSpace::DeviceGray,
        );
        // Unknown family.
        let unknown = Object::Array(vec![name_obj(b"Bogus")]);
        assert_eq!(resolve(&doc, &unknown), ColorSpace::DeviceGray);
    }

    #[test]
    fn ncomponents_matches_spec() {
        assert_eq!(ColorSpace::DeviceGray.ncomponents(), 1);
        assert_eq!(ColorSpace::DeviceRgb.ncomponents(), 3);
        assert_eq!(ColorSpace::DeviceCmyk.ncomponents(), 4);
        assert_eq!(lab_d65().ncomponents(), 3);
        assert_eq!(
            ColorSpace::IccBased {
                n: 4,
                stream_id: None
            }
            .ncomponents(),
            4,
        );
        assert_eq!(
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 255,
                lookup: None,
            }
            .ncomponents(),
            1,
        );
        assert_eq!(ColorSpace::Pattern { base: None }.ncomponents(), 0);
        assert_eq!(
            ColorSpace::Separation {
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint: None,
                none: false,
            }
            .ncomponents(),
            1,
        );
        assert_eq!(
            ColorSpace::DeviceN {
                ncomps: 5,
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint: None,
                none: false,
            }
            .ncomponents(),
            5,
        );
    }

    // ── convert_to_rgb ──────────────────────────────────────────────────────

    #[test]
    fn convert_devicegray_full_range() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[0.0]),
            [0, 0, 0]
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[1.0]),
            [255, 255, 255],
        );
        assert_eq!(
            ColorSpace::DeviceGray.convert_to_rgb(&doc, &[0.5]),
            [128, 128, 128], // gray_to_u8 rounds 0.5*255 = 127.5 → 128
        );
    }

    #[test]
    fn convert_devicergb_passes_through() {
        let doc = empty_doc();
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]),
            [255, 0, 0],
        );
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[0.0, 1.0, 0.0]),
            [0, 255, 0],
        );
    }

    #[test]
    fn convert_devicecmyk_uses_process_ink_model() {
        let doc = empty_doc();
        // (0, 0, 0, 0) = paper white.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[0.0, 0.0, 0.0, 0.0]),
            [255, 255, 255],
        );
        // (1, 1, 1, 1) = four-colour solid = black.
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[1.0, 1.0, 1.0, 1.0]),
            [0, 0, 0],
        );
        // Solid cyan is the printed process cyan, not the ideal (0, 255, 255).
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[1.0, 0.0, 0.0, 0.0]),
            color::cmyk::CORNERS[4],
        );
    }

    #[test]
    fn convert_lab_black_is_black() {
        let doc = empty_doc();
        // L*=0 ≡ black; a*/b* irrelevant.
        let black = lab_d65().convert_to_rgb(&doc, &[0.0, 0.0, 0.0]);
        assert_eq!(black, [0, 0, 0]);
    }

    #[test]
    fn convert_lab_white_is_white() {
        let doc = empty_doc();
        // L*=100, a*=b*=0 ≡ D65 white.  sRGB encoding may not be exactly 255
        // due to floating-point rounding; allow ±2 LSB.
        let [r, g, b] = lab_d65().convert_to_rgb(&doc, &[100.0, 0.0, 0.0]);
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
        assert_eq!(icc1.convert_to_rgb(&doc, &[1.0]), [255, 255, 255]);
        assert_eq!(icc3.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
        assert_eq!(
            icc4.convert_to_rgb(&doc, &[0.0, 0.0, 0.0, 1.0]),
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
        assert_eq!(icc.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]), [255, 0, 0]);
    }

    #[test]
    fn convert_iccbased_n1_without_a_profile_stream_falls_back() {
        // Object 1 of the empty document is not a stream: no profile bytes.
        let doc = empty_doc();
        let icc = ColorSpace::IccBased {
            n: 1,
            stream_id: Some((1, 0)),
        };
        assert_eq!(icc.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
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
        assert_eq!(icc.convert_to_rgb(&doc, &[128.0 / 255.0]), [146; 3]);

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
        assert_eq!(icc.convert_to_rgb(&doc, &[half, half, half]), [146; 3]);
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
            icc.convert_to_rgb(&doc, &[1.0, 0.0, 0.0]),
            [255, 0, 0],
            "pure red",
        );
        channels_match(
            icc.convert_to_rgb(&doc, &[0.0, 1.0, 0.0]),
            [0, 255, 0],
            "pure green",
        );
        channels_match(
            icc.convert_to_rgb(&doc, &[0.0, 0.0, 1.0]),
            [0, 0, 255],
            "pure blue",
        );
        // Mid grey (0.5, 0.5, 0.5).  sRGB encodes 0.5-linear ≈ 188 but the
        // input is *already* sRGB-encoded; identity should keep it at
        // ~128 modulo transform rounding.  Additionally assert neutrality
        // — moxcms shouldn't introduce a colour cast on grey inputs.
        let grey = icc.convert_to_rgb(&doc, &[0.5, 0.5, 0.5]);
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
            none: false,
        };
        assert_eq!(sep.convert_to_rgb(&doc, &[0.0]), [255, 255, 255]); // no ink
        assert_eq!(sep.convert_to_rgb(&doc, &[1.0]), [0, 0, 0]); // full ink
    }

    #[test]
    fn convert_separation_unresolvable_tint_fn_falls_back() {
        // The tint transform references a missing object → evaluation
        // fails → the gray heuristic.
        let doc = empty_doc();
        let sep = ColorSpace::Separation {
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((999, 0)))),
            none: false,
        };
        assert_eq!(sep.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
    }

    /// Type 4 tint transform of a two-colorant `[/Black /Spot]` space into
    /// CMYK: Black adds to K; the spot is a teal `(1, 0.08, 0.26, 0.38)` per
    /// unit tint.
    const DEVICEN_TEAL_FN: &str = "{ exch 1 index 0.08 mul 2 index 0.26 mul 3 2 roll \
                                   3 index 0.38 mul add dup 1 gt { pop 1 } if }";

    fn devicen_teal_doc() -> Document {
        let body = format!(
            "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1 0 1 0 1 0 1] /Length {} >>\n\
             stream\n{DEVICEN_TEAL_FN}\nendstream",
            DEVICEN_TEAL_FN.len()
        );
        crate::test_helpers::make_doc_with_objects(&[(4, &body)])
    }

    #[test]
    fn convert_devicen_multi_input_evaluates_the_tint_transform() {
        let doc = devicen_teal_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 2,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((4, 0)))),
            none: false,
        };
        // Full spot, no black: the teal CMYK, not the gray of comps[0].
        let teal = dn.convert_to_rgb(&doc, &[0.0, 1.0]);
        let want = ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[1.0, 0.08, 0.26, 0.38]);
        assert_eq!(teal, want);
        assert!(teal[2] > teal[0] + 40, "teal, got {teal:?}");
        // Full black on top saturates K.
        assert_eq!(
            dn.convert_to_rgb(&doc, &[1.0, 1.0]),
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[1.0, 0.08, 0.26, 1.0])
        );
        // No ink at all is paper white.
        assert_eq!(dn.convert_to_rgb(&doc, &[0.0, 0.0]), [255, 255, 255]);
    }

    #[test]
    fn convert_indexed_looks_up_the_palette_through_the_base() {
        // Two-entry RGB palette: red, blue.
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            &b"<< /Length 6 >>\nstream\n\xff\x00\x00\x00\x00\xff\nendstream"[..],
        )]);
        let idx = ColorSpace::Indexed {
            base: Box::new(ColorSpace::DeviceRgb),
            hival: 1,
            lookup: indexed_lookup(&doc, &Object::Reference((4, 0))),
        };
        assert_eq!(idx.convert_to_rgb(&doc, &[0.0]), [255, 0, 0]);
        assert_eq!(idx.convert_to_rgb(&doc, &[1.0]), [0, 0, 255]);
        // An out-of-range index clamps to hival.
        assert_eq!(idx.convert_to_rgb(&doc, &[7.0]), [0, 0, 255]);
    }

    /// The colour space of the PDF object source `src`.
    fn parse_space(doc: &Document, src: &str) -> ColorSpace {
        let obj = pdf::parse_object(src.as_bytes(), &mut 0).expect("colour-space object parses");
        resolve(doc, &obj)
    }

    #[test]
    fn inline_tint_transform_converts_through_the_alternate() {
        let doc = empty_doc();
        let tint = "<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>";
        let sep = parse_space(&doc, &format!("[/Separation /Spot /DeviceRGB {tint}]"));
        assert_eq!(sep.convert_to_rgb(&doc, &[1.0]), [255, 0, 0]);
        assert_eq!(sep.convert_to_rgb(&doc, &[0.0]), [255, 255, 255]);
        let dn = parse_space(&doc, &format!("[/DeviceN [/Spot] /DeviceRGB {tint}]"));
        assert_eq!(dn.convert_to_rgb(&doc, &[1.0]), [255, 0, 0]);
    }

    #[test]
    fn inline_indexed_palette_is_looked_up() {
        let doc = empty_doc();
        let idx = parse_space(&doc, "[/Indexed /DeviceRGB 1 <ff000000ff00>]");
        assert_eq!(idx.convert_to_rgb(&doc, &[0.0]), [255, 0, 0]);
        assert_eq!(idx.convert_to_rgb(&doc, &[1.0]), [0, 255, 0]);
    }

    #[test]
    fn indirect_tint_transform_is_shared_across_resolutions() {
        let doc = devicen_teal_doc();
        let src = "[/DeviceN [/Black /Spot] /DeviceCMYK 4 0 R]";
        assert_eq!(parse_space(&doc, src), parse_space(&doc, src));
    }

    #[test]
    fn convert_devicen_multi_input_without_tint_fn_falls_back_to_gray() {
        // No tint transform → gray heuristic on comps[0].
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 3,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: None,
            none: false,
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[1.0, 0.5, 0.5]), [0, 0, 0]);
    }

    #[test]
    fn convert_devicen_single_input_no_tint_fn_falls_back_to_gray() {
        // ncomps=1 with no resolvable tint function → gray heuristic on the
        // single tint component (1.0 → "full ink" → black).
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 1,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: None,
            none: false,
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[1.0]), [0, 0, 0]);
        assert_eq!(dn.convert_to_rgb(&doc, &[0.0]), [255, 255, 255]);
    }

    #[test]
    fn convert_devicen_single_input_unresolvable_tint_fn_falls_back() {
        // ncomps=1 with a tint reference that doesn't resolve → evaluation
        // returns None → fall back to the gray heuristic.
        let doc = empty_doc();
        let dn = ColorSpace::DeviceN {
            ncomps: 1,
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(TintTransform::load(&doc, &Object::Reference((999, 0)))),
            none: false,
        };
        assert_eq!(dn.convert_to_rgb(&doc, &[0.5]), [128, 128, 128]);
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
            none: false,
        };
        let outer = ColorSpace::Separation {
            alternate: Box::new(inner),
            tint: None,
            none: false,
        };
        // No infinite loop, deterministic byte triple.
        let result = outer.convert_to_rgb(&doc, &[0.5]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn convert_wrong_component_count_falls_back_to_black() {
        let doc = empty_doc();
        // DeviceRgb wants 3 components; passing 1 must not panic.
        assert_eq!(
            ColorSpace::DeviceRgb.convert_to_rgb(&doc, &[0.5]),
            [0, 0, 0]
        );
        assert_eq!(ColorSpace::DeviceGray.convert_to_rgb(&doc, &[]), [0, 0, 0]);
        assert_eq!(
            ColorSpace::DeviceCmyk.convert_to_rgb(&doc, &[0.5, 0.5]),
            [0, 0, 0],
        );
    }

    #[test]
    fn convert_pattern_returns_black() {
        let doc = empty_doc();
        // Pattern is never directly convertible to RGB.
        assert_eq!(
            ColorSpace::Pattern { base: None }.convert_to_rgb(&doc, &[]),
            [0, 0, 0],
        );
    }
}
