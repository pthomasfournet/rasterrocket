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

use pdf::{Document, Object, ObjectId};

use super::shading::function::Function;
use crate::ColourIntent;

mod cal;
mod convert;
mod group;
pub(crate) mod icc;
mod lab;
mod resolve;
#[cfg(test)]
mod test_support;

pub(crate) use cal::CalSpace;
pub use group::{GroupSpace, RgbGroupSpace};
use icc::IccConversion;
pub(crate) use lab::LabSpace;
pub(crate) use resolve::{default_space, resolve, resolve_device_name};

/// Interpreted operations (see [`ColorSpace::conversion_cost`]) one image's
/// or one shading's colour conversion may spend; past it, colours are
/// converted at the points of a lattice and interpolated between them.
pub(crate) const EXACT_WORK_BUDGET: u64 = 1 << 28;

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
        /// The colorant the tint paints.
        colorant: Colorant,
    },
    /// Multi-channel spot-colour set. ISO 32000-1 §8.6.6.5. One tint input
    /// per colorant; `alternate` and `tint` mirror `Separation`.
    DeviceN {
        /// The colorants, one per tint component, from the `Names` array;
        /// 1 to 255 of them.
        colorants: Arc<[Colorant]>,
        /// Device space the tint converts into when spot colorants
        /// aren't available.
        alternate: Box<Self>,
        /// The compiled tint transform; `None` when the array has none.
        tint: Option<TintTransform>,
    },
}

/// A colorant of a `Separation` or `DeviceN` space (ISO 32000-1 §8.6.6.4,
/// §8.6.6.5), classified by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Colorant {
    /// `/None`: the tint paints nothing.
    None,
    /// `/All`: the tint paints every colorant of the device (a
    /// registration colour).
    All,
    /// A process colorant, `/Cyan`, `/Magenta`, `/Yellow` or `/Black`.
    Process(ProcessInk),
    /// Any other name: a spot colorant.
    Spot(Arc<[u8]>),
}

impl Colorant {
    /// Classify the colorant `name`.
    #[must_use]
    pub fn parse(name: &[u8]) -> Self {
        match name {
            b"None" => Self::None,
            b"All" => Self::All,
            b"Cyan" => Self::Process(ProcessInk::Cyan),
            b"Magenta" => Self::Process(ProcessInk::Magenta),
            b"Yellow" => Self::Process(ProcessInk::Yellow),
            b"Black" => Self::Process(ProcessInk::Black),
            spot => Self::Spot(Arc::from(spot)),
        }
    }
}

/// A process ink of a CMYK device, in the channel order of `DeviceCMYK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessInk {
    Cyan,
    Magenta,
    Yellow,
    Black,
}

impl ProcessInk {
    /// The ink's channel, 0–3, in a `DeviceCMYK` colour or an ink pixel.
    #[must_use]
    pub const fn channel(self) -> usize {
        match self {
            Self::Cyan => 0,
            Self::Magenta => 1,
            Self::Yellow => 2,
            Self::Black => 3,
        }
    }
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
            Self::DeviceN { colorants, .. } => colorants.len(),
        }
    }

    /// A bound on the interpreted work of converting one colour of the
    /// space: for a `Separation` or `DeviceN` space, the tint transform's
    /// cost (see [`TintTransform::cost`]) plus the cost of converting in
    /// the alternate space; for an `Indexed` space, the base space's cost;
    /// one for any other.
    #[must_use]
    pub(crate) fn conversion_cost(&self) -> u64 {
        match self {
            Self::Separation {
                tint, alternate, ..
            }
            | Self::DeviceN {
                tint, alternate, ..
            } => tint
                .as_ref()
                .map_or(0, TintTransform::cost)
                .saturating_add(alternate.conversion_cost()),
            Self::Indexed { base, .. } => base.conversion_cost(),
            _ => 1,
        }
    }

    /// A bound on the work of [`process_ink`](Self::process_ink) of one
    /// colour of the space, in the units of
    /// [`conversion_cost`](Self::conversion_cost): for a four-component
    /// ICC space, the conversion and the solve for the reproducing inks
    /// ([`color::cmyk::INK_REPRODUCING_COST`]); for a `Separation` or
    /// `DeviceN` space, the tint transform's cost plus the process-ink cost
    /// of the alternate space; the conversion cost for any other.
    #[must_use]
    pub(crate) fn process_ink_cost(&self) -> u64 {
        match self {
            Self::IccBased { n: 4, .. } => {
                color::cmyk::INK_REPRODUCING_COST.saturating_add(self.conversion_cost())
            }
            Self::Separation {
                tint, alternate, ..
            }
            | Self::DeviceN {
                tint, alternate, ..
            } => tint
                .as_ref()
                .map_or(0, TintTransform::cost)
                .saturating_add(alternate.process_ink_cost()),
            _ => self.conversion_cost(),
        }
    }

    /// Whether the colours of the space are grays: `DeviceGray`, `CalGray`,
    /// a one-component ICC space, or an `Indexed` space over one of them.
    /// The device conversion to CMYK prints a gray in black ink alone
    /// (ISO 32000-2 §10.4.2).
    #[must_use]
    pub(crate) fn is_gray(&self) -> bool {
        match self {
            Self::DeviceGray | Self::IccBased { n: 1, .. } => true,
            Self::Cal(cal) => cal.ncomponents() == 1,
            Self::Indexed { base, .. } => base.is_gray(),
            _ => false,
        }
    }

    /// Whether the space is a process CMYK space — `DeviceCMYK` or a
    /// four-component ICC space — whose components are the inks of a CMYK
    /// device.
    #[must_use]
    pub(crate) const fn is_process_cmyk(&self) -> bool {
        matches!(self, Self::DeviceCmyk | Self::IccBased { n: 4, .. })
    }

    /// The colorants of a `Separation` or `DeviceN` space, one per tint
    /// component; empty for every other space.
    #[must_use]
    pub(crate) fn colorants(&self) -> &[Colorant] {
        match self {
            Self::Separation { colorant, .. } => std::slice::from_ref(colorant),
            Self::DeviceN { colorants, .. } => colorants,
            _ => &[],
        }
    }

    /// The components of the initial colour of this space (PDF §8.6.3 and
    /// §8.6.8), to which `cs`/`CS` set the current colour: black — every
    /// component 0, or 0 0 0 1 for `DeviceCMYK` — for the device, CIE-based
    /// and ICC-based spaces, index 0 for `Indexed`, every tint 1 for
    /// `Separation` and `DeviceN`.  A `Pattern` space has no components:
    /// its initial colour is a pattern that paints nothing.
    #[must_use]
    pub fn initial_components(&self) -> Vec<f64> {
        match self {
            Self::DeviceCmyk => vec![0.0, 0.0, 0.0, 1.0],
            Self::Separation { .. } | Self::DeviceN { .. } => vec![1.0; self.ncomponents()],
            _ => vec![0.0; self.ncomponents()],
        }
    }

    /// Whether this space converts its colours alike under the intents `a`
    /// and `b`: every ICC-based space it converts through selects the same
    /// conversion under both.
    #[must_use]
    pub(crate) fn converts_alike(&self, doc: &Document, a: ColourIntent, b: ColourIntent) -> bool {
        match self {
            Self::IccBased {
                n,
                stream_id: Some(id),
            } => IccConversion::converts_alike(doc, *id, *n, a, b),
            Self::Indexed { base, .. } => base.converts_alike(doc, a, b),
            Self::Separation { alternate, .. } | Self::DeviceN { alternate, .. } => {
                alternate.converts_alike(doc, a, b)
            }
            Self::Pattern { base } => base
                .as_deref()
                .is_none_or(|base| base.converts_alike(doc, a, b)),
            Self::IccBased {
                stream_id: None, ..
            }
            | Self::DeviceGray
            | Self::DeviceRgb
            | Self::DeviceCmyk
            | Self::Cal(_)
            | Self::Lab(_) => true,
        }
    }

    /// Whether painting in this space produces no output: a `Separation`
    /// space or a `DeviceN` space whose colorants are all `/None`
    /// (PDF §8.6.6.4, §8.6.6.5), or an uncoloured-pattern space over one.
    #[must_use]
    pub fn paints_nothing(&self) -> bool {
        match self {
            Self::Separation { .. } | Self::DeviceN { .. } => {
                self.colorants().iter().all(|c| *c == Colorant::None)
            }
            Self::Pattern { base: Some(base) } => base.paints_nothing(),
            _ => false,
        }
    }
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

    /// A bound on the work of one evaluation (see [`Function::cost`]).
    #[must_use]
    pub(crate) fn cost(&self) -> u64 {
        self.0.cost()
    }

    /// The components in `alternate` of the colour of `tints`: `tints`
    /// passed through the function.  `None` when evaluation fails.
    pub(crate) fn alternate_components(
        &self,
        alternate: &ColorSpace,
        tints: &[f64],
    ) -> Option<Vec<f64>> {
        self.0.eval_n(tints, alternate.ncomponents())
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

impl ColorSpace {
    /// The base space of an `Indexed` space and the components in it of
    /// the palette entry the index `comps` selects (ISO 32000-1 §8.6.6.3):
    /// the index is rounded and clamped to `hival`, and byte `b` of the
    /// entry maps linearly onto the base component's range (`[0, 1]` for
    /// device and tint spaces, `L*` in `[0, 100]` and `a*`, `b*` over the
    /// Lab `Range`).  `None` for any other space, for a component count
    /// other than 1, and when the lookup is missing or shorter than the
    /// entry.
    #[must_use]
    pub(crate) fn indexed_base_components(&self, comps: &[f64]) -> Option<(&Self, Vec<f64>)> {
        let Self::Indexed {
            base,
            hival,
            lookup,
        } = self
        else {
            return None;
        };
        let &[index] = comps else {
            return None;
        };
        let n = base.ncomponents();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to [0, hival] ⊂ [0, 255] first"
        )]
        let entry = index.round().clamp(0.0, f64::from(*hival)) as usize;
        let raw = lookup.as_deref()?.get(entry * n..(entry + 1) * n)?;
        Some((base, sample_components(base, raw)))
    }
}

/// The components of one 8-bit sample tuple of `space`: sample byte `b`
/// of component `i` maps linearly onto that component's default range
/// ([`ColorSpace::default_decode`] at 8 bits).
fn sample_components(space: &ColorSpace, samples: &[u8]) -> Vec<f64> {
    samples
        .iter()
        .zip(space.default_decode(8))
        .map(|(&b, (dmin, dmax))| decoded_sample(b, (dmin, dmax), 255.0))
        .collect()
}

/// The component value of sample `s` under the `Decode` pair
/// `(dmin, dmax)` when samples run from 0 to `max_sample` (PDF §8.9.5.2):
/// `dmin + s · (dmax − dmin) / max_sample`.
pub(crate) fn decoded_sample(s: u8, (dmin, dmax): (f64, f64), max_sample: f64) -> f64 {
    (f64::from(s) / max_sample).mul_add(dmax - dmin, dmin)
}

#[cfg(test)]
mod tests {
    use super::test_support::{devicen_teal_doc, lab_d65, spot, spots};
    use super::*;

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
                colorant: spot(),
            }
            .ncomponents(),
            1,
        );
        assert_eq!(
            ColorSpace::DeviceN {
                colorants: spots(5),
                alternate: Box::new(ColorSpace::DeviceCmyk),
                tint: None,
            }
            .ncomponents(),
            5,
        );
    }

    /// A tint space converts through its alternate, so its cost counts the
    /// alternate's tint transform too; a palette costs its base's.
    #[test]
    fn conversion_cost_counts_nested_tint_transforms() {
        let doc = devicen_teal_doc();
        let teal = TintTransform::load(&doc, &Object::Reference((4, 0)));
        let inner = ColorSpace::DeviceN {
            colorants: spots(2),
            alternate: Box::new(ColorSpace::DeviceCmyk),
            tint: Some(teal.clone()),
        };
        assert_eq!(inner.conversion_cost(), teal.cost() + 1);
        let outer = ColorSpace::Separation {
            alternate: Box::new(inner),
            tint: Some(teal.clone()),
            colorant: spot(),
        };
        assert_eq!(outer.conversion_cost(), 2 * teal.cost() + 1);
        let indexed = ColorSpace::Indexed {
            base: Box::new(outer),
            hival: 0,
            lookup: None,
        };
        assert_eq!(indexed.conversion_cost(), 2 * teal.cost() + 1);
    }

    /// The process ink of a four-component ICC colour solves for the inks
    /// reproducing it, which its process-ink cost counts, through a tint
    /// space's alternate as well; a device space's is its conversion's.
    #[test]
    fn process_ink_cost_counts_the_reproducing_solve() {
        let doc = devicen_teal_doc();
        let teal = TintTransform::load(&doc, &Object::Reference((4, 0)));
        let icc = ColorSpace::IccBased {
            n: 4,
            stream_id: None,
        };
        let solve = color::cmyk::INK_REPRODUCING_COST + icc.conversion_cost();
        assert_eq!(icc.process_ink_cost(), solve);
        let dn = ColorSpace::DeviceN {
            colorants: spots(2),
            alternate: Box::new(icc),
            tint: Some(teal.clone()),
        };
        assert_eq!(dn.conversion_cost(), teal.cost() + 1);
        assert_eq!(dn.process_ink_cost(), teal.cost() + solve);
        assert_eq!(ColorSpace::DeviceCmyk.process_ink_cost(), 1);
    }
}
