//! PDF colour space and colour value conversions.
//!
//! Maps PDF colour operator values (f64 in `[0,1]`) to raw u8 pixel components
//! understood by the raster crate.
//!
//! The pure conversion functions live in the `color` crate: `gray_to_u8` and
//! `rgb_to_bytes` in `color::convert` (re-exported here), `DeviceCMYK` in
//! `color::cmyk`.

use std::sync::{Arc, OnceLock};

pub(crate) use color::convert::gray_to_u8;
use color::{PixelMode, Sample};
use raster::pipe::ChannelMask;

use crate::resources::OverprintMode;
use crate::resources::transfer::Transfer;
use crate::separation::Separator;

/// The process channels C, M, Y and K of an ink canvas.
pub(crate) const PROCESS_CHANNELS: ChannelMask = ChannelMask::new(0b1111);

/// The channels of an ink canvas that a set of paints wrote (`paints`),
/// and those in which their colours carry ink (`carries`): a colour's
/// components occupy the channels of [`RasterColor::carries`], and a paint
/// that knocks out writes its other channels as no ink.  A transfer
/// function maps the channels a colour carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InkChannels {
    /// The channels written.
    pub(crate) paints: ChannelMask,
    /// The channels carrying the colours' components.
    pub(crate) carries: ChannelMask,
}

impl InkChannels {
    /// No channel, the channels of no paint.
    pub(crate) const NONE: Self = Self {
        paints: ChannelMask::new(0),
        carries: ChannelMask::new(0),
    };

    /// Every channel, the channels of a paint of sRGB samples, whose
    /// channels are not known.
    pub(crate) const ALL: Self = Self {
        paints: ChannelMask::ALL,
        carries: ChannelMask::ALL,
    };

    /// The channels of the paints of `self` and of `other`.
    #[must_use]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self {
            paints: ChannelMask::new(self.paints.bits() | other.paints.bits()),
            carries: ChannelMask::new(self.carries.bits() | other.carries.bits()),
        }
    }
}

/// The ink a `Separation` or `DeviceN` colour of a given space paints on an
/// ink canvas, fixed for a page once the page's spot slots hold the
/// space's spot colorants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ColorantPlan {
    /// Component `i` paints its tint in the channels of bit mask
    /// `channels[i]`: one bit for a process or slotted spot colorant, every
    /// bit for `/All`, none for `/None`.  A channel several components
    /// paint takes the largest tint.
    Channels(Box<[u8]>),
    /// The space's spot colorants do not all hold a slot: the colour is the
    /// process ink of its alternate colour, a colour that is not process
    /// ink converted by the separator.
    Process(Separator),
}

impl ColorantPlan {
    /// The channels a colour of the space paints.
    pub(crate) fn paints(&self) -> ChannelMask {
        match self {
            Self::Channels(channels) => ChannelMask::new(channels.iter().fold(0, |a, &b| a | b)),
            Self::Process(_) => PROCESS_CHANNELS,
        }
    }

    /// The ink of the colour `comps` ([0, 1] each) of `space`, the space
    /// the plan was made for, converted under `intent`, at the depth of
    /// `S`.
    pub(crate) fn ink<S: Sample>(
        &self,
        doc: &pdf::Document,
        intent: crate::ColourIntent,
        space: &crate::resources::ColorSpace,
        comps: &[f64],
    ) -> [S; 8] {
        match self {
            Self::Channels(channels) => {
                let mut ink = [S::ZERO; 8];
                for (&bits, &tint) in channels.iter().zip(comps) {
                    paint_tint(&mut ink, bits, S::from_unit(tint));
                }
                ink
            }
            Self::Process(separator) => {
                no_spots(space.process_ink::<S>(doc, intent, comps, separator))
            }
        }
    }

    /// The ink of the tints `tints`, one sample per component, under a plan
    /// of [`Self::Channels`]; `None` under [`Self::Process`].
    pub(crate) fn ink_of_tints<S: Sample>(&self, tints: &[S]) -> Option<[S; 8]> {
        let Self::Channels(channels) = self else {
            return None;
        };
        let mut ink = [S::ZERO; 8];
        for (&bits, &tint) in channels.iter().zip(tints) {
            paint_tint(&mut ink, bits, tint);
        }
        Some(ink)
    }
}

/// The process ink `cmyk` as the ink `[C, M, Y, K, S0, S1, S2, S3]`, no
/// spot ink.
pub(crate) fn no_spots<S: Sample>(cmyk: [S; 4]) -> [S; 8] {
    std::array::from_fn(|channel| cmyk.get(channel).copied().unwrap_or(S::ZERO))
}

/// Raise each channel of `ink` in the bit mask `bits` to at least `tint`.
pub(crate) fn paint_tint<S: Sample>(ink: &mut [S; 8], bits: u8, tint: S) {
    for (i, v) in ink.iter_mut().enumerate() {
        if bits & 1 << i != 0 {
            *v = (*v).max(tint);
        }
    }
}

/// Current fill or stroke colour, in the forms the raster pipeline accepts.
///
/// The colour is held as sRGB bytes, painted on an RGB canvas, and as ink
/// amounts `[C, M, Y, K, S0, S1, S2, S3]` (0 = no ink), painted on an ink
/// canvas.  A colour given in a four-component process space keeps its
/// components as the ink, and a colour given in a `Separation` or `DeviceN`
/// space its colorants' tints; any other colour's ink is the page's
/// [`Separator`] of its sRGB bytes, with no spot ink, computed on first use
/// so that a page painted on an RGB canvas never evaluates it.  A colour
/// lives within one page, whose canvas has one separator, so the ink it
/// keeps is that separator's.
///
/// On an ink canvas the colour paints the channels of [`Self::paints`] and
/// keeps the backdrop in the others: the process channels for a
/// four-component process colour, the colorants' channels for a
/// `Separation` or `DeviceN` colour, every channel for any other colour.
///
/// A colour mapped through a transfer function keeps its channels and its
/// overprint fields: its sRGB bytes map through the RGB tables and its ink
/// through the ink tables (see [`Self::transferred`]).
///
/// Beside its 8-bit bytes the colour carries its sRGB value at 16 bits,
/// quantised from the same conversion, for a 16-bit canvas; a transfer
/// function maps it through the function itself, evaluated on first use.
/// Its ink is held at 16 bits too ([`Inks`]): a given ink as the same
/// tints rounded at 16 bits, or the 16-bit solve of a 16-bit page; a
/// derived ink converts from the 16-bit sRGB value on first use.
///
/// `PipeSrc::Solid` takes the pixel's samples and asserts their count
/// equals the pixel's channel count; [`Self::bytes_for`] returns them for
/// an 8-bit mode, [`Self::rgb16`] for a 16-bit RGB pixel and
/// [`Self::ink16`] for a 16-bit ink pixel.
#[derive(Debug, Clone)]
pub(crate) struct RasterColor {
    bytes: [u8; 3],
    /// The 16-bit sRGB value the 16-bit transfer maps from: the colour's
    /// own before any transfer function.
    wide_source: [u16; 3],
    /// The transfer function the 16-bit value maps through.
    wide_transfer: Option<Arc<Transfer>>,
    /// The 16-bit sRGB value: `wide_source` mapped through
    /// `wide_transfer`, computed on first use.
    wide: OnceLock<[u16; 3]>,
    /// The ink the colour was given in, if any.
    given: Option<Inks>,
    /// The sRGB colour a derived ink converts from: the colour before any
    /// transfer function.
    source: Srgb,
    /// The transfer function a derived ink maps through after conversion.
    ink_transfer: Option<Arc<Transfer>>,
    /// The ink amounts: `given`, otherwise derived from `source` on first
    /// use.
    ink: OnceLock<[u8; 8]>,
    /// The 16-bit ink amounts: `given`, otherwise derived from the 16-bit
    /// `source` on first use.
    ink16: OnceLock<[u16; 8]>,
    /// The channels the colour paints on an ink canvas.
    paints: ChannelMask,
    /// Whether nonzero overprint mode (ISO 32000-1 §8.6.7) applies to the
    /// colour: it is a solid colour given in a four-component process
    /// space.
    opm_eligible: bool,
    /// The process channels whose component is exactly 0, bit `i` for
    /// channel `i`; nonzero overprint mode leaves them unpainted.  Zero
    /// for a colour that is not eligible.
    zero_components: u8,
}

impl PartialEq for RasterColor {
    /// The ink is a function of `given`, `source` and `ink_transfer`, so
    /// they decide equality with the bytes, the channel mask and the
    /// overprint fields.  Transfer functions compare by identity.
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.wide_source == other.wide_source
            && same_transfer(self.wide_transfer.as_ref(), other.wide_transfer.as_ref())
            && self.given == other.given
            && self.source == other.source
            && same_transfer(self.ink_transfer.as_ref(), other.ink_transfer.as_ref())
            && self.paints == other.paints
            && self.opm_eligible == other.opm_eligible
            && self.zero_components == other.zero_components
    }
}

impl Eq for RasterColor {}

/// Whether `a` and `b` are the same transfer function, or both none.
fn same_transfer(a: Option<&Arc<Transfer>>, b: Option<&Arc<Transfer>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (a, b) => a.is_none() && b.is_none(),
    }
}

impl std::hash::Hash for RasterColor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
        self.wide_source.hash(state);
        self.wide_transfer.as_ref().map(Arc::as_ptr).hash(state);
        self.given.hash(state);
        self.source.hash(state);
        self.ink_transfer.as_ref().map(Arc::as_ptr).hash(state);
        self.paints.bits().hash(state);
        self.opm_eligible.hash(state);
        self.zero_components.hash(state);
    }
}

impl RasterColor {
    /// A colour of sRGB `rgb` whose ink derives from its bytes; it paints
    /// every channel.
    const fn derived(rgb: Srgb) -> Self {
        Self {
            bytes: rgb.bytes,
            wide_source: rgb.wide,
            wide_transfer: None,
            wide: OnceLock::new(),
            given: None,
            source: rgb,
            ink_transfer: None,
            ink: OnceLock::new(),
            ink16: OnceLock::new(),
            paints: ChannelMask::ALL,
            opm_eligible: false,
            zero_components: 0,
        }
    }

    /// Grey [0, 1] — expanded to equal RGB components.
    #[must_use]
    pub(crate) fn gray(g: f64) -> Self {
        Self::derived(Srgb::from_unit([g; 3]))
    }

    /// RGB [0, 1] each.
    #[must_use]
    pub(crate) fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::derived(Srgb::from_unit([r, g, b]))
    }

    /// `DeviceCMYK` [0, 1] each — converted with [`color::cmyk::cmyk_f64_to_rgb`];
    /// the components are the ink.
    #[must_use]
    pub(crate) fn cmyk(c: f64, m: f64, y: f64, k: f64) -> Self {
        let rgb = Srgb {
            bytes: color::cmyk::cmyk_f64_to_rgb(c, m, y, k),
            wide: color::cmyk::cmyk_f64_to_rgb_at(c, m, y, k),
        };
        Self::with_process_inks(rgb, [c, m, y, k])
    }

    /// The colour of sRGB `rgb` given in a process space as the inks `cmyk`
    /// ([0, 1] each, clamped; NaN is no ink), which are the colour's ink.
    /// It paints the process channels and is eligible for nonzero
    /// overprint mode, which leaves the channels of its zero components
    /// unpainted.
    #[must_use]
    pub(crate) fn with_process_inks(rgb: Srgb, cmyk: [f64; 4]) -> Self {
        let [c, m, y, k] = cmyk;
        Self::with_process_ink_of(rgb, cmyk, Inks::from_unit([c, m, y, k, 0.0, 0.0, 0.0, 0.0]))
    }

    /// The colour of sRGB `rgb` given in a process space as the components
    /// `cmyk`, printed as the process ink `ink`, whose spot channels hold
    /// no ink: like [`Self::with_process_inks`], with the ink in place of
    /// the components.  The channels it paints, under either overprint
    /// mode, follow the components.
    #[must_use]
    pub(crate) fn with_process_ink_of(rgb: Srgb, cmyk: [f64; 4], ink: Inks) -> Self {
        let zero_components = cmyk
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v.is_nan() || v <= 0.0)
            .fold(0u8, |bits, (i, _)| bits | 1 << i);
        Self {
            opm_eligible: true,
            zero_components,
            ..Self::with_ink(rgb, ink, PROCESS_CHANNELS)
        }
    }

    /// The colour of sRGB `rgb` whose ink is `ink` and which paints the
    /// channels `paints`: a `Separation` or `DeviceN` colour, given as its
    /// colorants' tints.
    #[must_use]
    pub(crate) fn with_ink(rgb: Srgb, ink: Inks, paints: ChannelMask) -> Self {
        Self {
            bytes: rgb.bytes,
            wide_source: rgb.wide,
            wide_transfer: None,
            wide: OnceLock::new(),
            given: Some(ink),
            source: rgb,
            ink_transfer: None,
            ink: OnceLock::from(ink.bytes),
            ink16: OnceLock::from(ink.wide),
            paints,
            opm_eligible: false,
            zero_components: 0,
        }
    }

    /// The colour of an sRGB value already converted from its colour
    /// space (e.g. by `ColorSpace::convert_to_srgb`).
    #[must_use]
    pub(crate) const fn from_srgb(rgb: Srgb) -> Self {
        Self::derived(rgb)
    }

    /// The colour of the sRGB bytes `rgb`, its 16-bit value their exact
    /// widening.
    #[must_use]
    pub(crate) fn from_bytes(rgb: [u8; 3]) -> Self {
        Self::derived(Srgb::of_bytes(rgb))
    }

    /// The RGB bytes.
    #[must_use]
    pub(crate) const fn bytes(&self) -> [u8; 3] {
        self.bytes
    }

    /// The 16-bit sRGB value, mapped through the transfer function the
    /// colour was mapped through.
    #[must_use]
    pub(crate) fn rgb16(&self) -> &[u16; 3] {
        self.wide.get_or_init(|| {
            self.wide_transfer
                .as_ref()
                .map_or(self.wide_source, |t| t.apply16(self.wide_source))
        })
    }

    /// The ink amounts `[C, M, Y, K, S0, S1, S2, S3]`; a derived ink is
    /// `separator`'s conversion of the colour.
    #[must_use]
    pub(crate) fn ink(&self, separator: &Separator) -> &[u8; 8] {
        self.ink.get_or_init(|| {
            let [c, m, y, k] = separator.ink(self.source.bytes);
            let ink = [c, m, y, k, 0, 0, 0, 0];
            self.ink_transfer
                .as_ref()
                .map_or(ink, |t| t.apply_ink(ink, PROCESS_CHANNELS))
        })
    }

    /// The 16-bit ink amounts `[C, M, Y, K, S0, S1, S2, S3]`: a derived
    /// ink converts from the 16-bit colour by `separator` and maps through
    /// the transfer function at 16 bits.
    #[must_use]
    pub(crate) fn ink16(&self, separator: &Separator) -> &[u16; 8] {
        self.ink16.get_or_init(|| {
            let [c, m, y, k] = separator.ink(self.source.wide);
            let ink = [c, m, y, k, 0, 0, 0, 0];
            self.ink_transfer
                .as_ref()
                .map_or(ink, |t| t.apply_ink16(ink, PROCESS_CHANNELS))
        })
    }

    /// The channels the colour paints on an ink canvas.
    #[must_use]
    pub(crate) const fn paints(&self) -> ChannelMask {
        self.paints
    }

    /// The channels that carry the colour's components on an ink canvas:
    /// those it paints for a colour given as ink, the process channels for
    /// a derived ink, which holds no spot ink.
    #[must_use]
    pub(crate) const fn carries(&self) -> ChannelMask {
        if self.given.is_some() {
            self.paints
        } else {
            PROCESS_CHANNELS
        }
    }

    /// Whether nonzero overprint mode applies to the colour.
    #[must_use]
    pub(crate) const fn opm_eligible(&self) -> bool {
        self.opm_eligible
    }

    /// The channels the colour paints on an ink canvas under overprint
    /// mode `mode` (ISO 32000-1 §8.6.7): under the nonzero mode an
    /// eligible colour leaves unpainted the process channels whose
    /// component is exactly 0.
    #[must_use]
    pub(crate) const fn paints_with_opm(&self, mode: OverprintMode) -> ChannelMask {
        if matches!(mode, OverprintMode::NonZero) && self.opm_eligible() {
            ChannelMask::new(self.paints().bits() & !self.zero_components)
        } else {
            self.paints()
        }
    }

    /// The colour's bytes for a pixel of `mode`: the ink amounts for
    /// [`PixelMode::DeviceN8`], a derived ink by `separator`, the RGB
    /// bytes for every other mode.
    #[must_use]
    pub(crate) fn bytes_for(&self, mode: PixelMode, separator: &Separator) -> &[u8] {
        match mode {
            PixelMode::DeviceN8 => self.ink(separator),
            _ => &self.bytes,
        }
    }

    /// The colour mapped through the transfer function `transfer`: its
    /// bytes through the RGB tables, its ink through the ink tables — a
    /// derived ink after its conversion from the unmapped bytes.  The
    /// channels it paints and its overprint fields are the colour's own:
    /// nonzero overprint mode reads the components the colour was given
    /// in (ISO 32000-2 §8.6.7), not the device values the function makes
    /// of them.
    #[must_use]
    pub(crate) fn transferred(&self, transfer: Option<&Arc<Transfer>>) -> Self {
        let Some(t) = transfer else {
            return self.clone();
        };
        let bytes = t.apply(self.bytes);
        let wide = Self {
            bytes,
            wide_source: *self.rgb16(),
            wide_transfer: Some(Arc::clone(t)),
            wide: OnceLock::new(),
            ..self.clone()
        };
        let Some(given) = self.given else {
            return Self {
                given: None,
                source: self.source,
                ink_transfer: Some(Arc::clone(t)),
                ink: OnceLock::new(),
                ink16: OnceLock::new(),
                ..wide
            };
        };
        let ink = Inks {
            bytes: t.apply_ink(given.bytes, self.carries()),
            wide: t.apply_ink16(given.wide, self.carries()),
        };
        Self {
            given: Some(ink),
            source: Srgb {
                bytes,
                wide: *wide.rgb16(),
            },
            ink_transfer: None,
            ink: OnceLock::from(ink.bytes),
            ink16: OnceLock::from(ink.wide),
            ..wide
        }
    }

    /// Raw bytes slice for `PipeSrc::Solid` on an RGB canvas.
    ///
    /// Always 3 bytes (RGB), matching `Rgb8::BYTES`.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

/// An sRGB colour at both sample depths: its 8-bit bytes and its 16-bit
/// value, each quantised from the same conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Srgb {
    /// The 8-bit bytes.
    pub bytes: [u8; 3],
    /// The 16-bit value.
    pub wide: [u16; 3],
}

/// Ink amounts `[C, M, Y, K, S0, S1, S2, S3]` (0 = no ink) at both sample
/// depths: each quantised from the same tints, or one depth's own and the
/// other's taken from it ([`Self::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Inks {
    /// The 8-bit amounts.
    pub bytes: [u8; 8],
    /// The 16-bit amounts.
    pub wide: [u16; 8],
}

impl Inks {
    /// The tints `inks` ([0, 1] each, clamped; NaN is no ink), rounded at
    /// either depth.
    #[must_use]
    pub(crate) fn from_unit(inks: [f64; 8]) -> Self {
        Self {
            bytes: inks.map(gray_to_u8),
            wide: inks.map(u16::from_unit),
        }
    }

    /// The process ink `cmyk` at the depth of `S`, no spot ink, as
    /// [`Self::of`] takes it.
    #[must_use]
    pub(crate) fn process<S: Sample>(cmyk: [S; 4]) -> Self {
        Self::of(no_spots(cmyk))
    }

    /// The amounts `ink` at the depth of `S`, the other depth's the exact
    /// widening of 8-bit amounts or the nearest bytes to 16-bit ones.
    #[must_use]
    pub(crate) fn of<S: Sample>(ink: [S; 8]) -> Self {
        Self {
            bytes: ink.map(S::to_u8),
            wide: ink.map(u16::from_sample),
        }
    }
}

impl Srgb {
    /// The colour of the bytes `rgb`, its 16-bit value their exact
    /// widening.
    #[must_use]
    pub(crate) fn of_bytes(rgb: [u8; 3]) -> Self {
        Self {
            bytes: rgb,
            wide: rgb.map(u16::from_u8),
        }
    }

    /// The colour of the normalised components `rgb`, each clamped to
    /// `[0, 1]` and rounded at either depth.
    #[must_use]
    pub(crate) fn from_unit(rgb: [f64; 3]) -> Self {
        Self {
            bytes: rgb.map(gray_to_u8),
            wide: rgb.map(u16::from_unit),
        }
    }

    /// The colour at the depth of `S`.
    #[must_use]
    pub(crate) fn at<S: Sample>(&self) -> [S; 3] {
        match S::from_u8_slice(&self.bytes) {
            Some(&[r, g, b]) => [r, g, b],
            _ => self
                .wide
                .map(|v| S::from_unit(f64::from(v) / f64::from(u16::MAX))),
        }
    }
}

impl Default for RasterColor {
    /// Default fill/stroke colour is black (PDF §8.4.4 initial value).
    fn default() -> Self {
        Self::gray(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_color_rgb_slice() {
        let c = RasterColor::rgb(1.0, 0.0, 0.0);
        assert_eq!(c.as_slice(), &[255, 0, 0]);
    }

    #[test]
    fn raster_color_carries_its_16_bit_value() {
        assert_eq!(RasterColor::gray(0.5).rgb16(), &[32_768; 3]);
        assert_eq!(
            RasterColor::rgb(1.0, 0.25, 0.0).rgb16(),
            &[65_535, 16_384, 0]
        );
        let cmyk = RasterColor::cmyk(0.2, 0.5, 0.1, 0.3);
        for (w, n) in cmyk.rgb16().iter().zip(cmyk.bytes()) {
            assert!(
                w.to_u8().abs_diff(n) <= 1,
                "{:?} {:?}",
                cmyk.rgb16(),
                cmyk.bytes()
            );
        }
        assert_ne!(RasterColor::gray(0.5), RasterColor::gray(0.501));
    }

    #[test]
    fn raster_color_gray_expands_to_rgb() {
        // Gray must always produce a 3-byte slice for Rgb8 pipeline compatibility.
        let black = RasterColor::gray(0.0);
        assert_eq!(black.as_slice(), &[0, 0, 0]);
        let white = RasterColor::gray(1.0);
        assert_eq!(white.as_slice(), &[255, 255, 255]);
    }

    #[test]
    fn raster_color_ink_is_given_cmyk_or_derived() {
        let c = RasterColor::cmyk(0.2, 0.5, 0.1, 0.3);
        assert_eq!(
            c.bytes_for(PixelMode::DeviceN8, &Separator::Model),
            &[51, 128, 26, 77, 0, 0, 0, 0]
        );
        assert_eq!(
            c.bytes_for(PixelMode::Rgb8, &Separator::Model),
            c.as_slice()
        );
        let white = RasterColor::gray(1.0);
        assert_eq!(white.ink(&Separator::Model), &[0; 8]);
        // A grey prints in black ink alone, the black printing that grey.
        let grey = RasterColor::gray(0.2);
        let grey = grey.ink(&Separator::Model);
        assert_eq!(grey[..3], [0, 0, 0]);
        assert_eq!(grey[4..], [0; 4]);
        assert!(color::cmyk::cmyk_to_rgb(0, 0, 0, grey[3])[0].abs_diff(51) <= 1);
        // Equal RGB given as process inks differs from the derived colour.
        let given = RasterColor::with_process_inks(Srgb::of_bytes([255; 3]), [0.0; 4]);
        assert_eq!(given.ink(&Separator::Model), white.ink(&Separator::Model));
        assert_ne!(given, white);
    }

    #[test]
    fn raster_color_paints_its_space_channels() {
        let process = RasterColor::cmyk(0.2, 0.0, 0.6, 0.0);
        assert_eq!(process.paints(), PROCESS_CHANNELS);
        assert!(process.opm_eligible());
        let spot = RasterColor::with_ink(
            Srgb::of_bytes([200, 30, 60]),
            Inks::of::<u8>([0, 0, 0, 0, 153, 0, 0, 0]),
            ChannelMask::new(0b1_0000),
        );
        assert_eq!(spot.paints(), ChannelMask::new(0b1_0000));
        assert!(!spot.opm_eligible());
        assert_eq!(spot.ink(&Separator::Model), &[0, 0, 0, 0, 153, 0, 0, 0]);
        // The same ink in a different channel is a different colour.
        assert_ne!(
            spot,
            RasterColor::with_ink(
                Srgb::of_bytes([200, 30, 60]),
                Inks::of::<u8>([0, 0, 0, 0, 153, 0, 0, 0]),
                ChannelMask::new(0b1_0001)
            )
        );
        for derived in [
            RasterColor::gray(0.3),
            RasterColor::rgb(0.1, 0.2, 0.3),
            RasterColor::from_bytes([1, 2, 3]),
        ] {
            assert_eq!(derived.paints(), ChannelMask::ALL);
            assert!(!derived.opm_eligible());
        }
    }

    #[test]
    fn raster_color_nonzero_overprint_mode_skips_zero_components() {
        let black = RasterColor::cmyk(0.0, 0.0, 0.0, 1.0);
        assert_eq!(black.paints_with_opm(OverprintMode::Zero), PROCESS_CHANNELS);
        assert_eq!(
            black.paints_with_opm(OverprintMode::NonZero),
            ChannelMask::new(0b1000)
        );
        // A component however small is painted; only an exact 0 is not.
        let faint = RasterColor::cmyk(0.001, 0.0, 0.5, 0.0);
        assert_eq!(faint.ink(&Separator::Model)[0], 0);
        assert_eq!(
            faint.paints_with_opm(OverprintMode::NonZero),
            ChannelMask::new(0b0101)
        );
        assert_ne!(faint, RasterColor::cmyk(0.0, 0.0, 0.5, 0.0));
        // A component that clamps to no ink counts as zero.
        let clamped = RasterColor::cmyk(-0.25, f64::NAN, 0.5, 0.0);
        assert_eq!(clamped.ink(&Separator::Model)[..2], [0, 0]);
        assert_eq!(
            clamped.paints_with_opm(OverprintMode::NonZero),
            ChannelMask::new(0b0100)
        );
        assert_eq!(clamped, RasterColor::cmyk(0.0, 0.0, 0.5, 0.0));
        // Only a process colour is eligible.
        let spot = RasterColor::with_ink(
            Srgb::of_bytes([200, 30, 60]),
            Inks::of::<u8>([0; 8]),
            ChannelMask::new(0b1_0000),
        );
        assert_eq!(
            spot.paints_with_opm(OverprintMode::NonZero),
            ChannelMask::new(0b1_0000)
        );
        assert_eq!(
            RasterColor::gray(1.0).paints_with_opm(OverprintMode::NonZero),
            ChannelMask::ALL
        );
    }

    #[test]
    fn raster_color_transferred_keeps_its_channels() {
        use crate::resources::transfer::{TransferSetting, ext_gstate_transfer};

        // The inverting transfer function.
        let doc = crate::test_helpers::empty_doc();
        let gs = pdf::parse_object(
            b"<< /TR << /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >> >>",
            &mut 0,
        )
        .expect("the dictionary parses");
        let pdf::Object::Dictionary(gs) = gs else {
            panic!("a dictionary");
        };
        let Some(TransferSetting::Functions(transfer)) = ext_gstate_transfer(&doc, &gs) else {
            panic!("a transfer function");
        };
        let spot = RasterColor::with_ink(
            Srgb::of_bytes([200, 30, 60]),
            Inks::of::<u8>([0, 0, 0, 0, 153, 0, 0, 0]),
            ChannelMask::new(0b1_0000),
        );
        let mapped = spot.transferred(Some(&transfer));
        assert_eq!(mapped.bytes(), [55, 225, 195]);
        assert_eq!(mapped.paints(), ChannelMask::new(0b1_0000));
        assert_eq!(mapped.ink(&Separator::Model), &[0, 0, 0, 0, 102, 0, 0, 0]);
        // The 16-bit value maps through the function at 16 bits.
        assert_eq!(mapped.rgb16(), &transfer.apply16(*spot.rgb16()));
        assert_eq!(spot.rgb16(), &[200, 30, 60].map(u16::from_u8));
        assert_eq!(spot.transferred(None), spot);
        // A process colour keeps its eligibility and its zero components.
        let black = RasterColor::cmyk(0.0, 0.0, 0.0, 1.0).transferred(Some(&transfer));
        assert_eq!(black.ink(&Separator::Model)[..4], [255, 255, 255, 0]);
        assert_eq!(
            black.paints_with_opm(OverprintMode::NonZero),
            ChannelMask::new(0b1000)
        );
        // A derived ink converts from the unmapped bytes, then maps.
        let grey = RasterColor::gray(0.4);
        let mapped = grey.transferred(Some(&transfer));
        assert_eq!(mapped.paints(), ChannelMask::ALL);
        assert_eq!(
            mapped.ink(&Separator::Model),
            &transfer.apply_ink(*grey.ink(&Separator::Model), PROCESS_CHANNELS)
        );
        assert_eq!(
            mapped.ink(&Separator::Model)[4..],
            [0; 4],
            "a derived ink holds no spot ink"
        );
    }

    #[test]
    fn raster_color_default_is_black() {
        let c = RasterColor::default();
        assert_eq!(c.as_slice(), &[0, 0, 0]);
    }
}
