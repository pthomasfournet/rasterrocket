//! PDF colour space and colour value conversions.
//!
//! Maps PDF colour operator values (f64 in `[0,1]`) to raw u8 pixel components
//! understood by the raster crate.
//!
//! The pure conversion functions live in the `color` crate: `gray_to_u8` and
//! `rgb_to_bytes` in `color::convert` (re-exported here), `DeviceCMYK` in
//! `color::cmyk`.

use std::sync::OnceLock;

use color::PixelMode;
pub(crate) use color::convert::{gray_to_u8, rgb_to_bytes};
use raster::pipe::ChannelMask;

use crate::resources::transfer::Transfer;

/// The process channels C, M, Y and K of an ink canvas.
pub(crate) const PROCESS_CHANNELS: ChannelMask = ChannelMask::new(0b1111);

/// Current fill or stroke colour, in the forms the raster pipeline accepts.
///
/// The colour is held as sRGB bytes, painted on an RGB canvas, and as ink
/// amounts `[C, M, Y, K, S0, S1, S2, S3]` (0 = no ink), painted on an ink
/// canvas.  A colour given in a four-component process space keeps its
/// components as the ink, and a colour given in a `Separation` or `DeviceN`
/// space its colorants' tints; any other colour's ink is
/// [`color::cmyk::rgb_to_ink`] of its sRGB bytes — black alone for a grey,
/// the nearest in-gamut CMY otherwise — with no spot ink, computed on first
/// use so that a page painted on an RGB canvas never evaluates it.
///
/// On an ink canvas the colour paints the channels of [`Self::paints`] and
/// keeps the backdrop in the others: the process channels for a
/// four-component process colour, the colorants' channels for a
/// `Separation` or `DeviceN` colour, every channel for any other colour.
///
/// `PipeSrc::Solid` takes `&[u8]` and asserts its length equals the pixel
/// component count; [`Self::bytes_for`] returns the slice for a mode.
#[derive(Debug, Clone)]
pub(crate) struct RasterColor {
    bytes: [u8; 3],
    /// The ink the colour was given in, if any.
    given: Option<[u8; 8]>,
    /// The ink amounts: `given`, otherwise derived from `bytes` on first
    /// use.
    ink: OnceLock<[u8; 8]>,
    /// The channels the colour paints on an ink canvas.
    paints: ChannelMask,
    /// Whether nonzero overprint mode (PDF §8.6.7) applies to the colour:
    /// it is a solid colour given in a four-component process space.
    opm_eligible: bool,
}

impl PartialEq for RasterColor {
    /// The ink is a function of `bytes` and `given`, so they decide
    /// equality with the channel mask and the overprint eligibility.
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.given == other.given
            && self.paints == other.paints
            && self.opm_eligible == other.opm_eligible
    }
}

impl Eq for RasterColor {}

impl std::hash::Hash for RasterColor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bytes.hash(state);
        self.given.hash(state);
        self.paints.bits().hash(state);
        self.opm_eligible.hash(state);
    }
}

impl RasterColor {
    /// A colour of sRGB `bytes` whose ink derives from them; it paints
    /// every channel.
    const fn derived(bytes: [u8; 3]) -> Self {
        Self {
            bytes,
            given: None,
            ink: OnceLock::new(),
            paints: ChannelMask::ALL,
            opm_eligible: false,
        }
    }

    /// Grey [0, 1] — expanded to equal RGB components.
    #[must_use]
    pub(crate) fn gray(g: f64) -> Self {
        let v = gray_to_u8(g);
        Self::derived([v, v, v])
    }

    /// RGB [0, 1] each.
    #[must_use]
    pub(crate) fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::derived(rgb_to_bytes(r, g, b))
    }

    /// `DeviceCMYK` [0, 1] each — converted with [`color::cmyk::cmyk_f64_to_rgb`];
    /// the components are the ink.
    #[must_use]
    pub(crate) fn cmyk(c: f64, m: f64, y: f64, k: f64) -> Self {
        Self::with_process_inks(color::cmyk::cmyk_f64_to_rgb(c, m, y, k), [c, m, y, k])
    }

    /// The colour of sRGB `rgb` given in a process space as the inks `cmyk`
    /// ([0, 1] each, clamped; NaN is no ink), which are the colour's ink.
    /// It paints the process channels and is eligible for nonzero
    /// overprint mode.
    #[must_use]
    pub(crate) fn with_process_inks(rgb: [u8; 3], cmyk: [f64; 4]) -> Self {
        let [c, m, y, k] = cmyk.map(gray_to_u8);
        Self {
            opm_eligible: true,
            ..Self::with_ink(rgb, [c, m, y, k, 0, 0, 0, 0], PROCESS_CHANNELS)
        }
    }

    /// The colour of sRGB `rgb` whose ink is `ink` and which paints the
    /// channels `paints`: a `Separation` or `DeviceN` colour, given as its
    /// colorants' tints.
    #[must_use]
    pub(crate) fn with_ink(rgb: [u8; 3], ink: [u8; 8], paints: ChannelMask) -> Self {
        Self {
            bytes: rgb,
            given: Some(ink),
            ink: OnceLock::from(ink),
            paints,
            opm_eligible: false,
        }
    }

    /// Construct directly from a pre-converted RGB byte triple. Useful when
    /// the caller has already done a colour-space conversion (e.g.
    /// `ColorSpace::convert_to_rgb`) and would otherwise round-trip through
    /// `f64 / 255.0` → `rgb_to_bytes` for no reason.
    #[must_use]
    pub(crate) const fn from_bytes(rgb: [u8; 3]) -> Self {
        Self::derived(rgb)
    }

    /// The RGB bytes.
    #[must_use]
    pub(crate) const fn bytes(&self) -> [u8; 3] {
        self.bytes
    }

    /// The ink amounts `[C, M, Y, K, S0, S1, S2, S3]`.
    #[must_use]
    pub(crate) fn ink(&self) -> &[u8; 8] {
        self.ink.get_or_init(|| {
            let [c, m, y, k] = color::cmyk::rgb_to_ink(self.bytes);
            [c, m, y, k, 0, 0, 0, 0]
        })
    }

    /// The channels the colour paints on an ink canvas.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read when a paint's channel mask is built")
    )]
    #[must_use]
    pub(crate) const fn paints(&self) -> ChannelMask {
        self.paints
    }

    /// Whether nonzero overprint mode applies to the colour.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read when a paint's channel mask is built")
    )]
    #[must_use]
    pub(crate) const fn opm_eligible(&self) -> bool {
        self.opm_eligible
    }

    /// The colour's bytes for a pixel of `mode`: the ink amounts for
    /// [`PixelMode::DeviceN8`], the RGB bytes for every other mode.
    #[must_use]
    pub(crate) fn bytes_for(&self, mode: PixelMode) -> &[u8] {
        match mode {
            PixelMode::DeviceN8 => self.ink(),
            _ => &self.bytes,
        }
    }

    /// The colour mapped through the RGB transfer tables `transfer`.  A
    /// mapped colour is in no colour space of its own: its ink derives
    /// from its mapped RGB bytes and it paints every channel.
    #[must_use]
    pub(crate) fn transferred(&self, transfer: Option<&Transfer>) -> Self {
        transfer.map_or_else(|| self.clone(), |t| Self::derived(t.apply(self.bytes)))
    }

    /// Raw bytes slice for `PipeSrc::Solid` on an RGB canvas.
    ///
    /// Always 3 bytes (RGB), matching `Rgb8::BYTES`.
    #[must_use]
    pub(crate) const fn as_slice(&self) -> &[u8] {
        &self.bytes
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
            c.bytes_for(PixelMode::DeviceN8),
            &[51, 128, 26, 77, 0, 0, 0, 0]
        );
        assert_eq!(c.bytes_for(PixelMode::Rgb8), c.as_slice());
        let white = RasterColor::gray(1.0);
        assert_eq!(white.ink(), &[0; 8]);
        // A grey prints in black ink alone.
        assert_eq!(RasterColor::gray(0.2).ink(), &[0, 0, 0, 204, 0, 0, 0, 0]);
        // Equal RGB given as process inks differs from the derived colour.
        let given = RasterColor::with_process_inks([255; 3], [0.0; 4]);
        assert_eq!(given.ink(), white.ink());
        assert_ne!(given, white);
    }

    #[test]
    fn raster_color_paints_its_space_channels() {
        let process = RasterColor::cmyk(0.2, 0.0, 0.6, 0.0);
        assert_eq!(process.paints(), PROCESS_CHANNELS);
        assert!(process.opm_eligible());
        let spot = RasterColor::with_ink(
            [200, 30, 60],
            [0, 0, 0, 0, 153, 0, 0, 0],
            ChannelMask::new(0b1_0000),
        );
        assert_eq!(spot.paints(), ChannelMask::new(0b1_0000));
        assert!(!spot.opm_eligible());
        assert_eq!(spot.ink(), &[0, 0, 0, 0, 153, 0, 0, 0]);
        // The same ink in a different channel is a different colour.
        assert_ne!(
            spot,
            RasterColor::with_ink(
                [200, 30, 60],
                [0, 0, 0, 0, 153, 0, 0, 0],
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
    fn raster_color_transferred_is_derived() {
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
            [200, 30, 60],
            [0, 0, 0, 0, 153, 0, 0, 0],
            ChannelMask::new(0b1_0000),
        );
        let mapped = spot.transferred(Some(&transfer));
        assert_eq!(mapped.bytes(), [55, 225, 195]);
        assert_eq!(mapped.paints(), ChannelMask::ALL);
        assert!(!mapped.opm_eligible());
        assert_eq!(
            mapped.ink(),
            &RasterColor::from_bytes([55, 225, 195]).ink()[..]
        );
        assert_eq!(spot.transferred(None), spot);
    }

    #[test]
    fn raster_color_default_is_black() {
        let c = RasterColor::default();
        assert_eq!(c.as_slice(), &[0, 0, 0]);
    }
}
