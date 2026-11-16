//! Colour-value conversion through an ICC profile stream (PDF §8.6.5.5).
//!
//! A profile stream is parsed once per document into an [`IccProfile`],
//! which compiles an [`IccConversion`] per device-to-PCS model its
//! rendering intents select (ISO 32000-1 §8.6.5.8): the `AToB0`, `AToB1`
//! or `AToB2` table of the intent (ICC.1:2022 §8.10.2, `AToB0` standing in
//! for a table the profile lacks), or, in a profile without such tables,
//! its matrix/TRC model.  Each model compiles, per working space, at most
//! once in each form (media-relative, ICC-absolute, black-point
//! compensated); intents that select the same model share one conversion.
//!
//! A gray or RGB matrix/TRC model compiles into the shared shaper tables of
//! `super::super::icc_shaper` for sRGB; for the wide space an RGB one
//! compiles into a 16-bit [`LinearRgb`] through the composite of
//! [`crate::working_space::wide_composite`], and a gray one, whose colours
//! are neutral, into the sRGB shaper, a neutral having the same value in
//! both spaces.  The device → PCS table of a three- or four-channel
//! profile compiles into a grid over the device cube evaluated in floating
//! point and interpolated in the simplices of its cells (see [`table`]),
//! which serves both depths.  Any other 3- or 4-channel conversion (an
//! RGB matrix/TRC model in a form the shaper does not take, a table the
//! evaluator declines) compiles into a `moxcms` 8-bit
//! transform to the working space, beside which a 16-bit transform is
//! created on first use by a 16-bit render; ICC-absolute colorimetry there
//! scales the PCS by the source's media white (see [`crate::pcs`]), folded
//! into the working-space destination's colorants.  A profile that admits neither
//! compiles to its `/Alternate` space when that space takes the same
//! number of components (§8.6.5.5, Table 66).  Converting a colour value
//! is then a table lookup, one transform call, or a conversion in the
//! alternate space.

use std::sync::{Arc, OnceLock};

use color::Sample;
#[cfg(feature = "gpu-icc")]
use color::grid::CmykGrid;
use moxcms::{
    ColorProfile, Layout, Transform8BitExecutor, Transform16BitExecutor, TransformOptions,
};
use pdf::{Document, ObjectId};

use super::ColorSpace;
use crate::resources::icc_shaper::{ProfileConversion, eval_trc, fetch_profile_bytes};
use crate::working_space::{LinearRgb, wide_composite};
use crate::{ColourIntent, ColourTarget, RenderingIntent, WorkingSpace};

mod table;

use table::PcsAdjustment;

/// The device-to-PCS model of a profile: its matrix/TRC model, or one of
/// its `AToB` tables.  The discriminant indexes the slots of a [`Form`].
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

    /// The rendering intent whose table the model is.  The matrix/TRC
    /// model, of every intent, is taken as media-relative.
    const fn intent(self) -> moxcms::RenderingIntent {
        match self {
            Self::Matrix | Self::AToB1 => moxcms::RenderingIntent::RelativeColorimetric,
            Self::AToB0 => moxcms::RenderingIntent::Perceptual,
            Self::AToB2 => moxcms::RenderingIntent::Saturation,
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

/// The forms in which an [`IccProfile`] compiles each [`Model`]: plain,
/// ICC-absolute, and with black-point compensation.  The conversion of
/// model `m` in form `f` to working space `w` is held in slot
/// `(w · FORMS + f) · MODELS + m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Media-relative, the model's PCS converted as it is.
    Plain = 0,
    /// ICC-absolute: the PCS scaled by the media white.
    Absolute = 1,
    /// Media-relative with the model's black point mapped to sRGB black.
    Compensated = 2,
}

/// Number of models.
const MODELS: usize = 4;

/// Number of forms.
const FORMS: usize = 3;

/// Number of conversions an [`IccProfile`] holds: every [`Model`] in
/// every [`Form`] to every working space.
const SLOTS: usize = WorkingSpace::ALL.len() * FORMS * MODELS;

/// The largest black-point luminance compensation leaves alone, that of
/// `L* = 0.1`: the resolution at which a black point is found, so a black
/// within it is the PCS origin and the compensated conversion the plain
/// one.
const BLACK_EPSILON: f64 = 0.1 / 903.3;

/// One ICC profile stream, parsed, and the conversions of its colour
/// values compiled for the rendering intents that use it.
pub(crate) struct IccProfile {
    /// The component count of the colour values converted.
    n: u8,
    /// The parsed profile; `None` when the stream is missing or is not a
    /// profile.
    profile: Option<Arc<ColorProfile>>,
    /// The profile's [`crate::pcs::tag_fingerprint`].
    fingerprint: u64,
    /// The PCS scale of ICC-absolute colorimetry (see
    /// [`crate::pcs::scale_from_white`]); `None` when the media white is
    /// D50, under which ICC-absolute is media-relative colorimetry.
    absolute: Option<[f64; 3]>,
    /// The model each intent selects, by [`RenderingIntent::index`].
    models: [Model; 4],
    /// The black-point luminance of each model as a conversion source (see
    /// [`crate::pcs::black::source_black`]), detected on first use.
    blacks: [OnceLock<f64>; MODELS],
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
        let models = RenderingIntent::ALL.map(|intent| {
            profile
                .as_ref()
                .map_or(Model::Matrix, |p| Model::of(p, intent))
        });
        Self {
            n,
            profile: profile.map(Arc::new),
            fingerprint: bytes.map_or(0, crate::pcs::tag_fingerprint),
            absolute,
            models,
            blacks: Default::default(),
            conversions: Default::default(),
        }
    }

    /// The slot of the conversion to `target`: the model the rendering
    /// intent selects, ICC-absolute when the intent is and the media white
    /// is not D50, compensated when compensation is on, the intent is not
    /// absolute and the model's black is not the PCS origin, else plain,
    /// into the target's working space.
    fn slot(&self, target: ColourTarget) -> usize {
        let intent = target.intent;
        let model = self.models[intent.intent.index()];
        let form = match intent.intent {
            RenderingIntent::AbsoluteColorimetric if self.absolute.is_some() => Form::Absolute,
            RenderingIntent::AbsoluteColorimetric => Form::Plain,
            _ if intent.bpc && self.black(model) > BLACK_EPSILON => Form::Compensated,
            _ => Form::Plain,
        };
        (target.space.index() * FORMS + form as usize) * MODELS + model as usize
    }

    /// The black-point luminance of `model` as a conversion source,
    /// detected on first use; 0 without a profile, or when detection fails,
    /// which is logged.
    fn black(&self, model: Model) -> f64 {
        *self.blacks[model as usize].get_or_init(|| {
            let Some(profile) = &self.profile else {
                return 0.0;
            };
            crate::pcs::black::source_black(profile, model.intent())
                .map_err(|e| {
                    log::warn!(
                        "colorspace: ICC black point not found, converting without \
                         black-point compensation: {e}"
                    );
                })
                .unwrap_or(0.0)
        })
    }

    /// The conversion of the profile's colour values to `target`,
    /// compiled on first use; `alternate` supplies the `/Alternate` space
    /// a profile without a conversion of its own converts through.
    fn conversion(
        &self,
        target: ColourTarget,
        alternate: impl FnOnce() -> Option<ColorSpace>,
    ) -> Option<Arc<IccConversion>> {
        let slot = self.slot(target);
        self.conversions[slot]
            .get_or_init(|| {
                self.compile(slot)
                    .or_else(|| {
                        alternate().map(|space| IccConversion::Alternate(space, target.space))
                    })
                    .map(Arc::new)
            })
            .clone()
    }

    /// Compile the conversion of `slot` from the profile.  `None` when
    /// there is no profile, `n` has no layout, or the converter refuses the
    /// transform.
    fn compile(&self, slot: usize) -> Option<IccConversion> {
        let profile = self.profile.as_ref()?;
        let model = Model::ALL[slot % MODELS];
        let form = [Form::Plain, Form::Absolute, Form::Compensated][slot / MODELS % FORMS];
        let space = WorkingSpace::ALL[slot / (MODELS * FORMS)];
        // A three- or four-channel device → PCS table is tabulated here:
        // the converter interpolates such a table multilinearly.
        if model != Model::Matrix && matches!(self.n, 3 | 4) {
            let adjustment = match form {
                Form::Plain => PcsAdjustment::None,
                Form::Absolute => PcsAdjustment::Absolute(self.absolute?),
                Form::Compensated => PcsAdjustment::Compensated(self.black(model)),
            };
            let tabulated = if self.n == 3 {
                tabulated::<3>(profile, model.intent(), adjustment, space)
            } else {
                tabulated::<4>(profile, model.intent(), adjustment, space)
            };
            if tabulated.is_some() {
                return tabulated;
            }
        }
        let dest = match form {
            Form::Plain => {
                if model == Model::Matrix {
                    let shaper = ProfileConversion::from_profile(profile);
                    match (&shaper, self.n, space) {
                        (ProfileConversion::Gray(_), 1, _)
                        | (ProfileConversion::Rgb(_), 3, WorkingSpace::Srgb) => {
                            return Some(IccConversion::Shaper(shaper));
                        }
                        (ProfileConversion::Rgb(_), 3, WorkingSpace::Wide) => {
                            if let Some(wide) = wide_rgb(profile) {
                                return Some(IccConversion::Wide(wide));
                            }
                        }
                        _ => {}
                    }
                }
                space.profile()
            }
            Form::Absolute => {
                let scale = self.absolute?;
                if model == Model::Matrix
                    && self.n == 1
                    && let Some(gray) = absolute_gray(profile, scale, space)
                {
                    return Some(gray);
                }
                crate::pcs::scaled_working(space, scale)
            }
            Form::Compensated => crate::pcs::black::compensating(space, self.black(model)),
        };
        let layout = layout(self.n)?;
        // Floating-point table evaluation: the fixed-point one misses by up
        // to two levels at 8 bits in the shadows, which compensation
        // stretches.
        let options = TransformOptions {
            rendering_intent: model.cms(),
            prefer_fixed_point: false,
            ..TransformOptions::default()
        };
        let xform = profile
            .create_transform_8bit(layout, &dest, Layout::Rgb, options)
            .map_err(|e| log::debug!("colorspace: ICC transform creation failed: {e}"))
            .ok()?;
        Some(IccConversion::Transform {
            n: self.n,
            xform,
            recipe: Some(Box::new(Recipe {
                profile: Arc::clone(profile),
                dest,
                options,
            })),
            xform16: OnceLock::new(),
            #[cfg(feature = "gpu-icc")]
            device: DeviceGrid::default(),
        })
    }

    /// Whether the profile's colour values convert as those of the device
    /// space of the same component count under every intent: one
    /// conversion serves every intent, with compensation on or off, and it
    /// is device-equivalent (see [`IccConversion::is_device_equivalent`]).
    fn is_device_equivalent(&self, alternate: impl FnOnce() -> Option<ColorSpace>) -> bool {
        let first = ColourIntent::INITIAL;
        let first = ColourTarget::new(first, WorkingSpace::Srgb);
        let one = RenderingIntent::ALL.iter().all(|&intent| {
            [false, true].iter().all(|&bpc| {
                self.slot(ColourTarget::new(
                    ColourIntent::new(intent, bpc),
                    WorkingSpace::Srgb,
                )) == self.slot(first)
            })
        });
        one && self
            .conversion(first, alternate)
            .is_some_and(|conv| conv.is_device_equivalent())
    }
}

/// The conversion of `profile`'s `N`-channel device → PCS table to
/// `space`, tabulated ([`table::tabulate`]), as a transform whose 8- and
/// 16-bit executors are the one grid.
fn tabulated<const N: usize>(
    profile: &ColorProfile,
    intent: moxcms::RenderingIntent,
    adjustment: PcsAdjustment,
    space: WorkingSpace,
) -> Option<IccConversion> {
    let grid = Arc::new(table::tabulate::<N>(profile, intent, adjustment, space)?);
    let xform16: Arc<Transform16BitExecutor> = grid.clone();
    #[cfg(feature = "gpu-icc")]
    let device = DeviceGrid {
        table: (Arc::clone(&grid) as Arc<dyn std::any::Any + Send + Sync>)
            .downcast()
            .ok(),
        grid: OnceLock::new(),
    };
    Some(IccConversion::Transform {
        n: u8::try_from(N).ok()?,
        xform: grid,
        recipe: None,
        xform16: OnceLock::from(Some(xform16)),
        #[cfg(feature = "gpu-icc")]
        device,
    })
}

/// The ICC-absolute conversion to `space` of the gray matrix/TRC
/// `profile`, whose PCS scale is `scale` (`D50 / w`): the RGB matrix/TRC
/// profile with the gray tone curve on every channel and the colorants of
/// `space` scaled by `w / D50`, which maps the gray `v` on every channel to
/// the absolute PCS `w · TRC(v)` of `v`.  `None` for a profile that is not
/// a gray TRC one.
fn absolute_gray(
    profile: &ColorProfile,
    scale: [f64; 3],
    space: WorkingSpace,
) -> Option<IccConversion> {
    let trc = profile
        .gray_trc
        .as_ref()
        .filter(|_| profile.pcs == moxcms::DataColorSpace::Xyz)?;
    let mut rgb = ColorProfile::new_srgb();
    rgb.cicp = None;
    rgb.red_trc = Some(trc.clone());
    rgb.green_trc = Some(trc.clone());
    rgb.blue_trc = Some(trc.clone());
    crate::pcs::scale_colorants(&mut rgb, scale.map(f64::recip));
    match space {
        WorkingSpace::Srgb => {
            let conversion = ProfileConversion::from_profile(&rgb);
            matches!(conversion, ProfileConversion::Rgb(_))
                .then_some(IccConversion::GrayAbsolute(conversion))
        }
        WorkingSpace::Wide => wide_rgb(&rgb).map(IccConversion::WideGray),
    }
}

/// The 16-bit conversion to the wide space of the RGB matrix/TRC
/// `profile`: each tone curve tabulated at every 16-bit level, then the
/// composite of [`wide_composite`].  `None` when a tag is missing, the
/// colorant matrix is singular or a curve does not evaluate.
fn wide_rgb(profile: &ColorProfile) -> Option<Arc<LinearRgb>> {
    let trcs = [&profile.red_trc, &profile.green_trc, &profile.blue_trc];
    let column = |c: &moxcms::Xyzd| [c.x, c.y, c.z];
    let cols = [
        column(&profile.red_colorant),
        column(&profile.green_colorant),
        column(&profile.blue_colorant),
    ];
    let colorants = [0, 1, 2].map(|r| [cols[0][r], cols[1][r], cols[2][r]]);
    // A singular colorant matrix covers the all-zero default of absent tags.
    crate::resources::icc_shaper::mat_inverse(&colorants).map(drop)?;
    let mut tables: Vec<Arc<[f32]>> = Vec::with_capacity(3);
    for (i, trc) in trcs.iter().enumerate() {
        let trc = trc.as_ref()?;
        let shared = (0..i).find(|&j| trcs[j].as_ref() == Some(trc));
        let table = match shared {
            Some(j) => Arc::clone(&tables[j]),
            None => LinearRgb::table(|x| eval_trc(trc, x))?,
        };
        tables.push(table);
    }
    let linear: [Arc<[f32]>; 3] = tables.try_into().ok()?;
    Some(Arc::new(LinearRgb::new(
        linear,
        &wide_composite(&colorants),
    )))
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

/// A converter transform of a source profile to a working space.
pub(crate) struct Recipe {
    /// The source profile.
    profile: Arc<ColorProfile>,
    /// The destination profile, the working space.
    dest: ColorProfile,
    /// The transform's options.
    options: TransformOptions,
}

/// The grid of a tabulated four-channel conversion and its [`CmykGrid`],
/// made on first use.
#[cfg(feature = "gpu-icc")]
#[derive(Default)]
pub(crate) struct DeviceGrid {
    /// The conversion's table; `None` for a conversion not tabulated on
    /// four inputs.
    table: Option<Arc<crate::pcs::grid::GridTable<4>>>,
    grid: OnceLock<Option<Arc<CmykGrid>>>,
}

/// The compiled conversion to a working space of one ICC profile stream
/// for colour values of its declared component count under one
/// device-to-PCS model.
pub(crate) enum IccConversion {
    /// A gray (`n = 1`) or RGB (`n = 3`) matrix/TRC profile, to sRGB; a
    /// gray one to either space.
    Shaper(ProfileConversion),
    /// An RGB matrix/TRC profile, to the wide space.
    Wide(Arc<LinearRgb>),
    /// A gray matrix/TRC profile under ICC-absolute colorimetry, to the
    /// wide space: as [`Self::GrayAbsolute`].
    WideGray(Arc<LinearRgb>),
    /// A gray matrix/TRC profile under ICC-absolute colorimetry: an RGB
    /// shaper evaluated at the gray value on every channel (see
    /// [`absolute_gray`]).  The converter maps a gray source to a neutral,
    /// whatever the media white, so the tint of that white is carried by
    /// the colorants of this shaper instead.
    GrayAbsolute(ProfileConversion),
    /// Any other profile with `n` of 1, 3 or 4: an 8-bit transform whose
    /// source pixel holds `n` channels, and its 16-bit counterpart.
    Transform {
        n: u8,
        xform: Arc<Transform8BitExecutor>,
        /// The converter transform both executors are, from which the
        /// 16-bit one is created; `None` for a tabulated conversion, whose
        /// 16-bit executor is given at compilation.
        recipe: Option<Box<Recipe>>,
        /// The 16-bit transform, given or created on first use from
        /// `recipe`; `None` when moxcms refuses it.
        xform16: OnceLock<Option<Arc<Transform16BitExecutor>>>,
        /// The grid of a tabulated four-channel conversion, for the
        /// device conversion (see [`Self::cmyk_grid`]).
        #[cfg(feature = "gpu-icc")]
        device: DeviceGrid,
    },
    /// A profile with no conversion of its own: the `/Alternate` space of
    /// the stream, a device or CIE-based space of the same component count,
    /// converting to the working space given.
    Alternate(ColorSpace, WorkingSpace),
}

impl IccConversion {
    /// The conversion of profile stream `id` for `n`-component values to
    /// `target`, compiled on first use and shared for the lifetime of
    /// `doc`.  `None` when neither the profile nor the `/Alternate` space
    /// converts `n` components.
    pub(crate) fn load(
        doc: &Document,
        id: ObjectId,
        n: u8,
        target: ColourTarget,
    ) -> Option<Arc<Self>> {
        Self::profile(doc, id, n)?.conversion(target, || Self::alternate(doc, id, n))
    }

    /// Whether profile stream `id` converts `n`-component values as the
    /// device space of `n` components does under every intent.
    pub(super) fn is_device_equivalent_stream(doc: &Document, id: ObjectId, n: u8) -> bool {
        Self::profile(doc, id, n)
            .is_some_and(|p| p.is_device_equivalent(|| Self::alternate(doc, id, n)))
    }

    /// Whether profile stream `id` converts `n`-component values through
    /// the same conversion under the intents `a` and `b`: both select the
    /// same model, media-relative or ICC-absolute alike.
    pub(super) fn converts_alike(
        doc: &Document,
        id: ObjectId,
        n: u8,
        a: ColourIntent,
        b: ColourIntent,
    ) -> bool {
        let srgb = |intent| ColourTarget::new(intent, WorkingSpace::Srgb);
        Self::profile(doc, id, n).is_none_or(|p| p.slot(srgb(a)) == p.slot(srgb(b)))
    }

    /// The profile of stream `id` for `n`-component values, parsed, and its
    /// [`crate::pcs::tag_fingerprint`]; `None` when the stream is not a
    /// profile.
    pub(crate) fn source_profile(
        doc: &Document,
        id: ObjectId,
        n: u8,
    ) -> Option<(Arc<ColorProfile>, u64)> {
        let profile = Self::profile(doc, id, n)?;
        Some((Arc::clone(profile.profile.as_ref()?), profile.fingerprint))
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

    /// Compile profile `bytes` for `n`-component values to `target`
    /// without consulting a cache.
    pub(crate) fn compile(bytes: &[u8], n: u8, target: ColourTarget) -> Option<Arc<Self>> {
        IccProfile::new(Some(bytes), n).conversion(target, || None)
    }

    /// The [`CmykGrid`] of a tabulated four-channel conversion, made on
    /// first use and shared: the conversion of 8-bit samples of every
    /// evaluator of the grid is this conversion's.  `None` for any other
    /// conversion.
    #[cfg(feature = "gpu-icc")]
    pub(crate) fn cmyk_grid(&self) -> Option<Arc<CmykGrid>> {
        let Self::Transform {
            n: 4,
            device:
                DeviceGrid {
                    table: Some(table),
                    grid,
                },
            ..
        } = self
        else {
            return None;
        };
        grid.get_or_init(|| table.cmyk_grid().map(Arc::new)).clone()
    }

    /// Whether this conversion yields the device conversion of the same
    /// component count: a shaper profile equal to sRGB at 8 bits, or a
    /// device `/Alternate` space.
    pub(super) const fn is_device_equivalent(&self) -> bool {
        matches!(
            self,
            Self::Shaper(ProfileConversion::Gray(None) | ProfileConversion::Rgb(None))
                | Self::Alternate(
                    ColorSpace::DeviceGray | ColorSpace::DeviceRgb | ColorSpace::DeviceCmyk,
                    _
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

    /// Convert normalised components to the working space at the depth of
    /// `S`.
    /// Components are quantised to that depth for the transform, which is
    /// the 8-bit or the 16-bit one; the shaper evaluates at that depth (see
    /// `ProfileConversion::convert_components_at`).  `None` when `comps`
    /// does not hold the compiled component count or the transform fails.
    pub(crate) fn convert<S: Sample>(&self, doc: &Document, comps: &[f64]) -> Option<[S; 3]> {
        match self {
            Self::Alternate(space, working) => {
                // An alternate is a device or CIE-based space, whose conversion
                // is independent of the intent.
                (comps.len() == space.ncomponents()).then(|| {
                    space.convert(
                        doc,
                        ColourTarget::new(ColourIntent::INITIAL, *working),
                        comps,
                    )
                })
            }
            Self::Shaper(shaper) => shaper.convert_components_at(comps),
            Self::GrayAbsolute(shaper) => match *comps {
                [v] => shaper.convert_components_at(&[v, v, v]),
                _ => None,
            },
            Self::Wide(wide) => match *comps {
                [r, g, b] => Some(
                    wide.convert([r, g, b].map(u16::from_unit))
                        .map(S::from_sample),
                ),
                _ => None,
            },
            Self::WideGray(wide) => match *comps {
                [v] => Some(wide.convert([u16::from_unit(v); 3]).map(S::from_sample)),
                _ => None,
            },
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
            n, recipe, xform16, ..
        } = self
        else {
            return None;
        };
        xform16
            .get_or_init(|| {
                let Recipe {
                    profile,
                    dest,
                    options,
                } = recipe.as_deref()?;
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
mod tests;
