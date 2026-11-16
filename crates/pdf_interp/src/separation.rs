//! The canvas a page is painted on, and how colours that are not ink
//! become ink on a canvas of inks.
//!
//! A page is painted on an sRGB canvas, or on a canvas of process and
//! spot inks `[C, M, Y, K, S0, S1, S2, S3]` (see
//! [`crate::renderer::FinishedCanvas`]).  An ink canvas serves two ends:
//! the overprint preview, which simulates overprint and finishes as sRGB,
//! and CMYK output, which finishes as the page's process ink, its spots
//! folded in.  A colour given as ink — `DeviceCMYK`, the colorants of a
//! `Separation` or `DeviceN` space — paints its own ink on either.  Any
//! other colour is converted from its sRGB value by a [`Separator`]: the
//! inverse of the process-ink model for the preview, so that the preview
//! reproduces the colour, and for CMYK output the conversion of the
//! output profile ([`ProfileSeparation`]) or, without one, the device
//! conversion of ISO 32000-2 §10.4.2 under the graphics state's black
//! generation and undercolour removal.  The separator of a paint is the
//! page's with the graphics state's parameters taken in (see
//! [`Separator::for_paint`]): its black-generation and undercolour-removal
//! functions, and the rendering intent a profile separates it under.

mod press;

pub use press::PressSeparation;

use std::sync::Arc;

use color::Sample;

use crate::RenderingIntent;
use crate::pcs::grid::GridTable;
use crate::resources::ColorSpace;
use crate::resources::black_generation::DeviceBlack;
use crate::resources::colorspace::icc::IccConversion;

/// The canvas a page is painted on and the pixels it finishes as.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CanvasRequest {
    /// sRGB pixels.
    #[default]
    Rgb,
    /// 16-bit pixels of the wide working space
    /// ([`WorkingSpace::Wide`](crate::WorkingSpace::Wide)).
    WideRgb,
    /// Process and spot inks on which overprint (ISO 32000-2 §8.6.7) is
    /// simulated, finished as sRGB.
    OverprintPreview,
    /// Process and spot inks finished as CMYK, overprint simulated when
    /// `overprint` and every paint knocking out every channel otherwise;
    /// colours not given as ink are separated by `separation`.
    Cmyk {
        /// Whether overprint is simulated.
        overprint: bool,
        /// How colours not given as ink become process ink.
        separation: Separation,
    },
}

impl CanvasRequest {
    /// Whether the page is painted on an ink canvas.
    #[must_use]
    pub const fn is_ink(&self) -> bool {
        !matches!(self, Self::Rgb | Self::WideRgb)
    }

    /// The working space the page's colours convert into: that of the
    /// separation of a CMYK page, the wide space for
    /// [`Self::WideRgb`], sRGB otherwise.
    #[must_use]
    pub fn space(&self) -> crate::WorkingSpace {
        match self {
            Self::Rgb | Self::OverprintPreview => crate::WorkingSpace::Srgb,
            Self::WideRgb => crate::WorkingSpace::Wide,
            Self::Cmyk { separation, .. } => separation.space(),
        }
    }

    /// Whether the page finishes as CMYK.
    #[must_use]
    pub const fn is_cmyk(&self) -> bool {
        matches!(self, Self::Cmyk { .. })
    }

    /// The ink-canvas settings of the request.
    pub(crate) fn ink_setup(&self) -> InkSetup {
        match self {
            Self::Rgb | Self::WideRgb | Self::OverprintPreview => InkSetup::PREVIEW,
            Self::Cmyk {
                overprint,
                separation,
            } => InkSetup {
                overprint: *overprint,
                cmyk: true,
                separator: separation.separator(),
            },
        }
    }
}

/// How CMYK output converts a colour not given as ink to process ink.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Separation {
    /// The conversion ISO 32000-2 §10.4.2.3 defines for a device without
    /// a colour profile ([`color::cmyk::rgb_to_device_cmyk`]): gray is
    /// black ink alone, and RGB loses its gray component to black.
    #[default]
    Device,
    /// The conversion of a CMYK output profile from the working space,
    /// under the intent of each paint
    /// (see [`OutputProfile::separation`](crate::OutputProfile::separation)).
    Profile(Arc<PressSeparation>),
}

impl Separation {
    /// The working space the separation converts from.
    #[must_use]
    pub fn space(&self) -> crate::WorkingSpace {
        match self {
            Self::Device => crate::WorkingSpace::Srgb,
            Self::Profile(press) => press.space(),
        }
    }

    /// The separator of the separation under the initial graphics state.
    fn separator(&self) -> Separator {
        match self {
            Self::Device => Separator::Device(DeviceBlack::default()),
            Self::Profile(press) => Separator::Profile(ProfileInk::new(
                press,
                RenderingIntent::RelativeColorimetric,
            )),
        }
    }
}

/// The separation of a paint through a CMYK output profile: the table of
/// the intent the paint separates under.
#[derive(Clone, Debug)]
pub(crate) struct ProfileInk {
    press: Arc<PressSeparation>,
    intent: RenderingIntent,
    table: Arc<ProfileSeparation>,
}

impl PartialEq for ProfileInk {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.press, &other.press)
            && self.intent == other.intent
            && Arc::ptr_eq(&self.table, &other.table)
    }
}

impl Eq for ProfileInk {}

impl ProfileInk {
    /// The separation through `press` of a paint under `intent`.
    fn new(press: &Arc<PressSeparation>, intent: RenderingIntent) -> Self {
        let intent = press.intent(intent);
        Self {
            press: Arc::clone(press),
            intent,
            table: press.table(intent),
        }
    }
}

/// The number of nodes along each axis of a [`ProfileSeparation`].
pub(crate) const PROFILE_GRID: usize = 65;

/// The CMYK of a profile's conversion from a working space, tabulated on a
/// `65 × 65 × 65` grid over the space's cube and interpolated in
/// tetrahedra.
///
/// Node `(r, g, b)` holds the 16-bit CMYK of the colour of channels
/// `v · 65535 / 64`, rounded, for `v` = `r`, `g`, `b`.
#[derive(Clone, PartialEq, Eq)]
pub struct ProfileSeparation {
    /// The nodes, `r` slowest, `b` fastest.
    nodes: Box<[[u16; 4]]>,
    /// The working space of the colours separated.
    space: crate::WorkingSpace,
}

impl std::fmt::Debug for ProfileSeparation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileSeparation")
            .field("grid", &PROFILE_GRID)
            .finish_non_exhaustive()
    }
}

impl ProfileSeparation {
    /// The sRGB channel value of node index `i` along an axis.
    pub(crate) fn node_value(i: usize) -> u16 {
        let span = PROFILE_GRID as u64 - 1;
        u16::try_from((i as u64 * u64::from(u16::MAX) + span / 2) / span)
            .expect("a node value lies within the sample range")
    }

    /// The table of `nodes`, `PROFILE_GRID³` of them, `r` slowest, of
    /// colours of `space`; `None` for another count.
    pub(crate) fn from_nodes(nodes: Vec<[u16; 4]>, space: crate::WorkingSpace) -> Option<Self> {
        (nodes.len() == PROFILE_GRID.pow(3)).then(|| Self {
            nodes: nodes.into_boxed_slice(),
            space,
        })
    }

    /// The CMYK of the working-space colour `rgb`, at its depth: the
    /// interpolation in the tetrahedron of its grid cell whose vertices are
    /// visited in decreasing order of the colour's fractional positions
    /// within the cell.  A neutral interpolates the nodes of the neutral
    /// axis alone, so the gray-component replacement of a profile, which
    /// changes its inks across that axis, is not blurred into the grays.
    #[must_use]
    pub fn ink<S: Sample>(&self, rgb: [S; 3]) -> [S; 4] {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the grid's span, 64, is exact in f64"
        )]
        let span = (PROFILE_GRID - 1) as f64;
        let pos = rgb.map(|v| v.to_unit() * span);
        let mut vertex = pos.map(|p| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "p lies in [0, 64], so its floor is a node index"
            )]
            let i = p.floor() as usize;
            i.min(PROFILE_GRID - 2)
        });
        #[expect(
            clippy::cast_precision_loss,
            reason = "node indices are below 64, exact in f64"
        )]
        let frac = [0, 1, 2].map(|i| pos[i] - vertex[i] as f64);
        let mut axes = [0usize, 1, 2];
        axes.sort_by(|&a, &b| frac[b].total_cmp(&frac[a]));
        let node = |[r, g, b]: [usize; 3]| self.nodes[(r * PROFILE_GRID + g) * PROFILE_GRID + b];
        let mut ink = [0.0f64; 4];
        let mut previous = 1.0;
        for step in 0..=3 {
            let f = axes.get(step).map_or(0.0, |&a| frac[a]);
            let weight = previous - f;
            if weight != 0.0 {
                for (o, &v) in ink.iter_mut().zip(&node(vertex)) {
                    *o = weight.mul_add(f64::from(v), *o);
                }
            }
            if let Some(&a) = axes.get(step) {
                vertex[a] += 1;
            }
            previous = f;
        }
        ink.map(|v| S::from_unit(v / f64::from(u16::MAX)))
    }
}

/// The conversion of an sRGB colour to the process ink an ink canvas
/// paints it with.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum Separator {
    /// The inverse of the process-ink model,
    /// [`color::cmyk::rgb_to_ink_at`]: the ink the overprint preview
    /// converts back to the colour.
    #[default]
    Model,
    /// The device conversion under black-generation and undercolour-removal
    /// functions ([`DeviceBlack::separate`]); under the device's own,
    /// [`color::cmyk::rgb_to_device_cmyk`].
    Device(DeviceBlack),
    /// An output profile's conversion under one intent.
    Profile(ProfileInk),
}

impl Separator {
    /// Whether a colour given in a four-component ICC space paints the
    /// process ink reproducing the profile's colour in the process-ink
    /// model ([`color::cmyk::ink_reproducing`]), so that the preview shows
    /// that colour; otherwise, for CMYK output, it paints its components,
    /// which are already a CMYK device's values.
    #[must_use]
    pub(crate) const fn reproduces_process_colour(&self) -> bool {
        matches!(self, Self::Model)
    }

    /// The working space of the colours the separator converts: that of
    /// a profile's table, sRGB for the models.
    #[must_use]
    pub(crate) fn space(&self) -> crate::WorkingSpace {
        match self {
            Self::Model | Self::Device(_) => crate::WorkingSpace::Srgb,
            Self::Profile(ink) => ink.table.space,
        }
    }

    /// The process ink `[C, M, Y, K]` of the colour `rgb` of the
    /// separator's working space ([`Self::space`]), at its depth.
    #[must_use]
    pub(crate) fn ink<S: Sample>(&self, rgb: [S; 3]) -> [S; 4] {
        match self {
            Self::Model => color::cmyk::rgb_to_ink_at(rgb),
            Self::Device(black) => black.separate(rgb),
            Self::Profile(ink) => ink.table.ink(rgb),
        }
    }

    /// The conversion of the colours of `space`, a four-component ICC
    /// space, to the ink of the separator's output profile, when its
    /// profile is not the output profile (see
    /// [`OutputProfile::cmyk_link`](crate::OutputProfile::cmyk_link));
    /// `None` for any other separator or space, whose components are the
    /// ink they paint.
    #[must_use]
    pub(crate) fn cmyk_link(&self, doc: &pdf::Document, space: &ColorSpace) -> Option<CmykLink> {
        let Self::Profile(ink) = self else {
            return None;
        };
        let ColorSpace::IccBased {
            n: 4,
            stream_id: Some(id),
        } = space
        else {
            return None;
        };
        let (profile, fingerprint) = IccConversion::source_profile(doc, *id, 4)?;
        ink.press
            .profile()
            .cmyk_link(&profile, fingerprint, ink.intent, ink.press.bpc())
            .map(CmykLink)
    }

    /// The separator of a paint under `black`, the graphics state's
    /// black-generation and undercolour-removal functions, and `intent`:
    /// the device conversion under the functions, a profile's conversion
    /// under the intent the paint separates under
    /// ([`PressSeparation::intent`]); the preview's model as it is.
    #[must_use]
    pub(crate) fn for_paint(&self, black: &DeviceBlack, intent: RenderingIntent) -> Self {
        match self {
            Self::Model => Self::Model,
            Self::Device(_) => Self::Device(black.clone()),
            Self::Profile(ink) if ink.press.intent(intent) == ink.intent => self.clone(),
            Self::Profile(ink) => Self::Profile(ProfileInk::new(&ink.press, intent)),
        }
    }
}

/// The conversion of the ink of a CMYK profile to the ink of an output
/// profile ([`Separator::cmyk_link`]).  Two are equal when they are the same
/// table.
#[derive(Clone, Debug)]
pub(crate) struct CmykLink(Arc<GridTable<4>>);

impl PartialEq for CmykLink {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CmykLink {}

impl CmykLink {
    /// The output profile's ink of the source ink `cmyk`, at its depth.
    #[must_use]
    pub(crate) fn ink<S: Sample>(&self, cmyk: [S; 4]) -> [S; 4] {
        self.0.convert(cmyk)
    }
}

/// The settings of a page's ink canvas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InkSetup {
    /// Whether overprint is simulated: a paint under overprint writes only
    /// the channels its colour paints.
    pub(crate) overprint: bool,
    /// Whether the canvas finishes as CMYK, not sRGB.
    pub(crate) cmyk: bool,
    /// The conversion of colours not given as ink.
    pub(crate) separator: Separator,
}

impl InkSetup {
    /// The overprint preview's settings; those of an RGB canvas too, which
    /// never reads them.
    pub(crate) const PREVIEW: Self = Self {
        overprint: true,
        cmyk: false,
        separator: Separator::Model,
    };
}

impl Default for InkSetup {
    fn default() -> Self {
        Self::PREVIEW
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_set_up_the_ink_canvas() {
        assert!(!CanvasRequest::Rgb.is_ink());
        assert_eq!(
            CanvasRequest::OverprintPreview.ink_setup(),
            InkSetup::PREVIEW
        );
        let cmyk = CanvasRequest::Cmyk {
            overprint: false,
            separation: Separation::Device,
        };
        assert!(cmyk.is_ink() && cmyk.is_cmyk());
        let setup = cmyk.ink_setup();
        assert!(setup.cmyk && !setup.overprint);
        assert_eq!(setup.separator.ink([0u8; 3]), [0, 0, 0, 255]);
        assert_eq!(
            Separator::Model.ink([0u8; 3]),
            color::cmyk::rgb_to_ink([0; 3])
        );
    }

    /// A table of the device conversion of each node reproduces the
    /// conversion at the nodes and between them, where it is linear in each
    /// tetrahedron of a cell.
    #[test]
    fn a_profile_separation_interpolates_its_nodes() {
        let mut nodes = Vec::new();
        for r in 0..PROFILE_GRID {
            for g in 0..PROFILE_GRID {
                for b in 0..PROFILE_GRID {
                    let rgb = [r, g, b].map(ProfileSeparation::node_value);
                    nodes.push(color::cmyk::rgb_to_device_cmyk(rgb));
                }
            }
        }
        let table =
            ProfileSeparation::from_nodes(nodes, crate::WorkingSpace::Srgb).expect("a whole grid");
        assert_eq!(ProfileSeparation::node_value(PROFILE_GRID - 1), u16::MAX);
        for rgb in [[0u8; 3], [255; 3], [255, 0, 0], [12, 200, 77], [128; 3]] {
            let want = color::cmyk::rgb_to_device_cmyk(rgb);
            let got = table.ink(rgb);
            for (g, w) in got.iter().zip(want) {
                assert!(g.abs_diff(w) <= 1, "{rgb:?}: {got:?} for {want:?}");
            }
        }
        // Off the nodes, near the neutral axis, where the black the
        // conversion takes switches between the channels.
        for rgb in [
            [30_000u16, 30_100, 30_050],
            [1_000, 900, 1_100],
            [40_000, 40_000, 39_000],
        ] {
            let want = color::cmyk::rgb_to_device_cmyk(rgb);
            let got = table.ink(rgb);
            for (g, w) in got.iter().zip(want) {
                assert!(g.abs_diff(w) <= 1, "{rgb:?}: {got:?} for {want:?}");
            }
        }
        assert_eq!(table.ink([u16::MAX; 3]), [0; 4]);
        assert!(
            ProfileSeparation::from_nodes(vec![[0; 4]; 8], crate::WorkingSpace::Srgb).is_none()
        );
    }
}
