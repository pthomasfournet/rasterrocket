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
//! reproduces the colour, and the device conversion of ISO 32000-2
//! §10.4.2.3 for CMYK output.

use color::Sample;

/// The canvas a page is painted on and the pixels it finishes as.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CanvasRequest {
    /// sRGB pixels.
    #[default]
    Rgb,
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
        !matches!(self, Self::Rgb)
    }

    /// Whether the page finishes as CMYK.
    #[must_use]
    pub const fn is_cmyk(&self) -> bool {
        matches!(self, Self::Cmyk { .. })
    }

    /// The ink-canvas settings of the request.
    pub(crate) const fn ink_setup(&self) -> InkSetup {
        match self {
            Self::Rgb | Self::OverprintPreview => InkSetup::PREVIEW,
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
}

impl Separation {
    /// The separator of the separation.
    const fn separator(&self) -> Separator {
        match self {
            Self::Device => Separator::Device,
        }
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
    /// [`color::cmyk::rgb_to_device_cmyk`].
    Device,
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

    /// The process ink `[C, M, Y, K]` of the sRGB colour `rgb`, at its
    /// depth.
    #[must_use]
    pub(crate) fn ink<S: Sample>(&self, rgb: [S; 3]) -> [S; 4] {
        match self {
            Self::Model => color::cmyk::rgb_to_ink_at(rgb),
            Self::Device => color::cmyk::rgb_to_device_cmyk(rgb),
        }
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
}
