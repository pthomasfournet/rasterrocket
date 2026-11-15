//! Conversion of a finished page from the working space to an output ICC
//! profile.
//!
//! An [`OutputProfile`] is an RGB or gray profile of the output, display
//! or colour-space class whose PCS is XYZ or Lab; matrix/TRC and table
//! (`BToA`) destinations alike are converted through `moxcms`.  A
//! conversion is compiled once per rendering intent and black-point
//! compensation, on first use, and shared by every page of the profile.
//! It runs on 16-bit samples at both depths: an 8-bit sample is widened
//! exactly (`257 · v`) and the result rounded back to 8 bits, which keeps
//! the 8-bit result within rounding of the 16-bit one.  The source of every conversion is the working space
//! (sRGB) with the intent's adjustments folded in (see
//! [`working_space`]).
//!
//! [`OutputProfile::convert`] converts a [`FinishedCanvas`] in place for an
//! RGB profile and into a gray bitmap for a gray one, row by row, the
//! alpha plane carried unchanged; colour beside alpha is unassociated, so
//! a pixel converts independently of its alpha.  A pixel of alpha 0
//! keeps colour 0.

mod working_space;

use std::sync::{Arc, OnceLock};

use color::{Gray8, Gray16, Pixel, Rgb8, Rgb16, Sample};
use moxcms::{
    CmsError, ColorProfile, DataColorSpace, Layout, ProfileClass, TransformExecutor,
    TransformOptions,
};
use raster::Bitmap;

use crate::RenderingIntent;
use crate::renderer::{CanvasBuffers, FinishedCanvas};

/// The colour space of an output profile's device values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSpace {
    /// Three channels, red, green and blue.
    Rgb,
    /// One channel, 0 black.
    Gray,
}

impl OutputSpace {
    /// The channels of a pixel.
    #[must_use]
    pub const fn channels(self) -> u8 {
        match self {
            Self::Rgb => 3,
            Self::Gray => 1,
        }
    }
}

/// Why a profile cannot be an output profile, or a conversion to it was
/// not built.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OutputProfileError {
    /// The bytes are not an ICC profile the parser reads.
    Unreadable(String),
    /// A profile class that does not describe an output: device link,
    /// abstract, named colour or input.
    Class(&'static str),
    /// A CMYK profile: CMYK device values are not an output.
    Cmyk,
    /// A data colour space other than RGB, gray and CMYK, named.
    ColorSpace(String),
    /// A profile connection space other than XYZ and Lab.
    Pcs,
    /// The conversion for `intent` could not be built.
    Transform {
        /// The intent of the conversion.
        intent: RenderingIntent,
        /// The reason the converter gave.
        reason: String,
    },
    /// A conversion failed on a page's pixels.
    Conversion(String),
}

impl std::fmt::Display for OutputProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreadable(e) => write!(f, "not an ICC profile: {e}"),
            Self::Class(class) => write!(
                f,
                "a {class} profile; an output profile is of the output, display \
                 or colour-space class"
            ),
            Self::Cmyk => f.write_str(
                "a CMYK profile; CMYK raster output is not written, give an RGB or \
                 gray output profile",
            ),
            Self::ColorSpace(space) => write!(
                f,
                "a profile of {space} device values; an output profile is RGB or gray"
            ),
            Self::Pcs => f.write_str("a profile connection space other than XYZ and Lab"),
            Self::Transform { intent, reason } => {
                write!(f, "no {} conversion: {reason}", intent.pdf_name())
            }
            Self::Conversion(e) => write!(f, "the conversion failed: {e}"),
        }
    }
}

impl std::error::Error for OutputProfileError {}

/// A compiled conversion of 16-bit samples.
type Executor = Arc<dyn TransformExecutor<u16> + Send + Sync>;

/// The conversions of one intent and compensation, in the order of
/// [`slot`].
const SLOTS: usize = RenderingIntent::ALL.len() * 2;

/// A compiled conversion, or why it could not be compiled.
type Compiled = Result<Executor, OutputProfileError>;

/// An ICC profile pages are converted to and tagged with.
pub struct OutputProfile {
    bytes: Arc<[u8]>,
    profile: ColorProfile,
    space: OutputSpace,
    conversions: [OnceLock<Compiled>; SLOTS],
}

impl std::fmt::Debug for OutputProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputProfile")
            .field("space", &self.space)
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

/// The index of the conversion of `intent`, with compensation when `bpc`.
fn slot(intent: RenderingIntent, bpc: bool) -> usize {
    let intent = RenderingIntent::ALL
        .iter()
        .position(|&i| i == intent)
        .expect("ALL lists every intent");
    intent * 2 + usize::from(bpc)
}

/// The converter's form of `intent`.
const fn cms_intent(intent: RenderingIntent) -> moxcms::RenderingIntent {
    match intent {
        RenderingIntent::RelativeColorimetric => moxcms::RenderingIntent::RelativeColorimetric,
        RenderingIntent::AbsoluteColorimetric => moxcms::RenderingIntent::AbsoluteColorimetric,
        RenderingIntent::Perceptual => moxcms::RenderingIntent::Perceptual,
        RenderingIntent::Saturation => moxcms::RenderingIntent::Saturation,
    }
}

impl OutputProfile {
    /// The output profile of `bytes`.
    ///
    /// # Errors
    ///
    /// [`OutputProfileError`] for bytes that are not a profile, or a
    /// profile not of RGB or gray device values, not of the output,
    /// display or colour-space class, or whose PCS is neither XYZ nor Lab,
    /// checked in that order.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, OutputProfileError> {
        let bytes = bytes.into();
        let profile = ColorProfile::new_from_slice(&bytes)
            .map_err(|e| OutputProfileError::Unreadable(e.to_string()))?;
        let space = match profile.color_space {
            DataColorSpace::Rgb => OutputSpace::Rgb,
            DataColorSpace::Gray => OutputSpace::Gray,
            DataColorSpace::Cmyk => return Err(OutputProfileError::Cmyk),
            other => return Err(OutputProfileError::ColorSpace(format!("{other:?}"))),
        };
        match profile.profile_class {
            ProfileClass::OutputDevice | ProfileClass::DisplayDevice | ProfileClass::ColorSpace => {
            }
            ProfileClass::InputDevice => return Err(OutputProfileError::Class("input")),
            ProfileClass::DeviceLink => return Err(OutputProfileError::Class("device link")),
            ProfileClass::Abstract => return Err(OutputProfileError::Class("abstract")),
            ProfileClass::Named => return Err(OutputProfileError::Class("named colour")),
        }
        if !matches!(profile.pcs, DataColorSpace::Xyz | DataColorSpace::Lab) {
            return Err(OutputProfileError::Pcs);
        }
        Ok(Self {
            bytes,
            profile,
            space,
            conversions: std::array::from_fn(|_| OnceLock::new()),
        })
    }

    /// The profile's bytes.
    #[must_use]
    pub const fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    /// The colour space of the profile's device values.
    #[must_use]
    pub const fn space(&self) -> OutputSpace {
        self.space
    }

    /// Compile the conversion of `intent`, with black-point compensation
    /// when `bpc`, unless already compiled.
    ///
    /// # Errors
    ///
    /// [`OutputProfileError::Transform`] when the profile admits no such
    /// conversion; the error is kept and returned again on every later
    /// call.
    pub fn prepare(&self, intent: RenderingIntent, bpc: bool) -> Result<(), OutputProfileError> {
        self.executor(intent, bpc).map(drop)
    }

    /// The conversion of `intent` and `bpc`, compiled on first use.
    fn executor(&self, intent: RenderingIntent, bpc: bool) -> Compiled {
        self.conversions[slot(intent, bpc)]
            .get_or_init(|| {
                self.compile(intent, bpc)
                    .map_err(|e| OutputProfileError::Transform {
                        intent,
                        reason: e.to_string(),
                    })
            })
            .clone()
    }

    /// Build the conversion of `intent` and `bpc`.
    fn compile(&self, intent: RenderingIntent, bpc: bool) -> Result<Executor, CmsError> {
        let intent = cms_intent(intent);
        let black = working_space::black_lift(&self.profile, intent, bpc)?;
        let source = working_space::source_profile(&self.profile, intent, black)?;
        let layout = match self.space {
            OutputSpace::Rgb => Layout::Rgb,
            OutputSpace::Gray => Layout::Gray,
        };
        source.create_transform_16bit(
            Layout::Rgb,
            &self.profile,
            layout,
            TransformOptions {
                rendering_intent: intent,
                allow_use_cicp_transfer: false,
                ..TransformOptions::default()
            },
        )
    }

    /// `canvas`, a working-space canvas, converted to the profile's device
    /// values under `intent`, with black-point compensation when `bpc`: in
    /// place for an RGB profile, into a gray canvas built in `buffers` for
    /// a gray one, the working-space storage then reclaimed into
    /// `buffers`.  A canvas already in a gray space is returned as it is,
    /// and so is a CMYK canvas, which holds device values.
    ///
    /// # Errors
    ///
    /// The conversion could not be built ([`OutputProfileError::Transform`]),
    /// or failed, or a gray canvas could not be allocated
    /// ([`OutputProfileError::Conversion`]).
    pub fn convert(
        &self,
        canvas: FinishedCanvas,
        (intent, bpc): (RenderingIntent, bool),
        buffers: &mut CanvasBuffers,
    ) -> Result<FinishedCanvas, OutputProfileError> {
        if matches!(
            canvas,
            FinishedCanvas::Gray8(_)
                | FinishedCanvas::Gray16(_)
                | FinishedCanvas::Cmyk8(_)
                | FinishedCanvas::Cmyk16(_)
        ) {
            return Ok(canvas);
        }
        let xform = self.executor(intent, bpc)?;
        let xform = &*xform;
        Ok(match (canvas, self.space) {
            (FinishedCanvas::Rgb8(mut rgb), OutputSpace::Rgb) => {
                convert_in_place(xform, &mut rgb)?;
                FinishedCanvas::Rgb8(rgb)
            }
            (FinishedCanvas::Rgb16(mut rgb), OutputSpace::Rgb) => {
                convert_in_place(xform, &mut rgb)?;
                FinishedCanvas::Rgb16(rgb)
            }
            (FinishedCanvas::Rgb8(rgb), OutputSpace::Gray) => {
                let storage = buffers.gray.take_for(rgb.has_alpha());
                let gray = convert_to::<Rgb8, Gray8>(xform, &rgb, storage)?;
                buffers.rgb.reclaim(rgb);
                FinishedCanvas::Gray8(gray)
            }
            (FinishedCanvas::Rgb16(rgb), OutputSpace::Gray) => {
                let storage = buffers.gray16.take_for(rgb.has_alpha());
                let gray = convert_to::<Rgb16, Gray16>(xform, &rgb, storage)?;
                buffers.rgb16.reclaim(rgb);
                FinishedCanvas::Gray16(gray)
            }
            (
                canvas @ (FinishedCanvas::Gray8(_)
                | FinishedCanvas::Gray16(_)
                | FinishedCanvas::Cmyk8(_)
                | FinishedCanvas::Cmyk16(_)),
                _,
            ) => canvas,
        })
    }
}

/// A row converter: the 16-bit working rows of a conversion.
#[derive(Default)]
struct Rows {
    src: Vec<u16>,
    dst: Vec<u16>,
}

impl Rows {
    /// Convert `src` through `xform` into `dst`, samples of any depth,
    /// through 16-bit samples.
    fn convert<T: Sample>(
        &mut self,
        xform: &(dyn TransformExecutor<u16> + Send + Sync),
        src: &[T],
        dst: &mut [T],
    ) -> Result<(), OutputProfileError> {
        self.src.clear();
        self.src.extend(src.iter().map(|&v| u16::from_sample(v)));
        self.dst.clear();
        self.dst.resize(dst.len(), 0);
        xform
            .transform(&self.src, &mut self.dst)
            .map_err(|e| OutputProfileError::Conversion(e.to_string()))?;
        for (d, &v) in dst.iter_mut().zip(&self.dst) {
            *d = T::from_sample(v);
        }
        Ok(())
    }
}

/// Convert every row of `bitmap` in place through `xform`, of equal source
/// and destination channels; a pixel of alpha 0 is left 0.
fn convert_in_place<P: Pixel>(
    xform: &(dyn TransformExecutor<u16> + Send + Sync),
    bitmap: &mut Bitmap<P>,
) -> Result<(), OutputProfileError> {
    let live = bitmap.width as usize * P::NCOMPS;
    let mut rows = Rows::default();
    let mut row_in = Vec::with_capacity(live);
    for y in 0..bitmap.height {
        let (row, alpha) = bitmap.row_and_alpha_mut(y);
        let row = &mut row[..live];
        row_in.clear();
        row_in.extend_from_slice(row);
        rows.convert(xform, &row_in, row)?;
        if let Some(alpha) = alpha {
            clear_transparent::<P>(row, alpha);
        }
    }
    Ok(())
}

/// `src` converted through `xform` into a bitmap of pixels `D` built in
/// `storage`, `src`'s alpha plane copied; a pixel of alpha 0 is 0.
fn convert_to<S: Pixel, D: Pixel<Sample = S::Sample>>(
    xform: &(dyn TransformExecutor<u16> + Send + Sync),
    src: &Bitmap<S>,
    storage: raster::BitmapBuffers<S::Sample>,
) -> Result<Bitmap<D>, OutputProfileError> {
    let mut dst = Bitmap::<D>::try_new_filled_in(
        storage,
        src.width,
        src.height,
        1,
        src.has_alpha(),
        S::Sample::default(),
    )
    .map_err(|e| OutputProfileError::Conversion(e.to_string()))?;
    if let (Some(from), Some(to)) = (src.alpha_plane(), dst.alpha_plane_mut()) {
        to.copy_from_slice(from);
    }
    let (s_live, d_live) = (
        src.width as usize * S::NCOMPS,
        src.width as usize * D::NCOMPS,
    );
    let mut rows = Rows::default();
    for y in 0..src.height {
        let (row, alpha) = dst.row_and_alpha_mut(y);
        let row = &mut row[..d_live];
        rows.convert(xform, &src.row_samples(y)[..s_live], row)?;
        if let Some(alpha) = alpha {
            clear_transparent::<D>(row, alpha);
        }
    }
    Ok(dst)
}

/// Set the colour of every pixel of `row` whose `alpha` is 0 to 0.
fn clear_transparent<P: Pixel>(row: &mut [P::Sample], alpha: &[P::Sample]) {
    for (px, &a) in row.chunks_exact_mut(P::NCOMPS).zip(alpha) {
        if a == P::Sample::default() {
            px.fill(P::Sample::default());
        }
    }
}

#[cfg(test)]
mod tests;
