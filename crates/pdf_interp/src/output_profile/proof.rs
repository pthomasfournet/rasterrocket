//! Soft proofing: a page painted as the process ink of a press, converted
//! to a display space through the press's profile.
//!
//! A [`SoftProof`] pairs a CMYK press profile, the output intent a
//! document was prepared for, with the display space the proof is shown
//! in: the working space, or an RGB or gray output profile.  The page is
//! painted on an ink canvas whose colours not given as ink are separated
//! by the press profile ([`OutputProfile::separation`]), so the finished
//! CMYK canvas holds the ink the press would print.  The proof converts
//! that ink to the display: the press's device → PCS conversion under
//! relative colorimetry, then the display's PCS → device conversion under
//! relative colorimetry, without black-point compensation.  Under
//! [`ProofIntent::AbsoluteColorimetric`] the relative PCS is scaled
//! between the two by the ratio of the press's media white to the
//! display's ([`crate::pcs::absolute_scale`]), which reproduces the paper
//! white and the press's black as measured; the scale folds into the
//! colorants of the display (scaled by its inverse), which must therefore
//! be the working space or a matrix/TRC RGB profile.  To a gray display the
//! ink converts through lightness-encoded XYZ
//! ([`crate::pcs::black::pcs_lightness`]), into whose colorants the scale
//! folds.
//!
//! The conversion is evaluated in floating point at the nodes of a
//! [`PROOF_GRID`]⁴ grid over the ink cube, once per proof intent, and
//! interpolated in the 4-simplex of each pixel.

use std::sync::{Arc, OnceLock};

use color::{Cmyk8, Cmyk16, Gray8, Gray16, Rgb8, Rgb16};
use moxcms::{CmsError, ColorProfile, Layout, TransformExecutor, TransformOptions};

use super::{OutputProfile, OutputProfileError, OutputSpace, convert_to};
use crate::intent::ProofIntent;
use crate::pcs::black::{as_version_2, layout_of, pcs_lightness};
use crate::renderer::{CanvasBuffers, FinishedCanvas};

/// The number of cells along each ink axis of a proof's table.
const SPAN: u32 = 32;

/// The number of nodes along each ink axis of a proof's table.
pub const PROOF_GRID: usize = SPAN as usize + 1;

/// The distance between neighbouring nodes along each ink axis, `c`
/// slowest.
const STRIDES: [usize; 4] = [PROOF_GRID.pow(3), PROOF_GRID.pow(2), PROOF_GRID, 1];

/// A tabulated proof conversion, or why it could not be built.
type Tabulated = Result<Arc<ProofTable>, OutputProfileError>;

/// The conversion of a press's process ink to a display space.
pub struct SoftProof {
    press: Arc<OutputProfile>,
    display: Option<Arc<OutputProfile>>,
    tables: [OnceLock<Tabulated>; ProofIntent::ALL.len()],
}

impl std::fmt::Debug for SoftProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoftProof")
            .field("press", &self.press)
            .field("display", &self.display)
            .finish_non_exhaustive()
    }
}

impl SoftProof {
    /// The proof of the CMYK `press` on `display`, an RGB or gray output
    /// profile, or the working space when `None`.
    ///
    /// # Errors
    ///
    /// [`OutputProfileError::Mismatch`] when `press` is not CMYK or
    /// `display` is.
    pub fn new(
        press: Arc<OutputProfile>,
        display: Option<Arc<OutputProfile>>,
    ) -> Result<Self, OutputProfileError> {
        if press.space() != OutputSpace::Cmyk {
            return Err(OutputProfileError::Mismatch(press.space()));
        }
        if let Some(display) = &display
            && display.space() == OutputSpace::Cmyk
        {
            return Err(OutputProfileError::Mismatch(OutputSpace::Cmyk));
        }
        Ok(Self {
            press,
            display,
            tables: std::array::from_fn(|_| OnceLock::new()),
        })
    }

    /// The press profile.
    #[must_use]
    pub const fn press(&self) -> &Arc<OutputProfile> {
        &self.press
    }

    /// The display's output profile; `None` for the working space.
    #[must_use]
    pub const fn display(&self) -> Option<&Arc<OutputProfile>> {
        self.display.as_ref()
    }

    /// Tabulate the conversion of `intent`, unless already tabulated.
    ///
    /// # Errors
    ///
    /// [`OutputProfileError::Transform`] when a conversion of the chain
    /// cannot be built, or [`OutputProfileError::Conversion`] when it
    /// fails; the error is kept and returned again on every later call.
    pub fn prepare(&self, intent: ProofIntent) -> Result<(), OutputProfileError> {
        self.table(intent).map(drop)
    }

    /// The table of `intent`, tabulated on first use.
    fn table(&self, intent: ProofIntent) -> Tabulated {
        self.tables[intent.index()]
            .get_or_init(|| self.tabulate(intent).map(Arc::new))
            .clone()
    }

    /// The display profile and its media white.
    fn display_profile(&self) -> (ColorProfile, moxcms::Xyzd) {
        self.display.as_ref().map_or_else(
            || {
                let mut srgb = ColorProfile::new_srgb();
                srgb.cicp = None;
                (srgb, crate::pcs::d50())
            },
            |p| (p.profile.clone(), crate::pcs::media_white(&p.profile)),
        )
    }

    /// The floating-point conversion of ink to the display under
    /// `intent`: to an RGB display directly, the scale of absolute
    /// colorimetry folded into the display's colorants; to a gray display
    /// through lightness-encoded XYZ, the scale folded into its colorants.
    fn leg(&self, intent: ProofIntent) -> Result<Leg, OutputProfileError> {
        let error = |reason: String| OutputProfileError::Transform {
            intent: intent.rendering_intent(),
            reason,
        };
        let cms = |e: CmsError| error(e.to_string());
        let press = &self.press.profile;
        let (mut display, display_white) = self.display_profile();
        let scale = intent
            .simulates_paper()
            .then(|| crate::pcs::absolute_scale(crate::pcs::media_white(press), display_white))
            .flatten();
        let options = TransformOptions {
            rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
            allow_use_cicp_transfer: false,
            prefer_fixed_point: false,
            ..TransformOptions::default()
        };
        let (layout, channels) = layout_of(&display);
        if display.color_space == moxcms::DataColorSpace::Gray {
            let pcs = as_version_2(&pcs_lightness()).map_err(cms)?;
            let mut source = pcs_lightness();
            if let Some(scale) = scale {
                crate::pcs::scale_colorants(&mut source, scale);
            }
            let source = as_version_2(&source).map_err(cms)?;
            let display = as_version_2(&display).map_err(cms)?;
            return Ok(Leg {
                channels,
                to_pcs: Some(
                    press
                        .create_transform_f64(Layout::Rgba, &pcs, Layout::Rgb, options)
                        .map_err(cms)?,
                ),
                to_display: source
                    .create_transform_f64(Layout::Rgb, &display, layout, options)
                    .map_err(cms)?,
            });
        }
        if let Some(scale) = scale {
            if !is_matrix_rgb(&display) {
                return Err(error(
                    "simulating the paper white needs the working space or a matrix/TRC \
                     display profile; this RGB one converts through tables"
                        .to_owned(),
                ));
            }
            crate::pcs::scale_colorants(&mut display, scale.map(f64::recip));
        }
        let display = as_version_2(&display).map_err(cms)?;
        Ok(Leg {
            channels,
            to_pcs: None,
            to_display: press
                .create_transform_f64(Layout::Rgba, &display, layout, options)
                .map_err(cms)?,
        })
    }

    /// Evaluate the conversion of `intent` at every node of the grid.
    fn tabulate(&self, intent: ProofIntent) -> Result<ProofTable, OutputProfileError> {
        let leg = self.leg(intent)?;
        let channels = leg.channels;
        let conversion = |e: CmsError| OutputProfileError::Conversion(e.to_string());
        let span = f64::from(SPAN);
        let row = PROOF_GRID * PROOF_GRID * PROOF_GRID;
        let mut ink = Vec::with_capacity(row * 4);
        let mut out = vec![0.0; row * channels];
        let mut nodes = Vec::with_capacity(row * PROOF_GRID * channels);
        for c in 0..=SPAN {
            ink.clear();
            for m in 0..=SPAN {
                for y in 0..=SPAN {
                    for k in 0..=SPAN {
                        ink.extend([c, m, y, k].map(|i| f64::from(i) / span));
                    }
                }
            }
            leg.transform(&ink, &mut out).map_err(conversion)?;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a value clamped to [0, 1] scaled to the u16 range"
            )]
            nodes.extend(
                out.iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16),
            );
        }
        Ok(ProofTable {
            channels,
            nodes: nodes.into_boxed_slice(),
        })
    }

    /// `canvas`, a page painted as the press's ink, converted to the
    /// display under `intent` at the ink's depth: into an RGB canvas for
    /// the working space or an RGB display profile, a gray one for a gray
    /// profile, built in `buffers`, into which the ink canvas's storage is
    /// then reclaimed.
    ///
    /// # Errors
    ///
    /// [`OutputProfileError::Mismatch`] for a canvas that is not CMYK; the
    /// errors of [`Self::prepare`]; [`OutputProfileError::Conversion`]
    /// when the display canvas cannot be allocated.
    pub fn convert(
        &self,
        canvas: FinishedCanvas,
        intent: ProofIntent,
        buffers: &mut CanvasBuffers,
    ) -> Result<FinishedCanvas, OutputProfileError> {
        let table = self.table(intent)?;
        let table = &*table;
        let gray = table.channels == 1;
        Ok(match canvas {
            FinishedCanvas::Cmyk8(cmyk) if gray => {
                let storage = buffers.gray.take_for(cmyk.has_alpha());
                let out = convert_to::<Cmyk8, Gray8>(table, &cmyk, storage)?;
                buffers.cmyk.reclaim(cmyk);
                FinishedCanvas::Gray8(out)
            }
            FinishedCanvas::Cmyk8(cmyk) => {
                let storage = buffers.rgb.take_for(cmyk.has_alpha());
                let out = convert_to::<Cmyk8, Rgb8>(table, &cmyk, storage)?;
                buffers.cmyk.reclaim(cmyk);
                FinishedCanvas::Rgb8(out)
            }
            FinishedCanvas::Cmyk16(cmyk) if gray => {
                let storage = buffers.gray16.take_for(cmyk.has_alpha());
                let out = convert_to::<Cmyk16, Gray16>(table, &cmyk, storage)?;
                buffers.cmyk16.reclaim(cmyk);
                FinishedCanvas::Gray16(out)
            }
            FinishedCanvas::Cmyk16(cmyk) => {
                let storage = buffers.rgb16.take_for(cmyk.has_alpha());
                let out = convert_to::<Cmyk16, Rgb16>(table, &cmyk, storage)?;
                buffers.cmyk16.reclaim(cmyk);
                FinishedCanvas::Rgb16(out)
            }
            _ => return Err(OutputProfileError::Mismatch(OutputSpace::Cmyk)),
        })
    }
}

/// A compiled floating-point conversion.
type F64Executor = Arc<dyn TransformExecutor<f64> + Send + Sync>;

/// The floating-point conversion of ink to `channels` display values:
/// through `to_pcs` into lightness-encoded XYZ when given, then
/// `to_display`.
struct Leg {
    channels: usize,
    to_pcs: Option<F64Executor>,
    to_display: F64Executor,
}

impl Leg {
    /// Convert the ink `src` into the display values `dst`.
    fn transform(&self, src: &[f64], dst: &mut [f64]) -> Result<(), CmsError> {
        let Some(to_pcs) = &self.to_pcs else {
            return self.to_display.transform(src, dst);
        };
        let mut pcs = vec![0.0; src.len() / 4 * 3];
        to_pcs.transform(src, &mut pcs)?;
        self.to_display.transform(&pcs, dst)
    }
}

/// Whether `profile` converts from the PCS to RGB through its colorant
/// matrix and tone curves alone, so that a scale of its PCS folds into
/// its colorants.
fn is_matrix_rgb(profile: &ColorProfile) -> bool {
    profile.color_space == moxcms::DataColorSpace::Rgb
        && profile.lut_b_to_a_perceptual.is_none()
        && profile.lut_b_to_a_colorimetric.is_none()
        && profile.lut_b_to_a_saturation.is_none()
        && profile.red_trc.is_some()
        && profile.green_trc.is_some()
        && profile.blue_trc.is_some()
}

/// A proof conversion tabulated over the ink cube: [`PROOF_GRID`]⁴ nodes
/// of `channels` 16-bit display values, `c` slowest and `k` fastest; node
/// `(c, m, y, k)` holds the display value of the ink `v / (PROOF_GRID −
/// 1)` for `v` = `c`, `m`, `y`, `k`.
struct ProofTable {
    channels: usize,
    nodes: Box<[u16]>,
}

impl ProofTable {
    /// The display value of the 16-bit ink `ink` into `out`: the
    /// interpolation in the simplex of the grid cell around it whose
    /// vertices are visited in decreasing order of the ink's fractional
    /// positions within the cell.
    fn eval(&self, ink: [u16; 4], out: &mut [u16]) {
        let full = u32::from(u16::MAX);
        let mut base = 0usize;
        let mut frac = [(0u32, 0usize); 4];
        for ((&v, &stride), f) in ink.iter().zip(&STRIDES).zip(&mut frac) {
            let scaled = u32::from(v) * SPAN;
            let cell = (scaled / full).min(SPAN - 1);
            base += cell as usize * stride;
            *f = (scaled - cell * full, stride);
        }
        frac.sort_unstable_by_key(|f| std::cmp::Reverse(f.0));
        let n = self.channels;
        let mut acc = [0u64; 3];
        let mut vertex = base;
        let mut previous = full;
        for &(f, stride) in &frac {
            let weight = u64::from(previous - f);
            for (a, &node) in acc.iter_mut().zip(&self.nodes[vertex * n..vertex * n + n]) {
                *a += weight * u64::from(node);
            }
            previous = f;
            vertex += stride;
        }
        for (a, &node) in acc.iter_mut().zip(&self.nodes[vertex * n..vertex * n + n]) {
            *a += u64::from(previous) * u64::from(node);
        }
        let half = u64::from(full) / 2;
        for (o, &a) in out.iter_mut().zip(&acc) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a convex combination of u16 nodes, divided by its weight sum, fits u16"
            )]
            let value = ((a + half) / u64::from(full)) as u16;
            *o = value;
        }
    }
}

impl TransformExecutor<u16> for ProofTable {
    fn transform(&self, src: &[u16], dst: &mut [u16]) -> Result<(), CmsError> {
        let (pixels, []) = src.as_chunks::<4>() else {
            return Err(CmsError::LaneSizeMismatch);
        };
        if dst.len() != pixels.len() * self.channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        for (&ink, out) in pixels.iter().zip(dst.chunks_exact_mut(self.channels)) {
            self.eval(ink, out);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
