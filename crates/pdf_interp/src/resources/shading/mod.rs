//! PDF shading resource resolution (Types 1–7).
//!
//! [`resolve_shading`] looks up a named `Shading` resource and returns a
//! [`ShadingResult`] which is either a [`raster::pipe::Pattern`] (for smooth
//! gradient types 1–3) or a pre-decoded triangle mesh (for mesh types 4–7).
//!
//! # Supported shading types
//!
//! | PDF type | Name | Support |
//! |---|---|---|
//! | 1 | Function-based | yes (evaluated per device pixel; sampled when very large) |
//! | 2 | Axial (linear) | yes |
//! | 3 | Radial | yes |
//! | 4 | Free-form Gouraud triangle mesh | yes |
//! | 5 | Lattice-form Gouraud mesh | yes |
//! | 6 | Coons patch mesh | yes |
//! | 7 | Tensor-product patch mesh | yes |
//!
//! # Colour spaces
//!
//! A shading's `ColorSpace` is resolved once into a [`ColorSpace`] — a device
//! space being replaced by the default space of the resources in effect
//! (PDF §8.6.5.6) — and every colour (function output or mesh vertex) is
//! converted to sRGB through [`ColorSpace::convert_to_rgb`], the conversion
//! used for fill colours.  A function output carries exactly
//! [`ColorSpace::ncomponents`] channels, as does a mesh vertex of a shading
//! without a `Function`.  A shading in a space whose colorants are all
//! `/None` paints nothing: it resolves to an empty mesh with no background.
//!
//! A shading in a `Separation` or `DeviceN` space resolved for an ink
//! canvas converts each colour instead to the ink of a [`ColorantPlan`], so
//! that it paints its colorants' channels alone; so does one in the space
//! of a `CalRGB` or ICC RGB transparency group on the group's canvas,
//! which paints its components.
//!
//! A shading resolved for a 16-bit canvas converts each colour at 16 bits
//! ([`ColorSpace::convert`]), and an axial or radial shading samples its
//! function more densely, so that a gradient keeps the levels of 16-bit
//! samples between its stops.
//!
//! # PDF Function objects
//!
//! The `Function` key maps the shading parameter to colour components
//! (evaluated by the `function` module).  Axial and radial shadings
//! evaluate it at uniformly spaced parameter values and interpolate the
//! converted colours, so every stop of a stitching function and any
//! non-linearity of the colour conversion are preserved.  A mesh shading
//! with a `Function` carries one parametric value per vertex, interpolated
//! across the mesh before the function applies (see the `mesh` module).

pub(crate) mod function;
mod gouraud;
mod mesh;
mod patch;
mod stream;
mod triangles;
mod type1;

use color::{Sample, SampleDepth};
use pdf::{Dictionary, Document, Object, Stream};
use raster::pipe::{ChannelMask, Pattern};
use raster::shading::ColorRamp;
use raster::shading::axial::AxialPattern;
use raster::shading::gouraud::GouraudVertex;
use raster::shading::radial::RadialPattern;

use super::colorspace::{self, ColorSpace, EXACT_WORK_BUDGET};
use super::dict_ext::DictExt;
use super::{PageResources, resolve_dict};
use crate::renderer::color::{ColorantPlan, Inks, PROCESS_CHANNELS, Srgb};
use crate::renderer::gstate::invert_affine;
use function::{Function, read_f64_array, read_fn_domain};
use mesh::{CornerSource, DirectColors, ParamColors, TriangleSink};
use patch::PatchKind;
use stream::{VertexFormat, read_decode_array};
pub(crate) use triangles::Mesh;
use type1::resolve_function_based;

// ── Public types ──────────────────────────────────────────────────────────────

/// The device canvas a shading is resolved for, in pixels, the depth of
/// its samples and the working space and rendering intent its colours
/// convert to.
/// Device y runs down from the top edge, so `height` places the page's
/// origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Canvas {
    pub width: f64,
    pub height: f64,
    pub depth: SampleDepth,
    pub target: crate::ColourTarget,
}

/// The colour source of a resolved shading.
///
/// - `Pattern`: smooth gradient — use with [`raster::shading::shaded_fill`].
///   Pixels outside the shading's extent are left unpainted by the pattern.
/// - `Mesh`: pre-decoded Gouraud triangles, indexed by device row — call
///   [`raster::shading::gouraud::gouraud_triangle_fill`] for each triangle
///   [`Mesh::meeting`] the fill's clip.
///
/// The colours are sRGB, of 16-bit samples for `Pattern16` and `Mesh16`,
/// or for `InkPattern` and `InkMesh` the eight ink amounts of an ink pixel.
pub enum ShadingResult {
    /// Smooth gradient — use with [`raster::shading::shaded_fill`].
    Pattern(Box<dyn Pattern + Send + Sync>),
    /// Pre-decoded Gouraud triangles, indexed by device row.
    Mesh(Mesh),
    /// Smooth gradient of 16-bit sRGB.
    Pattern16(Box<dyn Pattern<u16> + Send + Sync>),
    /// Gouraud triangles of 16-bit sRGB, indexed by device row.
    Mesh16(Mesh<3, u16>),
    /// Smooth gradient of ink.
    InkPattern(Box<dyn Pattern + Send + Sync>),
    /// Gouraud triangles of ink, indexed by device row.
    InkMesh(Mesh<8>),
    /// Smooth gradient of 16-bit ink.
    InkPattern16(Box<dyn Pattern<u16> + Send + Sync>),
    /// Gouraud triangles of 16-bit ink, indexed by device row.
    InkMesh16(Mesh<8, u16>),
}

/// A shading's colours of `N` samples of type `S`: a pattern or a mesh.
enum Colours<const N: usize, S: Sample> {
    Pattern(Box<dyn Pattern<S> + Send + Sync>),
    Mesh(Mesh<N, S>),
}

/// A shading dictionary resolved for painting (PDF §8.7.4.3).
pub struct ResolvedShading {
    /// The shading's colours.
    pub result: ShadingResult,
    /// `BBox`, the temporary clipping boundary of the shading, as the
    /// device-space corners of its rectangle in the shading's target space.
    pub bbox: Option<[(f64, f64); 4]>,
    /// `Background` converted to sRGB, at 16 bits for a shading resolved
    /// for a 16-bit canvas.  It applies only when the shading is painted as
    /// a shading pattern, never through the `sh` operator.
    pub background: Option<Srgb>,
    /// `Background` as ink, for a shading whose colours are ink, at the
    /// depth the shading was resolved at.
    pub background_ink: Option<Inks>,
    /// The channels the shading paints on an ink canvas: the process
    /// channels for a shading in a process CMYK space, the channels of its
    /// [`ColorantPlan`] for a shading of ink, every channel otherwise.  The
    /// sRGB colours of any other shading reach the ink canvas through their
    /// conversion to process ink, which writes every channel.
    pub paints: ChannelMask,
    /// Heap bytes `result` holds: a mesh's triangles or a function-based
    /// shading's samples.
    pub held_bytes: usize,
    /// Whether the shading's colour space is a gray one
    /// ([`ColorSpace::is_gray`]), whose colours separate to black ink
    /// alone.
    pub gray: bool,
}

impl ResolvedShading {
    /// The channels that carry the shading's colours on an ink canvas:
    /// those it paints for colours of ink, the process channels for sRGB
    /// colours, which reach the canvas as process ink.
    #[must_use]
    pub(crate) const fn carries(&self) -> ChannelMask {
        match self.result {
            ShadingResult::InkPattern(_)
            | ShadingResult::InkMesh(_)
            | ShadingResult::InkPattern16(_)
            | ShadingResult::InkMesh16(_) => self.paints,
            ShadingResult::Pattern(_)
            | ShadingResult::Mesh(_)
            | ShadingResult::Pattern16(_)
            | ShadingResult::Mesh16(_) => PROCESS_CHANNELS,
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// The plan by which the colours of a space paint as ink on the ink canvas
/// a shading is resolved for, `None` for a space whose colours paint their
/// sRGB conversion (see [`resolve_shading_object`]).
pub(crate) type PlanColorants<'a> = &'a mut dyn FnMut(&ColorSpace) -> Option<ColorantPlan>;

/// Resolve the named shading from `shadings`, the `/Shading` resource
/// sub-dictionary in effect, whose default colour spaces `resources`
/// supplies.
///
/// Returns `None` if the name is absent, the type is unsupported, or any
/// required key is missing.  A warning is logged for unsupported types so the
/// caller can fall back gracefully.
#[must_use]
pub(crate) fn resolve_shading(
    resources: &PageResources<'_>,
    shadings: &Dictionary,
    name: &[u8],
    ctm: &[f64; 6],
    canvas: Canvas,
    ink: Option<PlanColorants<'_>>,
) -> Option<ResolvedShading> {
    resolve_shading_object(resources, shadings.get(name)?, ctm, canvas, ink)
}

/// Resolve the shading `sh_obj` (a dictionary, a stream, or a reference to
/// either) whose target space `ctm` maps to the device, with the default
/// colour spaces of `resources`.
///
/// With `ink`, the shading is resolved for an ink canvas: a shading in a
/// space `ink` gives a plan — a `Separation` or `DeviceN` space, process
/// CMYK — converts its colours to the ink of that plan, and paints that
/// plan's channels.
///
/// Returns `None` if the type is unsupported or any required key is missing.
#[must_use]
pub(crate) fn resolve_shading_object(
    resources: &PageResources<'_>,
    sh_obj: &Object,
    ctm: &[f64; 6],
    canvas: Canvas,
    ink: Option<PlanColorants<'_>>,
) -> Option<ResolvedShading> {
    let doc = resources.doc();
    let page_h = canvas.height;
    // Shading types 2–3 are plain dicts; types 4–7 are streams.
    // Try stream first (stream has a dict too), then fall back to plain dict.
    // Owned Dictionary because the new pdf crate hands back Arc<Object> from
    // get_object — we can't keep a borrow into a temporary Arc.
    let (sh_dict, stream_data): (Dictionary, Option<Vec<u8>>) = match sh_obj {
        Object::Stream(s) => (s.dict.clone(), stream_content(s)),
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            match referent.as_ref() {
                Object::Stream(s) => (s.dict.clone(), stream_content(s)),
                obj => (resolve_dict(doc, obj)?, None),
            }
        }
        obj => (resolve_dict(doc, obj)?, None),
    };

    let shading_type = sh_dict.get_i64(b"ShadingType")?;

    let cs_space = resources.device_space(colorspace::resolve(doc, sh_dict.get(b"ColorSpace")?));
    if matches!(cs_space, ColorSpace::Pattern { .. }) {
        log::warn!("shading: Pattern colour space is not valid for a shading — skipping");
        return None;
    }
    let plan = ink.and_then(|plan| plan(&cs_space));
    let paints = match &plan {
        Some(plan) => plan.paints(),
        None if cs_space.is_process_cmyk() => PROCESS_CHANNELS,
        None => ChannelMask::ALL,
    };
    if cs_space.paints_nothing() {
        log::debug!("shading: every colorant is /None — the shading paints nothing");
        return Some(ResolvedShading {
            result: ShadingResult::Mesh(Mesh::new(Vec::new())),
            bbox: None,
            background: None,
            background_ink: None,
            paints,
            held_bytes: 0,
            gray: cs_space.is_gray(),
        });
    }
    let cs = ShadingCs {
        doc,
        space: &cs_space,
        target: canvas.target,
    };
    let bbox = read_bbox(&sh_dict, ctm, page_h);
    let source = Source {
        doc,
        sh: &sh_dict,
        data: stream_data,
        shading_type,
        ctm,
        canvas,
        bbox: bbox.as_ref(),
    };
    let background = read_background(&sh_dict, cs.channels());
    let (result, held_bytes, background_ink) =
        shading_colours(&source, cs, plan.as_ref(), background.as_deref())?;
    Some(ResolvedShading {
        result,
        bbox,
        background: background.as_deref().map(|c| cs.to_srgb(c, canvas.depth)),
        background_ink,
        paints,
        held_bytes,
        gray: cs_space.is_gray(),
    })
}

/// The colours of the shading `source` in `cs`, at the depth of its
/// canvas — the ink of `plan` when the shading is of ink — with the heap
/// bytes they hold and its `background`, as ink, for a shading of ink.
fn shading_colours(
    source: &Source<'_>,
    cs: ShadingCs<'_>,
    plan: Option<&ColorantPlan>,
    background: Option<&[f64]>,
) -> Option<(ShadingResult, usize, Option<Inks>)> {
    Some(match (plan, source.canvas.depth) {
        (None, SampleDepth::Eight) => {
            let (colours, held) = source.colours::<3, u8, _>(cs)?;
            let result = match colours {
                Colours::Pattern(p) => ShadingResult::Pattern(p),
                Colours::Mesh(m) => ShadingResult::Mesh(m),
            };
            (result, held, None)
        }
        (None, SampleDepth::Sixteen) => {
            let (colours, held) = source.colours::<3, u16, _>(cs)?;
            let result = match colours {
                Colours::Pattern(p) => ShadingResult::Pattern16(p),
                Colours::Mesh(m) => ShadingResult::Mesh16(m),
            };
            (result, held, None)
        }
        (Some(plan), SampleDepth::Eight) => {
            let ink = InkCs { cs, plan };
            let (colours, held) = source.colours::<8, u8, _>(ink)?;
            let result = match colours {
                Colours::Pattern(p) => ShadingResult::InkPattern(p),
                Colours::Mesh(m) => ShadingResult::InkMesh(m),
            };
            (
                result,
                held,
                background.map(|c| Inks::of::<u8>(ink.convert(c))),
            )
        }
        (Some(plan), SampleDepth::Sixteen) => {
            let ink = InkCs { cs, plan };
            let (colours, held) = source.colours::<8, u16, _>(ink)?;
            let result = match colours {
                Colours::Pattern(p) => ShadingResult::InkPattern16(p),
                Colours::Mesh(m) => ShadingResult::InkMesh16(m),
            };
            (
                result,
                held,
                background.map(|c| Inks::of::<u16>(ink.convert(c))),
            )
        }
    })
}

/// The dictionary and stream of a shading, with the device geometry it is
/// resolved for.
struct Source<'a> {
    doc: &'a Document,
    sh: &'a Dictionary,
    /// The decompressed stream of a mesh shading.
    data: Option<Vec<u8>>,
    shading_type: i64,
    ctm: &'a [f64; 6],
    canvas: Canvas,
    bbox: Option<&'a [(f64, f64); 4]>,
}

impl Source<'_> {
    /// The shading's colours converted through `cs`, with the heap bytes
    /// they hold; `None` when the shading cannot be resolved.
    fn colours<const N: usize, S: Sample, C: ShadingColour<N, S>>(
        &self,
        cs: C,
    ) -> Option<(Colours<N, S>, usize)> {
        let (doc, sh, ctm) = (self.doc, self.sh, self.ctm);
        let page_h = self.canvas.height;
        let n_channels = cs.channels();
        match self.shading_type {
            1 => resolve_function_based(doc, sh, cs, n_channels, ctm, self.canvas, self.bbox)
                .map(|(pattern, bytes)| (Colours::Pattern(pattern), bytes)),
            2 => resolve_axial(doc, sh, cs, n_channels, ctm, page_h)
                .map(|p| (Colours::Pattern(p), 0)),
            3 => resolve_radial(doc, sh, cs, n_channels, ctm, page_h)
                .map(|p| (Colours::Pattern(p), 0)),
            shading_type @ 4..=7 => {
                let Some(data) = &self.data else {
                    log::warn!("shading: ShadingType {shading_type} must be a stream — skipping");
                    return None;
                };
                let mesh_type = u8::try_from(shading_type).ok()?;
                // PDF §8.7.4.5.5: with a Function, each vertex carries one
                // parametric value t, mapped through the function; its Decode
                // range is the interval the function is evaluated over.
                let mesh = if let Some(fn_obj) = sh.get(b"Function") {
                    let decode = read_decode_array(sh, 1);
                    let ramp = sample_ramp(doc, fn_obj, cs, n_channels, decode[4], decode[5])?;
                    decode_mesh(
                        mesh_type,
                        sh,
                        data,
                        &ParamColors { ramp },
                        &decode,
                        ctm,
                        page_h,
                    )
                } else {
                    let decode = read_decode_array(sh, n_channels);
                    decode_mesh(
                        mesh_type,
                        sh,
                        data,
                        &DirectColors::new(cs),
                        &decode,
                        ctm,
                        page_h,
                    )
                };
                let mesh = Mesh::new(mesh);
                let held = std::mem::size_of_val(mesh.triangles());
                Some((Colours::Mesh(mesh), held))
            }
            other => {
                log::warn!("shading: ShadingType {other} is not a shading type (1–7) — skipping");
                None
            }
        }
    }
}

/// Read `BBox` as the device-space corners of its rectangle, whose
/// coordinates lie in the space `ctm` maps to the device.  A malformed
/// `BBox` is ignored.
fn read_bbox(sh: &Dictionary, ctm: &[f64; 6], page_h: f64) -> Option<[(f64, f64); 4]> {
    let bb = read_f64_array(sh, b"BBox", 4)?;
    let corners = [
        (bb[0], bb[1]),
        (bb[2], bb[1]),
        (bb[2], bb[3]),
        (bb[0], bb[3]),
    ]
    .map(|(x, y)| transform_point(ctm, x, y, page_h));
    if corners.iter().all(|(x, y)| x.is_finite() && y.is_finite()) {
        Some(corners)
    } else {
        log::warn!("shading: non-finite BBox — ignoring BBox");
        None
    }
}

/// Read `Background`, one colour of `n` components in the shading's colour
/// space.  A `Background` of the wrong length is ignored.
fn read_background(sh: &Dictionary, n: usize) -> Option<Vec<f64>> {
    let arr = sh.get(b"Background")?.as_array()?;
    if arr.len() != n {
        log::warn!(
            "shading: Background has {} components, the colour space {n} — ignoring Background",
            arr.len()
        );
        return None;
    }
    read_f64_array(sh, b"Background", n)
}

/// Decode the stream of mesh shading `shading_type` (4–7) with vertex
/// colours from `source` and the shading's `Decode` array `decode`.
fn decode_mesh<const N: usize, S: CornerSource<N>>(
    shading_type: u8,
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    decode: &[f64],
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex<N, S::Sample>; 3]> {
    let Some(format) = VertexFormat::parse(sh, shading_type, decode, ctm, page_h) else {
        return Vec::new();
    };
    let mut sink = TriangleSink::new();
    match shading_type {
        4 => gouraud::decode_free_form(sh, data, source, &format, &mut sink),
        5 => gouraud::decode_lattice(sh, data, source, &format, &mut sink),
        6 => patch::decode_patch_mesh(sh, data, source, &format, PatchKind::Coons, &mut sink),
        _ => patch::decode_patch_mesh(sh, data, source, &format, PatchKind::Tensor, &mut sink),
    }
    sink.finish()
}

/// Decompress a stream, logging a warning on failure.
fn stream_content(s: &Stream) -> Option<Vec<u8>> {
    match s.decompressed_content() {
        Ok(data) => Some(data),
        Err(e) => {
            log::warn!("shading: failed to decompress stream — {e}");
            None
        }
    }
}

// ── Axial (Type 2) ────────────────────────────────────────────────────────────

fn resolve_axial<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    doc: &Document,
    sh: &Dictionary,
    cs: C,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Box<dyn Pattern<S> + Send + Sync>> {
    // Coords: [x0, y0, x1, y1] in user space.
    let coords = read_f64_array(sh, b"Coords", 4)?;

    // Domain: [t0, t1] — defaults to [0, 1].
    let (t0, t1) = read_fn_domain(sh);

    // Extend: [extend_start, extend_end].
    let (ext_s, ext_e) = read_extend(sh);

    let ramp = sample_ramp(doc, sh.get(b"Function")?, cs, n, t0, t1)?;

    if !coords.iter().all(|v| v.is_finite()) {
        log::warn!("shading/axial: non-finite Coords — skipping");
        return None;
    }
    let Some(inverse) = device_to_space(ctm, page_h) else {
        log::warn!("shading/axial: singular or non-finite CTM — skipping");
        return None;
    };

    // The axis stays in the shading's target space; each device pixel is
    // projected onto it there.
    let pattern = AxialPattern::new(
        ramp, coords[0], coords[1], coords[2], coords[3], ext_s, ext_e,
    )
    .in_shading_space(&inverse);
    Some(Box::new(pattern))
}

/// Number of parameter values at which an axial or radial shading's
/// function is evaluated for samples of `depth`: 256 at 8 bits; 1024 at 16
/// bits, where the piecewise-linear ramp between them stays within a few
/// 16-bit levels of the function (see the bound in this module's tests).
const fn ramp_samples(depth: SampleDepth) -> u16 {
    match depth {
        SampleDepth::Eight => 256,
        SampleDepth::Sixteen => 1024,
    }
}

/// Evaluate the shading function at [`ramp_samples`] uniformly spaced
/// parameter values from `t0` to `t1` and convert each colour through `cs`.
///
/// PDF §8.7.4.5.3: the colour at parameter `t` is the colour-space
/// conversion of `Function(t)`; between samples the ramp interpolates the
/// converted samples.  `None` when the function fails to evaluate at any
/// sample.
fn sample_ramp<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    doc: &Document,
    fn_obj: &Object,
    cs: C,
    n: usize,
    t0: f64,
    t1: f64,
) -> Option<ColorRamp<N, S>> {
    let function = Function::load(doc, fn_obj);
    let count = ramp_samples(S::DEPTH);
    let last = f64::from(count - 1);
    let samples = (0..count)
        .map(|k| {
            let t = (f64::from(k) / last).mul_add(t1 - t0, t0);
            function.eval(t, n).map(|c| cs.convert(&c)).ok_or(t)
        })
        .collect::<Result<Vec<_>, f64>>()
        .map_err(|t| {
            log::warn!("shading: Function yields no {n}-component colour at t = {t} — skipping");
        })
        .ok()?;
    ColorRamp::from_samples(samples)
}

// ── Radial (Type 3) ───────────────────────────────────────────────────────────

fn resolve_radial<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    doc: &Document,
    sh: &Dictionary,
    cs: C,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Box<dyn Pattern<S> + Send + Sync>> {
    // Coords: [x0, y0, r0, x1, y1, r1] in user space.
    let coords = read_f64_array(sh, b"Coords", 6)?;

    let (t0, t1) = read_fn_domain(sh);
    let (ext_s, ext_e) = read_extend(sh);

    let ramp = sample_ramp(doc, sh.get(b"Function")?, cs, n, t0, t1)?;

    if !coords.iter().all(|v| v.is_finite()) {
        log::warn!("shading/radial: non-finite Coords — skipping");
        return None;
    }
    let Some(inverse) = device_to_space(ctm, page_h) else {
        log::warn!("shading/radial: singular or non-finite CTM — skipping");
        return None;
    };

    // The circles stay in the shading's target space, so the CTM draws
    // them as their affine images.  PDF requires r ≥ 0; a negative radius
    // is taken by magnitude.
    let pattern = RadialPattern::new(
        ramp,
        coords[0],
        coords[1],
        coords[2].abs(),
        coords[3],
        coords[4],
        coords[5].abs(),
        ext_s,
        ext_e,
    )
    .in_shading_space(&inverse);
    Some(Box::new(pattern))
}

// ── Colour conversion helpers ─────────────────────────────────────────────────

/// A shading's resolved colour space, bound to the document its
/// parameterised variants (ICC profile, tint transform) dereference into.
#[derive(Clone, Copy)]
pub(super) struct ShadingCs<'a> {
    doc: &'a Document,
    space: &'a ColorSpace,
    /// The working space and rendering intent the colours convert to.
    target: crate::ColourTarget,
}

impl ShadingCs<'_> {
    /// Number of colour components a function output or mesh vertex carries.
    pub(super) fn channels(self) -> usize {
        self.space.ncomponents()
    }

    /// Convert colour components (nominally in `[0, 1]`) to a triple of the
    /// working space.
    ///
    /// A slice shorter than [`Self::channels`] reads the missing components
    /// as `0.0`; surplus components are ignored.  Out-of-range values clamp
    /// and `NaN` maps to the low end of the range, as in
    /// [`ColorSpace::convert_to_rgb`].
    pub(super) fn to_rgb<S: Sample>(self, channels: &[f64]) -> [S; 3] {
        let n = self.channels();
        if channels.len() == n {
            return self.space.convert(self.doc, self.target, channels);
        }
        let mut fitted = channels[..channels.len().min(n)].to_vec();
        fitted.resize(n, 0.0);
        self.space.convert(self.doc, self.target, &fitted)
    }

    /// [`Self::to_rgb`] at both depths for a canvas of `depth` (see
    /// [`ColorSpace::convert_to_srgb`]).
    fn to_srgb(self, channels: &[f64], depth: SampleDepth) -> Srgb {
        let bytes = self.to_rgb(channels);
        match depth {
            SampleDepth::Eight => Srgb::of_bytes(bytes),
            SampleDepth::Sixteen => Srgb {
                bytes,
                wide: self.to_rgb(channels),
            },
        }
    }
}

/// The conversion of a shading's colour components to the `N` samples of
/// type `S` of a pixel.
pub(super) trait ShadingColour<const N: usize, S: Sample>: Copy {
    /// Number of colour components a function output or mesh vertex carries.
    fn channels(self) -> usize;

    /// The pixel samples of the colour components `comps`, fitted to
    /// [`Self::channels`] as [`ShadingCs::to_rgb`] fits them.
    fn convert(self, comps: &[f64]) -> [S; N];

    /// The work of one conversion, in the units of
    /// [`ColorSpace::conversion_cost`].
    fn conversion_cost(self) -> u64;
}

impl<S: Sample> ShadingColour<3, S> for ShadingCs<'_> {
    fn channels(self) -> usize {
        Self::channels(self)
    }

    fn convert(self, comps: &[f64]) -> [S; 3] {
        self.to_rgb(comps)
    }

    fn conversion_cost(self) -> u64 {
        self.space.conversion_cost()
    }
}

/// A shading space of ink — `Separation`, `DeviceN`, process CMYK — or of
/// the group whose canvas the shading paints, converting to the ink of
/// `plan`.
#[derive(Clone, Copy)]
struct InkCs<'a> {
    cs: ShadingCs<'a>,
    plan: &'a ColorantPlan,
}

impl<S: Sample> ShadingColour<8, S> for InkCs<'_> {
    fn channels(self) -> usize {
        self.cs.channels()
    }

    fn convert(self, comps: &[f64]) -> [S; 8] {
        let (doc, space) = (self.cs.doc, self.cs.space);
        let n = self.cs.channels();
        if comps.len() == n {
            return self.plan.ink(doc, self.cs.target, space, comps);
        }
        let mut fitted = comps[..comps.len().min(n)].to_vec();
        fitted.resize(n, 0.0);
        self.plan.ink(doc, self.cs.target, space, &fitted)
    }

    fn conversion_cost(self) -> u64 {
        match self.plan {
            ColorantPlan::Channels(_) | ColorantPlan::Linked(_) | ColorantPlan::Complements => 1,
            ColorantPlan::Process(_) => self.cs.space.process_ink_cost(),
        }
    }
}

// ── Coordinate transforms ─────────────────────────────────────────────────────

/// Transform a user-space point through the CTM and y-flip to device space.
///
/// Applies the affine CTM `[a b c d e f]` to `(x, y)` then flips the y axis
/// with `page_h − dy`.  This differs from `gstate::ctm_transform` which does
/// not include the y-flip.
pub(super) fn transform_point(ctm: &[f64; 6], x: f64, y: f64, page_h: f64) -> (f64, f64) {
    let dx = ctm[0].mul_add(x, ctm[2] * y) + ctm[4];
    let dy = ctm[1].mul_add(x, ctm[3] * y) + ctm[5];
    (dx, page_h - dy)
}

/// The affine map from device pixels to the space `ctm` maps to the device:
/// the inverse of [`transform_point`].  `None` when `ctm` is singular or not
/// finite.
pub(super) fn device_to_space(ctm: &[f64; 6], page_h: f64) -> Option<[f64; 6]> {
    invert_affine(&[ctm[0], -ctm[1], ctm[2], -ctm[3], ctm[4], page_h - ctm[5]])
}

/// Read the `Extend` array `[extend_start, extend_end]`; both default to `false`.
fn read_extend(sh: &Dictionary) -> (bool, bool) {
    let arr = sh.get(b"Extend").and_then(|o| o.as_array());
    let Some(a) = arr.filter(|a| a.len() >= 2) else {
        return (false, false);
    };
    let b = |i: usize| a.get(i).and_then(pdf::Object::as_bool).unwrap_or(false);
    (b(0), b(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::gstate::ctm_transform;

    use crate::test_helpers::empty_doc;

    /// The largest difference, in 16-bit levels, between the ramp
    /// `sample_ramp` builds of the gray function `function` and the function
    /// evaluated and converted at each of 65 536 positions over its domain.
    fn ramp_error_16(function: &[u8]) -> u16 {
        let doc = empty_doc();
        let obj = pdf::parse_object(function, &mut 0).expect("the function parses");
        let cs = ShadingCs {
            doc: &doc,
            space: &ColorSpace::DeviceGray,
            target: crate::ColourTarget::SRGB,
        };
        let ramp: ColorRamp<3, u16> = sample_ramp(&doc, &obj, cs, 1, 0.0, 1.0).expect("a ramp");
        let exact = Function::load(&doc, &obj);
        (0..=u16::MAX)
            .map(|i| {
                let t = i.to_unit();
                let mut got = [0u16; 3];
                ramp.write(t, &mut got);
                let want = u16::from_unit(exact.eval(t, 1).expect("evaluates")[0]);
                got[0].abs_diff(want)
            })
            .max()
            .unwrap_or(0)
    }

    /// The 16-bit ramp of an axial or radial shading, 1024 samples of the
    /// function interpolated linearly, tracks the function within 1 level
    /// of 16 bits for an exponential function of N = 2.2 (the
    /// interpolation error is below 0.02 level, the rest the rounding of
    /// the samples and of the blend).  A stitching function's kink between
    /// two samples is cut by the chord across it: the bound is then
    /// `Δslope / (4 · 1023)` of full scale, 16 levels for slopes ½ and 1½
    /// meeting at the middle of the domain, which falls midway between two
    /// samples, plus the same rounding.
    #[test]
    fn a_16_bit_ramp_tracks_its_function() {
        let exponential =
            ramp_error_16(b"<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 2.2 >>");
        assert!(
            exponential <= 1,
            "exponential N = 2.2: {exponential} levels"
        );
        let stitching = ramp_error_16(
            b"<< /FunctionType 3 /Domain [0 1] /Bounds [0.5] /Encode [0 1 0 1] \
              /Functions [<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [0.25] /N 1 >> \
                          << /FunctionType 2 /Domain [0 1] /C0 [0.25] /C1 [1] /N 1 >>] >>",
        );
        assert!(stitching <= 17, "stitching at 0.5: {stitching} levels");
        eprintln!("16-bit ramp error: exponential {exponential}, stitching {stitching} levels");
    }

    fn to_rgb(space: &ColorSpace, channels: &[f64]) -> [u8; 3] {
        let doc = empty_doc();
        ShadingCs {
            doc: &doc,
            space,
            target: crate::ColourTarget::SRGB,
        }
        .to_rgb(channels)
    }

    #[test]
    fn to_rgb_gray_midpoint() {
        // round(0.5 * 255) = 128
        assert_eq!(to_rgb(&ColorSpace::DeviceGray, &[0.5]), [128, 128, 128]);
    }

    #[test]
    fn to_rgb_rgb_passthrough() {
        assert_eq!(
            to_rgb(&ColorSpace::DeviceRgb, &[1.0, 0.0, 0.5]),
            [255, 0, 128]
        );
    }

    #[test]
    fn to_rgb_cmyk_reads_four_ink_channels() {
        assert_eq!(
            to_rgb(&ColorSpace::DeviceCmyk, &[0.85, 0.69, 0.0, 0.0]),
            color::cmyk::cmyk_f64_to_rgb(0.85, 0.69, 0.0, 0.0)
        );
        assert_eq!(
            to_rgb(&ColorSpace::DeviceCmyk, &[0.0, 0.0, 0.0, 1.0]),
            color::cmyk::CORNERS[8]
        );
        assert_eq!(
            to_rgb(&ColorSpace::DeviceCmyk, &[0.0, 0.0, 1.0, 0.0]),
            color::cmyk::CORNERS[1]
        );
    }

    /// Short channel slices default missing channels to 0, not panic —
    /// malformed shading functions sometimes return undersized arrays.
    #[test]
    fn to_rgb_short_slice_defaults_to_zero() {
        assert_eq!(to_rgb(&ColorSpace::DeviceRgb, &[]), [0, 0, 0]);
        assert_eq!(to_rgb(&ColorSpace::DeviceRgb, &[1.0]), [255, 0, 0]);
        assert_eq!(to_rgb(&ColorSpace::DeviceCmyk, &[]), [255, 255, 255]);
    }

    #[test]
    fn to_rgb_ignores_surplus_channels() {
        assert_eq!(
            to_rgb(&ColorSpace::DeviceGray, &[1.0, 0.0, 0.0]),
            [255, 255, 255]
        );
    }

    #[test]
    fn to_rgb_out_of_range_and_nan() {
        assert_eq!(
            to_rgb(&ColorSpace::DeviceRgb, &[2.0, -0.5, 0.5]),
            [255, 0, 128]
        );
        assert_eq!(
            to_rgb(&ColorSpace::DeviceRgb, &[f64::NAN, 1.0, 1.0]),
            [0, 255, 255]
        );
    }

    #[test]
    fn transform_point_identity_with_yflip() {
        let ctm = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (dx, dy) = transform_point(&ctm, 10.0, 20.0, 100.0);
        assert!((dx - 10.0).abs() < 1e-9);
        assert!((dy - 80.0).abs() < 1e-9); // y-flip: 100 - 20
    }

    #[test]
    fn device_to_space_inverts_transform_point() {
        let ctm = [2.0, 0.5, -1.0, 3.0, 7.0, 11.0];
        let inv = device_to_space(&ctm, 100.0).expect("invertible");
        let (dx, dy) = transform_point(&ctm, 4.0, -6.0, 100.0);
        let back = ctm_transform(&inv, dx, dy);
        assert!((back.0 - 4.0).abs() < 1e-9 && (back.1 + 6.0).abs() < 1e-9);
    }

    #[test]
    fn device_to_space_rejects_singular_ctm() {
        assert_eq!(device_to_space(&[1.0, 2.0, 2.0, 4.0, 0.0, 0.0], 10.0), None);
        assert_eq!(
            device_to_space(&[f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0], 10.0),
            None
        );
    }
}
