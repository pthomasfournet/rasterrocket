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
//! that it paints its colorants' channels alone.
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

use color::{Sample, SampleDepth};
use pdf::{Dictionary, Document, Object, Stream};
use raster::pipe::{ChannelMask, Pattern};
use raster::shading::ColorRamp;
use raster::shading::axial::AxialPattern;
use raster::shading::function::FunctionPattern;
use raster::shading::gouraud::GouraudVertex;
use raster::shading::radial::RadialPattern;

use super::colorspace::{self, ColorSpace, EXACT_WORK_BUDGET};
use super::dict_ext::DictExt;
use super::lattice::{Axis, for_each_corner};
use super::{PageResources, read_matrix, resolve_dict};
use crate::renderer::color::{ColorantPlan, Inks, PROCESS_CHANNELS, Srgb};
use crate::renderer::gstate::{ctm_multiply, ctm_transform, invert_affine};
use function::{Function, read_f64_array, read_fn_domain};
use mesh::{CornerSource, DirectColors, ParamColors, TriangleSink};
use patch::PatchKind;
use stream::{VertexFormat, read_decode_array};
pub(crate) use triangles::Mesh;

// ── Public types ──────────────────────────────────────────────────────────────

/// The device canvas a shading is resolved for, in pixels, and the depth
/// of its samples.  Device y runs down from the top edge, so `height`
/// places the page's origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Canvas {
    pub width: f64,
    pub height: f64,
    pub depth: SampleDepth,
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
        });
    }
    let cs = ShadingCs {
        doc,
        space: &cs_space,
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

// ── Function-based (Type 1) ───────────────────────────────────────────────────

/// Most samples of a Type 1 shading's device lattice (12 MiB of RGB).
const TYPE1_LATTICE_MAX: usize = 1 << 22;

/// Tolerance, in units of the Domain's extent, admitting a point on a
/// Type 1 shading's Domain edge despite rounding error.
const TYPE1_EDGE_TOLERANCE: f64 = 1e-9;

/// The device pixels a Type 1 shading is sampled at: every `step`-th
/// pixel from `(x0, y0)`, `cols × rows` of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lattice {
    x0: i64,
    y0: i64,
    step: usize,
    cols: usize,
    rows: usize,
}

impl Lattice {
    /// The lattice over the integer device pixels of `area`, the inclusive
    /// pixel bounds `[x0, y0, x1, y1]`: step 1 while the area holds at most
    /// `max_points` pixels, else the least step that keeps the lattice
    /// within it, the last row and column reaching the far edges.  `None`
    /// when the area is empty.  `max_points` is at least 4, which any area
    /// fits at a step of its longer side.
    fn covering([x0, y0, x1, y1]: [i64; 4], max_points: usize) -> Option<Self> {
        debug_assert!(max_points >= 4, "a lattice needs its four corners");
        let w = usize::try_from(x1 - x0 + 1).ok().filter(|&w| w > 0)?;
        let h = usize::try_from(y1 - y0 + 1).ok().filter(|&h| h > 0)?;
        let dims = |step: usize| ((w - 1).div_ceil(step) + 1, (h - 1).div_ceil(step) + 1);
        let fits = |step: usize| {
            let (cols, rows) = dims(step);
            cols.saturating_mul(rows) <= max_points
        };
        // `fits` is monotone in the step and holds at the longer side.
        let (mut lo, mut hi) = (1, w.max(h));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let (cols, rows) = dims(lo);
        Some(Self {
            x0,
            y0,
            step: lo,
            cols,
            rows,
        })
    }

    /// Device pixel of lattice point `(col, row)`.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the lattice lies within the canvas, far below 2^52 pixels"
    )]
    fn pixel(self, col: usize, row: usize) -> (f64, f64) {
        (
            self.x0 as f64 + (self.step * col) as f64,
            self.y0 as f64 + (self.step * row) as f64,
        )
    }

    /// The interpolation axes of device pixel `(px, py)`: the lattice
    /// columns and rows around it, clamped to the lattice.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the lattice lies within the canvas, far below 2^52 pixels"
    )]
    fn axes(self, px: f64, py: f64) -> [Axis; 2] {
        let step = self.step as f64;
        [
            Axis::at((px - self.x0 as f64) / step, self.cols, 1),
            Axis::at((py - self.y0 as f64) / step, self.rows, self.cols),
        ]
    }
}

/// The inclusive integer pixel bounds `[x0, y0, x1, y1]` of the device
/// points `corners` that lie on `canvas`, widened by one pixel so a pixel
/// on the canvas edge has an interpolation neighbour; within `bbox`'s
/// corners too when given.  Empty bounds have `x1 < x0` or `y1 < y0`.
/// Every corner is finite.
fn type1_area(corners: &[(f64, f64)], bbox: Option<&[(f64, f64); 4]>, canvas: Canvas) -> [i64; 4] {
    let span = |points: &[(f64, f64)], axis: fn(&(f64, f64)) -> f64| {
        let lo = points.iter().map(axis).fold(f64::INFINITY, f64::min);
        let hi = points.iter().map(axis).fold(f64::NEG_INFINITY, f64::max);
        (lo.floor(), hi.ceil())
    };
    let (mut x0, mut x1) = span(corners, |c| c.0);
    let (mut y0, mut y1) = span(corners, |c| c.1);
    if let Some(bbox) = bbox {
        let (bx0, bx1) = span(bbox, |c| c.0);
        let (by0, by1) = span(bbox, |c| c.1);
        (x0, x1, y0, y1) = (x0.max(bx0), x1.min(bx1), y0.max(by0), y1.min(by1));
    }
    // The low edge is clamped to [-1, extent + 1] and the high edge to
    // [-2, extent]: an empty range stays empty.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "whole numbers clamped to within two of the canvas"
    )]
    let edge = |lo: f64, hi: f64, extent: f64| {
        let extent = extent.ceil();
        (
            lo.clamp(-1.0, extent + 1.0) as i64,
            hi.clamp(-2.0, extent) as i64,
        )
    };
    let (x0, x1) = edge(x0, x1, canvas.width);
    let (y0, y1) = edge(y0, y1, canvas.height);
    [x0, y0, x1, y1]
}

/// The colours of a Type 1 shading at the points of `lattice`, row by row:
/// `function` at the shading-space point of each, its fractions of the
/// Domain `[x0, x1, y0, y1]` given by `domain_at`, converted through `cs`.
/// At step 1 a point outside the Domain is never read and is left black;
/// at a larger step it takes the colour of the nearest Domain point.
///
/// `None`, logged, when the function evaluates at none of the points it was
/// evaluated at; points where it fails take colour components 0, logged
/// once.  A lattice with no point in the Domain evaluates nothing and paints
/// nothing.
fn sample_lattice<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    lattice: Lattice,
    domain_at: &impl Fn(f64, f64) -> (f64, f64, bool),
    function: &Function,
    cs: C,
    n: usize,
    [xd0, xd1, yd0, yd1]: [f64; 4],
) -> Option<Box<[[S; N]]>> {
    let mut samples = Vec::with_capacity(lattice.cols * lattice.rows);
    let (mut evaluated, mut failures) = (0usize, 0usize);
    for row in 0..lattice.rows {
        for col in 0..lattice.cols {
            let (px, py) = lattice.pixel(col, row);
            let (tx, ty, inside) = domain_at(px, py);
            if lattice.step == 1 && !inside {
                samples.push([S::ZERO; N]);
                continue;
            }
            let sx = tx.mul_add(xd1 - xd0, xd0);
            let sy = ty.mul_add(yd1 - yd0, yd0);
            evaluated += 1;
            let channels = function.eval_n(&[sx, sy], n).unwrap_or_else(|| {
                failures += 1;
                vec![0.0; n]
            });
            samples.push(cs.convert(&channels));
        }
    }
    if evaluated > 0 && failures == evaluated {
        log::warn!("shading/type1: Function evaluates at no sampled point — skipping");
        return None;
    }
    if failures > 0 {
        log::warn!(
            "shading/type1: Function fails at {failures} of {evaluated} sampled points — \
             painting their colour components as 0"
        );
    }
    Some(samples.into_boxed_slice())
}

/// The Domain `[x0, x1, y0, y1]` of a Type 1 shading: `[0 1 0 1]` when
/// absent (PDF §8.7.4.5.2); a malformed one is logged and the default used.
fn type1_domain(sh: &Dictionary) -> [f64; 4] {
    const DEFAULT: [f64; 4] = [0.0, 1.0, 0.0, 1.0];
    if sh.get(b"Domain").is_none() {
        return DEFAULT;
    }
    read_f64_array(sh, b"Domain", 4)
        .and_then(|d| <[f64; 4]>::try_from(d.as_slice()).ok())
        .unwrap_or_else(|| {
            log::warn!("shading/type1: Domain is not four numbers — using [0 1 0 1]");
            DEFAULT
        })
}

/// A resolved Type 1 shading: its pattern and the heap bytes it holds.
type FunctionBased<S> = (Box<dyn Pattern<S> + Send + Sync>, usize);

/// A Type 1 shading ready to sample: its function, colour space and
/// output count, Domain `[x0, x1, y0, y1]`, and the matrix from shading
/// space to the device.
struct Type1<'a, C> {
    function: &'a Function,
    cs: C,
    n: usize,
    domain: [f64; 4],
    shading_ctm: [f64; 6],
}

fn resolve_function_based<const N: usize, S: Sample, C: ShadingColour<N, S>>(
    doc: &Document,
    sh: &Dictionary,
    cs: C,
    n: usize,
    ctm: &[f64; 6],
    canvas: Canvas,
    bbox: Option<&[(f64, f64); 4]>,
) -> Option<FunctionBased<S>> {
    let domain @ [xd0, xd1, yd0, yd1] = type1_domain(sh);
    if !domain.iter().all(|v| v.is_finite()) {
        log::warn!("shading/type1: non-finite Domain — skipping");
        return None;
    }
    if xd0 >= xd1 || yd0 >= yd1 {
        log::warn!("shading/type1: degenerate Domain [{xd0},{xd1}]×[{yd0},{yd1}] — skipping");
        return None;
    }
    let function = Function::load(doc, sh.get(b"Function")?);
    // The point budget is the smaller of a count and the conversion work
    // budget over the cost of one point.
    let point_cost = function.cost().saturating_add(cs.conversion_cost()).max(1);
    let affordable = usize::try_from(EXACT_WORK_BUDGET / point_cost).unwrap_or(usize::MAX);
    let shading = Type1 {
        function: &function,
        cs,
        n,
        domain,
        // PDF §8.7.4.5.2: `Matrix` maps shading space into the target
        // space the CTM maps to the device.
        shading_ctm: read_matrix(sh).map_or(*ctm, |m| ctm_multiply(&m, ctm)),
    };
    shading.pattern(canvas, bbox, TYPE1_LATTICE_MAX.min(affordable).max(4))
}

impl<C> Type1<'_, C> {
    /// The shading's pattern on `canvas`, within `bbox` when given, from at
    /// most `max_points` samples.
    ///
    /// The colour at shading-space point (x, y) is the conversion of
    /// Function(x, y).  It is sampled at the device pixels of a lattice
    /// over the part of the Domain's device bounding box on the canvas and
    /// within `bbox`: at every pixel, which the pattern then reads
    /// unchanged, unless that exceeds `max_points`, when every `step`-th
    /// pixel is sampled and the pixels between are interpolated.
    fn pattern<const N: usize, S: Sample>(
        &self,
        canvas: Canvas,
        bbox: Option<&[(f64, f64); 4]>,
        max_points: usize,
    ) -> Option<FunctionBased<S>>
    where
        C: ShadingColour<N, S>,
    {
        let [xd0, xd1, yd0, yd1] = self.domain;
        let page_h = canvas.height;
        let Some(inv) = device_to_space(&self.shading_ctm, page_h) else {
            log::warn!("shading/type1: degenerate Matrix × CTM — skipping");
            return None;
        };
        // The shading-space point of device pixel (px, py) as fractions of
        // the Domain, clamped to it, and whether it lies in the Domain (PDF
        // §8.7.4.5.2: points outside it are not painted).
        let domain_at = move |px: f64, py: f64| {
            let (sx, sy) = ctm_transform(&inv, px, py);
            let tx = (sx - xd0) / (xd1 - xd0);
            let ty = (sy - yd0) / (yd1 - yd0);
            let inside = |t: f64| (-TYPE1_EDGE_TOLERANCE..=1.0 + TYPE1_EDGE_TOLERANCE).contains(&t);
            (
                tx.clamp(0.0, 1.0),
                ty.clamp(0.0, 1.0),
                inside(tx) && inside(ty),
            )
        };
        let corner = |x, y| transform_point(&self.shading_ctm, x, y, page_h);
        let corners = [
            corner(xd0, yd0),
            corner(xd1, yd0),
            corner(xd0, yd1),
            corner(xd1, yd1),
        ];
        if corners
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            log::warn!("shading/type1: Domain maps to non-finite device coordinates — skipping");
            return None;
        }
        let Some(lattice) = Lattice::covering(type1_area(&corners, bbox, canvas), max_points)
        else {
            log::debug!(
                "shading/type1: the Domain covers no pixel of the canvas — nothing painted"
            );
            return Some((Box::new(FunctionPattern::<N, S>::new(|_, _| None)), 0));
        };
        if lattice.step > 1 {
            log::debug!(
                "shading/type1: sampled every {} pixels, {} points within a budget of {max_points}",
                lattice.step,
                lattice.cols * lattice.rows
            );
        }
        let samples = sample_lattice(
            lattice,
            &domain_at,
            self.function,
            self.cs,
            self.n,
            self.domain,
        )?;
        let held = std::mem::size_of_val(&*samples);
        let pattern = FunctionPattern::new(move |px, py| {
            if !domain_at(px, py).2 {
                return None;
            }
            let level = |v: S| f64::from(u32::try_from(v.signed()).unwrap_or(0));
            let mut colour = [0.0_f64; N];
            for_each_corner(&lattice.axes(px, py), |index, weight| {
                for (acc, &v) in colour.iter_mut().zip(&samples[index]) {
                    *acc = weight.mul_add(level(v), *acc);
                }
            });
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a convex combination of samples, rounded, lies in the sample range"
            )]
            Some(colour.map(|v| S::from_signed(v.round() as i64)))
        });
        Some((Box::new(pattern), held))
    }
}

// ── Colour conversion helpers ─────────────────────────────────────────────────

/// A shading's resolved colour space, bound to the document its
/// parameterised variants (ICC profile, tint transform) dereference into.
#[derive(Clone, Copy)]
pub(super) struct ShadingCs<'a> {
    doc: &'a Document,
    space: &'a ColorSpace,
}

impl ShadingCs<'_> {
    /// Number of colour components a function output or mesh vertex carries.
    pub(super) fn channels(self) -> usize {
        self.space.ncomponents()
    }

    /// Convert colour components (nominally in `[0, 1]`) to an sRGB triple.
    ///
    /// A slice shorter than [`Self::channels`] reads the missing components
    /// as `0.0`; surplus components are ignored.  Out-of-range values clamp
    /// and `NaN` maps to the low end of the range, as in
    /// [`ColorSpace::convert_to_rgb`].
    pub(super) fn to_rgb<S: Sample>(self, channels: &[f64]) -> [S; 3] {
        let n = self.channels();
        if channels.len() == n {
            return self.space.convert(self.doc, channels);
        }
        let mut fitted = channels[..channels.len().min(n)].to_vec();
        fitted.resize(n, 0.0);
        self.space.convert(self.doc, &fitted)
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

/// A shading space of ink — `Separation`, `DeviceN`, process CMYK —
/// converting to the ink of `plan`.
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
            return self.plan.ink(doc, space, comps);
        }
        let mut fitted = comps[..comps.len().min(n)].to_vec();
        fitted.resize(n, 0.0);
        self.plan.ink(doc, space, &fitted)
    }

    fn conversion_cost(self) -> u64 {
        match self.plan {
            ColorantPlan::Channels(_) => 1,
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

    use crate::test_helpers::empty_doc;

    /// A Type 1 shading whose function evaluates nowhere paints nothing
    /// rather than a black rectangle.
    #[test]
    fn type1_with_an_unusable_function_is_skipped() {
        let doc = empty_doc();
        let mut exponential = Dictionary::new();
        exponential.set("FunctionType", Object::Integer(2));
        exponential.set("N", Object::Integer(1));
        let mut sh = Dictionary::new();
        sh.set("ShadingType", Object::Integer(1));
        // A Type 2 function takes one input; Type 1 supplies two.
        sh.set("Function", Object::Dictionary(exponential));
        let cs = ShadingCs {
            doc: &doc,
            space: &ColorSpace::DeviceGray,
        };
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!(resolve_function_based::<3, u8, _>(&doc, &sh, cs, 1, &ctm, CANVAS, None).is_none());
    }

    /// A 100 × 100 pixel canvas.
    const CANVAS: Canvas = Canvas {
        width: 100.0,
        height: 100.0,
        depth: SampleDepth::Eight,
    };

    /// A Type 1 shading of Domain `[0 1 0 1]` under `matrix`, painting
    /// grey 0.5.
    fn reals(values: &[f32]) -> Object {
        Object::Array(values.iter().copied().map(Object::Real).collect())
    }

    /// A Type 4 function of `(x, y)` returning grey 0.5.
    fn constant_grey() -> Object {
        let program = b"{ pop pop 0.5 }".to_vec();
        let mut dict = Dictionary::new();
        dict.set("FunctionType", Object::Integer(4));
        dict.set("Domain", reals(&[0.0, 1.0, 0.0, 1.0]));
        dict.set("Range", reals(&[0.0, 1.0]));
        dict.set("Length", Object::Integer(program.len().try_into().unwrap()));
        Object::Stream(Stream::new(dict, program))
    }

    fn grey_type1(doc: &Document, matrix: [f32; 6]) -> Option<FunctionBased<u8>> {
        let constant = constant_grey();
        let mut sh = Dictionary::new();
        sh.set("ShadingType", Object::Integer(1));
        sh.set("Matrix", reals(&matrix));
        sh.set("Function", constant);
        let cs = ShadingCs {
            doc,
            space: &ColorSpace::DeviceGray,
        };
        let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        resolve_function_based(doc, &sh, cs, 1, &ctm, CANVAS, None)
    }

    /// The pixels of row `y` from `x0` to `x1` that `pattern` paints.
    fn painted(pattern: &dyn Pattern, y: i32, x0: i32, x1: i32) -> usize {
        let n = usize::try_from(x1 - x0 + 1).unwrap();
        let (mut out, mut coverage) = (vec![0; n * 3], vec![0; n]);
        pattern.fill_span_coverage(y, x0, x1, &mut out, &mut coverage);
        coverage.iter().filter(|&&c| c != 0).count()
    }

    /// Sampled every few pixels, a Domain rotated 45° paints the function's
    /// colour up to its edge: lattice points outside the Domain take the
    /// nearest Domain point's colour, never black.
    #[test]
    fn a_coarse_lattice_paints_the_function_up_to_a_rotated_edge() {
        let doc = empty_doc();
        let function = Function::load(&doc, &constant_grey());
        let shading = Type1 {
            function: &function,
            cs: ShadingCs {
                doc: &doc,
                space: &ColorSpace::DeviceGray,
            },
            n: 1,
            domain: [0.0, 1.0, 0.0, 1.0],
            shading_ctm: [42.43, 42.43, -42.43, 42.43, 50.0, 20.0],
        };
        let (pattern, held) = shading
            .pattern::<3, u8>(CANVAS, None, 100)
            .expect("resolves");
        assert!(held <= 100 * 3, "{held} bytes for 100 points");
        let mut painted = 0;
        for y in 0..100 {
            let (mut out, mut coverage) = ([0; 300], [0; 100]);
            pattern.fill_span_coverage(y, 0, 99, &mut out, &mut coverage);
            for (x, (&c, rgb)) in coverage.iter().zip(out.as_chunks::<3>().0).enumerate() {
                if c != 0 {
                    painted += 1;
                    assert_eq!(*rgb, [128; 3], "pixel ({x}, {y})");
                }
            }
        }
        assert!(painted > 3000, "{painted} pixels painted");
    }

    /// A Domain between pixel coordinates holds no lattice point: the
    /// shading resolves and paints nothing, rather than being dropped as
    /// if its function failed.
    #[test]
    fn a_sub_pixel_type1_domain_resolves_and_paints_nothing() {
        let doc = empty_doc();
        let (pattern, _) = grey_type1(&doc, [0.4, 0.0, 0.0, 0.4, 10.2, 50.2]).expect("resolves");
        assert_eq!(painted(&*pattern, 49, 0, 99), 0);
    }

    /// A Domain wholly off the canvas resolves, holds nothing and paints
    /// nothing.
    #[test]
    fn an_off_canvas_type1_domain_holds_nothing() {
        let doc = empty_doc();
        let (pattern, held) =
            grey_type1(&doc, [10.0, 0.0, 0.0, 10.0, -5_000.0, 0.0]).expect("resolves");
        assert_eq!(held, 0);
        assert_eq!(painted(&*pattern, 50, 0, 99), 0);
    }

    /// The samples of a Domain covering the canvas are bounded by the
    /// canvas, however far the Domain extends.
    #[test]
    fn a_huge_type1_domain_is_sampled_over_the_canvas_only() {
        let doc = empty_doc();
        let (pattern, held) = grey_type1(&doc, [1e9, 0.0, 0.0, 1e9, -5e8, -5e8]).expect("resolves");
        assert!(held <= 102 * 102 * 3, "{held} bytes held");
        assert_eq!(painted(&*pattern, 50, 0, 99), 100);
    }

    /// The largest difference, in 16-bit levels, between the ramp
    /// `sample_ramp` builds of the gray function `function` and the function
    /// evaluated and converted at each of 65 536 positions over its domain.
    fn ramp_error_16(function: &[u8]) -> u16 {
        let doc = empty_doc();
        let obj = pdf::parse_object(function, &mut 0).expect("the function parses");
        let cs = ShadingCs {
            doc: &doc,
            space: &ColorSpace::DeviceGray,
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
        ShadingCs { doc: &doc, space }.to_rgb(channels)
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
    fn a_lattice_covers_every_pixel_of_a_small_area() {
        let lattice = Lattice::covering([10, 5, 21, 9], TYPE1_LATTICE_MAX).unwrap();
        assert_eq!(
            lattice,
            Lattice {
                x0: 10,
                y0: 5,
                step: 1,
                cols: 12,
                rows: 5,
            }
        );
        assert_eq!(lattice.pixel(11, 4), (21.0, 9.0));
        let [x, y] = lattice.axes(13.0, 6.0);
        assert_eq!((x.lo, x.frac, y.lo, y.frac, y.stride), (3, 0.0, 1, 0.0, 12));
    }

    #[test]
    fn a_large_area_takes_the_least_step_within_the_budget() {
        let lattice = Lattice::covering([0, 0, 10_000, 6_000], TYPE1_LATTICE_MAX).unwrap();
        assert!(lattice.cols * lattice.rows <= TYPE1_LATTICE_MAX);
        let single = Lattice::covering([0, 0, 0, 0], 4).unwrap();
        assert_eq!((single.step, single.cols, single.rows), (1, 1, 1));
        let step = lattice.step;
        let dims = |s: usize| (10_000_usize.div_ceil(s) + 1) * (6_000_usize.div_ceil(s) + 1);
        assert!(
            dims(step - 1) > TYPE1_LATTICE_MAX,
            "step {step} is not the least"
        );
        let (far_x, far_y) = lattice.pixel(lattice.cols - 1, lattice.rows - 1);
        assert!(
            far_x >= 10_000.0 && far_y >= 6_000.0,
            "reaches the far edges"
        );
    }

    /// A thin area far longer than the budget still finds the least step.
    #[test]
    fn a_thin_area_takes_a_step_of_its_length_over_the_budget() {
        let lattice = Lattice::covering([0, 0, 99_999, 1], 100).unwrap();
        assert!(lattice.cols * lattice.rows <= 100);
        assert!(Lattice::covering([0, 0, 99_999, 1], 100).is_some());
        let dims = |s: usize| (99_999_usize.div_ceil(s) + 1) * 2;
        assert!(dims(lattice.step - 1) > 100);
    }

    #[test]
    fn an_empty_area_has_no_lattice() {
        assert_eq!(Lattice::covering([5, 0, 4, 3], 4), None);
        assert_eq!(Lattice::covering([0, 3, 4, 2], 4), None);
    }

    /// The sampled area is the Domain's box cut to the canvas, widened by
    /// one pixel, and to `BBox`; a box beside the canvas stays empty.
    #[test]
    fn the_type1_area_is_cut_to_the_canvas_and_bbox() {
        let domain = [
            (-50.5, -50.5),
            (500.5, -50.5),
            (-50.5, 500.5),
            (500.5, 500.5),
        ];
        assert_eq!(type1_area(&domain, None, CANVAS), [-1, -1, 100, 100]);
        let bbox = [(10.2, 20.0), (30.0, 20.0), (30.0, 40.7), (10.2, 40.7)];
        assert_eq!(type1_area(&domain, Some(&bbox), CANVAS), [10, 20, 30, 41]);
        let left = [(-900.0, 0.0), (-500.0, 0.0), (-900.0, 50.0), (-500.0, 50.0)];
        let [x0, _, x1, _] = type1_area(&left, None, CANVAS);
        assert!(x1 < x0, "a box left of the canvas is empty: {x0}..{x1}");
        let right = [(900.0, 0.0), (1500.0, 0.0), (900.0, 50.0), (1500.0, 50.0)];
        let [x0, _, x1, _] = type1_area(&right, None, CANVAS);
        assert!(x1 < x0, "a box right of the canvas is empty: {x0}..{x1}");
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
