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

use std::sync::Arc;

use pdf::{Dictionary, Document, Object, Stream};
use raster::pipe::{ChannelMask, Pattern};
use raster::shading::ColorRamp;
use raster::shading::axial::AxialPattern;
use raster::shading::function::FunctionPattern;
use raster::shading::gouraud::GouraudVertex;
use raster::shading::radial::RadialPattern;

use super::colorspace::{self, ColorSpace};
use super::dict_ext::DictExt;
use super::{PageResources, read_matrix, resolve_dict};
use crate::renderer::color::PROCESS_CHANNELS;
use crate::renderer::gstate::{ctm_multiply, ctm_transform, invert_affine};
use function::{Function, read_f64_array, read_fn_domain};
use mesh::{CornerSource, DirectColors, ParamColors, TriangleSink};
use patch::PatchKind;
use stream::{VertexFormat, read_decode_array};
pub(crate) use triangles::Mesh;

// ── Public types ──────────────────────────────────────────────────────────────

/// The colour source of a resolved shading.
///
/// - `Pattern`: smooth gradient — use with [`raster::shading::shaded_fill`].
///   Pixels outside the shading's extent are left unpainted by the pattern.
/// - `Mesh`: pre-decoded Gouraud triangles, indexed by device row — call
///   [`raster::shading::gouraud::gouraud_triangle_fill`] for each triangle
///   [`Mesh::meeting`] the fill's clip.
pub enum ShadingResult {
    /// Smooth gradient — use with [`raster::shading::shaded_fill`].
    Pattern(Box<dyn Pattern + Send + Sync>),
    /// Pre-decoded Gouraud triangles, indexed by device row.
    Mesh(Mesh),
}

/// A shading dictionary resolved for painting (PDF §8.7.4.3).
pub struct ResolvedShading {
    /// The shading's colours.
    pub result: ShadingResult,
    /// `BBox`, the temporary clipping boundary of the shading, as the
    /// device-space corners of its rectangle in the shading's target space.
    pub bbox: Option<[(f64, f64); 4]>,
    /// `Background` converted to sRGB.  It applies only when the shading is
    /// painted as a shading pattern, never through the `sh` operator.
    pub background: Option<[u8; 3]>,
    /// The channels the shading paints on an ink canvas: the process
    /// channels for a shading in a process CMYK space, every channel
    /// otherwise.  A shading's colours reach the ink canvas through their
    /// sRGB conversion, so a `Separation` or `DeviceN` shading paints
    /// process ink and writes every channel.
    pub paints: ChannelMask,
}

// ── Public entry point ────────────────────────────────────────────────────────

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
    page_h: f64,
) -> Option<ResolvedShading> {
    resolve_shading_object(resources, shadings.get(name)?, ctm, page_h)
}

/// Resolve the shading `sh_obj` (a dictionary, a stream, or a reference to
/// either) whose target space `ctm` maps to the device, with the default
/// colour spaces of `resources`.
///
/// Returns `None` if the type is unsupported or any required key is missing.
#[must_use]
pub(crate) fn resolve_shading_object(
    resources: &PageResources<'_>,
    sh_obj: &Object,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<ResolvedShading> {
    let doc = resources.doc();
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
    let paints = if cs_space.is_process_cmyk() {
        PROCESS_CHANNELS
    } else {
        ChannelMask::ALL
    };
    if cs_space.paints_nothing() {
        log::debug!("shading: every colorant is /None — the shading paints nothing");
        return Some(ResolvedShading {
            result: ShadingResult::Mesh(Mesh::new(Vec::new())),
            bbox: None,
            background: None,
            paints,
        });
    }
    let cs = ShadingCs {
        doc,
        space: &cs_space,
    };
    let n_channels = cs.channels();

    let result = match shading_type {
        1 => resolve_function_based(doc, &sh_dict, cs, n_channels, ctm, page_h)
            .map(ShadingResult::Pattern),
        2 => resolve_axial(doc, &sh_dict, cs, n_channels, ctm, page_h).map(ShadingResult::Pattern),
        3 => resolve_radial(doc, &sh_dict, cs, n_channels, ctm, page_h).map(ShadingResult::Pattern),
        4..=7 => {
            let Some(data) = stream_data else {
                log::warn!("shading: ShadingType {shading_type} must be a stream — skipping");
                return None;
            };
            let mesh_type = u8::try_from(shading_type).ok()?;
            // PDF §8.7.4.5.5: with a Function, each vertex carries one
            // parametric value t, mapped through the function; its Decode
            // range is the interval the function is evaluated over.
            let mesh = if let Some(fn_obj) = sh_dict.get(b"Function") {
                let decode = read_decode_array(&sh_dict, 1);
                let ramp = sample_ramp(doc, fn_obj, cs, n_channels, decode[4], decode[5])?;
                decode_mesh(
                    mesh_type,
                    &sh_dict,
                    &data,
                    &ParamColors { ramp },
                    &decode,
                    ctm,
                    page_h,
                )
            } else {
                let decode = read_decode_array(&sh_dict, n_channels);
                decode_mesh(
                    mesh_type,
                    &sh_dict,
                    &data,
                    &DirectColors { cs },
                    &decode,
                    ctm,
                    page_h,
                )
            };
            Some(ShadingResult::Mesh(Mesh::new(mesh)))
        }
        other => {
            log::warn!("shading: ShadingType {other} is not a shading type (1–7) — skipping");
            None
        }
    }?;

    Some(ResolvedShading {
        result,
        bbox: read_bbox(&sh_dict, ctm, page_h),
        background: read_background(&sh_dict, cs, n_channels),
        paints,
    })
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

/// Read `Background`, one colour in the shading's colour space, as sRGB.
/// A `Background` of the wrong length is ignored.
fn read_background(sh: &Dictionary, cs: ShadingCs<'_>, n: usize) -> Option<[u8; 3]> {
    let arr = sh.get(b"Background")?.as_array()?;
    if arr.len() != n {
        log::warn!(
            "shading: Background has {} components, the colour space {n} — ignoring Background",
            arr.len()
        );
        return None;
    }
    let comps = read_f64_array(sh, b"Background", n)?;
    Some(cs.to_rgb(&comps))
}

/// Decode the stream of mesh shading `shading_type` (4–7) with vertex
/// colours from `source` and the shading's `Decode` array `decode`.
fn decode_mesh<S: CornerSource>(
    shading_type: u8,
    sh: &Dictionary,
    data: &[u8],
    source: &S,
    decode: &[f64],
    ctm: &[f64; 6],
    page_h: f64,
) -> Vec<[GouraudVertex; 3]> {
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

fn resolve_axial(
    doc: &Document,
    sh: &Dictionary,
    cs: ShadingCs<'_>,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Box<dyn Pattern + Send + Sync>> {
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
/// function is evaluated.
const RAMP_SAMPLES: u16 = 256;

/// Evaluate the shading function at [`RAMP_SAMPLES`] uniformly spaced
/// parameter values from `t0` to `t1` and convert each colour to sRGB.
///
/// PDF §8.7.4.5.3: the colour at parameter `t` is the colour-space
/// conversion of `Function(t)`; between samples the ramp interpolates in
/// sRGB.  `None` when the function fails to evaluate at any sample.
fn sample_ramp(
    doc: &Document,
    fn_obj: &Object,
    cs: ShadingCs<'_>,
    n: usize,
    t0: f64,
    t1: f64,
) -> Option<ColorRamp> {
    let function = Function::load(doc, fn_obj);
    let last = f64::from(RAMP_SAMPLES - 1);
    let samples = (0..RAMP_SAMPLES)
        .map(|k| {
            let t = (f64::from(k) / last).mul_add(t1 - t0, t0);
            function.eval(t, n).map(|c| cs.to_rgb(&c)).ok_or(t)
        })
        .collect::<Result<Vec<_>, f64>>()
        .map_err(|t| {
            log::warn!("shading: Function yields no {n}-component colour at t = {t} — skipping");
        })
        .ok()?;
    ColorRamp::from_samples(samples)
}

// ── Radial (Type 3) ───────────────────────────────────────────────────────────

fn resolve_radial(
    doc: &Document,
    sh: &Dictionary,
    cs: ShadingCs<'_>,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Box<dyn Pattern + Send + Sync>> {
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
const TYPE1_LATTICE_MAX: u64 = 1 << 22;

/// Tolerance, in units of the Domain's extent, admitting a point on a
/// Type 1 shading's Domain edge despite rounding error.
const TYPE1_EDGE_TOLERANCE: f64 = 1e-9;

/// The device pixels a Type 1 shading is sampled at: every `step`-th
/// pixel from `(x0, y0)`, `cols × rows` of them, covering the device
/// bounding box of the Domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lattice {
    x0: i64,
    y0: i64,
    step: i64,
    cols: usize,
    rows: usize,
}

impl Lattice {
    /// The lattice over the integer device pixels of the box spanned by
    /// `corners`: step 1 while the box holds at most [`TYPE1_LATTICE_MAX`]
    /// pixels, else the least step keeping the lattice within it, the last
    /// row and column reaching the box's far edges.  `None` when a corner
    /// is not finite.
    fn covering(corners: &[(f64, f64); 4]) -> Option<Self> {
        if corners
            .iter()
            .any(|&(x, y)| !x.is_finite() || !y.is_finite())
        {
            return None;
        }
        let bound = |pick: fn(&(f64, f64)) -> f64, lo: bool| {
            let v = corners.iter().map(pick);
            let v = if lo {
                v.fold(f64::INFINITY, f64::min).floor()
            } else {
                v.fold(f64::NEG_INFINITY, f64::max).ceil()
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "clamped to ±2^31 first; the cast is exact"
            )]
            let v = v.clamp(-2_147_483_648.0, 2_147_483_647.0) as i64;
            v
        };
        let (x0, x1) = (bound(|c| c.0, true), bound(|c| c.0, false));
        let (y0, y1) = (bound(|c| c.1, true), bound(|c| c.1, false));
        #[expect(clippy::cast_sign_loss, reason = "x1 ≥ x0 and y1 ≥ y0")]
        let (w, h) = ((x1 - x0 + 1) as u64, (y1 - y0 + 1) as u64);
        let dims = |step: u64| ((w - 1).div_ceil(step) + 1, (h - 1).div_ceil(step) + 1);
        let fits = |step: u64| {
            let (c, r) = dims(step);
            c.saturating_mul(r) <= TYPE1_LATTICE_MAX
        };
        // The box's area over the budget, square-rooted, is within a few
        // steps of the least step that fits.
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "an estimate of a step below 2^32; corrected by the loop"
        )]
        let mut step = ((w as f64 * h as f64 / TYPE1_LATTICE_MAX as f64).sqrt() as u64).max(1);
        while step > 1 && fits(step - 1) {
            step -= 1;
        }
        while !fits(step) {
            step += 1;
        }
        let (cols, rows) = dims(step);
        Some(Self {
            x0,
            y0,
            step: i64::try_from(step).ok()?,
            cols: usize::try_from(cols).ok()?,
            rows: usize::try_from(rows).ok()?,
        })
    }

    /// Device pixel of lattice point `(col, row)`.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        reason = "col < cols, row < rows ≤ 2^22 and the pixel within ±2^33: exact"
    )]
    fn pixel(self, col: usize, row: usize) -> (f64, f64) {
        let at = |o: i64, i: usize| (o + self.step * i as i64) as f64;
        (at(self.x0, col), at(self.y0, row))
    }

    /// Lattice coordinates of device pixel `(px, py)`, clamped to the
    /// lattice: whole numbers at the lattice's own pixels.
    #[expect(
        clippy::cast_precision_loss,
        reason = "x0, y0, step within ±2^31; cols, rows ≤ 2^22; exact in f64"
    )]
    fn coords(self, px: f64, py: f64) -> (f64, f64) {
        let step = self.step as f64;
        (
            ((px - self.x0 as f64) / step).clamp(0.0, (self.cols - 1) as f64),
            ((py - self.y0 as f64) / step).clamp(0.0, (self.rows - 1) as f64),
        )
    }
}

/// The colours of a Type 1 shading at the points of `lattice`, row by row:
/// `Function` at the shading-space point of each, its fractions of the
/// Domain `[x0, x1, y0, y1]` given by `domain_at`, converted through `cs`.
/// At step 1 a point outside the Domain is never read and is left black;
/// at a larger step it takes the colour of the nearest Domain point.
///
/// `None`, logged, when the function evaluates at no point; points where
/// it fails take colour components 0, logged once.
fn sample_lattice(
    lattice: Lattice,
    domain_at: &impl Fn(f64, f64) -> (f64, f64, bool),
    function: &Function,
    cs: ShadingCs<'_>,
    n: usize,
    [xd0, xd1, yd0, yd1]: [f64; 4],
) -> Option<Arc<[[u8; 3]]>> {
    let mut samples = Vec::with_capacity(lattice.cols * lattice.rows);
    let (mut evaluated, mut failures) = (0usize, 0usize);
    for row in 0..lattice.rows {
        for col in 0..lattice.cols {
            let (px, py) = lattice.pixel(col, row);
            let (tx, ty, inside) = domain_at(px, py);
            if lattice.step == 1 && !inside {
                samples.push([0; 3]);
                continue;
            }
            let sx = tx.mul_add(xd1 - xd0, xd0);
            let sy = ty.mul_add(yd1 - yd0, yd0);
            evaluated += 1;
            let channels = function.eval_n(&[sx, sy], n).unwrap_or_else(|| {
                failures += 1;
                vec![0.0; n]
            });
            samples.push(cs.to_rgb(&channels));
        }
    }
    if failures == evaluated {
        log::warn!("shading/type1: Function evaluates at no sampled point — skipping");
        return None;
    }
    if failures > 0 {
        log::warn!(
            "shading/type1: Function fails at {failures} of {evaluated} sampled points — \
             painting their colour components as 0"
        );
    }
    Some(Arc::from(samples.into_boxed_slice()))
}

fn resolve_function_based(
    doc: &Document,
    sh: &Dictionary,
    cs: ShadingCs<'_>,
    n: usize,
    ctm: &[f64; 6],
    page_h: f64,
) -> Option<Box<dyn Pattern + Send + Sync>> {
    // Domain: [xmin, xmax, ymin, ymax] in shading space. PDF spec default is [0,1,0,1].
    let domain = read_f64_array(sh, b"Domain", 4).unwrap_or_else(|| vec![0.0, 1.0, 0.0, 1.0]);
    let (xd0, xd1, yd0, yd1) = (domain[0], domain[1], domain[2], domain[3]);

    if !xd0.is_finite() || !xd1.is_finite() || !yd0.is_finite() || !yd1.is_finite() {
        log::warn!("shading/type1: non-finite Domain — skipping");
        return None;
    }
    if xd0 >= xd1 || yd0 >= yd1 {
        log::warn!("shading/type1: degenerate Domain [{xd0},{xd1}]×[{yd0},{yd1}] — skipping");
        return None;
    }

    let function = Function::load(doc, sh.get(b"Function")?);

    // PDF §8.7.4.5.2: `Matrix` maps shading space into the target space the
    // CTM maps to the device.
    let shading_ctm = read_matrix(sh).map_or(*ctm, |m| ctm_multiply(&m, ctm));

    // Device pixels map back to shading space through the inverse.
    let Some(inv) = device_to_space(&shading_ctm, page_h) else {
        log::warn!("shading/type1: degenerate Matrix × CTM — skipping");
        return None;
    };

    // The shading-space point of device pixel (px, py) as fractions of the
    // Domain, and whether it lies in the Domain (PDF §8.7.4.5.2: points
    // outside it are not painted).
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

    // The colour at shading-space point (x, y) is the conversion of
    // Function(x, y).  It is evaluated at the device pixels of a lattice
    // over the Domain's device bounding box: at every pixel (exact, the
    // closure below then reads the sample of its own pixel) unless the box
    // exceeds the lattice budget, when every `step`-th pixel is sampled and
    // the pixels between are interpolated.  A lattice point outside the
    // Domain samples the nearest Domain point, so interpolation next to the
    // Domain's edge has a value on both sides; at step 1 none is read and
    // none is evaluated.  All document references are resolved here; the
    // closure captures only the finished samples.
    let corner = |x, y| transform_point(&shading_ctm, x, y, page_h);
    let Some(lattice) = Lattice::covering(&[
        corner(xd0, yd0),
        corner(xd1, yd0),
        corner(xd0, yd1),
        corner(xd1, yd1),
    ]) else {
        log::warn!("shading/type1: Domain maps to non-finite device coordinates — skipping");
        return None;
    };
    let samples = sample_lattice(lattice, &domain_at, &function, cs, n, [xd0, xd1, yd0, yd1])?;
    let (cols, rows) = (lattice.cols, lattice.rows);

    let pattern = FunctionPattern::new(move |px, py| {
        if !domain_at(px, py).2 {
            return None;
        }
        // Bilinear interpolation between the lattice points around the
        // pixel; at a lattice point both weights are 0 and the sample is
        // returned unchanged.
        let (fx, fy) = lattice.coords(px, py);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "fx ∈ [0, cols − 1], fy ∈ [0, rows − 1]; floor fits in usize"
        )]
        let (c0, r0) = (fx.floor() as usize, fy.floor() as usize);
        let c1 = (c0 + 1).min(cols - 1);
        let r1 = (r0 + 1).min(rows - 1);
        #[expect(clippy::cast_precision_loss, reason = "c0, r0 < 2^22; exact in f64")]
        let (wc, wr) = (fx - c0 as f64, fy - r0 as f64);

        let s00 = samples[r0 * cols + c0];
        let s10 = samples[r0 * cols + c1];
        let s01 = samples[r1 * cols + c0];
        let s11 = samples[r1 * cols + c1];

        // Lerp a single u8 channel; t ∈ [0,1), result ∈ [0,255].
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a,b ∈ [0,255] and t ∈ [0,1); result rounds into [0,255]"
        )]
        let lerp_ch = |a: u8, b: u8, t: f64| -> u8 {
            t.mul_add(f64::from(b) - f64::from(a), f64::from(a)).round() as u8
        };

        let top = [
            lerp_ch(s00[0], s10[0], wc),
            lerp_ch(s00[1], s10[1], wc),
            lerp_ch(s00[2], s10[2], wc),
        ];
        let bot = [
            lerp_ch(s01[0], s11[0], wc),
            lerp_ch(s01[1], s11[1], wc),
            lerp_ch(s01[2], s11[2], wc),
        ];
        Some([
            lerp_ch(top[0], bot[0], wr),
            lerp_ch(top[1], bot[1], wr),
            lerp_ch(top[2], bot[2], wr),
        ])
    });

    Some(Box::new(pattern))
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
    pub(super) fn to_rgb(self, channels: &[f64]) -> [u8; 3] {
        let n = self.channels();
        if channels.len() == n {
            return self.space.convert_to_rgb(self.doc, channels);
        }
        let mut fitted = channels[..channels.len().min(n)].to_vec();
        fitted.resize(n, 0.0);
        self.space.convert_to_rgb(self.doc, &fitted)
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
        assert!(resolve_function_based(&doc, &sh, cs, 1, &ctm, 10.0).is_none());
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
    fn a_lattice_covers_every_pixel_of_a_small_box() {
        let lattice =
            Lattice::covering(&[(10.2, 5.7), (20.9, 5.7), (10.2, 8.1), (20.9, 8.1)]).unwrap();
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
        assert_eq!(lattice.coords(13.0, 6.0), (3.0, 1.0));
    }

    #[test]
    fn a_large_box_takes_the_least_step_within_the_budget() {
        let lattice = Lattice::covering(&[
            (0.0, 0.0),
            (10_000.0, 0.0),
            (0.0, 6_000.0),
            (10_000.0, 6_000.0),
        ])
        .unwrap();
        let cells = |l: &Lattice| l.cols as u64 * l.rows as u64;
        assert!(cells(&lattice) <= TYPE1_LATTICE_MAX);
        let finer = Lattice::covering(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]).unwrap();
        assert_eq!(finer.step, 1);
        let step = u64::try_from(lattice.step).unwrap();
        let dims = |s: u64| (10_000_u64.div_ceil(s) + 1) * (6_000_u64.div_ceil(s) + 1);
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

    #[test]
    fn a_lattice_needs_finite_corners() {
        assert_eq!(
            Lattice::covering(&[(0.0, 0.0), (f64::INFINITY, 0.0), (0.0, 1.0), (1.0, 1.0)]),
            None
        );
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
