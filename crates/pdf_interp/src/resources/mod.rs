//! PDF resource resolution — fonts, colour spaces, `XObject`s, shadings.
//!
//! The [`PageResources`] struct holds a reference to the [`pdf::Document`]
//! and the current page's [`ObjectId`], and provides lazy-loading helpers
//! for each resource category.
//!
//! # Design
//!
//! `pdf::Document` is immutable after load; resource dicts are accessed
//! via `get_page_fonts()` and friends, which traverse the page tree and
//! dereference indirect objects.  All caching lives in the caller
//! ([`super::renderer::font_cache`]) rather than here, so this module
//! stays stateless and easy to test.

pub mod cmap;
pub mod cmap_predefined;
pub mod colorspace;
pub(crate) mod dict_ext;
pub mod font;
pub mod font_substitute;
pub(crate) mod icc_shaper;
pub mod image;
pub mod shading;
pub mod tiling;
pub mod transfer;

use std::sync::{Arc, OnceLock};

use dict_ext::DictExt;
use pdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

pub use colorspace::ColorSpace;
pub use font::{BaseEncoding, FontDescriptor, PdfFontKind, resolve_font};
/// Re-exported from [`gpu::nvjpeg`] for callers that enable the `nvjpeg` feature.
/// Create via [`gpu::nvjpeg::NvJpegDecoder::new`] with a raw `CUstream` handle,
/// then pass to the page renderer's `set_nvjpeg` configuration entry point.
#[cfg(feature = "nvjpeg")]
pub use gpu::nvjpeg::NvJpegDecoder;
pub use image::{
    DecodedImage, IMAGE_FILTER_COUNT, ImageColorSpace, ImageDescriptor, ImageFilter, ImageRequest,
    ImageResolution, SoftMask, resolve_image,
};

/// Selected parameters extracted from a PDF `ExtGState` resource dictionary.
///
/// Only the subset needed by the current rasteriser is extracted; unknown keys
/// are silently ignored so that future additions do not break existing pages.
#[derive(Debug, Clone)]
pub struct ExtGStateParams {
    /// Non-stroking (fill) opacity `ca` in [0, 255].  `None` = unchanged.
    pub fill_alpha: Option<u8>,
    /// Stroking opacity `CA` in [0, 255].  `None` = unchanged.
    pub stroke_alpha: Option<u8>,
    /// Line width `LW` in user-space units.  `None` = unchanged.
    pub line_width: Option<f64>,
    /// Line cap style `LC` (0–2).  `None` = unchanged.
    pub line_cap: Option<i32>,
    /// Line join style `LJ` (0–2).  `None` = unchanged.
    pub line_join: Option<i32>,
    /// Miter limit `ML`.  `None` = unchanged.
    pub miter_limit: Option<f64>,
    /// Flatness `FL`.  `None` = unchanged.
    pub flatness: Option<f64>,
    /// Blend mode `BM`.  `None` = unchanged (keep current).
    pub blend_mode: Option<BlendMode>,
    /// Soft mask `SMask`.  `None` = unchanged.
    pub soft_mask: Option<SoftMaskParams>,
    /// Transfer function `TR2`, else `TR`.  `None` = unchanged.
    pub transfer: Option<transfer::TransferSetting>,
}

/// A shading pattern (`PatternType` 2, PDF §8.7.3.3) resolved for painting.
pub struct ShadingPattern {
    /// The pattern's shading, placed by `matrix`.
    pub shading: shading::ResolvedShading,
    /// The pattern's `ExtGState`: graphics state parameters in effect while
    /// the shading is painted.
    pub ext_gstate: Option<ExtGStateParams>,
    /// The pattern matrix concatenated onto the default coordinate space of
    /// the pattern's parent content stream: pattern space to device space,
    /// before the device y-flip.
    pub matrix: [f64; 6],
}

/// The `SMask` entry of an `ExtGState` dictionary (PDF §11.6.5.2).
#[derive(Debug, Clone)]
pub enum SoftMaskParams {
    /// `/None`: no soft mask.
    None,
    /// A soft-mask dictionary.
    Group {
        /// `S`: `true` for `/Luminosity`, `false` for `/Alpha`.
        luminosity: bool,
        /// `G`: the transparency group `XObject` that defines the mask.
        group: ObjectId,
        /// `BC`: backdrop colour components in the group's colour space;
        /// empty means black.
        backdrop: Vec<f64>,
        /// `TR`: transfer function applied to the mask values; `None` is
        /// the identity.
        transfer: Option<Object>,
    },
}

impl ExtGStateParams {
    fn from_dict(doc: &Document, d: &Dictionary) -> Self {
        Self {
            fill_alpha: real_to_u8(d, b"ca"),
            stroke_alpha: real_to_u8(d, b"CA"),
            line_width: real_or_int(d, b"LW"),
            line_cap: int_val(d, b"LC"),
            line_join: int_val(d, b"LJ"),
            miter_limit: real_or_int(d, b"ML"),
            flatness: real_or_int(d, b"FL"),
            blend_mode: parse_blend_mode(d),
            soft_mask: parse_soft_mask(doc, d),
            transfer: transfer::ext_gstate_transfer(doc, d),
        }
    }
}

/// Parse the `SMask` key from an `ExtGState` dictionary.
///
/// A dictionary without a stream reference in `G`, or with an `S` other
/// than `Luminosity` / `Alpha`, is malformed and leaves the soft mask
/// unchanged.
fn parse_soft_mask(doc: &Document, d: &Dictionary) -> Option<SoftMaskParams> {
    let obj = d.get(b"SMask")?;
    if obj.as_name() == Some(b"None".as_slice()) {
        return Some(SoftMaskParams::None);
    }
    let sm = resolve_dict(doc, obj)?;
    let luminosity = match sm.get(b"S")?.as_name()? {
        b"Luminosity" => true,
        b"Alpha" => false,
        other => {
            log::warn!(
                "ExtGState: soft mask subtype {:?} unknown — ignoring the soft mask",
                String::from_utf8_lossy(other)
            );
            return None;
        }
    };
    let Object::Reference(group) = sm.get(b"G")? else {
        return None;
    };
    let backdrop = match sm.get(b"BC") {
        Some(Object::Array(a)) => a
            .iter()
            .filter_map(|o| match o {
                Object::Real(r) => Some(f64::from(*r)),
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "colour components are small integers"
                )]
                Object::Integer(n) => Some(*n as f64),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let transfer = sm
        .get(b"TR")
        .filter(|o| o.as_name() != Some(b"Identity".as_slice()))
        .cloned();
    Some(SoftMaskParams::Group {
        luminosity,
        group: *group,
        backdrop,
        transfer,
    })
}

/// Parse the `BM` key from an `ExtGState` dictionary.
///
/// `BM` is either a single `Name` or an array of names (priority list; the
/// viewer uses the first one it supports).  Unknown names are silently ignored
/// and `None` is returned, leaving the current blend mode unchanged.
fn parse_blend_mode(d: &Dictionary) -> Option<BlendMode> {
    let obj = d.get(b"BM")?;
    match obj {
        Object::Name(n) => bm_name_to_mode(n),
        // Array is a viewer-preference priority list; use the first recognised name.
        Object::Array(arr) => arr
            .iter()
            .find_map(|o| o.as_name().and_then(bm_name_to_mode)),
        _ => None,
    }
}

/// Map a PDF blend mode name to [`BlendMode`].
///
/// Returns `None` for unrecognised names so the caller can keep the current mode.
const fn bm_name_to_mode(name: &[u8]) -> Option<BlendMode> {
    Some(match name {
        b"Normal" | b"Compatible" => BlendMode::Normal,
        b"Multiply" => BlendMode::Multiply,
        b"Screen" => BlendMode::Screen,
        b"Overlay" => BlendMode::Overlay,
        b"Darken" => BlendMode::Darken,
        b"Lighten" => BlendMode::Lighten,
        b"ColorDodge" => BlendMode::ColorDodge,
        b"ColorBurn" => BlendMode::ColorBurn,
        b"HardLight" => BlendMode::HardLight,
        b"SoftLight" => BlendMode::SoftLight,
        b"Difference" => BlendMode::Difference,
        b"Exclusion" => BlendMode::Exclusion,
        b"Hue" => BlendMode::Hue,
        b"Saturation" => BlendMode::Saturation,
        b"Color" => BlendMode::Color,
        b"Luminosity" => BlendMode::Luminosity,
        _ => return None,
    })
}

/// Read a real or integer key from a dictionary as `f64`.
fn real_or_int(d: &Dictionary, key: &[u8]) -> Option<f64> {
    match d.get(key)? {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "ExtGState numeric params are small integers"
        )]
        Object::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

/// Read a real-valued opacity key and convert to u8 in [0, 255].
fn real_to_u8(d: &Dictionary, key: &[u8]) -> Option<u8> {
    let v = real_or_int(d, key)?.clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v clamped to [0,1], scaled to [0,255]; round() fits in u8"
    )]
    Some((v * 255.0).round() as u8)
}

/// Read an integer key.
fn int_val(d: &Dictionary, key: &[u8]) -> Option<i32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "line cap/join values are 0–2"
    )]
    d.get(key)?.as_i64().map(|n| n as i32)
}

/// Parsed Form `XObject` — content bytes and the CTM matrix to apply.
///
/// The `resources_id` is the `ObjectId` of the form stream itself; callers
/// construct a child [`PageResources`] from it via [`PageResources::for_form`].
pub struct FormXObject {
    /// Decoded content bytes of the form's content stream.
    pub content: Vec<u8>,
    /// Optional form Matrix `[a b c d e f]` (default: identity).
    pub matrix: [f64; 6],
    /// Form bounding box in form space `[llx, lly, urx, ury]`, which clips
    /// the form's content and bounds its transparency group; `None` when
    /// absent or malformed.
    pub bbox: Option<[f64; 4]>,
    /// The object ID of the form stream, used to build a child `PageResources`.
    pub resources_id: ObjectId,
    /// True if the form has its own `Resources` dict; false means it inherits
    /// from the parent context (the caller should keep the parent resources).
    pub has_own_resources: bool,
    /// Transparency group parameters, present when `Group /S /Transparency` is set.
    pub transparency: Option<TransparencyGroupParams>,
}

/// The attributes of a transparency group dictionary (PDF §11.6.6), of a
/// form `XObject` or of a page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransparencyGroupParams {
    /// `I`: the group is isolated, so blend modes inside it do not see the
    /// backdrop under it (default: false).
    pub isolated: bool,
    /// `K`: the group is a knockout group, in which each object composites
    /// with the group's initial backdrop rather than over the objects
    /// before it (default: false).
    pub knockout: bool,
}

/// The identity of a resource object (a font, pattern or shading
/// dictionary) within one document.
///
/// Two resource names that refer to the same indirect object share a key,
/// and one name bound to different direct objects in different resource
/// dictionaries has distinct keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKey {
    /// An indirect object.
    Object(ObjectId),
    /// A direct object, held under `name` in one category of the resources
    /// of the object `context`.
    Direct {
        /// The page or form whose resources hold the dictionary.
        context: ObjectId,
        /// The resource name.
        name: Box<[u8]>,
    },
}

/// Thin accessor wrapping a `(Document, resource_context_id)` pair.
///
/// `resource_context_id` is the `ObjectId` of the object (page or form stream)
/// whose `Resources` dict is used to resolve fonts and `XObjects`.  For a top-level
/// page render this is the page object; for a form `XObject` it is the form stream.
pub struct PageResources<'doc> {
    doc: &'doc Document,
    /// Object whose `Resources` key is consulted for font/XObject lookups.
    resource_context_id: ObjectId,
    /// Each category sub-dictionary, resolved on first use; `None` when the
    /// category is absent or empty.
    subdicts: [OnceLock<Option<Arc<Dictionary>>>; Category::COUNT],
    /// The default colour spaces for `DeviceGray`, `DeviceRGB` and
    /// `DeviceCMYK`, in that order, resolved on first use.
    default_spaces: OnceLock<[Option<ColorSpace>; 3]>,
}

/// A sub-dictionary key of a resource dictionary (ISO 32000-1 §7.8.3).
#[derive(Clone, Copy)]
enum Category {
    Font,
    XObject,
    ExtGState,
    ColorSpace,
    Pattern,
    Shading,
    Properties,
}

impl Category {
    const COUNT: usize = 7;

    /// Every category, in discriminant order.
    const ALL: [Self; Self::COUNT] = [
        Self::Font,
        Self::XObject,
        Self::ExtGState,
        Self::ColorSpace,
        Self::Pattern,
        Self::Shading,
        Self::Properties,
    ];

    const fn key(self) -> &'static [u8] {
        match self {
            Self::Font => b"Font",
            Self::XObject => b"XObject",
            Self::ExtGState => b"ExtGState",
            Self::ColorSpace => b"ColorSpace",
            Self::Pattern => b"Pattern",
            Self::Shading => b"Shading",
            Self::Properties => b"Properties",
        }
    }
}

impl<'doc> PageResources<'doc> {
    /// Construct a [`PageResources`] for the given page.
    #[must_use]
    pub const fn new(doc: &'doc Document, page_id: ObjectId) -> Self {
        Self {
            doc,
            resource_context_id: page_id,
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
        }
    }

    /// Construct a child [`PageResources`] scoped to the given form's resources.
    ///
    /// A form without its own `Resources` inherits the resources in effect,
    /// including any already resolved.
    #[must_use]
    pub fn for_form(&self, form: &FormXObject) -> Self {
        if !form.has_own_resources {
            return Self {
                doc: self.doc,
                resource_context_id: self.resource_context_id,
                subdicts: self.subdicts.clone(),
                default_spaces: self.default_spaces.clone(),
            };
        }
        Self {
            doc: self.doc,
            resource_context_id: form.resources_id,
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
        }
    }

    /// Construct a [`PageResources`] over the resource dictionary
    /// `resources` itself, such as an interactive form's default resources
    /// (`/AcroForm /DR`), rather than over an object's `/Resources` entry.
    /// `context` identifies the dictionary to the font cache, which keys
    /// direct font dictionaries by it and their resource name.
    #[must_use]
    pub fn with_resources(&self, context: ObjectId, resources: &Dictionary) -> Self {
        let subdicts = Category::ALL.map(|category| {
            OnceLock::from(
                resources
                    .get(category.key())
                    .and_then(|o| resolve_dict(self.doc, o))
                    .filter(|d| !d.is_empty())
                    .map(Arc::new),
            )
        });
        Self {
            doc: self.doc,
            resource_context_id: context,
            subdicts,
            default_spaces: OnceLock::new(),
        }
    }

    /// Construct a child [`PageResources`] scoped to the `Resources` of the
    /// object `context` (a Type 3 font dictionary carrying its own).
    #[must_use]
    pub const fn for_context(&self, context: ObjectId) -> Self {
        Self {
            doc: self.doc,
            resource_context_id: context,
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
        }
    }

    /// The underlying [`pdf::Document`] (read-only).
    #[must_use]
    pub const fn doc(&self) -> &'doc Document {
        self.doc
    }

    /// The object ID of the current resource context (page or form stream).
    ///
    /// Used by tiling-pattern rendering to inherit the parent context when the
    /// pattern stream carries no own `Resources` dict.
    #[must_use]
    pub const fn resource_context_id(&self) -> ObjectId {
        self.resource_context_id
    }

    /// The `category` sub-dictionary (`XObject`, `ExtGState`, `Pattern`, …) of
    /// the resources in effect.
    ///
    /// `/Resources` is inheritable (ISO 32000-1 §7.7.3.4): a page without its
    /// own entry takes the nearest ancestor's in the page tree.  Every resource
    /// category resolves through this one lookup so that inheritance applies
    /// uniformly.  `None` when no context in the chain defines `category`.
    /// The lookup runs once per category; later calls borrow the result.
    fn resource_subdict(&self, category: Category) -> Option<&Dictionary> {
        self.subdicts[category as usize]
            .get_or_init(|| {
                self.doc
                    .get_page_resource_dict(self.resource_context_id, category.key())
                    .ok()
                    .filter(|d| !d.is_empty())
                    .map(Arc::new)
            })
            .as_deref()
    }

    /// The identity of the font dictionary the named resource (e.g. `b"F1"`)
    /// refers to, with the resource entry itself.
    ///
    /// Returns `None` if the resource name is not present in the resource dict.
    #[must_use]
    pub fn font_entry(&self, name: &[u8]) -> Option<(ResourceKey, Object)> {
        let entry = self.resource_subdict(Category::Font)?.get(name)?.clone();
        Some((self.key_of(&entry, name), entry))
    }

    /// Resolve a font resource entry returned by [`Self::font_entry`].
    ///
    /// Returns `None` if the entry is not a dictionary.
    #[must_use]
    pub fn resolve_font_entry(&self, entry: &Object) -> Option<font::FontDescriptor> {
        let dict = resolve_dict(self.doc, entry)?;
        Some(font::resolve_font(self.doc, &dict, entry.as_reference()))
    }

    /// Resolve the named entry of the page's `/ColorSpace` resource dictionary
    /// into a [`ColorSpace`] tracking value.
    ///
    /// Per PDF §8.6.3, the operand to `cs`/`CS` may be one of the four
    /// device-space names (`DeviceGray`, `DeviceRGB`, `DeviceCMYK`, `Pattern`)
    /// which resolve directly, or a key into the page's `/ColorSpace`
    /// resource dict that resolves to a parameterised colour-space array.
    ///
    /// A device-space name selects the space [`Self::device_space`] maps it
    /// to.
    ///
    /// Returns [`ColorSpace::DeviceGray`] when the name isn't found in the
    /// resource dict and isn't a device-space name — the same safe-fallback
    /// policy the rest of the resource layer uses on missing inputs.
    #[must_use]
    pub fn color_space(&self, name: &[u8]) -> ColorSpace {
        // Device-space names short-circuit the resource lookup per §8.6.3.
        if matches!(name, b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK") {
            return self.device_space(colorspace::resolve_device_name(name));
        }
        if name == b"Pattern" {
            return colorspace::resolve_device_name(name);
        }
        let cs_dict = self.resource_subdict(Category::ColorSpace);
        let Some(entry) = cs_dict.and_then(|d| d.get(name)) else {
            log::debug!(
                "colorspace: name {:?} not found in /ColorSpace resource dict — fallback DeviceGray",
                String::from_utf8_lossy(name)
            );
            return ColorSpace::DeviceGray;
        };
        colorspace::resolve(self.doc, entry)
    }

    /// The colour space that selecting the device space `device` yields:
    /// the usable `/DefaultGray`, `/DefaultRGB` or `/DefaultCMYK` entry of
    /// the `/ColorSpace` resource dictionary (PDF §8.6.5.6), otherwise
    /// `device` itself.  Spaces other than the three device spaces are
    /// returned unchanged.
    #[must_use]
    pub fn device_space(&self, device: ColorSpace) -> ColorSpace {
        let slot = match device {
            ColorSpace::DeviceGray => 0,
            ColorSpace::DeviceRgb => 1,
            ColorSpace::DeviceCmyk => 2,
            _ => return device,
        };
        let defaults = self.default_spaces.get_or_init(|| {
            let Some(dict) = self.color_space_dict() else {
                return [None, None, None];
            };
            [
                ColorSpace::DeviceGray,
                ColorSpace::DeviceRgb,
                ColorSpace::DeviceCmyk,
            ]
            .map(|d| colorspace::default_space(self.doc, dict, &d))
        });
        defaults[slot].clone().unwrap_or(device)
    }

    /// The `/ColorSpace` subdictionary of the resources in effect, if any.
    #[must_use]
    pub fn color_space_dict(&self) -> Option<&Dictionary> {
        self.resource_subdict(Category::ColorSpace)
    }

    /// The component count of the colour space `name` names in the
    /// `/ColorSpace` resources, for the data length of an inline image
    /// (see [`crate::content::parse_with`]); `None` for a name the
    /// resources do not define or a `Pattern` space.
    #[must_use]
    pub fn inline_image_components(&self, name: &[u8]) -> Option<usize> {
        let entry = self.color_space_dict()?.get(name)?;
        let n = colorspace::resolve(self.doc, entry).ncomponents();
        (n > 0).then_some(n)
    }

    /// Decode the image `XObject` `request` names from the resource dictionary,
    /// at no less than `request.draw_size` when the codec can reduce.
    ///
    /// Returns [`image::ImageResolution::Absent`] if the name is absent or not
    /// an image, and [`image::ImageResolution::DecodeFailed`] if decoding fails.
    ///
    /// `ctx` carries the GPU decoders, ICC handles and image cache the decode
    /// may use; [`image::ImageDecodeCtx::cpu`] decodes on the CPU only.
    #[must_use]
    pub fn image<#[cfg(feature = "gpu-jpeg-huffman")] B: gpu::backend::GpuBackend>(
        &self,
        request: image::ImageRequest<'_>,
        ctx: &mut image::ImageDecodeCtx<'_>,
        #[cfg(feature = "gpu-jpeg-huffman")] jpeg_gpu: Option<
            &mut gpu::jpeg_decoder::JpegGpuDecoder<B>,
        >,
    ) -> image::ImageResolution {
        let empty = Dictionary::new();
        let xobjects = self.resource_subdict(Category::XObject).unwrap_or(&empty);
        image::resolve_image(
            ctx,
            xobjects,
            request,
            #[cfg(feature = "gpu-jpeg-huffman")]
            jpeg_gpu,
        )
    }

    /// Look up a named `ExtGState` resource and return selected parameters.
    ///
    /// Returns `None` if the name is absent or the resource dict is unreadable.
    /// Unknown or unsupported keys in the dict are silently ignored.
    #[must_use]
    pub fn ext_gstate(&self, name: &[u8]) -> Option<ExtGStateParams> {
        let eg_dict = self.resource_subdict(Category::ExtGState)?;
        let gs_ref_or_dict = eg_dict.get(name)?;
        let gs = resolve_dict(self.doc, gs_ref_or_dict)?;
        Some(ExtGStateParams::from_dict(self.doc, &gs))
    }

    /// The object the `XObject` resource `name` refers to, when the entry is
    /// an indirect reference (as `XObject` streams must be).
    #[must_use]
    pub fn xobject_id(&self, name: &[u8]) -> Option<ObjectId> {
        match self.resource_subdict(Category::XObject)?.get(name)? {
            Object::Reference(id) => Some(*id),
            _ => None,
        }
    }

    /// Resolve the named `XObject` and return it as a [`FormXObject`] if its
    /// subtype is `Form`.  Returns `None` for Image `XObjects`, missing names,
    /// or unreadable streams.
    #[must_use]
    pub fn form_xobject(&self, name: &[u8]) -> Option<FormXObject> {
        let stream_id = self.xobject_id(name)?;
        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream()?;

        // Must be a Form subtype.
        if stream.dict.get(b"Subtype")?.as_name()? != b"Form" {
            return None;
        }

        let content = stream.decompressed_content().ok()?;

        // Optional Matrix — defaults to identity if absent or malformed.
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        let bbox = read_bbox(&stream.dict);

        let has_own_resources = stream.dict.get(b"Resources").is_some();

        let transparency = read_transparency_group(self.doc, &stream.dict);

        Some(FormXObject {
            content,
            matrix,
            bbox,
            resources_id: stream_id,
            has_own_resources,
            transparency,
        })
    }

    /// Build a [`FormXObject`] directly from a stream object ID.
    ///
    /// Used for annotation appearance streams, which are referenced by object ID
    /// rather than through the page `Resources/XObject` dict.
    #[must_use]
    pub fn form_from_stream_id(&self, stream_id: ObjectId) -> Option<FormXObject> {
        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream()?;
        let content = stream.decompressed_content().ok()?;
        let matrix = read_matrix(&stream.dict).unwrap_or([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let bbox = read_bbox(&stream.dict);
        let has_own_resources = stream.dict.get(b"Resources").is_some();
        let transparency = read_transparency_group(self.doc, &stream.dict);
        Some(FormXObject {
            content,
            matrix,
            bbox,
            resources_id: stream_id,
            has_own_resources,
            transparency,
        })
    }

    /// The identity of the object the named `Pattern` resource refers to;
    /// `None` when the name is absent.
    #[must_use]
    pub fn pattern_key(&self, name: &[u8]) -> Option<ResourceKey> {
        let entry = self.resource_subdict(Category::Pattern)?.get(name)?;
        Some(self.key_of(entry, name))
    }

    /// The identity of the object the named `Shading` resource refers to;
    /// `None` when the name is absent.
    #[must_use]
    pub fn shading_key(&self, name: &[u8]) -> Option<ResourceKey> {
        let entry = self.resource_subdict(Category::Shading)?.get(name)?;
        Some(self.key_of(entry, name))
    }

    /// The identity of `entry`, held under `name` in a category of the
    /// resources in effect.
    fn key_of(&self, entry: &Object, name: &[u8]) -> ResourceKey {
        match entry {
            Object::Reference(id) => ResourceKey::Object(*id),
            _ => ResourceKey::Direct {
                context: self.resource_context_id,
                name: name.into(),
            },
        }
    }

    /// Resolve the named `Pattern` resource as a tiling descriptor.
    ///
    /// Returns `None` if the name is absent, the pattern is not `PatternType 1`
    /// (tiling), or any required key is missing.  A debug message is logged for
    /// `PatternType` 2 (shading patterns referenced via `scn`).
    #[must_use]
    pub fn tiling_pattern(&self, name: &[u8]) -> Option<tiling::TilingDescriptor> {
        let patterns = self.resource_subdict(Category::Pattern)?;
        tiling::resolve_tiling(self.doc, patterns, name)
    }

    /// Resolve the named `Shading` resource.
    ///
    /// Returns `None` if the name is absent, the type is unsupported, or any
    /// required key is missing.
    #[must_use]
    pub fn shading(
        &self,
        name: &[u8],
        ctm: &[f64; 6],
        page_h: f64,
    ) -> Option<shading::ResolvedShading> {
        let shadings = self.resource_subdict(Category::Shading)?;
        shading::resolve_shading(self.doc, shadings, name, ctm, page_h)
    }

    /// Resolve the named `Pattern` resource as a shading pattern
    /// (`PatternType` 2, PDF §8.7.4.2).
    ///
    /// `pattern_ctm` maps the default coordinate space of the pattern's
    /// parent content stream to the device; the pattern's `Matrix` is
    /// concatenated onto it.  Returns `None` if the name is absent, the
    /// pattern is not a shading pattern, or its shading does not resolve.
    #[must_use]
    pub fn shading_pattern(
        &self,
        name: &[u8],
        pattern_ctm: &[f64; 6],
        page_h: f64,
    ) -> Option<ShadingPattern> {
        let patterns = self.resource_subdict(Category::Pattern)?;
        let pattern = resolve_stream_dict(self.doc, patterns.get(name)?)?;
        if pattern.get_i64(b"PatternType") != Some(2) {
            return None;
        }
        let matrix = read_matrix(&pattern).map_or(*pattern_ctm, |m| {
            crate::renderer::gstate::ctm_multiply(&m, pattern_ctm)
        });
        let Some(shading) =
            shading::resolve_shading_object(self.doc, pattern.get(b"Shading")?, &matrix, page_h)
        else {
            log::warn!(
                "rasterrocket-interp: shading pattern /{} has no usable Shading — skipping",
                String::from_utf8_lossy(name)
            );
            return None;
        };
        let ext_gstate = pattern
            .get(b"ExtGState")
            .and_then(|obj| resolve_dict(self.doc, obj))
            .map(|d| ExtGStateParams::from_dict(self.doc, &d));
        Some(ShadingPattern {
            shading,
            ext_gstate,
            matrix,
        })
    }

    /// The `Properties` resource `key` (PDF §14.6.2): the property list a
    /// `BDC` names, such as the optional content group or membership
    /// dictionary of `BDC /OC /key`.
    #[must_use]
    pub fn property(&self, key: &[u8]) -> Option<Object> {
        self.resource_subdict(Category::Properties)?
            .get(key)
            .cloned()
    }

    /// The `/OC` entry of the `XObject` resource `name` (PDF §8.11.3.3): the
    /// optional content group or membership dictionary governing it.
    #[must_use]
    pub fn xobject_optional_content(&self, name: &[u8]) -> Option<Object> {
        let stream = self.doc.get_object(self.xobject_id(name)?).ok()?;
        stream.as_stream()?.dict.get(b"OC").cloned()
    }
}

/// Convert a [`pdf::Object`] (Real or Integer) to `f64`.
///
/// Returns `None` for any non-numeric object type.
pub(crate) fn obj_to_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Real(r) => Some(f64::from(*r)),
        #[expect(
            clippy::cast_precision_loss,
            reason = "PDF numeric values fit within f64 mantissa in all real-world uses"
        )]
        Object::Integer(n) => Some(*n as f64),
        _ => None,
    }
}

/// Read a single numeric value from a dictionary key.
///
/// Returns `None` if the key is absent or the value is not a Real or Integer.
pub(crate) fn read_f64_1(dict: &Dictionary, key: &[u8]) -> Option<f64> {
    obj_to_f64(dict.get(key)?)
}

/// Parse a fixed-length array of `f64` values from a PDF dictionary.
///
/// Returns `None` if the key is absent, the value is not an array, the array
/// has fewer than `N` elements, or any of the first `N` elements is not numeric.
pub(crate) fn read_f64_n<const N: usize>(dict: &Dictionary, key: &[u8]) -> Option<[f64; N]> {
    let arr = dict.get(key)?.as_array()?;
    if arr.len() < N {
        return None;
    }
    let mut out = [0.0f64; N];
    for (i, obj) in arr.iter().take(N).enumerate() {
        out[i] = obj_to_f64(obj)?;
    }
    Some(out)
}

/// Read the `BBox` array `[llx, lly, urx, ury]` from a dictionary.
///
/// Returns `None` if absent or fewer than 4 numeric entries.
/// Normalises the result so that `llx ≤ urx` and `lly ≤ ury` — PDF allows
/// inverted `BBox` values.
pub(crate) fn read_bbox(dict: &Dictionary) -> Option<[f64; 4]> {
    let mut r = read_f64_n::<4>(dict, b"BBox")?;
    if r[0] > r[2] {
        r.swap(0, 2);
    }
    if r[1] > r[3] {
        r.swap(1, 3);
    }
    Some(r)
}

/// The transparency group attributes of `dict`, a form `XObject` stream
/// dictionary or a page dictionary: `Some` when its `Group` has
/// `/S /Transparency`, `None` otherwise.
#[must_use]
pub fn read_transparency_group(
    doc: &Document,
    dict: &Dictionary,
) -> Option<TransparencyGroupParams> {
    let grp_obj = dict.get(b"Group")?;
    let grp = resolve_dict(doc, grp_obj)?;
    if grp.get(b"S")?.as_name()? != b"Transparency" {
        return None;
    }
    let bool_flag = |key: &[u8]| grp.get(key).and_then(Object::as_bool).unwrap_or(false);
    Some(TransparencyGroupParams {
        isolated: bool_flag(b"I"),
        knockout: bool_flag(b"K"),
    })
}

/// Read a 6-element `Matrix` array from a dictionary, returning `None` if the
/// key is absent or has fewer than 6 numeric entries.
pub(crate) fn read_matrix(dict: &Dictionary) -> Option<[f64; 6]> {
    read_f64_n::<6>(dict, b"Matrix")
}

/// Dereference a PDF `Object` to an owned `Dictionary`.
///
/// Accepts `Dictionary` (cloned) or `Reference` (dereferenced via `doc` and
/// cloned).  Returns owned rather than borrowed because the underlying lazy
/// document parser materialises objects on demand and hands back `Arc`s, not
/// long-lived borrows.  Dicts are small (a few hundred entries at most), so
/// the clone cost is negligible.
///
/// Use [`resolve_stream_dict`] when the referent may be a stream object.
pub(crate) fn resolve_dict(doc: &Document, obj: &Object) -> Option<Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Reference(id) => doc.get_dictionary(*id).ok().map(|a| (*a).clone()),
        _ => None,
    }
}

/// Like [`resolve_dict`] but also handles `Reference → Stream → stream.dict`.
///
/// Needed for `ICCBased` colour spaces where the second array element is a
/// `Reference` to a stream whose dictionary carries the ICC metadata.
pub(crate) fn resolve_stream_dict(doc: &Document, obj: &Object) -> Option<Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d.clone()),
        Object::Reference(id) => {
            let referent = doc.get_object(*id).ok()?;
            match referent.as_ref() {
                Object::Stream(s) => Some(s.dict.clone()),
                // Reference to a plain dictionary: share the plain-dict path.
                other => resolve_dict(doc, other),
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm_name_all_variants_round_trip() {
        let pairs: &[(&[u8], BlendMode)] = &[
            (b"Normal", BlendMode::Normal),
            (b"Compatible", BlendMode::Normal),
            (b"Multiply", BlendMode::Multiply),
            (b"Screen", BlendMode::Screen),
            (b"Overlay", BlendMode::Overlay),
            (b"Darken", BlendMode::Darken),
            (b"Lighten", BlendMode::Lighten),
            (b"ColorDodge", BlendMode::ColorDodge),
            (b"ColorBurn", BlendMode::ColorBurn),
            (b"HardLight", BlendMode::HardLight),
            (b"SoftLight", BlendMode::SoftLight),
            (b"Difference", BlendMode::Difference),
            (b"Exclusion", BlendMode::Exclusion),
            (b"Hue", BlendMode::Hue),
            (b"Saturation", BlendMode::Saturation),
            (b"Color", BlendMode::Color),
            (b"Luminosity", BlendMode::Luminosity),
        ];
        for &(name, expected) in pairs {
            assert_eq!(
                bm_name_to_mode(name),
                Some(expected),
                "bm_name_to_mode({:?})",
                String::from_utf8_lossy(name)
            );
        }
    }

    #[test]
    fn bm_name_unknown_returns_none() {
        assert!(bm_name_to_mode(b"Dissolve").is_none());
        assert!(bm_name_to_mode(b"").is_none());
    }

    #[test]
    fn parse_blend_mode_bare_name() {
        let mut dict = Dictionary::new();
        dict.set("BM", Object::Name(b"Multiply".to_vec()));
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Multiply));
    }

    #[test]
    fn parse_blend_mode_array_picks_first_known() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Unknown1".to_vec()),
                Object::Name(b"Screen".to_vec()),
                Object::Name(b"Normal".to_vec()),
            ]),
        );
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Screen));
    }

    #[test]
    fn parse_blend_mode_absent_returns_none() {
        let dict = Dictionary::new();
        assert!(parse_blend_mode(&dict).is_none());
    }

    #[test]
    fn parse_blend_mode_all_unknown_array_returns_none() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Foo".to_vec()),
                Object::Name(b"Bar".to_vec()),
            ]),
        );
        assert!(parse_blend_mode(&dict).is_none());
    }

    // ── color_space() resolution ────────────────────────────────────────────

    #[test]
    fn color_space_device_names_short_circuit_resource_dict() {
        // Even with no /ColorSpace resource dict, the four device names must
        // resolve to their direct variants per §8.6.3.
        let doc = crate::test_helpers::empty_doc();
        let resources = PageResources::new(&doc, (1, 0));
        assert_eq!(resources.color_space(b"DeviceGray"), ColorSpace::DeviceGray);
        assert_eq!(resources.color_space(b"DeviceRGB"), ColorSpace::DeviceRgb);
        assert_eq!(resources.color_space(b"DeviceCMYK"), ColorSpace::DeviceCmyk,);
        assert_eq!(
            resources.color_space(b"Pattern"),
            ColorSpace::Pattern { base: None },
        );
    }

    #[test]
    fn color_space_unknown_name_falls_back_to_devicegray() {
        let doc = crate::test_helpers::empty_doc();
        let resources = PageResources::new(&doc, (1, 0));
        assert_eq!(resources.color_space(b"CS0"), ColorSpace::DeviceGray);
        assert_eq!(resources.color_space(b""), ColorSpace::DeviceGray);
    }
}
