//! The resource dictionaries in effect for a content stream (ISO 32000-1
//! §7.8.3) and the lookup of named resources through them.

use std::sync::{Arc, OnceLock};

use pdf::{Dictionary, Document, Object, ObjectId};

use super::dict_ext::DictExt;
use super::{
    ColorSpace, ExtGStateParams, ShadingPattern, colorspace, font, read_matrix, resolve_dict,
    resolve_stream_dict, shading, tiling,
};

mod xobject;

pub use xobject::FormXObject;

/// The identity of a resource object (a font, pattern or shading
/// dictionary) within one document.
///
/// Two resource names that refer to the same indirect object share a key,
/// and one name bound to different direct objects in different resource
/// dictionaries has distinct keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

/// The identity of the resource dictionaries in effect.
///
/// It keys memos of content that resolves names through them: the object
/// whose `/Resources` they are, or, for a set built over a dictionary
/// itself ([`PageResources::with_resources`]), an identity unique to that
/// set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceIdentity {
    /// The `/Resources` of this object (page, form, pattern or font).
    Object(ObjectId),
    /// A set built over a dictionary, numbered in creation order.
    Synthetic(u64),
}

impl ResourceIdentity {
    /// A synthetic identity no earlier set holds.
    fn synthetic() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self::Synthetic(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
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
    /// See [`ResourceIdentity`].
    identity: ResourceIdentity,
    /// Each category sub-dictionary, resolved on first use; `None` when the
    /// category is absent or empty.
    subdicts: [OnceLock<Option<Arc<Dictionary>>>; Category::COUNT],
    /// The default colour spaces for `DeviceGray`, `DeviceRGB` and
    /// `DeviceCMYK`, in that order, resolved on first use.
    default_spaces: OnceLock<[Option<ColorSpace>; 3]>,
    /// The object identifying the `/ColorSpace` sub-dictionary in effect
    /// (see [`Document::page_resource_entry`]), resolved on first use.
    color_space_source: OnceLock<Option<ObjectId>>,
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
            identity: ResourceIdentity::Object(page_id),
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
            color_space_source: OnceLock::new(),
        }
    }

    /// Construct a child [`PageResources`] scoped to the given form's resources.
    ///
    /// A form without its own `Resources` inherits the resources in effect,
    /// including any already resolved.
    #[must_use]
    pub fn for_form(&self, form: &FormXObject) -> Self {
        if !form.has_own_resources {
            return self.inherited();
        }
        Self {
            doc: self.doc,
            resource_context_id: form.resources_id,
            identity: ResourceIdentity::Object(form.resources_id),
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
            color_space_source: OnceLock::new(),
        }
    }

    /// The resources in effect as a nested content stream without
    /// resources of its own (a form or a tiling pattern) inherits them: the
    /// same dictionaries, whichever object or dictionary they came from,
    /// with what is already resolved.
    #[must_use]
    pub fn inherited(&self) -> Self {
        Self {
            doc: self.doc,
            resource_context_id: self.resource_context_id,
            identity: self.identity,
            subdicts: self.subdicts.clone(),
            default_spaces: self.default_spaces.clone(),
            color_space_source: self.color_space_source.clone(),
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
        let color_space_source = resources
            .get(Category::ColorSpace.key())
            .map(|entry| entry.as_reference().unwrap_or(context));
        Self {
            doc: self.doc,
            resource_context_id: context,
            identity: ResourceIdentity::synthetic(),
            subdicts,
            default_spaces: OnceLock::new(),
            color_space_source: OnceLock::from(color_space_source),
        }
    }

    /// Construct a child [`PageResources`] scoped to the `Resources` of the
    /// object `context` (a Type 3 font dictionary carrying its own).
    #[must_use]
    pub const fn for_context(&self, context: ObjectId) -> Self {
        Self {
            doc: self.doc,
            resource_context_id: context,
            identity: ResourceIdentity::Object(context),
            subdicts: [const { OnceLock::new() }; Category::COUNT],
            default_spaces: OnceLock::new(),
            color_space_source: OnceLock::new(),
        }
    }

    /// The underlying [`pdf::Document`] (read-only).
    #[must_use]
    pub const fn doc(&self) -> &'doc Document {
        self.doc
    }

    /// The object ID of the current resource context (page or form stream);
    /// every set built over a dictionary itself shares the `context` it was
    /// given (see [`Self::with_resources`]).
    #[must_use]
    pub const fn resource_context_id(&self) -> ObjectId {
        self.resource_context_id
    }

    /// The identity of the dictionaries in effect (see [`ResourceIdentity`]),
    /// distinct for every set [`Self::with_resources`] builds.
    #[must_use]
    pub const fn identity(&self) -> ResourceIdentity {
        self.identity
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
    pub fn font_entry(&self, name: &[u8]) -> Option<(ResourceKey, &Object)> {
        let entry = self.resource_subdict(Category::Font)?.get(name)?;
        Some((self.key_of(entry, name), entry))
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
    /// to, as does the underlying device space of an uncoloured `Pattern`
    /// space (PDF §8.6.5.6).
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
        match colorspace::resolve(self.doc, entry) {
            ColorSpace::Pattern { base: Some(base) } => ColorSpace::Pattern {
                base: Some(Box::new(self.device_space(*base))),
            },
            space => space,
        }
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

    /// The object identifying the `/ColorSpace` sub-dictionary in effect,
    /// shared by every context that inherits the same one; `None` when there
    /// is none.
    #[must_use]
    pub fn color_space_source(&self) -> Option<ObjectId> {
        *self.color_space_source.get_or_init(|| {
            self.doc
                .page_resource_entry(self.resource_context_id, Category::ColorSpace.key())
                .ok()
                .flatten()
                .map(|(_, source)| source)
        })
    }

    /// Whether the resources in effect replace any device colour space with
    /// a default space, so that a shading's device space resolves
    /// differently here than under resources without one.
    #[must_use]
    pub fn has_default_spaces(&self) -> bool {
        self.device_space(ColorSpace::DeviceGray) != ColorSpace::DeviceGray
            || self.device_space(ColorSpace::DeviceRgb) != ColorSpace::DeviceRgb
            || self.device_space(ColorSpace::DeviceCmyk) != ColorSpace::DeviceCmyk
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

    /// Resolve the named `Shading` resource, for an ink canvas when `ink`
    /// plans its colorants (see [`shading::resolve_shading_object`]).
    ///
    /// Returns `None` if the name is absent, the type is unsupported, or any
    /// required key is missing.
    #[must_use]
    pub(crate) fn shading(
        &self,
        name: &[u8],
        ctm: &[f64; 6],
        canvas: shading::Canvas,
        ink: Option<shading::PlanColorants<'_>>,
    ) -> Option<shading::ResolvedShading> {
        let shadings = self.resource_subdict(Category::Shading)?;
        shading::resolve_shading(self, shadings, name, ctm, canvas, ink)
    }

    /// Resolve the named `Pattern` resource as a shading pattern
    /// (`PatternType` 2, PDF §8.7.4.2).
    ///
    /// `pattern_ctm` maps the default coordinate space of the pattern's
    /// parent content stream to the device; the pattern's `Matrix` is
    /// concatenated onto it.  The shading is resolved for an ink canvas
    /// when `ink` plans its colorants.  Returns `None` if the name is
    /// absent, the pattern is not a shading pattern, or its shading does
    /// not resolve.
    #[must_use]
    pub(crate) fn shading_pattern(
        &self,
        name: &[u8],
        pattern_ctm: &[f64; 6],
        canvas: shading::Canvas,
        ink: Option<shading::PlanColorants<'_>>,
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
            shading::resolve_shading_object(self, pattern.get(b"Shading")?, &matrix, canvas, ink)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
