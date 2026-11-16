//! Colour-space resources (ISO 32000-1 §8.6.3, §8.6.5.6) of the resource
//! dictionaries in effect.

use pdf::{Dictionary, ObjectId};

use super::{Category, PageResources};
use crate::resources::{ColorSpace, colorspace};

impl PageResources<'_> {
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
