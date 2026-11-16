//! The colour space of a transparency group (ISO 32000-2 §11.6.6, `/CS`).
//!
//! A group space is a device or CIE-based space whose components are
//! independent additive or subtractive values in `[0, 1]` (§11.3.4):
//! `DeviceGray`, `DeviceRGB`, `DeviceCMYK`, `CalGray`, `CalRGB` and
//! `ICCBased` spaces of one, three or four components whose data space is
//! not Lab.  `Lab`, a Lab-data ICC profile and the special spaces
//! (`Indexed`, `Pattern`, `Separation`, `DeviceN`) are not group spaces; a
//! group that names one has no space of its own and inherits its parent's.

use std::sync::atomic::{AtomicBool, Ordering};

use pdf::{Document, Object, ObjectId};

use super::ColorSpace;
use super::cal::CalSpace;
use super::icc::IccConversion;

/// The family of a transparency group's colour space: the space the
/// group's colours convert into and composite in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GroupSpace {
    /// One additive component: `DeviceGray`, `CalGray`, a gray ICC space.
    Gray,
    /// Three additive components.
    Rgb(RgbGroupSpace),
    /// Four subtractive components: `DeviceCMYK` or a CMYK ICC space.
    Cmyk {
        /// The profile stream of an ICC space; `None` for `DeviceCMYK` and
        /// an ICC space without a stream, which converts as `DeviceCMYK`.
        profile: Option<ObjectId>,
    },
}

/// Which three-component space a group composites in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RgbGroupSpace {
    /// `DeviceRGB`, which the `DefaultRGB` space of the group's resources
    /// replaces when it names one (ISO 32000-2 §8.6.5.6).
    Device,
    /// A CIE-based space that converts as sRGB does.
    Srgb,
    /// An RGB ICC space whose profile, the stream of this identifier, is
    /// not sRGB.
    Icc(ObjectId),
    /// A `CalRGB` space.
    Cal(CalSpace),
}

impl GroupSpace {
    /// The family of `space`, a space resolved from a group's `/CS` or from
    /// a default colour space replacing a device one; `None` for a space
    /// that is not a group space.
    #[must_use]
    pub(crate) fn of(doc: &Document, space: &ColorSpace) -> Option<Self> {
        match space {
            ColorSpace::DeviceGray => Some(Self::Gray),
            ColorSpace::DeviceRgb => Some(Self::Rgb(RgbGroupSpace::Device)),
            ColorSpace::DeviceCmyk => Some(Self::Cmyk { profile: None }),
            ColorSpace::Cal(cal) => match cal.ncomponents() {
                1 => Some(Self::Gray),
                _ => Some(Self::Rgb(RgbGroupSpace::Cal(**cal))),
            },
            ColorSpace::IccBased { n, stream_id } => match (n, stream_id) {
                (1, _) => Some(Self::Gray),
                (3, Some(id)) => icc_rgb(doc, *id),
                (3, None) => Some(Self::Rgb(RgbGroupSpace::Device)),
                (4, id) => Some(Self::Cmyk { profile: *id }),
                _ => None,
            },
            ColorSpace::Lab(_)
            | ColorSpace::Indexed { .. }
            | ColorSpace::Pattern { .. }
            | ColorSpace::Separation { .. }
            | ColorSpace::DeviceN { .. } => None,
        }
    }

    /// The family a group's `/CS` entry `cs` names; `None`, logged once per
    /// process, when it names no group space.
    #[must_use]
    pub(crate) fn read(doc: &Document, cs: &Object) -> Option<Self> {
        let space = named_family(doc, cs, 0).and_then(|_| Self::of(doc, &super::resolve(doc, cs)));
        if space.is_none() {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "rasterrocket-interp: a transparency group's /CS is not a group colour space \
                     — the group composites in its parent's space"
                );
            }
        }
        space
    }

    /// The space with a `DeviceRGB` space replaced by `default_rgb`, the
    /// RGB space of the resources in effect (`DefaultRGB`, ISO 32000-2
    /// §8.6.5.6, or `DeviceRGB` itself), which is sRGB when it is
    /// `DeviceRGB` or not a group space.
    #[must_use]
    pub(crate) fn with_default_rgb(
        self,
        doc: &Document,
        default_rgb: impl FnOnce() -> ColorSpace,
    ) -> Self {
        if self != Self::Rgb(RgbGroupSpace::Device) {
            return self;
        }
        match default_rgb() {
            ColorSpace::DeviceRgb => Self::Rgb(RgbGroupSpace::Srgb),
            other => Self::of(doc, &other).unwrap_or(Self::Rgb(RgbGroupSpace::Srgb)),
        }
    }

    /// Whether the space is sRGB or gray, whose colours sRGB holds.
    #[must_use]
    pub const fn within_srgb(self) -> bool {
        matches!(
            self,
            Self::Gray | Self::Rgb(RgbGroupSpace::Device | RgbGroupSpace::Srgb)
        )
    }
}

/// The family name of `cs` when it names one of the group-space families,
/// following at most a few references; `None` otherwise.  [`super::resolve`]
/// replaces an unknown family with `DeviceGray`, so the family is checked
/// before the space is resolved.
fn named_family(doc: &Document, cs: &Object, depth: u8) -> Option<Vec<u8>> {
    const FAMILIES: [&[u8]; 6] = [
        b"DeviceGray",
        b"DeviceRGB",
        b"DeviceCMYK",
        b"CalGray",
        b"CalRGB",
        b"ICCBased",
    ];
    let name = match cs {
        Object::Name(n) => n.clone(),
        Object::Array(a) => a.first()?.as_name()?.to_vec(),
        Object::Reference(id) if depth < 4 => {
            let obj = doc.get_object(*id).ok()?;
            return named_family(doc, obj.as_ref(), depth + 1);
        }
        _ => return None,
    };
    FAMILIES.contains(&name.as_slice()).then_some(name)
}

/// The family of the RGB ICC profile stream `id`: sRGB when it converts as
/// `DeviceRGB` does under every intent, `None` when its data space is Lab.
fn icc_rgb(doc: &Document, id: ObjectId) -> Option<GroupSpace> {
    if IccConversion::is_device_equivalent_stream(doc, id, 3) {
        return Some(GroupSpace::Rgb(RgbGroupSpace::Srgb));
    }
    let lab = crate::resources::icc_shaper::fetch_profile_bytes(doc, id)
        .is_some_and(|bytes| bytes.get(16..20) == Some(b"Lab ".as_slice()));
    (!lab).then_some(GroupSpace::Rgb(RgbGroupSpace::Icc(id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_objects;

    /// A stream object of `dict` entries holding `data`.
    fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut s = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
        s.extend_from_slice(data);
        s.extend_from_slice(b"\nendstream");
        s
    }

    /// The family `cs` names in a document of `objects`, numbered from 4.
    fn read_in(cs: &str, objects: &[(u32, Vec<u8>)]) -> Option<GroupSpace> {
        let doc = make_doc_with_objects(objects);
        let obj = pdf::parse_object(cs.as_bytes(), &mut 0).expect("a /CS object");
        GroupSpace::read(&doc, &obj)
    }

    fn read(cs: &str) -> Option<GroupSpace> {
        read_in(cs, &[])
    }

    /// A profile stream of `n` components holding `profile`.
    fn profile(n: u8, profile: &[u8]) -> Vec<(u32, Vec<u8>)> {
        vec![(4, stream(&format!("/N {n}"), profile))]
    }

    #[test]
    fn device_names_are_their_families() {
        assert_eq!(read("/DeviceGray"), Some(GroupSpace::Gray));
        assert_eq!(
            read("/DeviceRGB"),
            Some(GroupSpace::Rgb(RgbGroupSpace::Device))
        );
        assert_eq!(
            read("/DeviceCMYK"),
            Some(GroupSpace::Cmyk { profile: None })
        );
    }

    #[test]
    fn special_lab_and_unknown_spaces_are_not_group_spaces() {
        assert_eq!(read("/Pattern"), None);
        assert_eq!(read("/Foo"), None);
        assert_eq!(read("[/Lab << /WhitePoint [0.9505 1 1.089] >>]"), None);
        assert_eq!(read("[/Indexed /DeviceRGB 1 <000000FFFFFF>]"), None);
        assert_eq!(
            read(
                "[/Separation /Spot /DeviceCMYK << /FunctionType 2 /Domain [0 1] \
                 /C0 [0 0 0 0] /C1 [1 0 0 0] /N 1 >>]"
            ),
            None
        );
    }

    #[test]
    fn cal_spaces_take_their_component_count() {
        assert_eq!(
            read("[/CalGray << /WhitePoint [0.9505 1 1.089] >>]"),
            Some(GroupSpace::Gray)
        );
        assert!(matches!(
            read("[/CalRGB << /WhitePoint [0.9505 1 1.089] /Gamma [1.8 1.8 1.8] >>]"),
            Some(GroupSpace::Rgb(RgbGroupSpace::Cal(_)))
        ));
    }

    #[test]
    fn icc_spaces_take_their_component_count_and_srgb_is_recognised() {
        let srgb = {
            let mut p = moxcms::ColorProfile::new_srgb();
            p.cicp = None;
            p.encode().expect("encodes")
        };
        let adobe = moxcms::ColorProfile::new_adobe_rgb()
            .encode()
            .expect("encodes");
        assert_eq!(
            read_in("[/ICCBased 4 0 R]", &profile(3, &srgb)),
            Some(GroupSpace::Rgb(RgbGroupSpace::Srgb))
        );
        assert_eq!(
            read_in("[/ICCBased 4 0 R]", &profile(3, &adobe)),
            Some(GroupSpace::Rgb(RgbGroupSpace::Icc((4, 0))))
        );
        assert_eq!(
            read_in("[/ICCBased 4 0 R]", &profile(1, b"")),
            Some(GroupSpace::Gray)
        );
        assert_eq!(
            read_in("[/ICCBased 4 0 R]", &profile(4, b"")),
            Some(GroupSpace::Cmyk {
                profile: Some((4, 0))
            })
        );
    }

    #[test]
    fn an_icc_profile_of_lab_data_is_not_a_group_space() {
        let mut lab = vec![0u8; 128];
        lab[16..20].copy_from_slice(b"Lab ");
        assert_eq!(read_in("[/ICCBased 4 0 R]", &profile(3, &lab)), None);
    }

    #[test]
    fn a_reference_resolves_to_its_space() {
        assert_eq!(
            read_in("4 0 R", &[(4, b"/DeviceCMYK".to_vec())]),
            Some(GroupSpace::Cmyk { profile: None })
        );
    }
}
