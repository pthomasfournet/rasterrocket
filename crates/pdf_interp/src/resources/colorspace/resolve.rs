//! The resolution of a colour-space object into a [`ColorSpace`].

use std::sync::Arc;

use pdf::{Dictionary, Document, Object, ObjectId};

use super::icc::IccConversion;
use super::{CalSpace, ColorSpace, Colorant, LabSpace, TintTransform};
use crate::resources::dict_ext::DictExt as _;

/// Maximum recursion depth for nested colour-space chains
/// (Indexed/Pattern/Separation/DeviceN's `alternate` base). PDF spec
/// forbids cycles but adversarial PDFs are not required to be
/// spec-compliant; this bound prevents stack overflow on crafted input.
const MAX_CS_DEPTH: u8 = 8;

/// Resolve a PDF `ColorSpace` object into a [`ColorSpace`] tracking value.
///
/// Returns [`ColorSpace::DeviceGray`] on unknown, malformed, or
/// recursion-bounded inputs.
///
/// `cs_obj` may be:
/// - a `Name` for the device families (e.g. `/DeviceRGB`, `/CMYK`),
/// - an `Array` whose first element names a parameterised family
///   (`[/ICCBased <ref>]`, `[/Indexed base hival lookup]`, etc.), or
/// - a `Reference` to either of the above.
#[must_use]
pub(crate) fn resolve(doc: &Document, cs_obj: &Object) -> ColorSpace {
    resolve_depth(doc, cs_obj, 0)
}

fn resolve_depth(doc: &Document, cs_obj: &Object, depth: u8) -> ColorSpace {
    if depth >= MAX_CS_DEPTH {
        log::debug!("colorspace: nested too deep (depth {depth}); falling back to DeviceGray");
        return ColorSpace::DeviceGray;
    }
    match cs_obj {
        Object::Name(n) => resolve_device_name(n),
        Object::Reference(id) => doc
            .get_object(*id)
            .ok()
            .map_or(ColorSpace::DeviceGray, |obj| {
                resolve_depth(doc, obj.as_ref(), depth + 1)
            }),
        Object::Array(arr) => resolve_array(doc, arr, depth),
        _ => {
            log::debug!("colorspace: unexpected object type — fallback DeviceGray");
            ColorSpace::DeviceGray
        }
    }
}

/// Map a PDF device-space name (including the `cs`-operator abbreviations
/// from PDF §8.9.7 Table 89) to a [`ColorSpace`]. Unknown names fall back
/// to `DeviceGray`.
///
/// Public so callers that hold a `&[u8]` name straight from the operator
/// stream can short-circuit the four device-space cases without allocating
/// an `Object::Name`. Internal callers in this module use it directly from
/// the `resolve_*` dispatch.
#[must_use]
pub(crate) fn resolve_device_name(name: &[u8]) -> ColorSpace {
    match name {
        b"DeviceGray" | b"G" | b"CalGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" | b"RGB" | b"CalRGB" => ColorSpace::DeviceRgb,
        b"DeviceCMYK" | b"CMYK" => ColorSpace::DeviceCmyk,
        b"Pattern" => ColorSpace::Pattern { base: None },
        _ => {
            log::debug!(
                "colorspace: unknown device name {:?} — fallback DeviceGray",
                String::from_utf8_lossy(name)
            );
            ColorSpace::DeviceGray
        }
    }
}

/// The default colour space that the `/ColorSpace` resource dictionary
/// `cs_dict` declares for the device space `device` (PDF §8.6.5.6):
/// `/DefaultGray`, `/DefaultRGB` or `/DefaultCMYK`.
///
/// `None` when `device` is not a device space, the entry is absent, or the
/// entry is unusable: a space of a different component count, a device
/// space, or an `Indexed` or `Pattern` space.  `None` also when the entry
/// is an ICC-based space whose conversion equals the device conversion, so
/// the device path stays in effect for sRGB-equivalent profiles.
#[must_use]
pub(crate) fn default_space(
    doc: &Document,
    cs_dict: &Dictionary,
    device: &ColorSpace,
) -> Option<ColorSpace> {
    let key: &[u8] = match device {
        ColorSpace::DeviceGray => b"DefaultGray",
        ColorSpace::DeviceRgb => b"DefaultRGB",
        ColorSpace::DeviceCmyk => b"DefaultCMYK",
        _ => return None,
    };
    let space = resolve(doc, cs_dict.get(key)?);
    let usable = space.ncomponents() == device.ncomponents()
        && !matches!(
            space,
            ColorSpace::DeviceGray
                | ColorSpace::DeviceRgb
                | ColorSpace::DeviceCmyk
                | ColorSpace::Indexed { .. }
                | ColorSpace::Pattern { .. }
        );
    if !usable {
        log::debug!(
            "colorspace: /{} {space:?} unusable for {device:?}",
            String::from_utf8_lossy(key)
        );
        return None;
    }
    if let ColorSpace::IccBased {
        n,
        stream_id: Some(id),
    } = space
        && IccConversion::is_device_equivalent_stream(doc, id, n)
    {
        return None;
    }
    Some(space)
}

fn resolve_array(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    let Some(family) = arr.first().and_then(Object::as_name) else {
        log::debug!("colorspace: array missing leading family name — fallback");
        return ColorSpace::DeviceGray;
    };
    match family {
        b"DeviceGray" => ColorSpace::DeviceGray,
        b"DeviceRGB" => ColorSpace::DeviceRgb,
        b"CalGray" => resolve_cal(doc, arr, 1, ColorSpace::DeviceGray),
        b"CalRGB" => resolve_cal(doc, arr, 3, ColorSpace::DeviceRgb),
        b"DeviceCMYK" => ColorSpace::DeviceCmyk,
        b"Lab" => ColorSpace::Lab(Box::new(LabSpace::resolve(doc, arr.get(1)))),
        b"ICCBased" => resolve_icc_based(doc, arr.get(1)),
        b"Indexed" | b"I" => resolve_indexed(doc, arr, depth),
        b"Pattern" => ColorSpace::Pattern {
            base: arr
                .get(1)
                .map(|o| Box::new(resolve_depth(doc, o, depth + 1))),
        },
        b"Separation" => resolve_separation(doc, arr, depth),
        b"DeviceN" => resolve_device_n(doc, arr, depth),
        _ => {
            log::debug!(
                "colorspace: unknown array family {:?} — fallback DeviceGray",
                String::from_utf8_lossy(family)
            );
            ColorSpace::DeviceGray
        }
    }
}

/// `[/CalGray dict]` (`n = 1`) or `[/CalRGB dict]` (`n = 3`); `device`
/// when the parameter dictionary is missing or invalid.
fn resolve_cal(doc: &Document, arr: &[Object], n: u8, device: ColorSpace) -> ColorSpace {
    arr.get(1)
        .and_then(|params| CalSpace::resolve(doc, params, n))
        .map_or(device, |cal| ColorSpace::Cal(Box::new(cal)))
}

fn resolve_icc_based(doc: &Document, ref_obj: Option<&Object>) -> ColorSpace {
    let Some(ref_obj) = ref_obj else {
        return ColorSpace::DeviceGray;
    };
    let stream_id = reference_id(ref_obj);
    // Read N from the stream dict if reachable.
    let n = crate::resources::resolve_stream_dict(doc, ref_obj)
        .and_then(|d| d.get_i64(b"N"))
        .and_then(|n| u8::try_from(n).ok())
        .filter(|&n| matches!(n, 1 | 3 | 4))
        .unwrap_or(0);
    if n == 0 {
        log::debug!("colorspace: ICCBased stream has missing or invalid /N");
        return ColorSpace::DeviceGray;
    }
    ColorSpace::IccBased { n, stream_id }
}

fn resolve_indexed(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Indexed base hival lookup]
    let Some(base_obj) = arr.get(1) else {
        return ColorSpace::DeviceGray;
    };
    let base = Box::new(resolve_depth(doc, base_obj, depth + 1));
    let hival = arr
        .get(2)
        .and_then(Object::as_i64)
        .map(|n| n.clamp(0, 255))
        .and_then(|n| u8::try_from(n).ok())
        .unwrap_or(0);
    ColorSpace::Indexed {
        base,
        hival,
        lookup: arr.get(3).and_then(|o| indexed_lookup(doc, o)),
    }
}

/// The palette bytes of an `Indexed` lookup `obj` (PDF §8.6.6.3): a string,
/// or a reference to a string or stream, read once per document.
fn indexed_lookup(doc: &Document, obj: &Object) -> Option<Arc<[u8]>> {
    /// Document-cache entry for an indirect lookup.
    struct Lookup(Arc<[u8]>);

    let bytes = |o: &Object| match o {
        Object::String(b, _) => Some(Arc::<[u8]>::from(b.as_slice())),
        Object::Stream(s) => s
            .decompressed_content()
            .map_err(|e| log::debug!("colorspace: Indexed lookup stream: {e}"))
            .ok()
            .map(Arc::from),
        _ => None,
    };
    match obj {
        Object::Reference(id) => doc
            .derived(*id, || {
                let o = doc.get_object(*id).ok()?;
                bytes(o.as_ref()).map(Lookup)
            })
            .map(|l| Arc::clone(&l.0)),
        direct => bytes(direct),
    }
}

fn resolve_separation(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/Separation name alternate tint_transform]
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    ColorSpace::Separation {
        alternate,
        tint: arr.get(3).map(|o| TintTransform::load(doc, o)),
        colorant: colorant_of(arr.get(1)),
    }
}

/// The colorant the name object `name` declares; a spot colorant with an
/// empty name when the object is missing or not a name.
fn colorant_of(name: Option<&Object>) -> Colorant {
    Colorant::parse(name.and_then(Object::as_name).unwrap_or_default())
}

fn resolve_device_n(doc: &Document, arr: &[Object], depth: u8) -> ColorSpace {
    // [/DeviceN names alternate tint_transform attributes?]
    let Some(names) = arr.get(1).and_then(Object::as_array) else {
        return ColorSpace::DeviceGray;
    };
    if names.is_empty() || u8::try_from(names.len()).is_err() {
        log::debug!(
            "colorspace: DeviceN has {} colorant names, not 1–255 — fallback DeviceGray",
            names.len()
        );
        return ColorSpace::DeviceGray;
    }
    let Some(alt_obj) = arr.get(2) else {
        return ColorSpace::DeviceGray;
    };
    let alternate = Box::new(resolve_depth(doc, alt_obj, depth + 1));
    ColorSpace::DeviceN {
        colorants: names.iter().map(|n| colorant_of(Some(n))).collect(),
        alternate,
        tint: arr.get(3).map(|o| TintTransform::load(doc, o)),
    }
}

const fn reference_id(obj: &Object) -> Option<ObjectId> {
    match obj {
        Object::Reference(id) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::colorspace::ProcessInk;
    use crate::resources::colorspace::test_support::{
        RI, devicen_teal_doc, lab_d65, name_obj, parse_space,
    };
    use crate::test_helpers::empty_doc;

    #[test]
    fn device_names_resolve_to_device_variants() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceGray")),
            ColorSpace::DeviceGray
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceRGB")),
            ColorSpace::DeviceRgb
        );
        assert_eq!(
            resolve(&doc, &name_obj(b"DeviceCMYK")),
            ColorSpace::DeviceCmyk
        );
    }

    #[test]
    fn abbreviated_names_resolve_per_table_89() {
        let doc = empty_doc();
        // PDF §8.9.7 Table 89 abbreviations (used in inline images but also
        // accepted by `cs`/`CS` per most decoders).
        assert_eq!(resolve(&doc, &name_obj(b"G")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"RGB")), ColorSpace::DeviceRgb);
        assert_eq!(resolve(&doc, &name_obj(b"CMYK")), ColorSpace::DeviceCmyk);
    }

    #[test]
    fn cal_spaces_approximate_to_device() {
        let doc = empty_doc();
        // The bare names carry no parameters; they map to the device spaces.
        assert_eq!(resolve(&doc, &name_obj(b"CalGray")), ColorSpace::DeviceGray);
        assert_eq!(resolve(&doc, &name_obj(b"CalRGB")), ColorSpace::DeviceRgb);
    }

    #[test]
    fn pattern_name_yields_pattern_no_base() {
        let doc = empty_doc();
        assert_eq!(
            resolve(&doc, &name_obj(b"Pattern")),
            ColorSpace::Pattern { base: None },
        );
    }

    #[test]
    fn unknown_name_falls_back_to_devicegray() {
        let doc = empty_doc();
        assert_eq!(resolve(&doc, &name_obj(b"Bogus")), ColorSpace::DeviceGray);
    }

    #[test]
    fn lab_array_resolves_its_parameters_by_value_or_reference() {
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            "<< /WhitePoint [0.9505 1 1.089] /Range [-50 50 -50 50] >>",
        )]);
        let by_ref = resolve(
            &doc,
            &Object::Array(vec![name_obj(b"Lab"), Object::Reference((4, 0))]),
        );
        let inline = parse_space(
            &doc,
            "[/Lab << /WhitePoint [0.9505 1 1.089] /Range [-50 50 -50 50] >>]",
        );
        assert_eq!(by_ref, inline);
        let ColorSpace::Lab(space) = &by_ref else {
            panic!("Lab expected");
        };
        assert_eq!(space.range(), [-50.0, 50.0, -50.0, 50.0]);
        assert_ne!(by_ref, lab_d65());
    }

    #[test]
    fn pattern_array_with_base_recurses_into_base() {
        let doc = empty_doc();
        let arr = vec![name_obj(b"Pattern"), name_obj(b"DeviceRGB")];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Pattern {
                base: Some(Box::new(ColorSpace::DeviceRgb)),
            },
        );
    }

    #[test]
    fn indexed_array_resolves_base_and_clamps_hival() {
        let doc = empty_doc();
        // [/Indexed /DeviceRGB 255 <lookup>]
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(255),
            Object::String(b"\x00\x01\x02".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 255,
                lookup: Some(Arc::from(&b"\x00\x01\x02"[..])),
            },
        );
    }

    #[test]
    fn indexed_hival_out_of_range_clamps_to_255() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceGray"),
            Object::Integer(99_999), // adversarial
            Object::String(b"".to_vec(), pdf::StringFormat::Hexadecimal),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceGray),
                hival: 255,
                lookup: Some(Arc::from(&[][..])),
            },
        );
    }

    #[test]
    fn indexed_lookup_reference_to_a_missing_object_has_no_palette() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Indexed"),
            name_obj(b"DeviceRGB"),
            Object::Integer(15),
            Object::Reference((42, 0)),
        ];
        assert_eq!(
            resolve(&doc, &Object::Array(arr)),
            ColorSpace::Indexed {
                base: Box::new(ColorSpace::DeviceRgb),
                hival: 15,
                lookup: None,
            },
        );
    }

    #[test]
    fn separation_extracts_alternate_and_tint() {
        let doc = empty_doc();
        let arr = vec![
            name_obj(b"Separation"),
            name_obj(b"PANTONE_185_C"),
            name_obj(b"DeviceCMYK"),
            Object::Reference((42, 0)),
        ];
        let ColorSpace::Separation {
            alternate,
            tint,
            colorant,
        } = resolve(&doc, &Object::Array(arr))
        else {
            panic!("expected Separation");
        };
        assert_eq!(*alternate, ColorSpace::DeviceCmyk);
        assert!(tint.is_some());
        assert_eq!(colorant, Colorant::parse(b"PANTONE_185_C"));
    }

    #[test]
    fn device_n_takes_a_colorant_per_name() {
        let doc = empty_doc();
        let names = Object::Array(vec![
            name_obj(b"Cyan"),
            name_obj(b"Magenta"),
            name_obj(b"Spot1"),
        ]);
        let arr = vec![
            name_obj(b"DeviceN"),
            names,
            name_obj(b"DeviceCMYK"),
            Object::Reference((7, 0)),
        ];
        let ColorSpace::DeviceN {
            colorants,
            alternate,
            tint,
        } = resolve(&doc, &Object::Array(arr))
        else {
            panic!("expected DeviceN");
        };
        assert_eq!(*alternate, ColorSpace::DeviceCmyk);
        assert!(tint.is_some());
        assert_eq!(
            *colorants,
            [
                Colorant::Process(ProcessInk::Cyan),
                Colorant::Process(ProcessInk::Magenta),
                Colorant::parse(b"Spot1")
            ]
        );
    }

    #[test]
    fn malformed_array_falls_back_to_devicegray() {
        let doc = empty_doc();
        // Empty array, no family name.
        assert_eq!(
            resolve(&doc, &Object::Array(vec![])),
            ColorSpace::DeviceGray,
        );
        // Unknown family.
        let unknown = Object::Array(vec![name_obj(b"Bogus")]);
        assert_eq!(resolve(&doc, &unknown), ColorSpace::DeviceGray);
    }

    #[test]
    fn convert_indexed_looks_up_the_palette_through_the_base() {
        // Two-entry RGB palette: red, blue.
        let doc = crate::test_helpers::make_doc_with_objects(&[(
            4,
            &b"<< /Length 6 >>\nstream\n\xff\x00\x00\x00\x00\xff\nendstream"[..],
        )]);
        let idx = ColorSpace::Indexed {
            base: Box::new(ColorSpace::DeviceRgb),
            hival: 1,
            lookup: indexed_lookup(&doc, &Object::Reference((4, 0))),
        };
        assert_eq!(idx.convert_to_rgb(&doc, RI, &[0.0]), [255, 0, 0]);
        assert_eq!(idx.convert_to_rgb(&doc, RI, &[1.0]), [0, 0, 255]);
        // An out-of-range index clamps to hival.
        assert_eq!(idx.convert_to_rgb(&doc, RI, &[7.0]), [0, 0, 255]);
    }

    #[test]
    fn inline_tint_transform_converts_through_the_alternate() {
        let doc = empty_doc();
        let tint = "<< /FunctionType 2 /Domain [0 1] /C0 [1 1 1] /C1 [1 0 0] /N 1 >>";
        let sep = parse_space(&doc, &format!("[/Separation /Spot /DeviceRGB {tint}]"));
        assert_eq!(sep.convert_to_rgb(&doc, RI, &[1.0]), [255, 0, 0]);
        assert_eq!(sep.convert_to_rgb(&doc, RI, &[0.0]), [255, 255, 255]);
        let dn = parse_space(&doc, &format!("[/DeviceN [/Spot] /DeviceRGB {tint}]"));
        assert_eq!(dn.convert_to_rgb(&doc, RI, &[1.0]), [255, 0, 0]);
    }

    #[test]
    fn inline_indexed_palette_is_looked_up() {
        let doc = empty_doc();
        let idx = parse_space(&doc, "[/Indexed /DeviceRGB 1 <ff000000ff00>]");
        assert_eq!(idx.convert_to_rgb(&doc, RI, &[0.0]), [255, 0, 0]);
        assert_eq!(idx.convert_to_rgb(&doc, RI, &[1.0]), [0, 255, 0]);
    }

    #[test]
    fn indirect_tint_transform_is_shared_across_resolutions() {
        let doc = devicen_teal_doc();
        let src = "[/DeviceN [/Black /Spot] /DeviceCMYK 4 0 R]";
        assert_eq!(parse_space(&doc, src), parse_space(&doc, src));
    }
}
