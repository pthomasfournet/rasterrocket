//! PDF resource resolution — fonts, colour spaces, `XObject`s, shadings.
//!
//! [`PageResources`] resolves the named resources of a content stream
//! through the resource dictionaries in effect for it; the submodules parse
//! each kind of resource.  This module holds the dictionary readers they
//! share.

pub(crate) mod black_generation;
pub(crate) mod cmap;
pub(crate) mod cmap_predefined;
pub(crate) mod colorspace;
pub(crate) mod dict_ext;
mod ext_gstate;
pub(crate) mod font;
pub(crate) mod font_substitute;
pub(crate) mod icc_shaper;
pub(crate) mod image;
pub(crate) mod lattice;
mod page_resources;
pub(crate) mod shading;
pub(crate) mod tiling;
pub(crate) mod transfer;
mod transparency_group;

use pdf::{Dictionary, Document, Object};

pub use colorspace::{ColorSpace, GroupSpace, RgbGroupSpace};
pub(crate) use ext_gstate::parse_blend_mode;
pub use ext_gstate::{ExtGStateParams, OverprintMode, SoftMaskParams};
pub use font::{BaseEncoding, FontDescriptor, PdfFontKind, resolve_font};
/// Re-exported from [`gpu::nvjpeg`] for callers that enable the `nvjpeg` feature.
/// Create via [`gpu::nvjpeg::NvJpegDecoder::new`] with a raw `CUstream` handle,
/// then pass to the page renderer's `set_nvjpeg` configuration entry point.
#[cfg(feature = "nvjpeg")]
pub use gpu::nvjpeg::NvJpegDecoder;
pub use image::{
    DecodedImage, IMAGE_FILTER_COUNT, ImageColorSpace, ImageDescriptor, ImageFilter, ImageRequest,
    ImageResolution, ImageTints, SoftMask, TintData, resolve_image,
};
pub use page_resources::{
    FormXObject, PageResources, ResourceIdentity, ResourceKey, ShadingPattern,
};
pub use transparency_group::{TransparencyGroupParams, read_transparency_group};

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

/// The elements of the array `obj`, following one indirect reference;
/// empty when it is absent or not an array.
pub(crate) fn resolve_array(doc: &Document, obj: Option<&Object>) -> Vec<Object> {
    match obj {
        Some(Object::Array(a)) => a.clone(),
        Some(Object::Reference(id)) => doc
            .get_object(*id)
            .ok()
            .and_then(|o| o.as_array().map(<[Object]>::to_vec))
            .unwrap_or_default(),
        _ => Vec::new(),
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
