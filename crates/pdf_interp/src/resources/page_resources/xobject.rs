//! Image and form `XObject` resources (ISO 32000-1 §8.8, §8.10) of the
//! resource dictionaries in effect.

use pdf::{Dictionary, Object, ObjectId};

use super::{Category, PageResources};
#[cfg(feature = "nvjpeg2k")]
use crate::resources::dict_ext::DictExt;
use crate::resources::{
    TransparencyGroupParams, image, read_bbox, read_matrix, read_transparency_group,
};

/// Parsed Form `XObject` — content bytes and the CTM matrix to apply.
///
/// The `resources_id` is the `ObjectId` of the form stream itself; callers
/// construct a child [`PageResources`] from it via [`PageResources::for_form`].
#[non_exhaustive]
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

impl FormXObject {
    /// The form's dictionary entries, with no content.
    #[must_use]
    pub const fn without_content(&self) -> Self {
        Self {
            content: Vec::new(),
            matrix: self.matrix,
            bbox: self.bbox,
            resources_id: self.resources_id,
            has_own_resources: self.has_own_resources,
            transparency: self.transparency,
        }
    }
}

impl PageResources<'_> {
    /// Decode the image `XObject` `request` names from the resource dictionary,
    /// at no less than `request.draw_size` when the codec can reduce.
    ///
    /// Returns [`image::ImageResolution::Absent`] if the name is absent or not
    /// an image, and [`image::ImageResolution::DecodeFailed`] if decoding fails.
    ///
    /// `ctx` carries the GPU decoders, ICC handles and image cache the decode
    /// may use; `image::ImageDecodeCtx::cpu` decodes on the CPU only.
    #[must_use]
    pub fn image(
        &self,
        request: image::ImageRequest<'_>,
        ctx: &mut image::ImageDecodeCtx<'_>,
    ) -> image::ImageResolution {
        let empty = Dictionary::new();
        let xobjects = self.resource_subdict(Category::XObject).unwrap_or(&empty);
        image::resolve_image(ctx, xobjects, request)
    }

    /// Whether the GPU JPEG 2000 decoder is worth offering the `XObject`
    /// resource `name` drawn at `draw_size`: an image with terminal filter
    /// `JPXDecode` that meets the dispatch threshold `threshold`
    /// ([`image::jpx_area_dispatches`]) by its dictionary, whose CPU decode
    /// would not skip a resolution level (see `draw_halves`) at the size its
    /// masks let it reduce to (see `reduced_draw_size`).
    #[cfg(feature = "nvjpeg2k")]
    #[must_use]
    pub(crate) fn xobject_takes_device_jpx(
        &self,
        name: &[u8],
        draw_size: Option<(u32, u32)>,
        threshold: u32,
    ) -> bool {
        let Some(obj) = self
            .xobject_id(name)
            .and_then(|id| self.doc.get_object(id).ok())
        else {
            return false;
        };
        let Some(dict) = obj.as_ref().as_stream().map(|s| &s.dict) else {
            return false;
        };
        let Some(full) = device_jpx_size(dict, threshold) else {
            return false;
        };
        !image::reduced_draw_size(self.doc, dict, draw_size)
            .is_some_and(|draw| image::draw_halves(full, draw))
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
        let (stream_id, _) = self.form_xobject_entry(name)?;
        self.form_from_stream_id(stream_id)
    }

    /// The stream object of the named `XObject` when its subtype is `Form`,
    /// and whether the form has a `Resources` dictionary of its own; its
    /// content is not decoded.
    #[must_use]
    pub fn form_xobject_entry(&self, name: &[u8]) -> Option<(ObjectId, bool)> {
        let stream_id = self.xobject_id(name)?;
        let obj = self.doc.get_object(stream_id).ok()?;
        let dict = &obj.as_stream()?.dict;
        if dict.get(b"Subtype")?.as_name()? != b"Form" {
            return None;
        }
        Some((stream_id, dict.get(b"Resources").is_some()))
    }

    /// Build a [`FormXObject`] directly from a stream object ID.
    ///
    /// Used for annotation appearance streams, which are referenced by object ID
    /// rather than through the page `Resources/XObject` dict.  `None` when the
    /// object is not a stream or its content does not decode (logged).
    #[must_use]
    pub fn form_from_stream_id(&self, stream_id: ObjectId) -> Option<FormXObject> {
        let obj = self.doc.get_object(stream_id).ok()?;
        let stream = obj.as_stream()?;
        let content = match stream.decompressed_content() {
            Ok(content) => content,
            Err(e) => {
                log::warn!(
                    "rasterrocket-interp: form object {} does not decode ({e}) — skipping",
                    stream_id.0
                );
                return None;
            }
        };
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

    /// The `/OC` entry of the `XObject` resource `name` (PDF §8.11.3.3): the
    /// optional content group or membership dictionary governing it.
    #[must_use]
    pub fn xobject_optional_content(&self, name: &[u8]) -> Option<Object> {
        let stream = self.doc.get_object(self.xobject_id(name)?).ok()?;
        stream.as_stream()?.dict.get(b"OC").cloned()
    }

    /// The `Intent` of the `XObject` resource `name`: an image's own
    /// rendering intent (§8.9.5.1).  `None` when absent.
    #[must_use]
    pub fn xobject_intent(&self, name: &[u8]) -> Option<crate::RenderingIntent> {
        let stream = self.doc.get_object(self.xobject_id(name)?).ok()?;
        let name = stream.as_stream()?.dict.get(b"Intent")?.as_name()?;
        Some(crate::RenderingIntent::from_pdf_name(name))
    }
}

/// The dimensions of the image dictionary `dict` when the GPU JPEG 2000
/// decoder takes it: terminal filter `JPXDecode` and an area meeting the
/// dispatch threshold `threshold` ([`image::jpx_area_dispatches`]).
#[cfg(feature = "nvjpeg2k")]
fn device_jpx_size(dict: &Dictionary, threshold: u32) -> Option<(u32, u32)> {
    let chain = image::image_filter_chain(dict.get(b"Filter")?);
    if chain.last().is_none_or(|f| f != "JPXDecode") {
        return None;
    }
    let dim = |key: &[u8]| dict.get_i64(key).and_then(|v| u32::try_from(v).ok());
    let (w, h) = (dim(b"Width")?, dim(b"Height")?);
    image::jpx_area_dispatches(w, h, threshold).then_some((w, h))
}

#[cfg(all(test, feature = "nvjpeg2k"))]
mod tests {
    use super::*;

    /// The GPU JPEG 2000 decoder is declined for a draw at half size or
    /// less only when the CPU decode reduces: a matte soft mask or a
    /// colour-key mask decodes in full, so the device decoder is offered.
    #[test]
    fn device_jpx_is_offered_where_the_cpu_decode_cannot_reduce() {
        let jpx = "/Type /XObject /Subtype /Image /Width 4096 /Height 4096 \
                   /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /JPXDecode";
        let stream = |extra: &str| format!("<< {jpx} {extra} /Length 0 >>\nstream\n\nendstream");
        let doc = crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /Resources << /XObject << /Plain 5 0 R /Matted 6 0 R \
                 /Keyed 8 0 R >> >> >>"
                    .to_owned(),
            ),
            (5, stream("")),
            (6, stream("/SMask 7 0 R")),
            (
                7,
                "<< /Type /XObject /Subtype /Image /Width 4096 /Height 4096 /ColorSpace \
                 /DeviceGray /BitsPerComponent 8 /Matte [0 0 0] /Length 0 >>\nstream\n\nendstream"
                    .to_owned(),
            ),
            (8, stream("/Mask [0 0 0 0 0 0]")),
        ]);
        let resources = PageResources::new(&doc, (4, 0));
        let takes = |name: &[u8], draw| resources.xobject_takes_device_jpx(name, draw, 2_097_152);
        let half = Some((2048, 2048));
        assert!(!takes(b"Plain", half));
        assert!(takes(b"Plain", Some((2049, 2048))));
        assert!(takes(b"Plain", None));
        assert!(takes(b"Matted", half));
        assert!(takes(b"Keyed", half));
        assert!(!takes(b"Missing", None));
        assert!(!resources.xobject_takes_device_jpx(b"Plain", None, u32::MAX));
    }
}
