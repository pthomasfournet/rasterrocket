//! The ink of an `Indexed` image over a space whose components an ink
//! canvas paints.
//!
//! The samples of an `Indexed` image whose base space is a space of ink —
//! `DeviceCMYK`, a four-component ICC space, a `Separation` or a `DeviceN`
//! space — select palette entries that are amounts of that ink; over a
//! `CalRGB` or ICC RGB space, entries that are its components.  Each
//! sample maps through `/Decode` to an index, rounded and clamped to
//! `hival`, whose entry's bytes map onto the base's component ranges
//! (ISO 32000-2 §8.6.6.3), as the conversion of the colour does; the
//! entry is then the tints of the base, painted on an ink canvas as an
//! image in the base space is.

use super::{ImageColour, KeptTints, Layout, Samples, keeps};
use crate::resources::colorspace::{ColorSpace, decoded_sample};
use crate::resources::image::{ImageTints, TintData};

impl ImageColour {
    /// The tints of the base space of an `Indexed` image at each of its
    /// samples, when `kept` keeps the base's components (see the module
    /// documentation); `None` for an image in another space, for samples that are not one host-resident 8-bit
    /// index a pixel, and when a sample's entry is missing from the
    /// palette, which the conversion of the colour draws as black.
    pub(super) fn palette_tints(&self, samples: &Samples, kept: KeptTints) -> Option<ImageTints> {
        let indexed = self.space.as_ref()?;
        let ColorSpace::Indexed { base, .. } = indexed else {
            return None;
        };
        if !keeps(base, kept) || samples.layout != Layout::Color(1) {
            return None;
        }
        let &[pair] = self.decode.as_slice() else {
            return None;
        };
        let n = base.ncomponents();
        let mut table = vec![0u8; 256 * n];
        for (v, entry) in (0..=u8::MAX).zip(table.chunks_exact_mut(n)) {
            let index = decoded_sample(v, pair, self.max_sample);
            let (_, comps) = indexed.indexed_base_components(&[index])?;
            for (out, c) in entry.iter_mut().zip(comps) {
                *out = color::Sample::from_unit(c);
            }
        }
        let indices = samples.data.as_cpu()?;
        let mut data = Vec::new();
        data.try_reserve_exact(indices.len() * n).ok()?;
        data.extend(
            indices
                .iter()
                .flat_map(|&i| &table[usize::from(i) * n..(usize::from(i) + 1) * n]),
        );
        Some(ImageTints {
            space: (**base).clone(),
            data: TintData::U8(data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::empty_doc;

    fn colour(space: &str, decode: &str) -> ImageColour {
        let doc = empty_doc();
        let dict = pdf::parse_object(
            format!("<< /ColorSpace {space} /BitsPerComponent 8 {decode} >>").as_bytes(),
            &mut 0,
        )
        .expect("parses");
        let pdf::Object::Dictionary(dict) = dict else {
            panic!("a dictionary")
        };
        ImageColour::resolve(&doc, &dict, None)
    }

    #[test]
    fn an_indexed_cmyk_image_paints_its_palette_ink() {
        let c = colour("[/Indexed /DeviceCMYK 1 <ff000000 00000080>]", "");
        let tints = c
            .palette_tints(&Samples::gray(3, 1, vec![0, 1, 7]), KeptTints::Ink)
            .expect("tints");
        assert_eq!(tints.space, ColorSpace::DeviceCmyk);
        let TintData::U8(data) = tints.data else {
            panic!("8-bit tints")
        };
        // Index 7 clamps to hival.
        assert_eq!(data, [255, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 128]);
        // /Decode [1 0] reverses the indices.
        let c = colour(
            "[/Indexed /DeviceCMYK 1 <ff000000 00000080>]",
            "/Decode [1 0]",
        );
        let TintData::U8(data) = c
            .palette_tints(&Samples::gray(1, 1, vec![0]), KeptTints::Ink)
            .expect("tints")
            .data
        else {
            panic!("8-bit tints")
        };
        assert_eq!(data, [0, 0, 0, 128]);
    }

    #[test]
    fn only_an_indexed_image_of_ink_has_palette_tints() {
        let samples = Samples::gray(1, 1, vec![1]);
        assert!(
            colour("[/Indexed /DeviceRGB 1 <ff0000 00ff00>]", "")
                .palette_tints(&samples, KeptTints::Ink)
                .is_none()
        );
        assert!(
            colour("/DeviceCMYK", "")
                .palette_tints(&samples, KeptTints::Ink)
                .is_none()
        );
        // A short palette has no entry for index 1.
        assert!(
            colour("[/Indexed /DeviceCMYK 1 <ff000000>]", "")
                .palette_tints(&samples, KeptTints::Ink)
                .is_none()
        );
    }
}
