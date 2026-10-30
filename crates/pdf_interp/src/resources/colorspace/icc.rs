//! Colour-value conversion through an ICC profile stream (PDF §8.6.5.5).
//!
//! A profile stream is compiled once per document into an
//! [`IccConversion`]: a gray or RGB matrix/TRC profile into the shared
//! shaper tables of `super::super::icc_shaper`, any other 3- or 4-channel
//! profile into a `moxcms` 8-bit transform to sRGB.  Converting a colour
//! value is then a table lookup or one transform call.

use std::sync::Arc;

use moxcms::{ColorProfile, Layout, Transform8BitExecutor, TransformOptions};
use pdf::{Document, ObjectId};

use crate::resources::icc_shaper::{ProfileConversion, fetch_profile_bytes};

/// The compiled sRGB conversion of one ICC profile stream for colour values
/// of its declared component count.
pub(super) enum IccConversion {
    /// A gray (`n = 1`) or RGB (`n = 3`) matrix/TRC profile.
    Shaper(ProfileConversion),
    /// Any other profile with `n` of 3 or 4: an 8-bit transform whose source
    /// pixel holds `n` channels.
    Transform {
        n: u8,
        xform: Arc<Transform8BitExecutor>,
    },
}

impl IccConversion {
    /// The conversion of profile stream `id` for `n`-component values,
    /// compiled on first use and shared for the lifetime of `doc`.  `None`
    /// when the stream cannot be read, the profile does not parse, or no
    /// conversion exists for `n` channels.
    pub(super) fn load(doc: &Document, id: ObjectId, n: u8) -> Option<Arc<Self>> {
        doc.derived(id, || Self::compile(&fetch_profile_bytes(doc, id)?, n))
    }

    /// Compile `bytes` for `n`-component values without consulting a cache.
    fn compile(bytes: &[u8], n: u8) -> Option<Self> {
        let shaper = ProfileConversion::from_profile_bytes(bytes);
        if matches!(
            (&shaper, n),
            (ProfileConversion::Gray(_), 1) | (ProfileConversion::Rgb(_), 3)
        ) {
            return Some(Self::Shaper(shaper));
        }
        let layout = match n {
            3 => Layout::Rgb,
            4 => Layout::Rgba,
            _ => return None,
        };
        let src = ColorProfile::new_from_slice(bytes)
            .map_err(|e| log::debug!("colorspace: ICC profile parse failed: {e}"))
            .ok()?;
        let xform = src
            .create_transform_8bit(
                layout,
                &ColorProfile::new_srgb(),
                Layout::Rgb,
                TransformOptions::default(),
            )
            .map_err(|e| log::debug!("colorspace: ICC transform creation failed: {e}"))
            .ok()?;
        Some(Self::Transform { n, xform })
    }

    /// Convert normalised components to sRGB bytes.  Components are
    /// quantised to 8 bits.  `None` when `comps` does not hold the compiled
    /// component count or the transform fails.
    pub(super) fn convert(&self, comps: &[f64]) -> Option<[u8; 3]> {
        match self {
            Self::Shaper(shaper) => shaper.convert_components(comps),
            Self::Transform { n, xform } => {
                if comps.len() != usize::from(*n) {
                    return None;
                }
                let mut src = [0u8; 4];
                for (dst, &c) in src.iter_mut().zip(comps) {
                    *dst = color::convert::gray_to_u8(c);
                }
                let mut rgb = [0u8; 3];
                xform
                    .transform(&src[..comps.len()], &mut rgb)
                    .map_err(|e| log::debug!("colorspace: ICC transform execution failed: {e}"))
                    .ok()?;
                Some(rgb)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_stream;

    fn srgb_doc() -> Document {
        let bytes = ColorProfile::new_srgb()
            .encode()
            .expect("moxcms encodes its own sRGB profile");
        make_doc_with_stream(&bytes, " /N 3")
    }

    #[test]
    fn profile_stream_is_compiled_once_per_document() {
        let doc = srgb_doc();
        let first = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        let second = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn matrix_trc_profile_compiles_to_the_shaper() {
        let doc = srgb_doc();
        let conv = IccConversion::load(&doc, (2, 0), 3).expect("sRGB profile compiles");
        assert!(matches!(*conv, IccConversion::Shaper(_)));
        assert_eq!(conv.convert(&[1.0, 0.0, 0.0]), Some([255, 0, 0]));
        assert_eq!(conv.convert(&[1.0, 0.0]), None);
    }

    #[test]
    fn unusable_profiles_do_not_compile() {
        let bytes = ColorProfile::new_srgb().encode().expect("encodes");
        assert!(IccConversion::compile(&bytes, 1).is_none());
        assert!(IccConversion::compile(b"not a profile", 3).is_none());
    }
}
