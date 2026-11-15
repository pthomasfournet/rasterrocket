//! Installed faces chosen by the name of a font without a program.
//!
//! A [`FaceName`] matches an installed face at one of three tiers, the
//! lowest winning: the face's PostScript name equals the font's name
//! without its subset tag; the face's family key ([`family_key`] of its
//! typographic family name) equals the name's family key; or, when the
//! name runs style words onto its family (`CarlitoBold`), it equals the
//! name's stem.  Among the faces of the lowest tier, a visible face ranks
//! before a hidden one, then the face of the nearest weight to the
//! variant asked for (700 bold, else 400), then a face of the slant asked
//! for, then the earlier by path: the choice depends on the installed
//! files alone.

use std::sync::{Arc, PoisonError};

use font::program::FontProgram;

use super::{Face, FontLibrary};
use crate::resources::font_substitute::{FaceName, SubstituteStyle, family_key};

/// One name request: the name and the bold and italic variant asked for.
pub(super) type Request = (FaceName, bool, bool);

/// The ranking key of a face matching a name: match tier, hidden, weight
/// distance, slant mismatch, position in the face list.
type NameRank = (u8, bool, u16, bool, usize);

/// The tier at which `face` matches `name`, `None` when it does not.
fn tier(face: &Face, name: &FaceName) -> Option<u8> {
    if face.postscript.as_deref() == Some(name.postscript.as_str()) {
        return Some(0);
    }
    let key = family_key(&face.family);
    if key == name.family {
        Some(1)
    } else if name.stem != name.family && key == name.stem {
        Some(2)
    } else {
        None
    }
}

/// The index of the face of `faces` that best matches `name` in the
/// variant `bold` / `italic`; see the module documentation.
fn best_match(faces: &[Face], name: &FaceName, bold: bool, italic: bool) -> Option<usize> {
    let want_weight: u16 = if bold { 700 } else { 400 };
    faces
        .iter()
        .enumerate()
        .filter_map(|(i, face)| {
            let rank: NameRank = (
                tier(face, name)?,
                face.hidden(),
                face.weight_class.abs_diff(want_weight),
                face.italic != italic,
                i,
            );
            Some(rank)
        })
        .min()
        .map(|(.., i)| i)
}

impl FontLibrary {
    /// The installed face `name` names, in the variant of `style`, parsed;
    /// `None` when no installed face matches or the matching one does not
    /// parse.  The face a request resolves to is chosen once per library
    /// and logged.
    pub(crate) fn named_program(
        &self,
        name: &FaceName,
        style: SubstituteStyle,
    ) -> Option<Arc<FontProgram>> {
        let request: Request = (name.clone(), style.bold, style.italic);
        let cached = self
            .named
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&request)
            .copied();
        let index = cached.unwrap_or_else(|| {
            let faces = self.faces();
            let index = best_match(faces, name, style.bold, style.italic);
            match index.map(|i| &faces[i]) {
                Some(face) => log::info!(
                    "system_fonts: font {} drawn with the installed face {} ({}#{})",
                    name.postscript,
                    face.postscript.as_deref().unwrap_or(&face.family),
                    face.path.display(),
                    face.index
                ),
                None => log::debug!(
                    "system_fonts: no installed face named {}; drawing a substitute",
                    name.postscript
                ),
            }
            let _ = self
                .named
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(request, index);
            index
        })?;
        self.face_program(index)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use font::sfnt_info::FaceInfo;

    use super::*;

    fn face(family: &str, postscript: &str, weight: u16, italic: bool) -> Face {
        Face::new(
            Path::new("/nonexistent"),
            FaceInfo {
                index: 0,
                family: family.to_owned(),
                postscript: Some(postscript.to_owned()),
                code_page_ranges: [0, 0],
                weight_class: weight,
                italic,
                family_class: 0,
            },
        )
    }

    fn name(n: &str) -> FaceName {
        let mut dict = pdf::Dictionary::new();
        dict.set(b"BaseFont", pdf::Object::Name(n.as_bytes().to_vec()));
        FaceName::of(&dict, None).expect("a family")
    }

    fn faces() -> Vec<Face> {
        vec![
            face("Carlito", "Carlito-Bold", 700, false),
            face("Carlito", "Carlito-Italic", 400, true),
            face("Carlito", "Carlito-Regular", 400, false),
            face("Myriad Pro", "MyriadPro-Regular", 400, false),
            face("Calibri", "Calibri-Light", 300, false),
            face("Calibri", "Calibri", 400, false),
        ]
    }

    /// The family and variant select the face, whatever the name's style
    /// suffix, subset tag, case or separators.
    #[test]
    fn a_family_name_selects_the_face_of_the_variant() {
        let faces = faces();
        let pick = |n: &str, bold, italic| best_match(&faces, &name(n), bold, italic);
        assert_eq!(pick("ABCDEF+Carlito", false, false), Some(2));
        assert_eq!(pick("Carlito,Bold", true, false), Some(0));
        assert_eq!(pick("CARLITO-Italic", false, true), Some(1));
        assert_eq!(pick("CarlitoBold", true, false), Some(0), "the stem");
        assert_eq!(pick("Myriad Pro", false, false), Some(3));
        assert_eq!(
            pick("Myriad_Pro-Bold", true, false),
            Some(3),
            "the nearest weight"
        );
        assert_eq!(pick("Cambria", false, false), None);
    }

    /// A PostScript name names its face exactly, before the family.
    #[test]
    fn a_postscript_name_selects_its_face() {
        let faces = faces();
        assert_eq!(
            best_match(&faces, &name("Calibri-Light"), false, false),
            Some(4)
        );
        assert_eq!(best_match(&faces, &name("Calibri"), false, false), Some(5));
    }

    /// A library over a directory holding a face draws a font named by the
    /// face's family with it, whatever the file is called.
    #[test]
    fn an_installed_face_is_found_by_its_name() {
        let dir: PathBuf =
            std::env::temp_dir().join(format!("rr-named-face-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let ahem = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../font_test_data/test_data/ttf/ahem.ttf"
        );
        let _bytes = std::fs::copy(ahem, dir.join("renamed.ttf")).expect("install");
        let library = FontLibrary::shared(std::slice::from_ref(&dir), false);
        let style = SubstituteStyle::default();
        let found = library.named_program(&name("XYZABC+Ahem,Bold"), style);
        let missing = library.named_program(&name("Aheme"), style);
        let _ = std::fs::remove_dir_all(&dir);
        let found = found.expect("the installed face");
        assert_eq!(found.data(), std::fs::read(ahem).expect("ahem").as_slice());
        assert!(missing.is_none());
    }
}
