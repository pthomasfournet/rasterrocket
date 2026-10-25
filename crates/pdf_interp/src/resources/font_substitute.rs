//! Typeface classification for non-embedded fonts.
//!
//! A font without an embedded program is drawn with an installed substitute.
//! [`SubstituteStyle`] records which substitute fits: the family class
//! (sans-serif, serif, monospaced) and the bold / italic variant.  The class
//! is read from the `BaseFont` name first, since it names the intended
//! typeface, and from the font descriptor's `Flags`, `FontWeight` and
//! `ItalicAngle` (PDF §9.8.2) where the name is silent.

use pdf::{Dictionary, Document, Object};

use crate::resources::dict_ext::DictExt as _;

/// Family class of a substitute typeface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubstituteFamily {
    /// Sans-serif (the class of Helvetica / Arial).
    #[default]
    Sans,
    /// Serif (the class of Times).
    Serif,
    /// Monospaced (the class of Courier).
    Mono,
}

/// Family class and variant of the substitute for a non-embedded font.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubstituteStyle {
    /// Family class.
    pub family: SubstituteFamily,
    /// Bold variant.
    pub bold: bool,
    /// Italic / oblique variant.
    pub italic: bool,
}

/// `FixedPitch` flag (bit 1), PDF §9.8.2 Table 121.
const FLAG_FIXED_PITCH: i64 = 1;
/// `Serif` flag (bit 2).
const FLAG_SERIF: i64 = 1 << 1;
/// `Italic` flag (bit 7).
const FLAG_ITALIC: i64 = 1 << 6;
/// `ForceBold` flag (bit 19).
const FLAG_FORCE_BOLD: i64 = 1 << 18;

/// Name fragments (lower case) that identify a monospaced typeface.
const MONO_NAMES: &[&str] = &["courier", "mono", "consol", "typewriter", "lucidaconsole"];
/// Name fragments that identify a sans-serif typeface.  Checked before
/// [`SERIF_NAMES`] so that e.g. `"MinionSans"` or `"DejaVuSerifSans"` style
/// names with an explicit `sans` stay sans-serif.
const SANS_NAMES: &[&str] = &[
    "sans",
    "arial",
    "helvetica",
    "verdana",
    "tahoma",
    "calibri",
    "carlito",
    "gothic",
    "futura",
    "frutiger",
    "univers",
    "myriad",
    "segoe",
    "trebuchet",
    "gill",
    "optima",
    "lucida",
];
/// Name fragments that identify a serif typeface.
const SERIF_NAMES: &[&str] = &[
    "times",
    "roman",
    "serif",
    "garamond",
    "georgia",
    "palatino",
    "bookman",
    "century",
    "minion",
    "cambria",
    "caladea",
    "baskerville",
    "caslon",
    "bodoni",
    "didot",
    "schoolbook",
    "goudy",
    "plantin",
    "sabon",
    "janson",
    "bembo",
    "utopia",
    "charter",
    "book",
    "antiqua",
    "cochin",
    "perpetua",
    "constantia",
];
/// Name fragments that mark a bold weight.
const BOLD_NAMES: &[&str] = &["bold", "black", "heavy", "demi", "semibold", "extrabold"];
/// Name fragments that mark an italic or oblique style.
const ITALIC_NAMES: &[&str] = &["italic", "oblique", "slanted", "kursiv"];

/// Classify the substitute for the font described by `font_dict`, whose
/// font descriptor lives in `fd_owner` (the font dictionary itself for a
/// simple font, the descendant `CIDFont` for a Type 0 font).
#[must_use]
pub fn substitute_style(
    doc: &Document,
    font_dict: &Dictionary,
    fd_owner: Option<&Dictionary>,
) -> SubstituteStyle {
    let name = font_dict
        .get_name(b"BaseFont")
        .map(|n| String::from_utf8_lossy(n).to_ascii_lowercase())
        .unwrap_or_default();
    // A subset tag (`ABCDEF+`) carries no style information.
    let name = name.split_once('+').map_or(name.as_str(), |(_, rest)| rest);

    let fd = fd_owner.and_then(|d| resolve_descriptor(doc, d));
    let flags = fd.as_ref().and_then(|d| d.get_i64(b"Flags")).unwrap_or(0);
    let weight = fd
        .as_ref()
        .and_then(|d| d.get(b"FontWeight"))
        .and_then(Object::as_f64)
        .unwrap_or(0.0);
    let italic_angle = fd
        .as_ref()
        .and_then(|d| d.get(b"ItalicAngle"))
        .and_then(Object::as_f64)
        .unwrap_or(0.0);

    classify(name, flags, weight, italic_angle)
}

/// Classification core: `name` is the lower-cased `BaseFont` without its
/// subset tag; `flags`, `weight` and `italic_angle` come from the font
/// descriptor (0 when absent).
fn classify(name: &str, flags: i64, weight: f64, italic_angle: f64) -> SubstituteStyle {
    let has = |frags: &[&str]| frags.iter().any(|f| name.contains(f));
    let family = if has(MONO_NAMES) {
        SubstituteFamily::Mono
    } else if has(SANS_NAMES) {
        SubstituteFamily::Sans
    } else if has(SERIF_NAMES) {
        SubstituteFamily::Serif
    } else if flags & FLAG_FIXED_PITCH != 0 {
        SubstituteFamily::Mono
    } else if flags & FLAG_SERIF != 0 {
        SubstituteFamily::Serif
    } else {
        SubstituteFamily::Sans
    };
    SubstituteStyle {
        family,
        bold: has(BOLD_NAMES) || flags & FLAG_FORCE_BOLD != 0 || weight >= 600.0,
        italic: has(ITALIC_NAMES) || flags & FLAG_ITALIC != 0 || italic_angle.abs() > 0.5,
    }
}

/// Dereference the `FontDescriptor` entry of `dict`.
fn resolve_descriptor(doc: &Document, dict: &Dictionary) -> Option<Dictionary> {
    match dict.get(b"FontDescriptor")? {
        Object::Reference(id) => doc.get_dictionary(*id).ok().map(|d| (*d).clone()),
        Object::Dictionary(d) => Some(d.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(family: SubstituteFamily, bold: bool, italic: bool) -> SubstituteStyle {
        SubstituteStyle {
            family,
            bold,
            italic,
        }
    }

    #[test]
    fn standard_fourteen_names_classify() {
        use SubstituteFamily::{Mono, Sans, Serif};
        let cases = [
            ("helvetica", style(Sans, false, false)),
            ("helvetica-boldoblique", style(Sans, true, true)),
            ("times-roman", style(Serif, false, false)),
            ("times-bolditalic", style(Serif, true, true)),
            ("courier-bold", style(Mono, true, false)),
            ("courier-oblique", style(Mono, false, true)),
        ];
        for (name, want) in cases {
            assert_eq!(classify(name, 0, 0.0, 0.0), want, "{name}");
        }
    }

    #[test]
    fn producer_names_classify() {
        use SubstituteFamily::{Sans, Serif};
        assert_eq!(
            classify("timesnewromanps-boldmt", 0, 0.0, 0.0),
            style(Serif, true, false)
        );
        assert_eq!(classify("arialmt", 0, 0.0, 0.0), style(Sans, false, false));
        assert_eq!(
            classify("arial,bolditalic", 0, 0.0, 0.0),
            style(Sans, true, true)
        );
        // `Sans` in the name wins over a serif-family fragment.
        assert_eq!(
            classify("notosans-regular", 0, 0.0, 0.0),
            style(Sans, false, false)
        );
    }

    #[test]
    fn descriptor_fields_decide_when_the_name_is_silent() {
        use SubstituteFamily::{Mono, Sans, Serif};
        assert_eq!(
            classify("xyz", FLAG_SERIF, 0.0, 0.0),
            style(Serif, false, false)
        );
        assert_eq!(
            classify("xyz", FLAG_FIXED_PITCH, 700.0, 0.0),
            style(Mono, true, false)
        );
        assert_eq!(classify("xyz", 0, 0.0, -12.0), style(Sans, false, true));
        assert_eq!(
            classify("xyz", FLAG_FORCE_BOLD | FLAG_ITALIC, 0.0, 0.0),
            style(Sans, true, true)
        );
    }

    #[test]
    fn subset_tag_is_ignored() {
        let doc = crate::test_helpers::empty_doc();
        let mut dict = Dictionary::new();
        dict.set("BaseFont", Object::Name(b"ABCDEF+Courier".to_vec()));
        assert_eq!(
            substitute_style(&doc, &dict, None).family,
            SubstituteFamily::Mono
        );
    }
}
