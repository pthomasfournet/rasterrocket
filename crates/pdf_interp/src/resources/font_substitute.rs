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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SubstituteFamily {
    /// Sans-serif (the class of Helvetica / Arial).
    #[default]
    Sans,
    /// Serif (the class of Times).
    Serif,
    /// Monospaced (the class of Courier).
    Mono,
    /// The standard Symbol font (Greek letters, mathematical and
    /// arrow symbols).
    Symbol,
    /// The standard `ZapfDingbats` font.
    Dingbats,
}

/// A writing system the Latin substitutes do not cover, drawn from an
/// installed font of that script.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    /// Adobe-Japan1.
    Japanese,
    /// Adobe-GB1.
    SimplifiedChinese,
    /// Adobe-CNS1.
    TraditionalChinese,
    /// Adobe-Korea1 / Adobe-KR.
    Korean,
    /// Arabic.
    Arabic,
    /// Thai.
    Thai,
}

impl Script {
    /// The script of a `CIDSystemInfo` `Ordering`; `None` for `Identity`
    /// and any other ordering.
    #[must_use]
    pub const fn from_ordering(ordering: &[u8]) -> Option<Self> {
        match ordering {
            b"Japan1" => Some(Self::Japanese),
            b"GB1" => Some(Self::SimplifiedChinese),
            b"CNS1" => Some(Self::TraditionalChinese),
            b"Korea1" | b"KR" => Some(Self::Korean),
            _ => None,
        }
    }

    /// The script a Unicode scalar value belongs to, for the scripts
    /// whose block identifies them (unified Han ideographs do not).
    #[must_use]
    pub const fn of_char(c: u32) -> Option<Self> {
        match c {
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
                Some(Self::Arabic)
            }
            0x0E00..=0x0E7F => Some(Self::Thai),
            0x3040..=0x30FF | 0x31F0..=0x31FF => Some(Self::Japanese),
            0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => Some(Self::Korean),
            _ => None,
        }
    }
}

/// Family class and variant of the substitute for a non-embedded font.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SubstituteStyle {
    /// Family class.
    pub family: SubstituteFamily,
    /// Bold variant.
    pub bold: bool,
    /// Italic / oblique variant.
    pub italic: bool,
    /// Script the font is for, when it is one the Latin substitutes do
    /// not cover.
    pub script: Option<Script>,
    /// The font is one of the standard 14 or a metric-compatible alias
    /// (Arial, Times New Roman, Courier New): the substitute's advances
    /// are the font's own, so its glyphs are drawn unfitted.
    pub standard: bool,
}

/// `FixedPitch` flag (bit 1), PDF §9.8.2 Table 121.
const FLAG_FIXED_PITCH: i64 = 1;
/// `Serif` flag (bit 2).
const FLAG_SERIF: i64 = 1 << 1;
/// `Italic` flag (bit 7).
const FLAG_ITALIC: i64 = 1 << 6;
/// `ForceBold` flag (bit 19).
const FLAG_FORCE_BOLD: i64 = 1 << 18;

/// Family names (lower case, separators removed, style suffix cut at the
/// first `,` or `-`) of the standard 14 fonts and their metric-compatible
/// aliases.
const STANDARD_NAMES: &[&str] = &[
    "helvetica",
    "arial",
    "arialmt",
    "times",
    "timesroman",
    "timesnewroman",
    "timesnewromanps",
    "timesnewromanpsmt",
    "courier",
    "couriernew",
    "couriernewps",
    "couriernewpsmt",
    "symbol",
    "zapfdingbats",
];

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
/// Names (lower case, style suffix after `,` removed) of the standard
/// Symbol font and its TrueType namesake.
const SYMBOL_NAMES: &[&str] = &["symbol", "symbolmt"];
/// Names of the standard `ZapfDingbats` font and its common aliases.
const DINGBATS_NAMES: &[&str] = &[
    "zapfdingbats",
    "dingbats",
    "zapfdingbatsitc",
    "itczapfdingbats",
];
/// Name fragments that identify a typeface of each [`Script`].
const SCRIPT_NAMES: &[(Script, &[&str])] = &[
    (
        Script::Japanese,
        &[
            "mincho", "gothic", "meiryo", "hiragino", "kozuka", "ipaex", "ipag", "ipam", "yugoth",
            "yumin",
        ],
    ),
    (
        Script::SimplifiedChinese,
        &[
            "simsun", "simhei", "simkai", "fangsong", "kaiti", "stsong", "stheiti", "yahei",
            "songti", "heiti",
        ],
    ),
    (
        Script::TraditionalChinese,
        &["mingliu", "pmingliu", "dfkai", "jhenghei", "kaiu", "ming"],
    ),
    (
        Script::Korean,
        &[
            "batang", "gulim", "dotum", "gungsuh", "malgun", "nanum", "myeongjo",
        ],
    ),
    (Script::Arabic, &["arabic", "naskh", "kufi", "arab"]),
    (
        Script::Thai,
        &[
            "thai",
            "angsana",
            "cordia",
            "browallia",
            "garuda",
            "norasi",
            "kinnari",
            "sawasdee",
            "tlwg",
        ],
    ),
];

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

    let fd = fd_owner.and_then(|d| super::font::resolve_fd(doc, d));
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
    // The family name proper: the part before a `,Bold`-style suffix,
    // without the separators producers insert (`Zapf-Dingbats`, `Symbol MT`).
    let base: String = name
        .split(',')
        .next()
        .unwrap_or_default()
        .chars()
        .filter(|c| !matches!(c, '-' | ' ' | '_'))
        .collect();
    let standard_base: String = name
        .split([',', '-'])
        .next()
        .unwrap_or_default()
        .chars()
        .filter(|c| !matches!(c, ' ' | '_'))
        .collect();
    let standard = STANDARD_NAMES.contains(&standard_base.as_str());
    let family = if SYMBOL_NAMES.contains(&base.as_str()) {
        SubstituteFamily::Symbol
    } else if DINGBATS_NAMES.contains(&base.as_str()) {
        SubstituteFamily::Dingbats
    } else if has(MONO_NAMES) {
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
        script: SCRIPT_NAMES
            .iter()
            .find(|(_, frags)| has(frags))
            .map(|&(script, _)| script),
        standard,
    }
}

/// The script of the `Ordering` in the `CIDSystemInfo` of `cid_font`
/// (`None` for `Identity` or an absent entry).
#[must_use]
pub fn ordering_script(doc: &Document, cid_font: &Dictionary) -> Option<Script> {
    let info = super::resolve_dict(doc, cid_font.get(b"CIDSystemInfo")?)?;
    Script::from_ordering(info.get(b"Ordering")?.as_str()?)
}

/// The script most of the glyph names in `differences` belong to.
///
/// `None` unless that script outnumbers the Latin content: the Latin
/// names among them plus the `base_codes` the font leaves to its base
/// encoding.  Names outside the identifying blocks and outside Latin do
/// not count.
#[must_use]
pub fn differences_script(differences: &[Option<Box<str>>], base_codes: usize) -> Option<Script> {
    let mut counts: Vec<(Script, usize)> = Vec::new();
    let mut latin = base_codes;
    for unicode in differences
        .iter()
        .flatten()
        .filter_map(|name| font::glyph_name_to_unicode(name))
    {
        match Script::of_char(unicode) {
            Some(script) => match counts.iter_mut().find(|(s, _)| *s == script) {
                Some((_, n)) => *n += 1,
                None => counts.push((script, 1)),
            },
            None if matches!(unicode, 0x41..=0x24F) => latin += 1,
            None => {}
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, n)| n)
        .filter(|&(_, n)| n > latin)
        .map(|(script, _)| script)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style(family: SubstituteFamily, bold: bool, italic: bool) -> SubstituteStyle {
        SubstituteStyle {
            family,
            bold,
            italic,
            script: None,
            standard: false,
        }
    }

    /// The standard families and their aliases are recognised whatever
    /// style suffix follows; a narrow cut of one is not metric-compatible.
    #[test]
    fn standard_families_are_recognised_by_name() {
        for name in [
            "helvetica",
            "helvetica-bold",
            "arial,bold",
            "arial-boldmt",
            "arialmt",
            "times-roman",
            "timesnewromanps-boldmt",
            "times new roman",
            "courier-boldoblique",
            "couriernewpsmt",
            "symbol",
        ] {
            assert!(classify(name, 0, 0.0, 0.0).standard, "{name}");
        }
        for name in [
            "arialnarrow",
            "arial narrow",
            "calibri",
            "minion-regular",
            "carlito",
        ] {
            assert!(!classify(name, 0, 0.0, 0.0).standard, "{name}");
        }
    }

    #[test]
    fn script_names_and_orderings_classify() {
        use Script::{Arabic, Japanese, Korean, SimplifiedChinese, Thai, TraditionalChinese};
        assert_eq!(classify("ms-mincho", 0, 0.0, 0.0).script, Some(Japanese));
        assert_eq!(
            classify("ms-gothic", 0, 0.0, 0.0),
            SubstituteStyle {
                family: SubstituteFamily::Sans,
                script: Some(Japanese),
                ..SubstituteStyle::default()
            }
        );
        assert_eq!(
            classify("simsun", 0, 0.0, 0.0).script,
            Some(SimplifiedChinese)
        );
        assert_eq!(
            classify("pmingliu", 0, 0.0, 0.0).script,
            Some(TraditionalChinese)
        );
        assert_eq!(classify("batang", 0, 0.0, 0.0).script, Some(Korean));
        assert_eq!(
            classify("notonaskharabic", 0, 0.0, 0.0).script,
            Some(Arabic)
        );
        assert_eq!(classify("angsananew", 0, 0.0, 0.0).script, Some(Thai));
        assert_eq!(classify("arial", 0, 0.0, 0.0).script, None);
        assert_eq!(Script::from_ordering(b"Japan1"), Some(Japanese));
        assert_eq!(Script::from_ordering(b"KR"), Some(Korean));
        assert_eq!(Script::from_ordering(b"Identity"), None);
    }

    #[test]
    fn differences_vote_for_a_script() {
        let diffs: Vec<Option<Box<str>>> = vec![
            Some("A".into()),
            Some("uni0E01".into()),
            Some("uni0E02".into()),
            Some("afii57415".into()),
            None,
        ];
        assert_eq!(differences_script(&diffs, 0), Some(Script::Thai));
        assert_eq!(differences_script(&[Some("A".into()), None], 0), None);
        assert_eq!(differences_script(&[], 0), None);
        // A Latin font with a stray Thai name stays Latin, whether the
        // Latin content is named or left to the base encoding.
        let mostly_latin: Vec<Option<Box<str>>> = vec![
            Some("eacute".into()),
            Some("uni0E01".into()),
            Some("egrave".into()),
        ];
        assert_eq!(differences_script(&mostly_latin, 0), None);
        assert_eq!(differences_script(&[Some("uni0E01".into())], 90), None);
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
            let want = SubstituteStyle {
                standard: true,
                ..want
            };
            assert_eq!(classify(name, 0, 0.0, 0.0), want, "{name}");
        }
    }

    #[test]
    fn producer_names_classify() {
        use SubstituteFamily::{Sans, Serif};
        let standard = |want: SubstituteStyle| SubstituteStyle {
            standard: true,
            ..want
        };
        assert_eq!(
            classify("timesnewromanps-boldmt", 0, 0.0, 0.0),
            standard(style(Serif, true, false))
        );
        assert_eq!(
            classify("arialmt", 0, 0.0, 0.0),
            standard(style(Sans, false, false))
        );
        assert_eq!(
            classify("arial,bolditalic", 0, 0.0, 0.0),
            standard(style(Sans, true, true))
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
    fn symbol_and_dingbats_names_classify() {
        use SubstituteFamily::{Dingbats, Sans, Symbol};
        for name in ["symbol", "symbolmt", "symbol,bold", "symbol mt"] {
            assert_eq!(classify(name, 0, 0.0, 0.0).family, Symbol, "{name}");
        }
        for name in [
            "zapfdingbats",
            "dingbats",
            "zapf-dingbats",
            "zapfdingbatsitc",
        ] {
            assert_eq!(classify(name, 0, 0.0, 0.0).family, Dingbats, "{name}");
        }
        // Only the standard fonts themselves, not every name containing
        // the word.
        assert_eq!(classify("symbolneu", 0, 0.0, 0.0).family, Sans);
        assert_eq!(classify("wingdings", 0, 0.0, 0.0).family, Sans);
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
