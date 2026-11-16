//! `ExtGState` resource dictionaries (ISO 32000-1 §8.4.5): the graphics
//! state parameters a `gs` operator sets.

use pdf::{Dictionary, Document, Object, ObjectId};
use raster::types::BlendMode;

use super::dict_ext::DictExt;
use super::{black_generation, obj_to_f64, read_f64_1, resolve_dict, transfer};

/// Selected parameters extracted from a PDF `ExtGState` resource dictionary.
///
/// Only the subset needed by the current rasteriser is extracted; unknown keys
/// are silently ignored so that future additions do not break existing pages.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExtGStateParams {
    /// Non-stroking (fill) opacity `ca` in [0, 255].  `None` = unchanged.
    pub fill_alpha: Option<u8>,
    /// Stroking opacity `CA` in [0, 255].  `None` = unchanged.
    pub stroke_alpha: Option<u8>,
    /// Line width `LW` in user-space units.  `None` = unchanged.
    pub line_width: Option<f64>,
    /// Line cap style `LC` (0–2).  `None` = unchanged.
    pub line_cap: Option<i32>,
    /// Line join style `LJ` (0–2).  `None` = unchanged.
    pub line_join: Option<i32>,
    /// Miter limit `ML`.  `None` = unchanged.
    pub miter_limit: Option<f64>,
    /// Flatness `FL`.  `None` = unchanged.
    pub flatness: Option<f64>,
    /// Blend mode `BM`.  `None` = unchanged (keep current).
    pub blend_mode: Option<BlendMode>,
    /// Soft mask `SMask`.  `None` = unchanged.
    pub soft_mask: Option<SoftMaskParams>,
    /// Transfer function `TR2`, else `TR`.  `None` = unchanged.
    pub transfer: Option<transfer::TransferSetting>,
    /// Black-generation function `BG2`, else `BG`.  `None` = unchanged.
    pub(crate) black_generation: Option<black_generation::KSetting>,
    /// Undercolour-removal function `UCR2`, else `UCR`.  `None` =
    /// unchanged.
    pub(crate) undercolour_removal: Option<black_generation::KSetting>,
    /// Stroking overprint `OP`.  `None` = unchanged.
    pub overprint_stroke: Option<bool>,
    /// Non-stroking overprint `op`, which defaults to `OP` when absent.
    /// `None` = unchanged.
    pub overprint_fill: Option<bool>,
    /// Overprint mode `OPM`.  `None` = unchanged, also for a value other
    /// than 0 and 1, the only defined ones.
    pub overprint_mode: Option<OverprintMode>,
    /// Rendering intent `RI`.  `None` = unchanged.
    pub rendering_intent: Option<crate::RenderingIntent>,
    /// Black-point compensation `UseBlackPtComp` (ISO 32000-2 §8.4.5):
    /// `/ON` and `/Default` on, `/OFF` off.  `None` = unchanged, also for
    /// another value.
    pub black_point_compensation: Option<bool>,
}

/// The overprint mode `OPM` (ISO 32000-1 §8.6.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverprintMode {
    /// `OPM 0`: an overprinting colour paints every component of its space.
    #[default]
    Zero,
    /// `OPM 1`: an overprinting colour of a process CMYK space leaves the
    /// components it gives as exactly 0 unpainted.
    NonZero,
}

/// The `SMask` entry of an `ExtGState` dictionary (PDF §11.6.5.2).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SoftMaskParams {
    /// `/None`: no soft mask.
    None,
    /// A soft-mask dictionary.
    Group {
        /// `S`: `true` for `/Luminosity`, `false` for `/Alpha`.
        luminosity: bool,
        /// `G`: the transparency group `XObject` that defines the mask.
        group: ObjectId,
        /// `BC`: backdrop colour components in the group's colour space;
        /// empty means black.
        backdrop: Vec<f64>,
        /// `TR`: transfer function applied to the mask values; `None` is
        /// the identity.
        transfer: Option<Object>,
    },
}

impl ExtGStateParams {
    /// The parameters of the `ExtGState` dictionary `obj` refers to;
    /// `None` when it is not a dictionary.
    pub(super) fn from_object(doc: &Document, obj: &Object) -> Option<Self> {
        resolve_dict(doc, obj).map(|d| Self::from_dict(doc, &d))
    }

    fn from_dict(doc: &Document, d: &Dictionary) -> Self {
        let overprint_stroke = d.get_bool(b"OP");
        let (black_generation, undercolour_removal) = black_generation::ext_gstate_black(doc, d);
        Self {
            fill_alpha: real_to_u8(d, b"ca"),
            stroke_alpha: real_to_u8(d, b"CA"),
            line_width: read_f64_1(d, b"LW"),
            line_cap: int_val(d, b"LC"),
            line_join: int_val(d, b"LJ"),
            miter_limit: read_f64_1(d, b"ML"),
            flatness: read_f64_1(d, b"FL"),
            blend_mode: parse_blend_mode(d),
            soft_mask: parse_soft_mask(doc, d),
            transfer: transfer::ext_gstate_transfer(doc, d),
            black_generation,
            undercolour_removal,
            overprint_stroke,
            overprint_fill: d.get_bool(b"op").or(overprint_stroke),
            overprint_mode: d.get_i64(b"OPM").and_then(|m| match m {
                0 => Some(OverprintMode::Zero),
                1 => Some(OverprintMode::NonZero),
                _ => None,
            }),
            rendering_intent: d
                .get(b"RI")
                .and_then(Object::as_name)
                .map(crate::RenderingIntent::from_pdf_name),
            black_point_compensation: d.get(b"UseBlackPtComp").and_then(Object::as_name).and_then(
                |name| match name {
                    b"ON" | b"Default" => Some(true),
                    b"OFF" => Some(false),
                    _ => None,
                },
            ),
        }
    }
}

/// Parse the `SMask` key from an `ExtGState` dictionary.
///
/// A dictionary without a stream reference in `G`, or with an `S` other
/// than `Luminosity` / `Alpha`, is malformed and leaves the soft mask
/// unchanged.
fn parse_soft_mask(doc: &Document, d: &Dictionary) -> Option<SoftMaskParams> {
    let obj = d.get(b"SMask")?;
    if obj.as_name() == Some(b"None".as_slice()) {
        return Some(SoftMaskParams::None);
    }
    let sm = resolve_dict(doc, obj)?;
    let luminosity = match sm.get(b"S")?.as_name()? {
        b"Luminosity" => true,
        b"Alpha" => false,
        other => {
            log::warn!(
                "ExtGState: soft mask subtype {:?} unknown — ignoring the soft mask",
                String::from_utf8_lossy(other)
            );
            return None;
        }
    };
    let Object::Reference(group) = sm.get(b"G")? else {
        return None;
    };
    let backdrop = match sm.get(b"BC") {
        Some(Object::Array(a)) => a.iter().filter_map(obj_to_f64).collect(),
        _ => Vec::new(),
    };
    let transfer = sm
        .get(b"TR")
        .filter(|o| o.as_name() != Some(b"Identity".as_slice()))
        .cloned();
    Some(SoftMaskParams::Group {
        luminosity,
        group: *group,
        backdrop,
        transfer,
    })
}

/// Parse the `BM` key from an `ExtGState` dictionary.
///
/// `BM` is either a single `Name` or an array of names (priority list; the
/// viewer uses the first one it supports).  Unknown names are silently ignored
/// and `None` is returned, leaving the current blend mode unchanged.
pub(crate) fn parse_blend_mode(d: &Dictionary) -> Option<BlendMode> {
    let obj = d.get(b"BM")?;
    match obj {
        Object::Name(n) => bm_name_to_mode(n),
        // Array is a viewer-preference priority list; use the first recognised name.
        Object::Array(arr) => arr
            .iter()
            .find_map(|o| o.as_name().and_then(bm_name_to_mode)),
        _ => None,
    }
}

/// Map a PDF blend mode name to [`BlendMode`].
///
/// Returns `None` for unrecognised names so the caller can keep the current mode.
const fn bm_name_to_mode(name: &[u8]) -> Option<BlendMode> {
    Some(match name {
        b"Normal" | b"Compatible" => BlendMode::Normal,
        b"Multiply" => BlendMode::Multiply,
        b"Screen" => BlendMode::Screen,
        b"Overlay" => BlendMode::Overlay,
        b"Darken" => BlendMode::Darken,
        b"Lighten" => BlendMode::Lighten,
        b"ColorDodge" => BlendMode::ColorDodge,
        b"ColorBurn" => BlendMode::ColorBurn,
        b"HardLight" => BlendMode::HardLight,
        b"SoftLight" => BlendMode::SoftLight,
        b"Difference" => BlendMode::Difference,
        b"Exclusion" => BlendMode::Exclusion,
        b"Hue" => BlendMode::Hue,
        b"Saturation" => BlendMode::Saturation,
        b"Color" => BlendMode::Color,
        b"Luminosity" => BlendMode::Luminosity,
        _ => return None,
    })
}

/// Read a real-valued opacity key and convert to u8 in [0, 255].
fn real_to_u8(d: &Dictionary, key: &[u8]) -> Option<u8> {
    let v = read_f64_1(d, key)?.clamp(0.0, 1.0);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "v clamped to [0,1], scaled to [0,255]; round() fits in u8"
    )]
    Some((v * 255.0).round() as u8)
}

/// Read an integer key, saturated to the `i32` range so that an
/// out-of-range value stays out of range rather than wrapping into it.
fn int_val(d: &Dictionary, key: &[u8]) -> Option<i32> {
    let n = d.get(key)?.as_i64()?;
    Some(i32::try_from(n).unwrap_or(if n < 0 { i32::MIN } else { i32::MAX }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `ExtGState` parameters of the dictionary `body`.
    fn ext_gstate(body: &str) -> ExtGStateParams {
        let doc = crate::test_helpers::make_doc_with_object(body);
        let d = doc.get_dictionary((2, 0)).expect("dictionary");
        ExtGStateParams::from_dict(&doc, &d)
    }

    #[test]
    fn use_black_pt_comp_reads_on_off_and_default() {
        let bpc =
            |v: &str| ext_gstate(&format!("<< /UseBlackPtComp {v} >>")).black_point_compensation;
        assert_eq!(bpc("/ON"), Some(true));
        assert_eq!(bpc("/Default"), Some(true));
        assert_eq!(bpc("/OFF"), Some(false));
        assert_eq!(bpc("/Maybe"), None);
        assert_eq!(ext_gstate("<< >>").black_point_compensation, None);
    }

    #[test]
    fn op_defaults_to_op_stroke() {
        let p = ext_gstate("<< /OP true >>");
        assert_eq!(p.overprint_stroke, Some(true));
        assert_eq!(p.overprint_fill, Some(true));
    }

    #[test]
    fn op_false_overrides() {
        let p = ext_gstate("<< /OP true /op false >>");
        assert_eq!(p.overprint_stroke, Some(true));
        assert_eq!(p.overprint_fill, Some(false));
        let p = ext_gstate("<< /op true >>");
        assert_eq!(p.overprint_stroke, None);
        assert_eq!(p.overprint_fill, Some(true));
    }

    #[test]
    fn non_boolean_overprint_is_ignored() {
        let p = ext_gstate("<< /OP 1 /op (x) >>");
        assert_eq!(p.overprint_stroke, None);
        assert_eq!(p.overprint_fill, None);
    }

    #[test]
    fn opm_parsed() {
        let opm = |src| ext_gstate(src).overprint_mode;
        assert_eq!(opm("<< /OPM 1 >>"), Some(OverprintMode::NonZero));
        assert_eq!(opm("<< /OPM 0 >>"), Some(OverprintMode::Zero));
        assert_eq!(opm("<< /OPM 7 >>"), None);
        assert_eq!(opm("<< /OPM -1 >>"), None);
        assert_eq!(opm("<< >>"), None);
    }

    #[test]
    fn bm_name_all_variants_round_trip() {
        let pairs: &[(&[u8], BlendMode)] = &[
            (b"Normal", BlendMode::Normal),
            (b"Compatible", BlendMode::Normal),
            (b"Multiply", BlendMode::Multiply),
            (b"Screen", BlendMode::Screen),
            (b"Overlay", BlendMode::Overlay),
            (b"Darken", BlendMode::Darken),
            (b"Lighten", BlendMode::Lighten),
            (b"ColorDodge", BlendMode::ColorDodge),
            (b"ColorBurn", BlendMode::ColorBurn),
            (b"HardLight", BlendMode::HardLight),
            (b"SoftLight", BlendMode::SoftLight),
            (b"Difference", BlendMode::Difference),
            (b"Exclusion", BlendMode::Exclusion),
            (b"Hue", BlendMode::Hue),
            (b"Saturation", BlendMode::Saturation),
            (b"Color", BlendMode::Color),
            (b"Luminosity", BlendMode::Luminosity),
        ];
        for &(name, expected) in pairs {
            assert_eq!(
                bm_name_to_mode(name),
                Some(expected),
                "bm_name_to_mode({:?})",
                String::from_utf8_lossy(name)
            );
        }
    }

    #[test]
    fn bm_name_unknown_returns_none() {
        assert!(bm_name_to_mode(b"Dissolve").is_none());
        assert!(bm_name_to_mode(b"").is_none());
    }

    #[test]
    fn parse_blend_mode_bare_name() {
        let mut dict = Dictionary::new();
        dict.set("BM", Object::Name(b"Multiply".to_vec()));
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Multiply));
    }

    #[test]
    fn parse_blend_mode_array_picks_first_known() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Unknown1".to_vec()),
                Object::Name(b"Screen".to_vec()),
                Object::Name(b"Normal".to_vec()),
            ]),
        );
        assert_eq!(parse_blend_mode(&dict), Some(BlendMode::Screen));
    }

    #[test]
    fn parse_blend_mode_absent_returns_none() {
        let dict = Dictionary::new();
        assert!(parse_blend_mode(&dict).is_none());
    }

    #[test]
    fn parse_blend_mode_all_unknown_array_returns_none() {
        let mut dict = Dictionary::new();
        dict.set(
            "BM",
            Object::Array(vec![
                Object::Name(b"Foo".to_vec()),
                Object::Name(b"Bar".to_vec()),
            ]),
        );
        assert!(parse_blend_mode(&dict).is_none());
    }
}
