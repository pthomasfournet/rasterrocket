//! Transfer functions (PDF §10.5).
//!
//! A transfer function maps each device colour component in `[0, 1]` to a
//! new value in `[0, 1]`.  Components are 8-bit on the page bitmap, so a
//! function is tabulated once into a 256-entry table and applied by lookup.

use std::sync::Arc;

use pdf::{Dictionary, Document, Object};

use super::shading::function::Function;

/// The identity table `v → v`.
const IDENTITY: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        #[expect(clippy::cast_possible_truncation, reason = "i < 256, so it fits in u8")]
        {
            t[i] = i as u8;
        }
        i += 1;
    }
    t
};

/// Tabulate the one-input, one-output function `f` over the 256 component
/// values.  `None` when `f` does not compile or an evaluation fails.
#[must_use]
pub fn function_lut(doc: &Document, f: &Object) -> Option<[u8; 256]> {
    let function = Function::load(doc, f);
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let t = f64::from(u8::try_from(i).ok()?) / 255.0;
        let y = function.eval(t, 1)?.first().copied()?;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to [0, 1] and scaled to [0, 255]"
        )]
        let b = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        *v = b;
    }
    Some(lut)
}

/// Transfer tables for the red, green and blue components of the page
/// bitmap, at least one of which differs from the identity.
#[derive(Debug, PartialEq, Eq)]
pub struct Transfer {
    rgb: [[u8; 256]; 3],
}

impl Transfer {
    /// The red, green and blue tables.
    #[must_use]
    pub const fn tables(&self) -> [&[u8; 256]; 3] {
        let [r, g, b] = &self.rgb;
        [r, g, b]
    }

    /// Map an RGB colour through the tables.
    #[must_use]
    pub fn apply(&self, rgb: [u8; 3]) -> [u8; 3] {
        let [r, g, b] = &self.rgb;
        [
            r[usize::from(rgb[0])],
            g[usize::from(rgb[1])],
            b[usize::from(rgb[2])],
        ]
    }
}

/// The transfer function an `ExtGState` installs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSetting {
    /// Components pass unchanged: `/Identity`, `/Default`, or functions
    /// that tabulate to the identity.
    Identity,
    /// Components map through the tables.
    Functions(Arc<Transfer>),
}

/// The transfer function of `ExtGState` dictionary `d`: `TR2` when present,
/// otherwise `TR` (PDF §8.4.5, Table 57).
///
/// The value is a function applied to every component, an array of four
/// functions of which the first three apply to red, green and blue, or the
/// name `/Identity` (or `/Default`, which for this device is the identity).
/// Array entries may also be `/Identity`.  `None` when neither key is
/// present or the value is malformed, which leaves the transfer function
/// unchanged.
#[must_use]
pub fn ext_gstate_transfer(doc: &Document, d: &Dictionary) -> Option<TransferSetting> {
    let obj = d.get(b"TR2").or_else(|| d.get(b"TR"))?;
    let setting = match obj {
        Object::Reference(id) => {
            struct Cached(Option<TransferSetting>);
            doc.derived(*id, || {
                let target = doc.get_object(*id).ok()?;
                let target = target.as_ref();
                let setting = if matches!(target, Object::Array(_) | Object::Name(_)) {
                    setting_of(doc, target)
                } else {
                    setting_of(doc, obj)
                };
                Some(Cached(setting))
            })
            .and_then(|c| c.0.clone())
        }
        _ => setting_of(doc, obj),
    };
    if setting.is_none() {
        log::debug!("ExtGState: transfer function {obj:?} unusable — left unchanged");
    }
    setting
}

/// The setting of a direct `TR`/`TR2` value, or of a function given by
/// reference.
fn setting_of(doc: &Document, obj: &Object) -> Option<TransferSetting> {
    let rgb = if let Object::Array(fns) = obj {
        let [r, g, b, _gray] = fns.as_slice() else {
            return None;
        };
        [
            component_lut(doc, r)?,
            component_lut(doc, g)?,
            component_lut(doc, b)?,
        ]
    } else {
        [component_lut(doc, obj)?; 3]
    };
    Some(if rgb.iter().all(|lut| *lut == IDENTITY) {
        TransferSetting::Identity
    } else {
        TransferSetting::Functions(Arc::new(Transfer { rgb }))
    })
}

/// The table of one transfer-function entry: a function, `/Identity` or
/// `/Default`.
fn component_lut(doc: &Document, obj: &Object) -> Option<[u8; 256]> {
    match obj.as_name() {
        Some(b"Identity" | b"Default") => Some(IDENTITY),
        Some(_) => None,
        None => function_lut(doc, obj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_doc_with_objects;

    const INVERT: &str = "{ 1 exch sub }";

    /// Object 4: an inverting Type 4 function.
    fn invert_doc() -> Document {
        make_doc_with_objects(&[(
            4,
            format!(
                "<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length {} >>\n\
                 stream\n{INVERT}\nendstream",
                INVERT.len()
            ),
        )])
    }

    fn gs(src: &str) -> Dictionary {
        let obj = pdf::parse_object(src.as_bytes(), &mut 0).expect("dictionary parses");
        obj.as_dict().expect("a dictionary").clone()
    }

    fn inverted(setting: Option<TransferSetting>) -> [u8; 3] {
        let Some(TransferSetting::Functions(t)) = setting else {
            panic!("expected functions, got {setting:?}");
        };
        t.apply([0, 51, 255])
    }

    #[test]
    fn a_single_function_maps_every_component() {
        let doc = invert_doc();
        assert_eq!(
            inverted(ext_gstate_transfer(&doc, &gs("<< /TR 4 0 R >>"))),
            [255, 204, 0]
        );
    }

    #[test]
    fn a_four_function_array_maps_red_green_blue() {
        let doc = invert_doc();
        let setting = ext_gstate_transfer(&doc, &gs("<< /TR [4 0 R /Identity 4 0 R 4 0 R] >>"));
        let Some(TransferSetting::Functions(t)) = setting else {
            panic!("expected functions, got {setting:?}");
        };
        assert_eq!(t.apply([0, 51, 255]), [255, 51, 0]);
    }

    #[test]
    fn tr2_takes_precedence_over_tr() {
        let doc = invert_doc();
        assert_eq!(
            ext_gstate_transfer(&doc, &gs("<< /TR 4 0 R /TR2 /Default >>")),
            Some(TransferSetting::Identity)
        );
        assert_eq!(
            inverted(ext_gstate_transfer(
                &doc,
                &gs("<< /TR /Identity /TR2 4 0 R >>")
            )),
            [255, 204, 0]
        );
    }

    #[test]
    fn identity_functions_and_malformed_values() {
        let doc = invert_doc();
        let identity = "<< /TR << /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [1] /N 1 >> >>";
        assert_eq!(
            ext_gstate_transfer(&doc, &gs(identity)),
            Some(TransferSetting::Identity)
        );
        assert_eq!(ext_gstate_transfer(&doc, &gs("<< >>")), None);
        assert_eq!(ext_gstate_transfer(&doc, &gs("<< /TR /Unknown >>")), None);
        assert_eq!(
            ext_gstate_transfer(&doc, &gs("<< /TR [4 0 R 4 0 R] >>")),
            None
        );
    }
}
