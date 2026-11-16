//! Black generation and undercolour removal (ISO 32000-2 §8.4.5, Table 58;
//! §10.4.2).
//!
//! A device without a colour profile converts an RGB colour to CMYK as
//! `c = 1 − r`, `m = 1 − g`, `y = 1 − b`, `k = min(c, m, y)`, then
//! `C = c − UCR(k)`, `M = m − UCR(k)`, `Y = y − UCR(k)` and `K = BG(k)`,
//! each clamped to `[0, 1]`.  `BG`, the black-generation function, maps
//! `k` into `[0, 1]`; `UCR`, the undercolour-removal function, maps it
//! into `[−1, 1]`.  A gray converts to `K = 1 − gray` alone: neither
//! function takes part.  The functions in effect at the start of a page
//! are the device's own, here `BG(k) = k` and `UCR(k) = k`, which remove
//! the whole gray component to black.
//!
//! An `ExtGState` sets them through `BG` and `UCR`, a function each, or
//! `BG2` and `UCR2`, a function or `/Default`, the function in effect at
//! the start of the page; a `…2` entry takes precedence over the other.
//! A function is tabulated at the 256 values of an 8-bit `k` when it is
//! read, and at the 65 536 values of a 16-bit one on first use.

use std::sync::{Arc, OnceLock};

use color::{Sample, SampleDepth};
use pdf::{Dictionary, Document, Object};

use super::shading::function::Function;

/// A black-generation or undercolour-removal function of `k`, tabulated.
#[derive(Debug)]
pub(crate) struct KFunction {
    /// The function.
    function: Arc<Function>,
    /// The lower bound of its output: 0 for `BG`, −1 for `UCR`.
    floor: f64,
    /// Its value at each 8-bit `k`.
    narrow: Box<[f32; 256]>,
    /// Its value at each 16-bit `k`, tabulated on first use.
    wide: OnceLock<Box<[f32]>>,
}

impl KFunction {
    /// `function` of outputs in `[floor, 1]`, tabulated at 8 bits; `None`
    /// when an evaluation fails.
    fn new(function: Arc<Function>, floor: f64) -> Option<Self> {
        let mut narrow = Box::new([0.0f32; 256]);
        for (v, out) in (0u16..).zip(narrow.iter_mut()) {
            *out = eval(&function, f64::from(v) / 255.0, floor)?;
        }
        Some(Self {
            function,
            floor,
            narrow,
            wide: OnceLock::new(),
        })
    }

    /// The function's value at `k`.
    fn at<S: Sample>(&self, k: S) -> f64 {
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "a sample's value is non-negative and below 65 536"
        )]
        let i = k.signed() as usize;
        let table: &[f32] = match S::DEPTH {
            SampleDepth::Eight => &self.narrow[..],
            SampleDepth::Sixteen => self.wide.get_or_init(|| {
                (0..=u16::MAX)
                    .map(|v| {
                        eval(&self.function, v.to_unit(), self.floor)
                            .unwrap_or_else(|| self.narrow[usize::from(v.to_u8())])
                    })
                    .collect()
            }),
        };
        f64::from(table[i])
    }
}

/// `function` at `t`, clamped to `[floor, 1]`; `None` when it fails.
fn eval(function: &Function, t: f64, floor: f64) -> Option<f32> {
    let y = *function.eval(t, 1)?.first()?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a value clamped to [-1, 1] is held in f32"
    )]
    Some(if y.is_nan() {
        floor
    } else {
        y.clamp(floor, 1.0)
    } as f32)
}

/// The black-generation and undercolour-removal functions of a graphics
/// state; `None` for the device's own.  Two are equal when they hold the
/// same tabulated functions.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeviceBlack {
    /// `BG`.
    bg: Option<Arc<KFunction>>,
    /// `UCR`.
    ucr: Option<Arc<KFunction>>,
}

impl PartialEq for DeviceBlack {
    fn eq(&self, other: &Self) -> bool {
        same(self.bg.as_ref(), other.bg.as_ref()) && same(self.ucr.as_ref(), other.ucr.as_ref())
    }
}

impl Eq for DeviceBlack {}

/// Whether `a` and `b` are the same function, or both the device's.
fn same(a: Option<&Arc<KFunction>>, b: Option<&Arc<KFunction>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (a, b) => a.is_none() && b.is_none(),
    }
}

impl DeviceBlack {
    /// Whether both functions are the device's own.
    pub(crate) const fn is_device(&self) -> bool {
        self.bg.is_none() && self.ucr.is_none()
    }

    /// `self` with the settings of an `ExtGState` applied: each given one
    /// replaces its function.
    #[must_use]
    pub(crate) fn with(&self, bg: Option<&KSetting>, ucr: Option<&KSetting>) -> Self {
        let pick = |setting: Option<&KSetting>, current: &Option<Arc<KFunction>>| match setting {
            None => current.clone(),
            Some(KSetting::Default) => None,
            Some(KSetting::Function(f)) => Some(Arc::clone(f)),
        };
        Self {
            bg: pick(bg, &self.bg),
            ucr: pick(ucr, &self.ucr),
        }
    }

    /// The device CMYK of the RGB colour `rgb` under the functions (see
    /// the module documentation), at the depth of `S`.  Under the device's
    /// own functions it is [`color::cmyk::rgb_to_device_cmyk`], exact in
    /// integers.
    #[must_use]
    pub(crate) fn separate<S: Sample>(&self, rgb: [S; 3]) -> [S; 4] {
        if self.is_device() {
            return color::cmyk::rgb_to_device_cmyk(rgb);
        }
        let [c, m, y] = rgb.map(|v| S::from_signed(S::MAX.signed() - v.signed()));
        let k = c.min(m).min(y);
        let ucr = self.ucr.as_ref().map_or_else(|| k.to_unit(), |f| f.at(k));
        let bg = self.bg.as_ref().map_or_else(|| k.to_unit(), |f| f.at(k));
        let [c, m, y] = [c, m, y].map(|v| S::from_unit(v.to_unit() - ucr));
        [c, m, y, S::from_unit(bg)]
    }
}

/// A `BG`/`BG2` or `UCR`/`UCR2` value: the device's function or one given.
#[derive(Debug, Clone)]
pub(crate) enum KSetting {
    /// `/Default`: the function in effect at the start of the page.
    Default,
    /// A function.
    Function(Arc<KFunction>),
}

/// The black-generation and undercolour-removal settings of the
/// `ExtGState` dictionary `d`: `BG2` when present, else `BG`; `UCR2` when
/// present, else `UCR`.  `None` for an absent or malformed entry, which
/// leaves the function unchanged.
pub(crate) fn ext_gstate_black(
    doc: &Document,
    d: &Dictionary,
) -> (Option<KSetting>, Option<KSetting>) {
    let read = |key2: &[u8], key: &[u8], floor: f64| {
        let obj = d.get(key2).or_else(|| d.get(key))?;
        let setting = setting_of(doc, obj, floor);
        if setting.is_none() {
            log::debug!(
                "ExtGState: {} {obj:?} unusable — left unchanged",
                String::from_utf8_lossy(key)
            );
        }
        setting
    };
    (read(b"BG2", b"BG", 0.0), read(b"UCR2", b"UCR", -1.0))
}

/// The setting of one entry: `/Default` (and `/Identity`, which is the
/// device's function here) or a function of outputs in `[floor, 1]`.
fn setting_of(doc: &Document, obj: &Object, floor: f64) -> Option<KSetting> {
    match obj.as_name() {
        Some(b"Default" | b"Identity") => Some(KSetting::Default),
        Some(_) => None,
        None => {
            KFunction::new(Function::load(doc, obj), floor).map(|f| KSetting::Function(Arc::new(f)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::empty_doc;

    fn gs(src: &str) -> Dictionary {
        let Object::Dictionary(d) = pdf::parse_object(src.as_bytes(), &mut 0).expect("parses")
        else {
            panic!("a dictionary")
        };
        d
    }

    /// The device black of the settings of `src`.
    fn black(src: &str) -> DeviceBlack {
        let (bg, ucr) = ext_gstate_black(&empty_doc(), &gs(src));
        DeviceBlack::default().with(bg.as_ref(), ucr.as_ref())
    }

    const ZERO: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [0] /N 1 >>";
    const HALF: &str = "<< /FunctionType 2 /Domain [0 1] /C0 [0] /C1 [0.5] /N 1 >>";

    #[test]
    fn the_device_functions_remove_the_whole_gray_component() {
        let device = DeviceBlack::default();
        assert!(device.is_device());
        for rgb in [[51u8, 102, 153], [0, 0, 0], [255, 255, 255], [10, 200, 90]] {
            assert_eq!(device.separate(rgb), color::cmyk::rgb_to_device_cmyk(rgb));
        }
        assert_eq!(black("<< /BG2 /Default /UCR2 /Default >>"), device);
    }

    /// `c = 0.8, m = 0.6, y = 0.4`, `k = 0.4`: without black generation
    /// or undercolour removal the colour is cyan, magenta and yellow alone.
    #[test]
    fn functions_apply_as_the_conversion_defines() {
        let none = black(&format!("<< /BG {ZERO} /UCR {ZERO} >>"));
        assert!(!none.is_device());
        assert_eq!(none.separate([51u8, 102, 153]), [204, 153, 102, 0]);
        assert_eq!(
            none.separate([51u16 * 257, 102 * 257, 153 * 257]),
            [204 * 257, 153 * 257, 102 * 257, 0]
        );
        // Half the black generated, half the undercolour removed.
        let half = black(&format!("<< /BG {HALF} /UCR {HALF} >>"));
        assert_eq!(half.separate([51u8, 102, 153]), [153, 102, 51, 51]);
        // An undercolour removal below 0 adds ink, clamped at full.
        let add = black("<< /UCR << /FunctionType 2 /Domain [0 1] /C0 [-1] /C1 [-1] /N 1 >> >>");
        assert_eq!(add.separate([51u8, 102, 153]), [255, 255, 255, 102]);
    }

    #[test]
    fn the_second_entries_take_precedence() {
        let d = black(&format!(
            "<< /BG {ZERO} /BG2 /Default /UCR2 {ZERO} /UCR /Default >>"
        ));
        // BG is the device's, UCR removes nothing.
        assert_eq!(d.separate([51u8, 102, 153]), [204, 153, 102, 102]);
        // A /Default resets a function set earlier; an absent entry keeps it.
        let (bg, ucr) = ext_gstate_black(&empty_doc(), &gs("<< /BG2 /Default >>"));
        assert!(ucr.is_none());
        let reset = d.with(bg.as_ref(), ucr.as_ref());
        assert_eq!(reset.separate([51u8, 102, 153]), [204, 153, 102, 102]);
        let malformed = black("<< /BG /Nonsense /UCR 3 >>");
        assert!(malformed.is_device());
    }
}
