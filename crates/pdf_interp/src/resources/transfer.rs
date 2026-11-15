//! Transfer functions (PDF §10.5).
//!
//! A transfer function maps each device colour component in `[0, 1]` to a
//! new value in `[0, 1]`.  On an 8-bit bitmap a function is tabulated once
//! into a 256-entry table and applied by lookup.  On a 16-bit bitmap the
//! function itself maps a colour, evaluated in `f64` at the 16-bit value,
//! and the pipe's tables hold it at all 65 536 values, evaluated the same
//! way on first use.
//!
//! Transfer functions are defined on additive values (§10.5): on an ink
//! canvas, whose channels are amounts of a subtractive colorant, a function
//! maps the complement of the ink and the result is complemented back.  The
//! red, green and blue functions of a four-function array apply to cyan,
//! magenta and yellow, and the gray function to black and to the spot
//! colorants.

use std::sync::{Arc, OnceLock};

use color::Sample;
use pdf::{Dictionary, Document, Object};
use raster::pipe::ChannelMask;

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

/// Tabulate `function`, of one input and one output, at every value of
/// the sample type of `table`: entry `v` is `f(v / MAX)` clamped to
/// `[0, 1]` and rounded.  `None` when an evaluation fails.
pub(crate) fn tabulate<S: Sample>(function: &Function, table: &mut [S]) -> Option<()> {
    for (v, out) in (0u32..).zip(table.iter_mut()) {
        *out = evaluate::<S>(
            function,
            S::from_wide(<S::Wide as color::Wide>::of(v)).to_unit(),
        )?;
    }
    Some(())
}

/// `f(t)` of the one-input, one-output `function`, clamped to `[0, 1]`
/// and rounded to a sample; `None` when the evaluation fails.
fn evaluate<S: Sample>(function: &Function, t: f64) -> Option<S> {
    function
        .eval(t, 1)?
        .first()
        .copied()
        .map(|y| S::from_unit(y))
}

/// Transfer tables for the red, green and blue components of an RGB
/// canvas and for the eight ink channels of an ink canvas, at least one of
/// the four functions differing from the identity.
#[derive(Debug)]
#[non_exhaustive]
pub struct Transfer {
    rgb: [[u8; 256]; 3],
    /// Ink channel `i`'s table: `255 − f(255 − v)` for the function `f`
    /// of the channel's colorant.
    ink: [[u8; 256]; 8],
    /// The red, green, blue and gray functions; `None` for the identity.
    functions: [Option<Arc<Function>>; 4],
    /// The red, green and blue functions at every 16-bit value, one table
    /// after the other, tabulated on first use; an identity function takes
    /// the identity table.
    rgb16: OnceLock<Box<[u16]>>,
    /// The ink tables at 16 bits of the cyan, magenta, yellow and black
    /// channels, one after the other, tabulated on first use; the spot
    /// channels take the black table.
    ink16: OnceLock<Box<[u16]>>,
}

impl PartialEq for Transfer {
    /// Two transfers are equal when their tables are and their functions
    /// are the same compiled functions.
    fn eq(&self, other: &Self) -> bool {
        self.rgb == other.rgb
            && self.ink == other.ink
            && self
                .functions
                .iter()
                .zip(&other.functions)
                .all(|(a, b)| match (a, b) {
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    (a, b) => a.is_none() && b.is_none(),
                })
    }
}

impl Eq for Transfer {}

impl Transfer {
    /// The tables of the red, green, blue and gray functions, in that
    /// order, and the functions themselves, `None` for the identity.
    fn new(tables: &[[u8; 256]; 4], functions: [Option<Arc<Function>>; 4]) -> Self {
        let [red, green, blue, _] = *tables;
        let [cyan, magenta, yellow, black]: [[u8; 256]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|v| u8::MAX - tables[i][usize::from(u8::MAX) - v])
        });
        Self {
            rgb: [red, green, blue],
            ink: [cyan, magenta, yellow, black, black, black, black, black],
            functions,
            rgb16: OnceLock::new(),
            ink16: OnceLock::new(),
        }
    }

    /// The function of ink channel `channel`: red, green and blue for
    /// cyan, magenta and yellow, gray for black and the spots.
    fn ink_function(&self, channel: usize) -> Option<&Arc<Function>> {
        self.functions[channel.min(3)].as_ref()
    }

    /// The 16-bit ink value `v` of channel `channel` mapped through its
    /// function: `MAX − f(MAX − v)`, the function evaluated at the
    /// complement.  A function that fails maps the complement to 0.
    fn ink16_at(&self, channel: usize, v: u16) -> u16 {
        self.ink_function(channel).map_or(v, |f| {
            u16::MAX - evaluate::<u16>(f, (u16::MAX - v).to_unit()).unwrap_or(0)
        })
    }

    /// [`Self::apply_ink`] at 16 bits: each channel of `channels` mapped
    /// through its function at the 16-bit value ([`Self::ink_tables16`]
    /// holds the same values).
    #[must_use]
    pub fn apply_ink16(&self, ink: [u16; 8], channels: ChannelMask) -> [u16; 8] {
        std::array::from_fn(|i| {
            if channels.paints(i) {
                self.ink16_at(i, ink[i])
            } else {
                ink[i]
            }
        })
    }

    /// The ink tables at 16 bits of the eight ink channels: entry `v` of
    /// table `i` is channel `i`'s value `v` mapped through its function;
    /// tabulated on first use.
    #[must_use]
    pub fn ink_tables16(&self) -> [&[u16; 65_536]; 8] {
        let tables = self.ink16.get_or_init(|| {
            let mut tables = vec![0u16; 4 * 65_536].into_boxed_slice();
            for (channel, table) in tables.as_chunks_mut::<65_536>().0.iter_mut().enumerate() {
                for (v, out) in (0..=u16::MAX).zip(table.iter_mut()) {
                    *out = self.ink16_at(channel, v);
                }
            }
            tables
        });
        let [c, m, y, k] = tables.as_chunks::<65_536>().0 else {
            unreachable!("four tables of 65 536 entries");
        };
        [c, m, y, k, k, k, k, k]
    }

    /// Map the channels `channels` of the ink `[C, M, Y, K, S0, S1, S2,
    /// S3]` through the ink tables.  The other channels carry no component
    /// of the colour and pass unchanged: a function with `f(1) ≠ 1` maps
    /// an absent ink to some ink, which only the colour's own components
    /// receive.
    #[must_use]
    pub fn apply_ink(&self, ink: [u8; 8], channels: ChannelMask) -> [u8; 8] {
        std::array::from_fn(|i| {
            if channels.paints(i) {
                self.ink[i][usize::from(ink[i])]
            } else {
                ink[i]
            }
        })
    }

    /// The red, green and blue tables.
    #[must_use]
    pub const fn tables(&self) -> [&[u8; 256]; 3] {
        let [r, g, b] = &self.rgb;
        [r, g, b]
    }

    /// The red, green and blue tables at 16 bits: entry `v` of each is
    /// [`Self::apply16`] of `v`, the function evaluated at the 16-bit
    /// value; tabulated on first use.
    #[must_use]
    pub fn tables16(&self) -> [&[u16; 65_536]; 3] {
        let tables = self.rgb16.get_or_init(|| {
            let mut tables = vec![0u16; 3 * 65_536].into_boxed_slice();
            let chunks = tables.as_chunks_mut::<65_536>().0;
            for (table, function) in chunks.iter_mut().zip(&self.functions[..3]) {
                match function {
                    Some(f) => {
                        for (v, out) in (0..=u16::MAX).zip(table.iter_mut()) {
                            *out = evaluate(f, v.to_unit()).unwrap_or(0);
                        }
                    }
                    None => table.copy_from_slice(u16::identity_table()),
                }
            }
            tables
        });
        let [r, g, b] = tables.as_chunks::<65_536>().0 else {
            unreachable!("three tables of 65 536 entries");
        };
        [r, g, b]
    }

    /// Map a 16-bit RGB colour through the functions, evaluated at the
    /// colour itself.  A function that fails maps its component to 0.
    #[must_use]
    pub fn apply16(&self, rgb: [u16; 3]) -> [u16; 3] {
        std::array::from_fn(|i| {
            self.functions[i]
                .as_ref()
                .map_or(rgb[i], |f| evaluate(f, rgb[i].to_unit()).unwrap_or(0))
        })
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
#[non_exhaustive]
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
pub(crate) fn ext_gstate_transfer(doc: &Document, d: &Dictionary) -> Option<TransferSetting> {
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
    let components = if let Object::Array(fns) = obj {
        let [r, g, b, gray] = fns.as_slice() else {
            return None;
        };
        [
            component(doc, r)?,
            component(doc, g)?,
            component(doc, b)?,
            component(doc, gray)?,
        ]
    } else {
        let one = component(doc, obj)?;
        [one.clone(), one.clone(), one.clone(), one]
    };
    let tables = components.each_ref().map(|(lut, _)| *lut);
    Some(if tables.iter().all(|lut| *lut == IDENTITY) {
        TransferSetting::Identity
    } else {
        let functions =
            components.map(|(lut, function)| (lut != IDENTITY).then_some(function).flatten());
        TransferSetting::Functions(Arc::new(Transfer::new(&tables, functions)))
    })
}

/// The table of one transfer-function entry — a function, `/Identity` or
/// `/Default` — and the function, `None` for the identity.
fn component(doc: &Document, obj: &Object) -> Option<([u8; 256], Option<Arc<Function>>)> {
    match obj.as_name() {
        Some(b"Identity" | b"Default") => Some((IDENTITY, None)),
        Some(_) => None,
        None => {
            let function = Function::load(doc, obj);
            let mut lut = [0u8; 256];
            tabulate(&function, &mut lut)?;
            Some((lut, Some(function)))
        }
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

    /// At 16 bits a function maps the 16-bit value itself, through
    /// `apply16` and its tables alike, and an identity entry passes it.
    #[test]
    fn sixteen_bit_transfer_evaluates_the_function_at_the_value() {
        let doc = invert_doc();
        let setting = ext_gstate_transfer(&doc, &gs("<< /TR [4 0 R /Identity 4 0 R 4 0 R] >>"));
        let Some(TransferSetting::Functions(t)) = setting else {
            panic!("expected functions, got {setting:?}");
        };
        assert_eq!(t.apply16([0, 1000, 65_535]), [65_535, 1000, 0]);
        let [r, g, b] = t.tables16();
        assert_eq!((r[1000], g[1000], b[1000]), (64_535, 1000, 64_535));
        assert_eq!(r[12_345], t.apply16([12_345, 0, 0])[0]);
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

    /// On ink a function maps the complement: the inverting function maps
    /// no ink to full ink.  The gray function of an array maps black and
    /// the spots; a gray function alone is not the identity.
    /// At 16 bits the ink maps through the functions themselves, as the
    /// 8-bit tables do: the inverting gray function takes a 16-bit ink to
    /// its 16-bit complement on the black and spot channels, and the
    /// tables hold the same values.
    #[test]
    fn ink_maps_at_16_bits_through_the_functions() {
        let doc = invert_doc();
        let Some(TransferSetting::Functions(t)) =
            ext_gstate_transfer(&doc, &gs("<< /TR [/Identity /Identity /Identity 4 0 R] >>"))
        else {
            panic!("a gray function alone maps ink");
        };
        let ink = [10, 20, 30, 1000, 50_000, 0, 7, 65_535];
        assert_eq!(
            t.apply_ink16(ink, ChannelMask::new(0b0111_1111)),
            [10, 20, 30, 64_535, 15_535, 65_535, 65_528, 65_535]
        );
        let tables = t.ink_tables16();
        for (i, &v) in ink.iter().enumerate() {
            assert_eq!(
                tables[i][usize::from(v)],
                t.apply_ink16(ink, ChannelMask::ALL)[i]
            );
        }
    }

    #[test]
    fn ink_tables_map_the_complement_per_colorant() {
        let doc = invert_doc();
        let Some(TransferSetting::Functions(t)) = ext_gstate_transfer(&doc, &gs("<< /TR 4 0 R >>"))
        else {
            panic!("expected functions");
        };
        assert_eq!(
            t.apply_ink([0, 51, 255, 0, 0, 0, 0, 10], ChannelMask::ALL),
            [255, 204, 0, 255, 255, 255, 255, 245]
        );
        assert_eq!(
            t.apply_ink([0, 51, 255, 0, 0, 0, 0, 10], ChannelMask::new(0b1000_1111)),
            [255, 204, 0, 255, 0, 0, 0, 245],
            "channels outside the mask pass unchanged"
        );
        let Some(TransferSetting::Functions(t)) =
            ext_gstate_transfer(&doc, &gs("<< /TR [/Identity /Identity /Identity 4 0 R] >>"))
        else {
            panic!("a gray function alone maps ink");
        };
        assert_eq!(t.apply([0, 51, 255]), [0, 51, 255]);
        assert_eq!(
            t.apply_ink([10, 20, 30, 40, 50, 0, 0, 0], ChannelMask::ALL),
            [10, 20, 30, 215, 205, 255, 255, 255]
        );
        let Some(TransferSetting::Functions(t)) =
            ext_gstate_transfer(&doc, &gs("<< /TR [4 0 R /Identity /Identity /Identity] >>"))
        else {
            panic!("expected functions");
        };
        assert_eq!(
            t.apply_ink([10, 20, 30, 40, 0, 0, 0, 0], ChannelMask::ALL),
            [245, 20, 30, 40, 0, 0, 0, 0]
        );
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
