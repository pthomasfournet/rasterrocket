//! The conversion of CMYK ink of one profile to the ink of a CMYK output
//! profile.
//!
//! A colour given in a four-component ICC space is a CMYK device's values
//! in that space's profile.  On a page separated through a different CMYK
//! profile it converts to that profile's ink (ISO 32000-2 §8.6.5.5): the
//! source's device → PCS table of the intent, evaluated at the ink, then
//! the output profile's PCS → device table, evaluated at the PCS value
//! ([`crate::pcs::lut::ProfileLut`]).  Between them the PCS value is
//! scaled by the ratio of the media whites under absolute colorimetry
//! (see [`crate::pcs`]), or, under black-point compensation, mapped by
//! the neutral affine map of ISO 18619 that takes the source's black to
//! the output profile's and keeps white: `X' = s · X + (Y_d − s · Y_s) ·
//! D50` with `s = (1 − Y_d) / (1 − Y_s)`, `Y_s` and `Y_d` the luminances of
//! the two blacks: the source's as a conversion source finds it, the
//! output profile's as a separation to it does ([`super::source`]), which
//! a perceptual or saturation conversion to a version 4 profile always
//! compensates to.  The conversion is tabulated on a
//! grid of [`LINK_SPAN`] cells along each ink axis
//! ([`super::ink_table::InkTable`]).

use moxcms::{ColorProfile, RenderingIntent};

use super::ink_table::InkTable;
use crate::pcs::D50;
use crate::pcs::black::{is_v4, source_black};
use crate::pcs::lut::{Direction, ProfileLut};

/// The number of cells along each ink axis of a link's table.
pub(super) const LINK_SPAN: u32 = 16;

/// The conversion of the ink of `source` to the ink of `dest` under
/// `intent`, with black-point compensation when `bpc`; `None` when either
/// profile lacks the table of the intent or a black point cannot be
/// found.
pub(super) fn tabulate(
    source: &ColorProfile,
    dest: &ColorProfile,
    intent: RenderingIntent,
    bpc: bool,
) -> Option<InkTable> {
    let to_pcs = ProfileLut::new(source, intent, Direction::ToPcs, 4)?;
    let to_device = ProfileLut::new(dest, intent, Direction::ToDevice, 4)?;
    let adjust = pcs_adjustment(source, dest, intent, bpc)?;
    InkTable::tabulate(LINK_SPAN, 4, |inks, out| {
        for (ink, o) in inks
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<4>().0)
        {
            to_device.to_device(adjust.apply(to_pcs.to_pcs(ink)), o);
        }
        Ok::<(), std::convert::Infallible>(())
    })
    .ok()
}

/// An affine map of the PCS, component by component: `X' = a · X + b`.
#[derive(Debug, Clone, Copy)]
struct PcsMap {
    a: [f64; 3],
    b: [f64; 3],
}

impl PcsMap {
    const IDENTITY: Self = Self {
        a: [1.0; 3],
        b: [0.0; 3],
    };

    fn apply(&self, x: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|i| self.a[i].mul_add(x[i], self.b[i]))
    }
}

/// The map of the PCS between the two profiles' tables (see the module
/// documentation); `None` when a black point cannot be found.
fn pcs_adjustment(
    source: &ColorProfile,
    dest: &ColorProfile,
    intent: RenderingIntent,
    bpc: bool,
) -> Option<PcsMap> {
    if intent == RenderingIntent::AbsoluteColorimetric {
        let white = |p| crate::pcs::media_white(p);
        return Some(
            crate::pcs::absolute_scale(white(source), white(dest))
                .map_or(PcsMap::IDENTITY, |a| PcsMap { a, b: [0.0; 3] }),
        );
    }
    let perceptual = matches!(
        intent,
        RenderingIntent::Perceptual | RenderingIntent::Saturation
    );
    if !(bpc || perceptual && is_v4(dest)) {
        return Some(PcsMap::IDENTITY);
    }
    let ys = source_black(source, intent).ok()?;
    let yd = super::source::black_lift(dest, intent, true).ok()??;
    if (ys - yd).abs() < 1e-9 || ys >= 1.0 {
        return Some(PcsMap::IDENTITY);
    }
    let s = (1.0 - yd) / (1.0 - ys);
    let offset = s.mul_add(-ys, yd);
    Some(PcsMap {
        a: [s; 3],
        b: D50.map(|w| offset * w),
    })
}
