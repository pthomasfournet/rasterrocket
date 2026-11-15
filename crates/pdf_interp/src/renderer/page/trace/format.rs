//! The text of a trace's values: rounded numbers, names, resource keys,
//! path summaries, colour spaces and clip names.

use std::fmt::Write as _;
use std::sync::Arc;

use raster::fnv::Fnv1a;
use raster::path::Path;

use crate::resources::ResourceKey;
use crate::resources::colorspace::{ColorSpace, Colorant};

/// The most points a path prints in full.
pub(super) const INLINE_POINTS: usize = 8;

/// Fraction bits of a printed device coordinate, line width or font size.
const COORD_BITS: u32 = 8;

/// Decimal places of a printed colour component or matrix entry.
const UNIT_DIGITS: u32 = 6;

/// `v` rounded to the nearest integer, half away from zero; `None` when
/// `v` is not finite.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the float-to-int conversion saturates, and a value beyond 2^63 \
              quanta has no device meaning"
)]
fn finite_round(v: f64) -> Option<i64> {
    v.is_finite().then(|| v.round() as i64)
}

/// `v` rounded to the nearest multiple of `2^-bits`, in units of that
/// multiple; `None` when `v` is not finite.
fn quantise(v: f64, bits: u32) -> Option<i64> {
    finite_round(v * f64::from(1_u32 << bits))
}

/// Append the non-finite `v` as `nan`, `inf` or `-inf`.
fn push_non_finite(out: &mut String, v: f64) {
    out.push_str(if v.is_nan() {
        "nan"
    } else if v > 0.0 {
        "inf"
    } else {
        "-inf"
    });
}

/// Append `v`, rounded to the nearest multiple of `2^-bits`, as the exact
/// decimal of the rounded value; `nan`, `inf` or `-inf` when not finite.
fn push_quantised(out: &mut String, v: f64, bits: u32) {
    let Some(q) = quantise(v, bits) else {
        push_non_finite(out, v);
        return;
    };
    push_fixed(out, q, bits);
}

/// Append the fixed-point value `q`, of `bits` fraction bits, as its exact
/// decimal: `2^-bits = 5^bits · 10^-bits`, so the fraction has `bits`
/// decimal digits, trailing zeros dropped.
fn push_fixed(out: &mut String, q: i64, bits: u32) {
    let a = i128::from(q).abs();
    let one = 1_i128 << bits;
    if q < 0 {
        out.push('-');
    }
    let _ = write!(out, "{}", a / one);
    let frac = a % one;
    if frac != 0 {
        let digits = frac * 5_i128.pow(bits);
        let text = format!("{digits:0width$}", width = bits as usize);
        out.push('.');
        out.push_str(text.trim_end_matches('0'));
    }
}

/// Append the device coordinate `v`.
pub(super) fn push_coord(out: &mut String, v: f64) {
    push_quantised(out, v, COORD_BITS);
}

/// Append the colour component or matrix entry `v`.
pub(super) fn push_unit(out: &mut String, v: f64) {
    let scale = 10_i64.pow(UNIT_DIGITS);
    #[expect(
        clippy::cast_precision_loss,
        reason = "10^6 is exactly representable in f64"
    )]
    let Some(q) = finite_round(v * scale as f64) else {
        push_non_finite(out, v);
        return;
    };
    let a = i128::from(q).abs();
    let one = i128::from(scale);
    if q < 0 {
        out.push('-');
    }
    let _ = write!(out, "{}", a / one);
    let frac = a % one;
    if frac != 0 {
        let text = format!("{frac:0width$}", width = UNIT_DIGITS as usize);
        out.push('.');
        out.push_str(text.trim_end_matches('0'));
    }
}

/// Append the values `vs`, comma-separated, each by `push`.
pub(super) fn push_list(out: &mut String, vs: &[f64], push: fn(&mut String, f64)) {
    for (i, &v) in vs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push(out, v);
    }
}

/// Append `name` as a PDF name: `/`, then each byte outside `!`..=`~`, and
/// `#` itself, as `#` and two hexadecimal digits.
pub(super) fn push_name(out: &mut String, name: &[u8]) {
    out.push('/');
    for &b in name {
        if (b'!'..=b'~').contains(&b) && b != b'#' {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "#{b:02x}");
        }
    }
}

/// Append the identity of the object a resource refers to: `(n g R)`, or
/// for a direct object, the name it is held under in the resources of the
/// object `(n g)`.
pub(super) fn push_key(out: &mut String, key: &ResourceKey) {
    match key {
        ResourceKey::Object((n, g)) => {
            let _ = write!(out, "({n} {g} R)");
        }
        ResourceKey::Direct {
            context: (n, g),
            name,
        } => {
            let _ = write!(out, "(direct {n} {g} ");
            push_name(out, name);
            out.push(')');
        }
    }
}

/// Append the line of the glyph of `code`, drawn as the glyph `selector`
/// (`(selector, by_gid)`; `None` when the font draws nothing for it), at
/// the pen `pen` in the page's device space.
pub(in crate::renderer::page) fn push_glyph(
    out: &mut String,
    code: u32,
    selector: Option<(u32, bool)>,
    pen: (f64, f64),
) {
    let _ = write!(out, "glyph c={code:x}");
    match selector {
        Some((selector, by_gid)) => {
            let _ = write!(out, " sel={selector}{}", if by_gid { "g" } else { "" });
        }
        None => out.push_str(" none"),
    }
    out.push_str(" pen=");
    push_coord(out, pen.0);
    out.push(',');
    push_coord(out, pen.1);
    out.push('\n');
}

/// Append the summary of `path`, whose points lie in a window whose
/// top-left pixel is the page's pixel `origin`: its points' counts,
/// bounding box and hash in the page's device space, and the points of a
/// path of at most [`INLINE_POINTS`] points.
pub(super) fn push_path(out: &mut String, path: &Path, origin: (u32, u32)) {
    let (ox, oy) = (f64::from(origin.0), f64::from(origin.1));
    let mut hash = Fnv1a::new();
    let mut bbox: Option<[i64; 4]> = None;
    let (mut subpaths, mut curve_points) = (0_usize, 0_usize);
    for (pt, flags) in path.pts.iter().zip(&path.flags) {
        let (x, y) = (
            quantise(pt.x + ox, COORD_BITS),
            quantise(pt.y + oy, COORD_BITS),
        );
        hash.update(&x.unwrap_or(i64::MIN).to_le_bytes());
        hash.update(&y.unwrap_or(i64::MIN).to_le_bytes());
        hash.update(&[flags.bits()]);
        subpaths += usize::from(flags.is_first());
        curve_points += usize::from(flags.is_curve());
        if let (Some(x), Some(y)) = (x, y) {
            bbox = Some(bbox.map_or([x, y, x, y], |[x0, y0, x1, y1]| {
                [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
            }));
        }
    }
    let _ = write!(
        out,
        "n={} sub={subpaths} cub={}",
        path.pts.len(),
        curve_points / 2
    );
    out.push_str(" bbox=");
    match bbox {
        Some(b) => {
            for (i, &v) in b.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_fixed(out, v, COORD_BITS);
            }
        }
        None => out.push('-'),
    }
    let _ = write!(out, " h={:016x}", hash.finish());
    if path.pts.len() <= INLINE_POINTS && !path.pts.is_empty() {
        out.push_str(" pts=");
        for (i, (pt, flags)) in path.pts.iter().zip(&path.flags).enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push(if flags.is_first() {
                'm'
            } else if flags.is_curve() {
                'c'
            } else {
                'l'
            });
            push_coord(out, pt.x + ox);
            out.push(',');
            push_coord(out, pt.y + oy);
            if flags.is_last() && flags.is_closed() {
                out.push_str(" z");
            }
        }
    }
}

/// The name of colour space `cs` in a trace: its family, and the
/// parameters that identify it.
pub(super) fn space_name(cs: &ColorSpace) -> String {
    match cs {
        ColorSpace::DeviceGray => "DeviceGray".to_owned(),
        ColorSpace::DeviceRgb => "DeviceRGB".to_owned(),
        ColorSpace::DeviceCmyk => "DeviceCMYK".to_owned(),
        ColorSpace::Cal(_) => "Cal".to_owned(),
        ColorSpace::Lab(_) => "Lab".to_owned(),
        ColorSpace::IccBased { n, stream_id } => match stream_id {
            Some((id, g)) => format!("ICCBased[{n} ({id} {g} R)]"),
            None => format!("ICCBased[{n}]"),
        },
        ColorSpace::Indexed { base, hival, .. } => {
            format!("Indexed[{} {hival}]", space_name(base))
        }
        ColorSpace::Pattern { base } => base.as_ref().map_or_else(
            || "Pattern".to_owned(),
            |base| format!("Pattern[{}]", space_name(base)),
        ),
        ColorSpace::Separation {
            alternate,
            colorant,
            ..
        } => {
            let mut s = "Separation[".to_owned();
            push_colorant(&mut s, colorant);
            let _ = write!(s, " {}]", space_name(alternate));
            s
        }
        ColorSpace::DeviceN {
            colorants,
            alternate,
            ..
        } => {
            let mut s = "DeviceN[".to_owned();
            for colorant in colorants.iter() {
                push_colorant(&mut s, colorant);
                s.push(' ');
            }
            let _ = write!(s, "{}]", space_name(alternate));
            s
        }
    }
}

/// Append the name of `colorant`.
fn push_colorant(out: &mut String, colorant: &Colorant) {
    match colorant {
        Colorant::None => out.push_str("/None"),
        Colorant::All => out.push_str("/All"),
        Colorant::Process(ink) => {
            let _ = write!(out, "/{ink:?}");
        }
        Colorant::Spot(name) => push_name(out, name),
    }
}

/// The colour `comps` in the space `cs`, as a paint prints it.
pub(super) fn colour_text(cs: &ColorSpace, comps: &[f64]) -> Arc<str> {
    let mut s = space_name(cs);
    s.push('(');
    push_list(&mut s, comps, push_unit);
    s.push(')');
    Arc::from(s)
}

/// The name of the clip that narrowing the clip named `parent` by the
/// operation printed as `body` produces.
pub(super) fn chained_clip(parent: u64, body: &str) -> u64 {
    let mut hash = Fnv1a::new();
    hash.update(&parent.to_le_bytes());
    hash.update(body.as_bytes());
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(q: i64, bits: u32) -> String {
        let mut s = String::new();
        push_fixed(&mut s, q, bits);
        s
    }

    fn coord(v: f64) -> String {
        let mut s = String::new();
        push_coord(&mut s, v);
        s
    }

    #[test]
    fn fixed_point_values_print_as_exact_decimals() {
        assert_eq!(fixed(0, 8), "0");
        assert_eq!(fixed(256 * 12 + 128, 8), "12.5");
        assert_eq!(fixed(256 * 12 + 1, 8), "12.00390625");
        assert_eq!(fixed(-1, 8), "-0.00390625");
        assert_eq!(fixed(1, 16), "0.0000152587890625");
        let mut unit = String::new();
        push_list(&mut unit, &[0.2, -0.5, 1.0, 1e-7, 0.123_456_5], push_unit);
        assert_eq!(unit, "0.2,-0.5,1,0,0.123457");
    }

    #[test]
    fn coordinates_round_to_the_nearest_quantum() {
        assert_eq!(coord(10.0), "10");
        assert_eq!(coord(10.0 - 1e-9), "10");
        assert_eq!(coord(10.0 + 1e-9), "10");
        assert_eq!(coord(0.5 / 256.0 - 1e-9), "0");
        assert_eq!(coord(f64::NAN), "nan");
        assert_eq!(coord(f64::NEG_INFINITY), "-inf");
    }

    #[test]
    fn names_escape_delimiters_and_controls() {
        let mut s = String::new();
        push_name(&mut s, b"F1 #a\x01");
        assert_eq!(s, "/F1#20#23a#01");
    }

    #[test]
    fn a_small_path_prints_its_points_in_the_page_space() {
        let mut pb = raster::path::PathBuilder::new();
        let _ = pb.move_to(1.0, 2.0);
        let _ = pb.line_to(3.5, 2.0);
        let _ = pb.line_to(3.5, 4.0);
        let _ = pb.close(true);
        let mut s = String::new();
        push_path(&mut s, &pb.build(), (10, 20));
        assert!(
            s.starts_with("n=4 sub=1 cub=0 bbox=11,22,13.5,24 h="),
            "{s}"
        );
        assert!(s.ends_with(" pts=m11,22 l13.5,22 l13.5,24 l11,22 z"), "{s}");
    }

    #[test]
    fn a_path_hash_ignores_last_bit_differences() {
        let path = |dx: f64| {
            let mut pb = raster::path::PathBuilder::new();
            let _ = pb.move_to(1.0 + dx, 2.0);
            let _ = pb.line_to(30.0, 2.0 - dx);
            pb.build()
        };
        let summary = |p: &Path| {
            let mut s = String::new();
            push_path(&mut s, p, (0, 0));
            s
        };
        assert_eq!(summary(&path(0.0)), summary(&path(1e-12)));
        assert_ne!(summary(&path(0.0)), summary(&path(1.0 / 256.0)));
    }

    #[test]
    fn a_clip_name_depends_on_its_parent_and_its_body() {
        assert_eq!(chained_clip(0, "nz a"), chained_clip(0, "nz a"));
        assert_ne!(chained_clip(0, "nz a"), chained_clip(1, "nz a"));
        assert_ne!(chained_clip(0, "nz a"), chained_clip(0, "eo a"));
    }
}
