//! One image, many scan layouts.
//!
//! Each group of files under `images/scans/` holds the same quantised
//! coefficients: the first was written by an encoder with every component
//! in one scan, the others were derived from it by a lossless transcode
//! that only changes how the coefficients are laid out in scans (one scan
//! per component, components grouped two ways, restart intervals,
//! progressive).  A decoder must produce the same pixels for all of them.

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Decodes `data` strictly to its natural colour space: gray, RGB or the
/// four components unchanged.
fn decode(data: &[u8], name: &str) -> (usize, usize, Vec<u8>) {
    let mut headers = JpegDecoder::new(ZCursor::new(data));
    headers
        .decode_headers()
        .unwrap_or_else(|e| panic!("{name}: headers: {e:?}"));
    let input = headers.input_colorspace().unwrap();
    let out = match input.num_components() {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        _ => input,
    };
    let options = DecoderOptions::default()
        .jpeg_set_out_colorspace(out)
        .set_strict_mode(true);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    let pixels = decoder
        .decode()
        .unwrap_or_else(|e| panic!("{name}: decode: {e:?}"));
    let (w, h) = decoder.dimensions().unwrap();
    assert_eq!(pixels.len(), w * h * out.num_components(), "{name}");
    (w, h, pixels)
}

macro_rules! image {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("images/scans/", $name)).as_slice(),
        )
    };
}

type Group = &'static [(&'static str, &'static [u8])];

/// Colour images at four chroma subsamplings; the top third is noise and
/// the rest is flat, so the last rows of every scan are a few bytes long.
const COLOUR: &[Group] = &[
    &[
        image!("mix_444.jpg"),
        image!("mix_444_ni.jpg"),
        image!("mix_444_ni_rst.jpg"),
        image!("mix_444_y_cbcr.jpg"),
        image!("mix_444_ycb_cr.jpg"),
    ],
    &[
        image!("mix_420.jpg"),
        image!("mix_420_ni.jpg"),
        image!("mix_420_ni_rst.jpg"),
        image!("mix_420_y_cbcr.jpg"),
        image!("mix_420_ycb_cr.jpg"),
    ],
    &[
        image!("mix_422.jpg"),
        image!("mix_422_ni.jpg"),
        image!("mix_422_ni_rst.jpg"),
        image!("mix_422_y_cbcr.jpg"),
        image!("mix_422_ycb_cr.jpg"),
    ],
    &[
        image!("mix_440.jpg"),
        image!("mix_440_ni.jpg"),
        image!("mix_440_ni_rst.jpg"),
        image!("mix_440_y_cbcr.jpg"),
        image!("mix_440_ycb_cr.jpg"),
    ],
];

const CMYK: Group = &[
    image!("cmyk.jpg"),
    image!("cmyk_ni.jpg"),
    image!("cmyk_k_cmy.jpg"),
];

fn assert_group_decodes_alike(group: Group) {
    let (first, rest) = group.split_first().unwrap();
    let want = decode(first.1, first.0);
    for (name, data) in rest {
        let got = decode(data, name);
        assert_eq!((got.0, got.1), (want.0, want.1), "{name}: size");
        let differing = got.2.iter().zip(&want.2).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} samples differ from {}",
            first.0
        );
    }
}

#[test]
fn sequential_scan_layouts_decode_alike() {
    for group in COLOUR {
        assert_group_decodes_alike(group);
    }
}

#[test]
fn four_component_scan_layouts_decode_alike() {
    assert_group_decodes_alike(CMYK);
}

/// The largest difference between two decodes of the same coefficients.
fn max_difference(a: (&str, &[u8]), b: (&str, &[u8])) -> u8 {
    let (a, b) = (decode(a.1, a.0), decode(b.1, b.0));
    assert_eq!((a.0, a.1), (b.0, b.1));
    a.2.iter()
        .zip(&b.2)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap()
}

/// A progressive stream of the same coefficients decodes to the same
/// image; the two paths pick their inverse transforms differently, which
/// may round a sample one level apart.
#[test]
fn progressive_matches_sequential() {
    let pairs = [
        (image!("mix_444.jpg"), image!("mix_444_prog.jpg")),
        (image!("mix_420.jpg"), image!("mix_420_prog.jpg")),
        (image!("mix_422.jpg"), image!("mix_422_prog.jpg")),
        (image!("mix_440.jpg"), image!("mix_440_prog.jpg")),
        (image!("gray.jpg"), image!("gray_prog.jpg")),
        (image!("cmyk.jpg"), image!("cmyk_prog.jpg")),
    ];
    for (sequential, progressive) in pairs {
        let d = max_difference(sequential, progressive);
        assert!(
            d <= 1,
            "{}: differs from {} by {d}",
            progressive.0,
            sequential.0
        );
    }
}

/// A flat 16×16 image with one scan per component: each scan is two bytes
/// long, so the marker after it is in the bit reader's lookahead from the
/// first block on.
#[test]
fn flat_image_with_a_scan_per_component_is_flat() {
    for (name, data) in [image!("flat_444_ni.jpg"), image!("flat_420_ni.jpg")] {
        let (w, h, pixels) = decode(data, name);
        assert_eq!((w, h), (16, 16), "{name}");
        for (i, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
            for (c, want) in px.iter().zip([206_u8, 81, 103]) {
                assert!(
                    c.abs_diff(want) <= 3,
                    "{name}: pixel {i} is {px:?}, not (206, 81, 103)"
                );
            }
        }
    }
}
