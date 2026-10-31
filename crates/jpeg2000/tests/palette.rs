//! Palettes (I.5.3.4, I.5.3.5).

use crate::*;

#[test]
fn basic() {
    check("palette_basic", reference("palette_basic"));
}

#[test]
fn tiled() {
    check("palette_tiled", reference("palette_basic"));
}

#[test]
fn small() {
    check("palette_small", reference("palette_small"));
}

#[test]
fn four_bit_entries() {
    check(
        "palette_four_bit_entries",
        reference("palette_four_bit_entries"),
    );
}

#[test]
fn out_of_range_indices() {
    check(
        "palette_out_of_range_indices",
        reference("palette_out_of_range_indices"),
    );
}

#[test]
fn cdef() {
    check("palette_cdef", reference("palette_basic"));
}

#[test]
fn cmap_outside_jp2h() {
    check("palette_cmap_outside_jp2h", reference("palette_basic"));
}

#[test]
fn unresolved() {
    check_with(
        "palette_basic",
        DecodeSettings {
            resolve_palette_indices: false,
            ..Default::default()
        },
        reference("palette_unresolved"),
    );
}

/// `palette_basic` with the contents of its `name` box (a child of `jp2h`)
/// replaced by `contents`.
fn palette_basic_with_box(name: &[u8; 4], contents: &[u8]) -> Vec<u8> {
    let mut data = read_stream("palette_basic");
    let find = |data: &[u8], name: &[u8]| data.windows(4).position(|w| w == name).unwrap() - 4;
    let box_len = |data: &[u8], at: usize| u32::from_be_bytes(data[at..at + 4].try_into().unwrap());
    let (jp2h, child) = (find(&data, b"jp2h"), find(&data, name));
    let old_len = box_len(&data, child);
    let new_len = 8 + contents.len() as u32;
    let jp2h_len = box_len(&data, jp2h) - old_len + new_len;
    data.splice(
        child + 8..child + old_len as usize,
        contents.iter().copied(),
    );
    data[child..child + 4].copy_from_slice(&new_len.to_be_bytes());
    data[jp2h..jp2h + 4].copy_from_slice(&jp2h_len.to_be_bytes());
    data
}

/// A palette whose columns are deeper than the 31 bits the sample
/// conversions support is ignored, as an unreadable palette box is: the
/// indices decode as a grayscale image.
#[test]
fn palette_deeper_than_supported() {
    let (entries, columns) = (256_u16, 3_u8);
    let mut pclr = entries.to_be_bytes().to_vec();
    pclr.push(columns);
    pclr.extend([31; 3]);
    pclr.extend((0..usize::from(entries) * usize::from(columns)).flat_map(|i| [i as u8, 0, 0, 0]));
    let data = palette_basic_with_box(b"pclr", &pclr);

    let image = Image::new(&data, &DecodeSettings::default()).unwrap();
    let mut ctx = DecoderContext::default();
    let decoded = image.decode(&mut ctx).unwrap();
    assert_eq!(decoded.components().len(), 1);
    assert_eq!(
        decoded.data_u8().len(),
        (image.width() * image.height()) as usize
    );
}

/// `palette_basic` with its component mapping box emptied: the palette
/// then maps no component, which leaves no image to return.
#[test]
fn empty_cmap() {
    let data = palette_basic_with_box(b"cmap", &[]);

    let image = Image::new(&data, &DecodeSettings::default()).unwrap();
    let mut ctx = DecoderContext::default();
    match image.decode(&mut ctx) {
        Ok(_) => panic!("decoding succeeded, but should have failed"),
        Err(e) => assert_eq!(e.to_string(), "failed to resolve palette indices"),
    }
}
