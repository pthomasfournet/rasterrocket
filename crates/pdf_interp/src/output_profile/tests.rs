use super::*;

/// The bytes of moxcms's sRGB profile.
fn srgb_bytes() -> Vec<u8> {
    ColorProfile::new_srgb().encode().expect("encodes")
}

/// The bytes of a gray profile of gamma 2.2.
fn gray_bytes() -> Vec<u8> {
    ColorProfile::new_gray_with_gamma(2.2)
        .encode()
        .expect("encodes")
}

/// A `width × 1` 8-bit canvas of `px`, with alpha `alpha` per pixel when
/// given.
fn rgb8(px: &[[u8; 3]], alpha: Option<&[u8]>) -> Bitmap<Rgb8> {
    let width = u32::try_from(px.len()).unwrap();
    let mut bmp = Bitmap::<Rgb8>::new(width, 1, 1, alpha.is_some());
    for (dst, src) in bmp
        .row_samples_mut(0)
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(px)
    {
        *dst = *src;
    }
    if let (Some(a), Some(plane)) = (alpha, bmp.alpha_plane_mut()) {
        plane.copy_from_slice(a);
    }
    bmp
}

const PATCHES: [[u8; 3]; 6] = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [12, 200, 77],
    [128, 128, 128],
    [250, 3, 190],
];

#[test]
fn profiles_that_are_not_rgb_or_gray_outputs_are_refused() {
    assert_eq!(
        OutputProfile::from_bytes(srgb_bytes()).unwrap().space(),
        OutputSpace::Rgb
    );
    assert_eq!(
        OutputProfile::from_bytes(gray_bytes()).unwrap().space(),
        OutputSpace::Gray
    );
    let assets = concat!(env!("CARGO_MANIFEST_DIR"), "/../jpeg2000/assets/");
    let cmyk = std::fs::read(format!("{assets}CGATS001Compat-v2-micro.icc")).unwrap();
    assert_eq!(
        OutputProfile::from_bytes(cmyk).unwrap_err(),
        OutputProfileError::Cmyk
    );
    let lab = std::fs::read(format!("{assets}LAB.icc")).unwrap();
    assert!(OutputProfile::from_bytes(lab).is_err());
    assert!(matches!(
        OutputProfile::from_bytes(b"not a profile".to_vec()),
        Err(OutputProfileError::Unreadable(_))
    ));
    let mut link = srgb_bytes();
    link[12..16].copy_from_slice(b"link");
    assert_eq!(
        OutputProfile::from_bytes(link).unwrap_err(),
        OutputProfileError::Class("device link")
    );
}

/// The working space converted to itself is unchanged, under every intent
/// and with compensation (its black is the PCS origin).
#[test]
fn srgb_to_srgb_is_the_identity() {
    let profile = OutputProfile::from_bytes(srgb_bytes()).unwrap();
    for intent in RenderingIntent::ALL {
        for bpc in [false, true] {
            let canvas = FinishedCanvas::Rgb8(rgb8(&PATCHES, None));
            let out = profile
                .convert(canvas, (intent, bpc), &mut CanvasBuffers::default())
                .unwrap();
            let FinishedCanvas::Rgb8(out) = out else {
                panic!("RGB")
            };
            for (got, want) in out.row_samples(0).iter().zip(PATCHES.as_flattened()) {
                assert!(got.abs_diff(*want) <= 1, "{intent} {bpc}: {got} vs {want}");
            }
        }
    }
}

#[test]
fn a_gray_profile_converts_to_gray_keeping_alpha() {
    let profile = OutputProfile::from_bytes(gray_bytes()).unwrap();
    let alpha = [255, 255, 0, 255, 128, 255];
    let canvas = FinishedCanvas::Rgb8(rgb8(&PATCHES, Some(&alpha)));
    let mut buffers = CanvasBuffers::default();
    let FinishedCanvas::Gray8(gray) = profile
        .convert(
            canvas,
            (RenderingIntent::RelativeColorimetric, false),
            &mut buffers,
        )
        .unwrap()
    else {
        panic!("gray")
    };
    let row = gray.row_samples(0);
    assert_eq!(row[0], 0, "black");
    assert!(row[1] >= 254, "white {}", row[1]);
    assert_eq!(row[2], 0, "alpha 0 is colour 0");
    assert!(
        row[4].abs_diff(128) <= 2,
        "sRGB mid gray at gamma 2.2: {}",
        row[4]
    );
    assert_eq!(gray.alpha_plane(), Some(&alpha[..]));
    assert!(
        buffers.rgb.data.capacity() > 0,
        "the RGB storage is reclaimed"
    );
}

#[test]
fn sixteen_bit_canvases_convert_through_the_16_bit_conversion() {
    let profile = OutputProfile::from_bytes(gray_bytes()).unwrap();
    let mut rgb = Bitmap::<Rgb16>::new(2, 1, 1, false);
    rgb.row_samples_mut(0)
        .copy_from_slice(&[0, 0, 0, u16::MAX, u16::MAX, u16::MAX]);
    let FinishedCanvas::Gray16(gray) = profile
        .convert(
            FinishedCanvas::Rgb16(rgb),
            (RenderingIntent::Perceptual, false),
            &mut CanvasBuffers::default(),
        )
        .unwrap()
    else {
        panic!("16-bit gray")
    };
    let row = gray.row_samples(0);
    assert!(row[0] < 1024 && row[1] > u16::MAX - 64, "{row:?}");
}

#[test]
fn conversions_are_compiled_once_and_slots_are_distinct() {
    let profile = OutputProfile::from_bytes(srgb_bytes()).unwrap();
    let a = profile.executor(RenderingIntent::Saturation, true).unwrap();
    let b = profile.executor(RenderingIntent::Saturation, true).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    let mut seen = std::collections::HashSet::new();
    for intent in RenderingIntent::ALL {
        for bpc in [false, true] {
            assert!(seen.insert(slot(intent, bpc)));
        }
    }
    assert_eq!(seen.len(), SLOTS);
}
