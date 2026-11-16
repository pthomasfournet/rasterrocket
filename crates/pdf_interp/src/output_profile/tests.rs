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
fn profiles_that_are_not_rgb_gray_or_cmyk_outputs_are_refused() {
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
        OutputProfileError::Class("input")
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
                .convert(
                    canvas,
                    WorkingSpace::Srgb,
                    (intent, bpc),
                    &mut CanvasBuffers::default(),
                )
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
            WorkingSpace::Srgb,
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
            WorkingSpace::Srgb,
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
    let a = profile
        .executor(WorkingSpace::Wide, RenderingIntent::Saturation, true)
        .unwrap();
    let b = profile
        .executor(WorkingSpace::Wide, RenderingIntent::Saturation, true)
        .unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    let mut seen = std::collections::HashSet::new();
    for space in WorkingSpace::ALL {
        for intent in RenderingIntent::ALL {
            for bpc in [false, true] {
                assert!(seen.insert(slot(intent, bpc, space)));
            }
        }
    }
    assert_eq!(seen.len(), SLOTS);
}

/// The synthetic CMYK printer of `tests/fixtures/icc`.
fn cmyk_printer() -> OutputProfile {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
    );
    OutputProfile::from_bytes(std::fs::read(path).expect("the fixture is present"))
        .expect("a CMYK output profile")
}

/// The separation of wide-space colours to the CMYK printer under
/// relative colorimetry agrees with Little CMS's `transicc -t 1` from the
/// wide space within 1.5 % of each ink: Adobe RGB (1998) colours, two of
/// them outside sRGB, converted to the wide space by `transicc` too.
#[test]
fn a_cmyk_profile_separates_the_wide_space_like_transicc() {
    let profile = cmyk_printer();
    let table = profile
        .separation_table(
            WorkingSpace::Wide,
            RenderingIntent::RelativeColorimetric,
            false,
        )
        .expect("separates");
    // The wide value (0–255) and its ink (percent), for the Adobe RGB
    // colours (100, 180, 160), (60, 140, 200), (150, 200, 90), (200, 120, 160).
    let reference: [([f64; 3], [f64; 4]); 4] = [
        ([123.4434, 172.5023, 161.9424], [100.0, 0.0, 13.5561, 26.45]),
        (
            [106.9496, 135.8517, 195.6185],
            [100.0, 35.3658, 0.0, 16.1761],
        ),
        (
            [151.452, 193.1052, 105.2139],
            [45.8854, 0.0, 82.5956, 17.3938],
        ),
        (
            [188.8293, 136.9135, 160.015],
            [0.0, 58.0056, 32.5109, 8.7648],
        ),
    ];
    for (wide, want) in reference {
        let got = table.ink(wide.map(|v| u16::from_unit(v / 255.0)));
        for (g, w) in got.iter().zip(want) {
            let g = f64::from(*g) / 655.35;
            assert!((g - w).abs() <= 1.5, "{wide:?}: {got:?} for {want:?} %");
        }
    }
}

/// The separation of sRGB colours to the CMYK printer under relative
/// colorimetry agrees with Little CMS's `transicc -t 1` within 1.5 % of
/// each ink; gray separates to black ink alone, as the profile's tables
/// replace the gray component.
#[test]
fn a_cmyk_profile_separates_like_transicc() {
    let profile = cmyk_printer();
    assert_eq!(profile.space(), OutputSpace::Cmyk);
    assert_eq!(profile.space().channels(), 4);
    let table = profile
        .separation_table(
            WorkingSpace::Srgb,
            RenderingIntent::RelativeColorimetric,
            false,
        )
        .expect("separates");
    // transicc -n -i '*sRGB' -o printer-cmyk-v4.icc -t 1, in percent.
    let reference: [([u8; 3], [f64; 4]); 4] = [
        ([255, 0, 0], [0.0, 100.0, 100.0, 0.0]),
        ([128, 128, 128], [1.8067, 1.5991, 1.6236, 50.0679]),
        ([0, 0, 0], [0.0, 4.7059, 4.7059, 99.8672]),
        ([255, 255, 255], [0.6226, 0.1862, 0.4593, 0.0]),
    ];
    for (rgb, want) in reference {
        let got = table.ink(rgb.map(u16::from_u8));
        for (g, w) in got.iter().zip(want) {
            let g = f64::from(*g) / 655.35;
            assert!((g - w).abs() <= 1.5, "{rgb:?}: {got:?} for {want:?} %");
        }
    }
    let profile = Arc::new(profile);
    let Separation::Profile(press) =
        OutputProfile::separation(&profile, WorkingSpace::Srgb, None, false).expect("separates")
    else {
        panic!("a profile separation")
    };
    assert!(
        Arc::ptr_eq(&press.table(RenderingIntent::RelativeColorimetric), &table),
        "the table is built once"
    );
}

/// Below the printer's black, where its relative colorimetric table is
/// ill-conditioned (a few tenths of a* move the inks by percents), the
/// separation evaluates the table at each colour's exact PCS value and
/// agrees with Little CMS within 0.5 % of each ink.
#[test]
fn a_cmyk_separation_below_the_press_black_matches_transicc() {
    let table = cmyk_printer()
        .separation_table(
            WorkingSpace::Srgb,
            RenderingIntent::RelativeColorimetric,
            false,
        )
        .expect("separates");
    // transicc -n -i '*sRGB' -o printer-cmyk-v4.icc -t 1, in percent.
    let reference: [(u8, [f64; 4]); 4] = [
        (4, [0.0, 4.7059, 4.7059, 99.7894]),
        (8, [0.0, 4.7059, 4.7059, 99.7116]),
        (12, [0.0, 4.7059, 4.7059, 99.6307]),
        (16, [11.1864, 7.0420, 5.1148, 99.0570]),
    ];
    for (gray, want) in reference {
        let got = table.ink([u16::from_u8(gray); 3]);
        for (g, w) in got.iter().zip(want) {
            let g = f64::from(*g) / 655.35;
            assert!((g - w).abs() <= 0.5, "gray {gray}: {got:?} for {want:?} %");
        }
    }
}

/// Without a fixed intent a paint separates through the table of its own
/// intent; a fixed intent serves every paint.
#[test]
fn a_press_separation_takes_each_paint_s_intent_unless_one_is_fixed() {
    let profile = Arc::new(cmyk_printer());
    let table = |intent| {
        profile
            .separation_table(WorkingSpace::Srgb, intent, false)
            .expect("separates")
    };
    let Separation::Profile(own) =
        OutputProfile::separation(&profile, WorkingSpace::Srgb, None, false).expect("separates")
    else {
        panic!("a profile separation")
    };
    let Separation::Profile(fixed) = OutputProfile::separation(
        &profile,
        WorkingSpace::Srgb,
        Some(RenderingIntent::Saturation),
        false,
    )
    .expect("separates") else {
        panic!("a profile separation")
    };
    for intent in RenderingIntent::ALL {
        assert!(Arc::ptr_eq(&own.table(intent), &table(intent)), "{intent}");
        assert!(
            Arc::ptr_eq(&fixed.table(intent), &table(RenderingIntent::Saturation)),
            "{intent}"
        );
    }
    let perceptual = table(RenderingIntent::Perceptual);
    let relative = table(RenderingIntent::RelativeColorimetric);
    assert_ne!(
        perceptual.ink([30u8, 120, 200]),
        relative.ink([30u8, 120, 200]),
        "the test printer's intents separate differently"
    );
}

/// The compensated separation to the CMYK printer prints the colours of
/// `transicc -t 1 -b`'s within 1.5 levels of 8-bit sRGB: the working-space
/// black maps to the printer's black as a conversion source detects it,
/// the round trip of the perceptual black (`L*` 10.6), not its darkest
/// colorant (`L*` 8).  Near black the inks themselves may differ, as many
/// inks print one dark colour.
#[test]
fn a_compensated_cmyk_separation_maps_black_to_the_perceptual_black() {
    let profile = cmyk_printer();
    let table = profile
        .separation_table(
            WorkingSpace::Srgb,
            RenderingIntent::RelativeColorimetric,
            true,
        )
        .expect("separates");
    let mut srgb = ColorProfile::new_srgb();
    srgb.cicp = None;
    let printed = profile
        .profile
        .create_transform_f64(
            Layout::Rgba,
            &srgb,
            Layout::Rgb,
            TransformOptions {
                rendering_intent: moxcms::RenderingIntent::RelativeColorimetric,
                allow_use_cicp_transfer: false,
                ..TransformOptions::default()
            },
        )
        .unwrap();
    // transicc -n -i '*sRGB' -o printer-cmyk-v4.icc -t 1 -b, in percent.
    let reference: [([u8; 3], [f64; 4]); 3] = [
        ([0, 0, 0], [45.1850, 14.2992, 7.1870, 91.9539]),
        ([64, 64, 64], [3.0594, 2.5376, 2.5940, 72.5948]),
        ([0, 0, 255], [100.0, 100.0, 0.0, 0.0]),
    ];
    for (rgb, want) in reference {
        let got = table
            .ink(rgb.map(u16::from_u8))
            .map(|v| f64::from(v) / 65535.0);
        let mut colours = [0.0; 6];
        printed
            .transform(&[got, want.map(|w| w / 100.0)].concat(), &mut colours)
            .unwrap();
        let (ours, theirs) = colours.split_at(3);
        for (o, t) in ours.iter().zip(theirs) {
            assert!(
                (o - t).abs() * 255.0 <= 1.5,
                "{rgb:?}: ink {got:?} prints {ours:?}, transicc's {want:?} % prints {theirs:?}"
            );
        }
    }
}

/// A CMYK profile separates CMYK pages and converts no sRGB page; an RGB
/// one has no separation and converts no CMYK page.
#[test]
fn profiles_and_pages_of_other_colour_spaces_are_refused() {
    let cmyk = cmyk_printer();
    let rgb = OutputProfile::from_bytes(srgb_bytes()).unwrap();
    let relative = (RenderingIntent::RelativeColorimetric, false);
    let mut buffers = CanvasBuffers::default();
    assert_eq!(
        rgb.separation_table(WorkingSpace::Srgb, relative.0, relative.1)
            .unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Rgb)
    );
    let page = FinishedCanvas::Rgb8(rgb8(&PATCHES, None));
    assert_eq!(
        cmyk.convert(page, WorkingSpace::Srgb, relative, &mut buffers)
            .unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Cmyk)
    );
    let mut ink = Bitmap::<color::Cmyk8>::new(2, 1, 1, false);
    ink.row_samples_mut(0)
        .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let FinishedCanvas::Cmyk8(same) = cmyk
        .convert(
            FinishedCanvas::Cmyk8(ink),
            WorkingSpace::Srgb,
            relative,
            &mut buffers,
        )
        .unwrap()
    else {
        panic!("CMYK")
    };
    assert_eq!(same.row_samples(0), [1, 2, 3, 4, 5, 6, 7, 8]);
    let ink = Bitmap::<color::Cmyk8>::new(2, 1, 1, false);
    assert_eq!(
        rgb.convert(
            FinishedCanvas::Cmyk8(ink),
            WorkingSpace::Srgb,
            relative,
            &mut buffers
        )
        .unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Rgb)
    );
}
