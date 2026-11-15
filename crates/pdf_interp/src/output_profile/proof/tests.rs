use super::*;

/// The synthetic CMYK printer of `tests/fixtures/icc`.
fn press() -> Arc<OutputProfile> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/icc/printer-cmyk-v4.icc"
    );
    Arc::new(
        OutputProfile::from_bytes(std::fs::read(path).expect("the fixture is present"))
            .expect("a CMYK output profile"),
    )
}

/// A `width × 1` 8-bit ink canvas of `px`.
fn ink8(px: &[[u8; 4]]) -> FinishedCanvas {
    let width = u32::try_from(px.len()).unwrap();
    let mut bmp = raster::Bitmap::<Cmyk8>::new(width, 1, 1, false);
    for (dst, src) in bmp
        .row_samples_mut(0)
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(px)
    {
        *dst = *src;
    }
    FinishedCanvas::Cmyk8(bmp)
}

/// Paper, solid black, solid cyan and red (magenta + yellow) ink.
const INKS: [[u8; 4]; 4] = [
    [0, 0, 0, 0],
    [0, 0, 0, 255],
    [255, 0, 0, 0],
    [0, 255, 255, 0],
];

/// The working-space proof of the press agrees with Little CMS's
/// `transicc -n -i printer-cmyk-v4.icc -o '*sRGB' -t 1` (relative) and
/// `-t 3` (absolute) within two levels: relative colorimetry maps the
/// paper to white, absolute colorimetry shows the paper's tint.
#[test]
fn the_proof_of_the_press_agrees_with_transicc() {
    let proof = SoftProof::new(press(), None).expect("a CMYK press");
    let reference: [(ProofIntent, [[f64; 3]; 4]); 2] = [
        (
            ProofIntent::RelativeColorimetric,
            [
                [254.99, 255.0, 255.05],
                [23.51, 23.51, 23.51],
                [104.85, 247.44, 247.49],
                [239.58, 77.37, 77.39],
            ],
        ),
        (
            ProofIntent::AbsoluteColorimetric,
            [
                [252.88, 250.95, 245.19],
                [23.22, 22.95, 22.14],
                [108.48, 243.49, 238.0],
                [235.96, 76.07, 73.67],
            ],
        ),
    ];
    let mut buffers = CanvasBuffers::default();
    for (intent, want) in reference {
        let FinishedCanvas::Rgb8(rgb) = proof
            .convert(ink8(&INKS), intent, &mut buffers)
            .expect("converts")
        else {
            panic!("an RGB proof")
        };
        for (got, want) in rgb.row_samples(0).as_chunks::<3>().0.iter().zip(want) {
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (f64::from(*g) - w).abs() <= 2.0,
                    "{intent}: {got:?} for {want:?}"
                );
            }
        }
    }
}

/// The 16-bit proof of an ink is the 8-bit one within a level, and the
/// table of an intent is built once.
#[test]
fn sixteen_bit_proofs_agree_with_eight_bit_ones() {
    let proof = SoftProof::new(press(), None).unwrap();
    let mut buffers = CanvasBuffers::default();
    let intent = ProofIntent::AbsoluteColorimetric;
    let FinishedCanvas::Rgb8(eight) = proof.convert(ink8(&INKS), intent, &mut buffers).unwrap()
    else {
        panic!("RGB")
    };
    let ink16 = || {
        let mut bmp = raster::Bitmap::<Cmyk16>::new(4, 1, 1, false);
        for (dst, &v) in bmp.row_samples_mut(0).iter_mut().zip(INKS.as_flattened()) {
            *dst = u16::from(v) * 257;
        }
        FinishedCanvas::Cmyk16(bmp)
    };
    let FinishedCanvas::Rgb16(sixteen) = proof.convert(ink16(), intent, &mut buffers).unwrap()
    else {
        panic!("RGB")
    };
    for (&e, &s) in eight.row_samples(0).iter().zip(sixteen.row_samples(0)) {
        assert!(
            (f64::from(e) - f64::from(s) / 257.0).abs() <= 1.0,
            "{e} {s}"
        );
    }
    let (a, b) = (proof.table(intent).unwrap(), proof.table(intent).unwrap());
    assert!(Arc::ptr_eq(&a, &b), "the table is built once");
    assert!(!Arc::ptr_eq(
        &a,
        &proof.table(ProofIntent::RelativeColorimetric).unwrap()
    ));
}

/// A gray display profile proofs into a gray canvas.
#[test]
fn a_gray_display_proofs_to_gray() {
    let gray = moxcms::ColorProfile::new_gray_with_gamma(2.2)
        .encode()
        .unwrap();
    let display = Arc::new(OutputProfile::from_bytes(gray).unwrap());
    let proof = SoftProof::new(press(), Some(display)).unwrap();
    let mut buffers = CanvasBuffers::default();
    let FinishedCanvas::Gray8(out) = proof
        .convert(ink8(&INKS), ProofIntent::RelativeColorimetric, &mut buffers)
        .unwrap()
    else {
        panic!("a gray proof")
    };
    let px = out.row_samples(0);
    assert!(px[0] >= 253, "paper is white: {px:?}");
    assert!(px[1] < 40, "solid black is dark: {px:?}");
}

/// At the 16-bit ink nearest a node the table's interpolation is the
/// node's value within a level: a node lies at `v · 65535 / 16`, which
/// 16-bit ink reaches only within half a level.
#[test]
fn the_interpolation_reproduces_the_nodes() {
    let proof = SoftProof::new(press(), None).unwrap();
    let table = proof.table(ProofIntent::RelativeColorimetric).unwrap();
    let node = |i: u32| u16::try_from((i * u32::from(u16::MAX) + SPAN / 2) / SPAN).unwrap();
    for (c, m, y, k) in [(0, 0, 0, 0), (16, 16, 16, 16), (3, 7, 11, 5), (16, 0, 9, 2)] {
        let mut out = [0u16; 3];
        table.eval([c, m, y, k].map(node), &mut out);
        let at = [c, m, y, k]
            .iter()
            .zip(&STRIDES)
            .map(|(&i, &s)| i as usize * s)
            .sum::<usize>()
            * 3;
        for (o, n) in out.iter().zip(&table.nodes[at..at + 3]) {
            assert!(o.abs_diff(*n) <= 1, "node {c} {m} {y} {k}: {out:?}");
        }
    }
}

/// The press is CMYK and the display not; the canvas is ink.
#[test]
fn proofs_of_other_colour_spaces_are_refused() {
    let srgb = Arc::new(
        OutputProfile::from_bytes(moxcms::ColorProfile::new_srgb().encode().unwrap()).unwrap(),
    );
    assert_eq!(
        SoftProof::new(Arc::clone(&srgb), None).unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Rgb)
    );
    assert_eq!(
        SoftProof::new(press(), Some(press())).unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Cmyk)
    );
    let proof = SoftProof::new(press(), Some(srgb)).unwrap();
    let page = FinishedCanvas::Rgb8(raster::Bitmap::new(1, 1, 1, false));
    assert_eq!(
        proof
            .convert(
                page,
                ProofIntent::RelativeColorimetric,
                &mut CanvasBuffers::default()
            )
            .unwrap_err(),
        OutputProfileError::Mismatch(OutputSpace::Cmyk)
    );
}
