use color::{Gray8, Gray16, Sample};
use raster::Bitmap;

use super::*;

#[test]
fn a_page_past_the_deskew_bound_is_refused_and_left_as_it_is() {
    for (w, h) in [(DESKEW_MAX_SIDE + 1, 2), (2, DESKEW_MAX_SIDE + 1)] {
        let mut img = Bitmap::<Gray8>::new_filled(w, h, 1, false, 7);
        let err = apply(&mut img, BackendPolicy::CpuOnly).expect_err("over the bound");
        assert!(err.to_string().contains("32768 px a side"), "{err}");
        assert!(detect(&img).is_err());
        assert!(img.data().iter().all(|&v| v == 7));
        let mut out = vec![1u8; img.data().len()];
        assert!(apply_into(&img, &mut out, w as usize, BackendPolicy::CpuOnly).is_err());
        assert!(out.iter().all(|&v| v == 1), "out is untouched");
    }
    let mut img = Bitmap::<Gray8>::new_filled(DESKEW_MAX_SIDE, 2, 1, false, 255);
    assert!(apply(&mut img, BackendPolicy::CpuOnly).is_ok());
}

/// FNV-1a 64 of `bytes`.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A `w × h` page of text-like bars on white, lines `pitch` px apart and
/// `7 · pitch / 16` px tall, the bars `margin` px from the page's edges
/// and broken where `gap(row, col)`,
/// turned by `deg` degrees clockwise on the page (y downward) about its
/// centre: a page whose lines [`detect`] measures at `−deg`.
fn tilted(
    (w, h, margin): (u32, u32, u32),
    (deg, pitch): (f32, u32),
    gap: impl Fn(u32, u32) -> bool,
) -> Bitmap<Gray8> {
    let (sin, cos) = deg.to_radians().sin_cos();
    let mut img = Bitmap::<Gray8>::new_filled(w, h, 1, false, 250);
    #[expect(clippy::cast_precision_loss, reason = "sides below 2^12")]
    let (cx, cy, limit) = (w as f32 / 2.0, h as f32 / 2.0, 2.0 * w.max(h) as f32);
    for y in 0..h {
        for (x, px) in img.row_bytes_mut(y).iter_mut().enumerate().take(w as usize) {
            #[expect(clippy::cast_precision_loss, reason = "coordinates below 2^12")]
            let (dx, dy) = (x as f32 - cx, y as f32 - cy);
            let across = cos.mul_add(dy, -sin * dx) + cy;
            let along = cos.mul_add(dx, sin * dy) + cx;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the coordinates are clamped to [0, 2 · side] first"
            )]
            let (row, col) = (
                across.clamp(0.0, limit) as u32,
                along.clamp(0.0, limit) as u32,
            );
            if (margin..h - margin).contains(&row)
                && (margin..w - margin).contains(&col)
                && 16 * (row % pitch) < 7 * pitch
                && !gap(row, col)
            {
                *px = 20;
            }
        }
    }
    img
}

/// A 600 × 400 page of bars broken every 23 columns, tilted by `deg`.
fn tilted_text(deg: f32) -> Bitmap<Gray8> {
    tilted((600, 400, 40), (deg, 16), |_, col| col % 23 == 0)
}

/// A `w × h` page of text-like lines 36 px apart, about 12-point text at
/// 150 dpi, broken into words of irregular lengths, tilted by `deg`.  The
/// breaks do not line up from line to line, so the page has no alignment
/// but its lines.
fn tilted_page(w: u32, h: u32, margin: u32, deg: f32) -> Bitmap<Gray8> {
    tilted((w, h, margin), (deg, 36), |row, col| {
        let cell = (row / 36).wrapping_mul(0x9e37_79b9) ^ (col / 11).wrapping_mul(0x85eb_ca6b);
        cell.wrapping_mul(0xc2b2_ae35) >> 29 == 0
    })
}

/// `img` turned upside down about its horizontal axis.
fn flipped(img: &Bitmap<Gray8>) -> Bitmap<Gray8> {
    let mut out = Bitmap::<Gray8>::new(img.width, img.height, 1, false);
    for y in 0..img.height {
        let w = img.width as usize;
        out.row_bytes_mut(img.height - 1 - y)[..w].copy_from_slice(&img.row_bytes(y)[..w]);
    }
    out
}

/// The 8-bit rotation of fixed inputs and the deskew of a fixed page.
#[test]
fn eight_bit_deskew_matches_its_golden_digests() {
    let page = tilted_text(2.5);
    let mut fixed = tilted_text(2.5);
    let skew = apply(&mut fixed, BackendPolicy::CpuOnly)
        .expect("within bounds")
        .expect("a confident angle");
    let got = [
        digest(rotate::rotate_cpu(&page, 3.25).data()),
        digest(rotate::rotate_cpu(&page, -6.0).data()),
        digest(fixed.data()),
    ];
    assert_ne!(fixed.data(), page.data(), "the page is deskewed");
    assert_eq!(got, GOLDEN, "{got:#x?} at {skew:?}");
}

const GOLDEN: [u64; 3] = [
    0x7e57_dda9_e717_4282,
    0xe770_cd2a_b45a_c65e,
    0x6d61_a832_3158_a711,
];

/// A 16-bit page is measured on its 8-bit levels and rotated at 16 bits:
/// widened from an 8-bit page, it finds the same angle, and its rotation
/// rounds to within one 8-bit level of the 8-bit rotation.
#[test]
fn a_16_bit_page_deskews_like_its_8_bit_levels() {
    let eight = tilted_text(2.5);
    let wide: Vec<u16> = eight.data().iter().map(|&v| u16::from_u8(v)).collect();
    let mut sixteen = Bitmap::<Gray16>::from_packed(eight.width, eight.height, wide);
    let skew = detect(&eight).expect("within bounds").expect("confident");
    assert!(skew.angle_deg.abs() >= MIN_CORRECT_DEG, "{skew:?}");
    assert_eq!(detect(&sixteen).expect("within bounds"), Some(skew));
    let mut fixed = tilted_text(2.5);
    let _ = apply(&mut fixed, BackendPolicy::CpuOnly).expect("within bounds");
    let applied = apply(&mut sixteen, BackendPolicy::ForceCuda).expect("within bounds");
    assert_eq!(applied, Some(skew));
    let mut finer = false;
    for (&deep, &narrow) in sixteen.data().iter().zip(fixed.data()) {
        assert!(deep.to_u8().abs_diff(narrow) <= 1, "{deep} vs {narrow}");
        finer |= !deep.is_multiple_of(257);
    }
    assert!(finer, "the blend keeps levels between the 8-bit ones");
    let mut big = Bitmap::<Gray16>::new(DESKEW_MAX_SIDE + 1, 2, 1, false);
    assert!(apply(&mut big, BackendPolicy::CpuOnly).is_err());
}

/// On a page `W` px wide the measured angle is within the documented
/// `2 / W` radians of the true one, with a tolerance of a quarter of that
/// for the quantisation of the synthetic page, over a grid of angles
/// across the sweep; the confidence is at least [`MIN_CONFIDENCE`].
#[test]
fn the_angle_is_within_the_documented_resolution() {
    for w in [1200u32, 1600] {
        #[expect(clippy::cast_precision_loss, reason = "w is exact in f32")]
        let bound = (2.5 / w as f32).to_degrees();
        for i in 0..40u8 {
            let deg = f32::from(i).mul_add(0.3037, -6.0);
            let skew = detect(&tilted_page(w, w * 11 / 16 / 4 * 4, 60, deg))
                .expect("within bounds")
                .unwrap_or_else(|| panic!("no angle at {deg}°"));
            assert!(
                (skew.angle_deg + deg).abs() <= bound,
                "{w} px, {deg}°: measured {skew:?}, bound {bound}°"
            );
            assert!(skew.confidence >= MIN_CONFIDENCE, "{skew:?}");
        }
    }
}

/// A page and its upside-down image, whose lines tilt the other way,
/// measure opposite angles of the same confidence, with ink up to the top
/// and bottom edges.
#[test]
fn a_mirrored_page_measures_the_opposite_angle() {
    for deg in [1.7, 3.2, 5.1] {
        let page = tilted_page(800, 560, 2, deg);
        let up = detect(&page).expect("within bounds").expect("confident");
        let down = detect(&flipped(&page))
            .expect("within bounds")
            .expect("confident");
        assert_eq!(down.angle_deg, -up.angle_deg, "{deg}°: {up:?} vs {down:?}");
        assert_eq!(down.confidence, up.confidence, "{deg}°");
    }
}

/// A page whose profile varies little with the angle, noise or a smooth
/// photograph-like gradient, has no confident angle and is not rotated.
#[test]
fn a_page_without_lines_is_left_as_it_is() {
    let mut noise = Bitmap::<Gray8>::new(640, 480, 1, false);
    let mut state = 0x2545_f491_u32;
    for px in noise.data_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *px = state.to_le_bytes()[1];
    }
    let mut photo = Bitmap::<Gray8>::new(640, 480, 1, false);
    for y in 0..480u32 {
        for (x, px) in photo.row_bytes_mut(y).iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "coordinates below 2^10")]
            let (dx, dy) = (x as f32 - 260.0, y as f32 - 300.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to [0, 255]"
            )]
            {
                *px = (dx.hypot(dy) * 0.6).clamp(0.0, 255.0) as u8;
            }
        }
    }
    for (name, page) in [("noise", noise), ("photo", photo)] {
        let mut img = Bitmap::<Gray8>::from_packed(page.width, page.height, page.data().to_vec());
        assert_eq!(
            apply(&mut img, BackendPolicy::CpuOnly).expect("within bounds"),
            None,
            "{name}"
        );
        assert_eq!(img.data(), page.data(), "{name} is unchanged");
    }
}

/// A best angle within one sweep step of either end is declined: the
/// score may still rise beyond the range searched.
#[test]
fn a_maximum_at_the_ends_of_the_sweep_is_declined() {
    for deg in [-8.5, -6.9, 6.9, 8.5] {
        let page = tilted_page(800, 600, 40, deg);
        let mut img = Bitmap::<Gray8>::from_packed(page.width, page.height, page.data().to_vec());
        assert_eq!(
            apply(&mut img, BackendPolicy::CpuOnly).expect("within bounds"),
            None,
            "{deg}°"
        );
        assert_eq!(img.data(), page.data(), "{deg}° is unchanged");
    }
    let inside = detect(&tilted_page(800, 600, 40, 6.4)).expect("within bounds");
    assert!(
        inside.is_some_and(|s| (s.angle_deg + 6.4).abs() < 0.1),
        "{inside:?}"
    );
}
