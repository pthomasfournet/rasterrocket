//! Soft-masked objects in the host model against the soft mask's formula.
#![expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "test pages are a few tiles wide"
)]

use color::SampleDepth;

use super::PlaneValues;
use crate::page::reference::render_page;
use crate::page::test_support::{Rng, binned_bytes, object_over_tiles, page_plane, random_plane};
use crate::page::{MaskBits, PAGE_TILE, PageCommands, PixelBounds};

/// `x / M` rounded to nearest, for an independent statement of the blend.
fn rounded(x: u64, max: u64) -> u64 {
    (2 * x + max) / (2 * max)
}

/// The page model of `page` with every command painted plainly, objects
/// ignored: the painted values an object's blend starts from.
fn plain(page: &PageCommands) -> Vec<u16> {
    let mut p = page.clone();
    for c in &mut p.cmds {
        c.object = 0;
    }
    render_page(&p).expect("host model")
}

#[test]
fn an_object_blends_back_by_its_mask_at_8_and_16_bits() {
    let mut rng = Rng(0x50F7_3A5C);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        let (width, height) = (53, 37);
        for round in 0..12 {
            let paper = [0; 3].map(|_| rng.sample(max));
            let values = random_plane(&mut rng, width, height, depth);
            let mut page = PageCommands::new(width, height, depth, paper);
            let plane = page_plane(&mut page, &values);
            let rect = PixelBounds {
                x0: rng.below(20) - 3,
                y0: rng.below(15) - 3,
                x1: 20 + rng.below(40),
                y1: 15 + rng.below(30),
            };
            page.begin_object(plane, rect).expect("begin");
            // Overlapping translucent masks: the object is their sum.
            for _ in 0..3 {
                let data: Vec<u8> = (0..30 * 30).map(|_| rng.byte()).collect();
                let mask = page
                    .add_mask(&data, 30, 30, MaskBits::Coverage)
                    .expect("mask");
                let origin = (rng.below(40) - 5, rng.below(30) - 5);
                let all = PixelBounds {
                    x0: 0,
                    y0: 0,
                    x1: 60,
                    y1: 40,
                };
                page.push_mask(
                    mask,
                    origin,
                    all,
                    [0; 3].map(|_| rng.sample(max)),
                    rng.sample(max),
                )
                .expect("push");
            }
            page.end_object();
            assert_eq!(page.bytes(), binned_bytes(&page));
            let painted = plain(&page);
            let got = render_page(&page).expect("host model");
            for (i, (&g, &c)) in got.iter().zip(&painted).enumerate() {
                let (x, y) = (
                    i32::try_from(i / 3 % width as usize).unwrap(),
                    i32::try_from(i / 3 / width as usize).unwrap(),
                );
                let p = u64::from(paper[i % 3]);
                let m = u64::from(values[i / 3]);
                let inside = (rect.x0..=rect.x1).contains(&x) && (rect.y0..=rect.y1).contains(&y);
                let want = if inside {
                    rounded(u64::from(c) * m + p * (u64::from(max) - m), u64::from(max))
                } else {
                    u64::from(c)
                };
                assert_eq!(
                    u64::from(g),
                    want,
                    "{}-bit round {round}: ({x}, {y}) m {m}",
                    depth.bits()
                );
            }
        }
    }
}

#[test]
fn a_plane_is_held_as_the_rectangle_that_differs() {
    let mut page = PageCommands::new(6, 4, SampleDepth::Eight, [255; 3]);
    let mut v = vec![7u8; 24];
    v[6 + 2] = 1;
    v[2 * 6 + 4] = 9;
    let before = page.bytes.len();
    let plane = page.add_mask_plane(PlaneValues::Eight(&v)).expect("plane");
    assert_eq!(
        page.bytes.len() - before,
        6,
        "the 3 × 2 rectangle of (2, 1) - (4, 2)"
    );
    page.begin_object(
        plane,
        PixelBounds {
            x0: 0,
            y0: 0,
            x1: 5,
            y1: 3,
        },
    )
    .expect("begin");
    let object = page.modifiers.object.expect("an object");
    for y in 0..4 {
        for x in 0..6 {
            assert_eq!(
                page.object_mask(object, x, y),
                Some(u16::from(v[(y * 6 + x) as usize]))
            );
        }
    }
    let uniform = page
        .add_mask_plane(PlaneValues::Eight(&[3; 24]))
        .expect("plane");
    page.begin_object(
        uniform,
        PixelBounds {
            x0: 1,
            y0: 1,
            x1: 2,
            y1: 2,
        },
    )
    .expect("begin");
    let object = page.modifiers.object.expect("an object");
    assert_eq!(page.object_mask(object, 1, 1), Some(3));
    assert_eq!(
        page.object_mask(object, 0, 1),
        None,
        "outside the object's rectangle"
    );
    let mut page16 = PageCommands::new(3, 1, SampleDepth::Sixteen, [0; 3]);
    let wide = page16
        .add_mask_plane(PlaneValues::Sixteen(&[0, 0x1234, 0]))
        .expect("plane");
    page16
        .begin_object(
            wide,
            PixelBounds {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 0,
            },
        )
        .expect("begin");
    let object = page16.modifiers.object.expect("an object");
    assert_eq!(page16.object_mask(object, 1, 0), Some(0x1234));
    assert_eq!(page16.object_mask(object, 2, 0), Some(0));
}

/// An opaque outline covering whole tiles inside an object, over a
/// painted page: the object's blend needs the pixels below it, so it hides
/// nothing, and its members split by binning (absent from some tiles, an
/// object spanning tiles) blend from the pixels before the object in each.
#[test]
fn an_object_hides_nothing_and_blends_from_its_first_item_in_each_tile() {
    let tile = PAGE_TILE as i32;
    let (width, height) = (3 * PAGE_TILE, 2 * PAGE_TILE);
    let page = object_over_tiles(SampleDepth::Eight);
    let values: Vec<u8> = (0..width * height)
        .map(|i| u8::try_from(i % 256).unwrap())
        .collect();
    let bins = page.bin().expect("bin");
    let n_tiles = (width / PAGE_TILE * (height / PAGE_TILE)) as usize;
    assert!(
        (0..n_tiles).all(
            |t| bins.items[bins.offsets[t] as usize..bins.offsets[t + 1] as usize]
                .iter()
                .any(|i| i[0] == 0)
        ),
        "the stripe below the object is listed in every tile"
    );
    let got = render_page(&page).expect("host model");
    let w = width as i32;
    for y in 0..2 * tile {
        for x in 0..3 * tile {
            let o = (y * w + x) as usize * 3;
            let below = if (4..3 * tile - 4).contains(&x) {
                [10, 20, 30]
            } else {
                [200, 100, 50]
            };
            let m = u32::from(values[(y * w + x) as usize]);
            let want = if y == 3 && (tile..2 * tile).contains(&x) {
                [1, 2, 3]
            } else {
                [0, 1, 2].map(|ch| {
                    let painted = [0u32, 0, 255][ch];
                    u16::try_from(rounded(u64::from(painted * m + below[ch] * (255 - m)), 255))
                        .unwrap()
                })
            };
            assert_eq!(got[o..o + 3], want, "({x}, {y})");
        }
    }
}
