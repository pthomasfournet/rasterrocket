use color::SampleDepth;

use super::super::reference::render_page;
use super::super::test_support::{
    Rng, affine_tables, binned_bytes, push_random_images, table_entry,
};
use super::super::{PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds};
use super::{ImageDraw, ImageLayout, ImageMask, SampleMap, Samples};

/// `x / max` rounded to nearest; `max` is odd, so no quotient is a half.
fn rounded(x: u64, max: u64) -> u64 {
    (2 * x + max) / (2 * max)
}

/// The tables mapping pixel `(x, y)` of `0..w × 0..h` to the centre of
/// sample `(x, y)`: one sample per pixel.
fn identity(w: i32, h: i32) -> (Vec<u32>, Vec<u32>) {
    affine_tables((0, 0, w, h), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
}

/// The one pixel of a 1 × 1 page of `depth` on `paper`, painted by a 1 × 1
/// RGB image `rgb` (samples of the page's depth) under a mask of value
/// `mask` (of the page's depth), composited directly or through the pipe
/// at `a_input`.
fn one_pixel(
    depth: SampleDepth,
    paper: u16,
    rgb: [u16; 3],
    mask: u16,
    a_input: u16,
    direct: bool,
) -> Vec<u16> {
    let mut page = PageCommands::new(1, 1, depth, [paper; 3]);
    let (cols, rows) = identity(1, 1);
    let deep = depth == SampleDepth::Sixteen;
    let add = |page: &mut PageCommands, s: &[u16], ch| {
        if deep {
            page.add_samples(Samples::Sixteen(s), 1, 1, ch)
        } else {
            let s: Vec<u8> = s.iter().map(|&v| u8::try_from(v).unwrap()).collect();
            page.add_samples(Samples::Eight(&s), 1, 1, ch)
        }
        .expect("add samples")
    };
    let samples = add(&mut page, &rgb, 3);
    let mask = add(&mut page, &[mask], 1);
    page.push_image(&ImageDraw {
        samples,
        layout: ImageLayout::Rgb,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
            cols: &cols,
            rows: &rows,
            filter_x: false,
            filter_y: false,
        },
        mask: ImageMask::Taps(mask),
        bounds: PixelBounds {
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
        },
        colour: [0; 3],
        a_input,
        direct,
    })
    .expect("push image");
    render_page(&page).expect("render")
}

#[test]
fn the_two_composites_at_their_boundaries_at_8_and_16_bits() {
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        let top = u64::from(max);
        let paper = max / 3;
        let rgb = [max, max / 2 + 1, 7];
        for shape in [max, max - 1, max / 2, 1, 0] {
            for a_input in [max, max - 1, max / 2, 1] {
                for direct in [true, false] {
                    let got = one_pixel(depth, paper, rgb, shape, a_input, direct);
                    let want: Vec<u16> = rgb
                        .iter()
                        .map(|&c| {
                            let (src, dst, sh) = (u64::from(c), u64::from(paper), u64::from(shape));
                            let v = if sh == 0 {
                                dst
                            } else if direct {
                                if sh == top {
                                    src
                                } else {
                                    (src * sh + dst * (top - sh)) / top
                                }
                            } else {
                                let alpha = rounded(u64::from(a_input) * sh, top);
                                rounded((top - alpha) * dst + alpha * src, top)
                            };
                            u16::try_from(v).unwrap()
                        })
                        .collect();
                    assert_eq!(
                        got,
                        want,
                        "{}-bit shape {shape} alpha {a_input} direct {direct}",
                        depth.bits()
                    );
                }
            }
        }
    }
}

/// A direct composite at less than a full shape truncates where the pipe
/// rounds: the two differ, so a command taking the other composite fails
/// the test above.
#[test]
fn the_direct_composite_truncates() {
    for (depth, paper, colour, shape, direct, pipe) in [
        (SampleDepth::Eight, 1, 2, 254, 1, 2),
        (SampleDepth::Eight, 200, 100, 3, 198, 199),
        (SampleDepth::Sixteen, 1, 2, 65_534, 1, 2),
    ] {
        let max = depth.max_sample();
        let d = one_pixel(depth, paper, [colour; 3], shape, max, true);
        let p = one_pixel(depth, paper, [colour; 3], shape, max, false);
        assert_eq!((d[0], p[0]), (direct, pipe), "{}-bit", depth.bits());
    }
}

/// A 2 × 2 gray image `[10, 50; 90, 250]` over a 4 × 4 page, magnified
/// twice and drawn by `filter`, then sampled at pixel `(x, y)`.
fn two_by_two(filter: (bool, bool), at: (usize, usize)) -> u16 {
    let mut page = PageCommands::new(4, 4, SampleDepth::Eight, [0; 3]);
    let samples = page
        .add_samples(Samples::Eight(&[10, 50, 90, 250]), 2, 2, 1)
        .expect("add");
    let (cols, rows) = affine_tables((0, 0, 4, 4), [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]);
    page.push_image(&ImageDraw {
        samples,
        layout: ImageLayout::Gray,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 4,
            cols: &cols,
            rows: &rows,
            filter_x: filter.0,
            filter_y: filter.1,
        },
        mask: ImageMask::None,
        bounds: PixelBounds {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 3,
        },
        colour: [0; 3],
        a_input: 255,
        direct: true,
    })
    .expect("push");
    render_page(&page).expect("render")[(at.1 * 4 + at.0) * 3]
}

#[test]
fn taps_and_weights_follow_the_tables() {
    // Nearest: pixel 1 lies at source 0.75, in sample 0.
    assert_eq!(two_by_two((false, false), (1, 1)), 10);
    assert_eq!(two_by_two((false, false), (2, 1)), 50);
    // Bilinear columns: pixel 1 at 0.75 is 0.25 past sample 0's centre:
    // weight 64 of 256 on sample 1.
    assert_eq!(
        u32::from(two_by_two((true, false), (1, 0))),
        (10 * 192 * 256 + 50 * 64 * 256 + 32_768_u32) >> 16
    );
    // Pixel 0 at 0.25 lies before the first centre: both taps clamp.
    assert_eq!(two_by_two((true, true), (0, 0)), 10);
    // Both axes at (1.25, 1.25): weights 192 on the second sample.
    let row = |l: u32, r: u32| l * 64 + r * 192;
    assert_eq!(
        u32::from(two_by_two((true, true), (2, 2))),
        (row(10, 50) * 64 + row(90, 250) * 192 + 32_768) >> 16
    );
}

#[test]
fn pixels_off_the_image_paint_nothing_and_an_own_mask_clamps() {
    let mut page = PageCommands::new(8, 1, SampleDepth::Eight, [3; 3]);
    let samples = page
        .add_samples(Samples::Eight(&[200, 201]), 2, 1, 1)
        .expect("add");
    let mask = page
        .add_samples(Samples::Eight(&[255, 64]), 2, 1, 1)
        .expect("add");
    // The image: pixel x samples source x − 2, so pixels 2 and 3 only.
    let (cols, rows) = affine_tables((0, 0, 8, 1), [1.0, 0.0, -2.0, 0.0, 1.0, 0.0]);
    // The mask: pixel x samples x over its own rectangle 0..2, pixels
    // past it clamped onto its last column.
    let (mcols, mrows) = identity(2, 1);
    page.push_image(&ImageDraw {
        samples,
        layout: ImageLayout::Gray,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 8,
            y1: 1,
            cols: &cols,
            rows: &rows,
            filter_x: false,
            filter_y: false,
        },
        mask: ImageMask::Own(
            mask,
            SampleMap {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 1,
                cols: &mcols,
                rows: &mrows,
                filter_x: false,
                filter_y: false,
            },
        ),
        bounds: PixelBounds {
            x0: 0,
            y0: 0,
            x1: 7,
            y1: 0,
        },
        colour: [0; 3],
        a_input: 255,
        direct: true,
    })
    .expect("push");
    let px: Vec<u16> = render_page(&page)
        .expect("render")
        .chunks(3)
        .map(|p| p[0])
        .collect();
    let blend = |c: u32, d: u32, s: u32| u16::try_from((c * s + d * (255 - s)) / 255).unwrap();
    assert_eq!(px, [3, 3, blend(200, 3, 64), blend(201, 3, 64), 3, 3, 3, 3]);
}

#[test]
fn stencils_paint_their_colour_by_the_complement_of_their_samples() {
    let mut page = PageCommands::new(3, 1, SampleDepth::Sixteen, [65_535; 3]);
    let samples = page
        .add_samples(Samples::Eight(&[0, 255, 128]), 3, 1, 1)
        .expect("add");
    let (cols, rows) = identity(3, 1);
    page.push_image(&ImageDraw {
        samples,
        layout: ImageLayout::Stencil,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 1,
            cols: &cols,
            rows: &rows,
            filter_x: false,
            filter_y: false,
        },
        mask: ImageMask::None,
        bounds: PixelBounds {
            x0: 0,
            y0: 0,
            x1: 2,
            y1: 0,
        },
        colour: [1, 2, 3],
        a_input: 65_535,
        direct: true,
    })
    .expect("push");
    let out = render_page(&page).expect("render");
    let shape = u32::from(65_535 - 128 * 257_u16);
    let partial = |c: u32| u16::try_from((c * shape + 65_535 * (65_535 - shape)) / 65_535).unwrap();
    assert_eq!(
        out,
        [
            1,
            2,
            3,
            65_535,
            65_535,
            65_535,
            partial(1),
            partial(2),
            partial(3)
        ]
    );
}

#[test]
fn sixteen_bit_samples_are_read_low_byte_first() {
    let mut page = PageCommands::new(2, 1, SampleDepth::Sixteen, [0; 3]);
    let samples = page
        .add_samples(Samples::Sixteen(&[0x1234, 0xFEDC]), 2, 1, 1)
        .expect("add");
    let eight = page
        .add_samples(Samples::Eight(&[0x12, 0xFE]), 2, 1, 1)
        .expect("add");
    let (cols, rows) = identity(2, 1);
    for (s, want) in [
        (samples, [0x1234, 0xFEDC]),
        (eight, [0x12 * 257, 0xFE * 257]),
    ] {
        let mut page = page.clone();
        page.push_image(&ImageDraw {
            samples: s,
            layout: ImageLayout::Gray,
            map: SampleMap {
                x0: 0,
                y0: 0,
                x1: 2,
                y1: 1,
                cols: &cols,
                rows: &rows,
                filter_x: false,
                filter_y: false,
            },
            mask: ImageMask::None,
            bounds: PixelBounds {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 0,
            },
            colour: [0; 3],
            a_input: 65_535,
            direct: true,
        })
        .expect("push");
        let out = render_page(&page).expect("render");
        assert_eq!([out[0], out[3]], want);
    }
}

#[test]
fn the_byte_count_is_that_of_the_binned_inputs_with_images() {
    let mut rng = Rng(0x1A6E_5BCD);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let mut page = PageCommands::new(150, 90, depth, [depth.max_sample(); 3]);
        for i in 0..40 {
            push_random_images(&mut rng, &mut page, 1);
            assert_eq!(page.bytes(), binned_bytes(&page), "after push {i}");
        }
    }
}

#[test]
fn samples_past_the_limit_are_refused_and_leave_the_page_unchanged() {
    let mut page = PageCommands::new(16, 16, SampleDepth::Eight, [255; 3]);
    let before = page.bytes();
    let n = PAGE_COMMANDS_MAX_BYTES / 2 + 1;
    let big = vec![0u16; n];
    assert_eq!(
        page.add_samples(Samples::Sixteen(&big), u32::try_from(n).unwrap(), 1, 1),
        Err(PageCommandsError::TooLarge)
    );
    assert_eq!(page.bytes(), before);
}

#[test]
fn a_draw_beside_the_page_or_its_tables_pushes_nothing() {
    let mut page = PageCommands::new(8, 8, SampleDepth::Eight, [255; 3]);
    let samples = page
        .add_samples(Samples::Eight(&[1]), 1, 1, 1)
        .expect("add");
    let (cols, rows) = identity(4, 4);
    let draw = |bounds| ImageDraw {
        samples,
        layout: ImageLayout::Gray,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 4,
            cols: &cols,
            rows: &rows,
            filter_x: false,
            filter_y: false,
        },
        mask: ImageMask::None,
        bounds,
        colour: [0; 3],
        a_input: 255,
        direct: true,
    };
    let b = |x0, y0, x1, y1| PixelBounds { x0, y0, x1, y1 };
    page.push_image(&draw(b(4, 0, 7, 7))).expect("push");
    page.push_image(&draw(b(-5, -5, -1, -1))).expect("push");
    assert!(page.is_empty());
    page.push_image(&draw(b(2, 2, 100, 100))).expect("push");
    let c = page.cmds()[0];
    assert_eq!((c.x0, c.y0, c.x1, c.y1), (2, 2, 3, 3));
}

#[test]
#[should_panic(expected = "16-bit samples on an 8-bit page")]
fn sixteen_bit_samples_on_an_8_bit_page_panic() {
    let mut page = PageCommands::new(2, 2, SampleDepth::Eight, [255; 3]);
    let samples = page
        .add_samples(Samples::Sixteen(&[1]), 1, 1, 1)
        .expect("add");
    let cols = table_entry(0.5, 0.0);
    let rows = table_entry(0.0, 0.5);
    let _ = page.push_image(&ImageDraw {
        samples,
        layout: ImageLayout::Gray,
        map: SampleMap {
            x0: 0,
            y0: 0,
            x1: 1,
            y1: 1,
            cols: &cols,
            rows: &rows,
            filter_x: false,
            filter_y: false,
        },
        mask: ImageMask::None,
        bounds: PixelBounds {
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
        },
        colour: [0; 3],
        a_input: 255,
        direct: true,
    });
}
