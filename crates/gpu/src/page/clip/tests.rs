//! Clips in the host model against the CPU's clip, fill and glyph blit.

use color::{Pixel, Rgb8, Rgb16, Sample, SampleDepth};
use raster::types::BlendMode;
use raster::{Bitmap, Clip, ClipRows, GlyphBitmap, PipeSrc, PipeState, RowClip, TransferSet};

use crate::page::reference::render_page;
use crate::page::test_support::{
    IDENTITY, Rng, binned_bytes, clip_bounds, page_clip, push_fill, random_clip, random_path,
};
use crate::page::{CoverageRows, MaskBits, PageCommands, PixelBounds};

/// The region counts the tests clip by: one, several, and past the
/// CPU's fold of eight.
const REGIONS: [usize; 5] = [1, 2, 3, 5, 9];

/// `page` rendered by the host model against `want`, samples widened to
/// 16 bits.
fn assert_model_is(page: &PageCommands, want: &[u16], what: &str) {
    let got = render_page(page).expect("host model");
    let width = page.width() as usize;
    if let Some(i) = (0..want.len()).find(|&i| got[i] != want[i]) {
        panic!(
            "{what}: pixel ({}, {}) sample {} is {} in the model, {} on the CPU",
            i / 3 % width,
            i / 3 / width,
            i % 3,
            got[i],
            want[i]
        );
    }
}

/// A bitmap of `paper` and its samples widened to 16 bits.
fn paper_bitmap<P: Pixel>(width: u32, height: u32, paper: [u16; 3]) -> Bitmap<P> {
    let s = |v: u16| P::Sample::from_signed(i64::from(v));
    let data = (0..width * height).flat_map(|_| paper.map(s)).collect();
    Bitmap::<P>::from_packed(width, height, data)
}

fn widened<P: Pixel>(bitmap: &Bitmap<P>) -> Vec<u16> {
    bitmap
        .data()
        .iter()
        .map(|&v| u16::try_from(v.signed()).unwrap())
        .collect()
}

fn pipe<S: Sample>(alpha: u16) -> PipeState<'static, S> {
    PipeState {
        blend_mode: BlendMode::Normal,
        a_input: S::from_signed(i64::from(alpha)),
        transfer: TransferSet::identity_rgb(),
        backdrop: None,
        overprint: None,
        gray: None,
    }
}

#[test]
fn a_clip_scales_as_the_cpu_clip_scales_shapes_and_rows() {
    let mut rng = Rng(0xC11B_5CA1);
    let (width, height) = (70, 50);
    for n in REGIONS {
        let clip = random_clip(&mut rng, width, height, n);
        let mut page = PageCommands::new(width, height, SampleDepth::Eight, [255; 3]);
        let id = page_clip(&mut page, &clip).expect("a path clip");
        let mut rows = ClipRows::new(&clip, 0, 0, 69, 49);
        for y in 0..50 {
            let row = rows.row(y);
            for x in 0..70 {
                // Combination B: the clip's coverage of a full shape.
                let want = row.map_or(0, |r: RowClip<'_>| r.coverage(x));
                let inside = (clip.x_min_i..=clip.x_max_i).contains(&x)
                    && (clip.y_min_i..=clip.y_max_i).contains(&y);
                let got = if inside {
                    page.clip_scale(id.word, x, y, 255)
                } else {
                    0
                };
                assert_eq!(got, want, "{n} regions: clip coverage at ({x}, {y})");
                // Combination A: a shape scaled region by region.
                for v in [1, 127, 128, 254, rng.byte()] {
                    let mut shape = [v];
                    clip.mul_row(y, x, &mut shape);
                    let got = if inside {
                        page.clip_scale(id.word, x, y, v)
                    } else {
                        0
                    };
                    assert_eq!(got, shape[0], "{n} regions: shape {v} at ({x}, {y})");
                }
            }
        }
    }
}

/// `n` random polygons filled by the CPU, antialiased, under `clip`.
fn cpu_fills<P: Pixel>(
    paper: [u16; 3],
    clip: &Clip,
    fills: &[(raster::Path, bool, [u16; 3], u16)],
    (width, height): (u32, u32),
) -> Vec<u16> {
    let mut bitmap = paper_bitmap::<P>(width, height, paper);
    for (path, eo, rgb, alpha) in fills {
        let colour = rgb.map(|v| P::Sample::from_signed(i64::from(v)));
        let pipe = pipe::<P::Sample>(*alpha);
        let src = PipeSrc::Solid(&colour);
        if *eo {
            raster::eo_fill(&mut bitmap, clip, path, &pipe, &src, &IDENTITY, 0.1, true);
        } else {
            raster::fill(&mut bitmap, clip, path, &pipe, &src, &IDENTITY, 0.1, true);
        }
    }
    widened(&bitmap)
}

/// The fills as clipped commands: outlines, or their tables when
/// `tables`.
fn fill_page(
    depth: SampleDepth,
    paper: [u16; 3],
    clip: &Clip,
    fills: &[(raster::Path, bool, [u16; 3], u16)],
    (width, height): (u32, u32),
    tables: bool,
) -> PageCommands {
    let mut page = PageCommands::new(width, height, depth, paper);
    let id = page_clip(&mut page, clip);
    page.set_clip(id);
    for (path, eo, rgb, alpha) in fills {
        push_fill(&mut page, (path, *eo), clip, (*rgb, *alpha), tables);
    }
    page
}

#[test]
fn clipped_fills_composite_as_the_cpu_fills() {
    let mut rng = Rng(0xF111_C11B);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        let paper = [max, max - 40, max / 3];
        for n in REGIONS {
            let size = (61, 45);
            let clip = random_clip(&mut rng, size.0, size.1, n);
            let fills: Vec<_> = (0..12)
                .map(|_| {
                    let alpha = [max, max, max / 2 + 1, 1, rng.sample(max)]
                        [usize::try_from(rng.below(5)).unwrap()];
                    let rgb = [0; 3].map(|_| rng.sample(max));
                    (
                        random_path(&mut rng, size.0, size.1),
                        rng.below(2) == 0,
                        rgb,
                        alpha,
                    )
                })
                .collect();
            let want = match depth {
                SampleDepth::Eight => cpu_fills::<Rgb8>(paper, &clip, &fills, size),
                SampleDepth::Sixteen => cpu_fills::<Rgb16>(paper, &clip, &fills, size),
            };
            for tables in [false, true] {
                let page = fill_page(depth, paper, &clip, &fills, size, tables);
                assert_eq!(page.bytes(), binned_bytes(&page));
                let what = format!("{}-bit, {n} regions, tables {tables}", depth.bits());
                assert_model_is(&page, &want, &what);
            }
        }
    }
}

/// Random glyph bitmaps blitted by the CPU under `clip` at random pens.
#[expect(clippy::type_complexity, reason = "a test's list of glyphs")]
fn cpu_glyphs<P: Pixel>(
    paper: [u16; 3],
    clip: &Clip,
    glyphs: &[(Vec<u8>, u32, u32, bool, (i32, i32), [u16; 3], u16)],
    (width, height): (u32, u32),
) -> Vec<u16> {
    let mut bitmap = paper_bitmap::<P>(width, height, paper);
    for (data, w, h, aa, pen, rgb, alpha) in glyphs {
        let colour = rgb.map(|v| P::Sample::from_signed(i64::from(v)));
        let glyph = GlyphBitmap {
            data,
            x: 0,
            y: 0,
            w: i32::try_from(*w).unwrap(),
            h: i32::try_from(*h).unwrap(),
            aa: *aa,
        };
        let _ = raster::fill_glyph(
            &mut bitmap,
            clip,
            &pipe::<P::Sample>(*alpha),
            &PipeSrc::Solid(&colour),
            pen.0,
            pen.1,
            &glyph,
        );
    }
    widened(&bitmap)
}

#[test]
fn clipped_glyphs_composite_as_the_cpu_blits_them() {
    let mut rng = Rng(0x6171_C11B);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        let paper = [max - 5, max, max / 2];
        for n in REGIONS {
            let size = (64, 40);
            let clip = random_clip(&mut rng, size.0, size.1, n);
            let glyphs: Vec<_> = (0..40)
                .map(|_| {
                    let (w, h) = (
                        rng.below(30).unsigned_abs() + 1,
                        rng.below(25).unsigned_abs() + 1,
                    );
                    let aa = rng.below(3) != 0;
                    let len = if aa { w * h } else { w.div_ceil(8) * h };
                    let data = (0..len).map(|_| rng.byte()).collect();
                    let pen = (rng.below(80) - 10, rng.below(56) - 10);
                    let alpha =
                        [max, max / 2 + 1, rng.sample(max)][usize::try_from(rng.below(3)).unwrap()];
                    (data, w, h, aa, pen, [0; 3].map(|_| rng.sample(max)), alpha)
                })
                .collect();
            let want = match depth {
                SampleDepth::Eight => cpu_glyphs::<Rgb8>(paper, &clip, &glyphs, size),
                SampleDepth::Sixteen => cpu_glyphs::<Rgb16>(paper, &clip, &glyphs, size),
            };
            let mut page = PageCommands::new(size.0, size.1, depth, paper);
            let id = page_clip(&mut page, &clip);
            page.set_clip(id);
            for (data, w, h, aa, pen, rgb, alpha) in &glyphs {
                let bits = if *aa {
                    MaskBits::Coverage
                } else {
                    MaskBits::Mono
                };
                let mask = page.add_mask(data, *w, *h, bits).expect("add mask");
                page.push_mask(mask, *pen, clip_bounds(&clip), *rgb, *alpha)
                    .expect("push mask");
            }
            assert_model_is(&page, &want, &format!("{}-bit, {n} regions", depth.bits()));
        }
    }
}

#[test]
fn a_clipped_opaque_outline_hides_nothing_below_it() {
    let (width, height) = (48, 48);
    let mut rng = Rng(0x0CC1_0DE5);
    for n in [1, 3] {
        let clip = random_clip(&mut rng, width, height, n);
        // A page-covering opaque square under the clip, over a fill.
        let mut square = raster::PathBuilder::new();
        square.move_to(-4.0, -4.0);
        for (x, y) in [(52.0, -4.0), (52.0, 52.0), (-4.0, 52.0)] {
            square.line_to(x, y).expect("line_to");
        }
        square.close(false).expect("close");
        let fills = [
            (
                random_path(&mut rng, width, height),
                false,
                [10, 200, 30],
                255,
            ),
            (square.build(), false, [250, 5, 90], 255),
        ];
        let want = cpu_fills::<Rgb8>(
            [255; 3],
            &Clip::new(0.0, 0.0, 48.0, 48.0, true),
            &fills[..1],
            (width, height),
        );
        let mut page = PageCommands::new(width, height, SampleDepth::Eight, [255; 3]);
        let rect = Clip::new(0.0, 0.0, 48.0, 48.0, true);
        push_fill(
            &mut page,
            (&fills[0].0, false),
            &rect,
            (fills[0].2, 255),
            false,
        );
        let before = page.len();
        let id = page_clip(&mut page, &clip);
        page.set_clip(id);
        push_fill(
            &mut page,
            (&fills[1].0, false),
            &clip,
            (fills[1].2, 255),
            false,
        );
        let bins = page.bin().expect("bin");
        let listed = bins
            .items
            .iter()
            .filter(|i| (i[0] as usize) < before)
            .count();
        let unclipped = {
            let mut p = PageCommands::new(width, height, SampleDepth::Eight, [255; 3]);
            push_fill(
                &mut p,
                (&fills[0].0, false),
                &rect,
                (fills[0].2, 255),
                false,
            );
            p.bin().expect("bin").items.len()
        };
        assert_eq!(
            listed, unclipped,
            "{n} regions: the clipped square hid items below it"
        );
        // The CPU: the fill, then the square under the clip.
        let mut bitmap = paper_bitmap::<Rgb8>(width, height, [255; 3]);
        let colour = |rgb: [u16; 3]| rgb.map(|v| u8::try_from(v).unwrap());
        let (c0, c1) = (colour(fills[0].2), colour(fills[1].2));
        raster::fill(
            &mut bitmap,
            &rect,
            &fills[0].0,
            &pipe::<u8>(255),
            &PipeSrc::Solid(&c0),
            &IDENTITY,
            0.1,
            true,
        );
        raster::fill(
            &mut bitmap,
            &clip,
            &fills[1].0,
            &pipe::<u8>(255),
            &PipeSrc::Solid(&c1),
            &IDENTITY,
            0.1,
            true,
        );
        let want_both = widened(&bitmap);
        assert_ne!(want, want_both);
        assert_model_is(&page, &want_both, &format!("{n} regions"));
    }
}

#[test]
fn commands_outside_every_region_push_nothing() {
    let mut page = PageCommands::new(40, 40, SampleDepth::Eight, [255; 3]);
    // A region covering (5..=9, 5..=9), and one covering (20..=24, 20..=24).
    let square = |page: &mut PageCommands, at: i32| {
        let row_starts: Vec<u32> = (0..=5).collect();
        let spans: Vec<[i32; 3]> = (0..5).map(|_| [at, at + 4, -1]).collect();
        page.add_clip_region(CoverageRows {
            y_min: at,
            row_starts: &row_starts,
            spans: &spans,
            bytes: &[],
        })
        .expect("add region")
    };
    let (a, b) = (square(&mut page, 5), square(&mut page, 20));
    let disjoint = page.add_clip(&[a, b]).expect("add clip");
    let one = page.add_clip(&[a]).expect("add clip");
    let full = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 39,
        y1: 39,
    };
    let mask = page
        .add_mask(&[255; 4], 2, 2, MaskBits::Coverage)
        .expect("mask");
    page.set_clip(Some(disjoint));
    page.push_mask(mask, (6, 6), full, [0; 3], 255)
        .expect("push");
    assert!(page.is_empty(), "a clip of disjoint regions paints nothing");
    page.set_clip(Some(one));
    page.push_mask(mask, (20, 20), full, [0; 3], 255)
        .expect("push");
    assert!(page.is_empty(), "a mask outside the region paints nothing");
    page.push_mask(mask, (9, 9), full, [0; 3], 255)
        .expect("push");
    let c = page.cmds()[0];
    assert_eq!(
        (c.x0, c.y0, c.x1, c.y1),
        (9, 9, 9, 9),
        "bounds narrowed to the region"
    );
    page.set_clip(None);
    page.push_opaque_runs(&mut [(30, 0, 39)], [0; 3])
        .expect("runs");
    assert_eq!(page.cmds()[1].clip, 0);
    page.set_clip(Some(one));
    page.push_opaque_runs(&mut [(30, 0, 39)], [0; 3])
        .expect("runs");
    assert_eq!(page.cmds()[2].clip, 0, "runs are never clipped");
    assert_eq!(page.bytes(), binned_bytes(&page));
}
