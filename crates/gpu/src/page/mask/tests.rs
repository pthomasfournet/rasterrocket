//! Masks in the host model against the CPU's glyph blit.

use color::{Pixel, Rgb8, Rgb16, Sample, SampleDepth};
use raster::pipe::render_span;
use raster::types::BlendMode;
use raster::{Bitmap, Clip, GlyphBitmap, PipeSrc, PipeState, TransferSet, fill_glyph};

use super::{MaskBits, MaskId};
use crate::page::reference::render_page;
use crate::page::test_support::Rng;
use crate::page::{PAGE_COMMANDS_MAX_BYTES, PageCommands, PageCommandsError, PixelBounds};

/// A glyph as the CPU blits it.
struct Glyph {
    data: Vec<u8>,
    w: u32,
    h: u32,
    aa: bool,
    x_off: i32,
    y_off: i32,
    pen: (i32, i32),
    /// The clip rectangle, `[x0, x1) × [y0, y1)` in device pixels.
    clip: [f64; 4],
    rgb: [u16; 3],
    alpha: u16,
}

fn random_glyph(rng: &mut Rng, width: u32, height: u32, max: u16) -> Glyph {
    let (w, h) = (
        u32::try_from(rng.range(1, 40)).unwrap(),
        u32::try_from(rng.range(1, 30)).unwrap(),
    );
    let aa = rng.range(0, 3) != 0;
    let row_bytes = if aa { w } else { w.div_ceil(8) } as usize;
    // Short mono data reads as zero; an AA bitmap is always whole.
    let short = if aa {
        0
    } else {
        usize::try_from(rng.range(0, 3)).unwrap()
    };
    let len = (row_bytes * h as usize).saturating_sub(short);
    let data = (0..len)
        .map(|_| match rng.range(0, 4) {
            0 => 0,
            1 => 255,
            _ => rng.byte(),
        })
        .collect();
    let coord =
        |rng: &mut Rng, extent: u32| i32::try_from(rng.range(-30, i64::from(extent) + 30)).unwrap();
    let edge = |rng: &mut Rng, extent: u32| {
        f64::from(i32::try_from(rng.range(-8, i64::from(extent) * 4 + 8)).unwrap()) / 4.0
    };
    let (cx0, cx1) = (edge(rng, width), edge(rng, width));
    let (cy0, cy1) = (edge(rng, height), edge(rng, height));
    Glyph {
        data,
        w,
        h,
        aa,
        x_off: i32::try_from(rng.range(-5, 20)).unwrap(),
        y_off: i32::try_from(rng.range(-5, 20)).unwrap(),
        pen: (coord(rng, width), coord(rng, height)),
        clip: [cx0, cy0, cx1, cy1],
        rgb: [0; 3].map(|_| rng.sample(max)),
        alpha: [max, max, max / 2 + 1, 1, rng.sample(max)]
            [usize::try_from(rng.range(0, 5)).unwrap()],
    }
}

/// The glyphs blitted by the CPU onto a page of `paper`, samples widened
/// to 16 bits.
fn cpu_page<P: Pixel>(width: u32, height: u32, paper: [u16; 3], glyphs: &[Glyph]) -> Vec<u16> {
    let s = |v: u16| P::Sample::from_signed(i64::from(v));
    let data = (0..width * height)
        .flat_map(|_| paper.map(s))
        .collect::<Vec<_>>();
    let mut bitmap = Bitmap::<P>::from_packed(width, height, data);
    for g in glyphs {
        let [x0, y0, x1, y1] = g.clip;
        let clip = Clip::new(x0, y0, x1, y1, true);
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: s(g.alpha),
            transfer: TransferSet::identity_rgb(),
            backdrop: None,
            overprint: None,
            gray: None,
        };
        let colour = g.rgb.map(s);
        let src = PipeSrc::Solid(&colour);
        let glyph = GlyphBitmap {
            data: &g.data,
            x: g.x_off,
            y: g.y_off,
            w: i32::try_from(g.w).unwrap(),
            h: i32::try_from(g.h).unwrap(),
            aa: g.aa,
        };
        let _ = fill_glyph(&mut bitmap, &clip, &pipe, &src, g.pen.0, g.pen.1, &glyph);
    }
    bitmap
        .data()
        .iter()
        .map(|&v| u16::try_from(v.signed()).unwrap())
        .collect()
}

/// The glyphs as mask commands, placed and bounded as the CPU blits them;
/// glyphs of equal bitmaps share one mask.
fn mask_page(
    width: u32,
    height: u32,
    depth: SampleDepth,
    paper: [u16; 3],
    glyphs: &[Glyph],
) -> PageCommands {
    let mut page = PageCommands::new(width, height, depth, paper);
    let mut held: Vec<(usize, MaskId)> = Vec::new();
    for (i, g) in glyphs.iter().enumerate() {
        let shared = held
            .iter()
            .find(|&&(j, _)| {
                glyphs[j].data == g.data
                    && (glyphs[j].w, glyphs[j].h, glyphs[j].aa) == (g.w, g.h, g.aa)
            })
            .map(|&(_, m)| m);
        let mask = shared.unwrap_or_else(|| {
            let bits = if g.aa {
                MaskBits::Coverage
            } else {
                MaskBits::Mono
            };
            let m = page.add_mask(&g.data, g.w, g.h, bits).expect("add mask");
            held.push((i, m));
            m
        });
        let [x0, y0, x1, y1] = g.clip;
        let clip = Clip::new(x0, y0, x1, y1, true);
        let bounds = PixelBounds {
            x0: clip.x_min_i,
            y0: clip.y_min_i,
            x1: clip.x_max_i,
            y1: clip.y_max_i,
        };
        let origin = (
            g.pen.0.saturating_sub(g.x_off),
            g.pen.1.saturating_sub(g.y_off),
        );
        page.push_mask(mask, origin, bounds, g.rgb, g.alpha)
            .expect("push mask");
    }
    page
}

#[test]
fn masks_composite_as_the_cpu_blits_glyphs() {
    let mut rng = Rng(0x61F_0A11_A5ED);
    for depth in [SampleDepth::Eight, SampleDepth::Sixteen] {
        let max = depth.max_sample();
        for (width, height, n) in [(1, 1, 4), (23, 17, 40), (64, 48, 120), (130, 41, 200)] {
            let paper = [max - 3, max / 3, max];
            let mut glyphs: Vec<Glyph> = (0..n)
                .map(|_| random_glyph(&mut rng, width, height, max))
                .collect();
            // Every third glyph repeats the bitmap before it, as a font's
            // glyphs repeat in text: one mask placed twice.
            for k in (3..glyphs.len()).step_by(3) {
                let (before, rest) = glyphs.split_at_mut(k);
                let (prev, g) = (&before[k - 1], &mut rest[0]);
                (g.data, g.w, g.h, g.aa) = (prev.data.clone(), prev.w, prev.h, prev.aa);
            }
            let want = match depth {
                SampleDepth::Eight => cpu_page::<Rgb8>(width, height, paper, &glyphs),
                SampleDepth::Sixteen => cpu_page::<Rgb16>(width, height, paper, &glyphs),
            };
            let painted = want.chunks(3).filter(|&p| p != paper).count();
            assert!(
                painted > 0 || width == 1,
                "{width}×{height}: no glyph painted"
            );
            let page = mask_page(width, height, depth, paper, &glyphs);
            let got = render_page(&page).expect("host model");
            if let Some(i) = (0..want.len()).find(|&i| got[i] != want[i]) {
                panic!(
                    "{}-bit {width}×{height}: pixel ({}, {}) sample {} is {} in the model, {} on the CPU",
                    depth.bits(),
                    i / 3 % width as usize,
                    i / 3 / width as usize,
                    i % 3,
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// A mono glyph's set bits composite, on the CPU, without a shape (the
/// general pipe when translucent, the simple one when opaque), and a mask
/// composites them at full coverage: the two agree for every alpha.
fn full_runs_match_full_coverage<P: Pixel>(rng: &mut Rng, alphas: impl Iterator<Item = u16>) {
    let max = P::Sample::MAX.signed();
    let s = |v: i64| P::Sample::from_signed(v);
    for alpha in alphas {
        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: s(i64::from(alpha)),
            transfer: TransferSet::identity_rgb(),
            backdrop: None,
            overprint: None,
            gray: None,
        };
        let colour = [0; 3].map(|_| s(rng.range(0, max + 1)));
        let dst: Vec<P::Sample> = (0..48).map(|_| s(rng.range(0, max + 1))).collect();
        let full = [P::Sample::MAX; 16];
        let (mut bare, mut shaped) = (dst.clone(), dst);
        render_span::<P>(
            &pipe,
            &PipeSrc::Solid(&colour),
            &mut bare,
            None,
            None,
            0,
            15,
            0,
        );
        render_span::<P>(
            &pipe,
            &PipeSrc::Solid(&colour),
            &mut shaped,
            None,
            Some(&full),
            0,
            15,
            0,
        );
        assert_eq!(
            bare, shaped,
            "alpha {alpha}: a full run differs from full coverage"
        );
    }
}

#[test]
fn full_runs_composite_as_full_coverage_at_8_and_16_bits() {
    let mut rng = Rng(0xF011_0F0F);
    full_runs_match_full_coverage::<Rgb8>(&mut rng, 0..=255);
    let sampled: Vec<u16> = (0..2_000).map(|_| rng.sample(u16::MAX)).collect();
    full_runs_match_full_coverage::<Rgb16>(
        &mut rng,
        [0, 1, 257, 32_768, 65_534, 65_535]
            .into_iter()
            .chain(sampled),
    );
}

#[test]
fn masks_are_held_once_and_counted() {
    let mut page = PageCommands::new(40, 40, SampleDepth::Eight, [255; 3]);
    let before = page.bytes();
    let mask = page
        .add_mask(&[1, 2, 3, 4, 5, 6], 3, 2, MaskBits::Coverage)
        .expect("add");
    assert_eq!((mask.width(), mask.height()), (3, 2));
    let all = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 39,
        y1: 39,
    };
    for x in [0, 14, 28] {
        page.push_mask(mask, (x, 15), all, [0; 3], 255)
            .expect("push");
    }
    assert_eq!(page.len(), 3);
    // Six coverage bytes in two words, three commands, and the items of
    // the tiles they touch: columns 0, 0–1 and 1 of tile rows 0–1.
    let bins = page.bin().expect("bin");
    assert_eq!(bins.items.len(), 2 + 4 + 2);
    assert_eq!(page.bytes(), before + 2 * 4 + 3 * 48 + 8 * 16);
}

#[test]
fn mono_masks_expand_their_bits() {
    let mut page = PageCommands::new(10, 2, SampleDepth::Eight, [255; 3]);
    let mask = page
        .add_mask(&[0b1010_0000, 0b0100_0000, 0xFF], 10, 2, MaskBits::Mono)
        .expect("add");
    let all = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 9,
        y1: 1,
    };
    page.push_mask(mask, (0, 0), all, [0; 3], 255)
        .expect("push");
    let px = render_page(&page).expect("model");
    let painted: Vec<bool> = px.chunks(3).map(|p| p[0] == 0).collect();
    let want = [
        true, false, true, false, false, false, false, false, false, true, // row 0
        true, true, true, true, true, true, true, true, false,
        false, // row 1: 0xFF, then short data
    ];
    assert_eq!(painted, want);
}

#[test]
fn masks_off_the_page_or_empty_push_nothing() {
    let mut page = PageCommands::new(10, 10, SampleDepth::Eight, [255; 3]);
    let mask = page
        .add_mask(&[255; 4], 2, 2, MaskBits::Coverage)
        .expect("add");
    let empty = page.add_mask(&[], 0, 5, MaskBits::Coverage).expect("add");
    let all = PixelBounds {
        x0: 0,
        y0: 0,
        x1: 9,
        y1: 9,
    };
    page.push_mask(mask, (10, 0), all, [0; 3], 255)
        .expect("push");
    page.push_mask(mask, (-2, 3), all, [0; 3], 255)
        .expect("push");
    page.push_mask(mask, (i32::MAX, i32::MIN), all, [0; 3], 255)
        .expect("push");
    page.push_mask(empty, (0, 0), all, [0; 3], 255)
        .expect("push");
    page.push_mask(
        mask,
        (3, 3),
        PixelBounds {
            x0: 5,
            y0: 0,
            x1: 9,
            y1: 9,
        },
        [0; 3],
        255,
    )
    .expect("push");
    assert!(page.is_empty());
}

#[test]
fn a_mask_past_the_limit_is_refused() {
    let mut page = PageCommands::new(16, 16, SampleDepth::Eight, [255; 3]);
    let side = u32::try_from(PAGE_COMMANDS_MAX_BYTES.isqrt() + 1).unwrap();
    assert_eq!(
        page.add_mask(&[], side, side, MaskBits::Coverage),
        Err(PageCommandsError::TooLarge)
    );
    assert_eq!(
        page.bytes(),
        PageCommands::new(16, 16, SampleDepth::Eight, [255; 3]).bytes()
    );
}
