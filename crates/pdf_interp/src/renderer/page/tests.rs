//! Unit tests of the page renderer, most rendering small content streams.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use color::Rgb8;
use color::convert::div255;
use pdf::{Document, ObjectId};
use raster::Bitmap;
use raster::pipe::{PipeSrc, PipeState};
use raster::types::BlendMode;

use super::super::gstate::{Ctm, ctm_transform};
use super::image_blit::{
    ImageCompositor, ImageLenError, blend_rgb, blend_u8, check_image_bytes_len,
};
use super::pixel_rect::PixelRect;
use super::view::build_initial_ctm;
use super::xobject::image_draw_size;
use super::{
    DEADLINE_CHECK_INTERVAL, MAX_WARNED_UNKNOWN_OPS, PageDiagnostics, PageGeometry, PageRenderer,
    PageView, RenderAbort,
};

#[test]
fn image_draw_size_measures_the_transformed_unit_square() {
    assert_eq!(
        image_draw_size(&[200.0, 0.0, 0.0, 100.0, 5.0, 7.0]),
        Some((200, 100))
    );
    // Rotated 90°: the image's width lies along device y.
    assert_eq!(
        image_draw_size(&[0.0, 200.0, -100.0, 0.0, 0.0, 0.0]),
        Some((200, 100))
    );
    // A fractional extent rounds up.
    assert_eq!(
        image_draw_size(&[30.0, 40.0, 0.0, 10.4, 0.0, 0.0]),
        Some((50, 11))
    );
    // Flipped (negative scale) measures the same.
    assert_eq!(
        image_draw_size(&[-200.0, 0.0, 0.0, -100.0, 0.0, 0.0]),
        Some((200, 100))
    );
}

#[test]
fn image_draw_size_rejects_degenerate_matrices() {
    assert_eq!(image_draw_size(&[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]), None);
    assert_eq!(image_draw_size(&[f64::NAN, 0.0, 0.0, 1.0, 0.0, 0.0]), None);
    assert_eq!(image_draw_size(&[1e300, 0.0, 0.0, 1.0, 0.0, 0.0]), None);
}

fn diag_with_ppi(ppi: f32) -> PageDiagnostics {
    PageDiagnostics {
        has_images: true,
        source_ppi_hint: Some(ppi),
        ..PageDiagnostics::default()
    }
}

#[test]
fn suggested_dpi_rounds_up_to_nearest_step() {
    // 67 PPI → next step is 72.
    assert_eq!(diag_with_ppi(67.0).suggested_dpi(72.0, 600.0), Some(72.0));
    // 73 PPI → next step is 96.
    assert_eq!(diag_with_ppi(73.0).suggested_dpi(72.0, 600.0), Some(96.0));
    // 150 PPI exactly → stays at 150.
    assert_eq!(diag_with_ppi(150.0).suggested_dpi(72.0, 600.0), Some(150.0));
    // 280 PPI → next step is 300.
    assert_eq!(diag_with_ppi(280.0).suggested_dpi(72.0, 600.0), Some(300.0));
}

#[test]
fn suggested_dpi_respects_min_clamp() {
    // 67 PPI → step is 72, but min=150 clamps it up.
    assert_eq!(diag_with_ppi(67.0).suggested_dpi(150.0, 300.0), Some(150.0));
}

#[test]
fn suggested_dpi_respects_max_clamp() {
    // 700 PPI → above all steps; last step is 600, then clamped to max=300.
    assert_eq!(diag_with_ppi(700.0).suggested_dpi(72.0, 300.0), Some(300.0));
}

#[test]
fn suggested_dpi_none_when_no_images() {
    let diag = PageDiagnostics::default(); // source_ppi_hint is None
    assert_eq!(diag.suggested_dpi(150.0, 300.0), None);
}

// ── build_initial_ctm: per-/Rotate-branch + page-box origin ────────────
//
// Each expected matrix below is hand-computed from ISO 32000-2 §8.3.4 by
// composing the box-at-origin scale+rotate with an innermost pre-translation
// T(-llx, -lly).  The locked render-regression baseline only exercises
// (0,0)-origin pages, so it cannot catch an origin-sign error; these tests
// pin the sign for every rotation against an independent derivation.

fn assert_ctm_eq(got: Ctm, want: Ctm) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-9,
            "ctm[{i}]: got {g}, want {w} (full got {got:?}, want {want:?})"
        );
    }
}

#[test]
fn initial_ctm_zero_origin_is_pixel_neutral_for_every_rotation() {
    // With llx = lly = 0 each branch must equal the pre-origin-fix matrix,
    // proving origin-at-zero PDFs (the overwhelming majority) are unchanged.
    let (wpx, hpx, s) = (600u32, 800u32, 2.0);
    let w = f64::from(wpx) / s;
    let h = f64::from(hpx) / s;
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 0, 0.0, 0.0),
        [s, 0.0, 0.0, s, 0.0, 0.0],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 90, 0.0, 0.0),
        [0.0, -s, s, 0.0, 0.0, h * s],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 180, 0.0, 0.0),
        [-s, 0.0, 0.0, -s, w * s, h * s],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 270, 0.0, 0.0),
        [0.0, s, -s, 0.0, w * s, 0.0],
    );
}

#[test]
fn initial_ctm_nonzero_origin_pretranslates_per_rotation() {
    // CropBox like the lecouteux repro: non-zero lower-left origin.
    let (wpx, hpx, s) = (794u32, 1286u32, 2.0);
    let (llx, lly) = (462.0_f64, 23.0_f64);
    let w = f64::from(wpx) / s;
    let h = f64::from(hpx) / s;

    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 0, llx, lly),
        [s, 0.0, 0.0, s, -llx * s, -lly * s],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 90, llx, lly),
        [0.0, -s, s, 0.0, -lly * s, (h + llx) * s],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 180, llx, lly),
        [-s, 0.0, 0.0, -s, (w + llx) * s, (h + lly) * s],
    );
    assert_ctm_eq(
        build_initial_ctm(wpx, hpx, (s, s), 270, llx, lly),
        [0.0, s, -s, 0.0, (w + lly) * s, -llx * s],
    );
}

/// Page geometry input for `assert_box_corners_map_to`.
///
/// `wpx`/`hpx` are the output bitmap pixel dimensions, already swapped for
/// 90°/270° exactly as the renderer's caller swaps them before `PageView::whole`.
struct CornerCase {
    wpx: u32,
    hpx: u32,
    /// Device pixels per point along device x and device y.
    scale: (f64, f64),
    rotate: u16,
    llx: f64,
    lly: f64,
}

/// Expected device-pixel landing of each selected-box corner (origin
/// top-left): lower-left, lower-right, upper-left, upper-right in PDF user
/// space.  Named fields make a transposed corner a compile-readable error
/// rather than a silent geometry mismatch.
struct ExpectedCorners {
    ll: (f64, f64),
    lr: (f64, f64),
    ul: (f64, f64),
    ur: (f64, f64),
}

/// Map the four selected-box corners through `build_initial_ctm` + the
/// `to_device` Y-flip and assert each lands at its expected device pixel.
///
/// Expected positions are an independent cross-check: they match both
/// `pdftoppm` and `mutool` (verified on identical fixtures), neither of
/// which shares code with this renderer.
fn assert_box_corners_map_to(case: &CornerCase, want: &ExpectedCorners) {
    let CornerCase {
        wpx,
        hpx,
        scale: (sx, sy),
        rotate,
        llx,
        lly,
    } = *case;
    let ctm = build_initial_ctm(wpx, hpx, (sx, sy), rotate, llx, lly);
    // For 0°/180° the box is wpx×hpx; for 90°/270° the caller swapped the
    // bitmap dims, so the box in points is hpx×wpx.  Resolve box pts from
    // the unrotated orientation.
    let (bw, bh) = if rotate % 180 == 0 {
        (f64::from(wpx) / sx, f64::from(hpx) / sy)
    } else {
        (f64::from(hpx) / sy, f64::from(wpx) / sx)
    };
    let page_h = f64::from(hpx);
    let to_dev = |x: f64, y: f64| {
        let (dx, dy) = ctm_transform(&ctm, x, y);
        (dx, page_h - dy)
    };
    let corners = [
        ("LL", (llx, lly), want.ll),
        ("LR", (llx + bw, lly), want.lr),
        ("UL", (llx, lly + bh), want.ul),
        ("UR", (llx + bw, lly + bh), want.ur),
    ];
    for (name, (x, y), (wx, wy)) in corners {
        let (px, py) = to_dev(x, y);
        assert!(
            (px - wx).abs() < 1e-9 && (py - wy).abs() < 1e-9,
            "rotate {rotate} {name}: got ({px}, {py}), want ({wx}, {wy})"
        );
    }
}

#[test]
fn initial_ctm_rotate0_maps_box_corners_to_device_pixels() {
    // Unrotated, non-(0,0)-origin box: lower-left → device bottom-left,
    // upper-right → device top-right.  The sign check the locked
    // (0,0)-origin baseline structurally cannot perform.
    assert_box_corners_map_to(
        &CornerCase {
            wpx: 400,
            hpx: 600,
            scale: (2.0, 2.0),
            rotate: 0,
            llx: 50.0,
            lly: 30.0,
        },
        &ExpectedCorners {
            ll: (0.0, 600.0),   // → bottom-left
            lr: (400.0, 600.0), // → bottom-right
            ul: (0.0, 0.0),     // → top-left
            ur: (400.0, 0.0),   // → top-right
        },
    );
}

#[test]
fn initial_ctm_rotate90_maps_box_corners_to_device_pixels() {
    // Caller swaps pixel dims for 90°: bitmap is 600×400.
    assert_box_corners_map_to(
        &CornerCase {
            wpx: 600,
            hpx: 400,
            scale: (2.0, 2.0),
            rotate: 90,
            llx: 50.0,
            lly: 30.0,
        },
        &ExpectedCorners {
            ll: (0.0, 0.0),     // → top-left
            lr: (0.0, 400.0),   // → bottom-left
            ul: (600.0, 0.0),   // → top-right
            ur: (600.0, 400.0), // → bottom-right
        },
    );
}

#[test]
fn initial_ctm_rotate180_maps_box_corners_to_device_pixels() {
    assert_box_corners_map_to(
        &CornerCase {
            wpx: 400,
            hpx: 600,
            scale: (2.0, 2.0),
            rotate: 180,
            llx: 50.0,
            lly: 30.0,
        },
        &ExpectedCorners {
            ll: (400.0, 0.0),   // → top-right
            lr: (0.0, 0.0),     // → top-left
            ul: (400.0, 600.0), // → bottom-right
            ur: (0.0, 600.0),   // → bottom-left
        },
    );
}

#[test]
fn initial_ctm_rotate270_maps_box_corners_to_device_pixels() {
    // Caller swaps pixel dims for 270°: bitmap is 600×400.
    assert_box_corners_map_to(
        &CornerCase {
            wpx: 600,
            hpx: 400,
            scale: (2.0, 2.0),
            rotate: 270,
            llx: 50.0,
            lly: 30.0,
        },
        &ExpectedCorners {
            ll: (600.0, 400.0), // → bottom-right
            lr: (600.0, 0.0),   // → top-right
            ul: (0.0, 400.0),   // → bottom-left
            ur: (0.0, 0.0),     // → top-left
        },
    );
}

/// With different device scales per axis, every rotation still maps the
/// box corners onto the bitmap corners: the scales apply along the device
/// axes, after rotation.
#[test]
fn initial_ctm_anisotropic_maps_box_corners_to_bitmap_corners() {
    let (w, h) = (300.0, 100.0);
    for rotate in [0, 90, 180, 270] {
        let want = match rotate {
            0 => ExpectedCorners {
                ll: (0.0, h),
                lr: (w, h),
                ul: (0.0, 0.0),
                ur: (w, 0.0),
            },
            90 => ExpectedCorners {
                ll: (0.0, 0.0),
                lr: (0.0, h),
                ul: (w, 0.0),
                ur: (w, h),
            },
            180 => ExpectedCorners {
                ll: (w, 0.0),
                lr: (0.0, 0.0),
                ul: (w, h),
                ur: (0.0, h),
            },
            _ => ExpectedCorners {
                ll: (w, h),
                lr: (w, 0.0),
                ul: (0.0, h),
                ur: (0.0, 0.0),
            },
        };
        assert_box_corners_map_to(
            &CornerCase {
                wpx: 300,
                hpx: 100,
                scale: (3.0, 0.5),
                rotate,
                llx: 50.0,
                lly: 30.0,
            },
            &want,
        );
    }
}

/// `/UserUnit` scales the page's pixel extent and the CTM alike: a
/// 200 × 100 pt page with `/UserUnit 2` at one pixel per point is
/// 400 × 200 px, and its content point (200, 100) lands on the far
/// corner.  A window of that page keeps the same scale.
#[test]
fn page_view_user_unit_scales_the_ctm_with_the_extent() {
    let geom = PageGeometry {
        width_pts: 400.0,
        height_pts: 200.0,
        rotate_cw: 0,
        user_unit: 2.0,
        origin_x: 0.0,
        origin_y: 0.0,
        media_box: Some([0.0, 0.0, 200.0, 100.0]),
        crop_box: None,
    };
    let whole = PageView::whole(&geom, (1.0, 1.0));
    assert_eq!((whole.page_width, whole.page_height), (400, 200));
    assert_ctm_eq(whole.initial_ctm(), [2.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let window = whole.window(100, 50, 200, 100);
    assert_ctm_eq(window.initial_ctm(), [2.0, 0.0, 0.0, 2.0, -100.0, -50.0]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "rotate_cw must be normalised")]
fn initial_ctm_non_spec_rotation_trips_debug_assert() {
    // Callers must normalise /Rotate to {0,90,180,270}; a non-spec value is a
    // contract violation and must fail loudly in debug rather than silently
    // selecting a rotation arm and rendering a wrong-but-plausible page.
    let _ = build_initial_ctm(100, 100, (1.0, 1.0), 45, 0.0, 0.0);
}

#[test]
fn initial_ctm_non_spec_rotation_release_falls_back_to_unrotated() {
    // Release builds (debug_assertions off) must not panic on a stray value:
    // the fallback is the unrotated matrix, so the page renders upright.
    // Only meaningful without debug assertions; debug builds panic (above).
    if cfg!(debug_assertions) {
        return;
    }
    assert_ctm_eq(
        build_initial_ctm(100, 100, (1.0, 1.0), 45, 0.0, 0.0),
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
}

// ── check_image_bytes_len ──────────────────────────────────────────────

#[test]
fn check_image_bytes_len_accepts_exact() {
    // 2 × 2 × Rgb (bpp=3) → expected 12 bytes.
    assert_eq!(check_image_bytes_len(2, 2, 3, 12), Ok(12));
}

#[test]
fn check_image_bytes_len_accepts_over_long_buffer() {
    // Over-long buffers are fine — only short buffers are rejected.
    assert_eq!(check_image_bytes_len(2, 2, 3, 15), Ok(12));
}

#[test]
fn check_image_bytes_len_rejects_short_buffer() {
    assert_eq!(
        check_image_bytes_len(2, 2, 3, 6),
        Err(ImageLenError::ShortBuffer { expected: 12 }),
    );
}

#[test]
fn check_image_bytes_len_rejects_dimension_overflow() {
    // u32::MAX × u32::MAX × 3 overflows usize on 64-bit targets.
    assert_eq!(
        check_image_bytes_len(u32::MAX, u32::MAX, 3, 0),
        Err(ImageLenError::DimensionOverflow),
    );
}

#[test]
fn check_image_bytes_len_one_by_one_gray() {
    // 1 × 1 × Gray (bpp=1) → expected 1 byte.
    assert_eq!(check_image_bytes_len(1, 1, 1, 1), Ok(1));
}

// ── ImageCompositor ───────────────────────────────────────────────────

/// Composite one row of samples pixel by pixel: each painted sample
/// alone, with source alpha `div255(coverage × a_input)`.
fn paint_row_per_pixel(
    bitmap: &mut Bitmap<Rgb8>,
    pipe: &PipeState<'_>,
    y: u32,
    colours: &[u8],
    shape: &[u8],
) {
    for (x, (src, &coverage)) in colours.as_chunks::<3>().0.iter().zip(shape).enumerate() {
        let a_input = div255(u32::from(coverage) * u32::from(pipe.a_input));
        if a_input == 0 {
            continue;
        }
        let pixel_pipe = PipeState { a_input, ..*pipe };
        let x = i32::try_from(x).expect("small");
        let y = i32::try_from(y).expect("small");
        raster::pipe::render_bitmap_span::<Rgb8>(
            bitmap,
            &pixel_pipe,
            &PipeSrc::Solid(src),
            None,
            x,
            x,
            y,
        );
    }
}

/// Compositing a row of image samples in runs gives the bytes of
/// compositing each sample alone, with and without an alpha plane,
/// for constant alpha and a separable and a non-separable blend mode.
#[test]
fn image_rows_composite_like_single_samples() {
    let width = 13u32;
    let mut seed = 7u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed.to_be_bytes()[0]
    };
    let colours: Vec<u8> = (0..width * 3).map(|_| next()).collect();
    // Runs of unpainted samples, and coverages small enough that the
    // scaled alpha rounds to zero.
    let shape: Vec<u8> = (0..width)
        .map(|i| match i % 5 {
            0 => 0,
            1 => 1,
            _ => next(),
        })
        .collect();
    let backdrop: Vec<u8> = (0..width * 3).map(|_| next()).collect();
    let backdrop_alpha: Vec<u8> = (0..width).map(|_| next()).collect();
    for has_alpha in [false, true] {
        for (fill_alpha, mode) in [
            (255, BlendMode::Normal),
            (128, BlendMode::Normal),
            (200, BlendMode::Multiply),
            (255, BlendMode::Luminosity),
        ] {
            let make_base = || {
                let mut base = Bitmap::<Rgb8>::new(width, 1, 1, has_alpha);
                base.row_bytes_mut(0).copy_from_slice(&backdrop);
                if let Some(a) = base.alpha_plane_mut() {
                    a.copy_from_slice(&backdrop_alpha);
                }
                base
            };
            let base = make_base();
            let compositor = ImageCompositor::new(&base, fill_alpha, mode, None);
            let mut runs = make_base();
            let (row, alpha_row) = runs.row_and_alpha_mut(0);
            compositor.paint_row(row, alpha_row, 0, 0, &colours, &shape);
            let mut single = make_base();
            if let Some(pipe) = &compositor.pipe {
                paint_row_per_pixel(&mut single, pipe, 0, &colours, &shape);
            } else {
                let row = single.row_bytes_mut(0);
                for (x, &c) in shape.iter().enumerate() {
                    if c != 0 {
                        let src = [colours[x * 3], colours[x * 3 + 1], colours[x * 3 + 2]];
                        blend_rgb(&mut row[x * 3..x * 3 + 3], src, c);
                    }
                }
            }
            assert_eq!(
                runs.data(),
                single.data(),
                "{mode:?} at {fill_alpha}, alpha {has_alpha}"
            );
            assert_eq!(
                runs.alpha_plane(),
                single.alpha_plane(),
                "{mode:?} at {fill_alpha}, alpha {has_alpha}"
            );
        }
    }
}

// ── blend_u8 ──────────────────────────────────────────────────────────────

#[test]
fn blend_u8_opaque_returns_src() {
    assert_eq!(blend_u8(200, 50, 255), 200);
    assert_eq!(blend_u8(0, 255, 255), 0);
    assert_eq!(blend_u8(255, 0, 255), 255);
}

#[test]
fn blend_u8_transparent_returns_dst() {
    // alpha=0 is short-circuited by the caller (continue), but the function
    // itself must still be well-defined (returns dst).
    assert_eq!(blend_u8(200, 50, 0), 50);
}

#[test]
fn blend_u8_50pct_midpoints_correctly() {
    // src=255, dst=0, alpha=128 → (255*128 + 0*127)/255 = 128.
    assert_eq!(blend_u8(255, 0, 128), 128);
    // src=0, dst=255, alpha=128 → (0*128 + 255*127)/255 = 127.
    assert_eq!(blend_u8(0, 255, 128), 127);
}

#[test]
fn blend_u8_no_overflow_at_max_values() {
    // Worst-case intermediate: 255*254 + 255*1 = 65025, fits u16.
    assert_eq!(blend_u8(255, 255, 254), 255);
    assert_eq!(blend_u8(255, 255, 1), 255);
}

// ── per-page work watchdog ────────────────────────────────────────────────

/// Verify the operator-count cap fires: a tiny `op_budget` is tripped,
/// fewer than all ops run, and `budget_status()` reports the breach.
#[test]
fn watchdog_op_budget_fires() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    // Construct a minimal renderer; (1,1) = 1-pixel white page at 72 dpi.
    // empty_doc has no pages, so we use object id (1,0) (the catalog) as
    // the page_id — no real page resources are needed because EndPath is a
    // pure graphics-state no-op that touches no resources.
    let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

    // Override op_budget to a small value so the test terminates instantly.
    renderer.set_op_budget_for_test(10);
    renderer.set_deadline_for_test(None); // disable wall-clock cap; only op count fires

    // Build a vector of 50 no-op EndPath operators — well over the budget.
    let ops: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 50).collect();
    renderer.execute(&ops);

    assert!(
        renderer.budget_status().is_some(),
        "budget_status() must be Some after exceeding op_budget"
    );
    assert!(
        renderer.ops_executed_for_test() <= 11, // budget=10 → fires at op 11
        "no more than budget+1 ops should have run, got {}",
        renderer.ops_executed_for_test()
    );
}

/// A raised cancel flag stops the render before its next operator and
/// reaches a pattern-cell renderer through the shared budget.
#[test]
fn cancel_flag_stops_execute_and_is_inherited() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let flag = Arc::new(AtomicBool::new(false));
    let mut parent = PageRenderer::new(1, 1, &doc, (1, 0));
    parent.set_cancel(Arc::clone(&flag));
    let ops: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 8).collect();
    parent.execute(&ops);
    assert!(parent.budget_status().is_none(), "a lowered flag is inert");

    flag.store(true, Ordering::Relaxed);
    let mut child = PageRenderer::new(1, 1, &doc, (1, 0));
    child.adopt_parent_budget(&parent);
    child.execute(&ops);
    assert_eq!(
        child.budget_status(),
        Some(&RenderAbort::Cancelled { ops: 0 }),
        "the child observes the parent's flag before its first operator"
    );
    parent.execute(&ops);
    assert_eq!(
        parent.budget_status(),
        Some(&RenderAbort::Cancelled { ops: 8 })
    );
}

/// Verify the wall-clock deadline fires when set to an already-elapsed instant.
#[test]
fn watchdog_deadline_fires() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

    // Set an already-elapsed deadline and a very large op budget so the
    // deadline is the only thing that can fire.
    renderer.set_op_budget_for_test(u64::MAX);
    renderer.set_deadline_for_test(Some(
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap(),
    ));

    // Build exactly DEADLINE_CHECK_INTERVAL no-op ops so the periodic check
    // is guaranteed to run at least once.
    let ops: Vec<Operator> = std::iter::repeat_n(
        Operator::EndPath,
        usize::try_from(DEADLINE_CHECK_INTERVAL).unwrap(),
    )
    .collect();
    renderer.execute(&ops);

    assert!(
        renderer.budget_status().is_some(),
        "budget_status() must be Some after deadline is exceeded"
    );
}

/// The page work-budget MUST be shared with — not reset for — a child
/// renderer (tiling-pattern tile).  A fresh `PageRenderer` would otherwise
/// get its own full budget and a fresh deadline, letting a pathological
/// pattern content stream escape the per-page watchdog entirely.  This
/// pins both halves of the contract: `adopt_parent_budget` carries the
/// parent's *remaining* allowance + exact deadline (and inherits an
/// already-tripped breach), and `fold_child_into` rolls the child's
/// consumption and any breach back into the parent.
/// An image skipped or a font missing inside a child renderer (a
/// tiling-pattern cell) is reported on the page that owns the cell, once
/// however many times the cell is rasterised.
#[test]
fn child_diagnostics_fold_into_parent_once() {
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let mut parent = PageRenderer::new(1, 1, &doc, (1, 0));
    parent.record_image_failure(Some(b"Im0"), "parent".to_owned());
    let mut child = PageRenderer::new(1, 1, &doc, (1, 0));
    child.record_image_failure(None, "child".to_owned());
    child.record_missing_font(b"F9");
    child.diag.has_images = true;

    child.fold_child_into(&mut parent);
    child.fold_child_into(&mut parent);

    let diag = parent.finish().1;
    let reasons: Vec<_> = diag
        .image_failures
        .into_iter()
        .map(|f| (f.name, f.reason))
        .collect();
    assert_eq!(
        reasons,
        [
            (Some("Im0".to_owned()), "parent".to_owned()),
            (None, "child".to_owned())
        ]
    );
    assert_eq!(diag.missing_fonts, ["F9"]);
    assert!(diag.has_images);
}

#[test]
fn watchdog_budget_shared_with_child_renderer() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();

    // Parent with a 100-op budget that has already spent 90 ops.
    let mut parent = PageRenderer::new(1, 1, &doc, (1, 0));
    parent.set_op_budget_for_test(100);
    parent.set_deadline_for_test(None);
    let ops90: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 90).collect();
    parent.execute(&ops90);
    assert!(parent.budget_status().is_none(), "90 < 100 must not trip");
    assert_eq!(parent.ops_executed_for_test(), 90);

    // Child adopts the *remaining* 10-op allowance, not a fresh 100.
    let mut child = PageRenderer::new(1, 1, &doc, (1, 0));
    child.adopt_parent_budget(&parent);
    assert_eq!(
        child.op_budget, 10,
        "child must inherit remaining allowance"
    );
    assert_eq!(child.ops_executed_for_test(), 0);
    assert!(
        child.deadline.is_none(),
        "child must inherit exact deadline"
    );

    // Child runs 50 ops — well over its 10-op share — and trips.
    let ops50: Vec<Operator> = std::iter::repeat_n(Operator::EndPath, 50).collect();
    child.execute(&ops50);
    assert!(
        child.budget_status().is_some(),
        "child must trip on the shared remaining budget"
    );

    // Folding back: parent's counter advances by the child's work and the
    // breach propagates so the parent's execute() bails loudly next.
    child.fold_child_into(&mut parent);
    assert!(
        parent.ops_executed_for_test() >= 90 + 11,
        "parent must account the child's work, got {}",
        parent.ops_executed_for_test()
    );
    assert!(
        parent.budget_status().is_some(),
        "a child breach must propagate to the parent"
    );

    // An already-tripped parent must hand the child a zero allowance so it
    // does no work at all.
    let mut child2 = PageRenderer::new(1, 1, &doc, (1, 0));
    child2.adopt_parent_budget(&parent);
    assert!(
        child2.budget_status().is_some(),
        "child must inherit an already-tripped parent breach"
    );
    let one: Vec<Operator> = vec![Operator::EndPath];
    child2.execute(&one);
    assert_eq!(
        child2.ops_executed_for_test(),
        0,
        "a child of a tripped parent must execute zero ops"
    );
}

// ── unsupported-operator warn-once dedup ──────────────────────────────────

/// A hostile content stream with far more distinct junk operators than the
/// cap must not grow `warned_unknown_ops` without bound: the set is capped
/// at `MAX_WARNED_UNKNOWN_OPS` and the cap-reached flag is set (so the
/// summary warning fires exactly once).  Exact repeats of a tracked keyword
/// must not inflate the set.
#[test]
fn unknown_op_dedup_set_is_bounded_under_adversarial_input() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));
    renderer.set_op_budget_for_test(u64::MAX);
    renderer.set_deadline_for_test(None);

    // Ten times the cap of *distinct* junk keywords, each repeated twice to
    // also exercise the dedup-hit (no-allocation) path.
    let distinct = MAX_WARNED_UNKNOWN_OPS * 10;
    let mut ops: Vec<Operator> = Vec::with_capacity(distinct * 2);
    for i in 0..distinct {
        let kw = format!("junk{i}").into_bytes();
        ops.push(Operator::Unknown(kw.clone()));
        ops.push(Operator::Unknown(kw));
    }
    renderer.execute(&ops);

    let (tracked, capped) = renderer.warned_unknown_ops_state_for_test();
    assert_eq!(
        tracked, MAX_WARNED_UNKNOWN_OPS,
        "warned set must be hard-capped, got {tracked}"
    );
    assert!(
        capped,
        "cap-reached flag must be set so the summary warning fires"
    );
}

/// The first occurrence of every distinct unsupported operator (below the
/// cap) is tracked — proving the first-occurrence WARN still surfaces and
/// the dedup never swallows the signal that content may be incomplete.
#[test]
fn unknown_op_first_occurrence_is_tracked_per_keyword() {
    use crate::content::Operator;
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let mut renderer = PageRenderer::new(1, 1, &doc, (1, 0));

    // Three distinct keywords, the first repeated three times.
    renderer.execute(&[
        Operator::Unknown(b"foo".to_vec()),
        Operator::Unknown(b"foo".to_vec()),
        Operator::Unknown(b"foo".to_vec()),
        Operator::Unknown(b"bar".to_vec()),
        Operator::Unknown(b"baz".to_vec()),
    ]);

    let (tracked, capped) = renderer.warned_unknown_ops_state_for_test();
    assert_eq!(tracked, 3, "one tracked entry per distinct keyword");
    assert!(!capped, "cap must not trip below MAX_WARNED_UNKNOWN_OPS");
}

/// Each page gets a fresh `PageRenderer`, so the dedup set is inherently
/// per-page.  Verify a freshly constructed renderer starts with an empty
/// set (no cross-page leakage of suppression state).
#[test]
fn unknown_op_dedup_state_is_per_renderer() {
    use crate::test_helpers::empty_doc;

    let doc = empty_doc();
    let renderer = PageRenderer::new(1, 1, &doc, (1, 0));
    assert_eq!(
        renderer.warned_unknown_ops_state_for_test(),
        (0, false),
        "a new per-page renderer must start with an empty, uncapped dedup set"
    );
}

// ── Image clipping ────────────────────────────────────────────────────────

/// Run `content` on a `size × size` white page at 72 dpi and return the
/// red channel of every pixel, row-major.  The page has no resources, so
/// the content must paint through inline images only.
fn render_gray(size: u32, content: &[u8]) -> Vec<u8> {
    render_gray_on(&crate::test_helpers::empty_doc(), (1, 0), size, content)
}

/// [`render_gray`] on page `page_id` of `doc`, whose resources the
/// content may name.
fn render_gray_on(doc: &Document, page_id: ObjectId, size: u32, content: &[u8]) -> Vec<u8> {
    let mut renderer = PageRenderer::new(size, size, doc, page_id);
    renderer.execute(&crate::content::parse(content));
    let stride = renderer.bitmap.stride;
    let data = renderer.bitmap.data();
    (0..size)
        .flat_map(|y| (0..size).map(move |x| (x, y)))
        .map(|(x, y)| data[y as usize * stride + x as usize * 3])
        .collect()
}

/// A 2×2 all-black `DeviceGray` inline image stretched over the whole
/// 8 pt page.  Without a clip it would blacken every pixel.
const BLACK_IMAGE_OVER_PAGE: &[u8] =
    b"q 8 0 0 8 0 0 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\x00\x00\x00 EI Q";

#[test]
fn image_blit_honours_rectangular_clip() {
    let content = [
        b"q 0 0 4 8 re W n ".as_slice(),
        BLACK_IMAGE_OVER_PAGE,
        b" Q",
    ]
    .concat();
    let px = render_gray(8, &content);
    for y in 0..8 {
        for x in 0..8 {
            let want = if x < 4 { 0 } else { 255 };
            assert_eq!(px[y * 8 + x], want, "pixel ({x}, {y})");
        }
    }
}

/// An L-shaped path clip over a black image: the image shows through the
/// L only.  Column 4 of the top half lies on the clip's vertical edge and
/// carries the clip's antialiased coverage (4 of 16 sub-samples, 223
/// after the gamma table), exactly like a fill through the same clip.
#[test]
fn image_blit_honours_path_clip() {
    let content = [
        b"q 0 0 m 8 0 l 8 4 l 4 4 l 4 8 l 0 8 l h W n ".as_slice(),
        BLACK_IMAGE_OVER_PAGE,
        b" Q",
    ]
    .concat();
    let px = render_gray(8, &content);
    for y in 0..8 {
        for x in 0..8 {
            let want = match (y < 4, x) {
                (true, 4) => 223,
                (true, 5..) => 255,
                _ => 0,
            };
            assert_eq!(px[y * 8 + x], want, "pixel ({x}, {y})");
        }
    }
}

/// The line width is a user-space distance: under a 4× CTM a 1-unit
/// horizontal stroke covers 4 device rows, centred on the path.
#[test]
fn stroke_width_scales_with_ctm() {
    let px = render_gray(16, b"4 0 0 4 0 0 cm 0 G 1 w 0 2 m 4 2 l S");
    let dark: Vec<usize> = (0..16).filter(|&y| px[y * 16 + 8] < 128).collect();
    assert_eq!(dark, [6, 7, 8, 9], "stroke rows at x = 8");
}

// ── Colorant /None ────────────────────────────────────────────────────────

/// Page `(4, 0)` naming `/N` = `[/Separation /None …]`, `/A` =
/// `[/Separation /All …]`, `/NN` = `[/DeviceN [/None /None] …]` and
/// `/NC` = `[/DeviceN [/None /Cyan] …]`, all over `DeviceGray`, where
/// a tint of 1 in the first colorant is black, and `/I`, an `Indexed`
/// space whose index 0 is black and 1 white.
fn none_colorant_doc() -> Document {
    const PS: &str = "{ pop 1 exch sub }";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /ColorSpace << \
             /N [/Separation /None /DeviceGray 5 0 R] \
             /A [/Separation /All /DeviceGray 5 0 R] \
             /NN [/DeviceN [/None /None] /DeviceGray 6 0 R] \
             /NC [/DeviceN [/None /Cyan] /DeviceGray 6 0 R] \
             /I [/Indexed /DeviceGray 1 <00ff>] >> >> >>"
                .to_owned(),
        ),
        (
            5,
            "<< /FunctionType 2 /Domain [0 1] /C0 [1] /C1 [0] /N 1 >>".to_owned(),
        ),
        (
            6,
            format!(
                "<< /FunctionType 4 /Domain [0 1 0 1] /Range [0 1] /Length {} >>\n\
                 stream\n{PS}\nendstream",
                PS.len()
            ),
        ),
    ])
}

fn render_none(content: &[u8]) -> Vec<u8> {
    render_gray_on(&none_colorant_doc(), (4, 0), 8, content)
}

#[test]
fn separation_none_paints_nothing() {
    let fill = render_none(b"/N cs 1 scn 0 0 8 8 re f");
    assert!(fill.iter().all(|&v| v == 255), "fill painted: {fill:?}");
    let stroke = render_none(b"/N CS 1 SCN 4 w 0 4 m 8 4 l S");
    assert!(
        stroke.iter().all(|&v| v == 255),
        "stroke painted: {stroke:?}"
    );
    let stencil =
        render_none(b"/N cs 1 scn q 8 0 0 8 0 0 cm BI /W 1 /H 1 /IM true /BPC 1 ID \x00 EI Q");
    assert!(
        stencil.iter().all(|&v| v == 255),
        "stencil painted: {stencil:?}"
    );
}

#[test]
fn separation_all_paints_its_tint() {
    let px = render_none(b"/A cs 1 scn 0 0 8 8 re f");
    assert!(px.iter().all(|&v| v == 0), "got {px:?}");
}

#[test]
fn device_n_paints_nothing_only_when_every_colorant_is_none() {
    let none = render_none(b"/NN cs 1 0 scn 0 0 8 8 re f");
    assert!(none.iter().all(|&v| v == 255), "got {none:?}");
    let cyan = render_none(b"/NC cs 1 0 scn 0 0 8 8 re f");
    assert!(cyan.iter().all(|&v| v == 0), "got {cyan:?}");
}

/// `cs` and `CS` set the current colour to the new space's initial
/// colour (PDF §8.6.8): black for a device space, a tint of 1 for a
/// `Separation`, index 0 for an `Indexed` space, and no pattern — which
/// paints nothing — for a `Pattern` space.
#[test]
fn selecting_a_colour_space_sets_its_initial_colour() {
    for content in [
        b"1 g /DeviceRGB cs 0 0 8 8 re f".as_slice(),
        b"1 g /A cs 0 0 8 8 re f",
        b"1 g /I cs 0 0 8 8 re f",
        b"1 G /DeviceRGB CS 8 w 0 4 m 8 4 l S",
        b"1 G /A CS 8 w 0 4 m 8 4 l S",
    ] {
        let px = render_none(content);
        assert!(
            px.iter().all(|&v| v == 0),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
    // 0 0 0 1 in the process-ink model of DeviceCMYK.
    let k = render_none(b"0 0 0 1 k 0 0 8 8 re f");
    assert_eq!(render_none(b"1 g /DeviceCMYK cs 0 0 8 8 re f"), k);
    for content in [
        b"0 g /Pattern cs 0 0 8 8 re f".as_slice(),
        b"0 G /Pattern CS 8 w 0 4 m 8 4 l S",
        b"0 g /Pattern cs 0.5 sc 0 0 8 8 re f",
    ] {
        let px = render_none(content);
        assert!(
            px.iter().all(|&v| v == 255),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
}

/// `g` and `G` select `DeviceGray` (PDF §8.6.8), leaving the `/None` space.
#[test]
fn device_colour_operators_leave_a_none_space() {
    let fill = render_none(b"/N cs 1 scn 0 g 0 0 8 8 re f");
    assert!(fill.iter().all(|&v| v == 0), "got {fill:?}");
    let stroke = render_none(b"/N CS 1 SCN 0 G 8 w 0 4 m 8 4 l S");
    assert!(stroke.iter().all(|&v| v == 0), "got {stroke:?}");
}

// ── Default colour spaces ─────────────────────────────────────────────────

/// Page `(4, 0)` whose `/ColorSpace` resources hold `/DefaultGray` =
/// `default_gray`.
fn default_gray_doc(default_gray: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[(
        4,
        format!("<< /Type /Page /Resources << /ColorSpace << /DefaultGray {default_gray} >> >> >>"),
    )])
}

/// A linear (`/Gamma 1`) D65 `CalGray` space: a value of 0.5 is half
/// luminance, 188 after sRGB encoding, where `DeviceGray` 0.5 is 128.
const LINEAR_CAL_GRAY: &str = "[/CalGray << /WhitePoint [0.9505 1 1.089] /Gamma 1 >>]";

/// `DefaultGray` replaces `DeviceGray` whether it is selected by `cs`,
/// `CS`, `g` or `G` (PDF §8.6.5.6).
#[test]
fn default_gray_replaces_every_selection_of_device_gray() {
    let doc = default_gray_doc(LINEAR_CAL_GRAY);
    for content in [
        b"0.5 g 0 0 8 8 re f".as_slice(),
        b"/DeviceGray cs 0.5 sc 0 0 8 8 re f",
        b"0.5 G 8 w 0 4 m 8 4 l S",
        b"/DeviceGray CS 0.5 SC 8 w 0 4 m 8 4 l S",
    ] {
        let px = render_gray_on(&doc, (4, 0), 8, content);
        assert!(
            px.iter().all(|v| (186..=189).contains(v)),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
}

/// The underlying space of an uncoloured pattern takes the default
/// space too (PDF §8.6.5.6): a tint of 0.5 paints 188 in the linear
/// `/DefaultGray`.
#[test]
fn an_uncoloured_pattern_tint_takes_the_default_space() {
    const CELL: &str = "0 0 8 8 re f";
    let doc = crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!(
                "<< /Type /Page /Resources << /ColorSpace << /DefaultGray {LINEAR_CAL_GRAY} \
                 /PG [/Pattern /DeviceGray] >> /Pattern << /U 5 0 R >> >> >>"
            ),
        ),
        (
            5,
            format!(
                "<< /PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 8 8] \
                 /XStep 8 /YStep 8 /Length {} >>\nstream\n{CELL}\nendstream",
                CELL.len()
            ),
        ),
    ]);
    let px = render_gray_on(&doc, (4, 0), 8, b"/PG cs 0.5 /U scn 0 0 8 8 re f");
    assert!(px.iter().all(|v| (186..=189).contains(v)), "got {px:?}");
}

/// A default space of the wrong component count leaves the device space
/// in effect.
#[test]
fn default_space_of_another_component_count_is_ignored() {
    let doc = default_gray_doc("[/CalRGB << /WhitePoint [0.9505 1 1.089] >>]");
    let px = render_gray_on(&doc, (4, 0), 8, b"0.5 g 0 0 8 8 re f");
    assert!(px.iter().all(|&v| v == 128), "got {px:?}");
}

// ── Transfer functions ────────────────────────────────────────────────────

/// Page `(4, 0)` with `ExtGState` `/Inv` = `<< /TR 5 0 R >>` and `/Id` =
/// `<< /TR /Identity >>`, where object 5 inverts its input.
fn transfer_doc() -> Document {
    const PS: &str = "{ 1 exch sub }";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /ExtGState << \
             /Inv << /TR 5 0 R >> /Id << /TR /Identity >> >> >> >>"
                .to_owned(),
        ),
        (
            5,
            format!(
                "<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length {} >>\n\
                 stream\n{PS}\nendstream",
                PS.len()
            ),
        ),
    ])
}

fn render_transfer(content: &[u8]) -> Vec<u8> {
    render_gray_on(&transfer_doc(), (4, 0), 8, content)
}

/// Fills, strokes and image samples map through the transfer function
/// (PDF §10.5): 0.2 gray (51) paints 204.
#[test]
fn transfer_function_maps_painted_colours() {
    for content in [
        b"/Inv gs 0.2 g 0 0 8 8 re f".as_slice(),
        b"/Inv gs 0.2 G 8 w 0 4 m 8 4 l S",
        b"/Inv gs q 8 0 0 8 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x33 EI Q",
    ] {
        let px = render_transfer(content);
        assert!(
            px.iter().all(|&v| v == 204),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
}

/// The transfer function is graphics state: `Q` and `/TR /Identity`
/// both end it.
#[test]
fn transfer_function_is_restored_and_reset() {
    let restored = render_transfer(b"q /Inv gs Q 0.2 g 0 0 8 8 re f");
    assert!(restored.iter().all(|&v| v == 51), "got {restored:?}");
    let reset = render_transfer(b"/Inv gs /Id gs 0.2 g 0 0 8 8 re f");
    assert!(reset.iter().all(|&v| v == 51), "got {reset:?}");
}

/// The transfer function maps the object's colour, not the composited
/// result: black under an inverting transfer paints white everywhere,
/// including the partially covered edge pixels of a fill.
#[test]
fn transfer_function_leaves_the_backdrop_alone() {
    let px = render_transfer(b"/Inv gs 0 g 0.5 0.5 6.3 6.3 re f");
    assert!(px.iter().all(|&v| v == 255), "got {px:?}");
}

/// A stroke thinner than a device pixel is widened to one pixel: a
/// 0.1-unit rule centred in a pixel row paints that row solid.
#[test]
fn thin_stroke_widens_to_one_device_pixel() {
    let px = render_gray(16, b"0 G 0.1 w 0 4.5 m 16 4.5 l S");
    let column: Vec<u8> = (0..16).map(|y| px[y * 16 + 8]).collect();
    assert_eq!(column[11], 0, "rule row solid, column {column:?}");
    assert!(
        column
            .iter()
            .enumerate()
            .all(|(y, &g)| (10..=12).contains(&y) || g == 255),
        "no paint beyond the rule's neighbour rows, column {column:?}"
    );
}

/// A negative line width strokes as its magnitude, widened to one
/// device pixel like any thin stroke.
#[test]
fn negative_thin_stroke_widens_to_one_device_pixel() {
    let px = render_gray(16, b"0 G -0.1 w 0 4.5 m 16 4.5 l S");
    assert_eq!(px[11 * 16 + 8], 0, "rule row solid");
}

/// An image whose bounding box misses the clip entirely paints nothing.
#[test]
fn image_blit_outside_clip_paints_nothing() {
    let content = [
        b"q 0 0 1 1 re W n q 4 0 0 4 4 4 cm ".as_slice(),
        b"BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\x00\x00\x00 EI Q Q",
    ]
    .concat();
    let px = render_gray(8, &content);
    assert!(px.iter().all(|&g| g == 255), "no pixel may be painted");
}

// ── Shadings and patterns: /None, default spaces, transfer ───────────────

/// Page `(4, 0)` whose `/ColorSpace` resources are `color_spaces`, with
/// the axial shading `/Sh` (and the shading pattern `/P` over it) of the
/// constant `value` in the colour space `shading_space`, the coloured
/// tiling pattern `/T` painting 0.2 gray, and the inverting transfer
/// `ExtGState` `/Inv`.
fn shading_doc(color_spaces: &str, shading_space: &str, value: f64) -> Document {
    const PS: &str = "{ 1 exch sub }";
    const CELL: &str = "0.2 g 0 0 8 8 re f";
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            format!(
                "<< /Type /Page /Resources << /ColorSpace << {color_spaces} >> \
                 /Shading << /Sh 5 0 R >> /Pattern << /P 6 0 R /T 7 0 R >> \
                 /ExtGState << /Inv << /TR 8 0 R >> >> >> >>"
            ),
        ),
        (
            5,
            format!(
                "<< /ShadingType 2 /ColorSpace {shading_space} /Coords [0 0 8 0] \
                 /Function 9 0 R /Extend [true true] >>"
            ),
        ),
        (6, "<< /PatternType 2 /Shading 5 0 R >>".to_owned()),
        (
            7,
            format!(
                "<< /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 8 8] \
                 /XStep 8 /YStep 8 /Length {} >>\nstream\n{CELL}\nendstream",
                CELL.len()
            ),
        ),
        (
            8,
            format!(
                "<< /FunctionType 4 /Domain [0 1] /Range [0 1] /Length {} >>\n\
                 stream\n{PS}\nendstream",
                PS.len()
            ),
        ),
        (
            9,
            format!("<< /FunctionType 2 /Domain [0 1] /C0 [{value}] /C1 [{value}] /N 1 >>"),
        ),
    ])
}

/// A shading in a space whose colorants are all `/None` paints nothing,
/// through `sh` and through a shading pattern alike.
#[test]
fn shading_in_a_none_space_paints_nothing() {
    let doc = shading_doc("", "[/Separation /None /DeviceGray 9 0 R]", 0.2);
    for content in [b"/Sh sh".as_slice(), b"/Pattern cs /P scn 0 0 8 8 re f"] {
        let px = render_gray_on(&doc, (4, 0), 8, content);
        assert!(
            px.iter().all(|&v| v == 255),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
}

/// A shading in `DeviceGray` takes the `/DefaultGray` of the resources
/// in effect (PDF §8.6.5.6): 0.5 in the linear space is 188, not 128.
#[test]
fn shading_takes_the_default_colour_space() {
    let doc = shading_doc(
        &format!("/DefaultGray {LINEAR_CAL_GRAY}"),
        "/DeviceGray",
        0.5,
    );
    for content in [b"/Sh sh".as_slice(), b"/Pattern cs /P scn 0 0 8 8 re f"] {
        let px = render_gray_on(&doc, (4, 0), 8, content);
        assert!(
            px.iter().all(|v| (186..=189).contains(v)),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
}

/// A shading and a tiling pattern paint through the transfer function
/// in effect (PDF §10.5): 0.2 gray (51) paints 204 under the inverting
/// transfer, whether painted by `sh`, a shading pattern or a tiling
/// pattern.
#[test]
fn shadings_and_patterns_map_through_the_transfer_function() {
    let doc = shading_doc("", "/DeviceGray", 0.2);
    for content in [
        b"/Inv gs /Sh sh".as_slice(),
        b"/Inv gs /Pattern cs /P scn 0 0 8 8 re f",
        b"/Inv gs /Pattern cs /T scn 0 0 8 8 re f",
        b"/Inv gs /Pattern CS /T SCN 8 w 0 4 m 8 4 l S",
    ] {
        let px = render_gray_on(&doc, (4, 0), 8, content);
        assert!(
            px.iter().all(|&v| v == 204),
            "{}: {px:?}",
            String::from_utf8_lossy(content)
        );
    }
    let px = render_gray_on(&doc, (4, 0), 8, b"/Pattern cs /T scn 0 0 8 8 re f");
    assert!(px.iter().all(|&v| v == 51), "without the transfer: {px:?}");
}

// ── Nested content streams ───────────────────────────────────────────────

/// A stream object `<< dict >>` holding `content`.
fn stream_object(dict: &str, content: &str) -> String {
    format!(
        "<< {dict} /Length {} >>\nstream\n{content}\nendstream",
        content.len()
    )
}

/// Page `(4, 0)` drawing the form `/F` of content `form`, and the Type 3
/// font `/T` whose glyph `a` has the description `glyph`.
fn nested_stream_doc(form: &str, glyph: &str) -> Document {
    crate::test_helpers::make_doc_with_objects(&[
        (
            4,
            "<< /Type /Page /Resources << /XObject << /F 5 0 R >> \
             /Font << /T 6 0 R >> >> >>"
                .to_owned(),
        ),
        (
            5,
            stream_object("/Type /XObject /Subtype /Form /BBox [0 0 8 8]", form),
        ),
        (
            6,
            "<< /Type /Font /Subtype /Type3 /FontBBox [0 0 1 1] \
             /FontMatrix [1 0 0 1 0 0] /CharProcs << /a 7 0 R >> \
             /Encoding << /Type /Encoding /Differences [97 /a] >> \
             /FirstChar 97 /LastChar 97 /Widths [1] >>"
                .to_owned(),
        ),
        (7, stream_object("", glyph)),
    ])
}

/// The state a form leaves saved by an unbalanced `q` is discarded with
/// the form: the fill colour it set does not reach the page.
#[test]
fn a_form_s_unbalanced_save_does_not_outlive_it() {
    let doc = nested_stream_doc("1 0 0 rg q", "1 0 d0");
    let px = render_gray_on(&doc, (4, 0), 8, b"/F Do 0 0 8 8 re f");
    assert!(
        px.iter().all(|&v| v == 0),
        "page fill must stay black: {px:?}"
    );
}

/// The `Q`s of a Type 3 glyph description cannot pop the states the
/// page saved: the red fill set before the text is still in effect.
#[test]
fn a_glyph_description_s_restores_cannot_pop_the_page_s_states() {
    let doc = nested_stream_doc("", "1 0 d0 Q Q");
    let px = render_gray_on(
        &doc,
        (4, 0),
        8,
        b"q 1 0 0 rg BT /T 1 Tf (a) Tj ET 0 0 8 8 re f Q",
    );
    assert!(
        px.iter().all(|&v| v == 255),
        "page fill must stay red: {px:?}"
    );
}

/// Annotations are drawn in the initial graphics state of the vector
/// antialiasing mode selected for the page, not the default mode.
#[test]
fn annotations_clip_in_the_page_s_antialiasing_mode() {
    let doc = crate::test_helpers::empty_doc();
    let mut renderer = PageRenderer::new(8, 8, &doc, (1, 0));
    renderer.set_antialias(true, false);
    renderer.render_annotations((1, 0));
    assert!(!renderer.gstate.current().clip.antialias);
}

/// The clipping outlines of a text object the page left unterminated
/// do not reach the `ET` of an annotation appearance.
#[test]
fn annotations_start_without_the_page_s_pending_text_clip() {
    let doc = crate::test_helpers::empty_doc();
    let mut renderer = PageRenderer::new(8, 8, &doc, (1, 0));
    renderer.text_clip = Some(PixelRect::new(0, 0, 1, 1).expect("non-empty").to_path());
    renderer.render_annotations((1, 0));
    assert!(renderer.text_clip.is_none());
}
