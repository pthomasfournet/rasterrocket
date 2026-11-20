use color::convert::div255;

use super::*;
use crate::path::PathBuilder;
use crate::xpath::XPath;

/// An antialiased path clip selects the device-space region the path
/// covers.
#[test]
fn aa_path_clip_selects_device_space_region() {
    let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);

    // Diamond centred on (16, 16): non-axis-aligned, so it stays a
    // scanner instead of reducing to a rectangle.
    let mut builder = PathBuilder::new();
    builder.move_to(16.0, 2.0);
    builder.line_to(30.0, 16.0).expect("line_to");
    builder.line_to(16.0, 30.0).expect("line_to");
    builder.line_to(2.0, 16.0).expect("line_to");
    builder.close(false).expect("close");
    let path = builder.build();
    let xpath = XPath::new(&path, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true);
    clip.clip_to_path(&xpath, false);
    assert_eq!(clip.regions().count(), 1, "diamond must become a path clip");

    assert!(clip.test(16, 16), "diamond centre must be inside the clip");
    assert!(clip.test(16, 8), "upper interior point must be inside");
    assert!(clip.test(9, 16), "left interior point must be inside");
    assert!(clip.test(23, 16), "right interior point must be inside");
    assert!(clip.test(16, 24), "lower interior point must be inside");
    assert!(
        !clip.test(3, 3),
        "corner outside the diamond must be clipped"
    );
    assert!(
        !clip.test(4, 4),
        "point outside the diamond must be clipped"
    );
}

/// Diamond with vertices at `(cx, cy ± r)` and `(cx ± r, cy)`, clipped
/// into `clip` as a path.
fn clip_to_diamond(clip: &mut Clip, cx: f64, cy: f64, r: f64) {
    let mut builder = PathBuilder::new();
    builder.move_to(cx, cy - r);
    builder.line_to(cx + r, cy).expect("line_to");
    builder.line_to(cx, cy + r).expect("line_to");
    builder.line_to(cx - r, cy).expect("line_to");
    builder.close(false).expect("close");
    let xpath = XPath::new(&builder.build(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true);
    clip.clip_to_path(&xpath, false);
}

/// The path of a random star polygon of 5 to 12 vertices around
/// `(32, 32)`, its radii between 20 and 48 so it is concave and crosses
/// the clip rectangle's edges; one in three visits its vertices out of
/// angular order, crossing itself.  Drawn from the generator state `seed`.
fn random_polygon(seed: &mut u64) -> XPath {
    let mut next = || {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[expect(clippy::cast_precision_loss, reason = "31-bit value")]
        let unit = (*seed >> 33) as f64 / f64::from(1u32 << 31);
        unit
    };
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0..8"
    )]
    let n = 5 + (next() * 8.0) as usize;
    let crossing = next() < 1.0 / 3.0;
    let mut angles: Vec<f64> = (0..n).map(|_| next() * std::f64::consts::TAU).collect();
    if !crossing {
        angles.sort_by(f64::total_cmp);
    }
    let mut builder = PathBuilder::new();
    for (k, angle) in angles.into_iter().enumerate() {
        let r = next().mul_add(28.0, 20.0);
        let (x, y) = (r.mul_add(angle.cos(), 32.0), r.mul_add(angle.sin(), 32.0));
        if k == 0 {
            builder.move_to(x, y);
        } else {
            builder.line_to(x, y).expect("line_to");
        }
    }
    builder.close(false).expect("close");
    XPath::new(&builder.build(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, true)
}

#[test]
fn a_folded_clip_answers_as_the_conjunction_of_its_paths() {
    let mut seed = 7;
    // Trials whose intersection is neither empty nor the whole rectangle.
    let mut partly_inside = 0;
    for antialias in [false, true] {
        for eo in [false, true] {
            for _ in 0..12 {
                let paths: Vec<XPath> = (0..10).map(|_| random_polygon(&mut seed)).collect();
                let mut folded = Clip::new(0.0, 0.0, 64.0, 64.0, antialias);
                let singles: Vec<Clip> = paths
                    .iter()
                    .map(|path| {
                        folded.clip_to_path(path, eo);
                        let mut single = Clip::new(0.0, 0.0, 64.0, 64.0, antialias);
                        single.clip_to_path(path, eo);
                        single
                    })
                    .collect();
                assert!(folded.regions().count() < FOLD_DEPTH);
                let inside = (0..64)
                    .flat_map(|y| (0..64).map(move |x| (x, y)))
                    .filter(|&(x, y)| folded.test(x, y))
                    .count();
                if (1..64 * 64).contains(&inside) {
                    partly_inside += 1;
                }
                for y in -2..66 {
                    for x in (-2..66).filter(|_| !antialias) {
                        assert_eq!(
                            folded.test(x, y),
                            singles.iter().all(|c| c.test(x, y)),
                            "test ({x}, {y}), aa {antialias}, eo {eo}"
                        );
                    }
                    for x0 in (-2..66).step_by(3) {
                        for len in [0, 1, 4, 9, 30] {
                            let results: Vec<ClipResult> = singles
                                .iter()
                                .map(|c| c.test_span(x0, x0 + len, y))
                                .collect();
                            let want = if results.iter().all(|&r| r == ClipResult::AllInside) {
                                ClipResult::AllInside
                            } else if results[0] == ClipResult::AllOutside {
                                // The rectangle, shared by every clip, excludes the span.
                                ClipResult::AllOutside
                            } else {
                                ClipResult::Partial
                            };
                            assert_eq!(
                                folded.test_span(x0, x0 + len, y),
                                want,
                                "test_span {x0}..={} on {y}",
                                x0 + len
                            );
                        }
                    }
                    // The folded coverage is the product of the singles',
                    // rounded once per factor in another order.
                    let mut got = [u8::MAX; 70];
                    folded.mul_row(y, -2, &mut got);
                    let mut want = [u8::MAX; 70];
                    for single in &singles {
                        single.mul_row(y, -2, &mut want);
                    }
                    for (x, (&g, &w)) in (-2..).zip(got.iter().zip(&want)) {
                        assert!(
                            g.abs_diff(w) <= u8::try_from(FOLD_DEPTH).unwrap(),
                            "mul_row ({x}, {y}): {g} vs {w}, aa {antialias}"
                        );
                        if !antialias {
                            assert_eq!(g, w, "mul_row ({x}, {y})");
                        }
                    }
                }
            }
        }
    }
    // Of the 48 trials, enough must cross a boundary of the intersection.
    assert!(
        partly_inside >= 20,
        "{partly_inside} of 48 trials had a boundary"
    );
}

/// The clip's coverage of pixels `x..x + n` on row `y`.
fn coverage_row(clip: &Clip, y: i32, x: i32, n: usize) -> Vec<u8> {
    let mut row = vec![u8::MAX; n];
    clip.mul_row(y, x, &mut row);
    row
}
/// The exported regions are the clip's, newest first, the order
/// `mul_row` scales by, with identities shared by `clone_shared`
/// copies and distinct between regions.
#[test]
fn path_regions_are_exported_newest_first() {
    let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
    assert!(clip.path_key().is_none());
    assert_eq!(clip.path_regions().count(), 0);
    clip_to_diamond(&mut clip, 20.0, 20.0, 15.0);
    let first = clip.path_key().expect("a path clip");
    let saved = clip.clone_shared();
    clip_to_diamond(&mut clip, 30.0, 26.0, 18.0);
    let regions: Vec<_> = clip.path_regions().collect();
    assert_eq!(regions.len(), 2);
    assert_eq!(Some(regions[0].0), clip.path_key());
    assert_eq!(regions[1].0, first);
    assert_ne!(regions[0].0, regions[1].0);
    assert_eq!(saved.path_key(), Some(first));
    let (newest, oldest) = (regions[0].1.expect("area"), regions[1].1.expect("area"));
    for y in 0..64 {
        for x in 0..64 {
            for v in [1u8, 200, 255] {
                let mut row = [v];
                clip.mul_row(y, x, &mut row);
                let scaled = div255(u32::from(v) * u32::from(newest.coverage(x, y)));
                let want = div255(u32::from(scaled) * u32::from(oldest.coverage(x, y)));
                assert_eq!(row[0], want, "({x}, {y}) {v}");
            }
        }
    }
    let mut aliased = Clip::new(0.0, 0.0, 64.0, 64.0, false);
    clip_to_diamond(&mut aliased, 20.0, 20.0, 15.0);
    assert!(aliased.path_regions().all(|(_, t)| t.is_none()));
}

/// `mul_row` scales the coverage already in the row by the clip's:
/// pixels inside the path clip keep their value, those outside become
/// 0, pixels on its edge take the product.
#[test]
fn mul_row_scales_by_the_clip_coverage() {
    let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
    clip_to_diamond(&mut clip, 16.0, 16.5, 14.0);
    let mut row = [0u8; 64];
    row[10..50].fill(200);
    clip.mul_row(16, 0, &mut row);
    assert_eq!(row[5], 0, "inside the clip, never covered");
    assert_eq!(row[16], 200, "inside the clip and the row");
    assert_eq!(row[40], 0, "row coverage outside the clip");
    let edge = coverage_row(&clip, 16, 2, 1)[0];
    assert!(
        (1..255).contains(&edge),
        "the diamond's tip is partly covered: {edge}"
    );
}

/// Nested path clips multiply: a pixel keeps coverage only where every
/// clip covers it.
#[test]
fn nested_clips_multiply() {
    let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, true);
    clip_to_diamond(&mut clip, 16.0, 16.0, 14.0);
    clip_to_diamond(&mut clip, 28.0, 16.0, 14.0);
    assert_eq!(clip.regions().count(), 2);
    let row = coverage_row(&clip, 16, 0, 64);
    assert_eq!(row[8], 0, "only in the first diamond");
    assert_eq!(row[22], 255, "in both diamonds");
    assert_eq!(row[36], 0, "only in the second diamond");
}

/// A translated clip passes and clips exactly what the clip passes and
/// clips at the shifted coordinates, pixel by pixel, with the same
/// coverage.
#[test]
fn translated_clip_tests_the_shifted_pixels() {
    let (dx, dy) = (-9, -5);
    for antialias in [false, true] {
        let mut clip = Clip::new(2.5, 1.0, 60.0, 40.25, antialias);
        clip_to_diamond(&mut clip, 24.0, 18.0, 13.5);
        let moved = clip.translated(dx, dy);
        assert_eq!(
            (moved.x_min_i, moved.y_min_i, moved.x_max_i, moved.y_max_i),
            (
                clip.x_min_i + dx,
                clip.y_min_i + dy,
                clip.x_max_i + dx,
                clip.y_max_i + dy
            )
        );
        for y in 0..48 {
            for x in 0..64 {
                assert_eq!(moved.test(x + dx, y + dy), clip.test(x, y), "({x}, {y})");
            }
        }
        for y in 0..48 {
            assert_eq!(
                coverage_row(&clip, y, 0, 64),
                coverage_row(&moved, y + dy, dx, 64),
                "row {y}, aa {antialias}"
            );
        }
    }
}

#[test]
fn new_clip_rect_bounds() {
    let c = Clip::new(1.5, 2.5, 10.5, 8.5, false);
    assert_eq!(c.x_min_i, 1); // floor(1.5) = 1
    assert_eq!(c.y_min_i, 2); // floor(2.5) = 2
    assert_eq!(c.x_max_i, 10); // ceil(10.5) - 1 = 11 - 1 = 10
    assert_eq!(c.y_max_i, 8); // ceil(8.5) - 1 = 9 - 1 = 8
}

#[test]
fn test_inside() {
    let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
    assert!(c.test(5, 5));
}

#[test]
fn test_outside() {
    let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
    assert!(!c.test(15, 5));
    assert!(!c.test(5, 15));
}

#[test]
fn clip_to_rect_shrinks() {
    let mut c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
    c.clip_to_rect(2.0, 3.0, 8.0, 7.0);
    assert_eq!(c.x_min_i, 2);
    assert_eq!(c.y_min_i, 3);
}

#[test]
fn test_rect_all_inside() {
    let c = Clip::new(0.0, 0.0, 20.0, 20.0, false);
    assert_eq!(c.test_rect(1, 1, 5, 5), ClipResult::AllInside);
}

#[test]
fn test_rect_all_outside() {
    let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
    assert_eq!(c.test_rect(15, 15, 20, 20), ClipResult::AllOutside);
}

/// A save shares the whole path-clip list (one pointer, not one `Arc`
/// per clip), and intersecting the copy leaves the original unchanged.
#[test]
fn clone_shares_the_path_clip_list() {
    let mut c = Clip::new(0.0, 0.0, 64.0, 64.0, false);
    for i in 0..3 {
        clip_to_diamond(&mut c, 32.0, 32.0, 20.0 + f64::from(i));
    }
    let mut saved = c.clone_shared();
    assert_eq!(saved.x_min_i, c.x_min_i);
    let (head, saved_head) = (c.scanners.as_ref(), saved.scanners.as_ref());
    assert!(head.zip(saved_head).is_some_and(|(a, b)| Arc::ptr_eq(a, b)));
    assert_eq!(head.map(Arc::strong_count), Some(2));

    clip_to_diamond(&mut saved, 10.0, 32.0, 6.0);
    assert_eq!(c.regions().count(), 3);
    assert_eq!(saved.regions().count(), 4);
    // (32, 32) is inside the three shared diamonds but outside the
    // fourth, (10, 32) the other way round.
    assert!(c.test(32, 32) && !saved.test(32, 32));
    assert!(!c.test(10, 32) && !saved.test(10, 32));
    assert!(c.test(30, 32) && !saved.test(30, 32));
    assert!(saved.test(14, 32) && c.test(14, 32));
}

/// Intersecting many path clips keeps fewer than `FOLD_DEPTH` scanners,
/// and the folded clip still passes the common interior.
#[test]
fn a_clip_holds_at_most_fold_depth_scanners() {
    let mut c = Clip::new(0.0, 0.0, 4.0, 4.0, false);
    for _ in 0..10_000 {
        clip_to_diamond(&mut c, 2.0, 2.0, 1.5);
        assert!(c.regions().count() < FOLD_DEPTH);
    }
    assert!(c.test(2, 2));
}

/// `inside_runs` yields the maximal runs inside the clip, including runs
/// touching either end of the span, and nothing for an empty span.
#[test]
fn inside_runs_are_maximal_and_ordered() {
    let mut clip = Clip::new(0.0, 0.0, 64.0, 64.0, false);
    clip_to_diamond(&mut clip, 16.0, 16.0, 6.0);
    let runs: Vec<_> = clip.inside_runs(0, 63, 16).collect();
    assert_eq!(runs.len(), 1, "{runs:?}");
    let (a, b) = runs[0];
    assert!((10..=11).contains(&a) && (21..=22).contains(&b), "{runs:?}");
    assert!((a..=b).all(|x| clip.test(x, 16)));
    assert_eq!(clip.inside_runs(a, b, 16).collect::<Vec<_>>(), [(a, b)]);
    assert_eq!(clip.inside_runs(5, 4, 16).count(), 0);
    assert_eq!(clip.inside_runs(0, 5, 16).count(), 0);

    let rect = Clip::new(0.0, 0.0, 8.0, 8.0, false);
    assert_eq!(rect.inside_runs(-3, 20, 2).collect::<Vec<_>>(), [(0, 7)]);
}

/// A rectangle reaching `i32::MAX`, a saturated non-finite coordinate,
/// is classified without overflowing.
#[test]
fn test_rect_accepts_i32_max_edges() {
    let c = Clip::new(0.0, 0.0, 10.0, 10.0, false);
    assert_eq!(c.test_rect(2, 2, i32::MAX, i32::MAX), ClipResult::Partial);
    assert_eq!(
        c.test_rect(i32::MAX, 2, i32::MAX, 3),
        ClipResult::AllOutside
    );
}

/// A rectangle whose edges lie far past the `i32` range of device space
/// saturates its integer bounds and contains no pixel near the origin.
#[test]
fn rect_beyond_the_i32_range_saturates_its_bounds() {
    let far = 1e18;
    let below = Clip::new(-far, -far, -far / 2.0, -far / 2.0, false);
    assert_eq!(
        (below.x_min_i, below.y_min_i, below.x_max_i, below.y_max_i),
        (i32::MIN, i32::MIN, i32::MIN, i32::MIN)
    );
    assert!(!below.test(0, 0));
    let mut clip = Clip::new(0.0, 0.0, 16.0, 16.0, false);
    clip.clip_to_rect(-far, -far, -far / 2.0, -far / 2.0);
    assert!(!clip.test(0, 0));
    let above = Clip::new(far / 2.0, far / 2.0, far, far, false);
    assert_eq!(
        (above.x_min_i, above.y_min_i, above.x_max_i, above.y_max_i),
        (i32::MAX, i32::MAX, i32::MAX - 1, i32::MAX - 1)
    );
}
