use super::*;

/// Coverage of the closed polygons `polys` over `[0, w) × [0, h)` with the
/// raster window `[wx0, wx1] × [wy0, wy1]`, row-major over the full grid.
fn render_window(
    polys: &[&[(f64, f64)]],
    eo: bool,
    w: usize,
    h: usize,
    window: (i32, i32, i32, i32),
) -> Vec<u8> {
    let mut raster = AreaRaster::new(window.0, window.1, window.2, window.3);
    for poly in polys {
        for (i, &(x0, y0)) in poly.iter().enumerate() {
            let (x1, y1) = poly[(i + 1) % poly.len()];
            raster.add_line(x0, y0, x1, y1);
        }
    }
    let mut out = vec![0u8; w * h];
    raster.rows(eo, |y, x, cov| {
        for (px, &c) in (i64::from(x)..).zip(cov) {
            if let (Ok(px), Ok(py)) = (usize::try_from(px), usize::try_from(y))
                && px < w
                && py < h
            {
                out[py * w + px] = c;
            }
        }
    });
    out
}

fn render(polys: &[&[(f64, f64)]], eo: bool, w: usize, h: usize) -> Vec<u8> {
    let (wi, hi) = (i32::try_from(w).unwrap(), i32::try_from(h).unwrap());
    render_window(polys, eo, w, h, (0, 0, wi - 1, hi - 1))
}

fn ink(cov: &[u8]) -> f64 {
    cov.iter().map(|&c| f64::from(c) / 255.0).sum()
}

fn triangle_area(t: &[(f64, f64); 3]) -> f64 {
    0.5 * (t[1].0 - t[0].0)
        .mul_add(t[2].1 - t[0].1, -(t[2].0 - t[0].0) * (t[1].1 - t[0].1))
        .abs()
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> [(f64, f64); 4] {
    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

#[test]
fn pixel_aligned_square_is_fully_covered() {
    let cov = render(&[&rect(2.0, 3.0, 5.0, 7.0)], false, 8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let inside = (2..5).contains(&x) && (3..7).contains(&y);
            assert_eq!(cov[y * 8 + x], if inside { 255 } else { 0 }, "({x}, {y})");
        }
    }
}

#[test]
fn half_pixel_offset_square_has_exact_partial_coverage() {
    let cov = render(&[&rect(2.5, 2.5, 6.5, 6.5)], false, 8, 8);
    let at = |x: usize, y: usize| cov[y * 8 + x];
    assert_eq!(at(2, 2), 64, "corner: a quarter");
    assert_eq!(at(6, 6), 64, "corner: a quarter");
    assert_eq!(at(4, 2), 128, "top edge: a half");
    assert_eq!(at(4, 6), 128, "bottom edge: a half");
    assert_eq!(at(2, 4), 128, "left edge: a half");
    assert_eq!(at(6, 4), 128, "right edge: a half");
    assert_eq!(at(4, 4), 255);
    assert_eq!(at(1, 4) | at(7, 4) | at(4, 1) | at(4, 7), 0);
}

#[test]
fn band_on_a_row_boundary_splits_evenly() {
    let cov = render(&[&rect(0.0, 9.5, 16.0, 10.5)], false, 16, 16);
    for x in 0..16 {
        assert_eq!(cov[9 * 16 + x], 128, "row 9, column {x}");
        assert_eq!(cov[10 * 16 + x], 128, "row 10, column {x}");
        assert_eq!(cov[8 * 16 + x] | cov[11 * 16 + x], 0, "column {x}");
    }
}

#[test]
fn winding_direction_does_not_matter() {
    let tri = [(1.3, 0.7), (9.1, 3.4), (4.2, 8.8)];
    let rev = [tri[2], tri[1], tri[0]];
    assert_eq!(
        render(&[&tri], false, 10, 10),
        render(&[&rev], false, 10, 10)
    );
    assert_eq!(render(&[&tri], true, 10, 10), render(&[&rev], true, 10, 10));
}

#[test]
fn ink_equals_area() {
    let tri = [(1.3, 0.7), (29.1, 6.4), (8.2, 27.8)];
    let area = triangle_area(&tri);
    let cov = render(&[&tri], false, 32, 32);
    // Rounding to bytes moves each edge pixel by at most half a level.
    let edge = cov.iter().filter(|&&c| c != 0 && c != 255).count();
    let tolerance = (0.5 / 255.0f64).mul_add(f64::from(u32::try_from(edge).unwrap()), 0.01);
    assert!(
        (ink(&cov) - area).abs() <= tolerance,
        "ink {} vs area {area}",
        ink(&cov)
    );
}

#[test]
fn steep_and_shallow_edges_conserve_area() {
    for tri in [
        [(0.2, 0.1), (31.7, 1.3), (0.4, 2.9)],
        [(0.3, 0.2), (1.1, 31.6), (2.7, 0.4)],
        [(5.25, 5.75), (25.5, 5.75), (15.125, 25.375)],
    ] {
        let area = triangle_area(&tri);
        let cov = render(&[&tri], false, 32, 32);
        let edge = cov.iter().filter(|&&c| c != 0 && c != 255).count();
        let tolerance = (0.5 / 255.0f64).mul_add(f64::from(u32::try_from(edge).unwrap()), 0.02);
        assert!(
            (ink(&cov) - area).abs() <= tolerance,
            "{tri:?}: ink {} vs {area}",
            ink(&cov)
        );
    }
}

#[test]
fn even_odd_overlap_is_a_hole_and_non_zero_is_not() {
    let a = rect(1.0, 1.0, 6.0, 6.0);
    let b = rect(3.0, 3.0, 8.0, 8.0);
    let eo = render(&[&a, &b], true, 10, 10);
    let nz = render(&[&a, &b], false, 10, 10);
    assert_eq!(eo[4 * 10 + 4], 0, "overlap under even-odd");
    assert_eq!(nz[4 * 10 + 4], 255, "overlap under non-zero");
    assert_eq!(eo[2 * 10 + 2], 255);
    assert_eq!(eo[7 * 10 + 7], 255);
}

#[test]
fn even_odd_hole_edge_has_exact_coverage() {
    let outer = rect(0.0, 0.0, 10.0, 10.0);
    let inner = rect(2.5, 2.5, 7.5, 7.5);
    let cov = render(&[&outer, &inner], true, 10, 10);
    assert_eq!(cov[4 * 10 + 2], 128, "half of the pixel lies in the hole");
    assert_eq!(cov[4 * 10 + 4], 0);
    assert_eq!(cov[4 * 10 + 1], 255);
    // A hole wound the same way is a hole under even-odd only.
    let nz = render(&[&outer, &inner], false, 10, 10);
    assert_eq!(nz[4 * 10 + 4], 255);
}

#[test]
fn window_clipping_keeps_coverage_exact() {
    let tri = [(-7.3, 1.2), (19.6, 4.9), (3.1, 14.4)];
    let full = render_window(&[&tri], false, 16, 16, (-20, -20, 40, 40));
    let clipped = render_window(&[&tri], false, 16, 16, (2, 3, 11, 9));
    for y in 0..16 {
        for x in 0..16 {
            let inside = (2..=11).contains(&x) && (3..=9).contains(&y);
            let want = if inside { full[y * 16 + x] } else { 0 };
            assert_eq!(clipped[y * 16 + x], want, "({x}, {y})");
        }
    }
}

#[test]
fn a_shape_past_the_right_edge_fills_to_it() {
    let cov = render_window(&[&rect(2.0, 1.0, 50.0, 3.0)], false, 8, 4, (0, 0, 7, 3));
    assert_eq!(&cov[8..16], &[0, 0, 255, 255, 255, 255, 255, 255]);
}

#[test]
fn whole_pixel_translation_moves_coverage_exactly() {
    let tri = [(1.37, 0.71), (9.13, 3.41), (4.29, 8.83)];
    let moved: Vec<(f64, f64)> = tri.iter().map(|&(x, y)| (x + 5.0, y + 3.0)).collect();
    let a = render(&[&tri], false, 20, 20);
    let b = render(&[&moved], false, 20, 20);
    for y in 0..17 {
        for x in 0..15 {
            assert_eq!(a[y * 20 + x], b[(y + 3) * 20 + x + 5], "({x}, {y})");
        }
    }
}

#[test]
fn non_finite_segments_are_skipped() {
    let mut raster = AreaRaster::new(0, 0, 7, 7);
    raster.add_line(f64::NAN, 0.0, 3.0, 3.0);
    raster.add_line(f64::INFINITY, 0.0, 3.0, f64::NEG_INFINITY);
    let mut rows = 0;
    raster.rows(false, |_, _, _| rows += 1);
    assert!(rows <= 8);
}

#[test]
fn table_products_and_queries() {
    let mut raster = AreaRaster::new(0, 0, 15, 15);
    for (i, &(x0, y0)) in rect(2.5, 2.0, 10.0, 8.0).iter().enumerate() {
        let (x1, y1) = rect(2.5, 2.0, 10.0, 8.0)[(i + 1) % 4];
        raster.add_line(x0, y0, x1, y1);
    }
    let table = CoverageTable::from_raster(&mut raster, false);
    assert_eq!(table.coverage(2, 4), 128);
    assert_eq!(table.coverage(5, 4), 255);
    assert_eq!(table.coverage(10, 4), 0);
    assert_eq!(table.coverage(5, 8), 0);
    assert!(table.is_full(3, 9, 4));
    assert!(!table.is_full(2, 9, 4));
    let mut row = [200u8; 12];
    table.mul_row(4, 0, &mut row);
    assert_eq!(row, [0, 0, 100, 200, 200, 200, 200, 200, 200, 200, 0, 0]);

    let squared = CoverageTable::product([&table, &table]);
    assert_eq!(squared.coverage(2, 4), 64);
    assert_eq!(squared.coverage(5, 4), 255);
    assert!(squared.is_full(3, 9, 4));

    let moved = table.translated(3, -1);
    assert_eq!(moved.coverage(5, 3), 128);
    assert_eq!(moved.coverage(12, 3), 255);
}
