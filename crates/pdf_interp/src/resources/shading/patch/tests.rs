//! Tests of patch subdivision, tessellation and assembly.

use super::*;

#[expect(
    clippy::too_many_arguments,
    reason = "test helper: 4 corner positions + 4 corner colours map naturally to 8 args; bundling into a struct just shifts the boilerplate to the call sites"
)]
fn flat_patch(
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    c00: [u8; 3],
    c01: [u8; 3],
    c10: [u8; 3],
    c11: [u8; 3],
) -> Patch<[u8; 3]> {
    // Bilinear interpolation of control points across the 4×4 grid.
    let mut p = Patch::zero();
    for r in 0..4u32 {
        let tr = f64::from(r) / 3.0;
        for c in 0..4u32 {
            let tc = f64::from(c) / 3.0;
            let (ri, ci) = (r as usize, c as usize);
            p.xy[ri][ci][0] = (x1 - x0).mul_add(tc, x0);
            p.xy[ri][ci][1] = (y1 - y0).mul_add(tr, y0);
        }
    }
    p.color[0][0] = c00;
    p.color[0][1] = c01;
    p.color[1][0] = c10;
    p.color[1][1] = c11;
    p
}

#[test]
fn split_patch_u_flat_corners_unchanged() {
    let p = flat_patch(
        0.0,
        0.0,
        6.0,
        4.0,
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    );
    let (left, right) = split_patch_u(&p);
    // Top-left corner should be exactly preserved in both halves.
    assert_eq!(left.xy[0][0], p.xy[0][0]);
    assert_eq!(right.xy[3][0], p.xy[3][0]);
    // The u-split seam (left's u=1 edge, right's u=0 edge) sits at the
    // y-midpoint of the patch's u=0 / u=3 control rows.
    let mid_y = f64::midpoint(p.xy[0][0][1], p.xy[3][0][1]);
    assert!(
        (left.xy[3][0][1] - mid_y).abs() < 1e-9,
        "left bottom y={} expected={}",
        left.xy[3][0][1],
        mid_y
    );
    assert!(
        (right.xy[0][0][1] - mid_y).abs() < 1e-9,
        "right top y={} expected={}",
        right.xy[0][0][1],
        mid_y
    );
    // Colors: left's u=1 edge should be midpoint of original u edges.
    for ch in 0..3 {
        let expected = u8::midpoint(p.color[0][0][ch], p.color[1][0][ch]);
        assert!(
            (i16::from(left.color[1][0][ch]) - i16::from(expected)).abs() <= 1,
            "color mismatch ch={ch}"
        );
    }
}

#[test]
fn split_patch_v_flat_corners_unchanged() {
    let p = flat_patch(
        0.0,
        0.0,
        8.0,
        6.0,
        [10, 20, 30],
        [200, 100, 50],
        [30, 60, 90],
        [180, 90, 45],
    );
    let (left, right) = split_patch_v(&p);
    assert_eq!(left.xy[0][0], p.xy[0][0]);
    assert_eq!(right.xy[0][3], p.xy[0][3]);
    let mid_x = f64::midpoint(p.xy[0][0][0], p.xy[0][3][0]);
    assert!((left.xy[0][3][0] - mid_x).abs() < 1e-9);
    assert!((right.xy[0][0][0] - mid_x).abs() < 1e-9);
}

#[test]
fn corner_spread_uniform_is_zero() {
    let mut p = Patch::zero();
    let c = [128u8, 64, 32];
    p.color[0][0] = c;
    p.color[0][1] = c;
    p.color[1][0] = c;
    p.color[1][1] = c;
    assert_eq!(corner_spread(&p), 0.0);
}

#[test]
fn corner_spread_max_diff_is_one() {
    let mut p = Patch::<[u8; 3]>::zero();
    p.color[0][0] = [0, 0, 0];
    p.color[0][1] = [255, 0, 0];
    p.color[1][0] = [0, 0, 0];
    p.color[1][1] = [0, 0, 0];
    let d = corner_spread(&p);
    assert!((d - 1.0).abs() < 1e-9, "expected 1.0, got {d}");
}

#[test]
fn tessellate_patch_uniform_emits_two_triangles() {
    // A patch where all corners have the same colour → depth=0 leaf immediately.
    let p = flat_patch(
        0.0,
        0.0,
        1.0,
        1.0,
        [128, 128, 128],
        [128, 128, 128],
        [128, 128, 128],
        [128, 128, 128],
    );
    let doc = crate::test_helpers::empty_doc();
    let source = super::super::mesh::DirectColors::new(super::super::ShadingCs {
        doc: &doc,
        space: &super::super::ColorSpace::DeviceRgb,
        target: crate::ColourTarget::SRGB,
    });
    let mut sink = TriangleSink::new();
    tessellate_patch(&source, p, &mut sink);
    assert_eq!(
        sink.finish().len(),
        2,
        "expected exactly 2 triangles for uniform-colour patch"
    );
}

/// The triangles `tessellate_patch` emits for the 8-bit RGB patch `p`.
fn triangles(p: &Patch<[u8; 3]>) -> Vec<[[f64; 2]; 3]> {
    let doc = crate::test_helpers::empty_doc();
    let source = super::super::mesh::DirectColors::new(super::super::ShadingCs {
        doc: &doc,
        space: &super::super::ColorSpace::DeviceRgb,
        target: crate::ColourTarget::SRGB,
    });
    let mut sink = TriangleSink::new();
    tessellate_patch(&source, *p, &mut sink);
    sink.finish()
        .iter()
        .map(|t| t.each_ref().map(|v| [v.x, v.y]))
        .collect()
}

/// Whether a triangle of `tris` contains `p`.
fn covered(tris: &[[[f64; 2]; 3]], p: [f64; 2]) -> bool {
    let cross = |a: [f64; 2], b: [f64; 2]| {
        (b[0] - a[0]).mul_add(p[1] - a[1], -(b[1] - a[1]) * (p[0] - a[0]))
    };
    tris.iter().any(|&[a, b, c]| {
        let d = [cross(a, b), cross(b, c), cross(c, a)];
        d.iter().all(|&v| v >= 0.0) || d.iter().all(|&v| v <= 0.0)
    })
}

/// A patch of one colour whose side bulges is drawn to the curve, not as
/// its corner quadrilateral: the bulge is covered and the area outside it
/// is not.
#[test]
fn a_uniform_patch_covers_its_curved_side() {
    let grey = [128, 128, 128];
    let mut p = flat_patch(0.0, 0.0, 30.0, 30.0, grey, grey, grey, grey);
    // The u = 0 side (row 0, y = 0) bulges to y = -9 at its middle.
    p.xy[0][1][1] = -12.0;
    p.xy[0][2][1] = -12.0;
    let tris = triangles(&p);
    assert!(covered(&tris, [15.0, -8.0]), "inside the bulge");
    assert!(covered(&tris, [15.0, 15.0]), "inside the square");
    assert!(!covered(&tris, [15.0, -9.6]), "beyond the bulge");
    assert!(
        !covered(&tris, [1.0, -3.0]),
        "outside the curve near a corner"
    );
}

/// A patch whose sides are straight is its corner quadrilateral, however
/// its control points are spaced along them.
#[test]
fn straight_sides_with_clustered_control_points_are_flat() {
    let grey = [128, 128, 128];
    let mut p = flat_patch(0.0, 0.0, 30.0, 30.0, grey, grey, grey, grey);
    for i in [1, 2] {
        let near = if i == 1 { 0 } else { 3 };
        p.xy[0][i] = p.xy[0][near];
        p.xy[3][i] = p.xy[3][near];
        p.xy[i][0] = p.xy[near][0];
        p.xy[i][3] = p.xy[near][3];
    }
    fill_type6_interior(&mut p);
    assert_eq!(triangles(&p).len(), 2);
}

#[test]
fn tessellate_patch_stops_at_the_pixel() {
    // Four distinct corner colours over half a pixel: one leaf.
    let p = flat_patch(
        0.0,
        0.0,
        0.5,
        0.5,
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    );
    let doc = crate::test_helpers::empty_doc();
    let source = super::super::mesh::DirectColors::new(super::super::ShadingCs {
        doc: &doc,
        space: &super::super::ColorSpace::DeviceRgb,
        target: crate::ColourTarget::SRGB,
    });
    let mut sink = TriangleSink::new();
    tessellate_patch(&source, p, &mut sink);
    assert_eq!(sink.finish().len(), 2);
}

/// The subdivision tolerance is three 8-bit levels at 8 bits and one
/// 8-bit level at 16 bits.
#[test]
fn the_tolerance_follows_the_depth() {
    assert_eq!(color_delta::<u8>().to_bits(), (3.0f64 / 255.0).to_bits());
    assert_eq!(color_delta::<u16>().to_bits(), (1.0f64 / 255.0).to_bits());
}

/// A 64-pixel patch whose corners differ by twelve 8-bit levels stops
/// subdividing at 8 bits once they differ by three, and subdivides
/// further at 16 bits, to at most sixteen times as many triangles.
#[test]
fn a_16_bit_patch_subdivides_to_its_finer_tolerance() {
    let p = flat_patch(
        0.0,
        0.0,
        64.0,
        64.0,
        [100, 100, 100],
        [112, 100, 100],
        [100, 112, 100],
        [100, 100, 112],
    );
    let doc = crate::test_helpers::empty_doc();
    let cs = super::super::ShadingCs {
        doc: &doc,
        space: &super::super::ColorSpace::DeviceRgb,
        target: crate::ColourTarget::SRGB,
    };
    let mut narrow = TriangleSink::new();
    tessellate_patch(
        &super::super::mesh::DirectColors::<_, u8>::new(cs),
        p,
        &mut narrow,
    );
    let wide_patch = Patch {
        xy: p.xy,
        color: p
            .color
            .map(|row| row.map(|c| c.map(<u16 as color::Sample>::from_u8))),
    };
    let mut wide = TriangleSink::new();
    tessellate_patch(
        &super::super::mesh::DirectColors::<_, u16>::new(cs),
        wide_patch,
        &mut wide,
    );
    let (narrow, wide) = (narrow.finish().len(), wide.finish().len());
    eprintln!("64-pixel patch: {narrow} triangles at 8 bits, {wide} at 16");
    assert!(narrow < wide && wide <= 16 * narrow, "{narrow} and {wide}");
}

/// The boundary of the square `[x0, x0 + 3] × [y0, y0 + 3]` in stream
/// order, one control point per unit.
fn square(x0: f64, y0: f64) -> Vec<[f64; 2]> {
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (2, 3),
        (1, 3),
        (0, 3),
        (0, 2),
        (0, 1),
    ]
    .map(|(x, y)| [x0 + f64::from(x), y0 + f64::from(y)])
    .to_vec()
}

/// Flags 1, 2 and 3 continue the boundary from corners p03, p33 and
/// p30 of the previous patch, with that corner's colour and the next.
#[test]
fn continued_outline_shares_the_flagged_edge() {
    let prev = Outline::new(
        PatchKind::Coons,
        &square(0.0, 0.0),
        &[[10u8; 3], [20; 3], [30; 3], [40; 3]],
    )
    .expect("outline");
    for (edge, shared, colors) in [
        (
            1,
            [[3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [3.0, 3.0]],
            [20, 30],
        ),
        (
            2,
            [[3.0, 3.0], [2.0, 3.0], [1.0, 3.0], [0.0, 3.0]],
            [30, 40],
        ),
        (
            3,
            [[0.0, 3.0], [0.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
            [40, 10],
        ),
    ] {
        let next = prev
            .continued(
                PatchKind::Coons,
                edge,
                &[[9.0, 9.0]; 8],
                &[[50u8; 3], [60; 3]],
            )
            .expect("outline");
        assert_eq!(next.boundary[..4], shared, "flag {edge}");
        assert_eq!(
            next.colors,
            [[colors[0]; 3], [colors[1]; 3], [50; 3], [60; 3]],
            "flag {edge}"
        );
    }
}

/// A Coons patch beside a unit square, sharing its right edge: the
/// continuation record lists the rest of the boundary from the end of
/// the shared edge, so the patch is the square `[3, 6] × [0, 3]`.
#[test]
fn a_flag_1_record_continues_from_the_shared_edge() {
    let mut sh = Dictionary::new();
    for (k, v) in [
        ("BitsPerCoordinate", 8),
        ("BitsPerComponent", 8),
        ("BitsPerFlag", 8),
    ] {
        sh.set(k, pdf::Object::Integer(v));
    }
    let coord = |pts: &[[f64; 2]]| -> Vec<u8> {
        pts.iter()
            .flatten()
            .map(|&v| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "test coordinates are small non-negative integers"
                )]
                let b = v as u8;
                b
            })
            .collect()
    };
    let mut data = vec![0];
    data.extend(coord(&square(0.0, 0.0)));
    data.extend([128; 4]);
    data.push(1);
    data.extend(coord(&[
        [4.0, 3.0],
        [5.0, 3.0],
        [6.0, 3.0],
        [6.0, 2.0],
        [6.0, 1.0],
        [6.0, 0.0],
        [5.0, 0.0],
        [4.0, 0.0],
    ]));
    data.extend([128; 2]);

    let doc = crate::test_helpers::empty_doc();
    let source = super::super::mesh::DirectColors::new(super::super::ShadingCs {
        doc: &doc,
        space: &super::super::ColorSpace::DeviceGray,
        target: crate::ColourTarget::SRGB,
    });
    let decode = [0.0, 255.0, 0.0, 255.0, 0.0, 1.0];
    // page_h = 0: device y is the negated shading y.
    let ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let format = VertexFormat::parse(&sh, 6, &decode, &ctm, 0.0).expect("format");
    let mut sink = TriangleSink::new();
    decode_patch_mesh(&sh, &data, &source, &format, PatchKind::Coons, &mut sink);
    let tris = sink.finish();
    assert_eq!(tris.len(), 4, "two uniform patches, two triangles each");
    let corners = |t: &[GouraudVertex; 3]| t.map(|v| [v.x, -v.y]);
    // Leaf triangles are (p00, p03, p30) and (p03, p33, p30).
    assert_eq!(
        corners(&tris[2]),
        [[3.0, 0.0], [3.0, 3.0], [6.0, 0.0]],
        "the second patch covers [3, 6] × [0, 3]"
    );
    assert_eq!(corners(&tris[3]), [[3.0, 3.0], [6.0, 3.0], [6.0, 0.0]]);
}

#[test]
fn build_patch_type6_interior_formula() {
    // Use a simple square boundary with known analytic interior.
    // p[0][0]=(0,0)  p[0][1]=(1,0) p[0][2]=(2,0) p[0][3]=(3,0)
    // p[1][3]=(3,1)  p[2][3]=(3,2)
    // p[3][3]=(3,3)  p[3][2]=(2,3) p[3][1]=(1,3) p[3][0]=(0,3)
    // p[2][0]=(0,2)  p[1][0]=(0,1)
    //
    // For a uniform-spaced rectilinear grid the Coons interior formula collapses to
    // simple bilinear: p[1][1]=(1,1), p[1][2]=(2,1), p[2][1]=(1,2), p[2][2]=(2,2).
    let pts: Vec<[f64; 2]> = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0], // p[0][0..3]
        [3.0, 1.0],
        [3.0, 2.0], // p[1][3], p[2][3]
        [3.0, 3.0],
        [2.0, 3.0],
        [1.0, 3.0],
        [0.0, 3.0], // p[3][3..0]
        [0.0, 2.0],
        [0.0, 1.0], // p[2][0], p[1][0]
    ];
    let colors: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
    let p = Outline::new(PatchKind::Coons, &pts, &colors)
        .expect("12 points and 4 colours")
        .patch();
    for (r, c, ex, ey) in [
        (1, 1, 1.0, 1.0),
        (1, 2, 2.0, 1.0),
        (2, 1, 1.0, 2.0),
        (2, 2, 2.0, 2.0),
    ] {
        assert!(
            (p.xy[r][c][0] - ex).abs() < 1e-9 && (p.xy[r][c][1] - ey).abs() < 1e-9,
            "interior[{r}][{c}] = {:?} expected ({ex},{ey})",
            p.xy[r][c]
        );
    }
}
