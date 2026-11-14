//! The extent of a polyline whose points lie on one line.

/// A point `(x, y)`.
pub(crate) type Point = (f64, f64);

/// The disjoint intervals of a line that the segments `segs` cover, each
/// as its two ends, the end with the smaller coordinates first and the
/// intervals in that order along the line; `None` when the segments'
/// points do not all lie on one line, or coincide.
///
/// A point is on the line through the extreme points `lo` and `hi` when
/// its distance from it is within `10⁻⁹ · |hi − lo|`.  The intervals do
/// not depend on the order or direction of the segments, and each part of
/// the line covered by several segments is covered once.
pub(crate) fn intervals(segs: &[(Point, Point)]) -> Option<Vec<[Point; 2]>> {
    let points = || segs.iter().flat_map(|&seg| <[Point; 2]>::from(seg));
    let by_coords = |a: &Point, b: &Point| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1));
    let lo = points().min_by(by_coords)?;
    let hi = points().max_by(by_coords)?;
    let (dx, dy) = (hi.0 - lo.0, hi.1 - lo.1);
    let extent = dx.hypot(dy);
    if !(extent > 0.0 && extent.is_finite()) {
        return None;
    }
    let off_line = |p: Point| dx.mul_add(p.1 - lo.1, -(dy * (p.0 - lo.0))).abs();
    if points().any(|p| off_line(p) > 1e-9 * extent * extent) {
        return None;
    }
    let along = |p: Point| dx.mul_add(p.0 - lo.0, dy * (p.1 - lo.1));
    let mut ends: Vec<[(f64, Point); 2]> = segs
        .iter()
        .map(|&(a, b)| {
            let (ta, tb) = (along(a), along(b));
            if ta <= tb {
                [(ta, a), (tb, b)]
            } else {
                [(tb, b), (ta, a)]
            }
        })
        .collect();
    ends.sort_by(|a, b| a[0].0.total_cmp(&b[0].0));
    let mut out = Vec::new();
    let mut merged = ends[0];
    for &next in &ends[1..] {
        if next[0].0 <= merged[1].0 {
            if next[1].0 > merged[1].0 {
                merged[1] = next[1];
            }
        } else {
            out.push([merged[0].1, merged[1].1]);
            merged = next;
        }
    }
    out.push([merged[0].1, merged[1].1]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_and_reversed_segments_merge() {
        let segs = [
            ((0.0, 0.0), (4.0, 2.0)),
            ((6.0, 3.0), (2.0, 1.0)),
            ((8.0, 4.0), (10.0, 5.0)),
        ];
        assert_eq!(
            intervals(&segs),
            Some(vec![[(0.0, 0.0), (6.0, 3.0)], [(8.0, 4.0), (10.0, 5.0)]])
        );
    }

    #[test]
    fn off_line_and_degenerate_segments_have_no_intervals() {
        assert_eq!(
            intervals(&[((0.0, 0.0), (4.0, 0.0)), ((4.0, 0.0), (2.0, 1.0))]),
            None
        );
        assert_eq!(intervals(&[((1.0, 1.0), (1.0, 1.0))]), None);
        assert_eq!(intervals(&[]), None);
    }
}
