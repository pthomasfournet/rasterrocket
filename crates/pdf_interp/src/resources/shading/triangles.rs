//! The triangles of a decoded mesh shading, indexed by device row so that a
//! fill visits only the triangles that can meet it.
//!
//! A mesh is resolved once per page and painted by every fill that uses it;
//! a small fill over a large mesh would otherwise walk every triangle.  The
//! triangles are bucketed into bands of [`BAND_ROWS`] device rows from row 0;
//! a query for a rectangle gathers the triangles of the bands it spans whose
//! bounds meet the rectangle and yields them once each, in mesh order, so
//! that overlapping triangles paint in the order the mesh gives them.  A
//! triangle spanning more than [`MAX_TRIANGLE_BANDS`] bands, or lying below
//! the banded rows, is held once in an unbanded list and tested against
//! every query; a triangle with a non-finite vertex has no bounds and is
//! yielded for every query.

use raster::Clip;
use raster::shading::gouraud::GouraudVertex;

/// Device rows per band.
const BAND_ROWS: i32 = 32;

/// Bands cover rows `0..MAX_BANDS × BAND_ROWS`; a triangle entirely below
/// them is unbanded.
const MAX_BANDS: u32 = 1 << 14;

/// Bands a triangle may span and still be banded.  The index holds one
/// entry per band a triangle spans, so without this bound a mesh of tall
/// triangles costs up to [`MAX_BANDS`] entries per triangle.
const MAX_TRIANGLE_BANDS: usize = 64;

/// A triangle list with a row index.
pub struct Mesh {
    triangles: Vec<[GouraudVertex; 3]>,
    /// `(x_min, y_min, x_max, y_max)` of each triangle.
    bounds: Vec<[f64; 4]>,
    /// For each band, the triangles meeting it, in mesh order.
    bands: Vec<Vec<u32>>,
    /// Triangles with non-finite bounds, bounds below the banded rows, or
    /// bounds spanning more than [`MAX_TRIANGLE_BANDS`] bands.
    unbanded: Vec<u32>,
}

impl Mesh {
    /// Index `triangles`.
    #[must_use]
    pub fn new(triangles: Vec<[GouraudVertex; 3]>) -> Self {
        let bounds: Vec<[f64; 4]> = triangles
            .iter()
            .map(|t| {
                // `min` and `max` skip a NaN operand, so a non-finite
                // vertex is checked for explicitly.
                if t.iter().any(|v| !v.x.is_finite() || !v.y.is_finite()) {
                    return [f64::NAN; 4];
                }
                t.iter().fold(
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |b, v| [b[0].min(v.x), b[1].min(v.y), b[2].max(v.x), b[3].max(v.y)],
                )
            })
            .collect();
        let mut bands: Vec<Vec<u32>> = Vec::new();
        let mut unbanded = Vec::new();
        for (i, b) in (0u32..).zip(&bounds) {
            let Some((first, last)) =
                band_range(b[1], b[3]).filter(|(first, last)| last - first < MAX_TRIANGLE_BANDS)
            else {
                unbanded.push(i);
                continue;
            };
            if bands.len() <= last {
                bands.resize_with(last + 1, Vec::new);
            }
            for band in &mut bands[first..=last] {
                band.push(i);
            }
        }
        Self {
            triangles,
            bounds,
            bands,
            unbanded,
        }
    }

    /// Every triangle, in mesh order.
    #[must_use]
    pub fn triangles(&self) -> &[[GouraudVertex; 3]] {
        &self.triangles
    }

    /// The triangles whose bounds meet the rectangle of `clip`, and every
    /// triangle without bounds, each once, in mesh order.
    pub fn meeting(&self, clip: &Clip) -> impl Iterator<Item = [GouraudVertex; 3]> + '_ {
        let (x0, y0) = (f64::from(clip.x_min_i), f64::from(clip.y_min_i));
        let (x1, y1) = (f64::from(clip.x_max_i) + 1.0, f64::from(clip.y_max_i) + 1.0);
        let range = band_range(y0, y1 - 1.0)
            .map(|(first, last)| (first, last.min(self.bands.len().saturating_sub(1))))
            .filter(|(first, _)| *first < self.bands.len());
        // Non-finite bounds are NaN throughout.
        let meets = |&i: &u32| {
            let b = self.bounds[i as usize];
            b[0].is_nan() || (b[2] >= x0 && b[0] <= x1 && b[3] >= y0 && b[1] <= y1)
        };
        let mut found: Vec<u32> = range
            .into_iter()
            .flat_map(|(first, last)| self.bands[first..=last].iter().flatten().copied())
            .chain(self.unbanded.iter().copied())
            .filter(meets)
            .collect();
        found.sort_unstable();
        found.dedup();
        found.into_iter().map(move |i| self.triangles[i as usize])
    }
}

/// The bands rows `y_min..=y_max` span, or `None` when the range is not
/// finite or lies entirely below the banded rows.
fn band_range(y_min: f64, y_max: f64) -> Option<(usize, usize)> {
    if !y_min.is_finite() || !y_max.is_finite() || y_max < 0.0 {
        return None;
    }
    let max_rows = f64::from(BAND_ROWS) * f64::from(MAX_BANDS);
    if y_min >= max_rows {
        return None;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "both values are clamped to [0, MAX_BANDS × BAND_ROWS) first"
    )]
    let band = |y: f64| (y.clamp(0.0, max_rows - 1.0) / f64::from(BAND_ROWS)) as usize;
    Some((band(y_min), band(y_max)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(x0: f64, y0: f64, x1: f64, y1: f64) -> [GouraudVertex; 3] {
        let v = |x, y| GouraudVertex {
            x,
            y,
            color: [0; 3],
        };
        [v(x0, y0), v(x1, y0), v(x0, y1)]
    }

    fn clip(x0: f64, y0: f64, x1: f64, y1: f64) -> Clip {
        Clip::new(x0, y0, x1, y1, false)
    }

    fn ids(mesh: &Mesh, c: &Clip) -> Vec<f64> {
        let mut v: Vec<f64> = mesh.meeting(c).map(|t| t[0].x).collect();
        v.sort_by(f64::total_cmp);
        v
    }

    #[test]
    fn a_query_yields_each_meeting_triangle_once() {
        // Triangle 1 spans rows 10..100 (three bands); triangle 2 rows
        // 200..210; triangle 3 columns 500..600 only.
        let mesh = Mesh::new(vec![
            tri(1.0, 10.0, 50.0, 100.0),
            tri(2.0, 200.0, 50.0, 210.0),
            tri(500.0, 10.0, 600.0, 100.0),
        ]);
        assert_eq!(ids(&mesh, &clip(0.0, 0.0, 100.0, 300.0)), [1.0, 2.0]);
        assert_eq!(ids(&mesh, &clip(0.0, 40.0, 100.0, 90.0)), [1.0]);
        assert_eq!(ids(&mesh, &clip(0.0, 120.0, 100.0, 190.0)), []);
        assert_eq!(ids(&mesh, &clip(0.0, 0.0, 1000.0, 20.0)), [1.0, 500.0]);
        assert_eq!(
            ids(&mesh, &clip(0.0, 0.0, 1000.0, 1000.0)),
            [1.0, 2.0, 500.0]
        );
    }

    #[test]
    fn a_triangle_above_the_page_is_banded_from_row_zero() {
        let mesh = Mesh::new(vec![tri(1.0, -50.0, 10.0, 5.0)]);
        assert_eq!(ids(&mesh, &clip(0.0, 0.0, 100.0, 100.0)), [1.0]);
        assert_eq!(ids(&mesh, &clip(0.0, 10.0, 100.0, 100.0)), []);
    }

    #[test]
    fn a_tall_triangle_takes_one_index_entry_and_meets_by_bounds() {
        let mesh = Mesh::new(vec![tri(3.0, 0.0, 10.0, 400_000.0)]);
        assert_eq!(mesh.bands.iter().map(Vec::len).sum::<usize>(), 0);
        assert_eq!(mesh.unbanded, [0]);
        assert_eq!(ids(&mesh, &clip(0.0, 9_000.0, 5.0, 9_100.0)), [3.0]);
        assert_eq!(ids(&mesh, &clip(50.0, 9_000.0, 90.0, 9_100.0)), []);
    }

    #[test]
    fn a_non_finite_triangle_is_yielded_for_every_query() {
        let mesh = Mesh::new(vec![tri(7.0, f64::NAN, 10.0, 5.0)]);
        assert_eq!(ids(&mesh, &clip(900.0, 900.0, 950.0, 950.0)), [7.0]);
        assert_eq!(mesh.triangles().len(), 1);
    }
}
