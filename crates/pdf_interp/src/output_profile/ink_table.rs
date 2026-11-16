//! A conversion of process ink tabulated over the ink cube and
//! interpolated in 4-simplices.

use moxcms::{CmsError, TransformExecutor};

/// A conversion of CMYK ink to at most four channels, tabulated at the
/// `(span + 1)⁴` nodes of a grid over the ink cube: node `(c, m, y, k)`,
/// `c` slowest and `k` fastest, holds the 16-bit value of the ink
/// `v / span` for `v` = `c`, `m`, `y`, `k`.
pub(crate) struct InkTable {
    span: u32,
    channels: usize,
    strides: [usize; 4],
    nodes: Box<[u16]>,
}

impl InkTable {
    /// The conversion `convert` tabulated on a grid of `span` cells along
    /// each ink axis into `channels` values: `convert` takes the inks of a
    /// run of nodes, four values of `[0, 1]` each, and writes their
    /// `channels` values of `[0, 1]`.
    ///
    /// # Errors
    ///
    /// The first error `convert` returns.
    ///
    /// # Panics
    ///
    /// For no channel or more than four.
    pub(crate) fn tabulate<E>(
        span: u32,
        channels: usize,
        mut convert: impl FnMut(&[f64], &mut [f64]) -> Result<(), E>,
    ) -> Result<Self, E> {
        assert!((1..=4).contains(&channels), "one to four channels");
        let grid = span as usize + 1;
        let row = grid * grid * grid;
        let full = f64::from(span);
        let mut ink = Vec::with_capacity(row * 4);
        let mut out = vec![0.0; row * channels];
        let mut nodes = Vec::with_capacity(row * grid * channels);
        for c in 0..=span {
            ink.clear();
            for m in 0..=span {
                for y in 0..=span {
                    for k in 0..=span {
                        ink.extend([c, m, y, k].map(|i| f64::from(i) / full));
                    }
                }
            }
            convert(&ink, &mut out)?;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a value clamped to [0, 1] scaled to the u16 range"
            )]
            nodes.extend(
                out.iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16),
            );
        }
        Ok(Self {
            span,
            channels,
            strides: [grid * grid * grid, grid * grid, grid, 1],
            nodes: nodes.into_boxed_slice(),
        })
    }

    /// The four values of the ink `ink` at the depth of `S`, converted at
    /// 16 bits ([`Self::eval`]).
    ///
    /// # Panics
    ///
    /// For a table of other than four channels.
    pub(crate) fn convert<S: color::Sample>(&self, ink: [S; 4]) -> [S; 4] {
        assert_eq!(self.channels, 4, "a table of four channels");
        let mut out = [0u16; 4];
        self.eval(ink.map(<u16 as color::Sample>::from_sample), &mut out);
        out.map(S::from_sample)
    }

    /// The number of values of a node.
    pub(crate) const fn channels(&self) -> usize {
        self.channels
    }

    /// The value of the 16-bit ink `ink` into `out`: the interpolation in
    /// the simplex of the grid cell around it whose vertices are visited in
    /// decreasing order of the ink's fractional positions within the cell.
    pub(crate) fn eval(&self, ink: [u16; 4], out: &mut [u16]) {
        let full = u32::from(u16::MAX);
        let mut base = 0usize;
        let mut frac = [(0u32, 0usize); 4];
        for ((&v, &stride), f) in ink.iter().zip(&self.strides).zip(&mut frac) {
            let scaled = u32::from(v) * self.span;
            let cell = (scaled / full).min(self.span - 1);
            base += cell as usize * stride;
            *f = (scaled - cell * full, stride);
        }
        frac.sort_unstable_by_key(|f| std::cmp::Reverse(f.0));
        let n = self.channels;
        let mut acc = [0u64; 4];
        let mut vertex = base;
        let mut previous = full;
        for &(f, stride) in &frac {
            let weight = u64::from(previous - f);
            for (a, &node) in acc.iter_mut().zip(&self.nodes[vertex * n..vertex * n + n]) {
                *a += weight * u64::from(node);
            }
            previous = f;
            vertex += stride;
        }
        for (a, &node) in acc.iter_mut().zip(&self.nodes[vertex * n..vertex * n + n]) {
            *a += u64::from(previous) * u64::from(node);
        }
        let half = u64::from(full) / 2;
        for (o, &a) in out.iter_mut().zip(&acc) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a convex combination of u16 nodes, divided by its weight sum, fits u16"
            )]
            let value = ((a + half) / u64::from(full)) as u16;
            *o = value;
        }
    }
}

impl std::fmt::Debug for InkTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InkTable")
            .field("span", &self.span)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl TransformExecutor<u16> for InkTable {
    fn transform(&self, src: &[u16], dst: &mut [u16]) -> Result<(), CmsError> {
        let (pixels, []) = src.as_chunks::<4>() else {
            return Err(CmsError::LaneSizeMismatch);
        };
        if dst.len() != pixels.len() * self.channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        for (&ink, out) in pixels.iter().zip(dst.chunks_exact_mut(self.channels)) {
            self.eval(ink, out);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A conversion that is not linear in the inks.
    fn convert(ink: &[f64], out: &mut [f64]) {
        for (px, o) in ink
            .as_chunks::<4>()
            .0
            .iter()
            .zip(out.as_chunks_mut::<3>().0)
        {
            *o = [px[0] * px[1], 1.0 - px[2], px[3] * px[3]];
        }
    }

    /// At the 16-bit ink nearest a node the interpolation is the node's
    /// value within a level, a node lying at `v · 65535 / span`, which
    /// 16-bit ink reaches only within half a level; between nodes it is a
    /// convex combination of the cell's.
    #[test]
    fn the_interpolation_reproduces_the_nodes() {
        const SPAN: u32 = 16;
        let table = InkTable::tabulate(SPAN, 3, |i, o| {
            convert(i, o);
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        assert_eq!(table.channels(), 3);
        let node = |i: u32| u16::try_from((i * u32::from(u16::MAX) + SPAN / 2) / SPAN).unwrap();
        for (c, m, y, k) in [(0, 0, 0, 0), (16, 16, 16, 16), (3, 7, 11, 5), (16, 0, 9, 2)] {
            let mut out = [0u16; 3];
            table.eval([c, m, y, k].map(node), &mut out);
            let mut want = [0.0; 3];
            convert(
                &[c, m, y, k].map(|v| f64::from(v) / f64::from(SPAN)),
                &mut want,
            );
            for (o, w) in out.iter().zip(want) {
                assert!(
                    w.mul_add(-65535.0, f64::from(*o)).abs() <= 1.0,
                    "node {c} {m} {y} {k}: {out:?} for {want:?}"
                );
            }
        }
        let mut mid = [0u16; 3];
        table.eval([32_768, 32_768, 0, 65_535], &mut mid);
        assert!(mid[0].abs_diff(16_384) <= 64 && mid[1] == 65_535, "{mid:?}");
    }
}
