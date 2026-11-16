//! A conversion of `N` device channels tabulated over the device cube and
//! interpolated in its simplices.

use moxcms::{CmsError, TransformExecutor};

/// A conversion of `N` device channels to at most four channels,
/// tabulated at the `(span + 1)^N` nodes of a grid over the device cube:
/// the node of indices `(i₀, …, i_{N−1})`, `i₀` slowest, holds the 16-bit
/// value of the device value `i / span`.
pub(crate) struct GridTable<const N: usize> {
    span: u32,
    channels: usize,
    strides: [usize; N],
    nodes: Box<[u16]>,
}

impl<const N: usize> GridTable<N> {
    /// The conversion `convert` tabulated on a grid of `span` cells along
    /// each device axis into `channels` values: `convert` takes the device
    /// values of a run of nodes, `N` values of `[0, 1]` each, and writes
    /// their `channels` values of `[0, 1]`.
    ///
    /// # Errors
    ///
    /// The first error `convert` returns.
    ///
    /// # Panics
    ///
    /// For no input or no channel, more than four channels, or a span
    /// below 1.
    pub(crate) fn tabulate<E>(
        span: u32,
        channels: usize,
        mut convert: impl FnMut(&[f64], &mut [f64]) -> Result<(), E>,
    ) -> Result<Self, E> {
        assert!(N >= 1, "at least one input");
        assert!((1..=4).contains(&channels), "one to four channels");
        assert!(span >= 1, "at least one cell");
        let grid = span as usize + 1;
        let mut strides = [1usize; N];
        for i in (0..N - 1).rev() {
            strides[i] = strides[i + 1] * grid;
        }
        // One call per node of the slowest axis: a slice of the grid.
        let slice = strides[0];
        let full = f64::from(span);
        let mut input = Vec::with_capacity(slice * N);
        let mut out = vec![0.0; slice * channels];
        let mut nodes = Vec::with_capacity(slice * grid * channels);
        for first in 0..grid {
            input.clear();
            for rest in 0..slice {
                let index = first * slice + rest;
                input.extend(strides.iter().map(|&stride| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "a node index is below span + 1 ≤ 2^32"
                    )]
                    let i = (index / stride % grid) as f64;
                    i / full
                }));
            }
            convert(&input, &mut out)?;
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
            strides,
            nodes: nodes.into_boxed_slice(),
        })
    }

    /// The `M` values of the device value `px` at the depth of `S`,
    /// converted at 16 bits ([`Self::eval`]).
    ///
    /// # Panics
    ///
    /// For a table of other than `M` channels.
    pub(crate) fn convert<S: color::Sample, const M: usize>(&self, px: [S; N]) -> [S; M] {
        assert_eq!(self.channels, M, "a table of {M} channels");
        let mut out = [0u16; M];
        self.eval(px.map(<u16 as color::Sample>::from_sample), &mut out);
        out.map(S::from_sample)
    }

    /// The number of values of a node.
    pub(crate) const fn channels(&self) -> usize {
        self.channels
    }

    /// The value of the 16-bit device value `px` into `out`: the
    /// interpolation in the simplex of the grid cell around it whose
    /// vertices are visited in decreasing order of the value's fractional
    /// positions within the cell.  The simplex containing the cell's main
    /// diagonal is shared by every point of it, so a value of equal
    /// components interpolates between the nodes of equal components alone.
    pub(crate) fn eval(&self, px: [u16; N], out: &mut [u16]) {
        let full = u32::from(u16::MAX);
        let mut base = 0usize;
        let mut frac = [(0u32, 0usize); N];
        for ((&v, &stride), f) in px.iter().zip(&self.strides).zip(&mut frac) {
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

    /// [`Self::eval`] of each pixel of `src`, `N` samples of `S` each,
    /// into `dst`, [`Self::channels`] samples a pixel.
    fn run<S: color::Sample>(&self, src: &[S], dst: &mut [S]) -> Result<(), CmsError> {
        let (pixels, []) = src.as_chunks::<N>() else {
            return Err(CmsError::LaneSizeMismatch);
        };
        if dst.len() != pixels.len() * self.channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        let mut value = [0u16; 4];
        let value = &mut value[..self.channels];
        for (px, out) in pixels.iter().zip(dst.chunks_exact_mut(self.channels)) {
            self.eval(px.map(<u16 as color::Sample>::from_sample), value);
            for (o, &v) in out.iter_mut().zip(&*value) {
                *o = S::from_sample(v);
            }
        }
        Ok(())
    }
}

impl<const N: usize> std::fmt::Debug for GridTable<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridTable")
            .field("inputs", &N)
            .field("span", &self.span)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl<const N: usize> TransformExecutor<u16> for GridTable<N> {
    fn transform(&self, src: &[u16], dst: &mut [u16]) -> Result<(), CmsError> {
        self.run(src, dst)
    }
}

impl<const N: usize> TransformExecutor<u8> for GridTable<N> {
    fn transform(&self, src: &[u8], dst: &mut [u8]) -> Result<(), CmsError> {
        self.run(src, dst)
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
        let table = GridTable::<4>::tabulate(SPAN, 3, |i, o| {
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

    /// In three inputs a value of equal components takes the nodes of
    /// equal components alone, whatever the others hold.
    #[test]
    fn equal_components_interpolate_along_the_diagonal() {
        const SPAN: u32 = 4;
        let table = GridTable::<3>::tabulate(SPAN, 3, |rgb, out| {
            for (px, o) in rgb
                .as_chunks::<3>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<3>().0)
            {
                let gray = px[0].to_bits() == px[1].to_bits() && px[1].to_bits() == px[2].to_bits();
                *o = if gray { [px[0]; 3] } else { [1.0, 0.0, 1.0] };
            }
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        for v in [0u16, 1000, 12_345, 40_000, 65_535] {
            let out: [u16; 3] = table.convert([v; 3]);
            assert!(out.iter().all(|o| o.abs_diff(v) <= 1), "{v}: {out:?}");
        }
        let mut dst = [0u8; 6];
        table
            .transform(&[0u8, 0, 0, 255, 255, 255], &mut dst)
            .unwrap();
        assert_eq!(dst, [0, 0, 0, 255, 255, 255]);
        assert!(table.transform(&[0u8; 4], &mut [0u8; 3]).is_err());
    }
}
