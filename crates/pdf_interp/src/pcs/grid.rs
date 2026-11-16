//! A conversion of `N` device channels tabulated over the device cube and
//! interpolated in its simplices.

use moxcms::{CmsError, TransformExecutor};

/// A conversion of `N` device channels to at most four channels,
/// tabulated at the `(span + 1)^N` nodes of a grid over the device cube:
/// the node of indices `(i₀, …, i_{N−1})`, `i₀` slowest, holds the 16-bit
/// value of the device value `i / span`, padded with zeros to four
/// channels.
pub(crate) struct GridTable<const N: usize> {
    span: u32,
    channels: usize,
    /// The node stride of each input, zero beyond the `N`th.
    strides: [usize; 8],
    nodes: Box<[[u16; 4]]>,
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
    /// For no input or more than eight, no channel or more than four, or a
    /// span outside `[1, 65 536]`, beyond which a scaled 16-bit device
    /// value overflows a `u32`.
    pub(crate) fn tabulate<E>(
        span: u32,
        channels: usize,
        mut convert: impl FnMut(&[f64], &mut [f64]) -> Result<(), E>,
    ) -> Result<Self, E> {
        assert!((1..=8).contains(&N), "one to eight inputs");
        assert!((1..=4).contains(&channels), "one to four channels");
        assert!((1..=65_536).contains(&span), "one to 65 536 cells");
        let grid = span as usize + 1;
        let mut strides = [0usize; 8];
        strides[N - 1] = 1;
        for i in (0..N - 1).rev() {
            strides[i] = strides[i + 1] * grid;
        }
        // One call per node of the slowest axis: a slice of the grid.
        let slice = strides[0];
        let full = f64::from(span);
        let mut input = Vec::with_capacity(slice * N);
        let mut out = vec![0.0; slice * channels];
        let mut nodes = Vec::with_capacity(slice * grid);
        for first in 0..grid {
            input.clear();
            for rest in 0..slice {
                let index = first * slice + rest;
                input.extend(strides[..N].iter().map(|&stride| {
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
            nodes.extend(out.chunks_exact(channels).map(|v| {
                let mut node = [0u16; 4];
                for (n, &v) in node.iter_mut().zip(v) {
                    *n = (v.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16;
                }
                node
            }));
        }
        Ok(Self {
            span,
            channels,
            strides,
            nodes: nodes.into_boxed_slice(),
        })
    }

    /// The `M` values of the device value `px` at the depth of `S`,
    /// converted at 16 bits ([`Self::value`]).
    ///
    /// # Panics
    ///
    /// For a table of other than `M` channels.
    pub(crate) fn convert<S: color::Sample, const M: usize>(&self, px: [S; N]) -> [S; M] {
        assert_eq!(self.channels, M, "a table of {M} channels");
        self.value::<M>(px.map(<u16 as color::Sample>::from_sample))
            .map(S::from_sample)
    }

    /// The number of values of a node.
    pub(crate) const fn channels(&self) -> usize {
        self.channels
    }

    /// The `M` values of the 16-bit device value `px`, `M` being
    /// [`Self::channels`]: the interpolation in the simplex of the grid
    /// cell around it whose vertices are visited in decreasing order of the
    /// value's fractional positions within the cell, each vertex weighted
    /// by the difference of successive positions.  The simplex containing
    /// the cell's main diagonal is shared by every point of it, so a value
    /// of equal components interpolates between the nodes of equal
    /// components alone.  Vertices of equal positions are visited in either
    /// order, the vertex between them taking weight 0.  A position is
    /// at most 65 535, so it fits a `u32` above three bits of index.
    ///
    /// The weights sum to 65 535, so a node sum is at most 65 535² plus
    /// the rounding half, below 2³².
    #[inline]
    fn value<const M: usize>(&self, px: [u16; N]) -> [u16; M] {
        let mut base = 0usize;
        let mut key = [0u32; N];
        for (i, (&v, k)) in px.iter().zip(&mut key).enumerate() {
            let (offset, input) = self.key(i, locate(self.span, v));
            base += offset;
            *k = input;
        }
        sort_decreasing(&mut key);
        self.interpolate(base, key)
    }

    /// The offset along input `i` of the cell `(cell, frac)` ([`locate`])
    /// and the input's sort key: its position within the cell above the
    /// three bits of `i`.  Sorted by key, the inputs are in decreasing
    /// order of position, those of equal positions by decreasing index.
    #[inline]
    const fn key(&self, i: usize, (cell, frac): (u32, u32)) -> (usize, u32) {
        #[expect(clippy::cast_possible_truncation, reason = "i is below N ≤ 8")]
        let index = i as u32;
        (cell as usize * self.strides[i & 7], frac << 3 | index)
    }

    /// The interpolation from node `base` along the inputs of `key`,
    /// sorted in decreasing order ([`Self::key`]).
    #[inline]
    fn interpolate<const M: usize>(&self, base: usize, key: [u32; N]) -> [u16; M] {
        const FULL: u32 = u16::MAX as u32;
        let nodes = &*self.nodes;
        let mut acc = [0u32; 4];
        let mut vertex = base;
        let mut previous = FULL;
        for &k in &key {
            let f = k >> 3;
            let weight = previous - f;
            for (a, &v) in acc.iter_mut().zip(&nodes[vertex]) {
                *a += weight * u32::from(v);
            }
            previous = f;
            vertex += self.strides[(k & 7) as usize];
        }
        for (a, &v) in acc.iter_mut().zip(&nodes[vertex]) {
            *a += previous * u32::from(v);
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a convex combination of u16 nodes, divided by its weight sum, fits u16"
        )]
        std::array::from_fn(|c| ((acc[c] + FULL / 2) / FULL) as u16)
    }

    /// [`Self::value`] of each pixel of `src`, `N` samples of `S` each,
    /// into `dst`, `M` samples a pixel, a block of [`BLOCK`] pixels at a
    /// time: the keys of the block's inputs held input by input and sorted
    /// across the block, each pixel interpolated, a pixel equal to the one
    /// before it taking its value, then the block's values converted to
    /// `S`.
    fn run_into<S: color::Sample, const M: usize>(&self, pixels: &[[S; N]], dst: &mut [S]) {
        let mut base = [0usize; BLOCK];
        let mut key = [[0u32; BLOCK]; N];
        let mut value = [[0u16; M]; BLOCK];
        let mut last: Option<([S; N], [u16; M])> = None;
        for (block, out) in pixels.chunks(BLOCK).zip(dst.chunks_mut(BLOCK * M)) {
            for (p, px) in block.iter().enumerate() {
                let mut b = 0;
                for (i, &v) in px.iter().enumerate() {
                    let cell = locate(self.span, <u16 as color::Sample>::from_sample(v));
                    let (offset, k) = self.key(i, cell);
                    b += offset;
                    key[i][p] = k;
                }
                base[p] = b;
            }
            sort_block_decreasing(&mut key);
            for (p, (px, v)) in block.iter().zip(&mut value).enumerate() {
                *v = match last {
                    Some((seen, value)) if seen == *px => value,
                    _ => {
                        let value =
                            self.interpolate::<M>(base[p], std::array::from_fn(|i| key[i][p]));
                        last = Some((*px, value));
                        value
                    }
                };
            }
            for (o, v) in out.as_chunks_mut::<M>().0.iter_mut().zip(&value) {
                *o = v.map(S::from_sample);
            }
        }
    }

    /// [`Self::value`] of each pixel of `src`, `N` samples of `S` each,
    /// into `dst`, [`Self::channels`] samples a pixel.
    fn run<S: color::Sample>(&self, src: &[S], dst: &mut [S]) -> Result<(), CmsError> {
        let (pixels, []) = src.as_chunks::<N>() else {
            return Err(CmsError::LaneSizeMismatch);
        };
        if dst.len() != pixels.len() * self.channels {
            return Err(CmsError::LaneSizeMismatch);
        }
        match self.channels {
            1 => self.run_into::<S, 1>(pixels, dst),
            2 => self.run_into::<S, 2>(pixels, dst),
            3 => self.run_into::<S, 3>(pixels, dst),
            _ => self.run_into::<S, 4>(pixels, dst),
        }
        Ok(())
    }
}

/// The index of the cell of a grid of `span` cells holding the 16-bit
/// device value `v` along one input, and the position of `v` within it in
/// `[0, 65 535]`.
#[inline]
const fn locate(span: u32, v: u16) -> (u32, u32) {
    const FULL: u32 = u16::MAX as u32;
    let scaled = v as u32 * span;
    let cell = scaled / FULL;
    let cell = if cell < span - 1 { cell } else { span - 1 };
    (cell, scaled - cell * FULL)
}

/// The number of pixels [`GridTable::run_into`] converts at a time.
const BLOCK: usize = 64;

/// The compare-exchanges of the optimal sorting network of `n` values for
/// `n` of 3 or 4, of the exchange sort otherwise.
fn network(n: usize) -> impl Iterator<Item = (usize, usize)> {
    let fixed: &[(usize, usize)] = match n {
        4 => &[(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)],
        3 => &[(0, 1), (1, 2), (0, 1)],
        _ => &[],
    };
    let exchange = (!matches!(n, 3 | 4))
        .then(|| (0..n).flat_map(move |i| (0..n - 1 - i).map(|j| (j, j + 1))))
        .into_iter()
        .flatten();
    fixed.iter().copied().chain(exchange)
}

/// Each column of `key` sorted in decreasing order by the
/// compare-exchanges of [`network`], free of branches.
#[inline]
fn sort_block_decreasing<const N: usize>(key: &mut [[u32; BLOCK]; N]) {
    for (i, j) in network(N) {
        for p in 0..BLOCK {
            let (a, b) = (key[i][p], key[j][p]);
            key[i][p] = a.max(b);
            key[j][p] = a.min(b);
        }
    }
}

/// `key` sorted in decreasing order by the compare-exchanges of
/// [`network`], free of branches.
#[inline]
fn sort_decreasing<const N: usize>(key: &mut [u32; N]) {
    for (i, j) in network(N) {
        let (a, b) = (key[i], key[j]);
        key[i] = a.max(b);
        key[j] = a.min(b);
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
            let out: [u16; 3] = table.value([c, m, y, k].map(node));
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
        let mid: [u16; 3] = table.value([32_768, 32_768, 0, 65_535]);
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

    /// A run of pixels converts, block by block, to each pixel's own
    /// value at either depth, repeated pixels and the last partial block
    /// included.
    #[test]
    fn a_run_converts_each_pixel_to_its_value() {
        let table = GridTable::<4>::tabulate(16, 3, |i, o| {
            convert(i, o);
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut src: Vec<u16> = (0..4 * 150).map(|_| (next() & 0xffff) as u16).collect();
        src.copy_within(0..4, 4);
        src.copy_within(4 * 63..4 * 64, 4 * 64);
        let mut dst = vec![0u16; 3 * 150];
        TransformExecutor::<u16>::transform(&table, &src, &mut dst).unwrap();
        let bytes: Vec<u8> = src.iter().map(|&v| (v >> 8) as u8).collect();
        let mut dst8 = vec![0u8; 3 * 150];
        TransformExecutor::<u8>::transform(&table, &bytes, &mut dst8).unwrap();
        for p in 0..150 {
            let px: [u16; 4] = src[4 * p..4 * p + 4].try_into().unwrap();
            let want: [u16; 3] = table.value(px);
            assert_eq!(dst[3 * p..3 * p + 3], want, "pixel {p}");
            let px8: [u8; 4] = bytes[4 * p..4 * p + 4].try_into().unwrap();
            let want8: [u8; 3] = table.convert(px8);
            assert_eq!(dst8[3 * p..3 * p + 3], want8, "byte pixel {p}");
        }
    }
}
