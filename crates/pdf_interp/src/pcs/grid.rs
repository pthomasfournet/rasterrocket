//! A conversion of `N` device channels tabulated over the device cube and
//! interpolated in its simplices.

use moxcms::{CmsError, TransformExecutor};

/// The values a grid's nodes hold, and how an interpolated value maps to
/// an output in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeRange {
    /// `[0, 1]`, the value `v` held as the 16-bit `v · 65 535`; an
    /// interpolated value is the output.
    Unit,
    /// `[−1/2, 3/2]`, the value `v` held as the 16-bit `(v + 1/2) / 2 ·
    /// 65 535`; an interpolated value is clamped to `[0, 1]`, so an output
    /// that leaves `[0, 1]` within a cell is clamped at the point and not
    /// at the cell's nodes.
    Extended,
}

impl NodeRange {
    /// The 16-bit node of the value `v`, clamped to the range.
    fn node(self, v: f64) -> u16 {
        match self {
            Self::Unit => to_u16(v),
            Self::Extended => to_u16(f64::midpoint(v, 0.5)),
        }
    }

    /// The 16-bit output of the sum `acc` of nodes weighted by weights
    /// summing to 65 535, `F`, rounded to nearest: for [`Self::Extended`],
    /// `(4 · acc − F² + F) / 2F` clamped to `[0, F]`, computed as
    /// `(2 (acc′ − L) + 1) / F` for `acc′` the sum clamped to
    /// `[L, L + (F² − 1) / 2]`, `L = (F (F − 1) / 2 + 1) / 2`: every sum
    /// up to `L` has output 0, every sum from the upper bound output `F`,
    /// and `2 (acc′ − L) + 1` lies in `[1, F²]`.
    #[inline]
    fn output(self, acc: u32) -> u16 {
        const FULL: u32 = u16::MAX as u32;
        const LOW: u32 = (FULL * (FULL - 1) / 2).div_ceil(2);
        const HIGH: u32 = LOW + (FULL * FULL - 1) / 2;
        let v = match self {
            Self::Unit => (acc + FULL / 2) / FULL,
            Self::Extended => (2 * (acc.clamp(LOW, HIGH) - LOW) + 1) / FULL,
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a convex combination of u16 nodes divided by its weight sum, or clamped to [0, 65 535]"
        )]
        let v = v as u16;
        v
    }
}

/// A conversion of `N` device channels to at most four channels,
/// tabulated at the nodes of a grid of `spans[i]` cells along input `i`,
/// over the device cube or over the image of it under a curve per input
/// (the shaper): the node of indices `(i₀, …, i_{N−1})`, `i₀` slowest,
/// holds the 16-bit value in `range` of the device value whose shaped
/// value along input `k` is `i_k / spans[k]`, padded with zeros to four
/// channels.
pub(crate) struct GridTable<const N: usize> {
    spans: [u32; N],
    channels: usize,
    range: NodeRange,
    /// The node stride of each input, zero beyond the `N`th.
    strides: [usize; 8],
    /// The curve of each input, if any.
    shaper: Option<Shaper<N>>,
    nodes: Box<[[u16; 4]]>,
}

impl<const N: usize> GridTable<N> {
    /// The conversion `convert` tabulated on a grid of `span` cells along
    /// each device axis into `channels` values of `[0, 1]`
    /// ([`Self::tabulate_shaped`] with no shaper, [`NodeRange::Unit`]).
    ///
    /// # Errors
    ///
    /// The first error `convert` returns.
    ///
    /// # Panics
    ///
    /// As [`Self::tabulate_shaped`].
    pub(crate) fn tabulate<E>(
        span: u32,
        channels: usize,
        convert: impl FnMut(&[f64], &mut [f64]) -> Result<(), E>,
    ) -> Result<Self, E> {
        Self::tabulate_shaped([span; N], None, NodeRange::Unit, channels, convert)
    }

    /// The conversion `convert` tabulated on a grid of `spans[i]` cells
    /// along input `i` into `channels` values.  Without a shaper the grid
    /// lies over the device cube and `convert` takes device values; with
    /// one, `shaper[i]` holds the curve of input `i` at evenly spaced
    /// device values spanning `[0, 1]`, each in `[0, 1]`, the grid lies over
    /// the curves' outputs and `convert` takes those.  `convert` takes the
    /// values of a run of nodes, `N` values of `[0, 1]` each, and writes
    /// their `channels` values, each clamped to `range`.
    ///
    /// # Errors
    ///
    /// The first error `convert` returns.
    ///
    /// # Panics
    ///
    /// For no input or more than eight, no channel or more than four, a
    /// span outside `[1, 65 536]`, beyond which a scaled 16-bit value
    /// overflows a `u32`, or a curve of fewer than two values or more than
    /// 65 537.
    pub(crate) fn tabulate_shaped<E>(
        spans: [u32; N],
        shaper: Option<[&[f64]; N]>,
        range: NodeRange,
        channels: usize,
        mut convert: impl FnMut(&[f64], &mut [f64]) -> Result<(), E>,
    ) -> Result<Self, E> {
        assert!((1..=8).contains(&N), "one to eight inputs");
        assert!((1..=4).contains(&channels), "one to four channels");
        assert!(
            spans.iter().all(|s| (1..=65_536).contains(s)),
            "one to 65 536 cells"
        );
        let shaper = shaper.map(|curves| Shaper::new(curves, spans));
        let grids = spans.map(|s| s as usize + 1);
        let mut strides = [0usize; 8];
        strides[N - 1] = 1;
        for i in (0..N - 1).rev() {
            strides[i] = strides[i + 1] * grids[i + 1];
        }
        // One call per node of the slowest axis: a slice of the grid.
        let slice = strides[0];
        let mut input = Vec::with_capacity(slice * N);
        let mut out = vec![0.0; slice * channels];
        let mut nodes = Vec::with_capacity(slice * grids[0]);
        for first in 0..grids[0] {
            input.clear();
            for rest in 0..slice {
                let index = first * slice + rest;
                input.extend((0..N).map(|i| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "a node index is below span + 1 ≤ 2^32"
                    )]
                    let node = (index / strides[i] % grids[i]) as f64;
                    node / f64::from(spans[i])
                }));
            }
            convert(&input, &mut out)?;
            nodes.extend(out.chunks_exact(channels).map(|v| {
                let mut node = [0u16; 4];
                for (n, &v) in node.iter_mut().zip(v) {
                    *n = range.node(v);
                }
                node
            }));
        }
        Ok(Self {
            spans,
            channels,
            range,
            strides,
            shaper,
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

    /// Whether the grid lies over the outputs of a shaper.
    #[cfg(test)]
    pub(crate) const fn is_shaped(&self) -> bool {
        self.shaper.is_some()
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
    /// The weights sum to 65 535, so a node sum is at most 65 535², below
    /// 2³², mapped to the output by [`NodeRange::output`].
    #[inline]
    fn value<const M: usize>(&self, px: [u16; N]) -> [u16; M] {
        let mut base = 0usize;
        let mut key = [0u32; N];
        for (i, (&v, k)) in px.iter().zip(&mut key).enumerate() {
            let cell = self
                .shaper
                .as_ref()
                .map_or_else(|| locate(self.spans[i], v), |shaper| shaper.cell(i, v));
            let (offset, input) = self.key(i, cell);
            base += offset;
            *k = input;
        }
        sort_decreasing(&mut key);
        self.interpolate::<M>(base, key)
            .map(|acc| self.range.output(acc))
    }

    /// The offset along input `i` of the cell `(cell, frac)` of its axis
    /// ([`locate`], [`Shaper::cell`]) and the input's sort key: its
    /// position within the cell above the three bits of `i`.  Sorted by
    /// key, the inputs are in decreasing order of position, those of equal
    /// positions by decreasing index.
    #[inline]
    const fn key(&self, i: usize, (cell, frac): (u32, u32)) -> (usize, u32) {
        #[expect(clippy::cast_possible_truncation, reason = "i is below N ≤ 8")]
        let index = i as u32;
        (cell as usize * self.strides[i & 7], frac << 3 | index)
    }

    /// The node sums of the interpolation from node `base` along the
    /// inputs of `key`, sorted in decreasing order ([`Self::key`]), each
    /// the output's [`NodeRange::output`].
    #[inline]
    fn interpolate<const M: usize>(&self, base: usize, key: [u32; N]) -> [u32; M] {
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
        std::array::from_fn(|c| acc[c])
    }

    /// The base node of each pixel of `block` into `base` and its keys
    /// into `key`, input by input, `key_of` giving the offset and key
    /// ([`Self::key`]) of the sample of input `i`.
    #[inline]
    fn block_keys<S: Copy>(
        block: &[[S; N]],
        base: &mut [usize; BLOCK],
        key: &mut [[u32; BLOCK]; N],
        key_of: impl Fn(usize, S) -> (usize, u32),
    ) {
        for (p, px) in block.iter().enumerate() {
            let mut b = 0;
            for (i, &v) in px.iter().enumerate() {
                let (offset, k) = key_of(i, v);
                b += offset;
                key[i][p] = k;
            }
            base[p] = b;
        }
    }

    /// [`Self::value`] of each pixel of `src`, `N` samples of `S` each,
    /// into `dst`, `M` samples a pixel ([`Self::run_blocks`]), the values of
    /// a shaped grid's block shaped first.
    fn run_into<S: color::Sample, const M: usize>(&self, pixels: &[[S; N]], dst: &mut [S]) {
        match &self.shaper {
            Some(shaper) => {
                let mut cells = [[0u32; N]; BLOCK];
                self.run_blocks::<S, M>(pixels, dst, |block, base, key| {
                    for (c, px) in cells.iter_mut().zip(block) {
                        *c = std::array::from_fn(|i| shaper.packed_sample(i, px[i]));
                    }
                    let cells = &cells[..block.len()];
                    Self::block_keys(cells, base, key, |i, c| self.key(i, (c >> 16, c & 0xffff)));
                });
            }
            None => self.run_blocks::<S, M>(pixels, dst, |block, base, key| {
                Self::block_keys(block, base, key, |i, v| {
                    let v = <u16 as color::Sample>::from_sample(v);
                    self.key(i, locate(self.spans[i], v))
                });
            }),
        }
    }

    /// [`Self::value`] of each pixel of `src`, `N` samples of `S` each,
    /// into `dst`, `M` samples a pixel, a block of [`BLOCK`] pixels at a
    /// time: the block's base nodes and keys, input by input, from
    /// `keys` ([`Self::block_keys`]), the keys sorted across the block,
    /// each pixel's node sums interpolated, a pixel equal to the one before
    /// it taking its sums, then the block's sums mapped to outputs
    /// ([`NodeRange::output`]) and converted to `S`.
    #[inline]
    fn run_blocks<S: color::Sample, const M: usize>(
        &self,
        pixels: &[[S; N]],
        dst: &mut [S],
        keys: impl FnMut(&[[S; N]], &mut [usize; BLOCK], &mut [[u32; BLOCK]; N]),
    ) {
        match self.range {
            NodeRange::Unit => self.run_blocks_in::<S, M, false>(pixels, dst, keys),
            NodeRange::Extended => self.run_blocks_in::<S, M, true>(pixels, dst, keys),
        }
    }

    /// [`Self::run_blocks`] in the node range [`NodeRange::Extended`] or
    /// not, as `EXTENDED`.
    #[inline]
    fn run_blocks_in<S: color::Sample, const M: usize, const EXTENDED: bool>(
        &self,
        pixels: &[[S; N]],
        dst: &mut [S],
        mut keys: impl FnMut(&[[S; N]], &mut [usize; BLOCK], &mut [[u32; BLOCK]; N]),
    ) {
        let mut base = [0usize; BLOCK];
        let mut key = [[0u32; BLOCK]; N];
        let range = if EXTENDED {
            NodeRange::Extended
        } else {
            NodeRange::Unit
        };
        let mut sums = [[0u32; M]; BLOCK];
        let mut last: Option<([S; N], [u32; M])> = None;
        for (block, out) in pixels.chunks(BLOCK).zip(dst.chunks_mut(BLOCK * M)) {
            keys(block, &mut base, &mut key);
            sort_block_decreasing(&mut key);
            for (p, (px, v)) in block.iter().zip(&mut sums).enumerate() {
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
            for (o, v) in out.as_chunks_mut::<M>().0.iter_mut().zip(&sums) {
                *o = v.map(|acc| S::from_sample(range.output(acc)));
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
/// value `v` along one input, and the position of `v` within it in
/// `[0, 65 535]`.
#[inline]
const fn locate(span: u32, v: u16) -> (u32, u32) {
    const FULL: u32 = u16::MAX as u32;
    let scaled = v as u32 * span;
    let cell = scaled / FULL;
    let cell = if cell < span - 1 { cell } else { span - 1 };
    (cell, scaled - cell * FULL)
}

/// The curve of each input of a shaped grid, tabulated at every 16-bit
/// value as the cell of the grid its output lies in: the cell's index
/// above 16 bits, the position within it in `[0, 65 535]` below, found
/// from the curve's output in floating point, so a curve that flattens
/// keeps the resolution of its input.
struct Shaper<const N: usize> {
    /// The cell of input `i` at the 16-bit value `v` at `i · 2¹⁶ + v`.
    cells: Box<[u32]>,
    /// The cell of input `i` at the 8-bit value `v` (the 16-bit `257 · v`)
    /// at `256 · i + v`.
    bytes: Box<[u32]>,
}

impl<const N: usize> Shaper<N> {
    /// The shaper of the curves `curves`, each of values in `[0, 1]` at
    /// evenly spaced inputs spanning `[0, 1]`, interpolated linearly, onto
    /// a grid of `spans[i]` cells along input `i`.
    ///
    /// # Panics
    ///
    /// For a curve of fewer than two values or more than 65 537.
    fn new(curves: [&[f64]; N], spans: [u32; N]) -> Self {
        let mut cells = Vec::with_capacity(N << 16);
        for (curve, span) in curves.iter().zip(spans) {
            assert!(
                (2..=65_537).contains(&curve.len()),
                "two to 65 537 curve values"
            );
            #[expect(
                clippy::cast_precision_loss,
                reason = "a curve holds at most 65 537 values"
            )]
            let last = (curve.len() - 1) as f64;
            cells.extend((0..=u16::MAX).map(|v| {
                let pos = f64::from(v) / f64::from(u16::MAX) * last;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "pos lies in [0, len - 1]"
                )]
                let lo = (pos as usize).min(curve.len() - 2);
                #[expect(clippy::cast_precision_loss, reason = "lo < 65 537")]
                let f = pos - lo as f64;
                let u = f
                    .mul_add(curve[lo + 1] - curve[lo], curve[lo])
                    .clamp(0.0, 1.0);
                let pos = u * f64::from(span);
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "pos lies in [0, span] and span ≤ 65 536"
                )]
                let cell = (pos as u32).min(span - 1);
                let frac = to_u16(pos - f64::from(cell));
                cell << 16 | u32::from(frac)
            }));
        }
        let bytes = (0..N)
            .flat_map(|i| (0..256).map(move |v| (i << 16) | (v * 257)))
            .map(|at| cells[at])
            .collect();
        Self {
            cells: cells.into_boxed_slice(),
            bytes,
        }
    }

    /// The cell `(cell, frac)` of input `i` at the 16-bit value `v`.
    #[inline]
    fn cell(&self, i: usize, v: u16) -> (u32, u32) {
        let c = self.cells[i << 16 | usize::from(v)];
        (c >> 16, c & 0xffff)
    }

    /// The packed cell of input `i` at the sample `v`.
    #[inline]
    fn packed_sample<S: color::Sample>(&self, i: usize, v: S) -> u32 {
        if S::BITS == 8 {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "an 8-bit sample lies in [0, 255]"
            )]
            let v = v.signed() as usize;
            self.bytes[i << 8 | v]
        } else {
            self.cells[i << 16 | usize::from(<u16 as color::Sample>::from_sample(v))]
        }
    }
}

/// `v` clamped to `[0, 1]` and scaled to the nearest 16-bit value.
fn to_u16(v: f64) -> u16 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a value clamped to [0, 1] scaled to the u16 range"
    )]
    let v = (v.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16;
    v
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
            .field("spans", &self.spans)
            .field("shaped", &self.shaper.is_some())
            .field("range", &self.range)
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

    /// The extended range's output is the rounded `(4 · acc − F² + F) /
    /// 2F` clamped to `[0, F]` at every sum, those around both clamps and
    /// the extremes included.
    #[test]
    fn the_extended_output_is_the_clamped_rescaled_sum() {
        const F: u64 = u16::MAX as u64;
        let want = |acc: u64| ((4 * acc).saturating_sub(F * F) + F) / (2 * F);
        let low = (F * (F - 1) / 2).div_ceil(2);
        let high = low + (F * F - 1) / 2;
        let sums = (0..=F * F)
            .step_by(9_973)
            .chain(low - 70_000..low + 70_000)
            .chain(high - 70_000..high + 70_000)
            .chain(F * F - 70_000..=F * F);
        for acc in sums {
            let got = NodeRange::Extended.output(u32::try_from(acc).unwrap());
            assert_eq!(u64::from(got), want(acc).min(F), "sum {acc}");
        }
        assert_eq!(NodeRange::Extended.node(-0.5), 0);
        assert_eq!(NodeRange::Extended.node(1.5), u16::MAX);
        assert_eq!(NodeRange::Extended.node(0.0), 16_384);
    }

    /// A conversion leaving `[0, 1]` within a cell, tabulated in the
    /// extended range, is clamped where it leaves it, within two levels;
    /// in the unit range the cell interpolates between its nodes' clamped
    /// values and misses by the overshoot.
    #[test]
    fn an_extended_grid_clamps_within_its_cells() {
        // Linear in the inputs, leaving [0, 1] at x₀ = 3/16 and 13/16,
        // inside the cells [0, 1/4] and [3/4, 1].
        let ramp = |x: &[f64]| [1.6f64.mul_add(x[0], -0.3), 0.5, x[3]];
        let table = |range| {
            GridTable::<4>::tabulate_shaped([4; 4], None, range, 3, |x, o| {
                for (px, o) in x.as_chunks::<4>().0.iter().zip(o.as_chunks_mut::<3>().0) {
                    *o = ramp(px);
                }
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap()
        };
        let (extended, unit) = (table(NodeRange::Extended), table(NodeRange::Unit));
        let mut worst = 0.0f64;
        for x in (0..=u16::MAX).step_by(97) {
            let px = [x, 20_000, 40_000, 65_535 - x];
            let want = ramp(&px.map(|v| f64::from(v) / 65535.0)).map(|v| v.clamp(0.0, 1.0));
            let got: [u16; 3] = extended.value(px);
            let plain: [u16; 3] = unit.value(px);
            for ((g, p), w) in got.iter().zip(plain).zip(want) {
                assert!(
                    w.mul_add(-65535.0, f64::from(*g)).abs() <= 2.0,
                    "{px:?}: {got:?} for {want:?}"
                );
                worst = worst.max(w.mul_add(-65535.0, f64::from(p)).abs());
            }
        }
        assert!(worst > 4000.0, "the unit grid misses by {worst}");
    }

    /// The square of each input, at 257 values.
    fn squares() -> Vec<f64> {
        (0..=256).map(|i| (f64::from(i) / 256.0).powi(2)).collect()
    }

    /// A run of pixels converts, block by block, to each pixel's own
    /// value at either depth, repeated pixels and the last partial block
    /// included, through a grid over the device cube, a shaped one of
    /// uneven spans and one in the extended range alike.
    #[test]
    fn a_run_converts_each_pixel_to_its_value() {
        let square = squares();
        let unit = |i: &[f64], o: &mut [f64]| {
            convert(i, o);
            Ok::<(), std::convert::Infallible>(())
        };
        let plain = GridTable::<4>::tabulate(16, 3, unit).unwrap();
        let shaped = GridTable::<4>::tabulate_shaped(
            [20, 15, 16, 7],
            Some([&*square; 4]),
            NodeRange::Unit,
            3,
            unit,
        )
        .unwrap();
        let overshoot = |i: &[f64], o: &mut [f64]| {
            convert(i, o);
            for v in o.iter_mut() {
                *v = 1.8f64.mul_add(*v, -0.4);
            }
            Ok::<(), std::convert::Infallible>(())
        };
        let extended =
            GridTable::<4>::tabulate_shaped([16; 4], None, NodeRange::Extended, 3, overshoot)
                .unwrap();
        for table in [plain, shaped, extended] {
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
                assert_eq!(dst[3 * p..3 * p + 3], want, "{table:?} pixel {p}");
                let px8: [u8; 4] = bytes[4 * p..4 * p + 4].try_into().unwrap();
                let want8: [u8; 3] = table.convert(px8);
                assert_eq!(dst8[3 * p..3 * p + 3], want8, "{table:?} byte pixel {p}");
            }
        }
    }

    /// A grid over the outputs of its shaper interpolates a conversion
    /// linear in those outputs within two levels wherever the inputs lie,
    /// along uneven spans, where a grid over the device cube misses by
    /// the curves' bend within its cells.
    #[test]
    fn a_shaped_grid_interpolates_over_the_curves() {
        let square = squares();
        // Linear in each input: reproduced by any simplex of any grid.
        let linear = |u: &[f64], o: &mut [f64]| {
            for (px, o) in u.as_chunks::<4>().0.iter().zip(o.as_chunks_mut::<3>().0) {
                *o = [px[0], f64::midpoint(px[1], px[2]), px[3]];
            }
        };
        let shaped = GridTable::<4>::tabulate_shaped(
            [3, 5, 7, 2],
            Some([&*square; 4]),
            NodeRange::Unit,
            3,
            |u, o| {
                linear(u, o);
                Ok::<(), std::convert::Infallible>(())
            },
        )
        .unwrap();
        let plain = GridTable::<4>::tabulate(4, 3, |x, o| {
            let u: Vec<f64> = x.iter().map(|v| v * v).collect();
            linear(&u, o);
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        let mut worst = 0.0f64;
        for px in [
            [1000u16, 20_000, 33_333, 65_535],
            [65_000, 0, 7, 40_000],
            [12_345; 4],
        ] {
            let x = px.map(|v| f64::from(v) / 65535.0);
            let mut want = [0.0; 3];
            linear(&x.map(|v| v * v), &mut want);
            let got: [u16; 3] = shaped.value(px);
            let plain: [u16; 3] = plain.value(px);
            for ((g, p), w) in got.iter().zip(plain).zip(want) {
                assert!(
                    w.mul_add(-65535.0, f64::from(*g)).abs() <= 2.0,
                    "{px:?}: {got:?} for {want:?}"
                );
                worst = worst.max(w.mul_add(-65535.0, f64::from(p)).abs());
            }
        }
        assert!(worst > 500.0, "the device grid misses by {worst}");
    }
}
