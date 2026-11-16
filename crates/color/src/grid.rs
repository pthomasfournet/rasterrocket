//! Interpolation in the simplices of a grid of 16-bit nodes, in integer
//! arithmetic.
//!
//! A grid of `N` inputs holds, at each node, up to four 16-bit values.  A
//! point of the grid lies in a cell, the base node of the cell, and has a
//! position in `[0, 65 535]` within the cell along each input.  The point's
//! value is the interpolation in the simplex of the cell whose vertices
//! are visited from the base node in decreasing order of the positions,
//! each vertex weighted by the difference of successive positions
//! ([`interpolate`]); every operation is on integers, so every evaluator of
//! the same nodes and positions, host or device, yields the same value.
//!
//! [`CmykGrid`] holds a grid of four 8-bit inputs and three outputs in the
//! layout the device kernels read.

use crate::Sample;

/// The largest position within a cell, and the sum of a point's weights.
const FULL: u32 = u16::MAX as u32;

/// The values a grid's nodes hold, and how an interpolated value maps to
/// an output in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRange {
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
    #[must_use]
    pub fn node(self, v: f64) -> u16 {
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
    #[must_use]
    pub fn output(self, acc: u32) -> u16 {
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

    /// The range's code in a [`CmykGrid`]: 0 for [`Self::Unit`], 1 for
    /// [`Self::Extended`].
    const fn code(self) -> u32 {
        match self {
            Self::Unit => 0,
            Self::Extended => 1,
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

/// The index of the cell of a grid of `span` cells, `span` at least 1,
/// holding the 16-bit value `v` along one input, and the position of `v`
/// within it in `[0, 65 535]`.
#[inline]
#[must_use]
pub const fn locate(span: u32, v: u16) -> (u32, u32) {
    let scaled = v as u32 * span;
    let cell = scaled / FULL;
    let cell = if cell < span - 1 { cell } else { span - 1 };
    (cell, scaled - cell * FULL)
}

/// The sort key of input `input`, below 8, at the position `frac` within
/// its cell.
///
/// The key holds the position above the three bits of the input, so keys
/// sorted in decreasing order visit the inputs in decreasing order of
/// position, those of equal positions by decreasing index.
#[inline]
#[must_use]
pub const fn sort_key(frac: u32, input: u32) -> u32 {
    frac << 3 | input
}

/// The compare-exchanges of the optimal sorting network of `n` values for
/// `n` of 3 or 4, of the exchange sort otherwise; each puts the larger of
/// its pair first.
#[inline]
pub fn network(n: usize) -> impl Iterator<Item = (usize, usize)> {
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

/// `key` sorted in decreasing order by the compare-exchanges of
/// [`network`], free of branches.
#[inline]
pub fn sort_decreasing<const N: usize>(key: &mut [u32; N]) {
    for (i, j) in network(N) {
        let (a, b) = (key[i], key[j]);
        key[i] = a.max(b);
        key[j] = a.min(b);
    }
}

/// The node sums of the interpolation from node `base` of `nodes` along
/// the inputs of `key`, sorted in decreasing order ([`sort_key`],
/// [`sort_decreasing`]).
///
/// The vertex reached from `base` by the strides (`strides[i]` the node
/// stride of input `i`) of the inputs of the first `j` keys has weight
/// `f_{j−1} − f_j`, the positions `f` of the keys between 65 535 before
/// the first and 0 after the last.  The sums of the first `M` values, at
/// most four, are returned.  The weights sum to 65 535, so a sum is at most
/// 65 535², below 2³², and maps to an output by [`NodeRange::output`].
#[inline]
#[must_use]
pub fn interpolate<const N: usize, const M: usize>(
    nodes: &[[u16; 4]],
    strides: &[usize; 8],
    base: usize,
    key: [u32; N],
) -> [u32; M] {
    // All four values are summed: four lanes vectorise where three do
    // not.
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
        vertex += strides[(k & 7) as usize];
    }
    for (a, &v) in acc.iter_mut().zip(&nodes[vertex]) {
        *a += previous * u32::from(v);
    }
    std::array::from_fn(|c| acc[c])
}

/// A grid of four 8-bit inputs and three outputs, interpolated as
/// [`interpolate`] does, held as 32-bit words:
///
/// - words 0 to 3: the node stride of each input;
/// - word 4: the [`NodeRange`], 0 for unit and 1 for extended; words 5 to
///   7 are zero;
/// - from word [`Self::CELLS`], `256 · i + v`: the cell of input `i` at
///   the sample `v`, its index above 16 bits and the position within it
///   below;
/// - from word [`Self::NODES`], two words a node, its four 16-bit values
///   in order: on a little-endian host the first value in the low half of
///   the first word, the second in its high half, the third in the low half
///   of the second word; the fourth is not an output.
///
/// Every vertex a sample reaches lies within the nodes.
#[derive(Clone, PartialEq, Eq)]
pub struct CmykGrid {
    words: Box<[u32]>,
}

impl CmykGrid {
    /// The word at which the cells begin.
    pub const CELLS: usize = 8;

    /// The word at which the nodes begin.
    pub const NODES: usize = Self::CELLS + 4 * 256;

    /// The grid of the cells `cells[i][v]` of input `i` at the sample `v`
    /// (index above 16 bits, position below), the node strides `strides`,
    /// in `range`, over `nodes`, of which the first three values are
    /// outputs.  `None` when some sample reaches a vertex beyond the nodes
    /// or the grid would exceed `i32::MAX` bytes.
    #[must_use]
    pub fn new(
        cells: &[[u32; 256]; 4],
        strides: [u32; 4],
        range: NodeRange,
        nodes: &[[u16; 4]],
    ) -> Option<Self> {
        let len = Self::NODES
            .checked_add(nodes.len().checked_mul(2)?)
            .filter(|&l| l.checked_mul(4).is_some_and(|b| i32::try_from(b).is_ok()))?;
        // The farthest vertex of a sample is its base node plus every
        // input's stride.
        let last: u64 = cells
            .iter()
            .zip(strides)
            .map(|(axis, stride)| {
                let cell = axis.iter().map(|c| c >> 16).max().unwrap_or(0);
                (u64::from(cell) + 1) * u64::from(stride)
            })
            .sum();
        if usize::try_from(last).ok()? >= nodes.len() {
            return None;
        }
        let mut words = Vec::with_capacity(len);
        words.extend(strides);
        words.extend([range.code(), 0, 0, 0]);
        words.extend_from_slice(cells.as_flattened());
        words.extend_from_slice(bytemuck::cast_slice(nodes));
        Some(Self {
            words: words.into_boxed_slice(),
        })
    }

    /// The grid's words in the host's byte order.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    /// The number of nodes of a grid of `len` bytes, or `None` for a
    /// length no grid has: below one node, not a whole number of nodes
    /// after the cells, or beyond `i32::MAX`.
    #[must_use]
    pub const fn node_count(len: usize) -> Option<u32> {
        let head = Self::NODES * 4;
        if len < head + 8 || !(len - head).is_multiple_of(8) || len > i32::MAX as usize {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "len ≤ i32::MAX, so the count is below 2^28"
        )]
        Some(((len - head) / 8) as u32)
    }

    /// The RGB of each whole CMYK pixel of `cmyk`, three bytes a pixel:
    /// the 8-bit value nearest each output of [`Self::convert16`].
    #[must_use]
    pub fn convert(&self, cmyk: &[u8]) -> Vec<u8> {
        self.outputs(cmyk)
            .flat_map(|rgb| rgb.map(Sample::to_u8))
            .collect()
    }

    /// The RGB of each whole CMYK pixel of `cmyk`, three 16-bit values a
    /// pixel: the interpolation of the pixel's cells, each output its
    /// [`NodeRange::output`].
    #[must_use]
    pub fn convert16(&self, cmyk: &[u8]) -> Vec<u16> {
        self.outputs(cmyk).flatten().collect()
    }

    /// The 16-bit outputs of each whole CMYK pixel of `cmyk`.
    fn outputs<'a>(&'a self, cmyk: &'a [u8]) -> impl Iterator<Item = [u16; 3]> + 'a {
        let range = if self.words[4] == 1 {
            NodeRange::Extended
        } else {
            NodeRange::Unit
        };
        let mut strides = [0usize; 8];
        for (s, &w) in strides.iter_mut().zip(&self.words[..4]) {
            *s = w as usize;
        }
        let cells = &self.words[Self::CELLS..Self::NODES];
        let nodes: &[[u16; 4]] = bytemuck::cast_slice(&self.words[Self::NODES..]);
        cmyk.as_chunks::<4>().0.iter().map(move |px| {
            let mut base = 0;
            let mut key = [0u32; 4];
            for (i, (&v, k)) in px.iter().zip(&mut key).enumerate() {
                let cell = cells[256 * i + usize::from(v)];
                base += (cell >> 16) as usize * strides[i];
                #[expect(clippy::cast_possible_truncation, reason = "i is below 4")]
                let input = i as u32;
                *k = sort_key(cell & 0xffff, input);
            }
            sort_decreasing(&mut key);
            interpolate::<4, 3>(nodes, &strides, base, key).map(|acc| range.output(acc))
        })
    }
}

impl std::fmt::Debug for CmykGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmykGrid")
            .field("words", &self.words.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A grid of one cell per input whose nodes hold `(c + m + y + k) / 4`
    /// and its complements, every sample in the cell.
    fn mean_grid(range: NodeRange) -> CmykGrid {
        let mut cells = [[0u32; 256]; 4];
        for axis in &mut cells {
            for (v, c) in (0u16..).zip(axis.iter_mut()) {
                let (cell, frac) = locate(1, v * 257);
                *c = cell << 16 | frac;
            }
        }
        let nodes: Vec<[u16; 4]> = (0..16u32)
            .map(|i| {
                let mean = f64::from(i.count_ones()) / 4.0;
                let [a, b, c] = [mean, 1.0 - mean, 0.5].map(|v| range.node(v));
                [a, b, c, 0]
            })
            .collect();
        CmykGrid::new(&cells, [8, 4, 2, 1], range, &nodes).expect("a valid grid")
    }

    /// A grid linear in its inputs reproduces the linear function at every
    /// sample, within a level, in either range.
    #[test]
    fn a_linear_grid_interpolates_exactly() {
        for range in [NodeRange::Unit, NodeRange::Extended] {
            let grid = mean_grid(range);
            for px in [
                [0u8; 4],
                [255; 4],
                [10, 200, 30, 99],
                [255, 0, 0, 0],
                [7, 7, 7, 7],
            ] {
                let rgb = grid.convert(&px);
                let mean = px.iter().map(|&v| f64::from(v)).sum::<f64>() / 4.0;
                assert!(
                    (f64::from(rgb[0]) - mean).abs() <= 1.0,
                    "{range:?} {px:?}: {rgb:?}"
                );
                assert!((f64::from(rgb[1]) - (255.0 - mean)).abs() <= 1.0, "{rgb:?}");
                assert_eq!(rgb[2], 128, "{rgb:?}");
            }
        }
    }

    /// The 8-bit conversion is the nearest byte of the 16-bit one.
    #[test]
    fn the_8_bit_conversion_narrows_the_16_bit_one() {
        for range in [NodeRange::Unit, NodeRange::Extended] {
            let grid = mean_grid(range);
            let cmyk: Vec<u8> = (0..=255u8)
                .flat_map(|v| [v, 255 - v, v / 3, v / 2])
                .collect();
            let wide = grid.convert16(&cmyk);
            assert_eq!(wide.len(), 256 * 3);
            let narrowed: Vec<u8> = wide.iter().map(|&v| v.to_u8()).collect();
            assert_eq!(grid.convert(&cmyk), narrowed, "{range:?}");
            assert!(
                wide.iter().any(|&v| v % 257 != 0),
                "{range:?}: no 16-bit level"
            );
        }
    }

    /// A grid whose cells reach beyond its nodes is refused, and the byte
    /// length of a grid gives back its node count.
    #[test]
    fn a_grid_reaching_beyond_its_nodes_is_refused() {
        let cells = [[0u32; 256]; 4];
        let nodes = [[0u16; 4]; 16];
        assert!(CmykGrid::new(&cells, [8, 4, 2, 1], NodeRange::Unit, &nodes).is_some());
        assert!(CmykGrid::new(&cells, [8, 4, 2, 2], NodeRange::Unit, &nodes).is_none());
        let mut far = cells;
        far[0][255] = 1 << 16;
        assert!(CmykGrid::new(&far, [8, 4, 2, 1], NodeRange::Unit, &nodes).is_none());
        let grid = CmykGrid::new(&cells, [8, 4, 2, 1], NodeRange::Unit, &nodes).unwrap();
        assert_eq!(CmykGrid::node_count(grid.as_bytes().len()), Some(16));
        assert_eq!(CmykGrid::node_count(grid.as_bytes().len() - 4), None);
        assert_eq!(CmykGrid::node_count(CmykGrid::NODES * 4), None);
    }

    /// A sample at a cell's corner takes the corner node alone.
    #[test]
    fn locate_puts_the_top_value_at_the_last_cell_s_end() {
        assert_eq!(locate(16, 0), (0, 0));
        assert_eq!(locate(16, u16::MAX), (15, 65_535));
        assert_eq!(locate(1, 32_768), (0, 32_768));
    }

    /// The sorting network orders every arrangement of distinct and equal
    /// keys.
    #[test]
    fn the_networks_sort_decreasing() {
        for n in 0..4u32.pow(4) {
            let mut key: [u32; 4] = std::array::from_fn(|i| n >> (2 * i) & 3);
            let mut want = key;
            want.sort_unstable_by(|a, b| b.cmp(a));
            sort_decreasing(&mut key);
            assert_eq!(key, want);
            let mut three = [key[3], key[1], key[2]];
            let mut want3 = three;
            want3.sort_unstable_by(|a, b| b.cmp(a));
            sort_decreasing(&mut three);
            assert_eq!(three, want3);
        }
    }
}
