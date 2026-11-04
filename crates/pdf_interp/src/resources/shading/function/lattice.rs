//! Multilinear interpolation over a regular grid: the `2^m` corners of the
//! grid cell holding a point, each with its weight.

/// One axis of the cell holding a point: the grid indices below and above
/// the point, the weight of the upper one, and the axis stride in grid
/// nodes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Axis {
    pub lo: usize,
    pub hi: usize,
    pub frac: f64,
    pub stride: usize,
}

impl Axis {
    /// The axis of a grid of `size` nodes, `stride` apart, at position
    /// `pos`, which is clamped to `[0, size − 1]`.  `size` is at least one.
    pub(crate) fn at(pos: f64, size: usize, stride: usize) -> Self {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a grid axis is far below 2^52 nodes"
        )]
        let last = (size - 1) as f64;
        let pos = pos.clamp(0.0, last);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos is clamped to [0, size − 1]; NaN clamps to NaN and casts to 0"
        )]
        let lo = pos.floor() as usize;
        Self {
            lo,
            hi: (lo + 1).min(size - 1),
            frac: pos - pos.floor(),
            stride,
        }
    }
}

/// Call `visit(index, weight)` for each corner of the cell `axes` describe
/// whose weight is positive: `index` is the corner's node index (the sum
/// of each axis's chosen grid index times its stride), and the weights sum
/// to one.
pub(crate) fn for_each_corner(axes: &[Axis], mut visit: impl FnMut(usize, f64)) {
    for corner in 0..1usize << axes.len() {
        let mut weight = 1.0;
        let mut index = 0usize;
        for (bit, axis) in axes.iter().enumerate() {
            let (grid_index, axis_weight) = if corner >> bit & 1 == 1 {
                (axis.hi, axis.frac)
            } else {
                (axis.lo, 1.0 - axis.frac)
            };
            weight *= axis_weight;
            index += grid_index * axis.stride;
        }
        if weight > 0.0 {
            visit(index, weight);
        }
    }
}
