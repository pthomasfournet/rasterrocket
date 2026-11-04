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
    /// `pos`, which is clamped to `[0, size − 1]`; a NaN position is the
    /// first node.  `size` is at least one.
    pub(crate) fn at(pos: f64, size: usize, stride: usize) -> Self {
        debug_assert!(size >= 1, "a grid axis holds at least one node");
        let last_index = size.saturating_sub(1);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a grid axis is far below 2^52 nodes"
        )]
        let pos = if pos.is_nan() {
            0.0
        } else {
            pos.clamp(0.0, last_index as f64)
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos lies in [0, size − 1]"
        )]
        let lo = pos.floor() as usize;
        Self {
            lo,
            hi: (lo + 1).min(last_index),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A NaN position is the first node, so the corner weights still sum
    /// to one.
    #[test]
    fn a_nan_position_is_the_first_node() {
        let axis = Axis::at(f64::NAN, 3, 1);
        assert_eq!((axis.lo, axis.hi), (0, 1));
        assert!(axis.frac.abs() < f64::EPSILON);
        let mut total = 0.0;
        for_each_corner(&[axis, Axis::at(0.5, 2, 3)], |_, w| total += w);
        assert!((total - 1.0).abs() < 1e-12, "weights sum to {total}");
    }

    #[test]
    fn positions_clamp_to_the_axis() {
        let below = Axis::at(-4.0, 3, 1);
        assert_eq!((below.lo, below.hi), (0, 1));
        let above = Axis::at(9.0, 3, 1);
        assert_eq!((above.lo, above.hi), (2, 2));
        let single = Axis::at(0.7, 1, 1);
        assert_eq!((single.lo, single.hi), (0, 0));
    }
}
