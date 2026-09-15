//! Tile records, Halton sampling, and CPU anti-aliased fill.

use cudarc::driver::DeviceRepr;

use super::{TILE_H, TILE_W};

/// One tile record per (segment, tile-row) crossing.
///
/// Layout must match `struct TileRecord` in `tile_fill.cu` exactly.
/// The struct is `repr(C)` and 32 bytes so that `bytemuck::cast_slice` can
/// transmit it to the GPU without additional copying.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TileRecord {
    /// Sort key: `(tile_y << 16) | tile_x`.  Set by [`build_tile_records`].
    pub key: u32,
    /// Segment x at the top of the segment's y-extent within this tile (tile-local).
    pub x_enter: f32,
    /// Slope: dx/dy in device pixels.
    pub dxdy: f32,
    /// Segment start y within this tile row (0..`TILE_H`).
    pub y0_tile: f32,
    /// Segment end y within this tile row (0..`TILE_H`).
    pub y1_tile: f32,
    /// Sign: `+1.0` for an upward-crossing segment, `-1.0` for downward.
    pub sign: f32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad: u32,
    /// Padding (must be 0).
    #[expect(
        clippy::pub_underscore_fields,
        reason = "padding field required for repr(C) alignment to match tile_fill.cu struct layout"
    )]
    pub _pad2: u32,
}

// SAFETY: TileRecord is repr(C), all fields are primitive types (u32, f32), no
// uninitialised padding — bytemuck::Pod and cudarc::DeviceRepr are safe.
unsafe impl bytemuck::Pod for TileRecord {}
unsafe impl bytemuck::Zeroable for TileRecord {}
// SAFETY: TileRecord has no pointer types or other non-device-representable fields;
// all fields are plain u32/f32 with repr(C) alignment.
unsafe impl DeviceRepr for TileRecord {}

/// f32 aliases for tile dimensions — values are 16.0, exact in f32.
/// Avoids repeated `TILE_W/H as f32` casts inside `build_tile_records` that
/// would fire `cast_precision_loss` despite being trivially safe.
#[expect(
    clippy::cast_precision_loss,
    reason = "TILE_W/H = 16, exact in f32 (24-bit mantissa)"
)]
const TILE_W_F: f32 = TILE_W as f32;
#[expect(
    clippy::cast_precision_loss,
    reason = "TILE_W/H = 16, exact in f32 (24-bit mantissa)"
)]
const TILE_H_F: f32 = TILE_H as f32;

/// Build a sorted list of [`TileRecord`]s from a flat segment list, plus the
/// `tile_starts` / `tile_counts` index arrays required by [`crate::GpuCtx::tile_fill`].
///
/// `segs` is packed `[x0, y0, x1, y1]` per segment in device pixels, same
/// format as [`crate::GpuCtx::aa_fill`].  `x_min`, `y_min`, `width`, `height` define
/// the fill bounding box in device pixels.
///
/// Returns `(records, tile_starts, tile_counts, grid_w)`.
///
/// # Panics
///
/// Panics if `segs.len()` is not a multiple of 4, or if `width` or `height`
/// require more than 65535 tiles in either dimension (i.e. exceed `65535 × TILE_W`
/// or `65535 × TILE_H` pixels) — the sort key packs tile coordinates into 16 bits each.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "tile-record builder is a single coherent algorithm; splitting would obscure the data flow"
)]
pub fn build_tile_records(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
) -> (Vec<TileRecord>, Vec<u32>, Vec<u32>, u32) {
    assert!(
        segs.len().is_multiple_of(4),
        "segs.len() must be a multiple of 4 (got {})",
        segs.len()
    );

    let grid_w = width.div_ceil(TILE_W);
    let grid_h = height.div_ceil(TILE_H);
    assert!(
        grid_w <= 0xFFFF && grid_h <= 0xFFFF,
        "raster too large: grid {grid_w}×{grid_h} tiles exceeds 16-bit tile key range",
    );
    // grid_w, grid_h ≤ 0xFFFF; product ≤ 65535² = 4_294_836_225 < u32::MAX — no overflow.
    let n_tiles = (grid_w * grid_h) as usize;

    let mut records: Vec<TileRecord> = Vec::new();

    for seg in segs.as_chunks::<4>().0 {
        let (mut sx0, mut sy0, mut sx1, mut sy1) = (
            seg[0] - x_min,
            seg[1] - y_min,
            seg[2] - x_min,
            seg[3] - y_min,
        );

        // Skip horizontal segments (no winding contribution) and any segment
        // with non-finite coordinates.  `!is_finite` catches NaN and Inf from
        // a malformed or adversarial path; unguarded NaN would propagate into
        // TileRecord fields and corrupt GPU coverage computation.
        let dy = (sy1 - sy0).abs();
        if !dy.is_finite() || dy < 1e-6_f32 || !sx0.is_finite() || !sx1.is_finite() {
            continue;
        }

        // Enforce sy0 ≤ sy1; record crossing direction as sign.
        let sign = if sy0 > sy1 {
            std::mem::swap(&mut sx0, &mut sx1);
            std::mem::swap(&mut sy0, &mut sy1);
            -1.0f32
        } else {
            1.0f32
        };

        let dxdy = (sx1 - sx0) / (sy1 - sy0);
        // Two individually finite x coordinates can still overflow the
        // difference or the quotient (e.g. ±2e38 endpoints); an infinite
        // slope would propagate NaN through the fma in x_enter and paint
        // silent full-coverage bands. Such a segment is effectively
        // vertical at the scale of the raster and outside any real page
        // geometry — skip it.
        if !dxdy.is_finite() {
            continue;
        }

        // Clamp to output bounds.
        let ey0 = sy0.max(0.0);
        // height ≤ u32::MAX px; page heights are always ≤ 32768 at supported DPIs —
        // exact in f32 (which has a 24-bit mantissa, covering integers to 16M).
        #[expect(
            clippy::cast_precision_loss,
            reason = "height ≤ 32768 px in practice; exact in f32 (24-bit mantissa)"
        )]
        let ey1 = sy1.min(height as f32);
        if ey0 >= ey1 {
            continue;
        }

        // First and last tile rows the segment (after clamping) crosses.
        // Subtract a small epsilon from ey1 so a segment ending exactly on a
        // tile boundary doesn't bleed into the next tile row.
        // ey0/ey1 are non-negative and floored ≤ height/TILE_H_F ≤ grid_h ≤ 0xFFFF — fits u32.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "floor(non-negative / TILE_H_F) is ≥ 0 and ≤ grid_h ≤ 0xFFFF — fits u32"
        )]
        let ty0 = (ey0 / TILE_H_F).floor() as u32;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "floor(non-negative / TILE_H_F) is ≥ 0 and ≤ grid_h ≤ 0xFFFF — fits u32"
        )]
        let ty1 = ((ey1 - 1e-6).max(0.0) / TILE_H_F).floor() as u32;

        for ty in ty0..=ty1.min(grid_h - 1) {
            // ty ≤ 0xFFFF, TILE_H_F = 16.0; product ≤ ~1M — exact in f32 (24-bit mantissa).
            #[expect(
                clippy::cast_precision_loss,
                reason = "ty * TILE_H ≤ 0xFFFF*16 ≈ 1M; exact in f32"
            )]
            let tile_top = (ty * TILE_H) as f32;
            let tile_bot = tile_top + TILE_H_F;

            // Segment y-extent clipped to this tile row, in tile-local coords.
            let seg_y0_tile = ey0.max(tile_top) - tile_top;
            let seg_y1_tile = ey1.min(tile_bot) - tile_top;
            if seg_y0_tile >= seg_y1_tile {
                continue;
            }

            // Global x where the segment enters this tile row (at the clipped ey0).
            let x_enter_global = dxdy.mul_add(ey0.max(tile_top) - sy0, sx0);
            // Global x at the exit of this tile row.
            let x_at_exit = dxdy.mul_add(seg_y1_tile - seg_y0_tile, x_enter_global);

            // Rightmost x the segment reaches within this tile row.
            let xr = x_enter_global.max(x_at_exit);

            // Rightmost tile column the segment reaches. A negative value means the
            // segment lies entirely left of the raster, so the tile row has nothing
            // to record.
            // xr is an f32 page coordinate; floor→i32 is safe for any realistic page width.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "floor(f32) for page-coordinate x; realistic page widths fit comfortably in i32"
            )]
            let tx1_i = (xr / TILE_W_F).floor() as i32;
            if tx1_i < 0 {
                continue;
            }
            // tx1_i ≥ 0 checked above — safe to cast to u32.
            #[expect(
                clippy::cast_sign_loss,
                reason = "tx1_i ≥ 0 verified by the guard above"
            )]
            let tx1 = (tx1_i as u32).min(grid_w - 1);

            // Emit from tile column 0 rather than from the first column the segment
            // crosses. A pixel's winding number is the signed sum over every segment
            // to its right, so a segment must be visible to all tile columns left of
            // it, not only to those it passes through. `segment_pixel_area` discards
            // segments that are genuinely left of a pixel via its `xr <= px` branch,
            // so widening the range cannot add coverage.
            for tx in 0..=tx1 {
                // tx ≤ grid_w-1 ≤ 0xFFFE, TILE_W_F = 16.0; product ≤ ~1M — exact in f32.
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "tx * TILE_W ≤ 0xFFFE*16 ≈ 1M; exact in f32"
                )]
                records.push(TileRecord {
                    key: (ty << 16) | tx,
                    // tile-local x: subtract this tile column's left edge.
                    x_enter: x_enter_global - (tx * TILE_W) as f32,
                    dxdy,
                    y0_tile: seg_y0_tile,
                    y1_tile: seg_y1_tile,
                    sign,
                    _pad: 0,
                    _pad2: 0,
                });
            }
        }
    }

    // Sort records by (tile_y, tile_x) key — CPU sort is faster than CUB radix
    // sort for typical PDF segment counts (O(100–1000) records).
    records.sort_unstable_by_key(|r| r.key);

    // Build exclusive prefix-sum index: tile_starts[i] = first record index for tile i.
    let mut tile_starts = vec![0u32; n_tiles];
    let mut tile_counts = vec![0u32; n_tiles];

    for rec in &records {
        let tile_idx = ((rec.key >> 16) * grid_w + (rec.key & 0xFFFF)) as usize;
        // tile_idx is always < n_tiles: key components are clamped to [0, grid_w/h-1]
        // at record-build time above.
        debug_assert!(tile_idx < n_tiles, "record key out of tile grid range");
        if tile_idx < n_tiles {
            tile_counts[tile_idx] += 1;
        }
    }

    let mut running = 0u32;
    for (start, count) in tile_starts.iter_mut().zip(tile_counts.iter()) {
        *start = running;
        running = running
            .checked_add(*count)
            .expect("tile record count overflows u32");
    }

    (records, tile_starts, tile_counts, grid_w)
}

/// Halton(2) jitter X offsets within [0,1) for 64-sample AA.
const HALTON2: [f32; 64] = [
    0.5,
    0.25,
    0.75,
    0.125,
    0.625,
    0.375,
    0.875,
    0.062_5,
    0.562_5,
    0.312_5,
    0.812_5,
    0.187_5,
    0.687_5,
    0.437_5,
    0.937_5,
    0.031_25,
    0.531_25,
    0.281_25,
    0.781_25,
    0.156_25,
    0.656_25,
    0.406_25,
    0.906_25,
    0.093_75,
    0.593_75,
    0.343_75,
    0.843_75,
    0.218_75,
    0.718_75,
    0.468_75,
    0.968_75,
    0.015_625,
    0.515_625,
    0.265_625,
    0.765_625,
    0.140_625,
    0.640_625,
    0.390_625,
    0.890_625,
    0.078_125,
    0.578_125,
    0.328_125,
    0.828_125,
    0.203_125,
    0.703_125,
    0.453_125,
    0.953_125,
    0.046_875,
    0.546_875,
    0.296_875,
    0.796_875,
    0.171_875,
    0.671_875,
    0.421_875,
    0.921_875,
    0.109_375,
    0.609_375,
    0.359_375,
    0.859_375,
    0.234_375,
    0.734_375,
    0.484_375,
    0.984_375,
    0.007_812_5,
];

/// Halton(3) jitter Y offsets within [0,1) for 64-sample AA.
///
/// `HALTON3[i]` = halton(3, i+1).  All 64 values are distinct and cover [0,1)
/// with a low-discrepancy distribution.
const HALTON3: [f32; 64] = [
    // n=1..8
    0.333_333, 0.666_667, 0.111_111, 0.444_444, 0.777_778, 0.222_222, 0.555_556, 0.888_889,
    // n=9..16
    0.037_037, 0.370_370, 0.703_704, 0.148_148, 0.481_481, 0.814_815, 0.259_259, 0.592_593,
    // n=17..24
    0.925_926, 0.074_074, 0.407_407, 0.740_741, 0.185_185, 0.518_519, 0.851_852, 0.296_296,
    // n=25..32
    0.629_630, 0.962_963, 0.012_346, 0.345_679, 0.679_012, 0.123_457, 0.456_790, 0.790_123,
    // n=33..40
    0.234_568, 0.567_901, 0.901_235, 0.049_383, 0.382_716, 0.716_049, 0.160_494, 0.493_827,
    // n=41..48
    0.827_160, 0.271_605, 0.604_938, 0.938_272, 0.086_420, 0.419_753, 0.753_086, 0.197_531,
    // n=49..56
    0.530_864, 0.864_198, 0.308_642, 0.641_975, 0.975_309, 0.024_691, 0.358_025, 0.691_358,
    // n=57..64
    0.135_802, 0.469_136, 0.802_469, 0.246_914, 0.580_247, 0.913_580, 0.061_728, 0.395_062,
];

/// CPU fallback for `aa_fill` using 64-sample Halton jitter per pixel.
///
/// Matches the GPU kernel's coverage computation exactly: same Halton(2,3)
/// sample offsets, same winding-number / even-odd logic, same scale formula.
/// Used when `n_pixels < GPU_AA_FILL_THRESHOLD` or when no CUDA device is present.
#[must_use]
pub fn aa_fill_cpu(
    segs: &[f32],
    x_min: f32,
    y_min: f32,
    width: u32,
    height: u32,
    eo: bool,
) -> Vec<u8> {
    let n_pixels = width as usize * height as usize;
    let mut out = vec![0u8; n_pixels];

    for py in 0..height {
        for px in 0..width {
            #[expect(
                clippy::cast_precision_loss,
                reason = "px/py ≤ width/height ≤ u32::MAX; at typical DPIs (≤ 32768 px) \
                          the f32 precision loss is sub-pixel and irrelevant for AA coverage"
            )]
            let (cx, cy) = (x_min + px as f32 + 0.5, y_min + py as f32 + 0.5);
            let mut hits = 0u32;
            for s in 0..64usize {
                let sx = cx + HALTON2[s] - 0.5;
                let sy = cy + HALTON3[s] - 0.5;
                if aa_fill_cpu_sample(segs, sx, sy, eo) {
                    hits += 1;
                }
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "hits ≤ 64; (64*255+32)>>6 = 255 — always fits u8"
            )]
            {
                out[py as usize * width as usize + px as usize] = ((hits * 255 + 32) >> 6) as u8;
            }
        }
    }
    out
}

fn aa_fill_cpu_sample(segs: &[f32], sx: f32, sy: f32, eo: bool) -> bool {
    let mut winding = 0i32;
    for seg in segs.as_chunks::<4>().0 {
        let (x0, y0, x1, y1) = (seg[0], seg[1], seg[2], seg[3]);
        if y0 <= sy && sy < y1 {
            let t = (sy - y0) / (y1 - y0);
            let xi = t.mul_add(x1 - x0, x0);
            if xi >= sx {
                winding += 1;
            }
        } else if y1 <= sy && sy < y0 {
            let t = (sy - y1) / (y0 - y1);
            let xi = t.mul_add(x0 - x1, x1);
            if xi >= sx {
                winding -= 1;
            }
        }
    }
    if eo { (winding & 1) != 0 } else { winding != 0 }
}

/// CPU model of the `tile_fill` kernel's per-pixel arithmetic, shared by
/// the unit tests below and the GPU parity tests in `lib.rs`. Kept
/// operation-for-operation compatible with `kernels/tile_fill.cu` /
/// `.slang` so a formula drift in either direction surfaces as a parity
/// failure.
#[cfg(test)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(crate) is intentional: the parent `fill` module is private; \
              explicit visibility documents that lib.rs's GPU parity tests \
              consume these helpers"
)]
pub(crate) mod test_model {
    use super::{TILE_H, TILE_W, TileRecord};

    /// CPU model of `segment_pixel_area` in `kernels/tile_fill.cu`.
    ///
    /// Kept operation-for-operation compatible with the device code so the
    /// coverage tests below exercise the same arithmetic the kernel performs.
    /// Inputs are finite by precondition (`build_tile_records` filters
    /// non-finite coordinates and slopes); on NaN the CUDA fmin/fmax and
    /// Rust min/max/clamp families diverge, so no parity claim is made
    /// there. The model bakes `mul_add` where nvcc contracts to fma; the
    /// GPU parity test's ±1-byte budget absorbs the ulp-level differences
    /// (a future Slang/SPIR-V twin may pair contractions differently but
    /// stays within the same budget).
    ///
    /// Computes the exact clipped-trapezoid integral
    /// `sign × ∫_{iy0}^{iy1} clamp(x(y) − px, 0, 1) dy` — the winding-count
    /// contribution of the segment integrated over the pixel's interior,
    /// not a per-y step function sampled at the pixel's left edge.
    pub(crate) fn segment_pixel_area(
        x_at_iy0: f32,
        dxdy: f32,
        iy0: f32,
        iy1: f32,
        sign: f32,
        px: f32,
    ) -> f32 {
        let y_len = iy1 - iy0;
        if y_len <= 0.0 {
            return 0.0;
        }
        let x0 = x_at_iy0;
        let x1 = dxdy.mul_add(iy1 - iy0, x_at_iy0);
        let xl = x0.min(x1);
        let xr = x0.max(x1);
        let cover = if xr <= px {
            0.0
        } else if xl >= px + 1.0 {
            y_len
        } else {
            let dx = xr - xl;
            if dx < 1e-6 {
                // Near-vertical: constant x ≈ xmid across the row.
                let xmid = 0.5 * (x0 + x1);
                y_len * (xmid - px).clamp(0.0, 1.0)
            } else {
                // Split the y-interval where x(y) crosses px and px + 1.
                // x(y) is monotone over the sorted range [xl, xr]; mapping
                // y-fractions through the sorted range is direction-safe
                // because the integrand depends only on the distribution
                // of x values, which is uniform along the segment.
                let left = xl.max(px);
                let right = xr.min(px + 1.0);
                let yf_left = (left - xl) / dx * y_len;
                let yf_right = (right - xl) / dx * y_len;
                // y-span with x ≥ px + 1 contributes 1 per unit y; the
                // partial span contributes the mean of (x − px) over its
                // linear sweep from `left` to `right`.
                let above = y_len - yf_right;
                (yf_right - yf_left).mul_add(0.5f32.mul_add(left + right, -px), above)
            }
        };
        cover * sign
    }

    /// CPU model of the `tile_fill` kernel's per-pixel accumulation loop.
    ///
    /// Returns the winding/area value the kernel would compute for `(px, py)`.
    pub(crate) fn kernel_area_at(
        recs: &[TileRecord],
        starts: &[u32],
        counts: &[u32],
        grid_w: u32,
        px: u32,
        py: u32,
    ) -> f32 {
        let (tile_x, tile_y) = (px / TILE_W, py / TILE_H);
        let (px_local, py_local) = (px % TILE_W, py % TILE_H);
        let idx = (tile_y * grid_w + tile_x) as usize;
        let (start, count) = (starts[idx] as usize, counts[idx] as usize);
        let py_f = py_local as f32;
        let mut area = 0.0f32;
        for rec in &recs[start..start + count] {
            let iy0 = rec.y0_tile.max(py_f);
            let iy1 = rec.y1_tile.min(py_f + 1.0);
            if iy0 >= iy1 {
                continue;
            }
            let x_at_iy0 = rec.dxdy.mul_add(iy0 - rec.y0_tile, rec.x_enter);
            area += segment_pixel_area(x_at_iy0, rec.dxdy, iy0, iy1, rec.sign, px_local as f32);
        }
        area
    }

    /// Kernel-model coverage byte for `(px, py)`, mirroring the
    /// kernel's area → byte conversion for both fill rules.
    ///
    /// Non-zero winding: `min(|area|, 1) × 255.5`, truncated.
    /// Even-odd: the accumulated signed area folds with the period-2
    /// triangle wave peaking at odd integers — `t = |area| mod 2`,
    /// coverage `t` for `t ≤ 1` and `2 − t` above — so a fully
    /// interior pixel of a simple path (`|area| = 1`) maps to full
    /// coverage and winding-2 overlap regions back to zero.
    pub(crate) fn kernel_coverage_at(
        recs: &[TileRecord],
        starts: &[u32],
        counts: &[u32],
        grid_w: u32,
        px: u32,
        py: u32,
        eo: bool,
    ) -> u8 {
        let area = kernel_area_at(recs, starts, counts, grid_w, px, py);
        let a = if eo {
            let t = area.abs() % 2.0;
            if t > 1.0 { 2.0 - t } else { t }
        } else {
            area.abs().min(1.0)
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a is in [0, 1]; × 255.5 truncated to 0..=255 is the kernel's own conversion"
        )]
        {
            ((a * 255.5) as i32).min(255) as u8
        }
    }

    /// The parity shapes shared by the CPU-model-vs-aa test here and
    /// the GPU-vs-model test in `lib.rs` — one list so the two oracles
    /// cannot drift apart.
    pub(crate) fn parity_shapes() -> [(&'static str, Vec<f32>, u32, u32); 3] {
        [
            (
                "interior rect",
                vec![10.0, 10.0, 10.0, 110.0, 210.0, 110.0, 210.0, 10.0],
                224,
                128,
            ),
            (
                "boundary rect",
                vec![0.0, 0.0, 0.0, 16.0, 48.0, 16.0, 48.0, 0.0],
                48,
                16,
            ),
            (
                "right triangle",
                vec![
                    0.0, 0.0, 32.0, 32.0, 32.0, 32.0, 0.0, 32.0, 0.0, 32.0, 0.0, 0.0,
                ],
                32,
                32,
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::test_model::{kernel_area_at, kernel_coverage_at, segment_pixel_area};
    use super::{aa_fill_cpu, build_tile_records};

    #[test]
    fn aa_fill_cpu_solid_rect_full_coverage() {
        let segs: Vec<f32> = vec![
            -100.0, -100.0, 100.0, -100.0, 100.0, -100.0, 100.0, 100.0, 100.0, 100.0, -100.0,
            100.0, -100.0, 100.0, -100.0, -100.0,
        ];
        let cov = aa_fill_cpu(&segs, 0.0, 0.0, 1, 1, false);
        assert_eq!(cov.len(), 1);
        assert_eq!(cov[0], 255, "fully covered pixel should be 255");
    }

    #[test]
    fn aa_fill_cpu_outside_rect_zero_coverage() {
        let segs: Vec<f32> = vec![
            0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0,
        ];
        let cov = aa_fill_cpu(&segs, 200.0, 200.0, 1, 1, false);
        assert_eq!(cov[0], 0, "pixel outside rect should be 0");
    }

    #[test]
    fn aa_fill_cpu_empty_segs_zero_coverage() {
        let cov = aa_fill_cpu(&[], 0.0, 0.0, 4, 4, false);
        assert_eq!(cov.len(), 16);
        assert!(
            cov.iter().all(|&v| v == 0),
            "empty segs → all zero coverage"
        );
    }

    #[test]
    fn aa_fill_cpu_eo_donut_inner_zero() {
        let outer: [f32; 16] = [
            -10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 10.0, 10.0, 10.0, -10.0, 10.0, -10.0,
            10.0, -10.0, -10.0,
        ];
        let inner: [f32; 16] = [
            -5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0, -5.0,
        ];
        let segs: Vec<f32> = outer.iter().chain(inner.iter()).copied().collect();
        let cov = aa_fill_cpu(&segs, -0.5, -0.5, 1, 1, true);
        assert_eq!(cov[0], 0, "EO donut centre should be 0");
    }

    #[test]
    fn tile_records_empty_segs() {
        let (recs, starts, counts, grid_w) = build_tile_records(&[], 0.0, 0.0, 32, 32);
        assert!(recs.is_empty());
        assert_eq!(grid_w, 2);
        assert!(starts.iter().all(|&s| s == 0));
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn tile_records_single_vertical_segment() {
        // Segment from (8,0) to (8,17): crosses tile rows 0 (y 0..16) and 1 (y 16..17).
        // It is emitted to every tile column at or left of the one it spans, so each
        // of the two tile rows contributes one record for tile column 0.
        let segs = [8.0f32, 0.0, 8.0, 17.0];
        let (recs, _, _, _) = build_tile_records(&segs, 0.0, 0.0, 64, 64);
        assert_eq!(recs.len(), 2, "one record per tile row crossed");
    }

    #[test]
    fn tile_records_reject_overflowing_slopes() {
        // Individually finite endpoints whose difference overflows f32:
        // the quotient is ±inf and the first-row x_enter fma would be
        // NaN. No record may carry a non-finite field (a NaN x_enter
        // renders as a silent full-coverage band).
        let segs = [
            3.0e38f32, 0.0, -3.0e38, 16.0, // dx overflows → dxdy = -inf
            1.0e30f32, 0.0, -1.0e30,
            1e-5, // finite dx, dy near the floor → quotient overflows
        ];
        let (recs, _, _, _) = build_tile_records(&segs, 0.0, 0.0, 64, 64);
        for rec in &recs {
            assert!(
                rec.x_enter.is_finite() && rec.dxdy.is_finite(),
                "non-finite record emitted: x_enter={} dxdy={}",
                rec.x_enter,
                rec.dxdy,
            );
        }
    }

    #[test]
    fn tile_records_diagonal_segment() {
        let segs = [0.0f32, 0.0, 32.0, 32.0];
        let (recs, _, _, _) = build_tile_records(&segs, 0.0, 0.0, 128, 128);
        assert!(recs.len() >= 2, "diagonal must produce at least 2 records");
    }

    /// `segment_pixel_area` computes the exact clipped-trapezoid integral
    /// `∫ clamp(x(y) − px, 0, 1) dy` over the pixel row — pin its value on
    /// the shapes where the old `frac × x_right_frac` product (and the
    /// kernels' span-sum variant) over-weighted partial crossings.
    #[test]
    fn segment_pixel_area_is_the_exact_clipped_trapezoid_integral() {
        // Diagonal sweeping the full pixel width within one scanline
        // (x: 0 → 1 over the row): true covered area is 0.5, not 1.0.
        let diag = segment_pixel_area(0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
        assert!((diag - 0.5).abs() < 1e-6, "diagonal: got {diag}");

        // Near-vertical edge at x = 0.25: covers a quarter of the pixel,
        // not the old all-or-nothing midpoint step.
        let vert = segment_pixel_area(0.25, 0.0, 0.0, 1.0, 1.0, 0.0);
        assert!((vert - 0.25).abs() < 1e-6, "vertical: got {vert}");

        // Partial crossing exiting right: x: 0.5 → 2.0 over the row.
        // ∫₀^⅓ (0.5 + 1.5y) dy + ∫_⅓^1 1 dy = 1/4 + 2/3 = 11/12.
        let cross = segment_pixel_area(0.5, 1.5, 0.0, 1.0, 1.0, 0.0);
        assert!(
            (cross - 11.0 / 12.0).abs() < 1e-6,
            "partial crossing: got {cross}"
        );

        // Fully right of the column: full y-span; fully left: nothing.
        let right = segment_pixel_area(2.0, 0.5, 0.0, 1.0, 1.0, 0.0);
        assert!((right - 1.0).abs() < 1e-6, "fully right: got {right}");
        let left = segment_pixel_area(-2.0, 0.5, 0.0, 1.0, 1.0, 0.0);
        assert!(left.abs() < 1e-6, "fully left: got {left}");
    }

    /// A fill wider than one tile must be solid across its interior.
    ///
    /// Winding at a pixel is the signed sum over every segment to its right, so a
    /// segment has to reach tile columns left of the ones it crosses. When records
    /// were emitted only to the columns a segment physically spanned, every interior
    /// pixel more than one tile from an edge accumulated nothing and rendered as
    /// background — losing the interior of any fill wider than `TILE_W`.
    #[test]
    fn tile_fill_interior_of_a_wide_rect_is_covered() {
        // Rect x 10..210, y 10..110 — 200 px wide, i.e. many tile columns.
        let segs = [
            10.0f32, 10.0, 10.0, 110.0, // left edge, downward
            210.0f32, 110.0, 210.0, 10.0, // right edge, upward
        ];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 320, 200);

        for px in [12u32, 40, 80, 120, 160, 200] {
            let area = kernel_area_at(&recs, &starts, &counts, grid_w, px, 50);
            assert!(
                area.abs() > 0.5,
                "interior pixel ({px},50) has area {area}, expected full coverage"
            );
        }

        for px in [0u32, 5, 215, 300] {
            let area = kernel_area_at(&recs, &starts, &counts, grid_w, px, 50);
            assert!(
                area.abs() < 0.5,
                "exterior pixel ({px},50) has area {area}, expected none"
            );
        }
    }

    /// A rect whose right edge lands exactly at `x == bbox width` — with the
    /// width a multiple of `TILE_W`, as `gpu_fill_segs` produces for every
    /// axis-aligned shape with integral coordinates — must still cover its
    /// rightmost pixel column. A lower clamp on the emit range once culled
    /// exactly this configuration, rendering the whole rect blank.
    #[test]
    fn tile_fill_right_edge_at_tile_boundary_is_covered() {
        let segs = [
            0.0f32, 0.0, 0.0, 16.0, // left edge, downward
            48.0f32, 16.0, 48.0, 0.0, // right edge, upward
        ];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 48, 16);
        for px in 0..48u32 {
            let area = kernel_area_at(&recs, &starts, &counts, grid_w, px, 8);
            assert!(
                area.abs() > 0.5,
                "pixel ({px},8) has area {area}, expected full coverage"
            );
        }
    }

    /// The tile kernel model must agree with `aa_fill_cpu` on every pixel:
    /// exactly on pixels `aa_fill_cpu` resolves as fully inside or fully
    /// outside, and within a sampling-noise bound on antialiased boundary
    /// pixels (the exact clipped-trapezoid integral vs the 64-sample
    /// jittered estimate).
    ///
    /// This is the guard against the total-content-loss class of record
    /// bugs (a culled or missing record flips whole interior columns to 0)
    /// *and* against edge-formula drift — the old `segment_pixel_area`
    /// over-weighted diagonally-crossed pixels by up to 2×, which the
    /// boundary bound now catches.
    #[test]
    fn tile_fill_matches_aa_fill_cpu_on_solid_pixels() {
        // Both fill rules: the shapes are simple paths, so even-odd and
        // non-zero winding must agree everywhere — which pins the eo
        // fold's period (a period-1 fold zeroes every interior pixel).
        for eo in [false, true] {
            for (name, segs, w, h) in super::test_model::parity_shapes() {
                let case_name = format!("{name} (eo={eo})");
                let aa = aa_fill_cpu(&segs, 0.0, 0.0, w, h, eo);
                let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, w, h);
                for py in 0..h {
                    for px in 0..w {
                        let t = kernel_coverage_at(&recs, &starts, &counts, grid_w, px, py, eo);
                        match aa[(py * w + px) as usize] {
                            255 => assert!(
                                t >= 240,
                                "{case_name}: interior pixel ({px},{py}) tile={t}, aa=255",
                            ),
                            0 => assert!(
                                t <= 16,
                                "{case_name}: exterior pixel ({px},{py}) tile={t}, aa=0",
                            ),
                            a => {
                                // 48 ≈ 2× the observed worst-case gap
                                // between the exact analytic integral and
                                // the 64-sample jittered estimate on these
                                // shapes (Halton discrepancy ~0.05–0.10
                                // area units, ~13–26 bytes, doubled at
                                // corners); the old formula's 2× diagonal
                                // overweight produced diffs up to ~127.
                                let diff = (i16::from(t) - i16::from(a)).abs();
                                assert!(
                                    diff <= 48,
                                    "{case_name}: boundary pixel ({px},{py}) tile={t}, aa={a}, \
                                     |diff|={diff}",
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every tile column of a wide fill must receive the records it needs.
    #[test]
    fn tile_records_reach_columns_left_of_the_segment() {
        // One vertical edge at x=200, i.e. tile column 12.
        let segs = [200.0f32, 0.0, 200.0, 16.0];
        let (_, _, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 320, 32);
        for tx in 0..=12u32 {
            assert!(
                counts[tx as usize] > 0,
                "tile column {tx} has no record for a segment at its right"
            );
        }
        let _ = grid_w;
    }

    #[test]
    fn tile_records_sort_order() {
        let segs = [24.0f32, 0.0, 24.0, 8.0, 8.0f32, 16.0, 8.0, 24.0];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 80, 80);
        assert_eq!(grid_w, 5);
        for w in recs.windows(2) {
            assert!(w[0].key <= w[1].key, "records must be sorted by key");
        }
        let grid_w_us = grid_w as usize;
        let tile_01 = 1; // (tile_y=0, tile_x=1)
        let tile_10 = grid_w_us; // (tile_y=1, tile_x=0)
        assert_eq!(counts[tile_01], 1);
        assert_eq!(counts[tile_10], 1);
        let _ = starts;
    }

    #[test]
    fn tile_records_prefix_sum_consistent() {
        let segs = [4.0f32, 0.0, 60.0, 63.0];
        let (recs, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, 64, 64);
        let _ = grid_w;
        let total: u32 = counts.iter().sum();
        assert_eq!(total as usize, recs.len(), "sum of counts == total records");
        for i in 0..counts.len() - 1 {
            assert_eq!(
                starts[i] + counts[i],
                starts[i + 1],
                "prefix sum broken at {i}"
            );
        }
    }
}
