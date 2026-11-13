//! Parity of the Vulkan IDCT + dequantise + colour-conversion kernel with
//! its CPU twin: the same integer transform and YCbCr → RGB evaluated on
//! the host must give byte-identical RGBA8 output, for one and for three
//! components, on every repeat of the dispatch.
//!
//! Skipped when no Vulkan device opens.
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features vulkan,gpu-jpeg-huffman --test idct_parity`

#![cfg(all(feature = "vulkan", feature = "gpu-jpeg-huffman"))]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "test fixture: bounded block and pixel indices, LCG draws below \
              i32::MAX, clamped bytes; the twin mirrors the kernel's indexing"
)]

use rasterrocket_gpu::backend::GpuBackend;
use rasterrocket_gpu::backend::params::IdctParams;
use rasterrocket_gpu::backend::vulkan::VulkanBackend;

/// The kernel's 1-D integer transform: 13-bit constants, 64-bit
/// intermediates, round-half-up descale by `shift`.
fn idct_1d(v: &mut [i32; 8], shift: u32) {
    let descale = |x: i64| ((x + (1i64 << (shift - 1))) >> shift) as i32;
    let (z2, z3) = (i64::from(v[2]), i64::from(v[6]));
    let z1 = (z2 + z3) * 4433;
    let tmp2 = z1 - z3 * 15137;
    let tmp3 = z1 + z2 * 6270;
    let tmp0 = i64::from(v[0] + v[4]) << 13;
    let tmp1 = i64::from(v[0] - v[4]) << 13;
    let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
    let (w0, w1, w2, w3) = (
        i64::from(v[7]),
        i64::from(v[5]),
        i64::from(v[3]),
        i64::from(v[1]),
    );
    let za = (w0 + w3) * -7373;
    let zb = (w1 + w2) * -20995;
    let (zc, zd) = (w0 + w2, w1 + w3);
    let z5 = (zc + zd) * 9633;
    let zc = z5 - zc * 16069;
    let zd = z5 - zd * 3196;
    let w0 = w0 * 2446 + za + zc;
    let w1 = w1 * 16819 + zb + zd;
    let w2 = w2 * 25172 + zb + zc;
    let w3 = w3 * 12299 + za + zd;
    *v = [
        descale(t10 + w3),
        descale(t11 + w2),
        descale(t12 + w1),
        descale(t13 + w0),
        descale(t13 - w0),
        descale(t12 - w1),
        descale(t11 - w2),
        descale(t10 - w3),
    ];
}

/// Zigzag position → natural (row-major) position in an 8×8 block.
const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// The inputs of one dispatch: `nc` components on a `bw × bh` block grid.
struct Case {
    /// Zigzag coefficients, `[comp][block_y][block_x][64]`.
    coefficients: Vec<i32>,
    /// Zigzag quantisation tables, `[table][64]`.
    qtables: Vec<i32>,
    /// Absolute DC per block, replacing zigzag slot 0.
    dc_values: Vec<i32>,
    nc: usize,
    bw: usize,
    bh: usize,
}

impl Case {
    /// Pseudo-random JPEG-like blocks: DC across the full level range, AC
    /// magnitudes falling with zigzag position, two thirds of them zero.
    fn new(nc: usize, bw: usize, bh: usize) -> Self {
        let blocks = nc * bw * bh;
        let mut rng = 0x5eed_1dc7_u32 ^ nc as u32;
        let mut next = move |m: u32| {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (rng >> 8) % m
        };
        let mut coefficients = vec![0i32; blocks * 64];
        let mut dc_values = vec![0i32; blocks];
        for b in 0..blocks {
            dc_values[b] = next(256) as i32 - 128;
            for zz in 1..64 {
                if next(3) == 0 {
                    let span = 1 + 400 / zz as u32;
                    coefficients[b * 64 + zz] = next(2 * span + 1) as i32 - span as i32;
                }
            }
        }
        let qtables = (0..64 * nc.min(2))
            .map(|i| 1 + (i % 64) as i32 / 4 + next(4) as i32)
            .collect();
        Self {
            coefficients,
            qtables,
            dc_values,
            nc,
            bw,
            bh,
        }
    }

    /// RGBA8 output of the CPU twin of the kernel.
    fn cpu(&self) -> Vec<u8> {
        let (w, h) = (self.bw * 8, self.bh * 8);
        let n_qt = self.qtables.len() / 64;
        let mut planes = vec![0u8; self.nc * w * h];
        for comp in 0..self.nc {
            let qt = &self.qtables[comp.min(1).min(n_qt - 1) * 64..][..64];
            for by in 0..self.bh {
                for bx in 0..self.bw {
                    let block = (comp * self.bh + by) * self.bw + bx;
                    let mut s = [[0i32; 8]; 8];
                    for zz in 0..64 {
                        let coef = if zz == 0 {
                            self.dc_values[block]
                        } else {
                            self.coefficients[block * 64 + zz]
                        };
                        let nat = ZIGZAG_TO_NATURAL[zz];
                        s[nat / 8][nat % 8] = coef * qt[zz];
                    }
                    for row in &mut s {
                        idct_1d(row, 11);
                    }
                    for col in 0..8 {
                        let mut v = [0i32; 8];
                        for k in 0..8 {
                            v[k] = s[k][col];
                        }
                        idct_1d(&mut v, 18);
                        for k in 0..8 {
                            let px = (by * 8 + k) * w + bx * 8 + col;
                            planes[comp * w * h + px] = (v[k] + 128).clamp(0, 255) as u8;
                        }
                    }
                }
            }
        }
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in 0..w * h {
            let y = i32::from(planes[px]);
            let rgb = if self.nc == 1 {
                [y; 3]
            } else {
                let cb = i32::from(planes[w * h + px]) - 128;
                let cr = i32::from(planes[2 * w * h + px]) - 128;
                let y0 = (y << 14) + 8192;
                [
                    (y0 + 22970 * cr) >> 14,
                    (y0 - 11700 * cr - 5638 * cb) >> 14,
                    (y0 + 29032 * cb) >> 14,
                ]
            };
            rgba.extend(rgb.map(|c| c.clamp(0, 255) as u8));
            rgba.push(0xFF);
        }
        rgba
    }

    /// RGBA8 output of the kernel on `backend`.
    fn vulkan(&self, backend: &VulkanBackend) -> Vec<u8> {
        let (w, h) = ((self.bw * 8) as u32, (self.bh * 8) as u32);
        let out_bytes = (w * h * 4) as usize;
        let upload = |data: &[i32]| {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            let buf = backend.alloc_device(bytes.len()).expect("alloc");
            backend.upload(&buf, bytes).expect("upload");
            buf
        };
        let coefficients = upload(&self.coefficients);
        let qtables = upload(&self.qtables);
        let dc_values = upload(&self.dc_values);
        let pixels = backend.alloc_device_readback(out_bytes).expect("alloc");
        backend
            .run_page(|b| {
                b.record_idct(IdctParams {
                    coefficients: &coefficients,
                    qtables: &qtables,
                    dc_values: &dc_values,
                    pixels_rgba: &pixels,
                    width: w,
                    height: h,
                    num_components: self.nc as u32,
                    blocks_wide: self.bw as u32,
                    blocks_high: self.bh as u32,
                    num_qtables: (self.qtables.len() / 64) as u32,
                })
            })
            .expect("record_idct");
        let mut out = vec![0u8; out_bytes];
        backend.download(&pixels, &mut out).expect("download");
        for buf in [coefficients, qtables, dc_values, pixels] {
            backend.free_device(buf);
        }
        out
    }
}

/// Four dispatches of a 96 × 64-block image with `nc` components, each
/// byte-identical to the CPU twin.
fn check(nc: usize) {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("skipping: no Vulkan device");
        return;
    };
    let case = Case::new(nc, 96, 64);
    let expect = case.cpu();
    for round in 0..4 {
        let got = case.vulkan(&backend);
        let first = got.iter().zip(&expect).position(|(g, e)| g != e);
        assert!(
            first.is_none(),
            "{nc}-component IDCT, dispatch {round}: first differing byte {first:?} of {}",
            expect.len()
        );
    }
}

#[test]
fn idct_ycbcr_matches_cpu_twin() {
    check(3);
}

#[test]
fn idct_gray_matches_cpu_twin() {
    check(1);
}
