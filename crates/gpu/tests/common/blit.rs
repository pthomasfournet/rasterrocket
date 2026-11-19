//! The image blit on a backend against its CPU reference.
//!
//! [`run_blit`] draws a [`Blit`] through [`GpuBackend::record_blit_image`]
//! and [`blit_cpu_reference`] through the kernel's table walk on the host;
//! the cases below compare the two byte for byte.

use rasterrocket_gpu::backend::GpuBackend;
use rasterrocket_gpu::backend::layout::word_padded;
use rasterrocket_gpu::backend::params::{BLIT_FILTER_X, BLIT_FILTER_Y, BlitParams};

/// A source image: `width × height` samples, row-major.
#[derive(Clone, Copy)]
pub(crate) struct Image<'a> {
    /// The samples, `width × height × bpp` bytes.
    pub(crate) pixels: &'a [u8],
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `0` for RGB (3 bytes per sample), `1` for gray (1 byte).
    pub(crate) layout: u32,
}

/// One blit of `image` into a zeroed `page_w × page_h` RGBA8 page.
pub(crate) struct Blit<'a> {
    pub(crate) image: Image<'a>,
    pub(crate) page_w: u32,
    pub(crate) page_h: u32,
    /// `[x0, y0, x1, y1]` in page pixels.
    pub(crate) bbox: [i32; 4],
    /// One sampling-table entry of four words per box column.
    pub(crate) cols: &'a [u32],
    /// One sampling-table entry of four words per box row.
    pub(crate) rows: &'a [u32],
    /// [`BLIT_FILTER_X`] and/or [`BLIT_FILTER_Y`], or `0`.
    pub(crate) filter: u32,
}

impl Blit<'_> {
    const fn page_bytes(&self) -> usize {
        self.page_w as usize * self.page_h as usize * 4
    }
}

/// Q32 split of one sampling-table term: `(hi as u32, lo)` with
/// `hi = floor(v)`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "`hi` is a small integer and `v - hi` lies in [0, 1), so the scaled \
              fraction is non-negative and below 2^32"
)]
fn split_q32(v: f64) -> (u32, u32) {
    let hi = v.floor();
    (
        (hi as i32).cast_unsigned(),
        ((v - hi) * 4_294_967_296.0) as u32,
    )
}

/// Sampling tables for the separable source map
/// `X = x_col(dx) + x_row(dy)`, `Y = y_col(dx) + y_row(dy)`, evaluated at
/// pixel centres over `bbox` (`[x0, y0, x1, y1]`).  Same layout the
/// kernel reads: four `u32` words per entry, `[x_hi, x_lo, y_hi, y_lo]`.
fn blit_tables(
    bbox: [i32; 4],
    col_terms: impl Fn(f64) -> (f64, f64),
    row_terms: impl Fn(f64) -> (f64, f64),
) -> (Vec<u32>, Vec<u32>) {
    let entry = |(x, y): (f64, f64)| {
        let (xh, xl) = split_q32(x);
        let (yh, yl) = split_q32(y);
        [xh, xl, yh, yl]
    };
    let cols = (bbox[0]..bbox[2])
        .flat_map(|dx| entry(col_terms(f64::from(dx) + 0.5)))
        .collect();
    let rows = (bbox[1]..bbox[3])
        .flat_map(|dy| entry(row_terms(f64::from(dy) + 0.5)))
        .collect();
    (cols, rows)
}

/// Taps `(lo, hi, weight of hi)` along one axis of `n` samples for the
/// in-range integer part `i` and Q32 fraction `frac` of a source
/// coordinate: the sample `i` alone, or, when `blend` is set, the samples
/// whose centres bracket the coordinate, clamped onto the axis, weighted by
/// the top 8 bits of the coordinate's fraction measured from sample centres.
fn blit_axis_taps(i: i32, frac: u32, n: u32, blend: bool) -> (u32, u32, u32) {
    if !blend {
        return (i.cast_unsigned(), i.cast_unsigned(), 0);
    }
    let q = (i64::from(i) << 32) + i64::from(frac) - (1 << 31);
    let lower = q >> 32;
    let index = |v: i64| u32::try_from(v).expect("tap clamped onto the axis");
    (
        index(lower.max(0)),
        index((lower + 1).min(i64::from(n) - 1)),
        index((q >> 24) & 0xFF),
    )
}

/// The page the kernel draws for `blit`, by its table walk on the host:
/// sum column and row terms with carry, reject out-of-image samples, blend
/// the taps `filter` selects and write the pixel with alpha 255.
pub(crate) fn blit_cpu_reference(blit: &Blit<'_>) -> Vec<u8> {
    let Image {
        pixels: src,
        width: src_w,
        height: src_h,
        layout,
    } = blit.image;
    let mut page = vec![0; blit.page_bytes()];
    let bpp = if layout == 0 { 3 } else { 1 };
    let [bx0, by0, bx1, by1] = blit.bbox;
    // Table index is the position within the bbox; page bounds are the
    // kernel's own reject test.
    for (r_idx, dy) in (by0..by1).enumerate() {
        let Ok(dy) = u32::try_from(dy) else {
            continue;
        };
        if dy >= blit.page_h {
            continue;
        }
        for (c_idx, dx) in (bx0..bx1).enumerate() {
            let Ok(dx) = u32::try_from(dx) else {
                continue;
            };
            if dx >= blit.page_w {
                continue;
            }
            let c = &blit.cols[c_idx * 4..][..4];
            let r = &blit.rows[r_idx * 4..][..4];
            let (frac_x, carry_x) = c[1].overflowing_add(r[1]);
            let ix = c[0]
                .cast_signed()
                .wrapping_add(r[0].cast_signed())
                .wrapping_add(i32::from(carry_x));
            let (frac_y, carry_y) = c[3].overflowing_add(r[3]);
            let iy = c[2]
                .cast_signed()
                .wrapping_add(r[2].cast_signed())
                .wrapping_add(i32::from(carry_y));
            let (Ok(ux), Ok(uy)) = (u32::try_from(ix), u32::try_from(iy)) else {
                continue;
            };
            if ux >= src_w || uy >= src_h {
                continue;
            }
            let (x0, x1, wx) = blit_axis_taps(ix, frac_x, src_w, blit.filter & BLIT_FILTER_X != 0);
            let (y0, y1, wy) = blit_axis_taps(iy, frac_y, src_h, blit.filter & BLIT_FILTER_Y != 0);
            let at =
                |x: u32, y: u32, ch: usize| u32::from(src[(y * src_w + x) as usize * bpp + ch]);
            let blend = |ch: usize| {
                let row = |y| at(x0, y, ch) * (256 - wx) + at(x1, y, ch) * wx;
                let v = (row(y0) * (256 - wy) + row(y1) * wy + (1 << 15)) >> 16;
                u8::try_from(v).expect("a blend of bytes is a byte")
            };
            let rgb = if layout == 0 {
                [blend(0), blend(1), blend(2)]
            } else {
                [blend(0); 3]
            };
            let dst_off = (dy * blit.page_w + dx) as usize * 4;
            page[dst_off..dst_off + 3].copy_from_slice(&rgb);
            page[dst_off + 3] = 255;
        }
    }
    page
}

/// The page `backend` draws for `blit`, in one page of its own.  The
/// source buffer is the image's samples followed by the bytes up to a
/// whole word, each set to `pad`.
pub(crate) fn run_blit<B: GpuBackend>(backend: &B, blit: &Blit<'_>, pad: u8) -> Vec<u8> {
    let dst_bytes = blit.page_bytes();
    let cols_bytes: &[u8] = bytemuck::cast_slice(blit.cols);
    let rows_bytes: &[u8] = bytemuck::cast_slice(blit.rows);
    let mut padded = blit.image.pixels.to_vec();
    padded.resize(word_padded(padded.len()), pad);
    let mut d_src = backend.alloc_device(padded.len()).expect("alloc src");
    let mut d_dst = backend.alloc_device(dst_bytes).expect("alloc dst");
    let mut d_cols = backend.alloc_device(cols_bytes.len()).expect("alloc cols");
    let mut d_rows = backend.alloc_device(rows_bytes.len()).expect("alloc rows");
    backend.upload(&mut d_src, &padded).expect("upload src");
    backend
        .upload(&mut d_dst, &vec![0; dst_bytes])
        .expect("zero dst");
    backend
        .upload(&mut d_cols, cols_bytes)
        .expect("upload cols");
    backend
        .upload(&mut d_rows, rows_bytes)
        .expect("upload rows");
    backend
        .run_page(|b| {
            b.record_blit_image(BlitParams {
                src: &d_src,
                dst: &mut d_dst,
                cols: &d_cols,
                rows: &d_rows,
                src_w: blit.image.width,
                src_h: blit.image.height,
                src_layout: blit.image.layout,
                dst_w: blit.page_w,
                dst_h: blit.page_h,
                bbox: blit.bbox,
                filter: blit.filter,
            })
        })
        .expect("blit page");
    let mut out = vec![0; dst_bytes];
    backend.download(&d_dst, &mut out).expect("download");
    for buf in [d_src, d_dst, d_cols, d_rows] {
        backend.free_device(buf);
    }
    out
}

/// Blit `image` into a `page × page` page over the whole page; see
/// [`check_blit_parity_in`].
fn check_blit_parity<B: GpuBackend>(
    backend: &B,
    image: Image<'_>,
    page: u32,
    col_terms: impl Fn(f64) -> (f64, f64),
    row_terms: impl Fn(f64) -> (f64, f64),
    what: &str,
) {
    let side = i32::try_from(page).expect("a test page side fits i32");
    check_blit_parity_in(
        backend,
        image,
        (page, page),
        [0, 0, side, side],
        col_terms,
        row_terms,
        what,
    );
}

/// Blit `image` over `bbox` into a `page.0 × page.1` page, through the
/// CPU reference and `backend`, nearest-neighbour and then interpolated on
/// both axes, with the source's pad bytes zero and then `0xFF`, asserting
/// byte equality.
fn check_blit_parity_in<B: GpuBackend>(
    backend: &B,
    image: Image<'_>,
    (page_w, page_h): (u32, u32),
    bbox: [i32; 4],
    col_terms: impl Fn(f64) -> (f64, f64),
    row_terms: impl Fn(f64) -> (f64, f64),
    what: &str,
) {
    let (cols, rows) = blit_tables(bbox, col_terms, row_terms);
    for filter in [0, BLIT_FILTER_X | BLIT_FILTER_Y] {
        let blit = Blit {
            image,
            page_w,
            page_h,
            bbox,
            cols: &cols,
            rows: &rows,
            filter,
        };
        let cpu = blit_cpu_reference(&blit);
        for pad in [0x00, 0xFF] {
            let device = run_blit(backend, &blit, pad);
            assert_eq!(
                cpu,
                device,
                "{what} on {} (filter {filter}, pad {pad:#04x}) diverges",
                B::NAME
            );
        }
        assert!(
            cpu.as_chunks::<4>().0.iter().any(|px| px[3] == 255),
            "{what}: no pixel was written"
        );
    }
}

/// `n` bytes `(i · mul + add) mod modulus` for `i = 0, 1, …`.
fn ramp(n: u32, mul: u32, add: u32, modulus: u32) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i * mul + add) % modulus).expect("modulus at most 256"))
        .collect()
}

/// A 4×4 RGB image placed at identity into a 4×4 page: page pixel
/// `(dx, dy)` reads source `(dx, dy)`.
pub(crate) fn identity_rgb<B: GpuBackend>(backend: &B) {
    let pixels = ramp(4 * 4 * 3, 7, 3, 251);
    let image = Image {
        pixels: &pixels,
        width: 4,
        height: 4,
        layout: 0,
    };
    check_blit_parity(
        backend,
        image,
        4,
        |px| (px, 0.0),
        |py| (0.0, py),
        "blit identity RGB",
    );
}

/// Fractional scale and offset in both axes with a y-flip: exercises the
/// fraction-word carry, negative integer parts, and rejection of page
/// pixels outside the image.
#[expect(
    clippy::suboptimal_flops,
    reason = "test geometry: the source map is spelled as scale-then-offset for readability"
)]
pub(crate) fn scaled_offset_with_carry<B: GpuBackend>(backend: &B) {
    let pixels = ramp(4 * 4 * 3, 11, 5, 253);
    let image = Image {
        pixels: &pixels,
        width: 4,
        height: 4,
        layout: 0,
    };
    check_blit_parity(
        backend,
        image,
        6,
        |px| ((px - 1.3) * 0.7, 0.0),
        |py| (0.0, 4.0 - (py - 0.6) * 0.9),
        "blit scaled/offset RGB",
    );
}

/// A 4×4 gray image placed at identity into a 4×4 page.
pub(crate) fn gray_layout<B: GpuBackend>(backend: &B) {
    let pixels = ramp(4 * 4, 17, 0, 256);
    let image = Image {
        pixels: &pixels,
        width: 4,
        height: 4,
        layout: 1,
    };
    check_blit_parity(
        backend,
        image,
        4,
        |px| (px, 0.0),
        |py| (0.0, py),
        "blit gray-layout",
    );
}

/// Sources whose bytes end inside a word: the last word's unused bytes are
/// padding, zero and then `0xFF`.  Each is blitted at identity and scaled
/// 2×, each nearest-neighbour and interpolated on both axes.
pub(crate) fn unpadded_tail_source<B: GpuBackend>(backend: &B) {
    let gray_1x1 = [0xA7_u8];
    let rgb_3x1 = ramp(9, 0x17, 0x31, 256);
    for (pixels, width, layout, what) in [
        (&gray_1x1[..], 1, 1, "1×1 gray"),
        (&rgb_3x1[..], 3, 0, "3×1 RGB"),
    ] {
        let image = Image {
            pixels,
            width,
            height: 1,
            layout,
        };
        check_blit_parity(
            backend,
            image,
            width,
            |px| (px, 0.0),
            |py| (0.0, py),
            &format!("{what} identity"),
        );
        check_blit_parity(
            backend,
            image,
            2 * width,
            |px| (px / 2.0, 0.0),
            |py| (0.0, py / 2.0),
            &format!("{what} 2×"),
        );
    }
}

/// A 5×3 RGB image scaled into the 6×4 box `[3, 1, 9, 5)` of a 10×7 page:
/// the box's origin, extent and the image's sides differ between the axes,
/// so a kernel that exchanges x and y anywhere writes other pixels, or
/// none, and the pixels outside the box stay zero.
pub(crate) fn offset_box_tells_the_axes_apart<B: GpuBackend>(backend: &B) {
    let pixels = ramp(5 * 3 * 3, 13, 9, 251);
    let image = Image {
        pixels: &pixels,
        width: 5,
        height: 3,
        layout: 0,
    };
    check_blit_parity_in(
        backend,
        image,
        (10, 7),
        [3, 1, 9, 5],
        |px| ((px - 3.0) * 0.9, 0.0),
        |py| (0.0, (py - 1.0) * 0.8),
        "blit offset box",
    );
}
