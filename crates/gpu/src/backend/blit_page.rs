//! The image blit as one backend page, into a target the size of the
//! blit's bounding box.
//!
//! The blit kernel writes output pixel `(dx, dy)` at index
//! `dy · dst_w + dx` of its destination and reads the sampling tables at
//! `dx − x0` and `dy − y0`.  Dispatching it over the bounding box moved to
//! the origin, `[0, w) × [0, h)`, with `dst_w = w` and `dst_h = h`, reads
//! the same table entries for every pixel and writes the pixel of page
//! position `(x0 + i, y0 + j)` at index `j · w + i`: the destination holds
//! exactly the bounding box, row-major.  The page bounds the kernel
//! checks are those of the box, which a caller's box lies within.

use super::params::{BlitParams, blit_extents};
use super::staged::{Direction, Staged, ensure, host_view, record_download, record_upload, stage};
use super::{BackendError, GpuBackend, Result};
use crate::RGBA_BPP;

/// The source of a blit: a device buffer of `width × height` samples in
/// the layout `layout` ([`BlitParams::src_layout`]).
pub struct BlitSource<'a, B: GpuBackend> {
    /// The samples, row-major, in a buffer of at least `width × height ×
    /// bpp` bytes rounded up to a multiple of 4
    /// ([`super::layout::word_padded`], [`BlitParams::src`]).
    pub pixels: &'a B::DeviceBuffer,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
    /// `0` for RGB, `1` for gray.
    pub layout: u32,
}

/// Reusable host and device buffers for [`blit_image_box`].
///
/// The two sampling tables and the RGBA8 target are each grown to the
/// largest blit seen and reused by later blits.  The buffers are freed when
/// the scratch is dropped; [`blit_image_box`] returns only after its page has
/// completed, so no device work references them then.  [`blit_image`] lends
/// one from the backend's pool ([`GpuBackend::scratch`]).
pub struct BlitScratch<B: GpuBackend> {
    cols: Option<Staged<B>>,
    rows: Option<Staged<B>>,
    target: Option<Staged<B>>,
}

/// An empty scratch; the first blit allocates.
impl<B: GpuBackend> Default for BlitScratch<B> {
    fn default() -> Self {
        Self {
            cols: None,
            rows: None,
            target: None,
        }
    }
}

/// [`blit_image_box`] with a scratch lent by `backend`'s pool
/// ([`GpuBackend::scratch`]): `consume` receives the box's RGBA8 rows and
/// its result is returned.
///
/// # Errors
/// As [`blit_image_box`].
pub fn blit_image<B: GpuBackend, R>(
    backend: &B,
    src: &BlitSource<'_, B>,
    bbox: [i32; 4],
    cols: &[u32],
    rows: &[u32],
    filter: u32,
    consume: impl FnOnce(&[u8]) -> R,
) -> Result<R> {
    backend.scratch().blit.with(|scratch| {
        blit_image_box(backend, scratch, src, bbox, cols, rows, filter).map(consume)
    })
}

/// The pixels of the bounding box `bbox` blitted from `src`, as RGBA8 rows.
///
/// `bbox = [x0, y0, x1, y1]` is in page pixels, `x0 <= x1`, `y0 <= y1`; the
/// samples are read through the tables `cols` (one entry per box column)
/// and `rows` (one per box row) with the `filter` bits, as [`BlitParams`]
/// describes.  The result is `(x1 − x0) × (y1 − y0)`
/// RGBA8 pixels, row-major, alpha `255` where the kernel wrote a sample and
/// all-zero elsewhere.  An empty box yields an empty slice without a device
/// round trip.
///
/// The tables are written into `scratch`'s host buffers, and their uploads,
/// the zero-fill of the target, the kernel and the download of the target
/// are recorded into one page ([`GpuBackend::run_page`]).  On a device
/// whose buffers the host reaches in place the tables are written into the
/// device buffers and the target read from them, with no copies.
///
/// # Errors
/// [`BackendError::InvariantViolation`] when a table does not cover the
/// box, the box or the source exceeds the kernel's 32-bit offsets, or
/// [`BlitParams::validate`] rejects the blit; the backend's error when an
/// allocation or a device call fails.
pub fn blit_image_box<'s, B: GpuBackend>(
    backend: &B,
    scratch: &'s mut BlitScratch<B>,
    src: &BlitSource<'_, B>,
    bbox: [i32; 4],
    cols: &[u32],
    rows: &[u32],
    filter: u32,
) -> Result<&'s [u8]> {
    const KIND: &str = "BlitInvariantViolation";
    let invariant = |detail: &'static str| BackendError::invariant(KIND, detail);
    let [x0, y0, x1, y1] = bbox;
    if x0 > x1 || y0 > y1 {
        return Err(invariant("bbox must satisfy x0 <= x1 and y0 <= y1"));
    }
    let width = x1.abs_diff(x0);
    let height = y1.abs_diff(y0);
    if width == 0 || height == 0 {
        return Ok(&[]);
    }
    let entry_words = BlitParams::<B>::TABLE_ENTRY_BYTES / size_of::<u32>();
    let (cols_need, rows_need) = (width as usize * entry_words, height as usize * entry_words);
    if cols.len() != cols_need || rows.len() != rows_need {
        return Err(BackendError::invariant(
            KIND,
            format!(
                "sampling tables of {} and {} words do not cover the {width} × {height} bbox \
                 ({cols_need} and {rows_need} words)",
                cols.len(),
                rows.len()
            ),
        ));
    }
    let (_, target_len) = blit_extents((src.width, src.height), src.layout, (width, height))
        .ok_or_else(|| invariant("src or bbox exceeds the kernels' 32-bit offset range"))?;
    let box_end = |n: u32| {
        i32::try_from(n).map_err(|_| invariant("bbox exceeds the kernels' 32-bit offsets"))
    };
    let (box_w, box_h) = (box_end(width)?, box_end(height)?);
    debug_assert_eq!(target_len, width as usize * height as usize * RGBA_BPP);

    let col_bytes: &[u8] = bytemuck::cast_slice(cols);
    let row_bytes: &[u8] = bytemuck::cast_slice(rows);
    let BlitScratch {
        cols: cols_slot,
        rows: rows_slot,
        target: target_slot,
    } = scratch;
    let d_cols = stage(backend, cols_slot, col_bytes)?;
    let d_rows = stage(backend, rows_slot, row_bytes)?;
    let target = ensure(backend, target_slot, target_len, Direction::ToHost)?;
    // SAFETY (the recorded copies and the in-place read below): the
    // scratch's buffers are borrowed mutably for this whole call and
    // `run_page` returns only once its page has completed or, on failure,
    // been waited on; nothing touches them in between.
    backend.run_page(|backend| {
        unsafe { record_upload(backend, d_cols, col_bytes.len()) }?;
        unsafe { record_upload(backend, d_rows, row_bytes.len()) }?;
        backend.record_zero_buffer(&mut target.device, target_len)?;
        backend.record_blit_image(BlitParams {
            src: src.pixels,
            dst: &mut target.device,
            cols: &d_cols.device,
            rows: &d_rows.device,
            src_w: src.width,
            src_h: src.height,
            src_layout: src.layout,
            dst_w: width,
            dst_h: height,
            bbox: [0, 0, box_w, box_h],
            filter,
        })?;
        unsafe { record_download(backend, target, target_len) }
    })?;
    let pixels = unsafe { host_view(backend, target) }?;
    Ok(&pixels[..target_len])
}

#[cfg(all(test, any(feature = "gpu-validation", feature = "vulkan")))]
mod tests {
    use super::super::layout::word_padded;
    use super::*;

    /// Sampling-table entry of the integer coordinate `v` along x (`x`
    /// true) or y, sampled at the pixel centre.
    const fn entry(v: i32, x: bool) -> [u32; 4] {
        let (hi, lo) = (v.cast_unsigned(), 1u32 << 31);
        if x { [hi, lo, 0, 0] } else { [0, 0, hi, lo] }
    }

    /// Blit a 3×2 RGB image, one sample per pixel, into the box
    /// `[1, 1, 5, 4)` whose columns map to source columns `-1..3` and rows
    /// to source rows `0..3`, and check every target pixel: the samples
    /// inside the image with alpha 255, zero outside it.  A larger box is
    /// blitted first and the box twice, so each check sees a target an
    /// earlier blit wrote beyond the box's length.
    fn check_box_blit<B: GpuBackend>(backend: &B) {
        let image: Vec<u8> = (0..3 * 2 * 3).map(|i| 10 + i).collect();
        let mut src_buf = backend
            .alloc_device(word_padded(image.len()))
            .expect("alloc");
        backend.upload(&mut src_buf, &image).expect("upload");
        let src = BlitSource::<B> {
            pixels: &src_buf,
            width: 3,
            height: 2,
            layout: 0,
        };
        let cols: Vec<u32> = (-1..3).flat_map(|x| entry(x, true)).collect();
        let rows: Vec<u32> = (0..3).flat_map(|y| entry(y, false)).collect();
        let mut scratch = BlitScratch::default();
        let wide_cols: Vec<u32> = (0..8).flat_map(|x| entry(x % 3, true)).collect();
        let wide_rows: Vec<u32> = (0..8).flat_map(|y| entry(y % 2, false)).collect();
        let wide = blit_image_box(
            backend,
            &mut scratch,
            &src,
            [0, 0, 8, 8],
            &wide_cols,
            &wide_rows,
            0,
        )
        .expect("wide blit");
        assert_eq!(wide.len(), 8 * 8 * RGBA_BPP);
        for _ in 0..2 {
            let out = blit_image_box(backend, &mut scratch, &src, [1, 1, 5, 4], &cols, &rows, 0)
                .expect("blit")
                .to_vec();
            assert_eq!(out.len(), 4 * 3 * RGBA_BPP);
            for (sy, row) in (0_usize..).zip(out.as_chunks::<{ 4 * RGBA_BPP }>().0) {
                for (sx, px) in (-1_isize..).zip(row.as_chunks::<RGBA_BPP>().0) {
                    let expected = match usize::try_from(sx) {
                        Ok(sx) if sx < 3 && sy < 2 => {
                            let o = (sy * 3 + sx) * 3;
                            [image[o], image[o + 1], image[o + 2], 255]
                        }
                        _ => [0; 4],
                    };
                    assert_eq!(*px, expected, "source pixel ({sx}, {sy})");
                }
            }
        }
        let empty = blit_image_box(backend, &mut scratch, &src, [2, 2, 2, 5], &[], &[], 0)
            .expect("empty box");
        assert_eq!(empty, &[0_u8; 0][..]);
    }

    /// [`blit_image`] hands the box to its consumer and returns its scratch
    /// to the backend's pool, buffers kept.
    fn check_pooled_blit<B: GpuBackend>(backend: &B) {
        let image = [7_u8, 8, 9];
        let mut src_buf = backend
            .alloc_device(word_padded(image.len()))
            .expect("alloc");
        backend.upload(&mut src_buf, &image).expect("upload");
        let src = BlitSource::<B> {
            pixels: &src_buf,
            width: 1,
            height: 1,
            layout: 0,
        };
        for _ in 0..2 {
            let px = blit_image(
                backend,
                &src,
                [0, 0, 1, 1],
                &entry(0, true),
                &entry(0, false),
                0,
                <[u8]>::to_vec,
            )
            .expect("blit");
            assert_eq!(px, [7, 8, 9, 255]);
            assert_eq!(backend.scratch().blit.idle_len(), 1);
        }
        assert!(backend.scratch().blit.take_idle().target.is_some());
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn cuda_box_blit_writes_the_box() {
        let backend = crate::backend::cuda::CudaBackend::new().expect("CUDA device 0");
        check_box_blit(&backend);
        check_pooled_blit(&backend);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_box_blit_writes_the_box() {
        if let Some(backend) = crate::backend::vulkan::test_backend() {
            check_box_blit(&backend);
            check_pooled_blit(&backend);
        }
    }
}
