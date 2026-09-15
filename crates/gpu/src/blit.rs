//! Image blit kernel dispatcher.
//!
//! Wraps `kernels/blit_image.cu`.  Copies a cached source image into a
//! [`DevicePageBuffer`] through caller-supplied sampling tables: the
//! source pixel for output `(dx, dy)` is `cols[dx] + rows[dy]` in Q32
//! fixed point.  The kernel performs no transform of its own, so the
//! caller that builds the tables (the interpreter's image sampler, which
//! also drives the CPU path) fully determines which source pixel every
//! output pixel reads.  Output pixels whose sample falls outside the
//! image are left untouched (the page buffer is zero-initialised, so
//! they composite as fully transparent).
//!
//! # Table layout
//!
//! One entry per bounding-box column (`cols`) and per row (`rows`), each
//! [`SAMPLE_ENTRY_WORDS`] `u32` words: `[x_hi, x_lo, y_hi, y_lo]`.  `hi`
//! words carry the two's-complement integer part of the term, `lo` words
//! the fraction; the kernel sums column and row terms with carry from the
//! fraction word.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::GpuCtx;
use crate::cache::{CachedDeviceImage, DevicePageBuffer, ImageLayout};

/// `u32` words per sampling-table entry: `[x_hi, x_lo, y_hi, y_lo]`.
pub const SAMPLE_ENTRY_WORDS: usize = 4;

/// Output bounding box for the blit kernel — only pixels within this
/// rectangle are dispatched, and the sampling tables cover exactly its
/// columns and rows.
#[derive(Debug, Clone, Copy)]
pub struct BlitBbox {
    /// Inclusive left edge (page pixels).
    pub x0: i32,
    /// Inclusive top edge (page pixels).
    pub y0: i32,
    /// Exclusive right edge (page pixels).
    pub x1: i32,
    /// Exclusive bottom edge (page pixels).
    pub y1: i32,
}

impl BlitBbox {
    /// Width in pixels, saturating to 0 for an inverted or zero-area
    /// bbox.  `saturating_sub` rules out the i32-overflow case that
    /// `(x1 - x0).max(0)` could panic on for adversarial inputs.
    #[must_use]
    pub fn width(self) -> u32 {
        u32::try_from(self.x1.saturating_sub(self.x0)).unwrap_or(0)
    }

    /// Height in pixels; same saturation rule as [`Self::width`].
    #[must_use]
    pub fn height(self) -> u32 {
        u32::try_from(self.y1.saturating_sub(self.y0)).unwrap_or(0)
    }

    /// Byte lengths the column and row tables must have for this bbox.
    #[must_use]
    pub fn table_bytes(self) -> (usize, usize) {
        let entry = SAMPLE_ENTRY_WORDS * std::mem::size_of::<u32>();
        (
            self.width() as usize * entry,
            self.height() as usize * entry,
        )
    }
}

impl GpuCtx {
    /// Dispatch the image-blit kernel: copy `src` into `dst` through the
    /// `cols`/`rows` sampling tables, writing only pixels inside `bbox`.
    ///
    /// `cols` must hold `bbox.width()` entries and `rows` `bbox.height()`
    /// entries of [`SAMPLE_ENTRY_WORDS`] words each.
    ///
    /// Mask-layout images aren't supported by this kernel — the caller
    /// must route them through the CPU path.
    ///
    /// # Errors
    /// - [`BlitError::UnsupportedLayout`] for `ImageLayout::Mask`.
    /// - [`BlitError::TableSize`] if a table doesn't match the bbox.
    /// - [`BlitError::DimensionsTooLarge`] if any dim doesn't fit i32.
    /// - [`BlitError::Cuda`] for any underlying CUDA failure (PTX
    ///   launch, table upload, etc.).
    pub fn blit_image_to_buffer(
        &self,
        src: &CachedDeviceImage,
        dst: &mut DevicePageBuffer,
        bbox: BlitBbox,
        cols: &[u32],
        rows: &[u32],
    ) -> Result<(), BlitError> {
        // Caller contract: bbox is non-inverted.  An inverted bbox
        // would silently render zero pixels (BlitBbox::width returns
        // 0 via saturating_sub), masking a programming error.
        debug_assert!(
            bbox.x0 <= bbox.x1 && bbox.y0 <= bbox.y1,
            "blit bbox is inverted: ({}..{}, {}..{})",
            bbox.x0,
            bbox.x1,
            bbox.y0,
            bbox.y1,
        );

        let layout_code: i32 = match src.layout {
            ImageLayout::Rgb => 0,
            ImageLayout::Gray => 1,
            ImageLayout::Mask => return Err(BlitError::UnsupportedLayout),
        };
        if bbox.width() == 0 || bbox.height() == 0 {
            return Ok(());
        }
        if cols.len() != bbox.width() as usize * SAMPLE_ENTRY_WORDS
            || rows.len() != bbox.height() as usize * SAMPLE_ENTRY_WORDS
        {
            return Err(BlitError::TableSize);
        }

        let stream = &self.stream;
        let d_cols = stream
            .clone_htod(bytemuck::cast_slice::<u32, u8>(cols))
            .map_err(BlitError::cuda)?;
        let d_rows = stream
            .clone_htod(bytemuck::cast_slice::<u32, u8>(rows))
            .map_err(BlitError::cuda)?;

        self.launch_blit_image_async(
            &src.device_ptr,
            (src.width, src.height),
            layout_code,
            &dst.rgba,
            (dst.width, dst.height),
            bbox,
            &d_cols,
            &d_rows,
        )
    }

    /// Async kernel launch for the image-blit kernel.
    ///
    /// This is the trait-facing variant: it takes raw device byte
    /// buffers and dimensions instead of the cache wrappers
    /// `CachedDeviceImage` / `DevicePageBuffer`, and the sampling tables
    /// already resident on the device (`d_cols` / `d_rows`, each the
    /// byte image of `u32` words as described in the module docs).  It
    /// does **not** synchronise and does **not** touch host memory; it
    /// is the helper `CudaBackend::record_blit_image` calls.
    ///
    /// The caller is responsible for keeping every device buffer alive
    /// until the stream has executed the launch.  cudarc frees a dropped
    /// `CudaSlice` in stream order, so dropping a table buffer right after
    /// this call is safe on the same stream.
    ///
    /// `layout_code` is the kernel's enum value: `0 = Rgb`, `1 = Gray`.
    /// Mask layout is rejected by the public wrapper before this
    /// helper is reached.
    ///
    /// # Errors
    /// - [`BlitError::DimensionsTooLarge`] if any dim doesn't fit i32.
    /// - [`BlitError::Cuda`] for any underlying cudarc failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "kernel arg count is fixed by the PTX signature; grouping into a struct would just hide the mapping"
    )]
    #[expect(
        unused_results,
        reason = "cudarc LaunchArgs::arg returns &mut Self for chaining; chain output is intentionally discarded"
    )]
    pub(crate) fn launch_blit_image_async(
        &self,
        d_src: &CudaSlice<u8>,
        src_dims: (u32, u32),
        layout_code: i32,
        d_dst: &CudaSlice<u8>,
        dst_dims: (u32, u32),
        bbox: BlitBbox,
        d_cols: &CudaSlice<u8>,
        d_rows: &CudaSlice<u8>,
    ) -> Result<(), BlitError> {
        // 16×16 blocks: 256 threads / block, full-warp aligned, fits
        // in any modern SM's register budget for this kernel.
        const TILE: u32 = 16;

        let bw = bbox.width();
        let bh = bbox.height();
        if bw == 0 || bh == 0 {
            return Ok(());
        }
        let (cols_bytes, rows_bytes) = bbox.table_bytes();
        if d_cols.len() < cols_bytes || d_rows.len() < rows_bytes {
            return Err(BlitError::TableSize);
        }

        let grid_x = bw.div_ceil(TILE);
        let grid_y = bh.div_ceil(TILE);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };

        let src_w = i32::try_from(src_dims.0).map_err(|_| BlitError::DimensionsTooLarge)?;
        let src_h = i32::try_from(src_dims.1).map_err(|_| BlitError::DimensionsTooLarge)?;
        let dst_w = i32::try_from(dst_dims.0).map_err(|_| BlitError::DimensionsTooLarge)?;
        let dst_h = i32::try_from(dst_dims.1).map_err(|_| BlitError::DimensionsTooLarge)?;

        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.kernels.blit_image);
        builder
            .arg(d_src)
            .arg(&src_w)
            .arg(&src_h)
            .arg(&layout_code)
            .arg(d_dst)
            .arg(&dst_w)
            .arg(&dst_h)
            .arg(&bbox.x0)
            .arg(&bbox.y0)
            .arg(&bbox.x1)
            .arg(&bbox.y1)
            .arg(d_cols)
            .arg(d_rows);

        // SAFETY: kernel signature in kernels/blit_image.cu matches
        // the argument list above; src/dst bounds are validated by the
        // checked casts on src_w/src_h/dst_w/dst_h and the table
        // buffers cover the bbox by the length check above.
        unsafe { builder.launch(cfg) }.map_err(BlitError::cuda)?;
        // Don't synchronise here — the caller decides when to read back.
        Ok(())
    }
}

/// Errors from [`GpuCtx::blit_image_to_buffer`].
#[derive(Debug)]
pub enum BlitError {
    /// `ImageLayout::Mask` — caller must route through the CPU path.
    UnsupportedLayout,
    /// A sampling table doesn't cover the bbox (`width × 4` / `height × 4`
    /// words).
    TableSize,
    /// Source or destination dimensions don't fit in `i32`.  Indicates
    /// a malformed image or page; should never happen at PDF-realistic
    /// resolutions (max u32 wide ≈ 4.3B px ≫ any real page).
    DimensionsTooLarge,
    /// Underlying cudarc driver error.
    Cuda(cudarc::driver::DriverError),
}

impl BlitError {
    const fn cuda(e: cudarc::driver::DriverError) -> Self {
        Self::Cuda(e)
    }
}

impl std::fmt::Display for BlitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedLayout => write!(f, "blit kernel does not handle Mask layout"),
            Self::TableSize => write!(f, "blit sampling tables do not match the bbox"),
            Self::DimensionsTooLarge => {
                write!(f, "image or page dimensions exceed i32::MAX")
            }
            Self::Cuda(e) => write!(f, "cuda: {e}"),
        }
    }
}

impl std::error::Error for BlitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cuda(e) => Some(e),
            Self::UnsupportedLayout | Self::TableSize | Self::DimensionsTooLarge => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_width_height_clamped_to_u32() {
        let b = BlitBbox {
            x0: 5,
            y0: 5,
            x1: 3,
            y1: 3,
        };
        assert_eq!(b.width(), 0);
        assert_eq!(b.height(), 0);
        assert_eq!(b.table_bytes(), (0, 0));
    }

    #[test]
    fn table_bytes_follow_bbox_extent() {
        let b = BlitBbox {
            x0: 2,
            y0: 1,
            x1: 7,
            y1: 4,
        };
        assert_eq!(b.table_bytes(), (5 * 16, 3 * 16));
    }

    /// Q32 split of one table term: `(hi as u32, lo)` with `hi = floor(v)`.
    #[cfg(feature = "gpu-validation")]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test builder: values are small; floor fits i32 and the fraction is in [0, 2^32)"
    )]
    fn split_q32(v: f64) -> (u32, u32) {
        let hi = v.floor();
        (
            (hi as i32) as u32,
            ((v - hi) * f64::from(1u32 << 31) * 2.0) as u32,
        )
    }

    /// Build sampling tables for the affine source map
    /// `X = x_col(dx) + x_row(dy)`, `Y = y_col(dx) + y_row(dy)` over
    /// `bbox`, sampling at pixel centres.
    #[cfg(feature = "gpu-validation")]
    fn build_tables(
        bbox: BlitBbox,
        col_terms: impl Fn(f64) -> (f64, f64),
        row_terms: impl Fn(f64) -> (f64, f64),
    ) -> (Vec<u32>, Vec<u32>) {
        let mut cols = Vec::new();
        for dx in bbox.x0..bbox.x1 {
            let (x, y) = col_terms(f64::from(dx) + 0.5);
            let (xh, xl) = split_q32(x);
            let (yh, yl) = split_q32(y);
            cols.extend_from_slice(&[xh, xl, yh, yl]);
        }
        let mut rows = Vec::new();
        for dy in bbox.y0..bbox.y1 {
            let (x, y) = row_terms(f64::from(dy) + 0.5);
            let (xh, xl) = split_q32(x);
            let (yh, yl) = split_q32(y);
            rows.extend_from_slice(&[xh, xl, yh, yl]);
        }
        (cols, rows)
    }

    /// Run the kernel on a `SRC×SRC` RGB image with the given source map
    /// and compare every page pixel against the f64 reference.
    #[cfg(feature = "gpu-validation")]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "test arithmetic on small constant dimensions; all values fit losslessly"
    )]
    fn check_kernel_against_reference(
        col_terms: impl Fn(f64) -> (f64, f64) + Copy,
        row_terms: impl Fn(f64) -> (f64, f64) + Copy,
    ) {
        use crate::cache::{DeviceImageCache, ImageLayout, InsertRequest};
        use crate::cache::{DocId, HostBudget, ObjId, VramBudget};
        use std::sync::Arc;

        const SRC: u32 = 4;
        const PAGE: u32 = 6;

        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let gpu = GpuCtx::init().expect("gpu init");

        // Distinct bytes per RGB channel so a one-pixel mismatch is
        // visible.  Step 7 stays away from any obvious row/col stride.
        let pixels: Vec<u8> = (0..SRC * SRC * 3)
            .map(|i| u8::try_from(((i * 7) + 3) % 251).expect("fits"))
            .collect();
        let h = DeviceImageCache::hash_bytes(&pixels);

        let cache = DeviceImageCache::new(
            Arc::clone(&stream),
            VramBudget {
                vram_bytes: 1 << 16,
            },
            HostBudget { host_bytes: 0 },
        );
        let cached = cache
            .insert(InsertRequest {
                doc: DocId([0; 32]),
                obj: ObjId(1),
                hash: h,
                width: SRC,
                height: SRC,
                layout: ImageLayout::Rgb,
                pixels: &pixels,
            })
            .expect("insert");

        let bbox = BlitBbox {
            x0: 0,
            y0: 0,
            x1: PAGE as i32,
            y1: PAGE as i32,
        };
        let (cols, rows) = build_tables(bbox, col_terms, row_terms);
        let mut page = DevicePageBuffer::new(Arc::clone(&stream), PAGE, PAGE).expect("page");
        gpu.blit_image_to_buffer(&cached, &mut page, bbox, &cols, &rows)
            .expect("blit");
        let host = page.download().expect("download");

        let mut compared = 0;
        for dy in 0..PAGE {
            for dx in 0..PAGE {
                let (xc, yc) = col_terms(f64::from(dx) + 0.5);
                let (xr, yr) = row_terms(f64::from(dy) + 0.5);
                let (x, y) = ((xc + xr).floor(), (yc + yr).floor());
                let off = ((dy * PAGE + dx) * 4) as usize;
                let alpha = host[off + 3];
                if x < 0.0 || y < 0.0 || x >= f64::from(SRC) || y >= f64::from(SRC) {
                    assert_eq!(alpha, 0, "alpha at ({dx},{dy}) — out-of-bounds sample");
                    continue;
                }
                let (ix, iy) = (x as u32, y as u32);
                assert_eq!(alpha, 255, "alpha at ({dx},{dy}) — in-bounds sample");
                let src_off = ((iy * SRC + ix) * 3) as usize;
                assert_eq!(
                    &host[off..off + 3],
                    &pixels[src_off..src_off + 3],
                    "RGB at ({dx},{dy}) (ix={ix}, iy={iy})",
                );
                compared += 1;
            }
        }
        // Sanity: at least some pixels must have been in-bounds.
        assert!(compared > 0, "test inputs degenerate; compared zero pixels");
    }

    /// Identity placement: page pixel `(dx, dy)` reads source `(dx, dy)`
    /// and the two page columns/rows past the image stay transparent.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn blit_kernel_identity_matches_reference() {
        check_kernel_against_reference(|px| (px, 0.0), |py| (0.0, py));
    }

    /// Fractional scale and offset in both axes, with a y-flip, so the
    /// per-pixel fraction-word carry and negative integer parts are
    /// exercised.
    #[cfg(feature = "gpu-validation")]
    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "test geometry: the source map is spelled as scale-then-offset for readability"
    )]
    fn blit_kernel_scaled_offset_matches_reference() {
        check_kernel_against_reference(
            |px| ((px - 1.3) * 0.7, 0.0),
            |py| (0.0, 4.0 - (py - 0.6) * 0.9),
        );
    }

    /// A rotated placement: the column term contributes to `Y` and the row
    /// term to `X`, and bbox corners fall outside the image.
    #[cfg(feature = "gpu-validation")]
    #[test]
    #[expect(
        clippy::suboptimal_flops,
        reason = "test geometry: the source map is spelled as rotate-then-offset for readability"
    )]
    fn blit_kernel_rotated_matches_reference() {
        check_kernel_against_reference(
            |px| ((px - 3.0) * 0.6, (px - 3.0) * 0.4),
            |py| ((py - 3.0) * -0.4 + 2.0, (py - 3.0) * 0.6 + 2.0),
        );
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn blit_rejects_mismatched_tables() {
        use crate::cache::{DeviceImageCache, ImageLayout, InsertRequest};
        use crate::cache::{DocId, HostBudget, ObjId, VramBudget};
        use std::sync::Arc;

        let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.default_stream();
        let gpu = GpuCtx::init().expect("gpu init");
        let pixels = vec![0u8; 4 * 4 * 3];
        let cache = DeviceImageCache::new(
            Arc::clone(&stream),
            VramBudget {
                vram_bytes: 1 << 16,
            },
            HostBudget { host_bytes: 0 },
        );
        let cached = cache
            .insert(InsertRequest {
                doc: DocId([0; 32]),
                obj: ObjId(1),
                hash: DeviceImageCache::hash_bytes(&pixels),
                width: 4,
                height: 4,
                layout: ImageLayout::Rgb,
                pixels: &pixels,
            })
            .expect("insert");
        let bbox = BlitBbox {
            x0: 0,
            y0: 0,
            x1: 4,
            y1: 4,
        };
        let mut page = DevicePageBuffer::new(Arc::clone(&stream), 4, 4).expect("page");
        let err = gpu
            .blit_image_to_buffer(&cached, &mut page, bbox, &[0; 8], &[0; 16])
            .unwrap_err();
        assert!(matches!(err, BlitError::TableSize), "{err}");
    }
}
