//! Parameter structs for `GpuBackend::record_*` calls.
//!
//! Each struct holds references (not owned values) to caller-owned device
//! buffers and small scalar args. Backends resolve buffer references to
//! their native pointer/handle type at record time.
//!
//! A buffer the kernel writes in any phase is held by `&mut`, one it only
//! reads by `&`, so a backend that orders work across queues by recorded
//! reads and writes (the CUDA backend's per-buffer events) sees every
//! write.

use super::GpuBackend;

/// [`BlitParams::filter`] bit: interpolate between source columns.
pub const BLIT_FILTER_X: u32 = 1;
/// [`BlitParams::filter`] bit: interpolate between source rows.
pub const BLIT_FILTER_Y: u32 = 2;

/// Byte lengths `(source, destination)` a blit addresses.
///
/// `src` and `dst` are (width, height); the destination is RGBA8.  `None`
/// when the source exceeds `i32::MAX` bytes or the destination `i32::MAX`
/// pixels.  The blit kernels compute source byte offsets and destination pixel
/// indices in 32-bit signed arithmetic; within these bounds neither can
/// wrap.  `src_layout` `0` is RGB (3 bytes per pixel), any other value
/// Gray (1 byte per pixel).
#[must_use]
pub fn blit_extents(src: (u32, u32), src_layout: u32, dst: (u32, u32)) -> Option<(usize, usize)> {
    let bytes_per_pixel = if src_layout == 0 { 3 } else { 1 };
    let src_bytes = (src.0 as usize)
        .checked_mul(src.1 as usize)?
        .checked_mul(bytes_per_pixel)?;
    let dst_pixels = (dst.0 as usize).checked_mul(dst.1 as usize)?;
    let limit = i32::MAX as usize;
    (src_bytes <= limit && dst_pixels <= limit).then_some((src_bytes, dst_pixels * 4))
}

/// Parameters for a GPU image blit (table-driven sampling onto the page
/// buffer).
///
/// The kernel samples the source at `cols[dx] + rows[dy]` (Q32 fixed point,
/// four `u32` words per entry: `[x_hi, x_lo, y_hi, y_lo]`) for every page
/// pixel inside `bbox`, by nearest neighbour or, on the axes `filter`
/// selects, bilinearly; see `crate::blit` for the table layout.
///
/// # Invariants enforced by `BlitParams::validate`
/// - `src_w > 0`, `src_h > 0`, `dst_w > 0`, `dst_h > 0`
/// - `src_layout` is `0` (RGB, 3 bytes/pixel) or `1` (Gray, 1 byte/pixel).
///   Mask images (layout `2`) are CPU-only and must not reach the GPU.
/// - `bbox` is `[x0, y0, x1, y1]` with `x0 <= x1` and `y0 <= y1`
/// - `cols` holds at least `(x1 - x0)` entries and `rows` at least
///   `(y1 - y0)` entries of 16 bytes each
/// - `filter` has no bits beyond [`BLIT_FILTER_X`] | [`BLIT_FILTER_Y`]
/// - the source and destination fit the kernels' 32-bit offsets and `src`
///   and `dst` hold the bytes they address (see [`blit_extents`])
///
/// Backends should call `validate()` at record time. Violating these
/// invariants would read past a sampling table, route Mask images through
/// the Gray code path silently, or produce garbage pixels (or panics in
/// debug builds via `debug_assert`).
pub struct BlitParams<'a, B: GpuBackend + ?Sized> {
    /// Source image in device memory.
    pub src: &'a B::DeviceBuffer,
    /// Destination page buffer in device memory.
    pub dst: &'a mut B::DeviceBuffer,
    /// Per-column sampling table in device memory (`bbox` width entries).
    pub cols: &'a B::DeviceBuffer,
    /// Per-row sampling table in device memory (`bbox` height entries).
    pub rows: &'a B::DeviceBuffer,
    /// Source image width in pixels (must be > 0).
    pub src_w: u32,
    /// Source image height in pixels (must be > 0).
    pub src_h: u32,
    /// Source image channel layout: `0` = RGB, `1` = Gray. Other values
    /// are rejected by `validate()`.
    pub src_layout: u32,
    /// Destination buffer width in pixels (must be > 0).
    pub dst_w: u32,
    /// Destination buffer height in pixels (must be > 0).
    pub dst_h: u32,
    /// Destination bounding box `[x0, y0, x1, y1]` in page-space pixels (`x0 <= x1`, `y0 <= y1`).
    pub bbox: [i32; 4],
    /// Interpolated axes: [`BLIT_FILTER_X`] and/or [`BLIT_FILTER_Y`].
    pub filter: u32,
}

impl<B: GpuBackend + ?Sized> BlitParams<'_, B> {
    /// Bytes per sampling-table entry: four `u32` words.
    pub const TABLE_ENTRY_BYTES: usize = 16;

    /// Validate the invariants documented on `BlitParams`.
    ///
    /// Returns a [`super::BackendError`] describing the first violated
    /// invariant. Backends should call this at the top of `record_blit_image`
    /// so misuse fails loudly with a clear message rather than producing
    /// undefined kernel behaviour.
    ///
    /// # Errors
    /// Returns a `BackendError` if any documented invariant is violated.
    pub fn validate(&self, backend: &B) -> super::Result<()> {
        const KIND: &str = "BlitInvariantViolation";
        let invariant =
            |detail: &'static str| super::BackendError::InvariantViolation { kind: KIND, detail };
        if self.src_w == 0 || self.src_h == 0 {
            return Err(invariant("src dimensions must be > 0"));
        }
        if self.dst_w == 0 || self.dst_h == 0 {
            return Err(invariant("dst dimensions must be > 0"));
        }
        let [x0, y0, x1, y1] = self.bbox;
        if x0 > x1 || y0 > y1 {
            return Err(invariant("bbox must satisfy x0 <= x1 and y0 <= y1"));
        }
        // Table coverage: the kernel indexes `cols` by `dx - x0` and `rows`
        // by `dy - y0` without a bounds check of its own.
        let width = usize::try_from(x1.saturating_sub(x0)).unwrap_or(0);
        let height = usize::try_from(y1.saturating_sub(y0)).unwrap_or(0);
        if backend.device_buffer_len(self.cols) < width.saturating_mul(Self::TABLE_ENTRY_BYTES) {
            return Err(invariant("cols table shorter than bbox width × 16 bytes"));
        }
        if backend.device_buffer_len(self.rows) < height.saturating_mul(Self::TABLE_ENTRY_BYTES) {
            return Err(invariant("rows table shorter than bbox height × 16 bytes"));
        }
        // src_layout: 0 = RGB, 1 = Gray. Anything else (especially 2 = Mask)
        // would fall through the kernel's `if (src_layout == 0) { RGB } else { Gray }`
        // dispatch and silently produce wrong pixels.
        if self.src_layout > 1 {
            return Err(invariant(
                "src_layout must be 0 (RGB) or 1 (Gray); Mask layout is CPU-only",
            ));
        }
        if self.filter & !(BLIT_FILTER_X | BLIT_FILTER_Y) != 0 {
            return Err(invariant(
                "filter has bits beyond BLIT_FILTER_X | BLIT_FILTER_Y",
            ));
        }
        let (src_bytes, dst_bytes) = blit_extents(
            (self.src_w, self.src_h),
            self.src_layout,
            (self.dst_w, self.dst_h),
        )
        .ok_or_else(|| invariant("src or dst exceeds the kernels' 32-bit offset range"))?;
        if backend.device_buffer_len(self.src) < src_bytes {
            return Err(invariant("src buffer shorter than src_w × src_h × bpp"));
        }
        if backend.device_buffer_len(self.dst) < dst_bytes {
            return Err(invariant("dst buffer shorter than dst_w × dst_h × 4"));
        }
        Ok(())
    }
}

/// Largest pixel count a per-pixel kernel accepts: the kernels address
/// pixel `i` at byte `i × 4` in 32-bit arithmetic, which stays exact only
/// below 2³⁰ pixels.
pub const MAX_KERNEL_PIXELS: u32 = u32::MAX / 4;

/// Check that `n_pixels` is addressable by a per-pixel kernel and that each
/// `(buffer, bytes per pixel, message)` holds `n_pixels × bytes` bytes.
fn validate_per_pixel<B: GpuBackend + ?Sized>(
    backend: &B,
    kind: &'static str,
    n_pixels: u32,
    buffers: &[(&B::DeviceBuffer, usize, &'static str)],
) -> super::Result<()> {
    let invariant = |detail: &'static str| super::BackendError::InvariantViolation { kind, detail };
    if n_pixels > MAX_KERNEL_PIXELS {
        return Err(invariant(
            "n_pixels exceeds MAX_KERNEL_PIXELS (32-bit byte addressing)",
        ));
    }
    for &(buf, bytes_per_pixel, detail) in buffers {
        if backend.device_buffer_len(buf) < (n_pixels as usize).saturating_mul(bytes_per_pixel) {
            return Err(invariant(detail));
        }
    }
    Ok(())
}

/// Parameters for a GPU ICC CMYK→RGB conversion through a grid
/// ([`color::grid::CmykGrid`]).
///
/// The matrix fast-path is handled on the CPU (AVX-512); only the grid
/// kernel runs on the GPU, so `clut` is required.
///
/// # Invariants enforced by `IccClutParams::validate`
/// - `n_pixels` ≤ [`MAX_KERNEL_PIXELS`], and with `wide` at most
///   [`MAX_WIDE_CLUT_PIXELS`]
/// - `cmyk` holds at least `n_pixels × 4` bytes and `rgb` at least
///   `n_pixels × 4`, `n_pixels × 8` with `wide`
/// - `clut` is as long as a grid of some number of nodes
///   ([`color::grid::CmykGrid::node_count`]), at most `i32::MAX` bytes;
///   the kernels read no node beyond that number, whatever the buffer
///   holds
pub struct IccClutParams<'a, B: GpuBackend + ?Sized> {
    /// Input CMYK pixels in device memory, one 32-bit word each:
    /// `c | m << 8 | y << 16 | k << 24`, the four bytes in order.
    pub cmyk: &'a B::DeviceBuffer,
    /// Output RGB pixels in device memory, whole 32-bit words: one word
    /// `r | g << 8 | b << 16` a pixel, or with `wide` two, `r | g << 16`
    /// then `b`.
    pub rgb: &'a mut B::DeviceBuffer,
    /// The grid's bytes ([`color::grid::CmykGrid::as_bytes`]) in device
    /// memory.
    pub clut: &'a B::DeviceBuffer,
    /// Number of pixels to transform.
    pub n_pixels: u32,
    /// Whether `rgb` receives the 16-bit outputs
    /// ([`color::grid::CmykGrid::convert16`]) rather than the 8-bit ones
    /// ([`color::grid::CmykGrid::convert`]).
    pub wide: bool,
}

/// Largest pixel count of a 16-bit [`IccClutParams`] conversion: the
/// output of pixel `i` lies at bytes `[i × 8, i × 8 + 8)`, addressed in
/// 32-bit arithmetic.
pub const MAX_WIDE_CLUT_PIXELS: u32 = u32::MAX / 8;

impl<B: GpuBackend + ?Sized> IccClutParams<'_, B> {
    /// Check the invariants documented on `IccClutParams` and return the
    /// grid's node count.
    ///
    /// # Errors
    /// Returns a `BackendError` if any documented invariant is violated.
    pub fn validate(&self, backend: &B) -> super::Result<u32> {
        const KIND: &str = "IccClutInvariantViolation";
        if self.wide && self.n_pixels > MAX_WIDE_CLUT_PIXELS {
            return Err(super::BackendError::InvariantViolation {
                kind: KIND,
                detail: "n_pixels exceeds MAX_WIDE_CLUT_PIXELS (32-bit byte addressing)",
            });
        }
        let (rgb_bytes, rgb_short) = if self.wide {
            (
                4 * super::layout::device_pixel_words(2),
                "rgb buffer shorter than n_pixels × 8 bytes",
            )
        } else {
            (
                4 * super::layout::device_pixel_words(1),
                "rgb buffer shorter than n_pixels × 4 bytes",
            )
        };
        validate_per_pixel(
            backend,
            KIND,
            self.n_pixels,
            &[
                (self.cmyk, 4, "cmyk buffer shorter than n_pixels × 4 bytes"),
                (&*self.rgb, rgb_bytes, rgb_short),
            ],
        )?;
        color::grid::CmykGrid::node_count(backend.device_buffer_len(self.clut)).ok_or(
            super::BackendError::InvariantViolation {
                kind: KIND,
                detail: "clut length is not that of a grid of at most i32::MAX bytes",
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phantom backend used solely for validating param-struct invariants in
    /// pure-CPU tests (no cudarc required).
    ///
    /// `DeviceBuffer = usize` — the buffer "is" its capacity in bytes,
    /// so `device_buffer_len` is a trivial identity. Most other trait
    /// methods are `unreachable!()` because validate-only tests don't
    /// route through them.
    struct FakeBackend;

    impl super::super::sealed::Sealed for FakeBackend {}

    impl GpuBackend for FakeBackend {
        const NAME: &'static str = "fake";
        type DeviceBuffer = usize;
        type HostBuffer = ();
        type PageFence = ();

        fn alloc_device(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!("FakeBackend allocation paths are not exercised by validate-only tests")
        }
        fn alloc_device_readback(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!()
        }
        fn free_device(&self, _buf: Self::DeviceBuffer) {}
        fn alloc_host_pinned(&self, _size: usize) -> super::super::Result<Self::HostBuffer> {
            unreachable!()
        }
        fn alloc_host_readback(&self, _size: usize) -> super::super::Result<Self::HostBuffer> {
            unreachable!()
        }
        fn free_host_pinned(&self, _buf: Self::HostBuffer) {}
        fn begin_page(&self) -> super::super::Result<()> {
            Ok(())
        }
        fn abandon_page(&self) {
            ABANDONED.set(ABANDONED.get() + 1);
        }
        fn record_blit_image(&self, _p: BlitParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_icc_clut(&self, _p: IccClutParams<'_, Self>) -> super::super::Result<()> {
            unreachable!()
        }
        fn record_zero_buffer(
            &self,
            _buf: &mut Self::DeviceBuffer,
            _len: usize,
        ) -> super::super::Result<()> {
            unreachable!()
        }
        unsafe fn record_copy_to_device(
            &self,
            _src: &Self::HostBuffer,
            _dst: &mut Self::DeviceBuffer,
            _len: usize,
        ) -> super::super::Result<()> {
            unreachable!()
        }
        unsafe fn record_copy_to_host(
            &self,
            _src: &Self::DeviceBuffer,
            _dst: &mut Self::HostBuffer,
            _len: usize,
        ) -> super::super::Result<()> {
            unreachable!()
        }
        fn device_memory_is_host_mapped(&self) -> bool {
            unreachable!()
        }
        unsafe fn device_bytes<'a>(&self, _buf: &'a Self::DeviceBuffer) -> Option<&'a [u8]> {
            unreachable!()
        }
        unsafe fn device_bytes_mut<'a>(
            &self,
            _buf: &'a mut Self::DeviceBuffer,
        ) -> Option<&'a mut [u8]> {
            unreachable!()
        }
        fn host_buffer_len(&self, _buf: &Self::HostBuffer) -> usize {
            unreachable!()
        }
        fn host_bytes<'a>(&self, _buf: &'a Self::HostBuffer) -> super::super::Result<&'a [u8]> {
            unreachable!()
        }
        fn host_bytes_mut<'a>(
            &self,
            _buf: &'a mut Self::HostBuffer,
        ) -> super::super::Result<&'a mut [u8]> {
            unreachable!()
        }
        fn submit_page(&self) -> super::super::Result<Self::PageFence> {
            Ok(())
        }
        fn wait_page(&self, _fence: Self::PageFence) -> super::super::Result<()> {
            Ok(())
        }
        fn upload(&self, _dst: &mut Self::DeviceBuffer, _src: &[u8]) -> super::super::Result<()> {
            unreachable!()
        }
        fn alloc_device_zeroed(&self, _size: usize) -> super::super::Result<Self::DeviceBuffer> {
            unreachable!()
        }
        fn device_buffer_len(&self, buf: &Self::DeviceBuffer) -> usize {
            *buf
        }
        fn download(&self, _src: &Self::DeviceBuffer, _dst: &mut [u8]) -> super::super::Result<()> {
            unreachable!()
        }
        fn detect_vram_budget(&self) -> super::super::Result<super::super::VramBudget> {
            unreachable!()
        }
        fn scratch(&self) -> &super::super::DeviceScratch<Self> {
            unreachable!()
        }
    }

    thread_local! {
        /// [`FakeBackend::abandon_page`] calls made on this thread.
        static ABANDONED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn default_run_page_abandons_a_failed_or_unwinding_page() {
        let before = ABANDONED.get();
        FakeBackend.run_page(|_| Ok(())).expect("empty page");
        assert_eq!(ABANDONED.get(), before, "a completed page was abandoned");
        let err = FakeBackend
            .run_page(|_| Err(super::super::BackendError::msg("injected")))
            .expect_err("the record error is returned");
        assert_eq!(err.to_string(), "injected");
        assert_eq!(
            ABANDONED.get(),
            before + 1,
            "a failed page was not abandoned"
        );
        let unwound = std::panic::catch_unwind(|| {
            FakeBackend.run_page(|_| -> super::super::Result<()> { panic!("page closure") })
        });
        assert!(unwound.is_err());
        assert_eq!(
            ABANDONED.get(),
            before + 2,
            "an unwinding page was not abandoned"
        );
    }

    /// Static stand-in for a device buffer of "large enough" capacity.
    /// The numeric value is only consulted by `device_buffer_len`.
    static FAKE_BUF: usize = usize::MAX;

    /// A "large enough" fake buffer a kernel writes: a leaked unique
    /// reference, as the parameter structs take written buffers by `&mut`.
    fn fake_out() -> &'static mut usize {
        fake_out_of(FAKE_BUF)
    }

    /// A fake written buffer of `len` bytes, as [`fake_out`].
    fn fake_out_of(len: usize) -> &'static mut usize {
        Box::leak(Box::new(len))
    }

    /// Sampling tables sized exactly for `ok_blit`'s 100×100 bbox.
    static BLIT_TABLE: usize = 100 * 16;

    fn ok_blit() -> BlitParams<'static, FakeBackend> {
        BlitParams {
            src: &FAKE_BUF,
            dst: fake_out(),
            cols: &BLIT_TABLE,
            rows: &BLIT_TABLE,
            src_w: 100,
            src_h: 100,
            src_layout: 0,
            dst_w: 200,
            dst_h: 200,
            bbox: [0, 0, 100, 100],
            filter: BLIT_FILTER_X | BLIT_FILTER_Y,
        }
    }

    #[test]
    fn validate_accepts_valid_blit_params() {
        ok_blit()
            .validate(&FakeBackend)
            .expect("valid params should pass");
    }

    #[test]
    fn validate_rejects_zero_src_dim() {
        let mut p = ok_blit();
        p.src_w = 0;
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("src dimensions"), "{err}");
    }

    #[test]
    fn validate_rejects_zero_dst_dim() {
        let mut p = ok_blit();
        p.dst_h = 0;
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("dst dimensions"), "{err}");
    }

    #[test]
    fn validate_rejects_inverted_bbox() {
        let mut p = ok_blit();
        p.bbox = [50, 0, 10, 100]; // x0 > x1
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("bbox"), "{err}");
    }

    #[test]
    fn validate_rejects_short_cols_table() {
        let mut p = ok_blit();
        p.bbox = [0, 0, 101, 100];
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("cols table"), "{err}");
    }

    #[test]
    fn validate_rejects_short_rows_table() {
        let mut p = ok_blit();
        p.bbox = [0, 0, 100, 101];
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("rows table"), "{err}");
    }

    #[test]
    fn validate_accepts_empty_bbox_with_empty_tables() {
        static EMPTY: usize = 0;
        let mut p = ok_blit();
        p.bbox = [10, 10, 10, 10];
        p.cols = &EMPTY;
        p.rows = &EMPTY;
        p.validate(&FakeBackend)
            .expect("an empty bbox needs no table entries");
    }

    #[test]
    fn validate_accepts_layout_0_rgb() {
        let mut p = ok_blit();
        p.src_layout = 0;
        p.validate(&FakeBackend).expect("layout 0 (RGB) is valid");
    }

    #[test]
    fn validate_accepts_layout_1_gray() {
        let mut p = ok_blit();
        p.src_layout = 1;
        p.validate(&FakeBackend).expect("layout 1 (Gray) is valid");
    }

    #[test]
    fn validate_rejects_layout_2_mask() {
        let mut p = ok_blit();
        p.src_layout = 2;
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("Mask"), "{err}");
        assert!(err.contains("CPU-only"), "{err}");
    }

    #[test]
    fn validate_rejects_unknown_layout() {
        let mut p = ok_blit();
        p.src_layout = 999;
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("src_layout"), "{err}");
    }

    #[test]
    fn validate_rejects_unknown_filter_bits() {
        let mut p = ok_blit();
        p.filter = 4;
        let err = p.validate(&FakeBackend).unwrap_err().to_string();
        assert!(err.contains("filter"), "{err}");
    }

    #[test]
    fn validate_rejects_short_src_and_dst_buffers() {
        static SRC_SHORT: usize = 100 * 100 * 3 - 1;
        let mut dst_short: usize = 200 * 200 * 4 - 1;
        let err = BlitParams {
            src: &SRC_SHORT,
            ..ok_blit()
        }
        .validate(&FakeBackend)
        .unwrap_err()
        .to_string();
        assert!(err.contains("src buffer"), "{err}");
        let err = BlitParams {
            dst: &mut dst_short,
            ..ok_blit()
        }
        .validate(&FakeBackend)
        .unwrap_err()
        .to_string();
        assert!(err.contains("dst buffer"), "{err}");
    }

    #[test]
    fn validate_rejects_sources_past_32_bit_offsets() {
        // 30 000 × 30 000 RGB is 2.7 GB: its byte offsets overflow i32.
        let err = BlitParams {
            src_w: 30_000,
            src_h: 30_000,
            ..ok_blit()
        }
        .validate(&FakeBackend)
        .unwrap_err()
        .to_string();
        assert!(err.contains("32-bit"), "{err}");
        assert_eq!(
            blit_extents((30_000, 30_000), 1, (1, 1)),
            Some((900_000_000, 4))
        );
        assert_eq!(blit_extents((1, 1), 0, (50_000, 50_000)), None);
    }

    // --- per-pixel kernels ---

    #[test]
    fn icc_clut_validate_returns_the_node_count_and_checks_buffers() {
        static GRID: usize = (color::grid::CmykGrid::NODES + 2 * 17) * 4;
        static ODD_GRID: usize = (color::grid::CmykGrid::NODES + 2 * 17) * 4 + 4;
        let ok = || IccClutParams::<FakeBackend> {
            cmyk: &FAKE_BUF,
            rgb: fake_out(),
            clut: &GRID,
            n_pixels: 4,
            wide: false,
        };
        assert_eq!(ok().validate(&FakeBackend).unwrap(), 17);
        let err =
            |p: IccClutParams<'_, FakeBackend>| p.validate(&FakeBackend).unwrap_err().to_string();
        let e = err(IccClutParams {
            rgb: fake_out_of(15),
            ..ok()
        });
        assert!(e.contains("rgb buffer"), "{e}");
        for clut in [&ODD_GRID, &FAKE_BUF] {
            let e = err(IccClutParams { clut, ..ok() });
            assert!(e.contains("clut length"), "{e}");
        }
        let e = err(IccClutParams {
            n_pixels: MAX_KERNEL_PIXELS + 1,
            ..ok()
        });
        assert!(e.contains("MAX_KERNEL_PIXELS"), "{e}");
        let wide = || IccClutParams {
            wide: true,
            rgb: fake_out_of(32),
            ..ok()
        };
        assert_eq!(wide().validate(&FakeBackend).unwrap(), 17);
        let e = err(IccClutParams {
            rgb: fake_out_of(31),
            ..wide()
        });
        assert!(e.contains("n_pixels × 8"), "{e}");
        let e = err(IccClutParams {
            n_pixels: MAX_WIDE_CLUT_PIXELS + 1,
            rgb: fake_out(),
            ..wide()
        });
        assert!(e.contains("MAX_WIDE_CLUT_PIXELS"), "{e}");
    }

    #[test]
    fn icc_clut_params_require_a_word_per_pixel() {
        static GRID: usize = (color::grid::CmykGrid::NODES + 2 * 16) * 4;
        const N: u32 = 5;
        let n = N as usize;
        let params = |wide, rgb_len| IccClutParams::<FakeBackend> {
            cmyk: &FAKE_BUF,
            rgb: fake_out_of(rgb_len),
            clut: &GRID,
            n_pixels: N,
            wide,
        };
        for (wide, words) in [(false, 1), (true, 2)] {
            let device = n * 4 * words;
            assert_eq!(params(wide, device).validate(&FakeBackend).unwrap(), 16);
            let e = params(wide, device - 1)
                .validate(&FakeBackend)
                .unwrap_err()
                .to_string();
            assert!(e.contains("rgb buffer"), "{e}");
        }
    }
}
