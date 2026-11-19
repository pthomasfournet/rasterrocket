//! Direct3D 12 compute backend.
//!
//! Adapter selection (`select`) is plain data and builds on every host;
//! everything that calls Direct3D 12 or DXGI builds on Windows only.
//!
//! Module split (all private; named here for orientation only):
//! - `commands` — a command list, its buffer states and barriers, and the
//!   objects it holds until its fence.
//! - `device` — DXGI factory, adapter enumeration, device, compute queue,
//!   debug layer.
//! - `diag` — the loaded WARP module and the CPU's vector instruction set.
//! - `error` — `windows::core::Error → BackendError` adaptor.
//! - `fence` — queue submission and the fence each submission signals.
//! - `memory` — committed device and host buffers, the view-element limit.
//! - `page` — whole-page raster from a command list; `PageDevice`.
//! - `pipeline` — root signatures and compute pipeline states per kernel.
//! - `recorder` — the page's command list, buffer states and descriptor
//!   heap.
//! - `select` — adapter eligibility, ranking and selector resolution.
//! - `timestamps` — device timestamps between the stages of a page.
//! - `transfer` — synchronous upload and download through staging buffers.

#[cfg(windows)]
mod commands;
#[cfg(windows)]
mod device;
#[cfg(windows)]
mod diag;
#[cfg(windows)]
mod error;
#[cfg(windows)]
mod fence;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod page;
#[cfg(windows)]
mod pipeline;
#[cfg(windows)]
mod recorder;
#[cfg_attr(
    not(any(windows, test)),
    expect(dead_code, reason = "only the Windows host code consumes it")
)]
mod select;
#[cfg(all(test, windows))]
mod tests;
#[cfg(windows)]
mod timestamps;
#[cfg(windows)]
mod transfer;

#[cfg(windows)]
pub use fence::PageFence;
#[cfg(windows)]
pub use host::D3d12Backend;
#[cfg(windows)]
pub use memory::{DeviceBuffer, HostBuffer};

/// The backend on the adapter `PDF_RASTER_D3D12_DEVICE` names, or on the
/// automatically chosen one when it is unset or empty; `None` (the calling
/// test is skipped) when none opens, unless `PDF_RASTER_REQUIRE_D3D12` is
/// `1`.  On WARP, the loaded module is logged, and with `RR_WARP_VERSION`
/// set it must be the copy beside the test binary and of that version.
///
/// # Panics
/// When the variable holds a selector that does not parse, when no adapter
/// opens and `PDF_RASTER_REQUIRE_D3D12` is `1`, or when the WARP check fails.
#[cfg(all(test, windows))]
pub(crate) fn test_backend() -> Option<D3d12Backend> {
    use crate::backend::page_device::PageDevice as _;
    use crate::backend::select::DeviceSelector;

    let selector = std::env::var("PDF_RASTER_D3D12_DEVICE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.parse::<DeviceSelector>()
                .expect("PDF_RASTER_D3D12_DEVICE holds a device selector")
        });
    match D3d12Backend::open(selector.as_ref()) {
        Ok(backend) => {
            eprintln!(
                "adapter {} \"{}\"",
                backend.device_index(),
                backend.device_name()
            );
            if backend.is_warp() {
                let loaded = D3d12Backend::loaded_warp();
                match &loaded {
                    Some((path, v)) => eprintln!("WARP module: {} version {v}", path.display()),
                    None => eprintln!("WARP module: not loaded"),
                }
                if let Ok(expected) = std::env::var("RR_WARP_VERSION") {
                    let (path, version) = loaded.expect("d3d10warp.dll is loaded");
                    let exe = std::env::current_exe().expect("test binary path");
                    assert_eq!(
                        path.parent(),
                        exe.parent(),
                        "the loaded WARP is not the one beside the test binary: {}",
                        path.display()
                    );
                    assert_eq!(version, expected.trim(), "WARP version");
                }
            }
            Some(backend)
        }
        Err(e) => {
            assert!(
                !std::env::var("PDF_RASTER_REQUIRE_D3D12").is_ok_and(|v| v == "1"),
                "PDF_RASTER_REQUIRE_D3D12=1, but no D3D12 backend opens: {e}"
            );
            eprintln!("skipping: no D3D12 backend: {e}");
            None
        }
    }
}

#[cfg(windows)]
mod host {
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use windows::Win32::Graphics::Dxgi::{
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
    };
    use windows::core::Interface as _;

    use super::device::{self, D3d12Device};
    use super::error::hr_err;
    use super::fence::{PageFence, Submitter};
    use super::memory::{self, DeviceBuffer, HostBuffer, HostHeap};
    use super::page::{PAGE_SETS, PageScratch};
    use super::pipeline::KernelId;
    use super::recorder::Recorder;
    use super::transfer::Transfer;
    use crate::backend::page_job::PageClock;
    use crate::backend::select::DeviceSelector;
    use crate::backend::{
        DeviceScratch, GpuBackend, Result, ScratchPool, VramBudget, params, recorded_copy_len,
        reject_zero_size, run_page_steps,
    };

    /// Direct3D 12 compute backend on one DXGI adapter.
    ///
    /// Adapters are named by the [`DeviceSelector`] rules over DXGI's
    /// high-performance enumeration order, except that the selector `warp`
    /// names the WARP software adapter and no other selector names a
    /// software adapter.  `PDF_RASTER_D3D12_DEBUG=1` enables the D3D12
    /// debug layer before the first device of the process is created.
    pub struct D3d12Backend {
        // Field order is drop order: buffers and the recorder go before
        // the queue and device they were created on.
        scratch: DeviceScratch<Self>,
        /// Buffers of [`PageDevice::render_page`], kept between pages: one
        /// set per page being staged, rendered or read back.
        ///
        /// [`PageDevice::render_page`]: crate::backend::page_device::PageDevice::render_page
        pub(super) page_scratch: ScratchPool<PageScratch>,
        pub(super) recorder: Recorder,
        transfer: Transfer,
        /// Held for the whole of a [`GpuBackend::run_page`]: the recorder
        /// has one recording state, so pages from several threads run one
        /// at a time.
        pub(super) page: Mutex<()>,
        /// The time page rasters hold the device
        /// ([`PageDevice::page_mean`]).
        ///
        /// [`PageDevice::page_mean`]: crate::backend::page_device::PageDevice::page_mean
        pub(super) page_clock: PageClock,
        /// Whether the ICC grid kernel reproduces the host on this
        /// device; measured on first use ([`Self::icc_clut_is_exact`]).
        icc_clut_exact: OnceLock<bool>,
        pub(super) submitter: Arc<Submitter>,
        pub(super) device: D3d12Device,
    }

    impl D3d12Backend {
        /// Open the adapter `selector` names, or without one the first
        /// usable hardware adapter with dedicated video memory, else the
        /// first usable hardware adapter.
        ///
        /// # Errors
        ///
        /// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
        /// when the selector names no usable adapter;
        /// [`BackendError::Other`](crate::backend::BackendError::Other) when
        /// no hardware adapter is usable or a Direct3D 12 call fails.
        pub(super) fn open_adapter(selector: Option<&DeviceSelector>) -> Result<Self> {
            let device = device::open_adapter(selector)?;
            let submitter = Arc::new(Submitter::new(&device.device, &device.queue)?);
            let recorder = Recorder::new(&device.device, submitter.clone())?;
            let transfer = Transfer::new(&device.device, submitter.clone())?;
            Ok(Self {
                scratch: DeviceScratch::new(),
                page_scratch: ScratchPool::bounded(PAGE_SETS),
                recorder,
                transfer,
                page: Mutex::new(()),
                page_clock: PageClock::default(),
                icc_clut_exact: OnceLock::new(),
                submitter,
                device,
            })
        }

        /// Open the WARP software adapter.
        ///
        /// # Errors
        ///
        /// As [`Self::open`] with the selector `warp`.
        pub fn warp() -> Result<Self> {
            Self::open_adapter(Some(&"warp".parse()?))
        }

        /// Whether the ICC grid kernel reproduces
        /// [`color::grid::CmykGrid::convert`] byte for byte on this device.
        ///
        /// The kernel interpolates in 32-bit integers and reads the pixels
        /// and the grid through 32-bit structured views.  The first call
        /// runs [`crate::backend::icc_clut_matches_cpu`], which catches a
        /// device whose integer arithmetic, view access or shader
        /// translation is faulty, and logs a mismatch or a device error at
        /// `warn`; later calls return the same answer.
        #[must_use]
        pub fn icc_clut_is_exact(&self) -> bool {
            *self
                .icc_clut_exact
                .get_or_init(|| match crate::backend::icc_clut_matches_cpu(self) {
                    Ok(true) => true,
                    Ok(false) => {
                        log::warn!(
                            "d3d12: the ICC grid kernel does not reproduce the host on this \
                             device; ICC CMYK images convert on the CPU"
                        );
                        false
                    }
                    Err(e) => {
                        log::warn!(
                            "d3d12: the ICC grid kernel check failed: {e}; ICC CMYK images \
                             convert on the CPU"
                        );
                        false
                    }
                })
        }

        /// Whether the adapter is WARP.
        #[must_use]
        pub const fn is_warp(&self) -> bool {
            self.device.warp
        }

        /// The path and file version of the `d3d10warp.dll` loaded in the
        /// process, or `None` when none is loaded.
        #[must_use]
        pub fn loaded_warp() -> Option<(std::path::PathBuf, String)> {
            super::diag::warp_module()
        }

        /// The widest vector instruction set of the host CPU (`avx512`,
        /// `avx2`, `sse` or `neon`), which WARP runs on.
        #[must_use]
        pub fn cpu_tier() -> &'static str {
            super::diag::cpu_tier()
        }

        /// Whether the debug layer is enabled: `PDF_RASTER_D3D12_DEBUG=1`
        /// was set at the first open and the layer was available.
        #[must_use]
        pub fn debug_layer_enabled() -> bool {
            device::debug_layer_enabled()
        }

        /// Copy `src` into `dst[..src.len()]` and zeros into the bytes after
        /// it up to the next multiple of 4
        /// ([`word_padded`](crate::backend::layout::word_padded)), in one
        /// copy, as [`GpuBackend::upload`] copies.
        ///
        /// # Errors
        /// As [`GpuBackend::upload`], with the padded length checked
        /// against `dst`'s size.
        #[cfg(any(test, feature = "cache"))]
        pub(crate) fn upload_word_padded(&self, dst: &DeviceBuffer, src: &[u8]) -> Result<()> {
            let zeros = crate::backend::layout::word_padded(src.len()) - src.len();
            self.transfer.upload(dst, src, zeros)
        }

        /// Whether kernels may use 64-bit integers (`Int64ShaderOps`).
        #[must_use]
        pub const fn has_int64(&self) -> bool {
            self.device.int64
        }
    }

    impl Drop for D3d12Backend {
        fn drop(&mut self) {
            if let Err(e) = self.submitter.wait_idle() {
                log::warn!("d3d12: waiting for the queue to drain on drop failed: {e}");
                self.recorder.strand_in_flight();
            }
        }
    }

    impl crate::backend::sealed::Sealed for D3d12Backend {}

    impl GpuBackend for D3d12Backend {
        const NAME: &'static str = "d3d12";
        type DeviceBuffer = DeviceBuffer;
        type HostBuffer = HostBuffer;
        type PageFence = PageFence;

        /// Not zeroed (`CREATE_NOT_ZEROED`).
        fn alloc_device(&self, size: usize) -> Result<DeviceBuffer> {
            reject_zero_size(size, "alloc_device")?;
            memory::alloc_device(&self.device.device, size, false, "alloc_device")
        }

        /// A committed resource created zeroed; sizes that are not a
        /// multiple of 4 are refused, as the zero-fill refuses them.
        fn alloc_device_zeroed(&self, size: usize) -> Result<DeviceBuffer> {
            reject_zero_size(size, "alloc_device_zeroed")?;
            memory::fill_size(size as u64).map(drop)?;
            memory::alloc_device(&self.device.device, size, true, "alloc_device_zeroed")
        }

        /// The same DEFAULT-heap buffer as [`Self::alloc_device`]: the
        /// host reads device buffers only through READBACK copies.
        fn alloc_device_readback(&self, size: usize) -> Result<DeviceBuffer> {
            reject_zero_size(size, "alloc_device_readback")?;
            memory::alloc_device(&self.device.device, size, false, "alloc_device_readback")
        }

        fn device_buffer_len(&self, buf: &DeviceBuffer) -> usize {
            usize::try_from(buf.size()).expect("a buffer's size fits usize on 64-bit targets")
        }

        /// The resource is released once no page in flight references it.
        fn free_device(&self, buf: DeviceBuffer) {
            drop(buf);
        }

        /// An UPLOAD-heap buffer.
        fn alloc_host_pinned(&self, size: usize) -> Result<HostBuffer> {
            reject_zero_size(size, "alloc_host_pinned")?;
            memory::alloc_host(
                &self.device.device,
                size,
                HostHeap::Upload,
                "alloc_host_pinned",
            )
        }

        /// A READBACK-heap buffer.
        fn alloc_host_readback(&self, size: usize) -> Result<HostBuffer> {
            reject_zero_size(size, "alloc_host_readback")?;
            memory::alloc_host(
                &self.device.device,
                size,
                HostHeap::Readback,
                "alloc_host_readback",
            )
        }

        fn free_host_pinned(&self, buf: HostBuffer) {
            drop(buf);
        }

        fn begin_page(&self) -> Result<()> {
            self.recorder.begin_page()
        }

        fn record_blit_image(&self, p: params::BlitParams<'_, Self>) -> Result<()> {
            p.validate(self)?;
            let int = |v: u32| i32::try_from(v).unwrap_or(i32::MAX).cast_unsigned();
            // The order of `blit_image.slang`'s uniforms.
            let constants = [
                int(p.src_w),
                int(p.src_h),
                p.src_layout,
                int(p.dst_w),
                int(p.dst_h),
                p.bbox[0].cast_unsigned(),
                p.bbox[1].cast_unsigned(),
                p.bbox[2].cast_unsigned(),
                p.bbox[3].cast_unsigned(),
                p.filter,
            ];
            // One 16×16 thread group per tile of the box.
            let extent = |lo: i32, hi: i32| u32::try_from((hi - lo).max(0)).unwrap_or(0);
            let groups = (
                extent(p.bbox[0], p.bbox[2]).div_ceil(16),
                extent(p.bbox[1], p.bbox[3]).div_ceil(16),
                1,
            );
            self.recorder.dispatch(
                KernelId::BlitImage,
                &[(p.src, 4), (p.cols, 4), (p.rows, 4)],
                &[(p.dst, 4)],
                &constants,
                groups,
            )
        }

        fn record_icc_clut(&self, p: params::IccClutParams<'_, Self>) -> Result<()> {
            let n_nodes = p.validate(self)?;
            // The order of `icc_clut.slang`'s uniforms.
            let constants = [n_nodes, p.n_pixels, u32::from(p.wide)];
            self.recorder.dispatch(
                KernelId::IccClut,
                &[(p.cmyk, 4), (p.clut, 4)],
                &[(p.rgb, 4)],
                &constants,
                (p.n_pixels.div_ceil(256), 1, 1),
            )
        }

        fn record_zero_buffer(&self, buf: &mut DeviceBuffer, len: usize) -> Result<()> {
            self.recorder.record_zero_buffer(buf, len)
        }

        unsafe fn record_copy_to_device(
            &self,
            src: &HostBuffer,
            dst: &mut DeviceBuffer,
            len: usize,
        ) -> Result<()> {
            recorded_copy_len(len, src.size(), dst.size(), "record_copy_to_device")?
                .map_or(Ok(()), |n| self.recorder.record_copy_to_device(src, dst, n))
        }

        unsafe fn record_copy_to_host(
            &self,
            src: &DeviceBuffer,
            dst: &mut HostBuffer,
            len: usize,
        ) -> Result<()> {
            recorded_copy_len(len, src.size(), dst.size(), "record_copy_to_host")?
                .map_or(Ok(()), |n| self.recorder.record_copy_to_host(src, dst, n))
        }

        fn device_memory_is_host_mapped(&self) -> bool {
            false
        }

        unsafe fn device_bytes<'a>(&self, _buf: &'a DeviceBuffer) -> Option<&'a [u8]> {
            None
        }

        unsafe fn device_bytes_mut<'a>(&self, _buf: &'a mut DeviceBuffer) -> Option<&'a mut [u8]> {
            None
        }

        fn host_buffer_len(&self, buf: &HostBuffer) -> usize {
            buf.as_slice().len()
        }

        fn host_bytes<'a>(&self, buf: &'a HostBuffer) -> Result<&'a [u8]> {
            Ok(buf.as_slice())
        }

        fn host_bytes_mut<'a>(&self, buf: &'a mut HostBuffer) -> Result<&'a mut [u8]> {
            Ok(buf.as_mut_slice())
        }

        fn submit_page(&self) -> Result<PageFence> {
            self.recorder.submit_page()
        }

        fn wait_page(&self, fence: PageFence) -> Result<()> {
            self.recorder.wait_page(&fence)
        }

        fn abandon_page(&self) {
            self.recorder.abandon_page();
        }

        /// As the trait's default, under the backend's page lock.
        fn run_page<F>(&self, record: F) -> Result<()>
        where
            F: FnOnce(&Self) -> Result<()>,
        {
            let _page = self.page.lock().unwrap_or_else(PoisonError::into_inner);
            run_page_steps(self, record)
        }

        /// A copy through an UPLOAD staging buffer, in a submission of its
        /// own.
        fn upload(&self, dst: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
            self.transfer.upload(dst, src, 0)
        }

        /// A copy through a READBACK staging buffer, in a submission of
        /// its own.
        fn download(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
            self.transfer.download(src, dst)
        }

        fn scratch(&self) -> &DeviceScratch<Self> {
            &self.scratch
        }

        /// `QueryVideoMemoryInfo` of the local segment: three quarters of
        /// the budget left; the total is the adapter's dedicated video
        /// memory, or its shared system memory when it has none.
        fn detect_vram_budget(&self) -> Result<VramBudget> {
            let adapter: IDXGIAdapter3 = self
                .device
                .adapter
                .cast()
                .map_err(hr_err("QueryInterface(IDXGIAdapter3)"))?;
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: `adapter` is live; `info` is a valid out-pointer.
            unsafe {
                adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &raw mut info)
            }
            .map_err(hr_err("QueryVideoMemoryInfo"))?;
            // SAFETY: `adapter` is live.
            let desc = unsafe { self.device.adapter.GetDesc1() }.map_err(hr_err("GetDesc1"))?;
            let total = if desc.DedicatedVideoMemory > 0 {
                desc.DedicatedVideoMemory as u64
            } else {
                desc.SharedSystemMemory as u64
            };
            let usable = info
                .Budget
                .saturating_sub(info.CurrentUsage)
                .saturating_mul(3)
                / 4;
            Ok(VramBudget::new(total, usable.min(total)))
        }
    }

    /// The quotients of a [`D3d12Backend::floor_div_probe`], and the operands
    /// `(a, b)` of the first division it recorded as a fault.
    #[cfg(test)]
    pub(crate) type FloorDivProbe = (Vec<i64>, Option<(i64, i64)>);

    #[cfg(test)]
    impl D3d12Backend {
        /// The device's floor division of each pair `(a, b)` of `pairs`, with
        /// `max_steps` correction steps in each direction, and the operands
        /// of the first division it records as a fault; one page.
        pub(crate) fn floor_div_probe(
            &self,
            pairs: &[(i64, i64)],
            max_steps: u32,
        ) -> Result<FloorDivProbe> {
            let (quotients, status) =
                self.floor_div_probe_on(KernelId::FloorDivProbe, pairs, max_steps)?;
            Ok((
                quotients,
                crate::backend::kernel_status::floor_div_fault(&status),
            ))
        }

        /// The quotients of [`Self::floor_div_probe`] by the probe kernel
        /// `kernel`, and the status words of the page.  The operands go to
        /// the read-only register and the status and quotients to the
        /// writable ones, in the order the register tables give.
        pub(super) fn floor_div_probe_on(
            &self,
            kernel: KernelId,
            pairs: &[(i64, i64)],
            max_steps: u32,
        ) -> Result<(Vec<i64>, [u32; crate::backend::kernel_status::STATUS_WORDS])> {
            use crate::backend::kernel_status::{STATUS_BYTES, status_words};

            let n = u32::try_from(pairs.len()).map_err(crate::backend::BackendError::new)?;
            let operands: Vec<u8> = pairs
                .iter()
                .copied()
                .flat_map(<[i64; 2]>::from)
                .flat_map(i64::to_le_bytes)
                .collect();
            let mut input = self.alloc_device(operands.len())?;
            self.upload(&mut input, &operands)?;
            let mut status = self.alloc_device_zeroed(STATUS_BYTES)?;
            let mut quotients = self.alloc_device(pairs.len() * 8)?;
            self.run_page(|b| {
                b.recorder.dispatch(
                    kernel,
                    &[(&input, 4)],
                    &[(&mut status, 4), (&mut quotients, 4)],
                    &[n, max_steps],
                    (n.div_ceil(64), 1, 1),
                )
            })?;
            let mut out = vec![0; pairs.len() * 8];
            self.download(&quotients, &mut out)?;
            let quotients = out
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            let mut words = [0; STATUS_BYTES];
            self.download(&status, &mut words)?;
            Ok((quotients, status_words(&words)))
        }
    }
}
