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
//! - `pipeline` — root signatures and compute pipeline states per kernel.
//! - `recorder` — the page's command list, buffer states and descriptor
//!   heap.
//! - `select` — adapter eligibility, ranking and selector resolution.
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
mod pipeline;
#[cfg(windows)]
mod recorder;
#[cfg_attr(
    not(any(windows, test)),
    expect(dead_code, reason = "only the Windows host code consumes it")
)]
mod select;
#[cfg(windows)]
mod transfer;

#[cfg(windows)]
pub use fence::PageFence;
#[cfg(windows)]
pub use host::D3d12Backend;
#[cfg(windows)]
pub use memory::{DeviceBuffer, HostBuffer};

#[cfg(windows)]
mod host {
    use std::sync::{Arc, Mutex, PoisonError};

    use windows::Win32::Graphics::Dxgi::{
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
    };
    use windows::core::Interface as _;

    use super::device::{self, D3d12Device};
    use super::error::hr_err;
    use super::fence::{PageFence, Submitter};
    use super::memory::{self, DeviceBuffer, HostBuffer, HostHeap};
    use super::pipeline::KernelId;
    use super::recorder::Recorder;
    use super::transfer::Transfer;
    use crate::backend::select::DeviceSelector;
    use crate::backend::{
        DeviceScratch, GpuBackend, Result, VramBudget, params, recorded_copy_len, reject_zero_size,
        run_page_steps,
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
        recorder: Recorder,
        transfer: Transfer,
        /// Held for the whole of a [`GpuBackend::run_page`]: the recorder
        /// has one recording state, so pages from several threads run one
        /// at a time.
        page: Mutex<()>,
        submitter: Arc<Submitter>,
        device: D3d12Device,
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
        pub fn open(selector: Option<&DeviceSelector>) -> Result<Self> {
            let device = device::open_adapter(selector)?;
            let submitter = Arc::new(Submitter::new(&device.device, &device.queue)?);
            let recorder = Recorder::new(&device.device, submitter.clone())?;
            let transfer = Transfer::new(&device.device, submitter.clone())?;
            Ok(Self {
                scratch: DeviceScratch::new(),
                recorder,
                transfer,
                page: Mutex::new(()),
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
            Self::open(Some(&"warp".parse()?))
        }

        /// The indices of every usable hardware adapter: those with
        /// dedicated video memory first, each group in enumeration order.
        ///
        /// # Errors
        ///
        /// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
        /// when there is none.
        pub fn usable_gpus() -> Result<Vec<usize>> {
            device::usable_gpus()
        }

        /// The adapter's index in the enumeration selectors resolve against.
        #[must_use]
        pub const fn device_index(&self) -> usize {
            self.device.index
        }

        /// The adapter's name (`DXGI_ADAPTER_DESC1::Description`).
        #[must_use]
        pub fn device_name(&self) -> &str {
            &self.device.name
        }

        /// Whether the adapter is WARP.
        #[must_use]
        pub const fn is_warp(&self) -> bool {
            self.device.warp
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
            self.transfer.upload(dst, src)
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

    #[cfg(test)]
    mod tests {
        use super::super::diag::{cpu_tier, warp_module};
        use super::*;

        /// The floor-division probe binds its status at `u0` and its
        /// quotients at `u1`; a binding taken from the tables puts each
        /// where the kernel reads it.
        #[test]
        fn floor_div_probe_binds_from_the_register_tables() {
            use crate::backend::kernel_status::{FLOOR_DIV_STEPS, STATUS_BYTES};

            let backend = D3d12Backend::warp().expect("the WARP adapter opens");
            if !backend.has_int64() {
                eprintln!("skipping: WARP lacks Int64ShaderOps");
                return;
            }
            // Operands within floor division's domain: `0 < b < 2^32`,
            // `|a / b| < 2^40`.
            let pairs: [(i64, i64); 5] = [
                (7, 2),
                (-7, 2),
                (0, 3),
                (-1, (1 << 32) - 1),
                (-(1 << 40) - 5, 256),
            ];
            let floor = |a: i64, b: i64| {
                let q = a / b;
                if a % b != 0 && (a < 0) != (b < 0) {
                    q - 1
                } else {
                    q
                }
            };
            let words: Vec<u8> = pairs
                .iter()
                .copied()
                .flat_map(<[i64; 2]>::from)
                .flat_map(i64::to_le_bytes)
                .collect();
            let n = u32::try_from(pairs.len()).expect("five pairs");
            let mut input = backend.alloc_device(words.len()).expect("input");
            backend.upload(&mut input, &words).expect("upload");
            let mut status = backend.alloc_device_zeroed(STATUS_BYTES).expect("status");
            let mut quotients = backend.alloc_device(pairs.len() * 8).expect("quotients");
            backend
                .run_page(|b| {
                    b.recorder.dispatch(
                        KernelId::FloorDivProbe,
                        &[(&input, 4)],
                        &[(&mut status, 4), (&mut quotients, 4)],
                        &[n, FLOOR_DIV_STEPS],
                        (n.div_ceil(64), 1, 1),
                    )
                })
                .expect("probe page");
            let mut out = vec![0; pairs.len() * 8];
            backend.download(&quotients, &mut out).expect("quotients");
            let got: Vec<i64> = out
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect();
            let want: Vec<i64> = pairs.iter().map(|&(a, b)| floor(a, b)).collect();
            assert_eq!(got, want);
            let mut words = [0xFF; STATUS_BYTES];
            backend.download(&status, &mut words).expect("status");
            assert_eq!(words, [0; STATUS_BYTES], "the probe recorded a fault");
        }

        /// The reference count of `object`.
        fn refs(object: &windows::core::IUnknown) -> u32 {
            let vtable = object.vtable();
            // SAFETY: `object` is live; the reference added is released at
            // once, and `Release` returns the count left.
            unsafe {
                let _added = (vtable.AddRef)(object.as_raw());
                (vtable.Release)(object.as_raw())
            }
        }

        /// Stranded commands refuse a new list and never release what the
        /// list references, even when dropped; released commands do.
        #[test]
        fn stranded_commands_keep_their_references() {
            use super::super::commands::Commands;

            let backend = D3d12Backend::warp().expect("the WARP adapter opens");
            let buf = backend.alloc_device(64).expect("alloc_device");
            let unknown: windows::core::IUnknown = buf.resource().clone().into();
            let base = refs(&unknown);

            let mut released = Commands::new(&backend.device.device).expect("commands");
            released.begin().expect("begin");
            released.hold(buf.resource());
            assert_eq!(refs(&unknown), base + 1);
            released.discard();
            assert_eq!(refs(&unknown), base, "discard releases the hold");
            drop(released);

            let mut stranded = Commands::new(&backend.device.device).expect("commands");
            stranded.begin().expect("begin");
            stranded.hold(buf.resource());
            stranded.strand();
            stranded.release();
            stranded.discard();
            assert!(stranded.begin().is_err(), "stranded commands are not reset");
            drop(stranded);
            assert_eq!(refs(&unknown), base + 1, "the stranded hold is leaked");
        }

        /// A page whose own fence cannot be waited on keeps everything its
        /// list references, through `wait_page` and the backend's drop, and
        /// no further page begins; a page dropped unwaited does the same
        /// when the queue cannot be drained.
        #[test]
        fn unconfirmed_pages_keep_their_references() {
            for waited in [true, false] {
                let backend = D3d12Backend::warp().expect("the WARP adapter opens");
                let mut buf = backend.alloc_device(64).expect("alloc_device");
                let unknown: windows::core::IUnknown = buf.resource().clone().into();
                let base = refs(&unknown);
                backend.begin_page().expect("begin_page");
                backend.record_zero_buffer(&mut buf, 64).expect("fill");
                let fence = backend.submit_page().expect("submit_page");
                backend.submitter.fail_waits();
                if waited {
                    assert!(backend.wait_page(fence).is_err(), "the wait fails");
                    let err = backend.begin_page().expect_err("the recorder is lost");
                    assert!(err.to_string().contains("could not be confirmed"), "{err}");
                    backend.abandon_page();
                }
                drop(backend);
                assert_eq!(
                    refs(&unknown),
                    base + 1,
                    "waited {waited}: the list's hold on the buffer was released"
                );
            }
        }

        /// Every kernel's root signature and pipeline state build.
        #[test]
        fn every_kernel_pipeline_builds_on_warp() {
            let backend = D3d12Backend::warp().expect("the WARP adapter opens");
            for &k in KernelId::ALL {
                if matches!(k, KernelId::PageRasterPaths | KernelId::FloorDivProbe)
                    && !backend.has_int64()
                {
                    continue;
                }
                backend
                    .recorder
                    .pipeline(k)
                    .unwrap_or_else(|e| panic!("{}: {e}", k.label()));
            }
        }

        #[test]
        fn warp_opens_and_logs_its_module() {
            let backend = D3d12Backend::warp().expect("the WARP adapter opens");
            assert!(backend.is_warp(), "{}", backend.device_name());
            let (path, version) = warp_module().expect("d3d10warp.dll is loaded");
            let file = path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or_default();
            assert!(
                file.eq_ignore_ascii_case("d3d10warp.dll"),
                "{}",
                path.display()
            );
            eprintln!(
                "WARP {version} from {} on a CPU with {}; adapter {} \"{}\", int64 {}",
                path.display(),
                cpu_tier(),
                backend.device_index(),
                backend.device_name(),
                backend.has_int64()
            );
        }

        #[test]
        fn hardware_adapters_open_by_index_and_automatically() {
            let gpus = match D3d12Backend::usable_gpus() {
                Ok(gpus) => gpus,
                Err(e) => {
                    eprintln!("skipping: {e}");
                    return;
                }
            };
            let auto = D3d12Backend::open(None).expect("the automatic choice opens");
            assert_eq!(auto.device_index(), gpus[0]);
            assert!(!auto.is_warp());
            for &i in &gpus {
                let sel: DeviceSelector = i.to_string().parse().expect("selector");
                let backend = D3d12Backend::open(Some(&sel)).expect("a usable adapter opens");
                assert_eq!(backend.device_index(), i);
                assert!(!backend.is_warp());
                eprintln!(
                    "adapter {i} \"{}\", int64 {}",
                    backend.device_name(),
                    backend.has_int64()
                );
            }
        }
    }
}
