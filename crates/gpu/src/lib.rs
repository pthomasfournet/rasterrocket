//! GPU compute for the rasteriser: a CUDA context ([`GpuCtx`], via cudarc)
//! and the CUDA and Vulkan (`vulkan` feature) backends of
//! [`backend::GpuBackend`], which run the ICC CLUT and image blit kernels; the whole-page raster (`page`, `vulkan` feature); the
//! device-resident image cache (`cache`, `cache` feature); and the nvJPEG,
//! nvJPEG2000 and VA-API decoders.  The ICC kernel's host evaluator is
//! [`color::grid::CmykGrid::convert`].
//!
//! The interpreter depends on this crate through its `gpu-icc`,
//! `cache`, `vulkan`, `nvjpeg`, `nvjpeg2k` and `vaapi` features.
//!
//! # Usage
//!
//! ```no_run
//! use rasterrocket_gpu::GpuCtx;
//! let ctx = GpuCtx::init().expect("no CUDA device");
//! ```
//!
//! # Feature flags
//!
//! - `nvjpeg` — enables the `nvjpeg` module for GPU-accelerated JPEG decoding
//!   via NVIDIA nvJPEG.  Requires `libnvjpeg.so` at link time.
//! - `nvjpeg2k` — enables the `nvjpeg2k` module for GPU-accelerated JPEG 2000
//!   decoding via NVIDIA nvJPEG2000.  Requires `libnvjpeg2k.so` at link time.
//! - `gpu-deskew` — enables the `npp_rotate` module for GPU rotation via CUDA NPP
//!   (`nppiRotate_8u_C1R_Ctx`).  Requires `libnppig.so`, `libnppc.so`, and
//!   `libcudart.so` at link time.
//! - `vaapi` — enables the `vaapi` module for VA-API hardware JPEG decoding on AMD
//!   and Intel iGPU/discrete hardware.  Requires `libva.so.2` and `libva-drm.so.2`
//!   at link time.

pub(crate) mod cuda;

pub mod backend;
#[cfg_attr(
    ptx_placeholder,
    expect(
        dead_code,
        reason = "the context init that uses it is skipped on placeholder PTX"
    )
)]
mod cuda_diag;
mod lib_kernels;
mod live;

#[cfg(feature = "cache")]
pub mod blit;
#[cfg(feature = "cache")]
pub mod cache;

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub(crate) mod decode_queue;
#[cfg(feature = "vaapi")]
pub(crate) mod jpeg;
pub(crate) mod jpeg_sof;
#[cfg(feature = "vulkan")]
pub mod page;
#[cfg(feature = "cache")]
mod staging;
pub(crate) mod traits;

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub use decode_queue::{DecodeQueue, JpegQueueHandle};
pub use jpeg_sof::{JpegSof, JpegVariant, jpeg_sof_info, jpeg_sof_type};
pub use live::{open_contexts, release_thread_state};
pub use traits::{DecodedImage, GpuDecodeError, GpuJpeg2kDecoder, GpuJpegDecoder};

#[cfg(feature = "nvjpeg")]
pub mod nvjpeg;

#[cfg(feature = "nvjpeg2k")]
pub mod nvjpeg2k;

#[cfg(feature = "gpu-deskew")]
pub mod npp_rotate;

#[cfg(feature = "vaapi")]
pub mod vaapi;

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig};

// The embedded PTX strings are only consumed by `GpuCtx::init_inner`, which
// is itself gated on `not(ptx_placeholder)`.  Gate the consts (and the
// imports they need) the same way so a build without NVCC compiles cleanly
// without dead-code warnings.
#[cfg(not(ptx_placeholder))]
use cudarc::driver::CudaModule;
#[cfg(not(ptx_placeholder))]
use cudarc::nvrtc::Ptx;

#[cfg(not(ptx_placeholder))]
const PTX_ICC_CLUT: &str = include_str!(concat!(env!("OUT_DIR"), "/icc_clut.ptx"));
#[cfg(all(not(ptx_placeholder), feature = "cache"))]
const PTX_BLIT_IMAGE: &str = include_str!(concat!(env!("OUT_DIR"), "/blit_image.ptx"));

/// Bytes per pixel for the RGBA8 pixel format the renderer composites in.
///
/// Lives at the crate root rather than under `cache::` because every
/// path that touches pixel buffers (renderer, kernels, blit, tests) needs
/// it; gating it behind the `cache` feature was an accident of where it
/// was first defined.  `cache::RGBA_BPP` is re-exported for back-compat.
pub const RGBA_BPP: usize = 4;
/// Minimum pixel count for the device ICC CMYK→RGB CLUT conversion.
///
/// The device kernels reproduce [`color::grid::CmykGrid::convert`] byte for
/// byte; the renderer converts a smaller ICC CMYK image through the profile
/// transform, as without a GPU backend.
///
/// Applies only when a grid is available (`grid` is `Some`).  The matrix path
/// (`grid` is `None`) always uses the CPU AVX-512 fallback — `threshold_bench` showed the GPU
/// matrix kernel never beats AVX-512 across 256–4M pixels on this machine (`PCIe`
/// round-trip cost exceeds the cheap per-pixel computation at all measured sizes).
///
/// Measured with `threshold_bench --only clut` (one call against the
/// single-threaded evaluator: the device is faster from 4 000 px on CUDA,
/// Vulkan NVIDIA and RADV, from 16 000 px under `MoltenVK`) and end to end
/// (8-page documents, one thread: the device is faster from 128 000 px on
/// CUDA and RADV and level with Vulkan NVIDIA; on every core, level below
/// 1 Mpx).  The renderer reads an override from
/// `PDF_RASTER_ICC_CLUT_THRESHOLD`.
pub const GPU_ICC_CLUT_THRESHOLD: usize = 128_000;

pub(crate) struct GpuKernels {
    pub(crate) icc_cmyk_matrix: CudaFunction,
    pub(crate) icc_cmyk_clut: CudaFunction,
    #[cfg(feature = "cache")]
    pub(crate) blit_image: CudaFunction,
}

/// An initialised CUDA context and compiled kernel set.
///
/// Created with [`GpuCtx::init`] and shared across threads via `Arc`.  Its
/// kernels and scratch buffers are freed when it drops; the CUDA context
/// itself is released once the threads' streams on it are dropped too
/// ([`release_thread_state`], or a thread's next stream on another context).
///
/// # Concurrency
///
/// All methods take `&self` and are safe to call from multiple threads.  Work is
/// queued on the calling thread's stream ([`GpuCtx::stream`]), so concurrent
/// callers execute independently on the device and a synchronising call waits
/// only for the calling thread's own work.
pub struct GpuCtx {
    pub(crate) cuda: Arc<CudaContext>,
    pub(crate) kernels: GpuKernels,
    /// Scratch buffers of the backend operations run on this context.
    pub(crate) scratch: backend::DeviceScratch<backend::cuda::CudaBackend>,
}

thread_local! {
    /// The calling thread's stream for each [`GpuCtx`] context it has used,
    /// created on first use.  Each stream holds an `Arc` to its context, so a
    /// context cannot be freed (and its address reused) while an entry names it.
    /// A stream whose context no live [`GpuCtx`] holds is dropped when the
    /// thread next creates one ([`live`]).
    static THREAD_STREAMS: RefCell<Vec<Arc<CudaStream>>> = const { RefCell::new(Vec::new()) };
}

impl GpuCtx {
    /// Initialise CUDA device 0 and compile the embedded kernels.
    ///
    /// # Lazy-loading audit
    ///
    /// Both backends already defer all driver work to first dispatch:
    ///
    /// - `cudarc` resolves CUDA driver symbols through `libloading` on first
    ///   call — the `dlopen("libcuda.so")` and `dlsym(...)` happen here, in
    ///   `init_inner`, not at process startup.
    /// - `ash::Entry::load` (in `backend::vulkan::device::init`) similarly
    ///   `dlopen`s `libvulkan.so` only when invoked.
    /// - `init` is not on any static-init path: it's called from
    ///   `rasterrocket::render::init_gpu_ctx` during `open_session`, after the
    ///   CLI has parsed args.
    /// - The only `OnceLock<GpuCtx>` in this crate lives behind
    ///   `#[cfg(feature = "gpu-validation")]` test code; runtime contexts are
    ///   constructed explicitly by callers and shared via `Arc`.
    ///
    /// As of 2026-05-09, `rasterrocket --help` (closest proxy for `--version`)
    /// runs in ~1.0 ms median (hyperfine, 30 runs, prewarmed); the dlopen cost
    /// would only be paid once a real render starts. No deferral work needed.
    ///
    /// # Errors
    ///
    /// Returns an error if no CUDA device is present or kernel load fails.
    /// Also returns an error if the `gpu` crate was built without NVCC
    /// available (placeholder PTX); the message points at the build
    /// step rather than the driver, which is what's actually wrong.
    pub fn init() -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(ptx_placeholder)]
        return Err(
            "GpuCtx::init: PTX kernels are placeholders — the `gpu` crate was \
             built on a host without NVCC.  Rebuild on a machine with the CUDA \
             toolkit (`/usr/local/cuda/bin/nvcc`) or set the NVCC env var to \
             the toolkit's nvcc binary.  CPU paths still work; only `--backend \
             cuda` and CUDA-feature-gated paths require real PTX."
                .into(),
        );

        #[cfg(not(ptx_placeholder))]
        Self::init_inner()
    }

    /// Real init body — separate so the `cfg(ptx_placeholder)` early-return
    /// in `init` doesn't get tangled with the success path's borrow scopes.
    #[cfg(not(ptx_placeholder))]
    fn init_inner() -> Result<Self, Box<dyn std::error::Error>> {
        if !cuda_diag::driver_library_present() {
            return Err(cuda_diag::NO_DRIVER_LIBRARY.into());
        }
        let driver = cuda_diag::driver_cuda_version();
        let ctx: Arc<CudaContext> =
            CudaContext::new(0).map_err(|e| cuda_diag::context_failure(e, driver))?;
        let device = ctx
            .compute_capability()
            .ok()
            .and_then(|(m, n)| Some((u32::try_from(m).ok()?, u32::try_from(n).ok()?)));

        let load =
            |ptx_src: &str, name: &str| -> Result<CudaFunction, Box<dyn std::error::Error>> {
                let header = cuda_diag::ptx_header(ptx_src);
                if let Some(message) = device.and_then(|d| cuda_diag::arch_mismatch(&header, d)) {
                    return Err(message.into());
                }
                let ptx = Ptx::from_src(ptx_src);
                let module: Arc<CudaModule> = ctx
                    .load_module(ptx)
                    .map_err(|e| cuda_diag::load_failure(e, &header, driver, device))?;
                Ok(module.load_function(name)?)
            };

        let kernels = GpuKernels {
            icc_cmyk_matrix: load(PTX_ICC_CLUT, "icc_cmyk_matrix")?,
            icc_cmyk_clut: load(PTX_ICC_CLUT, "icc_cmyk_clut")?,
            #[cfg(feature = "cache")]
            blit_image: load(PTX_BLIT_IMAGE, "blit_image")?,
        };
        live::register(&ctx);
        Ok(Self {
            cuda: ctx,
            scratch: backend::DeviceScratch::new(),
            kernels,
        })
    }

    /// The calling thread's CUDA stream on this context.
    ///
    /// Every kernel launch, transfer and device allocation made through
    /// this context is queued on the stream of the thread that makes it.
    /// Each thread owns one non-blocking stream per context, created on
    /// first use and destroyed at thread exit, so synchronising it waits
    /// only for that thread's work.
    ///
    /// A device buffer used from more than one stream is ordered by
    /// cudarc's per-allocation read/write events: a launch or copy waits
    /// for the buffer's last recorded write, and records a write only when
    /// it takes the buffer by `&mut` (copies into it, `&mut` kernel
    /// arguments).  A kernel that writes through a `&CudaSlice` argument
    /// or a raw device pointer records no write, so its output is ordered
    /// only against later work on the same stream — i.e. the same thread.
    /// Every caller in the renderer queues a page's GPU work from the
    /// thread rendering that page, which satisfies this.
    ///
    /// If stream creation fails, the thread uses the context's legacy
    /// default stream instead; the event ordering above still applies.
    #[must_use]
    pub fn stream(&self) -> Arc<CudaStream> {
        THREAD_STREAMS.with_borrow_mut(|streams| {
            if let Some(s) = streams
                .iter()
                .find(|s| Arc::ptr_eq(s.context(), &self.cuda))
            {
                return Arc::clone(s);
            }
            live::retain_live(streams, |s| s.context());
            let stream = self.cuda.new_stream().unwrap_or_else(|e| {
                log::warn!(
                    "gpu: per-thread stream creation failed ({e}); using the default stream"
                );
                self.cuda.default_stream()
            });
            streams.push(Arc::clone(&stream));
            stream
        })
    }

    /// A new non-blocking stream on this context, owned by the caller.
    ///
    /// For components that outlive any one thread's work and queue
    /// transfers of their own, such as the device image cache.
    ///
    /// # Errors
    ///
    /// Returns the driver error if stream creation fails.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>, cudarc::driver::DriverError> {
        self.cuda.new_stream()
    }
}

impl Drop for GpuCtx {
    fn drop(&mut self) {
        live::unregister(&self.cuda);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "callers validate n ≤ u32::MAX via u32::try_from before calling this; n.div_ceil(256) is therefore also ≤ u32::MAX"
)]
pub(crate) const fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
