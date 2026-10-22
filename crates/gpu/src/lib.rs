//! GPU-accelerated rasterizer helpers (CUDA via cudarc).
//!
//! Gate with `feature = "gpu"` in crates that depend on this.
//!
//! # Usage
//!
//! ```no_run
//! use gpu::GpuCtx;
//! let ctx = GpuCtx::init().expect("no CUDA device");
//! // … call ctx.composite_rgba8(src, dst) etc.
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

mod cmyk;
mod composite;
pub(crate) mod cuda;

pub mod backend;
pub mod lib_kernels;

#[cfg(feature = "cache")]
pub mod blit;
#[cfg(feature = "cache")]
pub mod cache;

#[cfg(feature = "gpu-jpeg-huffman")]
pub mod jpeg_decoder;

#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub mod decode_queue;
mod fill;
pub mod jpeg;
pub mod jpeg_sof;
pub mod traits;

pub use cmyk::icc_cmyk_to_rgb_cpu;
pub use composite::{apply_soft_mask_cpu, composite_rgba8_cpu};
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub use decode_queue::{DecodeQueue, JpegQueueHandle};
pub use fill::{AA_SIZE, AaFillRect, AaScanTable, ScanIntersect, aa_fill_cpu, pixel_coverage};
pub use jpeg_sof::{JpegSof, JpegVariant, jpeg_sof_info, jpeg_sof_type};
pub use traits::{DecodedImage, GpuCompute, GpuDecodeError, GpuJpeg2kDecoder, GpuJpegDecoder};

#[cfg(feature = "nvjpeg")]
pub mod nvjpeg;

#[cfg(feature = "nvjpeg2k")]
pub mod nvjpeg2k;

#[cfg(feature = "gpu-deskew")]
pub mod npp_rotate;

#[cfg(feature = "vaapi")]
pub mod vaapi;

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig};

// The embedded PTX strings are only consumed by `GpuCtx::init_inner`, which
// is itself gated on `not(ptx_placeholder)`.  Gate the consts (and the
// imports they need) the same way so a build without NVCC compiles cleanly
// without dead-code warnings.
#[cfg(not(ptx_placeholder))]
use cudarc::driver::{CudaContext, CudaModule};
#[cfg(not(ptx_placeholder))]
use cudarc::nvrtc::Ptx;

#[cfg(not(ptx_placeholder))]
const PTX_COMPOSITE: &str = include_str!(concat!(env!("OUT_DIR"), "/composite_rgba8.ptx"));
#[cfg(not(ptx_placeholder))]
const PTX_SOFT_MASK: &str = include_str!(concat!(env!("OUT_DIR"), "/apply_soft_mask.ptx"));
#[cfg(not(ptx_placeholder))]
const PTX_AA_FILL: &str = include_str!(concat!(env!("OUT_DIR"), "/aa_fill.ptx"));
#[cfg(not(ptx_placeholder))]
const PTX_ICC_CLUT: &str = include_str!(concat!(env!("OUT_DIR"), "/icc_clut.ptx"));
#[cfg(all(not(ptx_placeholder), feature = "cache"))]
const PTX_BLIT_IMAGE: &str = include_str!(concat!(env!("OUT_DIR"), "/blit_image.ptx"));
#[cfg(all(not(ptx_placeholder), feature = "gpu-jpeg-huffman"))]
const PTX_BLELLOCH_SCAN: &str = include_str!(concat!(env!("OUT_DIR"), "/blelloch_scan.ptx"));
#[cfg(all(not(ptx_placeholder), feature = "gpu-jpeg-huffman"))]
const PTX_PARALLEL_HUFFMAN: &str = include_str!(concat!(env!("OUT_DIR"), "/parallel_huffman.ptx"));
#[cfg(all(not(ptx_placeholder), feature = "gpu-jpeg-huffman"))]
const PTX_IDCT_COLOR: &str = include_str!(concat!(env!("OUT_DIR"), "/idct_color.ptx"));

/// Threshold in pixels below which CPU is faster than GPU dispatch overhead.
pub const GPU_COMPOSITE_THRESHOLD: usize = 500_000;
/// Threshold for soft-mask application.
pub const GPU_SOFTMASK_THRESHOLD: usize = 500_000;
/// Minimum fill bounding-box area (pixels) for a fill to be dispatched to the
/// GPU scanline-AA kernel rather than composited from the CPU's own
/// coverage walk.
///
/// `usize::MAX`: the path is dormant.  The kernel replaces only the span walk
/// and sub-sample count, which the CPU performs as bit-run fills and SIMD
/// popcounts; scan conversion and compositing stay on the CPU either way, so
/// the table upload, launch and readback are pure overhead and end-to-end
/// renders are slower on the GPU at every fill size measured.  The renderer
/// reads an override from `PDF_RASTER_AA_FILL_THRESHOLD`.
pub const GPU_AA_FILL_THRESHOLD: usize = usize::MAX;
/// Bytes per pixel for the RGBA8 pixel format the renderer composites in.
///
/// Lives at the crate root rather than under `cache::` because every
/// path that touches pixel buffers (renderer, kernels, blit, tests) needs
/// it; gating it behind the `cache` feature was an accident of where it
/// was first defined.  `cache::RGBA_BPP` is re-exported for back-compat.
pub const RGBA_BPP: usize = 4;
/// Minimum pixel count for GPU ICC CMYK→RGB CLUT transform to beat CPU + `PCIe` overhead.
///
/// Applies only when a full ICC CLUT is available (clut=Some).  The matrix path
/// (clut=None) always uses the CPU AVX-512 fallback — `threshold_bench` showed the GPU
/// matrix kernel never beats AVX-512 across 256–4M pixels on this machine (`PCIe`
/// round-trip cost exceeds the cheap per-pixel computation at all measured sizes).
///
/// The CLUT path is not yet benchmarked; 500 000 px is a conservative placeholder.
/// Run `threshold_bench` with a CLUT workload to calibrate once baking is in the hot path.
pub const GPU_ICC_CLUT_THRESHOLD: usize = 500_000;

pub(crate) struct GpuKernels {
    pub(crate) composite_rgba8: CudaFunction,
    pub(crate) apply_soft_mask: CudaFunction,
    pub(crate) aa_fill: CudaFunction,
    pub(crate) icc_cmyk_matrix: CudaFunction,
    pub(crate) icc_cmyk_clut: CudaFunction,
    #[cfg(feature = "cache")]
    pub(crate) blit_image: CudaFunction,
    /// Per-workgroup Blelloch scan entry (phase 1).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) scan_per_workgroup: CudaFunction,
    /// Single-workgroup scan of block sums (phase 2).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) scan_block_sums: CudaFunction,
    /// Scatter scanned block sums back into tiles (phase 3).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) scan_scatter: CudaFunction,
    /// JPEG Huffman Phase 1 (intra-sequence sync).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) phase1_intra_sync: CudaFunction,
    /// JPEG Huffman Phase 2 (inter-sequence sync, bounded retry).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) phase2_inter_sync: CudaFunction,
    /// JPEG Huffman Phase 4 (re-decode + write final symbols).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) phase4_redecode: CudaFunction,
    /// JPEG-framed Phase 1.  Same kernel shape as
    /// [`Self::phase1_intra_sync`] but uses the real JPEG state
    /// machine (DC magnitude skip + AC run/size/EOB/ZRL framing).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) jpeg_phase1_intra_sync: CudaFunction,
    /// JPEG-framed Phase 2.  Same kernel shape as
    /// [`Self::phase2_inter_sync`] but uses the real JPEG state
    /// machine and the JPEG sync predicate (`block_in_mcu` + `z_in_block`).
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) jpeg_phase2_inter_sync: CudaFunction,
    /// JPEG-framed Phase 4 (re-decode + write final symbols).  Same
    /// kernel shape as [`Self::phase4_redecode`] but uses the JPEG
    /// state machine and inherits `(block_in_mcu, z_in_block)` from
    /// the predecessor's Phase-1 snapshot instead of `(c, z)`.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) jpeg_phase4_redecode: CudaFunction,
    /// IDCT + dequant + colour-conversion kernel (Phase 5).
    /// One 8×8×3 workgroup per 8×8 block; produces RGBA8 pixels from
    /// zigzag DCT coefficients and quantisation tables.
    #[cfg(feature = "gpu-jpeg-huffman")]
    pub(crate) idct_dequant_colour: CudaFunction,
}

/// An initialised CUDA context and compiled kernel set.
///
/// Create once per process with [`GpuCtx::init`] and share across threads via `Arc`.
///
/// # Concurrency
///
/// All methods take `&self` and are safe to call from multiple threads, but they
/// serialize on the single internal `CudaStream`: a call from thread B blocks until
/// thread A's `stream.synchronize()` returns.  This is correct for the current
/// architecture where GPU work is occasional (threshold-gated) and pages are the
/// unit of parallelism.  If image decode is ever parallelized within a page, replace
/// the shared stream with a `thread_local!` `Arc<CudaStream>` per rayon worker.
pub struct GpuCtx {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) kernels: GpuKernels,
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
        let ctx: Arc<CudaContext> = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let load =
            |ptx_src: &str, name: &str| -> Result<CudaFunction, Box<dyn std::error::Error>> {
                let ptx = Ptx::from_src(ptx_src);
                let module: Arc<CudaModule> = ctx.load_module(ptx)?;
                Ok(module.load_function(name)?)
            };

        Ok(Self {
            stream,
            kernels: GpuKernels {
                composite_rgba8: load(PTX_COMPOSITE, "composite_rgba8")?,
                apply_soft_mask: load(PTX_SOFT_MASK, "apply_soft_mask")?,
                aa_fill: load(PTX_AA_FILL, "aa_fill")?,
                icc_cmyk_matrix: load(PTX_ICC_CLUT, "icc_cmyk_matrix")?,
                icc_cmyk_clut: load(PTX_ICC_CLUT, "icc_cmyk_clut")?,
                #[cfg(feature = "cache")]
                blit_image: load(PTX_BLIT_IMAGE, "blit_image")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                scan_per_workgroup: load(PTX_BLELLOCH_SCAN, "scan_per_workgroup")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                scan_block_sums: load(PTX_BLELLOCH_SCAN, "scan_block_sums")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                scan_scatter: load(PTX_BLELLOCH_SCAN, "scan_scatter")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                phase1_intra_sync: load(PTX_PARALLEL_HUFFMAN, "phase1_intra_sync")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                phase2_inter_sync: load(PTX_PARALLEL_HUFFMAN, "phase2_inter_sync")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                phase4_redecode: load(PTX_PARALLEL_HUFFMAN, "phase4_redecode")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                jpeg_phase1_intra_sync: load(PTX_PARALLEL_HUFFMAN, "jpeg_phase1_intra_sync")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                jpeg_phase2_inter_sync: load(PTX_PARALLEL_HUFFMAN, "jpeg_phase2_inter_sync")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                jpeg_phase4_redecode: load(PTX_PARALLEL_HUFFMAN, "jpeg_phase4_redecode")?,
                #[cfg(feature = "gpu-jpeg-huffman")]
                idct_dequant_colour: load(PTX_IDCT_COLOR, "idct_dequant_colour")?,
            },
        })
    }

    /// The CUDA stream this context binds all kernel launches and
    /// device allocations to.
    ///
    /// Exposed so callers (e.g. the Phase 9 image cache) can share
    /// the same stream for upload + dispatch + download — single-
    /// stream serialisation eliminates the need for explicit
    /// cross-stream synchronisation.
    #[must_use]
    pub const fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
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
