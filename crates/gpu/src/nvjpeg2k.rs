//! nvJPEG2000 GPU-accelerated JPEG 2000 decoding.
//!
//! Wraps the NVIDIA nvJPEG2000 library (installed at
//! `/usr/lib/x86_64-linux-gnu/libnvjpeg2k/13/`) to decode JPEG 2000 bitstreams
//! on the GPU, then copies the finished planar component planes back to host
//! memory via `cudaMemcpy2D`.
//!
//! PDF's `JPXDecode` filter embeds either raw `.j2k` codestreams or full JP2
//! container files; `nvjpeg2kStreamParse` auto-detects both.
//!
//! The CPU JPEG 2000 decoder is used as fallback for small images (below
//! the per-call dispatch threshold), when no GPU is available, and always
//! for inline images in the content stream (which are typically small
//! thumbnails not worth the `PCIe` dispatch overhead).
//!
//! # Key differences from nvJPEG (baseline JPEG)
//!
//! | | nvJPEG | nvJPEG2000 |
//! |---|---|---|
//! | Output memory | Host pinned (`cuMemAllocHost`) | Device (`cudaMalloc`) |
//! | Copy to host | `to_vec()` directly | `cudaMemcpy2D` per component |
//! | Image layout | Interleaved (RGBRGB…) | Planar — one ptr per component |
//! | Handle create | `nvjpegCreateEx(backend, …)` | `nvjpeg2kCreateSimple` |
//! | Backends | HARDWARE or DEFAULT | DEFAULT only |
//! | Per-decode objects | 1 state | `decode_state` + `stream` (bitstream handle) |
//! | Parse step | None | `nvjpeg2kStreamParse` before decode |
//!
//! # Thread safety
//!
//! [`NvJpeg2kDecoder`] is `Send` but not `Sync`: each thread must own its own
//! instance.  Creating one per thread is cheap — it retains the primary GPU
//! context (reference-counted) and creates one CUDA stream.

// All nvJPEG2000 calls are FFI through raw pointers.
#![cfg(feature = "nvjpeg2k")]

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr;

// ── CUDA driver API ───────────────────────────────────────────────────────────

use crate::cuda::{
    CUstream, DeviceBuf, cuCtxSetCurrent, cuDevicePrimaryCtxRelease, cuStreamDestroy,
    cuStreamSynchronize, init_primary_ctx_and_stream,
};

mod ffi;

use ffi::{
    CUDAMEMCPY_D2H, NVJPEG2K_STATUS_BAD_JPEG, NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED,
    NVJPEG2K_STATUS_SUCCESS, NVJPEG2K_UINT8, NVJPEG2K_UINT16, Nvjpeg2kImage,
    Nvjpeg2kImageComponentInfo, Nvjpeg2kImageInfo, cudaMemcpy2D, nvjpeg2k_shim_create_simple,
    nvjpeg2k_shim_decode, nvjpeg2k_shim_decode_state_create, nvjpeg2k_shim_decode_state_destroy,
    nvjpeg2k_shim_destroy, nvjpeg2k_shim_stream_create, nvjpeg2k_shim_stream_destroy,
    nvjpeg2k_shim_stream_get_image_component_info, nvjpeg2k_shim_stream_get_image_info,
    nvjpeg2k_shim_stream_parse, nvjpeg2kDecodeState_t, nvjpeg2kHandle_t, nvjpeg2kStream_t,
};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors returned by the nvJPEG2000 wrapper.
#[derive(Debug)]
pub enum NvJpeg2kError {
    /// nvJPEG2000 API returned a non-zero status.
    ///
    /// Codes from `nvjpeg2k.h`:
    /// - 1: `NVJPEG2K_STATUS_NOT_INITIALIZED`
    /// - 2: `NVJPEG2K_STATUS_INVALID_PARAMETER`
    /// - 3: `NVJPEG2K_STATUS_BAD_JPEG`
    /// - 4: `NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED`
    /// - 5: `NVJPEG2K_STATUS_ALLOCATOR_FAILURE`
    /// - 6: `NVJPEG2K_STATUS_EXECUTION_FAILED`
    /// - 7: `NVJPEG2K_STATUS_ARCH_MISMATCH`
    /// - 8: `NVJPEG2K_STATUS_INTERNAL_ERROR`
    /// - 9: `NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED`
    ///
    /// Note: code 9 is also returned by the C++ exception shim when a C++
    /// exception is thrown during decode (e.g. on a malformed codestream).
    /// A caller cannot distinguish that case from the library returning 9
    /// normally; both are treated as "fall back to CPU decoder."
    Nvjpeg2kStatus(i32),
    /// CUDA driver API error (`CUresult`).
    CudaError(i32),
    /// CUDA runtime API error (`cudaError_t`).
    ///
    /// From `cudaMalloc` or `cudaMemcpy2D`.  Common codes (shared with driver):
    /// - 2: `cudaErrorMemoryAllocation`
    /// - 35: `cudaErrorInsufficientDriver`
    /// - 100: `cudaErrorNoDevice`
    CudartError(i32),
    /// Component count is not 1 (Gray) or 3 (RGB); CMYK, Gray+Alpha, LAB, and
    /// N-channel images fall through to the CPU JPEG 2000 path.
    UnsupportedComponents(u32),
    /// At least one image component has a smaller width or height than the
    /// full image, indicating sub-sampled chroma (e.g. YUV 4:2:0).  Bare
    /// `nvjpeg2kDecode` does not upsample — fall through to CPU.
    SubSampledComponents,
    /// The decoded image has a zero dimension.
    ZeroDimension {
        /// Reported pixel width (0 when degenerate).
        width: u32,
        /// Reported pixel height (0 when degenerate).
        height: u32,
    },
    /// Arithmetic overflow computing pixel buffer size.
    Overflow,
}

impl std::fmt::Display for NvJpeg2kError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvjpeg2kStatus(code) => {
                let name = match *code {
                    1 => "NVJPEG2K_STATUS_NOT_INITIALIZED",
                    2 => "NVJPEG2K_STATUS_INVALID_PARAMETER",
                    3 => "NVJPEG2K_STATUS_BAD_JPEG",
                    4 => "NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED",
                    5 => "NVJPEG2K_STATUS_ALLOCATOR_FAILURE",
                    6 => "NVJPEG2K_STATUS_EXECUTION_FAILED",
                    7 => "NVJPEG2K_STATUS_ARCH_MISMATCH",
                    8 => "NVJPEG2K_STATUS_INTERNAL_ERROR",
                    NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED => {
                        "NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED"
                    }
                    _ => "NVJPEG2K_STATUS_UNKNOWN",
                };
                write!(f, "nvJPEG2000 error {code} ({name})")
            }
            Self::CudaError(code) => {
                let name = match *code {
                    1 => "CUDA_ERROR_INVALID_VALUE",
                    2 => "CUDA_ERROR_OUT_OF_MEMORY",
                    3 => "CUDA_ERROR_NOT_INITIALIZED",
                    35 => "CUDA_ERROR_INSUFFICIENT_DRIVER",
                    100 => "CUDA_ERROR_NO_DEVICE",
                    101 => "CUDA_ERROR_INVALID_DEVICE",
                    200 => "CUDA_ERROR_INVALID_IMAGE",
                    201 => "CUDA_ERROR_INVALID_CONTEXT",
                    400 => "CUDA_ERROR_INVALID_HANDLE",
                    700 => "CUDA_ERROR_ILLEGAL_ADDRESS",
                    _ => "CUDA_ERROR_UNKNOWN",
                };
                write!(f, "CUDA driver error {code} ({name})")
            }
            Self::CudartError(code) => write!(f, "CUDA runtime error {code}"),
            Self::UnsupportedComponents(n) => {
                write!(
                    f,
                    "unsupported JPEG 2000 component count {n} \
                     (only Gray/1 and RGB/3 are GPU-accelerated)"
                )
            }
            Self::SubSampledComponents => write!(
                f,
                "sub-sampled chroma components (e.g. YUV 4:2:0) — \
                 bare nvjpeg2kDecode does not upsample; falling back to CPU"
            ),
            Self::ZeroDimension { width, height } => {
                write!(f, "JPEG 2000 reported zero dimension {width}×{height}")
            }
            Self::Overflow => write!(f, "pixel buffer size overflow (image too large)"),
        }
    }
}

impl std::error::Error for NvJpeg2kError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, NvJpeg2kError>;

// ── Output type ───────────────────────────────────────────────────────────────

/// Colour space of the pixels returned by [`NvJpeg2kDecoder::decode_sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jpeg2kColorSpace {
    /// One sample per pixel (luma / grayscale).
    Gray,
    /// Three samples per pixel, interleaved R G B.
    Rgb,
}

/// A sample type nvJPEG2000 writes: `u8` (`NVJPEG2K_UINT8`) or `u16`
/// (`NVJPEG2K_UINT16`).
pub trait Jpeg2kSample: bytemuck::Pod + sealed::Sealed {
    /// The `nvjpeg2kImageType_t` of the sample.
    const PIXEL_TYPE: i32;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
}

impl Jpeg2kSample for u8 {
    const PIXEL_TYPE: i32 = NVJPEG2K_UINT8;
}

impl Jpeg2kSample for u16 {
    const PIXEL_TYPE: i32 = NVJPEG2K_UINT16;
}

/// Decoded JPEG 2000 image, host-resident, of samples `S`.
#[derive(Debug)]
pub struct DecodedJpeg2k<S = u8> {
    /// The samples, interleaved.  Layout matches
    /// [`color_space`](Self::color_space).
    pub data: Vec<S>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Colour space of the samples.
    pub color_space: Jpeg2kColorSpace,
    /// The precision in bits of each component, from its `SIZ` entry.
    pub precisions: Vec<u8>,
}

// ── NvJpeg2k ─────────────────────────────────────────────────────────────────

/// nvJPEG2000 decoder inner context.
///
/// Holds the library handle, per-decode state, and a reusable bitstream handle.
/// This is an implementation detail of [`NvJpeg2kDecoder`]; `pub(crate)` rather
/// than private so that integration tests in this crate can construct one
/// directly without going through the full CUDA context setup.
pub(crate) struct NvJpeg2k {
    handle: nvjpeg2kHandle_t,
    decode_state: nvjpeg2kDecodeState_t,
    /// Reusable bitstream handle.  `nvjpeg2kStreamParse` overwrites it each
    /// call, so it is safe to reuse across decodes.
    j2k_stream: nvjpeg2kStream_t,
}

// SAFETY: nvJPEG2000 guarantees that a handle used from one host thread at a
// time is safe; the internal CUDA stream serialises device-side work.
unsafe impl Send for NvJpeg2k {}

impl NvJpeg2k {
    /// Initialise nvJPEG2000 (single DEFAULT backend — no hardware variant).
    ///
    /// # Errors
    ///
    /// Returns an error if nvJPEG2000 initialisation or state allocation fails.
    pub(crate) fn new() -> Result<Self> {
        let mut handle: nvjpeg2kHandle_t = ptr::null_mut();
        // SAFETY: nvjpeg2k_shim_create_simple wraps nvjpeg2kCreateSimple in try/catch.
        let status = unsafe { nvjpeg2k_shim_create_simple(&raw mut handle) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !handle.is_null(),
            "nvjpeg2kCreateSimple succeeded but returned null handle"
        );

        let mut decode_state: nvjpeg2kDecodeState_t = ptr::null_mut();
        // SAFETY: handle is valid; shim wraps in try/catch.
        let status = unsafe { nvjpeg2k_shim_decode_state_create(handle, &raw mut decode_state) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            // SAFETY: handle is valid; decode_state was never initialised.
            let _ = unsafe { nvjpeg2k_shim_destroy(handle) };
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !decode_state.is_null(),
            "nvjpeg2kDecodeStateCreate succeeded but returned null state"
        );

        let mut j2k_stream: nvjpeg2kStream_t = ptr::null_mut();
        // SAFETY: no preconditions; shim wraps nvjpeg2kStreamCreate in try/catch.
        let status = unsafe { nvjpeg2k_shim_stream_create(&raw mut j2k_stream) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            // SAFETY: both are valid; clean up before returning.
            unsafe {
                let _ = nvjpeg2k_shim_decode_state_destroy(decode_state);
                let _ = nvjpeg2k_shim_destroy(handle);
            }
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        assert!(
            !j2k_stream.is_null(),
            "nvjpeg2kStreamCreate succeeded but returned null stream"
        );

        Ok(Self {
            handle,
            decode_state,
            j2k_stream,
        })
    }

    /// Decode a JPEG 2000 bitstream on `cu_stream` into samples `S`,
    /// returning host-resident pixels.
    ///
    /// Rejects images with component counts other than 1 (Gray) or 3 (RGB) —
    /// CMYK, LAB, and N-channel images fall through to the CPU path.
    ///
    /// Sub-sampled images (chroma components narrower/shorter than the image)
    /// are also rejected — bare `nvjpeg2kDecode` writes each component at its
    /// native dimensions without upsampling, which would break the planar→RGB
    /// interleave.  Call-site falls back to CPU for these.
    ///
    /// The samples are the components as reconstructed, at their own
    /// precision, `0 ..= 2^p − 1`; they are not scaled to the width of
    /// `S`.  A destination narrower than a component's precision is
    /// refused by the library (`NVJPEG2K_STATUS_INVALID_PARAMETER`).
    ///
    /// # Errors
    ///
    /// Returns an error if parsing, decode, or memory allocation fails.
    #[expect(
        clippy::too_many_lines,
        reason = "CUDA/FFI decode pipeline — phases are sequential and cannot be split without fragmenting the flow"
    )]
    fn decode<S: Jpeg2kSample>(
        &mut self,
        data: &[u8],
        cu_stream: CUstream,
    ) -> Result<DecodedJpeg2k<S>> {
        // Fast-path: the library returns BAD_JPEG (3) on empty input, but an FFI
        // call for a trivially-invalid input is unnecessary noise.
        if data.is_empty() {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(NVJPEG2K_STATUS_BAD_JPEG));
        }

        // ── Phase 1: parse bitstream headers ─────────────────────────────────
        // SAFETY: handle and j2k_stream are valid; data is a valid slice.
        // save_metadata = 0, save_stream = 0: standard decode (no caching).
        // Uses the shim to catch C++ exceptions thrown by the library.
        let status = unsafe {
            nvjpeg2k_shim_stream_parse(
                self.handle,
                data.as_ptr(),
                data.len(),
                0,
                0,
                self.j2k_stream,
            )
        };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }

        // ── Phase 2: read image-level info ────────────────────────────────────
        let mut info = Nvjpeg2kImageInfo {
            image_width: 0,
            image_height: 0,
            tile_width: 0,
            tile_height: 0,
            num_tiles_x: 0,
            num_tiles_y: 0,
            num_components: 0,
        };
        // SAFETY: j2k_stream is valid and freshly parsed; shim wraps in try/catch.
        let status = unsafe { nvjpeg2k_shim_stream_get_image_info(self.j2k_stream, &raw mut info) };
        if status != NVJPEG2K_STATUS_SUCCESS {
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }

        let w = info.image_width;
        let h = info.image_height;
        let nc = info.num_components;

        if w == 0 || h == 0 {
            return Err(NvJpeg2kError::ZeroDimension {
                width: w,
                height: h,
            });
        }

        // Cap component count before any allocation.  Legitimate JPEG 2000
        // images have at most a handful of components (≤ 16 per standard, ≤ 4
        // in practice for Gray / RGB / CMYK).  A corrupted header returning
        // u32::MAX would otherwise cause Vec::with_capacity to attempt a ~68 GB
        // allocation.
        if nc > 4 {
            return Err(NvJpeg2kError::UnsupportedComponents(nc));
        }

        // Only Gray (1) and RGB (3) are supported.  CMYK (4), LAB (3, but
        // typically with a different ICC profile), and other multi-channel
        // images fall through to the CPU decoder which handles them correctly.
        let color_space = match nc {
            1 => Jpeg2kColorSpace::Gray,
            3 => Jpeg2kColorSpace::Rgb,
            _ => return Err(NvJpeg2kError::UnsupportedComponents(nc)),
        };

        // ── Phase 3: per-component info + device allocation ───────────────────
        // nc is 1 or 3 here (enforced by the color_space match above).
        // The `as usize` cast is lossless: nc ≤ 3, which fits every target's usize.
        let mut comp_infos: Vec<Nvjpeg2kImageComponentInfo> = Vec::with_capacity(nc as usize);
        for i in 0..nc {
            let mut ci = Nvjpeg2kImageComponentInfo {
                component_width: 0,
                component_height: 0,
                precision: 0,
                sgn: 0,
            };
            // SAFETY: j2k_stream valid; i < nc; shim wraps in try/catch.
            let status = unsafe {
                nvjpeg2k_shim_stream_get_image_component_info(self.j2k_stream, &raw mut ci, i)
            };
            if status != NVJPEG2K_STATUS_SUCCESS {
                return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
            }
            // Guard degenerate component dimensions — a zero-width or zero-height
            // component would cause empty device allocations and then panic on
            // RGB interleave indexing.
            if ci.component_width == 0 || ci.component_height == 0 {
                return Err(NvJpeg2kError::ZeroDimension {
                    width: ci.component_width,
                    height: ci.component_height,
                });
            }
            // nvjpeg2kDecode (without nvjpeg2kDecodeParamsSetRGBOutput) writes each
            // component at its native dimensions — sub-sampled chroma components are
            // NOT upsampled to the full image size.  Attempting to interleave
            // mis-sized planes would index out of bounds.  Fall through to the CPU
            // JPEG 2000 path which handles sub-sampling correctly.
            if ci.component_width != w || ci.component_height != h {
                return Err(NvJpeg2kError::SubSampledComponents);
            }
            comp_infos.push(ci);
        }

        // Allocate one device buffer per component.
        // pitch = component_width (= image width, validated above) samples of
        // `S`.  The caller sets the pitch in the Nvjpeg2kImage struct; the
        // library writes exactly at the pitch we specify.
        //
        // comp_infos has exactly nc entries (the loop above pushes exactly nc times).
        // pixel_ptrs[i] aliases dev_bufs[i].ptr (a device pointer owned by DeviceBuf).
        // dev_bufs must not be dropped before nvjpeg2k_shim_decode completes.
        // Both live until end of this scope, so no use-after-free is possible.
        // Vec capacity == nc means no reallocation occurs during push, so
        // as_mut_ptr() into pixel_ptrs/pitches is stable across the decode call.
        let mut dev_bufs: Vec<DeviceBuf> = Vec::with_capacity(nc as usize);
        let mut pixel_ptrs: Vec<*mut c_void> = Vec::with_capacity(comp_infos.len());
        let mut pitches: Vec<usize> = Vec::with_capacity(comp_infos.len());
        for ci in &comp_infos {
            // component_width/height == image_width/height (guarded in Phase 3).
            // u32 values ≤ image dimensions; usize is at least 32 bits on all targets.
            let pitch = (ci.component_width as usize)
                .checked_mul(size_of::<S>())
                .ok_or(NvJpeg2kError::Overflow)?;
            let size = pitch
                .checked_mul(ci.component_height as usize)
                .ok_or(NvJpeg2kError::Overflow)?;
            let buf = DeviceBuf::alloc(size).map_err(NvJpeg2kError::CudartError)?;
            pixel_ptrs.push(buf.ptr);
            pitches.push(pitch);
            dev_bufs.push(buf);
        }

        // ── Phase 4: build Nvjpeg2kImage and decode ───────────────────────────
        // pixel_ptrs and pitches Vecs are pre-sized to nc (≤ 3); no reallocation
        // occurs between construction and nvjpeg2kDecode, so as_mut_ptr() is stable.

        // nc is u32; validated ≤ 3 in Phase 2.  u32 → u32 cast is a no-op but
        // the field type matches the C ABI declaration.
        let mut img = Nvjpeg2kImage {
            pixel_data: pixel_ptrs.as_mut_ptr(),
            pitch_in_bytes: pitches.as_mut_ptr(),
            pixel_type: S::PIXEL_TYPE,
            num_components: nc,
        };

        // SAFETY: handle, decode_state, j2k_stream are valid; img pointers are
        // valid for the duration of the call; cu_stream is owned by the caller.
        // nvjpeg2kDecode enqueues GPU work asynchronously — caller must sync.
        // Uses the shim to catch C++ exceptions thrown by the library.
        let status = unsafe {
            nvjpeg2k_shim_decode(
                self.handle,
                self.decode_state,
                self.j2k_stream,
                &raw mut img,
                cu_stream,
            )
        };

        // Synchronise unconditionally, even on error: partial GPU work may have
        // been enqueued and could write into `dev_bufs` after we free them unless
        // we wait for the stream to drain.
        // SAFETY: cu_stream is valid.
        let sync_code = unsafe { cuStreamSynchronize(cu_stream) };

        if status != NVJPEG2K_STATUS_SUCCESS {
            // Decode failed.  Report any additional stream sync failure via the
            // log facade so it is visible for debugging, but return the decode
            // error — it is the root cause, and the type does not support
            // carrying two errors.
            if sync_code != 0 {
                log::error!(
                    "nvjpeg2k: cuStreamSynchronize failed (code {sync_code}) \
                     while handling decode error (status {status})"
                );
            }
            return Err(NvJpeg2kError::Nvjpeg2kStatus(status));
        }
        if sync_code != 0 {
            return Err(NvJpeg2kError::CudaError(sync_code));
        }

        // ── Phase 5: copy component planes device → host, then interleave ─────
        Self::copy_and_interleave(&comp_infos, &dev_bufs, w, h, color_space)
    }

    /// Copy device planes to host and produce packed, interleaved samples.
    ///
    /// For Gray (1 component) the output is the single plane straight.
    /// For RGB (3 components) the planes are interleaved:
    /// `R[0] G[0] B[0]  R[1] G[1] B[1]  …`.
    fn copy_and_interleave<S: Jpeg2kSample>(
        comp_infos: &[Nvjpeg2kImageComponentInfo],
        dev_bufs: &[DeviceBuf],
        width: u32,
        height: u32,
        color_space: Jpeg2kColorSpace,
    ) -> Result<DecodedJpeg2k<S>> {
        let nc = comp_infos.len();

        // Invariant: nc matches color_space (1 for Gray, 3 for Rgb).
        // These assertions are cheap (one match per decode) and guard against
        // future refactoring bugs that could cause silent data corruption if the
        // interleave logic received the wrong number of planes.
        match color_space {
            Jpeg2kColorSpace::Gray => {
                assert_eq!(nc, 1, "Gray requires exactly 1 component plane");
            }
            Jpeg2kColorSpace::Rgb => {
                assert_eq!(nc, 3, "Rgb requires exactly 3 component planes");
            }
        }

        // Allocate host planes (one per component) and copy device → host.
        // All components have the same dimensions as the image (validated in
        // Phase 3).  The pitch passed to nvjpeg2kDecode is the component width
        // in samples of `S`, so the device layout is tightly packed.
        let mut host_planes: Vec<Vec<S>> = Vec::with_capacity(nc);
        for (ci, buf) in comp_infos.iter().zip(dev_bufs.iter()) {
            // component_width/height == image_width/height (Phase 3 invariant).
            let cw = ci.component_width as usize;
            let ch = ci.component_height as usize;
            let plane_len = cw.checked_mul(ch).ok_or(NvJpeg2kError::Overflow)?;
            let pitch = cw * size_of::<S>();
            let mut host = vec![S::zeroed(); plane_len];

            // SAFETY: buf.ptr is valid device memory of exactly pitch*ch bytes,
            // allocated in Phase 3 and written by nvjpeg2kDecode.
            // host.as_mut_ptr() is valid for plane_len samples == pitch*ch bytes.
            // The CUDA stream was synchronised by cuStreamSynchronize in decode()
            // immediately before this function is called — the GPU write is complete.
            let code = unsafe {
                cudaMemcpy2D(
                    host.as_mut_ptr().cast::<c_void>(),
                    pitch, // destination pitch (host, tightly packed)
                    buf.ptr.cast::<c_void>(),
                    pitch, // source pitch (the pitch we told nvjpeg2k to use)
                    pitch, // width in bytes to copy per row
                    ch,    // number of rows
                    CUDAMEMCPY_D2H,
                )
            };
            if code != 0 {
                return Err(NvJpeg2kError::CudartError(code));
            }
            host_planes.push(host);
        }

        let data = match color_space {
            // For Gray: exactly one plane; take it without shifting the Vec.
            Jpeg2kColorSpace::Gray => host_planes.swap_remove(0),
            Jpeg2kColorSpace::Rgb => {
                // Interleave three planar buffers (R, G, B) into packed RGBRGB…
                // All three planes hold width×height samples (validated in Phase 3).
                let total = (width as usize)
                    .checked_mul(height as usize)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or(NvJpeg2kError::Overflow)?;
                let mut interleaved = vec![S::zeroed(); total];
                let (plane_r, plane_g, plane_b) =
                    (&host_planes[0], &host_planes[1], &host_planes[2]);
                interleaved
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(plane_r.iter().zip(plane_g.iter().zip(plane_b.iter())))
                    .for_each(|(px, (&rv, (&gv, &bv)))| *px = [rv, gv, bv]);
                interleaved
            }
        };

        Ok(DecodedJpeg2k {
            data,
            width,
            height,
            color_space,
            precisions: comp_infos.iter().map(|c| c.precision).collect(),
        })
    }
}

impl Drop for NvJpeg2k {
    fn drop(&mut self) {
        // Destroy in reverse creation order:
        //   1. decode_state — per-decode scratch buffers (created last, freed first)
        //   2. j2k_stream   — bitstream handle (holds parsed tile/header data)
        //   3. handle       — top-level library handle (must outlive both above)
        //
        // nvJPEG2000 documentation requires nvjpeg2kDecodeStateDestroy before
        // nvjpeg2kDestroy.  nvjpeg2kStreamDestroy is independent but follows the
        // same reverse-creation convention for clarity.
        //
        // `new()` asserts non-null after each successful create, so these
        // pointers are always valid here.  The null checks are defence-in-depth
        // only and cannot fire through the public constructor.
        //
        // SAFETY: non-null handles are valid and exclusively owned; no aliases.
        unsafe {
            if !self.decode_state.is_null() {
                let _ = nvjpeg2k_shim_decode_state_destroy(self.decode_state);
            }
            if !self.j2k_stream.is_null() {
                let _ = nvjpeg2k_shim_stream_destroy(self.j2k_stream);
            }
            if !self.handle.is_null() {
                let _ = nvjpeg2k_shim_destroy(self.handle);
            }
        }
    }
}

// ── NvJpeg2kDecoder ───────────────────────────────────────────────────────────

/// Safe, self-contained GPU JPEG 2000 decoder.
///
/// Manages its own CUDA primary context and stream, independent of cudarc.
/// This is required because nvJPEG2000 captures the CUDA context current at
/// `nvjpeg2kCreateSimple` time; mixing cudarc's context management causes
/// `CUDA_ERROR_INVALID_CONTEXT (201)` on subsequent stream synchronisations.
///
/// The initialisation sequence mirrors the C reference:
/// ```text
/// cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain → cuCtxSetCurrent →
/// cuStreamCreate → NvJpeg2k::new() → (decode → cuStreamSynchronize)*
/// ```
///
/// `Send` but not `Sync`: each thread must own its own instance.
pub struct NvJpeg2kDecoder {
    /// Wrapped in `ManuallyDrop` so `Drop::drop` can explicitly run
    /// nvJPEG2000 teardown before releasing the primary context.
    dec: ManuallyDrop<NvJpeg2k>,
    /// Raw CUDA device handle (i32 ordinal).
    device: i32,
    /// Retained primary context for `device`.
    cu_ctx: *mut c_void,
    /// CUDA stream created in `cu_ctx`.
    stream: CUstream,
}

// SAFETY: NvJpeg2k is Send; raw pointers are only accessed from the owning
// thread (decode_sync takes &mut self, preventing concurrent access).
unsafe impl Send for NvJpeg2kDecoder {}

impl NvJpeg2kDecoder {
    /// Create an `NvJpeg2kDecoder` on the given GPU device (0-indexed).
    ///
    /// # Errors
    ///
    /// Returns an error if no CUDA device is available, or if any CUDA or
    /// nvJPEG2000 initialisation step fails.
    ///
    /// # Panics
    ///
    /// Panics if a CUDA or nvJPEG2000 function reports success but returns a
    /// null handle — that would indicate a driver bug.
    pub fn new(ordinal: usize) -> Result<Self> {
        // Steps 1–5: cuInit → cuDeviceGet → cuDevicePrimaryCtxRetain →
        // cuCtxSetCurrent → cuStreamCreate.  Stream is created before nvJPEG2000
        // init so it belongs to the primary context, not any internal context the
        // library might push.
        // CUDA_ERROR_INVALID_VALUE = 1: the ordinal is out of i32 range, which is
        // equivalent to an invalid device ordinal from the driver's perspective.
        let ordinal_i32 = i32::try_from(ordinal).map_err(|_| NvJpeg2kError::CudaError(1))?;
        let init = init_primary_ctx_and_stream(ordinal_i32)
            .map_err(|e| NvJpeg2kError::CudaError(e.code))?;

        // Step 6 — initialise nvJPEG2000.  nvjpeg2kCreateSimple captures the
        // current context (set in step 4), so handle and stream share a context.
        let dec = NvJpeg2k::new().inspect_err(|_| {
            // SAFETY: stream was created in init; no work is enqueued yet.
            // Rebind context first: nvjpeg2kCreateSimple may have pushed its own
            // internal context, leaving the primary context no longer current.
            unsafe {
                let _ = cuCtxSetCurrent(init.cu_ctx);
                let _ = cuStreamDestroy(init.stream);
                let _ = cuDevicePrimaryCtxRelease(init.device);
            }
        })?;

        Ok(Self {
            dec: ManuallyDrop::new(dec),
            device: init.device,
            cu_ctx: init.cu_ctx,
            stream: init.stream,
        })
    }

    /// Decode a JPEG 2000 bitstream synchronously into 8-bit samples
    /// (`NVJPEG2K_UINT8`), returning host-resident pixels.
    ///
    /// Rebinds the primary CUDA context to the calling thread (required in
    /// thread-pool scenarios where threads may change between calls), enqueues
    /// GPU decode work, blocks until complete, then copies results to host.
    ///
    /// # Errors
    ///
    /// - [`NvJpeg2kError::CudaError`] if `cuCtxSetCurrent` fails (driver error).
    /// - [`NvJpeg2kError::Nvjpeg2kStatus`] if the bitstream is invalid or
    ///   the library reports a decode failure.
    /// - [`NvJpeg2kError::UnsupportedComponents`] or
    ///   [`NvJpeg2kError::SubSampledComponents`] if the image geometry is not
    ///   supported by the GPU path (falls back to CPU at the call site).
    /// - [`NvJpeg2kError::CudartError`] if `cudaMemcpy2D` fails during D→H copy.
    pub fn decode_sync(&mut self, data: &[u8]) -> Result<DecodedJpeg2k> {
        self.decode_sync_as(data)
    }

    /// [`Self::decode_sync`] into 16-bit samples (`NVJPEG2K_UINT16`), for
    /// codestreams of more than 8 bits: each sample at its component's
    /// precision ([`DecodedJpeg2k::precisions`]).
    ///
    /// # Errors
    ///
    /// As [`Self::decode_sync`].
    pub fn decode_sync_u16(&mut self, data: &[u8]) -> Result<DecodedJpeg2k<u16>> {
        self.decode_sync_as(data)
    }

    /// [`Self::decode_sync`] into samples `S`.
    fn decode_sync_as<S: Jpeg2kSample>(&mut self, data: &[u8]) -> Result<DecodedJpeg2k<S>> {
        // Rebind primary context — necessary when called across thread-pool threads.
        // SAFETY: cu_ctx is valid for the lifetime of this struct.
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            return Err(NvJpeg2kError::CudaError(r));
        }

        // decode() handles both the async GPU work and the stream synchronise
        // internally (sync is required before cudaMemcpy2D).
        self.dec.decode(data, self.stream)
    }
}

impl Drop for NvJpeg2kDecoder {
    fn drop(&mut self) {
        // Teardown order is critical: nvjpeg2k resources must be destroyed while
        // the primary context is still current and our retain is held.
        //
        // SAFETY: cu_ctx is valid; we hold the primary-context retain.
        unsafe {
            // 1. Bind context so CUDA calls below land in the right context.
            let _ = cuCtxSetCurrent(self.cu_ctx);

            // 2. Destroy nvJPEG2000 handle/state/stream (ManuallyDrop — runs Drop<NvJpeg2k>).
            ManuallyDrop::drop(&mut self.dec);

            // 3. Destroy the CUDA stream.
            if !self.stream.is_null() {
                let _ = cuStreamDestroy(self.stream);
            }

            // 4. Release our retain of the primary context.
            let _ = cuDevicePrimaryCtxRelease(self.device);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
