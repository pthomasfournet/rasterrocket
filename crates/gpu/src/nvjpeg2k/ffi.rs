//! The CUDA runtime and nvJPEG2000 surface the decoder calls: the types
//! and constants of `nvjpeg2k.h` it uses, and the exception-safe shims.

use std::ffi::c_void;

use crate::cuda::CUstream;

// ── CUDA runtime API ──────────────────────────────────────────────────────────
//
// cudaMalloc / cudaFree are shared via crate::cuda::DeviceBuf.
// cudaMemcpy2D is used directly here for D→H component copies.

/// `cudaMemcpyDeviceToHost = 2` — direction flag for `cudaMemcpy2D`.
pub(super) const CUDAMEMCPY_D2H: i32 = 2;

unsafe extern "C" {
    pub(super) fn cudaMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
}

// ── nvJPEG2000 FFI declarations ───────────────────────────────────────────────
//
// Only the surface used by this module.  Avoids a bindgen dependency while
// keeping the binding surface minimal and auditable against nvjpeg2k.h.

/// nvJPEG2000 status code type.  0 is success; all other values are errors.
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
pub(super) type nvjpeg2kStatus_t = i32;

/// Opaque library handle.
#[repr(C)]
pub(super) struct Nvjpeg2kHandle_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
pub(super) type nvjpeg2kHandle_t = *mut Nvjpeg2kHandle_;

/// Opaque per-decode state (decoder scratch buffers, tiling state, etc.).
#[repr(C)]
pub(super) struct Nvjpeg2kDecodeState_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
pub(super) type nvjpeg2kDecodeState_t = *mut Nvjpeg2kDecodeState_;

/// Opaque bitstream handle (`nvjpeg2kStream_t`).
///
/// Called "stream" in the nvJPEG2000 API but this is the *bitstream* handle,
/// not a CUDA stream.  It holds the parsed headers and tile indices.  It is
/// reused across decodes: `nvjpeg2kStreamParse` overwrites its contents each
/// time it is called.
#[repr(C)]
pub(super) struct Nvjpeg2kStream_ {
    _opaque: [u8; 0],
}
#[expect(
    non_camel_case_types,
    reason = "C FFI type alias; must match the nvJPEG2000 ABI name"
)]
pub(super) type nvjpeg2kStream_t = *mut Nvjpeg2kStream_;

/// Image-level metadata returned by `nvjpeg2kStreamGetImageInfo`.
#[repr(C)]
pub(super) struct Nvjpeg2kImageInfo {
    pub(super) image_width: u32,
    pub(super) image_height: u32,
    pub(super) tile_width: u32,
    pub(super) tile_height: u32,
    pub(super) num_tiles_x: u32,
    pub(super) num_tiles_y: u32,
    pub(super) num_components: u32,
}

/// Per-component metadata returned by `nvjpeg2kStreamGetImageComponentInfo`.
#[repr(C)]
pub(super) struct Nvjpeg2kImageComponentInfo {
    pub(super) component_width: u32,
    pub(super) component_height: u32,
    /// Bit-depth of each sample (1–16).
    pub(super) precision: u8,
    /// Non-zero if samples are signed (two's-complement).
    pub(super) sgn: u8,
}

/// Output descriptor passed to `nvjpeg2kDecode`.
///
/// `pixel_data[i]` is a **device** pointer for component `i`.
/// `pitch_in_bytes[i]` is the row stride in bytes for component `i`.
/// `pixel_type` selects the output element type: [`NVJPEG2K_UINT8`] or
/// [`NVJPEG2K_UINT16`].
#[repr(C)]
pub(super) struct Nvjpeg2kImage {
    pub(super) pixel_data: *mut *mut c_void,
    pub(super) pitch_in_bytes: *mut usize,
    pub(super) pixel_type: i32,
    pub(super) num_components: u32,
}

/// `NVJPEG2K_UINT8` — one unsigned byte per sample per component.
pub(super) const NVJPEG2K_UINT8: i32 = 0;

/// `NVJPEG2K_UINT16` — one unsigned 16-bit integer per sample per component.
pub(super) const NVJPEG2K_UINT16: i32 = 1;

pub(super) const NVJPEG2K_STATUS_SUCCESS: nvjpeg2kStatus_t = 0;
/// Returned by the library for a malformed/truncated codestream, and also
/// returned by the shims when a C++ exception is caught during decode.
pub(super) const NVJPEG2K_STATUS_BAD_JPEG: nvjpeg2kStatus_t = 3;
/// Returned when the requested feature is not supported, and also used by the
/// C++ exception shims as the sentinel value for any caught exception.  Callers
/// that receive this code cannot distinguish the two cases; both are treated as
/// "fall back to the CPU decoder".
pub(super) const NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED: nvjpeg2kStatus_t = 9;

unsafe extern "C" {
    // ── Exception-safe shims (from shim/nvjpeg2k_shim.cpp) ───────────────────
    //
    // All nvjpeg2k functions that can throw C++ exceptions are routed through
    // the C++ shim which wraps each call in try/catch and returns
    // NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED on any exception.
    // C++ exceptions must not propagate through Rust FFI (undefined behaviour).
    //
    // Destroy functions are shimmed for defence-in-depth: they are called from
    // Drop (which must not unwind), so catching any unexpected exception is
    // critical even though the library documents them as non-throwing.

    pub(super) fn nvjpeg2k_shim_create_simple(handle: *mut nvjpeg2kHandle_t) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_decode_state_create(
        handle: nvjpeg2kHandle_t,
        decode_state: *mut nvjpeg2kDecodeState_t,
    ) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_stream_create(stream: *mut nvjpeg2kStream_t) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_stream_get_image_info(
        stream: nvjpeg2kStream_t,
        image_info: *mut Nvjpeg2kImageInfo,
    ) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_stream_get_image_component_info(
        stream: nvjpeg2kStream_t,
        component_info: *mut Nvjpeg2kImageComponentInfo,
        component_id: u32,
    ) -> nvjpeg2kStatus_t;

    /// Exception-safe wrapper for `nvjpeg2kStreamParse`.
    ///
    /// `save_metadata` and `save_stream` must both be 0 for standard decodes
    /// (metadata and tile data are not persistently cached).
    pub(super) fn nvjpeg2k_shim_stream_parse(
        handle: nvjpeg2kHandle_t,
        data: *const u8,
        length: usize,
        save_metadata: i32,
        save_stream: i32,
        stream: nvjpeg2kStream_t,
    ) -> nvjpeg2kStatus_t;

    /// Exception-safe wrapper for `nvjpeg2kDecode`.
    ///
    /// The caller must synchronise `cuda_stream` before reading the device
    /// buffers pointed to by `output.pixel_data`.
    pub(super) fn nvjpeg2k_shim_decode(
        handle: nvjpeg2kHandle_t,
        decode_state: nvjpeg2kDecodeState_t,
        j2k_stream: nvjpeg2kStream_t,
        output: *mut Nvjpeg2kImage,
        cuda_stream: CUstream,
    ) -> nvjpeg2kStatus_t;

    pub(super) fn nvjpeg2k_shim_decode_state_destroy(
        decode_state: nvjpeg2kDecodeState_t,
    ) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_stream_destroy(stream: nvjpeg2kStream_t) -> nvjpeg2kStatus_t;
    pub(super) fn nvjpeg2k_shim_destroy(handle: nvjpeg2kHandle_t) -> nvjpeg2kStatus_t;
}
