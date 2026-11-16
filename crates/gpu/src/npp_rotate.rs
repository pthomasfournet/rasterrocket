//! GPU rotation via CUDA NPP `nppiRotate_8u_C1R_Ctx` and
//! `nppiRotate_16u_C1R_Ctx`.
//!
//! Rotates a single-channel 8- or 16-bit grayscale image by a given angle
//! using NVIDIA NPP hardware bilinear interpolation.  Expected throughput on
//! RTX 5070: ~0.3–0.5 ms for a 2550×3300 8-bit image (texture fill rate
//! limited).
//!
//! # Angle convention
//!
//! [`rotate_gray8`] and [`rotate_gray16`] take a **clockwise-positive**
//! angle in degrees, matching
//! the deskew pipeline convention.  NPP also uses clockwise-positive in
//! image-space (Y axis points down), so the angle is passed through unchanged.
//! The post-rotation shift is computed to keep the image centred on its
//! original centre point.
//!
//! # Thread safety
//!
//! The internal `NppRotator` (private) is `Send` but not `Sync`.  The
//! recommended entry points are [`rotate_gray8`] and [`rotate_gray16`], which
//! manage a per-thread
//! instance via `thread_local!` and shields callers from the thread-binding
//! constraint.  CUDA primary contexts are thread-bound and a `NppRotator`
//! must only be used from the thread that constructed it.

#![cfg(feature = "gpu-deskew")]

use std::cell::RefCell;
use std::ffi::c_void;

use crate::cuda::{
    CUstream as CudaHandle, DeviceBuf, cuCtxSetCurrent, cuDevicePrimaryCtxRelease, cuStreamDestroy,
    cuStreamSynchronize, init_primary_ctx_and_stream,
};

// ── CUDA runtime API (libcudart.so) ───────────────────────────────────────────

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

unsafe extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: CudaHandle,
    ) -> i32;
    fn cudaMemsetAsync(dev_ptr: *mut c_void, value: i32, count: usize, stream: CudaHandle) -> i32;
}

// ── NPP core (libnppc.so) ─────────────────────────────────────────────────────

/// NPP status code: 0 = success, negative = error, positive = warning.
type NppStatus = i32;
const NPP_SUCCESS: i32 = 0;
/// NPP warning: rotated source quad does not intersect the destination ROI;
/// **no pixels were written**.  Value confirmed in `nppdefs.h`.
const NPP_WRONG_INTERSECTION_QUAD_WARNING: i32 = 30;

/// NPP stream context; populated by [`nppGetStreamContext`] after
/// [`nppSetStream`] points NPP at the correct stream.
#[repr(C)]
#[derive(Default)]
struct NppStreamContext {
    h_stream: CudaHandle,
    cuda_device_id: i32,
    multi_processor_count: i32,
    max_threads_per_multi_processor: i32,
    max_threads_per_block: i32,
    shared_mem_per_block: usize,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    stream_flags: u32,
    reserved0: i32,
}

unsafe extern "C" {
    // No warning codes are defined for these two calls; any non-zero is an error.
    fn nppSetStream(stream: CudaHandle) -> NppStatus;
    fn nppGetStreamContext(ctx: *mut NppStreamContext) -> NppStatus;
}

// ── NPP geometry (libnppig.so) ────────────────────────────────────────────────

/// 2-D integer size (width × height).
#[repr(C)]
struct NppiSize {
    width: i32,
    height: i32,
}

/// 2-D integer rectangle: top-left corner (x, y) and dimensions.
#[repr(C)]
#[derive(Clone, Copy)]
struct NppiRect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

/// Bilinear interpolation mode (`NPPI_INTER_LINEAR = 2`).
const NPPI_INTER_LINEAR: i32 = 2;

unsafe extern "C" {
    fn nppiRotate_16u_C1R_Ctx(
        p_src: *const u16,
        src_size: NppiSize,
        src_step: i32,
        src_roi: NppiRect,
        p_dst: *mut u16,
        dst_step: i32,
        dst_roi: NppiRect,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
        interpolation: i32,
        npp_ctx: NppStreamContext,
    ) -> NppStatus;
    fn nppiRotate_8u_C1R_Ctx(
        p_src: *const u8,
        src_size: NppiSize,
        src_step: i32,
        src_roi: NppiRect,
        p_dst: *mut u8,
        dst_step: i32,
        dst_roi: NppiRect,
        angle: f64,
        shift_x: f64,
        shift_y: f64,
        interpolation: i32,
        npp_ctx: NppStreamContext,
    ) -> NppStatus;
}

/// A gray sample NPP rotates, `u8` or `u16`, and its NPP rotation.
trait NppSample: Copy + Default {
    /// The name of the NPP call.
    const ROTATE: &'static str;

    /// The NPP rotation of the whole image from `p_src`, of the size and
    /// step `src`, to the pointer and step `dst`, steps in bytes, by
    /// `angle` degrees and then `shift`, with bilinear interpolation; the
    /// NPP status.
    ///
    /// # Safety
    ///
    /// As the NPP call: both pointers are device allocations of the
    /// image's size at the steps given, and `npp_ctx` is the context of
    /// the current device and stream.
    unsafe fn rotate(
        p_src: *const Self,
        src: (NppiSize, i32),
        dst: (*mut Self, i32),
        angle: f64,
        shift: (f64, f64),
        npp_ctx: NppStreamContext,
    ) -> i32;
}

/// The rectangle of the whole image `size`.
const fn full_roi(size: &NppiSize) -> NppiRect {
    NppiRect {
        x: 0,
        y: 0,
        width: size.width,
        height: size.height,
    }
}

impl NppSample for u8 {
    const ROTATE: &'static str = "nppiRotate_8u_C1R_Ctx";

    unsafe fn rotate(
        p_src: *const Self,
        (size, src_step): (NppiSize, i32),
        (p_dst, dst_step): (*mut Self, i32),
        angle: f64,
        (shift_x, shift_y): (f64, f64),
        npp_ctx: NppStreamContext,
    ) -> i32 {
        let roi = full_roi(&size);
        // SAFETY: the caller's contract.
        unsafe {
            nppiRotate_8u_C1R_Ctx(
                p_src,
                size,
                src_step,
                roi,
                p_dst,
                dst_step,
                roi,
                angle,
                shift_x,
                shift_y,
                NPPI_INTER_LINEAR,
                npp_ctx,
            )
        }
    }
}

impl NppSample for u16 {
    const ROTATE: &'static str = "nppiRotate_16u_C1R_Ctx";

    unsafe fn rotate(
        p_src: *const Self,
        (size, src_step): (NppiSize, i32),
        (p_dst, dst_step): (*mut Self, i32),
        angle: f64,
        (shift_x, shift_y): (f64, f64),
        npp_ctx: NppStreamContext,
    ) -> i32 {
        let roi = full_roi(&size);
        // SAFETY: the caller's contract.
        unsafe {
            nppiRotate_16u_C1R_Ctx(
                p_src,
                size,
                src_step,
                roi,
                p_dst,
                dst_step,
                roi,
                angle,
                shift_x,
                shift_y,
                NPPI_INTER_LINEAR,
                npp_ctx,
            )
        }
    }
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error returned by GPU rotation.
#[derive(Debug)]
pub struct NppRotateError(pub(crate) String);

impl std::fmt::Display for NppRotateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NppRotateError {}

type Result<T> = std::result::Result<T, NppRotateError>;

// ── NppRotator ────────────────────────────────────────────────────────────────

/// Owns a CUDA primary context, a stream, and the device source and
/// destination buffers for NPP rotation calls.
///
/// Reuse across pages — context and stream creation are expensive.  The
/// device buffers persist between calls and are reallocated only when a
/// larger image arrives, so a rotation of an image no larger than any
/// before it costs one upload, one fill, the NPP kernel and one download,
/// all queued on the stream, and a single stream synchronisation.
///
/// # Safety
///
/// `NppRotator` must only be used from the thread that called [`NppRotator::new`].
/// CUDA primary contexts are thread-bound; using the same instance from a
/// different thread produces `CUDA_ERROR_INVALID_CONTEXT`.  The recommended
/// entry point is [`rotate_gray8`], which enforces this via `thread_local!`.
pub(crate) struct NppRotator {
    cu_ctx: *mut c_void,
    device: i32,
    stream: CudaHandle,
    d_src: GrowBuf,
    d_dst: GrowBuf,
}

/// A device allocation reused across calls and replaced by a larger one
/// when a request exceeds its capacity.
#[derive(Default)]
struct GrowBuf {
    buf: Option<DeviceBuf>,
    capacity: usize,
}

impl GrowBuf {
    /// A device buffer of at least `len` bytes.  Its contents are
    /// unspecified.
    fn get(&mut self, len: usize) -> std::result::Result<&DeviceBuf, i32> {
        if self.capacity < len || self.buf.is_none() {
            // Free before allocating so the old and new buffers never
            // coexist in device memory.
            self.buf = None;
            self.capacity = 0;
            self.buf = Some(DeviceBuf::alloc(len)?);
            self.capacity = len;
        }
        Ok(self.buf.as_ref().expect("allocated above"))
    }
}

// SAFETY: NppRotator is accessed exclusively through the per-thread
// `thread_local!` slot in `rotate_gray8`; it is never shared across threads.
unsafe impl Send for NppRotator {}

impl NppRotator {
    /// Initialise CUDA device 0 and create a stream for NPP use.
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if any CUDA driver call fails (e.g. no GPU
    /// is present, or the driver is not loaded).
    ///
    /// # Panics
    ///
    /// Panics if a CUDA function reports success but returns a null handle —
    /// that indicates a driver bug.
    pub(crate) fn new() -> Result<Self> {
        let init = init_primary_ctx_and_stream(0)
            .map_err(|e| NppRotateError(format!("{}: code {}", e.step, e.code)))?;
        Ok(Self {
            cu_ctx: init.cu_ctx,
            device: init.device,
            stream: init.stream,
            d_src: GrowBuf::default(),
            d_dst: GrowBuf::default(),
        })
    }

    /// Rotate a single-channel gray image of samples `T` by `angle_deg`
    /// degrees (clockwise positive) via NPP bilinear.
    ///
    /// - `src`: samples; must be at least `src_stride * height` of them.
    /// - `src_stride`: samples per row in `src`; must be ≥ `width`.
    /// - `width`, `height`: image dimensions in pixels; both must be non-zero
    ///   and fit in `i32` (i.e. ≤ 2 147 483 647).
    /// - `angle_deg`: rotation angle in degrees (clockwise positive); must be
    ///   finite (not NaN or ±infinity).
    ///
    /// Returns tightly packed output samples (`width * height`, no padding).
    ///
    /// # Errors
    ///
    /// Returns [`NppRotateError`] if any precondition is violated or any CUDA
    /// or NPP call fails.
    fn rotate<T: NppSample>(
        &mut self,
        src: &[T],
        src_stride: usize,
        width: u32,
        height: u32,
        angle_deg: f32,
    ) -> Result<Vec<T>> {
        let RotateLayout {
            w,
            h,
            src_step,
            dst_step,
            dst_len,
            dst_bytes,
        } = RotateLayout::of::<T>(src.len(), src_stride, (width, height), angle_deg)?;
        let src_bytes = size_of_val(src);

        // ── Computation ──────────────────────────────────────────────────────

        let (n_angle, shift_x, shift_y) = centre_pivot_shift(w, h, angle_deg);

        // Restore CUDA context (rayon workers may interleave threads between calls).
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            return Err(NppRotateError(format!(
                "cuCtxSetCurrent (rotate): code {r}"
            )));
        }

        let stream = self.stream;
        let npp_ctx = self.build_npp_ctx()?;
        let d_src = self
            .d_src
            .get(src_bytes)
            .map_err(|code| NppRotateError(format!("cudaMalloc(src, {src_bytes}): code {code}")))?;
        let d_dst = self
            .d_dst
            .get(dst_bytes)
            .map_err(|code| NppRotateError(format!("cudaMalloc(dst, {dst_bytes}): code {code}")))?;

        // nppiRotate writes only the pixels covered by the rotated source
        // quad; pre-fill the destination with white (every byte 0xFF, the
        // largest sample of either depth) so the uncovered corner wedges
        // match the CPU path's white background instead of exposing stale
        // device memory.
        // SAFETY: d_dst holds at least dst_bytes bytes.
        let r = unsafe { cudaMemsetAsync(d_dst.ptr, 0xFF, dst_bytes, stream) };
        if r != 0 {
            return Err(NppRotateError(format!("cudaMemsetAsync(dst): code {r}")));
        }

        // H → D upload.  From pageable memory the call returns once `src`
        // has been staged, so `src` is not read after it returns.
        // SAFETY: src is valid for `src_bytes` bytes and d_src holds at
        // least that many; src.len() >= src_stride * h was validated above,
        // so NPP does not read past the copied bytes.
        let r = unsafe {
            cudaMemcpyAsync(
                d_src.ptr,
                src.as_ptr().cast(),
                src_bytes,
                CUDA_MEMCPY_H2D,
                stream,
            )
        };
        if r != 0 {
            return Err(NppRotateError(format!("cudaMemcpyAsync H2D: code {r}")));
        }

        Self::call_nppi_rotate::<T>(
            (d_src, src_step),
            (d_dst, dst_step),
            (w, h),
            (n_angle, shift_x, shift_y),
            npp_ctx,
        )?;

        // D → H download, ordered after the kernel on the stream.
        let mut dst_pixels = vec![T::default(); dst_len];
        // SAFETY: d_dst holds at least dst_bytes bytes; dst_pixels has
        // dst_bytes bytes and outlives the synchronisation below.
        let r = unsafe {
            cudaMemcpyAsync(
                dst_pixels.as_mut_ptr().cast(),
                d_dst.ptr,
                dst_bytes,
                CUDA_MEMCPY_D2H,
                stream,
            )
        };
        if r != 0 {
            return Err(NppRotateError(format!("cudaMemcpyAsync D2H: code {r}")));
        }
        // SAFETY: stream is the live stream created in `new`.
        let r = unsafe { cuStreamSynchronize(stream) };
        if r != 0 {
            return Err(NppRotateError(format!("cuStreamSynchronize: code {r}")));
        }

        Ok(dst_pixels)
    }

    /// Build the NPP stream context for the current stream.
    fn build_npp_ctx(&self) -> Result<NppStreamContext> {
        // nppSetStream + nppGetStreamContext fills in device properties,
        // avoiding manual cudaGetDeviceProperties calls.
        // No warning codes are defined for either call; any non-zero is a hard error.
        let r = unsafe { nppSetStream(self.stream) };
        if r != NPP_SUCCESS {
            return Err(NppRotateError(format!("nppSetStream: code {r}")));
        }
        let mut npp_ctx = NppStreamContext::default();
        let r = unsafe { nppGetStreamContext(&raw mut npp_ctx) };
        if r != NPP_SUCCESS {
            return Err(NppRotateError(format!("nppGetStreamContext: code {r}")));
        }
        Ok(npp_ctx)
    }

    /// Launch the NPP rotation of `T` over the whole `w × h` image, steps
    /// in bytes.
    ///
    /// Precondition: `w`, `h` and both steps fit in `i32` (caller must
    /// validate before calling — see [`NppRotator::rotate`]).
    fn call_nppi_rotate<T: NppSample>(
        (d_src, src_step): (&DeviceBuf, usize),
        (d_dst, dst_step): (&DeviceBuf, usize),
        (w, h): (usize, usize),
        (n_angle, shift_x, shift_y): (f64, f64, f64),
        npp_ctx: NppStreamContext,
    ) -> Result<()> {
        // All dimension/step values were range-checked in rotate() before this call.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "bounds checked against i32::MAX in rotate() before this call"
        )]
        let (wi, hi, src_step, dst_step) = (w as i32, h as i32, src_step as i32, dst_step as i32);
        let size = NppiSize {
            width: wi,
            height: hi,
        };
        // SAFETY: d_src/d_dst are valid device allocations sized for the pixel data;
        //         size and steps are consistent with width/height; npp_ctx was filled
        //         by nppGetStreamContext for the current device and stream.
        let status = unsafe {
            T::rotate(
                d_src.ptr.cast::<T>(),
                (size, src_step),
                (d_dst.ptr.cast::<T>(), dst_step),
                n_angle,
                (shift_x, shift_y),
                npp_ctx,
            )
        };

        if status == NPP_WRONG_INTERSECTION_QUAD_WARNING {
            // No pixels were written; the rotated quad fell entirely outside the
            // destination ROI.  This should not occur for small deskew angles
            // (≤ ±7°) on non-trivial images, but log it so it is diagnosable.
            log::warn!(
                "npp_rotate: {} returned NPP_WRONG_INTERSECTION_QUAD_WARNING (30); \
                 destination image will be all white",
                T::ROTATE
            );
        } else if status < 0 {
            return Err(NppRotateError(format!("{}: status {status}", T::ROTATE)));
        }
        Ok(())
    }
}

impl Drop for NppRotator {
    fn drop(&mut self) {
        // The device buffers belong to `cu_ctx`; free them while it is
        // current and retained.
        // SAFETY: cu_ctx was retained in new() and is released below.
        let r = unsafe { cuCtxSetCurrent(self.cu_ctx) };
        if r != 0 {
            log::warn!("npp_rotate: cuCtxSetCurrent (drop): code {r}");
        }
        self.d_src = GrowBuf::default();
        self.d_dst = GrowBuf::default();
        // SAFETY: stream and cu_ctx were initialised in new(); this is the unique owner.
        unsafe {
            let r = cuStreamDestroy(self.stream);
            if r != 0 {
                log::warn!("npp_rotate: cuStreamDestroy failed: code {r}");
            }
            let r = cuDevicePrimaryCtxRelease(self.device);
            if r != 0 {
                log::warn!("npp_rotate: cuDevicePrimaryCtxRelease failed: code {r}");
            }
        }
    }
}

/// The validated geometry of one rotation: dimensions, steps in bytes, and
/// the output's length in samples and bytes.
struct RotateLayout {
    w: usize,
    h: usize,
    src_step: usize,
    dst_step: usize,
    dst_len: usize,
    dst_bytes: usize,
}

impl RotateLayout {
    /// The layout of the rotation of `src_len` samples of `T`, rows
    /// `src_stride` samples apart, of a `width × height` image by
    /// `angle_deg`, tightly packed out; an error for an angle that is not
    /// finite, an empty image, a source too short, or a dimension or step
    /// beyond NPP's `i32`.
    fn of<T>(
        src_len: usize,
        src_stride: usize,
        (width, height): (u32, u32),
        angle_deg: f32,
    ) -> Result<Self> {
        let fits_i32 = |b: usize| i32::try_from(b).is_ok();
        if !angle_deg.is_finite() {
            return Err(NppRotateError(format!(
                "angle_deg is not finite: {angle_deg}"
            )));
        }
        let w = width as usize;
        let h = height as usize;
        let size = size_of::<T>();
        if w == 0 || h == 0 {
            return Err(NppRotateError(format!(
                "degenerate image: {width}×{height}"
            )));
        }
        // NPP takes dimensions and steps in bytes as i32.  (PDF pages are
        // well under this limit, but the public API must enforce it.)
        if !fits_i32(w) || !fits_i32(h) {
            return Err(NppRotateError(format!(
                "image {width}×{height} exceeds NPP i32 dimension limit"
            )));
        }
        if src_stride < w {
            return Err(NppRotateError(format!(
                "src_stride {src_stride} < width {w}"
            )));
        }
        let src_step = src_stride
            .checked_mul(size)
            .filter(|&b| fits_i32(b))
            .ok_or_else(|| {
                NppRotateError(format!(
                    "src_stride {src_stride} exceeds NPP i32 step limit"
                ))
            })?;
        let required_src = src_stride
            .checked_mul(h)
            .ok_or_else(|| NppRotateError("src_stride * height overflows usize".into()))?;
        if src_len < required_src {
            return Err(NppRotateError(format!(
                "src too short: {src_len} samples for stride {src_stride} × {h} rows"
            )));
        }
        let dst_step = w
            .checked_mul(size)
            .filter(|&b| fits_i32(b))
            .ok_or_else(|| NppRotateError(format!("width {w} exceeds NPP i32 step limit")))?;
        let dst_len = w
            .checked_mul(h)
            .ok_or_else(|| NppRotateError("width * height overflows usize".into()))?;
        let dst_bytes = dst_len
            .checked_mul(size)
            .ok_or_else(|| NppRotateError("width * height overflows usize".into()))?;
        Ok(Self {
            w,
            h,
            src_step,
            dst_step,
            dst_len,
            dst_bytes,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the NPP angle and post-rotation shift needed to rotate a `w × h`
/// image CW by `angle_deg` around its centre.
///
/// NPP uses a CW-positive angle convention (image-space: Y axis points down).
/// The post-rotation shift keeps the image anchored to its original centre:
///
/// ```text
/// nAngle  = +θ                            (CW-positive, same as deskew convention)
/// shiftX  = cx − cx·cos(θ) − cy·sin(θ)
/// shiftY  = cy + cx·sin(θ) − cy·cos(θ)
/// ```
///
/// Derivation: NPP applies `dst = R_CW(θ)·src + shift`.  Setting the centre
/// point `(cx, cy)` to map to itself gives `shift = (I − R_CW(θ))·(cx, cy)`.
fn centre_pivot_shift(w: usize, h: usize, angle_deg: f32) -> (f64, f64, f64) {
    let theta = f64::from(angle_deg).to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // w and h are ≤ i32::MAX (< 2^31); f64 mantissa is 52 bits; exact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "dimensions validated ≤ i32::MAX < 2^31; f64 mantissa is 52 bits; exact"
    )]
    let (cx, cy) = ((w as f64 - 1.0) * 0.5, (h as f64 - 1.0) * 0.5);
    let n_angle = f64::from(angle_deg); // NPP is CW-positive (image-space Y-down)
    let shift_x = cy.mul_add(-sin_t, cx.mul_add(-cos_t, cx));
    let shift_y = cx.mul_add(sin_t, cy.mul_add(-cos_t, cy));
    (n_angle, shift_x, shift_y)
}

// ── Thread-local entry point ──────────────────────────────────────────────────

/// Per-thread rotator state.
enum RotatorSlot {
    /// Not yet initialised.
    Uninit,
    /// Initialisation failed permanently; do not retry.
    Failed,
    /// Ready to use.
    Ready(NppRotator),
}

thread_local! {
    static ROTATOR: RefCell<RotatorSlot> = const { RefCell::new(RotatorSlot::Uninit) };
}

/// Rotate a single-channel 8-bit grayscale image by `angle_deg` (clockwise
/// positive) on the GPU.
///
/// - `src`: pixel data; must be at least `src_stride * height` bytes.
/// - `src_stride`: bytes per row in `src`; must be ≥ `width`.
/// - `width`, `height`: image dimensions in pixels; both must be non-zero.
/// - `angle_deg`: rotation angle in degrees (clockwise positive); must be finite.
///
/// Returns tightly packed output pixels (`width * height` bytes).
///
/// On the first call per thread, initialises the CUDA context and stream.
/// If GPU initialisation fails, logs a warning and returns `Err` — the slot
/// is permanently marked as failed and no further init is attempted.
/// If the NPP kernel fails on a subsequent call, returns `Err` without
/// poisoning the slot (a transient error may be recoverable).
///
/// # Errors
///
/// Returns [`NppRotateError`] if GPU initialisation or any CUDA/NPP call fails.
pub fn rotate_gray8(
    src: &[u8],
    src_stride: usize,
    width: u32,
    height: u32,
    angle_deg: f32,
) -> std::result::Result<Vec<u8>, NppRotateError> {
    rotate_gray(src, src_stride, width, height, angle_deg)
}

/// [`rotate_gray8`] of a 16-bit image: `src_stride` in samples, the
/// output `width * height` samples.
///
/// # Errors
///
/// As [`rotate_gray8`].
pub fn rotate_gray16(
    src: &[u16],
    src_stride: usize,
    width: u32,
    height: u32,
    angle_deg: f32,
) -> std::result::Result<Vec<u16>, NppRotateError> {
    rotate_gray(src, src_stride, width, height, angle_deg)
}

/// [`rotate_gray8`] at the depth of `T`, through the thread's rotator.
fn rotate_gray<T: NppSample>(
    src: &[T],
    src_stride: usize,
    width: u32,
    height: u32,
    angle_deg: f32,
) -> std::result::Result<Vec<T>, NppRotateError> {
    ROTATOR.with(|cell| {
        let mut slot = cell.borrow_mut();

        if matches!(*slot, RotatorSlot::Uninit) {
            match NppRotator::new() {
                Ok(r) => *slot = RotatorSlot::Ready(r),
                Err(e) => {
                    log::warn!("npp_rotate: GPU init failed ({e}); later rotations on this thread fail without retrying");
                    *slot = RotatorSlot::Failed;
                    return Err(e);
                }
            }
        }

        match &mut *slot {
            RotatorSlot::Ready(r) => r.rotate(src, src_stride, width, height, angle_deg),
            RotatorSlot::Failed => Err(NppRotateError("GPU init previously failed".into())),
            RotatorSlot::Uninit => unreachable!("slot was just initialised"),
        }
    })
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;

    /// Destination pixels the rotated source quad does not cover must come
    /// back as white (255), matching the CPU rotate path's background fill.
    ///
    /// The 0° priming call makes a stale-VRAM regression deterministic: it
    /// writes an all-black destination of the same size, which the thread's
    /// rotator reuses as the next same-size destination, so an uncleared
    /// destination reads 0 at the corners.
    #[test]
    fn uncovered_corners_are_white() {
        const W: u32 = 512;
        const H: u32 = 512;
        let black = vec![0u8; (W * H) as usize];

        let _primed = rotate_gray8(&black, W as usize, W, H, 0.0).expect("priming rotate");
        let out = rotate_gray8(&black, W as usize, W, H, 3.0).expect("rotate");

        let (w, h) = (W as usize, H as usize);
        // At 3° about the centre, all four destination corners lie outside
        // the rotated source quad.
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            assert_eq!(
                out[y * w + x],
                255,
                "uncovered corner ({x}, {y}) must be white"
            );
        }

        let black = vec![0u16; (W * H) as usize];
        let _primed = rotate_gray16(&black, W as usize, W, H, 0.0).expect("priming rotate");
        let out = rotate_gray16(&black, W as usize, W, H, 3.0).expect("rotate");
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            assert_eq!(
                out[y * w + x],
                u16::MAX,
                "uncovered 16-bit corner ({x}, {y}) must be white"
            );
        }
        assert_eq!(
            out[(h / 2) * w + w / 2],
            0,
            "the 16-bit centre is the source's"
        );
    }
}
