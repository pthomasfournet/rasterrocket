//! The scanline-AA fill as one backend page: the one sequence every caller
//! of the fill kernel runs, on either backend.

use std::sync::{Mutex, PoisonError};

use super::params::{AaFillParams, aa_fill_invariant};
use super::{BackendError, GpuBackend, Result, reject_zero_size};
use crate::fill::{AaFillRect, AaScanTable};

/// A device buffer and, unless the host reaches the device buffer in place
/// ([`GpuBackend::device_memory_is_host_mapped`]), a host buffer of at least its length that
/// transfers go through.
struct Staged<B: GpuBackend> {
    host: Option<B::HostBuffer>,
    device: B::DeviceBuffer,
}

/// Reusable host and device buffers for [`aa_fill_page`].
///
/// Each of the three transfers of a fill (intersections, row offsets,
/// counts) keeps one device buffer and, on a device whose buffers the host
/// cannot reach in place, one host buffer, grown to the largest fill seen
/// and reused by later fills, so a fill allocates only when it is larger
/// than every earlier one.  The buffers are freed when the scratch
/// is dropped; [`aa_fill_page`] returns only after its page has completed,
/// so no device work references them then.
pub struct AaFillScratch<B: GpuBackend> {
    intersects: Option<Staged<B>>,
    row_starts: Option<Staged<B>>,
    coverage: Option<Staged<B>>,
}

impl<B: GpuBackend> AaFillScratch<B> {
    /// An empty scratch; the first fill allocates.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intersects: None,
            row_starts: None,
            coverage: None,
        }
    }
}

impl<B: GpuBackend> Default for AaFillScratch<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Scratches lent to the fills a backend runs ([`aa_fill`]), one per
/// concurrent fill.
///
/// Owned by the backend ([`GpuBackend::aa_fill_scratch`]), so its buffers
/// belong to the device they were allocated on and live as long as the
/// backend: a fill takes an idle scratch, or a new one when every scratch
/// is lent, and returns it when done.  The pool holds as many scratches as
/// fills have run at once, each at the size of the largest fill it served.
pub struct AaFillScratchPool<B: GpuBackend> {
    idle: Mutex<Vec<AaFillScratch<B>>>,
}

impl<B: GpuBackend> AaFillScratchPool<B> {
    /// An empty pool.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
        }
    }

    fn take(&self) -> AaFillScratch<B> {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default()
    }

    fn put(&self, scratch: AaFillScratch<B>) {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }
}

impl<B: GpuBackend> Default for AaFillScratchPool<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// [`aa_fill_page`] with a scratch lent by `backend`'s pool
/// ([`GpuBackend::aa_fill_scratch`]).
///
/// # Errors
/// As [`aa_fill_page`].
pub fn aa_fill<B: GpuBackend>(
    backend: &B,
    table: &AaScanTable<'_>,
    rect: AaFillRect,
    aa_width: u32,
    even_odd: bool,
) -> Result<Vec<u8>> {
    let pool = backend.aa_fill_scratch();
    let mut scratch = pool.take();
    let counts = aa_fill_page(backend, &mut scratch, table, rect, aa_width, even_odd);
    pool.put(scratch);
    counts
}

/// Direction of a staged transfer, which picks the memory kind of both
/// buffers.
#[derive(Clone, Copy)]
enum Direction {
    /// Host writes, device reads: [`GpuBackend::alloc_device`] and
    /// [`GpuBackend::alloc_host_pinned`].
    ToDevice,
    /// Device writes, host reads: [`GpuBackend::alloc_device_readback`] and
    /// [`GpuBackend::alloc_host_readback`].
    ToHost,
}

/// Largest number of bytes a slot is grown by beyond the size requested.
const GROWTH_CAP: usize = 64 << 20;

/// The size a slot holding `capacity` bytes is reallocated to for a
/// transfer of `len > capacity` bytes: doubled, but by at most
/// [`GROWTH_CAP`] beyond `len`, so `len <= size <= len + GROWTH_CAP`.
fn grown_size(capacity: usize, len: usize) -> usize {
    len.max(
        capacity
            .saturating_mul(2)
            .min(len.saturating_add(GROWTH_CAP)),
    )
}

/// `slot`'s buffers, reallocated first when either holds fewer than
/// `len > 0` bytes, to [`grown_size`]: a run of slowly growing fills
/// reallocates logarithmically often below [`GROWTH_CAP`].
fn ensure<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    len: usize,
    direction: Direction,
) -> Result<&'a mut Staged<B>> {
    reject_zero_size(len, "aa_fill scratch")?;
    let capacity = slot.as_ref().map_or(0, |s| {
        let device = backend.device_buffer_len(&s.device);
        s.host
            .as_ref()
            .map_or(device, |host| device.min(backend.host_buffer_len(host)))
    });
    if capacity < len {
        let size = grown_size(capacity, len);
        if let Some(old) = slot.take() {
            if let Some(host) = old.host {
                backend.free_host_pinned(host);
            }
            backend.free_device(old.device);
        }
        let device = match direction {
            Direction::ToDevice => backend.alloc_device(size)?,
            Direction::ToHost => backend.alloc_device_readback(size)?,
        };
        let host = if backend.device_memory_is_host_mapped() {
            None
        } else {
            Some(match direction {
                Direction::ToDevice => backend.alloc_host_pinned(size)?,
                Direction::ToHost => backend.alloc_host_readback(size)?,
            })
        };
        return Ok(slot.insert(Staged { host, device }));
    }
    // `capacity >= len > 0` holds only for a populated slot.
    slot.as_mut()
        .ok_or_else(|| BackendError::msg("aa_fill: scratch slot is empty"))
}

/// The error of a staged transfer without a host buffer on a device whose
/// buffers the host cannot reach in place.
fn not_mapped() -> BackendError {
    BackendError::msg("aa_fill: device buffer is not host-mapped")
}

/// The host-side bytes of `staged`: its host buffer, or its device buffer in
/// place when there is none.
///
/// # Safety
/// No device work that writes `staged`'s buffers is in flight.
unsafe fn host_view<'a, B: GpuBackend>(backend: &B, staged: &'a Staged<B>) -> Result<&'a [u8]> {
    staged.host.as_ref().map_or_else(
        // SAFETY: forwarded from this function's contract.
        || unsafe { backend.device_bytes(&staged.device) }.ok_or_else(not_mapped),
        |host| backend.host_bytes(host),
    )
}

/// The writable host-side bytes of `staged`; see [`host_view`].
///
/// # Safety
/// No device work that accesses `staged`'s buffers is in flight.
unsafe fn host_view_mut<'a, B: GpuBackend>(
    backend: &B,
    staged: &'a mut Staged<B>,
) -> Result<&'a mut [u8]> {
    match staged.host.as_mut() {
        Some(host) => backend.host_bytes_mut(host),
        // SAFETY: forwarded from this function's contract.
        None => unsafe { backend.device_bytes_mut(&mut staged.device) }.ok_or_else(not_mapped),
    }
}

/// Copy `bytes` into `slot`, growing it first as needed: into the host
/// buffer, or into the device buffer in place when there is none.
fn stage<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    bytes: &[u8],
) -> Result<&'a mut Staged<B>> {
    let staged = ensure(backend, slot, bytes.len(), Direction::ToDevice)?;
    // SAFETY: the scratch owns the buffers exclusively (`&mut`), and every
    // page that used them was waited on before `aa_fill_page` returned,
    // whether it succeeded or failed (`run_page`).
    let target = unsafe { host_view_mut(backend, staged) }?;
    target[..bytes.len()].copy_from_slice(bytes);
    Ok(staged)
}

/// Record the copy of `staged`'s first `len` bytes from host to device, if
/// it has a host buffer.
///
/// # Safety
/// The page this records into is waited on before `staged` is written again
/// or dropped.
unsafe fn record_upload<B: GpuBackend>(backend: &B, staged: &Staged<B>, len: usize) -> Result<()> {
    staged.host.as_ref().map_or(Ok(()), |host| {
        // SAFETY: forwarded from this function's contract.
        unsafe { backend.record_copy_to_device(host, &staged.device, len) }
    })
}

/// Record the copy of `staged`'s first `len` bytes from device to host, if
/// it has a host buffer.
///
/// # Safety
/// The page this records into is waited on before `staged` is accessed again
/// or dropped.
unsafe fn record_download<B: GpuBackend>(
    backend: &B,
    staged: &mut Staged<B>,
    len: usize,
) -> Result<()> {
    let Staged { host, device } = staged;
    host.as_mut().map_or(Ok(()), |host| {
        // SAFETY: forwarded from this function's contract.
        unsafe { backend.record_copy_to_host(device, host, len) }
    })
}

/// The per-pixel AA sub-sample counts of `rect` from `table`, computed on
/// `backend`.
///
/// The counts are `rect.width × rect.height` bytes, row-major, each the
/// count `0..=AA_SIZE²` of the pixel's covered sub-samples, as
/// [`crate::aa_fill_cpu`] produces them.  `aa_width` is the bitmap width ×
/// `AA_SIZE`.
///
/// The table is written into `scratch`'s host buffers, and the uploads, the
/// fill kernel and the download of the counts are recorded into one page
/// ([`GpuBackend::run_page`]): one device submission per fill.  On a device
/// whose buffers the host reaches in place, the table is written into the
/// device buffers and the counts read from them, with no copies.  A table
/// without intersections, or an empty rectangle, yields zero counts without
/// a device round trip.  The caller decides whether the fill is worth one
/// (see [`crate::GPU_AA_FILL_THRESHOLD`]).
///
/// # Errors
///
/// [`BackendError::InvariantViolation`] when `table` is malformed (see
/// [`AaScanTable::is_well_formed`]) or has more rows than `u32` holds, or
/// `rect` has more pixels than the kernel's 32-bit index addresses; the
/// backend's error when an allocation or a device call fails.
pub fn aa_fill_page<B: GpuBackend>(
    backend: &B,
    scratch: &mut AaFillScratch<B>,
    table: &AaScanTable<'_>,
    rect: AaFillRect,
    aa_width: u32,
    even_odd: bool,
) -> Result<Vec<u8>> {
    let invariant = aa_fill_invariant;
    if !table.is_well_formed() {
        return Err(invariant("malformed scan table"));
    }
    let n_pixels = rect
        .kernel_pixel_count()
        .ok_or_else(|| invariant("rect has more pixels than a 32-bit index addresses"))?;
    if n_pixels == 0 || table.intersects.is_empty() {
        return Ok(vec![0; n_pixels]);
    }
    let n_rows = u32::try_from(table.n_rows())
        .map_err(|_| invariant("scan table has more rows than u32 holds"))?;

    let intersects: &[u8] = bytemuck::cast_slice(table.intersects);
    let row_starts: &[u8] = bytemuck::cast_slice(table.row_starts);
    let AaFillScratch {
        intersects: intersects_slot,
        row_starts: row_starts_slot,
        coverage: coverage_slot,
    } = scratch;
    let d_intersects = stage(backend, intersects_slot, intersects)?;
    let d_row_starts = stage(backend, row_starts_slot, row_starts)?;
    let coverage = ensure(backend, coverage_slot, n_pixels, Direction::ToHost)?;
    // SAFETY (the recorded copies and the in-place reads below): the
    // scratch's buffers are borrowed mutably for this whole call and
    // `run_page` returns only once its page has completed or, on failure,
    // been waited on; nothing touches them in between.
    backend.run_page(|backend| {
        unsafe { record_upload(backend, d_intersects, intersects.len()) }?;
        unsafe { record_upload(backend, d_row_starts, row_starts.len()) }?;
        backend.record_aa_fill(AaFillParams {
            intersects: &d_intersects.device,
            row_starts: &d_row_starts.device,
            coverage: &coverage.device,
            n_rows,
            table_y0: table.y_min,
            rect,
            aa_width,
            fill_rule: u8::from(even_odd),
        })?;
        unsafe { record_download(backend, coverage, n_pixels) }
    })?;
    let counts = unsafe { host_view(backend, coverage) }?;
    Ok(counts[..n_pixels].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_doubles_below_the_cap_and_is_bounded_above_it() {
        assert_eq!(grown_size(0, 10), 10);
        assert_eq!(grown_size(100, 101), 200);
        assert_eq!(grown_size(100, 300), 300);
        let big = 300 << 20;
        assert_eq!(grown_size(big, big + 1), big + 1 + GROWTH_CAP);
        assert_eq!(grown_size(usize::MAX - 1, usize::MAX), usize::MAX);
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn a_zero_length_transfer_into_an_empty_slot_is_an_error() {
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let mut slot = None;
        let err = ensure(&backend, &mut slot, 0, Direction::ToDevice)
            .err()
            .expect("a zero-length slot is rejected");
        assert!(matches!(err, BackendError::ZeroSizeAlloc { .. }), "{err}");
        assert!(slot.is_none());
    }

    #[cfg(feature = "gpu-validation")]
    #[test]
    fn fills_reuse_the_backends_pooled_scratch() {
        use crate::fill::ScanIntersect;
        let Ok(backend) = crate::backend::cuda::CudaBackend::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let intersects = [ScanIntersect {
            x0: 0,
            x1: 7,
            count: 0,
        }];
        let row_starts = [0, 1];
        let table = AaScanTable {
            intersects: &intersects,
            row_starts: &row_starts,
            y_min: 0,
        };
        let rect = AaFillRect {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let idle = || backend.aa_fill_scratch().idle.lock().expect("pool").len();
        for _ in 0..3 {
            let counts = aa_fill(&backend, &table, rect, 8, false).expect("aa_fill");
            assert_eq!(counts.len(), 2);
            assert_eq!(idle(), 1, "the fill did not return its scratch");
        }
        let pooled = backend.aa_fill_scratch().take();
        assert!(
            pooled.coverage.is_some(),
            "the pooled scratch kept no buffers"
        );
    }
}
