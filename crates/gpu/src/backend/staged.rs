//! Device buffers paired with the host buffers their transfers go through,
//! grown on demand and reused across the pages that record those transfers.
//!
//! A [`Staged`] slot holds one device buffer and, unless the host reaches
//! device buffers in place ([`GpuBackend::device_memory_is_host_mapped`]),
//! one host buffer of at least its length.  Host data is written into the
//! host buffer (or the device buffer in place) before a page, the page
//! records the copies between the two, and results are read from the host
//! buffer (or the device buffer in place) once the page has completed.

use super::{BackendError, GpuBackend, Result, reject_zero_size};

/// A device buffer and, unless the host reaches the device buffer in place
/// ([`GpuBackend::device_memory_is_host_mapped`]), a host buffer of at least its length that
/// transfers go through.
pub(crate) struct Staged<B: GpuBackend> {
    pub(crate) host: Option<B::HostBuffer>,
    pub(crate) device: B::DeviceBuffer,
}

/// Direction of a staged transfer, which picks the memory kind of both
/// buffers.
#[derive(Clone, Copy)]
pub(crate) enum Direction {
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
/// When `capacity` and `len` are multiples of 4, so is the result.
pub(crate) fn grown_size(capacity: usize, len: usize) -> usize {
    len.max(
        capacity
            .saturating_mul(2)
            .min(len.saturating_add(GROWTH_CAP)),
    )
}

/// `slot`'s buffers, reallocated first when either holds fewer than
/// `len > 0` bytes, to [`grown_size`]: a run of slowly growing transfers
/// reallocates logarithmically often below [`GROWTH_CAP`].
///
/// # Errors
/// [`BackendError::ZeroSizeAlloc`] for a zero `len`; the backend's error
/// when an allocation fails, the slot then being empty.
pub(crate) fn ensure<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    len: usize,
    direction: Direction,
) -> Result<&'a mut Staged<B>> {
    reject_zero_size(len, "staged transfer")?;
    let capacity = slot.as_ref().map_or(0, |s| capacity(backend, s));
    if capacity < len {
        return replace(backend, slot, grown_size(capacity, len), direction);
    }
    // `capacity >= len > 0` holds only for a populated slot.
    slot.as_mut()
        .ok_or_else(|| BackendError::msg("staged transfer: slot is empty"))
}

/// The bytes both of `staged`'s buffers hold.
fn capacity<B: GpuBackend>(backend: &B, staged: &Staged<B>) -> usize {
    let device = backend.device_buffer_len(&staged.device);
    staged
        .host
        .as_ref()
        .map_or(device, |host| device.min(backend.host_buffer_len(host)))
}

/// `slot` with its buffers freed and reallocated at `size > 0` bytes.
///
/// # Errors
/// The backend's error when an allocation fails, the slot then being empty.
fn replace<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    size: usize,
    direction: Direction,
) -> Result<&'a mut Staged<B>> {
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
    Ok(slot.insert(Staged { host, device }))
}

/// The error of a staged transfer without a host buffer on a device whose
/// buffers the host cannot reach in place.
fn not_mapped() -> BackendError {
    BackendError::msg("staged transfer: device buffer is not host-mapped")
}

/// The host-side bytes of `staged`: its host buffer, or its device buffer in
/// place when there is none.
///
/// # Safety
/// No device work that writes `staged`'s buffers is in flight.
pub(crate) unsafe fn host_view<'a, B: GpuBackend>(
    backend: &B,
    staged: &'a Staged<B>,
) -> Result<&'a [u8]> {
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
///
/// The slot is held exclusively, and every page that used its buffers has
/// been waited on: [`GpuBackend::run_page`] returns only once its page has
/// completed or been abandoned.
///
/// # Errors
/// As [`ensure`]; the backend's error when the host view is unavailable.
pub(crate) fn stage<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    bytes: &[u8],
) -> Result<&'a mut Staged<B>> {
    let staged = ensure(backend, slot, bytes.len(), Direction::ToDevice)?;
    // SAFETY: the slot is borrowed exclusively (`&mut`), and every page
    // that used its buffers was waited on before the call that recorded it
    // returned, whether it succeeded or failed (`run_page`).
    let target = unsafe { host_view_mut(backend, staged) }?;
    target[..bytes.len()].copy_from_slice(bytes);
    Ok(staged)
}

/// [`stage`] into a slot whose device buffer holds exactly `bytes.len()`
/// bytes, for a kernel that reads its input's size from the buffer: the
/// slot is reallocated whenever its device length differs.
///
/// # Errors
/// As [`stage`].
pub(crate) fn stage_exact<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    bytes: &[u8],
) -> Result<&'a mut Staged<B>> {
    reject_zero_size(bytes.len(), "staged transfer")?;
    if slot
        .as_ref()
        .is_none_or(|s| backend.device_buffer_len(&s.device) != bytes.len())
    {
        replace(backend, slot, bytes.len(), Direction::ToDevice).map(drop)?;
    }
    stage(backend, slot, bytes)
}

/// Record the copy of `staged`'s first `len` bytes from host to device, if
/// it has a host buffer.
///
/// # Safety
/// The page this records into is waited on before `staged` is written again
/// or dropped.
pub(crate) unsafe fn record_upload<B: GpuBackend>(
    backend: &B,
    staged: &Staged<B>,
    len: usize,
) -> Result<()> {
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
pub(crate) unsafe fn record_download<B: GpuBackend>(
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
}
