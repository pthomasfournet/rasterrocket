//! The scanline-AA fill as one backend page: the one sequence every caller
//! of the fill kernel runs, on either backend.

use super::params::AaFillParams;
use super::{BackendError, GpuBackend, Result};
use crate::fill::{AaFillRect, AaScanTable};

/// One host buffer and one device buffer of at least the same length.
struct Staged<B: GpuBackend> {
    host: B::HostBuffer,
    device: B::DeviceBuffer,
}

/// Reusable host and device buffers for [`aa_fill_page`].
///
/// Each of the three transfers of a fill (intersections, row offsets,
/// counts) keeps one host and one device buffer, grown to the largest fill
/// seen and reused by later fills, so a fill allocates only when it is
/// larger than every earlier one.  The buffers are freed when the scratch
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

/// Direction of a staged transfer, which picks the host memory kind.
#[derive(Clone, Copy)]
enum Direction {
    /// Host writes, device reads: [`GpuBackend::alloc_host_pinned`].
    ToDevice,
    /// Device writes, host reads: [`GpuBackend::alloc_host_readback`].
    ToHost,
}

/// `slot`'s buffers, reallocated first when either holds fewer than `len`
/// bytes.  Growth is at least geometric, so a run of slowly growing fills
/// reallocates logarithmically often.
fn ensure<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    len: usize,
    direction: Direction,
) -> Result<&'a mut Staged<B>> {
    let capacity = slot.as_ref().map_or(0, |s| {
        backend
            .host_buffer_len(&s.host)
            .min(backend.device_buffer_len(&s.device))
    });
    if capacity < len {
        let size = len.max(capacity.saturating_mul(2));
        if let Some(old) = slot.take() {
            backend.free_host_pinned(old.host);
            backend.free_device(old.device);
        }
        let host = match direction {
            Direction::ToDevice => backend.alloc_host_pinned(size)?,
            Direction::ToHost => backend.alloc_host_readback(size)?,
        };
        *slot = Some(Staged {
            host,
            device: backend.alloc_device(size)?,
        });
    }
    Ok(slot.as_mut().expect("populated above"))
}

/// Copy `bytes` into the host half of `slot`, growing it first as needed.
fn stage<'a, B: GpuBackend>(
    backend: &B,
    slot: &'a mut Option<Staged<B>>,
    bytes: &[u8],
) -> Result<&'a mut Staged<B>> {
    let staged = ensure(backend, slot, bytes.len(), Direction::ToDevice)?;
    backend.host_bytes_mut(&mut staged.host)[..bytes.len()].copy_from_slice(bytes);
    Ok(staged)
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
/// ([`GpuBackend::run_page`]): one device submission per fill.  A table
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
    let invariant = |detail: &'static str| BackendError::InvariantViolation {
        kind: "AaFillInvariantViolation",
        detail,
    };
    if !table.is_well_formed() {
        return Err(invariant("malformed scan table"));
    }
    let n_pixels = rect
        .pixel_count()
        .filter(|&n| u32::try_from(n).is_ok())
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
    backend.run_page(|backend| {
        backend.record_copy_to_device(
            &d_intersects.host,
            &d_intersects.device,
            intersects.len(),
        )?;
        backend.record_copy_to_device(
            &d_row_starts.host,
            &d_row_starts.device,
            row_starts.len(),
        )?;
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
        backend.record_copy_to_host(&coverage.device, &mut coverage.host, n_pixels)
    })?;
    Ok(backend.host_bytes(&coverage.host)[..n_pixels].to_vec())
}
