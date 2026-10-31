//! Device→host downloads through per-thread page-locked staging buffers.
//!
//! A copy into pageable host memory is bounced by the driver through its own
//! page-locked buffer and returns only once the whole transfer is complete.
//! [`download_banded`] instead copies a row-major device range in bands of
//! at most [`BAND_BYTES`] into one of two page-locked buffers owned by the
//! calling thread, and hands each band to the caller while the copy of the
//! next band is in flight.  The caller consumes the rows in place; no
//! intermediate host buffer of the full range exists.
//!
//! The buffers are allocated without write-combining, so host reads of a
//! band run at cached-memory speed.  They are created on first use, grown
//! to one row when a row exceeds [`BAND_BYTES`], and released at thread
//! exit.  A range of at most [`DIRECT_MAX_BYTES`] is copied in one
//! synchronous transfer into a pageable buffer instead, and so is any range
//! when page-locked allocation fails.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream, CudaView, DriverError, PinnedHostSlice};

/// Upper bound on the bytes copied per band (rounded down to whole rows).
pub(crate) const BAND_BYTES: usize = 4 << 20;

/// Largest range copied directly into pageable memory.  Below this size
/// the driver's bounce copy costs less than allocating page-locked memory
/// on a thread that has none.
pub(crate) const DIRECT_MAX_BYTES: usize = 256 << 10;

/// Two page-locked band buffers and the events marking their copies done.
struct Staging {
    bufs: [PinnedHostSlice<u8>; 2],
    copied: [CudaEvent; 2],
}

impl Staging {
    fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self, DriverError> {
        // SAFETY: a band buffer is only read after a device→host copy has
        // written the bytes being read.
        let alloc = || unsafe { ctx.alloc_pinned_with_flags::<u8>(bytes, 0) };
        Ok(Self {
            bufs: [alloc()?, alloc()?],
            copied: [ctx.new_event(None)?, ctx.new_event(None)?],
        })
    }

    fn capacity(&self) -> usize {
        self.bufs[0].len()
    }
}

thread_local! {
    /// The calling thread's staging buffers, one set per CUDA context.
    static STAGING: RefCell<Vec<Staging>> = const { RefCell::new(Vec::new()) };
}

/// Take the calling thread's staging set for `ctx` out of the thread-local
/// store, allocating or growing it to hold `bytes`.  Returns `None` if
/// page-locked allocation fails.
fn take_staging(ctx: &Arc<CudaContext>, bytes: usize) -> Option<Staging> {
    let held = STAGING.with_borrow_mut(|sets| {
        let i = sets
            .iter()
            .position(|s| Arc::ptr_eq(s.bufs[0].context(), ctx))?;
        Some(sets.swap_remove(i))
    });
    if let Some(s) = held.filter(|s| s.capacity() >= bytes) {
        return Some(s);
    }
    Staging::new(ctx, bytes.max(BAND_BYTES))
        .map_err(|e| log::debug!("gpu: page-locked staging allocation failed ({e})"))
        .ok()
}

fn return_staging(staging: Staging) {
    STAGING.with_borrow_mut(|sets| sets.push(staging));
}

/// Copy `src`, `src.len() / row_bytes` rows of `row_bytes` bytes, to the
/// host in bands of whole rows, calling `f(first_row, band)` once per band
/// in row order.  `band` holds the band's rows contiguously and may be
/// modified.  Work queued on `stream` before the call is complete before
/// the first band is copied.
///
/// # Errors
/// Returns the driver error of a failed copy, event record or wait; bands
/// already passed to `f` stay consumed.
///
/// # Panics
/// Panics if `row_bytes` is zero or does not divide `src.len()`.
pub(crate) fn download_banded<F>(
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, u8>,
    row_bytes: usize,
    mut f: F,
) -> Result<(), DriverError>
where
    F: FnMut(usize, &mut [u8]),
{
    assert!(
        row_bytes != 0 && src.len().is_multiple_of(row_bytes),
        "download_banded: {} bytes is not a whole number of {row_bytes}-byte rows",
        src.len()
    );
    let n_rows = src.len() / row_bytes;
    if n_rows == 0 {
        return Ok(());
    }
    if src.len() <= DIRECT_MAX_BYTES {
        return download_banded_pageable(stream, src, row_bytes, n_rows, f);
    }
    let band_rows = (BAND_BYTES / row_bytes).clamp(1, n_rows);
    let band_bytes = band_rows * row_bytes;
    let Some(mut staging) = take_staging(stream.context(), band_bytes) else {
        return download_banded_pageable(stream, src, row_bytes, band_rows, f);
    };
    let result = run_banded(stream, src, row_bytes, band_rows, &mut staging, &mut f);
    // A failure part-way may leave a copy into the staging buffers queued;
    // they are reused only once the stream has drained.
    if result.is_err() && stream.synchronize().is_err() {
        return result;
    }
    return_staging(staging);
    result
}

/// Double-buffered band loop: the copy of band `k + 1` is queued before
/// the host waits for band `k`, so the transfer overlaps `f`.
fn run_banded<F>(
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, u8>,
    row_bytes: usize,
    band_rows: usize,
    staging: &mut Staging,
    f: &mut F,
) -> Result<(), DriverError>
where
    F: FnMut(usize, &mut [u8]),
{
    let n_rows = src.len() / row_bytes;
    let band_range = |k: usize| {
        let r0 = k * band_rows;
        r0..(r0 + band_rows).min(n_rows)
    };
    let n_bands = n_rows.div_ceil(band_rows);
    let enqueue = |staging: &mut Staging, k: usize| -> Result<(), DriverError> {
        let rows = band_range(k);
        let bytes = rows.len() * row_bytes;
        let slot = k % 2;
        let view = src.slice(rows.start * row_bytes..rows.end * row_bytes);
        // The previous use of this slot was band k - 2, which the host has
        // finished reading.
        let dst = &mut staging.bufs[slot].as_mut_slice()?[..bytes];
        stream.memcpy_dtoh(&view, dst)?;
        staging.copied[slot].record(stream)
    };

    enqueue(staging, 0)?;
    for k in 0..n_bands {
        if k + 1 < n_bands {
            enqueue(staging, k + 1)?;
        }
        let slot = k % 2;
        staging.copied[slot].synchronize()?;
        let rows = band_range(k);
        let bytes = rows.len() * row_bytes;
        f(rows.start, &mut staging.bufs[slot].as_mut_slice()?[..bytes]);
    }
    Ok(())
}

/// Fallback for [`download_banded`] when no page-locked memory is
/// available: synchronous copies through one pageable band buffer.
fn download_banded_pageable<F>(
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, u8>,
    row_bytes: usize,
    band_rows: usize,
    mut f: F,
) -> Result<(), DriverError>
where
    F: FnMut(usize, &mut [u8]),
{
    let n_rows = src.len() / row_bytes;
    let mut band = vec![0u8; band_rows * row_bytes];
    for r0 in (0..n_rows).step_by(band_rows) {
        let r1 = (r0 + band_rows).min(n_rows);
        let dst = &mut band[..(r1 - r0) * row_bytes];
        stream.memcpy_dtoh(&src.slice(r0 * row_bytes..r1 * row_bytes), dst)?;
        stream.synchronize()?;
        f(r0, dst);
    }
    Ok(())
}

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i * 7 + i / 251).to_le_bytes()[0])
            .collect()
    }

    /// Bands arrive in order, cover every row exactly once, and carry the
    /// device bytes — for a range smaller than one band, one spanning
    /// several, and rows wider than a band.
    #[test]
    fn bands_reassemble_the_source() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        for (row_bytes, n_rows) in [(64, 3), (4096, 2500), (BAND_BYTES + 4, 3)] {
            let host = pattern(row_bytes * n_rows);
            let dev = stream.clone_htod(&host).expect("upload");
            let mut seen = Vec::with_capacity(host.len());
            let mut next_row = 0;
            download_banded(&stream, &dev.as_view(), row_bytes, |r0, band| {
                assert_eq!(r0, next_row, "bands out of order");
                assert_eq!(band.len() % row_bytes, 0, "partial row in band");
                next_row += band.len() / row_bytes;
                seen.extend_from_slice(band);
            })
            .expect("download");
            assert_eq!(next_row, n_rows);
            assert!(seen == host, "{row_bytes}×{n_rows}: bytes differ");
        }
    }

    #[test]
    fn pageable_fallback_reassembles_the_source() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("stream");
        let host = pattern(256 * 40);
        let dev = stream.clone_htod(&host).expect("upload");
        let mut seen = Vec::new();
        download_banded_pageable(&stream, &dev.as_view(), 256, 7, |_, band| {
            seen.extend_from_slice(band);
        })
        .expect("download");
        assert_eq!(seen, host);
    }
}
