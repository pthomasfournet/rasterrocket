//! The scanline-AA fill as one backend page: the one sequence every caller
//! of the fill kernel runs, on either backend.

use super::params::{AaFillParams, aa_fill_invariant};
use super::staged::{Direction, Staged, ensure, host_view, record_download, record_upload, stage};
use super::{GpuBackend, Result};
use crate::fill::{AaFillRect, AaScanTable};

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

/// [`aa_fill_page`] with a scratch lent by `backend`'s pool
/// ([`GpuBackend::scratch`]).
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
    backend
        .scratch()
        .aa_fill
        .with(|scratch| aa_fill_page(backend, scratch, table, rect, aa_width, even_odd))
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
/// [`super::BackendError::InvariantViolation`] when `table` is malformed (see
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

#[cfg(all(test, feature = "gpu-validation"))]
mod tests {
    use super::*;

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
        let idle = || backend.scratch().aa_fill.idle_len();
        for _ in 0..3 {
            let counts = aa_fill(&backend, &table, rect, 8, false).expect("aa_fill");
            assert_eq!(counts.len(), 2);
            assert_eq!(idle(), 1, "the fill did not return its scratch");
        }
        let pooled = backend.scratch().aa_fill.take_idle();
        assert!(
            pooled.coverage.is_some(),
            "the pooled scratch kept no buffers"
        );
    }
}
