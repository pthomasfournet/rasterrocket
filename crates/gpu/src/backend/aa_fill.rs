//! The scanline-AA fill as one backend page: the one sequence every caller
//! of the fill kernel runs, on either backend.

use super::params::AaFillParams;
use super::{BackendError, GpuBackend, Result};
use crate::fill::{AaFillRect, AaScanTable};

/// The per-pixel AA sub-sample counts of `rect` from `table`, computed on
/// `backend`.
///
/// The counts are `rect.width × rect.height` bytes, row-major, each the
/// count `0..=AA_SIZE²` of the pixel's covered sub-samples, as
/// [`crate::aa_fill_cpu`] produces them.  `aa_width` is the bitmap width ×
/// `AA_SIZE`.
///
/// The table is uploaded, the fill kernel recorded and run as one page
/// ([`GpuBackend::run_page`]) and the counts downloaded.  A table without
/// intersections, or an empty rectangle, yields zero counts without a
/// device round trip.  The caller decides whether the fill is worth one
/// (see [`crate::GPU_AA_FILL_THRESHOLD`]).
///
/// # Errors
///
/// [`BackendError::InvariantViolation`] when `table` is malformed (see
/// [`AaScanTable::is_well_formed`]) or has more rows than `u32` holds, or
/// `rect` has more pixels than the kernel's 32-bit index addresses; the
/// backend's error when a device call fails.
pub fn aa_fill_page<B: GpuBackend>(
    backend: &B,
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
    let d_intersects = backend.alloc_device(intersects.len())?;
    let d_row_starts = backend.alloc_device(row_starts.len())?;
    let d_coverage = backend.alloc_device(n_pixels)?;
    backend.upload(&d_intersects, intersects)?;
    backend.upload(&d_row_starts, row_starts)?;
    backend.run_page(|backend| {
        backend.record_aa_fill(AaFillParams {
            intersects: &d_intersects,
            row_starts: &d_row_starts,
            coverage: &d_coverage,
            n_rows,
            table_y0: table.y_min,
            rect,
            aa_width,
            fill_rule: u8::from(even_odd),
        })
    })?;
    let mut counts = vec![0; n_pixels];
    backend.download(&d_coverage, &mut counts)?;
    Ok(counts)
}
