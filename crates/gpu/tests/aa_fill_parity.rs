//! `aa_fill` kernel parity: the CUDA kernel, the Vulkan kernel, and the
//! host twin [`rasterrocket_gpu::aa_fill_cpu`] must produce byte-identical
//! sub-sample counts for the same intersection table.  The kernel is pure
//! integer arithmetic, so the contract is exact equality, not an LSB budget.
//!
//! Run with:
//!   `cargo test -p rasterrocket-gpu --features gpu-validation --test aa_fill_parity`
//!   `cargo test -p rasterrocket-gpu --features vulkan --test aa_fill_parity`
//!   `cargo test -p rasterrocket-gpu --features 'vulkan,gpu-validation' --test aa_fill_parity`

#![cfg(any(feature = "gpu-validation", feature = "vulkan"))]

use rasterrocket_gpu::{AA_SIZE, AaFillRect, AaScanTable, ScanIntersect, aa_fill_cpu};

// ── fixtures ────────────────────────────────────────────────────────────

const fn ix(x0: i32, x1: i32, count: i32) -> ScanIntersect {
    ScanIntersect { x0, x1, count }
}

const fn rect(x: i32, y: i32, width: u32, height: u32) -> AaFillRect {
    AaFillRect {
        x,
        y,
        width,
        height,
    }
}

/// One fixture: a table plus the rectangle and clamp it is evaluated over.
struct Case {
    name: &'static str,
    intersects: Vec<ScanIntersect>,
    row_starts: Vec<u32>,
    y_min: i32,
    rect: AaFillRect,
    aa_width: u32,
}

impl Case {
    /// Assemble a case from per-row entry lists.
    fn from_rows(
        name: &'static str,
        rows: &[Vec<ScanIntersect>],
        y_min: i32,
        rect: AaFillRect,
        aa_width: u32,
    ) -> Self {
        let mut intersects = Vec::new();
        let mut row_starts = Vec::with_capacity(rows.len() + 1);
        for row in rows {
            row_starts.push(u32::try_from(intersects.len()).expect("fits u32"));
            intersects.extend_from_slice(row);
        }
        row_starts.push(u32::try_from(intersects.len()).expect("fits u32"));
        Self {
            name,
            intersects,
            row_starts,
            y_min,
            rect,
            aa_width,
        }
    }

    fn table(&self) -> AaScanTable<'_> {
        AaScanTable {
            intersects: &self.intersects,
            row_starts: &self.row_starts,
            y_min: self.y_min,
        }
    }
}

/// A 10 × 40 px rectangle with fractional edges: left edge at AA column 2
/// (pixel 0, coverage 2/4), right edge at AA column 41 (pixel 10, coverage
/// 2/4).  The first and last AA rows carry the horizontal edges collapsed
/// to count 0, as the CPU scanner emits them.
fn fractional_rect() -> Case {
    let mut rows = vec![vec![ix(2, 41, 0)]];
    rows.extend((0..158).map(|_| vec![ix(2, 2, -1), ix(41, 41, 1)]));
    rows.push(vec![ix(2, 41, 0)]);
    Case::from_rows("fractional rect", &rows, 5, rect(0, 1, 12, 42), 48)
}

/// A slanted edge: the left intersection walks one AA column per two AA
/// rows and spans two columns on the rows where it moves.
fn slanted_edge() -> Case {
    let rows: Vec<Vec<ScanIntersect>> = (0..64i32)
        .map(|r| {
            let left = r / 2;
            let entry = if r % 2 == 0 {
                ix(left, left, -1)
            } else {
                ix(left, left + 1, -1)
            };
            vec![entry, ix(60, 60, 1)]
        })
        .collect();
    Case::from_rows("slanted edge", &rows, 0, rect(0, 0, 17, 16), 68)
}

/// Nested squares with the same winding: even-odd punches the centre out,
/// non-zero keeps it.
fn nested_squares() -> Case {
    let rows: Vec<Vec<ScanIntersect>> = (0..32i32)
        .map(|r| {
            if (8..24).contains(&r) {
                vec![ix(0, 0, 1), ix(8, 8, 1), ix(23, 23, -1), ix(31, 31, -1)]
            } else {
                vec![ix(0, 0, 1), ix(31, 31, -1)]
            }
        })
        .collect();
    Case::from_rows("nested squares", &rows, 0, rect(0, 0, 8, 8), 32)
}

/// Unterminated winding: the span runs to `aa_width`, which lies inside the
/// evaluated rectangle, so the clamp is what stops it.
fn unterminated_span() -> Case {
    let rows: Vec<Vec<ScanIntersect>> = (0..16).map(|_| vec![ix(6, 6, 1)]).collect();
    Case::from_rows("unterminated span", &rows, 0, rect(0, 0, 8, 4), 22)
}

/// Unmerged overlapping entries: coverage is the union.
fn overlapping_entries() -> Case {
    let rows: Vec<Vec<ScanIntersect>> = (0..8)
        .map(|_| vec![ix(0, 5, 1), ix(2, 3, 1), ix(4, 4, -1), ix(7, 7, -1)])
        .collect();
    Case::from_rows("overlapping entries", &rows, 0, rect(0, 0, 4, 2), 16)
}

/// Wider than one 16×16 block in both axes, with an offset rectangle origin
/// and table origin: exercises the block/grid indexing and the origin
/// arithmetic.
fn multi_block() -> Case {
    let n_rows = 40 * AA_SIZE.unsigned_abs() as usize;
    let rows: Vec<Vec<ScanIntersect>> = (0..n_rows)
        .map(|_| vec![ix(9, 9, -1), ix(150, 150, 1)])
        .collect();
    Case::from_rows("multi-block", &rows, 12, rect(2, 3, 37, 41), 160)
}

fn cases() -> Vec<Case> {
    vec![
        fractional_rect(),
        slanted_edge(),
        nested_squares(),
        unterminated_span(),
        overlapping_entries(),
        multi_block(),
    ]
}

fn assert_identical(expected: &[u8], actual: &[u8], what: &str) {
    assert_eq!(expected.len(), actual.len(), "{what}: length mismatch");
    if let Some((i, (e, a))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (e, a))| e != a)
    {
        panic!("{what}: byte {i} differs (expected {e}, got {a})");
    }
}

// ── CUDA ────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu-validation")]
mod cuda {
    use std::sync::Arc;

    use super::{aa_fill_cpu, assert_identical, cases};
    use rasterrocket_gpu::GpuCtx;
    use rasterrocket_gpu::backend::aa_fill_page;
    use rasterrocket_gpu::backend::cuda::CudaBackend;

    /// The CUDA backend on a fresh context.
    pub(super) fn backend() -> CudaBackend {
        let ctx = GpuCtx::init().expect("gpu-validation tests require a CUDA device");
        CudaBackend::from_ctx(Arc::new(ctx))
    }

    #[test]
    fn cuda_kernel_matches_host_twin() {
        let backend = backend();
        for case in cases() {
            for eo in [false, true] {
                let table = case.table();
                let expected = aa_fill_cpu(&table, case.rect, case.aa_width, eo);
                let actual = aa_fill_page(&backend, &table, case.rect, case.aa_width, eo)
                    .unwrap_or_else(|e| panic!("{}: CUDA aa_fill failed: {e}", case.name));
                assert_identical(
                    &expected,
                    &actual,
                    &format!("{} (eo={eo}) CUDA vs host twin", case.name),
                );
                assert!(
                    expected.iter().any(|&c| c != 0),
                    "{}: fixture covers nothing",
                    case.name
                );
            }
        }
    }
}

// ── Vulkan ──────────────────────────────────────────────────────────────

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::{Case, aa_fill_cpu, assert_identical, cases};
    use rasterrocket_gpu::backend::aa_fill_page;
    use rasterrocket_gpu::backend::vulkan::VulkanBackend;

    fn run_vulkan(backend: &VulkanBackend, case: &Case, eo: bool) -> Vec<u8> {
        aa_fill_page(backend, &case.table(), case.rect, case.aa_width, eo)
            .unwrap_or_else(|e| panic!("{}: Vulkan aa_fill failed: {e}", case.name))
    }

    #[test]
    fn vulkan_kernel_matches_host_twin() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new");
        for case in cases() {
            for eo in [false, true] {
                let expected = aa_fill_cpu(&case.table(), case.rect, case.aa_width, eo);
                let actual = run_vulkan(&backend, &case, eo);
                assert_identical(
                    &expected,
                    &actual,
                    &format!("{} (eo={eo}) Vulkan vs host twin", case.name),
                );
            }
        }
    }

    /// Pages from several threads on one backend: each runs whole, none
    /// fails on another's recording state.
    #[test]
    fn a_shared_backend_runs_pages_from_several_threads() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new");
        let cases = cases();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                // The scope joins every thread, propagating a panic.
                let _ = scope.spawn(|| {
                    for case in &cases {
                        for eo in [false, true] {
                            let expected = aa_fill_cpu(&case.table(), case.rect, case.aa_width, eo);
                            assert_identical(
                                &expected,
                                &run_vulkan(&backend, case, eo),
                                &format!("{} (eo={eo}) from a thread", case.name),
                            );
                        }
                    }
                });
            }
        });
    }

    /// A page that fails while recording is abandoned, and the next page
    /// runs.
    #[test]
    fn a_failed_page_leaves_the_backend_usable() {
        use rasterrocket_gpu::backend::{BackendError, GpuBackend};

        let backend = VulkanBackend::new().expect("VulkanBackend::new");
        let failed = backend.run_page(|_| Err(BackendError::msg("injected")));
        assert!(failed.is_err());
        let case = &cases()[0];
        let expected = aa_fill_cpu(&case.table(), case.rect, case.aa_width, false);
        assert_identical(
            &expected,
            &run_vulkan(&backend, case, false),
            "the page after a failed one",
        );
    }

    /// The two device kernels against each other, independent of the twin.
    #[cfg(feature = "gpu-validation")]
    #[test]
    fn vulkan_kernel_matches_cuda_kernel() {
        let backend = VulkanBackend::new().expect("VulkanBackend::new");
        let cuda_backend = super::cuda::backend();
        for case in cases() {
            for eo in [false, true] {
                let cuda = aa_fill_page(&cuda_backend, &case.table(), case.rect, case.aa_width, eo)
                    .unwrap_or_else(|e| panic!("{}: CUDA aa_fill failed: {e}", case.name));
                let vk = run_vulkan(&backend, &case, eo);
                assert_identical(
                    &cuda,
                    &vk,
                    &format!("{} (eo={eo}) Vulkan vs CUDA", case.name),
                );
            }
        }
    }
}
