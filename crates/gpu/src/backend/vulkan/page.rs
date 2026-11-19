//! Whole-page raster from a [`PageCommands`] list: one submission and one
//! wait per page.
//!
//! The host bins the commands into tiles ([`PageCommands::bin`]) and stages
//! the five input arrays ([`crate::backend::staged`]): in place on a device
//! whose buffers are host-mapped, otherwise into host buffers copied by the
//! page's own command buffer.  The page then records the raster dispatch,
//! one workgroup per tile, and on a device whose buffers the host does not
//! map, a copy of the page into a host read-back buffer.  Every buffer is
//! kept between pages and grown on demand.
//!
//! The checks, the push constants, the inputs and the read-back are the
//! backend's [`PageJob`]; this module stages, records and runs it.
//!
//! A page with outlines also binds a status buffer
//! ([`crate::backend::kernel_status`]), zeroed by the page's command
//! buffer before the raster and read back with the pixels; a fault it
//! records fails the page before any pixel is copied.
//!
//! A timed page ([`PageDevice::render_page_timed`]) also reads the host
//! clock between its host stages and writes device timestamps between its
//! device stages; an untimed page does neither.

use std::time::Duration;

use crate::backend::kernel_status::{STATUS_BYTES, status_words};
use crate::backend::page_device::{PageDevice, PageTimes};
use crate::backend::page_job::{InputParts, Laps, PageJob};
use crate::backend::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage_parts,
};
use crate::backend::{BackendError, Result};
use crate::page::{PageCommands, PageSample};

use super::VulkanBackend;
use super::pipeline::KernelId;
use super::timestamps::StageClock;

/// The bytes staged for an empty input: a descriptor binds a non-empty
/// range, and the kernel reads no element of an empty array.
const EMPTY_INPUT: [u8; 4] = [0; 4];

/// The buffer sets of the page raster a backend keeps, one per page
/// between binning and the end of its read-back: while one page is on the
/// device, the others stage their inputs or read their pixels.
pub(super) const PAGE_SETS: usize = 3;

/// The buffers of the page raster, kept between pages.
#[derive(Default)]
pub(super) struct PageScratch {
    /// Commands, coverage table words, row offsets, tile offsets, tile
    /// commands.
    inputs: [Option<Staged<VulkanBackend>>; 5],
    /// The page.
    target: Option<Staged<VulkanBackend>>,
    /// The status words of a page with outlines
    /// ([`crate::backend::kernel_status`]).
    status: Option<Staged<VulkanBackend>>,
    /// The timestamp queries of timed pages, created by the first.
    clock: Option<StageClock>,
}

/// `len` bytes as a device size.
fn device_size(len: usize) -> u64 {
    u64::try_from(len).expect("usize fits u64 on supported targets")
}

impl VulkanBackend {
    /// [`PageDevice::render_page`], timed into `times` when given, a page
    /// with outlines rasterised by the kernel `paths`.
    #[expect(
        clippy::too_many_lines,
        reason = "one page's stages in order: checks, staging, recording, read-back"
    )]
    pub(super) fn render_page_with<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
        times: Option<&mut PageTimes>,
        paths: KernelId,
    ) -> Result<()> {
        let mut laps = Laps::new(times.is_some());
        let job = PageJob::prepare::<S>(page, out.len(), stride, self.page_paths(), paths.label())?;
        // A page with outlines runs `paths` and binds the status buffer.
        let with_status = job.kernel_paths;
        let kernel = if with_status {
            paths
        } else {
            KernelId::PageRaster
        };
        let page_len = job.device_page_len;
        // Before any device buffer of the page is allocated: see
        // `host_pages`.
        super::host_pages::populate_for_write(&mut out[..job.span]);
        let inputs = job.inputs();
        for input in &inputs {
            self.device
                .check_storage_range(input.what, device_size(input.len()))?;
        }
        self.device
            .check_storage_range("render_page pixels", device_size(page_len))?;

        let mut scratch = self.page_scratch.lend();
        let PageScratch {
            inputs: slots,
            target,
            status,
            clock,
        } = &mut *scratch;
        let clock = match (&times, clock) {
            (None, _) => None,
            (Some(_), Some(clock)) => Some(&*clock),
            (Some(_), clock @ None) => Some(&*clock.insert(StageClock::new(self.device.clone())?)),
        };
        let prepare = laps.lap();
        let staged_len = |input: &InputParts<'_>| input.len().max(EMPTY_INPUT.len());
        // Every buffer of the page is grown before any is written; when
        // memory runs out, the idle buffer sets are freed and the growth is
        // retried once.  Staging then allocates nothing.
        let reserve = |slots: &mut [Option<Staged<Self>>; 5],
                       target: &mut Option<Staged<Self>>,
                       status: &mut Option<Staged<Self>>| {
            for (slot, input) in slots.iter_mut().zip(&inputs) {
                ensure(self, slot, staged_len(input), Direction::ToDevice).map(drop)?;
            }
            ensure(self, target, page_len, Direction::ToHost).map(drop)?;
            if with_status {
                ensure(self, status, STATUS_BYTES, Direction::ToHost).map(drop)?;
            }
            Ok(())
        };
        match reserve(slots, target, status) {
            Err(BackendError::OutOfMemory { .. }) if self.page_scratch.free_idle() > 0 => {
                reserve(slots, target, status)?;
            }
            reserved => reserved?,
        }
        let mut staged = Vec::with_capacity(slots.len());
        for (slot, input) in slots.iter_mut().zip(&inputs) {
            let len = input.len();
            let parts = if len == 0 {
                &[&EMPTY_INPUT[..]]
            } else {
                input.parts()
            };
            staged.push((stage_parts(self, slot, parts)?, len));
        }
        let target = ensure(self, target, page_len, Direction::ToHost)?;
        let mut status = if with_status {
            Some(ensure(self, status, STATUS_BYTES, Direction::ToHost)?)
        } else {
            None
        };
        let write = laps.lap();

        // Query `i` of a timed page is written once every command before it
        // has completed: 0 before the uploads, 1 after them and the zeroing
        // of the status words, 2 after the raster, 3 after the downloads.
        let stamp = |backend: &Self, query: u32| {
            clock.map_or(Ok(()), |c| {
                backend.recorder.record_timestamp(c.pool(), query)
            })
        };
        let mut record = Duration::ZERO;
        // SAFETY (the recorded copies and the read below): the scratch is
        // lent to this call alone for its whole duration, and `run_page` returns
        // only once its page has completed or, on failure, been waited on;
        // nothing touches the buffers in between.
        let (run, held) = crate::backend::run_page_held(self, &self.page, |backend| {
            if let Some(c) = clock {
                backend
                    .recorder
                    .record_reset_queries(c.pool(), StageClock::COUNT)?;
            }
            stamp(backend, 0)?;
            let mut handles = Vec::with_capacity(staged.len() + 1);
            let mut sizes = Vec::with_capacity(staged.len() + 1);
            for (input, len) in &mut staged {
                let len = *len;
                unsafe { record_upload(backend, input, len) }?;
                handles.push(input.device.handle());
                sizes.push(device_size(len));
            }
            if let Some(status) = status.as_deref() {
                backend
                    .recorder
                    .record_zero_buffer(&status.device, STATUS_BYTES)?;
            }
            stamp(backend, 1)?;
            handles.push(target.device.handle());
            sizes.push(device_size(page_len));
            if let Some(status) = status.as_deref() {
                handles.push(status.device.handle());
                sizes.push(device_size(STATUS_BYTES));
            }
            backend.recorder.dispatch_kernel(
                kernel,
                &handles,
                &sizes,
                &job.push,
                (job.bins.tiles_x, job.bins.tiles_y, 1),
            )?;
            stamp(backend, 2)?;
            unsafe { record_download(backend, target, page_len) }?;
            if let Some(status) = status.as_deref_mut() {
                unsafe { record_download(backend, status, STATUS_BYTES) }?;
            }
            stamp(backend, 3)?;
            record = laps.lap();
            Ok(())
        });
        run?;
        let submit_wait = laps.lap();

        // A fault means some pixel is wrong: no pixel reaches `out`.
        let words = match status.as_deref() {
            Some(status) => Some(status_words(
                unsafe { host_view(self, status) }?
                    .first_chunk::<STATUS_BYTES>()
                    .ok_or_else(|| BackendError::msg("render_page: the status buffer is short"))?,
            )),
            None => None,
        };
        let pixels = unsafe { host_view(self, target) }?;
        job.finish(pixels, words.as_ref(), out, stride)?;
        self.page_clock.record(held);
        let read = laps.lap();
        if let (Some(times), Some(clock)) = (times, clock) {
            let [device_upload, device_raster, device_download] = clock.intervals()?;
            *times = PageTimes {
                prepare,
                write,
                record,
                submit_wait,
                read,
                device_upload,
                device_raster,
                device_download,
            };
        }
        drop(scratch);
        Ok(())
    }
}

/// The quotients of a [`VulkanBackend::floor_div_probe`], and the operands
/// `(a, b)` of the first division it recorded as a fault.
#[cfg(test)]
pub(crate) type FloorDivProbe = (Vec<i64>, Option<(i64, i64)>);

#[cfg(test)]
impl VulkanBackend {
    /// The device's floor division of each pair `(a, b)` of `pairs`, with
    /// `max_steps` correction steps in each direction, and the operands of
    /// the first division it records as a fault; one page.
    pub(crate) fn floor_div_probe(
        &self,
        pairs: &[(i64, i64)],
        max_steps: u32,
    ) -> Result<FloorDivProbe> {
        let (quotients, status) =
            self.floor_div_probe_on(KernelId::FloorDivProbe, pairs, max_steps)?;
        Ok((
            quotients,
            crate::backend::kernel_status::floor_div_fault(&status),
        ))
    }

    /// [`PageDevice::render_page`] with the floor divisions of its outlines'
    /// raster poisoned as by [`KernelId::FloorDivProbePoisoned`].
    pub(crate) fn render_page_poisoned<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None, KernelId::PageRasterPathsPoisoned)
    }

    /// The quotients of [`Self::floor_div_probe`] by the probe kernel
    /// `kernel`, and the status words of the page.
    pub(super) fn floor_div_probe_on(
        &self,
        kernel: KernelId,
        pairs: &[(i64, i64)],
        max_steps: u32,
    ) -> Result<(Vec<i64>, [u32; crate::backend::kernel_status::STATUS_WORDS])> {
        /// The low and high words of `v`.
        fn words(v: i64) -> [u32; 2] {
            let v = v.cast_unsigned();
            [
                u32::try_from(v & 0xFFFF_FFFF).expect("32 bits"),
                u32::try_from(v >> 32).expect("32 bits"),
            ]
        }
        let n = u32::try_from(pairs.len()).map_err(BackendError::new)?;
        let operands: Vec<u32> = pairs
            .iter()
            .flat_map(|&(a, b)| words(a).into_iter().chain(words(b)))
            .collect();
        let operands = bytemuck::cast_slice::<u32, u8>(&operands);
        let quotients_len = pairs.len() * 8;
        let mut push = [0u8; KernelId::FloorDivProbe.push_bytes()];
        push[..4].copy_from_slice(&n.to_ne_bytes());
        push[4..].copy_from_slice(&max_steps.to_ne_bytes());
        let (mut input, mut output, mut status) = (None, None, None);
        let input = stage_parts(self, &mut input, &[operands])?;
        let output = ensure(self, &mut output, quotients_len, Direction::ToHost)?;
        let status = ensure(self, &mut status, STATUS_BYTES, Direction::ToHost)?;
        // SAFETY: the three buffers are this call's own, and `run_page`
        // returns only once the page has completed or been waited on.
        crate::backend::GpuBackend::run_page(self, |backend| {
            unsafe { record_upload(backend, input, operands.len()) }?;
            backend
                .recorder
                .record_zero_buffer(&status.device, STATUS_BYTES)?;
            backend.recorder.dispatch_kernel(
                kernel,
                &[
                    input.device.handle(),
                    output.device.handle(),
                    status.device.handle(),
                ],
                &[
                    device_size(operands.len()),
                    device_size(quotients_len),
                    device_size(STATUS_BYTES),
                ],
                &push,
                (n.div_ceil(64), 1, 1),
            )?;
            unsafe { record_download(backend, output, quotients_len) }?;
            unsafe { record_download(backend, status, STATUS_BYTES) }
        })?;
        let quotients = unsafe { host_view(self, output) }?[..quotients_len]
            .as_chunks::<4>()
            .0
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[lo, hi]| {
                crate::backend::kernel_status::join(u32::from_ne_bytes(lo), u32::from_ne_bytes(hi))
            })
            .collect();
        let status = unsafe { host_view(self, status) }?
            .first_chunk::<STATUS_BYTES>()
            .ok_or_else(|| BackendError::msg("floor_div_probe: the status buffer is short"))?;
        Ok((quotients, status_words(status)))
    }
}

#[cfg(test)]
mod tests {
    use color::SampleDepth;

    use crate::backend::BackendError;
    use crate::backend::kernel_status::{
        FAULT_FLOOR_DIV, FAULT_FLOOR_DIV_WRAP, FLOOR_DIV_STEPS, floor_div_cases, floor_div_fault,
    };
    use crate::backend::page_device::PageDevice;
    use crate::backend::page_device::path_tests::random_page;
    use crate::backend::page_job::device_page_bytes;
    use crate::backend::vulkan::{test_backend, test_backend_with_int64};
    use crate::page::test_support::Rng;

    #[test]
    fn bindings_past_the_storage_range_are_refused() {
        let Some(backend) = test_backend() else {
            return;
        };
        let max = u64::from(backend.device.max_storage_buffer_range);
        assert!(backend.device.check_storage_range("input", max).is_ok());
        let err = backend
            .device
            .check_storage_range("input", max + 1)
            .expect_err("past the range");
        assert!(
            matches!(err, crate::backend::BackendError::DeviceLimit { need, .. } if need == max + 1),
            "{err}"
        );
    }

    #[test]
    fn the_page_range_check_sizes_the_word_page() {
        let Some(backend) = test_backend() else {
            return;
        };
        let max = usize::try_from(backend.device.max_storage_buffer_range)
            .expect("u32 fits usize on supported targets");
        // A row of `n` pixels whose packed samples fit the range and whose
        // device words do not.
        for (packed, n) in [(3, max / 4 + 1), (6, max / 8 + 1)] {
            assert!(n * packed <= max, "{n} pixels of {packed} bytes fit");
            let (_, len) = if packed == 3 {
                device_page_bytes::<u8>(n, 1)
            } else {
                device_page_bytes::<u16>(n, 1)
            }
            .expect("no overflow");
            let err = backend
                .device
                .check_storage_range("render_page pixels", super::device_size(len))
                .expect_err("the word page exceeds the range");
            assert!(
                matches!(err, BackendError::DeviceLimit { need, .. } if need == super::device_size(len)),
                "{err}"
            );
        }
        assert_eq!(device_page_bytes::<u8>(5, 7), Some((20, 140)));
        assert_eq!(device_page_bytes::<u16>(5, 7), Some((40, 280)));
        assert_eq!(device_page_bytes::<u8>(usize::MAX / 2, 1), None);
    }

    #[test]
    fn vulkan_floor_div_matches_the_host_model() {
        let Some(b) = test_backend_with_int64() else {
            return;
        };
        let cases = floor_div_cases(&mut Rng(0x5eed));
        let (q, fault) = b.floor_div_probe(&cases, FLOOR_DIV_STEPS).unwrap();
        assert_eq!(fault, None);
        for (&(a, d), q) in cases.iter().zip(q) {
            assert_eq!(q, crate::page::path::floor_div(a, d), "{a} / {d}");
        }
    }

    #[test]
    fn vulkan_floor_div_reports_an_exhausted_bound() {
        let Some(b) = test_backend_with_int64() else {
            return;
        };
        // Truncation toward zero leaves r = -1: one correction step is needed.
        let (_, fault) = b.floor_div_probe(&[(-1, 3)], 0).unwrap();
        assert_eq!(fault, Some((-1, 3)));
    }

    #[test]
    fn vulkan_floor_div_reports_an_exhausted_upward_bound() {
        let Some(b) = test_backend_with_int64() else {
            return;
        };
        // Outside the domain: with |a / b| near 2^61 the second estimate
        // divides a remainder of about 2^37, whose float rounding leaves
        // r = 3 123 above the quotient after it, past 16 upward steps.
        let pair = (2_404_050_171_065_969_715, 1);
        let (_, status) = b
            .floor_div_probe_on(super::KernelId::FloorDivProbe, &[pair], FLOOR_DIV_STEPS)
            .unwrap();
        assert_eq!(status[0], FAULT_FLOOR_DIV);
        assert_eq!(floor_div_fault(&status), Some(pair));
    }

    #[test]
    fn the_first_faulting_division_writes_the_operands() {
        let Some(b) = test_backend_with_int64() else {
            return;
        };
        // Every pair leaves r = -1 after the estimates: all fault at once.
        let pairs: Vec<(i64, i64)> = (0..4096)
            .map(|k: i64| {
                let d = 3 + k % 1000;
                (-(k * d + 1), d)
            })
            .collect();
        let (_, fault) = b.floor_div_probe(&pairs, 0).unwrap();
        let fault = fault.expect("a fault");
        assert!(pairs.contains(&fault), "{fault:?} is no single pair");
    }

    #[test]
    fn a_wrapped_estimate_is_a_fault() {
        let Some(b) = test_backend_with_int64() else {
            return;
        };
        let poisoned = |pairs: &[(i64, i64)]| {
            b.floor_div_probe_on(
                super::KernelId::FloorDivProbePoisoned,
                pairs,
                FLOOR_DIV_STEPS,
            )
            .unwrap()
        };
        // The poisoned first estimate makes q * b wrap; the remainder
        // correction then leaves r in [0, b), and only the wrap guard sees
        // the wrong quotient.  Each pair runs alone, so its own fault is
        // the one recorded.
        let mut cases = vec![
            (0, 2),
            (-5, 3),
            (12_345 * (1 << 31) + 7, 1 << 31),
            (1 << 62, (1 << 32) - 1),
        ];
        let drawn = floor_div_cases(&mut Rng(0x0bad));
        cases.extend(drawn.iter().filter(|&&(_, d)| d >= 2).step_by(997).take(64));
        for pair in cases {
            let (q, status) = poisoned(&[pair]);
            assert_ne!(
                q[0],
                crate::page::path::floor_div(pair.0, pair.1),
                "{pair:?}"
            );
            assert_eq!(status[0], FAULT_FLOOR_DIV_WRAP, "{pair:?}");
            assert_eq!(floor_div_fault(&status), Some(pair));
        }
        // Over the whole sweep the poison wrongs every quotient, and the
        // division recorded is one of the batch's.
        let batch: Vec<(i64, i64)> = drawn.into_iter().filter(|&(_, d)| d >= 2).collect();
        let (q, status) = poisoned(&batch);
        for (&(a, d), q) in batch.iter().zip(q) {
            assert_ne!(q, crate::page::path::floor_div(a, d), "{a} / {d}");
        }
        assert_eq!(status[0], FAULT_FLOOR_DIV_WRAP);
        assert!(batch.contains(&floor_div_fault(&status).expect("a fault")));
    }

    #[test]
    fn a_faulting_raster_leaves_the_page_untouched() {
        let Some(backend) = test_backend_with_int64() else {
            return;
        };
        let page = random_page(&mut Rng(0xFA17), 37, 21, 30, SampleDepth::Eight);
        let stride = 37 * 3 + 5;
        let mut out = vec![0xA5u8; stride * 21];
        let err = backend
            .render_page_poisoned(&page, &mut out, stride)
            .expect_err("a poisoned raster faults");
        assert!(
            matches!(
                err,
                BackendError::KernelFault {
                    kernel: "page_raster_paths_poisoned",
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("wrapped"), "{err}");
        assert!(
            out.iter().all(|&s| s == 0xA5),
            "a faulting page wrote a sample"
        );
        backend
            .render_page(&page, &mut out, stride)
            .expect("the unpoisoned raster renders");
    }
}
