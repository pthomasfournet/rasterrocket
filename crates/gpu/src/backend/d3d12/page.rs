//! Whole-page raster from a [`PageCommands`] list, and the backend's
//! [`PageDevice`] implementation: one command list, one submission and one
//! fence wait per page.
//!
//! The host bins the commands into tiles ([`PageCommands::bin`]) and stages
//! the five input arrays into persistently mapped UPLOAD buffers
//! ([`crate::backend::staged`]).  The page's command list copies them into
//! device buffers, runs the raster, one thread group per tile, and copies
//! the page into a persistently mapped READBACK buffer.  The inputs are
//! bound as structured SRVs (`t0..t4`): the commands in elements of
//! `size_of::<PageCmd>()` bytes, the other inputs in 32-bit words; the page
//! (`u0`) and, for a page with outlines, the status words (`u1`) as
//! structured UAVs of 32-bit words.  Each view covers the bytes the page
//! uses, not the whole of a buffer grown for an earlier page.  Every buffer
//! is kept between pages and grown on demand, one set per page between
//! staging and the end of its read-back.
//!
//! A view holds at most 2^27 elements: an input or a page past that is a
//! [`BackendError::DeviceLimit`] before anything is allocated.
//!
//! A page with outlines zeroes its status words before the raster and reads
//! them back with the pixels; a fault they record fails the page with
//! [`BackendError::KernelFault`] before any pixel is copied.
//!
//! A timed page ([`PageDevice::render_page_timed`]) also reads the host
//! clock between its host stages, writes timestamp queries between its
//! device stages and resolves them into a READBACK buffer at its end; an
//! untimed page does neither.

use std::sync::PoisonError;
use std::time::{Duration, Instant};

use super::D3d12Backend;
use super::memory::check_view_elements;
use super::pipeline::KernelId;
use super::recorder::Binding;
use super::timestamps::StageClock;
use crate::backend::kernel_status::{STATUS_BYTES, status_words};
use crate::backend::page_device::{PageDevice, PageTimes};
use crate::backend::page_job::{InputParts, Laps, PageJob};
use crate::backend::select::DeviceSelector;
use crate::backend::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage_parts,
};
use crate::backend::{BackendError, Result};
use crate::page::{PageCmd, PageCommands, PageSample};

/// The buffer sets of the page raster a backend keeps, one per page
/// between binning and the end of its read-back: while one page is on the
/// device, the others stage their inputs or read their pixels.
pub(super) const PAGE_SETS: usize = 3;

/// The bytes of a 32-bit word, the element of every view but the
/// commands'.
const WORD: u32 = 4;

/// The element stride of each input's view, in binding order: the
/// commands, the coverage table words, the row offsets, the tile offsets
/// and the tile items.
const INPUT_STRIDES: [u32; 5] = [CMD_STRIDE, WORD, WORD, WORD, WORD];

/// The bytes of one command, the element of the commands' view.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a command is 56 bytes, checked below"
)]
const CMD_STRIDE: u32 = size_of::<PageCmd>() as u32;
const _: () = assert!(size_of::<PageCmd>() == 56);

/// The buffers of the page raster, kept between pages.
#[derive(Default)]
pub(super) struct PageScratch {
    /// Commands, coverage table words, row offsets, tile offsets, tile
    /// items.
    inputs: [Option<Staged<D3d12Backend>>; 5],
    /// The page.
    target: Option<Staged<D3d12Backend>>,
    /// The status words of a page with outlines
    /// ([`crate::backend::kernel_status`]).
    status: Option<Staged<D3d12Backend>>,
    /// The timestamp queries of timed pages, created by the first.
    clock: Option<StageClock>,
}

/// The bytes the view of an input of `len` bytes in elements of `stride`
/// covers: at least one element, and a whole number of them.
fn view_len(len: usize, stride: u32) -> u64 {
    (len as u64).max(1).next_multiple_of(u64::from(stride))
}

/// The views of `inputs` in their order, checked with the page's view of
/// `page_len` bytes against the elements a view may hold.
///
/// # Errors
/// [`BackendError::DeviceLimit`] naming the first input, or the page, whose
/// view holds more.
fn check_views(inputs: &[InputParts<'_>; 5], page_len: usize) -> Result<[u64; 5]> {
    let lens: [u64; 5] = std::array::from_fn(|i| view_len(inputs[i].len(), INPUT_STRIDES[i]));
    for ((input, &len), &stride) in inputs.iter().zip(&lens).zip(&INPUT_STRIDES) {
        check_view_elements(input.what, len, stride)?;
    }
    check_view_elements("render_page pixels", page_len as u64, WORD)?;
    Ok(lens)
}

impl D3d12Backend {
    /// [`crate::backend::GpuBackend::run_page`] of `record` under the
    /// backend's page lock, and the time it held the lock: page runs hold
    /// the device one at a time, so that is the device's time for the run.
    fn run_held<F>(&self, record: F) -> (Result<()>, Duration)
    where
        F: FnOnce(&Self) -> Result<()>,
    {
        let _page = self.page.lock().unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        let run = crate::backend::run_page_steps(self, record);
        (run, started.elapsed())
    }

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
        let inputs = job.inputs();
        let view_lens = check_views(&inputs, page_len)?;
        let push: Vec<u32> = job
            .push
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_ne_bytes(*w))
            .collect();

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
            (Some(_), clock @ None) => {
                Some(&*clock.insert(StageClock::new(&self.device.device, &self.device.queue)?))
            }
        };
        let prepare = laps.lap();
        // Every buffer of the page is grown before any is written; when
        // memory runs out, the idle buffer sets are freed and the growth is
        // retried once.  Staging then allocates nothing.
        let reserve = |slots: &mut [Option<Staged<Self>>; 5],
                       target: &mut Option<Staged<Self>>,
                       status: &mut Option<Staged<Self>>| {
            for (slot, &len) in slots.iter_mut().zip(&view_lens) {
                let len = usize::try_from(len).map_err(BackendError::new)?;
                ensure(self, slot, len, Direction::ToDevice).map(drop)?;
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
            // An empty input is bound to a view of one element the kernel
            // never reads; nothing is written or copied.
            let staged_slot = if len == 0 {
                slot.as_mut()
                    .ok_or_else(|| BackendError::msg("render_page: an input slot is empty"))?
            } else {
                stage_parts(self, slot, input.parts())?
            };
            staged.push((staged_slot, len));
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
                backend.recorder.record_timestamp(c.heap(), query)
            })
        };
        let mut record = Duration::ZERO;
        let (run, held) = self.run_held(|backend| {
            stamp(backend, 0)?;
            for (input, len) in &mut staged {
                // SAFETY: the scratch is lent to this call alone for its
                // whole duration, and `run_held` returns only once the page
                // has completed or, on failure, been waited on or stranded
                // with its references held.
                unsafe { record_upload(backend, input, *len) }?;
            }
            if let Some(status) = status.as_deref() {
                backend
                    .recorder
                    .record_zero_buffer(&status.device, STATUS_BYTES)?;
            }
            stamp(backend, 1)?;
            {
                let srvs: Vec<Binding<'_>> = staged
                    .iter()
                    .zip(view_lens.iter().zip(INPUT_STRIDES))
                    .map(|((input, _), (&len, stride))| (&input.device, stride, len))
                    .collect();
                let mut uavs = Vec::with_capacity(2);
                uavs.push((&mut target.device, WORD, page_len as u64));
                if let Some(status) = status.as_deref_mut() {
                    uavs.push((&mut status.device, WORD, STATUS_BYTES as u64));
                }
                backend.recorder.dispatch_ranges(
                    kernel,
                    &srvs,
                    &uavs,
                    &push,
                    (job.bins.tiles_x, job.bins.tiles_y, 1),
                )?;
            }
            stamp(backend, 2)?;
            // SAFETY: as for the uploads.
            unsafe { record_download(backend, target, page_len) }?;
            if let Some(status) = status.as_deref_mut() {
                // SAFETY: as for the uploads.
                unsafe { record_download(backend, status, STATUS_BYTES) }?;
            }
            stamp(backend, 3)?;
            if let Some(c) = clock {
                backend.recorder.record_resolve_timestamps(
                    c.heap(),
                    StageClock::COUNT,
                    c.results(),
                )?;
            }
            record = laps.lap();
            Ok(())
        });
        run?;
        let submit_wait = laps.lap();

        // A fault means some pixel is wrong: no pixel reaches `out`.
        let words = match status.as_deref() {
            Some(status) => Some(status_words(
                // SAFETY: the page that wrote the buffer has completed.
                unsafe { host_view(self, status) }?
                    .first_chunk::<STATUS_BYTES>()
                    .ok_or_else(|| BackendError::msg("render_page: the status buffer is short"))?,
            )),
            None => None,
        };
        // SAFETY: the page that wrote the buffer has completed.
        let pixels = unsafe { host_view(self, target) }?;
        job.finish(pixels, words.as_ref(), out, stride)?;
        self.page_clock.record(held);
        let read = laps.lap();
        if let (Some(times), Some(clock)) = (times, clock) {
            let [device_upload, device_raster, device_download] = clock.intervals();
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

impl PageDevice for D3d12Backend {
    /// The adapter `selector` names (`warp` names the WARP adapter), or
    /// without one the first usable hardware adapter with dedicated video
    /// memory, else the first usable hardware adapter.
    fn open(selector: Option<&DeviceSelector>) -> Result<Self> {
        Self::open_adapter(selector)
    }

    /// The adapters are enumerated, a device being created and dropped on
    /// each for its capabilities; no queue or pipeline is built.
    fn check_selector(selector: &DeviceSelector) -> Result<()> {
        super::device::check_selector(selector)
    }

    fn resolve_devices(selectors: &[DeviceSelector]) -> Result<Vec<usize>> {
        super::device::resolve_devices(selectors)
    }

    /// The usable hardware adapters, those with dedicated video memory
    /// first, each group in enumeration order; never WARP.
    fn usable_gpus() -> Result<Vec<usize>> {
        super::device::usable_gpus()
    }

    /// The position in DXGI's high-performance enumeration order, WARP
    /// last.  WARP's index is not a selector of it: `warp` is.
    fn device_index(&self) -> usize {
        self.device.index
    }

    /// `DXGI_ADAPTER_DESC1::Description`.
    fn device_name(&self) -> String {
        self.device.name.clone()
    }

    /// The backend's first page raster includes the driver's first
    /// dispatch of the kernel.
    fn page_mean(&self) -> Option<Duration> {
        self.page_clock.mean()
    }

    /// `Int64ShaderOps`.
    fn page_paths(&self) -> bool {
        self.device.int64
    }

    /// Up to three calls stage their inputs and read their pixels at once,
    /// each with a buffer set of its own, the others waiting for one;
    /// other kernels' pages may run on the device between them.  An input
    /// or the page past the 2^27 elements of a view is a
    /// [`BackendError::DeviceLimit`]; when the page's buffers do not fit,
    /// the idle buffer sets of other pages are freed and the allocation is
    /// retried once before [`BackendError::OutOfMemory`].  A device removed
    /// during the page is an error naming the removal's reason, which
    /// retires the device.
    fn render_page<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None, KernelId::PageRasterPaths)
    }

    /// The device stages are timed by timestamp queries of the compute
    /// queue at its `GetTimestampFrequency`.
    fn render_page_timed<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<PageTimes> {
        let mut times = PageTimes::default();
        self.render_page_with(
            page,
            out,
            stride,
            Some(&mut times),
            KernelId::PageRasterPaths,
        )?;
        Ok(times)
    }
}

#[cfg(test)]
impl D3d12Backend {
    /// [`PageDevice::render_page`] with the floor divisions of its outlines'
    /// raster poisoned as by [`KernelId::PageRasterPathsPoisoned`].
    pub(crate) fn render_page_poisoned<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None, KernelId::PageRasterPathsPoisoned)
    }

    /// Whether the idle page buffer sets are within their bound.
    pub(crate) fn page_sets_within_bound(&self) -> bool {
        self.page_scratch.idle_len() <= PAGE_SETS
    }
}

#[cfg(test)]
mod tests {
    use color::SampleDepth;

    use super::{CMD_STRIDE, check_views, view_len};
    use crate::backend::BackendError;
    use crate::backend::d3d12::{D3d12Backend, test_backend};
    use crate::backend::page_device::PageDevice;
    use crate::backend::page_device::path_tests::random_page;
    use crate::backend::page_job::{PageJob, device_page_bytes};
    use crate::page::PageCommands;
    use crate::page::test_support::Rng;

    /// The most elements of a view.
    const MAX: usize = 1 << 27;

    #[test]
    fn views_cover_whole_elements_of_at_least_one() {
        assert_eq!(view_len(0, 56), 56);
        assert_eq!(view_len(56 * 3, 56), 56 * 3);
        assert_eq!(view_len(0, 4), 4);
        assert_eq!(view_len(9, 4), 12);
    }

    /// A row of `n` pixels whose packed samples fit a view of words and
    /// whose device words do not is refused, at both depths.
    #[test]
    fn the_page_view_check_sizes_the_word_page() {
        let page = PageCommands::new(1, 1, SampleDepth::Eight, [0; 3]);
        let job = PageJob::prepare::<u8>(&page, 3, 3, true, "page_raster_paths").expect("prepare");
        let inputs = job.inputs();
        for (packed, n) in [(3, MAX + 1), (6, MAX / 2 + 1)] {
            assert!(n * packed <= MAX * 4, "{n} pixels of {packed} bytes fit");
            let (_, len) = if packed == 3 {
                device_page_bytes::<u8>(n, 1)
            } else {
                device_page_bytes::<u16>(n, 1)
            }
            .expect("no overflow");
            let err = check_views(&inputs, len).expect_err("the word page exceeds the view");
            assert!(
                matches!(err, BackendError::DeviceLimit { what: "render_page pixels", need, .. }
                    if need == len as u64),
                "{err}"
            );
            let _ = check_views(&inputs, len - 8).expect("a page of whole words within the view");
        }
        assert_eq!(
            check_views(&inputs, 4).expect("a one-pixel page")[0],
            u64::from(CMD_STRIDE)
        );
    }

    /// The commands' view holds 56-byte elements, the others words.
    #[test]
    fn input_views_are_checked_in_their_own_elements() {
        assert!(super::check_view_elements("cmds", (MAX * 56) as u64, CMD_STRIDE).is_ok());
        assert!(super::check_view_elements("cmds", (MAX * 56 + 56) as u64, CMD_STRIDE).is_err());
        assert!(super::check_view_elements("table", (MAX * 4 + 4) as u64, 4).is_err());
    }

    /// A page past the view limit is a limit, not a fault of the device:
    /// it is refused before anything is staged, and keeps the device.
    #[test]
    fn an_oversized_page_is_a_device_limit() {
        let Some(backend) = test_backend() else {
            return;
        };
        let side = 11_586_u32;
        assert!((side as usize).pow(2) > MAX);
        let page = PageCommands::new(side, side, SampleDepth::Eight, [1, 2, 3]);
        let mut out = vec![0u8; (side as usize).pow(2) * 3];
        let err = backend
            .render_page(&page, &mut out, side as usize * 3)
            .expect_err("past the view limit");
        assert!(
            matches!(
                err,
                BackendError::DeviceLimit {
                    what: "render_page pixels",
                    ..
                }
            ),
            "{err}"
        );
        assert!(!D3d12Backend::retires_device(&err));
        assert!(out.iter().all(|&s| s == 0), "out is unchanged");
        assert!(
            backend.page_sets_within_bound(),
            "more buffer sets than the bound"
        );
    }

    #[test]
    fn a_faulting_raster_leaves_the_page_untouched() {
        let Some(backend) = test_backend() else {
            return;
        };
        if !backend.page_paths() {
            eprintln!("skipping: the adapter lacks Int64ShaderOps");
            return;
        }
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
        assert!(D3d12Backend::retires_device(&err));
        assert!(
            out.iter().all(|&s| s == 0xA5),
            "a faulting page wrote a sample"
        );
        backend
            .render_page(&page, &mut out, stride)
            .expect("the unpoisoned raster renders");
    }

    /// A page whose completion cannot be confirmed, as on a removed
    /// device, fails with an error that retires the device, leaves `out`
    /// unchanged, and every later page fails naming the likely cause.
    #[test]
    fn a_lost_page_retires_the_device() {
        let Some(backend) = test_backend() else {
            return;
        };
        let page = random_page(&mut Rng(0x7D2), 37, 21, 30, SampleDepth::Eight);
        let mut out = vec![0x5Au8; 37 * 3 * 21];
        backend
            .render_page(&page, &mut out, 37 * 3)
            .expect("a page before the loss");
        out.fill(0x5A);
        backend.submitter.fail_waits();
        let err = backend
            .render_page(&page, &mut out, 37 * 3)
            .expect_err("the wait fails");
        assert!(D3d12Backend::retires_device(&err), "{err}");
        assert!(out.iter().all(|&s| s == 0x5A), "a lost page wrote a sample");
        let err = backend
            .render_page(&page, &mut out, 37 * 3)
            .expect_err("the device is lost");
        assert!(D3d12Backend::retires_device(&err), "{err}");
        assert!(err.to_string().contains("TDR"), "{err}");
    }

    /// The timed page's device stages lie within its wait.
    #[test]
    fn d3d12_render_page_timed_reports_device_stages() {
        let Some(backend) = test_backend() else {
            return;
        };
        let depth = SampleDepth::Sixteen;
        let mut rng = Rng(0x71_3ED);
        let page = if backend.page_paths() {
            random_page(&mut rng, 403, 260, 300, depth)
        } else {
            PageCommands::new(403, 260, depth, [9; 3])
        };
        let mut out = vec![0u16; 403 * 3 * 260];
        for _ in 0..2 {
            let times = backend
                .render_page_timed(&page, &mut out, 403 * 3)
                .expect("render_page_timed");
            eprintln!(
                "adapter {} \"{}\", outlines on the device {}: {times:?}",
                backend.device_index(),
                backend.device_name(),
                backend.page_paths()
            );
            assert!(times.device_raster > std::time::Duration::ZERO, "{times:?}");
            assert!(
                times.device_upload + times.device_raster + times.device_download
                    <= times.submit_wait,
                "the device stages lie within the wait: {times:?}"
            );
        }
        assert!(backend.page_mean().is_some(), "the second page is counted");
    }
}
