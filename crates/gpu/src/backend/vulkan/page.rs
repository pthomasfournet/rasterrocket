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
//! The device page holds one device pixel per pixel, in whole 32-bit words
//! ([`crate::backend::layout`]); the read-back drops each pixel's pad
//! bytes and writes each row's packed samples at its stride in the
//! caller's buffer, leaving the samples between rows unchanged.
//!
//! A page with outlines also binds a status buffer
//! ([`crate::backend::kernel_status`]), zeroed by the page's command
//! buffer before the raster and read back with the pixels; a fault it
//! records fails the page before any pixel is copied.
//!
//! A timed page ([`VulkanBackend::render_page_timed`]) also reads the host
//! clock between its host stages and writes device timestamps between its
//! device stages; an untimed page does neither.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::backend::kernel_status::{self, STATUS_BYTES, STATUS_WORDS, status_words};
use crate::backend::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage_parts,
};
use crate::backend::{BackendError, Result};
use crate::page::{PageCommands, PageSample, TableBases, TileBins, pack_rgb};

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

/// The time page rasters hold the device: an exponentially weighted mean
/// of the page rasters that succeeded, the first excluded.
#[derive(Default)]
pub(super) struct PageClock {
    /// Set by the first page raster that succeeded, which is not counted.
    warm: AtomicBool,
    /// The mean in nanoseconds, at least 1; 0 while none is counted.
    mean_nanos: AtomicU64,
}

impl PageClock {
    /// The weight of the newest run is `1 / WEIGHT`.
    const WEIGHT: u64 = 8;

    /// Count a page raster that held the device for `held`.
    fn record(&self, held: Duration) {
        if !self.warm.swap(true, Ordering::Relaxed) {
            return;
        }
        let new = u64::try_from(held.as_nanos()).unwrap_or(u64::MAX).max(1);
        let _ = self
            .mean_nanos
            .update(Ordering::Relaxed, Ordering::Relaxed, |mean| {
                if mean == 0 {
                    return new;
                }
                let total = u128::from(mean) * u128::from(Self::WEIGHT - 1) + u128::from(new);
                u64::try_from(total / u128::from(Self::WEIGHT))
                    .unwrap_or(u64::MAX)
                    .max(1)
            });
    }

    /// The mean, once a run is counted.
    pub(super) fn mean(&self) -> Option<Duration> {
        match self.mean_nanos.load(Ordering::Relaxed) {
            0 => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }
}

/// The duration of each stage of one page rendered by
/// [`VulkanBackend::render_page_timed`].
///
/// The host stages run in order and together span the call; the device
/// stages are measured by timestamps the page's command buffer writes
/// between them, so they lie within `submit_wait`.  A transfer the device
/// does not need (its buffers are host-mapped) takes no device time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageTimes {
    /// Host: binning the commands into tiles and packing the coverage
    /// tables.
    pub prepare: Duration,
    /// Host: writing the inputs into the staging buffers, or into the
    /// device buffers in place.
    pub write: Duration,
    /// Host: beginning and recording the page's command buffer.
    pub record: Duration,
    /// Host: submitting the page and waiting for it to complete.
    pub submit_wait: Duration,
    /// Host: copying the pixels into the caller's samples.
    pub read: Duration,
    /// Device: copying the inputs from the staging buffers, and zeroing
    /// the status words of a page with outlines.
    pub device_upload: Duration,
    /// Device: the raster dispatch.
    pub device_raster: Duration,
    /// Device: copying the page into the read-back buffer.
    pub device_download: Duration,
}

/// The host clock of a timed page: the time since the previous lap, or
/// nothing for an untimed page, which never reads the clock.
struct Laps(Option<Instant>);

impl Laps {
    fn new(timed: bool) -> Self {
        Self(timed.then(Instant::now))
    }

    /// The time since the previous lap; zero when untimed.
    fn lap(&mut self) -> Duration {
        self.0.as_mut().map_or(Duration::ZERO, |last| {
            let now = Instant::now();
            let lap = now - *last;
            *last = now;
            lap
        })
    }
}

/// The push constants of the raster of `page` binned as `bins`, its table
/// words laid out as `bases`, in the order of `Params` in
/// `kernels/page_raster.slang`.
fn push_constants(page: &PageCommands, bins: &TileBins, bases: TableBases) -> [u8; 56] {
    let n_cmds = u32::try_from(page.len()).expect("the push limit bounds the commands");
    let [paper_rg, paper_b] = pack_rgb(page.paper());
    let values = [
        n_cmds,
        page.width(),
        page.height(),
        bins.tiles_x,
        bins.tiles_y,
        bases.coverage,
        paper_rg,
        paper_b,
        u32::from(page.depth().bits()),
        bases.segs,
        bases.seg_lists,
        bases.backdrops,
        bases.images,
        bases.mods,
    ];
    let mut push = [0u8; 56];
    for (dst, v) in push.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        dst.copy_from_slice(&v.to_ne_bytes());
    }
    push
}

/// `Ok` when the status words of the raster `kernel` of a page with
/// outlines record no fault.
///
/// # Errors
/// [`BackendError::KernelFault`] of `kernel` naming the fault.
fn check_paths_status(kernel: KernelId, words: &[u32; STATUS_WORDS]) -> Result<()> {
    kernel_status::check(kernel.label(), words)
}

/// `len` bytes as a device size.
fn device_size(len: usize) -> u64 {
    u64::try_from(len).expect("usize fits u64 on supported targets")
}

impl VulkanBackend {
    /// [`GpuBackend::run_page`] of `record` under the backend's page lock,
    /// and the time it held the lock: page runs hold the device one at a
    /// time, so that is the device's time for the run.
    pub(super) fn run_held<F>(&self, record: F) -> (Result<()>, Duration)
    where
        F: FnOnce(&Self) -> Result<()>,
    {
        let _page = self
            .page
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let started = Instant::now();
        let run = crate::backend::run_page_steps(self, record);
        (run, started.elapsed())
    }

    /// Whether the device rasterises outlines ([`crate::page::AreaPath`]):
    /// it supports 64-bit integers in kernels.  A page with outlines
    /// renders only on such a device.
    #[must_use]
    pub fn page_paths(&self) -> bool {
        self.device.has_int64
    }

    /// Render `page` on the device and write its pixels into `out`, RGB
    /// samples of the page's depth: row `y` at
    /// `out[y × stride..][..width × 3]`, `stride` in samples.  Samples
    /// between rows are left unchanged.
    ///
    /// Concurrent calls stage their inputs and read their pixels in
    /// parallel, up to three at once (each with buffers of its own, the
    /// others waiting for one), and run on the device one at a
    /// time; other kernels' pages may run between them.
    ///
    /// # Errors
    ///
    /// [`BackendError::DeviceLimit`] when an input or the page exceeds the
    /// device's `maxStorageBufferRange`, before anything is allocated;
    /// [`BackendError::OutOfMemory`] when the page's buffers do not fit in
    /// the memory left, also once the idle buffer sets of other pages are
    /// freed; [`BackendError::KernelFault`] when the raster of the page's
    /// outlines reports a division it could not complete exactly, its
    /// pixels then being discarded; otherwise an error when the page has no
    /// pixel, its depth is not that of `S`, `out` or `stride` is too small
    /// for it, an allocation fails otherwise, or the device reports an
    /// error.  `out` is unchanged then.
    pub fn render_page<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None, KernelId::PageRasterPaths)
    }

    /// As [`Self::render_page`], and return the duration of each of the
    /// page's stages.
    ///
    /// # Errors
    ///
    /// As [`Self::render_page`]; also when the device's compute queue
    /// writes no timestamps, before anything is staged.
    pub fn render_page_timed<S: PageSample>(
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

    /// [`Self::render_page`], timed into `times` when given, a page with
    /// outlines rasterised by the kernel `paths`.
    #[expect(
        clippy::too_many_lines,
        reason = "one page's stages in order: checks, staging, recording, read-back"
    )]
    fn render_page_with<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
        times: Option<&mut PageTimes>,
        paths: KernelId,
    ) -> Result<()> {
        let mut laps = Laps::new(times.is_some());
        let (width, height) = (page.width() as usize, page.height() as usize);
        if width == 0 || height == 0 {
            return Err(BackendError::msg("render_page: the page has no pixel"));
        }
        let overflow = || {
            BackendError::msg(format!(
                "render_page: a {width}×{height} page of {}-bit samples at stride {stride} \
                 exceeds the address space",
                S::DEPTH.bits()
            ))
        };
        // The samples of a row, the bytes of a device row and of the device
        // page, and the samples of `out` the page spans: the last row ends
        // `row` samples after the start of row `height - 1`.
        let row = width.checked_mul(3).ok_or_else(overflow)?;
        let device_row = width
            .checked_mul(S::DEVICE_PIXEL_BYTES)
            .ok_or_else(overflow)?;
        let page_len = device_row.checked_mul(height).ok_or_else(overflow)?;
        let span = stride
            .checked_mul(height - 1)
            .and_then(|n| n.checked_add(row))
            .ok_or_else(overflow)?;
        if page.depth() != S::DEPTH {
            return Err(BackendError::msg(format!(
                "render_page: a {}-bit page does not render into {}-bit samples",
                page.depth().bits(),
                S::DEPTH.bits()
            )));
        }
        if stride < row || out.len() < span {
            return Err(BackendError::msg(format!(
                "render_page: a {width}×{height} page does not fit {} samples at stride {stride}",
                out.len()
            )));
        }
        let kernel = if page.has_paths() {
            if !self.page_paths() {
                return Err(BackendError::msg(
                    "render_page: the page has outlines, which this device does not rasterise",
                ));
            }
            paths
        } else {
            KernelId::PageRaster
        };
        // Before any device buffer of the page is allocated: see
        // `host_pages`.
        super::host_pages::populate_for_write(&mut out[..span]);
        let bins = page.bin().map_err(BackendError::new)?;
        let (table, bases) = page.table_parts();
        let push = push_constants(page, &bins, bases);
        let inputs: [(&'static str, &[&[u8]]); 5] = [
            ("render_page commands", &[bytemuck::cast_slice(page.cmds())]),
            ("render_page coverage table", &table),
            (
                "render_page row offsets",
                &[bytemuck::cast_slice(page.row_starts())],
            ),
            (
                "render_page tile offsets",
                &[bytemuck::cast_slice(&bins.offsets)],
            ),
            (
                "render_page tile items",
                &[bytemuck::cast_slice(&bins.items)],
            ),
        ];
        let input_len = |parts: &[&[u8]]| parts.iter().map(|p| p.len()).sum::<usize>();
        for (what, parts) in inputs {
            self.device
                .check_storage_range(what, device_size(input_len(parts)))?;
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
        let staged_len = |parts: &[&[u8]]| input_len(parts).max(EMPTY_INPUT.len());
        // Every buffer of the page is grown before any is written; when
        // memory runs out, the idle buffer sets are freed and the growth is
        // retried once.  Staging then allocates nothing.
        let reserve = |slots: &mut [Option<Staged<Self>>; 5],
                       target: &mut Option<Staged<Self>>,
                       status: &mut Option<Staged<Self>>| {
            for (slot, (_, parts)) in slots.iter_mut().zip(inputs) {
                ensure(self, slot, staged_len(parts), Direction::ToDevice).map(drop)?;
            }
            ensure(self, target, page_len, Direction::ToHost).map(drop)?;
            if kernel == KernelId::PageRasterPaths {
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
        for (slot, (_, parts)) in slots.iter_mut().zip(inputs) {
            let parts = if input_len(parts) == 0 {
                &[&EMPTY_INPUT[..]]
            } else {
                parts
            };
            staged.push((stage_parts(self, slot, parts)?, input_len(parts)));
        }
        let target = ensure(self, target, page_len, Direction::ToHost)?;
        let mut status = if page.has_paths() {
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
        let (run, held) = self.run_held(|backend| {
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
                &push,
                (bins.tiles_x, bins.tiles_y, 1),
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
        if let Some(status) = status.as_deref() {
            let bytes = unsafe { host_view(self, status) }?
                .first_chunk::<STATUS_BYTES>()
                .ok_or_else(|| BackendError::msg("render_page: the status buffer is short"))?;
            check_paths_status(kernel, &status_words(bytes))?;
        }
        self.page_clock.record(held);
        let pixels = &unsafe { host_view(self, target) }?[..page_len];
        for (dst, src) in out.chunks_mut(stride).zip(pixels.chunks_exact(device_row)) {
            S::copy_from_words(&mut dst[..row], src);
        }
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
        self.floor_div_probe_on(KernelId::FloorDivProbe, pairs, max_steps)
    }

    /// [`Self::floor_div_probe`] with every first estimate offset so that
    /// its product with the divisor wraps.
    pub(crate) fn floor_div_probe_poisoned(
        &self,
        pairs: &[(i64, i64)],
        max_steps: u32,
    ) -> Result<FloorDivProbe> {
        self.floor_div_probe_on(KernelId::FloorDivProbePoisoned, pairs, max_steps)
    }

    /// [`Self::render_page`] with the floor divisions of its outlines'
    /// raster poisoned as in [`Self::floor_div_probe_poisoned`].
    pub(crate) fn render_page_poisoned<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None, KernelId::PageRasterPathsPoisoned)
    }

    /// [`Self::floor_div_probe`] by the probe kernel `kernel`.
    fn floor_div_probe_on(
        &self,
        kernel: KernelId,
        pairs: &[(i64, i64)],
        max_steps: u32,
    ) -> Result<FloorDivProbe> {
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
        let mut push = [0u8; 8];
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
            .map(|&[lo, hi]| kernel_status::join(u32::from_ne_bytes(lo), u32::from_ne_bytes(hi)))
            .collect();
        let status = unsafe { host_view(self, status) }?
            .first_chunk::<STATUS_BYTES>()
            .ok_or_else(|| BackendError::msg("floor_div_probe: the status buffer is short"))?;
        Ok((
            quotients,
            kernel_status::floor_div_fault(&status_words(status)),
        ))
    }
}

#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod mask_tests;
#[cfg(test)]
mod modifier_tests;
#[cfg(test)]
mod path_tests;

#[cfg(test)]
mod tests {
    use color::{Sample, SampleDepth};

    use crate::backend::BackendError;
    use crate::backend::kernel_status::{FLOOR_DIV_STEPS, floor_div_cases};
    use crate::backend::vulkan::test_backend_with_int64;
    use crate::page::test_support::Rng;
    use crate::page::{CoverageRows, PageCommands, PageSample, PixelBounds};

    /// The clock ignores the first page raster, takes the second as its
    /// mean, and weighs each later one by 1/8, so a slow run is forgotten.
    #[test]
    fn the_page_clock_skips_the_first_run_and_forgets_slow_ones() {
        use std::time::Duration;
        let clock = super::PageClock::default();
        clock.record(Duration::from_millis(80));
        assert_eq!(clock.mean(), None);
        clock.record(Duration::from_millis(16));
        assert_eq!(clock.mean(), Some(Duration::from_millis(16)));
        clock.record(Duration::from_millis(80));
        assert_eq!(clock.mean(), Some(Duration::from_millis(24)));
        for _ in 0..64 {
            clock.record(Duration::from_millis(2));
        }
        let mean = clock.mean().expect("a mean");
        assert!(mean < Duration::from_micros(2010), "{mean:?}");
        clock.record(Duration::ZERO);
        assert!(clock.mean().is_some_and(|m| m > Duration::ZERO));
    }

    /// One command as the host model paints it.
    struct HostCmd {
        y_min: i32,
        row_starts: Vec<u32>,
        spans: Vec<[i32; 3]>,
        bytes: Vec<u8>,
        bounds: PixelBounds,
        rgb: [u16; 3],
        a_input: u16,
    }

    impl HostCmd {
        fn table(&self) -> CoverageRows<'_> {
            CoverageRows {
                y_min: self.y_min,
                row_starts: &self.row_starts,
                spans: &self.spans,
                bytes: &self.bytes,
            }
        }

        /// The coverage of pixel `(x, y)`, by a linear walk of its row.
        fn coverage(&self, x: i32, y: i32) -> u8 {
            let Some(row) = usize::try_from(y - self.y_min)
                .ok()
                .filter(|&r| r + 1 < self.row_starts.len())
            else {
                return 0;
            };
            let spans =
                &self.spans[self.row_starts[row] as usize..self.row_starts[row + 1] as usize];
            spans
                .iter()
                .find(|s| (s[0]..=s[1]).contains(&x))
                .map_or(0, |&[x0, _, offset]| {
                    if offset < 0 {
                        255
                    } else {
                        self.bytes[usize::try_from(offset + x - x0).unwrap()]
                    }
                })
        }
    }

    /// The sample of `S` of value `v`, which fits it.
    fn sample<S: Sample>(v: u16) -> S {
        S::from_wide(<S::Wide as color::Wide>::of(u32::from(v)))
    }

    /// The page `cmds` paint at the depth of `S`, composited on the host
    /// as the CPU's AA pipe does: the coverage byte widened exactly to a
    /// shape, `a = div_max(a_input · shape)` and
    /// `c = div_max((M − a) · c_dst + a · c_src)`.
    #[expect(
        clippy::many_single_char_names,
        reason = "pixel coordinates, coverage and alpha of the kernel's formulas"
    )]
    fn host_page<S: Sample>(width: u32, height: u32, paper: [u16; 3], cmds: &[HostCmd]) -> Vec<S> {
        let (w, h) = (width as usize, i32::try_from(height).unwrap());
        let max = S::max_wide();
        let mut out: Vec<S> = (0..width * height)
            .flat_map(|_| paper.map(sample::<S>))
            .collect();
        for c in cmds {
            for y in c.bounds.y0.max(0)..=c.bounds.y1.min(h - 1) {
                for x in c.bounds.x0.max(0)..=c.bounds.x1.min(i32::try_from(w).unwrap() - 1) {
                    let n = c.coverage(x, y);
                    if n == 0 {
                        continue;
                    }
                    let shape = S::from_u8(n).wide();
                    let a = S::div_max(sample::<S>(c.a_input).wide() * shape).wide();
                    let o = (usize::try_from(y).unwrap() * w + usize::try_from(x).unwrap()) * 3;
                    for (j, &s) in c.rgb.iter().enumerate() {
                        let d = out[o + j].wide();
                        out[o + j] = S::div_max((max - a) * d + a * sample::<S>(s).wide());
                    }
                }
            }
        }
        out
    }

    /// A random table over rows near the page: disjoint sorted spans, full
    /// or with random coverage bytes, some reaching off the page; its
    /// colour and alpha samples are at most `max`.
    fn random_cmd(rng: &mut Rng, width: u32, height: u32, max: u16) -> HostCmd {
        let y_min = rng.below(height + 8) - 4;
        let n_rows = rng.below(40);
        let (mut spans, mut bytes, mut row_starts) = (Vec::new(), Vec::new(), vec![0u32]);
        for _ in 0..n_rows {
            let mut x = rng.below(width + 10) - 10;
            for _ in 0..rng.below(6) {
                let x0 = x + rng.below(6);
                let x1 = x0 + rng.below(12);
                let offset = if rng.below(3) == 0 {
                    -1
                } else {
                    let at = i32::try_from(bytes.len()).unwrap();
                    bytes.extend((x0..=x1).map(|_| rng.byte()));
                    at
                };
                spans.push([x0, x1, offset]);
                x = x1 + 1;
            }
            row_starts.push(u32::try_from(spans.len()).unwrap());
        }
        let x0 = rng.below(width + 20) - 10;
        let y0 = y_min + rng.below(4) - 2;
        HostCmd {
            y_min,
            row_starts,
            spans,
            bytes,
            bounds: PixelBounds {
                x0,
                y0,
                x1: x0 + rng.below(width),
                y1: y0 + rng.below(height),
            },
            rgb: [0; 3].map(|_| rng.sample(max)),
            a_input: [max, max, max / 2 + 1, 1, 0, rng.sample(max)]
                [usize::try_from(rng.below(6)).unwrap()],
        }
    }

    /// A table painting every pixel of a `width × height` page `rgb` at
    /// alpha `a_input`.
    fn whole_page_cmd(width: u32, height: u32, rgb: [u16; 3], a_input: u16) -> HostCmd {
        let last = i32::try_from(width).unwrap() - 1;
        HostCmd {
            y_min: 0,
            row_starts: (0..=height).collect(),
            spans: vec![[0, last, -1]; height as usize],
            bytes: Vec::new(),
            bounds: PixelBounds {
                x0: 0,
                y0: 0,
                x1: last,
                y1: i32::try_from(height).unwrap() - 1,
            },
            rgb,
            a_input,
        }
    }

    /// `out`, a page `width` pixels wide read back `stride` samples a row
    /// into a buffer of `sentinel`, holds the rows of `want`, and every
    /// sample after each row, the last included, is still `sentinel`.
    /// `want` holds some pixel other than `paper`.
    fn assert_rows_at_stride<S: Sample>(
        out: &[S],
        want: &[S],
        paper: [S; 3],
        width: u32,
        stride: usize,
        sentinel: S,
        what: &str,
    ) {
        let row = width as usize * 3;
        assert!(
            want.as_chunks::<3>().0.iter().any(|p| *p != paper),
            "{what}: the commands paint no pixel"
        );
        assert_eq!(
            out.len() / stride,
            want.len() / row,
            "{what}: rows read back at stride {stride} ({} samples) against {} samples \
             of {row}-sample rows",
            out.len(),
            want.len()
        );
        for (y, (got, want)) in out
            .chunks_exact(stride)
            .zip(want.chunks_exact(row))
            .enumerate()
        {
            assert_eq!(&got[..row], want, "{what}: row {y}");
            assert!(
                got[row..].iter().all(|&s| s == sentinel),
                "{what}: samples after row {y} written"
            );
        }
    }

    /// Random pages rendered at the depth of `S` match the host model.
    fn page_raster_matches_the_host_composite_at<S: PageSample>(backend: &super::VulkanBackend) {
        let max = u16::try_from(S::MAX.signed()).unwrap();
        let pad = sample::<S>(max / 3);
        let mut rng = Rng(7);
        for (width, height, n) in [(1, 1, 3), (37, 21, 40), (130, 70, 300), (300, 17, 900)] {
            let paper = [max - 5, max - 15, max - 25];
            let cmds: Vec<HostCmd> = std::iter::once(whole_page_cmd(
                width,
                height,
                [max / 5, max / 7, max / 11],
                max / 2 + 1,
            ))
            .chain((0..n).map(|_| random_cmd(&mut rng, width, height, max)))
            .collect();
            let mut page = PageCommands::new(width, height, S::DEPTH, paper);
            for c in &cmds {
                page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
                    .expect("push");
            }
            let stride = width as usize * 3 + 5;
            let mut out = vec![pad; stride * height as usize];
            backend
                .render_page(&page, &mut out, stride)
                .expect("render_page");
            let want = host_page::<S>(width, height, paper, &cmds);
            assert_rows_at_stride(
                &out,
                &want,
                paper.map(sample::<S>),
                width,
                stride,
                pad,
                &format!("{} bits, {width}×{height}, {n} commands", S::DEPTH.bits()),
            );
        }
    }

    #[test]
    fn page_raster_matches_the_host_composite() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        page_raster_matches_the_host_composite_at::<u8>(&backend);
        page_raster_matches_the_host_composite_at::<u16>(&backend);
    }

    /// Pages at the depth of `S` whose widths make device rows of one
    /// pixel, of an odd pixel count and of more than two tiles read back
    /// as the reference raster renders them.
    fn word_readback_keeps_stride_gaps_at<S: PageSample>(backend: &super::VulkanBackend) {
        let max = u16::try_from(S::MAX.signed()).unwrap();
        let sentinel = sample::<S>(max / 3 + 1);
        let mut rng = Rng(41);
        for (width, height) in [(1, 1), (17, 3), (33, 17)] {
            let mut page = PageCommands::new(width, height, S::DEPTH, [max, max - 1, max - 2]);
            let cover = whole_page_cmd(width, height, [max / 5, max / 7, max / 11], max / 2 + 1);
            for c in std::iter::once(cover)
                .chain((0..30).map(|_| random_cmd(&mut rng, width, height, max)))
            {
                page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
                    .expect("push");
            }
            let want: Vec<S> = crate::page::reference::render_page(&page)
                .expect("reference")
                .into_iter()
                .map(sample::<S>)
                .collect();
            let stride = width as usize * 3 + 5;
            let mut out = vec![sentinel; stride * height as usize];
            backend
                .render_page(&page, &mut out, stride)
                .expect("render_page");
            assert_rows_at_stride(
                &out,
                &want,
                page.paper().map(sample::<S>),
                width,
                stride,
                sentinel,
                &format!("{} bits, {width}×{height}", S::DEPTH.bits()),
            );
        }
    }

    #[test]
    fn word_readback_keeps_stride_gaps_at_both_depths() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        word_readback_keeps_stride_gaps_at::<u8>(&backend);
        word_readback_keeps_stride_gaps_at::<u16>(&backend);
    }

    /// Pages of different sizes rendered at once from more threads than
    /// the backend has buffer sets each get their own pixels.
    #[test]
    fn concurrent_pages_each_get_their_own_pixels() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let pages: Vec<(u32, u32, Vec<HostCmd>)> = (0..12u32)
            .map(|i| {
                let mut rng = Rng(100 + u64::from(i));
                let (width, height) = (40 + 17 * i, 30 + 11 * (i % 5));
                let cmds = (0..60)
                    .map(|_| random_cmd(&mut rng, width, height, 255))
                    .collect();
                (width, height, cmds)
            })
            .collect();
        std::thread::scope(|scope| {
            for (width, height, cmds) in &pages {
                let backend = &backend;
                let _ = scope.spawn(move || {
                    let paper = [250, 240, 230];
                    let mut page = PageCommands::new(*width, *height, SampleDepth::Eight, paper);
                    for c in cmds {
                        page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
                            .expect("push");
                    }
                    let want = host_page::<u8>(*width, *height, paper, cmds);
                    for _ in 0..3 {
                        let mut out = vec![0u8; want.len()];
                        backend
                            .render_page(&page, &mut out, *width as usize * 3)
                            .expect("render_page");
                        assert_eq!(out, want, "{width}×{height}");
                    }
                });
            }
        });
        assert!(
            backend.page_scratch.idle_len() <= super::PAGE_SETS,
            "more buffer sets than the bound"
        );
    }

    #[test]
    fn a_timed_page_is_the_untimed_page() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let mut rng = Rng(11);
        let (width, height) = (130, 70);
        let mut page = PageCommands::new(width, height, SampleDepth::Eight, [250, 240, 230]);
        for _ in 0..200 {
            let c = random_cmd(&mut rng, width, height, 255);
            page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
                .expect("push");
        }
        let len = width as usize * height as usize * 3;
        let (mut plain, mut stamped) = (vec![0u8; len], vec![1u8; len]);
        let row = width as usize * 3;
        backend
            .render_page(&page, &mut plain, row)
            .expect("render_page");
        for _ in 0..2 {
            let times = backend
                .render_page_timed(&page, &mut stamped, row)
                .expect("render_page_timed");
            assert_eq!(plain, stamped);
            assert!(times.device_raster > std::time::Duration::ZERO, "{times:?}");
            assert!(
                times.device_raster < times.submit_wait,
                "the raster lies within the wait: {times:?}"
            );
        }
        backend
            .render_page(&page, &mut plain, row)
            .expect("render_page after a timed page");
        assert_eq!(plain, stamped);
    }

    #[test]
    fn a_page_renders_into_samples_of_its_depth_only() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let page = PageCommands::new(2, 2, SampleDepth::Sixteen, [9; 3]);
        let mut narrow = vec![0u8; 12];
        let err = backend
            .render_page(&page, &mut narrow, 6)
            .expect_err("a 16-bit page into bytes");
        assert!(err.to_string().contains("16-bit page"), "{err}");
        assert!(narrow.iter().all(|&b| b == 0));
    }

    #[test]
    fn opaque_runs_paint_whole_pixels() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let mut page = PageCommands::new(8, 4, SampleDepth::Sixteen, [65_535; 3]);
        page.push_opaque_runs(
            &mut [(1, 2, 4), (1, 3, 6), (3, 0, 0), (9, 0, 7)],
            [10, 2_000, 65_534],
        )
        .expect("push");
        let mut out = vec![0u16; 8 * 4 * 3];
        backend
            .render_page(&page, &mut out, 24)
            .expect("render_page");
        for y in 0..4 {
            for x in 0..8 {
                let painted = (y == 1 && (2..=6).contains(&x)) || (y == 3 && x == 0);
                let want = if painted {
                    [10, 2_000, 65_534]
                } else {
                    [65_535; 3]
                };
                assert_eq!(out[(y * 8 + x) * 3..][..3], want, "pixel ({x}, {y})");
            }
        }
    }

    #[test]
    fn a_page_without_commands_is_paper() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let page = PageCommands::new(19, 3, SampleDepth::Eight, [1, 2, 3]);
        let mut out = vec![0u8; 19 * 3 * 3];
        backend
            .render_page(&page, &mut out, 19 * 3)
            .expect("render_page");
        assert!(out.as_chunks::<3>().0.iter().all(|&p| p == [1, 2, 3]));
    }

    /// A stride whose span past the last row overflows `usize` is an
    /// error naming the page, not a wrapped bound and a panicking slice.
    #[test]
    fn a_stride_past_the_address_space_is_refused() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let page = PageCommands::new(4, 3, SampleDepth::Eight, [1, 2, 3]);
        let mut out = vec![0u8; 4 * 3 * 3];
        let err = backend
            .render_page(&page, &mut out, usize::MAX / 2 + 1)
            .expect_err("an overflowing stride");
        assert!(
            err.to_string().contains("exceeds the address space"),
            "{err}"
        );
        assert!(out.iter().all(|&s| s == 0), "out is unchanged");
    }

    #[test]
    fn bindings_past_the_storage_range_are_refused() {
        let Some(backend) = super::super::test_backend() else {
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
        let (_, fault) = b.floor_div_probe(&[pair], FLOOR_DIV_STEPS).unwrap();
        assert_eq!(fault, Some(pair));
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
        // The poisoned first estimate makes q * b wrap; the remainder
        // correction then leaves r in [0, b), and only the wrap guard
        // sees the wrong quotient.
        for pair in [
            (0, 2),
            (-5, 3),
            (12_345 * (1 << 31) + 7, 1 << 31),
            (1 << 62, (1 << 32) - 1),
        ] {
            let (q, fault) = b
                .floor_div_probe_poisoned(&[pair], FLOOR_DIV_STEPS)
                .unwrap();
            assert_ne!(
                q[0],
                crate::page::path::floor_div(pair.0, pair.1),
                "{pair:?}"
            );
            assert_eq!(fault, Some(pair));
        }
        let cases: Vec<(i64, i64)> = floor_div_cases(&mut Rng(0x0bad))
            .into_iter()
            .filter(|&(_, d)| d >= 2)
            .collect();
        let (_, fault) = b.floor_div_probe_poisoned(&cases, FLOOR_DIV_STEPS).unwrap();
        assert!(cases.contains(&fault.expect("a fault")));
    }

    #[test]
    fn a_set_status_word_fails_the_page_with_a_kernel_fault() {
        let a: i64 = -(1 << 40) + 7;
        let b: i64 = (1 << 31) + 5;
        let (a_bits, b_bits) = (a.cast_unsigned(), b.cast_unsigned());
        let words = [
            1,
            u32::try_from(a_bits & 0xFFFF_FFFF).unwrap(),
            u32::try_from(a_bits >> 32).unwrap(),
            u32::try_from(b_bits & 0xFFFF_FFFF).unwrap(),
            u32::try_from(b_bits >> 32).unwrap(),
        ];
        let err = super::check_paths_status(super::KernelId::PageRasterPaths, &words)
            .expect_err("a set fault code");
        assert!(
            matches!(
                err,
                BackendError::KernelFault {
                    kernel: "page_raster_paths",
                    ..
                }
            ),
            "{err:?}"
        );
        let message = err.to_string();
        assert!(message.contains(&format!("a = {a}")), "{message}");
        assert!(message.contains(&format!("b = {b}")), "{message}");
        assert!(super::check_paths_status(super::KernelId::PageRasterPaths, &[0; 5]).is_ok());
    }
}
