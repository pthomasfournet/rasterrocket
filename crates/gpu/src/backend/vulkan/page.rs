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
//! A timed page ([`VulkanBackend::render_page_timed`]) also reads the host
//! clock between its host stages and writes device timestamps between its
//! device stages; an untimed page does neither.

use std::time::{Duration, Instant};

use crate::backend::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage,
};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::page::{PageCommands, PageSample, TileBins, pack_rgb};

use super::VulkanBackend;
use super::pipeline::KernelId;
use super::timestamps::StageClock;

/// The bytes staged for an empty input: a descriptor binds a non-empty
/// range, and the kernel reads no element of an empty array.
const EMPTY_INPUT: [u8; 4] = [0; 4];

/// The buffers of the page raster, kept between pages.
#[derive(Default)]
pub(super) struct PageScratch {
    /// Commands, coverage table words, row offsets, tile offsets, tile
    /// commands.
    inputs: [Option<Staged<VulkanBackend>>; 5],
    /// The page.
    target: Option<Staged<VulkanBackend>>,
    /// The timestamp queries of timed pages, created by the first.
    clock: Option<StageClock>,
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
    /// Device: copying the inputs from the staging buffers.
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

/// The push constants of the raster of `page` binned as `bins`, in the
/// order of `Params` in `kernels/page_raster.slang`.
fn push_constants(page: &PageCommands, bins: &TileBins, coverage_base: u32) -> [u8; 36] {
    let n_cmds = u32::try_from(page.len()).expect("the push limit bounds the commands");
    let [paper_rg, paper_b] = pack_rgb(page.paper());
    let values = [
        n_cmds,
        page.width(),
        page.height(),
        bins.tiles_x,
        bins.tiles_y,
        coverage_base,
        paper_rg,
        paper_b,
        u32::from(page.depth().bits()),
    ];
    let mut push = [0u8; 36];
    for (dst, v) in push.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        dst.copy_from_slice(&v.to_ne_bytes());
    }
    push
}

/// `len` bytes as a device size.
fn device_size(len: usize) -> u64 {
    u64::try_from(len).expect("usize fits u64 on supported targets")
}

impl VulkanBackend {
    /// Render `page` on the device and write its pixels into `out`, RGB
    /// samples of the page's depth: row `y` at
    /// `out[y × stride..][..width × 3]`, `stride` in samples.  Samples
    /// between rows are left unchanged.
    ///
    /// Concurrent calls run one at a time; other kernels' pages may run
    /// between them.
    ///
    /// # Errors
    ///
    /// [`BackendError::DeviceLimit`] when an input or the page exceeds the
    /// device's `maxStorageBufferRange`, before anything is allocated;
    /// otherwise an error when the page has no pixel, its depth is not
    /// that of `S`, `out` or `stride` is too small for it, an allocation
    /// fails, or the device reports an error.  `out` is unchanged then.
    pub fn render_page<S: PageSample>(
        &self,
        page: &PageCommands,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        self.render_page_with(page, out, stride, None)
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
        self.render_page_with(page, out, stride, Some(&mut times))?;
        Ok(times)
    }

    /// [`Self::render_page`], timed into `times` when given.
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
    ) -> Result<()> {
        let mut laps = Laps::new(times.is_some());
        let (width, height) = (page.width() as usize, page.height() as usize);
        let row = width * 3;
        let row_bytes = row * S::DEPTH.bytes() as usize;
        if width == 0 || height == 0 {
            return Err(BackendError::msg("render_page: the page has no pixel"));
        }
        if page.depth() != S::DEPTH {
            return Err(BackendError::msg(format!(
                "render_page: a {}-bit page does not render into {}-bit samples",
                page.depth().bits(),
                S::DEPTH.bits()
            )));
        }
        if stride < row || out.len() < stride * (height - 1) + row {
            return Err(BackendError::msg(format!(
                "render_page: a {width}×{height} page does not fit {} samples at stride {stride}",
                out.len()
            )));
        }
        let bins = page.bin().map_err(BackendError::new)?;
        let (table, coverage_base) = page.table_words();
        let push = push_constants(page, &bins, coverage_base);
        let page_len = row_bytes * height;
        let inputs: [(&'static str, &[u8]); 5] = [
            ("render_page commands", bytemuck::cast_slice(page.cmds())),
            ("render_page coverage table", bytemuck::cast_slice(&table)),
            (
                "render_page row offsets",
                bytemuck::cast_slice(page.row_starts()),
            ),
            (
                "render_page tile offsets",
                bytemuck::cast_slice(&bins.offsets),
            ),
            (
                "render_page tile commands",
                bytemuck::cast_slice(&bins.cmds),
            ),
        ];
        for (what, bytes) in inputs {
            self.device
                .check_storage_range(what, device_size(bytes.len()))?;
        }
        self.device
            .check_storage_range("render_page pixels", device_size(page_len))?;

        let mut scratch = self
            .page_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let PageScratch {
            inputs: slots,
            target,
            clock,
        } = &mut *scratch;
        let clock = match (&times, clock) {
            (None, _) => None,
            (Some(_), Some(clock)) => Some(&*clock),
            (Some(_), clock @ None) => Some(&*clock.insert(StageClock::new(self.device.clone())?)),
        };
        let prepare = laps.lap();
        let mut staged = Vec::with_capacity(slots.len());
        for (slot, (_, bytes)) in slots.iter_mut().zip(inputs) {
            let bytes = if bytes.is_empty() {
                &EMPTY_INPUT[..]
            } else {
                bytes
            };
            staged.push((stage(self, slot, bytes)?, bytes.len()));
        }
        let target = ensure(self, target, page_len, Direction::ToHost)?;
        let write = laps.lap();

        // Query `i` of a timed page is written once every command before it
        // has completed: 0 before the uploads, 1 after them, 2 after the
        // raster, 3 after the download.
        let stamp = |backend: &Self, query: u32| {
            clock.map_or(Ok(()), |c| {
                backend.recorder.record_timestamp(c.pool(), query)
            })
        };
        let mut record = Duration::ZERO;
        // SAFETY (the recorded copies and the read below): the scratch is
        // held under its lock for this whole call, and `run_page` returns
        // only once its page has completed or, on failure, been waited on;
        // nothing touches the buffers in between.
        self.run_page(|backend| {
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
            stamp(backend, 1)?;
            handles.push(target.device.handle());
            sizes.push(device_size(page_len));
            backend.recorder.dispatch_kernel(
                KernelId::PageRaster,
                &handles,
                &sizes,
                &push,
                (bins.tiles_x, bins.tiles_y, 1),
            )?;
            stamp(backend, 2)?;
            unsafe { record_download(backend, target, page_len) }?;
            stamp(backend, 3)?;
            record = laps.lap();
            Ok(())
        })?;
        let submit_wait = laps.lap();

        let pixels = &unsafe { host_view(self, target) }?[..page_len];
        for (dst, src) in out.chunks_mut(stride).zip(pixels.chunks_exact(row_bytes)) {
            S::copy_from_le(&mut dst[..row], src);
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

#[cfg(test)]
mod tests {
    use color::{Sample, SampleDepth};

    use crate::page::{CoverageRows, PageCommands, PageSample, PixelBounds};

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

    /// `SplitMix64`.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u32) -> i32 {
            i32::try_from(self.next() % u64::from(n)).unwrap()
        }

        fn byte(&mut self) -> u8 {
            u8::try_from(self.below(256)).unwrap()
        }

        /// A sample of at most `max`.
        fn sample(&mut self, max: u16) -> u16 {
            u16::try_from(self.below(u32::from(max) + 1)).unwrap()
        }
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

    /// Random pages rendered at the depth of `S` match the host model.
    fn page_raster_matches_the_host_composite_at<S: PageSample>(backend: &super::VulkanBackend) {
        let max = u16::try_from(S::MAX.signed()).unwrap();
        let pad = sample::<S>(max / 3);
        let mut rng = Rng(7);
        for (width, height, n) in [(1, 1, 3), (37, 21, 40), (130, 70, 300), (300, 17, 900)] {
            let paper = [max - 5, max - 15, max - 25];
            let cmds: Vec<HostCmd> = (0..n)
                .map(|_| random_cmd(&mut rng, width, height, max))
                .collect();
            let mut page = PageCommands::new(width, height, S::DEPTH, paper);
            for c in &cmds {
                page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
                    .expect("push");
            }
            let row = width as usize * 3;
            let stride = row + 5;
            let mut out = vec![pad; stride * height as usize];
            backend
                .render_page(&page, &mut out, stride)
                .expect("render_page");
            let want = host_page::<S>(width, height, paper, &cmds);
            let bits = S::DEPTH.bits();
            for y in 0..height as usize {
                assert_eq!(
                    out[y * stride..][..row],
                    want[y * row..][..row],
                    "{bits} bits, {width}×{height}, {n} commands: row {y}"
                );
                if y + 1 < height as usize {
                    assert!(
                        out[y * stride + row..(y + 1) * stride]
                            .iter()
                            .all(|&b| b == pad),
                        "row padding written"
                    );
                }
            }
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
}
