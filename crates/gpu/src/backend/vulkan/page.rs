//! Whole-page raster from a [`PageCommands`] list: one submission and one
//! wait per page.
//!
//! The host bins the commands into tiles ([`PageCommands::bin`]) and writes
//! the five input arrays into device buffers: in place on a device whose
//! buffers are host-mapped, otherwise into host staging buffers copied by
//! the page's own command buffer.  The page then records the raster
//! dispatch, one workgroup per tile, and on a device whose page buffer the
//! host cannot read through its cache, a copy into a host read-back
//! buffer.  Every buffer is kept between pages and grown on demand.

use crate::backend::{BackendError, GpuBackend, Result};
use crate::page::{PageCommands, TileBins, pack_rgb};

use super::VulkanBackend;
use super::memory::{DeviceBuffer, HostAccess, HostBuffer, SlabAllocator};
use super::pipeline::KernelId;
use super::recorder::PageRecorder;

/// Smallest buffer the page raster allocates, in bytes; a buffer grows to
/// the next power of two of what a page needs.
const MIN_BUFFER: usize = 64 * 1024;

/// One kernel input: its device buffer and, on a device whose buffers the
/// host does not map, the host buffer it is copied from.
struct Input {
    device: DeviceBuffer,
    staging: Option<HostBuffer>,
}

/// The page buffer the kernel writes and, when the host cannot read it in
/// place through its cache, the host buffer it is copied into.
struct Target {
    device: DeviceBuffer,
    readback: Option<HostBuffer>,
}

/// The buffers of the page raster, kept between pages.
#[derive(Default)]
pub(super) struct PageScratch {
    /// Commands, intersections, row offsets, tile offsets, tile commands.
    inputs: [Option<Input>; 5],
    /// The page.
    target: Option<Target>,
}

/// The capacity a buffer holding `len` bytes is allocated with.
fn capacity(len: usize) -> usize {
    len.max(MIN_BUFFER).next_power_of_two()
}

/// Whether a buffer of `size` bytes holds `len` bytes.
fn fits(size: u64, len: usize) -> bool {
    usize::try_from(size).is_ok_and(|size| size >= len)
}

/// `slot`'s value when `keep` accepts it, otherwise a new one from `make`,
/// left in `slot`.
fn reuse_or<T>(
    slot: &mut Option<T>,
    keep: impl FnOnce(&T) -> bool,
    make: impl FnOnce() -> Result<T>,
) -> Result<&mut T> {
    let value = match slot.take() {
        Some(value) if keep(&value) => value,
        old => {
            drop(old);
            make()?
        }
    };
    Ok(slot.insert(value))
}

impl Input {
    /// An input of at least `len` bytes.
    fn new(memory: &SlabAllocator, len: usize) -> Result<Self> {
        let device = memory.alloc_device(capacity(len), HostAccess::Write)?;
        let staging = if device.is_mapped() {
            None
        } else {
            Some(memory.alloc_host(capacity(len))?)
        };
        Ok(Self { device, staging })
    }

    /// Write `bytes` at the start of the device buffer: in place when it is
    /// mapped, otherwise into the staging buffer and by a copy recorded on
    /// the page.  Called while the page is recording, after every earlier
    /// page that read the buffer has completed.
    fn write(&mut self, recorder: &PageRecorder, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if let Some(mapped) = self.device.mapped_mut() {
            mapped[..bytes.len()].copy_from_slice(bytes);
            return Ok(());
        }
        let staging = self
            .staging
            .as_mut()
            .ok_or_else(|| BackendError::msg("page input has neither mapping nor staging"))?;
        staging.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        recorder.record_copy(staging.handle(), self.device.handle(), bytes.len() as u64)
    }
}

impl Target {
    /// A page buffer of at least `len` bytes.
    fn new(memory: &SlabAllocator, len: usize) -> Result<Self> {
        let device = memory.alloc_device(capacity(len), HostAccess::ReadBack)?;
        let readback = if device.host_reads_cached() {
            None
        } else {
            Some(memory.alloc_host_readback(capacity(len))?)
        };
        Ok(Self { device, readback })
    }

    /// Record the copy of the page's first `len` bytes into the read-back
    /// buffer, if the page has one.
    fn record_readback(&self, recorder: &PageRecorder, len: usize) -> Result<()> {
        self.readback.as_ref().map_or(Ok(()), |r| {
            recorder.record_copy(self.device.handle(), r.handle(), len as u64)
        })
    }

    /// The page's bytes as the host reads them, once its page has completed.
    fn pixels(&self) -> Result<&[u8]> {
        self.readback.as_ref().map_or_else(
            || {
                self.device
                    .mapped()
                    .ok_or_else(|| BackendError::msg("render_page: page buffer lost its mapping"))
            },
            |r| Ok(r.as_slice()),
        )
    }
}

/// The push constants of the raster of `page` binned as `bins`, in the
/// order of `Params` in `kernels/page_raster.slang`.
fn push_constants(page: &PageCommands, bins: &TileBins) -> Result<[u8; 28]> {
    let aa_width = page.width().checked_mul(4).ok_or_else(|| {
        BackendError::msg(format!(
            "render_page: a page {} pixels wide is too wide",
            page.width()
        ))
    })?;
    let n_cmds = u32::try_from(page.len())
        .map_err(|_| BackendError::msg("render_page: too many commands"))?;
    let values = [
        n_cmds,
        page.width(),
        page.height(),
        bins.tiles_x,
        bins.tiles_y,
        aa_width,
        pack_rgb(page.paper()),
    ];
    let mut push = [0u8; 28];
    for (dst, v) in push.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        dst.copy_from_slice(&v.to_ne_bytes());
    }
    Ok(push)
}

impl VulkanBackend {
    /// Render `page` on the device and write its pixels into `out`, RGB8:
    /// row `y` at `out[y × stride..][..width × 3]`.  Bytes between rows are
    /// left unchanged.
    ///
    /// Concurrent calls run one at a time; other kernels' pages may run
    /// between them.
    ///
    /// # Errors
    ///
    /// When the page has no pixel, `out` or `stride` is too small for it,
    /// an allocation fails, or the device reports an error.  `out` is
    /// unchanged then.
    pub fn render_page(&self, page: &PageCommands, out: &mut [u8], stride: usize) -> Result<()> {
        let (width, height) = (page.width() as usize, page.height() as usize);
        let row = width * 3;
        if width == 0 || height == 0 {
            return Err(BackendError::msg("render_page: the page has no pixel"));
        }
        if stride < row || out.len() < stride * (height - 1) + row {
            return Err(BackendError::msg(format!(
                "render_page: a {width}×{height} page does not fit {} bytes at stride {stride}",
                out.len()
            )));
        }
        let bins = page.bin().map_err(BackendError::new)?;
        let push = push_constants(page, &bins)?;
        let inputs: [&[u8]; 5] = [
            bytemuck::cast_slice(page.cmds()),
            bytemuck::cast_slice(page.intersects()),
            bytemuck::cast_slice(page.row_starts()),
            bytemuck::cast_slice(&bins.offsets),
            bytemuck::cast_slice(&bins.cmds),
        ];
        let page_len = row * height;

        let mut scratch = self
            .page_scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let PageScratch {
            inputs: slots,
            target,
        } = &mut *scratch;
        let mut inputs_in_place = Vec::with_capacity(slots.len());
        for (slot, bytes) in slots.iter_mut().zip(inputs) {
            let input = reuse_or(
                slot,
                |i| fits(i.device.size(), bytes.len()),
                || Input::new(&self.memory, bytes.len()),
            )?;
            inputs_in_place.push((input, bytes));
        }
        let target = reuse_or(
            target,
            |t| fits(t.device.size(), page_len),
            || Target::new(&self.memory, page_len),
        )?;

        self.run_page(|_| {
            let mut handles = Vec::with_capacity(6);
            let mut sizes = Vec::with_capacity(6);
            for (input, bytes) in &mut inputs_in_place {
                input.write(&self.recorder, bytes)?;
                handles.push(input.device.handle());
                sizes.push(input.device.size());
            }
            handles.push(target.device.handle());
            sizes.push(target.device.size());
            self.recorder.dispatch_kernel(
                KernelId::PageRaster,
                &handles,
                &sizes,
                &push,
                (bins.tiles_x, bins.tiles_y, 1),
            )?;
            target.record_readback(&self.recorder, page_len)
        })?;

        let pixels = &target.pixels()?[..page_len];
        for (dst, src) in out.chunks_mut(stride).zip(pixels.chunks_exact(row)) {
            dst[..row].copy_from_slice(src);
        }
        drop(scratch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::fill::{AA_SIZE, AaScanTable, ScanIntersect, pixel_coverage};
    use crate::page::{PageCommands, PixelBounds};

    /// The AA gamma map of a sub-sample count, as the CPU fill applies it.
    const AA_GAMMA: [u32; 17] = [
        0, 4, 11, 21, 32, 45, 59, 74, 90, 108, 126, 145, 166, 187, 209, 231, 255,
    ];

    fn div255(x: u32) -> u32 {
        let t = x + 0x80;
        ((t + (t >> 8)) >> 8).min(255)
    }

    /// One command as the host model paints it.
    struct HostCmd {
        intersects: Vec<ScanIntersect>,
        row_starts: Vec<u32>,
        y_min: i32,
        eo: bool,
        bounds: PixelBounds,
        rgb: [u8; 3],
        a_input: u8,
    }

    impl HostCmd {
        fn table(&self) -> AaScanTable<'_> {
            AaScanTable {
                intersects: &self.intersects,
                row_starts: &self.row_starts,
                y_min: self.y_min,
            }
        }
    }

    /// The page `cmds` paint, by the per-pixel walk of the host twin.
    #[expect(
        clippy::many_single_char_names,
        reason = "pixel coordinates, coverage and alpha of the kernel's formulas"
    )]
    fn host_page(width: u32, height: u32, paper: [u8; 3], cmds: &[HostCmd]) -> Vec<u8> {
        let (w, h) = (width as usize, i32::try_from(height).unwrap());
        let mut out: Vec<u8> = (0..width * height).flat_map(|_| paper).collect();
        for c in cmds {
            for y in c.bounds.y0.max(0)..=c.bounds.y1.min(h - 1) {
                for x in c.bounds.x0.max(0)..=c.bounds.x1.min(i32::try_from(w).unwrap() - 1) {
                    let n = pixel_coverage(&c.table(), width * 4, c.eo, x, y);
                    if n == 0 {
                        continue;
                    }
                    let a = div255(u32::from(c.a_input) * AA_GAMMA[usize::from(n)]);
                    let o = (usize::try_from(y).unwrap() * w + usize::try_from(x).unwrap()) * 3;
                    for (j, &s) in c.rgb.iter().enumerate() {
                        let d = u32::from(out[o + j]);
                        out[o + j] =
                            u8::try_from(div255((255 - a) * d + a * u32::from(s))).unwrap();
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
    }

    /// A random table over AA rows near the page: sorted intersections with
    /// winding counts of -1, 0 and 1, some rows ending inside.
    fn random_cmd(rng: &mut Rng, width: u32, height: u32) -> HostCmd {
        let y_min = rng.below(height * 4 + 8) - 8;
        let n_rows = rng.below(160);
        let (mut intersects, mut row_starts) = (Vec::new(), vec![0u32]);
        for _ in 0..n_rows {
            let mut row: Vec<ScanIntersect> = (0..rng.below(7))
                .map(|_| {
                    let x0 = rng.below(width * 4 + 40) - 20;
                    ScanIntersect {
                        x0,
                        x1: x0 + rng.below(12),
                        count: rng.below(3) - 1,
                    }
                })
                .collect();
            row.sort_by_key(|i| i.x0);
            intersects.extend(row);
            row_starts.push(u32::try_from(intersects.len()).unwrap());
        }
        let x0 = rng.below(width + 20) - 10;
        let y0 = (y_min / AA_SIZE) + rng.below(4) - 2;
        HostCmd {
            intersects,
            row_starts,
            y_min,
            eo: rng.below(2) == 1,
            bounds: PixelBounds {
                x0,
                y0,
                x1: x0 + rng.below(width),
                y1: y0 + rng.below(height),
            },
            rgb: [rng.byte(), rng.byte(), rng.byte()],
            a_input: [255, 255, 128, 1, 0, 77][usize::try_from(rng.below(6)).unwrap()],
        }
    }

    #[test]
    fn page_raster_matches_the_host_walk() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let mut rng = Rng(7);
        for (width, height, n) in [(1, 1, 3), (37, 21, 40), (130, 70, 300), (300, 17, 900)] {
            let paper = [250, 240, 230];
            let cmds: Vec<HostCmd> = (0..n)
                .map(|_| random_cmd(&mut rng, width, height))
                .collect();
            let mut page = PageCommands::new(width, height, paper);
            for c in &cmds {
                page.push_coverage(c.table(), c.eo, c.bounds, c.rgb, c.a_input)
                    .expect("push");
            }
            let row = width as usize * 3;
            let stride = row + 5;
            let mut out = vec![0xAB; stride * height as usize];
            backend
                .render_page(&page, &mut out, stride)
                .expect("render_page");
            let want = host_page(width, height, paper, &cmds);
            for y in 0..height as usize {
                assert_eq!(
                    out[y * stride..][..row],
                    want[y * row..][..row],
                    "{width}×{height}, {n} commands: row {y}"
                );
                if y + 1 < height as usize {
                    assert!(
                        out[y * stride + row..(y + 1) * stride]
                            .iter()
                            .all(|&b| b == 0xAB),
                        "row padding written"
                    );
                }
            }
        }
    }

    #[test]
    fn opaque_runs_paint_whole_pixels() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let mut page = PageCommands::new(8, 4, [255; 3]);
        page.push_opaque_runs(
            &mut [(1, 2, 4), (1, 3, 6), (3, 0, 0), (9, 0, 7)],
            [10, 20, 30],
        )
        .expect("push");
        let mut out = vec![0; 8 * 4 * 3];
        backend
            .render_page(&page, &mut out, 24)
            .expect("render_page");
        for y in 0..4 {
            for x in 0..8 {
                let painted = (y == 1 && (2..=6).contains(&x)) || (y == 3 && x == 0);
                let want = if painted { [10, 20, 30] } else { [255; 3] };
                assert_eq!(out[(y * 8 + x) * 3..][..3], want, "pixel ({x}, {y})");
            }
        }
    }
}
