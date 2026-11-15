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

use crate::backend::staged::{
    Direction, Staged, ensure, host_view, record_download, record_upload, stage,
};
use crate::backend::{BackendError, GpuBackend, Result};
use crate::page::{PageCommands, TileBins, pack_rgb};

use super::VulkanBackend;
use super::pipeline::KernelId;

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
}

/// The push constants of the raster of `page` binned as `bins`, in the
/// order of `Params` in `kernels/page_raster.slang`.
fn push_constants(page: &PageCommands, bins: &TileBins, coverage_base: u32) -> [u8; 28] {
    let n_cmds = u32::try_from(page.len()).expect("the push limit bounds the commands");
    let values = [
        n_cmds,
        page.width(),
        page.height(),
        bins.tiles_x,
        bins.tiles_y,
        coverage_base,
        pack_rgb(page.paper()),
    ];
    let mut push = [0u8; 28];
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
    /// Render `page` on the device and write its pixels into `out`, RGB8:
    /// row `y` at `out[y × stride..][..width × 3]`.  Bytes between rows are
    /// left unchanged.
    ///
    /// Concurrent calls run one at a time; other kernels' pages may run
    /// between them.
    ///
    /// # Errors
    ///
    /// [`BackendError::DeviceLimit`] when an input or the page exceeds the
    /// device's `maxStorageBufferRange`, before anything is allocated;
    /// otherwise an error when the page has no pixel, `out` or `stride` is
    /// too small for it, an allocation fails, or the device reports an
    /// error.  `out` is unchanged then.
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
        let (table, coverage_base) = page.table_words();
        let push = push_constants(page, &bins, coverage_base);
        let page_len = row * height;
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
        } = &mut *scratch;
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

        // SAFETY (the recorded copies and the read below): the scratch is
        // held under its lock for this whole call, and `run_page` returns
        // only once its page has completed or, on failure, been waited on;
        // nothing touches the buffers in between.
        self.run_page(|backend| {
            let mut handles = Vec::with_capacity(staged.len() + 1);
            let mut sizes = Vec::with_capacity(staged.len() + 1);
            for (input, len) in &mut staged {
                let len = *len;
                unsafe { record_upload(backend, input, len) }?;
                handles.push(input.device.handle());
                sizes.push(device_size(len));
            }
            handles.push(target.device.handle());
            sizes.push(device_size(page_len));
            backend.recorder.dispatch_kernel(
                KernelId::PageRaster,
                &handles,
                &sizes,
                &push,
                (bins.tiles_x, bins.tiles_y, 1),
            )?;
            unsafe { record_download(backend, target, page_len) }
        })?;

        let pixels = &unsafe { host_view(self, target) }?[..page_len];
        for (dst, src) in out.chunks_mut(stride).zip(pixels.chunks_exact(row)) {
            dst[..row].copy_from_slice(src);
        }
        drop(scratch);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use color::convert::div255;

    use crate::page::{CoverageRows, PageCommands, PixelBounds};

    /// One command as the host model paints it.
    struct HostCmd {
        y_min: i32,
        row_starts: Vec<u32>,
        spans: Vec<[i32; 3]>,
        bytes: Vec<u8>,
        bounds: PixelBounds,
        rgb: [u8; 3],
        a_input: u8,
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

    /// The page `cmds` paint, composited on the host as the CPU pipe does.
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
                    let n = c.coverage(x, y);
                    if n == 0 {
                        continue;
                    }
                    let a = u32::from(div255(u32::from(c.a_input) * u32::from(n)));
                    let o = (usize::try_from(y).unwrap() * w + usize::try_from(x).unwrap()) * 3;
                    for (j, &s) in c.rgb.iter().enumerate() {
                        let d = u32::from(out[o + j]);
                        out[o + j] = div255((255 - a) * d + a * u32::from(s));
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

    /// A random table over rows near the page: disjoint sorted spans, full
    /// or with random coverage bytes, some reaching off the page.
    fn random_cmd(rng: &mut Rng, width: u32, height: u32) -> HostCmd {
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
            rgb: [rng.byte(), rng.byte(), rng.byte()],
            a_input: [255, 255, 128, 1, 0, 77][usize::try_from(rng.below(6)).unwrap()],
        }
    }

    #[test]
    fn page_raster_matches_the_host_composite() {
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
                page.push_coverage(c.table(), c.bounds, c.rgb, c.a_input)
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

    #[test]
    fn a_page_without_commands_is_paper() {
        let Some(backend) = super::super::test_backend() else {
            return;
        };
        let page = PageCommands::new(19, 3, [1, 2, 3]);
        let mut out = vec![0; 19 * 3 * 3];
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
