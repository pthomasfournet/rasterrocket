//! The page raster of any [`PageDevice`] against its host model and the
//! reference raster, and the contract of its read-back.

use color::{Sample, SampleDepth};

use super::PageDevice;
use crate::page::test_support::Rng;
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
        let spans = &self.spans[self.row_starts[row] as usize..self.row_starts[row + 1] as usize];
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
fn page_raster_matches_the_host_composite_at<S: PageSample, B: PageDevice>(backend: &B) {
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

fn page_raster_matches_the_host_composite_on<B: PageDevice>(backend: &B) {
    page_raster_matches_the_host_composite_at::<u8, B>(backend);
    page_raster_matches_the_host_composite_at::<u16, B>(backend);
}

/// Pages at the depth of `S` whose widths make device rows of one
/// pixel, of an odd pixel count and of more than two tiles read back
/// as the reference raster renders them.
fn word_readback_keeps_stride_gaps_at<S: PageSample, B: PageDevice>(backend: &B) {
    let max = u16::try_from(S::MAX.signed()).unwrap();
    let sentinel = sample::<S>(max / 3 + 1);
    let mut rng = Rng(41);
    for (width, height) in [(1, 1), (17, 3), (33, 17)] {
        let mut page = PageCommands::new(width, height, S::DEPTH, [max, max - 1, max - 2]);
        let cover = whole_page_cmd(width, height, [max / 5, max / 7, max / 11], max / 2 + 1);
        for c in
            std::iter::once(cover).chain((0..30).map(|_| random_cmd(&mut rng, width, height, max)))
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

fn word_readback_keeps_stride_gaps_at_both_depths_on<B: PageDevice>(backend: &B) {
    word_readback_keeps_stride_gaps_at::<u8, B>(backend);
    word_readback_keeps_stride_gaps_at::<u16, B>(backend);
}

/// Pages of different sizes rendered at once from more threads than
/// the backend has buffer sets each get their own pixels.
fn concurrent_pages_each_get_their_own_pixels_on<B: PageDevice>(backend: &B) {
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
}

fn a_timed_page_is_the_untimed_page_on<B: PageDevice>(backend: &B) {
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

fn a_page_renders_into_samples_of_its_depth_only_on<B: PageDevice>(backend: &B) {
    let page = PageCommands::new(2, 2, SampleDepth::Sixteen, [9; 3]);
    let mut narrow = vec![0u8; 12];
    let err = backend
        .render_page(&page, &mut narrow, 6)
        .expect_err("a 16-bit page into bytes");
    assert!(err.to_string().contains("16-bit page"), "{err}");
    assert!(narrow.iter().all(|&b| b == 0));
}

fn opaque_runs_paint_whole_pixels_on<B: PageDevice>(backend: &B) {
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

fn a_page_without_commands_is_paper_on<B: PageDevice>(backend: &B) {
    let page = PageCommands::new(19, 3, SampleDepth::Eight, [1, 2, 3]);
    let mut out = vec![0u8; 19 * 3 * 3];
    backend
        .render_page(&page, &mut out, 19 * 3)
        .expect("render_page");
    assert!(out.as_chunks::<3>().0.iter().all(|&p| p == [1, 2, 3]));
}

/// A stride whose span past the last row overflows `usize` is an
/// error naming the page, not a wrapped bound and a panicking slice.
fn a_stride_past_the_address_space_is_refused_on<B: PageDevice>(backend: &B) {
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

/// A page past a device limit or the memory left keeps the device in
/// service; any other error retires it.
fn retires_device_on_every_error_but_limits_and_memory_of<B: PageDevice>() {
    use crate::backend::BackendError;
    assert!(!B::retires_device(&BackendError::DeviceLimit {
        what: "render_page pixels",
        need: 2,
        limit: "a buffer limit",
        max: 1,
    }));
    assert!(!B::retires_device(&BackendError::OutOfMemory {
        what: "render_page pixels",
        bytes: 1,
    }));
    assert!(B::retires_device(&BackendError::KernelFault {
        kernel: "page_raster_paths",
        detail: String::new(),
    }));
    assert!(B::retires_device(&BackendError::msg("device lost")));
}

on_each_backend! {
    page_raster_matches_the_host_composite => page_raster_matches_the_host_composite_on;
    word_readback_keeps_stride_gaps_at_both_depths => word_readback_keeps_stride_gaps_at_both_depths_on;
    a_timed_page_is_the_untimed_page => a_timed_page_is_the_untimed_page_on;
    a_page_renders_into_samples_of_its_depth_only => a_page_renders_into_samples_of_its_depth_only_on;
    opaque_runs_paint_whole_pixels => opaque_runs_paint_whole_pixels_on;
    a_page_without_commands_is_paper => a_page_without_commands_is_paper_on;
    a_stride_past_the_address_space_is_refused => a_stride_past_the_address_space_is_refused_on;
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_concurrent_pages_each_get_their_own_pixels() {
    if let Some(backend) = crate::backend::vulkan::test_backend() {
        concurrent_pages_each_get_their_own_pixels_on(&backend);
        assert!(
            backend.page_sets_within_bound(),
            "more buffer sets than the bound"
        );
    }
}

#[cfg(all(windows, feature = "d3d12"))]
#[test]
fn d3d12_concurrent_pages_each_get_their_own_pixels() {
    if let Some(backend) = crate::backend::d3d12::test_backend() {
        concurrent_pages_each_get_their_own_pixels_on(&backend);
        assert!(
            backend.page_sets_within_bound(),
            "more buffer sets than the bound"
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_retires_device_on_every_error_but_limits_and_memory() {
    retires_device_on_every_error_but_limits_and_memory_of::<crate::backend::vulkan::VulkanBackend>(
    );
}

#[cfg(all(windows, feature = "d3d12"))]
#[test]
fn d3d12_retires_device_on_every_error_but_limits_and_memory() {
    retires_device_on_every_error_but_limits_and_memory_of::<crate::backend::d3d12::D3d12Backend>();
}
