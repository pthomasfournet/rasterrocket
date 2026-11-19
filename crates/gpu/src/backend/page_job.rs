//! The device-independent stages of one page raster: the checks of the
//! page against the caller's buffer, the choice of kernel, the push
//! constants, the input arrays, and, once the device has run, the check
//! of its status words and the repacking of its pixels.
//!
//! The device page holds one device pixel per pixel, in whole 32-bit words
//! ([`crate::backend::layout`]); [`PageJob::finish`] drops each pixel's pad
//! bytes and writes each row's packed samples at its stride in the
//! caller's buffer, leaving the samples between rows unchanged.

use crate::backend::kernel_status::{self, STATUS_WORDS};
use crate::backend::{BackendError, Result};
use crate::page::{PageCommands, PageSample, TableBases, TileBins, pack_rgb};

/// The bytes of the page raster's push constants: the 14 words of
/// `Params` in `kernels/page_raster.slang`.
pub(crate) const PUSH_BYTES: usize = 56;

/// The kernel that rasterises a page with outlines, as named in the
/// fault it reports.
const PAGE_RASTER_PATHS: &str = "page_raster_paths";

/// The most byte slices one input of the page raster is made of: the
/// parts of the coverage table ([`PageCommands::table_parts`]).
const MAX_PARTS: usize = 8;

/// One input array of the page raster: consecutive byte slices, staged
/// end to end into one device buffer.
#[derive(Clone, Copy)]
pub(crate) struct InputParts<'b> {
    /// The input, for diagnostic context.
    pub(crate) what: &'static str,
    parts: [&'b [u8]; MAX_PARTS],
    n: usize,
}

impl<'b> InputParts<'b> {
    const fn one(what: &'static str, bytes: &'b [u8]) -> Self {
        let mut parts: [&[u8]; MAX_PARTS] = [&[]; MAX_PARTS];
        parts[0] = bytes;
        Self { what, parts, n: 1 }
    }

    /// The slices, in order.
    pub(crate) fn parts(&self) -> &[&'b [u8]] {
        &self.parts[..self.n]
    }

    /// The bytes of the input.
    pub(crate) fn len(&self) -> usize {
        self.parts().iter().map(|p| p.len()).sum()
    }
}

/// One page raster of a [`PageCommands`] list into a caller's buffer of
/// samples, between the checks and binning on the host and the read-back.
pub(crate) struct PageJob<'a> {
    page: &'a PageCommands,
    table: [&'a [u8]; MAX_PARTS],
    /// The page's tiles and the commands binned to each.
    pub(crate) bins: TileBins,
    /// Whether the page has outlines: it runs the raster that rasterises
    /// them and binds the status words.
    pub(crate) kernel_paths: bool,
    /// The kernel a fault in the status words is reported for.
    pub(crate) fault_kernel: &'static str,
    /// The push constants.
    pub(crate) push: [u8; PUSH_BYTES],
    /// The bytes of the device page.
    pub(crate) device_page_len: usize,
    /// The bytes of a device row.
    device_row: usize,
    /// The samples of a packed row.
    row: usize,
    /// The caller's stride, in samples.
    stride: usize,
    /// The samples of the caller's buffer the page spans: the last row
    /// ends `row` samples after the start of row `height - 1`.
    pub(crate) span: usize,
}

impl<'a> PageJob<'a> {
    /// Check `page` against a buffer of `out_len` samples of `S` at
    /// `stride`, on a device that rasterises outlines when `paths`, and
    /// bin it.
    ///
    /// # Errors
    ///
    /// When the page has no pixel, its bytes or span exceed the address
    /// space, its depth is not that of `S`, `out_len` or `stride` is too
    /// small for it, it has outlines and `paths` is false, or its binning
    /// fails.
    pub(crate) fn prepare<S: PageSample>(
        page: &'a PageCommands,
        out_len: usize,
        stride: usize,
        paths: bool,
    ) -> Result<Self> {
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
        let row = width.checked_mul(3).ok_or_else(overflow)?;
        let (device_row, device_page_len) =
            device_page_bytes::<S>(width, height).ok_or_else(overflow)?;
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
        if stride < row || out_len < span {
            return Err(BackendError::msg(format!(
                "render_page: a {width}×{height} page does not fit {out_len} samples at stride \
                 {stride}"
            )));
        }
        let kernel_paths = page.has_paths();
        if kernel_paths && !paths {
            return Err(BackendError::msg(
                "render_page: the page has outlines, which this device does not rasterise",
            ));
        }
        let bins = page.bin().map_err(BackendError::new)?;
        let (table, bases) = page.table_parts();
        let push = push_constants(page, &bins, bases);
        Ok(Self {
            page,
            table,
            bins,
            kernel_paths,
            fault_kernel: PAGE_RASTER_PATHS,
            push,
            device_page_len,
            device_row,
            row,
            stride,
            span,
        })
    }

    /// The five input arrays, in binding order: the commands, the
    /// coverage table words, the row offsets, the tile offsets and the
    /// tile items.
    pub(crate) fn inputs(&self) -> [InputParts<'_>; 5] {
        [
            InputParts::one(
                "render_page commands",
                bytemuck::cast_slice(self.page.cmds()),
            ),
            InputParts {
                what: "render_page coverage table",
                parts: self.table,
                n: MAX_PARTS,
            },
            InputParts::one(
                "render_page row offsets",
                bytemuck::cast_slice(self.page.row_starts()),
            ),
            InputParts::one(
                "render_page tile offsets",
                bytemuck::cast_slice(&self.bins.offsets),
            ),
            InputParts::one(
                "render_page tile items",
                bytemuck::cast_slice(&self.bins.items),
            ),
        ]
    }

    /// Write the device page `pixels` into `out` at `stride`, once
    /// `status`, the status words of a page with outlines, records no
    /// fault: row `y` at `out[y × stride..][..width × 3]`, the samples
    /// between rows unchanged.  `out` is unchanged on an error.
    ///
    /// # Errors
    ///
    /// [`BackendError::KernelFault`] of [`Self::fault_kernel`] when
    /// `status` records a fault; an error when the page has outlines and
    /// `status` is `None`, `pixels` is shorter than the device page, or
    /// `out` and `stride` are not those the job was prepared for.
    pub(crate) fn finish<S: PageSample>(
        &self,
        pixels: &[u8],
        status: Option<&[u32; STATUS_WORDS]>,
        out: &mut [S],
        stride: usize,
    ) -> Result<()> {
        match status {
            Some(words) => kernel_status::check(self.fault_kernel, words)?,
            None if self.kernel_paths => {
                return Err(BackendError::msg(
                    "render_page: the status words of a page with outlines are missing",
                ));
            }
            None => {}
        }
        let pixels = pixels
            .get(..self.device_page_len)
            .ok_or_else(|| BackendError::msg("render_page: the device page is short"))?;
        if stride != self.stride || out.len() < self.span {
            return Err(BackendError::msg(format!(
                "render_page: {} samples at stride {stride} are not the buffer the page was \
                 prepared for",
                out.len()
            )));
        }
        for (dst, src) in out
            .chunks_mut(stride)
            .zip(pixels.chunks_exact(self.device_row))
        {
            S::copy_from_words(&mut dst[..self.row], src);
        }
        Ok(())
    }
}

/// The push constants of the raster of `page` binned as `bins`, its table
/// words laid out as `bases`, in the order of `Params` in
/// `kernels/page_raster.slang`.
fn push_constants(page: &PageCommands, bins: &TileBins, bases: TableBases) -> [u8; PUSH_BYTES] {
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
    let mut push = [0u8; PUSH_BYTES];
    for (dst, v) in push.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        dst.copy_from_slice(&v.to_ne_bytes());
    }
    push
}

/// The bytes of a device row and of the device page of a `width` × `height`
/// page of `S` samples, [`PageSample::DEVICE_PIXEL_BYTES`] a pixel; `None`
/// when either overflows `usize`.
pub(crate) fn device_page_bytes<S: PageSample>(
    width: usize,
    height: usize,
) -> Option<(usize, usize)> {
    let row = width.checked_mul(S::DEVICE_PIXEL_BYTES)?;
    Some((row, row.checked_mul(height)?))
}

#[cfg(test)]
mod tests {
    use color::SampleDepth;

    use super::{PAGE_RASTER_PATHS, PageJob};
    use crate::backend::BackendError;
    use crate::page::PageCommands;

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
        let page = PageCommands::new(2, 1, SampleDepth::Eight, [7; 3]);
        let job = PageJob::prepare::<u8>(&page, 6, 6, true).expect("prepare");
        assert_eq!(job.fault_kernel, PAGE_RASTER_PATHS);
        let pixels = [0u8; 8];
        let mut out = [0u8; 6];
        let err = job
            .finish(&pixels, Some(&words), &mut out, 6)
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
        assert_eq!(out, [0; 6], "a faulting page wrote a sample");
        job.finish(&pixels, Some(&[0; 5]), &mut out, 6)
            .expect("no fault");
    }
}
