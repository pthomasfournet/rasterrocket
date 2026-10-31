//! Glyph bitmap blitting.
//!
//! A glyph bitmap is either:
//! - **AA** (`aa = true`): one byte per pixel, containing a coverage value 0–255.
//! - **Mono** (`aa = false`): 1-bit-per-pixel, MSB-first packed, `ceil(w/8)` bytes per row.
//!
//! The blit clips to the destination bitmap bounds and optionally to the `Clip` region.

use crate::bitmap::Bitmap;
use crate::clip::{Clip, ClipResult, ClipRows, RowClip};
use crate::pipe::{self, PipeSrc, PipeState};
use crate::simd;
use color::Pixel;
use color::convert::div255;

/// A rasterized glyph bitmap as produced by a font renderer.
///
/// The origin `(x, y)` is the pen-position offset in device pixels:
/// the top-left of the bitmap maps to `(pen_x - x, pen_y - y)`.
pub struct GlyphBitmap<'a> {
    /// Raw bitmap data. For AA glyphs: `w` bytes per row. For mono: `ceil(w/8)` bytes per row.
    pub data: &'a [u8],
    /// Horizontal offset from the pen position to the left edge of the bitmap.
    pub x: i32,
    /// Vertical offset from the pen position to the top edge of the bitmap.
    pub y: i32,
    /// Width in pixels.
    pub w: i32,
    /// Height in pixels.
    pub h: i32,
    /// `true` for anti-aliased (one u8 coverage byte per pixel);
    /// `false` for 1-bit MSB-first packed mono.
    pub aa: bool,
}

impl GlyphBitmap<'_> {
    /// Number of bytes per row in `data`.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        let w = self.w.max(0);
        if self.aa {
            #[expect(clippy::cast_sign_loss, reason = "w = self.w.max(0) is non-negative")]
            {
                w as usize
            }
        } else {
            // Use saturating_add so a pathological w near i32::MAX doesn't wrap.
            // Glyph bitmaps are bounded far below 2^16 px wide by the font
            // crate's rasteriser budget.
            #[expect(clippy::cast_sign_loss, reason = "w = self.w.max(0) is non-negative")]
            {
                (w.saturating_add(7) / 8) as usize
            }
        }
    }
}

/// Blit a glyph at pen position `(pen_x, pen_y)` (device pixel coordinates).
///
/// `clip_all_inside` — if `true`, skip per-pixel clip tests (the caller has
/// already determined the entire glyph bbox is inside the clip region).
/// When `false`, each glyph pixel is scaled by the clip's coverage of that
/// pixel ([`ClipRows`]), so an antialiased clip edge crossing the glyph is
/// antialiased as it is for a path fill.
///
/// The `pipe` and `src` describe the fill colour and compositing parameters.
#[expect(
    clippy::too_many_arguments,
    reason = "the destination, clip, compositing state, pen position and glyph are independent inputs"
)]
pub fn blit_glyph<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    clip_all_inside: bool,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    pen_x: i32,
    pen_y: i32,
    glyph: &GlyphBitmap<'_>,
) {
    // Compute the top-left destination pixel.
    // Saturating: a far-off pen position clamps to the i32 range and then
    // misses the bitmap instead of wrapping around onto it.
    let x_start_raw = pen_x.saturating_sub(glyph.x);
    let y_start_raw = pen_y.saturating_sub(glyph.y);

    // Clamp to bitmap bounds.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_w = bitmap.width as i32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "bitmap dims ≤ i32::MAX in practice"
    )]
    let bmp_h = bitmap.height as i32;

    let x_clip = x_start_raw.max(0);
    let y_clip = y_start_raw.max(0);
    let x_end = x_start_raw.saturating_add(glyph.w).min(bmp_w);
    let y_end = y_start_raw.saturating_add(glyph.h).min(bmp_h);

    if x_clip >= x_end || y_clip >= y_end {
        return;
    }

    // All four differences are non-negative because of the checks above.
    #[expect(
        clippy::cast_sign_loss,
        reason = "x_clip ≥ x_start_raw.max(0) so difference ≥ 0"
    )]
    let x_data_skip = (x_clip - x_start_raw) as usize;
    #[expect(
        clippy::cast_sign_loss,
        reason = "y_clip ≥ y_start_raw.max(0) so difference ≥ 0"
    )]
    let y_data_skip = (y_clip - y_start_raw) as usize;
    #[expect(clippy::cast_sign_loss, reason = "x_end > x_clip by guard above")]
    let xx_limit = (x_end - x_clip) as usize;
    #[expect(clippy::cast_sign_loss, reason = "y_end > y_clip by guard above")]
    let yy_limit = (y_end - y_clip) as usize;
    let row_bytes = glyph.row_bytes();
    let mut clip_rows =
        (!clip_all_inside).then(|| ClipRows::new(clip, x_clip, y_clip, x_end - 1, y_end - 1));
    if clip_rows
        .as_ref()
        .is_some_and(|rows| rows.bbox() == ClipResult::AllOutside)
    {
        return;
    }

    if glyph.aa {
        blit_aa::<P>(
            bitmap,
            clip_rows.as_mut(),
            pipe,
            src,
            glyph,
            x_clip,
            y_clip,
            x_data_skip,
            y_data_skip,
            xx_limit,
            yy_limit,
            row_bytes,
        );
    } else {
        blit_mono::<P>(
            bitmap,
            clip_rows.as_mut(),
            pipe,
            src,
            glyph,
            x_clip,
            y_clip,
            x_data_skip,
            y_data_skip,
            xx_limit,
            yy_limit,
            row_bytes,
        );
    }
}

/// Blit an AA (per-byte coverage) glyph.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper; all params necessary for this blit path"
)]
fn blit_aa<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    mut clip_rows: Option<&mut ClipRows<'_>>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    glyph: &GlyphBitmap<'_>,
    x_start: i32,
    y_start: i32,
    x_data_skip: usize,
    y_data_skip: usize,
    xx_limit: usize,
    yy_limit: usize,
    row_bytes: usize,
) {
    let data = glyph.data;
    // Verify the glyph data buffer covers the visible region.  A mismatch
    // indicates a malformed GlyphBitmap (wrong w/h/row_bytes); we clamp via
    // get() in the inner loop, but the assert catches it in debug builds.
    debug_assert!(
        data.len() >= (y_data_skip + yy_limit) * row_bytes,
        "blit_aa: glyph data too short: len={} < (y_data_skip={y_data_skip} + yy_limit={yy_limit}) * row_bytes={row_bytes}",
        data.len(),
    );
    // Hoisted above the row loop to avoid per-row heap allocation.
    let mut run_shape: Vec<u8> = Vec::new();

    for yy in 0..yy_limit {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "yy < yy_limit ≤ bitmap.height ≤ i32::MAX"
        )]
        #[expect(clippy::cast_possible_wrap, reason = "yy < bitmap.height ≤ i32::MAX")]
        let y = y_start + yy as i32;
        let Some(row_clip) = row_clip(clip_rows.as_deref_mut(), y) else {
            continue;
        };
        let row_off = (y_data_skip + yy) * row_bytes + x_data_skip;

        let mut run_start: Option<i32> = None;
        run_shape.clear();

        for xx in 0..xx_limit {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "xx < xx_limit ≤ bitmap.width ≤ i32::MAX"
            )]
            #[expect(clippy::cast_possible_wrap, reason = "xx < bitmap.width ≤ i32::MAX")]
            let x = x_start + xx as i32;
            let data_idx = row_off + xx;
            let alpha = clipped(data.get(data_idx).copied().unwrap_or(0), row_clip, x);

            if alpha != 0 {
                if run_start.is_none() {
                    run_start = Some(x);
                    run_shape.clear();
                }
                run_shape.push(alpha);
            } else if let Some(rs) = run_start.take() {
                emit_aa_run::<P>(bitmap, pipe, src, rs, y, &run_shape);
            }
        }
        if let Some(rs) = run_start.take() {
            emit_aa_run::<P>(bitmap, pipe, src, rs, y, &run_shape);
        }
    }
}

/// Blit a mono (1-bit packed) glyph.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper; all params necessary"
)]
#[expect(
    clippy::too_many_lines,
    reason = "function handles both SIMD and scalar mono-unpack paths; splitting further would obscure the logic"
)]
fn blit_mono<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    mut clip_rows: Option<&mut ClipRows<'_>>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    glyph: &GlyphBitmap<'_>,
    x_start: i32,
    y_start: i32,
    x_data_skip: usize,
    y_data_skip: usize,
    xx_limit: usize,
    yy_limit: usize,
    row_bytes: usize,
) {
    let x_shift = x_data_skip % 8;
    let data = glyph.data;
    // An undersized `data` buffer is tolerated rather than asserted: bytes
    // past the end read as zero bits on every path below, matching the AA
    // path's clamped reads.

    // Scratch buffer for SIMD-expanded bits: one byte per pixel, 0x00 or 0xFF.
    // Sized to the maximum row width; reused across rows.
    let mut expanded: Vec<u8> = vec![0u8; xx_limit];
    // Run shape scratch for rows the clip edge crosses.
    let mut shape: Vec<u8> = Vec::new();

    for yy in 0..yy_limit {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "yy < yy_limit ≤ bitmap.height ≤ i32::MAX"
        )]
        #[expect(clippy::cast_possible_wrap, reason = "yy < bitmap.height ≤ i32::MAX")]
        let y = y_start + yy as i32;
        let Some(row_clip) = row_clip(clip_rows.as_deref_mut(), y) else {
            continue;
        };
        if let RowClip::Partial { .. } = row_clip {
            blit_mono_row_clipped::<P>(
                bitmap,
                pipe,
                src,
                data,
                (y_data_skip + yy) * row_bytes,
                x_data_skip,
                x_start,
                xx_limit,
                y,
                row_clip,
                &mut shape,
            );
            continue;
        }
        let row_off = (y_data_skip + yy) * row_bytes + x_data_skip / 8;

        // Use SIMD unpack when x_shift == 0 (bits are byte-aligned) and the
        // whole packed row is present — the unpacker indexes the row without
        // clamping. When x_shift > 0 the pixels straddle byte boundaries and
        // when the buffer is short the bytes must read as zero, so both
        // cases take the scalar bit-extraction path, which handles them.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let use_simd_unpack = x_shift == 0 && data.len() >= row_off + xx_limit.div_ceil(8);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let use_simd_unpack = false;

        if use_simd_unpack {
            let packed_row = &data[row_off..];
            simd::unpack_mono_row(packed_row, xx_limit, &mut expanded);

            let mut run_start: Option<i32> = None;
            for (xx, &cov) in expanded[..xx_limit].iter().enumerate() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx < xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(clippy::cast_possible_wrap, reason = "xx < bitmap.width ≤ i32::MAX")]
                let x = x_start + xx as i32;
                if cov != 0 {
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else if let Some(rs) = run_start.take() {
                    let rx1 = x - 1;
                    pipe::render_bitmap_span::<P>(bitmap, pipe, src, None, rs, rx1, y);
                }
            }
            if let Some(rs) = run_start.take() {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "xx_limit < bitmap.width ≤ i32::MAX"
                )]
                let rx1 = x_start + xx_limit as i32 - 1;
                pipe::render_bitmap_span::<P>(bitmap, pipe, src, None, rs, rx1, y);
            }
            continue;
        }

        let mut run_start: Option<i32> = None;
        let mut xx = 0usize;

        while xx < xx_limit {
            let byte_idx = row_off + xx / 8;

            // When x_shift > 0, straddle two source bytes to align the read
            // window — on every byte group, including the last: the `hi`
            // read clamps past the buffer end to 0, and bits shifted in from
            // beyond `xx_limit` are never consumed by the bit loop below.
            let alpha0 = if x_shift > 0 {
                let lo = data.get(byte_idx).copied().unwrap_or(0);
                let hi = data.get(byte_idx + 1).copied().unwrap_or(0);
                #[expect(clippy::cast_possible_truncation, reason = "shift result fits in u8")]
                {
                    (u16::from(lo) << x_shift | u16::from(hi) >> (8 - x_shift)) as u8
                }
            } else {
                data.get(byte_idx).copied().unwrap_or(0)
            };

            let bits_this_byte = (xx_limit - xx).min(8);
            for bit in 0..bits_this_byte {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "xx + bit < xx_limit ≤ bitmap.width ≤ i32::MAX"
                )]
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "xx + bit < bitmap.width ≤ i32::MAX"
                )]
                let x = x_start + (xx + bit) as i32;
                if (alpha0 >> (7 - bit)) & 1 != 0 {
                    if run_start.is_none() {
                        run_start = Some(x);
                    }
                } else if let Some(rs) = run_start.take() {
                    let rx1 = x - 1;
                    pipe::render_bitmap_span::<P>(bitmap, pipe, src, None, rs, rx1, y);
                }
            }
            xx += bits_this_byte;
        }
        if let Some(rs) = run_start.take() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "xx_limit ≤ bitmap.width ≤ i32::MAX"
            )]
            #[expect(
                clippy::cast_possible_wrap,
                reason = "xx_limit < bitmap.width ≤ i32::MAX"
            )]
            let rx1 = x_start + xx_limit as i32 - 1;
            pipe::render_bitmap_span::<P>(bitmap, pipe, src, None, rs, rx1, y);
        }
    }
}

/// The clip coverage of row `y`: full when no clip applies, `None` when the
/// clip excludes the row.
fn row_clip<'r>(clip_rows: Option<&'r mut ClipRows<'_>>, y: i32) -> Option<RowClip<'r>> {
    clip_rows.map_or(Some(RowClip::Full), |rows| rows.row(y))
}

/// Glyph coverage `alpha` at column `x` scaled by the clip's coverage.
#[inline]
fn clipped(alpha: u8, row_clip: RowClip<'_>, x: i32) -> u8 {
    match row_clip {
        RowClip::Full => alpha,
        RowClip::Partial { .. } => div255(u32::from(alpha) * u32::from(row_clip.coverage(x))),
    }
}

/// Blit one row of a mono glyph that the clip edge crosses: each set bit
/// paints with the clip's coverage of its pixel as its shape.  Bit `i` of
/// the row lies at bit `x_bit_skip + i` from byte `row_start`; bytes past the
/// end of `data` read as zero.
#[expect(
    clippy::too_many_arguments,
    reason = "internal helper; all params necessary"
)]
fn blit_mono_row_clipped<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    data: &[u8],
    row_start: usize,
    x_bit_skip: usize,
    x_start: i32,
    xx_limit: usize,
    y: i32,
    row_clip: RowClip<'_>,
    shape: &mut Vec<u8>,
) {
    let mut run_start: Option<i32> = None;
    shape.clear();
    for xx in 0..xx_limit {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "xx < xx_limit ≤ bitmap.width ≤ i32::MAX"
        )]
        let x = x_start + xx as i32;
        let bit = x_bit_skip + xx;
        let set = data
            .get(row_start + bit / 8)
            .is_some_and(|&byte| (byte >> (7 - bit % 8)) & 1 != 0);
        let cov = if set { row_clip.coverage(x) } else { 0 };
        if cov != 0 {
            if run_start.is_none() {
                run_start = Some(x);
                shape.clear();
            }
            shape.push(cov);
        } else if let Some(rs) = run_start.take() {
            emit_aa_run::<P>(bitmap, pipe, src, rs, y, shape);
        }
    }
    if let Some(rs) = run_start.take() {
        emit_aa_run::<P>(bitmap, pipe, src, rs, y, shape);
    }
}

/// Emit one AA-shaped span starting at `x0`.
fn emit_aa_run<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    x0: i32,
    y: i32,
    shape: &[u8],
) {
    debug_assert_ne!(shape, []);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "shape.len() ≤ bitmap.width ≤ i32::MAX"
    )]
    let x1 = x0 + shape.len() as i32 - 1;
    pipe::render_bitmap_span::<P>(bitmap, pipe, src, Some(shape), x0, x1, y);
}

/// Clip-test a glyph bbox and blit it.
///
/// Classifies the glyph's bounding box against `clip`, blits the glyph unless
/// it lies wholly outside, and returns the classification.
pub fn fill_glyph<P: Pixel>(
    bitmap: &mut Bitmap<P>,
    clip: &Clip,
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    pen_x: i32,
    pen_y: i32,
    glyph: &GlyphBitmap<'_>,
) -> ClipResult {
    let x0 = pen_x.saturating_sub(glyph.x);
    let y0 = pen_y.saturating_sub(glyph.y);
    let x1 = x0.saturating_add(glyph.w).saturating_sub(1);
    let y1 = y0.saturating_add(glyph.h).saturating_sub(1);

    let clip_res = clip.test_rect(x0, y0, x1, y1);
    if clip_res != ClipResult::AllOutside {
        blit_glyph::<P>(
            bitmap,
            clip,
            clip_res == ClipResult::AllInside,
            pipe,
            src,
            pen_x,
            pen_y,
            glyph,
        );
    }
    clip_res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;
    use crate::pipe::PipeSrc;
    use crate::testutil::{make_clip, simple_pipe};
    use color::Rgb8;

    #[test]
    fn blit_aa_glyph_paints_pixels() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        // 4×4 AA glyph with full coverage (255) everywhere.
        let data = vec![255u8; 16];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 4,
            h: 4,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 2, 2, &glyph);

        // Rows 2..5, cols 2..5 should be red.
        for y in 2..6u32 {
            let row = bmp.row(y);
            for (x, px) in row.iter().enumerate().skip(2).take(4) {
                assert_eq!(px.r, 255, "y={y} x={x} R");
            }
        }
        assert_eq!(bmp.row(1)[1].r, 0);
    }

    #[test]
    fn blit_aa_glyph_zero_coverage_skips() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(4, 4, 4, false);
        let clip = make_clip(4, 4);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        let data = vec![0u8; 4];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 0, 0, &glyph);

        for y in 0..4u32 {
            let row = bmp.row(y);
            for px in row.iter().take(4) {
                assert_eq!(px.r, 0, "should be unpainted");
            }
        }
    }

    #[test]
    fn blit_mono_glyph_paints_set_bits() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 4, 4, false);
        let clip = make_clip(8, 4);
        let pipe = simple_pipe();
        let color = [0u8, 255, 0];
        let src = PipeSrc::Solid(&color);

        // 8×2 mono glyph: row 0 all-set (0xFF), row 1 all-clear (0x00).
        let data = [0xFFu8, 0x00u8];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 8,
            h: 2,
            aa: false,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 0, 0, &glyph);

        let row0 = bmp.row(0);
        for (x, px) in row0.iter().enumerate().take(8) {
            assert_eq!(px.g, 255, "row 0 x={x} should be painted");
        }
        let row1 = bmp.row(1);
        for (x, px) in row1.iter().enumerate().take(8) {
            assert_eq!(px.g, 0, "row 1 x={x} should be clear");
        }
    }

    /// Left-clipped mono glyphs must paint exactly the source bits
    /// `skip..w`, for every skip and width.
    ///
    /// Sweeping `x_data_skip` over 0..8 (via negative `pen_x`) across widths
    /// 1..=40 drives both `blit_mono` tiers — the byte-aligned unpack at
    /// `skip % 8 == 0` and the scalar straddle path otherwise — against the
    /// same external reference: the packed source bits themselves.
    #[test]
    fn blit_mono_left_clip_paints_source_bits() {
        let pipe = simple_pipe();
        let color = [0u8, 255, 0];

        for w in 1usize..=40 {
            let row_bytes = w.div_ceil(8);
            let data: Vec<u8> = (0..row_bytes)
                .map(|i| [0xAAu8, 0x5C, 0x93][i % 3])
                .collect();
            for skip in 0usize..8 {
                let mut bmp: Bitmap<Rgb8> = Bitmap::new(48, 1, 4, false);
                let clip = make_clip(48, 1);
                let src = PipeSrc::Solid(&color);
                let glyph = GlyphBitmap {
                    data: &data,
                    x: 0,
                    y: 0,
                    w: i32::try_from(w).expect("w ≤ 40"),
                    h: 1,
                    aa: false,
                };
                let x0 = -i32::try_from(skip).expect("skip < 8");

                blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, x0, 0, &glyph);

                let visible = w.saturating_sub(skip);
                let row = bmp.row(0);
                for (xx, px) in row.iter().enumerate().take(visible) {
                    let bit_idx = skip + xx;
                    let expected = (data[bit_idx / 8] >> (7 - bit_idx % 8)) & 1 != 0;
                    assert_eq!(
                        px.g == 255,
                        expected,
                        "w={w} skip={skip} xx={xx}: painted != source bit"
                    );
                }
                for (xx, px) in row.iter().enumerate().skip(visible).take(48 - visible) {
                    assert_eq!(px.g, 0, "w={w} skip={skip} x={xx} beyond the glyph");
                }
            }
        }
    }

    /// A mono glyph whose data buffer is shorter than `h * row_bytes` must
    /// not panic: rows past the end of the buffer read as zero bits, the
    /// same degraded behaviour the AA path's clamped reads produce.
    #[test]
    fn blit_mono_short_buffer_reads_missing_rows_as_zero() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(32, 4, 4, false);
        let clip = make_clip(32, 4);
        let pipe = simple_pipe();
        let color = [0u8, 255, 0];
        let src = PipeSrc::Solid(&color);

        // 32×4 mono glyph needs 16 bytes; supply rows 0–1 only.
        let data = vec![0xFFu8; 8];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 32,
            h: 4,
            aa: false,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, true, &pipe, &src, 0, 0, &glyph);

        for y in 0..2u32 {
            for (x, px) in bmp.row(y).iter().enumerate().take(32) {
                assert_eq!(px.g, 255, "({x}, {y}) inside supplied rows must paint");
            }
        }
        for y in 2..4u32 {
            for (x, px) in bmp.row(y).iter().enumerate().take(32) {
                assert_eq!(px.g, 0, "({x}, {y}) past the buffer must read as clear");
            }
        }
    }

    #[test]
    fn blit_glyph_clip_excludes_outside() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = Clip::new(2.0, 2.0, 5.0, 5.0, false);
        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let src = PipeSrc::Solid(&color);

        let data = vec![255u8; 16];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 4,
            h: 4,
            aa: true,
        };

        blit_glyph::<Rgb8>(&mut bmp, &clip, false, &pipe, &src, 0, 0, &glyph);

        assert_eq!(bmp.row(0)[0].r, 0, "row 0 should be clipped");
        assert_eq!(bmp.row(1)[0].r, 0, "row 1 should be clipped");
        assert_eq!(bmp.row(2)[2].r, 255, "(2,2) should be painted");
    }

    /// A glyph crossed by the diagonal edge of an antialiased path clip takes
    /// the clip's fractional coverage at the edge, for both glyph formats:
    /// the edge pixels are painted partially, the interior fully, the
    /// exterior not at all.
    #[test]
    fn aa_clip_edge_antialiases_glyph_pixels() {
        use crate::path::PathBuilder;
        use crate::xpath::XPath;

        let mut clip = Clip::new(0.0, 0.0, 16.0, 16.0, true);
        let mut builder = PathBuilder::new();
        builder.move_to(0.0, 0.0).expect("move_to");
        builder.line_to(16.0, 16.0).expect("line_to");
        builder.line_to(0.0, 16.0).expect("line_to");
        builder.close(false).expect("close");
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        clip.clip_to_path(&XPath::new(&builder.build(), &identity, 0.1, true), false);

        let pipe = simple_pipe();
        let color = [255u8, 0, 0];
        let aa_data = vec![255u8; 16 * 16];
        let mono_data = [0xFFu8; 2 * 16];
        for (aa, data) in [(true, &aa_data[..]), (false, &mono_data[..])] {
            let mut bmp: Bitmap<Rgb8> = Bitmap::new(16, 16, 4, false);
            let src = PipeSrc::Solid(&color);
            let glyph = GlyphBitmap {
                data,
                x: 0,
                y: 0,
                w: 16,
                h: 16,
                aa,
            };
            let res = fill_glyph::<Rgb8>(&mut bmp, &clip, &pipe, &src, 0, 0, &glyph);
            assert_eq!(res, ClipResult::Partial);

            // Below the diagonal y = x: inside; above: outside; on it: half.
            assert_eq!(bmp.row(10)[2].r, 255, "aa={aa}: interior pixel");
            assert_eq!(bmp.row(2)[10].r, 0, "aa={aa}: exterior pixel");
            let edge = bmp.row(8)[8].r;
            assert!(
                0 < edge && edge < 255,
                "aa={aa}: edge pixel must be partially covered, got {edge}"
            );
        }
    }

    #[test]
    fn fill_glyph_returns_clip_result() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(8, 8, 4, false);
        let clip = make_clip(8, 8);
        let pipe = simple_pipe();
        let color = [128u8, 128, 128];
        let src = PipeSrc::Solid(&color);

        let data = vec![128u8; 4];
        let glyph = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 2,
            h: 2,
            aa: true,
        };

        let res = fill_glyph::<Rgb8>(&mut bmp, &clip, &pipe, &src, 1, 1, &glyph);
        assert_eq!(res, ClipResult::AllInside);
    }

    #[test]
    fn glyph_row_bytes_aa() {
        let data = [];
        let g = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 7,
            h: 1,
            aa: true,
        };
        assert_eq!(g.row_bytes(), 7);
    }

    #[test]
    fn glyph_row_bytes_mono() {
        let data = [];
        let g = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 7,
            h: 1,
            aa: false,
        };
        assert_eq!(g.row_bytes(), 1); // ceil(7/8) = 1
        let g9 = GlyphBitmap {
            data: &data,
            x: 0,
            y: 0,
            w: 9,
            h: 1,
            aa: false,
        };
        assert_eq!(g9.row_bytes(), 2); // ceil(9/8) = 2
    }

    /// A pen position at the edge of the i32 range misses the bitmap
    /// instead of overflowing the glyph placement.
    #[test]
    fn far_pen_position_paints_nothing() {
        let mut bmp: Bitmap<Rgb8> = Bitmap::new(4, 4, 1, false);
        let data = [255u8; 4];
        let glyph = GlyphBitmap {
            data: &data,
            x: -2,
            y: -2,
            w: 2,
            h: 2,
            aa: true,
        };
        let src = PipeSrc::Solid(&[255, 255, 255]);
        for (px, py) in [(i32::MAX, 0), (0, i32::MAX), (i32::MIN, i32::MIN)] {
            let res = fill_glyph::<Rgb8>(
                &mut bmp,
                &make_clip(4, 4),
                &simple_pipe(),
                &src,
                px,
                py,
                &glyph,
            );
            assert_eq!(res, ClipResult::AllOutside, "pen ({px}, {py})");
        }
        assert!(bmp.data().iter().all(|&b| b == 0));
    }
}
