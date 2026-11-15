//! Rendering pages into memory the caller owns, through a page canvas
//! reused from one render to the next.

use color::Rgb8;
use pdf_interp::renderer::CanvasBuffers;
use raster::Bitmap;

use crate::CanvasLayout;
use crate::gray::{rgb_into_gray, rgb_rows_to_gray};
use crate::page::{rgb_rows_copy, rgb_rows_to_rgba};
use crate::render::{check_scale, page_view, render_page_canvas_with_geom, rgb8};
use crate::{
    PageDiagnostics, PixelFormat, RasterError, RasterOptions, RasterSession, RenderParams,
};

/// Storage reused across page renders, through which pages are rendered
/// into buffers the caller owns.
///
/// Every page is drawn on a canvas the size of the page: RGB, or ink when
/// the session simulates overprint on a page that sets it, with an alpha
/// plane when the page is painted as a page group.  A `PageBuffer` keeps
/// the allocations of both canvases and of the alpha plane between
/// renders, and each `render_*_into` method writes the finished page into
/// the caller's buffer in its pixel format.  Once they have grown to the
/// largest page rendered through them, a render allocates no buffer the
/// size of the page: neither a canvas, the alpha plane nor the output.
/// The allocations the drawing itself makes (decoded images, glyphs,
/// transparency groups, the skew profile of deskew) are unchanged, and so
/// are the two page-sized copies a GPU rotation makes under the
/// `gpu-deskew` feature.
///
/// The output is byte-identical to the owned-buffer entry points:
/// [`render_gray_into`](Self::render_gray_into) writes the pixels of
/// [`render_page_gray`](crate::render_page_gray),
/// [`render_rgba_into`](Self::render_rgba_into) those of
/// [`render_page_rgba`](crate::render_page_rgba) and
/// [`render_rgb_into`](Self::render_rgb_into) those of
/// [`render_page_rgb`](crate::render_page_rgb) at `opts.dpi / 72`.
///
/// The output buffer holds row `y` at `out[y * stride..]`; `stride` is in
/// bytes, at least the page width times the format's
/// [`bytes_per_pixel`](PixelFormat::bytes_per_pixel), and `out` holds at
/// least `(height - 1) * stride + width * bytes_per_pixel` bytes.  Bytes
/// between rows are left as they are.  [`RasterSession::page_size_px`]
/// gives the page's size before the render.  A buffer that is too small is
/// rejected before anything is drawn.
///
/// The methods take `&mut self`: a thread rendering pages in parallel keeps
/// its own `PageBuffer`.  A render that fails releases the canvas, which
/// the next render allocates again.
#[derive(Debug, Default)]
pub struct PageBuffer {
    /// The page canvas's storage between renders.
    canvas: CanvasBuffers,
}

/// What a `render_*_into` method of [`PageBuffer`] wrote.
#[derive(Debug)]
#[non_exhaustive]
pub struct PageInfo {
    /// Page number (1-based).
    pub page_num: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The layout of the pixels written.
    pub format: PixelFormat,
    /// Resolution in pixels per inch (`opts.dpi`); see
    /// [`RenderedPage::dpi`](crate::RenderedPage::dpi).
    pub dpi: f32,
    /// Metadata collected during the render; see
    /// [`RenderedPage::diagnostics`](crate::RenderedPage::diagnostics).
    pub diagnostics: PageDiagnostics,
}

impl PageBuffer {
    /// A buffer holding no storage yet; the first render allocates it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Render page `page_num` (1-based) of `session` as 8-bit gray into
    /// `out`, row `y` at `out[y * stride..]`.
    ///
    /// The pixels are those of [`render_page_gray`](crate::render_page_gray)
    /// with the same `opts`, of which only [`dpi`](RasterOptions::dpi),
    /// [`deskew`](RasterOptions::deskew) and
    /// [`cancel`](RasterOptions::cancel) apply.
    ///
    /// # Errors
    ///
    /// As [`render_page_rgb_with`](crate::render_page_rgb_with), plus
    /// [`RasterError::OutputStrideTooSmall`] when `stride < width` and
    /// [`RasterError::OutputBufferTooSmall`] when `out` is shorter than the
    /// page; `out` is then untouched.
    pub fn render_gray_into(
        &mut self,
        session: &RasterSession,
        page_num: u32,
        opts: &RasterOptions,
        out: &mut [u8],
        stride: usize,
    ) -> Result<PageInfo, RasterError> {
        let format = PixelFormat::Gray8;
        let (rgb, diagnostics) =
            self.render_canvas(session, page_num, opts, format, false, out.len(), stride)?;
        let (width, height) = (rgb.width, rgb.height);
        if opts.deskew {
            let gray = rgb_into_gray(rgb);
            let deskewed = crate::deskew::apply_into(&gray, out, stride);
            self.canvas.rgb.reclaim(gray);
            deskewed.map_err(|e| RasterError::Deskew(e.to_string()))?;
        } else {
            rgb_rows_to_gray(&rgb, out, stride);
            self.canvas.rgb.reclaim(rgb);
        }
        Ok(PageInfo {
            page_num,
            width,
            height,
            format,
            dpi: opts.dpi,
            diagnostics,
        })
    }

    /// Render page `page_num` (1-based) of `session` as packed RGB, three
    /// bytes per pixel, into `out`, row `y` at `out[y * stride..]`.
    ///
    /// The pixels are those of [`render_page_rgb`](crate::render_page_rgb)
    /// at `opts.dpi / 72`.  Of `opts`, only [`dpi`](RasterOptions::dpi) and
    /// [`cancel`](RasterOptions::cancel) apply.
    ///
    /// # Errors
    ///
    /// As [`render_page_rgb_with`](crate::render_page_rgb_with), plus
    /// [`RasterError::InvalidOptions`] when [`deskew`](RasterOptions::deskew)
    /// is set, and the output buffer errors of
    /// [`render_gray_into`](Self::render_gray_into) with `width * 3` bytes
    /// per row.
    pub fn render_rgb_into(
        &mut self,
        session: &RasterSession,
        page_num: u32,
        opts: &RasterOptions,
        out: &mut [u8],
        stride: usize,
    ) -> Result<PageInfo, RasterError> {
        self.render_colour_into(session, page_num, opts, PixelFormat::Rgb8, out, stride)
    }

    /// Render page `page_num` (1-based) of `session` over a transparent
    /// backdrop as RGBA, four bytes per pixel with alpha not premultiplied,
    /// into `out`, row `y` at `out[y * stride..]`.
    ///
    /// The pixels are those of [`render_page_rgba`](crate::render_page_rgba)
    /// with the same `opts`, of which only [`dpi`](RasterOptions::dpi) and
    /// [`cancel`](RasterOptions::cancel) apply.
    ///
    /// # Errors
    ///
    /// As [`render_page_rgba`](crate::render_page_rgba), plus the output
    /// buffer errors of [`render_gray_into`](Self::render_gray_into) with
    /// `width * 4` bytes per row.
    pub fn render_rgba_into(
        &mut self,
        session: &RasterSession,
        page_num: u32,
        opts: &RasterOptions,
        out: &mut [u8],
        stride: usize,
    ) -> Result<PageInfo, RasterError> {
        self.render_colour_into(session, page_num, opts, PixelFormat::Rgba8, out, stride)
    }

    /// [`Self::render_rgb_into`] or [`Self::render_rgba_into`], by `format`.
    fn render_colour_into(
        &mut self,
        session: &RasterSession,
        page_num: u32,
        opts: &RasterOptions,
        format: PixelFormat,
        out: &mut [u8],
        stride: usize,
    ) -> Result<PageInfo, RasterError> {
        if opts.deskew {
            return Err(RasterError::deskew_needs_gray());
        }
        let transparent = format == PixelFormat::Rgba8;
        let (rgb, diagnostics) = self.render_canvas(
            session,
            page_num,
            opts,
            format,
            transparent,
            out.len(),
            stride,
        )?;
        if transparent {
            rgb_rows_to_rgba(&rgb, out, stride);
        } else {
            rgb_rows_copy(&rgb, out, stride);
        }
        let (width, height) = (rgb.width, rgb.height);
        self.canvas.rgb.reclaim(rgb);
        Ok(PageInfo {
            page_num,
            width,
            height,
            format,
            dpi: opts.dpi,
            diagnostics,
        })
    }

    /// Check an output buffer of `out_len` bytes at `stride` against the
    /// page in `format`, then render the page whole at `opts.dpi` on the
    /// canvas held by `self`, over a transparent backdrop when
    /// `transparent`.
    #[expect(
        clippy::too_many_arguments,
        reason = "one private helper behind the three public entry points"
    )]
    fn render_canvas(
        &mut self,
        session: &RasterSession,
        page_num: u32,
        opts: &RasterOptions,
        format: PixelFormat,
        transparent: bool,
        out_len: usize,
        stride: usize,
    ) -> Result<(Bitmap<Rgb8>, PageDiagnostics), RasterError> {
        if opts.depth != color::SampleDepth::Eight {
            return Err(RasterError::InvalidOptions(
                "a page buffer holds 8-bit pixels: render 16-bit samples with \
                 render_page_canvas"
                    .to_owned(),
            ));
        }
        if session.output_profile.is_some() {
            return Err(RasterError::InvalidOptions(
                "a page buffer holds sRGB pixels: render the pages of a session with an \
                 output profile with render_page_canvas or render_page_bands"
                    .to_owned(),
            ));
        }
        let scale = f64::from(opts.dpi) / 72.0;
        check_scale(scale)?;
        let page_id = session.resolve_page(page_num)?;
        let geom = session.geometry_by_id(page_id)?;
        let view = page_view(
            &geom,
            (scale, scale),
            None,
            CanvasLayout::new(color::SampleDepth::Eight, transparent).bytes_per_px(),
            session.max_page_bytes,
        )?;
        check_output(view.width, view.height, format, out_len, stride)?;
        let mut params = RenderParams::new(scale);
        if transparent {
            params = params.with_transparent_background();
        }
        params.cancel.clone_from(&opts.cancel);
        let (canvas, diag) = render_page_canvas_with_geom(
            session,
            page_num,
            page_id,
            geom,
            &params,
            &mut self.canvas,
        )?;
        Ok((rgb8(canvas)?, diag))
    }
}

/// The output buffer errors of a `width × height` page in `format` written
/// to `len` bytes at `stride`.  `height ≥ 1`.
fn check_output(
    width: u32,
    height: u32,
    format: PixelFormat,
    len: usize,
    stride: usize,
) -> Result<(), RasterError> {
    let row_bytes = width as usize * format.bytes_per_pixel();
    if stride < row_bytes {
        return Err(RasterError::OutputStrideTooSmall { stride, row_bytes });
    }
    let needed = (height as usize - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(row_bytes))
        .unwrap_or(usize::MAX);
    if len < needed {
        return Err(RasterError::OutputBufferTooSmall { len, needed });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_pdf::{assemble, stream};
    use crate::{BackendPolicy, SessionConfig, open_session_from_bytes};

    /// A 50 × 40 pt page painted as a page group, its graphics state
    /// setting a blend mode, then a page of the same size whose graphics
    /// state sets overprint.
    fn group_and_overprint_pages() -> Vec<u8> {
        assemble(&[
            "<</Type /Catalog /Pages 2 0 R>>",
            "<</Type /Pages /Kids [3 0 R 4 0 R] /Count 2>>",
            "<</Type /Page /Parent 2 0 R /MediaBox [0 0 50 40] /Contents 5 0 R \
             /Resources <</ExtGState <</G <</BM /Multiply>>>>>>>>",
            "<</Type /Page /Parent 2 0 R /MediaBox [0 0 50 40] /Contents 6 0 R \
             /Resources <</ExtGState <</G <</OP true /op true>>>>>>>>",
            &stream("/G gs 0 0 1 rg 5 5 20 20 re f"),
            &stream("/G gs 1 0 0 rg 5 5 20 20 re f"),
        ])
    }

    /// The alpha plane of a page group composited onto paper and the ink
    /// canvas of an overprint page are kept and reused by the next render.
    #[test]
    fn the_alpha_plane_and_the_ink_canvas_are_reused() {
        let config = SessionConfig {
            overprint: true,
            ..SessionConfig::with_policy(BackendPolicy::CpuOnly)
        };
        let session = open_session_from_bytes(group_and_overprint_pages(), &config).unwrap();
        let opts = RasterOptions::new(72.0);
        let mut out = vec![0; 50 * 40];
        let mut buffer = PageBuffer::new();
        let mut render = |buffer: &mut PageBuffer, page| {
            let info = buffer
                .render_gray_into(&session, page, &opts, &mut out, 50)
                .unwrap();
            assert_eq!(info.page_num, page);
        };

        render(&mut buffer, 1);
        assert!(buffer.canvas.rgb.alpha.capacity() >= 50 * 40);
        let alpha = buffer.canvas.rgb.alpha.as_ptr();
        render(&mut buffer, 1);
        assert_eq!(
            buffer.canvas.rgb.alpha.as_ptr(),
            alpha,
            "alpha plane reused"
        );

        render(&mut buffer, 2);
        assert!(buffer.canvas.ink.data.capacity() >= 50 * 40 * 8);
        let ink = buffer.canvas.ink.data.as_ptr();
        render(&mut buffer, 2);
        assert_eq!(buffer.canvas.ink.data.as_ptr(), ink, "ink canvas reused");
        assert_eq!(buffer.canvas.rgb.alpha.as_ptr(), alpha, "alpha plane kept");
    }

    #[test]
    fn output_checks() {
        let gray = PixelFormat::Gray8;
        assert!(check_output(10, 3, gray, 30, 10).is_ok());
        assert!(
            check_output(10, 3, gray, 32, 11).is_ok(),
            "last row unpadded"
        );
        assert!(matches!(
            check_output(10, 3, gray, 31, 11),
            Err(RasterError::OutputBufferTooSmall {
                len: 31,
                needed: 32
            })
        ));
        assert!(matches!(
            check_output(10, 3, PixelFormat::Rgba8, 1000, 39),
            Err(RasterError::OutputStrideTooSmall {
                stride: 39,
                row_bytes: 40
            })
        ));
        assert!(matches!(
            check_output(1, 3, gray, usize::MAX - 1, usize::MAX),
            Err(RasterError::OutputBufferTooSmall {
                needed: usize::MAX,
                ..
            })
        ));
    }
}
