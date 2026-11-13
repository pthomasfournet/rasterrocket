//! Rendering pages into memory the caller owns, through a page canvas
//! reused from one render to the next.

use color::Rgb8;
use raster::{Bitmap, BitmapBuffers};

use crate::gray::{rgb_into_gray, rgb_rows_to_gray};
use crate::page::{rgb_rows_copy, rgb_rows_to_rgba};
use crate::render::{check_scale, page_view, render_page_rgb_with_geom};
use crate::{
    PageDiagnostics, PixelFormat, RasterError, RasterOptions, RasterSession, RenderParams,
};

/// Storage reused across page renders, through which pages are rendered
/// into buffers the caller owns.
///
/// Every page is drawn on an RGB canvas the size of the page.  A
/// `PageBuffer` keeps that canvas's allocation between renders, and each
/// `render_*_into` method writes the finished page into the caller's
/// buffer in its pixel format.  Once the canvas has grown to the largest
/// page rendered through it, a render allocates no buffer the size of the
/// page: neither the canvas nor the output.  The allocations the drawing
/// itself makes (decoded images, glyphs, transparency groups, the skew
/// profile of deskew) are unchanged.
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
    canvas: BitmapBuffers,
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
            crate::deskew::apply_into(&gray, out, stride);
            self.canvas = gray.into_buffers();
        } else {
            rgb_rows_to_gray(&rgb, out, stride);
            self.canvas = rgb.into_buffers();
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
            return Err(RasterError::InvalidOptions(
                "deskew applies to gray pages only".to_owned(),
            ));
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
        self.canvas = rgb.into_buffers();
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
        let scale = f64::from(opts.dpi) / 72.0;
        check_scale(scale)?;
        let page_id = session.resolve_page(page_num)?;
        let geom = pdf_interp::page_size_pts_by_id(&session.doc, page_id, session.page_box)?;
        let view = page_view(&geom, (scale, scale), None)?;
        check_output(view.width, view.height, format, out_len, stride)?;
        let mut params = RenderParams::new(scale);
        if transparent {
            params = params.with_transparent_background();
        }
        params.cancel.clone_from(&opts.cancel);
        render_page_rgb_with_geom(
            session,
            page_num,
            page_id,
            geom,
            &params,
            std::mem::take(&mut self.canvas),
        )
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
