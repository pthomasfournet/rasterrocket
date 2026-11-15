//! Pages written as they render, band by band: each band of rows is
//! brought to the output's colour model ([`tone::prepare_band`]) and handed
//! to a row-streamed writer, so memory holds a band, not the page.  A page
//! of one band takes the same writers, so the bytes of a file do not
//! depend on the band height.  A 16-bit page is always written by these
//! writers: as one band of the whole page when `--invert auto` measures it.

use std::fs::File;
use std::io::BufWriter;

use color::{Rgb8, SampleDepth};
use encode::{
    IccProfile, PngColor, PngWriter, PnmKind, PnmWriter, RowImage, RowLayout, TiffCompression,
    TiffStream,
};
use pdf_raster::{
    BandBitmap, BandError, FinishedCanvas, RasterError, RasterSession, RenderParams,
    render_page_bands, render_page_canvas,
};
use raster::Bitmap;

use crate::args::{Args, InvertWhen, OutputFormat};
use crate::cleanup::Page;
use crate::naming::output_path_with_prefix;
use crate::print::{PageSink, page_encoder};
use crate::ram::SpillPolicy;
use crate::render::{RenderError, write_output};
use crate::tone::{self, WidePage};

/// Where a page written band by band goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BandOutput {
    /// A file of its own.
    File(FileFormat),
    /// The print job.
    Print,
}

/// The format of a page file written band by band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileFormat {
    Ppm,
    Png,
    Tiff(TiffCompression),
}

impl BandOutput {
    /// The output of `args` when its pages are written band by band: a
    /// PPM, PNG or TIFF file or the print spool, under no stage that needs
    /// the whole page (the camera stages, `--invert auto` at 8 bits),
    /// outside a fax document.  `None` renders pages whole.  Every 16-bit
    /// output is one of these.
    pub(crate) fn of(args: &Args) -> Option<Self> {
        let camera = args.camera_options();
        if args.faxing()
            || camera.crop
            || camera.rectify
            || camera.dewarp
            || camera.split_spreads
            || (args.invert_when() == InvertWhen::Auto && args.depth() == SampleDepth::Eight)
        {
            return None;
        }
        if args.printing() {
            return Some(Self::Print);
        }
        let format = match args.output_format() {
            OutputFormat::Ppm => FileFormat::Ppm,
            OutputFormat::Png => FileFormat::Png,
            OutputFormat::Tiff(compression) => FileFormat::Tiff(compression),
            OutputFormat::Jpeg(_) => return None,
        };
        Some(Self::File(format))
    }
}

/// The colour model a page is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    /// Colour, with the render's alpha plane when `alpha`.
    Rgb { alpha: bool },
    /// Luma.
    Gray,
    /// Two levels.
    Mono,
    /// CMYK.
    Cmyk,
}

impl Model {
    fn of(args: &Args) -> Self {
        if args.cmyk() {
            Self::Cmyk
        } else if args.mono {
            Self::Mono
        } else if args.luma_output() {
            Self::Gray
        } else {
            Self::Rgb {
                alpha: args.transparent,
            }
        }
    }
}

/// A page file's writer.
enum FileWriter {
    Pnm(PnmWriter<BufWriter<File>>),
    Png(PngWriter<BufWriter<File>>),
    Tiff(TiffStream<BufWriter<File>>),
}

impl FileWriter {
    /// The writer of a `size` page in `model`, of samples of `depth`, to
    /// `out` in `format`, at `dpi` for the formats that record it, tagged
    /// with `icc` by the formats that carry a profile.
    fn new(
        out: BufWriter<File>,
        format: FileFormat,
        (model, depth): (Model, SampleDepth),
        (size, dpi): ((u32, u32), (f64, f64)),
        icc: Option<&IccProfile>,
    ) -> Result<Self, RenderError> {
        let wide = depth == SampleDepth::Sixteen;
        let no_cmyk = || -> RenderError {
            RasterError::InvalidOptions("CMYK pages are written as TIFF or JPEG".to_owned()).into()
        };
        Ok(match format {
            FileFormat::Ppm => {
                let kind = match (model, wide) {
                    (Model::Rgb { .. }, false) => PnmKind::Ppm,
                    (Model::Rgb { .. }, true) => PnmKind::Ppm16,
                    (Model::Gray, false) => PnmKind::Pgm,
                    (Model::Gray, true) => PnmKind::Pgm16,
                    (Model::Mono, _) => PnmKind::Pbm,
                    (Model::Cmyk, _) => return Err(no_cmyk()),
                };
                Self::Pnm(PnmWriter::new(out, size, kind)?)
            }
            FileFormat::Png => {
                let color = match (model, wide) {
                    (Model::Rgb { alpha: false }, false) => PngColor::Rgb,
                    (Model::Rgb { alpha: true }, false) => PngColor::Rgba,
                    (Model::Gray | Model::Mono, false) => PngColor::Gray,
                    (Model::Rgb { alpha: false }, true) => PngColor::Rgb16,
                    (Model::Rgb { alpha: true }, true) => PngColor::Rgba16,
                    (Model::Gray | Model::Mono, true) => PngColor::Gray16,
                    (Model::Cmyk, _) => return Err(no_cmyk()),
                };
                Self::Png(PngWriter::new(out, size, color, icc)?)
            }
            FileFormat::Tiff(compression) => {
                let layout = match model {
                    Model::Rgb { alpha: false } => RowLayout::Rgb,
                    Model::Rgb { alpha: true } => RowLayout::Rgba,
                    Model::Gray => RowLayout::Gray,
                    Model::Mono => RowLayout::Bilevel,
                    Model::Cmyk => RowLayout::Cmyk,
                };
                Self::Tiff(TiffStream::new(
                    out,
                    size,
                    layout.at_depth(depth),
                    dpi,
                    compression,
                    icc,
                )?)
            }
        })
    }

    fn push(&mut self, band: RowImage<'_>) -> Result<(), RenderError> {
        match self {
            Self::Pnm(w) => w.push(band)?,
            Self::Png(w) => w.push(band)?,
            Self::Tiff(w) => w.push(band)?,
        }
        Ok(())
    }

    fn finish(self) -> Result<BufWriter<File>, RenderError> {
        Ok(match self {
            Self::Pnm(w) => w.finish()?,
            Self::Png(w) => w.finish()?,
            Self::Tiff(w) => w.finish()?,
        })
    }
}

/// Render page `page_num` under `params`, a page of `size` pixels, in
/// bands written to `output` as each completes: brought to the output's
/// colour model, `push`ed, then the page finished.  Under `--invert auto`,
/// which measures the whole page, the page is rendered whole and pushed
/// as one band.
fn render_into(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
    args: &Args,
    mut push: impl FnMut(RowImage<'_>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    if args.invert_when() == InvertWhen::Auto {
        let canvas = render_page_canvas(session, page_num, params)?;
        let invert = match &canvas {
            FinishedCanvas::Rgb8(rgb) => tone::is_reversed_out8(rgb),
            FinishedCanvas::Rgb16(rgb) => tone::is_reversed_out16(rgb),
            FinishedCanvas::Gray8(gray) => tone::is_reversed_out(gray),
            FinishedCanvas::Gray16(gray) => tone::is_reversed_out(gray),
            FinishedCanvas::Cmyk8(_) | FinishedCanvas::Cmyk16(_) => false,
        };
        return prepared(BandBitmap::of(&canvas), 0, args, invert, &mut push);
    }
    let invert = args.invert_when() == InvertWhen::Always;
    render_page_bands(session, page_num, params, |band| {
        prepared(band.bitmap, band.y, args, invert, &mut push)
    })
    .map(drop)
    .map_err(|e| match e {
        BandError::Raster(e) => RenderError::Raster(e),
        BandError::Sink(e) => e,
    })
}

/// Bring `band`, rows from `y` of the page, to the output's colour model,
/// inverted when `invert`, and hand it to `push`.
fn prepared(
    band: BandBitmap<'_>,
    y: u32,
    args: &Args,
    invert: bool,
    push: &mut impl FnMut(RowImage<'_>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    match band {
        BandBitmap::Rgb8(band) => prepared8(band, y, args, invert, push),
        BandBitmap::Rgb16(band) => match tone::prepare_band16(band, invert, args) {
            None => push(RowImage::Rgb16(band)),
            Some(WidePage::Rgb(rgb)) => push(RowImage::Rgb16(&rgb)),
            Some(WidePage::Gray(luma)) => push(RowImage::Gray16(&luma)),
        },
        BandBitmap::Gray8(band) => match tone::prepare_gray_band(band, invert) {
            None => push(RowImage::Gray(band)),
            Some(gray) => push(RowImage::Gray(&gray)),
        },
        BandBitmap::Gray16(band) => match tone::prepare_gray_band(band, invert) {
            None => push(RowImage::Gray16(band)),
            Some(gray) => push(RowImage::Gray16(&gray)),
        },
        BandBitmap::Cmyk8(band) => push(RowImage::Cmyk(band)),
        BandBitmap::Cmyk16(band) => push(RowImage::Cmyk16(band)),
    }
}

/// [`prepared`] for an 8-bit band.
fn prepared8(
    band: &Bitmap<Rgb8>,
    y: u32,
    args: &Args,
    invert: bool,
    push: &mut impl FnMut(RowImage<'_>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    match tone::prepare_band(band, y, args, invert) {
        None => push(RowImage::Rgb(band)),
        Some(Page::Rgb(rgb)) => push(RowImage::Rgb(&rgb)),
        Some(Page::Gray(luma) | Page::Mono(luma)) => push(RowImage::Gray(&luma)),
    }
}

/// Render page `page_num` under `params`, a page of `size` pixels, in
/// bands, each written as it completes to its page file in `format`.
///
/// # Errors
///
/// The render, the writer or the output file failed; a page file left
/// incomplete is removed.
pub(crate) fn write_page_banded(
    session: &RasterSession,
    (page_num, total_pages): (u32, u32),
    args: &Args,
    (format, spill): (FileFormat, &SpillPolicy),
    params: &RenderParams,
    size: (u32, u32),
) -> Result<(), RenderError> {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "total_pages ≤ i32::MAX; validated in main before calling this function"
    )]
    let (number, total) = (page_num as i32, total_pages as i32);
    let path = output_path_with_prefix(
        spill.next_prefix(),
        args,
        number,
        total,
        args.output_format(),
        None,
    );
    write_output(&path, |out| {
        let model = (Model::of(args), args.depth());
        let mut writer = FileWriter::new(
            out,
            format,
            model,
            (size, args.dpi()),
            args.output_icc.as_ref(),
        )?;
        render_into(session, page_num, params, args, |band| writer.push(band))?;
        writer.finish()
    })
}

/// Render page `page_num` under `params`, a page of `size` pixels at
/// `dpi`, in bands, each compressed as it completes into the print job
/// `sink`'s page.
/// A page of more than one band holds its compressed image in a file of
/// its own until the spool takes it.
///
/// # Errors
///
/// The render or the encoder failed, or the page's file could not be
/// created.
pub(crate) fn print_page_banded(
    session: &RasterSession,
    page_num: u32,
    args: &Args,
    sink: &PageSink,
    params: &RenderParams,
    (size, dpi): ((u32, u32), (f64, f64)),
) -> Result<(), RenderError> {
    let model = Model::of(args);
    let page_bytes =
        u64::from(size.0) * u64::from(size.1) * u64::from(params.canvas_layout().bytes_per_px());
    let rgb = matches!(model, Model::Rgb { .. });
    let mut encoder = page_encoder(
        (size, dpi),
        (rgb, args.depth()),
        page_bytes > args.band_bytes(),
    )?;
    render_into(session, page_num, params, args, |band| {
        Ok(encoder.push(band)?)
    })?;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "page_num ≤ i32::MAX; validated in main before calling this function"
    )]
    sink.push(page_num as i32, None, encoder.finish()?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BandOutput, FileFormat};
    use crate::args::args_with;

    #[test]
    fn whole_page_stages_and_formats_render_pages_whole() {
        let file = |format| Some(BandOutput::File(format));
        assert_eq!(BandOutput::of(&args_with(&[])), file(FileFormat::Ppm));
        assert_eq!(
            BandOutput::of(&args_with(&["--png"])),
            file(FileFormat::Png)
        );
        assert!(matches!(
            BandOutput::of(&args_with(&["--tiff", "--mono"])),
            Some(BandOutput::File(FileFormat::Tiff(_)))
        ));
        assert!(BandOutput::of(&args_with(&["--invert=always"])).is_some());
        for argv in [
            &["--jpeg"][..],
            &["--invert=auto"],
            &["--crop"],
            &["--split-spreads"],
        ] {
            assert_eq!(BandOutput::of(&args_with(argv)), None, "{argv:?}");
        }
    }
}
