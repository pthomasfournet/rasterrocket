//! Pages written as they render, band by band: each band of rows is
//! brought to the output's colour model ([`tone::prepare_band`]) and handed
//! to a row-streamed writer, so memory holds a band, not the page.  A page
//! of one band takes the same writers, so the bytes of a file do not
//! depend on the band height.

use std::fs::File;
use std::io::BufWriter;

use color::Rgb8;
use encode::{
    PngColor, PngWriter, PnmKind, PnmWriter, RowImage, RowLayout, TiffCompression, TiffStream,
};
use pdf_raster::{BandError, RasterSession, RenderParams, render_page_bands};
use raster::Bitmap;

use crate::args::{Args, InvertWhen, OutputFormat};
use crate::cleanup::Page;
use crate::naming::output_path_with_prefix;
use crate::print::{PageSink, page_encoder};
use crate::ram::SpillPolicy;
use crate::render::{RenderError, write_output};
use crate::tone;

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
    /// the whole page (the camera stages, `--invert auto`), outside a fax
    /// document.  `None` renders pages whole.
    pub(crate) fn of(args: &Args) -> Option<Self> {
        let camera = args.camera_options();
        if args.faxing()
            || camera.crop
            || camera.rectify
            || camera.dewarp
            || camera.split_spreads
            || args.invert_when() == InvertWhen::Auto
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
}

impl Model {
    const fn of(args: &Args) -> Self {
        if args.mono {
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
    /// The writer of a `size` page in `model` to `out` in `format`, at
    /// `dpi` for the formats that record it.
    fn new(
        out: BufWriter<File>,
        format: FileFormat,
        model: Model,
        size: (u32, u32),
        dpi: (f64, f64),
    ) -> Result<Self, RenderError> {
        Ok(match format {
            FileFormat::Ppm => {
                let kind = match model {
                    Model::Rgb { .. } => PnmKind::Ppm,
                    Model::Gray => PnmKind::Pgm,
                    Model::Mono => PnmKind::Pbm,
                };
                Self::Pnm(PnmWriter::new(out, size, kind)?)
            }
            FileFormat::Png => {
                let color = match model {
                    Model::Rgb { alpha: false } => PngColor::Rgb,
                    Model::Rgb { alpha: true } => PngColor::Rgba,
                    Model::Gray | Model::Mono => PngColor::Gray,
                };
                Self::Png(PngWriter::new(out, size, color)?)
            }
            FileFormat::Tiff(compression) => {
                let layout = match model {
                    Model::Rgb { alpha: false } => RowLayout::Rgb,
                    Model::Rgb { alpha: true } => RowLayout::Rgba,
                    Model::Gray => RowLayout::Gray,
                    Model::Mono => RowLayout::Bilevel,
                };
                Self::Tiff(TiffStream::new(out, size, layout, dpi, compression)?)
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
/// colour model, `push`ed, then the page finished.
fn render_into(
    session: &RasterSession,
    page_num: u32,
    params: &RenderParams,
    args: &Args,
    mut push: impl FnMut(RowImage<'_>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    render_page_bands(session, page_num, params, |band| {
        prepared(band.bitmap, band.y, args, &mut push)
    })
    .map(drop)
    .map_err(|e| match e {
        BandError::Raster(e) => RenderError::Raster(e),
        BandError::Sink(e) => e,
    })
}

/// Bring `band`, rows from `y` of the page, to the output's colour model
/// and hand it to `push`.
fn prepared(
    band: &Bitmap<Rgb8>,
    y: u32,
    args: &Args,
    push: &mut impl FnMut(RowImage<'_>) -> Result<(), RenderError>,
) -> Result<(), RenderError> {
    match tone::prepare_band(band, y, args) {
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
        let mut writer = FileWriter::new(out, format, Model::of(args), size, args.dpi())?;
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
    let mut encoder = page_encoder((size, dpi), rgb, page_bytes > args.band_bytes())?;
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
