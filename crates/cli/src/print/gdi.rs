//! What rrocket does for a printer that takes page images through GDI,
//! where nothing fits them to the sheet: the orientation of each page, its
//! fit and place on the printable area, the resolution that makes the copy
//! to the device 1:1, the bands it is sent in and the rows of those bands;
//! and what is asked of the queue's driver and checked in its answer.
//!
//! A page whose aspect differs from the sheet's in the orientation the
//! driver gave is turned a quarter counter-clockwise, its top to the
//! sheet's left edge.  A page larger than the printable area is scaled to
//! fit it, keeping its aspect; a smaller one prints at its size; either is
//! centred on the printable area.

use std::ops::Range;

use super::Duplex;

/// The most bytes of bitmap one band of a page holds.
pub(crate) const MAX_BAND_BYTES: usize = 16 << 20;

/// `dmDuplex` of a two-sided sheet turning on its long edge
/// (`DMDUP_VERTICAL`).
pub(crate) const DMDUP_VERTICAL: i16 = 2;

/// `dmDuplex` of a two-sided sheet turning on its short edge
/// (`DMDUP_HORIZONTAL`).
pub(crate) const DMDUP_HORIZONTAL: i16 = 3;

/// The `dmDuplex` value of `duplex`.
pub(crate) const fn duplex_code(duplex: Duplex) -> i16 {
    match duplex {
        Duplex::Long => DMDUP_VERTICAL,
        Duplex::Short => DMDUP_HORIZONTAL,
    }
}

/// The orientation of a page or of a sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Orientation {
    /// Not wider than tall.
    Portrait,
    /// Wider than tall.
    Landscape,
}

impl Orientation {
    /// The orientation of a page of `width × height` points.
    pub(crate) fn of_page((width, height): (f64, f64)) -> Self {
        if width > height {
            Self::Landscape
        } else {
            Self::Portrait
        }
    }
}

/// The printable area of a sheet in one orientation, as the device
/// reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Surface {
    /// Device pixels per inch, horizontal and vertical (`LOGPIXELSX`,
    /// `LOGPIXELSY`).
    pub dpi: (u32, u32),
    /// The printable area's width and height in device pixels (`HORZRES`,
    /// `VERTRES`).
    pub printable: (u32, u32),
}

impl Surface {
    /// The printable area's width and height in inches.
    fn printable_in(&self) -> (f64, f64) {
        (
            f64::from(self.printable.0) / f64::from(self.dpi.0),
            f64::from(self.printable.1) / f64::from(self.dpi.1),
        )
    }

    /// The orientation of the printable area.
    fn orientation(&self) -> Orientation {
        let (w, h) = self.printable_in();
        if w > h {
            Orientation::Landscape
        } else {
            Orientation::Portrait
        }
    }

    /// Whether a page of `page_pts` is turned on this sheet, and the scale
    /// it prints at: 1, or less when it is larger than the printable area.
    fn fit(&self, page_pts: (f64, f64)) -> (bool, f64) {
        let rotate = Orientation::of_page(page_pts) != self.orientation();
        let (w, h) = if rotate {
            (page_pts.1, page_pts.0)
        } else {
            page_pts
        };
        let (area_w, area_h) = self.printable_in();
        let scale = (area_w / (w / 72.0)).min(area_h / (h / 72.0)).min(1.0);
        (rotate, scale)
    }
}

/// The printable areas of a printer's sheet in either orientation, as its
/// driver gives them.  A driver that cannot turn the sheet gives the same
/// area for both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Surfaces {
    pub portrait: Surface,
    pub landscape: Surface,
}

impl Surfaces {
    /// The sheet a page of `page_pts` prints on: in the page's orientation.
    pub(crate) const fn for_orientation(&self, orientation: Orientation) -> &Surface {
        match orientation {
            Orientation::Portrait => &self.portrait,
            Orientation::Landscape => &self.landscape,
        }
    }

    /// The resolution a page of `page_pts` renders at so that it copies
    /// 1:1 to the device: the device's lower resolution, at most `cap`,
    /// times the page's fit scale.
    pub(crate) fn render_dpi(&self, page_pts: (f64, f64), cap: f64) -> f64 {
        let surface = self.for_orientation(Orientation::of_page(page_pts));
        let (_, scale) = surface.fit(page_pts);
        f64::from(surface.dpi.0.min(surface.dpi.1)).min(cap) * scale
    }
}

/// Where a page image goes on the printable area, in device pixels from
/// its top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Placement {
    /// Whether the image is turned a quarter counter-clockwise first.
    pub rotate: bool,
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// The place on `surface` of a page of `page_pts`, an image of `image_px`
/// pixels: turned when its orientation is not the sheet's, fitted and
/// centred.  A size within a pixel of the image's, as rounding gives a
/// page rendered at [`Surfaces::render_dpi`], is the image's own, so the
/// copy is 1:1.
pub(crate) fn place(page_pts: (f64, f64), image_px: (u32, u32), surface: &Surface) -> Placement {
    let (rotate, scale) = surface.fit(page_pts);
    let ((w_pts, h_pts), (image_w, image_h)) = if rotate {
        ((page_pts.1, page_pts.0), (image_px.1, image_px.0))
    } else {
        (page_pts, image_px)
    };
    let device = |pts: f64, dpi: u32, image: u32, area: u32| {
        let px = (pts / 72.0 * scale * f64::from(dpi)).round();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative size, at most the printable area's, clamped below"
        )]
        let px = px.clamp(1.0, f64::from(area.max(1))) as u32;
        if px.abs_diff(image) <= 1 && image <= area {
            image
        } else {
            px
        }
    };
    let width = device(w_pts, surface.dpi.0, image_w, surface.printable.0);
    let height = device(h_pts, surface.dpi.1, image_h, surface.printable.1);
    let centre = |area: u32, size: u32| i32::try_from(area.saturating_sub(size) / 2).unwrap_or(0);
    Placement {
        rotate,
        x: centre(surface.printable.0, width),
        y: centre(surface.printable.1, height),
        width,
        height,
    }
}

/// The rows of each band of an image of `height` rows whose bitmap rows
/// are `stride` bytes: as many as [`MAX_BAND_BYTES`] holds, at least one.
pub(crate) fn band_rows(stride: usize, height: u32) -> u32 {
    u32::try_from(MAX_BAND_BYTES / stride.max(1))
        .unwrap_or(u32::MAX)
        .clamp(1, height.max(1))
}

/// The device rows `(y, height)` that the image rows `rows` of an image of
/// `image_height` rows cover in a placement of `dest_height` rows from
/// `dest_y`.  Consecutive bands abut, and at 1:1 each image row is one
/// device row.
pub(crate) fn band_dest(
    rows: &Range<u32>,
    image_height: u32,
    (dest_y, dest_height): (i32, u32),
) -> (i32, i32) {
    let edge = |row: u32| {
        let at = u64::from(row) * u64::from(dest_height) / u64::from(image_height.max(1));
        i32::try_from(at).unwrap_or(i32::MAX)
    };
    let (top, bottom) = (edge(rows.start), edge(rows.end));
    (dest_y.saturating_add(top), bottom - top)
}

/// The pixel format of a page's bitmap rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DibFormat {
    /// 24 bits a pixel, blue, green, red.
    Bgr24,
    /// 8 bits a pixel, an index into a 256-entry gray palette.
    Gray8,
    /// 1 bit a pixel, most significant first, an index into a palette of
    /// black (0) and white (1).
    Mono1,
}

impl DibFormat {
    /// The format of a page that is RGB when `rgb`, else gray, and
    /// bilevel when `mono`.
    pub(crate) const fn of(rgb: bool, mono: bool) -> Self {
        match (rgb, mono) {
            (true, _) => Self::Bgr24,
            (false, false) => Self::Gray8,
            (false, true) => Self::Mono1,
        }
    }

    /// The bits of a pixel.
    pub(crate) const fn bits(self) -> u16 {
        match self {
            Self::Bgr24 => 24,
            Self::Gray8 => 8,
            Self::Mono1 => 1,
        }
    }

    /// The bytes of a row of `width` pixels, padded to four.
    pub(crate) const fn stride(self, width: u32) -> usize {
        (width as usize * self.bits() as usize).div_ceil(32) * 4
    }
}

/// The bytes of a source pixel: `channels` samples of `bytes_per_sample`.
const fn source_bpp(rgb: bool, sixteen: bool) -> usize {
    (if rgb { 3 } else { 1 }) * (if sixteen { 2 } else { 1 })
}

/// A 16-bit sample narrowed to 8 bits by rounding.
fn narrow(hi: u8, lo: u8) -> u8 {
    let v = u32::from(u16::from_be_bytes([hi, lo]));
    u8::try_from((v * 255 + 32_767) / 65_535).unwrap_or(u8::MAX)
}

/// Write `src`, a row of pixels as [`encode::PdfPage::rows`] yields them
/// (RGB when `rgb`, else gray; 16-bit big-endian samples when `sixteen`),
/// into `out`, a bitmap row of `format` at least as long as its stride.
/// Bytes past the pixels are zero.
pub(crate) fn dib_row(src: &[u8], (rgb, sixteen): (bool, bool), format: DibFormat, out: &mut [u8]) {
    out.fill(0);
    let sample = |px: &[u8], i: usize| {
        if sixteen {
            narrow(px[2 * i], px[2 * i + 1])
        } else {
            px[i]
        }
    };
    let pixels = src.chunks_exact(source_bpp(rgb, sixteen));
    match format {
        DibFormat::Bgr24 => {
            for (px, bgr) in pixels.zip(out.as_chunks_mut::<3>().0) {
                bgr[0] = sample(px, 2);
                bgr[1] = sample(px, 1);
                bgr[2] = sample(px, 0);
            }
        }
        DibFormat::Gray8 => {
            for (px, gray) in pixels.zip(out.iter_mut()) {
                *gray = sample(px, 0);
            }
        }
        DibFormat::Mono1 => {
            for (x, px) in pixels.enumerate() {
                if sample(px, 0) >= encode::BLACK_BELOW {
                    out[x / 8] |= 0x80 >> (x % 8);
                }
            }
        }
    }
}

/// Write row `row` of `image`, `width × height` pixels of `bpp` bytes
/// turned a quarter counter-clockwise (`height` pixels a row, `width`
/// rows), into `out`.  The turned image's row `r` is the source's column
/// `width - 1 - r`, top to bottom.
pub(crate) fn turned_row(
    image: &[u8],
    (width, height): (u32, u32),
    bpp: usize,
    row: u32,
    out: &mut Vec<u8>,
) {
    out.clear();
    let (w, column) = (width as usize, (width - 1 - row) as usize);
    for y in 0..height as usize {
        let at = (y * w + column) * bpp;
        out.extend_from_slice(&image[at..at + bpp]);
    }
}

/// Whether a queue's port is a file: `PORTPROMPT:` and `FILE:`, which ask
/// for the file's name, or a path (`C:\…`, `\\?\…` aside, a drive letter, a
/// colon and a separator).
pub(crate) fn is_file_port(port: &str) -> bool {
    let port = port.trim();
    if port.eq_ignore_ascii_case("PORTPROMPT:") || port.eq_ignore_ascii_case("FILE:") {
        return true;
    }
    let bytes = port.as_bytes();
    bytes.len() > 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/')
}

/// Refuse a queue that writes to a file without `--print-file`, whose
/// driver would ask for the file's name in a dialog, and `--print-file`
/// for a queue that does not write to a file.
///
/// # Errors
///
/// The message naming the queue and its port.
pub(crate) fn check_print_file(printer: &str, port: &str, print_file: bool) -> Result<(), String> {
    match (is_file_port(port), print_file) {
        (true, false) => Err(format!(
            "{printer:?} prints to a file (port {port}) and would ask for its name in a \
             dialog; name the file with --print-file PATH"
        )),
        (false, true) => Err(format!(
            "--print-file names the file of a printer that prints to one; {printer:?} prints \
             to port {port}"
        )),
        _ => Ok(()),
    }
}

/// What a queue's driver says it can do (`DeviceCapabilitiesW`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Claims {
    /// Two-sided printing (`DC_DUPLEX`).
    pub duplex: bool,
    /// The most copies it makes of a job (`DC_COPIES`).
    pub max_copies: u32,
    /// Collated copies (`DC_COLLATE`).
    pub collate: bool,
}

/// How `--copies` is honoured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Copies {
    /// Not asked for: the queue's default.
    Default,
    /// The driver makes them, collated.
    Device(u32),
    /// rrocket sends the pages this many times in one job, the driver
    /// asked for one copy.
    Replay(u32),
}

impl Copies {
    /// The plan for `requested` copies on a queue that claims `claims`:
    /// the driver's when it makes that many, collated when more than one;
    /// else sent again.
    pub(crate) const fn plan(requested: Option<u32>, claims: Claims) -> Self {
        match requested {
            None => Self::Default,
            Some(n) if n <= 1 || (n <= claims.max_copies && claims.collate) => Self::Device(n),
            Some(n) => Self::Replay(n),
        }
    }

    /// The plan once the driver merged the request: `merged` is the copies
    /// and collation its answer holds, `None` for a field it dropped.  A
    /// driver that did not keep the copies asked for has them sent again.
    pub(crate) fn after_merge(self, merged: (Option<i16>, Option<bool>)) -> Self {
        match self {
            Self::Device(n) if n > 1 => {
                let kept = merged.0.and_then(|c| u32::try_from(c).ok()) == Some(n)
                    && merged.1 == Some(true);
                if kept { self } else { Self::Replay(n) }
            }
            plan => plan,
        }
    }

    /// The times the pages are sent.
    pub(crate) const fn passes(self) -> u32 {
        match self {
            Self::Replay(n) => n,
            Self::Default | Self::Device(_) => 1,
        }
    }
}

/// Refuse `--duplex` on a queue whose driver does not claim two-sided
/// printing.
///
/// # Errors
///
/// The message naming the queue.
pub(crate) fn check_duplex_claimed(
    printer: &str,
    requested: Option<Duplex>,
    claimed: bool,
) -> Result<(), String> {
    if requested.is_some() && !claimed {
        return Err(format!(
            "--duplex: {printer:?} does not print two-sided; drop --duplex to print one-sided"
        ));
    }
    Ok(())
}

/// Refuse `--duplex` when the driver's answer to the request dropped it:
/// `merged` is the answer's `dmDuplex`, `None` for a field it dropped.
///
/// # Errors
///
/// The message naming the queue.
pub(crate) fn check_duplex_merged(
    printer: &str,
    requested: Option<Duplex>,
    merged: Option<i16>,
) -> Result<(), String> {
    match requested {
        Some(duplex) if merged != Some(duplex_code(duplex)) => Err(format!(
            "--duplex: the driver of {printer:?} did not take two-sided printing; drop \
             --duplex to print one-sided"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Claims, Copies, DibFormat, Orientation, Placement, Surface, Surfaces, band_dest, band_rows,
        check_duplex_claimed, check_duplex_merged, check_print_file, dib_row, is_file_port, place,
        turned_row,
    };
    use crate::print::Duplex;

    /// A4 at 600 dpi with a 1/6 in margin (100 px) on every side, upright
    /// and turned.
    const A4_PORTRAIT: Surface = Surface {
        dpi: (600, 600),
        printable: (4760, 6815),
    };
    const A4_LANDSCAPE: Surface = Surface {
        dpi: (600, 600),
        printable: (6815, 4760),
    };
    const A4: Surfaces = Surfaces {
        portrait: A4_PORTRAIT,
        landscape: A4_LANDSCAPE,
    };

    #[test]
    fn a_small_page_prints_at_its_size_centred() {
        // A5 (420 × 595 pt) at 600 dpi: 3500 × 4958 px.
        let a5 = (420.0, 595.0);
        assert_eq!(A4.render_dpi(a5, 600.0), 600.0);
        let p = place(a5, (3500, 4958), &A4_PORTRAIT);
        assert_eq!(
            p,
            Placement {
                rotate: false,
                x: (4760 - 3500) / 2,
                y: (6815 - 4958) / 2,
                width: 3500,
                height: 4958,
            }
        );
    }

    #[test]
    fn a_large_page_is_fitted_keeping_its_aspect_and_copies_1_to_1() {
        // A3 (842 × 1191 pt) onto A4's printable area: the width binds.
        let a3 = (842.0, 1191.0);
        let dpi = A4.render_dpi(a3, 600.0);
        let scale = (4760.0 / (842.0 / 72.0 * 600.0_f64)).min(6815.0 / (1191.0 / 72.0 * 600.0));
        assert!((dpi / 600.0 - scale).abs() < 1e-12, "{dpi}");
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "page sides of a few thousand pixels"
        )]
        let px = |pts: f64| (pts / 72.0 * dpi).round() as u32;
        let image = (px(842.0), px(1191.0));
        let p = place(a3, image, &A4_PORTRAIT);
        assert!(!p.rotate);
        assert_eq!((p.width, p.height), image, "1:1");
        assert_eq!((p.x, p.width), (0, 4760));
        assert!(p.height <= 6815);
        assert_eq!(p.y, i32::try_from((6815 - p.height) / 2).unwrap());
    }

    #[test]
    fn a_landscape_page_takes_the_turned_sheet_or_is_turned_itself() {
        let landscape = (842.0, 595.0);
        assert_eq!(Orientation::of_page(landscape), Orientation::Landscape);
        assert_eq!(Orientation::of_page((595.0, 595.0)), Orientation::Portrait);
        let p = place(landscape, (7017, 4958), &A4_LANDSCAPE);
        assert!(!p.rotate);
        assert!(p.width <= 6815 && p.height <= 4760);
        // A driver that ignored the orientation: the page is turned onto
        // the upright sheet, and fitted the same.
        let ignored = Surfaces {
            portrait: A4_PORTRAIT,
            landscape: A4_PORTRAIT,
        };
        let dpi = ignored.render_dpi(landscape, 600.0);
        assert!((dpi - A4.render_dpi(landscape, 600.0)).abs() < 1e-9);
        let turned = place(landscape, (7017, 4958), &A4_PORTRAIT);
        assert!(turned.rotate);
        assert_eq!((turned.width, turned.height), (p.height, p.width));
    }

    #[test]
    fn the_resolution_is_the_lower_device_one_within_the_cap() {
        let uneven = Surface {
            dpi: (1200, 600),
            printable: (9600, 6000),
        };
        let both = Surfaces {
            portrait: uneven,
            landscape: uneven,
        };
        // 2 × 3 in fits a 8 × 10 in area.
        assert_eq!(both.render_dpi((144.0, 216.0), 600.0), 600.0);
        assert_eq!(both.render_dpi((144.0, 216.0), 360.0), 360.0);
        // Rendered at 600 dpi, the page stretches to 1200 dpi across.
        let p = place((144.0, 216.0), (1200, 1800), &uneven);
        assert_eq!((p.width, p.height), (2400, 1800));
    }

    #[test]
    fn bands_hold_at_most_16_mib_and_abut() {
        // 600 dpi A4 RGB: 4960 px a row.
        let stride = DibFormat::Bgr24.stride(4960);
        assert_eq!(stride, 14_880);
        let rows = band_rows(stride, 7016);
        assert_eq!(rows, u32::try_from((16 << 20) / 14_880).unwrap());
        assert!(rows as usize * stride <= 16 << 20);
        assert_eq!(band_rows(stride, 10), 10);
        assert_eq!(band_rows(usize::MAX, 10), 1);
        // 1:1: a band of rows 100..200 lands on device rows 100..200.
        assert_eq!(band_dest(&(100..200), 1000, (7, 1000)), (107, 100));
        // Doubled: consecutive bands cover the placement without gaps.
        let mut next = 7;
        for band in [0..3, 3..7, 7..10] {
            let (y, h) = band_dest(&band, 10, (7, 25));
            assert_eq!(y, next);
            next = y + h;
        }
        assert_eq!(next, 7 + 25);
    }

    #[test]
    fn bitmap_rows_are_bgr_gray_or_one_bit_padded_to_four_bytes() {
        assert_eq!(DibFormat::of(true, false), DibFormat::Bgr24);
        assert_eq!(DibFormat::of(false, false), DibFormat::Gray8);
        assert_eq!(DibFormat::of(false, true), DibFormat::Mono1);
        assert_eq!(DibFormat::Gray8.stride(5), 8);
        assert_eq!(DibFormat::Mono1.stride(33), 8);
        let mut out = [0xEE; 8];
        dib_row(
            &[1, 2, 3, 4, 5, 6],
            (true, false),
            DibFormat::Bgr24,
            &mut out,
        );
        assert_eq!(out, [3, 2, 1, 6, 5, 4, 0, 0]);
        // 16-bit samples round to 8 bits.
        let mut out = [0; 4];
        dib_row(
            &[0xFF, 0xFF, 0x80, 0x80, 0x00, 0x81, 0x00, 0x80],
            (false, true),
            DibFormat::Gray8,
            &mut out,
        );
        assert_eq!(out, [255, 128, 1, 0]);
        let mut out = [0; 4];
        dib_row(
            &[0, 255, 255, 0, 127, 128, 0, 0, 255],
            (false, false),
            DibFormat::Mono1,
            &mut out,
        );
        assert_eq!(out, [0b0110_0100, 0b1000_0000, 0, 0]);
    }

    #[test]
    fn a_turned_row_is_a_source_column_bottom_to_top_reversed() {
        // 3 × 2 gray: rows [1 2 3] and [4 5 6]; turned counter-clockwise
        // it is 2 × 3: [3 6], [2 5], [1 4].
        let image = [1, 2, 3, 4, 5, 6];
        let mut out = Vec::new();
        let rows: Vec<Vec<u8>> = (0..3)
            .map(|r| {
                turned_row(&image, (3, 2), 1, r, &mut out);
                out.clone()
            })
            .collect();
        assert_eq!(rows, [vec![3, 6], vec![2, 5], vec![1, 4]]);
    }

    #[test]
    fn file_ports_need_print_file_and_others_refuse_it() {
        for port in ["PORTPROMPT:", "FILE:", "file:", "C:\\out.prn", "d:/x.pdf"] {
            assert!(is_file_port(port), "{port}");
        }
        for port in [
            "LPT1:",
            "USB001",
            "SHRFAX:",
            "WSD-1234",
            "\\\\host\\queue",
            "nul:",
        ] {
            assert!(!is_file_port(port), "{port}");
        }
        let err = check_print_file("Microsoft Print to PDF", "PORTPROMPT:", false).unwrap_err();
        assert!(err.contains("--print-file PATH"), "{err}");
        let err = check_print_file("Office", "USB001", true).unwrap_err();
        assert!(err.contains("USB001"), "{err}");
        assert!(check_print_file("Office", "USB001", false).is_ok());
        assert!(check_print_file("Microsoft Print to PDF", "PORTPROMPT:", true).is_ok());
    }

    #[test]
    fn copies_beyond_the_driver_are_sent_again() {
        let claims = Claims {
            duplex: false,
            max_copies: 9,
            collate: true,
        };
        assert_eq!(Copies::plan(None, claims), Copies::Default);
        assert_eq!(Copies::plan(Some(3), claims), Copies::Device(3));
        assert_eq!(Copies::plan(Some(12), claims), Copies::Replay(12));
        let uncollated = Claims {
            collate: false,
            ..claims
        };
        assert_eq!(Copies::plan(Some(3), uncollated), Copies::Replay(3));
        assert_eq!(Copies::plan(Some(1), uncollated), Copies::Device(1));
        // The driver's answer decides.
        let device = Copies::Device(3);
        assert_eq!(device.after_merge((Some(3), Some(true))), device);
        assert_eq!(device.after_merge((Some(1), Some(true))), Copies::Replay(3));
        assert_eq!(device.after_merge((Some(3), None)), Copies::Replay(3));
        assert_eq!(
            Copies::Device(1).after_merge((None, None)),
            Copies::Device(1)
        );
        assert_eq!(Copies::Replay(4).passes(), 4);
        assert_eq!(Copies::Device(4).passes(), 1);
        assert_eq!(Copies::Default.passes(), 1);
    }

    #[test]
    fn duplex_is_refused_where_the_driver_cannot_or_did_not_take_it() {
        assert!(check_duplex_claimed("Lab", None, false).is_ok());
        let err = check_duplex_claimed("Lab", Some(Duplex::Long), false).unwrap_err();
        assert!(err.contains("\"Lab\" does not print two-sided"), "{err}");
        assert!(check_duplex_claimed("Lab", Some(Duplex::Long), true).is_ok());
        assert!(check_duplex_merged("Lab", None, None).is_ok());
        assert!(check_duplex_merged("Lab", Some(Duplex::Long), Some(2)).is_ok());
        assert!(check_duplex_merged("Lab", Some(Duplex::Short), Some(3)).is_ok());
        for merged in [None, Some(1), Some(2)] {
            let err = check_duplex_merged("Lab", Some(Duplex::Short), merged).unwrap_err();
            assert!(err.contains("did not take"), "{merged:?}: {err}");
        }
    }
}
