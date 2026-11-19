//! The Windows route: the pages go to the printer's driver through GDI.
//! The queue's DEVMODE, merged by its driver, carries the copies, two-sided
//! printing, colour and each page's orientation; rrocket fits and places
//! each page by the rules of [`super::gdi`] and draws it with
//! `StretchDIBits` in bands.  One thread, the one that appends pages to the
//! spool, owns the device context.
//!
//! The job is started when the printer is opened and ended once the last
//! page is drawn; a job dropped before then is aborted, and a file it was
//! writing is removed.

use std::ffi::OsStr;
use std::io::{self, Read as _, Write as _};
use std::os::windows::ffi::OsStrExt as _;
use std::path::PathBuf;
use std::ptr;

use color::SampleDepth;
use encode::PdfPage;
use windows_sys::Win32::Foundation::{
    ERROR_FILE_NOT_FOUND, ERROR_INSUFFICIENT_BUFFER, ERROR_SHARING_VIOLATION,
};
use windows_sys::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, COLORONCOLOR, CreateDCW, CreateICW, DEVMODEW,
    DEVMODEW_0_0, DIB_RGB_COLORS, DM_COLLATE, DM_COLOR, DM_COPIES, DM_DUPLEX, DM_FORMNAME,
    DM_IN_BUFFER, DM_ORIENTATION, DM_OUT_BUFFER, DM_PAPERLENGTH, DM_PAPERSIZE, DM_PAPERWIDTH,
    DMCOLLATE_TRUE, DMCOLOR_MONOCHROME, DeleteDC, GDI_ERROR, GetDeviceCaps, HALFTONE, HDC, HORZRES,
    IntersectClipRect, LOGPIXELSX, LOGPIXELSY, PHYSICALHEIGHT, PHYSICALOFFSETX, PHYSICALOFFSETY,
    PHYSICALWIDTH, RGBQUAD, ResetDCW, RestoreDC, SRCCOPY, SaveDC, SetBrushOrgEx, SetStretchBltMode,
    StretchDIBits, VERTRES,
};
use windows_sys::Win32::Graphics::Printing::{
    ClosePrinter, DocumentPropertiesW, EnumPrintersW, GetDefaultPrinterW, GetPrinterW,
    OpenPrinterW, PRINTER_ENUM_CONNECTIONS, PRINTER_ENUM_LOCAL, PRINTER_HANDLE, PRINTER_INFO_4W,
    PRINTER_INFO_5W,
};
use windows_sys::Win32::Storage::Xps::{
    AbortDoc, DC_COLLATE, DC_COPIES, DC_DUPLEX, DOCINFOW, DeviceCapabilitiesW, EndDoc, EndPage,
    StartDocW, StartPage,
};
use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler};
use windows_sys::core::BOOL;

use super::gdi::{self, Claims, Copies, DibFormat, Orientation, Placement, Surface, Surfaces};
use super::{Duplex, PageOut, Spool};
use crate::args::Args;
use crate::render::RenderError;

/// `dmOrientation` of an upright sheet (`DMORIENT_PORTRAIT`).
const DMORIENT_PORTRAIT: i16 = 1;
/// `dmOrientation` of a turned sheet (`DMORIENT_LANDSCAPE`).
const DMORIENT_LANDSCAPE: i16 = 2;
/// `dmPaperSize` of ISO A4 (`DMPAPER_A4`).
#[cfg(test)]
const DMPAPER_A4: i16 = 9;
/// `dmPaperSize` of US Letter (`DMPAPER_LETTER`).
#[cfg(test)]
const DMPAPER_LETTER: i16 = 1;

/// `s` as a NUL-terminated UTF-16 string.
fn wide(s: impl AsRef<OsStr>) -> Vec<u16> {
    s.as_ref().encode_wide().chain(Some(0)).collect()
}

/// The UTF-16 units a NUL-terminated pointer names, none for null.
///
/// # Safety
///
/// `p` is null or points at a NUL-terminated UTF-16 string that outlives
/// `'a`.
#[expect(
    unsafe_code,
    reason = "the spooler returns its strings as NUL-terminated pointers"
)]
const unsafe fn wide_units<'a>(p: *const u16) -> &'a [u16] {
    if p.is_null() {
        return &[];
    }
    let mut len = 0;
    // SAFETY: the string is NUL-terminated, so every unit up to the NUL is
    // readable.
    while unsafe { *p.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the `len` units before the NUL are readable for `'a`.
    unsafe { std::slice::from_raw_parts(p, len) }
}

/// A printer's name from its UTF-16 `units`.  A name that is not valid
/// UTF-16 is decoded with U+FFFD for its unpaired surrogates and logged:
/// `--printer` takes Unicode text, so it cannot name that printer.
fn printer_name(units: &[u16]) -> String {
    String::from_utf16(units).unwrap_or_else(|_| {
        let lossy = String::from_utf16_lossy(units);
        log::warn!(
            "the printer name {lossy:?} is not valid Unicode; --printer cannot name it, \
             rename the printer to print to it"
        );
        lossy
    })
}

/// Whether `error` is the system error `code`.
fn last_error_is(error: &io::Error, code: u32) -> bool {
    error.raw_os_error() == i32::try_from(code).ok()
}

/// The buffer a sized query fills: `call(buffer, bytes, needed)` is made
/// with no buffer, then with one of the bytes it reports needing, again
/// while the need grows between calls.  The buffer is 8-byte aligned.
fn sized_query(mut call: impl FnMut(*mut u8, u32, &mut u32) -> bool) -> io::Result<Vec<u64>> {
    let mut buf: Vec<u64> = Vec::new();
    for _ in 0..8 {
        let bytes = u32::try_from(buf.len() * 8).unwrap_or(u32::MAX);
        let mut needed = 0;
        if call(buf.as_mut_ptr().cast(), bytes, &mut needed) {
            return Ok(buf);
        }
        let error = io::Error::last_os_error();
        if !last_error_is(&error, ERROR_INSUFFICIENT_BUFFER) || needed <= bytes {
            return Err(error);
        }
        buf = vec![0; (needed as usize).div_ceil(8)];
    }
    Err(io::Error::other("the answer kept growing"))
}

/// The names of the local and connected printers.
#[expect(
    unsafe_code,
    reason = "EnumPrintersW is the interface to the spooler's printer list"
)]
fn printer_names() -> io::Result<Vec<String>> {
    let mut count = 0;
    let buf = sized_query(|buf, bytes, needed| {
        // SAFETY: `buf` holds `bytes` writable bytes (none when null), and
        // `needed` and `count` are writable.
        unsafe {
            EnumPrintersW(
                PRINTER_ENUM_LOCAL | PRINTER_ENUM_CONNECTIONS,
                ptr::null(),
                4,
                buf,
                bytes,
                needed,
                &raw mut count,
            ) != 0
        }
    })?;
    if count == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: the call wrote `count` `PRINTER_INFO_4W` at the start of the
    // 8-byte-aligned `buf`, whose strings point into `buf`.
    let infos = unsafe {
        std::slice::from_raw_parts(buf.as_ptr().cast::<PRINTER_INFO_4W>(), count as usize)
    };
    // SAFETY: each name is a NUL-terminated string within `buf`.
    Ok(infos
        .iter()
        .map(|info| printer_name(unsafe { wide_units(info.pPrinterName) }))
        .collect())
}

/// The user's default printer, if one is set.
#[expect(
    unsafe_code,
    reason = "GetDefaultPrinterW is the interface to the user's default printer"
)]
fn default_printer() -> io::Result<Option<String>> {
    let mut len = 0;
    // SAFETY: a null buffer asks for the length needed, written to `len`.
    if unsafe { GetDefaultPrinterW(ptr::null_mut(), &raw mut len) } != 0 {
        return Ok(None);
    }
    let error = io::Error::last_os_error();
    if last_error_is(&error, ERROR_FILE_NOT_FOUND) {
        return Ok(None);
    }
    if !last_error_is(&error, ERROR_INSUFFICIENT_BUFFER) {
        return Err(error);
    }
    let mut name = vec![0u16; len as usize];
    // SAFETY: `name` holds the `len` units asked for.
    if unsafe { GetDefaultPrinterW(name.as_mut_ptr(), &raw mut len) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let end = name.iter().position(|&u| u == 0).unwrap_or(name.len());
    Ok(Some(printer_name(&name[..end])))
}

/// What `--list-printers` writes on Windows: the local and connected
/// printers, the default one marked.
///
/// # Errors
///
/// The spooler could not be asked.
pub(crate) fn list_printers() -> Result<String, String> {
    let names = printer_names().map_err(|e| spooler_error("cannot list the printers", &e))?;
    let default =
        default_printer().map_err(|e| spooler_error("cannot ask for the default printer", &e))?;
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    Ok(super::printer_list(&names, default.as_deref()))
}

/// `what` failed with `e`, a pointer to the Print Spooler service added.
fn spooler_error(what: &str, e: &io::Error) -> String {
    format!("{what}: {e}; the Print Spooler service must be running")
}

/// A DEVMODE in a buffer of its own: the public fields, then the driver's
/// private bytes.  The buffer is 8-byte aligned and at least a `DEVMODEW`
/// long, so the public fields can be read and written in place whatever
/// `dmSize` the driver uses.
#[derive(Clone)]
struct DevMode {
    buf: Vec<u64>,
}

impl DevMode {
    /// A zeroed DEVMODE of `bytes`.
    fn zeroed(bytes: usize) -> Self {
        Self {
            buf: vec![0; bytes.max(size_of::<DEVMODEW>()).div_ceil(8)],
        }
    }

    const fn as_ptr(&self) -> *const DEVMODEW {
        self.buf.as_ptr().cast()
    }

    const fn as_mut_ptr(&mut self) -> *mut DEVMODEW {
        self.buf.as_mut_ptr().cast()
    }

    #[expect(
        unsafe_code,
        reason = "the DEVMODE is read in place in the buffer the driver filled"
    )]
    const fn fields(&self) -> &DEVMODEW {
        // SAFETY: the buffer is aligned for and at least as long as a
        // `DEVMODEW`, every bit pattern of which is valid.
        unsafe { &*self.as_ptr() }
    }

    #[expect(
        unsafe_code,
        reason = "the DEVMODE is written in place in the buffer the driver reads"
    )]
    fn fields_mut(&mut self) -> &mut DEVMODEW {
        // SAFETY: as in `fields`, and the borrow of `self` is exclusive.
        unsafe { &mut *self.as_mut_ptr() }
    }

    #[expect(
        unsafe_code,
        reason = "the printer fields share a union with the display fields"
    )]
    const fn printer_fields(&self) -> &DEVMODEW_0_0 {
        // SAFETY: both members of the union are plain integers, valid for
        // any bits.
        unsafe { &self.fields().Anonymous1.Anonymous1 }
    }

    #[expect(
        unsafe_code,
        reason = "the printer fields share a union with the display fields"
    )]
    fn printer_fields_mut(&mut self) -> &mut DEVMODEW_0_0 {
        // SAFETY: as in `printer_fields`.
        unsafe { &mut self.fields_mut().Anonymous1.Anonymous1 }
    }

    /// The field of `flag`, when the DEVMODE holds it.
    fn field<T>(&self, flag: u32, value: T) -> Option<T> {
        (self.fields().dmFields & flag != 0).then_some(value)
    }

    fn set_orientation(&mut self, orientation: Orientation) {
        self.printer_fields_mut().dmOrientation = match orientation {
            Orientation::Portrait => DMORIENT_PORTRAIT,
            Orientation::Landscape => DMORIENT_LANDSCAPE,
        };
        self.fields_mut().dmFields |= DM_ORIENTATION;
    }

    /// Ask for the job `settings` describes, its copies made as `copies`
    /// says, on an upright sheet.
    fn ask(&mut self, settings: &Settings, copies: Copies) {
        self.set_orientation(Orientation::Portrait);
        if let Some(paper) = settings.paper {
            self.printer_fields_mut().dmPaperSize = paper;
            let dm = self.fields_mut();
            dm.dmFields =
                (dm.dmFields | DM_PAPERSIZE) & !(DM_PAPERLENGTH | DM_PAPERWIDTH | DM_FORMNAME);
        }
        let device = match copies {
            Copies::Default => None,
            Copies::Device(n) => Some(n),
            Copies::Replay(_) => Some(1),
        };
        if let Some(n) = device {
            self.printer_fields_mut().dmCopies = i16::try_from(n).unwrap_or(i16::MAX);
            let dm = self.fields_mut();
            dm.dmFields |= DM_COPIES;
            if n > 1 {
                dm.dmCollate = DMCOLLATE_TRUE;
                dm.dmFields |= DM_COLLATE;
            }
        }
        let dm = self.fields_mut();
        if let Some(duplex) = settings.duplex {
            dm.dmDuplex = gdi::duplex_code(duplex);
            dm.dmFields |= DM_DUPLEX;
        }
        if settings.gray {
            dm.dmColor = DMCOLOR_MONOCHROME;
            dm.dmFields |= DM_COLOR;
        }
    }

    /// The copies and collation the DEVMODE holds.
    fn copies(&self) -> (Option<i16>, Option<bool>) {
        (
            self.field(DM_COPIES, self.printer_fields().dmCopies),
            self.field(DM_COLLATE, self.fields().dmCollate == DMCOLLATE_TRUE),
        )
    }

    fn duplex(&self) -> Option<i16> {
        self.field(DM_DUPLEX, self.fields().dmDuplex)
    }

    fn paper(&self) -> Option<i16> {
        self.field(DM_PAPERSIZE, self.printer_fields().dmPaperSize)
    }
}

/// An open printer.
struct Printer {
    handle: PRINTER_HANDLE,
    /// The name, NUL-terminated.
    name_w: Vec<u16>,
    name: String,
}

impl Drop for Printer {
    #[expect(unsafe_code, reason = "the printer handle is closed once")]
    fn drop(&mut self) {
        // SAFETY: the handle was opened by `OpenPrinterW` and is closed
        // only here.
        let _ = unsafe { ClosePrinter(self.handle) };
    }
}

impl Printer {
    /// Open the printer `name`.
    #[expect(unsafe_code, reason = "OpenPrinterW opens the printer")]
    fn open(name: &str) -> Result<Self, String> {
        let name_w = wide(name);
        let mut handle = PRINTER_HANDLE::default();
        // SAFETY: `name_w` is NUL-terminated and `handle` writable; no
        // defaults are passed.
        if unsafe { OpenPrinterW(name_w.as_ptr(), &raw mut handle, ptr::null()) } == 0 {
            return Err(format!(
                "cannot open the printer {name:?}: {}; --list-printers lists the printers",
                io::Error::last_os_error()
            ));
        }
        Ok(Self {
            handle,
            name_w,
            name: name.to_owned(),
        })
    }

    /// The printer's port (`PRINTER_INFO_5W.pPortName`).
    #[expect(unsafe_code, reason = "GetPrinterW reports the printer's port")]
    fn port(&self) -> io::Result<String> {
        let buf = sized_query(|buf, bytes, needed| {
            // SAFETY: `buf` holds `bytes` writable bytes, `needed` is
            // writable.
            unsafe { GetPrinterW(self.handle, 5, buf, bytes, needed) != 0 }
        })?;
        if buf.len() * 8 < size_of::<PRINTER_INFO_5W>() {
            return Err(io::Error::other("the spooler gave no printer information"));
        }
        // SAFETY: the call wrote a `PRINTER_INFO_5W` at the start of the
        // aligned `buf`, whose port name points into `buf`.
        Ok(String::from_utf16_lossy(unsafe {
            wide_units((*buf.as_ptr().cast::<PRINTER_INFO_5W>()).pPortName)
        }))
    }

    /// The driver's answer to `capability` for port `port_w`
    /// (NUL-terminated), -1 when it has none.
    #[expect(unsafe_code, reason = "DeviceCapabilitiesW asks the driver")]
    fn capability(&self, port_w: &[u16], capability: u16) -> i32 {
        // SAFETY: both strings are NUL-terminated; the asked capabilities
        // write no output.
        unsafe {
            DeviceCapabilitiesW(
                self.name_w.as_ptr(),
                port_w.as_ptr(),
                capability,
                ptr::null_mut(),
                ptr::null(),
            )
        }
    }

    /// What the driver says it can do.
    fn claims(&self, port: &str) -> Claims {
        let port_w = wide(port);
        Claims {
            duplex: self.capability(&port_w, DC_DUPLEX) == 1,
            max_copies: u32::try_from(self.capability(&port_w, DC_COPIES))
                .unwrap_or(1)
                .max(1),
            collate: self.capability(&port_w, DC_COLLATE) == 1,
        }
    }

    /// The bytes of the driver's DEVMODE.
    #[expect(unsafe_code, reason = "DocumentPropertiesW sizes the driver's DEVMODE")]
    fn devmode_bytes(&self) -> io::Result<usize> {
        // SAFETY: no buffers: the call returns the size.
        let bytes = unsafe {
            DocumentPropertiesW(
                ptr::null_mut(),
                self.handle,
                self.name_w.as_ptr(),
                ptr::null_mut(),
                ptr::null(),
                0,
            )
        };
        usize::try_from(bytes)
            .ok()
            .filter(|&b| b > 0)
            .ok_or_else(io::Error::last_os_error)
    }

    /// The driver's DEVMODE: the queue's defaults merged with `input`'s
    /// fields when given.
    #[expect(unsafe_code, reason = "DocumentPropertiesW fills the driver's DEVMODE")]
    fn devmode(&self, input: Option<&DevMode>) -> Result<DevMode, String> {
        let error = |e: io::Error| format!("the driver of {:?} gave no settings: {e}", self.name);
        let mut out = DevMode::zeroed(self.devmode_bytes().map_err(error)?);
        let mode = if input.is_some() {
            DM_IN_BUFFER | DM_OUT_BUFFER
        } else {
            DM_OUT_BUFFER
        };
        // SAFETY: `out` holds the bytes the driver asked for, `input` is a
        // DEVMODE the driver filled; no window and no prompt.
        let ok = unsafe {
            DocumentPropertiesW(
                ptr::null_mut(),
                self.handle,
                self.name_w.as_ptr(),
                out.as_mut_ptr(),
                input.map_or(ptr::null(), DevMode::as_ptr),
                mode,
            )
        };
        if ok < 0 {
            return Err(error(io::Error::last_os_error()));
        }
        Ok(out)
    }
}

/// The printable area `dc` reports.
#[expect(unsafe_code, reason = "GetDeviceCaps reads the device's page geometry")]
fn surface(dc: HDC, name: &str) -> Result<Surface, String> {
    let cap = |index: u32| {
        // SAFETY: `dc` is a live device context.
        let v = unsafe { GetDeviceCaps(dc, i32::try_from(index).unwrap_or(0)) };
        u32::try_from(v).unwrap_or(0)
    };
    let surface = Surface {
        dpi: (cap(LOGPIXELSX), cap(LOGPIXELSY)),
        printable: (cap(HORZRES), cap(VERTRES)),
        sheet: (cap(PHYSICALWIDTH), cap(PHYSICALHEIGHT)),
        offset: (cap(PHYSICALOFFSETX), cap(PHYSICALOFFSETY)),
    };
    log::debug!("{name:?}: {surface:?}");
    if surface.dpi.0 == 0
        || surface.dpi.1 == 0
        || surface.printable.0 == 0
        || surface.printable.1 == 0
    {
        return Err(format!("{name:?} reports no printable area: {surface:?}"));
    }
    Ok(surface)
}

/// The printable area of `printer` under `devmode`, read from an
/// information context.
fn info_surface(printer: &Printer, devmode: &DevMode) -> Result<Surface, String> {
    let ic = Dc::information(printer, devmode)?;
    surface(ic.raw(), &printer.name)
}

/// The signature of `CreateDCW` and `CreateICW`.
type CreateContext =
    unsafe extern "system" fn(*const u16, *const u16, *const u16, *const DEVMODEW) -> HDC;

/// A device or information context on a printer, deleted when dropped.
struct Dc {
    raw: HDC,
}

impl Drop for Dc {
    #[expect(unsafe_code, reason = "the context is deleted once")]
    fn drop(&mut self) {
        // SAFETY: `raw` was created by `CreateDCW` or `CreateICW` and is
        // deleted only here.
        let _ = unsafe { DeleteDC(self.raw) };
    }
}

impl Dc {
    /// A device context on `printer` under `devmode`, to print on.
    fn device(printer: &Printer, devmode: &DevMode) -> Result<Self, String> {
        Self::create(printer, devmode, CreateDCW, "open a device context on")
    }

    /// An information context on `printer` under `devmode`, to ask for
    /// its sheet.
    fn information(printer: &Printer, devmode: &DevMode) -> Result<Self, String> {
        Self::create(printer, devmode, CreateICW, "ask for the sheet of")
    }

    /// The context `create` makes on `printer` under `devmode`; the error
    /// names what could not be done (`what`).
    #[expect(unsafe_code, reason = "CreateDCW and CreateICW open the context")]
    fn create(
        printer: &Printer,
        devmode: &DevMode,
        create: CreateContext,
        what: &str,
    ) -> Result<Self, String> {
        let driver = wide("WINSPOOL");
        // SAFETY: the strings are NUL-terminated and the DEVMODE the
        // driver's.
        let raw = unsafe {
            create(
                driver.as_ptr(),
                printer.name_w.as_ptr(),
                ptr::null(),
                devmode.as_ptr(),
            )
        };
        if raw.is_null() {
            return Err(format!(
                "cannot {what} {:?}: {}",
                printer.name,
                io::Error::last_os_error()
            ));
        }
        Ok(Self { raw })
    }

    const fn raw(&self) -> HDC {
        self.raw
    }
}

/// Refuse a merged DEVMODE of the driver of `printer` that dropped the
/// two-sided printing or the paper `settings` asks for.
///
/// # Errors
///
/// The message naming the printer and the setting dropped.
fn check_merged(printer: &str, settings: &Settings, merged: &DevMode) -> Result<(), String> {
    gdi::check_duplex_merged(printer, settings.duplex, merged.duplex())?;
    match settings.paper {
        Some(paper) if merged.paper() != Some(paper) => Err(format!(
            "the driver of {printer:?} did not take paper {paper}: it holds {:?}",
            merged.paper()
        )),
        _ => Ok(()),
    }
}

/// The driver's DEVMODEs for the job `settings` describes on an upright
/// and on a turned sheet, and how its copies are made, once each of the
/// driver's answers is checked ([`check_merged`]): copies either answer did
/// not keep are sent again, the request then asking the driver for one.
fn settle(
    printer: &Printer,
    settings: &Settings,
    claims: Claims,
) -> Result<(DevMode, DevMode, Copies), String> {
    let name = &printer.name;
    let mut copies = Copies::plan(settings.copies, claims);
    let mut request = printer.devmode(None)?;
    loop {
        request.ask(settings, copies);
        let portrait = printer.devmode(Some(&request))?;
        let mut turned = portrait.clone();
        turned.set_orientation(Orientation::Landscape);
        let landscape = printer.devmode(Some(&turned))?;
        check_merged(name, settings, &portrait)?;
        check_merged(name, settings, &landscape)?;
        let kept = copies
            .after_merge(portrait.copies())
            .after_merge(landscape.copies());
        if kept == copies {
            if copies.passes() > 1 {
                log::info!(
                    "{name:?} makes at most {} copies, collated: {}; the pages are sent {} \
                     times",
                    claims.max_copies,
                    claims.collate,
                    copies.passes()
                );
            }
            return Ok((portrait, landscape, copies));
        }
        copies = kept;
    }
}

/// What a Windows print job is asked for.
pub(crate) struct Settings {
    /// The printer, else the default one.
    pub printer: Option<String>,
    pub copies: Option<u32>,
    pub duplex: Option<Duplex>,
    /// Whether the pages are gray, and so printed in black alone.
    pub gray: bool,
    /// Whether the pages are bilevel.
    pub mono: bool,
    pub title: String,
    /// The file a printer that prints to a file writes.
    pub print_file: Option<PathBuf>,
    /// The `dmPaperSize` asked for, else the queue's default.  The CLI
    /// has no paper flag and asks for none; the tests that print on A4
    /// and on Letter set it.
    pub paper: Option<i16>,
}

impl Settings {
    /// The job `args` asks for, on paper `paper` when given.
    fn of(args: &Args, paper: Option<i16>) -> Self {
        Self {
            printer: args.printer.clone(),
            copies: args.copies,
            duplex: args.duplex,
            gray: args.mono || args.luma_output(),
            mono: args.mono,
            title: super::job_title(args.input()),
            print_file: args.print_file.clone(),
            paper,
        }
    }
}

/// Open the printer `args` names and begin its job, on paper `paper` when
/// given, else the queue's; record the printer's printable areas in
/// `args`.
///
/// # Errors
///
/// The printer could not be opened, cannot print the job as asked, or the
/// job could not be started.
pub(crate) fn begin(args: &mut Args, paper: Option<i16>) -> Result<Spool<Job>, String> {
    catch_interrupts();
    let job = Job::open(&Settings::of(args, paper))?;
    args.print_surfaces = Some(job.surfaces);
    Ok(Spool::new(job))
}

/// Install [`on_console_event`] as a console control handler of the
/// process.  A failure is logged: the job then prints, and an interrupted
/// run leaves it to the spooler.
#[expect(
    unsafe_code,
    reason = "SetConsoleCtrlHandler installs the console control handler"
)]
fn catch_interrupts() {
    // SAFETY: the handler is a function that lives as long as the process
    // and touches only an atomic flag and standard error.
    if unsafe { SetConsoleCtrlHandler(Some(on_console_event), 1) } == 0 {
        log::warn!(
            "cannot catch Ctrl-C: {}; an interrupted run leaves its print job to the spooler",
            io::Error::last_os_error()
        );
    }
}

/// The console control handler of a print run.  The first Ctrl-C or
/// Ctrl-Break, the console's closing, the user's logging off and the
/// system's shutting down mark the run interrupted ([`super::interrupt`]):
/// no further page is begun, and `main` returns through the job's `Drop`,
/// which aborts it and removes the file it was writing.  A later Ctrl-C or
/// Ctrl-Break goes on to the default handler, which ends the process at
/// once.  On the other events the system ends the process when the handler
/// returns, so it does not return: the process ends when `main` does or
/// when the system's grace period runs out.
extern "system" fn on_console_event(event: u32) -> BOOL {
    let first = super::interrupt();
    match event {
        CTRL_C_EVENT | CTRL_BREAK_EVENT => {
            if first {
                let _ = writeln!(
                    io::stderr(),
                    "rrocket: interrupted: aborting the print job; interrupt again to quit at once"
                );
            }
            BOOL::from(first)
        }
        _ => loop {
            std::thread::park();
        },
    }
}

/// A print job being drawn.  Its fields drop in their order: the device
/// context, then the printer, then the file of an aborted job.
pub(crate) struct Job {
    dc: Dc,
    printer: Printer,
    /// The file the job writes, removed when the job is dropped started
    /// and not ended.
    output: Incomplete,
    /// The merged DEVMODEs of an upright and a turned sheet.
    devmodes: (DevMode, DevMode),
    surfaces: Surfaces,
    /// The orientation the device context is in, and its printable area.
    orientation: Orientation,
    surface: Surface,
    copies: Copies,
    mono: bool,
    /// Whether the sheets are printed on both sides.
    duplex: bool,
    pages: u32,
    /// The pages of the current pass.
    pass_pages: u32,
    phase: Phase,
}

/// The file a job writes, removed ([`remove_incomplete`]) when dropped
/// holding one.
struct Incomplete(Option<PathBuf>);

impl Drop for Incomplete {
    fn drop(&mut self) {
        if let Some(file) = &self.0 {
            remove_incomplete(file);
        }
    }
}

/// How far a [`Job`] has gone with the spooler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// `StartDocW` has not been called.
    Opened,
    /// The job is started and not ended: dropping it aborts it.
    Started,
    /// `EndDoc` handed the job to the spooler.
    Ended,
}

impl Drop for Job {
    /// Abort a job started and not ended; its file is removed once the
    /// device context and the printer are closed.  The file of a job not
    /// started is not the job's, and is left.
    #[expect(unsafe_code, reason = "the job is aborted once")]
    fn drop(&mut self) {
        if self.phase == Phase::Started {
            // SAFETY: the job was started on `dc` and not ended.
            let _ = unsafe { AbortDoc(self.dc.raw()) };
        } else {
            self.output.0 = None;
        }
    }
}

/// How long [`remove_incomplete`] waits for the spooler to let go of an
/// aborted job's file.
const RELEASE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// Remove `file`, the output of an aborted job.  The spooler closes the
/// file some time after the job is aborted; until then removing it fails
/// with a sharing violation, and it is retried for up to [`RELEASE_WAIT`].
/// A file that is still held, or cannot be removed, is logged.
fn remove_incomplete(file: &std::path::Path) {
    let deadline = std::time::Instant::now() + RELEASE_WAIT;
    loop {
        match std::fs::remove_file(file) {
            Ok(()) => return,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return,
            Err(e)
                if last_error_is(&e, ERROR_SHARING_VIOLATION)
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                log::warn!("could not remove the incomplete {}: {e}", file.display());
                return;
            }
        }
    }
}

impl Job {
    /// Open the printer and start the job `settings` describes.
    fn open(settings: &Settings) -> Result<Self, String> {
        let name = match &settings.printer {
            Some(name) => name.clone(),
            None => default_printer()
                .map_err(|e| spooler_error("cannot ask for the default printer", &e))?
                .ok_or(
                    "no default printer is set; name one with --printer (--list-printers lists \
                     them)",
                )?,
        };
        let printer = Printer::open(&name)?;
        let port = printer
            .port()
            .map_err(|e| spooler_error(&format!("cannot ask {name:?} for its port"), &e))?;
        gdi::check_print_file(&name, &port, settings.print_file.is_some())?;
        let claims = printer.claims(&port);
        log::debug!("{name:?} on port {port}: {claims:?}");
        gdi::check_duplex_claimed(&name, settings.duplex, claims.duplex)?;
        let (portrait, landscape, copies) = settle(&printer, settings, claims)?;
        let landscape_surface = info_surface(&printer, &landscape)?;
        let dc = Dc::device(&printer, &portrait)?;
        let portrait_surface = surface(dc.raw(), &name)?;
        let mut job = Self {
            dc,
            printer,
            output: Incomplete(None),
            devmodes: (portrait, landscape),
            surfaces: Surfaces {
                portrait: portrait_surface,
                landscape: landscape_surface,
            },
            orientation: Orientation::Portrait,
            surface: portrait_surface,
            copies,
            mono: settings.mono,
            duplex: settings.duplex.is_some(),
            pages: 0,
            pass_pages: 0,
            phase: Phase::Opened,
        };
        log::debug!("{name:?}: {:?}, copies {copies:?}", job.surfaces);
        job.start(settings)?;
        Ok(job)
    }

    /// Start the job, named `settings.title`, writing `settings.print_file`
    /// when given.
    #[expect(unsafe_code, reason = "StartDocW starts the GDI print job")]
    fn start(&mut self, settings: &Settings) -> Result<(), String> {
        self.output.0 = settings
            .print_file
            .as_deref()
            .map(std::path::absolute)
            .transpose()
            .map_err(|e| format!("--print-file: {e}"))?;
        let title = wide(&settings.title);
        let output_w = self.output.0.as_ref().map(wide);
        let info = DOCINFOW {
            cbSize: i32::try_from(size_of::<DOCINFOW>()).unwrap_or(i32::MAX),
            lpszDocName: title.as_ptr(),
            lpszOutput: output_w.as_ref().map_or(ptr::null(), Vec::as_ptr),
            lpszDatatype: ptr::null(),
            fwType: 0,
        };
        // SAFETY: `info` and the strings it points at outlive the call.
        if unsafe { StartDocW(self.dc.raw(), &raw const info) } <= 0 {
            return Err(format!(
                "{:?} did not start the job: {}",
                self.printer.name,
                io::Error::last_os_error()
            ));
        }
        self.phase = Phase::Started;
        Ok(())
    }

    /// End the job.
    ///
    /// # Errors
    ///
    /// No page was printed, or the spooler did not take the job.
    #[expect(unsafe_code, reason = "EndDoc ends the GDI print job")]
    pub(crate) fn finish(mut self) -> Result<(), String> {
        if self.pages == 0 {
            return Err("no page was printed".to_owned());
        }
        // SAFETY: the job was started on `dc`, and no page is open.
        if unsafe { EndDoc(self.dc.raw()) } <= 0 {
            return Err(format!(
                "{:?} did not take the job: {}",
                self.printer.name,
                io::Error::last_os_error()
            ));
        }
        self.phase = Phase::Ended;
        Ok(())
    }

    /// Turn the device context to `orientation` (between pages).
    #[expect(unsafe_code, reason = "ResetDCW turns the sheet between pages")]
    fn turn(&mut self, orientation: Orientation) -> Result<(), String> {
        let devmode = match orientation {
            Orientation::Portrait => &self.devmodes.0,
            Orientation::Landscape => &self.devmodes.1,
        };
        // SAFETY: `dc` is live with no page open; the DEVMODE is the
        // driver's.
        if unsafe { ResetDCW(self.dc.raw(), devmode.as_ptr()) }.is_null() {
            return Err(format!(
                "cannot turn the sheet to {orientation:?}: {}",
                io::Error::last_os_error()
            ));
        }
        self.surface = surface(self.dc.raw(), &self.printer.name)?;
        self.orientation = orientation;
        Ok(())
    }

    /// Print `page` on a side of its own.
    fn print(&mut self, page: &PdfPage) -> Result<(), String> {
        let orientation = Orientation::of_page(page.size_pts());
        if orientation != self.orientation {
            self.turn(orientation)?;
        }
        let placement = gdi::place(page.size_pts(), page.size(), &self.surface);
        self.page(|job| job.draw(page, &placement))
    }

    /// A page drawn by `draw` between `StartPage` and `EndPage`.
    #[expect(unsafe_code, reason = "StartPage and EndPage bound the page")]
    fn page(&mut self, draw: impl FnOnce(&Self) -> Result<(), String>) -> Result<(), String> {
        // SAFETY: `dc` is live, the job started and no page open.
        if unsafe { StartPage(self.dc.raw()) } <= 0 {
            return Err(format!(
                "cannot start a page: {}",
                io::Error::last_os_error()
            ));
        }
        draw(self)?;
        // SAFETY: the page was started above.
        if unsafe { EndPage(self.dc.raw()) } <= 0 {
            return Err(format!("cannot end a page: {}", io::Error::last_os_error()));
        }
        self.pages += 1;
        self.pass_pages += 1;
        Ok(())
    }

    /// Draw `page` at `placement` in bands of at most
    /// [`gdi::MAX_BAND_BYTES`], each a bottom-up bitmap.  A stretched page's
    /// band carries a source row past each inner edge
    /// ([`gdi::band_source`]) and is clipped to its own device rows, so the
    /// halftone filter sees the rows across the edge.
    #[expect(unsafe_code, reason = "the stretch mode and brush origin of the bands")]
    fn draw(&self, page: &PdfPage, placement: &Placement) -> Result<(), String> {
        if page.depth() != SampleDepth::Eight {
            return Err("a 16-bit page cannot be drawn: GDI print bitmaps are 8-bit".to_owned());
        }
        let rgb = page.is_rgb();
        let format = DibFormat::of(rgb, self.mono && !rgb);
        let bpp = format.source_bpp();
        let (width, height) = page.size();
        let (out_w, out_h) = if placement.rotate {
            (height, width)
        } else {
            (width, height)
        };
        let stride = format.stride(out_w);
        let mut info = DibInfo::new(format, out_w);
        let stretched = (placement.width, placement.height) != (out_w, out_h);
        let overlap = u32::from(stretched);
        let band = gdi::band_rows(stride, out_h)
            .saturating_sub(2 * overlap)
            .max(1);
        // SAFETY: `dc` is live with a page open.
        unsafe {
            let _ = SetStretchBltMode(
                self.dc.raw(),
                if stretched { HALFTONE } else { COLORONCOLOR },
            );
            let _ = SetBrushOrgEx(self.dc.raw(), 0, 0, ptr::null_mut());
        }
        let read_error = |e: io::Error| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                format!("the page's image read back short of its {width} × {height} pixels")
            } else {
                format!("cannot read the page's image back: {e}")
            }
        };
        let mut rows = page.rows();
        let whole = if placement.rotate {
            let mut image = vec![0; width as usize * height as usize * bpp];
            rows.read_exact(&mut image).map_err(read_error)?;
            Some(image)
        } else {
            None
        };
        let mut row = vec![0; out_w as usize * bpp];
        let mut bits = vec![0u8; stride * (band + 2 * overlap) as usize];
        // The bitmap rows of the band before that the next one reads
        // again, and how many source rows have been read.
        let mut carry: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut fetched = 0;
        let mut start = 0;
        while start < out_h {
            let range = start..(start + band).min(out_h);
            let src = gdi::band_source(&range, overlap, out_h);
            let n = (src.end - src.start) as usize;
            for (i, r) in src.clone().enumerate() {
                let at = (n - 1 - i) * stride;
                let out = &mut bits[at..at + stride];
                if let Some((_, kept)) = carry.iter().find(|(k, _)| *k == r) {
                    out.copy_from_slice(kept);
                    continue;
                }
                debug_assert_eq!(r, fetched, "source rows are read in order");
                match &whole {
                    Some(image) => gdi::turned_row(image, (width, height), bpp, r, &mut row),
                    None => rows.read_exact(&mut row).map_err(read_error)?,
                }
                fetched += 1;
                gdi::dib_row(&row, format, out);
            }
            carry = (src.end.saturating_sub(2 * overlap).max(src.start)..src.end)
                .map(|r| {
                    let at = (n - 1 - (r - src.start) as usize) * stride;
                    (r, bits[at..at + stride].to_vec())
                })
                .collect();
            let (y, dest_h) = gdi::band_dest(&src, out_h, (placement.y, placement.height));
            let (clip_y, clip_h) = gdi::band_dest(&range, out_h, (placement.y, placement.height));
            if clip_h > 0 {
                info.header.biHeight = i32::try_from(n).unwrap_or(i32::MAX);
                let band = Band {
                    bits: &bits,
                    info: &info,
                    size: (out_w, n),
                    dest: (y, dest_h),
                    clip: (clip_y, clip_h),
                };
                self.blit(&band, placement).map_err(|e| {
                    format!(
                        "StretchDIBits drew none of rows {}..{} of the page: {e}",
                        range.start, range.end
                    )
                })?;
            }
            start = range.end;
        }
        Ok(())
    }

    /// Draw `band` at its device rows within `placement`, clipped to its
    /// own rows.
    ///
    /// # Errors
    ///
    /// The driver drew none of it.
    #[expect(
        unsafe_code,
        reason = "StretchDIBits draws the band, clipped through SaveDC, IntersectClipRect \
                  and RestoreDC"
    )]
    fn blit(&self, band: &Band<'_>, placement: &Placement) -> io::Result<()> {
        let dc = self.dc.raw();
        let width = i32::try_from(placement.width).unwrap_or(i32::MAX);
        let (clip_y, clip_h) = band.clip;
        // SAFETY: `dc` is live with a page open; the state saved here is
        // restored below.
        let saved = unsafe {
            let saved = SaveDC(dc);
            let _ = IntersectClipRect(
                dc,
                placement.x,
                clip_y,
                placement.x.saturating_add(width),
                clip_y.saturating_add(clip_h),
            );
            saved
        };
        // SAFETY: `bits` holds `size.1` bitmap rows of `size.0` pixels as
        // `info` describes, with its palette.
        let copied = unsafe {
            StretchDIBits(
                dc,
                placement.x,
                band.dest.0,
                width,
                band.dest.1,
                0,
                0,
                i32::try_from(band.size.0).unwrap_or(i32::MAX),
                i32::try_from(band.size.1).unwrap_or(i32::MAX),
                band.bits.as_ptr().cast(),
                (&raw const *band.info).cast::<BITMAPINFO>(),
                DIB_RGB_COLORS,
                SRCCOPY,
            )
        };
        let error = io::Error::last_os_error();
        if saved != 0 {
            // SAFETY: restores the state `SaveDC` saved above.
            let _ = unsafe { RestoreDC(dc, saved) };
        }
        if copied == 0 || copied == GDI_ERROR {
            return Err(error);
        }
        Ok(())
    }
}

/// One band of a page as [`Job::blit`] draws it.
struct Band<'a> {
    /// The bitmap rows, bottom-up.
    bits: &'a [u8],
    info: &'a DibInfo,
    /// The bitmap's width and rows.
    size: (u32, usize),
    /// The device rows `(y, height)` the bitmap's rows cover.
    dest: (i32, i32),
    /// The device rows `(y, height)` drawn: the band's own.
    clip: (i32, i32),
}

impl PageOut for Job {
    fn write_page(&mut self, page: &PdfPage) -> Result<(), RenderError> {
        self.print(page)
            .map_err(|e| RenderError::Io(io::Error::other(format!("{:?}: {e}", self.printer.name))))
    }

    fn passes(&self) -> u32 {
        self.copies.passes()
    }

    /// A two-sided pass of an odd number of pages leaves the back of its
    /// last sheet blank, so the next pass starts on a sheet of its own.
    fn end_pass(&mut self) -> Result<(), RenderError> {
        if gdi::pass_needs_blank(self.duplex, self.pass_pages) {
            self.page(|_| Ok(())).map_err(|e| {
                RenderError::Io(io::Error::other(format!(
                    "{:?}: the blank back of a copy's last sheet: {e}",
                    self.printer.name
                )))
            })?;
        }
        self.pass_pages = 0;
        Ok(())
    }
}

/// A bitmap's header and its palette, as `StretchDIBits` reads them.
#[repr(C)]
struct DibInfo {
    header: BITMAPINFOHEADER,
    palette: [RGBQUAD; 256],
}

impl DibInfo {
    /// The header of bands of `format`, `width` pixels wide: a gray ramp
    /// for 8-bit gray, black and white for 1 bit.
    fn new(format: DibFormat, width: u32) -> Self {
        let colours = match format {
            DibFormat::Bgr24 => 0,
            DibFormat::Gray8 => 256,
            DibFormat::Mono1 => 2,
        };
        let mut palette = [RGBQUAD::default(); 256];
        for (i, entry) in palette.iter_mut().enumerate().take(colours) {
            let level = if colours == 2 { 255 * i } else { i };
            let level = u8::try_from(level).unwrap_or(u8::MAX);
            *entry = RGBQUAD {
                rgbBlue: level,
                rgbGreen: level,
                rgbRed: level,
                rgbReserved: 0,
            };
        }
        Self {
            header: BITMAPINFOHEADER {
                biSize: u32::try_from(size_of::<BITMAPINFOHEADER>()).unwrap_or(0),
                biWidth: i32::try_from(width).unwrap_or(i32::MAX),
                biHeight: 1,
                biPlanes: 1,
                biBitCount: format.bits(),
                biCompression: BI_RGB,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: u32::try_from(colours).unwrap_or(0),
                biClrImportant: 0,
            },
            palette,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::path::Path;

    use clap::Parser as _;

    use super::{DMPAPER_A4, DMPAPER_LETTER, Job, Settings, begin, list_printers, printer_names};
    use crate::args::Args;
    use crate::print::{Duplex, PageSink, complete};
    use crate::render::{self, Document};

    const PRINT_TO_PDF: &str = "Microsoft Print to PDF";

    /// Whether Microsoft Print to PDF is a printer here.  Its absence skips
    /// the test, logged, unless `RR_REQUIRE_PRINT_TO_PDF=1` makes it a
    /// failure.
    fn print_to_pdf(test: &str) -> bool {
        let required = std::env::var_os("RR_REQUIRE_PRINT_TO_PDF").is_some_and(|v| v == "1");
        let reason = match printer_names() {
            Ok(names) if names.iter().any(|n| n == PRINT_TO_PDF) => {
                eprintln!("{test}: ran");
                return true;
            }
            Ok(names) => format!("{PRINT_TO_PDF:?} is not among the printers {names:?}"),
            Err(e) => format!("the printers cannot be listed: {e}"),
        };
        assert!(!required, "RR_REQUIRE_PRINT_TO_PDF=1: {reason}");
        eprintln!("{test}: skipped: {reason}");
        false
    }

    /// A PDF of one page of each size in points, each painted black whole.
    fn black_pages(sizes: &[(f64, f64)]) -> Vec<u8> {
        let n = sizes.len();
        let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            format!("<< /Type /Pages /Kids [{}] /Count {n} >>", kids.join(" ")),
        ];
        for (i, (w, h)) in sizes.iter().enumerate() {
            let content = format!("0 g 0 0 {w} {h} re f");
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents {} 0 R >>",
                4 + 2 * i
            ));
            objects.push(format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ));
        }
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            let _ = write!(pdf, "{} 0 obj\n{object}\nendobj\n", i + 1);
        }
        let xref = pdf.len();
        let _ = write!(pdf, "xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
        for offset in offsets {
            let _ = writeln!(pdf, "{offset:010} 00000 n ");
        }
        let _ = write!(
            pdf,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        );
        pdf.into_bytes()
    }

    /// The arguments of a job to Print to PDF writing `file`, with `extra`.
    fn print_args(file: &Path, extra: &[&str]) -> Args {
        let mut command_line = vec!["rrocket", "--backend", "cpu", "--printer", PRINT_TO_PDF];
        command_line.extend_from_slice(extra);
        command_line.extend(["--print-file", file.to_str().unwrap(), "in.pdf"]);
        let mut args = Args::parse_from(command_line);
        args.validate_format_flags().unwrap();
        args.apply_print_preset();
        args
    }

    /// Print `pdf` on paper `paper` as `args` asks, as `main` does.
    fn print(pdf: Vec<u8>, mut args: Args, paper: i16) -> Result<(), String> {
        let (_guard, spill) = crate::ram::redirect_to_ram(&mut args).unwrap();
        let config = args.session_config().unwrap();
        let session = pdf_raster::open_session_from_bytes(pdf, &config).unwrap();
        let total = session.total_pages();
        let mut spool = begin(&mut args, Some(paper))?;
        let sink = PageSink::new();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let pages: Vec<i32> = (1..=i32::try_from(total).unwrap()).collect();
        let render = |&page: &i32| {
            let number = page.unsigned_abs();
            let document = Document::Print(&sink);
            render::render_page(&session, number, total, &args, &spill, document)
                .err()
                .map(|e| (page, e))
        };
        let (errors, write_error) = spool.render_job(&sink, &pages, &pool, render);
        assert!(errors.is_empty(), "pages failed: {errors:?}");
        assert!(write_error.is_none(), "{write_error:?}");
        complete(spool, &args)
    }

    /// The pages of the PDF `file`.
    fn written(file: &Path) -> pdf_raster::RasterSession {
        let pdf = std::fs::read(file).unwrap();
        pdf_raster::open_session_from_bytes(pdf, &pdf_raster::SessionConfig::default()).unwrap()
    }

    /// The ink's bounding box `(x0, y0, x1, y1)` on page `page` rendered
    /// at 72 dpi, and the page's size in points.
    fn ink_box(
        session: &pdf_raster::RasterSession,
        page: u32,
    ) -> ((u32, u32, u32, u32), (f64, f64)) {
        let geometry = session.page_geometry(page).unwrap();
        let bitmap = pdf_raster::render_page_rgb(session, page, 1.0).unwrap();
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..bitmap.height {
            let row = bitmap.row_bytes(y);
            for x in 0..bitmap.width {
                let at = x as usize * 3;
                if row[at..at + 3].iter().all(|&v| v < 128) {
                    (x0, y0) = (x0.min(x), y0.min(y));
                    (x1, y1) = (x1.max(x + 1), y1.max(y + 1));
                }
            }
        }
        ((x0, y0, x1, y1), (geometry.width_pts, geometry.height_pts))
    }

    /// A name that is not valid UTF-16 is decoded with U+FFFD, not refused.
    #[test]
    fn printer_names_decode_lossily_only_when_invalid() {
        let office: Vec<u16> = "Office é".encode_utf16().collect();
        assert_eq!(super::printer_name(&office), "Office é");
        assert_eq!(super::printer_name(&[0x41, 0xD800, 0x42]), "A\u{FFFD}B");
    }

    #[test]
    fn print_to_pdf_is_listed() {
        if !print_to_pdf("print_to_pdf_is_listed") {
            return;
        }
        let list = list_printers().unwrap();
        assert!(
            list.lines()
                .any(|l| l.split('\t').next() == Some(PRINT_TO_PDF)),
            "{list}"
        );
    }

    /// A printer that prints to a file is refused without `--print-file`,
    /// and `--duplex` by a printer that does not print two-sided.
    #[test]
    fn a_file_port_needs_print_file_and_one_sided_printers_refuse_duplex() {
        if !print_to_pdf("a_file_port_needs_print_file_and_one_sided_printers_refuse_duplex") {
            return;
        }
        let settings = |print_file: Option<&Path>, duplex| Settings {
            printer: Some(PRINT_TO_PDF.to_owned()),
            copies: None,
            duplex,
            gray: false,
            mono: false,
            title: "rrocket test".to_owned(),
            print_file: print_file.map(Path::to_path_buf),
            paper: None,
        };
        let err = Job::open(&settings(None, None)).err().unwrap();
        assert!(err.contains("--print-file PATH"), "{err}");
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("duplex.pdf");
        if let Err(e) = Job::open(&settings(Some(&file), Some(Duplex::Long))) {
            assert!(e.contains("two-sided"), "{e}");
        }
    }

    /// Print to PDF writes the pages on the paper asked for, each turned
    /// to the sheet's orientation, fitted when larger than the printable
    /// area and at its size otherwise, centred; `--copies 2` writes them
    /// twice, collated.
    #[test]
    fn print_to_pdf_writes_fitted_centred_pages_on_a4_and_letter() {
        if !print_to_pdf("print_to_pdf_writes_fitted_centred_pages_on_a4_and_letter") {
            return;
        }
        // A5, a landscape page larger than either sheet, A3, and a small
        // landscape page.
        let sizes = [
            (420.0, 595.0),
            (1000.0, 650.0),
            (842.0, 1191.0),
            (300.0, 200.0),
        ];
        let pdf = black_pages(&sizes);
        let dir = tempfile::tempdir().unwrap();
        for (paper, sheet) in [
            (DMPAPER_A4, (595.0, 842.0)),
            (DMPAPER_LETTER, (612.0, 792.0)),
        ] {
            let file = dir.path().join(format!("paper-{paper}.pdf"));
            print(pdf.clone(), print_args(&file, &[]), paper).unwrap();
            let session = written(&file);
            assert_eq!(session.total_pages(), 4, "paper {paper}");
            for (page, &(w, h)) in (1..).zip(&sizes) {
                let ((x0, y0, x1, y1), size) = ink_box(&session, page);
                let want = if w > h { (sheet.1, sheet.0) } else { sheet };
                assert!(
                    (size.0 - want.0).abs() <= 1.0 && (size.1 - want.1).abs() <= 1.0,
                    "paper {paper} page {page}: {size:?}, not {want:?}"
                );
                let ink = (f64::from(x1 - x0), f64::from(y1 - y0));
                eprintln!("paper {paper} page {page}: sheet {size:?}, ink {ink:?} at ({x0}, {y0})");
                // Fitted to the sheet at most: the printable area may be
                // smaller.
                let scale = (size.0 / w).min(size.1 / h).min(1.0);
                assert!(
                    ink.0 <= w.mul_add(scale, 2.0) && ink.1 <= h.mul_add(scale, 2.0),
                    "paper {paper} page {page}: {ink:?} larger than the page at {scale}"
                );
                assert!(
                    (ink.0 / ink.1 - w / h).abs() < 0.02,
                    "paper {paper} page {page}: {ink:?} lost the aspect"
                );
                if scale >= 1.0 {
                    assert!(
                        (ink.0 - w).abs() <= 2.0 && (ink.1 - h).abs() <= 2.0,
                        "paper {paper} page {page}: {ink:?} is not at 100 %"
                    );
                }
                let (left, right) = (f64::from(x0), size.0 - f64::from(x1));
                let (top, bottom) = (f64::from(y0), size.1 - f64::from(y1));
                assert!(
                    (left - right).abs() <= 2.0 && (top - bottom).abs() <= 2.0,
                    "paper {paper} page {page}: margins {left} {right} {top} {bottom}"
                );
            }
        }
        let file = dir.path().join("copies.pdf");
        let args = print_args(&file, &["--copies", "2", "--gray"]);
        print(black_pages(&sizes[..2]), args, DMPAPER_A4).unwrap();
        let session = written(&file);
        assert_eq!(session.total_pages(), 4, "two copies of two pages");
        for (page, landscape) in [(1, false), (2, true), (3, false), (4, true)] {
            let (_, (w, h)) = ink_box(&session, page);
            assert_eq!(w > h, landscape, "page {page}: {w} × {h}");
        }
    }
}
