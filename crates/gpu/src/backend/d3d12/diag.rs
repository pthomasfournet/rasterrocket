//! Diagnostics of the WARP adapter: the module that implements it and the
//! instruction-set tier of the CPU it runs on.

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt as _;
use std::path::PathBuf;

use windows::Win32::Storage::FileSystem::{
    GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW,
};
use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
use windows::core::{PCWSTR, w};

/// `VS_FIXEDFILEINFO::dwSignature` of a valid version resource.
const FIXED_FILE_INFO_SIGNATURE: u32 = 0xFEEF_04BD;

/// The path and file version (`a.b.c.d`, or `unknown` when the module has
/// no version resource) of the `d3d10warp.dll` loaded in the process, or
/// `None` when none is loaded.
pub(super) fn warp_module() -> Option<(PathBuf, String)> {
    // SAFETY: the name is a NUL-terminated literal; the returned handle is
    // not freed and is only passed to `GetModuleFileNameW` below.
    let module = unsafe { GetModuleHandleW(w!("d3d10warp.dll")) }.ok()?;
    // Long-path limit of the Win32 API, in UTF-16 units.
    let mut buf = vec![0u16; 32_768];
    // SAFETY: `buf` is a writable slice; the call writes at most its length.
    let len = unsafe { GetModuleFileNameW(Some(module), &mut buf) } as usize;
    if len == 0 || len >= buf.len() {
        return None;
    }
    buf.truncate(len);
    let path = PathBuf::from(OsString::from_wide(&buf));
    buf.push(0);
    let version = file_version(&buf).unwrap_or_else(|| "unknown".to_owned());
    Some((path, version))
}

/// The fixed file version of the file at the NUL-terminated wide `path`.
fn file_version(path: &[u16]) -> Option<String> {
    let path = PCWSTR(path.as_ptr());
    // SAFETY: `path` is NUL-terminated and outlives the call.
    let size = unsafe { GetFileVersionInfoSizeW(path, None) };
    if size == 0 {
        return None;
    }
    let mut block = vec![0u8; size as usize];
    // SAFETY: `block` holds `size` writable bytes.
    unsafe { GetFileVersionInfoW(path, None, size, block.as_mut_ptr().cast()) }.ok()?;
    let mut info: *mut core::ffi::c_void = core::ptr::null_mut();
    let mut info_len = 0u32;
    // SAFETY: `block` is the version resource `GetFileVersionInfoW` filled;
    // the out-pointers are valid for writes.
    let found = unsafe {
        VerQueryValueW(
            block.as_ptr().cast(),
            w!("\\"),
            &raw mut info,
            &raw mut info_len,
        )
    };
    if !found.as_bool()
        || info.is_null()
        || (info_len as usize) < core::mem::size_of::<VS_FIXEDFILEINFO>()
    {
        return None;
    }
    // SAFETY: `info` points into `block`, which is alive, at a
    // `VS_FIXEDFILEINFO` of `info_len` bytes; the read tolerates any
    // alignment.
    let fixed = unsafe { info.cast::<VS_FIXEDFILEINFO>().read_unaligned() };
    if fixed.dwSignature != FIXED_FILE_INFO_SIGNATURE {
        return None;
    }
    Some(format!(
        "{}.{}.{}.{}",
        fixed.dwFileVersionMS >> 16,
        fixed.dwFileVersionMS & 0xffff,
        fixed.dwFileVersionLS >> 16,
        fixed.dwFileVersionLS & 0xffff
    ))
}

/// The widest vector instruction set the host CPU supports: `avx512`
/// (AVX-512 F), `avx2` or `sse` on x86, `neon` on ARM64.
pub(super) fn cpu_tier() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            "avx512"
        } else if std::arch::is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "sse"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon"
    }
}
