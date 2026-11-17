//! Classification of CUDA initialisation failures.
//!
//! A CUDA context fails to come up for one of a few causes, each with its
//! own remedy: the driver library is absent, the driver sees no device, the
//! driver is older than the toolkit that compiled the kernels, or the
//! kernels target a newer architecture than the device.  The PTX text
//! records the toolkit release (`// Cuda compilation tools, release X.Y`),
//! the PTX ISA (`.version X.Y`) and the target (`.target sm_NN`); the driver
//! reports the newest CUDA version it supports and the device its compute
//! capability.  Each failure is described from its error code and these
//! facts; a kernel module and user-space driver of different versions (a
//! driver updated without a reboot) is told apart as well.

use cudarc::driver::DriverError;
use cudarc::driver::sys::{self, CUresult};

/// What a PTX module's header states about its origin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PtxHeader {
    /// Release of the CUDA toolkit that compiled it, `(major, minor)`.
    pub(crate) toolkit: Option<(u32, u32)>,
    /// PTX ISA version, `(major, minor)`.
    pub(crate) isa: Option<(u32, u32)>,
    /// Target architecture `N` of `.target sm_N` (`75` for `sm_75`).
    pub(crate) sm: Option<u32>,
}

/// The header facts of `ptx`; a fact the text does not state is `None`.
pub(crate) fn ptx_header(ptx: &str) -> PtxHeader {
    let mut header = PtxHeader::default();
    for line in ptx.lines().take(64) {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("// Cuda compilation tools, release ") {
            header.toolkit = rest.split(',').next().and_then(major_minor);
        } else if let Some(rest) = line.strip_prefix(".version ") {
            header.isa = major_minor(rest.trim());
        } else if let Some(rest) = line.strip_prefix(".target ") {
            header.sm = rest
                .split(',')
                .next()
                .and_then(|t| t.trim().strip_prefix("sm_"))
                .map(|n| n.trim_end_matches(|c: char| c.is_ascii_alphabetic()))
                .and_then(|n| n.parse().ok());
        }
    }
    header
}

/// `"X.Y"` as `(X, Y)`.
fn major_minor(s: &str) -> Option<(u32, u32)> {
    let (major, minor) = s.trim().split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Whether the CUDA driver library can be loaded.  Every other driver call
/// panics without it, so this is asked first.
pub(crate) fn driver_library_present() -> bool {
    // SAFETY: probes `dlopen` of the candidate library names and closes each
    // handle again; no symbol is called.
    unsafe { sys::is_culib_present() }
}

/// The newest CUDA version the installed driver supports, `(major, minor)`
/// (`cuDriverGetVersion`, which needs no `cuInit`).
pub(crate) fn driver_cuda_version() -> Option<(u32, u32)> {
    let mut version: core::ffi::c_int = 0;
    // SAFETY: `version` is a valid out-pointer for the duration of the call;
    // the library is present (callers check `driver_library_present` first).
    let status = unsafe { sys::cuDriverGetVersion(&raw mut version) };
    let version = u32::try_from(version).ok().filter(|v| *v > 0);
    (status == CUresult::CUDA_SUCCESS)
        .then_some(version)
        .flatten()
        .map(|v| (v / 1000, v % 1000 / 10))
}

/// The message for a missing driver library.
pub(crate) const NO_DRIVER_LIBRARY: &str = "the CUDA driver library (libcuda) is not installed \
     or not on the loader path: install the NVIDIA driver of a CUDA-capable GPU";

/// The message for `err`, returned while creating the context.
pub(crate) fn context_failure(err: DriverError, driver: Option<(u32, u32)>) -> String {
    match err.0 {
        CUresult::CUDA_ERROR_NO_DEVICE => format!(
            "the NVIDIA driver{} sees no CUDA device ({err:?}): check `nvidia-smi`; an empty or \
             wrong CUDA_VISIBLE_DEVICES hides every device",
            driver_suffix(driver)
        ),
        CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => format!(
            "the NVIDIA kernel module and the user-space driver{} are different versions \
             ({err:?}): reboot, or reload the nvidia module, after a driver update",
            driver_suffix(driver)
        ),
        _ => format!(
            "CUDA context creation failed ({err:?}){}: check `nvidia-smi`",
            driver_suffix(driver)
        ),
    }
}

/// The oldest architecture a toolkit of major release `major` compiles
/// for (`sm_N` as `N`): CUDA 13 dropped Maxwell, Pascal and Volta, CUDA 12
/// Kepler.  `None` for a release outside the table.
const fn oldest_target(major: u32) -> Option<u32> {
    match major {
        13.. => Some(75),
        12 => Some(50),
        11 => Some(35),
        _ => None,
    }
}

/// The message for kernels that target a newer architecture than the
/// device, or `None` when the header and the device do not show that.
///
/// A device older than every target of the toolkit that compiled the
/// kernels cannot run this build's CUDA kernels at all; any other is told
/// the `CUDA_ARCH` to rebuild with.
pub(crate) fn arch_mismatch(header: &PtxHeader, device: (u32, u32)) -> Option<String> {
    let target = header.sm?;
    let device_sm = device.0 * 10 + device.1;
    (target > device_sm).then(|| {
        let gpu = format!(
            "sm_{device_sm}, compute capability {}.{}",
            device.0, device.1
        );
        match header.toolkit {
            Some((major, minor)) if oldest_target(major).is_some_and(|o| device_sm < o) => {
                format!(
                    "the CUDA kernels were compiled for sm_{target} by CUDA {major}.{minor}, \
                     which compiles for no GPU as old as this one ({gpu}): this GPU cannot run \
                     this build's CUDA kernels; use the Vulkan backend or the CPU"
                )
            }
            _ => format!(
                "the CUDA kernels were compiled for sm_{target}, newer than this GPU ({gpu}): \
                 rebuild with CUDA_ARCH=sm_{device_sm}"
            ),
        }
    })
}

/// The message for `err`, returned while loading a kernel module whose
/// header is `header` on a device of compute capability `device`.
pub(crate) fn load_failure(
    err: DriverError,
    header: &PtxHeader,
    driver: Option<(u32, u32)>,
    device: Option<(u32, u32)>,
) -> String {
    let built = match (header.toolkit, header.isa) {
        (Some((tm, tn)), Some((im, in_))) => format!("CUDA {tm}.{tn} (PTX ISA {im}.{in_})"),
        (Some((tm, tn)), None) => format!("CUDA {tm}.{tn}"),
        (None, Some((im, in_))) => format!("PTX ISA {im}.{in_}"),
        (None, None) => "an unknown toolkit".to_owned(),
    };
    match err.0 {
        CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => {
            let needs = header.toolkit.map_or_else(String::new, |(m, n)| {
                format!(" one that supports CUDA {m}.{n} or later")
            });
            format!(
                "the CUDA kernels were compiled by {built}, newer than this NVIDIA driver \
                 supports{} ({err:?}): update the driver to{needs}",
                driver.map_or_else(String::new, |(m, n)| format!(" (up to CUDA {m}.{n})"))
            )
        }
        CUresult::CUDA_ERROR_INVALID_PTX | CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU => {
            let target = header
                .sm
                .map_or_else(String::new, |s| format!(" for sm_{s}"));
            let gpu = device.map_or_else(String::new, |(m, n)| format!(" (sm_{m}{n})"));
            format!(
                "the CUDA kernels compiled by {built}{target} could not be built for this GPU{gpu} \
                 ({err:?}): rebuild with CUDA_ARCH set to the GPU's architecture"
            )
        }
        _ => format!(
            "loading the CUDA kernels compiled by {built} failed ({err:?}){}",
            driver_suffix(driver)
        ),
    }
}

/// `" (supports CUDA X.Y)"`, or nothing when the version is unknown.
fn driver_suffix(driver: Option<(u32, u32)>) -> String {
    driver.map_or_else(String::new, |(m, n)| format!(" (supports CUDA {m}.{n})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "//\n// Generated by NVIDIA NVVM Compiler\n//\n\
        // Compiler Build ID: CL-38244171\n// Cuda compilation tools, release 13.3, V13.3.73\n\
        // Based on NVVM 23.0.0\n//\n\n.version 9.3\n.target sm_120\n.address_size 64\n";

    #[test]
    fn the_header_states_toolkit_isa_and_target() {
        assert_eq!(
            ptx_header(HEADER),
            PtxHeader {
                toolkit: Some((13, 3)),
                isa: Some((9, 3)),
                sm: Some(120)
            }
        );
        assert_eq!(ptx_header(".target sm_90a\n").sm, Some(90));
        assert_eq!(ptx_header(""), PtxHeader::default());
    }

    #[test]
    fn a_newer_target_than_the_device_is_named() {
        let h = PtxHeader {
            sm: Some(80),
            ..PtxHeader::default()
        };
        let m = arch_mismatch(&h, (7, 5)).expect("sm_80 on a 7.5 device");
        assert!(m.contains("sm_80") && m.contains("CUDA_ARCH=sm_75"), "{m}");
        assert_eq!(arch_mismatch(&h, (8, 0)), None);
        assert_eq!(arch_mismatch(&h, (12, 0)), None);
        assert_eq!(arch_mismatch(&PtxHeader::default(), (7, 5)), None);
    }

    /// A GPU older than every target of the build's toolkit is told it
    /// cannot run the kernels, not to rebuild for it.
    #[test]
    fn a_gpu_the_toolkit_dropped_is_told_to_use_vulkan_or_the_cpu() {
        let cuda13 = PtxHeader {
            toolkit: Some((13, 3)),
            isa: Some((9, 3)),
            sm: Some(80),
        };
        let m = arch_mismatch(&cuda13, (6, 1)).expect("sm_80 on a 6.1 device");
        assert!(m.contains("cannot run") && m.contains("Vulkan"), "{m}");
        assert!(!m.contains("CUDA_ARCH"), "{m}");
        let m = arch_mismatch(&cuda13, (7, 5)).expect("sm_80 on a 7.5 device");
        assert!(m.contains("CUDA_ARCH=sm_75"), "{m}");
        let cuda12 = PtxHeader {
            toolkit: Some((12, 4)),
            ..cuda13
        };
        let m = arch_mismatch(&cuda12, (6, 1)).expect("sm_80 on a 6.1 device");
        assert!(m.contains("CUDA_ARCH=sm_61"), "{m}");
    }

    #[test]
    fn a_newer_toolkit_than_the_driver_names_both() {
        let h = ptx_header(HEADER);
        let e = DriverError(CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
        let m = load_failure(e, &h, Some((12, 8)), Some((7, 5)));
        assert!(m.contains("CUDA 13.3 (PTX ISA 9.3)"), "{m}");
        assert!(m.contains("up to CUDA 12.8"), "{m}");
        assert!(m.contains("CUDA 13.3 or later"), "{m}");
    }

    #[test]
    fn no_device_points_at_visible_devices() {
        let e = DriverError(CUresult::CUDA_ERROR_NO_DEVICE);
        let m = context_failure(e, Some((12, 8)));
        assert!(
            m.contains("CUDA_VISIBLE_DEVICES") && m.contains("12.8"),
            "{m}"
        );
    }
}
