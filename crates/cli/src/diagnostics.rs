//! Human-readable error reporting for the CLI.
//!
//! All stderr output that is not progress reporting lives here.
//! Functions are free-standing so `main` can call them without
//! owning any render state.

#![expect(
    clippy::redundant_pub_crate,
    reason = "bin-only crate with private modules: pub(crate) is the real \
              visibility and cannot escape; widening to pub would misdescribe it"
)]

use pdf_raster::{BackendPolicy, RasterError};

use crate::args::Args;
use crate::render::RenderError;

/// The `source()` chain of `e` below `e` itself, one message per level.
///
/// A level whose message already appears in the text printed above it is
/// omitted: a wrapper that formats its source into its own `Display` (e.g.
/// `RenderError::Raster` printing "render error: {source}") would otherwise
/// print the same sentence again under "caused by:".
fn error_chain(e: &dyn std::error::Error) -> Vec<String> {
    let mut printed = e.to_string();
    let mut causes = Vec::new();
    let mut src = e.source();
    while let Some(cause) = src {
        let text = cause.to_string();
        if !printed.contains(&text) {
            printed.clone_from(&text);
            causes.push(text);
        }
        src = cause.source();
    }
    causes
}

/// Print [`error_chain`] of `e` to stderr, one "caused by:" line per level.
pub(crate) fn print_error_chain(e: &dyn std::error::Error) {
    for cause in error_chain(e) {
        eprintln!("  caused by: {cause}");
    }
}

/// Print a human-readable error (and actionable hints) when `open_session` fails.
///
/// `policy` is the resolved backend (after CLI flag → env var → default
/// fallback) so a hint can name the source the user actually controls
/// even when no `--backend` flag was passed.
pub(crate) fn report_open_error(e: &RasterError, args: &Args, policy: BackendPolicy) {
    if matches!(e, RasterError::BackendUnavailable(_)) {
        eprintln!("rrocket: {e}");
        print_backend_hint(policy, args.vaapi_device.as_deref(), args.backend.is_none());
    } else if matches!(e, RasterError::InvalidOptions(_)) {
        eprintln!("rrocket: {e}");
    } else {
        eprintln!("rrocket: failed to open PDF: {e}");
        print_error_chain(e);
    }
}

/// Print a backend-specific hint for the resolved policy.
///
/// `via_env` flags whether the policy came from `PDF_RASTER_BACKEND`
/// rather than `--backend`, so the hint can name the actual knob the
/// user reached for.
fn print_backend_hint(policy: BackendPolicy, vaapi_device: Option<&str>, via_env: bool) {
    #[cfg(not(feature = "vaapi"))]
    let _ = vaapi_device;
    let source_hint = |backend: &str| {
        if via_env {
            eprintln!("  hint: PDF_RASTER_BACKEND={backend} is set in your environment.");
        }
    };
    match policy {
        BackendPolicy::ForceCuda => {
            source_hint("cuda");
            print_cuda_hint();
        }
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => {
            source_hint("vaapi");
            print_vaapi_hint(vaapi_device);
        }
        BackendPolicy::ForceVulkan => {
            source_hint("vulkan");
            print_vulkan_hint();
        }
        _ => {
            eprintln!("  hint: use --backend auto to fall back to CPU when GPU is unavailable,");
            eprintln!("        or --backend cpu to force CPU-only mode.");
        }
    }
}

fn print_cuda_hint() {
    eprintln!("        --backend cuda requires a working CUDA driver and GPU.");
    eprintln!("        Run `nvidia-smi` to verify the driver is loaded.");
    eprintln!(
        "        Use --backend auto to fall back to Vulkan / CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
}

#[cfg(feature = "vaapi")]
fn print_vaapi_hint(device: Option<&str>) {
    if let Some(node) = device.filter(|d| *d != pdf_raster::DEFAULT_VAAPI_DEVICE) {
        eprintln!("        --backend vaapi could not use the DRM device {node}.");
        eprintln!("        Verify it is readable by your user, pass one of the render");
        eprintln!("        nodes listed above, or --vaapi-device auto to probe them all.");
    } else {
        eprintln!("        --backend vaapi found no render node with VA-API baseline JPEG");
        eprintln!("        decode.  Check that a VA-API driver for the device is installed");
        eprintln!("        (`vainfo --display drm --device <node>`).");
    }
    eprintln!(
        "        Use --backend auto to fall back to CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
}

fn print_vulkan_hint() {
    eprintln!("        --backend vulkan requires a Vulkan 1.3+ device and the loader.");
    eprintln!("        Run `vulkaninfo` to verify a usable Vulkan device is present.");
    eprintln!(
        "        Use --backend auto to fall back to CUDA / CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
}

/// Sort errors by page number and print each with its cause chain.
///
/// Returns the run's exit status: success when `errors` is empty, failure
/// otherwise.
pub(crate) fn report_errors(mut errors: Vec<(i32, RenderError)>) -> std::process::ExitCode {
    if errors.is_empty() {
        return std::process::ExitCode::SUCCESS;
    }
    errors.sort_by_key(|(p, _)| *p);
    let mut size_hint = false;
    for (page, err) in &errors {
        eprintln!("rrocket: page {page}: {err}");
        print_error_chain(err);
        size_hint |= matches!(
            err,
            RenderError::Raster(
                RasterError::PageTooLarge { .. } | RasterError::PageAreaTooLarge { .. }
            )
        );
    }
    if size_hint {
        eprintln!(
            "  hint: render such a page in tiles with -x/-y/-W/-H, each tile within the limit"
        );
    }
    std::process::ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_omits_a_cause_its_wrapper_already_printed() {
        let err = RenderError::Raster(RasterError::PageTooLarge {
            width: 40_000,
            height: 10,
        });
        assert!(
            err.to_string().contains(
                &RasterError::PageTooLarge {
                    width: 40_000,
                    height: 10
                }
                .to_string()
            )
        );
        assert_eq!(error_chain(&err), Vec::<String>::new());
    }

    #[test]
    fn chain_keeps_a_cause_with_new_information() {
        #[derive(Debug)]
        struct Wrapper(std::io::Error);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("write failed")
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let err = Wrapper(std::io::Error::other("disk full"));
        assert_eq!(error_chain(&err), vec!["disk full".to_owned()]);
    }
}
