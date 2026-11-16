//! Pixel-area thresholds for GPU image decoding and the environment
//! variables that override them.

/// Minimum pixel area (width × height) for GPU-accelerated `DCTDecode`
/// (nvJPEG or VA-API).
///
/// `u32::MAX` disables the dispatch: on a GPU with one hardware JPEG
/// engine, per-image decode calls serialise every worker thread through
/// that engine, and the hybrid backend decodes a single image on the CPU
/// behind a stream synchronisation, so both lose to the multithreaded CPU
/// decoder.  Batched decoding across pages, or a GPU with several JPEG
/// engines, is the configuration in which a finite threshold (on the order
/// of 512 × 512) pays off.  Measurements are kept with the project's
/// benchmarks.
///
/// `GPU_JPEG2K_THRESHOLD_PX` is independent of this value.
#[cfg(any(feature = "nvjpeg", feature = "vaapi"))]
pub(crate) const GPU_JPEG_THRESHOLD_PX: u32 = u32::MAX;

/// Minimum pixel area (width × height) for GPU-accelerated `JPXDecode`
/// through a decoder built on demand ([`JpxSupply::OnDemand`], as under
/// `BackendPolicy::Auto`).
///
/// `u32::MAX` disables the dispatch, and no such decoder is built.  A
/// thread's decoder build costs more process time than the device saves on
/// a single page at any area measured, and over a run of pages the outcome
/// is not ordered by page count or scheduled area: the device wins on some
/// runs and ties or loses on others of equal length.  Fewer rendering
/// threads, hence fewer serialised decoder builds, do not change that
/// ratio.  Measurements are kept with the project's benchmarks.
#[cfg(feature = "nvjpeg2k")]
const GPU_JPEG2K_THRESHOLD_PX: u32 = u32::MAX;

/// Minimum pixel area (width × height) for GPU-accelerated `JPXDecode`
/// through a decoder attached before the page renders
/// ([`JpxSupply::Attached`], as under `BackendPolicy::ForceCuda`): 2²¹
/// pixels, about 1448 × 1448.
///
/// Below it the multithreaded CPU decoder finishes an image in less time
/// than the device takes to transfer it, over a single page and over a run
/// of pages alike.  The encoded size per pixel does not order the outcome
/// at a given area.
#[cfg(feature = "nvjpeg2k")]
const GPU_JPEG2K_ATTACHED_THRESHOLD_PX: u32 = 2_097_152;

/// How a renderer's device JPEG 2000 decoder reaches it, which selects the
/// default dispatch threshold.
#[cfg(feature = "nvjpeg2k")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JpxSupply {
    /// Built by the first decode it is offered: [`GPU_JPEG2K_THRESHOLD_PX`].
    OnDemand,
    /// Attached before the page renders: [`GPU_JPEG2K_ATTACHED_THRESHOLD_PX`].
    Attached,
}

/// Returns the effective `JPXDecode` dispatch threshold for a decoder
/// supplied as `supply`: the decimal pixel area `PDF_RASTER_JPX_THRESHOLD`
/// holds (`0` = every eligible image on the device, `4294967295` = none),
/// which overrides both supplies, else the supply's default.  The variable
/// is read once, on first use.
#[cfg(feature = "nvjpeg2k")]
pub(crate) fn jpx_threshold(supply: JpxSupply) -> u32 {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| threshold_override_from_env("PDF_RASTER_JPX_THRESHOLD"))
        .unwrap_or(match supply {
            JpxSupply::OnDemand => GPU_JPEG2K_THRESHOLD_PX,
            JpxSupply::Attached => GPU_JPEG2K_ATTACHED_THRESHOLD_PX,
        })
}

/// Whether a `JPXDecode` image of `w` × `h` samples meets the dispatch
/// threshold `threshold`.  `u32::MAX` admits no image, including one whose
/// area saturates `u32`.
#[cfg(feature = "nvjpeg2k")]
pub(crate) const fn jpx_area_dispatches(w: u32, h: u32, threshold: u32) -> bool {
    threshold != u32::MAX && w.saturating_mul(h) >= threshold
}

/// Whether an image of `full` pixels drawn at `draw` is drawn at half its
/// size or less along both axes.  The device JPEG 2000 decoder reconstructs
/// at full resolution, and declines such a draw: its samples are resampled on the
/// host to at most a quarter of their number, and the CPU JPEG 2000
/// decoder skips at least one resolution level (each halves both axes).
#[cfg(feature = "nvjpeg2k")]
pub(crate) const fn draw_halves((w, h): (u32, u32), (dw, dh): (u32, u32)) -> bool {
    dw.saturating_mul(2) <= w && dh.saturating_mul(2) <= h
}

/// The pixel count the environment variable `var` holds in decimal, or
/// `default` when it is unset or does not parse.
#[cfg(feature = "gpu-icc")]
pub(super) fn threshold_from_env<T: std::str::FromStr>(var: &str, default: T) -> T {
    threshold_override_from_env(var).unwrap_or(default)
}

/// The pixel count the environment variable `var` holds in decimal; `None`
/// when it is unset, and when it does not parse, which is reported at warn.
#[cfg(any(feature = "nvjpeg2k", feature = "gpu-icc"))]
fn threshold_override_from_env<T: std::str::FromStr>(var: &str) -> Option<T> {
    let value = std::env::var(var).ok()?;
    let parsed = value.trim().parse().ok();
    if parsed.is_none() {
        log::warn!(
            "{var}={value:?} is not a pixel count (a non-negative integer); \
             using the default threshold"
        );
    }
    parsed
}

#[cfg(all(test, feature = "nvjpeg2k"))]
mod tests {
    use super::*;

    /// A finite threshold admits the areas at or above it; `u32::MAX`
    /// admits none, an area that saturates `u32` included.
    #[test]
    fn the_maximum_jpx_threshold_admits_no_area() {
        assert!(jpx_area_dispatches(2048, 1024, 2_097_152));
        assert!(!jpx_area_dispatches(2047, 1024, 2_097_152));
        assert!(jpx_area_dispatches(1, 1, 0));
        assert!(jpx_area_dispatches(u32::MAX, u32::MAX, u32::MAX - 1));
        assert!(!jpx_area_dispatches(u32::MAX, u32::MAX, u32::MAX));
        assert!(!jpx_area_dispatches(
            65_536,
            65_536,
            GPU_JPEG2K_THRESHOLD_PX
        ));
    }

    /// Only a decoder built on demand is dormant by default; one attached
    /// before the page renders dispatches from 2²¹ pixels.
    #[test]
    fn an_attached_jpx_decoder_is_not_dormant_by_default() {
        assert_eq!(GPU_JPEG2K_THRESHOLD_PX, u32::MAX);
        assert_ne!(GPU_JPEG2K_ATTACHED_THRESHOLD_PX, u32::MAX);
        if std::env::var_os("PDF_RASTER_JPX_THRESHOLD").is_none() {
            assert_eq!(jpx_threshold(JpxSupply::OnDemand), u32::MAX);
            assert_eq!(
                jpx_threshold(JpxSupply::Attached),
                GPU_JPEG2K_ATTACHED_THRESHOLD_PX
            );
        }
    }
}
