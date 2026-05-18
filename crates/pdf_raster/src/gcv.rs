//! Google Cloud Vision input optimization.
//!
//! Produces an upload-ready grayscale JPEG payload guaranteed to fit GCV's
//! binding limit — 10 MB of base64 inside the `images:annotate` JSON request
//! (NOT the 20 MB raw-file limit) — decided deterministically and locally.
//! No HTTP, auth, or batching: this crate renders pixels, it is not a GCV
//! API client.

use std::borrow::Cow;

use base64::Engine as _;
use color::Gray8;
use raster::Bitmap;

use crate::RenderedPage;

/// Budget for a single GCV `images:annotate` request.
#[derive(Debug, Clone)]
pub struct GcvBudget {
    /// Max length of the base64 string. GCV rejects JSON requests whose
    /// inline `content` exceeds 10 MB of base64.
    pub max_base64_bytes: usize,
    /// Lowest JPEG quality the step-down may use before falling back to
    /// downscaling. Below this, OCR accuracy degrades unacceptably.
    pub min_quality: u8,
    /// Starting JPEG quality.
    pub start_quality: u8,
}

impl Default for GcvBudget {
    fn default() -> Self {
        Self {
            max_base64_bytes: 10 * 1024 * 1024,
            min_quality: 60,
            start_quality: 90,
        }
    }
}

/// A GCV-ready encoded page.
#[derive(Debug, Clone)]
pub struct GcvImage {
    /// Fitted baseline grayscale JPEG bytes — guaranteed within budget.
    /// The universal artifact: write to disk, upload to GCS for
    /// `files:asyncBatchAnnotate`, or feed any other consumer.
    pub jpeg: Vec<u8>,
    /// Final JPEG quality used by the fitting algorithm.
    pub quality: u8,
    /// Final pixel dimensions (may be downscaled from the page if the
    /// quality floor was hit).
    pub width: u32,
    /// Final pixel height.
    pub height: u32,
}

impl GcvImage {
    /// base64 of `jpeg`, ready to drop into the GCV `images:annotate`
    /// request body. The fitting algorithm measured this exact length
    /// against the budget, so the result is within GCV's JSON ceiling.
    #[must_use]
    pub fn to_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.jpeg)
    }
}

/// Why a page could not be fitted into a [`GcvBudget`].
#[derive(Debug)]
pub enum GcvError {
    /// The page cannot fit the budget even at `min_quality` and the
    /// resolution floor, or it would exceed GCV's 75 MP OCR limit even
    /// after maximum downscale. The caller decides whether to split or skip
    /// — never an over-budget payload.
    Unfittable {
        /// Smallest base64 length achieved before giving up.
        smallest_base64: usize,
        /// The budget that could not be met.
        budget: usize,
    },
    /// The underlying JPEG codec failed.
    Encode(encode::EncodeError),
}

impl std::fmt::Display for GcvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unfittable {
                smallest_base64,
                budget,
            } => write!(
                f,
                "page does not fit GCV budget: smallest {smallest_base64} B base64 > {budget} B"
            ),
            Self::Encode(e) => write!(f, "JPEG encode failed: {e}"),
        }
    }
}

impl std::error::Error for GcvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(e) => Some(e),
            Self::Unfittable { .. } => None,
        }
    }
}

/// GCV's hard OCR pixel limit; above this GCV silently downscales server-side.
const GCV_MAX_PIXELS: u64 = 75_000_000;

/// Short-side floor — GCV's documented OCR sweet spot starts ~1024 px.
/// We never downscale below this; if the budget still can't be met, that is
/// `Unfittable` (caller's decision), not a silent sub-floor image.
const OCR_SHORT_SIDE_FLOOR: u32 = 1024;

/// Baseline JPEG's hard per-side ceiling. A dimension above this cannot be
/// encoded at all, so such a candidate is not a fitting option — it must be
/// filtered out exactly like the 75 MP cap, not surfaced as a codec failure.
const JPEG_MAX_SIDE: u32 = u16::MAX as u32;

/// base64 length for `n` raw bytes (STANDARD, with padding): ceil(n/3)*4.
const fn base64_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

/// Box-filter downscale of a tightly-packed L8 buffer to `dst_w`×`dst_h`.
/// Deterministic integer averaging — no float accumulation order issues.
#[expect(
    clippy::cast_possible_truncation,
    reason = "src/dst extents fit u32 by construction; box index products \
              stay within the source buffer range"
)]
fn downscale_gray(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h) as usize];
    for dy in 0..dst_h {
        let y0 = (u64::from(dy) * u64::from(src_h) / u64::from(dst_h)) as u32;
        let y1 = ((u64::from(dy + 1) * u64::from(src_h) / u64::from(dst_h)) as u32).max(y0 + 1);
        for dx in 0..dst_w {
            let x0 = (u64::from(dx) * u64::from(src_w) / u64::from(dst_w)) as u32;
            let x1 = ((u64::from(dx + 1) * u64::from(src_w) / u64::from(dst_w)) as u32).max(x0 + 1);
            let mut sum = 0u64;
            let mut count = 0u64;
            for y in y0..y1.min(src_h) {
                let row = (y * src_w) as usize;
                for x in x0..x1.min(src_w) {
                    sum += u64::from(src[row + x as usize]);
                    count += 1;
                }
            }
            // Averaged value is 0..=255 (mean of u8 samples); empty box → 0.
            dst[(dy * dst_w + dx) as usize] = sum.checked_div(count).map_or(0, |avg| avg as u8);
        }
    }
    dst
}

/// Encode `pixels` (tight L8, `w`×`h`) to JPEG at `quality` via the encode
/// crate. Wraps the buffer in a 1-byte-stride `Gray8` bitmap.
fn encode_jpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Result<Vec<u8>, GcvError> {
    let mut bmp = Bitmap::<Gray8>::new(w, h, 1, false);
    for y in 0..h {
        let src = &pixels[(y * w) as usize..((y + 1) * w) as usize];
        bmp.row_bytes_mut(y)[..w as usize].copy_from_slice(src);
    }
    encode::jpeg_gray::<Gray8>(&bmp, quality).map_err(GcvError::Encode)
}

/// Encode a [`RenderedPage`] into a GCV-ready base64 JPEG guaranteed to fit
/// `budget`, decided deterministically with no network call.
///
/// Algorithm: try `start_quality`; binary-search quality down toward
/// `min_quality`; only if the quality floor still overflows, box-downscale
/// (aspect-preserving, never below the OCR short-side floor) and retry.
/// Resolution is sacrificed last — GCV tolerates compression better than a
/// silent server-side resize.
///
/// # Errors
///
/// [`GcvError::Unfittable`] if the page cannot meet the budget without
/// dropping below the OCR resolution floor, or would exceed GCV's 75 MP
/// limit even downscaled. Never returns an over-budget or > 75 MP payload.
pub fn encode_for_gcv(page: &RenderedPage, budget: &GcvBudget) -> Result<GcvImage, GcvError> {
    // A degenerate (zero-area) page has no valid payload.
    if page.width == 0 || page.height == 0 {
        return Err(GcvError::Unfittable {
            smallest_base64: usize::MAX,
            budget: budget.max_base64_bytes,
        });
    }

    let lo_q = budget.min_quality.min(budget.start_quality);
    let hi_q = budget.start_quality.max(budget.min_quality);

    let mut dims: Vec<(u32, u32)> = Vec::new();
    let (mut cw, mut ch) = (page.width, page.height);
    let mut is_native = true;
    loop {
        let short = cw.min(ch);
        // Native is always allowed (it is the input, not a downscale).
        // Downscaled candidates must respect the OCR short-side floor.
        let floor_ok = is_native || short >= OCR_SHORT_SIDE_FLOOR;
        // A side above the baseline-JPEG ceiling cannot be encoded; treat it
        // as a non-candidate (it falls out to a smaller halving or, if none
        // qualifies, `Unfittable`) rather than letting `jpeg_gray` raise a
        // codec error the caller would misread as an encoder fault.
        let jpeg_ok = cw <= JPEG_MAX_SIDE && ch <= JPEG_MAX_SIDE;
        if floor_ok && jpeg_ok && u64::from(cw) * u64::from(ch) <= GCV_MAX_PIXELS {
            dims.push((cw, ch));
        }
        // Stop once we have reached/passed the floor or cannot halve further.
        if short <= OCR_SHORT_SIDE_FLOOR || cw < 2 || ch < 2 {
            break;
        }
        cw = (cw / 2).max(1);
        ch = (ch / 2).max(1);
        is_native = false;
    }
    if dims.is_empty() {
        return Err(GcvError::Unfittable {
            smallest_base64: usize::MAX,
            budget: budget.max_base64_bytes,
        });
    }

    let mut smallest = usize::MAX;

    for (di, &(dw, dh)) in dims.iter().enumerate() {
        let buf: Cow<[u8]> = if di == 0 {
            Cow::Borrowed(&page.pixels)
        } else {
            Cow::Owned(downscale_gray(
                &page.pixels,
                page.width,
                page.height,
                dw,
                dh,
            ))
        };

        let hi_jpeg = encode_jpeg(buf.as_ref(), dw, dh, hi_q)?;
        let hi_b64 = base64_len(hi_jpeg.len());
        smallest = smallest.min(hi_b64);
        if hi_b64 <= budget.max_base64_bytes {
            return Ok(GcvImage {
                jpeg: hi_jpeg,
                quality: hi_q,
                width: dw,
                height: dh,
            });
        }

        let mut lo = lo_q;
        // hi_q was already encoded above and did not fit; search strictly below it.
        let mut hi = hi_q.saturating_sub(1);
        let mut best: Option<(u8, Vec<u8>)> = None;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let jpeg = encode_jpeg(buf.as_ref(), dw, dh, mid)?;
            let b64 = base64_len(jpeg.len());
            smallest = smallest.min(b64);
            if b64 <= budget.max_base64_bytes {
                best = Some((mid, jpeg));
                lo = mid + 1;
            } else if mid == 0 {
                break;
            } else {
                hi = mid - 1;
            }
        }
        if let Some((q, jpeg)) = best {
            return Ok(GcvImage {
                jpeg,
                quality: q,
                width: dw,
                height: dh,
            });
        }
    }

    Err(GcvError::Unfittable {
        smallest_base64: smallest,
        budget: budget.max_base64_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_match_gcv_limits() {
        let b = GcvBudget::default();
        assert_eq!(b.max_base64_bytes, 10 * 1024 * 1024, "GCV JSON ceiling");
        assert_eq!(b.min_quality, 60);
        assert_eq!(b.start_quality, 90);
    }

    #[test]
    fn to_base64_roundtrips() {
        let img = GcvImage {
            jpeg: vec![0xFF, 0xD8, 0xFF, 0xD9],
            quality: 90,
            width: 2,
            height: 2,
        };
        let b64 = img.to_base64();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .unwrap();
        assert_eq!(decoded, img.jpeg);
    }

    #[test]
    fn base64_len_proxy_matches_actual_encoding() {
        // The budget guarantee depends on base64_len() exactly predicting
        // GcvImage::to_base64().len(). If the base64 engine ever changes to a
        // non-padded variant this must fail loudly, not silently break budgets.
        for n in 0..=64usize {
            let img = GcvImage {
                jpeg: vec![0xAB; n],
                quality: 80,
                width: 1,
                height: 1,
            };
            assert_eq!(
                base64_len(n),
                img.to_base64().len(),
                "base64_len({n}) must equal actual encoded length"
            );
        }
        // Larger sizes: each residue class (0/1/2) appears at least twice at
        // different scales to confirm the formula holds beyond the dense loop.
        for n in [255usize, 256, 257, 1_023, 1_024, 1_025] {
            let img = GcvImage {
                jpeg: vec![0xAB; n],
                quality: 80,
                width: 1,
                height: 1,
            };
            assert_eq!(base64_len(n), img.to_base64().len(), "n={n}");
        }
    }

    /// Build a RenderedPage of `w`×`h` filled with a deterministic noisy
    /// pattern (uniform fills compress to almost nothing and won't exercise
    /// the step-down).
    fn noisy_page(w: u32, h: u32) -> RenderedPage {
        // Low-frequency, non-flat, deterministic: compresses like a real
        // document page (large smooth regions), unlike incompressible noise.
        let mut pixels = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                // Coarse gradient + gentle block structure; stays smooth so
                // the JPEG DCT can actually compress it.
                let v = ((x / 8 + y / 8) & 0x3F) as u8 + 96;
                pixels[(y * w + x) as usize] = v;
            }
        }
        RenderedPage {
            page_num: 1,
            width: w,
            height: h,
            pixels,
            dpi: 300.0,
            effective_dpi: 300.0,
            diagnostics: crate::PageDiagnostics::default(),
        }
    }

    #[test]
    fn small_page_fits_at_start_quality() {
        let page = noisy_page(64, 64);
        let img = encode_for_gcv(&page, &GcvBudget::default()).unwrap();
        assert_eq!(img.quality, 90, "small page must not trigger step-down");
        assert_eq!((img.width, img.height), (64, 64));
        assert!(img.to_base64().len() <= GcvBudget::default().max_base64_bytes);
        assert_eq!(&img.jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn step_down_is_deterministic() {
        let page = noisy_page(256, 256);
        // Measure the achievable size envelope at native resolution so the
        // budget provably lands in the step-down band (below q90, above the
        // q-floor) without forcing a downscale or an Unfittable.
        let big = encode_for_gcv(
            &page,
            &GcvBudget {
                max_base64_bytes: usize::MAX,
                min_quality: 90,
                start_quality: 90,
            },
        )
        .unwrap();
        let small = encode_for_gcv(
            &page,
            &GcvBudget {
                max_base64_bytes: usize::MAX,
                min_quality: 10,
                start_quality: 10,
            },
        )
        .unwrap();
        let big_len = big.to_base64().len();
        let small_len = small.to_base64().len();
        assert!(
            small_len < big_len,
            "fixture must be compressible: q10 ({small_len}) should be < q90 ({big_len})"
        );
        // A budget strictly between the two forces a step-down but stays fittable.
        let budget = GcvBudget {
            max_base64_bytes: (big_len + small_len) / 2,
            min_quality: 10,
            start_quality: 90,
        };
        let a = encode_for_gcv(&page, &budget).unwrap();
        let b = encode_for_gcv(&page, &budget).unwrap();
        assert_eq!(a.jpeg, b.jpeg, "fitting must be byte-deterministic");
        assert_eq!(a.quality, b.quality);
        assert!(a.to_base64().len() <= budget.max_base64_bytes);
        assert!(a.quality >= budget.min_quality);
        assert!(a.quality < 90, "budget below q90 must have stepped down");
        assert_eq!((a.width, a.height), (256, 256), "must not downscale here");
    }

    #[test]
    fn downscale_when_quality_floor_insufficient() {
        let page = noisy_page(4096, 2048);
        // Size at the quality floor, native resolution. A budget below this
        // cannot be met without downscaling.
        let floor_native = encode_for_gcv(
            &page,
            &GcvBudget {
                max_base64_bytes: usize::MAX,
                min_quality: 60,
                start_quality: 60,
            },
        )
        .unwrap();
        assert_eq!(
            (floor_native.width, floor_native.height),
            (4096, 2048),
            "control: unbounded budget keeps native resolution"
        );
        let floor_len = floor_native.to_base64().len();
        // Strictly below the floor-quality native size → must downscale.
        let budget = GcvBudget {
            max_base64_bytes: floor_len - 1,
            min_quality: 60,
            start_quality: 90,
        };
        let img = encode_for_gcv(&page, &budget).unwrap();
        assert!(img.to_base64().len() <= budget.max_base64_bytes);
        assert!(
            img.width < 4096,
            "expected downscale, got {}x{}",
            img.width,
            img.height
        );
        // Aspect ratio preserved (2:1) within integer-halving rounding.
        let ratio = f64::from(img.width) / f64::from(img.height);
        assert!((ratio - 2.0).abs() < 0.05, "aspect not preserved: {ratio}");
    }

    #[test]
    fn unfittable_returns_error_never_oversized_payload() {
        let page = noisy_page(512, 512);
        let budget = GcvBudget {
            max_base64_bytes: 1,
            min_quality: 60,
            start_quality: 90,
        };
        let result = encode_for_gcv(&page, &budget);
        assert!(
            matches!(result, Err(GcvError::Unfittable { .. })),
            "must error, never return an over-budget payload: {result:?}"
        );
    }

    #[test]
    fn never_emits_over_75mp() {
        // 9000x9000 = 81 MP exceeds GCV's 75 MP OCR limit. The algorithm
        // must downscale (native is filtered) and never hand back >75 MP.
        let page = noisy_page(9000, 9000);
        let budget = GcvBudget::default();
        match encode_for_gcv(&page, &budget) {
            Ok(img) => {
                assert!(
                    u64::from(img.width) * u64::from(img.height) <= 75_000_000,
                    "must never emit > 75 MP: {}x{}",
                    img.width,
                    img.height
                );
                assert!(
                    img.width < 9000 && img.height < 9000,
                    "native 81 MP must be filtered, not emitted: {}x{}",
                    img.width,
                    img.height
                );
            }
            Err(GcvError::Unfittable { .. }) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn never_downscales_below_ocr_floor() {
        // A page whose halving chain would cross the 1024 short-side floor:
        // 1500 -> 750. The fitted result must never have a short side < 1024;
        // if the budget can't be met at/above the floor, that is Unfittable.
        let page = noisy_page(1500, 1500);
        // Tiny budget: native 1500x1500 at min quality will not fit, forcing
        // the algorithm toward downscaling — which must NOT cross the floor.
        let budget = GcvBudget {
            max_base64_bytes: 1_000,
            min_quality: 60,
            start_quality: 90,
        };
        match encode_for_gcv(&page, &budget) {
            Ok(img) => assert!(
                img.width.min(img.height) >= 1024,
                "emitted short side {} below OCR floor 1024 ({}x{})",
                img.width.min(img.height),
                img.width,
                img.height
            ),
            Err(GcvError::Unfittable { .. }) => {} // acceptable: can't fit ≥ floor
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn oversized_side_is_unfittable_not_encode_error() {
        // 70000 px exceeds baseline JPEG's 65535-per-side ceiling while the
        // page stays under 75 MP. The short side (16) is below the OCR floor,
        // so no halving produces an encodable candidate. The contract is a
        // clean `Unfittable`, never a `GcvError::Encode` codec fault leaked
        // from `jpeg_gray`.
        let page = noisy_page(70_000, 16);
        match encode_for_gcv(&page, &GcvBudget::default()) {
            Err(GcvError::Unfittable { .. }) => {}
            other => panic!("expected Unfittable, got {other:?}"),
        }
    }

    #[test]
    fn degenerate_zero_dim_is_unfittable() {
        let page = noisy_page(0, 100);
        assert!(matches!(
            encode_for_gcv(&page, &GcvBudget::default()),
            Err(GcvError::Unfittable { .. })
        ));
    }
}
