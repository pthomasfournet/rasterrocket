//! The status words a kernel writes when an operation it performs cannot be
//! completed exactly (`kernels/floor_div.slang`), and their decoding.
//!
//! The words are zero before the dispatch.  Word 0 holds the fault code,
//! zero for none; for [`FAULT_FLOOR_DIV`] and [`FAULT_FLOOR_DIV_WRAP`],
//! words 1 and 2 hold the low and high words of the dividend and words 3
//! and 4 those of the divisor, of the first faulting division.

use super::{BackendError, Result};

include!("../../kernels/floor_div_params.rs");

/// The bytes of a kernel's status buffer.
pub(crate) const STATUS_BYTES: usize = STATUS_WORDS * 4;

/// The status words of the status buffer's bytes `bytes`, in the device's
/// byte order, which is the host's.
pub(crate) fn status_words(bytes: &[u8; STATUS_BYTES]) -> [u32; STATUS_WORDS] {
    let mut words = [0; STATUS_WORDS];
    for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *word = u32::from_ne_bytes(*chunk);
    }
    words
}

/// The 64-bit integer of the words `lo`, `hi`.
pub(crate) fn join(lo: u32, hi: u32) -> i64 {
    (u64::from(lo) | u64::from(hi) << 32).cast_signed()
}

/// The operands `(a, b)` of the faulting floor division `words` record,
/// for either floor-division fault code; `None` when they record no
/// floor-division fault.
pub(crate) fn floor_div_fault(words: &[u32; STATUS_WORDS]) -> Option<(i64, i64)> {
    matches!(words[0], FAULT_FLOOR_DIV | FAULT_FLOOR_DIV_WRAP)
        .then(|| (join(words[1], words[2]), join(words[3], words[4])))
}

/// `Ok` when the status words of `kernel`, whose floor divisions take
/// [`FLOOR_DIV_STEPS`] correction steps, record no fault.
///
/// # Errors
/// [`BackendError::KernelFault`] naming `kernel` and the fault: a floor
/// division's operands and whether its corrections ran out or its
/// product wrapped, or an unknown fault code.
pub(crate) fn check(kernel: &'static str, words: &[u32; STATUS_WORDS]) -> Result<()> {
    if words[0] == 0 {
        return Ok(());
    }
    let detail = match (words[0], floor_div_fault(words)) {
        (FAULT_FLOOR_DIV, Some((a, b))) => format!(
            "floor_div(a = {a}, b = {b}) did not converge within {FLOOR_DIV_STEPS} \
             correction steps; the device's float division is unreliable"
        ),
        (_, Some((a, b))) => format!(
            "floor_div(a = {a}, b = {b}) produced a quotient whose product with b wrapped \
             64 bits; the device's float division is unreliable"
        ),
        (code, None) => format!("unknown fault code {code}"),
    };
    Err(BackendError::KernelFault { kernel, detail })
}

/// Operand pairs of floor division over its domain `0 < b < 2^32`,
/// `|a / b| < 2^40`, `|a| <= 2^62`: the edge cases, then 65 536 pairs drawn
/// from `rng`.
///
/// The edge cases are every `b` of `{1, 2, 3, 255, 256, 2^31 - 1, 2^31,
/// 2^32 - 1}` with every `a` of `{0, ±1, ±(b - 1), ±b, ±(b + 1), ±k·b,
/// ±(k·b + b - 1)}`, `k ∈ {1, 2^20, 2^40 - 1}`; an `a` past the domain is
/// replaced by the nearest one inside.  The drawn pairs take `b` and the
/// bound of `|a|` of uniformly random bit lengths, so that small and large
/// operands are equally frequent.
#[cfg(test)]
pub(crate) fn floor_div_cases(rng: &mut crate::page::test_support::Rng) -> Vec<(i64, i64)> {
    const QUOTIENT_LIMIT: i128 = 1 << 40;
    const A_LIMIT: i128 = 1 << 62;
    /// The largest `|a|` of the domain for divisor `b`.
    fn a_limit(b: i64) -> i128 {
        (QUOTIENT_LIMIT * i128::from(b) - 1).min(A_LIMIT)
    }
    let divisors = [1, 2, 3, 255, 256, (1 << 31) - 1, 1 << 31, (1 << 32) - 1];
    let mut cases = Vec::new();
    for b in divisors {
        let limit = a_limit(b);
        let wide = i128::from(b);
        let mut magnitudes = vec![0, 1, wide - 1, wide, wide + 1];
        for k in [1, 1 << 20, QUOTIENT_LIMIT - 1] {
            magnitudes.extend([k * wide, k * wide + wide - 1]);
        }
        for m in magnitudes {
            for a in [m, -m] {
                let a = i64::try_from(a.clamp(-limit, limit)).expect("the limit fits i64");
                cases.push((a, b));
            }
        }
    }
    for _ in 0..65_536 {
        let b_bits = rng.range(1, 33);
        let b = rng.range(1, 1 << b_bits);
        let a_bits = rng.range(0, 63);
        let bound = i64::try_from(a_limit(b).min(1 << a_bits)).expect("the bound fits i64");
        let magnitude = rng.range(0, bound + 1);
        let a = if rng.next() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        cases.push((a, b));
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_div_fault_decodes_the_operands() {
        assert_eq!(
            floor_div_fault(&[1, 0xFFFF_FFFF, 0xFFFF_FFFF, 3, 0]),
            Some((-1, 3))
        );
        assert_eq!(floor_div_fault(&[0; STATUS_WORDS]), None);
    }

    #[test]
    fn a_clear_status_passes_and_an_unknown_code_is_a_kernel_fault() {
        assert!(check("k", &[0; STATUS_WORDS]).is_ok());
        let err = check("k", &[7, 0, 0, 0, 0]).unwrap_err();
        assert!(matches!(err, BackendError::KernelFault { kernel: "k", .. }));
        assert_eq!(err.to_string(), "k: unknown fault code 7");
    }

    #[test]
    fn each_floor_div_fault_code_has_its_message() {
        let words = |code| [code, 7, 0, 3, 0];
        assert_eq!(floor_div_fault(&words(FAULT_FLOOR_DIV_WRAP)), Some((7, 3)));
        let steps = check("k", &words(FAULT_FLOOR_DIV)).unwrap_err().to_string();
        assert_eq!(
            steps,
            "k: floor_div(a = 7, b = 3) did not converge within 16 correction steps; \
             the device's float division is unreliable"
        );
        let wrap = check("k", &words(FAULT_FLOOR_DIV_WRAP))
            .unwrap_err()
            .to_string();
        assert_eq!(
            wrap,
            "k: floor_div(a = 7, b = 3) produced a quotient whose product with b wrapped \
             64 bits; the device's float division is unreliable"
        );
    }

    #[test]
    fn status_words_read_the_bytes_in_order() {
        let mut bytes = [0; STATUS_BYTES];
        for (i, chunk) in bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *chunk = (u32::try_from(i).unwrap() + 1).to_ne_bytes();
        }
        assert_eq!(status_words(&bytes), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn floor_div_cases_lie_in_the_domain() {
        let cases = floor_div_cases(&mut crate::page::test_support::Rng(1));
        assert!(cases.len() > 65_536);
        for &(a, b) in &cases {
            assert!(0 < b && b < 1 << 32, "{a} / {b}");
            assert!(a.unsigned_abs() <= 1 << 62, "{a} / {b}");
            assert!(a.div_euclid(b).unsigned_abs() <= 1 << 40, "{a} / {b}");
        }
        // The clamp reaches both of the domain's bounds.
        assert!(cases.contains(&((1 << 40) - 1, 1)));
        assert!(cases.contains(&(1 << 62, (1 << 32) - 1)));
    }
}
