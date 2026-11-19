// Floor division of 64-bit integers without a 64-bit division, and the
// status buffer through which a kernel reports a division it could not
// complete exactly.
//
// The includer defines STATUS_BINDING, the binding of set 0 the status
// buffer takes, before including this file; the build defines
// STATUS_WORDS, FLOOR_DIV_STEPS, FAULT_FLOOR_DIV and FAULT_FLOOR_DIV_WRAP
// (kernels/floor_div_params.rs).
//
// Status words, zero before the dispatch:
//   [0]        the fault code: 0 for none, FAULT_FLOOR_DIV when the
//              corrections leave the remainder outside [0, b),
//              FAULT_FLOOR_DIV_WRAP when the quotient's product wrapped;
//   [1], [2]   the low and high words of the faulting division's a;
//   [3], [4]   the low and high words of its b.
// The first faulting thread claims word 0 by compare-exchange and alone
// writes words 1 to 4; a later fault leaves them unchanged.

#ifndef STATUS_BINDING
#error "define STATUS_BINDING before including floor_div.slang"
#endif
#if !defined(FLOOR_DIV_STEPS) || !defined(FAULT_FLOOR_DIV) || !defined(FAULT_FLOOR_DIV_WRAP)
#error "the build passes FLOOR_DIV_STEPS and the fault codes (kernels/floor_div_params.rs)"
#endif
#if !defined(STATUS_WORDS) || STATUS_WORDS != 5
#error "the host's status buffer (STATUS_WORDS, kernels/floor_div_params.rs) is not the five words written below"
#endif

// |q * b - a| below which a quotient q is taken not to have wrapped: 2^62.
static const float FLOOR_DIV_WRAP = 4611686018427387904.0;

[[vk::binding(STATUS_BINDING, 0)]] RWStructuredBuffer<uint> status;

// Record the fault `code` of floor_div(a, b) unless an earlier one is
// recorded.
void floor_div_fault(uint code, int64_t a, int64_t b)
{
    uint prev;
    InterlockedCompareExchange(status[0], 0, code, prev);
    if (prev == 0) {
        uint64_t ua = uint64_t(a);
        uint64_t ub = uint64_t(b);
        status[1] = uint(ua);
        status[2] = uint(ua >> 32);
        status[3] = uint(ub);
        status[4] = uint(ub >> 32);
    }
}

// floor(a / b) for 0 < b < 2^32 and |a / b| < 2^40, |a| <= 2^62: two
// single-precision estimates of the quotient, each followed by its
// remainder, then at most max_steps unit corrections in each direction.
//
// Error bound, for a division within 2.5 ULP (the bound both Vulkan and
// D3D12 allow; the divisions are precise, so no compiler substitutes an
// approximate reciprocal): float(a) and float(b) each round by at most
// 2^-24 relative, the quotient by 2.5 * 2^-23, so the first estimate is
// within |a / b| * 3.5 * 2^-23 + 1 < 2^18.9 of a / b, and the remainder
// it leaves is below 2^18.9 * b < 2^51.  The second estimate, of that
// remainder, errs by less than 2^18.9 * 3.5 * 2^-23 < 1 before its
// truncation, so it leaves the quotient within 2, well inside the
// max_steps corrections.
//
// The result is checked, and a failed check is recorded as a fault in
// the status buffer (the returned quotient is then not floor(a / b)).
// The remainder r = a - q * b is computed modulo 2^64, so r in [0, b)
// alone proves q * b = a - r + m * 2^64 for some integer m; the wrap
// guard, |float(q) * float(b) - float(a)| < 2^62, rejects every m != 0
// and every NaN, and passes a correct quotient.  Five roundings enter
// the guard: float(q), float(b), their product, float(a) and the
// difference.  Each is at most 2^-24 relative to its operand, and the
// operands are capped by |a| <= 2^63 (any int64) and |q * b| <=
// |q * b - a| + 2^63; so they move the difference by less than
// 2^-21 * |q * b - a| + 2^43.  For m != 0, |q * b - a| >= 2^64 - b
// stays far above 2^62 after that; for a correct quotient,
// |q * b - a| = r < b < 2^32 stays far below it.  Both checks together
// prove q * b - a = -r exactly, so q = floor(a / b).
int64_t floor_div(int64_t a, int64_t b, uint max_steps)
{
    precise float fb = float(b);
    precise float e0 = float(a) / fb;
    int64_t q = int64_t(e0);
#ifdef FLOOR_DIV_POISON
    // Test builds only: the first estimate is off by floor((2^64 - 1) / b),
    // which wraps q * b and leaves the remainder nearly unchanged.
    q += int64_t(~uint64_t(0) / uint64_t(b));
#endif
    int64_t r = a - q * b;
    precise float e1 = float(r) / fb;
    q += int64_t(e1);
    r = a - q * b;
    for (uint s = 0; s < max_steps && r < 0; s++) {
        q -= 1;
        r += b;
    }
    for (uint s = 0; s < max_steps && r >= b; s++) {
        q += 1;
        r -= b;
    }
    precise float wrap = abs(float(q) * fb - float(a));
    if (r < 0 || r >= b) {
        floor_div_fault(FAULT_FLOOR_DIV, a, b);
    } else if (!(wrap < FLOOR_DIV_WRAP)) {
        floor_div_fault(FAULT_FLOOR_DIV_WRAP, a, b);
    }
    return q;
}
