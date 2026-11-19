// Floor division of 64-bit integers without a 64-bit division, and the
// status buffer through which a kernel reports a division it could not
// complete exactly.
//
// The includer defines STATUS_BINDING, the binding of set 0 the status
// buffer takes, before including this file.
//
// Status words, zero before the dispatch:
//   [0]        the fault code: 0 for none, FAULT_FLOOR_DIV;
//   [1], [2]   the low and high words of the faulting division's a;
//   [3], [4]   the low and high words of its b.
// The first faulting thread claims word 0 by compare-exchange and alone
// writes words 1 to 4; a later fault leaves them unchanged.

#ifndef STATUS_BINDING
#error "define STATUS_BINDING before including floor_div.slang"
#endif

static const uint FAULT_FLOOR_DIV = 1;

// The correction steps floor_div takes in each direction: the two
// estimates leave the quotient within 2 of floor(a / b) on a device whose
// single-precision division is correctly rounded.
static const uint FLOOR_DIV_STEPS = 16;

[[vk::binding(STATUS_BINDING, 0)]] RWStructuredBuffer<uint> status;

// Record a fault of floor_div(a, b) unless an earlier one is recorded.
void floor_div_fault(int64_t a, int64_t b)
{
    uint prev;
    InterlockedCompareExchange(status[0], 0, FAULT_FLOOR_DIV, prev);
    if (prev == 0) {
        uint64_t ua = uint64_t(a);
        uint64_t ub = uint64_t(b);
        status[1] = uint(ua);
        status[2] = uint(ua >> 32);
        status[3] = uint(ub);
        status[4] = uint(ub >> 32);
    }
}

// floor(a / b) for 0 < b < 2^32 and |a / b| < 2^40: two single-precision
// estimates of the quotient, each followed by its exact remainder, then
// at most max_steps unit corrections in each direction.  The first
// estimate is within |a / b| * 2^-23 + 1 of the quotient, so the second
// divides a remainder below 2^18 * b and is within 2 of it.  Both
// divisions are precise: a compiler may not substitute an approximate
// reciprocal.  When the corrections leave the remainder outside [0, b),
// the division is recorded as a fault in the status buffer and the
// returned quotient is not floor(a / b).
int64_t floor_div(int64_t a, int64_t b, uint max_steps)
{
    precise float fb = float(b);
    precise float e0 = float(a) / fb;
    int64_t q = int64_t(e0);
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
    if (r < 0 || r >= b) floor_div_fault(a, b);
    return q;
}
