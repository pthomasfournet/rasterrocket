// floor_div over a list of operand pairs, one pair per thread, for the
// tests that compare it with the host's floor division.
//
// Inputs:
//   pairs      four words per pair: a low, a high, b low, b high.
// Output:
//   quotients  two words per pair: the low and high word of
//              floor_div(a, b, max_steps).
//   status     floor_div's status words (floor_div.slang).

#define STATUS_BINDING 2
#include "floor_div.slang"

struct ProbeParams {
    uint n;          // pairs
    uint max_steps;  // floor_div's correction steps
};

[[vk::push_constant]] ConstantBuffer<ProbeParams> pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> pairs;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> quotients;

// The 64-bit integer of words lo, hi.
int64_t join(uint lo, uint hi)
{
    return int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void floor_div_probe(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.n) return;
    int64_t a = join(pairs[i * 4], pairs[i * 4 + 1]);
    int64_t b = join(pairs[i * 4 + 2], pairs[i * 4 + 3]);
    uint64_t q = uint64_t(floor_div(a, b, pc.max_steps));
    quotients[i * 2] = uint(q);
    quotients[i * 2 + 1] = uint(q >> 32);
}
