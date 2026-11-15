//! The 64-bit FNV-1a hash.
//!
//! A byte-serial, platform-independent hash: the digest of a byte sequence
//! is the same on every target and in every build, which makes it fit for
//! digests that are stored and compared (pinned test digests, resource
//! names derived from content, trace fingerprints).  It is not
//! collision-resistant against an adversary.

/// The FNV-1a offset basis for 64 bits.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// The FNV prime for 64 bits.
const PRIME: u64 = 0x0100_0000_01b3;

/// A running 64-bit FNV-1a hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fnv1a(u64);

impl Fnv1a {
    /// The hash of the empty sequence.
    #[must_use]
    pub const fn new() -> Self {
        Self(OFFSET_BASIS)
    }

    /// Extend the hashed sequence by `bytes`.
    pub fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(PRIME);
        }
    }

    /// The hash of the sequence so far.
    #[must_use]
    pub const fn finish(self) -> u64 {
        self.0
    }

    /// The hash of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> u64 {
        let mut h = Self::new();
        h.update(bytes);
        h.finish()
    }
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Fnv1a;

    /// The published FNV-1a 64 test vectors.
    #[test]
    fn matches_the_reference_vectors() {
        assert_eq!(Fnv1a::of(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(Fnv1a::of(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(Fnv1a::of(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn hashing_in_pieces_equals_hashing_at_once() {
        let mut h = Fnv1a::new();
        h.update(b"foo");
        h.update(b"bar");
        assert_eq!(h.finish(), Fnv1a::of(b"foobar"));
    }
}
