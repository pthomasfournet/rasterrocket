//! Deterministic inputs shared by the page raster's tests.

/// `SplitMix64`.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + i64::try_from(self.next() % u64::try_from(hi - lo).unwrap()).unwrap()
    }

    /// Uniform in `[0, n)`.
    pub(crate) fn below(&mut self, n: u32) -> i32 {
        i32::try_from(self.next() % u64::from(n)).unwrap()
    }

    pub(crate) fn byte(&mut self) -> u8 {
        u8::try_from(self.below(256)).unwrap()
    }

    /// A sample of at most `max`.
    pub(crate) fn sample(&mut self, max: u16) -> u16 {
        u16::try_from(self.below(u32::from(max) + 1)).unwrap()
    }
}

/// The closed polygon through `points`, as segments.
pub(crate) fn polygon(points: &[(i64, i64)]) -> Vec<[i64; 4]> {
    (0..points.len())
        .map(|i| {
            let (a, b) = (points[i], points[(i + 1) % points.len()]);
            [a.0, a.1, b.0, b.1]
        })
        .collect()
}
