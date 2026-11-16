//! The memoised per-pixel conversion of samples through a space: each
//! pixel exactly while the work fits the image's allowance, each
//! distinct tuple once past that, otherwise through a lattice of exact
//! conversions.

use std::collections::HashMap;

use super::decoded::Decoded;
use crate::resources::image::try_with_capacity;
use crate::resources::lattice::{Axis, for_each_corner};

/// Distinct sample tuples an image past the per-pixel budget may hold and
/// still convert each exactly; bounds the memory of that table.
const DISTINCT_LIMIT: usize = 1 << 20;

/// The lattice sizes, values per component, largest first: each `g − 1`
/// divides 255, so every node is a sample byte.  A conversion of up to four
/// components starts at the first, of five or six at the third, of seven
/// or eight at the fourth.
const LATTICE_GRIDS: [usize; 5] = [18, 16, 6, 4, 2];

/// The most components a lattice conversion takes: interpolating visits
/// `2^n` nodes per distinct tuple.
const LATTICE_MAX_COMPONENTS: usize = 8;

/// Distinct sample tuples memoised before the table is reset; bounds the
/// memory of a continuous-tone image's per-pixel conversion.  Also the
/// slot count of the [`TupleCache`] of a tuple of at most eight samples.
const MEMO_LIMIT: usize = 1 << 16;

/// A direct-mapped cache of the colours of sample tuples of at most eight
/// bytes, each packed into a `u64`: a tuple occupies the slot its key
/// hashes to, replacing the one there.  Conversion is a function of the
/// tuple, so a replaced entry is only recomputed, never stale.
struct TupleCache {
    /// `(key, [r, g, b, filled])` per slot.
    slots: Vec<(u64, [u8; 4])>,
    /// `64 − log2(slots.len())`: the hash's high bits index the slot.
    shift: u32,
}

impl TupleCache {
    /// A cache of at least 16 and at most [`MEMO_LIMIT`] slots, sized to
    /// `tuples`, the tuples it will be asked for, rounded up to a power of
    /// two; `None` when the table cannot be allocated.
    fn new(tuples: usize) -> Option<Self> {
        let len = tuples.clamp(16, MEMO_LIMIT).next_power_of_two();
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(len)
            .map_err(|e| log::warn!("image: cannot allocate a {len}-slot colour cache: {e}"))
            .ok()?;
        slots.resize(len, (0, [0; 4]));
        Some(Self {
            slots,
            shift: 64 - len.trailing_zeros(),
        })
    }

    /// The colour of `px`, from the cache or from `convert`.
    fn get_or_insert(&mut self, px: &[u8], convert: impl FnOnce() -> [u8; 3]) -> [u8; 3] {
        let key = px.iter().fold(0u64, |k, &b| (k << 8) | u64::from(b));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the shifted hash is below slots.len() ≤ MEMO_LIMIT"
        )]
        let slot =
            &mut self.slots[(key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize];
        if slot.1[3] != 0 && slot.0 == key {
            return [slot.1[0], slot.1[1], slot.1[2]];
        }
        let [r, g, b] = convert();
        *slot = (key, [r, g, b, 1]);
        [r, g, b]
    }
}

/// The exact colours at the nodes of a regular grid of `g` values per
/// component, `0, 255 / (g − 1), …, 255`, the first component varying
/// fastest.
pub(super) struct Lattice {
    g: usize,
    nodes: Vec<[u8; 3]>,
}

impl Lattice {
    /// The multilinear interpolation of the nodes around the sample tuple
    /// `px`, rounded; `axes` is scratch space.
    pub(super) fn interpolate(&self, px: &[u8], axes: &mut Vec<Axis>) -> [u8; 3] {
        let g = self.g;
        #[expect(clippy::cast_precision_loss, reason = "g ≤ 18")]
        let scale = (g - 1) as f64 / 255.0;
        axes.clear();
        let mut stride = 1;
        for &s in px {
            axes.push(Axis::at(f64::from(s) * scale, g, stride));
            stride *= g;
        }
        let mut acc = [0.0f64; 3];
        for_each_corner(axes, |index, weight| {
            for (a, &c) in acc.iter_mut().zip(&self.nodes[index]) {
                *a = weight.mul_add(f64::from(c), *a);
            }
        });
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a convex combination of bytes, rounded, is a byte"
        )]
        acc.map(|a| a.round().clamp(0.0, 255.0) as u8)
    }
}

impl Decoded<'_> {
    /// `n`-component samples through the space.
    ///
    /// Each pixel converts exactly when the work that takes fits
    /// [`Self::allowance`]: every pixel's conversion
    /// ([`Self::convert_cached`]), or, past that, each distinct sample
    /// tuple's once ([`Self::convert_distinct`]) when there are at most
    /// [`DISTINCT_LIMIT`] of them.  Otherwise the image converts through a
    /// lattice of exact conversions ([`Self::convert_lattice`]).  `None`
    /// when the output cannot be allocated or no lattice fits the
    /// allowance.
    pub(super) fn convert_memo(&self, bytes: &[u8], n: usize) -> Option<Vec<u8>> {
        let exact_conversions = self.allowance / self.cost;
        let pixels = bytes.len() / n;
        if u64::try_from(pixels).is_ok_and(|p| p <= exact_conversions) {
            return self.convert_cached(bytes, n);
        }
        let limit =
            usize::try_from(exact_conversions).map_or(DISTINCT_LIMIT, |e| e.min(DISTINCT_LIMIT));
        if let Some(out) = self.convert_distinct(bytes, n, limit) {
            return Some(out);
        }
        self.convert_lattice(bytes, n)
    }

    /// Each distinct `n`-sample tuple of `bytes` converted once, when there
    /// are at most `limit` of them; `None` past the limit or when the
    /// output cannot be allocated.
    fn convert_distinct(&self, bytes: &[u8], n: usize, limit: usize) -> Option<Vec<u8>> {
        let mut colours: HashMap<&[u8], [u8; 3]> = HashMap::new();
        for px in bytes.chunks_exact(n) {
            if colours.len() == limit && !colours.contains_key(px) {
                return None;
            }
            let _ = colours.entry(px).or_insert([0; 3]);
        }
        let mut comps = Vec::with_capacity(n);
        for (px, rgb) in &mut colours {
            *rgb = self.to_rgb_with(px, &mut comps);
        }
        let mut out = try_with_capacity(bytes.len() / n * 3)?;
        for px in bytes.chunks_exact(n) {
            out.extend_from_slice(&colours[px]);
        }
        Some(out)
    }

    /// The lattice of `n`-component samples: the space converted exactly
    /// at the nodes of a regular grid of `g` values per component, the
    /// largest of [`LATTICE_GRIDS`] for `n` whose `g^n` conversions fit
    /// [`Self::allowance`].  `None` (logged) when `n` exceeds
    /// [`LATTICE_MAX_COMPONENTS`], when not even a grid of two values per
    /// component fits, and when the nodes cannot be allocated.
    pub(super) fn lattice(&self, n: usize) -> Option<Lattice> {
        let start = match n {
            0 => return None,
            1..=4 => 0,
            5..=6 => 2,
            7..=LATTICE_MAX_COMPONENTS => 3,
            _ => {
                log::warn!(
                    "image: converting {n}-component samples exactly exceeds the work budget, \
                     and no lattice is kept past {LATTICE_MAX_COMPONENTS} components; image dropped"
                );
                return None;
            }
        };
        if let Some(page) = self.limiting {
            page.curtail();
        }
        let fits = |g: usize| {
            u32::try_from(n)
                .ok()
                .and_then(|n| (g as u64).checked_pow(n))
                .and_then(|nodes| nodes.checked_mul(self.cost))
                .is_some_and(|work| work <= self.allowance)
        };
        let Some(&g) = LATTICE_GRIDS[start..].iter().find(|&&g| fits(g)) else {
            log::warn!(
                "image: converting {n}-component samples at {} operations each exceeds the \
                 work left ({}) even through a lattice of 2 values per component; image dropped",
                self.cost,
                self.allowance,
            );
            return None;
        };
        log::warn!(
            "image: converting {n}-component samples exactly at {} operations each exceeds \
             the work budget; colours are interpolated in a lattice of {g} values per component",
            self.cost
        );
        let step = 255 / (g - 1);
        let count = g.pow(u32::try_from(n).ok()?);
        let mut nodes = Vec::new();
        nodes.try_reserve_exact(count).ok()?;
        let mut comps = Vec::with_capacity(n);
        let mut node = vec![0u8; n];
        for index in 0..count {
            let mut rest = index;
            for sample in &mut node {
                *sample = u8::try_from(rest % g * step).ok()?;
                rest /= g;
            }
            nodes.push(self.to_rgb_with(&node, &mut comps));
        }
        Some(Lattice { g, nodes })
    }

    /// `n`-component samples through [`Self::lattice`]: each pixel takes
    /// the multilinear interpolation of the nodes around its samples.
    /// `None` when there is no lattice or the output cannot be allocated.
    pub(super) fn convert_lattice(&self, bytes: &[u8], n: usize) -> Option<Vec<u8>> {
        let lattice = self.lattice(n)?;
        let mut axes = Vec::with_capacity(n);
        let mut out = try_with_capacity(bytes.len() / n * 3)?;
        let mut cache = TupleCache::new(bytes.len() / n)?;
        for px in bytes.chunks_exact(n) {
            out.extend_from_slice(&cache.get_or_insert(px, || lattice.interpolate(px, &mut axes)));
        }
        Some(out)
    }

    /// `n`-component samples through the space, a tuple converted again
    /// only once the cache has dropped it: a [`TupleCache`] for tuples of
    /// at most eight samples, otherwise a table of up to [`MEMO_LIMIT`]
    /// tuples reset when full.  `None` when the output cannot be allocated.
    fn convert_cached(&self, bytes: &[u8], n: usize) -> Option<Vec<u8>> {
        let pixels = bytes.len() / n;
        let mut out = try_with_capacity(pixels * 3)?;
        let mut comps = Vec::with_capacity(n);
        if n <= size_of::<u64>() {
            let mut cache = TupleCache::new(pixels)?;
            for px in bytes.chunks_exact(n) {
                let rgb = cache.get_or_insert(px, || self.to_rgb_with(px, &mut comps));
                out.extend_from_slice(&rgb);
            }
            return Some(out);
        }
        let mut memo: HashMap<Box<[u8]>, [u8; 3]> = HashMap::new();
        for px in bytes.chunks_exact(n) {
            let rgb = if let Some(&rgb) = memo.get(px) {
                rgb
            } else {
                if memo.len() >= MEMO_LIMIT {
                    memo.clear();
                }
                let rgb = self.to_rgb_with(px, &mut comps);
                let _ = memo.insert(px.into(), rgb);
                rgb
            };
            out.extend_from_slice(&rgb);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::resources::colorspace::EXACT_WORK_BUDGET;
    use crate::resources::image::ImageColorSpace;
    use crate::resources::image::colorspace::Layout;
    use crate::resources::image::colorspace::test_support::{
        convert, devicen_cmyk, devicen_teal, exact, noise, smooth_program, teal_rgb, unit_decoded,
    };

    #[test]
    fn multi_colorant_devicen_converts_each_pixel() {
        let (doc, cs) = devicen_teal();
        // Pixels (Black, Spot): spot only, no ink, black only.
        let img = convert(&doc, cs, &[], Layout::Color(2), &[0, 255, 0, 0, 255, 0]);
        assert_eq!(img.color_space, ImageColorSpace::Rgb);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(px[0..3], teal_rgb());
        assert_eq!(px[3..6], [255, 255, 255]);
        assert_eq!(px[6..9], color::cmyk::cmyk_f64_to_rgb(0.0, 0.0, 0.0, 1.0));
    }

    /// Every pixel of a multi-colorant image converts as its tuple does
    /// alone, whether the tuple is cached, replaces another in the cache or
    /// is converted afresh: four colorants take the packed-tuple cache,
    /// nine the table of byte tuples.
    #[test]
    fn cached_devicen_conversion_matches_per_tuple_conversion() {
        let cases: [(u8, &str); 2] = [
            (
                4,
                "{ 0.9 mul 4 1 roll 0.8 mul 4 1 roll 0.7 mul 4 1 roll 0.6 mul 4 1 roll }",
            ),
            (9, "{ pop pop pop pop pop }"),
        ];
        for (n, program) in cases {
            let width = usize::from(n);
            let (doc, cs) = devicen_cmyk(width, program);
            let space = crate::resources::colorspace::resolve(&doc, &cs);
            // 3000 pixels over 1000 distinct tuples: repeats, and more
            // tuples than the smallest cache has slots.
            let tuples = noise(width, 1000);
            let data: Vec<u8> = (0..3000)
                .flat_map(|i| tuples[i * 7 % 1000 * width..][..width].to_vec())
                .collect();
            let img = convert(&doc, cs, &[], Layout::Color(n), &data);
            assert_eq!(img.color_space, ImageColorSpace::Rgb);
            let px = img.data.as_cpu().unwrap();
            for (i, tuple) in data.chunks_exact(width).enumerate() {
                let want = exact(&doc, &space, tuple);
                assert_eq!(px[i * 3..i * 3 + 3], want, "n = {n}, pixel {i}: {tuple:?}");
            }
        }
    }

    #[test]
    fn a_lattice_conversion_interpolates_the_exact_one() {
        // Each grid the budget can select, on a smooth transform: nodes
        // convert exactly and every pixel stays within a few levels.
        let (doc, cs) = devicen_cmyk(4, &smooth_program(4, 0));
        let space = crate::resources::colorspace::resolve(&doc, &cs);
        let mut decoded = unit_decoded(&doc, &space, 4);
        let data = noise(4, 4000);
        for (g, tolerance) in [(18_u8, 2), (6, 8)] {
            decoded.allowance = u64::from(g).pow(4) * decoded.cost;
            let out = decoded.convert_lattice(&data, 4).expect("converts");
            let worst = data
                .as_chunks::<4>()
                .0
                .iter()
                .zip(out.as_chunks::<3>().0)
                .map(|(px, rgb)| {
                    let want = exact(&doc, &space, px);
                    (0..3).map(|c| rgb[c].abs_diff(want[c])).max().unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            assert!(worst <= tolerance, "g = {g}: off by {worst}");
            let step = 255 / (g - 1);
            let node = [0, step, 2 * step, 255];
            let at_node = decoded.convert_lattice(&node, 4).expect("converts");
            assert_eq!(at_node, exact(&doc, &space, &node), "g = {g}");
        }
    }

    #[test]
    fn costly_conversions_stay_exact_while_their_tuples_fit_the_budget() {
        // 8000 pixels of ten tuples through a 40 000-token program: past
        // the per-pixel budget, within the per-tuple one, so exact.
        let (doc, cs) = devicen_cmyk(4, &smooth_program(4, 20_000));
        let space = crate::resources::colorspace::resolve(&doc, &cs);
        let tuples = noise(4, 10);
        let data: Vec<u8> = (0..8000)
            .flat_map(|i| tuples[i % 10 * 4..i % 10 * 4 + 4].to_vec())
            .collect();
        assert!(8000 * space.conversion_cost() > EXACT_WORK_BUDGET);
        let img = convert(&doc, cs, &[], Layout::Color(4), &data);
        let px = img.data.as_cpu().unwrap();
        for (i, tuple) in data.as_chunks::<4>().0.iter().enumerate().step_by(97) {
            assert_eq!(
                px[i * 3..i * 3 + 3],
                exact(&doc, &space, tuple),
                "pixel {i}"
            );
        }
    }

    #[test]
    fn costly_conversions_of_many_tuples_go_through_a_lattice() {
        // Seven colorants through a 20 000-token program: 14 000 distinct
        // tuples exceed the budget, which then holds a lattice of two
        // values per colorant.  Its nodes convert exactly.
        let (doc, cs) = devicen_cmyk(7, &smooth_program(7, 10_000));
        let space = crate::resources::colorspace::resolve(&doc, &cs);
        let nodes = [[0u8; 7], [255; 7], [0, 255, 0, 255, 0, 255, 0]];
        let mut data = noise(7, 14_000);
        for (i, node) in nodes.iter().enumerate() {
            data[i * 7..i * 7 + 7].copy_from_slice(node);
        }
        let img = convert(&doc, cs, &[], Layout::Color(7), &data);
        let px = img.data.as_cpu().unwrap();
        assert_eq!(px.len(), 14_000 * 3);
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(px[i * 3..i * 3 + 3], exact(&doc, &space, node), "node {i}");
        }
    }

    #[test]
    fn no_lattice_past_eight_costly_colorants() {
        let (doc, cs) = devicen_cmyk(9, &smooth_program(9, 10_000));
        let space = crate::resources::colorspace::resolve(&doc, &cs);
        let decoded = unit_decoded(&doc, &space, 9);
        assert!(decoded.convert_lattice(&noise(9, 10), 9).is_none());
    }
}
