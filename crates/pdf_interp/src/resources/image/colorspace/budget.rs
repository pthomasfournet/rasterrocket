//! The tint-transform work the images of one page may spend.

use std::cell::Cell;

/// Interpreted operations the images of one page may spend through tint
/// transforms together: sixteen images' [`EXACT_WORK_BUDGET`].
///
/// [`EXACT_WORK_BUDGET`]: crate::resources::colorspace::EXACT_WORK_BUDGET
const PAGE_TINT_WORK_BUDGET: u64 = 1 << 32;

/// The tint-transform work left to the images of one page, in interpreted
/// operations (see [`ColorSpace::conversion_cost`]), starting at
/// [`PAGE_TINT_WORK_BUDGET`].
///
/// Each image converted through a space whose conversion costs more than
/// one operation spends the conversions it makes at that cost.  An image
/// may spend at most the smaller of [`EXACT_WORK_BUDGET`] and what is
/// left: once the page has spent most of its budget, an image converts
/// through a coarser lattice, and once a lattice of two values per
/// component no longer fits, it is not drawn.  Every such fallback is
/// logged.
///
/// A conversion the page's remaining work curtailed (a lattice or a drop
/// under an allowance below [`EXACT_WORK_BUDGET`]) depends on what the
/// page drew before it; [`Self::curtailments`] counts them, so that such
/// a decode is not shared with other pages.
///
/// [`ColorSpace::conversion_cost`]: crate::resources::colorspace::ColorSpace::conversion_cost
/// [`EXACT_WORK_BUDGET`]: crate::resources::colorspace::EXACT_WORK_BUDGET
#[derive(Debug)]
pub struct TintBudget {
    left: Cell<u64>,
    curtailed: Cell<u64>,
}

impl TintBudget {
    /// The full budget of one page.
    #[must_use]
    pub(crate) const fn page() -> Self {
        Self::with_work(PAGE_TINT_WORK_BUDGET)
    }

    /// A budget of `work` operations.
    pub(crate) const fn with_work(work: u64) -> Self {
        Self {
            left: Cell::new(work),
            curtailed: Cell::new(0),
        }
    }

    /// The work left.
    pub(super) const fn remaining(&self) -> u64 {
        self.left.get()
    }

    /// Spend `work`; the budget does not fall below zero.
    pub(super) fn spend(&self, work: u64) {
        self.left.set(self.left.get().saturating_sub(work));
    }

    /// The conversions the remaining work has curtailed so far: a decode
    /// during which this count rose depends on the page's earlier draws.
    pub(crate) const fn curtailments(&self) -> u64 {
        self.curtailed.get()
    }

    pub(super) fn curtail(&self) {
        self.curtailed.set(self.curtailed.get().saturating_add(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::colorspace::Layout;
    use crate::resources::image::colorspace::test_support::{
        convert_in, devicen_cmyk, exact, name_obj, noise, smooth_program,
    };

    /// Images convert exactly while the page's budget lasts, then through
    /// the lattice the work left affords, then not at all; an image whose
    /// conversion costs one operation is never limited.
    #[test]
    fn the_page_budget_bounds_the_tint_work_of_all_its_images() {
        let (doc, cs) = devicen_cmyk(4, &smooth_program(4, 1_000));
        let space = crate::resources::colorspace::resolve(&doc, &cs);
        let cost = space.conversion_cost();
        let nodes = [[0u8; 4], [255; 4]];
        let mut data = noise(4, 1000);
        data[..8].copy_from_slice(nodes.as_flattened());

        let budget = TintBudget::with_work(1000 * cost);
        let img = convert_in(
            &doc,
            cs.clone(),
            &[],
            Layout::Color(4),
            &data,
            Some(&budget),
        )
        .expect("exact within the budget");
        let px = img.data.as_cpu().unwrap();
        for (i, tuple) in data.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                px[i * 3..i * 3 + 3],
                exact(&doc, &space, tuple),
                "pixel {i}"
            );
        }
        assert!(
            budget.remaining() < 1000 * cost,
            "the exact conversion spent"
        );
        assert_eq!(
            budget.curtailments(),
            0,
            "an exact conversion is not curtailed"
        );

        budget.left.set(16 * cost);
        let img = convert_in(
            &doc,
            cs.clone(),
            &[],
            Layout::Color(4),
            &data,
            Some(&budget),
        )
        .expect("a two-value lattice fits");
        let px = img.data.as_cpu().unwrap();
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(px[i * 3..i * 3 + 3], exact(&doc, &space, node), "node {i}");
        }
        assert_eq!(budget.remaining(), 0);
        assert_eq!(
            budget.curtailments(),
            1,
            "the page's work chose the lattice"
        );

        assert!(convert_in(&doc, cs, &[], Layout::Color(4), &data, Some(&budget)).is_none());
        assert_eq!(
            budget.curtailments(),
            2,
            "the page's work dropped the image"
        );
        let gray = convert_in(
            &doc,
            name_obj(b"DeviceGray"),
            &[],
            Layout::Color(1),
            &[0, 128, 255],
            Some(&budget),
        )
        .expect("a device space spends no tint work");
        assert_eq!(gray.data.as_cpu().unwrap(), [0, 128, 255]);
        assert_eq!(
            budget.curtailments(),
            2,
            "a device space is never curtailed"
        );

        // One colorant: 256 exact conversions do not fit, a lattice of 18
        // values does.
        let (doc, cs) = devicen_cmyk(1, &format!("{{{} dup dup dup }}", " 1 pop".repeat(1_000)));
        let cost = crate::resources::colorspace::resolve(&doc, &cs).conversion_cost();
        budget.left.set(18 * cost);
        let one = convert_in(
            &doc,
            cs,
            &[],
            Layout::Color(1),
            &[0, 100, 255],
            Some(&budget),
        )
        .expect("a one-dimensional lattice fits");
        assert_eq!(one.data.as_cpu().unwrap().len(), 9);
        assert_eq!(budget.remaining(), 0);
    }
}
