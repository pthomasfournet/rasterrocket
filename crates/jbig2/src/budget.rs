//! Cumulative limits on the memory and compositing work of one decode.
//!
//! Every bitmap and every buffer derived from one is sized by header fields
//! before any coded data is read, so a short segment can declare an
//! arbitrarily large amount of state. A [`Budget`] is charged before each
//! such allocation, and before each symbol instance, text-region placement
//! and halftone rendering pass, and refuses the decode once it is spent. Charges are
//! never refunded: the memory limit bounds the total allocated over the
//! decode, which also bounds the work proportional to it.

use alloc::vec::Vec;
use core::cell::Cell;

use crate::error::{DecodeError, OverflowError, Result, bail};

/// Upper bound on the memory budget, in bits (1 GiB).
const MEMORY_CAP: u64 = 1 << 33;

/// Memory budget of one page beyond the page bitmap: this many page areas,
/// plus [`PAGE_MEMORY_BASE`], at most [`MEMORY_CAP`].
const PAGE_MEMORY_FACTOR: u64 = 16;

/// Memory budget of one page that does not scale with its area, in bits
/// (8 MiB).
const PAGE_MEMORY_BASE: u64 = 1 << 26;

/// Memory budget of the global segments, which belong to no page, in bits
/// (256 MiB).
const GLOBALS_MEMORY: u64 = 1 << 31;

/// Upper bound on the compositing budget, in pixels.
const WORK_CAP: u64 = 1 << 36;

/// Compositing budget of one page: this many page areas, plus
/// [`PAGE_WORK_BASE`], at most [`WORK_CAP`].
const PAGE_WORK_FACTOR: u64 = 64;

/// Compositing budget of one page that does not scale with its area, in
/// pixels.
const PAGE_WORK_BASE: u64 = 1 << 28;

/// Compositing budget of the global segments, in pixels.
const GLOBALS_WORK: u64 = 1 << 32;

/// Compositing work charged for decoding one symbol instance, in pixels,
/// on top of the pixels it writes. An instance that writes nothing, being
/// empty, out of range or outside its region, still costs its integer
/// decodes.
const INSTANCE_WORK: u64 = 64;

/// The memory and compositing work that a decode may still spend.
///
/// Memory is counted in bits of allocated storage; work in pixels written
/// by placing symbols and patterns into a region.
#[derive(Debug)]
pub(crate) struct Budget {
    memory: Cell<u64>,
    work: Cell<u64>,
}

impl Budget {
    /// The budget of a page of `width` × `height` pixels, excluding the page
    /// bitmap itself.
    pub(crate) fn for_page(width: u32, height: u32) -> Self {
        let area = u64::from(width) * u64::from(height);

        Self {
            memory: Cell::new((area * PAGE_MEMORY_FACTOR + PAGE_MEMORY_BASE).min(MEMORY_CAP)),
            work: Cell::new((area * PAGE_WORK_FACTOR + PAGE_WORK_BASE).min(WORK_CAP)),
        }
    }

    /// The budget of the global segments of a file or a PDF `JBIG2Globals`
    /// stream.
    pub(crate) fn for_globals() -> Self {
        Self {
            memory: Cell::new(GLOBALS_MEMORY),
            work: Cell::new(GLOBALS_WORK),
        }
    }

    /// A budget of `memory` bits and `work` pixels.
    pub(crate) fn with_limits(memory: u64, work: u64) -> Self {
        Self {
            memory: Cell::new(memory),
            work: Cell::new(work),
        }
    }

    /// Spend `bits` of memory.
    pub(crate) fn charge_memory(&self, bits: u64) -> Result<()> {
        match self.memory.get().checked_sub(bits) {
            Some(rest) => {
                self.memory.set(rest);
                Ok(())
            }
            None => bail!(OverflowError::MemoryBudget),
        }
    }

    /// Spend `pixels` of compositing work.
    pub(crate) fn charge_work(&self, pixels: u64) -> Result<()> {
        match self.work.get().checked_sub(pixels) {
            Some(rest) => {
                self.work.set(rest);
                Ok(())
            }
            None => bail!(OverflowError::WorkBudget),
        }
    }

    /// Spend the fixed work of decoding one symbol instance.
    pub(crate) fn charge_instance(&self) -> Result<()> {
        self.charge_work(INSTANCE_WORK)
    }

    /// A vector of `len` copies of `fill`, charged to the memory budget and
    /// allocated fallibly.
    pub(crate) fn filled_vec<T: Copy>(&self, len: usize, fill: T) -> Result<Vec<T>> {
        let mut vec = self.vec_with_capacity(len)?;
        vec.resize(len, fill);

        Ok(vec)
    }

    /// An empty vector with room for `len` elements, charged to the memory
    /// budget and allocated fallibly.
    pub(crate) fn vec_with_capacity<T>(&self, len: usize) -> Result<Vec<T>> {
        let bits = (len as u64).saturating_mul(core::mem::size_of::<T>() as u64 * 8);
        self.charge_memory(bits)?;

        let mut vec = Vec::new();
        try_reserve(&mut vec, len)?;

        Ok(vec)
    }
}

/// Reserve room for exactly `additional` more elements in `vec`, failing
/// with [`DecodeError::OutOfMemory`] if the allocation fails.
pub(crate) fn try_reserve<T>(vec: &mut Vec<T>, additional: usize) -> Result<()> {
    if vec.try_reserve_exact(additional).is_err() {
        bail!(DecodeError::OutOfMemory);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DecodeError;

    #[test]
    fn page_budget_scales_with_the_area_up_to_the_cap() {
        let small = Budget::for_page(100, 100);
        assert_eq!(small.memory.get(), 16 * 10_000 + (1 << 26));
        assert_eq!(small.work.get(), 64 * 10_000 + (1 << 28));

        let largest = Budget::for_page(65_535, 65_535);
        assert_eq!(largest.memory.get(), MEMORY_CAP);
        assert_eq!(largest.work.get(), WORK_CAP);
    }

    #[test]
    fn charges_beyond_the_budget_fail_and_leave_it_unchanged() {
        let budget = Budget::with_limits(100, 10);

        assert!(budget.charge_memory(60).is_ok());
        assert_eq!(
            budget.charge_memory(41),
            Err(DecodeError::Overflow(OverflowError::MemoryBudget))
        );
        assert!(budget.charge_memory(40).is_ok());

        assert_eq!(
            budget.charge_work(11),
            Err(DecodeError::Overflow(OverflowError::WorkBudget))
        );
        assert!(budget.charge_work(10).is_ok());
    }

    #[test]
    fn vectors_are_charged_by_their_storage() {
        let budget = Budget::with_limits(32 * 4, 0);

        assert_eq!(budget.filled_vec(4, 0_u32).unwrap(), [0; 4]);
        assert!(budget.filled_vec(1, 0_u8).is_err());
    }
}
