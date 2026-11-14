//! Per-thread scratch buffers of samples of any depth.

use std::cell::RefCell;

use color::Sample;

/// A grow-never-shrink buffer lent as a slice of samples of any type.
///
/// The storage is a `Vec<u64>`, whose alignment suffices for every sample
/// type, so the lent slice is a cast of it.  The contents of a lent slice
/// are unspecified: callers write every sample before reading it.
pub(crate) struct Scratch(RefCell<Vec<u64>>);

impl Scratch {
    /// An empty buffer.
    pub(crate) const fn new() -> Self {
        Self(RefCell::new(Vec::new()))
    }

    /// Run `f` on a slice of `len` samples of type `S` drawn from the
    /// buffer.
    ///
    /// # Panics
    ///
    /// When called again from within `f` on the same buffer.
    pub(crate) fn with<S: Sample, R>(&self, len: usize, f: impl FnOnce(&mut [S]) -> R) -> R {
        let mut words = self.0.borrow_mut();
        let need = (len * size_of::<S>()).div_ceil(size_of::<u64>());
        if words.len() < need {
            words.resize(need, 0);
        }
        let samples: &mut [S] = bytemuck::cast_slice_mut(&mut words[..need]);
        f(&mut samples[..len])
    }
}
