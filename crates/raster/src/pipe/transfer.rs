//! Borrowed per-channel transfer tables consumed by the compositing pipe.

use color::Sample;

/// Borrowed references to the per-channel transfer tables applied by the
/// pipe, one entry per sample value (256 for 8-bit samples, 65 536 for
/// 16-bit ones).
///
/// Stored in [`crate::PipeState`]; borrowing the tables avoids copying them
/// for each paint operation.
#[derive(Copy, Clone, Debug)]
pub struct TransferSet<'a, S: Sample = u8> {
    /// RGB transfer tables: `[R, G, B]`.
    pub rgb: [&'a S::Table; 3],
    /// Gray transfer table.
    pub gray: &'a S::Table,
    /// CMYK transfer tables: `[C, M, Y, K]`.
    pub cmyk: [&'a S::Table; 4],
    /// `DeviceN` transfer tables, one per channel of the 8-channel ink pixel.
    pub device_n: [&'a S::Table; 8],
    /// Whether all three `rgb` tables are the identity `v → v`; fixed at
    /// construction, so the per-span query is a field read.
    rgb_identity: bool,
}

impl<'a, S: Sample> TransferSet<'a, S> {
    /// Assemble a `TransferSet` from its tables.
    #[must_use]
    pub fn new(
        rgb: [&'a S::Table; 3],
        gray: &'a S::Table,
        cmyk: [&'a S::Table; 4],
        device_n: [&'a S::Table; 8],
    ) -> Self {
        let id = S::identity_table().as_ref();
        Self {
            rgb,
            gray,
            cmyk,
            device_n,
            rgb_identity: rgb.iter().all(|lut| lut.as_ref() == id),
        }
    }

    /// Return `true` if all three RGB transfer tables are the identity `v → v`.
    ///
    /// Used by the AA fast path to skip the transfer step entirely when it would
    /// be a no-op.
    #[must_use]
    pub const fn is_identity_rgb(&self) -> bool {
        self.rgb_identity
    }

    /// Return a `TransferSet` whose every table is the identity (pass-through).
    ///
    /// The returned value borrows from static memory.
    #[must_use]
    pub fn identity_rgb() -> TransferSet<'static, S> {
        let id = S::identity_table();
        TransferSet {
            rgb: [id; 3],
            gray: id,
            cmyk: [id; 4],
            device_n: [id; 8],
            rgb_identity: true,
        }
    }
}
