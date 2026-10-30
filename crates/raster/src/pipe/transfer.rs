//! Borrowed per-channel transfer tables consumed by the compositing pipe.

use color::TransferLut;

/// Borrowed references to the per-channel transfer LUTs applied by the pipe.
///
/// Stored in [`crate::PipeState`]; borrowing the tables avoids copying them
/// for each paint operation.
#[derive(Copy, Clone, Debug)]
pub struct TransferSet<'a> {
    /// RGB transfer tables: `[R, G, B]`, each 256 bytes.
    pub rgb: [&'a [u8; 256]; 3],
    /// Gray transfer table, 256 bytes.
    pub gray: &'a [u8; 256],
    /// CMYK transfer tables: `[C, M, Y, K]`, each 256 bytes.
    pub cmyk: [&'a [u8; 256]; 4],
    /// `DeviceN` transfer tables: `SPOT_NCOMPS + 4` tables of 256 bytes each.
    ///
    /// A slice of owned arrays (`&[[u8; 256]]`) rather than a slice of
    /// references, so a caller holding its tables in a `Vec<[u8; 256]>` can
    /// lend them without building a per-paint `Vec<&[u8; 256]>`.
    pub device_n: &'a [[u8; 256]],
    /// Whether all three `rgb` tables are the identity `v → v`; fixed at
    /// construction, so the per-span query is a field read.
    rgb_identity: bool,
}

impl<'a> TransferSet<'a> {
    /// Assemble a `TransferSet` from its tables.
    #[must_use]
    pub fn new(
        rgb: [&'a [u8; 256]; 3],
        gray: &'a [u8; 256],
        cmyk: [&'a [u8; 256]; 4],
        device_n: &'a [[u8; 256]],
    ) -> Self {
        let id = TransferLut::IDENTITY.as_array();
        Self {
            rgb,
            gray,
            cmyk,
            device_n,
            rgb_identity: rgb.iter().all(|lut| *lut == id),
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
    pub fn identity_rgb() -> TransferSet<'static> {
        /// Identity tables for all eight `DeviceN` channels.
        static DN: [[u8; 256]; 8] = [TransferLut::IDENTITY.0; 8];
        let id = TransferLut::IDENTITY.as_array();
        TransferSet {
            rgb: [id; 3],
            gray: id,
            cmyk: [id; 4],
            device_n: &DN,
            rgb_identity: true,
        }
    }
}
