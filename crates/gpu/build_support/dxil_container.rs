// The DXIL container check shared by `build.rs` (through `include!`) and
// the library's embedded blobs.

/// Whether `blob` is a hashed DXIL container: the magic `DXBC` at bytes
/// 0..4 and a 16-byte digest at 4..20 that is not all zero.  DXC writes
/// the digest only when its validator signed the container; a device
/// rejects an unsigned one.
pub(crate) fn container_is_hashed(blob: &[u8]) -> bool {
    blob.get(..4) == Some(b"DXBC".as_slice())
        && blob
            .get(4..20)
            .is_some_and(|digest| digest.iter().any(|&b| b != 0))
}
