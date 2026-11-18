//! Faulting in the host pages a page's read-back writes, before the device
//! buffers of the page are allocated.
//!
//! A canvas the host has not yet written is backed by no physical page; its
//! first write faults each page in.  In a region marked `MADV_HUGEPAGE` (the
//! arenas of some allocators) under the kernel's default THP defrag policy,
//! such a fault takes a 2 MiB page from the free lists, and when none is
//! free, compacts memory synchronously first.  The device buffers of a page
//! (on an integrated GPU, the read-back buffer is host memory the size of
//! the canvas) are taken from the same free lists: allocated before the
//! canvas is faulted in, they leave its faults compacting.  Populating the
//! canvas first gives it the free huge pages the device buffers would take.

/// Fault in, writable, every whole host page within `buf`, leaving its
/// contents unchanged; pages already present are kept.  Pages `buf` shares
/// with other memory at either end are not touched.
///
/// Linux and Android only (`MADV_POPULATE_WRITE`, Linux 5.14); elsewhere,
/// or when the kernel refuses the advice, the pages fault in on first
/// write instead.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(super) fn populate_for_write<T>(buf: &mut [T]) {
    // SAFETY: sysconf reads a constant of the system.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let Ok(page) = usize::try_from(page) else {
        return;
    };
    if !page.is_power_of_two() {
        return;
    }
    let start = buf.as_mut_ptr().cast::<u8>() as usize;
    let end = start + std::mem::size_of_val(buf);
    let first = start.next_multiple_of(page);
    let last = end & !(page - 1);
    if last <= first {
        return;
    }
    // SAFETY: `[first, last)` lies within `buf`, which is writable
    // memory borrowed exclusively for this call; MADV_POPULATE_WRITE
    // only faults in its pages, as a write would, and changes no byte
    // of them.
    let rc = unsafe {
        libc::madvise(
            first as *mut libc::c_void,
            last - first,
            libc::MADV_POPULATE_WRITE,
        )
    };
    if rc != 0 {
        log::debug!(
            "vulkan: MADV_POPULATE_WRITE of {} bytes: {}",
            last - first,
            std::io::Error::last_os_error()
        );
    }
}

/// [`populate_for_write`] where `MADV_POPULATE_WRITE` does not exist: the
/// pages fault in on first write.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(super) const fn populate_for_write<T>(_buf: &mut [T]) {}

#[cfg(test)]
mod tests {
    use super::populate_for_write;

    #[test]
    fn populating_keeps_every_byte() {
        let mut buf: Vec<u16> = (0..300_000u32)
            .map(|i| u16::try_from(i % 65_521).unwrap())
            .collect();
        let want = buf.clone();
        populate_for_write(&mut buf[3..]);
        populate_for_write(&mut buf[..1]);
        populate_for_write::<u16>(&mut []);
        assert_eq!(buf, want);
    }

    #[test]
    fn a_fresh_zeroed_buffer_stays_zero() {
        let mut buf = vec![0u8; 8 << 20];
        populate_for_write(&mut buf);
        assert!(buf.iter().all(|&b| b == 0));
    }
}
