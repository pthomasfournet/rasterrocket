//! Committed buffers: device buffers in the DEFAULT heap, host buffers in
//! the UPLOAD or READBACK heap, persistently mapped, and the limit on the
//! elements of a buffer view.

use windows::Win32::Foundation::E_OUTOFMEMORY;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_HEAP_FLAG_CREATE_NOT_ZEROED, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_RANGE, D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATES, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use super::error::device_err;
use crate::backend::{BackendError, Result};

/// The name of the limit [`check_view_elements`] enforces.
pub(super) const VIEW_LIMIT: &str =
    "2^27-element buffer view limit (D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP)";

/// The most elements one buffer view may hold.
const MAX_VIEW_ELEMENTS: u64 = 1 << D3D12_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP;

/// Fail when a view of `bytes` bytes in elements of `stride` bytes, over a
/// buffer holding `what`, holds more elements than a buffer view may.
///
/// # Errors
/// [`BackendError::DeviceLimit`] then, with the limit in bytes.
pub(super) fn check_view_elements(what: &'static str, bytes: u64, stride: u32) -> Result<()> {
    let max = MAX_VIEW_ELEMENTS.saturating_mul(u64::from(stride));
    if bytes > max {
        return Err(BackendError::DeviceLimit {
            what,
            need: bytes,
            limit: VIEW_LIMIT,
            max,
            unit: "bytes",
        });
    }
    Ok(())
}

/// A device buffer: a committed resource in the DEFAULT heap that kernels
/// read and write through views and copies read and write.
pub struct DeviceBuffer {
    resource: ID3D12Resource,
    /// The bytes requested.
    len: u64,
    /// The resource's width: `len` rounded up to a whole 32-bit word.
    width: u64,
}

impl DeviceBuffer {
    /// The size the buffer was allocated with, in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.len
    }

    /// The resource.
    pub(super) const fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// The resource's width in bytes, a multiple of 4 at least
    /// [`Self::size`]; views cover it.
    pub(super) const fn width(&self) -> u64 {
        self.width
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.len)
            .field("width", &self.width)
            .finish_non_exhaustive()
    }
}

/// Which host heap a [`HostBuffer`] lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HostHeap {
    /// UPLOAD: the host writes, copies read.
    Upload,
    /// READBACK: copies write, the host reads.
    Readback,
}

/// A host buffer: a committed resource in the UPLOAD or READBACK heap,
/// mapped for its whole life.
pub struct HostBuffer {
    resource: ID3D12Resource,
    /// The mapping of subresource 0, valid while `resource` lives.
    ptr: *mut u8,
    len: usize,
    heap: HostHeap,
}

impl std::fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostBuffer")
            .field("len", &self.len)
            .field("heap", &self.heap)
            .finish_non_exhaustive()
    }
}

// SAFETY: the mapping is owned by `resource`, which the buffer owns; the
// pointer is reached only through `&self` / `&mut self` accessors, so the
// borrow rules order host accesses across threads.
unsafe impl Send for HostBuffer {}
// SAFETY: as for `Send`; a shared borrow only reads the mapping.
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// The size the buffer was allocated with, in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.len as u64
    }

    /// The mapped bytes.
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps at least `len` bytes for as long as
        // `resource`, which `self` owns, lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The mapped bytes, writable.
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as for `as_slice`; `&mut self` makes the view unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The resource.
    pub(super) const fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    /// The heap the buffer lives in.
    pub(super) const fn heap(&self) -> HostHeap {
        self.heap
    }
}

/// A device buffer of `size` bytes.  Its bytes are zero when `zeroed`
/// (a committed resource created without `CREATE_NOT_ZEROED`), else
/// unspecified.
pub(super) fn alloc_device(
    device: &ID3D12Device,
    size: usize,
    zeroed: bool,
    what: &'static str,
) -> Result<DeviceBuffer> {
    let len = size as u64;
    let width = len
        .checked_next_multiple_of(4)
        .ok_or(BackendError::OutOfMemory { what, bytes: len })?;
    let flags = if zeroed {
        D3D12_HEAP_FLAG_NONE
    } else {
        D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
    };
    let resource = committed(
        device,
        D3D12_HEAP_TYPE_DEFAULT,
        flags,
        width,
        true,
        D3D12_RESOURCE_STATE_COMMON,
        what,
    )?;
    Ok(DeviceBuffer {
        resource,
        len,
        width,
    })
}

/// A host buffer of `size` bytes in `heap`, mapped.
pub(super) fn alloc_host(
    device: &ID3D12Device,
    size: usize,
    heap: HostHeap,
    what: &'static str,
) -> Result<HostBuffer> {
    let (heap_type, state) = match heap {
        HostHeap::Upload => (D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ),
        HostHeap::Readback => (D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST),
    };
    let resource = committed(
        device,
        heap_type,
        D3D12_HEAP_FLAG_NONE,
        size as u64,
        false,
        state,
        what,
    )?;
    // The host reads nothing of an upload buffer; it reads all of a
    // read-back one.
    let none_read = D3D12_RANGE { Begin: 0, End: 0 };
    let read = match heap {
        HostHeap::Upload => Some(&raw const none_read),
        HostHeap::Readback => None,
    };
    let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: `resource` is a buffer in a CPU-visible heap; `read` is null
    // or points at a range that outlives the call; `ptr` is a valid
    // out-pointer.
    unsafe { resource.Map(0, read, Some(&raw mut ptr)) }.map_err(device_err(device, "Map"))?;
    if ptr.is_null() {
        return Err(BackendError::msg("Map returned a null pointer"));
    }
    Ok(HostBuffer {
        resource,
        ptr: ptr.cast(),
        len: size,
        heap,
    })
}

/// A committed buffer resource of `width` bytes in a heap of `heap_type`,
/// created in `state`.
fn committed(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    heap_flags: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_FLAGS,
    width: u64,
    unordered: bool,
    state: D3D12_RESOURCE_STATES,
    what: &'static str,
) -> Result<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if unordered {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: `props` and `desc` outlive the call and describe a buffer;
    // `resource` is a valid out-pointer.
    unsafe {
        device.CreateCommittedResource(
            &raw const props,
            heap_flags,
            &raw const desc,
            state,
            None,
            &raw mut resource,
        )
    }
    .map_err(|e| {
        if e.code() == E_OUTOFMEMORY {
            BackendError::OutOfMemory { what, bytes: width }
        } else {
            device_err(device, "CreateCommittedResource")(e)
        }
    })?;
    resource.ok_or_else(|| BackendError::msg("CreateCommittedResource returned no resource"))
}

/// What a zero-fill of `size` bytes does: nothing for 0, else the fill of
/// a multiple of 4; any other size fails rather than leaving 1–3 trailing
/// bytes.
///
/// # Errors
/// [`BackendError::UnalignedFill`] for a size that is not a multiple of 4.
pub(super) const fn fill_size(size: u64) -> Result<Option<u64>> {
    if size == 0 {
        return Ok(None);
    }
    if !size.is_multiple_of(4) {
        return Err(BackendError::UnalignedFill {
            size,
            required_alignment: 4,
        });
    }
    Ok(Some(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_limit_is_two_to_the_27_elements_in_bytes() {
        let max4 = (1_u64 << 27) * 4;
        check_view_elements("page", max4, 4).expect("at the limit");
        let err = check_view_elements("page", max4 + 1, 4).expect_err("past the limit");
        assert!(
            matches!(
                err,
                BackendError::DeviceLimit { what: "page", need, limit: VIEW_LIMIT, max, unit: "bytes" }
                    if need == max4 + 1 && max == max4
            ),
            "{err:?}"
        );
        let max48 = (1_u64 << 27) * 48;
        check_view_elements("cmds", max48, 48).expect("at the limit");
        assert!(check_view_elements("cmds", max48 + 48, 48).is_err());
    }

    #[test]
    fn fill_sizes_are_whole_words() {
        assert_eq!(fill_size(0).expect("zero"), None);
        assert_eq!(fill_size(8).expect("eight"), Some(8));
        assert!(matches!(
            fill_size(6),
            Err(BackendError::UnalignedFill {
                size: 6,
                required_alignment: 4
            })
        ));
    }
}
