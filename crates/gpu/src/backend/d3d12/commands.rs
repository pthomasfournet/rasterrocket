//! A command allocator and its list, with the state of every buffer the
//! open list uses and every object it references.
//!
//! Every device buffer is in `COMMON` when a submission starts and ends
//! (a buffer decays to `COMMON` when the command list that used it
//! completes, and each list ends with explicit transitions back to it).
//! Within a list each buffer is transitioned to the state the next
//! operation needs: `COPY_DEST` and `COPY_SOURCE` for copies,
//! `NON_PIXEL_SHADER_RESOURCE` for a kernel's SRVs, `UNORDERED_ACCESS`
//! for its UAVs.  A transition orders the operations on either side of
//! it; two operations in the same writing state are separated by a UAV
//! barrier (dispatches) or a transition through `COMMON` (copies).
//!
//! Every object a list references (buffers, the descriptor heap, pipeline
//! objects, query heaps) is held by its [`Commands`] until a wait on the list's own
//! fence succeeds, so a buffer freed by its owner while a page is in
//! flight is released only after the page.  A list whose completion
//! cannot be confirmed (its fence was not signalled, a wait on it failed,
//! or the queue could not be drained when the backend was dropped)
//! strands its [`Commands`]: they are never reset, and their references
//! are leaked rather than released, even when the backend is dropped.

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATES,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER, ID3D12CommandAllocator,
    ID3D12DescriptorHeap, ID3D12Device, ID3D12GraphicsCommandList, ID3D12PipelineState,
    ID3D12QueryHeap, ID3D12Resource, ID3D12RootSignature,
};
use windows::core::Interface;

use super::error::device_err;
use crate::backend::{BackendError, Result};

/// A buffer's state within the list being recorded.
struct Tracked {
    resource: ID3D12Resource,
    state: D3D12_RESOURCE_STATES,
    /// Whether the last operation in `state` wrote the buffer.
    written: bool,
}

/// An object a list references other than a tracked buffer.
pub(super) enum Held {
    Resource(ID3D12Resource),
    Heap(ID3D12DescriptorHeap),
    Pipeline(ID3D12PipelineState),
    RootSignature(ID3D12RootSignature),
    QueryHeap(ID3D12QueryHeap),
}

impl Held {
    fn as_raw(&self) -> *mut core::ffi::c_void {
        match self {
            Self::Resource(o) => o.as_raw(),
            Self::Heap(o) => o.as_raw(),
            Self::Pipeline(o) => o.as_raw(),
            Self::RootSignature(o) => o.as_raw(),
            Self::QueryHeap(o) => o.as_raw(),
        }
    }
}

impl From<&ID3D12Resource> for Held {
    fn from(o: &ID3D12Resource) -> Self {
        Self::Resource(o.clone())
    }
}

impl From<&ID3D12DescriptorHeap> for Held {
    fn from(o: &ID3D12DescriptorHeap) -> Self {
        Self::Heap(o.clone())
    }
}

impl From<&ID3D12PipelineState> for Held {
    fn from(o: &ID3D12PipelineState) -> Self {
        Self::Pipeline(o.clone())
    }
}

impl From<&ID3D12QueryHeap> for Held {
    fn from(o: &ID3D12QueryHeap) -> Self {
        Self::QueryHeap(o.clone())
    }
}

impl From<&ID3D12RootSignature> for Held {
    fn from(o: &ID3D12RootSignature) -> Self {
        Self::RootSignature(o.clone())
    }
}

/// A command allocator, its list, and the objects the list references.
pub(super) struct Commands {
    device: ID3D12Device,
    allocator: ID3D12CommandAllocator,
    pub(super) list: ID3D12GraphicsCommandList,
    /// Whether `list` is open for recording.
    open: bool,
    /// Whether a submission of `list` may still run with no fence to
    /// confirm its end ([`Self::strand`]).
    stranded: bool,
    /// The device buffers the list uses, with their current states.
    tracked: Vec<Tracked>,
    /// The other objects the list references.
    held: Vec<Held>,
    /// Barriers to record before the next operation.
    pending: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Commands {
    pub(super) fn new(device: &ID3D12Device) -> Result<Self> {
        // SAFETY: `device` is live; the interfaces are returned by value.
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE) }
                .map_err(device_err(device, "CreateCommandAllocator"))?;
        // SAFETY: `allocator` is a live compute allocator.
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_COMPUTE,
                &allocator,
                None::<&ID3D12PipelineState>,
            )
        }
        .map_err(device_err(device, "CreateCommandList"))?;
        // SAFETY: the list was created open and records nothing.
        unsafe { list.Close() }.map_err(device_err(device, "ID3D12GraphicsCommandList::Close"))?;
        Ok(Self {
            device: device.clone(),
            allocator,
            list,
            open: false,
            stranded: false,
            tracked: Vec::new(),
            held: Vec::new(),
            pending: Vec::new(),
        })
    }

    /// Reset the allocator and open the list.  The caller guarantees that
    /// no submission of the list is still running; stranded commands are
    /// never reset.
    pub(super) fn begin(&mut self) -> Result<()> {
        if self.stranded {
            return Err(BackendError::msg(format!(
                "a D3D12 submission's completion could not be confirmed; {}",
                crate::backend::DEVICE_LOST_CAUSE
            )));
        }
        self.release();
        // SAFETY: the list is closed and no submission of it runs.
        unsafe { self.allocator.Reset() }
            .map_err(device_err(&self.device, "ID3D12CommandAllocator::Reset"))?;
        // SAFETY: as above.
        unsafe {
            self.list
                .Reset(&self.allocator, None::<&ID3D12PipelineState>)
        }
        .map_err(device_err(&self.device, "ID3D12GraphicsCommandList::Reset"))?;
        self.open = true;
        Ok(())
    }

    /// Return every tracked buffer to `COMMON` and close the list.
    pub(super) fn finish(&mut self) -> Result<()> {
        for i in 0..self.tracked.len() {
            let before = self.tracked[i].state;
            if before != D3D12_RESOURCE_STATE_COMMON {
                let b = transition(
                    &self.tracked[i].resource,
                    before,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                self.pending.push(b);
                self.tracked[i].state = D3D12_RESOURCE_STATE_COMMON;
            }
        }
        self.flush();
        self.open = false;
        // SAFETY: the list is open.
        unsafe { self.list.Close() }
            .map_err(device_err(&self.device, "ID3D12GraphicsCommandList::Close"))
    }

    /// Close the list if it is open, discarding what it recorded.  The
    /// caller guarantees that no submission of the list is still running.
    pub(super) fn discard(&mut self) {
        self.pending.clear();
        if self.open {
            self.open = false;
            // SAFETY: the list is open; a list closed in error is reset by
            // the next `begin`.
            if let Err(e) = unsafe { self.list.Close() } {
                log::debug!("d3d12: closing an abandoned list: {}", e.code());
            }
        }
        self.release();
    }

    /// Drop the references of the last list.  The caller guarantees that
    /// no submission of it is still running; stranded commands keep them.
    pub(super) fn release(&mut self) {
        if self.stranded {
            return;
        }
        self.tracked.clear();
        self.held.clear();
        self.pending.clear();
    }

    /// Keep the list and everything it references for the life of the
    /// process: a submission of it may still run.
    pub(super) fn strand(&mut self) {
        self.stranded = true;
        self.open = false;
        self.pending.clear();
    }

    /// Whether [`Self::strand`] was called.
    pub(super) const fn is_stranded(&self) -> bool {
        self.stranded
    }

    /// Queue the barriers that bring `resource` into `want` for an
    /// operation that writes it when `write`.
    pub(super) fn use_buffer(
        &mut self,
        resource: &ID3D12Resource,
        want: D3D12_RESOURCE_STATES,
        write: bool,
    ) {
        let raw = resource.as_raw();
        let i = if let Some(i) = self.tracked.iter().position(|t| t.resource.as_raw() == raw) {
            i
        } else {
            self.tracked.push(Tracked {
                resource: resource.clone(),
                state: D3D12_RESOURCE_STATE_COMMON,
                written: false,
            });
            self.tracked.len() - 1
        };
        let t = &self.tracked[i];
        if t.state != want {
            self.pending.push(transition(resource, t.state, want));
        } else if t.written && want == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            self.pending.push(uav_barrier(resource));
        } else if t.written {
            self.pending
                .push(transition(resource, want, D3D12_RESOURCE_STATE_COMMON));
            self.pending
                .push(transition(resource, D3D12_RESOURCE_STATE_COMMON, want));
        }
        let t = &mut self.tracked[i];
        t.state = want;
        t.written = write;
    }

    /// Hold `object` until a wait on the list's fence succeeds.
    pub(super) fn hold(&mut self, object: impl Into<Held>) {
        let object = object.into();
        let raw = object.as_raw();
        if !self.held.iter().any(|h| h.as_raw() == raw) {
            self.held.push(object);
        }
    }

    /// Record the queued barriers.
    pub(super) fn flush(&mut self) {
        if !self.pending.is_empty() {
            // SAFETY: the list is open; every resource a barrier names is
            // held in `tracked` beyond the call.
            unsafe { self.list.ResourceBarrier(&self.pending) };
            self.pending.clear();
        }
    }

    /// Record a copy of `len` bytes from the start of `src` to the start
    /// of `dst`, after the barriers queued for it.
    pub(super) fn copy(&mut self, dst: &ID3D12Resource, src: &ID3D12Resource, len: u64) {
        self.flush();
        // SAFETY: the list is open; both resources are held and at least
        // `len` bytes long, in states a copy accepts.
        unsafe { self.list.CopyBufferRegion(dst, 0, src, 0, len) };
    }
}

impl Drop for Commands {
    /// Stranded commands leak one reference to the allocator, the list and
    /// every object the list references, so that none is destroyed under a
    /// submission that may still run.
    fn drop(&mut self) {
        if self.stranded {
            std::mem::forget(self.allocator.clone());
            std::mem::forget(self.list.clone());
            std::mem::forget(std::mem::take(&mut self.tracked));
            std::mem::forget(std::mem::take(&mut self.held));
        }
    }
}

/// A transition barrier of all of `resource` from `before` to `after`.
const fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: borrowed(resource),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A UAV barrier on `resource`.
const fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: borrowed(resource),
            }),
        },
    }
}

/// A barrier holds a non-owning resource pointer of the resource's own
/// layout ([`borrowed`]).
const _: () =
    assert!(size_of::<ManuallyDrop<Option<ID3D12Resource>>>() == size_of::<ID3D12Resource>());

/// A non-owning copy of `resource`'s interface pointer, as barrier
/// structures hold it.
const fn borrowed(resource: &ID3D12Resource) -> ManuallyDrop<Option<ID3D12Resource>> {
    // SAFETY: `Option<ID3D12Resource>` is a nullable interface pointer of
    // the same layout as `ID3D12Resource`; the copy adds no reference and
    // `ManuallyDrop` never releases one.  Barriers are recorded while the
    // resource is held (`Commands::tracked`).
    unsafe { std::mem::transmute_copy(resource) }
}
