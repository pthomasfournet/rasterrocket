//! The page recorder: one command allocator and list, recorded between
//! `begin_page` and `submit_page` under one lock, and a second pair for
//! the synchronous transfers.
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
//! Every resource a list references is held by the recorder until the
//! list's fence has been waited on, so a buffer freed by its owner while a
//! page is in flight is released only after the page.

#![expect(
    clippy::significant_drop_tightening,
    reason = "the recording state's lock is held across the D3D12 calls that record into its list"
)]

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_NONE, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_NONE,
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_RESOURCE_BARRIER,
    D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATES,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER, D3D12_SHADER_RESOURCE_VIEW_DESC,
    D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_BUFFER, D3D12_UAV_DIMENSION_BUFFER,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC_0, ID3D12CommandAllocator,
    ID3D12DescriptorHeap, ID3D12Device, ID3D12GraphicsCommandList, ID3D12PipelineState,
    ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
use windows::core::Interface as _;

use super::error::device_err;
use super::fence::{PageFence, Submitter};
use super::memory::{self, DeviceBuffer, HostBuffer, HostHeap};
use super::pipeline::{KernelId, MAX_BINDINGS, PipelineCache, Shape};
use crate::backend::{BackendError, Result};

/// Dispatches one page may record: each takes [`MAX_BINDINGS`]
/// descriptors of the page's heap.
pub(super) const MAX_DISPATCHES_PER_PAGE: u32 = 64;

/// A staging buffer of a synchronous transfer larger than this is freed
/// after the transfer rather than kept for the next.
const KEPT_STAGING_BYTES: usize = 16 << 20;

/// The page's place in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No page; `begin_page` is valid.
    Idle,
    /// Between `begin_page` and `submit_page`.
    Recording,
    /// Submitted; only `wait_page` is valid.
    Submitted,
}

/// A buffer's state within the list being recorded.
struct Tracked {
    resource: ID3D12Resource,
    state: D3D12_RESOURCE_STATES,
    /// Whether the last operation in `state` wrote the buffer.
    written: bool,
}

/// A command allocator, its list, and the buffers the open list uses.
struct Commands {
    device: ID3D12Device,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Whether `list` is open for recording.
    open: bool,
    /// The device buffers the list uses, with their current states.
    tracked: Vec<Tracked>,
    /// The other resources the list references.
    held: Vec<ID3D12Resource>,
    /// Barriers to record before the next operation.
    pending: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Commands {
    fn new(device: &ID3D12Device) -> Result<Self> {
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
            tracked: Vec::new(),
            held: Vec::new(),
            pending: Vec::new(),
        })
    }

    /// Reset the allocator and open the list.  The caller guarantees that
    /// no submission of the list is still running.
    fn begin(&mut self) -> Result<()> {
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
    fn finish(&mut self) -> Result<()> {
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

    /// Close the list if it is open, discarding what it recorded.
    fn discard(&mut self) {
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
    /// no submission of it is still running.
    fn release(&mut self) {
        self.tracked.clear();
        self.held.clear();
        self.pending.clear();
    }

    /// Queue the barriers that bring `resource` into `want` for an
    /// operation that writes it when `write`.
    fn use_buffer(&mut self, resource: &ID3D12Resource, want: D3D12_RESOURCE_STATES, write: bool) {
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

    /// Hold `resource` until the list's fence has been waited on.
    fn hold(&mut self, resource: &ID3D12Resource) {
        let raw = resource.as_raw();
        if !self.held.iter().any(|h| h.as_raw() == raw) {
            self.held.push(resource.clone());
        }
    }

    /// Record the queued barriers.
    fn flush(&mut self) {
        if !self.pending.is_empty() {
            // SAFETY: the list is open; every resource a barrier names is
            // held in `tracked` beyond the call.
            unsafe { self.list.ResourceBarrier(&self.pending) };
            self.pending.clear();
        }
    }

    /// Record a copy of `len` bytes from the start of `src` to the start
    /// of `dst`, after the barriers queued for it.
    fn copy(&mut self, dst: &ID3D12Resource, src: &ID3D12Resource, len: u64) {
        self.flush();
        // SAFETY: the list is open; both resources are held and at least
        // `len` bytes long, in states a copy accepts.
        unsafe { self.list.CopyBufferRegion(dst, 0, src, 0, len) };
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

/// A non-owning copy of `resource`'s interface pointer, as barrier
/// structures hold it.
const fn borrowed(resource: &ID3D12Resource) -> ManuallyDrop<Option<ID3D12Resource>> {
    // SAFETY: `Option<ID3D12Resource>` is a nullable interface pointer of
    // the same layout as `ID3D12Resource`; the copy adds no reference and
    // `ManuallyDrop` never releases one.  Barriers are recorded while the
    // resource is held (`Commands::tracked`).
    unsafe { std::mem::transmute_copy(resource) }
}

/// The page's recording state.
struct Page {
    phase: Phase,
    commands: Commands,
    /// The shader-visible heap of the page's views:
    /// [`MAX_BINDINGS`] descriptors a dispatch.
    heap: ID3D12DescriptorHeap,
    /// Dispatches recorded this page.
    dispatches: u32,
    /// The fence of the submitted page.
    submitted: Option<PageFence>,
    /// A zeroed buffer that zero-fills copy from, grown to the largest.
    zeros: Option<DeviceBuffer>,
}

/// The synchronous transfers' state.
struct Transfer {
    commands: Commands,
    upload: Option<HostBuffer>,
    download: Option<HostBuffer>,
}

/// The per-page recorder and the synchronous transfers.
pub(super) struct Recorder {
    device: ID3D12Device,
    submitter: Arc<Submitter>,
    pipelines: PipelineCache,
    descriptor_size: u32,
    page: Mutex<Page>,
    transfer: Mutex<Transfer>,
}

impl Recorder {
    pub(super) fn new(device: &ID3D12Device, submitter: Arc<Submitter>) -> Result<Self> {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: MAX_DISPATCHES_PER_PAGE * MAX_BINDINGS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        // SAFETY: `desc` outlives the call.
        let heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&raw const desc) }
            .map_err(device_err(device, "CreateDescriptorHeap"))?;
        // SAFETY: `device` is live.
        let descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        Ok(Self {
            device: device.clone(),
            submitter,
            pipelines: PipelineCache::new(device),
            descriptor_size,
            page: Mutex::new(Page {
                phase: Phase::Idle,
                commands: Commands::new(device)?,
                heap,
                dispatches: 0,
                submitted: None,
                zeros: None,
            }),
            transfer: Mutex::new(Transfer {
                commands: Commands::new(device)?,
                upload: None,
                download: None,
            }),
        })
    }

    /// Build `kernel`'s pipeline if it is not built yet.
    #[cfg(test)]
    pub(super) fn pipeline(&self, kernel: KernelId) -> Result<()> {
        self.pipelines.get(kernel).map(drop)
    }

    fn lock(&self) -> MutexGuard<'_, Page> {
        self.page.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The page state, held, when a page is recording; `what` names the
    /// caller in the error otherwise.
    fn recording(&self, what: &str) -> Result<MutexGuard<'_, Page>> {
        let page = self.lock();
        if page.phase != Phase::Recording {
            return Err(BackendError::msg(format!(
                "{what} called from {:?} state; expected Recording",
                page.phase
            )));
        }
        Ok(page)
    }

    pub(super) fn begin_page(&self) -> Result<()> {
        let mut page = self.lock();
        if page.phase != Phase::Idle {
            return Err(BackendError::msg(format!(
                "begin_page called from {:?} state; expected Idle",
                page.phase
            )));
        }
        page.commands.begin()?;
        let heaps = [Some(page.heap.clone())];
        // SAFETY: the list is open; the heap is held by the recorder.
        unsafe { page.commands.list.SetDescriptorHeaps(&heaps) };
        page.dispatches = 0;
        page.phase = Phase::Recording;
        Ok(())
    }

    pub(super) fn submit_page(&self) -> Result<PageFence> {
        let mut page = self.recording("submit_page")?;
        let submitted = page
            .commands
            .finish()
            .and_then(|()| self.submitter.submit(&page.commands.list));
        match submitted {
            Ok(fence) => {
                page.submitted = Some(fence.clone());
                page.phase = Phase::Submitted;
                Ok(fence)
            }
            Err(e) => {
                page.commands.discard();
                page.phase = Phase::Idle;
                Err(e)
            }
        }
    }

    pub(super) fn wait_page(&self, fence: &PageFence) -> Result<()> {
        let mut page = self.lock();
        if page.phase != Phase::Submitted {
            return Err(BackendError::msg(format!(
                "wait_page called from {:?} state; expected Submitted",
                page.phase
            )));
        }
        self.submitter.wait(fence)?;
        page.commands.release();
        page.submitted = None;
        page.phase = Phase::Idle;
        Ok(())
    }

    /// Return to `Idle`: a recording page is discarded, a submitted one is
    /// waited for first; a failed wait is logged.
    pub(super) fn abandon_page(&self) {
        let mut page = self.lock();
        if page.phase == Phase::Submitted
            && let Some(fence) = page.submitted.take()
            && let Err(e) = self.submitter.wait(&fence)
        {
            log::warn!("d3d12: abandoning a submitted page: waiting for it failed: {e}");
        }
        page.commands.discard();
        page.phase = Phase::Idle;
    }

    /// Record a copy of `len` bytes from the upload buffer `src` to `dst`.
    pub(super) fn record_copy_to_device(
        &self,
        src: &HostBuffer,
        dst: &DeviceBuffer,
        len: u64,
    ) -> Result<()> {
        if src.heap() != HostHeap::Upload {
            return Err(BackendError::msg(
                "record_copy_to_device: the source is a read-back buffer; copies read only \
                 buffers from alloc_host_pinned",
            ));
        }
        let mut page = self.recording("record_copy_to_device")?;
        let c = &mut page.commands;
        c.hold(src.resource());
        c.use_buffer(dst.resource(), D3D12_RESOURCE_STATE_COPY_DEST, true);
        c.copy(dst.resource(), src.resource(), len);
        Ok(())
    }

    /// Record a copy of `len` bytes from `src` to the read-back buffer
    /// `dst`.
    pub(super) fn record_copy_to_host(
        &self,
        src: &DeviceBuffer,
        dst: &HostBuffer,
        len: u64,
    ) -> Result<()> {
        if dst.heap() != HostHeap::Readback {
            return Err(BackendError::msg(
                "record_copy_to_host: the destination is an upload buffer; copies write only \
                 buffers from alloc_host_readback",
            ));
        }
        let mut page = self.recording("record_copy_to_host")?;
        let c = &mut page.commands;
        c.hold(dst.resource());
        c.use_buffer(src.resource(), D3D12_RESOURCE_STATE_COPY_SOURCE, false);
        c.copy(dst.resource(), src.resource(), len);
        Ok(())
    }

    /// Record a zero-fill of the first `len` bytes of `buf`: a copy from
    /// the recorder's zeroed buffer.
    pub(super) fn record_zero_buffer(&self, buf: &DeviceBuffer, len: usize) -> Result<()> {
        let mut page = self.recording("record_zero_buffer")?;
        let len = crate::backend::copy_len(len, buf.size(), "record_zero_buffer", "destination")?;
        let Some(len) = memory::fill_size(len)? else {
            return Ok(());
        };
        let page = &mut *page;
        if page.zeros.as_ref().is_none_or(|z| z.width() < len) {
            let size = usize::try_from(len).map_err(BackendError::new)?;
            let zeros = memory::alloc_device(&self.device, size, true, "zero-fill source")?;
            if let Some(old) = page.zeros.replace(zeros) {
                page.commands.hold(old.resource());
            }
        }
        let zeros = page.zeros.as_ref().map(DeviceBuffer::resource).cloned();
        let zeros = zeros.ok_or_else(|| BackendError::msg("no zero-fill source"))?;
        let c = &mut page.commands;
        c.use_buffer(&zeros, D3D12_RESOURCE_STATE_COPY_SOURCE, false);
        c.use_buffer(buf.resource(), D3D12_RESOURCE_STATE_COPY_DEST, true);
        c.copy(buf.resource(), &zeros, len);
        Ok(())
    }

    /// Record a dispatch of `kernel` over `groups` thread groups with the
    /// root constants `constants`, the SRVs `srvs` (`t0..`) and the UAVs
    /// `uavs` (`u0..`), each a buffer and the stride of its elements.
    ///
    /// # Errors
    /// When the bindings do not match the kernel's registers, a view holds
    /// more elements than a view may ([`memory::check_view_elements`]),
    /// a dimension of `groups` exceeds the dispatch limit, or the page has
    /// recorded [`MAX_DISPATCHES_PER_PAGE`] dispatches
    /// ([`BackendError::DescriptorPoolExhausted`]).
    pub(super) fn dispatch(
        &self,
        kernel: KernelId,
        srvs: &[(&DeviceBuffer, u32)],
        uavs: &[(&mut DeviceBuffer, u32)],
        constants: &[u32],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let label = kernel.label();
        let pipeline = self.pipelines.get(kernel)?;
        let shape = pipeline.shape;
        let views = check_bindings(label, shape, srvs, uavs, constants, groups)?;

        let mut page = self.recording("dispatch")?;
        if page.dispatches >= MAX_DISPATCHES_PER_PAGE {
            return Err(BackendError::DescriptorPoolExhausted {
                allocated: page.dispatches,
                max: MAX_DISPATCHES_PER_PAGE,
            });
        }
        let first = page.dispatches * MAX_BINDINGS;
        // SAFETY: the heap is live.
        let cpu_start = unsafe { page.heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: the heap is live and shader-visible.
        let gpu_start = unsafe { page.heap.GetGPUDescriptorHandleForHeapStart() };
        for (i, &(buf, stride)) in views.iter().enumerate() {
            let slot = first + u32::try_from(i).unwrap_or(u32::MAX);
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + (slot * self.descriptor_size) as usize,
            };
            let elements = u32::try_from(buf.width() / u64::from(stride)).unwrap_or(u32::MAX);
            if i < srvs.len() {
                self.write_srv(buf.resource(), elements, stride, handle);
            } else {
                self.write_uav(buf.resource(), elements, stride, handle);
            }
        }
        let table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu_start.ptr + u64::from(first * self.descriptor_size),
        };

        let c = &mut page.commands;
        for &(buf, _) in srvs {
            c.use_buffer(
                buf.resource(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                false,
            );
        }
        for (buf, _) in uavs {
            c.use_buffer(buf.resource(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        }
        c.flush();
        // SAFETY: the list is open; the pipeline objects are held by the
        // cache, the views were written above into the bound heap, and
        // `constants` holds the root signature's constant words.
        unsafe {
            c.list.SetPipelineState(&pipeline.state);
            c.list.SetComputeRootSignature(&pipeline.root_signature);
            if shape.constant_words > 0 {
                c.list.SetComputeRoot32BitConstants(
                    0,
                    shape.constant_words,
                    constants.as_ptr().cast(),
                    0,
                );
            }
            if shape.srvs + shape.uavs > 0 {
                c.list
                    .SetComputeRootDescriptorTable(shape.table_index(), table);
            }
            c.list.Dispatch(groups.0, groups.1, groups.2);
        }
        page.dispatches += 1;
        Ok(())
    }

    /// Write a structured-buffer SRV of `elements` elements of `stride`
    /// bytes over `resource` at `handle`.
    fn write_srv(
        &self,
        resource: &ID3D12Resource,
        elements: u32,
        stride: u32,
        handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: stride,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        // SAFETY: `handle` is a slot of the page heap no recorded dispatch
        // of a running submission reads; `desc` outlives the call and
        // describes `elements × stride` bytes within `resource`.
        unsafe {
            self.device
                .CreateShaderResourceView(resource, Some(&raw const desc), handle);
        }
    }

    /// Write a structured-buffer UAV, as [`Self::write_srv`].
    fn write_uav(
        &self,
        resource: &ID3D12Resource,
        elements: u32,
        stride: u32,
        handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: stride,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                },
            },
        };
        // SAFETY: as for `write_srv`; `resource` allows unordered access.
        unsafe {
            self.device.CreateUnorderedAccessView(
                resource,
                None::<&ID3D12Resource>,
                Some(&raw const desc),
                handle,
            );
        }
    }

    /// Copy `src` into the start of `dst` and wait for the copy.
    pub(super) fn upload(&self, dst: &DeviceBuffer, src: &[u8]) -> Result<()> {
        let len = crate::backend::copy_len(src.len(), dst.size(), "upload", "destination")?;
        if len == 0 {
            return Ok(());
        }
        let mut t = self.transfer.lock().unwrap_or_else(PoisonError::into_inner);
        let t = &mut *t;
        let staging = staging(&self.device, &mut t.upload, src.len(), HostHeap::Upload)?;
        staging.as_mut_slice()[..src.len()].copy_from_slice(src);
        let resource = staging.resource().clone();
        let run = self.one_shot(&mut t.commands, |c| {
            c.hold(&resource);
            c.use_buffer(dst.resource(), D3D12_RESOURCE_STATE_COPY_DEST, true);
            c.copy(dst.resource(), &resource, len);
        });
        trim(&mut t.upload);
        run
    }

    /// Copy the first `dst.len()` bytes of `src` into `dst` and wait for
    /// the copy.
    pub(super) fn download(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        let len = crate::backend::copy_len(dst.len(), src.size(), "download", "source")?;
        if len == 0 {
            return Ok(());
        }
        let mut t = self.transfer.lock().unwrap_or_else(PoisonError::into_inner);
        let t = &mut *t;
        let resource = staging(&self.device, &mut t.download, dst.len(), HostHeap::Readback)?
            .resource()
            .clone();
        let run = self.one_shot(&mut t.commands, |c| {
            c.hold(&resource);
            c.use_buffer(src.resource(), D3D12_RESOURCE_STATE_COPY_SOURCE, false);
            c.copy(&resource, src.resource(), len);
        });
        if run.is_ok()
            && let Some(staging) = &t.download
        {
            dst.copy_from_slice(&staging.as_slice()[..dst.len()]);
        }
        trim(&mut t.download);
        run
    }

    /// Record `record` into `commands`, submit it and wait for it.
    fn one_shot(&self, commands: &mut Commands, record: impl FnOnce(&mut Commands)) -> Result<()> {
        commands.begin()?;
        record(commands);
        let run = commands
            .finish()
            .and_then(|()| self.submitter.submit(&commands.list))
            .and_then(|fence| self.submitter.wait(&fence));
        if run.is_err() {
            // The submission may still run: wait for the queue before the
            // allocator can be reset.
            if let Err(e) = self.submitter.wait_idle() {
                log::warn!("d3d12: waiting after a failed transfer failed: {e}");
            }
        }
        commands.discard();
        run
    }
}

/// The views of a dispatch of the kernel `label`, SRVs then UAVs, after
/// checking them, the root constants and `groups` against `shape` and the
/// device's limits.
fn check_bindings<'a>(
    label: &'static str,
    shape: Shape,
    srvs: &[(&'a DeviceBuffer, u32)],
    uavs: &'a [(&mut DeviceBuffer, u32)],
    constants: &[u32],
    groups: (u32, u32, u32),
) -> Result<Vec<(&'a DeviceBuffer, u32)>> {
    if srvs.len() != shape.srvs as usize
        || uavs.len() != shape.uavs as usize
        || constants.len() != shape.constant_words as usize
    {
        return Err(BackendError::msg(format!(
            "{label}: {} SRVs, {} UAVs and {} constant words bound; the kernel has {}, {} and {}",
            srvs.len(),
            uavs.len(),
            constants.len(),
            shape.srvs,
            shape.uavs,
            shape.constant_words
        )));
    }
    let max = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    if groups.0 > max || groups.1 > max || groups.2 > max {
        return Err(BackendError::msg(format!(
            "{label}: dispatch ({},{},{}) exceeds the {max} thread groups a dimension allows",
            groups.0, groups.1, groups.2
        )));
    }
    let views: Vec<(&DeviceBuffer, u32)> = srvs
        .iter()
        .copied()
        .chain(uavs.iter().map(|(b, s)| (&**b, *s)))
        .collect();
    for &(buf, stride) in &views {
        if stride == 0 || buf.width() < u64::from(stride) {
            return Err(BackendError::msg(format!(
                "{label}: a view of stride {stride} over a buffer of {} bytes",
                buf.width()
            )));
        }
        memory::check_view_elements(label, buf.width(), stride)?;
    }
    for (i, (s, _)) in srvs.iter().enumerate() {
        if uavs.iter().any(|(u, _)| u.resource() == s.resource()) {
            return Err(BackendError::msg(format!(
                "{label}: SRV t{i} is also bound as a UAV"
            )));
        }
    }
    Ok(views)
}

/// The staging buffer in `slot`, replaced by one of `len` bytes in `heap`
/// when it is smaller.
fn staging<'a>(
    device: &ID3D12Device,
    slot: &'a mut Option<HostBuffer>,
    len: usize,
    heap: HostHeap,
) -> Result<&'a mut HostBuffer> {
    if slot.as_ref().is_none_or(|s| s.as_slice().len() < len) {
        *slot = None;
        *slot = Some(memory::alloc_host(device, len, heap, "transfer staging")?);
    }
    slot.as_mut()
        .ok_or_else(|| BackendError::msg("no staging buffer"))
}

/// Free the staging buffer in `slot` when it is larger than
/// [`KEPT_STAGING_BYTES`].
fn trim(slot: &mut Option<HostBuffer>) {
    if slot
        .as_ref()
        .is_some_and(|s| s.as_slice().len() > KEPT_STAGING_BYTES)
    {
        *slot = None;
    }
}
