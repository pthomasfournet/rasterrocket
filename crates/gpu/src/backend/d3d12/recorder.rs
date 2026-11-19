//! The page recorder: one command allocator and list ([`Commands`]),
//! recorded between `begin_page` and `submit_page` under one lock, with
//! one shader-visible descriptor heap of [`MAX_BINDINGS`] views a
//! dispatch.  A page whose completion cannot be confirmed leaves the
//! recorder lost: its commands are stranded and no further page begins.

#![expect(
    clippy::significant_drop_tightening,
    reason = "the recording state's lock is held across the D3D12 calls that record into its list"
)]

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUFFER_SRV, D3D12_BUFFER_SRV_FLAG_NONE, D3D12_BUFFER_UAV, D3D12_BUFFER_UAV_FLAG_NONE,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, D3D12_DESCRIPTOR_HEAP_DESC,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0, D3D12_SRV_DIMENSION_BUFFER,
    D3D12_UAV_DIMENSION_BUFFER, D3D12_UNORDERED_ACCESS_VIEW_DESC,
    D3D12_UNORDERED_ACCESS_VIEW_DESC_0, ID3D12DescriptorHeap, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;

use super::commands::Commands;
use super::error::device_err;
use super::fence::{PageFence, SubmitError, Submitter};
use super::memory::{self, DeviceBuffer, HostBuffer, HostHeap};
use super::pipeline::{KernelId, MAX_BINDINGS, PipelineCache, Shape};
use crate::backend::{BackendError, Result};

/// Dispatches one page may record: each takes [`MAX_BINDINGS`]
/// descriptors of the page's heap.
pub(super) const MAX_DISPATCHES_PER_PAGE: u32 = 64;

/// The page's place in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No page; `begin_page` is valid.
    Idle,
    /// Between `begin_page` and `submit_page`.
    Recording,
    /// Submitted; only `wait_page` is valid.
    Submitted,
    /// A submitted page's completion could not be confirmed: its commands
    /// are stranded and no further page is accepted.
    Lost,
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

/// The per-page recorder.
pub(super) struct Recorder {
    device: ID3D12Device,
    submitter: Arc<Submitter>,
    pipelines: PipelineCache,
    descriptor_size: u32,
    page: Mutex<Page>,
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
        if page.phase == Phase::Lost {
            return Err(BackendError::msg(format!(
                "begin_page: an earlier page's completion could not be confirmed; {}",
                crate::backend::DEVICE_LOST_CAUSE
            )));
        }
        if page.phase != Phase::Idle {
            return Err(BackendError::msg(format!(
                "begin_page called from {:?} state; expected Idle",
                page.phase
            )));
        }
        let page = &mut *page;
        page.commands.begin()?;
        page.commands.hold(&page.heap);
        let heaps = [Some(page.heap.clone())];
        // SAFETY: the list is open; the heap is held by the list's commands.
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
            .map_err(|error| SubmitError {
                error,
                queued: false,
            })
            .and_then(|()| self.submitter.submit(&page.commands.list));
        match submitted {
            Ok(fence) => {
                page.submitted = Some(fence.clone());
                page.phase = Phase::Submitted;
                Ok(fence)
            }
            Err(SubmitError {
                error,
                queued: true,
            }) => {
                page.commands.strand();
                page.phase = Phase::Lost;
                Err(error)
            }
            Err(SubmitError { error, .. }) => {
                page.commands.discard();
                page.phase = Phase::Idle;
                Err(error)
            }
        }
    }

    /// Wait for the submitted page: on its own fence, and on `fence` as
    /// well when that is another submission's.  A failed wait on the
    /// page's own fence strands its commands; the recorder then accepts no
    /// further page.
    pub(super) fn wait_page(&self, fence: &PageFence) -> Result<()> {
        let mut page = self.lock();
        let own = match (&page.phase, &page.submitted) {
            (Phase::Submitted, Some(own)) => own.clone(),
            _ => {
                return Err(BackendError::msg(format!(
                    "wait_page called from {:?} state; expected Submitted",
                    page.phase
                )));
            }
        };
        if let Err(e) = self.submitter.wait(&own) {
            page.commands.strand();
            page.submitted = None;
            page.phase = Phase::Lost;
            return Err(e);
        }
        // The page has completed: its references may go whatever the wait
        // on the caller's fence reports.
        page.commands.release();
        page.submitted = None;
        page.phase = Phase::Idle;
        if own.same(fence) {
            Ok(())
        } else {
            self.submitter.wait(fence)
        }
    }

    /// Strand the commands of a submitted page whose completion cannot be
    /// confirmed, and accept no further page.
    pub(super) fn strand_in_flight(&self) {
        let mut page = self.lock();
        if page.phase == Phase::Submitted {
            page.commands.strand();
            page.submitted = None;
            page.phase = Phase::Lost;
        }
    }

    /// Return to `Idle`: a recording page is discarded, a submitted one is
    /// waited for first.  A failed wait is logged and strands the page's
    /// commands; the recorder then accepts no further page.
    pub(super) fn abandon_page(&self) {
        let mut page = self.lock();
        if page.phase == Phase::Lost {
            return;
        }
        if page.phase == Phase::Submitted
            && let Some(fence) = page.submitted.take()
            && let Err(e) = self.submitter.wait(&fence)
        {
            log::warn!("d3d12: abandoning a submitted page: waiting for it failed: {e}");
            page.commands.strand();
            page.phase = Phase::Lost;
            return;
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
        c.hold(&pipeline.state);
        c.hold(&pipeline.root_signature);
        c.flush();
        // SAFETY: the list is open; the pipeline objects are held by the
        // list's commands, the views were written above into the bound heap, and
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
