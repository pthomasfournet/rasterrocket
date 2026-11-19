//! Queue submission and its fence: every command list the backend
//! executes signals the next value of one fence, so a value names one
//! submission and its completion implies every earlier one's.

use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FENCE_FLAG_NONE, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence,
    ID3D12GraphicsCommandList,
};
use windows::core::Interface as _;

use super::error::device_err;
use crate::backend::{BackendError, DEVICE_LOST_CAUSE, Result};

/// The completion of one submission: the fence and the value the
/// submission signals.
#[derive(Clone)]
pub struct PageFence(Arc<(ID3D12Fence, u64)>);

impl std::fmt::Debug for PageFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PageFence").field(&self.0.1).finish()
    }
}

/// The compute queue and the fence its submissions signal.
pub(super) struct Submitter {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The value the last submission signalled; held across each
    /// execute-and-signal so that values rise in queue order.
    last: Mutex<u64>,
}

impl Submitter {
    /// A submitter on `queue` with a fresh fence at 0.
    pub(super) fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<Self> {
        // SAFETY: `device` is live; the fence is returned by value.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(device_err(device, "CreateFence"))?;
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            fence,
            last: Mutex::new(0),
        })
    }

    /// Execute the closed `list` and signal the next fence value after it.
    pub(super) fn submit(&self, list: &ID3D12GraphicsCommandList) -> Result<PageFence> {
        let list: ID3D12CommandList = list.cast().map_err(device_err(
            &self.device,
            "QueryInterface(ID3D12CommandList)",
        ))?;
        let mut signalled = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let value = signalled
            .checked_add(1)
            .ok_or_else(|| BackendError::msg("the D3D12 fence value overflowed u64"))?;
        // SAFETY: `list` is closed and every resource it references is
        // held by its recorder until the fence of this submission is
        // waited on.
        unsafe { self.queue.ExecuteCommandLists(&[Some(list)]) };
        // SAFETY: `queue` and `fence` are live.
        unsafe { self.queue.Signal(&self.fence, value) }
            .map_err(device_err(&self.device, "ID3D12CommandQueue::Signal"))?;
        *signalled = value;
        drop(signalled);
        Ok(PageFence(Arc::new((self.fence.clone(), value))))
    }

    /// Block until `fence` has been reached.
    pub(super) fn wait(&self, fence: &PageFence) -> Result<()> {
        self.wait_value(fence.0.1)
    }

    /// Block until every submission made so far has completed.
    pub(super) fn wait_idle(&self) -> Result<()> {
        let last = *self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.wait_value(last)
    }

    /// Block until the fence reaches `value`.  A removed device sets the
    /// fence to `u64::MAX`, which is reported as the loss it is.
    fn wait_value(&self, value: u64) -> Result<()> {
        // SAFETY: `fence` is live.
        let mut done = unsafe { self.fence.GetCompletedValue() };
        if done < value {
            // SAFETY: a null event makes the call return once the fence
            // reaches `value`.
            unsafe { self.fence.SetEventOnCompletion(value, HANDLE::default()) }
                .map_err(device_err(&self.device, "SetEventOnCompletion"))?;
            // SAFETY: `fence` is live.
            done = unsafe { self.fence.GetCompletedValue() };
        }
        if done == u64::MAX {
            // SAFETY: `device` is live; the call has no preconditions.
            let reason = match unsafe { self.device.GetDeviceRemovedReason() } {
                Ok(()) => "none reported".to_owned(),
                Err(e) => e.code().to_string(),
            };
            return Err(BackendError::msg(format!(
                "waiting for a D3D12 submission failed: the fence reads u64::MAX \
                 (removed reason: {reason}); {DEVICE_LOST_CAUSE}"
            )));
        }
        Ok(())
    }
}
