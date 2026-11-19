//! Synchronous upload and download: a copy through a staging buffer in a
//! submission of its own, waited for before the call returns.

use std::sync::{Arc, Mutex, PoisonError};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE, ID3D12Device,
};

use super::commands::Commands;
use super::fence::{SubmitError, Submitter};
use super::memory::{self, DeviceBuffer, HostBuffer, HostHeap};
use crate::backend::{BackendError, Result, copy_len};

/// A staging buffer larger than this is freed after its transfer rather
/// than kept for the next.
const KEPT_STAGING_BYTES: usize = 16 << 20;

/// The transfers' command state and staging buffers.
struct State {
    commands: Commands,
    upload: Option<HostBuffer>,
    download: Option<HostBuffer>,
}

/// Synchronous transfers, one at a time.
pub(super) struct Transfer {
    device: ID3D12Device,
    submitter: Arc<Submitter>,
    state: Mutex<State>,
}

impl Transfer {
    pub(super) fn new(device: &ID3D12Device, submitter: Arc<Submitter>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            submitter,
            state: Mutex::new(State {
                commands: Commands::new(device)?,
                upload: None,
                download: None,
            }),
        })
    }

    /// Copy `src` into the start of `dst` and wait for the copy.
    pub(super) fn upload(&self, dst: &DeviceBuffer, src: &[u8]) -> Result<()> {
        let len = copy_len(src.len(), dst.size(), "upload", "destination")?;
        if len == 0 {
            return Ok(());
        }
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let t = &mut *guard;
        refuse_stranded(&t.commands)?;
        let staging = staging(&self.device, &mut t.upload, src.len(), HostHeap::Upload)?;
        staging.as_mut_slice()[..src.len()].copy_from_slice(src);
        let resource = staging.resource().clone();
        let run = self.one_shot(&mut t.commands, |c| {
            c.hold(&resource);
            c.use_buffer(dst.resource(), D3D12_RESOURCE_STATE_COPY_DEST, true);
            c.copy(dst.resource(), &resource, len);
        });
        if !t.commands.is_stranded() {
            trim(&mut t.upload);
        }
        drop(guard);
        run
    }

    /// Copy the first `dst.len()` bytes of `src` into `dst` and wait for
    /// the copy.
    pub(super) fn download(&self, src: &DeviceBuffer, dst: &mut [u8]) -> Result<()> {
        let len = copy_len(dst.len(), src.size(), "download", "source")?;
        if len == 0 {
            return Ok(());
        }
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let t = &mut *guard;
        refuse_stranded(&t.commands)?;
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
        if !t.commands.is_stranded() {
            trim(&mut t.download);
        }
        drop(guard);
        run
    }

    /// Record `record` into `commands`, submit it and wait for it.  A
    /// submission whose completion cannot be confirmed strands `commands`.
    fn one_shot(&self, commands: &mut Commands, record: impl FnOnce(&mut Commands)) -> Result<()> {
        commands.begin()?;
        record(commands);
        if let Err(e) = commands.finish() {
            commands.discard();
            return Err(e);
        }
        match self.submitter.submit(&commands.list) {
            Ok(fence) => match self.submitter.wait(&fence) {
                Ok(()) => {
                    commands.release();
                    Ok(())
                }
                Err(e) => {
                    commands.strand();
                    Err(e)
                }
            },
            Err(SubmitError {
                error,
                queued: true,
            }) => {
                commands.strand();
                Err(error)
            }
            Err(SubmitError { error, .. }) => {
                commands.discard();
                Err(error)
            }
        }
    }
}

/// Fail when an earlier transfer's completion could not be confirmed: its
/// list and staging buffers may still be in use.
fn refuse_stranded(commands: &Commands) -> Result<()> {
    if commands.is_stranded() {
        return Err(BackendError::msg(format!(
            "an earlier D3D12 transfer's completion could not be confirmed; {}",
            crate::backend::DEVICE_LOST_CAUSE
        )));
    }
    Ok(())
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
