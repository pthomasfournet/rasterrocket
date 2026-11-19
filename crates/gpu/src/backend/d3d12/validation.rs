//! The debug layer's validation messages: with the layer enabled
//! (`PDF_RASTER_D3D12_DEBUG=1`), the device's info queue stores them, and a
//! page or transfer after which the queue holds a message of severity
//! `ERROR` or `CORRUPTION` fails with them.  Without the layer the device
//! has no info queue and nothing is read.

use std::sync::{Mutex, PoisonError};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_MESSAGE, D3D12_MESSAGE_SEVERITY_ERROR, ID3D12Device, ID3D12InfoQueue,
};
use windows::core::Interface as _;

use crate::backend::{BackendError, Result};

/// The info queue of a device created under the debug layer.
pub(crate) struct Validation {
    /// `None` without the debug layer.  The lock makes reading the stored
    /// messages and clearing them one step.
    queue: Option<Mutex<ID3D12InfoQueue>>,
}

impl Validation {
    /// The info queue of `device` when `debug_layer` is on; none otherwise.
    /// A device of a layer that offers no info queue is logged and read as
    /// none.
    pub(crate) fn new(device: &ID3D12Device, debug_layer: bool) -> Self {
        if !debug_layer {
            return Self { queue: None };
        }
        match device.cast::<ID3D12InfoQueue>() {
            Ok(queue) => Self {
                queue: Some(Mutex::new(queue)),
            },
            Err(e) => {
                log::warn!("d3d12: debug layer on, but the device has no info queue: {e}");
                Self { queue: None }
            }
        }
    }

    /// Whether validation messages are read.
    #[cfg(test)]
    pub(crate) const fn is_active(&self) -> bool {
        self.queue.is_some()
    }

    /// The stored messages of severity `ERROR` or `CORRUPTION`, each as
    /// its text; every stored message is then cleared.
    fn take_errors(&self) -> Vec<String> {
        let Some(queue) = &self.queue else {
            return Vec::new();
        };
        let queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `queue` is live.
        let n = unsafe { queue.GetNumStoredMessages() };
        let mut errors = Vec::new();
        for i in 0..n {
            match message(&queue, i) {
                Some((severity, text)) if severity <= D3D12_MESSAGE_SEVERITY_ERROR.0 => {
                    errors.push(text);
                }
                Some(_) => {}
                None => errors.push(format!("validation message {i} could not be read")),
            }
        }
        // SAFETY: `queue` is live.
        unsafe { queue.ClearStoredMessages() };
        errors
    }

    /// Fail `what` when the debug layer stored a message of severity
    /// `ERROR` or above since the last check.
    ///
    /// # Errors
    /// A [`BackendError::Other`] listing the messages.
    pub(crate) fn check(&self, what: &str) -> Result<()> {
        let errors = self.take_errors();
        if errors.is_empty() {
            return Ok(());
        }
        Err(BackendError::msg(format!(
            "{what}: the D3D12 debug layer reported {} error(s): {}",
            errors.len(),
            errors.join(" | ")
        )))
    }

    /// Store `text` as an application message of severity `ERROR`.
    #[cfg(test)]
    pub(crate) fn add_error(&self, text: &str) -> Result<()> {
        let Some(queue) = &self.queue else {
            return Err(BackendError::msg("no info queue"));
        };
        let text = std::ffi::CString::new(text).map_err(BackendError::new)?;
        let queue = queue.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `queue` is live and `text` a NUL-terminated string that
        // outlives the call, which copies it.
        unsafe {
            queue.AddApplicationMessage(
                D3D12_MESSAGE_SEVERITY_ERROR,
                windows::core::PCSTR(text.as_ptr().cast()),
            )
        }
        .map_err(BackendError::new)
    }
}

/// The severity and text of the stored message `index` of `queue`; `None`
/// when it cannot be read.
fn message(queue: &ID3D12InfoQueue, index: u64) -> Option<(i32, String)> {
    let mut len = 0_usize;
    // SAFETY: `queue` is live; with no message pointer the call writes the
    // message's byte length to `len` only.
    unsafe { queue.GetMessage(index, None, &raw mut len) }.ok()?;
    if len < size_of::<D3D12_MESSAGE>() {
        return None;
    }
    // `u64` elements align the buffer for `D3D12_MESSAGE`, whose fields
    // are pointer-sized at most.
    let mut buf = vec![0_u64; len.div_ceil(size_of::<u64>())];
    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
    // SAFETY: `msg` points to `len` writable bytes, aligned for
    // `D3D12_MESSAGE`, the size the first call reported.
    unsafe { queue.GetMessage(index, Some(msg), &raw mut len) }.ok()?;
    // SAFETY: the call wrote a `D3D12_MESSAGE` at `msg`.
    let msg = unsafe { &*msg };
    let text = if msg.pDescription.is_null() {
        String::new()
    } else {
        // SAFETY: `pDescription` points to `DescriptionByteLength` bytes
        // within `buf`, which is live and unchanged until the slice is
        // dropped.
        let bytes =
            unsafe { std::slice::from_raw_parts(msg.pDescription, msg.DescriptionByteLength) };
        String::from_utf8_lossy(bytes.strip_suffix(&[0]).unwrap_or(bytes)).into_owned()
    };
    Some((msg.Severity.0, text))
}
