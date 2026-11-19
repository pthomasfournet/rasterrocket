//! Adapter from `windows::core::Error` to `BackendError`.

use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{
    DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET,
};
use windows::core::{Error, HRESULT};

use crate::backend::{BackendError, DEVICE_LOST_CAUSE};

/// Wrap a failed call's `windows::core::Error` into a `BackendError` with
/// the call's name baked into the message.
///
/// Use as `unsafe { factory.EnumWarpAdapter() }.map_err(hr_err("EnumWarpAdapter"))?`.
pub(super) fn hr_err(call: &'static str) -> impl Fn(Error) -> BackendError + use<> {
    move |e| BackendError::msg(describe(call, e.code()))
}

/// [`hr_err`] for a call on `device`: a code that reports the device
/// removed also names the reason `GetDeviceRemovedReason` gives.
pub(super) fn device_err(
    device: &ID3D12Device,
    call: &'static str,
) -> impl Fn(Error) -> BackendError + use<> {
    let device = device.clone();
    move |e| {
        let code = e.code();
        let mut msg = describe(call, code);
        if is_removal(code) {
            // SAFETY: `device` is a live interface pointer; the call has no
            // preconditions.
            let reason = match unsafe { device.GetDeviceRemovedReason() } {
                Ok(()) => "none reported".to_owned(),
                Err(r) => code_text(r.code()),
            };
            msg.push_str("; removed reason: ");
            msg.push_str(&reason);
        }
        BackendError::msg(msg)
    }
}

/// Whether `code` reports the device removed, hung or reset.
fn is_removal(code: HRESULT) -> bool {
    [
        DXGI_ERROR_DEVICE_REMOVED,
        DXGI_ERROR_DEVICE_HUNG,
        DXGI_ERROR_DEVICE_RESET,
    ]
    .contains(&code)
}

/// `CODE` and, when the system has one, its message: `0x887A0005 (text)`.
fn code_text(code: HRESULT) -> String {
    let text = code.message();
    let text = text.trim();
    if text.is_empty() {
        code.to_string()
    } else {
        format!("{code} ({text})")
    }
}

/// `call failed: CODE (text)`, with [`DEVICE_LOST_CAUSE`] appended for the
/// codes that report the device removed, hung or reset.
pub(super) fn describe(call: &str, code: HRESULT) -> String {
    let text = code_text(code);
    if is_removal(code) {
        format!("{call} failed: {text}; {DEVICE_LOST_CAUSE}")
    } else {
        format!("{call} failed: {text}")
    }
}

#[cfg(test)]
mod tests {
    use windows::Win32::Foundation::E_OUTOFMEMORY;
    use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;

    use super::*;

    #[test]
    fn removed_device_names_the_tdr() {
        let msg = describe("Signal", DXGI_ERROR_DEVICE_REMOVED);
        assert!(msg.starts_with("Signal failed: 0x887A0005"), "{msg}");
        assert!(msg.contains("TDR"), "{msg}");
    }

    #[test]
    fn other_codes_name_only_the_call_and_code() {
        let msg = describe("CreateCommittedResource", E_OUTOFMEMORY);
        assert!(
            msg.starts_with("CreateCommittedResource failed: 0x8007000E"),
            "{msg}"
        );
        assert!(!msg.contains("TDR"), "{msg}");
    }
}
