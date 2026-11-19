//! Adapter from `ash`'s `vk::Result` to `BackendError`.

use ash::vk;

use crate::backend::{BackendError, DEVICE_LOST_CAUSE};

/// Wrap a Vulkan call's `vk::Result` failure into a `BackendError` with
/// the failing call's name baked into the message.
///
/// Use as `unsafe { instance.create_xyz(...) }.map_err(vk_err("vkCreateXyz"))?`.
pub(super) fn vk_err(call: &'static str) -> impl Fn(vk::Result) -> BackendError + use<> {
    move |code| BackendError::msg(describe(call, code))
}

/// `call failed: CODE`, with [`DEVICE_LOST_CAUSE`] appended for
/// `ERROR_DEVICE_LOST`.
pub(super) fn describe(call: &str, code: vk::Result) -> String {
    if code == vk::Result::ERROR_DEVICE_LOST {
        format!("{call} failed: {code:?} ({DEVICE_LOST_CAUSE})")
    } else {
        format!("{call} failed: {code:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lost_device_names_its_likely_cause() {
        let msg = describe("vkWaitSemaphores", vk::Result::ERROR_DEVICE_LOST);
        assert!(
            msg.starts_with("vkWaitSemaphores failed: ERROR_DEVICE_LOST ("),
            "{msg}"
        );
        assert_eq!(cfg!(windows), msg.contains("TDR"), "{msg}");
    }

    #[test]
    fn other_codes_name_only_the_call_and_code() {
        assert_eq!(
            describe("vkAllocateMemory", vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            "vkAllocateMemory failed: ERROR_OUT_OF_DEVICE_MEMORY"
        );
    }
}
