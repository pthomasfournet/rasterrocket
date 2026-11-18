//! Adapter from `ash`'s `vk::Result` to `BackendError`.

use ash::vk;

use crate::backend::BackendError;

/// Wrap a Vulkan call's `vk::Result` failure into a `BackendError` with
/// the failing call's name baked into the message.
///
/// Use as `unsafe { instance.create_xyz(...) }.map_err(vk_err("vkCreateXyz"))?`.
pub(super) fn vk_err(call: &'static str) -> impl Fn(vk::Result) -> BackendError + use<> {
    move |code| BackendError::msg(describe(call, code))
}

/// What a lost device means where the code runs.  Windows resets a GPU
/// whose submission runs past its timeout detection delay (TDR, 2 s by
/// default), and every call on the device then returns
/// `ERROR_DEVICE_LOST`.
const DEVICE_LOST_CAUSE: &str = if cfg!(windows) {
    "the device was lost; on Windows the likely cause is a GPU timeout \
     (TDR: a submission that runs past about 2 s resets the device)"
} else {
    "the device was lost: a driver reset or a GPU fault"
};

/// `call failed: CODE`, with [`DEVICE_LOST_CAUSE`] appended for
/// `ERROR_DEVICE_LOST`.
fn describe(call: &str, code: vk::Result) -> String {
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
