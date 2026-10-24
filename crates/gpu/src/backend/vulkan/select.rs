//! Physical-device selection over plain per-device data.
//!
//! [`Candidate`] holds the properties of one enumerated physical device that
//! selection depends on.  The functions here choose among candidates without
//! any driver call, so every matching and ordering rule is testable without
//! the hardware it describes.

use std::cmp::Reverse;
use std::fmt::{self, Write as _};
use std::str::FromStr;

use ash::vk;

use crate::backend::BackendError;

/// Explicit choice of Vulkan physical device.
///
/// Resolution against the enumerated devices: a selector of ASCII digits
/// below the device count is a position in `vkEnumeratePhysicalDevices`
/// order, counted from 0.  Any other selector is a case-insensitive
/// substring of the device name; an exact (case-insensitive) name match
/// takes precedence over substring matches, and more than one remaining
/// match is an error.  Digits that are not a valid position are matched
/// as a name substring, so a model number selects by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSelector(String);

impl FromStr for DeviceSelector {
    type Err = BackendError;

    /// Surrounding whitespace is trimmed; empty input is rejected.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(BackendError::DeviceSelection(
                "Vulkan device selector is empty; expected an index or a device-name substring"
                    .to_owned(),
            ));
        }
        Ok(Self(s.to_owned()))
    }
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Why an enumerated device cannot host the compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ineligible {
    /// The driver reports an API version below 1.3.
    ApiBelow13,
    /// No queue family advertises `VK_QUEUE_COMPUTE_BIT`.
    NoComputeQueue,
}

impl fmt::Display for Ineligible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ApiBelow13 => "Vulkan API below 1.3",
            Self::NoComputeQueue => "no compute queue family",
        })
    }
}

/// One enumerated physical device, reduced to the data selection reads.
#[derive(Debug, Clone)]
pub(super) struct Candidate {
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub(super) name: String,
    /// `VkPhysicalDeviceProperties::deviceType`.
    pub(super) device_type: vk::PhysicalDeviceType,
    /// Index of the first compute-capable queue family, or the reason the
    /// device is unusable.
    pub(super) compute_queue: Result<u32, Ineligible>,
    /// Sum of the `DEVICE_LOCAL` memory heap sizes, in bytes.
    pub(super) device_local_bytes: u64,
    /// PCI address, when the driver exposes `VK_EXT_pci_bus_info`.
    pub(super) pci: Option<PciAddress>,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`; all zero when the device
    /// reports an API version below 1.1 and the query is skipped.
    pub(super) uuid: [u8; 16],
}

/// PCI domain, bus, device, and function of a physical device.  Field
/// order makes the derived ordering the conventional bus order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PciAddress {
    pub(super) domain: u32,
    pub(super) bus: u32,
    pub(super) device: u32,
    pub(super) function: u32,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// Rank physical-device types: discrete < integrated < virtual < CPU < other.
/// Lower ranks are preferred.
const fn device_type_rank(t: vk::PhysicalDeviceType) -> u8 {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    }
}

/// Choose a device from `cands` (in enumeration order).
///
/// With a selector, the selected device is returned or a
/// [`BackendError::DeviceSelection`] listing every enumerated device; the
/// automatic ranking is not consulted.  Without one, the best-ranked
/// eligible device is returned.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] when the selector matches no device,
/// matches several, or names an ineligible device; [`BackendError::Other`]
/// when no selector is given and no device is eligible.
pub(super) fn pick(
    cands: &[Candidate],
    selector: Option<&DeviceSelector>,
) -> Result<usize, BackendError> {
    selector.map_or_else(
        || {
            pick_auto(cands).ok_or_else(|| {
                BackendError::msg(format!(
                    "no Vulkan 1.3+ device with a compute queue family was found; enumerated:\n{}",
                    describe(cands)
                ))
            })
        },
        |sel| {
            resolve_selector(cands, sel).map_err(|reason| {
                BackendError::DeviceSelection(format!(
                    "Vulkan device selector {sel}: {reason}\navailable Vulkan devices:\n{}",
                    describe(cands)
                ))
            })
        },
    )
}

/// The eligible candidate that sorts first by, in order: device-type rank;
/// larger device-local memory; known PCI address before unknown, then
/// ascending PCI address; ascending device UUID.  Every key is a property
/// of the device itself, so the choice does not depend on enumeration
/// order; the enumeration index decides only between candidates identical
/// in every key.
fn pick_auto(cands: &[Candidate]) -> Option<usize> {
    cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.compute_queue.is_ok())
        .min_by_key(|&(i, c)| {
            (
                device_type_rank(c.device_type),
                Reverse(c.device_local_bytes),
                c.pci.is_none(),
                c.pci,
                c.uuid,
                i,
            )
        })
        .map(|(i, _)| i)
}

/// Resolve `sel` to one eligible candidate index, or a reason it does not.
fn resolve_selector(cands: &[Candidate], sel: &DeviceSelector) -> Result<usize, String> {
    let raw = sel.0.as_str();
    let all_digits = raw.bytes().all(|b| b.is_ascii_digit());
    let as_index = if all_digits {
        raw.parse::<usize>().ok()
    } else {
        None
    };
    let idx = match as_index {
        Some(i) if i < cands.len() => i,
        _ => {
            let needle = raw.to_lowercase();
            let lowered: Vec<String> = cands.iter().map(|c| c.name.to_lowercase()).collect();
            let exact: Vec<usize> = (0..cands.len()).filter(|&i| lowered[i] == needle).collect();
            let partial: Vec<usize> = (0..cands.len())
                .filter(|&i| lowered[i].contains(&needle))
                .collect();
            match (exact.as_slice(), partial.as_slice()) {
                ([only], _) | ([], [only]) => *only,
                ([], []) if all_digits => {
                    return Err(format!(
                        "not a device index ({} device(s) enumerated) and no device name contains it",
                        cands.len()
                    ));
                }
                ([], []) => return Err("no device name contains it".to_owned()),
                ([], many) | (many, _) => {
                    let list: Vec<String> = many.iter().map(usize::to_string).collect();
                    return Err(format!(
                        "ambiguous, matches devices {}; select by index instead",
                        list.join(", ")
                    ));
                }
            }
        }
    };
    match cands[idx].compute_queue {
        Ok(_) => Ok(idx),
        Err(why) => Err(format!(
            "device {idx} \"{}\" is not usable: {why}",
            cands[idx].name
        )),
    }
}

/// One line per candidate: `  <index>: <name> (<type>, <MiB> MiB
/// device-local[, pci <address>])`, with the reason appended for
/// ineligible devices.
fn describe(cands: &[Candidate]) -> String {
    if cands.is_empty() {
        return "  (none)".to_owned();
    }
    let mut out = String::new();
    for (i, c) in cands.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        // Writing into a `String` cannot fail.
        let _ = write!(
            out,
            "  {i}: {} ({:?}, {} MiB device-local",
            c.name,
            c.device_type,
            c.device_local_bytes >> 20
        );
        if let Some(pci) = c.pci {
            let _ = write!(out, ", pci {pci}");
        }
        out.push(')');
        if let Err(why) = c.compute_queue {
            let _ = write!(out, ", unusable: {why}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn cand(name: &str, device_type: vk::PhysicalDeviceType) -> Candidate {
        Candidate {
            name: name.to_owned(),
            device_type,
            compute_queue: Ok(0),
            device_local_bytes: 0,
            pci: None,
            uuid: [0; 16],
        }
    }

    fn gpu(name: &str, heap_gib: u64, bus: u32, uuid_byte: u8) -> Candidate {
        Candidate {
            device_local_bytes: heap_gib * GIB,
            pci: Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: 0,
            }),
            uuid: [uuid_byte; 16],
            ..cand(name, DISCRETE)
        }
    }

    /// `pick_auto` over `cands` and over `cands` reversed must choose the
    /// same device; returns its name.
    fn auto_name_order_independent(cands: &[Candidate]) -> String {
        let forward = &cands[pick(cands, None).unwrap()].name;
        let reversed: Vec<Candidate> = cands.iter().rev().cloned().collect();
        let backward = &reversed[pick(&reversed, None).unwrap()].name;
        assert_eq!(forward, backward, "choice depends on enumeration order");
        forward.clone()
    }

    const DISCRETE: vk::PhysicalDeviceType = vk::PhysicalDeviceType::DISCRETE_GPU;
    const INTEGRATED: vk::PhysicalDeviceType = vk::PhysicalDeviceType::INTEGRATED_GPU;
    const CPU: vk::PhysicalDeviceType = vk::PhysicalDeviceType::CPU;

    fn sel(s: &str) -> DeviceSelector {
        s.parse().expect("valid selector")
    }

    fn selection_error(cands: &[Candidate], s: &str) -> String {
        match pick(cands, Some(&sel(s))) {
            Err(BackendError::DeviceSelection(msg)) => msg,
            other => panic!("expected DeviceSelection error, got {other:?}"),
        }
    }

    fn host() -> Vec<Candidate> {
        vec![
            cand("llvmpipe (LLVM 21.1.8, 256 bits)", CPU),
            cand("AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)", INTEGRATED),
            cand("NVIDIA GeForce RTX 5070", DISCRETE),
        ]
    }

    #[test]
    fn parse_trims_and_rejects_empty() {
        assert_eq!(sel(" 0 "), DeviceSelector("0".to_owned()));
        assert!(matches!(
            "  ".parse::<DeviceSelector>(),
            Err(BackendError::DeviceSelection(_))
        ));
    }

    #[test]
    fn out_of_range_digits_match_as_a_name() {
        assert_eq!(pick(&host(), Some(&sel("5070"))).unwrap(), 2);
        let msg = selection_error(&host(), "99999999999999999999999");
        assert!(msg.contains("not a device index"), "{msg}");
    }

    #[test]
    fn in_range_digits_are_an_index_even_if_a_name_contains_them() {
        // Device 2's name contains "0", but 0 is a valid position.
        assert_eq!(pick(&host(), Some(&sel("0"))).unwrap(), 0);
    }

    #[test]
    fn auto_prefers_discrete_over_enumeration_order() {
        assert_eq!(pick(&host(), None).unwrap(), 2);
    }

    #[test]
    fn auto_skips_ineligible_devices() {
        let mut cands = host();
        cands[2].compute_queue = Err(Ineligible::ApiBelow13);
        assert_eq!(pick(&cands, None).unwrap(), 1);
    }

    #[test]
    fn auto_with_no_eligible_device_is_not_a_selection_error() {
        let mut cands = host();
        for c in &mut cands {
            c.compute_queue = Err(Ineligible::NoComputeQueue);
        }
        assert!(matches!(pick(&cands, None), Err(BackendError::Other(_))));
    }

    #[test]
    fn auto_prefers_the_larger_heap_among_discrete_gpus() {
        let cands = vec![
            gpu("AMD Radeon RX 7600", 8, 0x03, 1),
            gpu("NVIDIA GeForce RTX 5070", 12, 0x01, 2),
        ];
        assert_eq!(
            auto_name_order_independent(&cands),
            "NVIDIA GeForce RTX 5070"
        );
        let cands = vec![
            gpu("NVIDIA GeForce RTX 5070", 12, 0x01, 2),
            gpu("AMD Radeon RX 7900 XTX", 24, 0x03, 1),
        ];
        assert_eq!(
            auto_name_order_independent(&cands),
            "AMD Radeon RX 7900 XTX"
        );
    }

    #[test]
    fn auto_breaks_equal_heaps_by_pci_bus_order() {
        let cands = vec![
            gpu("card on bus 0x41", 12, 0x41, 1),
            gpu("card on bus 0x01", 12, 0x01, 2),
        ];
        assert_eq!(auto_name_order_independent(&cands), "card on bus 0x01");
    }

    #[test]
    fn auto_prefers_a_known_pci_address_then_uuid_order() {
        let mut unknown = gpu("no pci info", 12, 0, 0);
        unknown.pci = None;
        let cands = vec![unknown.clone(), gpu("pci info", 12, 0x81, 9)];
        assert_eq!(auto_name_order_independent(&cands), "pci info");

        let mut a = unknown.clone();
        a.name = "uuid 07".to_owned();
        a.uuid = [7; 16];
        let mut b = unknown;
        b.name = "uuid 03".to_owned();
        b.uuid = [3; 16];
        assert_eq!(auto_name_order_independent(&[a, b]), "uuid 03");
    }

    #[test]
    fn heap_size_does_not_override_device_type() {
        let mut igpu = gpu("integrated", 32, 0x72, 1);
        igpu.device_type = INTEGRATED;
        let cands = vec![igpu, gpu("discrete", 4, 0x01, 2)];
        assert_eq!(auto_name_order_independent(&cands), "discrete");
    }

    #[test]
    fn device_list_shows_heap_and_pci_address() {
        let cands = vec![gpu("NVIDIA GeForce RTX 5070", 12, 0x01, 2)];
        let msg = selection_error(&cands, "nosuch");
        assert!(
            msg.contains("  0: NVIDIA GeForce RTX 5070 (DISCRETE_GPU, 12288 MiB device-local, pci 0000:01:00.0)"),
            "{msg}"
        );
    }

    #[test]
    fn index_selects_regardless_of_rank() {
        assert_eq!(pick(&host(), Some(&sel("0"))).unwrap(), 0);
        assert_eq!(pick(&host(), Some(&sel("1"))).unwrap(), 1);
    }

    #[test]
    fn index_out_of_range_lists_devices() {
        let msg = selection_error(&host(), "3");
        assert!(msg.contains("selector \"3\""), "{msg}");
        assert!(
            msg.contains("not a device index (3 device(s) enumerated)"),
            "{msg}"
        );
        assert!(
            msg.contains("  2: NVIDIA GeForce RTX 5070 (DISCRETE_GPU, 0 MiB device-local)"),
            "{msg}"
        );
        assert!(msg.contains("  0: llvmpipe"), "{msg}");
    }

    #[test]
    fn substring_match_is_case_insensitive() {
        assert_eq!(pick(&host(), Some(&sel("5070"))).unwrap(), 2);
        assert_eq!(pick(&host(), Some(&sel("radv"))).unwrap(), 1);
        assert_eq!(pick(&host(), Some(&sel("LLVMPIPE"))).unwrap(), 0);
    }

    #[test]
    fn unmatched_substring_lists_devices() {
        let msg = selection_error(&host(), "nosuch");
        assert!(msg.contains("selector \"nosuch\""), "{msg}");
        assert!(msg.contains("no device name contains it"), "{msg}");
        assert!(msg.contains("  1: AMD Radeon Graphics"), "{msg}");
    }

    #[test]
    fn ambiguous_substring_is_an_error_naming_the_matches() {
        let cands = vec![
            cand("NVIDIA GeForce RTX 5070", DISCRETE),
            cand("AMD Radeon RX 7900 XTX", DISCRETE),
            cand("NVIDIA GeForce RTX 4090", DISCRETE),
        ];
        let msg = selection_error(&cands, "nvidia");
        assert!(msg.contains("ambiguous, matches devices 0, 2"), "{msg}");
    }

    #[test]
    fn identical_names_are_ambiguous_even_on_exact_match() {
        let cands = vec![
            cand("NVIDIA GeForce RTX 5070", DISCRETE),
            cand("NVIDIA GeForce RTX 5070", DISCRETE),
        ];
        let msg = selection_error(&cands, "nvidia geforce rtx 5070");
        assert!(msg.contains("matches devices 0, 1"), "{msg}");
    }

    #[test]
    fn exact_name_wins_over_longer_names_containing_it() {
        let cands = vec![
            cand("Radeon RX 7900 XTX", DISCRETE),
            cand("Radeon RX 7900", DISCRETE),
        ];
        assert_eq!(pick(&cands, Some(&sel("radeon rx 7900"))).unwrap(), 1);
    }

    #[test]
    fn selecting_an_ineligible_device_is_an_error() {
        let mut cands = host();
        cands[0].compute_queue = Err(Ineligible::ApiBelow13);
        let msg = selection_error(&cands, "llvmpipe");
        assert!(msg.contains("device 0"), "{msg}");
        assert!(msg.contains("not usable: Vulkan API below 1.3"), "{msg}");
        assert!(msg.contains(", unusable: Vulkan API below 1.3"), "{msg}");
    }
}
