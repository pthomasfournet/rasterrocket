//! Device selectors, and the rules by which one names a device among the
//! devices a backend enumerates.

use std::fmt;
use std::str::FromStr;

use crate::backend::{BackendError, SelectionFailure};

/// Explicit choice of device.
///
/// Resolution against the enumerated devices: a selector of ASCII digits
/// below the device count is a position in the backend's enumeration
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
            return Err(BackendError::DeviceSelection {
                failure: SelectionFailure::Invalid,
                message: "device selector is empty; expected an index or a device-name substring"
                    .to_owned(),
            });
        }
        Ok(Self(s.to_owned()))
    }
}

impl DeviceSelector {
    /// The selector text, trimmed.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// The index of the device among the enumerated `names` that `selector`
/// names, by the rules of [`DeviceSelector`], or the kind of reason it
/// names none and the reason.  Whether the device can host the backend is
/// not considered.
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "d3d12", windows), test)),
    expect(dead_code, reason = "only the Vulkan and D3D12 backends consume it")
)]
pub(crate) fn match_selector(
    names: &[&str],
    selector: &DeviceSelector,
) -> Result<usize, (SelectionFailure, String)> {
    let raw = selector.0.as_str();
    let all_digits = raw.bytes().all(|b| b.is_ascii_digit());
    if all_digits && let Some(i) = raw.parse::<usize>().ok().filter(|&i| i < names.len()) {
        return Ok(i);
    }
    let needle = raw.to_lowercase();
    let lowered: Vec<String> = names.iter().map(|n| n.to_lowercase()).collect();
    let exact: Vec<usize> = (0..names.len()).filter(|&i| lowered[i] == needle).collect();
    let partial: Vec<usize> = (0..names.len())
        .filter(|&i| lowered[i].contains(&needle))
        .collect();
    match (exact.as_slice(), partial.as_slice()) {
        ([only], _) | ([], [only]) => Ok(*only),
        ([], []) if all_digits => Err((
            SelectionFailure::NoMatch,
            format!(
                "not a device index ({} device(s) enumerated) and no device name contains it",
                names.len()
            ),
        )),
        ([], []) => Err((
            SelectionFailure::NoMatch,
            "no device name contains it".to_owned(),
        )),
        ([], many) | (many, _) => {
            let list: Vec<String> = many.iter().map(usize::to_string).collect();
            Err((
                SelectionFailure::Ambiguous,
                format!(
                    "ambiguous, matches devices {}; select by index instead",
                    list.join(", ")
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: [&str; 3] = [
        "llvmpipe (LLVM 21.1.8, 256 bits)",
        "AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO)",
        "NVIDIA GeForce RTX 5070",
    ];

    fn sel(s: &str) -> DeviceSelector {
        s.parse().expect("valid selector")
    }

    fn matched(names: &[&str], s: &str) -> usize {
        match_selector(names, &sel(s)).expect("a match")
    }

    fn failure(names: &[&str], s: &str) -> (SelectionFailure, String) {
        match_selector(names, &sel(s)).expect_err("no single match")
    }

    #[test]
    fn parse_trims_and_rejects_empty() {
        assert_eq!(sel(" 0 "), DeviceSelector("0".to_owned()));
        assert!(matches!(
            "  ".parse::<DeviceSelector>(),
            Err(BackendError::DeviceSelection { .. })
        ));
    }

    #[test]
    fn out_of_range_digits_match_as_a_name() {
        assert_eq!(matched(&HOST, "5070"), 2);
        let (_, msg) = failure(&HOST, "99999999999999999999999");
        assert!(msg.contains("not a device index"), "{msg}");
    }

    #[test]
    fn in_range_digits_are_an_index_even_if_a_name_contains_them() {
        // Device 2's name contains "0", but 0 is a valid position.
        assert_eq!(matched(&HOST, "0"), 0);
    }

    #[test]
    fn substring_match_is_case_insensitive() {
        assert_eq!(matched(&HOST, "5070"), 2);
        assert_eq!(matched(&HOST, "radv"), 1);
        assert_eq!(matched(&HOST, "LLVMPIPE"), 0);
    }

    #[test]
    fn ambiguous_substring_is_an_error_naming_the_matches() {
        let names = [
            "NVIDIA GeForce RTX 5070",
            "AMD Radeon RX 7900 XTX",
            "NVIDIA GeForce RTX 4090",
        ];
        let (_, msg) = failure(&names, "nvidia");
        assert!(msg.contains("ambiguous, matches devices 0, 2"), "{msg}");
    }

    #[test]
    fn identical_names_are_ambiguous_even_on_exact_match() {
        let names = ["NVIDIA GeForce RTX 5070", "NVIDIA GeForce RTX 5070"];
        let (_, msg) = failure(&names, "nvidia geforce rtx 5070");
        assert!(msg.contains("matches devices 0, 1"), "{msg}");
    }

    #[test]
    fn exact_name_wins_over_longer_names_containing_it() {
        let names = ["Radeon RX 7900 XTX", "Radeon RX 7900"];
        assert_eq!(matched(&names, "radeon rx 7900"), 1);
    }
}
