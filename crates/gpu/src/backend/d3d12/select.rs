//! Adapter selection over plain per-adapter data.
//!
//! [`AdapterCandidate`] holds the properties of one enumerated DXGI adapter
//! that selection depends on.  The functions here choose among candidates
//! without any driver call, so every rule is testable on any host.  A
//! selector names an adapter by the rules of [`match_selector`], except
//! that the selector `warp` names the WARP adapter and no other selector
//! does.
//!
//! The candidate list is in DXGI enumeration order, with the WARP adapter,
//! when present, appended last: the last candidate is the WARP adapter
//! exactly when it is a software adapter.

use std::fmt::{self, Write as _};

use crate::backend::select::{DeviceSelector, match_selector};
use crate::backend::{BackendError, SelectionFailure};

/// The lowest shader model the kernels' DXIL targets, as
/// `D3D_SHADER_MODEL` encodes it (`0x60` is 6.0).
pub(super) const MIN_SHADER_MODEL: u32 = 0x60;

/// One enumerated adapter, reduced to the data selection reads.
#[derive(Debug, Clone)]
pub(super) struct AdapterCandidate {
    /// `DXGI_ADAPTER_DESC1::Description`.
    pub(super) name: String,
    /// Whether `DXGI_ADAPTER_FLAG_SOFTWARE` is set.
    pub(super) software: bool,
    /// `DXGI_ADAPTER_DESC1::AdapterLuid`, high part in the upper 32 bits.
    pub(super) luid: u64,
    /// `DXGI_ADAPTER_DESC1::DedicatedVideoMemory`; non-zero on a discrete
    /// adapter.
    pub(super) dedicated_bytes: u64,
    /// The highest shader model the device supports (`0x60` is 6.0), or
    /// `None` when no device at feature level `11_0` could be created.
    pub(super) shader_model: Option<u32>,
    /// Whether `D3D12_FEATURE_DATA_D3D12_OPTIONS1::Int64ShaderOps` is set.
    pub(super) int64: bool,
}

/// Why an adapter cannot host the compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unusable {
    /// No device at feature level `11_0` could be created on it.
    NoDevice,
    /// Its highest shader model is below [`MIN_SHADER_MODEL`].
    ShaderModel(u32),
}

impl fmt::Display for Unusable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => f.write_str("no D3D12 device at feature level 11_0"),
            Self::ShaderModel(sm) => write!(
                f,
                "shader model {} below the floor {}",
                ShaderModel(*sm),
                ShaderModel(MIN_SHADER_MODEL)
            ),
        }
    }
}

/// A `D3D_SHADER_MODEL` value shown as `major.minor`.
struct ShaderModel(u32);

impl fmt::Display for ShaderModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.0 >> 4, self.0 & 0xf)
    }
}

impl AdapterCandidate {
    /// The reason the adapter cannot host the backend, if any.
    const fn unusable(&self) -> Option<Unusable> {
        match self.shader_model {
            None => Some(Unusable::NoDevice),
            Some(sm) if sm < MIN_SHADER_MODEL => Some(Unusable::ShaderModel(sm)),
            Some(_) => None,
        }
    }

    /// Whether automatic choice and `all` may take the adapter: a usable
    /// hardware adapter.
    const fn is_usable_gpu(&self) -> bool {
        !self.software && self.unusable().is_none()
    }
}

/// The index of the WARP adapter: the last candidate, when it is a
/// software adapter.
fn warp_index(cands: &[AdapterCandidate]) -> Option<usize> {
    cands.len().checked_sub(1).filter(|&i| cands[i].software)
}

/// Choose an adapter from `cands`.
///
/// With a selector, the adapter it names is returned or a
/// [`BackendError::DeviceSelection`] listing every enumerated adapter.
/// Without one, the first usable hardware adapter with dedicated video
/// memory is returned, else the first usable hardware adapter.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] when the selector matches no adapter,
/// matches several, or names an unusable or software adapter (other than
/// WARP through `warp`); [`BackendError::Other`] when no selector is given
/// and no hardware adapter is usable.
pub(super) fn pick(
    cands: &[AdapterCandidate],
    selector: Option<&DeviceSelector>,
) -> Result<usize, BackendError> {
    selector.map_or_else(
        || {
            ranked_gpus(cands).first().copied().ok_or_else(|| {
                BackendError::msg(format!(
                    "no D3D12 hardware adapter with shader model {} or above was found; \
                     enumerated:\n{}",
                    ShaderModel(MIN_SHADER_MODEL),
                    describe(cands)
                ))
            })
        },
        |sel| {
            resolve_selector(cands, sel).map_err(|(failure, reason)| {
                BackendError::DeviceSelection {
                    failure,
                    message: format!(
                        "D3D12 adapter selector {sel}: {reason}\navailable D3D12 adapters:\n{}",
                        describe(cands)
                    ),
                }
            })
        },
    )
}

/// The adapters `selectors` name, in their order, each resolved as
/// [`pick`] resolves one.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] when `selectors` is empty, when one
/// of them does not resolve, or when two resolve to the same adapter; the
/// message lists every enumerated adapter.
pub(super) fn pick_many(
    cands: &[AdapterCandidate],
    selectors: &[DeviceSelector],
) -> Result<Vec<usize>, BackendError> {
    crate::backend::select::resolve_each(
        selectors,
        "D3D12 adapter",
        |sel| resolve_selector(cands, sel),
        |i| cands[i].name.as_str(),
    )
    .map_err(|(failure, reason)| BackendError::DeviceSelection {
        failure,
        message: format!("{reason}\navailable D3D12 adapters:\n{}", describe(cands)),
    })
}

/// Every usable hardware adapter, those with dedicated video memory first,
/// each group in enumeration order.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`] when there is none; the message lists
/// every enumerated adapter.
pub(super) fn usable_gpus(cands: &[AdapterCandidate]) -> Result<Vec<usize>, BackendError> {
    let gpus = ranked_gpus(cands);
    if gpus.is_empty() {
        return Err(BackendError::DeviceSelection {
            failure: SelectionFailure::NoneUsable,
            message: format!(
                "no usable D3D12 hardware adapter (shader model {} or above, not a software \
                 adapter) was found; enumerated:\n{}",
                ShaderModel(MIN_SHADER_MODEL),
                describe(cands)
            ),
        });
    }
    Ok(gpus)
}

/// The usable hardware adapters, those with dedicated video memory first,
/// each group in enumeration order.
fn ranked_gpus(cands: &[AdapterCandidate]) -> Vec<usize> {
    let mut gpus: Vec<usize> = (0..cands.len())
        .filter(|&i| cands[i].is_usable_gpu())
        .collect();
    // A stable sort keeps enumeration order within each group.
    gpus.sort_by_key(|&i| cands[i].dedicated_bytes == 0);
    gpus
}

/// Resolve `sel` to one usable candidate index, or the kind of reason it
/// does not and the reason.  `warp` names the WARP adapter; WARP's index
/// is refused; every other selector is matched against the candidates
/// before it.
fn resolve_selector(
    cands: &[AdapterCandidate],
    sel: &DeviceSelector,
) -> Result<usize, (SelectionFailure, String)> {
    let warp = warp_index(cands);
    let idx = if sel.as_str().eq_ignore_ascii_case("warp") {
        warp.ok_or_else(|| {
            (
                SelectionFailure::NoMatch,
                "no WARP adapter was enumerated".to_owned(),
            )
        })?
    } else {
        if let Some(w) = warp
            && sel.as_str().bytes().all(|b| b.is_ascii_digit())
            && sel.as_str().parse::<usize>().ok() == Some(w)
        {
            return Err((
                SelectionFailure::Ineligible,
                format!(
                    "device {w} \"{}\" is the WARP adapter; select it with \"warp\"",
                    cands[w].name
                ),
            ));
        }
        let names: Vec<&str> = cands[..warp.unwrap_or(cands.len())]
            .iter()
            .map(|c| c.name.as_str())
            .collect();
        let idx = match_selector(&names, sel)?;
        if cands[idx].software {
            return Err((
                SelectionFailure::Ineligible,
                format!(
                    "device {idx} \"{}\" is a software adapter; select WARP with \"warp\"",
                    cands[idx].name
                ),
            ));
        }
        idx
    };
    cands[idx].unusable().map_or(Ok(idx), |why| {
        Err((
            SelectionFailure::Ineligible,
            format!("device {idx} \"{}\" is not usable: {why}", cands[idx].name),
        ))
    })
}

/// One line per candidate: `  <index>: <name> (<kind>, <MiB> MiB
/// dedicated, shader model <x.y>[, int64], luid <luid>)`, with the reason
/// appended for unusable adapters.
fn describe(cands: &[AdapterCandidate]) -> String {
    if cands.is_empty() {
        return "  (none)".to_owned();
    }
    let warp = warp_index(cands);
    let mut out = String::new();
    for (i, c) in cands.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let kind = match (c.software, c.dedicated_bytes > 0) {
            (true, _) if warp == Some(i) => "WARP, select with \"warp\"",
            (true, _) => "software",
            (false, true) => "discrete",
            (false, false) => "integrated",
        };
        // Writing into a `String` cannot fail.
        let _ = write!(
            out,
            "  {i}: {} ({kind}, {} MiB dedicated",
            c.name,
            c.dedicated_bytes >> 20
        );
        if let Some(sm) = c.shader_model {
            let _ = write!(out, ", shader model {}", ShaderModel(sm));
        }
        if c.int64 {
            out.push_str(", int64");
        }
        let _ = write!(out, ", luid {:#x})", c.luid);
        if let Some(why) = c.unusable() {
            let _ = write!(out, ", unusable: {why}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn cand(
        name: &str,
        software: bool,
        dedicated_gib: u64,
        sm: u32,
        int64: bool,
    ) -> AdapterCandidate {
        AdapterCandidate {
            name: name.to_owned(),
            software,
            luid: 0,
            dedicated_bytes: dedicated_gib * GIB,
            shader_model: Some(sm),
            int64,
        }
    }

    fn host() -> Vec<AdapterCandidate> {
        vec![
            cand("NVIDIA GeForce RTX 2080 SUPER", false, 8, 0x68, true),
            cand("Intel(R) UHD Graphics 630", false, 0, 0x66, false),
            cand("Microsoft Basic Render Driver", true, 0, 0x62, true),
            cand("WARP", true, 0, 0x62, true),
        ]
    }

    fn sel(s: &str) -> DeviceSelector {
        s.parse().expect("valid selector")
    }

    fn picked(cands: &[AdapterCandidate], s: &str) -> usize {
        pick(cands, Some(&sel(s))).expect("a match")
    }

    fn selection_failure(cands: &[AdapterCandidate], s: &str) -> (SelectionFailure, String) {
        match pick(cands, Some(&sel(s))) {
            Err(BackendError::DeviceSelection { failure, message }) => (failure, message),
            other => panic!("expected DeviceSelection error, got {other:?}"),
        }
    }

    #[test]
    fn dxgi_selection_rules() {
        let cands = host();
        assert_eq!(pick(&cands, None).unwrap(), 0);
        assert_eq!(usable_gpus(&cands).unwrap(), [0, 1]);
        assert_eq!(picked(&cands, "warp"), 3);
        assert_eq!(picked(&cands, "WARP"), 3);
        assert_eq!(picked(&cands, "intel"), 1);
        assert_eq!(picked(&cands, "1"), 1);

        let (failure, msg) = selection_failure(&cands, "basic");
        assert_eq!(failure, SelectionFailure::Ineligible, "{msg}");
        assert!(msg.contains("software adapter"), "{msg}");
        assert!(
            msg.contains("device 2 \"Microsoft Basic Render Driver\""),
            "{msg}"
        );

        let twins = vec![
            cand("NVIDIA GeForce RTX 2080 SUPER", false, 8, 0x68, true),
            cand("NVIDIA GeForce RTX 2080 SUPER", false, 8, 0x68, true),
        ];
        let (failure, msg) = selection_failure(&twins, "2080");
        assert_eq!(failure, SelectionFailure::Ambiguous, "{msg}");
        assert!(msg.contains("matches devices 0, 1"), "{msg}");
        assert!(
            msg.contains("  0: NVIDIA GeForce RTX 2080 SUPER (discrete"),
            "{msg}"
        );
        assert!(
            msg.contains("  1: NVIDIA GeForce RTX 2080 SUPER (discrete"),
            "{msg}"
        );

        let mut old = host();
        old[0].shader_model = Some(0x51);
        assert_eq!(usable_gpus(&old).unwrap(), [1]);
        assert_eq!(pick(&old, None).unwrap(), 1);
        let (failure, msg) = selection_failure(&old, "nvidia");
        assert_eq!(failure, SelectionFailure::Ineligible, "{msg}");
        assert!(
            msg.contains("device 0 \"NVIDIA GeForce RTX 2080 SUPER\""),
            "{msg}"
        );
        assert!(
            msg.contains("shader model 5.1 below the floor 6.0"),
            "{msg}"
        );
    }

    #[test]
    fn dxgi_selection_warp_is_reachable_only_through_warp() {
        // DXGI names the WARP adapter as it names the basic render driver.
        let mut cands = host();
        cands[3].name = "Microsoft Basic Render Driver".to_owned();
        assert_eq!(picked(&cands, "warp"), 3);
        let (failure, msg) = selection_failure(&cands, "basic");
        assert_eq!(failure, SelectionFailure::Ineligible, "{msg}");
        // Index 3 is past the adapters a selector other than `warp` sees,
        // so the digits match as a name, and no name contains "4".
        let (failure, msg) = selection_failure(&cands, "4");
        assert_eq!(failure, SelectionFailure::NoMatch, "{msg}");
        assert!(msg.contains("3 device(s) enumerated"), "{msg}");
        // WARP's own index is refused, not matched as a name ("630").
        let (failure, msg3) = selection_failure(&cands, "3");
        assert_eq!(failure, SelectionFailure::Ineligible, "{msg3}");
        assert!(
            msg3.contains(
                "device 3 \"Microsoft Basic Render Driver\" is the WARP adapter; \
                 select it with \"warp\""
            ),
            "{msg3}"
        );
        let (failure, _) = selection_failure(&host(), "3");
        assert_eq!(failure, SelectionFailure::Ineligible);
        assert!(
            msg.contains("3: Microsoft Basic Render Driver (WARP"),
            "{msg}"
        );

        // Without a trailing software adapter there is no WARP adapter.
        let hardware = &host()[..2];
        let (failure, msg) = selection_failure(hardware, "warp");
        assert_eq!(failure, SelectionFailure::NoMatch, "{msg}");
    }

    #[test]
    fn dxgi_selection_prefers_dedicated_memory_then_enumeration_order() {
        let cands = vec![
            cand("integrated", false, 0, 0x66, false),
            cand("discrete a", false, 4, 0x66, false),
            cand("discrete b", false, 16, 0x66, false),
        ];
        assert_eq!(pick(&cands, None).unwrap(), 1);
        assert_eq!(usable_gpus(&cands).unwrap(), [1, 2, 0]);
    }

    #[test]
    fn dxgi_selection_without_usable_hardware() {
        let mut cands = host();
        cands[0].shader_model = None;
        cands[1].shader_model = Some(0x51);
        assert!(matches!(pick(&cands, None), Err(BackendError::Other(_))));
        match usable_gpus(&cands) {
            Err(BackendError::DeviceSelection {
                failure: SelectionFailure::NoneUsable,
                message,
            }) => {
                assert!(
                    message.contains("no usable D3D12 hardware adapter"),
                    "{message}"
                );
                assert!(
                    message.contains("unusable: no D3D12 device at feature level 11_0"),
                    "{message}"
                );
            }
            other => panic!("expected DeviceSelection error, got {other:?}"),
        }
        // WARP is still selectable by name.
        assert_eq!(picked(&cands, "warp"), 3);
    }

    #[test]
    fn dxgi_selection_warp_below_the_floor_is_an_error() {
        let mut cands = host();
        cands[3].shader_model = Some(0x51);
        let (failure, msg) = selection_failure(&cands, "warp");
        assert_eq!(failure, SelectionFailure::Ineligible, "{msg}");
        assert!(
            msg.contains("shader model 5.1 below the floor 6.0"),
            "{msg}"
        );
    }

    #[test]
    fn dxgi_selection_of_several_adapters() {
        let cands = host();
        let many = |list: &[&str]| {
            let sels: Vec<DeviceSelector> = list.iter().map(|s| sel(s)).collect();
            pick_many(&cands, &sels)
        };
        assert_eq!(many(&["intel", "0", "warp"]).unwrap(), [1, 0, 3]);
        let failure = |list: &[&str]| match many(list) {
            Err(BackendError::DeviceSelection { failure, message }) => (failure, message),
            other => panic!("expected DeviceSelection error, got {other:?}"),
        };
        let (kind, msg) = failure(&[]);
        assert_eq!(kind, SelectionFailure::Invalid, "{msg}");
        assert!(msg.contains("the D3D12 adapter list is empty"), "{msg}");
        let (kind, msg) = failure(&["1", "intel"]);
        assert_eq!(kind, SelectionFailure::Invalid, "{msg}");
        assert!(
            msg.contains(
                "D3D12 adapter selectors \"1\" and \"intel\" both name device 1 \
                 \"Intel(R) UHD Graphics 630\""
            ),
            "{msg}"
        );
        assert!(msg.contains("available D3D12 adapters:"), "{msg}");
        let (kind, msg) = failure(&["0", "basic"]);
        assert_eq!(kind, SelectionFailure::Ineligible, "{msg}");
        assert!(msg.contains("D3D12 adapter selector \"basic\""), "{msg}");
    }
}
