//! DRM render-node discovery and VA-API JPEG capability probing.
//!
//! Render nodes are `/dev/dri/renderD<N>`; probing walks them in ascending
//! `N` and takes the first whose driver initialises and advertises
//! `VAProfileJPEGBaseline` with `VAEntrypointVLD`.  The ordering and
//! first-match logic are plain functions over names and a probe callback,
//! testable without a VA-API device.

use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_int;
use std::path::Path;

use super::VapiDisplay;
use super::error::{Result, VapiError, check};
use super::ffi::{
    VA_ENTRYPOINT_VLD, VA_PROFILE_JPEG_BASELINE, vaMaxNumEntrypoints, vaQueryConfigEntrypoints,
    vaQueryVendorString,
};

/// Directory holding the DRM device nodes.
const DRI_DIR: &str = "/dev/dri";

/// One DRM render node, with the identity sysfs reports for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderNode {
    /// Device path, e.g. `/dev/dri/renderD128`.
    pub path: String,
    /// PCI address of the owning device (e.g. `0000:01:00.0`), when sysfs
    /// resolves it.
    pub pci: Option<String>,
    /// Kernel driver bound to the owning device (e.g. `amdgpu`), when
    /// sysfs resolves it.
    pub driver: Option<String>,
}

impl fmt::Display for RenderNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.path)?;
        match (&self.pci, &self.driver) {
            (Some(pci), Some(drv)) => write!(f, " (pci {pci}, {drv})"),
            (Some(pci), None) => write!(f, " (pci {pci})"),
            (None, Some(drv)) => write!(f, " ({drv})"),
            (None, None) => Ok(()),
        }
    }
}

/// The `N` of a `renderD<N>` file name; `None` for any other name.
fn render_node_number(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("renderD")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The render-node names among `names`, in ascending node number.
fn order_render_nodes<I, S>(names: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut nodes: Vec<(u32, String)> = names
        .into_iter()
        .map(Into::into)
        .filter_map(|n| render_node_number(&n).map(|num| (num, n)))
        .collect();
    nodes.sort_unstable();
    nodes.into_iter().map(|(_, n)| n).collect()
}

/// Last path component of the symlink at `link`, if it resolves.
fn link_basename(link: &Path) -> Option<String> {
    let target = std::fs::read_link(link).ok()?;
    target.file_name()?.to_str().map(str::to_owned)
}

/// The render nodes present on this host, in ascending node number.
///
/// Empty when `/dev/dri` does not exist or cannot be read.
#[must_use]
pub fn render_nodes() -> Vec<RenderNode> {
    let Ok(entries) = std::fs::read_dir(DRI_DIR) else {
        return Vec::new();
    };
    let names = entries
        .filter_map(std::result::Result::ok)
        .filter_map(|e| e.file_name().into_string().ok());
    order_render_nodes(names)
        .into_iter()
        .map(|name| {
            let sys = Path::new("/sys/class/drm").join(&name).join("device");
            RenderNode {
                path: format!("{DRI_DIR}/{name}"),
                pci: link_basename(&sys),
                driver: link_basename(&sys.join("driver")),
            }
        })
        .collect()
}

/// One-line-per-node description of `nodes` for error messages.
#[must_use]
pub fn describe_render_nodes(nodes: &[RenderNode]) -> String {
    if nodes.is_empty() {
        return format!("  (no render nodes under {DRI_DIR})");
    }
    nodes
        .iter()
        .map(|n| format!("  {n}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Open `path` and check that its driver can decode baseline JPEG.
///
/// Returns the driver's vendor string on success.
///
/// # Errors
///
/// [`VapiError::DrmOpen`] / [`VapiError::NullDisplay`] / [`VapiError::VaStatus`]
/// when the node cannot be opened or initialised or the profile query
/// fails; [`VapiError::NoJpegDecode`] when the driver initialises but does
/// not list `VAEntrypointVLD` for `VAProfileJPEGBaseline`.
pub fn supports_jpeg_decode(path: &str) -> Result<String> {
    let dpy = VapiDisplay::open(path)?;

    // SAFETY: dpy.dpy is an initialised display.
    let max = unsafe { vaMaxNumEntrypoints(dpy.dpy) };
    let mut entrypoints: Vec<c_int> = vec![0; usize::try_from(max.max(1)).unwrap_or(1)];
    let mut count: c_int = 0;
    // SAFETY: `entrypoints` holds `vaMaxNumEntrypoints` elements, the
    // documented capacity requirement; `count` is a valid out-pointer.
    check(
        unsafe {
            vaQueryConfigEntrypoints(
                dpy.dpy,
                VA_PROFILE_JPEG_BASELINE,
                entrypoints.as_mut_ptr(),
                &raw mut count,
            )
        },
        "vaQueryConfigEntrypoints(JPEGBaseline)",
    )?;
    let reported = usize::try_from(count).unwrap_or(0).min(entrypoints.len());
    if !entrypoints[..reported].contains(&VA_ENTRYPOINT_VLD) {
        return Err(VapiError::NoJpegDecode);
    }

    // SAFETY: dpy.dpy is initialised; the returned string is owned by the
    // display and outlives this read.
    let vendor = unsafe {
        let ptr = vaQueryVendorString(dpy.dpy);
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Ok(vendor)
}

/// The first of `nodes` for which `probe` succeeds, with its result.
///
/// # Errors
///
/// Every `(node, reason)` pair when no node succeeds; empty when `nodes`
/// is empty.
fn first_supported<T, E: fmt::Display>(
    nodes: &[RenderNode],
    mut probe: impl FnMut(&str) -> std::result::Result<T, E>,
) -> std::result::Result<(RenderNode, T), Vec<(RenderNode, String)>> {
    let mut failures = Vec::new();
    for node in nodes {
        match probe(&node.path) {
            Ok(v) => return Ok((node.clone(), v)),
            Err(e) => failures.push((node.clone(), e.to_string())),
        }
    }
    Err(failures)
}

/// A render node chosen by [`find_jpeg_render_node`].
#[derive(Debug, Clone)]
pub struct ProbedNode {
    /// The node.
    pub node: RenderNode,
    /// The driver's vendor string.
    pub vendor: String,
}

/// Probe the host's render nodes in ascending number and return the first
/// that supports VA-API baseline JPEG decode.
///
/// # Errors
///
/// A message naming every render node and why it was rejected (or that
/// none exist) when no node qualifies.
pub fn find_jpeg_render_node() -> std::result::Result<ProbedNode, String> {
    find_in(&render_nodes(), supports_jpeg_decode)
}

fn find_in<E: fmt::Display>(
    nodes: &[RenderNode],
    probe: impl FnMut(&str) -> std::result::Result<String, E>,
) -> std::result::Result<ProbedNode, String> {
    match first_supported(nodes, probe) {
        Ok((node, vendor)) => Ok(ProbedNode { node, vendor }),
        Err(failures) if failures.is_empty() => Err(format!(
            "no DRM render nodes under {DRI_DIR}; VA-API needs one"
        )),
        Err(failures) => {
            let lines: Vec<String> = failures
                .iter()
                .map(|(node, why)| format!("  {node}: {why}"))
                .collect();
            Err(format!(
                "no DRM render node supports VA-API baseline JPEG decode; probed:\n{}",
                lines.join("\n")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: u32) -> RenderNode {
        RenderNode {
            path: format!("/dev/dri/renderD{n}"),
            pci: Some(format!("0000:{n:02x}:00.0")),
            driver: Some("drv".to_owned()),
        }
    }

    #[test]
    fn render_node_numbers_parse_only_render_nodes() {
        assert_eq!(render_node_number("renderD128"), Some(128));
        assert_eq!(render_node_number("renderD"), None);
        assert_eq!(render_node_number("renderD12a"), None);
        assert_eq!(render_node_number("card0"), None);
        assert_eq!(render_node_number("by-path"), None);
    }

    #[test]
    fn render_nodes_sort_numerically_not_lexically() {
        let names = [
            "renderD1000",
            "card1",
            "renderD129",
            "by-path",
            "renderD128",
            "card2",
        ];
        assert_eq!(
            order_render_nodes(names),
            vec!["renderD128", "renderD129", "renderD1000"]
        );
    }

    #[test]
    fn probe_takes_the_first_supporting_node() {
        let nodes = [node(128), node(129), node(130)];
        let mut probed = Vec::new();
        let found = find_in(&nodes, |p: &str| {
            probed.push(p.to_owned());
            if p.ends_with("129") || p.ends_with("130") {
                Ok("vendor".to_owned())
            } else {
                Err("vaInitialize failed")
            }
        })
        .unwrap();
        assert_eq!(found.node.path, "/dev/dri/renderD129");
        assert_eq!(found.vendor, "vendor");
        assert_eq!(probed, ["/dev/dri/renderD128", "/dev/dri/renderD129"]);
    }

    #[test]
    fn probe_failure_names_every_node_and_reason() {
        let nodes = [node(128), node(129)];
        let err = find_in(&nodes, |p: &str| {
            Err::<String, _>(format!("nope on {}", &p[p.len() - 3..]))
        })
        .unwrap_err();
        assert!(
            err.contains("no DRM render node supports VA-API baseline JPEG decode"),
            "{err}"
        );
        assert!(
            err.contains("  /dev/dri/renderD128 (pci 0000:80:00.0, drv): nope on 128"),
            "{err}"
        );
        assert!(
            err.contains("  /dev/dri/renderD129 (pci 0000:81:00.0, drv): nope on 129"),
            "{err}"
        );
    }

    #[test]
    fn probe_with_no_nodes_says_so() {
        let err = find_in(&[], |_: &str| Ok::<_, String>(String::new())).unwrap_err();
        assert!(err.contains("no DRM render nodes"), "{err}");
    }

    #[test]
    fn render_node_display_omits_unknown_identity() {
        let mut n = node(128);
        n.pci = None;
        n.driver = None;
        assert_eq!(n.to_string(), "/dev/dri/renderD128");
        assert_eq!(
            describe_render_nodes(&[]),
            "  (no render nodes under /dev/dri)"
        );
    }
}
