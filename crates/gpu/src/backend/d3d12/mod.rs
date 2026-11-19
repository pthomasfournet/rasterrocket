//! Direct3D 12 compute backend.
//!
//! Adapter selection (`select`) is plain data and builds on every host;
//! everything that calls Direct3D 12 or DXGI builds on Windows only.
//!
//! Module split (all private; named here for orientation only):
//! - `device` — DXGI factory, adapter enumeration, device, compute queue,
//!   debug layer.
//! - `diag` — the loaded WARP module and the CPU's vector instruction set.
//! - `error` — `windows::core::Error → BackendError` adaptor.
//! - `select` — adapter eligibility, ranking and selector resolution.

#[cfg(windows)]
mod device;
#[cfg(windows)]
mod diag;
#[cfg(windows)]
mod error;
#[cfg_attr(
    not(any(windows, test)),
    expect(dead_code, reason = "only the Windows host code consumes it")
)]
mod select;

#[cfg(windows)]
pub use host::D3d12Backend;

#[cfg(windows)]
mod host {
    use super::device::{self, D3d12Device};
    use crate::backend::Result;
    use crate::backend::select::DeviceSelector;

    /// Direct3D 12 compute backend on one DXGI adapter.
    ///
    /// Adapters are named by the [`DeviceSelector`] rules over DXGI's
    /// high-performance enumeration order, except that the selector `warp`
    /// names the WARP software adapter and no other selector names a
    /// software adapter.  `PDF_RASTER_D3D12_DEBUG=1` enables the D3D12
    /// debug layer before the first device of the process is created.
    pub struct D3d12Backend {
        device: D3d12Device,
    }

    impl D3d12Backend {
        /// Open the adapter `selector` names, or without one the first
        /// usable hardware adapter with dedicated video memory, else the
        /// first usable hardware adapter.
        ///
        /// # Errors
        ///
        /// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
        /// when the selector names no usable adapter;
        /// [`BackendError::Other`](crate::backend::BackendError::Other) when
        /// no hardware adapter is usable or a Direct3D 12 call fails.
        pub fn open(selector: Option<&DeviceSelector>) -> Result<Self> {
            device::open_adapter(selector).map(|device| Self { device })
        }

        /// Open the WARP software adapter.
        ///
        /// # Errors
        ///
        /// As [`Self::open`] with the selector `warp`.
        pub fn warp() -> Result<Self> {
            Self::open(Some(&"warp".parse()?))
        }

        /// The indices of every usable hardware adapter: those with
        /// dedicated video memory first, each group in enumeration order.
        ///
        /// # Errors
        ///
        /// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
        /// when there is none.
        pub fn usable_gpus() -> Result<Vec<usize>> {
            device::usable_gpus()
        }

        /// The adapter's index in the enumeration selectors resolve against.
        #[must_use]
        pub const fn device_index(&self) -> usize {
            self.device.index
        }

        /// The adapter's name (`DXGI_ADAPTER_DESC1::Description`).
        #[must_use]
        pub fn device_name(&self) -> &str {
            &self.device.name
        }

        /// Whether the adapter is WARP.
        #[must_use]
        pub const fn is_warp(&self) -> bool {
            self.device.warp
        }

        /// Whether kernels may use 64-bit integers (`Int64ShaderOps`).
        #[must_use]
        pub const fn has_int64(&self) -> bool {
            self.device.int64
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::diag::{cpu_tier, warp_module};
        use super::*;

        #[test]
        fn warp_opens_and_logs_its_module() {
            let backend = D3d12Backend::warp().expect("the WARP adapter opens");
            assert!(backend.is_warp(), "{}", backend.device_name());
            let (path, version) = warp_module().expect("d3d10warp.dll is loaded");
            let file = path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or_default();
            assert!(
                file.eq_ignore_ascii_case("d3d10warp.dll"),
                "{}",
                path.display()
            );
            eprintln!(
                "WARP {version} from {} on a CPU with {}; adapter {} \"{}\", int64 {}",
                path.display(),
                cpu_tier(),
                backend.device_index(),
                backend.device_name(),
                backend.has_int64()
            );
        }

        #[test]
        fn hardware_adapters_open_by_index_and_automatically() {
            let gpus = match D3d12Backend::usable_gpus() {
                Ok(gpus) => gpus,
                Err(e) => {
                    eprintln!("skipping: {e}");
                    return;
                }
            };
            let auto = D3d12Backend::open(None).expect("the automatic choice opens");
            assert_eq!(auto.device_index(), gpus[0]);
            assert!(!auto.is_warp());
            for &i in &gpus {
                let sel: DeviceSelector = i.to_string().parse().expect("selector");
                let backend = D3d12Backend::open(Some(&sel)).expect("a usable adapter opens");
                assert_eq!(backend.device_index(), i);
                assert!(!backend.is_warp());
                eprintln!(
                    "adapter {i} \"{}\", int64 {}",
                    backend.device_name(),
                    backend.has_int64()
                );
            }
        }
    }
}
