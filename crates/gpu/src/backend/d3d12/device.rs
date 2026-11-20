//! DXGI factory and adapter enumeration, device and compute queue creation,
//! the capabilities selection reads, and the debug layer.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D_SHADER_MODEL, D3D_SHADER_MODEL_5_1, D3D_SHADER_MODEL_6_0, D3D_SHADER_MODEL_6_1,
    D3D_SHADER_MODEL_6_2, D3D_SHADER_MODEL_6_3, D3D_SHADER_MODEL_6_4, D3D_SHADER_MODEL_6_5,
    D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_8, D3D_SHADER_MODEL_6_9,
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_QUEUE_DESC, D3D12_FEATURE,
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_SHADER_MODEL, D3D12CreateDevice,
    D3D12GetDebugInterface, ID3D12CommandQueue, ID3D12Debug, ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
    DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1, IDXGIFactory6,
};

use super::diag::{cpu_tier, warp_module};
use super::error::{describe, device_err, hr_err};
use super::select::{self, AdapterCandidate};
use super::validation::Validation;
use crate::backend::Result;
use crate::backend::select::DeviceSelector;

/// Shader models from the highest this SDK names down to the lowest a
/// feature level `11_0` device supports.  `CheckFeatureSupport` rejects a
/// model the runtime does not know and lowers one it knows to the highest
/// it supports, so the first accepted query gives the device's model.
const SHADER_MODELS: [D3D_SHADER_MODEL; 11] = [
    D3D_SHADER_MODEL_6_9,
    D3D_SHADER_MODEL_6_8,
    D3D_SHADER_MODEL_6_7,
    D3D_SHADER_MODEL_6_6,
    D3D_SHADER_MODEL_6_5,
    D3D_SHADER_MODEL_6_4,
    D3D_SHADER_MODEL_6_3,
    D3D_SHADER_MODEL_6_2,
    D3D_SHADER_MODEL_6_1,
    D3D_SHADER_MODEL_6_0,
    D3D_SHADER_MODEL_5_1,
];

/// An open adapter: its device and one compute queue.
pub(crate) struct D3d12Device {
    /// The device, at feature level `11_0` or above.
    pub(super) device: ID3D12Device,
    /// The compute queue.
    pub(super) queue: ID3D12CommandQueue,
    /// The adapter the device was created on.
    pub(super) adapter: IDXGIAdapter1,
    /// The adapter's index in the enumeration selection resolves against.
    pub(super) index: usize,
    /// `DXGI_ADAPTER_DESC1::Description`.
    pub(super) name: String,
    /// Whether the adapter is WARP.
    pub(super) warp: bool,
    /// Whether kernels may use 64-bit integers (`Int64ShaderOps`).
    pub(super) int64: bool,
    /// The debug layer's validation messages.
    pub(super) validation: Validation,
}

/// Open the adapter `selector` names, or the automatically chosen
/// hardware adapter without one, with a device and a compute queue.
///
/// # Errors
///
/// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
/// when the selector names no usable adapter; [`BackendError::Other`](crate::backend::BackendError::Other)
/// when no hardware adapter is usable for the automatic choice or a call
/// fails.
pub(crate) fn open_adapter(selector: Option<&DeviceSelector>) -> Result<D3d12Device> {
    enable_debug_layer();
    let mut set = AdapterSet::enumerate()?;
    let index = match selector {
        None => {
            set.probe_first_gpu();
            set.decide(Vec::new(), |cands| select::pick(cands, None))?
        }
        Some(sel) => {
            let named = select::named(&set.cands, sel).into_iter().collect();
            set.decide(named, |cands| select::pick(cands, Some(sel)))?
        }
    };
    let device = match set.devices[index].take() {
        Some(device) => device,
        None => create_device(&set.adapters[index]).map_err(hr_err("D3D12CreateDevice"))?,
    };
    let cand = &set.cands[index];
    let desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
        ..Default::default()
    };
    // SAFETY: `desc` is a valid queue description that outlives the call.
    let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&raw const desc) }
        .map_err(device_err(&device, "CreateCommandQueue"))?;
    let warp = cand.software;
    if warp {
        match warp_module() {
            Some((path, version)) => log::info!(
                "d3d12: WARP {version} from {} on a CPU with {}",
                path.display(),
                cpu_tier()
            ),
            None => log::info!(
                "d3d12: WARP opened, but d3d10warp.dll is not loaded; CPU with {}",
                cpu_tier()
            ),
        }
    }
    log::debug!(
        "d3d12: opened adapter {index} \"{}\" (shader model {:#x}, int64 {})",
        cand.name,
        cand.shader_model.unwrap_or_default(),
        cand.int64
    );
    let validation = Validation::new(&device, debug_layer_enabled());
    Ok(D3d12Device {
        device,
        queue,
        adapter: set.adapters[index].clone(),
        index,
        name: cand.name.clone(),
        warp,
        int64: cand.int64,
        validation,
    })
}

/// The indices of every usable hardware adapter, by the order of
/// [`select::usable_gpus`].
///
/// # Errors
///
/// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
/// when there is none; [`BackendError::Other`](crate::backend::BackendError::Other)
/// when enumeration fails.
pub(crate) fn usable_gpus() -> Result<Vec<usize>> {
    enable_debug_layer();
    let mut set = AdapterSet::enumerate()?;
    let hardware = select::hardware_order(&set.cands);
    set.decide(hardware, select::usable_gpus)
}

/// Check that `selector` names one usable adapter, without creating a
/// device beyond those created to read capabilities.
///
/// # Errors
///
/// As [`open_adapter`] with `selector`, before its queue is created.
pub(crate) fn check_selector(selector: &DeviceSelector) -> Result<()> {
    enable_debug_layer();
    let mut set = AdapterSet::enumerate()?;
    let named = select::named(&set.cands, selector).into_iter().collect();
    set.decide(named, |cands| select::pick(cands, Some(selector)))
        .map(drop)
}

/// The index of the adapter `selector` names by [`select::named`], read
/// from the adapters' descriptions without creating a device; `None` when
/// enumeration fails or the selector fails on names alone.
pub(crate) fn named_index(selector: &DeviceSelector) -> Option<usize> {
    let set = AdapterSet::enumerate().ok()?;
    select::named(&set.cands, selector)
}

/// The indices of the adapters `selectors` name, in their order, by
/// [`select::pick_many`].
///
/// # Errors
///
/// [`BackendError::DeviceSelection`](crate::backend::BackendError::DeviceSelection)
/// when `selectors` is empty, one does not resolve, or two name the same
/// adapter; [`BackendError::Other`](crate::backend::BackendError::Other)
/// when enumeration fails.
pub(crate) fn resolve_devices(selectors: &[DeviceSelector]) -> Result<Vec<usize>> {
    enable_debug_layer();
    let mut set = AdapterSet::enumerate()?;
    let named = selectors
        .iter()
        .filter_map(|sel| select::named(&set.cands, sel))
        .collect();
    set.decide(named, |cands| select::pick_many(cands, selectors))
}

/// Whether [`enable_debug_layer`] enabled the layer.
static DEBUG_LAYER_ON: AtomicBool = AtomicBool::new(false);

/// Whether the debug layer is enabled in this process.
pub(crate) fn debug_layer_enabled() -> bool {
    DEBUG_LAYER_ON.load(Ordering::Acquire)
}

/// Enable the D3D12 debug layer once per process when
/// `PDF_RASTER_D3D12_DEBUG` is `1`, and log whether it was.  Every device
/// this module creates is created after the first call, as the layer
/// requires.
fn enable_debug_layer() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if !std::env::var("PDF_RASTER_D3D12_DEBUG").is_ok_and(|v| v == "1") {
            return;
        }
        let mut debug: Option<ID3D12Debug> = None;
        // SAFETY: `debug` is a valid out-pointer for the interface.
        match unsafe { D3D12GetDebugInterface(&raw mut debug) } {
            Ok(()) => {
                if let Some(debug) = debug {
                    // SAFETY: `debug` is a live interface; no device exists yet.
                    unsafe { debug.EnableDebugLayer() };
                    DEBUG_LAYER_ON.store(true, Ordering::Release);
                    log::info!("d3d12: debug layer enabled (PDF_RASTER_D3D12_DEBUG=1)");
                }
            }
            Err(e) => log::info!(
                "d3d12: debug layer not enabled (PDF_RASTER_D3D12_DEBUG=1): {}",
                describe("D3D12GetDebugInterface", e.code())
            ),
        }
    });
}

/// The enumerated adapters with their candidate data, each adapter's
/// capabilities read at most once, on first need.
///
/// Reading the capabilities creates a device on the adapter.  Selection succeeds on the capabilities
/// of the adapters it takes alone, so only those are read first; a
/// failing selection is repeated with every adapter read, so its error
/// describes each one.
struct AdapterSet {
    /// Every adapter DXGI lists, by high-performance preference, then the
    /// WARP adapter.  DXGI lists software adapters after hardware ones, so
    /// the last entry is the WARP adapter exactly when it is a software
    /// adapter.
    adapters: Vec<IDXGIAdapter1>,
    /// The candidate data of each adapter; `shader_model` is `None` and
    /// `int64` false until it is probed.
    cands: Vec<AdapterCandidate>,
    /// Whether each adapter has been probed.
    probed: Vec<bool>,
    /// The device each probe created, kept for [`open_adapter`] to use.
    devices: Vec<Option<ID3D12Device>>,
}

impl AdapterSet {
    /// List the adapters and read their descriptions; no device is created.
    fn enumerate() -> Result<Self> {
        // SAFETY: no flags; the factory interface is returned by value.
        let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(hr_err("CreateDXGIFactory2"))?;
        let mut adapters: Vec<IDXGIAdapter1> = Vec::new();
        for i in 0.. {
            // SAFETY: `factory` is live; an index past the last adapter
            // returns `DXGI_ERROR_NOT_FOUND`.
            match unsafe {
                factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                )
            } {
                Ok(adapter) => adapters.push(adapter),
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(hr_err("EnumAdapterByGpuPreference")(e)),
            }
        }
        // SAFETY: `factory` is live.
        match unsafe { factory.EnumWarpAdapter::<IDXGIAdapter1>() } {
            Ok(adapter) => adapters.push(adapter),
            Err(e) => log::debug!("d3d12: {}", describe("EnumWarpAdapter", e.code())),
        }
        let cands = adapters.iter().map(candidate).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            probed: vec![false; adapters.len()],
            devices: vec![None; adapters.len()],
            adapters,
            cands,
        })
    }

    /// Read the shader model and `Int64ShaderOps` of adapter `i` from a
    /// device created on it, unless already read.
    fn probe(&mut self, i: usize) {
        if std::mem::replace(&mut self.probed[i], true) {
            return;
        }
        let cand = &mut self.cands[i];
        match create_device(&self.adapters[i]) {
            Ok(device) => {
                cand.shader_model = Some(shader_model(&device));
                cand.int64 = int64_ops(&device);
                self.devices[i] = Some(device);
            }
            Err(e) => log::debug!(
                "d3d12: adapter \"{}\": {}",
                cand.name,
                describe("D3D12CreateDevice", e.code())
            ),
        }
    }

    /// Probe the hardware adapters in [`select::hardware_order`] up to the
    /// first usable one, the automatic choice.
    fn probe_first_gpu(&mut self) {
        for i in select::hardware_order(&self.cands) {
            self.probe(i);
            if self.cands[i].is_usable_gpu() {
                return;
            }
        }
    }

    /// `choose` over the candidates once `needed` are probed; when it
    /// fails, again once every adapter is probed.
    fn decide<T>(
        &mut self,
        needed: Vec<usize>,
        choose: impl Fn(&[AdapterCandidate]) -> Result<T>,
    ) -> Result<T> {
        for i in needed {
            self.probe(i);
        }
        choose(&self.cands).or_else(|_| {
            for i in 0..self.adapters.len() {
                self.probe(i);
            }
            choose(&self.cands)
        })
    }
}

/// The candidate data of `adapter` from its description, its capabilities
/// not yet read.
fn candidate(adapter: &IDXGIAdapter1) -> Result<AdapterCandidate> {
    // SAFETY: `adapter` is live.
    let desc = unsafe { adapter.GetDesc1() }.map_err(hr_err("GetDesc1"))?;
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    let name = String::from_utf16_lossy(&desc.Description[..len]);
    let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0.cast_unsigned() != 0;
    let luid = (u64::from(desc.AdapterLuid.HighPart.cast_unsigned()) << 32)
        | u64::from(desc.AdapterLuid.LowPart);
    Ok(AdapterCandidate {
        name,
        software,
        luid,
        dedicated_bytes: desc.DedicatedVideoMemory as u64,
        shader_model: None,
        int64: false,
    })
}

/// A device on `adapter` at feature level `11_0`.
fn create_device(adapter: &IDXGIAdapter1) -> windows::core::Result<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: `adapter` is live and `device` is a valid out-pointer.
    unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &raw mut device) }?;
    device.ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_POINTER))
}

/// Query `feature` of `device` into `data`.
fn check_feature<T>(
    device: &ID3D12Device,
    feature: D3D12_FEATURE,
    data: &mut T,
) -> windows::core::Result<()> {
    let size = u32::try_from(size_of::<T>()).unwrap_or(u32::MAX);
    // SAFETY: `data` is a writable `T` of `size` bytes, the structure
    // `feature` names.
    unsafe { device.CheckFeatureSupport(feature, (&raw mut *data).cast(), size) }
}

/// The highest shader model `device` supports, as `D3D_SHADER_MODEL`
/// encodes it.
fn shader_model(device: &ID3D12Device) -> u32 {
    for model in SHADER_MODELS {
        let mut data = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: model,
        };
        if check_feature(device, D3D12_FEATURE_SHADER_MODEL, &mut data).is_ok() {
            return data.HighestShaderModel.0.cast_unsigned();
        }
    }
    D3D_SHADER_MODEL_5_1.0.cast_unsigned()
}

/// Whether `device` reports `Int64ShaderOps`.
fn int64_ops(device: &ID3D12Device) -> bool {
    let mut data = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    check_feature(device, D3D12_FEATURE_D3D12_OPTIONS1, &mut data).is_ok()
        && data.Int64ShaderOps.as_bool()
}
