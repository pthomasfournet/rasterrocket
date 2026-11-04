//! Vulkan instance, physical-device selection, and logical-device creation.
//!
//! A physical device is eligible when it:
//! - Supports compute on at least one queue family.
//! - Reports `VK_API_VERSION_1_3` or higher (so we can rely on Vulkan 1.3
//!   core for `synchronization2`, `timelineSemaphore`, 8-bit storage,
//!   and buffer-device-address without enabling them as discrete extensions).
//! - Supports every one of those features (`vkGetPhysicalDeviceFeatures2`).
//!
//! The device is either named by a [`DeviceSelector`] or chosen by the
//! ranking in [`select`](super::select).
//!
//! No surface, no graphics queue — this is a headless compute backend.

use ash::ext::{memory_budget, pci_bus_info};
use ash::khr::{portability_enumeration, portability_subset};
use ash::vk;
use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::backend::{BackendError, Result};

use super::error::vk_err;
use super::select::{self, Candidate, DeviceSelector, Ineligible, PciAddress};

/// Owned Vulkan handles.
///
/// Field declaration order matters: Rust drops fields in declaration
/// order, so the `Drop` impl below tears them down explicitly to enforce
/// `device → instance → entry`.
pub(super) struct DeviceCtx {
    pub(super) device: ash::Device,
    /// Index into the device's queue-family list — used at command pool create time.
    pub(super) compute_queue_family: u32,
    /// The compute queue handle, wrapped in a `Mutex` to enforce the
    /// Vulkan "Threading Behavior" external-synchronization requirement
    /// (`vkQueueSubmit` / `vkQueueWaitIdle` / present-class calls on
    /// the same queue must not run concurrently from multiple threads).
    /// The mutex *is* the access path — there is no way to reach the
    /// raw `vk::Queue` without holding the lock.  Both submitters (the
    /// per-page recorder and the transfer context) go through
    /// [`Self::submit`] or [`Self::submit_and_wait_idle`].
    compute_queue: Mutex<vk::Queue>,
    /// Successful `vkQueueSubmit` calls on `compute_queue`.
    submits: AtomicU64,
    /// Cached physical-device handle for memory-property + budget queries.
    pub(super) phys: vk::PhysicalDevice,
    /// Cached memory properties (memoryTypeBits → `MemoryType` lookup).
    pub(super) mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Per-axis maximum workgroup count for `vkCmdDispatch`.  Vulkan's
    /// guaranteed minimum is `(65535, 65535, 65535)`; checked at dispatch
    /// time so the kernel-launch helpers can fail with a useful message
    /// instead of letting the driver return `ERROR_DEVICE_LOST`.
    pub(super) max_workgroup_count: [u32; 3],
    /// Bit `i` set for each memory heap `i` device buffers are placed in
    /// ([`buffer_heaps`]); the heaps the VRAM budget covers.
    pub(super) buffer_heaps: u32,
    /// Total size of the heaps in `buffer_heaps`, fixed across the device's
    /// lifetime.
    pub(super) vram_total: u64,
    /// Whether `VK_EXT_memory_budget` was successfully enabled (gates
    /// runtime budget queries).
    pub(super) has_memory_budget: bool,
    /// Whether the device is an integrated GPU: device memory is host
    /// memory, so device buffers are allocated host-visible and mapped.
    pub(super) integrated: bool,
    /// Whether `shaderInt64` is enabled: kernels declaring the `Int64`
    /// capability are built only when it is.
    pub(super) has_int64: bool,
    /// Borrowed PFNs onto the live instance for runtime queries (e.g. memory budget).
    pub(super) instance: ash::Instance,
    /// Vulkan loader; must outlive instance.
    #[expect(
        dead_code,
        reason = "kept alive for the lifetime of `instance`/`device`; ash::Entry holds the dlopen'd libvulkan handle"
    )]
    pub(super) entry: ash::Entry,
}

impl Drop for DeviceCtx {
    fn drop(&mut self) {
        // Safety: every owned handle was created via the matching create_*
        // call above; matching destroy_* calls here, in reverse order, are
        // sound because we have exclusive ownership and the recorder calls
        // `vkDeviceWaitIdle` before the backend drops.
        let _lifecycle = instance_lifecycle();
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Serialises instance creation and destruction across the process.
///
/// The Vulkan loader updates process-global state (its driver and implicit
/// layer lists, and the libraries behind them) during `vkCreateInstance` and
/// `vkDestroyInstance` under its own lock, while
/// `vkEnumerateInstanceExtensionProperties` reads that state without it:
/// creating backends on several threads at once calls through a freed
/// layer pointer.  Holding this lock from loading the loader to the end of
/// device creation, and around teardown, keeps each thread's view whole.
fn instance_lifecycle() -> MutexGuard<'static, ()> {
    static LIFECYCLE: Mutex<()> = Mutex::new(());
    LIFECYCLE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Initialise the Vulkan loader, instance, and device.
///
/// Returns an `Arc<DeviceCtx>` so the slab allocator, pipeline cache, and
/// per-page recorder can hold their own clones without lifetime-threading
/// the `ash::Device` reference everywhere.  `selector`, when given,
/// names the physical device; otherwise the automatic ranking applies.
pub(super) fn init(selector: Option<&DeviceSelector>) -> Result<Arc<DeviceCtx>> {
    let Picked {
        _lifecycle,
        entry,
        instance,
        phys,
        chosen,
        compute_queue_family,
    } = open_and_pick(selector)?;
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

    // Pull the dispatch group-count limit once.  Spec guarantees ≥ 65535
    // per axis; some software ICDs (lavapipe) report u32::MAX which is
    // fine — we only check against this to fail loudly, not to clamp.
    let dev_props = unsafe { instance.get_physical_device_properties(phys) };
    let max_workgroup_count = dev_props.limits.max_compute_work_group_count;

    log_chosen_device(&dev_props, &chosen, selector);

    let (device, compute_queue, setup) = destroy_instance_on_err(
        &instance,
        create_device(&instance, phys, compute_queue_family, chosen.int64),
    )?;

    let integrated = dev_props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
    let buffer_heaps = buffer_heaps(&mem_props, integrated);
    let vram_total = heap_bytes(&mem_props, buffer_heaps);

    Ok(Arc::new(DeviceCtx {
        device,
        compute_queue_family,
        compute_queue: Mutex::new(compute_queue),
        submits: AtomicU64::new(0),
        phys,
        mem_props,
        max_workgroup_count,
        buffer_heaps,
        vram_total,
        has_memory_budget: setup.has_memory_budget(),
        integrated,
        has_int64: setup.int64,
        instance,
        entry,
    }))
}

impl DeviceCtx {
    /// Submit `submit` to the compute queue with no fence.
    ///
    /// The queue lock is held for the `vkQueueSubmit` call only.  `what`
    /// names the submitter in the error.
    ///
    /// # Safety
    /// Every handle `submit` references is live and its command buffers are
    /// in the Executable state.
    pub(super) unsafe fn submit(
        &self,
        submit: &vk::SubmitInfo<'_>,
        what: &'static str,
    ) -> Result<()> {
        let queue = self.compute_queue.lock().expect("compute_queue poisoned");
        // Safety: the queue is held exclusively; the caller guarantees `submit`.
        unsafe {
            self.device
                .queue_submit(*queue, std::slice::from_ref(submit), vk::Fence::null())
        }
        .map_err(vk_err(what))?;
        let _ = self.submits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// [`Self::submit`], then `vkQueueWaitIdle` under the same lock hold, so
    /// that the queue is idle when the call returns.
    ///
    /// # Safety
    /// As [`Self::submit`].
    pub(super) unsafe fn submit_and_wait_idle(
        &self,
        submit: &vk::SubmitInfo<'_>,
        what: &'static str,
    ) -> Result<()> {
        let queue = self.compute_queue.lock().expect("compute_queue poisoned");
        // Safety: the queue is held exclusively; the caller guarantees `submit`.
        unsafe {
            self.device
                .queue_submit(*queue, std::slice::from_ref(submit), vk::Fence::null())
                .map_err(vk_err(what))?;
            let _ = self.submits.fetch_add(1, Ordering::Relaxed);
            self.device
                .queue_wait_idle(*queue)
                .map_err(vk_err("vkQueueWaitIdle"))
        }
    }

    /// Successful submissions to the compute queue so far.
    pub(super) fn submits(&self) -> u64 {
        self.submits.load(Ordering::Relaxed)
    }
}

/// Resolve `selector` against the enumerated physical devices without
/// creating a logical device: an instance is created, the devices are
/// enumerated and ranked as [`init`] does, and the instance is destroyed.
///
/// # Errors
/// [`BackendError::DeviceSelection`] when `selector` matches no device,
/// several, or one that cannot host the backend; the loader's or the
/// instance's error when Vulkan itself is unavailable.
pub(super) fn check_selector(selector: &DeviceSelector) -> Result<()> {
    let picked = open_and_pick(Some(selector))?;
    // Safety: the instance has no child objects; physical-device handles
    // are not owned.
    unsafe { picked.instance.destroy_instance(None) };
    Ok(())
}

/// A loaded loader, an instance on it, and the physical device chosen on
/// that instance, with the lifecycle lock still held.
struct Picked {
    _lifecycle: MutexGuard<'static, ()>,
    entry: ash::Entry,
    instance: ash::Instance,
    phys: vk::PhysicalDevice,
    chosen: Candidate,
    compute_queue_family: u32,
}

/// Take the lifecycle lock, load the loader, create an instance, and pick
/// the device `selector` names (or the best-ranked one).  On failure no
/// instance is left behind.
fn open_and_pick(selector: Option<&DeviceSelector>) -> Result<Picked> {
    let lifecycle = instance_lifecycle();
    let entry = load_entry()?;
    let instance = create_instance(&entry)?;
    let (phys, chosen, compute_queue_family) = destroy_instance_on_err(
        &instance,
        pick_physical_device(&instance, selector).and_then(|(phys, chosen)| {
            let family = chosen
                .compute_queue
                .map_err(|why| BackendError::msg(format!("picked an unusable device: {why}")))?;
            Ok((phys, chosen, family))
        }),
    )?;
    Ok(Picked {
        _lifecycle: lifecycle,
        entry,
        instance,
        phys,
        chosen,
        compute_queue_family,
    })
}

/// Pass `result` through, destroying `instance` first when it is an error.
///
/// Only valid while `instance` has no live child objects: every caller is
/// in [`open_and_pick`] or [`init`] before a logical device exists, and
/// [`create_device`] creates none on failure.
fn destroy_instance_on_err<T>(instance: &ash::Instance, result: Result<T>) -> Result<T> {
    if result.is_err() {
        // Safety: per the precondition above, no child objects exist.
        unsafe { instance.destroy_instance(None) };
    }
    result
}

/// Load the Vulkan loader: the platform's default name first, then the
/// paths of [`loader_candidates`].  The error names every path tried and
/// why each failed to load.
fn load_entry() -> Result<ash::Entry> {
    // Safety: `Entry::load` and `Entry::load_from` `dlopen` the loader; the
    // returned `Entry` owns the library handle and is `Send + Sync` by ash
    // design.
    let default_error = match unsafe { ash::Entry::load() } {
        Ok(entry) => return Ok(entry),
        Err(e) => e,
    };
    let sdk = std::env::var_os("VULKAN_SDK").map(PathBuf::from);
    let candidates = loader_candidates(sdk.as_deref());
    let mut failures = Vec::with_capacity(candidates.len());
    for path in &candidates {
        match unsafe { ash::Entry::load_from(path) } {
            Ok(entry) => return Ok(entry),
            Err(e) => failures.push(format!("{}: {e}", path.display())),
        }
    }
    Err(BackendError::msg(if failures.is_empty() {
        format!("Vulkan loader failed: {default_error}")
    } else {
        format!(
            "Vulkan loader failed: {default_error}; also tried {}",
            failures.join("; ")
        )
    }))
}

/// `sdk` when it is an absolute path.  An empty or relative `$VULKAN_SDK`
/// is ignored with a warning: the loader path joined to it would resolve
/// against the working directory.
fn absolute_sdk(sdk: Option<&Path>) -> Option<&Path> {
    let sdk = sdk?;
    if sdk.is_absolute() {
        Some(sdk)
    } else {
        log::warn!(
            "vulkan: ignoring VULKAN_SDK={:?}: not an absolute path",
            sdk.display()
        );
        None
    }
}

/// Loader libraries to try when the default name is not found: on macOS,
/// the Vulkan SDK's (`$VULKAN_SDK`, when absolute) and Homebrew's, whose directories the
/// dynamic linker does not search by default; none elsewhere.
fn loader_candidates(vulkan_sdk: Option<&Path>) -> Vec<PathBuf> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let mut candidates: Vec<PathBuf> = absolute_sdk(vulkan_sdk)
        .map(|sdk| sdk.join("lib/libvulkan.1.dylib"))
        .into_iter()
        .collect();
    candidates.extend(
        [
            "/opt/homebrew/lib/libvulkan.1.dylib",
            "/usr/local/lib/libvulkan.1.dylib",
        ]
        .map(PathBuf::from),
    );
    candidates
}

/// The instance extensions to enable and the create flags, from the
/// extensions the loader offers (`available`): portability enumeration
/// when offered, so that the loader lists portability implementations
/// such as `MoltenVK`.
fn instance_setup(available: &[&CStr]) -> (Vec<&'static CStr>, vk::InstanceCreateFlags) {
    if available.contains(&portability_enumeration::NAME) {
        (
            vec![portability_enumeration::NAME],
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
        )
    } else {
        (Vec::new(), vk::InstanceCreateFlags::empty())
    }
}

/// The device extensions to enable and the optional features to use, from
/// what the device offers.
#[derive(Debug, PartialEq, Eq)]
struct DeviceSetup {
    extensions: Vec<&'static CStr>,
    /// `shaderInt64` is supported and enabled.
    int64: bool,
}

impl DeviceSetup {
    fn has_memory_budget(&self) -> bool {
        self.extensions.contains(&memory_budget::NAME)
    }
}

/// [`DeviceSetup`] for a device offering the extensions `available`, with
/// `int64` whether it supports `shaderInt64`: memory budget when offered,
/// the portability subset whenever advertised (its enabling is then
/// required), and `shaderInt64` when supported.
fn device_setup(available: &[&CStr], int64: bool) -> DeviceSetup {
    let extensions = [memory_budget::NAME, portability_subset::NAME]
        .into_iter()
        .filter(|name| available.contains(name))
        .collect();
    DeviceSetup { extensions, int64 }
}

/// The names of `properties`, NUL-terminated per the specification.
fn extension_names(properties: &[vk::ExtensionProperties]) -> Vec<&CStr> {
    properties
        .iter()
        .map(|ext| {
            // Safety: extension_name is a NUL-terminated C string per spec.
            unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
        })
        .collect()
}

fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
    let app_name = c"rasterrocket";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(vk::make_api_version(0, 0, 7, 0))
        .engine_name(app_name)
        .engine_version(vk::make_api_version(0, 0, 7, 0))
        .api_version(vk::API_VERSION_1_3);

    // Headless compute: no surface extensions; portability enumeration when
    // the loader offers it.
    // Safety: no layer name is passed.
    let offered = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(vk_err("vkEnumerateInstanceExtensionProperties"))?;
    let (extensions, flags) = instance_setup(&extension_names(&offered));
    let extension_ptrs: Vec<*const c_char> = extensions.iter().map(|e| e.as_ptr()).collect();
    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_ptrs)
        .flags(flags);

    // Safety: `app_info` and `create_info` outlive this call; the pointers
    // captured by the ash::vk structs stay valid through the call.
    unsafe { entry.create_instance(&create_info, None) }.map_err(vk_err("vkCreateInstance"))
}

/// Enumerate physical devices, reduce each to a [`Candidate`], and return
/// the handle and candidate of the one [`select::pick`] chooses.
fn pick_physical_device(
    instance: &ash::Instance,
    selector: Option<&DeviceSelector>,
) -> Result<(vk::PhysicalDevice, Candidate)> {
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_err("vkEnumeratePhysicalDevices"))?;
    if physicals.is_empty() {
        return Err(BackendError::msg("no Vulkan physical devices found"));
    }

    let candidates = physicals
        .iter()
        .map(|&p| describe_physical_device(instance, p))
        .collect::<Result<Vec<_>>>()?;
    if selector.is_none() {
        for (i, c) in candidates.iter().enumerate() {
            if let Err(why) = c.compute_queue {
                log::info!("vulkan: device {i} \"{}\" skipped: {why}", c.name);
            }
        }
    }
    let idx = select::pick(&candidates, selector)?;
    let chosen = candidates.into_iter().nth(idx).ok_or_else(|| {
        BackendError::msg("device selection returned an index past the enumerated devices")
    })?;
    Ok((physicals[idx], chosen))
}

/// Feature support of a device, as far as the backend reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SupportedFeatures {
    /// The first feature [`create_device`] enables unconditionally that the
    /// device does not support, by its Vulkan name.
    missing_required: Option<&'static str>,
    /// Whether `shaderInt64` is supported.
    int64: bool,
}

impl SupportedFeatures {
    /// `vkGetPhysicalDeviceFeatures2` with the 1.2 and 1.3 structures
    /// chained; valid for a device reporting API 1.3 or higher.
    fn query(instance: &ash::Instance, phys: vk::PhysicalDevice) -> Self {
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut v12)
            .push_next(&mut v13);
        // Safety: every chained structure is live for the call and core in
        // the device's API version.
        unsafe { instance.get_physical_device_features2(phys, &mut features2) };
        let int64 = features2.features.shader_int64 == vk::TRUE;
        Self {
            missing_required: missing_required(&v12, &v13),
            int64,
        }
    }
}

/// The Vulkan name of the first feature [`create_device`] enables
/// unconditionally that `v12` / `v13` do not report as supported.
fn missing_required(
    v12: &vk::PhysicalDeviceVulkan12Features<'_>,
    v13: &vk::PhysicalDeviceVulkan13Features<'_>,
) -> Option<&'static str> {
    [
        ("bufferDeviceAddress", v12.buffer_device_address),
        ("timelineSemaphore", v12.timeline_semaphore),
        ("storageBuffer8BitAccess", v12.storage_buffer8_bit_access),
        (
            "uniformAndStorageBuffer8BitAccess",
            v12.uniform_and_storage_buffer8_bit_access,
        ),
        ("shaderInt8", v12.shader_int8),
        ("synchronization2", v13.synchronization2),
    ]
    .into_iter()
    .find_map(|(name, supported)| (supported != vk::TRUE).then_some(name))
}

/// Query the properties [`select`] ranks and matches on.
fn describe_physical_device(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
) -> Result<Candidate> {
    let props = unsafe { instance.get_physical_device_properties(phys) };
    let name = device_name(&props);

    // Vulkan 1.3 core is required: synchronization2, timelineSemaphore,
    // BDA, and 8-bit storage are used as core features.  `api_version` is
    // the maximum version the driver supports, so one packed-integer
    // comparison suffices.
    let (features, compute_queue) = if props.api_version < vk::API_VERSION_1_3 {
        (None, Err(Ineligible::ApiBelow13))
    } else {
        let features = SupportedFeatures::query(instance, phys);
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
        let queue = match features.missing_required {
            Some(name) => Err(Ineligible::MissingFeature(name)),
            None => match queue_families
                .iter()
                .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                Some(idx) => Ok(u32::try_from(idx)
                    .map_err(|_| BackendError::msg("queue family index does not fit in u32"))?),
                None => Err(Ineligible::NoComputeQueue),
            },
        };
        (Some(features), queue)
    };

    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };
    let (pci, uuid) = query_identity(instance, phys, props.api_version)?;

    Ok(Candidate {
        name,
        device_type: props.device_type,
        compute_queue,
        int64: features.is_some_and(|f| f.int64),
        device_local_bytes: total_device_local_bytes(&mem_props),
        pci,
        uuid,
    })
}

/// PCI address (when `VK_EXT_pci_bus_info` is supported) and device UUID.
///
/// `vkGetPhysicalDeviceProperties2` with `VkPhysicalDeviceIDProperties` is
/// core in 1.1; a device reporting a lower API version yields no PCI
/// address and an all-zero UUID.
fn query_identity(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    api_version: u32,
) -> Result<(Option<PciAddress>, [u8; 16])> {
    if api_version < vk::API_VERSION_1_1 {
        return Ok((None, [0; 16]));
    }
    let exts = unsafe { instance.enumerate_device_extension_properties(phys) }
        .map_err(vk_err("vkEnumerateDeviceExtensionProperties"))?;
    let has_pci = extension_names(&exts).contains(&pci_bus_info::NAME);

    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    if has_pci {
        props2 = props2.push_next(&mut pci);
    }
    // Safety: every struct in the chain is live for the call and of a type
    // the device supports (ID properties are core 1.1; the PCI struct is
    // chained only when the extension is advertised).
    unsafe { instance.get_physical_device_properties2(phys, &mut props2) };

    let address = has_pci.then_some(PciAddress {
        domain: pci.pci_domain,
        bus: pci.pci_bus,
        device: pci.pci_device,
        function: pci.pci_function,
    });
    Ok((address, id.device_uuid))
}

/// `deviceName` as an owned string; a name without a NUL terminator inside
/// the fixed-size array is reported as invalid rather than read past.
fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    props.device_name_as_c_str().map_or_else(
        |_| "<invalid device name>".to_owned(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Create the logical device and its compute queue.  The required features
/// were checked against the device's support at selection
/// ([`missing_required`]); `int64` is whether
/// `shaderInt64` is supported.
fn create_device(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    queue_family: u32,
    int64: bool,
) -> Result<(ash::Device, vk::Queue, DeviceSetup)> {
    let available_exts = unsafe { instance.enumerate_device_extension_properties(phys) }
        .map_err(vk_err("vkEnumerateDeviceExtensionProperties"))?;
    let setup = device_setup(&extension_names(&available_exts), int64);
    let enabled_exts: Vec<*const c_char> = setup.extensions.iter().map(|e| e.as_ptr()).collect();

    let priorities = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&priorities)];

    // Required core features — chained via push_next so the driver sees a
    // single VkPhysicalDeviceFeatures2 → 1.2 → 1.3 list.
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .timeline_semaphore(true)
        .storage_buffer8_bit_access(true)
        .uniform_and_storage_buffer8_bit_access(true)
        // The kernels declare `OpCapability Int8` because the Slang
        // sources use `uint8_t` for the byte-typed StructuredBuffers.
        // Without `shader_int8`, vkCreateShaderModule fails with a
        // missing-capability error from the validation layer.
        .shader_int8(true);
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let core = vk::PhysicalDeviceFeatures::default().shader_int64(setup.int64);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(core)
        .push_next(&mut f12)
        .push_next(&mut f13);

    let device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&enabled_exts)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(phys, &device_info, None) }
        .map_err(vk_err("vkCreateDevice"))?;

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    Ok((device, queue, setup))
}

/// PCI vendor ID for NVIDIA, per the PCI-SIG registry.  Vulkan's
/// `driver_version` field is only interpretable through `VK_VERSION_*`
/// for vendors that follow the core spec's suggested encoding; NVIDIA
/// packs a proprietary 10/8/8/6-bit major/minor/patch/build layout
/// instead, and other vendors are not guaranteed to match either.
const VENDOR_ID_NVIDIA: u32 = 0x10DE;

/// Decode `driver_version` the way `vendor_id` is known to encode it,
/// falling back to the raw integer for every vendor whose encoding this
/// module doesn't special-case.
fn format_driver_version(vendor_id: u32, driver_version: u32) -> String {
    if vendor_id == VENDOR_ID_NVIDIA {
        // NVIDIA's Vulkan driver_version is not VK_VERSION_*-shaped: it's
        // major(10) | minor(8) | patch(8) | build(6), matching the
        // `driver-version-major.minor` shown by `nvidia-smi`.
        let major = driver_version >> 22;
        let minor = (driver_version >> 14) & 0xFF;
        let patch = (driver_version >> 6) & 0xFF;
        let build = driver_version & 0x3F;
        format!("{major}.{minor}.{patch}.{build}")
    } else {
        format!(
            "{}.{}.{} (raw {driver_version})",
            vk::api_version_major(driver_version),
            vk::api_version_minor(driver_version),
            vk::api_version_patch(driver_version)
        )
    }
}

/// Log the physical device chosen by [`pick_physical_device`] at
/// `debug` level, with how it was chosen.  Called once per backend
/// construction.
fn log_chosen_device(
    props: &vk::PhysicalDeviceProperties,
    chosen: &Candidate,
    selector: Option<&DeviceSelector>,
) {
    let name = device_name(props);
    let driver_version = format_driver_version(props.vendor_id, props.driver_version);
    let api_version = format!(
        "{}.{}.{}",
        vk::api_version_major(props.api_version),
        vk::api_version_minor(props.api_version),
        vk::api_version_patch(props.api_version)
    );
    let device_type = props.device_type;
    let vendor_id = props.vendor_id;
    let device_local_mib = chosen.device_local_bytes >> 20;
    let pci = chosen
        .pci
        .map_or_else(|| "unknown".to_owned(), |a| a.to_string());
    let chosen_by = selector.map_or_else(|| "automatic".to_owned(), |s| format!("selector {s}"));
    log::debug!(
        "vulkan: chosen device \"{name}\" type={device_type:?} vendor_id=0x{vendor_id:04X} driver={driver_version} api={api_version} device_local={device_local_mib}MiB pci={pci} ({chosen_by})"
    );
}

/// Sum of every `DEVICE_LOCAL` heap described by `props`.
fn total_device_local_bytes(props: &vk::PhysicalDeviceMemoryProperties) -> u64 {
    heap_bytes(props, buffer_heaps(props, false))
}

/// `props`' memory heaps and types as slices of their declared counts.
fn heaps_and_types(
    props: &vk::PhysicalDeviceMemoryProperties,
) -> (&[vk::MemoryHeap], &[vk::MemoryType]) {
    let heaps = props
        .memory_heaps
        .get(..props.memory_heap_count as usize)
        .unwrap_or(&props.memory_heaps);
    let types = props
        .memory_types
        .get(..props.memory_type_count as usize)
        .unwrap_or(&props.memory_types);
    (heaps, types)
}

/// Bit `i` set for each memory heap `i` a device buffer can be placed in:
/// every `DEVICE_LOCAL` heap and, on an integrated GPU, the heap of the
/// type a read-back buffer (`MemoryLocation::GpuToCpu`) is allocated from.
/// That type is the first `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` one,
/// else the first `HOST_VISIBLE | HOST_COHERENT` one: the allocator's
/// choice for a buffer that every memory type may back.
fn buffer_heaps(props: &vk::PhysicalDeviceMemoryProperties, integrated: bool) -> u32 {
    let (heaps, types) = heaps_and_types(props);
    let bit = |heap: u32| 1u32.checked_shl(heap).unwrap_or(0);
    let mut mask = heaps
        .iter()
        .zip(0u32..)
        .filter(|(h, _)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .fold(0, |m, (_, i)| m | bit(i));
    if integrated {
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let first = |wanted: vk::MemoryPropertyFlags| {
            types.iter().find(|t| t.property_flags.contains(wanted))
        };
        if let Some(t) =
            first(coherent | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|| first(coherent))
        {
            mask |= bit(t.heap_index);
        }
    }
    mask
}

/// Total size of the heaps of `props` whose bit is set in `mask`.
fn heap_bytes(props: &vk::PhysicalDeviceMemoryProperties, mask: u32) -> u64 {
    let (heaps, _) = heaps_and_types(props);
    heaps
        .iter()
        .zip(0u32..)
        .filter(|&(_, i)| mask & 1u32.checked_shl(i).unwrap_or(0) != 0)
        .fold(0u64, |total, (h, _)| total.saturating_add(h.size))
}

/// Query the live `VK_EXT_memory_budget` snapshot of the heaps in
/// `buffer_heaps` if the extension is on, otherwise return `(used, budget)
/// = (0, vram_total)` — a safe fallback that lets the caller derive a
/// usable budget from the heaps' total size.
#[expect(
    clippy::unnecessary_wraps,
    reason = "Result kept for forward-compat: a future variant calls vkGetPhysicalDeviceMemoryProperties2 indirectly which can fail on lost devices"
)]
pub(super) fn query_memory_budget(ctx: &DeviceCtx) -> Result<(u64, u64)> {
    if !ctx.has_memory_budget {
        return Ok((0, ctx.vram_total));
    }

    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        ctx.instance
            .get_physical_device_memory_properties2(ctx.phys, &mut props2);
    }

    let mut used = 0_u64;
    let mut total_budget = 0_u64;
    let heaps = (ctx.mem_props.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    for i in 0..heaps {
        if ctx.buffer_heaps & (1 << i) != 0 {
            used = used.saturating_add(budget.heap_usage[i]);
            total_budget = total_budget.saturating_add(budget.heap_budget[i]);
        }
    }
    Ok((used, total_budget))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portability_enumeration_is_requested_only_when_the_loader_has_it() {
        let (exts, flags) = instance_setup(&[portability_enumeration::NAME]);
        assert_eq!(exts, [portability_enumeration::NAME]);
        assert!(flags.contains(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR));
        let (exts, flags) = instance_setup(&[]);
        assert_eq!(exts, Vec::<&CStr>::new());
        assert!(flags.is_empty());
    }

    #[test]
    fn portability_subset_and_memory_budget_are_enabled_when_advertised() {
        let setup = device_setup(&[portability_subset::NAME, memory_budget::NAME], false);
        assert!(setup.extensions.contains(&portability_subset::NAME));
        assert!(setup.has_memory_budget());
        let bare = device_setup(&[], false);
        assert!(bare.extensions.is_empty() && !bare.has_memory_budget());
    }

    #[test]
    fn int64_is_enabled_exactly_when_supported() {
        assert!(device_setup(&[], true).int64);
        assert!(!device_setup(&[], false).int64);
    }

    #[test]
    fn a_missing_required_feature_is_named() {
        let v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .shader_int8(true);
        let v13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        assert_eq!(missing_required(&v12, &v13), None);
        assert_eq!(
            missing_required(&v12.shader_int8(false), &v13),
            Some("shaderInt8")
        );
        assert_eq!(
            missing_required(&v12, &v13.synchronization2(false)),
            Some("synchronization2")
        );
        assert_eq!(
            missing_required(&v12.buffer_device_address(false), &v13),
            Some("bufferDeviceAddress")
        );
    }

    /// Memory properties with the heaps `heaps` (size, flags) and the types
    /// `types` (heap index, flags).
    fn mem_props(
        heaps: &[(u64, vk::MemoryHeapFlags)],
        types: &[(u32, vk::MemoryPropertyFlags)],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: u32::try_from(heaps.len()).unwrap(),
            memory_type_count: u32::try_from(types.len()).unwrap(),
            ..Default::default()
        };
        for (i, &(size, flags)) in heaps.iter().enumerate() {
            p.memory_heaps[i] = vk::MemoryHeap { size, flags };
        }
        for (i, &(heap_index, property_flags)) in types.iter().enumerate() {
            p.memory_types[i] = vk::MemoryType {
                property_flags,
                heap_index,
            };
        }
        p
    }

    /// An integrated GPU's layout: heap 0 host-only, heap 1 device-local;
    /// the device-local host-visible type precedes the cached one, which
    /// lives in heap 0.
    fn shared_memory_layout() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        mem_props(
            &[
                (5 << 30, vk::MemoryHeapFlags::empty()),
                (10 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL),
            ],
            &[
                (1, F::DEVICE_LOCAL),
                (0, F::HOST_VISIBLE | F::HOST_COHERENT),
                (1, F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT),
                (0, F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED),
            ],
        )
    }

    #[test]
    fn readback_heap_counts_towards_an_integrated_budget_only() {
        let p = shared_memory_layout();
        assert_eq!(buffer_heaps(&p, false), 0b10);
        assert_eq!(heap_bytes(&p, buffer_heaps(&p, false)), 10 << 30);
        assert_eq!(buffer_heaps(&p, true), 0b11);
        assert_eq!(heap_bytes(&p, buffer_heaps(&p, true)), 15 << 30);
        assert_eq!(total_device_local_bytes(&p), 10 << 30);
    }

    #[test]
    fn readback_heap_falls_back_to_the_first_coherent_type() {
        use vk::MemoryPropertyFlags as F;
        let p = mem_props(
            &[
                (1 << 30, vk::MemoryHeapFlags::empty()),
                (2 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (3 << 30, vk::MemoryHeapFlags::empty()),
            ],
            &[
                (1, F::DEVICE_LOCAL),
                (2, F::HOST_VISIBLE | F::HOST_COHERENT),
                (0, F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        );
        assert_eq!(buffer_heaps(&p, true), 0b110);
    }

    #[test]
    fn only_an_absolute_vulkan_sdk_is_used() {
        assert_eq!(
            absolute_sdk(Some(Path::new("/sdk"))),
            Some(Path::new("/sdk"))
        );
        assert_eq!(absolute_sdk(Some(Path::new(""))), None);
        assert_eq!(absolute_sdk(Some(Path::new("sdk"))), None);
        assert_eq!(absolute_sdk(None), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn loader_candidates_follow_the_sdk_then_homebrew() {
        let found = loader_candidates(Some(Path::new("/sdk")));
        assert_eq!(found[0], Path::new("/sdk/lib/libvulkan.1.dylib"));
        assert_eq!(found[1], Path::new("/opt/homebrew/lib/libvulkan.1.dylib"));
        assert_eq!(found.len(), 3);
        assert_eq!(loader_candidates(Some(Path::new("sdk"))).len(), 2);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn loader_candidates_are_macos_only() {
        assert_eq!(
            loader_candidates(Some(Path::new("/sdk"))),
            Vec::<PathBuf>::new()
        );
    }
}
