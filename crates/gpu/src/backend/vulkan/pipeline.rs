//! Pipeline cache: SPIR-V → `VkPipeline` per kernel.
//!
//! Each kernel (`icc_clut`, `blit_image`, `page_raster`) gets its own descriptor set layout + pipeline layout + compute pipeline,
//! lazy-created on first dispatch.
//! The SPIR-V blobs are baked into the binary via `include_bytes!` of the
//! build-script outputs.
//!
//! ## Descriptor model
//!
//! All kernels use a single descriptor set (set = 0) with N storage
//! buffers.  Scalar uniforms (`n_pixels`, width, height, eo, blit bbox)
//! are passed via push constants (max 128 bytes — Vulkan's guaranteed
//! minimum is 128, every desktop driver supports at least that).

use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::{describe, vk_err};

/// Identifier for one of the compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum KernelId {
    IccClut,
    BlitImage,
    /// Page raster from a binned command list of coverage tables
    /// (Vulkan only).
    PageRaster,
    /// [`Self::PageRaster`] that also rasterises outlines; needs
    /// `shaderInt64` (Vulkan only).
    PageRasterPaths,
    /// The page raster's floor division over a list of operand pairs, for
    /// the tests; needs `shaderInt64`.
    #[cfg(test)]
    FloorDivProbe,
    /// [`Self::FloorDivProbe`] whose first estimates are offset so that
    /// their products wrap, for the tests of the fault check.
    #[cfg(test)]
    FloorDivProbePoisoned,
    /// [`Self::PageRasterPaths`] whose floor divisions' first estimates are
    /// offset so that their products wrap, for the tests of the fault
    /// check.
    #[cfg(test)]
    PageRasterPathsPoisoned,
}

/// Total number of kernel slots, used to size the `OnceLock` array.
#[cfg(not(test))]
const NUM_KERNELS: usize = 4;
#[cfg(test)]
const NUM_KERNELS: usize = 7;

/// Every variant's `slot_index()` must be `< NUM_KERNELS`. Drift
/// here would surface as a runtime panic when `self.slots[idx]` is
/// accessed; the `const _` exercise below moves the failure to
/// build time. Each line evaluates a `const` expression that panics
/// via `assert!` if the invariant is violated.
const _: () = {
    assert!(KernelId::IccClut.slot_index() < NUM_KERNELS);
    assert!(KernelId::BlitImage.slot_index() < NUM_KERNELS);
    assert!(KernelId::PageRaster.slot_index() < NUM_KERNELS);
    assert!(KernelId::PageRasterPaths.slot_index() < NUM_KERNELS);
    #[cfg(test)]
    assert!(KernelId::FloorDivProbe.slot_index() < NUM_KERNELS);
    #[cfg(test)]
    assert!(KernelId::FloorDivProbePoisoned.slot_index() < NUM_KERNELS);
    #[cfg(test)]
    assert!(KernelId::PageRasterPathsPoisoned.slot_index() < NUM_KERNELS);
};

/// SPIR-V entry-point name of every kernel: each is compiled with
/// `slangc -entry NAME`, which renames its entry point to `main` in the
/// SPIR-V `OpEntryPoint` op.
const ENTRY_POINT: &CStr = c"main";

/// SPIR-V magic number, `OpCapability`, and the `Int64` capability operand.
const SPIRV_MAGIC: u32 = 0x0723_0203;
const OP_CAPABILITY: u32 = 17;
const CAPABILITY_INT64: u32 = 11;
#[cfg(test)]
const CAPABILITY_INT8: u32 = 39;
#[cfg(test)]
const CAPABILITY_STORAGE_BUFFER_8_BIT: u32 = 4448;
#[cfg(test)]
const CAPABILITY_UNIFORM_AND_STORAGE_BUFFER_8_BIT: u32 = 4449;

/// `OpDecorate` and the `FPFastMathMode` decoration; `OpExecutionMode`,
/// `OpExecutionModeId` and the `FPFastMathDefault` execution mode
/// (`SPV_KHR_float_controls2`).
#[cfg(test)]
const OP_DECORATE: u32 = 71;
#[cfg(test)]
const DECORATION_FP_FAST_MATH_MODE: u32 = 40;
#[cfg(test)]
const OP_EXECUTION_MODE: u32 = 16;
#[cfg(test)]
const OP_EXECUTION_MODE_ID: u32 = 331;
#[cfg(test)]
const EXECUTION_MODE_FP_FAST_MATH_DEFAULT: u32 = 6028;

/// The instructions of the SPIR-V module `words`, each with its operands,
/// after the five-word header.  The walk ends at a zero instruction length
/// or an instruction that runs past the module; a module without the
/// SPIR-V magic number has none.
fn spirv_instructions(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let body = if words.first() == Some(&SPIRV_MAGIC) {
        words.get(5..).unwrap_or_default()
    } else {
        &[]
    };
    let mut rest = body;
    std::iter::from_fn(move || {
        let count = (*rest.first()? >> 16) as usize;
        if count == 0 || count > rest.len() {
            return None;
        }
        let (instruction, tail) = rest.split_at(count);
        rest = tail;
        Some(instruction)
    })
}

/// Whether the SPIR-V module `words` declares `OpCapability capability`.
///
/// The `OpCapability` instructions are the first of a module, after the
/// five-word header (logical layout, SPIR-V specification §2.4), so the
/// walk stops at the first other instruction.
fn spirv_declares_capability(words: &[u32], capability: u32) -> bool {
    spirv_instructions(words)
        .take_while(|instruction| instruction[0] & 0xFFFF == OP_CAPABILITY)
        .any(|instruction| instruction.get(1) == Some(&capability))
}

/// Whether the SPIR-V module `words` permits fast math anywhere: an
/// `OpDecorate %id FPFastMathMode`, or an entry point's `FPFastMathDefault`
/// execution mode (`OpExecutionModeId`, or `OpExecutionMode`).  The
/// decoration or mode is the instruction's second operand.
#[cfg(test)]
fn spirv_permits_fast_math(words: &[u32]) -> bool {
    spirv_instructions(words).any(|instruction| {
        let operand = instruction.get(2);
        match instruction[0] & 0xFFFF {
            OP_DECORATE => operand == Some(&DECORATION_FP_FAST_MATH_MODE),
            OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID => {
                operand == Some(&EXECUTION_MODE_FP_FAST_MATH_DEFAULT)
            }
            _ => false,
        }
    })
}

/// The SPIR-V words of `spirv`, decoded as `vkCreateShaderModule` takes
/// them.
fn spirv_words(spirv: &[u8], label: &str) -> Result<Vec<u32>> {
    ash::util::read_spv(&mut std::io::Cursor::new(spirv))
        .map_err(|e| BackendError::msg(format!("read_spv({label}): {e}")))
}

impl KernelId {
    /// Every kernel of this build.
    #[cfg(test)]
    const ALL: &[Self] = &[
        Self::IccClut,
        Self::BlitImage,
        Self::PageRaster,
        Self::PageRasterPaths,
        Self::FloorDivProbe,
        Self::FloorDivProbePoisoned,
        Self::PageRasterPathsPoisoned,
    ];

    /// SPIR-V blob for this kernel, baked into the binary at build time.
    const fn spirv(self) -> &'static [u8] {
        match self {
            Self::IccClut => include_bytes!(concat!(env!("OUT_DIR"), "/icc_clut.spv")),
            Self::BlitImage => include_bytes!(concat!(env!("OUT_DIR"), "/blit_image.spv")),
            Self::PageRaster => include_bytes!(concat!(env!("OUT_DIR"), "/page_raster.spv")),
            Self::PageRasterPaths => {
                include_bytes!(concat!(env!("OUT_DIR"), "/page_raster_paths.spv"))
            }
            #[cfg(test)]
            Self::FloorDivProbe => {
                include_bytes!(concat!(env!("OUT_DIR"), "/floor_div_probe.spv"))
            }
            #[cfg(test)]
            Self::FloorDivProbePoisoned => {
                include_bytes!(concat!(env!("OUT_DIR"), "/floor_div_probe_poisoned.spv"))
            }
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => {
                include_bytes!(concat!(env!("OUT_DIR"), "/page_raster_paths_poisoned.spv"))
            }
        }
    }

    /// Index into the `slots` array. Must be unique per variant and
    /// `< NUM_KERNELS`.
    const fn slot_index(self) -> usize {
        match self {
            Self::IccClut => 0,
            Self::BlitImage => 1,
            Self::PageRaster => 2,
            Self::PageRasterPaths => 3,
            #[cfg(test)]
            Self::FloorDivProbe => 4,
            #[cfg(test)]
            Self::FloorDivProbePoisoned => 5,
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => 6,
        }
    }

    /// Human-readable kernel name for diagnostics.  Matches the source
    /// filename (and the function name in the .cu / .slang).
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::IccClut => "icc_cmyk_clut",
            Self::BlitImage => "blit_image",
            Self::PageRaster => "page_raster",
            Self::PageRasterPaths => "page_raster_paths",
            #[cfg(test)]
            Self::FloorDivProbe => "floor_div_probe",
            #[cfg(test)]
            Self::FloorDivProbePoisoned => "floor_div_probe_poisoned",
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => "page_raster_paths_poisoned",
        }
    }

    /// Descriptor-set binding slots for this kernel's storage buffers,
    /// in the order the recorder supplies them: `0..n_storage_buffers()`
    /// for every kernel.
    ///
    /// The SPIR-V's `OpDecorate %resource Binding N` directly drives
    /// what `Vk{Update,CmdBind}DescriptorSets` must target. Mismatch
    /// produces silent misbinding: kernel reads stale / wrong buffers
    /// at runtime, no validation-layer warning unless strict slot
    /// checking is enabled.
    pub(super) const fn binding_slots(self) -> &'static [u32] {
        match self {
            Self::BlitImage => &[0, 1, 2, 3],
            Self::IccClut => &[0, 1, 2],
            Self::PageRaster => &[0, 1, 2, 3, 4, 5],
            Self::PageRasterPaths => &[0, 1, 2, 3, 4, 5, 6],
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => &[0, 1, 2, 3, 4, 5, 6],
            #[cfg(test)]
            Self::FloorDivProbe | Self::FloorDivProbePoisoned => &[0, 1, 2],
        }
    }

    /// The bytes of this kernel's push constants, the size of its Slang
    /// uniforms, which every dispatch of it pushes in full.
    pub(super) const fn push_bytes(self) -> usize {
        match self {
            // n_nodes, n_pixels, wide
            Self::IccClut => 12,
            // src_w, src_h, src_layout, dst_w, dst_h, bbox[4], filter
            Self::BlitImage => 40,
            // the 14 words of `Params` in page_raster.slang
            Self::PageRaster | Self::PageRasterPaths => 56,
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => 56,
            // n, max_steps
            #[cfg(test)]
            Self::FloorDivProbe | Self::FloorDivProbePoisoned => 8,
        }
    }

    /// Number of `STORAGE_BUFFER` descriptors in set 0.
    ///
    /// Order matches the Slang signature so the recorder binds in the
    /// same order it builds the descriptor write list.
    const fn n_storage_buffers(self) -> u32 {
        match self {
            // (cmyk, rgb, grid)
            Self::IccClut => 3,
            // (src, dst_rgba, cols, rows) — dimensions and bbox travel as push constants
            Self::BlitImage => 4,
            // (cmds, table, row_starts, tile_offsets, tile_items, page)
            Self::PageRaster => 6,
            // PageRaster's, then status
            Self::PageRasterPaths => 7,
            #[cfg(test)]
            Self::PageRasterPathsPoisoned => 7,
            // (pairs, quotients, status)
            #[cfg(test)]
            Self::FloorDivProbe | Self::FloorDivProbePoisoned => 3,
        }
    }
}

/// One compiled kernel: shader module + descriptor layout + pipeline layout + pipeline.
struct CompiledKernel {
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader_module: vk::ShaderModule,
}

/// Lazy pipeline cache.  Uses `OnceLock<CompiledKernel>` per slot — one
/// dispatch builds the pipeline, the rest reuse it.  The host-side
/// `VkPipelineCache` (at `vk_cache`) accelerates compile time
/// across runs by persisting driver-internal pipeline state to a file.
pub(super) struct PipelineCache {
    device: Arc<DeviceCtx>,
    /// Slots indexed by `KernelId::slot_index()`. `OnceLock` so the first
    /// caller initialises and the rest read; thread-safe by construction.
    slots: [std::sync::OnceLock<CompiledKernel>; NUM_KERNELS],
    /// Driver-side pipeline cache — populated at startup from this
    /// device's file ([`CacheKey`]) when its header matches the device,
    /// written back at Drop.
    vk_cache: vk::PipelineCache,
    /// The device's identity, which names its cache file and which the
    /// file's header must match.
    key: CacheKey,
}

/// The identity of a device's pipeline-cache data: the fields of the
/// device properties that the data's header
/// (`VkPipelineCacheHeaderVersionOne`) carries, and the driver version.
///
/// Each key has a file of its own, so the devices of one host do not
/// overwrite one another's data; a file whose header names another vendor,
/// device or cache UUID, or that is shorter than the header, is not passed
/// to the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheKey {
    vendor_id: u32,
    device_id: u32,
    driver_version: u32,
    uuid: [u8; vk::UUID_SIZE],
}

/// Bytes of a version-one pipeline-cache header: length, version, vendor
/// ID and device ID as 32-bit words, then the cache UUID.
const CACHE_HEADER_BYTES: usize = 16 + vk::UUID_SIZE;

impl CacheKey {
    const fn of(props: &vk::PhysicalDeviceProperties) -> Self {
        Self {
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            driver_version: props.driver_version,
            uuid: props.pipeline_cache_uuid,
        }
    }

    /// The name of the key's cache file.
    fn file_name(&self) -> String {
        use std::fmt::Write as _;
        let uuid = self.uuid.iter().fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        });
        format!(
            "vulkan_pipeline_cache-{:04x}-{:04x}-{:08x}-{uuid}.bin",
            self.vendor_id, self.device_id, self.driver_version
        )
    }

    /// Whether `data` opens with a version-one header of this key; the
    /// header's words are stored least significant byte first.
    fn matches_header(&self, data: &[u8]) -> bool {
        let Some((words, uuid)) = data.get(..CACHE_HEADER_BYTES).map(|h| h.split_at(16)) else {
            return false;
        };
        let words: Vec<u32> = words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let version_one = u32::try_from(vk::PipelineCacheHeaderVersion::ONE.as_raw()).ok();
        usize::try_from(words[0]).is_ok_and(|len| len >= CACHE_HEADER_BYTES)
            && Some(words[1]) == version_one
            && words[2] == self.vendor_id
            && words[3] == self.device_id
            && uuid == self.uuid
    }
}

impl PipelineCache {
    /// Construct the cache, attempting to seed the driver-side
    /// `VkPipelineCache` from the on-disk blob.  A missing or unreadable
    /// file is logged at info level and treated as a cold start.
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Arc<Self>> {
        // SAFETY: the physical device belongs to the live instance.
        let props = unsafe { device.instance.get_physical_device_properties(device.phys) };
        let key = CacheKey::of(&props);
        if let Some(dir) = cache_path(&key)
            .as_deref()
            .and_then(std::path::Path::parent)
        {
            remove_stale_files(dir);
        }
        let initial_data = match read_cache_file(&key) {
            Ok(bytes) if key.matches_header(&bytes) => bytes,
            Ok(_) => {
                log::debug!(
                    "{}: header does not name this device; not loaded",
                    key.file_name()
                );
                Vec::new()
            }
            Err(e) => {
                log::debug!("{} not loaded: {e}", key.file_name());
                Vec::new()
            }
        };
        let mut info = vk::PipelineCacheCreateInfo::default();
        if !initial_data.is_empty() {
            info = info.initial_data(&initial_data);
        }
        // Safety: device is live; ash validates the create-info shape.
        let vk_cache = unsafe {
            device
                .device
                .create_pipeline_cache(&info, None)
                .map_err(vk_err("vkCreatePipelineCache"))?
        };
        Ok(Arc::new(Self {
            device,
            slots: Default::default(),
            vk_cache,
            key,
        }))
    }

    /// Get (or build, if first call) the compiled kernel for `id`.
    ///
    /// First-dispatch contention: a racing compile is correctness-safe;
    /// the loser's pipeline is destroyed before we return.
    /// TODO: switch to `OnceLock::get_or_try_init` once stable (rust-lang/rust#109737).
    fn get(&self, id: KernelId) -> Result<&CompiledKernel> {
        let slot = &self.slots[id.slot_index()];
        if let Some(c) = slot.get() {
            return Ok(c);
        }
        let compiled = self.compile(id)?;
        match slot.set(compiled) {
            Ok(()) => Ok(slot.get().expect("just set")),
            Err(extra) => {
                self.destroy_one(&extra);
                Ok(slot.get().expect("set by the racing thread"))
            }
        }
    }

    fn compile(&self, id: KernelId) -> Result<CompiledKernel> {
        let words = spirv_words(id.spirv(), id.label())?;
        if spirv_declares_capability(&words, CAPABILITY_INT64) && !self.device.has_int64 {
            return Err(BackendError::msg(format!(
                "kernel {id:?} uses 64-bit integers, which need shaderInt64; this device lacks it"
            )));
        }
        // Descriptor-set layout binding slots match the kernel's
        // SPIR-V `OpDecorate Binding N` exactly (see
        // `KernelId::binding_slots`).
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = id
            .binding_slots()
            .iter()
            .map(|&slot| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(slot)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // Safety: bindings outlives this call.
        let descriptor_set_layout = unsafe {
            self.device
                .device
                .create_descriptor_set_layout(&dsl_info, None)
        }
        .map_err(vk_err("vkCreateDescriptorSetLayout"))?;

        let layouts = [descriptor_set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            // 128 bytes: Vulkan's guaranteed minimum max push-constant size.
            // Every kernel's `push_bytes` is at most this.
            .size(128)];
        let pl_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { self.device.device.create_pipeline_layout(&pl_info, None) }
            .map_err(vk_err("vkCreatePipelineLayout"))
            .inspect_err(|_| unsafe {
                self.device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            })?;

        let sm_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let shader_module = unsafe { self.device.device.create_shader_module(&sm_info, None) }
            .map_err(vk_err("vkCreateShaderModule"))
            .inspect_err(|_| unsafe {
                self.device
                    .device
                    .destroy_pipeline_layout(pipeline_layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            })?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(ENTRY_POINT);
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout)];

        // Safety: pipeline_info outlives this call.  Pass our persistent
        // VkPipelineCache so the driver can warm-start compilation from
        // its on-disk blob.
        let pipelines = unsafe {
            self.device
                .device
                .create_compute_pipelines(self.vk_cache, &pipeline_info, None)
        };
        let pipeline = match pipelines {
            Ok(mut v) => v.remove(0),
            Err((_partial, code)) => {
                unsafe {
                    self.device
                        .device
                        .destroy_shader_module(shader_module, None);
                    self.device
                        .device
                        .destroy_pipeline_layout(pipeline_layout, None);
                    self.device
                        .device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
                return Err(BackendError::msg(describe(
                    &format!("vkCreateComputePipelines for {}", id.label()),
                    code,
                )));
            }
        };

        Ok(CompiledKernel {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            shader_module,
        })
    }

    fn destroy_one(&self, c: &CompiledKernel) {
        // Safety: handles owned by us and not in use (race-loser path; we
        // never submitted a command buffer using these handles).
        unsafe {
            self.device.device.destroy_pipeline(c.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(c.pipeline_layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(c.descriptor_set_layout, None);
            self.device
                .device
                .destroy_shader_module(c.shader_module, None);
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        // Persist the driver-side pipeline cache to disk before tearing
        // down handles.  Failures are logged but not propagated — we're
        // already on the destruction path and the cache is opportunistic.
        // Safety: vk_cache is owned by us, populated above.
        let data = unsafe { self.device.device.get_pipeline_cache_data(self.vk_cache) };
        match data {
            Ok(bytes) if !bytes.is_empty() => {
                if let Err(e) = write_cache_file(&self.key, &bytes) {
                    log::debug!("{} not written: {e}", self.key.file_name());
                }
            }
            Ok(_) => {} // empty cache, nothing to persist
            Err(e) => log::warn!("vkGetPipelineCacheData failed: {e:?}"),
        }
        // Safety: created via vkCreatePipelineCache in Self::new.
        unsafe {
            self.device
                .device
                .destroy_pipeline_cache(self.vk_cache, None);
        }

        // Take all kernels out of the OnceLocks first so we don't hold a
        // borrow into self.slots while calling self.destroy_one.
        let kernels: Vec<CompiledKernel> = self
            .slots
            .iter_mut()
            .filter_map(std::sync::OnceLock::take)
            .collect();
        for c in &kernels {
            self.destroy_one(c);
        }
    }
}

/// Resolve the absolute path to `key`'s on-disk pipeline cache:
/// `$XDG_CACHE_HOME/rasterrocket/<file>` (falling back to `$HOME/.cache/...`).
/// Returns `None` if neither env var is set — pure-batch builds with no
/// home directory just skip the cache.
fn cache_path(key: &CacheKey) -> Option<std::path::PathBuf> {
    let cache_root = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(cache_root.join("rasterrocket").join(key.file_name()))
}

fn read_cache_file(key: &CacheKey) -> std::io::Result<Vec<u8>> {
    let path =
        cache_path(key).ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    std::fs::read(&path)
}

fn write_cache_file(key: &CacheKey, bytes: &[u8]) -> std::io::Result<()> {
    let path =
        cache_path(key).ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    write_replacing(&path, bytes)
}

/// The file of the single cache all devices once shared.
const LEGACY_CACHE_FILE: &str = "vulkan_pipeline_cache.bin";

/// Age past which a temporary cache file is taken to be abandoned by a
/// writer that was killed between its write and its rename.
const STALE_TEMPORARY: std::time::Duration = std::time::Duration::from_secs(3600);

/// Whether `name` is the legacy cache file, or a temporary file of
/// [`write_replacing`] for a cache file of this module, last modified
/// `age` ago, at least [`STALE_TEMPORARY`].
fn is_stale(name: &str, age: Option<std::time::Duration>) -> bool {
    name == LEGACY_CACHE_FILE
        || (name.starts_with("vulkan_pipeline_cache-")
            && std::path::Path::new(name)
                .extension()
                .is_some_and(|e| e == "tmp")
            && name.contains(".bin.")
            && age.is_some_and(|age| age >= STALE_TEMPORARY))
}

/// Remove from `dir` the files [`is_stale`] names; every failure is
/// ignored, the files being only cache.
fn remove_stale_files(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let age = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok());
        if entry.file_type().is_ok_and(|t| t.is_file()) && is_stale(name, age) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Write `bytes` to a file of its own beside `path`, then rename it over
/// `path`: a rename within one directory replaces the file atomically, so
/// a reader sees the previous data or all of `bytes`, never a partial
/// write, and concurrent writers leave one whole file.  The temporary
/// file is named by process and a per-process count, and is removed when
/// the write or the rename fails.
fn write_replacing(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache path has no directory"))?;
    std::fs::create_dir_all(dir)?;
    let n = WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.{n}.tmp", std::process::id()));
    let tmp = dir.join(name);
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Pipeline handles needed to record a dispatch.  The Vulkan handles are
/// `Copy` and remain valid for the lifetime of the owning `PipelineCache`,
/// which is tied to the `Arc<PipelineCache>` the recorder holds; no
/// separate borrow is needed here.
#[derive(Clone, Copy)]
pub(super) struct PipelineHandles {
    pub(super) pipeline: vk::Pipeline,
    pub(super) layout: vk::PipelineLayout,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) n_storage_buffers: u32,
    /// Slot indices the kernel binds at (length == `n_storage_buffers`).
    pub(super) binding_slots: &'static [u32],
}

impl PipelineCache {
    /// Get pipeline handles for `id`, lazily compiling on first call.
    pub(super) fn handles(&self, id: KernelId) -> Result<PipelineHandles> {
        let c = self.get(id)?;
        Ok(PipelineHandles {
            pipeline: c.pipeline,
            layout: c.pipeline_layout,
            descriptor_set_layout: c.descriptor_set_layout,
            n_storage_buffers: id.n_storage_buffers(),
            binding_slots: id.binding_slots(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> CacheKey {
        CacheKey {
            vendor_id: 0x10de,
            device_id: 0x1e81,
            driver_version: 0x9b11_8000,
            uuid: std::array::from_fn(|i| u8::try_from(i * 17).expect("below 256")),
        }
    }

    fn header(key: &CacheKey) -> Vec<u8> {
        let mut h = Vec::new();
        for w in [32, 1, key.vendor_id, key.device_id] {
            h.extend_from_slice(&w.to_le_bytes());
        }
        h.extend_from_slice(&key.uuid);
        h
    }

    #[test]
    fn cache_files_are_named_per_device_and_driver() {
        let a = key();
        assert_eq!(
            a.file_name(),
            "vulkan_pipeline_cache-10de-1e81-9b118000-00112233445566778899aabbccddeeff.bin"
        );
        let others = [
            CacheKey {
                device_id: 0x3e92,
                ..a
            },
            CacheKey {
                driver_version: 1,
                ..a
            },
            CacheKey { uuid: [0; 16], ..a },
        ];
        for b in others {
            assert_ne!(a.file_name(), b.file_name());
        }
    }

    #[test]
    fn only_a_header_of_the_device_is_loaded() {
        let k = key();
        let mut data = header(&k);
        data.extend_from_slice(&[0xAB; 64]);
        assert!(k.matches_header(&data));
        assert!(!k.matches_header(&data[..CACHE_HEADER_BYTES - 1]));
        assert!(!k.matches_header(&[]));
        for (at, value) in [(0, 31_u8), (4, 2), (8, 0x02), (12, 0x00), (16, 0xFF)] {
            let mut bad = data.clone();
            bad[at] = value;
            assert!(!k.matches_header(&bad), "byte {at} = {value}");
        }
    }

    #[test]
    fn only_the_legacy_file_and_abandoned_temporaries_are_stale() {
        let hour = Some(STALE_TEMPORARY);
        let tmp = "vulkan_pipeline_cache-10de-1e81-9b118000-00.bin.4242.0.tmp";
        assert!(is_stale(LEGACY_CACHE_FILE, None));
        assert!(is_stale(tmp, hour));
        assert!(!is_stale(tmp, Some(std::time::Duration::from_secs(5))));
        assert!(!is_stale(tmp, None));
        assert!(!is_stale(&key().file_name(), hour));
        assert!(!is_stale("other.bin.1.0.tmp", hour));
        assert!(!is_stale("vulkan_pipeline_cache-x.tmp", hour));
    }

    #[test]
    fn a_replacing_write_leaves_only_the_whole_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("rasterrocket");
        let path = dir.join("cache.bin");
        write_replacing(&path, &[1; 100]).expect("first write");
        write_replacing(&path, &[2; 10]).expect("second write");
        assert_eq!(std::fs::read(&path).expect("read"), [2; 10]);
        let names: Vec<_> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert_eq!(names, ["cache.bin"]);
    }

    #[test]
    fn int64_is_declared_by_the_64_bit_kernels_only() {
        let declares = |id: KernelId, capability| {
            spirv_declares_capability(&spirv_words(id.spirv(), id.label()).unwrap(), capability)
        };
        assert!(!declares(KernelId::IccClut, CAPABILITY_INT64));
        assert!(!declares(KernelId::PageRaster, CAPABILITY_INT64));
        assert!(declares(KernelId::PageRasterPaths, CAPABILITY_INT64));
        assert!(declares(KernelId::FloorDivProbe, CAPABILITY_INT64));
        // Every module declares Shader (capability 1).
        assert!(declares(KernelId::IccClut, 1));
    }

    #[test]
    fn no_kernel_declares_8_bit_capabilities() {
        for &id in KernelId::ALL {
            let words = spirv_words(id.spirv(), id.label()).unwrap();
            for capability in [
                CAPABILITY_INT8,
                CAPABILITY_STORAGE_BUFFER_8_BIT,
                CAPABILITY_UNIFORM_AND_STORAGE_BUFFER_8_BIT,
            ] {
                assert!(
                    !spirv_declares_capability(&words, capability),
                    "{} declares capability {capability}",
                    id.label()
                );
            }
        }
    }

    #[test]
    fn capability_walk_stops_at_the_first_other_instruction() {
        let op = |count: u32, opcode: u32| (count << 16) | opcode;
        let header = [SPIRV_MAGIC, 0x0001_0300, 0, 1, 0];
        let module: Vec<u32> = header
            .into_iter()
            .chain([op(2, OP_CAPABILITY), 1])
            .chain([op(3, 14), 0, 1])
            .chain([op(2, OP_CAPABILITY), CAPABILITY_INT64])
            .collect();
        assert!(spirv_declares_capability(&module, 1));
        assert!(!spirv_declares_capability(&module, CAPABILITY_INT64));
        let mut no_magic = module;
        no_magic[0] = 0;
        assert!(!spirv_declares_capability(&no_magic, 1));
    }

    #[test]
    fn spirv_permits_no_fast_math() {
        for &id in KernelId::ALL {
            let words = spirv_words(id.spirv(), id.label()).unwrap();
            let walked: usize = spirv_instructions(&words).map(<[u32]>::len).sum();
            assert_eq!(
                walked,
                words.len() - 5,
                "{} is walked to its end",
                id.label()
            );
            assert!(!spirv_permits_fast_math(&words), "{}", id.label());
        }
    }

    #[test]
    fn fast_math_walk_reaches_decorations_past_the_capabilities() {
        let op = |count: u32, opcode: u32| (count << 16) | opcode;
        let header = [SPIRV_MAGIC, 0x0001_0300, 0, 8, 0];
        let module: Vec<u32> = header
            .into_iter()
            .chain([op(2, OP_CAPABILITY), 1])
            .chain([op(3, 14), 0, 1])
            .chain([op(3, OP_DECORATE), 7, 6])
            .chain([op(4, OP_DECORATE), 7, DECORATION_FP_FAST_MATH_MODE, 0])
            .collect();
        assert!(spirv_permits_fast_math(&module));
        // The same decoration with another opcode, and one cut short by
        // the module's end, are not found.
        let mut other = module.clone();
        other[13] = op(4, 72);
        assert!(!spirv_permits_fast_math(&other));
        let truncated = &module[..module.len() - 1];
        assert!(!spirv_permits_fast_math(truncated));
    }

    #[test]
    fn fast_math_walk_finds_a_fast_math_default_execution_mode() {
        let op = |count: u32, opcode: u32| (count << 16) | opcode;
        let header = [SPIRV_MAGIC, 0x0001_0600, 0, 8, 0];
        let with = |opcode: u32, mode: u32| -> Vec<u32> {
            header
                .into_iter()
                .chain([op(2, OP_CAPABILITY), 1])
                .chain([op(3, OP_EXECUTION_MODE), 4, 17])
                .chain([op(5, opcode), 4, mode, 2, 3])
                .collect()
        };
        assert!(spirv_permits_fast_math(&with(
            OP_EXECUTION_MODE_ID,
            EXECUTION_MODE_FP_FAST_MATH_DEFAULT
        )));
        assert!(spirv_permits_fast_math(&with(
            OP_EXECUTION_MODE,
            EXECUTION_MODE_FP_FAST_MATH_DEFAULT
        )));
        // LocalSizeId (38) is another mode; OpDecorateId (332) another opcode.
        assert!(!spirv_permits_fast_math(&with(OP_EXECUTION_MODE_ID, 38)));
        assert!(!spirv_permits_fast_math(&with(
            332,
            EXECUTION_MODE_FP_FAST_MATH_DEFAULT
        )));
    }

    /// The recorder feeds `binding_slots` to descriptor-set
    /// `vkUpdateDescriptorSets` while passing one buffer per slot
    /// from a list sized to `n_storage_buffers`.  If those two
    /// disagree, the kernel either reads stale data (some slot
    /// goes unbound) or the recorder over-runs its handle slice.
    /// Lock the invariant in a pure-CPU test so a future kernel
    /// addition can't silently drift.
    #[test]
    fn binding_slots_len_matches_n_storage_buffers_for_every_kernel() {
        let kernels = KernelId::ALL;
        for id in kernels {
            let slots = id.binding_slots();
            let n = id.n_storage_buffers();
            assert_eq!(
                slots.len(),
                n as usize,
                "{} binding_slots ({:?}) disagrees with n_storage_buffers ({n})",
                id.label(),
                slots,
            );
        }
    }

    /// The registers and root constants of the DXIL build
    /// (`kernels/d3d12_bindings.rs`), as the Vulkan host binds the same
    /// kernels: one SRV or UAV a storage buffer, and root constants of the
    /// push block's words.
    #[test]
    fn the_d3d12_tables_match_the_vulkan_kernel_shapes() {
        mod tables {
            include!("../../../kernels/d3d12_bindings.rs");
        }
        use tables::{KERNEL_CONSTANT_WORDS, KERNEL_REGISTERS, RegisterClass};

        let artifact = |id: KernelId| match id {
            KernelId::IccClut => "icc_clut",
            KernelId::BlitImage => "blit_image",
            KernelId::PageRaster => "page_raster",
            KernelId::PageRasterPaths => "page_raster_paths",
            KernelId::FloorDivProbe => "floor_div_probe",
            KernelId::FloorDivProbePoisoned => "floor_div_probe_poisoned",
            KernelId::PageRasterPathsPoisoned => "page_raster_paths_poisoned",
        };
        assert_eq!(KERNEL_REGISTERS.len(), KernelId::ALL.len());
        assert_eq!(KERNEL_CONSTANT_WORDS.len(), KernelId::ALL.len());
        for &id in KernelId::ALL {
            let name = artifact(id);
            let registers = KERNEL_REGISTERS
                .iter()
                .find(|(k, _)| *k == name)
                .unwrap_or_else(|| panic!("{name} has no registers"))
                .1;
            let count = |class| registers.iter().filter(|r| r.class == class).count();
            assert!(
                registers.iter().all(|r| r.space == 0 && !r.name.is_empty()),
                "{name}"
            );
            assert_eq!(
                count(RegisterClass::Srv) + count(RegisterClass::Uav),
                id.n_storage_buffers() as usize,
                "{name}: SRVs and UAVs against storage buffers"
            );
            let slots: Vec<u32> = (0..id.n_storage_buffers()).collect();
            assert_eq!(id.binding_slots(), slots, "{name}");
            let words = KERNEL_CONSTANT_WORDS
                .iter()
                .find(|(k, _)| *k == name)
                .unwrap_or_else(|| panic!("{name} has no root constants"))
                .1;
            assert_eq!(
                words as usize * 4,
                id.push_bytes(),
                "{name}: root constants against the push block"
            );
            assert!(
                id.push_bytes() <= 128,
                "{name}: past the push-constant range"
            );
            assert_eq!(
                count(RegisterClass::Constants),
                usize::from(words > 0),
                "{name}"
            );
            assert!(
                registers
                    .iter()
                    .filter(|r| r.class == RegisterClass::Constants)
                    .all(|r| r.index == 0),
                "{name}"
            );
        }
    }

    /// `slot_index()` is the array index into `PipelineCache::slots`;
    /// two kernels sharing the same index would silently overwrite
    /// each other's `OnceLock`.  The `const _` block already enforces
    /// `< NUM_KERNELS` at build time; this test confirms uniqueness.
    #[test]
    fn slot_indices_are_unique_across_kernels() {
        let kernels = KernelId::ALL;
        let mut indices: Vec<usize> = kernels.iter().map(|k| k.slot_index()).collect();
        indices.sort_unstable();
        let unique_count = indices.windows(2).filter(|w| w[0] != w[1]).count() + 1;
        assert_eq!(
            unique_count,
            kernels.len(),
            "slot_index() must be unique across kernels; got {indices:?}",
        );
    }
}
