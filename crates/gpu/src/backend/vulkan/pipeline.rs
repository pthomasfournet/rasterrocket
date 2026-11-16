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
use super::error::vk_err;

/// Identifier for one of the compute kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum KernelId {
    IccClut,
    BlitImage,
    /// Page raster from a binned command list (Vulkan only).
    PageRaster,
}

/// Total number of kernel slots, used to size the `OnceLock` array.
const NUM_KERNELS: usize = 3;

/// Every variant's `slot_index()` must be `< NUM_KERNELS`. Drift
/// here would surface as a runtime panic when `self.slots[idx]` is
/// accessed; the `const _` exercise below moves the failure to
/// build time. Each line evaluates a `const` expression that panics
/// via `assert!` if the invariant is violated.
const _: () = {
    assert!(KernelId::IccClut.slot_index() < NUM_KERNELS);
    assert!(KernelId::BlitImage.slot_index() < NUM_KERNELS);
    assert!(KernelId::PageRaster.slot_index() < NUM_KERNELS);
};

/// SPIR-V entry-point name of every kernel: each is compiled with
/// `slangc -entry NAME`, which renames its entry point to `main` in the
/// SPIR-V `OpEntryPoint` op.
const ENTRY_POINT: &CStr = c"main";

/// SPIR-V magic number, `OpCapability`, and the `Int64` capability operand.
const SPIRV_MAGIC: u32 = 0x0723_0203;
const OP_CAPABILITY: u32 = 17;
const CAPABILITY_INT64: u32 = 11;

/// Whether the SPIR-V module `words` declares `OpCapability capability`.
///
/// The `OpCapability` instructions are the first of a module, after the
/// five-word header (logical layout, SPIR-V specification §2.4), so the
/// walk stops at the first other instruction.  A module without the
/// SPIR-V magic number, or with a zero instruction length, declares none.
fn spirv_declares_capability(words: &[u32], capability: u32) -> bool {
    if words.first() != Some(&SPIRV_MAGIC) {
        return false;
    }
    let mut at = 5;
    while let Some(&first) = words.get(at) {
        let (count, opcode) = ((first >> 16) as usize, first & 0xFFFF);
        if count == 0 || opcode != OP_CAPABILITY {
            return false;
        }
        if words.get(at + 1) == Some(&capability) {
            return true;
        }
        at += count;
    }
    false
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
    const ALL: &[Self] = &[Self::IccClut, Self::BlitImage, Self::PageRaster];

    /// SPIR-V blob for this kernel, baked into the binary at build time.
    const fn spirv(self) -> &'static [u8] {
        match self {
            Self::IccClut => include_bytes!(concat!(env!("OUT_DIR"), "/icc_clut.spv")),
            Self::BlitImage => include_bytes!(concat!(env!("OUT_DIR"), "/blit_image.spv")),
            Self::PageRaster => include_bytes!(concat!(env!("OUT_DIR"), "/page_raster.spv")),
        }
    }

    /// Index into the `slots` array. Must be unique per variant and
    /// `< NUM_KERNELS`.
    const fn slot_index(self) -> usize {
        match self {
            Self::IccClut => 0,
            Self::BlitImage => 1,
            Self::PageRaster => 2,
        }
    }

    /// Human-readable kernel name for diagnostics.  Matches the source
    /// filename (and the function name in the .cu / .slang).
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::IccClut => "icc_cmyk_clut",
            Self::BlitImage => "blit_image",
            Self::PageRaster => "page_raster",
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
        }
    }

    /// Number of `STORAGE_BUFFER` descriptors in set 0.
    ///
    /// Order matches the Slang signature so the recorder binds in the
    /// same order it builds the descriptor write list.
    const fn n_storage_buffers(self) -> u32 {
        match self {
            // (cmyk, rgb, clut)
            Self::IccClut => 3,
            // (src, dst_rgba, cols, rows) — dimensions and bbox travel as push constants
            Self::BlitImage => 4,
            // (cmds, intersects, row_starts, tile_offsets, tile_cmds, page)
            Self::PageRaster => 6,
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
    /// Driver-side pipeline cache — populated from disk at startup if a
    /// matching file exists, written back at Drop.  Vulkan validates the
    /// cache header (driver UUID etc.) so a cache from a different driver
    /// version is silently ignored.  `vk::PipelineCache::null()` is a
    /// valid passthrough; we never expose this handle directly.
    vk_cache: vk::PipelineCache,
}

/// Filename of the on-disk pipeline-cache blob.  Lives under the user's
/// XDG cache root.  Vulkan's own header tags the device/driver, so a
/// shared filename is safe — mismatched caches are rejected at load.
const CACHE_FILENAME: &str = "vulkan_pipeline_cache.bin";

impl PipelineCache {
    /// Construct the cache, attempting to seed the driver-side
    /// `VkPipelineCache` from the on-disk blob.  A missing or unreadable
    /// file is logged at info level and treated as a cold start.
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Arc<Self>> {
        let initial_data = match read_cache_file() {
            Ok(bytes) => bytes,
            Err(e) => {
                log::debug!("vulkan_pipeline_cache.bin not loaded: {e}");
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
            // Our largest kernel push struct is well under this (icc_clut
            // pushes 12 bytes; blit_image pushes 40).
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
                return Err(BackendError::msg(format!(
                    "vkCreateComputePipelines for {} failed: {code:?}",
                    id.label()
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
                if let Err(e) = write_cache_file(&bytes) {
                    log::debug!("vulkan_pipeline_cache.bin not written: {e}");
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

/// Resolve the absolute path to the on-disk pipeline cache:
/// `$XDG_CACHE_HOME/rasterrocket/<file>` (falling back to `$HOME/.cache/...`).
/// Returns `None` if neither env var is set — pure-batch builds with no
/// home directory just skip the cache.
fn cache_path() -> Option<std::path::PathBuf> {
    let cache_root = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(cache_root.join("rasterrocket").join(CACHE_FILENAME))
}

fn read_cache_file() -> std::io::Result<Vec<u8>> {
    let path = cache_path().ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    std::fs::read(&path)
}

fn write_cache_file(bytes: &[u8]) -> std::io::Result<()> {
    let path = cache_path().ok_or_else(|| std::io::Error::other("no $XDG_CACHE_HOME or $HOME"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, bytes)
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

    #[test]
    fn int64_is_declared_by_the_64_bit_kernels_only() {
        let declares = |id: KernelId, capability| {
            spirv_declares_capability(&spirv_words(id.spirv(), id.label()).unwrap(), capability)
        };
        assert!(!declares(KernelId::IccClut, CAPABILITY_INT64));
        assert!(!declares(KernelId::PageRaster, CAPABILITY_INT64));
        // Every module declares Shader (capability 1).
        assert!(declares(KernelId::IccClut, 1));
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
