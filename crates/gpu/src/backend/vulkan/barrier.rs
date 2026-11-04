//! Global memory barriers between the commands of one queue.
//!
//! Every barrier the backend records is a single `VkMemoryBarrier2` over
//! all resources, described by the stage and access scope of each side.

use ash::vk;

/// Stage and access scope of one side of a memory dependency.
#[derive(Debug, Clone, Copy)]
pub(super) struct Scope {
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
}

impl Scope {
    pub(super) const fn new(stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Self {
        Self { stage, access }
    }
}

/// Stages of the commands a page records: dispatches, copies, and fills
/// (`vkCmdFillBuffer` runs in `CLEAR`, which `ALL_TRANSFER` contains).
const PAGE_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
);

/// Every access of every command a page records.
pub(super) const PAGE_COMMANDS: Scope = Scope::new(
    PAGE_STAGES,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::TRANSFER_READ.as_raw()
            | vk::AccessFlags2::TRANSFER_WRITE.as_raw(),
    ),
);

/// The writes of every command a page records.
pub(super) const PAGE_WRITES: Scope = Scope::new(
    PAGE_STAGES,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw() | vk::AccessFlags2::TRANSFER_WRITE.as_raw(),
    ),
);

/// Storage-buffer writes of a compute dispatch.
pub(super) const DISPATCH_WRITES: Scope = Scope::new(
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::SHADER_STORAGE_WRITE,
);

/// The reads and writes of a `vkCmdCopyBuffer`.
pub(super) const COPY_ACCESS: Scope = Scope::new(
    vk::PipelineStageFlags2::COPY,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::TRANSFER_READ.as_raw() | vk::AccessFlags2::TRANSFER_WRITE.as_raw(),
    ),
);

/// The writes of a `vkCmdCopyBuffer`.
pub(super) const COPY_WRITES: Scope = Scope::new(
    vk::PipelineStageFlags2::COPY,
    vk::AccessFlags2::TRANSFER_WRITE,
);

/// The writes of a `vkCmdFillBuffer`.
pub(super) const FILL_WRITES: Scope = Scope::new(
    vk::PipelineStageFlags2::CLEAR,
    vk::AccessFlags2::TRANSFER_WRITE,
);

/// Every device write, of any command.
pub(super) const ALL_WRITES: Scope = Scope::new(
    vk::PipelineStageFlags2::ALL_COMMANDS,
    vk::AccessFlags2::MEMORY_WRITE,
);

/// Host reads after the submission's completion has been waited on.
pub(super) const HOST_READS: Scope =
    Scope::new(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);

/// Record a global memory barrier from `src` to `dst` into `cmd`.
///
/// # Safety
/// `cmd` was allocated from `device` and is in the recording state, and no
/// other thread records into it.
pub(super) unsafe fn global(device: &ash::Device, cmd: vk::CommandBuffer, src: Scope, dst: Scope) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.stage)
        .src_access_mask(src.access)
        .dst_stage_mask(dst.stage)
        .dst_access_mask(dst.access)];
    // Safety: forwarded from this function's contract.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barrier),
        );
    }
}
