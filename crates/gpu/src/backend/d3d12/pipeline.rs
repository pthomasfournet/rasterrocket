//! Root signatures and compute pipeline states, one of each per kernel,
//! built on first use from the kernel's registers and its DXIL.
//!
//! A kernel's root signature is `[root constants at b0, descriptor table]`
//! with the table holding its SRVs (`t0..`) then its UAVs (`u0..`), one
//! range per class, all in space 0: the shape `kernels/d3d12_bindings.rs`
//! records and the build checks against the compiled HLSL.

use std::mem::ManuallyDrop;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::{
    D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_DESCRIPTOR_RANGE,
    D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND, D3D12_DESCRIPTOR_RANGE_TYPE,
    D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_ROOT_CONSTANTS,
    D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
    D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
    D3D12_ROOT_SIGNATURE_DESC, D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE,
    D3D12_SHADER_VISIBILITY_ALL, D3D12SerializeRootSignature, ID3D12Device, ID3D12PipelineState,
    ID3D12RootSignature,
};

use super::error::{device_err, hr_err};
use crate::backend::dxil::{container_is_hashed, dxil};
use crate::backend::{BackendError, Result};

mod tables {
    include!("../../../kernels/d3d12_bindings.rs");
}
use tables::{KERNEL_CONSTANT_WORDS, KERNEL_REGISTERS, Register, RegisterClass};

/// The most SRVs and UAVs one kernel binds.
pub(super) const MAX_BINDINGS: u32 = 8;

/// A kernel of the backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KernelId {
    IccClut,
    BlitImage,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the page raster dispatches it; nothing does yet")
    )]
    PageRaster,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the page raster dispatches it; nothing does yet")
    )]
    PageRasterPaths,
    /// The page raster's floor division over operand pairs, for the tests.
    #[cfg(test)]
    FloorDivProbe,
}

impl KernelId {
    /// Every kernel, by slot.
    #[cfg(test)]
    pub(super) const ALL: &[Self] = &[
        Self::IccClut,
        Self::BlitImage,
        Self::PageRaster,
        Self::PageRasterPaths,
        Self::FloorDivProbe,
    ];

    /// The number of kernels.
    const COUNT: usize = if cfg!(test) { 5 } else { 4 };

    /// The kernel's name: its DXIL artifact and its entry in the register
    /// tables.
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::IccClut => "icc_clut",
            Self::BlitImage => "blit_image",
            Self::PageRaster => "page_raster",
            Self::PageRasterPaths => "page_raster_paths",
            #[cfg(test)]
            Self::FloorDivProbe => "floor_div_probe",
        }
    }

    /// The kernel's slot in [`PipelineCache`].
    const fn slot(self) -> usize {
        match self {
            Self::IccClut => 0,
            Self::BlitImage => 1,
            Self::PageRaster => 2,
            Self::PageRasterPaths => 3,
            #[cfg(test)]
            Self::FloorDivProbe => 4,
        }
    }
}

/// The shape of a kernel's root signature, from the register tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    /// Root-constant words at `b0`; 0 for a kernel without constants.
    pub(super) constant_words: u32,
    /// SRVs, `t0..srvs`.
    pub(super) srvs: u32,
    /// UAVs, `u0..uavs`.
    pub(super) uavs: u32,
}

impl Shape {
    /// The root parameter index of the descriptor table.
    pub(super) const fn table_index(self) -> u32 {
        if self.constant_words > 0 { 1 } else { 0 }
    }
}

/// The shape of `kernel`'s root signature.
///
/// # Errors
/// When the tables lack the kernel, or its registers of a class are not
/// numbered `0..n` in space 0 or number more than [`MAX_BINDINGS`].
pub(super) fn shape(kernel: KernelId) -> Result<Shape> {
    let name = kernel.label();
    let registers: &[Register] = KERNEL_REGISTERS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|e| e.1)
        .ok_or_else(|| BackendError::msg(format!("{name}: no registers")))?;
    let constant_words = KERNEL_CONSTANT_WORDS
        .iter()
        .find(|(k, _)| *k == name)
        .map(|e| e.1)
        .ok_or_else(|| BackendError::msg(format!("{name}: no root-constant size")))?;
    let count = |class: RegisterClass| -> Result<u32> {
        let mut n = 0_u32;
        for r in registers.iter().filter(|r| r.class == class) {
            if r.index != n || r.space != 0 {
                return Err(BackendError::msg(format!(
                    "{name}: register of {} is {:?} {} in space {}, not {n} in space 0",
                    r.name, r.class, r.index, r.space
                )));
            }
            n += 1;
        }
        Ok(n)
    };
    let constants = count(RegisterClass::Constants)?;
    if constants != u32::from(constant_words > 0) {
        return Err(BackendError::msg(format!(
            "{name}: {constants} constant registers for {constant_words} root-constant words"
        )));
    }
    let shape = Shape {
        constant_words,
        srvs: count(RegisterClass::Srv)?,
        uavs: count(RegisterClass::Uav)?,
    };
    if shape.srvs + shape.uavs > MAX_BINDINGS {
        return Err(BackendError::msg(format!(
            "{name}: {} views, past the {MAX_BINDINGS} a dispatch reserves",
            shape.srvs + shape.uavs
        )));
    }
    Ok(shape)
}

/// The objects a dispatch of one kernel binds.
#[derive(Clone)]
pub(super) struct Pipeline {
    pub(super) shape: Shape,
    pub(super) root_signature: ID3D12RootSignature,
    pub(super) state: ID3D12PipelineState,
}

/// One [`Pipeline`] per kernel, each built on its first use.
pub(super) struct PipelineCache {
    device: ID3D12Device,
    slots: [OnceLock<Pipeline>; KernelId::COUNT],
}

impl PipelineCache {
    pub(super) fn new(device: &ID3D12Device) -> Self {
        Self {
            device: device.clone(),
            slots: std::array::from_fn(|_| OnceLock::new()),
        }
    }

    /// The pipeline of `kernel`, built when this is its first use.  A
    /// failed build is not cached; the next use tries again.
    pub(super) fn get(&self, kernel: KernelId) -> Result<Pipeline> {
        let slot = &self.slots[kernel.slot()];
        if let Some(p) = slot.get() {
            return Ok(p.clone());
        }
        let built = self.build(kernel)?;
        Ok(slot.get_or_init(|| built).clone())
    }

    fn build(&self, kernel: KernelId) -> Result<Pipeline> {
        let shape = shape(kernel)?;
        let root_signature = self.root_signature(kernel, shape)?;
        let blob = dxil(kernel.label());
        if !container_is_hashed(blob) {
            return Err(BackendError::msg(format!(
                "{}: the DXIL container is not signed",
                kernel.label()
            )));
        }
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a non-owning copy of the interface pointer; the
            // `ManuallyDrop` never releases it, and `root_signature`
            // outlives the call that reads it.
            pRootSignature: unsafe { std::mem::transmute_copy(&root_signature) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: blob.as_ptr().cast(),
                BytecodeLength: blob.len(),
            },
            ..Default::default()
        };
        // SAFETY: `desc` and the bytecode it points at outlive the call.
        let state: ID3D12PipelineState =
            unsafe { self.device.CreateComputePipelineState(&raw const desc) }
                .map_err(device_err(&self.device, "CreateComputePipelineState"))
                .map_err(|e| BackendError::msg(format!("{}: {e}", kernel.label())))?;
        Ok(Pipeline {
            shape,
            root_signature,
            state,
        })
    }

    /// Serialise and create the root signature of `shape`.
    fn root_signature(&self, kernel: KernelId, shape: Shape) -> Result<ID3D12RootSignature> {
        let range = |kind: D3D12_DESCRIPTOR_RANGE_TYPE, n: u32| D3D12_DESCRIPTOR_RANGE {
            RangeType: kind,
            NumDescriptors: n,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let ranges: Vec<D3D12_DESCRIPTOR_RANGE> = [
            (D3D12_DESCRIPTOR_RANGE_TYPE_SRV, shape.srvs),
            (D3D12_DESCRIPTOR_RANGE_TYPE_UAV, shape.uavs),
        ]
        .into_iter()
        .filter(|&(_, n)| n > 0)
        .map(|(kind, n)| range(kind, n))
        .collect();
        let mut params = Vec::with_capacity(2);
        if shape.constant_words > 0 {
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: shape.constant_words,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        if !ranges.is_empty() {
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: u32::try_from(ranges.len()).unwrap_or(u32::MAX),
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: u32::try_from(params.len()).unwrap_or(u32::MAX),
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: `desc`, `params` and `ranges` outlive the call; the
        // out-pointers are valid.
        let serialized = unsafe {
            D3D12SerializeRootSignature(
                &raw const desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &raw mut blob,
                Some(&raw mut errors),
            )
        };
        if let Err(e) = serialized {
            let detail = errors.as_ref().map_or_default(blob_text);
            return Err(BackendError::msg(format!(
                "{}: {}: {detail}",
                kernel.label(),
                hr_err("D3D12SerializeRootSignature")(e)
            )));
        }
        let blob = blob.ok_or_else(|| BackendError::msg("D3D12SerializeRootSignature: no blob"))?;
        // SAFETY: the blob is live; its pointer and size describe its bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        // SAFETY: `bytes` is a serialised root signature.
        unsafe { self.device.CreateRootSignature(0, bytes) }
            .map_err(device_err(&self.device, "CreateRootSignature"))
    }
}

/// The text of an error blob.
fn blob_text(blob: &ID3DBlob) -> String {
    // SAFETY: the blob is live; its pointer and size describe its bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    String::from_utf8_lossy(bytes)
        .trim_end_matches('\0')
        .trim()
        .to_owned()
}

/// A pipeline-state description holds a non-owning root-signature pointer.
const _: () = assert!(
    size_of::<ManuallyDrop<Option<ID3D12RootSignature>>>() == size_of::<ID3D12RootSignature>()
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_follow_the_register_tables() {
        let s = |k| shape(k).expect("shape");
        assert_eq!(
            s(KernelId::IccClut),
            Shape {
                constant_words: 3,
                srvs: 2,
                uavs: 1
            }
        );
        assert_eq!(
            s(KernelId::BlitImage),
            Shape {
                constant_words: 10,
                srvs: 3,
                uavs: 1
            }
        );
        assert_eq!(s(KernelId::PageRasterPaths).uavs, 2);
        assert_eq!(
            s(KernelId::FloorDivProbe),
            Shape {
                constant_words: 2,
                srvs: 1,
                uavs: 2
            }
        );
        for &k in KernelId::ALL {
            assert_eq!(s(k).table_index(), 1, "{}", k.label());
            assert_eq!(
                k.slot(),
                KernelId::ALL.iter().position(|&x| x == k).expect("listed")
            );
        }
    }
}
