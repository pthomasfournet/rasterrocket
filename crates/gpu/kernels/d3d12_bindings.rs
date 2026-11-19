// The HLSL registers of every kernel's parameters and the size of every
// kernel's root constants.  `build.rs` checks both against slangc's
// reflection of the HLSL it compiles to DXIL; the root signatures are
// built from them.

/// The HLSL register class of a kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RegisterClass {
    /// A constant buffer, `b` register: the root constants.
    Constants,
    /// A shader resource view, `t` register: a read-only buffer.
    Srv,
    /// An unordered access view, `u` register: a read-write buffer.
    Uav,
}

/// The register of one kernel parameter: `index` in `class`, in register
/// space `space`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Register {
    /// The parameter's name in the Slang source; `entryPointParams` for
    /// the constant buffer slangc builds from an entry point's `uniform`
    /// scalars.
    pub(crate) name: &'static str,
    /// Its register class.
    pub(crate) class: RegisterClass,
    /// Its register number within `class`.
    pub(crate) index: u32,
    /// Its register space.
    pub(crate) space: u32,
}

/// `name`'s register `index` of `class` in space 0.
const fn reg(name: &'static str, class: RegisterClass, index: u32) -> Register {
    Register {
        name,
        class,
        index,
        space: 0,
    }
}

const ICC_CLUT: &[Register] = &[
    reg("entryPointParams", RegisterClass::Constants, 0),
    reg("cmyk", RegisterClass::Srv, 0),
    reg("grid", RegisterClass::Srv, 1),
    reg("rgb", RegisterClass::Uav, 0),
];

const BLIT_IMAGE: &[Register] = &[
    reg("entryPointParams", RegisterClass::Constants, 0),
    reg("src", RegisterClass::Srv, 0),
    reg("cols", RegisterClass::Srv, 1),
    reg("rows", RegisterClass::Srv, 2),
    reg("dst_rgba", RegisterClass::Uav, 0),
];

const PAGE_RASTER: &[Register] = &[
    reg("pc", RegisterClass::Constants, 0),
    reg("cmds", RegisterClass::Srv, 0),
    reg("table", RegisterClass::Srv, 1),
    reg("row_starts", RegisterClass::Srv, 2),
    reg("tile_offsets", RegisterClass::Srv, 3),
    reg("tile_items", RegisterClass::Srv, 4),
    reg("page", RegisterClass::Uav, 0),
];

const PAGE_RASTER_PATHS: &[Register] = &[
    reg("pc", RegisterClass::Constants, 0),
    reg("cmds", RegisterClass::Srv, 0),
    reg("table", RegisterClass::Srv, 1),
    reg("row_starts", RegisterClass::Srv, 2),
    reg("tile_offsets", RegisterClass::Srv, 3),
    reg("tile_items", RegisterClass::Srv, 4),
    reg("page", RegisterClass::Uav, 0),
    reg("status", RegisterClass::Uav, 1),
];

const FLOOR_DIV_PROBE: &[Register] = &[
    reg("pc", RegisterClass::Constants, 0),
    reg("pairs", RegisterClass::Srv, 0),
    reg("status", RegisterClass::Uav, 0),
    reg("quotients", RegisterClass::Uav, 1),
];

/// The registers of every kernel compiled to DXIL: the `Constants`
/// register first, then the SRVs, then the UAVs, each class from register
/// 0 in space 0 without a gap.  The `Constants` register is the root
/// constants, a root parameter of its own; the descriptor table holds the
/// SRVs and the UAVs alone, in this order, one range per class.  A
/// poisoned variant has its base kernel's registers.
pub(crate) const KERNEL_REGISTERS: &[(&str, &[Register])] = &[
    ("icc_clut", ICC_CLUT),
    ("blit_image", BLIT_IMAGE),
    ("page_raster", PAGE_RASTER),
    ("page_raster_paths", PAGE_RASTER_PATHS),
    ("page_raster_paths_poisoned", PAGE_RASTER_PATHS),
    ("floor_div_probe", FLOOR_DIV_PROBE),
    ("floor_div_probe_poisoned", FLOOR_DIV_PROBE),
];

/// The size of every kernel's constant buffer (its root constants) in
/// 32-bit words.
pub(crate) const KERNEL_CONSTANT_WORDS: &[(&str, u32)] = &[
    ("icc_clut", 3),
    ("blit_image", 10),
    ("page_raster", 14),
    ("page_raster_paths", 14),
    ("page_raster_paths_poisoned", 14),
    ("floor_div_probe", 2),
    ("floor_div_probe_poisoned", 2),
];
