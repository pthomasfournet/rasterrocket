/// The 32-bit words of a kernel's status buffer
/// (`crate::backend::kernel_status`); the macro `STATUS_WORDS`, which
/// `kernels/floor_div.slang` requires to match the words it writes.
pub(crate) const STATUS_WORDS: usize = 5;

/// The correction steps the outline raster's floor division
/// (`kernels/floor_div.slang`) takes in each direction.  Every Slang
/// kernel is compiled with it as the macro `FLOOR_DIV_STEPS`.
pub(crate) const FLOOR_DIV_STEPS: u32 = 16;

/// The status fault code of a floor division whose bounded corrections
/// left its remainder outside `[0, b)`; the macro `FAULT_FLOOR_DIV`.
pub(crate) const FAULT_FLOOR_DIV: u32 = 1;

/// The status fault code of a floor division whose remainder lies in
/// `[0, b)` but whose quotient's product with `b` wrapped modulo `2^64`;
/// the macro `FAULT_FLOOR_DIV_WRAP`.
pub(crate) const FAULT_FLOOR_DIV_WRAP: u32 = 2;
