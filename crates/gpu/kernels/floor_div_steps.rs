/// The correction steps the outline raster's floor division
/// (`kernels/floor_div.slang`) takes in each direction.  Every Slang
/// kernel is compiled with it as the macro `FLOOR_DIV_STEPS`.
pub(crate) const FLOOR_DIV_STEPS: u32 = 16;
