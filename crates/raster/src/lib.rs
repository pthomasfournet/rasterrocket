//! # raster
//!
//! Software rasterizer for PDF page content: path fill and stroke, glyph and
//! shading fills, tiling patterns, transparency groups and the compositing
//! pipe. No I/O and no PDF parsing; the interpreter drives it with device-space
//! geometry and resolved colours.
//!
//! ## Module layout
//! - [`types`] — raster-local enums and constants (re-exports [`color`] types)
//! - [`bitmap`] — [`Bitmap<P>`]
//! - [`path`] — [`Path`], [`PathBuilder`], Bezier flattening, stroke adjustment
//! - [`xpath`] — [`XPath`] edge table (flattened, matrix-transformed segments)
//! - `scanner` — [`XPathScanner`] and [`crate::ScanIterator`] (scanline span emission)
//! - [`clip`] — [`Clip`] (rect + arbitrary path clip stack)
//! - [`coverage`] — exact-area coverage ([`coverage::AreaRaster`], [`coverage::CoverageTable`])
//! - [`fill`](mod@fill) — non-zero and even-odd path fill, aliased and exact-area antialiased
//! - [`fnv`] — the 64-bit FNV-1a hash ([`fnv::Fnv1a`])
//! - [`stroke`](mod@stroke) — stroke-to-path conversion and stroking
//! - [`glyph`] — glyph bitmap blitting and fill
//! - [`shading`] — axial, radial, function and Gouraud shadings
//! - `tiling` — [`TiledPattern`]
//! - [`transparency`] — transparency groups
//! - `screen` — [`HalftoneScreen`] (Bayer / stochastic threshold matrix)
//! - [`pipe`] — compositing pipeline (simple, AA, general; blend modes;
//!   [`Pattern`] trait; [`TransferSet`])
//! - `simd` — runtime-dispatched SIMD kernels used by the pipe and fill

pub mod bitmap;
pub mod clip;
pub mod coverage;
#[cfg(test)]
mod depth16;
pub mod fill;
pub mod fnv;
pub mod glyph;
pub mod path;
#[cfg(test)]
mod pinned;
pub mod pipe;
#[cfg(test)]
mod proptests;
pub(crate) mod scanner;
pub(crate) mod screen;
pub mod shading;
pub(crate) mod simd;
pub mod stroke;
pub(crate) mod tiling;
pub mod transparency;
mod turn;
pub mod types;
pub mod xpath;

#[cfg(test)]
pub(crate) mod testutil;

pub use bitmap::{Bitmap, BitmapAllocError, BitmapBuffers, interleave_alpha, push_with_alpha};
pub use clip::{Clip, ClipResult, ClipRows, RegionKey, RowClip};
pub use fill::{eo_fill, fill, fill_area};
pub use glyph::{GlyphBitmap, blit_glyph, fill_glyph};
pub use path::{Path, PathBuilder, PathFlags, PathPoint, StrokeAdjustHint};
pub use pipe::{ChannelMask, GrayBlend, Pattern, PipeSrc, PipeState, TransferSet};
pub use scanner::iter::ScanIterator;
pub use scanner::{Intersect, ScannerTooLarge, XPathScanner};
pub use screen::HalftoneScreen;
pub use shading::ColorRamp;
pub use shading::axial::AxialPattern;
pub use shading::function::FunctionPattern;
pub use shading::gouraud::{GouraudVertex, gouraud_triangle_fill, gouraud_triangle_spans};
pub use shading::radial::RadialPattern;
pub use shading::shaded_fill;
pub use simd::{blend_solid_gray8, blend_solid_rgb8, composite_aa_rgb8_opaque, unpack_mono_row};
pub use stroke::{
    StrokeParams, aa_stroke_coverage, aa_stroke_outline, hairline_runs, stroke, stroke_outline,
};
pub use tiling::{CELL_PHASES, CellPhase, Lattice, TiledPattern};
pub use types::*;
pub use xpath::{PixelModel, XPath, XPathFlags, XPathSeg};
