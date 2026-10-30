// Enable SVE2 intrinsics when the `nightly-sve2` feature is active.
// This is a no-op on stable (the cfg is false); the feature gate only fires
// on nightly when a caller explicitly opts in with `--features nightly-sve2`.
#![cfg_attr(
    all(target_arch = "aarch64", feature = "nightly-sve2"),
    feature(stdarch_aarch64_sve)
)]

//! # raster
//!
//! Software rasterizer for PDF page content: path fill and stroke, glyph and
//! shading fills, tiling patterns, transparency groups and the compositing
//! pipe. No I/O and no PDF parsing; the interpreter drives it with device-space
//! geometry and resolved colours.
//!
//! ## Module layout
//! - [`types`] — raster-local enums and constants (re-exports [`color`] types)
//! - [`bitmap`] — [`Bitmap<P>`] and [`AaBuf`]
//! - [`path`] — [`Path`], [`PathBuilder`], Bezier flattening, stroke adjustment
//! - [`xpath`] — [`XPath`] edge table (flattened, matrix-transformed segments)
//! - [`scanner`] — [`XPathScanner`] and [`crate::ScanIterator`] (scanline span emission)
//! - [`clip`] — [`Clip`] (rect + arbitrary path clip stack)
//! - [`fill`](mod@fill) — non-zero and even-odd path fill, aliased and 4× antialiased
//! - [`stroke`](mod@stroke) — stroke-to-path conversion and stroking
//! - [`glyph`] — glyph bitmap blitting and fill
//! - [`shading`] — axial, radial, function and Gouraud shadings
//! - [`tiling`] — [`TiledPattern`]
//! - [`transparency`] — transparency groups and soft masks
//! - [`screen`] — [`HalftoneScreen`] (Bayer / stochastic threshold matrix)
//! - [`pipe`] — compositing pipeline (simple, AA, general; blend modes;
//!   [`Pattern`] trait; [`TransferSet`])
//! - [`simd`] — runtime-dispatched SIMD kernels used by the pipe and fill

pub mod bitmap;
pub mod clip;
pub mod fill;
pub mod glyph;
pub mod path;
pub mod pipe;
pub mod scanner;
pub mod screen;
pub mod shading;
pub mod simd;
pub mod stroke;
pub mod tiling;
pub mod transparency;
pub mod types;
pub mod xpath;

#[cfg(test)]
pub(crate) mod testutil;

pub use bitmap::{AaBuf, Bitmap};
pub use clip::{Clip, ClipResult, ClipRows, RowClip};
pub use fill::{eo_fill, fill};
pub use glyph::{GlyphBitmap, blit_glyph, fill_glyph};
pub use path::{Path, PathBuilder, PathFlags, PathPoint, StrokeAdjustHint};
pub use pipe::{Pattern, PipeSrc, PipeState, TransferSet};
pub use scanner::iter::ScanIterator;
pub use scanner::{Intersect, XPathScanner};
pub use screen::HalftoneScreen;
pub use shading::ColorRamp;
pub use shading::axial::AxialPattern;
pub use shading::function::FunctionPattern;
pub use shading::gouraud::{GouraudVertex, gouraud_triangle_fill};
pub use shading::radial::RadialPattern;
pub use shading::shaded_fill;
pub use simd::{blend_solid_gray8, blend_solid_rgb8, composite_aa_rgb8_opaque, unpack_mono_row};
pub use stroke::{StrokeParams, stroke, stroke_outline};
pub use tiling::{CELL_PHASES, CellPhase, Lattice, TiledPattern};
pub use types::*;
pub use xpath::{XPath, XPathFlags, XPathSeg};
