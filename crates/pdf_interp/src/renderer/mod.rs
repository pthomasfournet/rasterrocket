//! Page rendering: operator dispatch → raster crate calls.

pub mod color;
pub mod font_cache;
pub mod gstate;
pub mod page;
pub mod text;

pub use font_cache::DocumentFonts;
pub use page::{ImageFailure, PageDiagnostics, PageRenderer};
