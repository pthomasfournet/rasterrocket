//! Page rendering: operator dispatch → raster crate calls.

#[cfg(feature = "bundled-fonts")]
mod bundled_fonts;
pub(crate) mod color;
pub(crate) mod encodings;
mod fallback;
pub(crate) mod font_cache;
pub(crate) mod gstate;
pub mod page;
pub(crate) mod substitutes;
mod system_fonts;
pub(crate) mod text;

pub use font_cache::DocumentFonts;
pub use page::{ImageFailure, PageDiagnostics, PageRenderer, PageView, RenderAbort};
pub use pdf::ContentFailure;
