//! Page rendering: operator dispatch → raster crate calls.

#[cfg(feature = "bundled-fonts")]
mod bundled_fonts;
pub(crate) mod color;
mod document_binding;
pub(crate) mod document_images;
pub(crate) mod encodings;
mod fallback;
pub(crate) mod font_cache;
pub(crate) mod gstate;
pub mod page;
pub(crate) mod substitutes;
mod system_fonts;
pub(crate) mod text;

pub use document_images::{DOCUMENT_IMAGE_BUDGET_BYTES, DocumentImages};
pub use font_cache::DocumentFonts;
pub use page::{
    CanvasBuffers, DevicePage, ImageFailure, PageDiagnostics, PageRenderer, PageView, RenderAbort,
};
#[cfg(feature = "vulkan")]
pub use page::{DeviceRecording, device_page_operators};
pub use pdf::ContentFailure;
