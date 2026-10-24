//! Font programs, glyph rasterisation and the glyph cache for the
//! `rasterrocket` pipeline.
//!
//! Font programs (TrueType, OpenType, bare CFF, Type 1) are parsed by the
//! local `read-fonts` / `skrifa` forks; outlines are unhinted, converted to
//! [`raster::Path`] in device space and rasterised by an exact-area coverage
//! rasteriser.
//!
//! # Structure
//!
//! | Module | Contents |
//! |--------|---------|
//! | [`key`] | [`key::FaceId`] and [`key::GlyphKey`] — cache key types |
//! | [`bitmap`] | [`bitmap::GlyphBitmap`] — rasterised glyph (owned data) |
//! | [`program`] | [`program::FontProgram`] — format detection, charmaps, names, outlines |
//! | [`outline`] | [`outline::PathPen`] — outline commands → `raster::Path` |
//! | [`coverage`] | [`coverage::rasterize`] — `raster::Path` → coverage bitmap |
//! | [`cache`] | [`cache::GlyphCache`] — concurrent glyph cache |
//! | [`face`] | [`face::FontFace`] — a program at one text matrix |
//! | [`engine`] | [`engine::FontEngine`] — parses programs, creates faces |
//!
//! # Usage
//!
//! ```no_run
//! use rasterrocket_font::cache::GlyphCache;
//! use rasterrocket_font::engine::{FaceParams, FontEngine};
//!
//! let mut engine = FontEngine::new(true);
//! let cache = GlyphCache::new();
//!
//! // A 40 px face: em units → device pixels, y up.
//! let params = FaceParams {
//!     code_to_gid: vec![],
//!     mat: [40.0, 0.0, 0.0, 40.0],
//! };
//! let face = engine
//!     .load_file_face("/path/to/font.ttf", 0, params)
//!     .expect("face load failed");
//!
//! // Look up or render the glyph for character code 65.
//! let bitmap = face.make_glyph_cached(&cache, 65, 0);
//! # let _ = bitmap;
//! ```

pub mod bitmap;
pub mod cache;
pub mod coverage;
pub mod engine;
pub mod face;
pub mod key;
pub mod outline;
pub mod program;

#[cfg(test)]
mod tests;
