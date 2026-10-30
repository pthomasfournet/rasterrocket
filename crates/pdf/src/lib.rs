//! Lazy, zero-copy PDF file parser for the rasterrocket render pipeline.
//!
//! # Design
//!
//! * [`Document::open`] parses only the xref table — no objects are loaded.
//! * Objects are resolved on first access and cached (`Arc`-wrapped) for reuse.
//! * All byte scanning uses hand-rolled primitives in the private `lexer`
//!   submodule.

#![expect(
    clippy::redundant_pub_crate,
    reason = "every module is private, so rustc's `unreachable_pub` requires \
              crate-internal items to be `pub(crate)`, which this lint then \
              calls redundant; the two lints cannot both hold"
)]

mod decrypt;
mod derived;
mod dictionary;
mod document;
mod error;
mod lexer;
mod linearization;
mod madvise;
mod object;
mod objstm;
mod page_tree;
mod repair;
mod stream;
mod xref;

pub use decrypt::{DecryptGuard, msg_gate_declined};
pub use dictionary::Dictionary;
pub use document::Document;
pub use error::PdfError;
pub use lexer::parse_number;
pub use linearization::LinearizationHints;
pub use object::{Object, ObjectId, Stream, StringFormat, parse_object};
pub use page_tree::descend_to_page_index;
pub use stream::inflate;
