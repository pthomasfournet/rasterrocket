//! Lazy, zero-copy PDF file parser for the rasterrocket render pipeline.
//!
//! # Design
//!
//! * [`Document::open`] parses only the xref table — no objects are loaded.
//! * Objects are resolved on first access and cached (`Arc`-wrapped) for reuse.
//! * All byte scanning uses hand-rolled primitives in the private `lexer`
//!   submodule.

mod crypt;
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

pub use crypt::{OpenOptions, Password, msg_gate_declined};
pub use dictionary::Dictionary;
pub use document::{ContentFailure, Document, PageContent, Resolved, is_encrypted};
pub use error::PdfError;
pub use lexer::{decode_name, parse_number};
pub use linearization::LinearizationHints;
pub use object::{Object, ObjectId, Stream, StringFormat, parse_object};
pub use page_tree::descend_to_page_index;
pub use stream::{filter_full_name, inflate, is_decoded_filter};
