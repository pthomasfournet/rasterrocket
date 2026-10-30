//! The top-level [`Document`] type: lazy PDF reader.
//!
//! Open a PDF with [`Document::open`].  Objects are resolved on demand from the
//! file bytes — the constructor only parses the xref table and trailer.  Files
//! of at least [`MMAP_MIN_LEN`] bytes are memory-mapped; smaller files are
//! read into memory.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use memmap2::Mmap;

use crate::{
    crypt::{OpenOptions, Password, SecurityHandler},
    derived::DerivedCache,
    dictionary::Dictionary,
    error::PdfError,
    linearization::LinearizationHints,
    object::{Object, ObjectId, parse_indirect_object},
    objstm::ObjStmCache,
    repair::scan_objects,
    stream::decode_stream,
    xref::{XrefEntry, XrefTable, read_xref},
};

// ── Limits ───────────────────────────────────────────────────────────────────

/// Aggregate cap on a page's concatenated, decompressed content streams.
///
/// A `/Contents` array of many refs (each individually bounded by the
/// per-stream decompression limit in `stream.rs`) can otherwise sum without
/// bound. This cap mirrors the per-stream `MAX_DECOMPRESSED` discipline.
const MAX_PAGE_CONTENT: usize = 512 * 1024 * 1024; // 512 MiB

/// Smallest file [`Document::open`] memory-maps; shorter files are read into
/// an owned buffer.
///
/// Opening touches only the trailer, the xref sections and the objects they
/// lead to, so a mapping costs a few page faults while a read copies the whole
/// file.  Below this length the copy is cheaper than setting up the mapping,
/// and an owned buffer is immune to the file being modified or truncated
/// while open (see [`map_file`]).
const MMAP_MIN_LEN: u64 = 256 * 1024;

/// The decoded content of a page, as returned by [`Document::page_content`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PageContent {
    /// The decoded `/Contents` elements, concatenated in order.
    pub bytes: Vec<u8>,
    /// The `/Contents` elements that contributed no bytes, in order.
    pub failures: Vec<ContentFailure>,
}

/// A `/Contents` element skipped because it holds no decodable content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentFailure {
    /// Object number of the element.
    pub object: u32,
    /// Why the element was skipped: a decode error, or the type of the
    /// object found where a stream was expected.
    pub reason: String,
}

impl std::fmt::Display for ContentFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "content stream obj {}: {}", self.object, self.reason)
    }
}

/// Collect the indirect references from a `/Contents` array.
///
/// Shared by the indirect-ref-to-array and direct-array arms of
/// [`Document::get_page_content`] so the extraction (and its diagnostics)
/// exist in exactly one place.  ISO 32000-2 §7.7.3.3 says every element of a
/// `/Contents` array is an indirect reference to a content stream, so any
/// element that is *not* a reference (an inline object, a nested array, a
/// number) cannot contribute renderable bytes.  Silently filtering such an
/// element is the v1 partial-blank failure mode, so each one is logged before
/// it is dropped; `page_id` is threaded purely so the message names the page.
fn collect_content_refs(arr: &[Object], page_id: ObjectId) -> Vec<ObjectId> {
    arr.iter()
        .filter_map(|el| {
            el.as_reference().or_else(|| {
                log::warn!(
                    "get_page_content: page {} /Contents array element is {}, \
                     not an indirect reference — skipping (page may be missing content)",
                    page_id.0,
                    el.enum_variant()
                );
                None
            })
        })
        .collect()
}

/// Whether a stream with dictionary `dict` is an object stream: its `/Type`
/// is `/ObjStm`, or it has no `/Type` and both `/N` and `/First` are
/// integers.  A stream of any other type is never decoded as a container.
fn is_object_stream(dict: &Dictionary) -> bool {
    dict.get(b"Type").and_then(Object::as_name).map_or_else(
        || {
            [b"N".as_slice(), b"First".as_slice()]
                .iter()
                .all(|key| dict.get(key).and_then(Object::as_i64).is_some())
        },
        |ty| ty == b"ObjStm",
    )
}

// ── Document ─────────────────────────────────────────────────────────────────

/// A lazily-parsed PDF document backed by a memory-mapped file or an owned
/// buffer.
pub struct Document {
    /// The file bytes.  `Arc` so callers can hold byte slices.
    data: Arc<MmapOrVec>,
    /// Position in `data` that the cross-reference offsets are relative to:
    /// 0, or the `%PDF-` header's position when bytes precede the header
    /// and the offsets were written without them.
    base: usize,
    xref: XrefTable,
    /// Per-object parse cache.  `None` = not yet resolved.
    cache: Mutex<HashMap<u32, Arc<Object>>>,
    /// Object-stream decompression cache (shared across threads).
    objstm: ObjStmCache,
    /// Catalogue object ID, resolved once from the trailer.
    catalog_id: OnceLock<ObjectId>,
    /// `/Pages` root reference, resolved once from the catalog.  Cached
    /// because every page-tree descent reads it.
    pages_root_id: OnceLock<ObjectId>,
    /// `/Pages /Count`, memoised on first read.  Falls back to the eager
    /// walk on malformed catalogs.
    page_count_cache: OnceLock<u32>,
    /// Cached linearization hints, parsed lazily on first access.
    lin_hints: OnceLock<Option<LinearizationHints>>,
    /// Object headers found by scanning the whole file, built the first time
    /// an xref entry does not lead to its object.  `None` when the scan
    /// exceeded its sanity limit.
    scanned_offsets: OnceLock<Option<HashMap<u32, XrefEntry>>>,
    /// Values derived from objects by higher layers ([`Self::derived`]).
    derived: DerivedCache,
    /// The authenticated security handler of an encrypted document; `None`
    /// for a document whose trailer has no `/Encrypt`.
    crypt: Option<SecurityHandler>,
    /// Object number of the indirect `/Encrypt` dictionary, whose strings
    /// are never encrypted.
    encrypt_id: Option<u32>,
}

/// Memory source: a mapping for files of at least [`MMAP_MIN_LEN`] bytes, an
/// owned buffer for shorter files and for bytes supplied by the caller.
enum MmapOrVec {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl MmapOrVec {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m,
            Self::Owned(v) => v,
        }
    }
}

/// Map `file` read-only into the address space.
///
/// The mapping shares the file's page-cache pages, so it is only as stable
/// as the file: a write by another process changes bytes the parser has
/// already borrowed, and a truncation makes any later access to a page past
/// the new end of file raise `SIGBUS`, which terminates the process.  The
/// parser never writes the file; callers must not modify or truncate a file
/// while a [`Document`] opened from it is alive.
fn map_file(file: &std::fs::File) -> std::io::Result<Mmap> {
    // SAFETY: `Mmap::map` is unsafe because the returned slice aliases
    // memory that another process can change or unmap by writing to or
    // truncating the file; the contract above makes that the caller's
    // obligation.  The mapping is read-only and this crate never writes
    // through it.
    #[expect(
        unsafe_code,
        reason = "a file mapping cannot be created without `unsafe`; the \
                  external-modification hazard is documented on `map_file`"
    )]
    unsafe {
        Mmap::map(file)
    }
}

/// The bytes of the file at `path`: a mapping for a file of at least
/// [`MMAP_MIN_LEN`] bytes, an owned buffer for a shorter one.
fn load_file(path: &Path) -> std::io::Result<MmapOrVec> {
    use std::io::Read as _;

    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len < MMAP_MIN_LEN {
        // `len < MMAP_MIN_LEN`, so the capacity conversion cannot fail.
        let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
        let _ = file.read_to_end(&mut bytes)?;
        Ok(MmapOrVec::Owned(bytes))
    } else {
        crate::madvise::advise_random(&file);
        Ok(MmapOrVec::Mapped(map_file(&file)?))
    }
}

/// The cross-reference table and the header position of `data`.
fn parse_structure(data: &[u8]) -> Result<(XrefTable, usize), PdfError> {
    let header = validate_pdf_input(data)?;
    read_xref(data, header).map_err(|e| match e {
        // A `%PDF-` header is present (validate_pdf_input passed) but the
        // xref could neither be parsed nor reconstructed: the file is
        // truncated or corrupt.  Replace the low-level "startxref not
        // found" with an accurate, operator-actionable diagnosis.
        PdfError::BadXref(detail) => PdfError::MissingXref(detail),
        other => other,
    })
}

/// True when the PDF at `path` carries an `/Encrypt` entry in its trailer.
///
/// A structural probe: the cross-reference table and trailer are parsed,
/// no object is read and no decryption is attempted, so the answer does
/// not depend on any password.
///
/// # Errors
/// [`PdfError::Io`] if the file cannot be read, and the structural errors of
/// [`Document::open`] (empty, not a PDF, no cross-reference table).
pub fn is_encrypted(path: &Path) -> Result<bool, PdfError> {
    let data = load_file(path)?;
    let (xref, _) = parse_structure(data.as_bytes())?;
    Ok(xref.trailer.contains_key(b"Encrypt"))
}

impl Document {
    /// Open a PDF file at `path`.
    ///
    /// Reads and parses only the xref table(s) and trailer.  Individual objects
    /// are parsed on first access and cached for subsequent calls.
    ///
    /// A file of at least [`MMAP_MIN_LEN`] bytes is memory-mapped, so it must
    /// not be modified or truncated while the returned `Document` is alive
    /// (see [`map_file`]); a shorter file is copied into memory.
    ///
    /// # Errors
    /// [`PdfError::Io`] if the file cannot be opened, read or mapped;
    /// otherwise the errors of [`Self::from_bytes_owned`].
    pub fn open(path: &Path) -> Result<Self, PdfError> {
        Self::open_with(path, &OpenOptions::default())
    }

    /// Open `path` with `options` deciding how an encrypted document is
    /// treated.
    ///
    /// An encrypted document is refused unless
    /// [`OpenOptions::decrypt_authorized`] is set; with it, the empty user
    /// password and then [`OpenOptions::password`] unlock the file key, and
    /// every string and stream is decrypted in process as its object is
    /// parsed.  An unencrypted document ignores `options`.
    ///
    /// # Errors
    /// The errors of [`Self::open`], and [`PdfError::EncryptedDocument`]
    /// when the document is encrypted and decryption is not authorised,
    /// uses a security handler other than `/Standard`, or needs a password
    /// other than the one given.
    pub fn open_with(path: &Path, options: &OpenOptions) -> Result<Self, PdfError> {
        Self::from_source(Arc::new(load_file(path)?), options)
    }

    /// Construct from PDF bytes already in memory.
    ///
    /// # Errors
    /// [`PdfError::EmptyInput`] or [`PdfError::NotPdf`] when the bytes are
    /// not a PDF, [`PdfError::MissingXref`] when no cross-reference table
    /// can be parsed or reconstructed, and [`PdfError::EncryptedDocument`]
    /// when the document is encrypted (see [`Self::from_bytes_with`]).
    pub fn from_bytes_owned(bytes: Vec<u8>) -> Result<Self, PdfError> {
        Self::from_bytes_with(bytes, &OpenOptions::default())
    }

    /// [`Self::from_bytes_owned`] with `options` deciding how an encrypted
    /// document is treated, as for [`Self::open_with`].
    ///
    /// # Errors
    /// The errors of [`Self::open_with`] other than I/O.
    pub fn from_bytes_with(bytes: Vec<u8>, options: &OpenOptions) -> Result<Self, PdfError> {
        Self::from_source(Arc::new(MmapOrVec::Owned(bytes)), options)
    }

    fn from_source(data: Arc<MmapOrVec>, options: &OpenOptions) -> Result<Self, PdfError> {
        let (xref, base) = parse_structure(data.as_bytes())?;
        let mut doc = Self {
            data,
            base,
            xref,
            cache: Mutex::new(HashMap::new()),
            objstm: ObjStmCache::new(),
            catalog_id: OnceLock::new(),
            pages_root_id: OnceLock::new(),
            page_count_cache: OnceLock::new(),
            lin_hints: OnceLock::new(),
            scanned_offsets: OnceLock::new(),
            derived: DerivedCache::default(),
            crypt: None,
            encrypt_id: None,
        };
        if let Some(encrypt) = doc.xref.trailer.get(b"Encrypt").cloned() {
            if !options.decrypt_authorized {
                return Err(PdfError::EncryptedDocument(
                    crate::crypt::msg_gate_declined(),
                ));
            }
            doc.install_security_handler(
                &encrypt,
                options.password.as_ref().map(Password::as_str),
            )?;
        }
        Ok(doc)
    }

    /// Build the document's [`SecurityHandler`] from the trailer's
    /// `/Encrypt` value and the trailer `/ID`.
    ///
    /// Runs before `crypt` is set, so the `/Encrypt` dictionary is parsed
    /// (and cached) with its strings untouched, as the specification
    /// requires.
    fn install_security_handler(
        &mut self,
        encrypt: &Object,
        password: Option<&str>,
    ) -> Result<(), PdfError> {
        let dict = match encrypt {
            Object::Reference(id) => {
                self.encrypt_id = Some(id.0);
                self.get_dict(*id).map_err(|e| {
                    PdfError::EncryptedDocument(format!(
                        "document is encrypted but its /Encrypt dictionary cannot be read: {e}"
                    ))
                })?
            }
            Object::Dictionary(d) => Arc::new(d.clone()),
            other => {
                return Err(PdfError::EncryptedDocument(format!(
                    "document is encrypted but /Encrypt is {}, not a dictionary",
                    other.enum_variant()
                )));
            }
        };
        let id0 = self
            .xref
            .trailer
            .get(b"ID")
            .and_then(Object::as_array)
            .and_then(|ids| ids.first())
            .and_then(Object::as_str)
            .map(<[u8]>::to_vec)
            .unwrap_or_default();
        self.crypt = Some(SecurityHandler::open(&dict, &id0, password)?);
        Ok(())
    }

    // ── Object resolution ────────────────────────────────────────────────────

    /// Resolve an indirect reference and return a shared reference to its value.
    ///
    /// First call parses the object from the file; subsequent calls return the
    /// cached value.
    ///
    /// # Errors
    /// [`PdfError::BadObject`] when neither the xref entry for `id` nor the
    /// header scan of the file leads to a parseable object `id`.
    pub fn get_object(&self, id: ObjectId) -> Result<Arc<Object>, PdfError> {
        {
            // Poison recovery: a previous panic while holding the lock left the
            // mutex poisoned. The cache is just a memo of immutable parses, so
            // the inner state is safe to keep using.
            let guard = self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(obj) = guard.get(&id.0) {
                return Ok(Arc::clone(obj));
            }
        }

        let mut obj = self.parse_object_uncached(id)?;
        if let Object::Stream(stream) = &mut obj {
            self.inline_filter_references(&mut stream.dict);
        }
        let arc = Arc::new(obj);
        let _ = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id.0, Arc::clone(&arc));
        Ok(arc)
    }

    /// The value of type `T` derived from object `id`, computed by `make` on
    /// the first request and shared by every later one for the lifetime of
    /// the document.  A `None` from `make` is memoised as well.
    ///
    /// For values that are expensive to compute and depend only on the
    /// object, such as a decoded dictionary that many streams refer to.
    pub fn derived<T: std::any::Any + Send + Sync>(
        &self,
        id: ObjectId,
        make: impl FnOnce() -> Option<T>,
    ) -> Option<Arc<T>> {
        self.derived.get_or_derive(id, make)
    }

    /// Replace indirect references in a stream's `/Filter` and `/DecodeParms`
    /// (the whole value, or elements of an array value) with the objects they
    /// name, so every consumer of the dictionary sees direct values.
    ///
    /// Referents are read with the uncached parser and accepted only as a
    /// name, array, dictionary or null, which no stream can be: a reference
    /// back to a stream (even this one) stays unresolved, without recursion.
    fn inline_filter_references(&self, dict: &mut Dictionary) {
        let direct = |id: ObjectId| match self.parse_object_uncached(id) {
            Ok(o @ (Object::Name(_) | Object::Array(_) | Object::Dictionary(_) | Object::Null)) => {
                Some(o)
            }
            _ => None,
        };
        for key in [b"Filter".as_slice(), b"DecodeParms".as_slice()] {
            let Some(value) = dict.get(key) else {
                continue;
            };
            let top = match value {
                Object::Reference(id) => match direct(*id) {
                    Some(o) => o,
                    None => continue,
                },
                other => other.clone(),
            };
            let resolved = match top {
                Object::Array(items) if items.iter().any(|o| matches!(o, Object::Reference(_))) => {
                    Object::Array(
                        items
                            .into_iter()
                            .map(|o| match o {
                                Object::Reference(id) => direct(id).unwrap_or(Object::Null),
                                other => other,
                            })
                            .collect(),
                    )
                }
                other => other,
            };
            if dict.get(key) != Some(&resolved) {
                dict.set(key, resolved);
            }
        }
    }

    /// Resolve a chain of indirect references until a non-Reference object is
    /// reached, following at most 32 references.
    ///
    /// # Errors
    /// The errors of [`Self::get_object`], and [`PdfError::BadObject`] when
    /// the chain is longer than 32 references.
    pub fn resolve(&self, obj: &Object) -> Result<Arc<Object>, PdfError> {
        let mut current = obj.clone();
        for _ in 0..32 {
            match current {
                Object::Reference(id) => current = (*self.get_object(id)?).clone(),
                other => return Ok(Arc::new(other)),
            }
        }
        Err(PdfError::BadObject {
            id: 0,
            detail: "infinite reference chain".into(),
        })
    }

    fn parse_object_uncached(&self, id: ObjectId) -> Result<Object, PdfError> {
        // A referenced object the xref does not list, or lists as free, is
        // what the header scan finds for its number, or nothing.
        let Some(entry) = self.xref.get(id.0) else {
            return self
                .recover_object(id.0, None)
                .ok_or_else(|| PdfError::BadObject {
                    id: id.0,
                    detail: "not in xref table".into(),
                });
        };

        match entry {
            XrefEntry::Direct { offset, .. } => {
                match self.parse_direct(offset) {
                    Some((found, obj)) if found.0 == id.0 => Ok(obj),
                    // The entry does not lead to object `id`: its offset is
                    // stale, points inside another object, or names a
                    // different object.  The object is what the header scan
                    // finds for its number, or nothing; what the entry led
                    // to is never returned under this number.
                    parsed => {
                        self.recover_object(id.0, Some(entry))
                            .ok_or_else(|| PdfError::BadObject {
                                id: id.0,
                                detail: parsed.map_or_else(
                                    || format!("parse failed at offset {offset}"),
                                    |(found, _)| {
                                        format!("xref offset {offset} holds object {}", found.0)
                                    },
                                ),
                            })
                    }
                }
            }
            // An object the container does not hold, or holds in an
            // unparseable form, or a container that is no object stream,
            // may still have a body elsewhere in the file.
            XrefEntry::InObjStm { container, index } => self
                .parse_compressed(container, index, id.0)
                .or_else(|e| self.recover_object(id.0, Some(entry)).ok_or(e)),
            XrefEntry::Free => {
                self.recover_object(id.0, Some(entry))
                    .ok_or_else(|| PdfError::BadObject {
                        id: id.0,
                        detail: "free in xref table".into(),
                    })
            }
        }
    }

    /// Object `num` at directory position `index` of object stream
    /// `container`, the position being a hint verified against the
    /// directory.
    ///
    /// No stream, an object stream included, is stored in an object stream
    /// (ISO 32000-2 §7.5.7): a container whose own entry is not direct is
    /// read from the header the scan finds for it.  The container must be an object stream by
    /// [`is_object_stream`]; any other stream is not decoded.
    fn parse_compressed(&self, container: u32, index: u32, num: u32) -> Result<Object, PdfError> {
        let container_obj = match self.xref.get(container) {
            Some(XrefEntry::Direct { .. }) => self.get_object((container, 0))?,
            _ => self
                .recover_direct(container, None)
                .map(Arc::new)
                .ok_or_else(|| PdfError::BadObject {
                    id: container,
                    detail: "object stream has no direct xref entry or header".into(),
                })?,
        };
        let Object::Stream(stream) = container_obj.as_ref() else {
            return Err(PdfError::BadObject {
                id: container,
                detail: "expected Stream for ObjStm container".into(),
            });
        };
        if !is_object_stream(&stream.dict) {
            return Err(PdfError::BadObject {
                id: container,
                detail: "container stream is not an ObjStm".into(),
            });
        }
        self.objstm
            .get_or_parse(container, index, num, &stream.content, &stream.dict)
    }

    /// Parse the indirect object whose header starts at `offset`.
    ///
    /// In an encrypted document the object's strings and streams are
    /// decrypted here, under the number and generation of its header; the
    /// `/Encrypt` dictionary itself is left as written.
    fn parse_direct(&self, offset: u64) -> Option<(ObjectId, Object)> {
        let (found, mut obj) = usize::try_from(offset)
            .ok()
            .and_then(|offset| parse_indirect_object(self.body(), offset))?;
        if let Some(crypt) = &self.crypt
            && self.encrypt_id != Some(found.0)
        {
            crypt.decrypt_object(found, &mut obj);
        }
        Some((found, obj))
    }

    /// The file from `base` on: the bytes cross-reference offsets index.
    fn body(&self) -> &[u8] {
        &self.data.as_bytes()[self.base..]
    }

    /// The entries a whole-file header scan finds, computed on first use.
    fn scanned_entries(&self) -> Option<&HashMap<u32, XrefEntry>> {
        self.scanned_offsets
            .get_or_init(|| scan_objects(self.body()).ok().map(|scan| scan.entries))
            .as_ref()
    }

    /// Object `num` as the whole-file scan locates it, when that location
    /// differs from `failed`, the xref entry that did not yield the object.
    ///
    /// A direct location is a header that must parse as object `num`; a
    /// compressed one is an object stream whose own scanned entry is
    /// direct, so recovery never descends through more than one stream.
    fn recover_object(&self, num: u32, failed: Option<XrefEntry>) -> Option<Object> {
        let scanned = self.scanned_entries()?;
        let obj = match (scanned.get(&num).copied()?, failed) {
            (XrefEntry::Direct { .. }, _) => self.recover_direct(num, failed)?,
            (
                XrefEntry::InObjStm { container, .. },
                Some(XrefEntry::InObjStm {
                    container: tried, ..
                }),
            ) if container == tried => return None,
            (XrefEntry::InObjStm { container, index }, _) => {
                if !matches!(scanned.get(&container), Some(XrefEntry::Direct { .. })) {
                    return None;
                }
                let obj = self.parse_compressed(container, index, num).ok()?;
                log::debug!("pdf: object {num} recovered from object stream {container}");
                obj
            }
            (XrefEntry::Free, _) => return None,
        };
        Some(obj)
    }

    /// Object `num` parsed from the header a whole-file scan finds for it,
    /// when that header is not at the offset of `failed` and parses as
    /// object `num`.
    fn recover_direct(&self, num: u32, failed: Option<XrefEntry>) -> Option<Object> {
        let Some(XrefEntry::Direct { offset, .. }) = self.scanned_entries()?.get(&num).copied()
        else {
            return None;
        };
        if matches!(failed, Some(XrefEntry::Direct { offset: tried, .. }) if tried == offset) {
            return None;
        }
        let (found, obj) = self.parse_direct(offset)?;
        (found.0 == num).then(|| {
            log::debug!("pdf: object {num} recovered at offset {offset} (xref entry: {failed:?})");
            obj
        })
    }

    // ── Typed dictionary accessors ────────────────────────────────────────────

    /// Resolve `id` and return it as a dictionary.
    ///
    /// Currently clones the dictionary out of the cached `Arc<Object>` because
    /// stable Rust has no zero-copy way to project an `Arc<Object>` to an
    /// `Arc<Dictionary>` (the dict lives inside the enum variant). Hot
    /// repeat-lookup callers (e.g. per-page resource resolution) should cache
    /// the result themselves; the dict is hashed and small (~tens of entries)
    /// so the clone cost is bounded.
    ///
    /// # Errors
    /// The errors of [`Self::get_object`], and [`PdfError::BadObject`] when
    /// the object is neither a dictionary nor a stream.
    pub fn get_dict(&self, id: ObjectId) -> Result<Arc<Dictionary>, PdfError> {
        let obj = self.get_object(id)?;
        match obj.as_ref() {
            Object::Dictionary(d) => Ok(Arc::new(d.clone())),
            Object::Stream(s) => Ok(Arc::new(s.dict.clone())),
            _ => Err(PdfError::BadObject {
                id: id.0,
                detail: format!("expected Dictionary or Stream, got {}", obj.enum_variant()),
            }),
        }
    }

    /// Alias for [`Self::get_dict`].
    ///
    /// # Errors
    /// The errors of [`Self::get_dict`].
    pub fn get_dictionary(&self, id: ObjectId) -> Result<Arc<Dictionary>, PdfError> {
        self.get_dict(id)
    }

    /// Resolve `id` and return the underlying `Arc<Object>` if it is a
    /// dictionary or stream-with-dict.
    ///
    /// Unlike [`Self::get_dict`], this does NOT clone the dictionary —
    /// the caller borrows directly from the cached `Arc<Object>`.  Use
    /// this on hot paths (e.g., page-tree descent) where every saved
    /// `HashMap<Vec<u8>, Object>` clone matters.  Callers must accept
    /// that the borrow lives only as long as the returned `Arc`.
    ///
    /// # Errors
    /// The errors of [`Self::get_dict`].
    pub fn get_dict_arc(&self, id: ObjectId) -> Result<Arc<Object>, PdfError> {
        let obj = self.get_object(id)?;
        match obj.as_ref() {
            Object::Dictionary(_) | Object::Stream(_) => Ok(obj),
            _ => Err(PdfError::BadObject {
                id: id.0,
                detail: format!("expected Dictionary or Stream, got {}", obj.enum_variant()),
            }),
        }
    }

    // ── Document structure ────────────────────────────────────────────────────

    /// Return the document catalogue dictionary.
    ///
    /// # Errors
    /// [`PdfError::MissingKey`] or [`PdfError::BadObject`] when the trailer
    /// `/Root` is absent or not a reference, and the errors of
    /// [`Self::get_dict`].
    pub fn catalog(&self) -> Result<Arc<Dictionary>, PdfError> {
        let id = self.catalog_id()?;
        self.get_dict(id)
    }

    fn catalog_id(&self) -> Result<ObjectId, PdfError> {
        if let Some(id) = self.catalog_id.get() {
            return Ok(*id);
        }
        let root = self
            .xref
            .trailer
            .get(b"Root")
            .ok_or(PdfError::MissingKey("Root"))?;
        let id = match root {
            Object::Reference(id) => *id,
            other => {
                // Some PDFs embed the catalog inline in the trailer.
                return Err(PdfError::BadObject {
                    id: 0,
                    detail: format!("trailer /Root is not a reference: {other:?}"),
                });
            }
        };
        let _ = self.catalog_id.set(id);
        Ok(id)
    }

    /// Return parsed linearization hints, or `None` if the document is
    /// not linearized.  Parsed once on first call and cached.
    ///
    /// `LinearizationHints` is `Copy` (24 bytes), so callers receive an
    /// owned value rather than a borrow that would tie up the document
    /// for its lifetime.
    ///
    /// # Errors
    /// Currently never errors — `try_load` collapses every failure mode
    /// (missing object 1, malformed dict, missing keys, invalid offsets)
    /// into `Ok(None)`.  The `Result` return type is reserved for the
    /// future hint-stream parser, which will gain real failure modes;
    /// the call site can switch to `OnceLock::get_or_try_init` when
    /// stable (rust-lang/rust#109737).
    pub fn linearization_hints(&self) -> Result<Option<LinearizationHints>, PdfError> {
        Ok(*self
            .lin_hints
            .get_or_init(|| LinearizationHints::try_load(self).ok().flatten()))
    }

    /// Raw byte view of the underlying file.  Used by hint-table parsing
    /// and by callers that want to hash the document content without
    /// re-reading the file.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    /// True when the document is encrypted, i.e. it was opened through its
    /// Standard security handler and its objects are decrypted as they are
    /// read.
    #[must_use]
    pub const fn is_encrypted(&self) -> bool {
        self.crypt.is_some()
    }

    /// Return the number of pages in the document by walking the page tree,
    /// saturating at `u32::MAX`.
    #[must_use]
    pub fn page_count(&self) -> u32 {
        u32::try_from(self.get_pages().count()).unwrap_or(u32::MAX)
    }

    /// Return the number of pages without walking the page tree.
    ///
    /// Reads `/Pages /Count` directly from the catalog and memoises it on
    /// first call.  Falls back to the eager [`Self::page_count`] (full tree
    /// walk) if the catalog is malformed.
    ///
    /// Cached, so repeated calls are a relaxed-load away from the answer.
    #[must_use]
    pub fn page_count_fast(&self) -> u32 {
        *self
            .page_count_cache
            .get_or_init(|| self.read_page_count_uncached())
    }

    /// Inner page-count read.  Used by `page_count_fast` (memoised) and
    /// the eager-walk fallback inside the page-tree descent.
    #[must_use]
    fn read_page_count_uncached(&self) -> u32 {
        // Linearized PDFs declare /N up front; trust it without descending
        // into the catalog dict.
        if let Ok(Some(hints)) = self.linearization_hints() {
            return hints.page_count;
        }
        let Ok(pages_root) = self.pages_root_id() else {
            return self.page_count();
        };
        let Ok(dict) = self.get_dict(pages_root) else {
            return self.page_count();
        };
        dict.get(b"Count")
            .and_then(Object::as_u32)
            .unwrap_or_else(|| self.page_count())
    }

    /// Return the catalog's `/Pages` reference, memoised on first call.
    ///
    /// `pages_root` is invariant for the document, so caching it avoids
    /// re-cloning the catalog dictionary on every page-tree descent.
    ///
    /// # Errors
    /// The errors of [`Self::catalog`], and [`PdfError::MissingKey`] when the
    /// catalog has no `/Pages` reference.
    pub fn pages_root_id(&self) -> Result<ObjectId, PdfError> {
        if let Some(id) = self.pages_root_id.get() {
            return Ok(*id);
        }
        let catalog = self.catalog()?;
        let id = catalog
            .get(b"Pages")
            .and_then(Object::as_reference)
            .ok_or(PdfError::MissingKey("Pages"))?;
        let _ = self.pages_root_id.set(id);
        Ok(id)
    }

    /// Resolve a 0-based page index to its `ObjectId` via logarithmic descent.
    ///
    /// See [`crate::descend_to_page_index`] for the full descent contract.
    ///
    /// # Errors
    /// The errors of [`crate::descend_to_page_index`].
    pub fn get_page(&self, idx: u32) -> Result<ObjectId, PdfError> {
        crate::page_tree::descend_to_page_index(self, idx)
    }

    /// Iterate over all pages in document order, yielding `(1-based page number, ObjectId)`.
    pub fn get_pages(&self) -> impl Iterator<Item = (u32, ObjectId)> + '_ {
        PageIter::new(self)
    }

    /// Return the decoded content stream bytes for a page.
    ///
    /// Equivalent to [`Self::page_content`] without the record of skipped
    /// elements.
    ///
    /// # Errors
    /// The errors of [`Self::page_content`].
    pub fn get_page_content(&self, page_id: ObjectId) -> Result<Vec<u8>, PdfError> {
        self.page_content(page_id).map(|content| content.bytes)
    }

    /// Return the decoded content stream bytes for a page, with the
    /// `/Contents` elements that contributed nothing.
    ///
    /// A `/Contents` element that is not a stream, or that cannot be decoded,
    /// is skipped with a warning and listed in [`PageContent::failures`].  A
    /// reference to an undefined object is the null object (ISO 32000-1
    /// §7.3.10) and is skipped without being listed.
    ///
    /// # Errors
    /// The errors of [`Self::get_object`], [`PdfError::BadObject`] when the
    /// page is not a dictionary, and [`PdfError::DecodeFailed`] when the
    /// decoded total exceeds the aggregate size cap or no content stream can
    /// be decoded.
    pub fn page_content(&self, page_id: ObjectId) -> Result<PageContent, PdfError> {
        let page = self.get_object(page_id)?;
        let page_dict = page.as_dict().ok_or_else(|| PdfError::BadObject {
            id: page_id.0,
            detail: "page is not a dict".into(),
        })?;

        let contents = match page_dict.get(b"Contents") {
            None => return Ok(PageContent::default()),
            Some(obj) => obj.clone(),
        };

        // /Contents (ISO 32000-2 §7.7.3.3) is either a single content stream
        // or an array of streams whose decoded bytes are concatenated.  Either
        // form may itself be reached through an indirect reference, so a
        // reference can resolve to a *stream* OR to an *array of stream
        // references* — modern producers (LibreOffice, many imposition tools)
        // emit `/Contents N 0 R` where object N is `[a 0 R b 0 R c 0 R]`.
        // The previous code treated such a reference as a single stream;
        // `get_object` returned the Array, the `Object::Stream` match failed,
        // and the page rendered blank.  Resolve one indirection level so a
        // ref-to-array expands into its element references.
        //
        // This expansion is intentionally a single, non-recursive pass: the
        // spec permits exactly one array of *direct stream references*, never
        // an array nested inside an array, so a flat collection both matches
        // the spec and makes a cyclic/self-referential `/Contents` ring
        // (an element ref pointing back at the array object) structurally
        // impossible to loop on — each `get_object` is visited at most once
        // and non-streams are diagnosed and skipped in the decode loop below.
        let refs: Vec<ObjectId> = match &contents {
            Object::Reference(id) => match self.get_object(*id).as_deref() {
                Ok(Object::Array(arr)) => collect_content_refs(arr, page_id),
                // Stream (or anything else, or unresolvable): keep the single
                // ref; the decode loop below filters and diagnoses it.
                _ => vec![*id],
            },
            Object::Array(arr) => collect_content_refs(arr, page_id),
            _ => return Ok(PageContent::default()),
        };

        let mut out = Vec::new();
        let mut failures = Vec::new();
        let mut decoded_any = false;
        let mut decode_failure = None;
        for r in refs {
            // A reference to an undefined object is the null object
            // (ISO 32000-1 §7.3.10): not content, and not an error.
            let stream_obj = match self.get_object(r) {
                Ok(obj) => obj,
                Err(e) => {
                    log::warn!(
                        "get_page_content: page {} /Contents element obj {} is missing \
                         ({e}) — skipping (page may be missing content)",
                        page_id.0,
                        r.0
                    );
                    continue;
                }
            };
            let Object::Stream(s) = stream_obj.as_ref() else {
                // A `/Contents` array element that does not resolve to a
                // stream (a dict, null, number, or — non-conformant — a
                // nested array) is not renderable content.  Dropping it
                // silently is the v1 blank/partial-page failure: the page
                // would render with missing operators and EXIT 0.  Surface
                // it loudly and continue with the streams that *are* valid
                // so the page degrades gracefully instead of vanishing.
                log::warn!(
                    "get_page_content: page {} /Contents element obj {} is {}, \
                     not a stream — skipping (page may be missing content)",
                    page_id.0,
                    r.0,
                    stream_obj.enum_variant()
                );
                failures.push(ContentFailure {
                    object: r.0,
                    reason: format!("{} is not a stream", stream_obj.enum_variant()),
                });
                continue;
            };
            // An element that cannot be decoded contributes no content; the
            // elements around it are independent streams and still render.
            let decoded = match decode_stream(&s.content, &s.dict) {
                Ok(decoded) => decoded,
                Err(e) => {
                    log::warn!(
                        "get_page_content: page {} /Contents element obj {} cannot be \
                         decoded ({e}) — skipping (page is missing content)",
                        page_id.0,
                        r.0
                    );
                    failures.push(ContentFailure {
                        object: r.0,
                        reason: e.clone(),
                    });
                    decode_failure = decode_failure.or(Some(e));
                    continue;
                }
            };
            decoded_any = true;
            // +1 accounts for the '\n' separator pushed below.
            if out.len().saturating_add(decoded.len()).saturating_add(1) > MAX_PAGE_CONTENT {
                return Err(PdfError::DecodeFailed(format!(
                    "page content exceeds {} MiB aggregate decompressed cap \
                     (possible decompression bomb)",
                    MAX_PAGE_CONTENT / (1024 * 1024)
                )));
            }
            // ISO 32000-2 §7.8.2: streams in a `/Contents` array are decoded
            // and *concatenated as if a single stream*, but a content stream
            // need not end at an operator/token boundary — the last token of
            // one stream can abut the first of the next (e.g. one stream ends
            // "...50 7", the next begins "00 cm").  A whitespace byte between
            // them is required so the lexer keeps the tokens distinct; a
            // single LF is the canonical, minimal separator.  Skip it only
            // when the accumulated buffer already ends in PDF whitespace
            // (LF/CR/space/tab/FF/NUL), which already provides the boundary.
            if out
                .last()
                .is_some_and(|b| !matches!(b, b'\n' | b'\r' | b' ' | b'\t' | 0x0C | 0x00))
            {
                out.push(b'\n');
            }
            out.extend_from_slice(&decoded);
        }
        // Content streams that all fail to decode leave nothing to render.
        match decode_failure {
            Some(e) if !decoded_any => Err(PdfError::DecodeFailed(e)),
            _ => Ok(PageContent {
                bytes: out,
                failures,
            }),
        }
    }

    /// Return a merged font dictionary for the page (all /Font entries from
    /// /Resources, with parent-chain inheritance).
    ///
    /// # Errors
    /// The errors of [`Self::get_page_resource_dict`].
    pub fn get_page_fonts(&self, page_id: ObjectId) -> Result<Dictionary, PdfError> {
        self.get_page_resource_dict(page_id, b"Font")
    }

    /// Return one named sub-dictionary from the page's /Resources, following
    /// parent page-tree inheritance.
    ///
    /// # Errors
    /// The errors of [`Self::get_object`] for the page, its ancestors, and
    /// the referenced resource dictionaries.
    pub fn get_page_resource_dict(
        &self,
        page_id: ObjectId,
        resource_name: &[u8],
    ) -> Result<Dictionary, PdfError> {
        // Walk up the page tree to find /Resources. Bound to 64 hops to defeat
        // cyclic /Parent references in malformed PDFs.
        let mut current_id = Some(page_id);
        let mut visited = std::collections::HashSet::new();
        for _ in 0..64 {
            let Some(id) = current_id else { break };
            if !visited.insert(id.0) {
                break;
            }
            let obj = self.get_object(id)?;
            let Some(dict) = obj.as_dict() else {
                break;
            };

            if let Some(res) = dict.get(b"Resources") {
                let res_dict = self.dict_from_object(res)?;
                if let Some(sub) = res_dict.get(resource_name) {
                    return self.dict_from_object(sub);
                }
            }

            // Walk to parent.
            current_id = dict.get(b"Parent").and_then(Object::as_reference);
        }
        Ok(Dictionary::new())
    }

    /// Return an inheritable page attribute value for a page object, following
    /// the `/Parent` chain per ISO 32000-2 §7.7.3.4.
    ///
    /// Checks the page leaf first, then walks up the `/Pages` ancestors until
    /// the key is found or the root is reached.  Returns `None` if the key is
    /// absent from every node in the chain.
    ///
    /// A hostile PDF can build a cyclic `/Parent` ring (page → pages → page) or
    /// a pathologically deep chain.  Both are defended: a visited-object set
    /// detects revisits on the first repeat, and the walk is hard-bounded to
    /// `PARENT_CHAIN_MAX_HOPS` (matching `page_tree.rs` `MAX_DEPTH`).  Either
    /// guard tripping is a malformed page tree, so it is logged loudly and the
    /// lookup fails closed (`None` → caller falls back to a default box) rather
    /// than spinning, recursing, or terminating silently.
    pub fn get_inherited_page_attr(&self, page_id: ObjectId, key: &[u8]) -> Option<Object> {
        // Matches `page_tree.rs` MAX_DEPTH so both /Parent walkers agree on the
        // legal page-tree depth bound.
        const PARENT_CHAIN_MAX_HOPS: usize = 64;

        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut current_id = Some(page_id);
        for _ in 0..PARENT_CHAIN_MAX_HOPS {
            let id = current_id?;
            if !visited.insert(id.0) {
                log::warn!(
                    "cyclic /Parent reference at object {} while resolving inheritable \
                     attribute {:?}; treating as absent (malformed PDF, ISO 32000-2 §7.7.3.4)",
                    id.0,
                    String::from_utf8_lossy(key),
                );
                return None;
            }
            let obj = self.get_object(id).ok()?;
            let dict = match obj.as_ref() {
                Object::Dictionary(d) => d,
                Object::Stream(s) => &s.dict,
                _ => return None,
            };
            if let Some(val) = dict.get(key) {
                // Resolve one level of indirection so callers always get the
                // concrete value (e.g. an Array, not a Reference to an Array).
                let resolved = match val {
                    Object::Reference(rid) => {
                        self.get_object(*rid).ok().map(|o| o.as_ref().clone())
                    }
                    other => Some(other.clone()),
                };
                return resolved;
            }
            current_id = dict.get(b"Parent").and_then(Object::as_reference);
        }
        // Ran out of hops without finding the key or hitting the root.  A
        // conforming page tree is far shallower than the bound, so this is a
        // hostile/over-deep chain — surface it instead of silently defaulting.
        log::warn!(
            "/Parent chain exceeded {PARENT_CHAIN_MAX_HOPS} hops while resolving inheritable \
             attribute {:?}; treating as absent (malformed PDF, ISO 32000-2 §7.7.3.4)",
            String::from_utf8_lossy(key),
        );
        None
    }

    /// Take a dictionary value out of an `Object`, resolving one level of
    /// indirection if needed. Non-dict values yield an empty map (matches the
    /// behaviour of malformed-but-recoverable PDFs).
    fn dict_from_object(&self, obj: &Object) -> Result<Dictionary, PdfError> {
        match obj {
            Object::Dictionary(d) => Ok(d.clone()),
            Object::Reference(rid) => {
                let r = self.get_object(*rid)?;
                Ok(r.as_dict().cloned().unwrap_or_default())
            }
            _ => Ok(Dictionary::new()),
        }
    }
}

// ── Page iterator ─────────────────────────────────────────────────────────────

struct PageIter<'a> {
    doc: &'a Document,
    /// Interior nodes on the traversal path, each with its resolved `/Kids`.
    stack: Vec<PageTreeFrame>,
    /// Object numbers of the nodes in `stack`; a kid already present is a
    /// cyclic `/Kids` reference and is skipped.
    on_stack: std::collections::HashSet<u32>,
    /// The `/Pages` root, visited on the first call to `next`.
    root: Option<ObjectId>,
    page_num: u32,
}

/// One interior page-tree node: its `/Kids` references, resolved once, and
/// the index of the next kid to visit.
struct PageTreeFrame {
    id: ObjectId,
    kids: Vec<ObjectId>,
    next: usize,
}

impl<'a> PageIter<'a> {
    fn new(doc: &'a Document) -> Self {
        // Catalog errors collapse to an empty iterator — same outcome as a PDF
        // with no /Pages root. Logged so corruption doesn't go silent.
        let catalog = doc.catalog().unwrap_or_else(|e| {
            log::warn!("PageIter: catalog unavailable, no pages will be yielded: {e:?}");
            Arc::default()
        });
        Self {
            doc,
            stack: Vec::new(),
            on_stack: std::collections::HashSet::new(),
            root: catalog.get(b"Pages").and_then(Object::as_reference),
            page_num: 0,
        }
    }

    /// Visit node `id`: a `/Type /Page` node is returned as the next page; any
    /// other dictionary with a `/Kids` array (direct, or an indirect reference
    /// to one) is pushed as an interior node.  Anything else is skipped.
    fn visit(&mut self, id: ObjectId) -> Option<(u32, ObjectId)> {
        let obj = self.doc.get_object(id).ok()?;
        let dict = obj.as_dict()?;
        if dict.get(b"Type").and_then(Object::as_name) == Some(b"Page") {
            self.page_num += 1;
            return Some((self.page_num, id));
        }
        let kids: Vec<ObjectId> = match dict.get(b"Kids") {
            Some(Object::Array(a)) => a.iter().filter_map(Object::as_reference).collect(),
            Some(Object::Reference(rid)) => {
                let arr = self.doc.get_object(*rid).ok()?;
                arr.as_array()?
                    .iter()
                    .filter_map(Object::as_reference)
                    .collect()
            }
            _ => return None,
        };
        let _ = self.on_stack.insert(id.0);
        self.stack.push(PageTreeFrame { id, kids, next: 0 });
        None
    }
}

impl Iterator for PageIter<'_> {
    type Item = (u32, ObjectId);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(root) = self.root.take()
            && let Some(page) = self.visit(root)
        {
            return Some(page);
        }
        loop {
            let frame = self.stack.last_mut()?;
            let Some(&kid) = frame.kids.get(frame.next) else {
                let id = frame.id;
                let _ = self.stack.pop();
                let _ = self.on_stack.remove(&id.0);
                continue;
            };
            frame.next += 1;
            if self.on_stack.contains(&kid.0) {
                log::warn!("PageIter: cyclic /Kids reference to obj {} skipped", kid.0);
                continue;
            }
            if let Some(page) = self.visit(kid) {
                return Some(page);
            }
        }
    }
}

/// How far into the file the `%PDF-` header may legally appear.
///
/// The spec puts the header at byte 0, but real-world files prepend a few
/// junk bytes (mail transfer artefacts, UTF-8 BOM, shebang lines), so the
/// first kilobyte is scanned.
const HEADER_SCAN_WINDOW: usize = 1024;

/// Reject obviously-unusable input *before* the xref parser runs so the
/// operator gets an accurate diagnosis ("file is empty" / "not a PDF")
/// instead of the low-level, misleading "'startxref' not found", and return
/// the position of the `%PDF-` header.
///
/// This is the single entry choke point shared by [`Document::open`] and
/// [`Document::from_bytes_owned`].
fn validate_pdf_input(data: &[u8]) -> Result<usize, PdfError> {
    if data.is_empty() {
        return Err(PdfError::EmptyInput);
    }
    let window = &data[..data.len().min(HEADER_SCAN_WINDOW)];
    window
        .windows(5)
        .position(|w| w == b"%PDF-")
        .ok_or(PdfError::NotPdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a PDF from `(object number, object body)` pairs, with a
    /// catalog as object 1 and a classic xref table.  Missing numbers are
    /// free entries.
    fn pdf_with_objects(objects: &[(u32, String)]) -> Vec<u8> {
        pdf_with_objects_after("", objects)
    }

    /// [`pdf_with_objects`] preceded by `prefix`, with every offset counted
    /// from the start of the prefix.
    fn pdf_with_objects_after(prefix: &str, objects: &[(u32, String)]) -> Vec<u8> {
        use std::fmt::Write as _;

        let mut out = format!("{prefix}%PDF-1.4\n");
        let max = objects.iter().map(|(n, _)| *n).max().unwrap_or(1).max(1);
        let mut offsets = vec![None; max as usize + 1];
        let catalog = (1, "<</Type /Catalog>>".to_owned());
        for (n, body) in std::iter::once(&catalog).chain(objects) {
            offsets[*n as usize] = Some(out.len());
            write!(out, "{n} 0 obj\n{body}\nendobj\n").unwrap();
        }
        let xref_start = out.len();
        write!(out, "xref\n0 {}\n0000000000 65535 f\r\n", max + 1).unwrap();
        for off in &offsets[1..] {
            match off {
                Some(o) => write!(out, "{o:010} 00000 n\r\n").unwrap(),
                None => out.push_str("0000000000 65535 f\r\n"),
            }
        }
        write!(
            out,
            "trailer\n<</Size {} /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF",
            max + 1
        )
        .unwrap();
        out.into_bytes()
    }

    /// Byte offset of the classic-xref entry of object `num` in a
    /// [`pdf_with_objects`] file.
    fn xref_entry_at(bytes: &[u8], num: usize) -> usize {
        let xref = bytes
            .windows(5)
            .position(|w| w == b"xref\n")
            .expect("xref keyword");
        let first_entry = xref
            + bytes[xref..]
                .iter()
                .position(|&b| b == b'\n')
                .and_then(|l1| {
                    bytes[xref + l1 + 1..]
                        .iter()
                        .position(|&b| b == b'\n')
                        .map(|l2| l1 + l2 + 2)
                })
                .expect("subsection header");
        first_entry + num * 20
    }

    /// Overwrite the classic-xref offset of object `num` in a
    /// [`pdf_with_objects`] file with `offset`.
    fn set_xref_offset(bytes: &mut [u8], num: usize, offset: usize) {
        let entry = xref_entry_at(bytes, num);
        bytes[entry..entry + 10].copy_from_slice(format!("{offset:010}").as_bytes());
    }

    /// Overwrite the classic-xref entry of object `num` in a
    /// [`pdf_with_objects`] file with an in-use entry at `offset`.
    fn set_xref_in_use(bytes: &mut [u8], num: usize, offset: usize) {
        let entry = xref_entry_at(bytes, num);
        bytes[entry..entry + 18].copy_from_slice(format!("{offset:010} 00000 n").as_bytes());
    }

    /// An xref entry that leads to another object, for an object with no
    /// body anywhere in the file, is a missing object: the other object is
    /// never returned under the wrong number.
    #[test]
    fn xref_offset_naming_another_object_with_no_body_is_missing() {
        let mut bytes = pdf_with_objects(&[(5, "(five)".to_owned())]);
        let five = find(&bytes, b"5 0 obj").expect("object 5");
        set_xref_in_use(&mut bytes, 4, five);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        let err = doc.get_object((4, 0)).expect_err("no object 4 exists");
        assert!(
            matches!(&err, PdfError::BadObject { id: 4, detail } if detail.contains("object 5")),
            "got {err:?}"
        );
        assert_eq!(
            doc.get_object((5, 0)).expect("object 5").as_str(),
            Some(&b"five"[..])
        );
    }

    /// An xref entry that leads to no object header at all, for an object
    /// with no body in the file, is a missing object.
    #[test]
    fn xref_offset_into_garbage_with_no_body_is_missing() {
        let mut bytes = pdf_with_objects(&[(5, "(five)".to_owned())]);
        let inside_five = find(&bytes, b"(five)").expect("object 5 body");
        set_xref_in_use(&mut bytes, 4, inside_five);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        let err = doc.get_object((4, 0)).expect_err("no object 4 exists");
        assert!(
            matches!(&err, PdfError::BadObject { id: 4, .. }),
            "got {err:?}"
        );
    }

    /// An xref offset that lands inside another object is recovered by
    /// scanning the file for the object's header.
    #[test]
    fn stale_xref_offset_is_recovered_by_scanning() {
        let mut bytes = pdf_with_objects(&[
            (2, "(two)".to_owned()),
            (3, "(three, a longer body)".to_owned()),
        ]);
        let into_three = find(&bytes, b"3 0 obj").expect("object 3") + 9;
        set_xref_offset(&mut bytes, 2, into_three);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
    }

    /// An xref offset that leads to a different object never returns that
    /// object under the wrong number.
    #[test]
    fn xref_offset_naming_another_object_is_corrected() {
        let mut bytes = pdf_with_objects(&[(2, "(two)".to_owned()), (3, "(three)".to_owned())]);
        let (two, three) = (
            find(&bytes, b"2 0 obj").expect("object 2"),
            find(&bytes, b"3 0 obj").expect("object 3"),
        );
        set_xref_offset(&mut bytes, 2, three);
        set_xref_offset(&mut bytes, 3, two);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
        assert_eq!(
            doc.get_object((3, 0)).expect("object 3").as_str(),
            Some(&b"three"[..])
        );
    }

    /// Append an incremental update to a [`pdf_with_objects`] file whose
    /// only entry frees object `num`.
    fn append_update_freeing(bytes: &mut Vec<u8>, num: u32) {
        use std::fmt::Write as _;

        let startxref = find(bytes, b"startxref\n").expect("startxref") + 10;
        let prev: usize = std::str::from_utf8(&bytes[startxref..])
            .ok()
            .and_then(|s| s.split_whitespace().next()?.parse().ok())
            .expect("startxref offset");
        bytes.push(b'\n');
        let xref = bytes.len();
        let mut update = String::new();
        write!(
            update,
            "xref\n{num} 1\n0000000000 00001 f\r\ntrailer\n<</Size 4 /Root 1 0 R /Prev {prev}>>\n\
             startxref\n{xref}\n%%EOF"
        )
        .unwrap();
        bytes.extend_from_slice(update.as_bytes());
    }

    /// A freed object that is still referenced, and whose body the update
    /// left in the file, resolves to that body.
    #[test]
    fn freed_object_with_a_body_is_recovered_by_scanning() {
        let mut bytes = pdf_with_objects(&[(2, "(two)".to_owned())]);
        append_update_freeing(&mut bytes, 2);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert!(matches!(doc.xref.get(2), Some(XrefEntry::Free)));
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
    }

    /// A free entry for an object with no body in the file is a missing
    /// object, not an error elsewhere and not another object.
    #[test]
    fn freed_object_without_a_body_is_missing() {
        let bytes = pdf_with_objects(&[(3, "(three)".to_owned())]);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert!(matches!(doc.xref.get(2), Some(XrefEntry::Free)));
        let err = doc.get_object((2, 0)).expect_err("object 2 is free");
        assert!(
            matches!(&err, PdfError::BadObject { id: 2, detail } if detail.contains("free")),
            "got {err:?}"
        );
    }

    /// A referenced object the xref does not list, whose body is in the
    /// file, resolves to that body.
    #[test]
    fn absent_object_with_a_body_is_recovered_by_scanning() {
        let mut bytes = pdf_with_objects(&[(2, "(two)".to_owned())]);
        bytes.extend_from_slice(b"\n7 0 obj\n(seven)\nendobj\n");
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert!(doc.xref.get(7).is_none());
        assert_eq!(
            doc.get_object((7, 0)).expect("object 7").as_str(),
            Some(&b"seven"[..])
        );
    }

    /// A referenced object neither the xref nor the file holds is missing.
    #[test]
    fn absent_object_without_a_body_is_missing() {
        let bytes = pdf_with_objects(&[(2, "(two)".to_owned())]);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        let err = doc.get_object((7, 0)).expect_err("object 7 is absent");
        assert!(
            matches!(&err, PdfError::BadObject { id: 7, detail } if detail.contains("not in xref")),
            "got {err:?}"
        );
    }

    /// The body of an object stream holding `members` (number, body), as a
    /// stream object whose dictionary opens with the entries `ty`.
    fn objstm_body(ty: &str, members: &[(u32, &str)]) -> String {
        use std::fmt::Write as _;

        let (mut directory, mut data) = (String::new(), String::new());
        for (num, body) in members {
            write!(directory, "{num} {} ", data.len()).unwrap();
            data.push_str(body);
            data.push('\n');
        }
        format!(
            "<<{ty} /N {} /First {} /Length {}>>\nstream\n{directory}{data}\nendstream",
            members.len(),
            directory.len(),
            directory.len() + data.len()
        )
    }

    /// A PDF-1.5 file holding the catalog (1) and `direct` objects, indexed
    /// by a cross-reference stream of `/Size size` whose entries are the
    /// direct objects' offsets and the type-2 entries `compressed`
    /// (number, container, index); every other number is free.
    fn xref_stream_pdf(
        direct: &[(u32, String)],
        compressed: &[(u32, u32, u32)],
        size: u32,
    ) -> Vec<u8> {
        let mut out = b"%PDF-1.5\n".to_vec();
        let mut rows = vec![[0u8; 7]; size as usize];
        let catalog = (1, "<</Type /Catalog>>".to_owned());
        for (n, body) in std::iter::once(&catalog).chain(direct) {
            let off = u32::try_from(out.len()).unwrap().to_be_bytes();
            rows[*n as usize] = [1, off[0], off[1], off[2], off[3], 0, 0];
            out.extend_from_slice(format!("{n} 0 obj\n{body}\nendobj\n").as_bytes());
        }
        for &(n, container, index) in compressed {
            let (c, i) = (
                container.to_be_bytes(),
                u16::try_from(index).unwrap().to_be_bytes(),
            );
            rows[n as usize] = [2, c[0], c[1], c[2], c[3], i[0], i[1]];
        }
        let xref_at = out.len();
        let data = rows.concat();
        out.extend_from_slice(
            format!(
                "{size} 0 obj\n<</Type /XRef /Size {size} /W [1 4 2] /Root 1 0 R /Length {}>>\nstream\n",
                data.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(&data);
        out.extend_from_slice(
            format!("\nendstream\nendobj\nstartxref\n{xref_at}\n%%EOF").as_bytes(),
        );
        out
    }

    /// A compressed object its own container lacks is recovered from
    /// another object stream that holds it, as is a compressed object the
    /// xref does not list.
    #[test]
    fn compressed_object_is_recovered_from_another_object_stream() {
        let bytes = xref_stream_pdf(
            &[
                (4, objstm_body("/Type /ObjStm", &[(2, "(two)")])),
                (
                    5,
                    objstm_body("/Type /ObjStm", &[(3, "(three)"), (7, "(seven)")]),
                ),
            ],
            &[(2, 4, 0), (3, 4, 1)],
            6,
        );
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
        assert_eq!(
            doc.get_object((3, 0)).expect("object 3").as_str(),
            Some(&b"three"[..])
        );
        assert!(doc.xref.get(7).is_none());
        assert_eq!(
            doc.get_object((7, 0)).expect("object 7").as_str(),
            Some(&b"seven"[..])
        );
    }

    /// A type-2 entry naming a stream of another `/Type`, or an untyped
    /// stream without `/N` and `/First`, does not decode that stream; an
    /// untyped stream with both is an object stream.
    #[test]
    fn container_must_be_an_object_stream() {
        for ty in ["/Type /XObject /Subtype /Image", "/Type /ObjStmX"] {
            let bytes = xref_stream_pdf(&[(4, objstm_body(ty, &[(2, "(two)")]))], &[(2, 4, 0)], 5);
            let doc = Document::from_bytes_owned(bytes).expect("opens");
            let err = doc
                .get_object((2, 0))
                .expect_err("object 4 is no object stream");
            assert!(
                matches!(&err, PdfError::BadObject { id: 4, detail } if detail.contains("ObjStm")),
                "{ty}: got {err:?}"
            );
        }
        let bytes = xref_stream_pdf(
            &[(4, "<</Length 9>>\nstream\n2 0 (two)\nendstream".to_owned())],
            &[(2, 4, 0)],
            5,
        );
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert!(doc.get_object((2, 0)).is_err());

        let bytes = xref_stream_pdf(&[(4, objstm_body("", &[(2, "(two)")]))], &[(2, 4, 0)], 5);
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
    }

    /// An object stream whose own entry places it inside an object stream
    /// (here itself) is read from its header in the file, without
    /// following the entry.
    #[test]
    fn self_contained_object_stream_is_read_from_the_file() {
        let bytes = xref_stream_pdf(
            &[(4, objstm_body("/Type /ObjStm", &[(2, "(two)")]))],
            &[(2, 4, 0), (4, 4, 0)],
            5,
        );
        let doc = Document::from_bytes_owned(bytes).expect("opens");
        assert_eq!(
            doc.get_object((2, 0)).expect("object 2").as_str(),
            Some(&b"two"[..])
        );
        assert!(matches!(
            doc.get_object((4, 0)).expect("object 4").as_ref(),
            Object::Stream(_)
        ));
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// Write `bytes` to a fresh temporary file and return its handle.
    fn temp_pdf(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write as _;

        let mut file = tempfile::NamedTempFile::new().expect("create temp file");
        file.write_all(bytes).expect("write temp file");
        file
    }

    /// A file shorter than `MMAP_MIN_LEN` is copied into memory, so
    /// truncating it after open leaves the document's bytes intact.
    #[test]
    fn small_file_is_read_and_survives_truncation() {
        let bytes = pdf_with_objects(&[(2, "(small)".to_owned())]);
        assert!((bytes.len() as u64) < MMAP_MIN_LEN);
        let file = temp_pdf(&bytes);
        let doc = Document::open(file.path()).expect("opens");
        assert!(matches!(*doc.data, MmapOrVec::Owned(_)));
        file.as_file().set_len(0).expect("truncate");
        let obj = doc.get_object((2, 0)).expect("object 2");
        assert_eq!(obj.as_str(), Some(&b"small"[..]));
        assert_eq!(doc.bytes(), &bytes[..]);
    }

    /// A file of at least `MMAP_MIN_LEN` bytes is memory-mapped.
    #[test]
    fn large_file_is_mapped() {
        let padding = "a".repeat(usize::try_from(MMAP_MIN_LEN).unwrap());
        let bytes = pdf_with_objects(&[(2, format!("({padding})"))]);
        let file = temp_pdf(&bytes);
        let doc = Document::open(file.path()).expect("opens");
        assert!(matches!(*doc.data, MmapOrVec::Mapped(_)));
        assert_eq!(doc.bytes(), &bytes[..]);
    }

    fn hex_stream(filter: &str) -> String {
        let data = "48656C6C6F>";
        format!(
            "<</Filter {filter} /Length {}>>\nstream\n{data}\nendstream",
            data.len()
        )
    }

    #[test]
    fn indirect_filter_is_resolved_into_the_stream_dict() {
        let bytes =
            pdf_with_objects(&[(5, "/ASCIIHexDecode".to_owned()), (6, hex_stream("5 0 R"))]);
        let doc = Document::from_bytes_owned(bytes).expect("parses");
        let obj = doc.get_object((6, 0)).expect("object 6");
        let stream = obj.as_stream().expect("a stream");
        assert_eq!(
            stream.dict.get(b"Filter"),
            Some(&Object::Name(b"ASCIIHexDecode".to_vec()))
        );
        assert_eq!(stream.decompressed_content().expect("decodes"), b"Hello");
    }

    #[test]
    fn indirect_filter_array_element_is_resolved() {
        let bytes = pdf_with_objects(&[
            (5, "/ASCIIHexDecode".to_owned()),
            (6, hex_stream("[5 0 R]")),
        ]);
        let doc = Document::from_bytes_owned(bytes).expect("parses");
        let obj = doc.get_object((6, 0)).expect("object 6");
        let stream = obj.as_stream().expect("a stream");
        assert_eq!(stream.decompressed_content().expect("decodes"), b"Hello");
    }

    #[test]
    fn self_referencing_filter_neither_hangs_nor_resolves() {
        let bytes = pdf_with_objects(&[(6, hex_stream("6 0 R"))]);
        let doc = Document::from_bytes_owned(bytes).expect("parses");
        let obj = doc.get_object((6, 0)).expect("object 6");
        let stream = obj.as_stream().expect("a stream");
        assert_eq!(stream.dict.get(b"Filter"), Some(&Object::Reference((6, 0))));
    }

    fn minimal_pdf() -> Vec<u8> {
        // Minimal valid PDF with one empty page.
        // Offsets verified by byte-counting the sections below.
        // %PDF-1.4\n = 9 bytes  → obj1 at 9
        // "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n" = 47 bytes → obj2 at 56
        // "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n" = 55 bytes → obj3 at 111
        // "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n" = 69 bytes → xref at 180
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn open_minimal_pdf() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        assert_eq!(doc.page_count(), 1);
    }

    #[test]
    fn page_count_fast_matches_eager() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        assert_eq!(doc.page_count_fast(), 1);
        assert_eq!(doc.page_count_fast(), doc.page_count());
    }

    #[test]
    fn get_pages_returns_one_page() {
        let doc = Document::from_bytes_owned(minimal_pdf()).unwrap();
        let pages: Vec<_> = doc.get_pages().collect();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].0, 1);
    }

    /// Regression: a Pages node whose `/Kids` is an indirect reference to an
    /// array (rather than an inline array) used to bail and report 0 pages.
    /// Real PDFs in the wild ship this layout (e.g. corpus-04).
    #[test]
    fn get_pages_handles_kids_as_reference() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        // Kids points to obj 4, which is an array containing obj 3 (the page).
        let obj2 = "2 0 obj\n<</Type /Pages /Kids 4 0 R /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n";
        let obj4 = "4 0 obj\n[3 0 R]\nendobj\n";
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off4 = off3 + obj3.len();
        let xref_start = off4 + obj4.len();
        let xref = format!(
            "xref\n0 5\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n{off4:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{obj4}{xref}{trailer}").into_bytes();
        let doc = Document::from_bytes_owned(bytes).expect("indirect-Kids PDF");
        assert_eq!(doc.page_count(), 1, "expected 1 page through indirect Kids");
    }

    /// Build a minimal PDF-1.5 whose cross-reference is a `/Type /XRef`
    /// stream (`/W [1 2 2]`) and whose Catalog / Pages / Page objects live
    /// inside a `/Type /ObjStm` compressed object stream.  This is the
    /// structure emitted by virtually every modern PDF toolchain (`pdfTeX`,
    /// `LibreOffice`, …); the streams here are left uncompressed (no `/Filter`)
    /// so the fixture needs no zlib round-trip.
    ///
    /// Regression for the PDF-1.5 blank-render class: the page object (obj 3)
    /// is reachable only via an xref-stream **type-2** entry → `ObjStm`
    /// extraction.  If `/W` decoding, type-2 `get_object` routing, or `ObjStm`
    /// directory parsing is wrong, the page object is unresolvable and the
    /// page renders blank.
    #[test]
    fn pdf15_xref_stream_with_objstm_resolves_in_objstm_page() {
        let header = b"%PDF-1.5\n";

        // Bodies for the three objects packed into the ObjStm.
        let b1 = b"<</Type/Catalog/Pages 2 0 R>>";
        let b2 = b"<</Type/Pages/Kids[3 0 R]/Count 1>>";
        let b3 = b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>";
        let o1 = 0usize;
        let o2 = o1 + b1.len() + 1; // +1 for the '\n' body separator
        let o3 = o2 + b2.len() + 1;
        let directory = format!("1 {o1} 2 {o2} 3 {o3} ").into_bytes();
        let first = directory.len();
        let mut objstm_data = directory;
        for body in [&b1[..], &b2[..], &b3[..]] {
            objstm_data.extend_from_slice(body);
            objstm_data.push(b'\n');
        }
        let mut objstm = format!(
            "4 0 obj\n<</Type/ObjStm/N 3/First {first}/Length {}>>\nstream\n",
            objstm_data.len()
        )
        .into_bytes();
        objstm.extend_from_slice(&objstm_data);
        objstm.extend_from_slice(b"\nendstream\nendobj\n");

        let off4 = header.len();
        let off5 = off4 + objstm.len();

        // /W [1 2 2] entries for object numbers 0..=5.
        let entry = |t: u8, f1: usize, f2: usize| -> [u8; 5] {
            let a = u16::try_from(f1).unwrap().to_be_bytes();
            let b = u16::try_from(f2).unwrap().to_be_bytes();
            [t, a[0], a[1], b[0], b[1]]
        };
        let mut entries = Vec::new();
        entries.extend_from_slice(&entry(0, 0, 65535)); // 0 free
        entries.extend_from_slice(&entry(2, 4, 0)); // 1 → ObjStm 4, idx 0
        entries.extend_from_slice(&entry(2, 4, 1)); // 2 → ObjStm 4, idx 1
        entries.extend_from_slice(&entry(2, 4, 2)); // 3 → ObjStm 4, idx 2 (PAGE)
        entries.extend_from_slice(&entry(1, off4, 0)); // 4 direct
        entries.extend_from_slice(&entry(1, off5, 0)); // 5 direct (self)

        let mut xref = format!(
            "5 0 obj\n<</Type/XRef/Size 6/W[1 2 2]/Root 1 0 R/Length {}>>\nstream\n",
            entries.len()
        )
        .into_bytes();
        xref.extend_from_slice(&entries);
        xref.extend_from_slice(b"\nendstream\nendobj\n");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&objstm);
        bytes.extend_from_slice(&xref);
        bytes.extend_from_slice(format!("startxref\n{off5}\n%%EOF").as_bytes());

        let doc = Document::from_bytes_owned(bytes).expect("PDF-1.5 xref-stream+ObjStm");

        // Catalog (obj 1) resolves out of the ObjStm.
        let catalog = doc.catalog().expect("catalog via ObjStm");
        assert!(catalog.get(b"Pages".as_slice()).is_some());

        // The page tree descends through ObjStm-resident Pages → Page.
        assert_eq!(doc.page_count(), 1, "one page via xref-stream type-2");
        let pages: Vec<_> = doc.get_pages().collect();
        assert_eq!(pages.len(), 1);

        // The page object itself (obj 3, an xref type-2 entry) resolves.
        let (_, page_id) = pages[0];
        assert_eq!(page_id.0, 3);
        let page = doc.get_object(page_id).expect("in-ObjStm page object");
        let pd = page.as_dict().expect("page is a dict");
        assert_eq!(
            pd.get(b"Type".as_slice()).and_then(Object::as_name),
            Some(&b"Page"[..])
        );
    }

    /// Direct unit test of the xref-stream `/W [1 2 2]` binary entry decoder
    /// for a **type-2** (in-object-stream) record.  Locks the big-endian
    /// field decode + the container/index assignment that the PDF-1.5
    /// resolution path depends on.
    #[test]
    fn xref_stream_w_type2_entry_decode() {
        // Single uncompressed xref stream, obj 1, /Size 2, covering objs 0..1.
        //   obj 0: type 0 (free)        → 00 0000 0000
        //   obj 1: type 2, container 7, index 3 → 02 0007 0003
        let entries: [u8; 10] = [0, 0, 0, 0, 0, 2, 0, 7, 0, 3];
        let body = format!(
            "1 0 obj\n<</Type/XRef/Size 2/W[1 2 2]/Root 9 0 R/Length {}>>\nstream\n",
            entries.len()
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"%PDF-1.5\n");
        let startxref = bytes.len();
        bytes.extend_from_slice(body.as_bytes());
        bytes.extend_from_slice(&entries);
        bytes.extend_from_slice(b"\nendstream\nendobj\n");
        bytes.extend_from_slice(format!("startxref\n{startxref}\n%%EOF").as_bytes());

        let (table, _) = crate::xref::read_xref(&bytes, 0).expect("xref stream parse");
        match table.get(1) {
            Some(crate::xref::XrefEntry::InObjStm { container, index }) => {
                assert_eq!(container, 7, "type-2 field2 = ObjStm container number");
                assert_eq!(index, 3, "type-2 field3 = index within ObjStm");
            }
            other => panic!("expected InObjStm entry for obj 1, got {other:?}"),
        }
    }

    /// Regression: a page whose `/Contents` is an *indirect reference to an
    /// array* of content-stream references (legal per ISO 32000-2 §7.7.3.3;
    /// emitted by `LibreOffice` and many imposition tools).  The pre-fix code
    /// dereferenced the single ref, found an `Object::Array` instead of an
    /// `Object::Stream`, skipped it, and returned an empty content buffer —
    /// every glyph-bearing page rendered pure white.  This asserts the array
    /// is expanded and all member streams are concatenated.
    #[test]
    fn get_page_content_contents_ref_to_array() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        // /Contents is a single indirect reference (obj 5), NOT an inline array.
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 5 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        // obj 5 resolves to an ARRAY of two content-stream references.
        let obj5 = "5 0 obj\n[6 0 R 7 0 R]\nendobj\n";
        let s6 = "q\n";
        let obj6 = format!(
            "6 0 obj\n<</Length {}>>\nstream\n{s6}\nendstream\nendobj\n",
            s6.len()
        );
        let s7 = "BT ET\n";
        let obj7 = format!(
            "7 0 obj\n<</Length {}>>\nstream\n{s7}\nendstream\nendobj\n",
            s7.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        // obj 4 is unused; reserve a free slot so xref indices stay simple.
        let off5 = off3 + obj3.len();
        let off6 = off5 + obj5.len();
        let off7 = off6 + obj6.len();
        let xref_start = off7 + obj7.len();
        let xref = format!(
            "xref\n0 8\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n\
             {off3:010} 00000 n\r\n0000000000 65535 f\r\n\
             {off5:010} 00000 n\r\n{off6:010} 00000 n\r\n{off7:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 8 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes =
            format!("{header}{obj1}{obj2}{obj3}{obj5}{obj6}{obj7}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("contents-ref-to-array PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        let content = doc.get_page_content(page_id).expect("page content");
        // Both member streams must be present, concatenated with a separator.
        assert!(
            !content.is_empty(),
            "ref-to-array /Contents must not yield empty content (the blank-page bug)"
        );
        let text = String::from_utf8_lossy(&content);
        assert!(
            text.contains('q'),
            "first content stream (obj 6) missing: {text:?}"
        );
        assert!(
            text.contains("BT ET"),
            "second content stream (obj 7) missing: {text:?}"
        );
    }

    /// ISO 32000-2 §7.8.2: the streams of a `/Contents` array are decoded and
    /// concatenated *as if a single stream*, and a stream need not break at a
    /// token boundary.  This builds a page whose two content streams split a
    /// number mid-digits — obj 6 ends `"10 0 0 10 50 7"`, obj 7 begins
    /// `"00 cm ..."`.  Without the inter-stream separator the lexer reads
    /// `"700 cm"` (a corrupt CTM) instead of `"50 700"` then `"cm"`, so the
    /// page transforms wrongly and content lands off-page (silent-wrong, the
    /// v1 sin).  Asserts the LF separator sits exactly at the join.
    #[test]
    fn get_page_content_array_inserts_separator_at_midtoken_split() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 9 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        let obj9 = "9 0 obj\n[10 0 R 11 0 R]\nendobj\n";
        // Stream 10 ends in the middle of the literal "700": "...50 7".
        let s10 = "q 1 0 0 1 0 0 cm 10 0 0 10 50 7";
        let obj10 = format!(
            "10 0 obj\n<</Length {}>>\nstream\n{s10}\nendstream\nendobj\n",
            s10.len()
        );
        // Stream 11 resumes with "00 cm ...".  Joined raw this reads "700 cm".
        let s11 = "00 cm BT ET Q";
        let obj11 = format!(
            "11 0 obj\n<</Length {}>>\nstream\n{s11}\nendstream\nendobj\n",
            s11.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off9 = off3 + obj3.len();
        let off10 = off9 + obj9.len();
        let off11 = off10 + obj10.len();
        let xref_start = off11 + obj11.len();
        // Objects 4..8 are free placeholders so xref indices stay literal.
        let xref = format!(
            "xref\n0 12\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n0000000000 65535 f\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n\
             {off9:010} 00000 n\r\n{off10:010} 00000 n\r\n{off11:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 12 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes =
            format!("{header}{obj1}{obj2}{obj3}{obj9}{obj10}{obj11}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("midtoken-split PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        let content = doc.get_page_content(page_id).expect("page content");
        let text = String::from_utf8_lossy(&content);
        // The separator must keep the "7" and "00" tokens apart: the corrupt
        // glued form "700 cm" must NOT appear, and the correctly separated
        // "50 7\n00 cm" boundary must.
        assert!(
            !text.contains("700 cm"),
            "missing §7.8.2 separator glued mid-token numbers: {text:?}"
        );
        assert!(
            text.contains("50 7\n00 cm"),
            "expected LF separator exactly at the stream join: {text:?}"
        );
    }

    /// A cyclic / self-referential `/Contents`: obj 9 is `[10 0 R 9 0 R]` —
    /// the second element points back at the array object itself.  A naive
    /// recursive expander would loop forever (a denial-of-service input).  The
    /// expander is a single non-recursive pass, so this must terminate
    /// promptly: obj 10's stream renders, and the back-reference (which
    /// resolves to an Array, not a Stream) is logged-and-skipped rather than
    /// recursed into or silently lost.
    #[test]
    fn get_page_content_cyclic_contents_terminates() {
        let header = "%PDF-1.4\n";
        let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
        let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
        let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /Contents 9 0 R \
                    /MediaBox [0 0 612 792]>>\nendobj\n";
        // The array's 2nd element (9 0 R) is the array object itself.
        let obj9 = "9 0 obj\n[10 0 R 9 0 R]\nendobj\n";
        let s10 = "q BT ET Q";
        let obj10 = format!(
            "10 0 obj\n<</Length {}>>\nstream\n{s10}\nendstream\nendobj\n",
            s10.len()
        );
        let off1 = header.len();
        let off2 = off1 + obj1.len();
        let off3 = off2 + obj2.len();
        let off9 = off3 + obj3.len();
        let off10 = off9 + obj9.len();
        let xref_start = off10 + obj10.len();
        let xref = format!(
            "xref\n0 11\n0000000000 65535 f\r\n\
             {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n0000000000 65535 f\r\n\
             0000000000 65535 f\r\n0000000000 65535 f\r\n\
             {off9:010} 00000 n\r\n{off10:010} 00000 n\r\n"
        );
        let trailer = format!("trailer\n<</Size 11 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
        let bytes = format!("{header}{obj1}{obj2}{obj3}{obj9}{obj10}{xref}{trailer}").into_bytes();

        let doc = Document::from_bytes_owned(bytes).expect("cyclic-/Contents PDF");
        let page_id = doc.get_pages().next().expect("one page").1;
        // Must return (not hang): the valid stream is kept, the self-ref is
        // skipped because it resolves to an Array rather than a Stream.
        let content = doc.get_page_content(page_id).expect("page content");
        let text = String::from_utf8_lossy(&content);
        assert!(
            text.contains("q BT ET Q"),
            "valid content stream must still render: {text:?}"
        );
    }

    // ── Input-boundary validation ────────────────────────────────────────

    /// `Document` is intentionally not `Debug`, so `Result::unwrap_err` is
    /// unavailable; pull the error out by hand.
    fn open_err(bytes: Vec<u8>) -> PdfError {
        match Document::from_bytes_owned(bytes) {
            Ok(_) => panic!("expected an error, document opened successfully"),
            Err(e) => e,
        }
    }

    #[test]
    fn empty_input_is_diagnosed_distinctly() {
        let err = open_err(Vec::new());
        assert!(
            matches!(err, PdfError::EmptyInput),
            "0-byte input must yield EmptyInput, got: {err}"
        );
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn non_pdf_input_is_diagnosed_distinctly() {
        let err = open_err(b"just some text, not a pdf\n".to_vec());
        assert!(
            matches!(err, PdfError::NotPdf),
            "non-PDF input must yield NotPdf, got: {err}"
        );
        // Binary garbage with no %PDF- header takes the same path.
        let err2 = open_err(vec![0u8; 2048]);
        assert!(matches!(err2, PdfError::NotPdf), "2KB NUL: {err2}");
    }

    #[test]
    fn header_only_without_xref_is_truncated_not_xref_jargon() {
        // Has a valid %PDF- header but no startxref/trailer/objects: the
        // operator must hear "truncated or corrupt", never the raw
        // "'startxref' not found" jargon.
        let err = open_err(b"%PDF-1.7\n".to_vec());
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "header-only must yield MissingXref, got: {err}"
        );
        assert!(err.to_string().contains("truncated or corrupt"));
    }

    const TRANSPORT_HEADER: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\n\r\n";

    /// Bytes prepended to a finished file shift every object; the offsets
    /// are read relative to the header and the cross-reference chain is
    /// parsed as written (its free entry 0 survives, which a reconstruction
    /// from an object scan does not record).
    #[test]
    fn offsets_written_without_a_prefix_are_read_from_the_header() {
        let page = "<</Type /Page>>".to_owned();
        let mut bytes = TRANSPORT_HEADER.as_bytes().to_vec();
        bytes.extend_from_slice(&pdf_with_objects(&[(2, page)]));
        let doc = Document::from_bytes_owned(bytes).expect("prefixed file opens");
        assert_eq!(doc.base, TRANSPORT_HEADER.len());
        assert!(matches!(doc.xref.get(0), Some(XrefEntry::Free)));
        let obj = doc.get_object((2, 0)).expect("object 2 resolves");
        assert!(
            obj.as_dict()
                .is_some_and(|d| d.get(b"Type").and_then(Object::as_name) == Some(b"Page"))
        );
    }

    /// Offsets that count the bytes before the header are absolute, as the
    /// spec measures them, and are used unchanged.
    #[test]
    fn offsets_counting_a_prefix_are_absolute() {
        let page = "<</Type /Page>>".to_owned();
        let bytes = pdf_with_objects_after(TRANSPORT_HEADER, &[(2, page)]);
        let doc = Document::from_bytes_owned(bytes).expect("prefixed file opens");
        assert_eq!(doc.base, 0);
        assert!(matches!(doc.xref.get(0), Some(XrefEntry::Free)));
        assert!(doc.get_object((2, 0)).is_ok());
    }

    #[test]
    fn header_late_within_scan_window_is_accepted_as_pdf() {
        // The %PDF- signature need not be at byte 0; a few junk bytes are
        // tolerated (matches poppler/mutool). It still has no xref, so the
        // failure is the truncation diagnosis, *not* NotPdf.
        let mut bytes = b"\xEF\xBB\xBFleading junk\n".to_vec();
        bytes.extend_from_slice(b"%PDF-1.4\n");
        let err = open_err(bytes);
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "header-after-junk must pass the NotPdf gate, got: {err}"
        );
    }

    #[test]
    fn broken_xref_is_reconstructed_from_object_scan() {
        // A structurally complete body with NO xref table and NO startxref
        // at all. The strict parser cannot find it; reconstruction must
        // rebuild the table from the object headers and synthesise a
        // trailer pointing at the /Catalog so the document opens.
        let body = b"%PDF-1.4\n\
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n\
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
            .to_vec();
        let doc = Document::from_bytes_owned(body)
            .expect("reconstruction must recover a usable catalogue");
        assert_eq!(doc.page_count(), 1, "rebuilt page tree must yield 1 page");
    }

    #[test]
    fn bare_catalog_stub_without_pages_is_not_falsely_recovered() {
        // A truncated file whose only object is a Catalog with no /Pages is
        // genuinely unrecoverable; repair must decline and report the
        // truncation, not the misleading "document has no pages".
        let body = b"%PDF-1.7\n1 0 obj<</Type/Catalog>>endobj\n".to_vec();
        let err = open_err(body);
        assert!(
            matches!(err, PdfError::MissingXref(_)),
            "bare-Catalog stub must yield MissingXref, got: {err}"
        );
    }
}
