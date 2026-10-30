//! Font engine — parses font programs and creates [`FontFace`] instances.
//!
//! A [`FontEngine`] assigns [`FaceId`]s and keeps every program it has
//! parsed, so faces built from identical bytes (one font resource drawn at
//! several sizes) or from the same file share one [`FontProgram`].  Its
//! program tables are plain data: callers that render pages in parallel give
//! each page its own engine, and engines whose faces share a glyph cache
//! share one [`FaceIds`] source.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::face::{FontFace, FontMatrix};
use crate::key::{FaceId, FaceIds};
use crate::program::FontProgram;

/// Parameters passed when loading a font face.
pub struct FaceParams {
    /// Glyph-index map: `code_to_gid[char_code]` → glyph index.  Pass an
    /// empty `Vec` to resolve codes through the program's charmap.
    pub code_to_gid: Vec<u32>,
    /// 2×2 matrix mapping em units to device pixels, y up (the PDF text
    /// rendering matrix without translation).
    pub mat: FontMatrix,
}

/// Parses font programs and assigns face IDs.
#[derive(Debug)]
pub struct FontEngine {
    /// Source of the identifiers of the faces this engine makes.
    ids: Arc<FaceIds>,
    /// Whether glyph bitmaps carry 8-bit coverage.
    aa: bool,
    /// Programs parsed from memory, keyed by a hash of their bytes.  Hash
    /// collisions are resolved by comparing the bytes.
    memory_programs: HashMap<u64, Vec<Arc<FontProgram>>>,
    /// Programs parsed from files, keyed by path and face index.
    file_programs: HashMap<(PathBuf, u32), Arc<FontProgram>>,
}

impl FontEngine {
    /// Create an engine whose faces render 8-bit coverage bitmaps when `aa`
    /// is `true` and 1-bit masks otherwise.
    #[must_use]
    pub fn new(aa: bool) -> Self {
        Self::with_face_ids(aa, Arc::default())
    }

    /// As [`Self::new`], drawing face identifiers from `ids`.
    #[must_use]
    pub fn with_face_ids(aa: bool, ids: Arc<FaceIds>) -> Self {
        Self {
            ids,
            aa,
            memory_programs: HashMap::new(),
            file_programs: HashMap::new(),
        }
    }

    /// Load a face from an in-memory font program (an embedded PDF font
    /// stream).  `face_index` selects the member of a TrueType collection.
    ///
    /// # Errors
    ///
    /// [`LoadError::UnsupportedFormat`] / [`LoadError::Malformed`] when the
    /// bytes are not a usable font program; [`LoadError::DegenerateSize`]
    /// when the matrix is degenerate.
    pub fn load_memory_face(
        &mut self,
        font_data: Vec<u8>,
        face_index: u32,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        let mut hasher = DefaultHasher::new();
        font_data.hash(&mut hasher);
        face_index.hash(&mut hasher);
        let key = hasher.finish();
        let cached = self
            .memory_programs
            .get(&key)
            .and_then(|bucket| bucket.iter().find(|p| p.data() == font_data.as_slice()))
            .cloned();
        let program = if let Some(program) = cached {
            program
        } else {
            let program = Arc::new(FontProgram::parse(font_data, face_index)?);
            self.memory_programs
                .entry(key)
                .or_default()
                .push(Arc::clone(&program));
            program
        };
        self.load_program_face(program, params)
    }

    /// Load a face from a font file.  `face_index` selects the member of a
    /// TrueType collection.
    ///
    /// # Errors
    ///
    /// [`LoadError::Io`] when the file cannot be read, otherwise as
    /// [`Self::load_memory_face`].
    pub fn load_file_face(
        &mut self,
        path: impl AsRef<Path>,
        face_index: u32,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        let key = (path.as_ref().to_path_buf(), face_index);
        let program = if let Some(program) = self.file_programs.get(&key) {
            Arc::clone(program)
        } else {
            let data = std::fs::read(&key.0).map_err(LoadError::Io)?;
            let program = Arc::new(FontProgram::parse(data, face_index)?);
            let _ = self.file_programs.insert(key, Arc::clone(&program));
            program
        };
        self.load_program_face(program, params)
    }

    /// Make a face of a program already parsed, possibly by another engine.
    ///
    /// # Errors
    ///
    /// [`LoadError::DegenerateSize`] when the matrix is degenerate.
    pub fn load_program_face(
        &self,
        program: Arc<FontProgram>,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        self.load_program_face_as(self.ids.next(), program, params)
    }

    /// As [`Self::load_program_face`], with the identifier `id`.  Two faces
    /// given one identifier must draw the same program at the same matrix:
    /// they share glyph bitmaps.
    ///
    /// # Errors
    ///
    /// [`LoadError::DegenerateSize`] when the matrix is degenerate.
    pub fn load_program_face_as(
        &self,
        id: FaceId,
        program: Arc<FontProgram>,
        params: FaceParams,
    ) -> Result<FontFace, LoadError> {
        FontFace::new(id, program, params, self.aa).ok_or(LoadError::DegenerateSize)
    }
}

/// Errors that can occur when loading a font face.
#[derive(Debug)]
pub enum LoadError {
    /// The font file could not be read.
    Io(std::io::Error),
    /// The bytes match none of the supported font formats.
    UnsupportedFormat,
    /// The bytes look like a supported format but cannot be parsed or carry
    /// no outlines.
    Malformed(String),
    /// The face matrix is degenerate (non-finite, or a pixel size outside
    /// `[1, 65535]`).
    DegenerateSize,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "font file unreadable: {e}"),
            Self::UnsupportedFormat => f.write_str("unrecognised font program format"),
            Self::Malformed(why) => write!(f, "malformed font program: {why}"),
            Self::DegenerateSize => f.write_str("font face has a degenerate size matrix"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
