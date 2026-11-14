//! The HEIF item structure of an AVIF file (ISO/IEC 23008-12 over the box
//! syntax of ISO/IEC 14496-12): which items exist, where their bytes lie,
//! which properties describe them and how they refer to one another.
//!
//! Parsing borrows from the file and copies nothing but small tables.  Every
//! offset and length is checked against the buffer it indexes, every table
//! is capped across the whole file, each box of `meta` appears at most once,
//! every table entry costs input bytes, and box nesting is bounded by
//! construction: the parser
//! descends a fixed set of containers (`meta`, `iprp`, `ipco`, `iinf`,
//! `iref`) and never recurses on input.

use std::borrow::Cow;

use super::DecodeError;

/// The most items, item locations, properties, references and extents a
/// file may declare, each counted over the whole file; property
/// associations are capped at 16 times as many.
const MAX_ENTRIES: usize = 4096;

/// A four-character box or item type.
pub(super) type FourCc = [u8; 4];

/// The boxes of `meta` that ISO/IEC 23008-12 allows at most once.
const SINGLE: [&FourCc; 7] = [
    b"hdlr", b"pitm", b"iinf", b"iloc", b"iprp", b"iref", b"idat",
];

fn malformed(what: impl std::fmt::Display) -> DecodeError {
    DecodeError::Codec(format!("avif: {what}"))
}

/// A forward reader over a byte slice; every read fails at the end of the
/// slice instead of indexing past it.
#[derive(Clone, Copy)]
struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let (head, tail) = self
            .data
            .split_at_checked(n)
            .ok_or_else(|| malformed("a box ends inside a field"))?;
        self.data = tail;
        Ok(head)
    }

    /// Pass over `n` bytes.
    fn skip(&mut self, n: usize) -> Result<(), DecodeError> {
        self.take(n).map(drop)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, DecodeError> {
        self.array().map(u16::from_be_bytes)
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        self.array().map(u32::from_be_bytes)
    }

    fn u64(&mut self) -> Result<u64, DecodeError> {
        self.array().map(u64::from_be_bytes)
    }

    /// A big-endian unsigned integer of `size` bytes: 0 (absent, read as
    /// zero), 4 or 8.
    fn sized(&mut self, size: u8) -> Result<u64, DecodeError> {
        match size {
            0 => Ok(0),
            4 => self.u32().map(u64::from),
            8 => self.u64(),
            _ => Err(malformed("an iloc field size other than 0, 4 or 8")),
        }
    }

    /// An item identifier: 32 bits when `wide`, else 16.
    fn id(&mut self, wide: bool) -> Result<u32, DecodeError> {
        if wide {
            self.u32()
        } else {
            self.u16().map(u32::from)
        }
    }

    /// The version and flags that open a full box.
    fn full_header(&mut self) -> Result<(u8, u32), DecodeError> {
        let word = self.u32()?;
        Ok(((word >> 24) as u8, word & 0x00FF_FFFF))
    }

    /// The next box: its type and payload.  A box that declares more bytes
    /// than remain is an error unless `lenient`, which clips it to the
    /// remainder (the last box of a truncated file).
    fn next_box(&mut self, lenient: bool) -> Result<(FourCc, &'a [u8]), DecodeError> {
        let available = self.data.len();
        let size = self.u32()?;
        let kind: FourCc = self.array()?;
        let mut header = 8usize;
        let total = match size {
            0 => available,
            1 => {
                header += 8;
                usize::try_from(self.u64()?).unwrap_or(usize::MAX)
            }
            n => n as usize,
        };
        if kind == *b"uuid" {
            header += 16;
            self.skip(16)?;
        }
        let payload = total
            .checked_sub(header)
            .ok_or_else(|| malformed("a box shorter than its own header"))?;
        let payload = if lenient {
            payload.min(self.data.len())
        } else {
            payload
        };
        Ok((kind, self.take(payload)?))
    }
}

/// Where an item's bytes come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// Offsets are from the start of the file.
    File,
    /// Offsets are into the payload of the `idat` box.
    Idat,
}

/// One `iloc` entry.
#[derive(Debug)]
struct Location {
    item: u32,
    source: Source,
    base: u64,
    /// (offset from `base`, length); a zero length runs to the end of the
    /// source.
    extents: Vec<(u64, u64)>,
}

/// One `infe` entry.
#[derive(Clone, Copy, Debug)]
pub(super) struct ItemInfo {
    pub(super) id: u32,
    pub(super) kind: FourCc,
}

/// A property of `ipco`: its box type and payload.
#[derive(Clone, Copy, Debug)]
pub(super) struct Property<'a> {
    pub(super) kind: FourCc,
    pub(super) payload: &'a [u8],
    /// Whether the association marks the property essential: a reader that
    /// does not implement it must not show the item.
    pub(super) essential: bool,
}

/// One typed reference of `iref`.
#[derive(Debug)]
struct Reference {
    kind: FourCc,
    from: u32,
    to: Vec<u32>,
}

/// The parsed `meta` box of a file.
#[derive(Debug)]
pub(super) struct Container<'a> {
    file: &'a [u8],
    idat: &'a [u8],
    primary: Option<u32>,
    items: Vec<ItemInfo>,
    locations: Vec<Location>,
    /// The extents of all `locations`.
    extents: usize,
    properties: Vec<(FourCc, &'a [u8])>,
    /// (item, 1-based property index, essential), in association order.
    associations: Vec<(u32, u16, bool)>,
    references: Vec<Reference>,
}

/// Whether `bytes` opens with an `ftyp` box naming an AVIF or generic HEIF
/// image brand (`avif`, `avis`, `mif1`), as major or compatible brand.
pub(in crate::decode) fn is_avif(bytes: &[u8]) -> bool {
    let mut reader = Reader::new(bytes);
    let Ok((kind, payload)) = reader.next_box(true) else {
        return false;
    };
    if kind != *b"ftyp" {
        return false;
    }
    // major_brand, minor_version, compatible_brands[]: every 4-byte word
    // but the second is a brand.
    let (brands, _) = payload.as_chunks::<4>();
    brands
        .iter()
        .enumerate()
        .any(|(i, brand)| i != 1 && matches!(brand, b"avif" | b"avis" | b"mif1"))
}

impl<'a> Container<'a> {
    /// Parse the item structure of `file`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Codec`] when the file has no `meta` box with a `pict`
    /// handler, or a box of it is malformed or over the table caps.
    pub(super) fn parse(file: &'a [u8]) -> Result<Self, DecodeError> {
        let mut top = Reader::new(file);
        while !top.is_empty() {
            // A truncated file ends inside its last top-level box (usually
            // the media data); item extents are checked on their own.
            let (kind, payload) = top.next_box(true)?;
            if kind == *b"meta" {
                return Self::parse_meta(file, payload);
            }
        }
        Err(malformed("no meta box: not a still image"))
    }

    fn parse_meta(file: &'a [u8], payload: &'a [u8]) -> Result<Self, DecodeError> {
        let mut meta = Reader::new(payload);
        meta.skip(4)?;
        let mut out = Self {
            file,
            idat: &[],
            primary: None,
            items: Vec::new(),
            locations: Vec::new(),
            extents: 0,
            properties: Vec::new(),
            associations: Vec::new(),
            references: Vec::new(),
        };
        let mut pict = false;
        let mut seen = [false; SINGLE.len()];
        while !meta.is_empty() {
            let (kind, payload) = meta.next_box(false)?;
            if let Some(at) = SINGLE.iter().position(|&single| *single == kind)
                && std::mem::replace(&mut seen[at], true)
            {
                return Err(malformed(format_args!(
                    "a second '{}' box in meta",
                    kind.escape_ascii()
                )));
            }
            let mut body = Reader::new(payload);
            match &kind {
                b"hdlr" => {
                    // Version and flags, then `pre_defined`.
                    body.skip(8)?;
                    pict = body.array::<4>()? == *b"pict";
                }
                b"pitm" => {
                    let (version, _) = body.full_header()?;
                    out.primary = Some(body.id(version > 0)?);
                }
                b"iinf" => out.parse_iinf(body)?,
                b"iloc" => out.parse_iloc(body)?,
                b"iprp" => out.parse_iprp(body)?,
                b"iref" => out.parse_iref(body)?,
                b"idat" => out.idat = payload,
                _ => {}
            }
        }
        if !pict {
            return Err(malformed("the meta box does not describe pictures"));
        }
        Ok(out)
    }

    fn parse_iinf(&mut self, mut body: Reader<'a>) -> Result<(), DecodeError> {
        let (version, _) = body.full_header()?;
        let count = body.id(version > 0)? as usize;
        if count > MAX_ENTRIES - self.items.len() {
            return Err(malformed("too many items"));
        }
        for _ in 0..count {
            let (kind, payload) = body.next_box(false)?;
            if kind != *b"infe" {
                continue;
            }
            let mut entry = Reader::new(payload);
            let (version, _) = entry.full_header()?;
            if version < 2 {
                // Versions 0 and 1 carry no item type.
                continue;
            }
            let id = entry.id(version > 2)?;
            // `item_protection_index`.
            entry.skip(2)?;
            let kind = entry.array()?;
            self.items.push(ItemInfo { id, kind });
        }
        Ok(())
    }

    fn parse_iloc(&mut self, mut body: Reader<'a>) -> Result<(), DecodeError> {
        let (version, _) = body.full_header()?;
        if version > 2 {
            return Err(malformed("unsupported iloc version"));
        }
        let sizes = body.u16()?;
        let nibble = |shift: u16| ((sizes >> shift) & 0xF) as u8;
        let (offset_size, length_size, base_size) = (nibble(12), nibble(8), nibble(4));
        let index_size = if version > 0 { nibble(0) } else { 0 };
        let count = body.id(version == 2)? as usize;
        if count > MAX_ENTRIES - self.locations.len() {
            return Err(malformed("too many item locations"));
        }
        for _ in 0..count {
            let item = body.id(version == 2)?;
            let method = if version > 0 { body.u16()? & 0xF } else { 0 };
            let source = match method {
                0 => Source::File,
                1 => Source::Idat,
                _ => return Err(malformed("unsupported item construction method")),
            };
            if body.u16()? != 0 {
                return Err(malformed("item data in another file"));
            }
            let base = body.sized(base_size)?;
            let extent_count = usize::from(body.u16()?);
            if extent_count > MAX_ENTRIES - self.extents {
                return Err(malformed("too many extents"));
            }
            // Without a length field every extent runs to the end of its
            // source, so only a single extent is meaningful; refusing more
            // also makes every extent cost input bytes.
            if extent_count > 1 && length_size == 0 {
                return Err(malformed("several extents without lengths"));
            }
            self.extents += extent_count;
            let mut extents = Vec::with_capacity(extent_count);
            for _ in 0..extent_count {
                let _index = body.sized(index_size)?;
                let offset = body.sized(offset_size)?;
                let length = body.sized(length_size)?;
                extents.push((offset, length));
            }
            self.locations.push(Location {
                item,
                source,
                base,
                extents,
            });
        }
        Ok(())
    }

    fn parse_iprp(&mut self, mut body: Reader<'a>) -> Result<(), DecodeError> {
        while !body.is_empty() {
            let (kind, payload) = body.next_box(false)?;
            let mut inner = Reader::new(payload);
            match &kind {
                b"ipco" => {
                    while !inner.is_empty() {
                        if self.properties.len() >= MAX_ENTRIES {
                            return Err(malformed("too many properties"));
                        }
                        self.properties.push(inner.next_box(false)?);
                    }
                }
                b"ipma" => {
                    let (version, flags) = inner.full_header()?;
                    let entries = inner.u32()? as usize;
                    if entries > MAX_ENTRIES {
                        return Err(malformed("too many property associations"));
                    }
                    for _ in 0..entries {
                        let item = inner.id(version > 0)?;
                        for _ in 0..inner.u8()? {
                            let (essential, index) = if flags & 1 == 1 {
                                let v = inner.u16()?;
                                (v & 0x8000 != 0, v & 0x7FFF)
                            } else {
                                let v = inner.u8()?;
                                (v & 0x80 != 0, u16::from(v & 0x7F))
                            };
                            if self.associations.len() >= MAX_ENTRIES * 16 {
                                return Err(malformed("too many property associations"));
                            }
                            self.associations.push((item, index, essential));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_iref(&mut self, mut body: Reader<'a>) -> Result<(), DecodeError> {
        let (version, _) = body.full_header()?;
        while !body.is_empty() {
            let (kind, payload) = body.next_box(false)?;
            let mut inner = Reader::new(payload);
            let from = inner.id(version > 0)?;
            let count = usize::from(inner.u16()?);
            if count > MAX_ENTRIES || self.references.len() >= MAX_ENTRIES {
                return Err(malformed("too many item references"));
            }
            let to = (0..count)
                .map(|_| inner.id(version > 0))
                .collect::<Result<_, _>>()?;
            self.references.push(Reference { kind, from, to });
        }
        Ok(())
    }

    /// The primary item.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Codec`] when the file names none (an image sequence
    /// without a still image) or names one it does not describe.
    pub(super) fn primary(&self) -> Result<ItemInfo, DecodeError> {
        let id = self
            .primary
            .ok_or_else(|| malformed("no primary item: not a still image"))?;
        self.item(id)
    }

    /// The item of identifier `id`.
    pub(super) fn item(&self, id: u32) -> Result<ItemInfo, DecodeError> {
        self.items
            .iter()
            .find(|item| item.id == id)
            .copied()
            .ok_or_else(|| malformed("a referenced item is not described"))
    }

    /// The properties associated with item `id`, in association order.
    pub(super) fn properties(&self, id: u32) -> Result<Vec<Property<'a>>, DecodeError> {
        self.associations
            .iter()
            .filter(|&&(item, index, _)| item == id && index != 0)
            .map(|&(_, index, essential)| {
                let (kind, payload) = *self
                    .properties
                    .get(usize::from(index) - 1)
                    .ok_or_else(|| malformed("a property association past the property list"))?;
                Ok(Property {
                    kind,
                    payload,
                    essential,
                })
            })
            .collect()
    }

    /// The items the first reference of type `kind` from item `id` names,
    /// in order.
    pub(super) fn references(&self, id: u32, kind: FourCc) -> &[u32] {
        self.references
            .iter()
            .find(|r| r.from == id && r.kind == kind)
            .map_or(&[], |r| r.to.as_slice())
    }

    /// The bytes of item `id`: borrowed when one extent holds them, the
    /// concatenation of the extents otherwise.  `budget` caps the length of
    /// a concatenation.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Codec`] when the item has no location or an extent
    /// lies outside its source.
    pub(super) fn data(&self, id: u32, budget: usize) -> Result<Cow<'a, [u8]>, DecodeError> {
        let location = self
            .locations
            .iter()
            .find(|l| l.item == id)
            .ok_or_else(|| malformed("an item has no location"))?;
        let source = match location.source {
            Source::File => self.file,
            Source::Idat => self.idat,
        };
        let extent = |&(offset, length): &(u64, u64)| {
            let out_of_bounds = || malformed("an item extent lies outside the file");
            let start = location
                .base
                .checked_add(offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(out_of_bounds)?;
            let rest = source.get(start..).ok_or_else(out_of_bounds)?;
            if length == 0 {
                return Ok(rest);
            }
            usize::try_from(length)
                .ok()
                .and_then(|n| rest.get(..n))
                .ok_or_else(out_of_bounds)
        };
        match location.extents.as_slice() {
            // No extent: the whole source from the base offset.
            [] => extent(&(0, 0)).map(Cow::Borrowed),
            [one] => extent(one).map(Cow::Borrowed),
            many => {
                let mut out = Vec::new();
                for e in many {
                    let piece = extent(e)?;
                    if piece.len() > budget.saturating_sub(out.len()) {
                        return Err(malformed("item extents over the size budget"));
                    }
                    out.extend_from_slice(piece);
                }
                Ok(Cow::Owned(out))
            }
        }
    }
}

/// The payload of a full-box property after its version and flags.
fn full_body(payload: &[u8]) -> Result<Reader<'_>, DecodeError> {
    let mut reader = Reader::new(payload);
    reader.skip(4)?;
    Ok(reader)
}

/// `ispe`: the width and height of the item's reconstructed image.
pub(super) fn parse_ispe(payload: &[u8]) -> Result<(u32, u32), DecodeError> {
    let mut body = full_body(payload)?;
    Ok((body.u32()?, body.u32()?))
}

/// The colour description of a `colr` box of type `nclx`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Nclx {
    pub(super) primaries: u16,
    pub(super) matrix: u16,
    pub(super) full_range: bool,
}

/// `colr`: `Some` for an `nclx` colour description, `None` for an ICC
/// profile or an unknown colour type.
pub(super) fn parse_colr(payload: &[u8]) -> Result<Option<Nclx>, DecodeError> {
    let mut body = Reader::new(payload);
    if body.array::<4>()? != *b"nclx" {
        return Ok(None);
    }
    let primaries = body.u16()?;
    let _transfer = body.u16()?;
    let matrix = body.u16()?;
    let full_range = body.u8()? & 0x80 != 0;
    Ok(Some(Nclx {
        primaries,
        matrix,
        full_range,
    }))
}

/// `av1C`: check the codec configuration holds its fixed fields (marker
/// and version, profile and level, then the format flags).  The format it
/// declares is not used: the stream's sequence header is authoritative.
pub(super) fn parse_av1c(payload: &[u8]) -> Result<(), DecodeError> {
    Reader::new(payload).skip(3)
}

/// `irot`: the number of anticlockwise quarter turns.
pub(super) fn parse_irot(payload: &[u8]) -> Result<u8, DecodeError> {
    Ok(Reader::new(payload).u8()? & 3)
}

/// `imir`: the axis, 0 exchanging top and bottom, 1 left and right.
pub(super) fn parse_imir(payload: &[u8]) -> Result<u8, DecodeError> {
    Ok(Reader::new(payload).u8()? & 1)
}

/// `clap`: the clean aperture as (x, y, width, height) within an image of
/// `width × height`, or `None` when its edges are not on whole pixels or
/// lie outside the image.
pub(super) fn parse_clap(
    payload: &[u8],
    width: u32,
    height: u32,
) -> Result<Option<(u32, u32, u32, u32)>, DecodeError> {
    let mut body = Reader::new(payload);
    let mut fraction = || -> Result<(i64, i64), DecodeError> {
        // Offsets are signed numerators; sizes are unsigned and positive in
        // any valid box, so one signed reading serves both.
        let n = i64::from(body.u32()?.cast_signed());
        Ok((n, i64::from(body.u32()?)))
    };
    let (w, h, dx, dy) = (fraction()?, fraction()?, fraction()?, fraction()?);
    // Along one axis of `size` pixels, an aperture of `len` whose centre is
    // `off` from the image's starts at (size − len) / 2 + off.
    let axis = |size: u32, (len_n, len_d): (i64, i64), (off_n, off_d): (i64, i64)| {
        if len_d == 0 || off_d == 0 || len_n <= 0 || len_n % len_d != 0 {
            return None;
        }
        let len = len_n / len_d;
        // start = (size − len) / 2 + off_n / off_d, over the denominator
        // 2 · off_d.
        let numerator = (i64::from(size) - len)
            .checked_mul(off_d)?
            .checked_add(off_n.checked_mul(2)?)?;
        let denominator = off_d * 2;
        if numerator % denominator != 0 {
            return None;
        }
        let start = numerator / denominator;
        let end = start.checked_add(len)?;
        if start < 0 || end > i64::from(size) {
            return None;
        }
        Some((u32::try_from(start).ok()?, u32::try_from(len).ok()?))
    };
    Ok(axis(width, w, dx)
        .zip(axis(height, h, dy))
        .map(|((x, w), (y, h))| (x, y, w, h)))
}

/// `a1op`: the operating point the item selects.
pub(super) fn parse_a1op(payload: &[u8]) -> Result<u8, DecodeError> {
    Reader::new(payload).u8()
}

/// `lsel`: the layer the item selects; `0xFFFF` is all layers.
pub(super) fn parse_lsel(payload: &[u8]) -> Result<u16, DecodeError> {
    Reader::new(payload).u16()
}

/// The data of a `grid` item.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Grid {
    pub(super) rows: u32,
    pub(super) columns: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// Parse the `ImageGrid` structure a `grid` item carries as its data.
pub(super) fn parse_grid(data: &[u8]) -> Result<Grid, DecodeError> {
    let mut body = Reader::new(data);
    if body.u8()? != 0 {
        return Err(malformed("unsupported grid version"));
    }
    let wide = body.u8()? & 1 == 1;
    let rows = u32::from(body.u8()?) + 1;
    let columns = u32::from(body.u8()?) + 1;
    let (width, height) = (body.id(wide)?, body.id(wide)?);
    Ok(Grid {
        rows,
        columns,
        width,
        height,
    })
}
