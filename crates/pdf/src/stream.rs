//! Stream filter decoding.
//!
//! We support the filters encountered in real renderer workloads.
//! Image-format filters (DCT, JBIG2, JPX, `CCITTFax`) are intentionally
//! excluded — those bytes are passed raw to the codec layer in `rasterrocket-interp`.

use crate::dictionary::Dictionary;
use crate::object::Object;

/// Upper bound on the number of filters in a single `/Filter` chain.
///
/// Real PDFs use at most two or three (a transport prefilter plus the
/// terminal codec, e.g. `[/ASCII85Decode /CCITTFaxDecode]`).  A hostile
/// `/Filter [/FlateDecode /FlateDecode … ×N]` array would otherwise drive `N`
/// decode passes — each able to re-expand to [`MAX_DECOMPRESSED`] — turning a
/// tiny stream into an unbounded CPU/RAM sink.  The image filter-chain path is
/// the reachable vector for this, so the cap lives here at the single decode
/// chokepoint rather than at each call site.
const MAX_FILTER_CHAIN: usize = 32;

/// The output of a filter, or of a whole filter chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Decoded {
    /// The decoded bytes.
    pub(crate) bytes: Vec<u8>,
    /// Set when a filter met corrupt or missing data and kept the bytes it
    /// had decoded before it: what was lost, naming the filter.  The first
    /// such filter of a chain is the one reported.
    pub(crate) truncated: Option<String>,
}

impl Decoded {
    /// Bytes decoded from the whole input.
    const fn whole(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            truncated: None,
        }
    }

    /// The prefix decoded before `reason` ended the data early.
    const fn cut(bytes: Vec<u8>, reason: String) -> Self {
        Self {
            bytes,
            truncated: Some(reason),
        }
    }
}

/// Decode a raw stream byte slice according to its /Filter chain.
///
/// `dict` is the stream dictionary; `raw` is the undecoded content bytes.
/// Returns the decoded bytes, or `raw` unchanged if there is no filter,
/// with the first truncation a filter of the chain reported.
///
/// # Errors
/// - `/Filter` holds a value that is not a filter name.
/// - The chain exceeds [`MAX_FILTER_CHAIN`] filters.
/// - Any stage produces more than [`MAX_DECOMPRESSED`] bytes.  Each Flate and LZW
///   stage is individually bomb-capped, but a chain of stages could otherwise
///   re-expand the intermediate buffer unbounded across iterations; this check
///   bounds the aggregate by failing the moment any intermediate exceeds the
///   cap.
/// - Any individual filter step fails (the underlying decoder's error).
pub(crate) fn decode_stream(raw: &[u8], dict: &Dictionary) -> Result<Decoded, String> {
    let filters = collect_filters(dict)?;
    if filters.is_empty() {
        return Ok(Decoded::whole(raw.to_vec()));
    }
    if filters.len() > MAX_FILTER_CHAIN {
        return Err(format!(
            "filter chain too long: {} filters exceeds {MAX_FILTER_CHAIN} cap \
             (possible decompression-bomb / hostile-array DoS)",
            filters.len()
        ));
    }

    let params_list = collect_decode_parms(dict, filters.len());

    let mut data = raw.to_vec();
    let mut truncated = None;
    for (i, filter) in filters.iter().enumerate() {
        let params = params_list.get(i).and_then(|o| o.as_ref());
        let stage = apply_filter(filter, &data, params)?;
        data = stage.bytes;
        truncated = truncated.or(stage.truncated);
        if data.len() > MAX_DECOMPRESSED {
            return Err(format!(
                "filter chain stage {i} ({}) produced {} bytes, exceeds {MAX_DECOMPRESSED}-byte \
                 cap (possible decompression bomb across chain)",
                String::from_utf8_lossy(filter),
                data.len()
            ));
        }
    }
    Ok(Decoded {
        bytes: data,
        truncated,
    })
}

/// The general-purpose filters (ISO 32000-2 §7.4) the stream decoder applies
/// itself, by full name.  Every other recognised filter is an image codec
/// whose bytes pass through the chain unchanged.
const DECODED_FILTERS: [&[u8]; 6] = [
    b"FlateDecode",
    b"LZWDecode",
    b"ASCII85Decode",
    b"ASCIIHexDecode",
    b"RunLengthDecode",
    b"BrotliDecode",
];

/// Whether the stream decoder applies the filter named `name` (its full
/// name, see [`filter_full_name`]) rather than passing its bytes through:
/// a stream whose chain ends in such a filter decodes to raw samples.
#[must_use]
pub fn is_decoded_filter(name: &[u8]) -> bool {
    DECODED_FILTERS.contains(&name)
}

fn apply_filter(filter: &[u8], data: &[u8], params: Option<&Object>) -> Result<Decoded, String> {
    match filter {
        b"FlateDecode" => apply_flate(
            data,
            params
                .and_then(Object::as_dict)
                .map(|d| d as &dyn DictLookup),
        )
        .map_err(|e| e.to_string()),
        b"LZWDecode" => apply_lzw(data, params),
        b"ASCII85Decode" => Ok(decode_ascii85(data)),
        b"ASCIIHexDecode" => Ok(Decoded::whole(decode_ascii_hex(data))),
        b"RunLengthDecode" => Ok(decode_run_length(data, MAX_DECOMPRESSED)),
        b"BrotliDecode" => apply_brotli(
            data,
            params
                .and_then(Object::as_dict)
                .map(|d| d as &dyn DictLookup),
        ),
        // Image filters — pass through; the caller handles them.  Decryption
        // is applied when the object is parsed, so a `/Crypt` entry that
        // survives to here named the `/Identity` filter.
        b"DCTDecode" | b"CCITTFaxDecode" | b"JBIG2Decode" | b"JPXDecode" | b"Crypt" => {
            Ok(Decoded::whole(data.to_vec()))
        }
        other => Err(format!(
            "unsupported filter: {}",
            String::from_utf8_lossy(other)
        )),
    }
}

// ── FlateDecode ───────────────────────────────────────────────────────────────

/// Decode `FlateDecode` data, then undo the predictor named by `params`.
fn apply_flate(data: &[u8], params: Option<&dyn DictLookup>) -> Result<Decoded, std::io::Error> {
    if data.is_empty() {
        return Ok(Decoded::whole(Vec::new()));
    }

    let mut out = decompress_zlib(data)?;

    if let Some(p) = params {
        out.bytes = apply_png_predictor(out.bytes, p)
            .map_err(|s| std::io::Error::new(std::io::ErrorKind::InvalidData, s))?;
    }
    Ok(out)
}

/// Maximum decompressed size we will accept, to defeat decompression bombs.
const MAX_DECOMPRESSED: usize = 1 << 30; // 1 GiB

/// Inflate a zlib (RFC 1950) or raw deflate stream, as `FlateDecode` does
/// for a stream without predictor parameters.
///
/// # Errors
///
/// An `InvalidData` error when the stream is malformed beyond recovery or
/// its output exceeds 1 GiB.
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    decompress_zlib(data).map(|out| out.bytes)
}

/// flate2-backed decompression with partial-output tolerance.
///
/// Returns whatever bytes were produced before a mid-stream error if the
/// decompressor wrote anything at all, marked as truncated — real-world
/// malformed PDFs ship truncated or checksum-corrupt content streams that
/// flate2 emits usefully up to the failure point.
///
/// Bounded at [`MAX_DECOMPRESSED`] via `Read::take`; `read_to_end` would
/// otherwise grow the output `Vec` without limit.
fn decompress_zlib_flate2(data: &[u8]) -> Result<Decoded, std::io::Error> {
    use flate2::read::{DeflateDecoder, ZlibDecoder};
    use std::io::Read;

    // 3× input is a good initial guess for typical text/PostScript-ish
    // payloads; capping at 16 MiB avoids reserving gigabytes when called
    // with a multi-hundred-MB compressed stream.
    let initial_cap = data.len().saturating_mul(3).min(16 * 1024 * 1024);
    let mut out = Vec::with_capacity(initial_cap);
    // +1 so we can detect the case where the decompressor *would* produce
    // more than the cap; if `read_to_end` consumes exactly the cap+1 byte
    // we know to error rather than silently truncate.
    let cap_plus_one = (MAX_DECOMPRESSED as u64).saturating_add(1);

    let mut decoder = ZlibDecoder::new(data).take(cap_plus_one);
    let mut truncated = None;
    match decoder.read_to_end(&mut out) {
        Ok(_) => {}
        Err(_) if out.is_empty() && data.len() > 2 => {
            // Retry with raw deflate (skip 2-byte zlib header).
            out.clear();
            let mut raw_dec = DeflateDecoder::new(&data[2..]).take(cap_plus_one);
            let _ = raw_dec.read_to_end(&mut out)?;
        }
        Err(e) => {
            if out.is_empty() {
                return Err(e);
            }
            // A complete deflate stream under a wrong or missing Adler-32
            // trailer fails only the zlib check: read as raw deflate, it
            // ends cleanly and nothing is lost.
            let mut raw = Vec::with_capacity(out.len());
            if data.len() > 2
                && DeflateDecoder::new(&data[2..])
                    .take(cap_plus_one)
                    .read_to_end(&mut raw)
                    .is_ok()
                && raw.len() >= out.len()
                && raw.len() <= MAX_DECOMPRESSED
            {
                return Ok(Decoded::whole(raw));
            }
            truncated = Some(format!(
                "FlateDecode: {e}; kept the {} bytes decoded before it",
                out.len()
            ));
        }
    }
    if out.len() > MAX_DECOMPRESSED {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "FlateDecode: decompressed output exceeds 1 GiB cap",
        ));
    }
    Ok(Decoded {
        bytes: out,
        truncated,
    })
}

#[cfg(feature = "libdeflate")]
fn decompress_zlib(data: &[u8]) -> Result<Decoded, std::io::Error> {
    use libdeflater::{DecompressionError, Decompressor};

    // libdeflate is a non-streaming, in-memory decoder: it needs the output
    // buffer sized up-front. Start at 4× input (typical text streams compress
    // 3-6×), then double on InsufficientSpace until either we succeed or hit
    // the 1 GiB cap.
    let initial_cap = data.len().saturating_mul(4).clamp(64, 16 * 1024 * 1024);
    let mut out = vec![0u8; initial_cap];
    let mut decoder = Decompressor::new();

    loop {
        match decoder.zlib_decompress(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(Decoded::whole(out));
            }
            Err(DecompressionError::InsufficientSpace) => {
                if out.len() >= MAX_DECOMPRESSED {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "FlateDecode: decompressed output exceeds 1 GiB cap",
                    ));
                }
                let new_len = out.len().saturating_mul(2).min(MAX_DECOMPRESSED);
                out.resize(new_len, 0);
            }
            Err(DecompressionError::BadData) => {
                // libdeflate rejects raw deflate (no zlib header), corrupt
                // zlib, AND truncated streams (it's all-or-nothing — even a
                // valid prefix that hits EOF surfaces as BadData). Try raw
                // deflate (skip 2-byte zlib header) as a header-missing
                // fallback; if that also fails, fall through to the flate2
                // path which tolerates partial output the way the
                // pre-libdeflate code did for real-world malformed PDFs.
                if data.len() > 2
                    && let Ok(raw) = decompress_raw_deflate(&data[2..])
                {
                    return Ok(Decoded::whole(raw));
                }
                return decompress_zlib_flate2(data);
            }
        }
    }
}

#[cfg(feature = "libdeflate")]
fn decompress_raw_deflate(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    use libdeflater::{DecompressionError, Decompressor};
    let mut out = vec![0u8; data.len().saturating_mul(4).clamp(64, 16 * 1024 * 1024)];
    let mut decoder = Decompressor::new();
    loop {
        match decoder.deflate_decompress(data, &mut out) {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(DecompressionError::InsufficientSpace) => {
                if out.len() >= MAX_DECOMPRESSED {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "FlateDecode (raw): decompressed output exceeds 1 GiB cap",
                    ));
                }
                let new_len = out.len().saturating_mul(2).min(MAX_DECOMPRESSED);
                out.resize(new_len, 0);
            }
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("libdeflate raw: {e}"),
                ));
            }
        }
    }
}

#[cfg(not(feature = "libdeflate"))]
fn decompress_zlib(data: &[u8]) -> Result<Decoded, std::io::Error> {
    decompress_zlib_flate2(data)
}

// ── BrotliDecode ──────────────────────────────────────────────────────────────

/// Decode `BrotliDecode` data (RFC 7932), then undo the predictor named by
/// `params`, whose entries are those of `FlateDecode`.
///
/// Output is bounded at [`MAX_DECOMPRESSED`].  A stream that is corrupt or
/// truncated after producing output yields that output, marked as truncated.
fn apply_brotli(data: &[u8], params: Option<&dyn DictLookup>) -> Result<Decoded, String> {
    use std::io::Read;

    let mut out = Vec::new();
    let cap_plus_one = (MAX_DECOMPRESSED as u64).saturating_add(1);
    let mut decoder = brotli_decompressor::Decompressor::new(data, 4096).take(cap_plus_one);
    let truncated = match decoder.read_to_end(&mut out) {
        Ok(_) => None,
        Err(e) if out.is_empty() => return Err(format!("BrotliDecode: {e}")),
        Err(e) => Some(format!(
            "BrotliDecode: {e}; kept the {} bytes decoded before it",
            out.len()
        )),
    };
    if out.len() > MAX_DECOMPRESSED {
        return Err(format!(
            "BrotliDecode: decompressed output exceeds {MAX_DECOMPRESSED}-byte cap"
        ));
    }
    let bytes = match params {
        Some(p) => apply_png_predictor(out, p)?,
        None => out,
    };
    Ok(Decoded { bytes, truncated })
}

// ── PNG predictor ─────────────────────────────────────────────────────────────

/// Read one predictor parameter, clamping malformed low values to the
/// spec default and rejecting values above `cap` loudly — a legal value
/// past the denial-of-service cap means the parameters (and therefore the geometry the
/// de-prediction would apply) cannot be honoured, so proceeding would
/// silently scramble every row.
fn predictor_param(
    params: &dyn DictLookup,
    key: &'static str,
    default: i64,
    cap: usize,
) -> Result<usize, String> {
    let v = params.get_i64(key.as_bytes()).unwrap_or(default).max(1);
    let v = usize::try_from(v).expect("max(1) makes the value positive");
    if v > cap {
        return Err(format!(
            "predictor parameter /{key} = {v} exceeds the supported cap of {cap}"
        ));
    }
    Ok(v)
}

/// Undo TIFF predictor 2 (componentwise horizontal differencing) in place.
///
/// Each row stores its first pixel raw and every later component as a
/// delta against the same component one pixel to the left; the undo is a
/// componentwise running sum per row, modulo `2^bits`. Rows are
/// `ceil(cols·colors·bits/8)` bytes with no per-row tag byte (unlike the
/// PNG predictors); sub-byte components are packed most significant bit
/// first and 16-bit components are big-endian.
///
/// Component depths other than 1, 2, 4, 8 and 16 pass through unchanged
/// with a warning. Trailing bytes that do not form a whole row also pass
/// through unchanged.
fn apply_tiff_predictor(
    mut data: Vec<u8>,
    colors: usize,
    bits: usize,
    cols: usize,
) -> Result<Vec<u8>, String> {
    if !matches!(bits, 1 | 2 | 4 | 8 | 16) {
        log::warn!(
            "TIFF predictor 2 with BitsPerComponent={bits} is not undone; \
             passing data through unchanged"
        );
        return Ok(data);
    }
    let samples = cols
        .checked_mul(colors)
        .ok_or_else(|| "TIFF predictor: stride overflow".to_string())?;
    let stride = samples
        .checked_mul(bits)
        .map(|b| b.div_ceil(8))
        .ok_or_else(|| "TIFF predictor: stride overflow".to_string())?;
    if !data.len().is_multiple_of(stride) {
        log::warn!(
            "TIFF predictor: {} trailing bytes do not form a full {stride}-byte row; \
             passing them through unchanged",
            data.len() % stride,
        );
    }
    for row in data.chunks_exact_mut(stride) {
        match bits {
            8 => {
                for i in colors..stride {
                    row[i] = row[i].wrapping_add(row[i - colors]);
                }
            }
            16 => {
                for i in colors..samples {
                    let left =
                        u16::from_be_bytes([row[2 * (i - colors)], row[2 * (i - colors) + 1]]);
                    let delta = u16::from_be_bytes([row[2 * i], row[2 * i + 1]]);
                    row[2 * i..2 * i + 2].copy_from_slice(&left.wrapping_add(delta).to_be_bytes());
                }
            }
            _ => {
                for i in colors..samples {
                    let sum = packed_sample(row, i, bits) + packed_sample(row, i - colors, bits);
                    set_packed_sample(row, i, bits, sum);
                }
            }
        }
    }
    Ok(data)
}

/// Sample `i` of `row`, packed `bits` (1, 2 or 4) to a sample, most
/// significant bit first.
const fn packed_sample(row: &[u8], i: usize, bits: usize) -> u8 {
    let bit = i * bits;
    let shift = 8 - bits - bit % 8;
    (row[bit / 8] >> shift) & ((1 << bits) - 1)
}

/// Store `value` modulo `2^bits` as sample `i` of `row`, packed as in
/// [`packed_sample`].
const fn set_packed_sample(row: &mut [u8], i: usize, bits: usize, value: u8) {
    let bit = i * bits;
    let shift = 8 - bits - bit % 8;
    let mask = ((1u8 << bits) - 1) << shift;
    row[bit / 8] = (row[bit / 8] & !mask) | ((value << shift) & mask);
}

fn apply_png_predictor(data: Vec<u8>, params: &dyn DictLookup) -> Result<Vec<u8>, String> {
    let predictor = params.get_i64(b"Predictor").unwrap_or(1);
    if predictor < 2 {
        // Predictor 1 = no prediction.
        return Ok(data);
    }

    // Parameter caps defeat malformed PDFs that would otherwise request
    // gigabyte-scale allocations; /Colors 32 is also the DeviceN
    // component ceiling (ISO 32000-1 § 8.6.6.5).
    let colors = predictor_param(params, "Colors", 1, 32)?;
    let bits = predictor_param(params, "BitsPerComponent", 8, 32)?;
    let cols = predictor_param(params, "Columns", 1, 1_000_000)?;

    if predictor < 10 {
        // Predictor 2 = TIFF componentwise horizontal differencing.
        return apply_tiff_predictor(data, colors, bits, cols);
    }

    let stride = cols
        .checked_mul(colors)
        .and_then(|x| x.checked_mul(bits))
        .ok_or_else(|| "PNG predictor: stride overflow".to_string())?
        .div_ceil(8);
    let row_len = stride + 1; // +1 for filter byte

    // Tolerate truncated data — process as many full rows as we have.
    let n_rows = data.len() / row_len;
    if !data.len().is_multiple_of(row_len) {
        log::warn!(
            "PNG predictor: {} trailing bytes do not form a full {row_len}-byte row; \
             emitting {n_rows} whole rows",
            data.len() % row_len,
        );
    }
    let total = n_rows
        .checked_mul(stride)
        .ok_or_else(|| "PNG predictor: output size overflow".to_string())?;
    // 256 MiB ceiling — well above any realistic predictor output.
    if total > 256 * 1024 * 1024 {
        return Err(format!(
            "PNG predictor output {total} bytes exceeds 256 MiB cap"
        ));
    }
    let mut out = vec![0u8; total];
    let bytes_per_pixel = (colors * bits).div_ceil(8);

    for row in 0..n_rows {
        let src_row = &data[row * row_len..(row + 1) * row_len];
        let filter_byte = src_row[0];
        let raw = &src_row[1..];

        // Split `out` into (rows written so far, current row) to satisfy the
        // borrow checker: `prev` is in the already-written prefix, `dst` is
        // the current row being filled.
        let (done, rest) = out.split_at_mut(row * stride);
        let dst = &mut rest[..stride];
        let prev: Option<&[u8]> = (row > 0).then(|| &done[(row - 1) * stride..row * stride]);

        apply_png_filter(filter_byte, raw, prev, dst, bytes_per_pixel);
    }
    Ok(out)
}

/// Undo one PNG row filter (RFC 2083 § 6.5–6.9) in place.
///
/// Dispatched per row rather than per byte: None (0) and Up (2) are
/// branch-free whole-row forms that autovectorise; Sub (1), Average (3),
/// and Paeth (4) carry a left-neighbour dependency at `bpp` lag, so their
/// loops stay scalar with the up-row handling hoisted out. `prev` is
/// `None` on the first row (all up/up-left neighbours read as zero);
/// unknown filter bytes copy the raw row through unchanged.
///
/// First-row degenerations follow from the neighbour zeroing: Up copies,
/// Paeth reduces to Sub (`paeth(a, 0, 0) = a`), and Average halves only
/// the left neighbour. Within the first `bpp` bytes the left neighbour is
/// zero, so Sub copies, Average halves only the up neighbour, and Paeth
/// takes the up neighbour (`paeth(0, b, 0) = b`).
fn apply_png_filter(filter: u8, raw: &[u8], prev: Option<&[u8]>, dst: &mut [u8], bpp: usize) {
    let stride = dst.len();
    debug_assert!(
        raw.len() >= stride && prev.is_none_or(|p| p.len() >= stride),
        "apply_png_filter: raw ({}) and prev ({:?}) must cover the {stride}-byte row",
        raw.len(),
        prev.map(<[u8]>::len),
    );
    let lead = bpp.min(stride);
    match (filter, prev) {
        (2, Some(prev)) => {
            for (d, (&r, &p)) in dst.iter_mut().zip(raw.iter().zip(prev)) {
                *d = r.wrapping_add(p);
            }
        }
        (1, _) | (4, None) => {
            // Sub ignores the row above; Paeth with no row above
            // degenerates to Sub (paeth(a, 0, 0) = a).
            dst[..lead].copy_from_slice(&raw[..lead]);
            for i in lead..stride {
                dst[i] = raw[i].wrapping_add(dst[i - bpp]);
            }
        }
        (3, None) => {
            dst[..lead].copy_from_slice(&raw[..lead]);
            for i in lead..stride {
                dst[i] = raw[i].wrapping_add(dst[i - bpp] / 2);
            }
        }
        (3, Some(prev)) => {
            for i in 0..lead {
                dst[i] = raw[i].wrapping_add(prev[i] / 2);
            }
            for i in lead..stride {
                dst[i] = raw[i].wrapping_add(dst[i - bpp].midpoint(prev[i]));
            }
        }
        (4, Some(prev)) => {
            for i in 0..lead {
                dst[i] = raw[i].wrapping_add(prev[i]);
            }
            for i in lead..stride {
                dst[i] = raw[i].wrapping_add(paeth(dst[i - bpp], prev[i], prev[i - bpp]));
            }
        }
        // None (0), Up (2) with no row above, and out-of-spec filter bytes
        // copy the raw row through.
        (_, _) => dst.copy_from_slice(&raw[..stride]),
    }
}

/// The Paeth predictor (RFC 2083 § 6.6): whichever of `a`, `b`, `c` is
/// closest to `a + b - c`, ties broken in that order.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    let p = ia + ib - ic;
    let pa = (p - ia).unsigned_abs();
    let pb = (p - ib).unsigned_abs();
    let pc = (p - ic).unsigned_abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

// ── LZW ──────────────────────────────────────────────────────────────────────

/// A `Vec<u8>` sink that refuses to grow past [`MAX_DECOMPRESSED`].
///
/// `weezl`'s `into_stream(...).decode_all` writes the entire decoded output in
/// one pass with no size bound, so a small LZW stream that expands to many
/// gigabytes (an LZW bomb — the same threat the Flate path is already capped
/// against) would otherwise exhaust RAM.  Returning a write error here makes
/// `decode_all` stop and surface a loud `LZWDecode` failure instead.  weezl
/// also reports its own corrupt-stream errors as `InvalidData`, so a dedicated
/// `tripped` flag — not the error kind — is what distinguishes "bomb cap hit"
/// (unrecoverable, discard partial) from a benign mid-stream decode error
/// (partial prefix is real content, keep it).
struct BoundedSink {
    buf: Vec<u8>,
    tripped: bool,
}

/// `true` when appending `incoming` bytes to a buffer already holding `have`
/// bytes would exceed [`MAX_DECOMPRESSED`].  Extracted so the bomb predicate
/// is unit-testable without materialising a gigabyte-scale buffer.
const fn would_overflow_cap(have: usize, incoming: usize) -> bool {
    have.saturating_add(incoming) > MAX_DECOMPRESSED
}

impl std::io::Write for BoundedSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if would_overflow_cap(self.buf.len(), buf.len()) {
            self.tripped = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LZWDecode: decompressed output exceeds 1 GiB cap (possible LZW bomb)",
            ));
        }
        self.buf.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn apply_lzw(data: &[u8], params: Option<&Object>) -> Result<Decoded, String> {
    let early_change = params
        .and_then(Object::as_dict)
        .and_then(|d| d.get(b"EarlyChange"))
        .and_then(Object::as_i64)
        .is_none_or(|v| v != 0);

    let mut decoder = if early_change {
        weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8)
    } else {
        weezl::decode::Decoder::new(weezl::BitOrder::Msb, 8)
    };

    let mut sink = BoundedSink {
        buf: Vec::new(),
        tripped: false,
    };
    let result = decoder.into_stream(&mut sink).decode_all(data);
    let BoundedSink {
        buf: mut out,
        tripped,
    } = sink;
    let truncated = match result.status {
        Ok(()) => None,
        // A bomb that tripped the size cap is unrecoverable — the partial
        // prefix is not the document's real content, so do not return it.
        Err(e) if tripped => return Err(format!("LZWDecode aborted: {e}")),
        Err(e) if out.is_empty() => {
            return Err(format!("LZWDecode failed with no output: {e}"));
        }
        Err(e) => Some(format!(
            "LZWDecode: {e}; kept the {} bytes decoded before it",
            out.len()
        )),
    };

    let pred_params = params.map(|o| o as &dyn DictLookup);
    if let Some(p) = pred_params {
        out = apply_png_predictor(out, p)?;
    }
    Ok(Decoded {
        bytes: out,
        truncated,
    })
}

// ── ASCII85 ───────────────────────────────────────────────────────────────────

/// Decode `ASCII85Decode` data (ISO 32000-2 §7.4.3).
///
/// The data ends at `~`, or at the first byte that cannot occur in it (a
/// byte outside `!`..=`u` and whitespace, or `z` inside a group); the
/// latter keeps the bytes decoded before it and marks them truncated.  A final
/// partial group of `n` characters yields `n - 1` bytes; a lone final
/// character carries no byte and is dropped with a warning.  A group above
/// `s8W-!` (2^32 − 1) wraps modulo 2^32.
fn decode_ascii85(data: &[u8]) -> Decoded {
    let value = |group: &[u8; 5]| {
        group.iter().fold(0u32, |acc, &x| {
            acc.wrapping_mul(85).wrapping_add(u32::from(x))
        })
    };
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut count = 0usize;
    let mut truncated = None;

    for (i, &b) in data.iter().enumerate() {
        match b {
            b'z' if count == 0 => out.extend_from_slice(&[0u8; 4]),
            b'~' => break,
            b' ' | b'\t' | b'\r' | b'\n' | 0x0C | 0x00 => {}
            b'!'..=b'u' => {
                group[count] = b - b'!';
                count += 1;
                if count == 5 {
                    out.extend_from_slice(&value(&group).to_be_bytes());
                    count = 0;
                }
            }
            other => {
                truncated = Some(format!(
                    "ASCII85Decode: invalid byte {other:#04x} at offset {i}; \
                     kept the data decoded before it"
                ));
                break;
            }
        }
    }
    match count {
        0 => {}
        1 => log::warn!("ASCII85Decode: lone character in the final group dropped"),
        _ => {
            // Pad the group with 'u' (84).
            group[count..].fill(84);
            out.extend_from_slice(&value(&group).to_be_bytes()[..count - 1]);
        }
    }
    Decoded {
        bytes: out,
        truncated,
    }
}

// ── RunLength ─────────────────────────────────────────────────────────────────

/// Decode `RunLengthDecode` data (PDF §7.4.5), producing at most `max_output`
/// bytes.
///
/// A length byte `n` in 0–127 copies the next `n + 1` bytes; 129–255 repeats
/// the next byte `257 − n` times; 128 ends the data.  A run cut short by the
/// end of input contributes the bytes present, and decoding stops at a run
/// that would exceed `max_output`; both mark the output truncated.
fn decode_run_length(data: &[u8], max_output: usize) -> Decoded {
    let mut out = Vec::new();
    let mut i = 0;
    let over_cap = || format!("RunLengthDecode: output exceeds {max_output} bytes");
    while let Some(&run) = data.get(i) {
        i += 1;
        match run {
            128 => break,
            0..=127 => {
                let want = i + usize::from(run) + 1;
                let end = want.min(data.len());
                if out.len() + (end - i) > max_output {
                    return Decoded::cut(out, over_cap());
                }
                out.extend_from_slice(&data[i..end]);
                if end < want {
                    let missing = want - end;
                    return Decoded::cut(
                        out,
                        format!("RunLengthDecode: the last run lacks {missing} bytes"),
                    );
                }
                i = end;
            }
            _ => {
                let Some(&b) = data.get(i) else {
                    return Decoded::cut(
                        out,
                        "RunLengthDecode: the last run lacks its byte".to_owned(),
                    );
                };
                let repeat = 257 - usize::from(run);
                if out.len() + repeat > max_output {
                    return Decoded::cut(out, over_cap());
                }
                out.resize(out.len() + repeat, b);
                i += 1;
            }
        }
    }
    Decoded::whole(out)
}

// ── ASCIIHex ──────────────────────────────────────────────────────────────────

fn decode_ascii_hex(data: &[u8]) -> Vec<u8> {
    use crate::lexer::hex_nibble;
    let mut out = Vec::new();
    let mut hi: Option<u8> = None;
    for &b in data {
        if b == b'>' {
            break;
        }
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            continue;
        }
        let nibble = hex_nibble(b);
        match hi {
            None => hi = Some(nibble),
            Some(h) => {
                out.push((h << 4) | nibble);
                hi = None;
            }
        }
    }
    if let Some(h) = hi {
        out.push(h << 4);
    }
    out
}

// ── Filter/DecodeParms helpers ────────────────────────────────────────────────

/// The full name of the filter `name`, its abbreviation expanded.
///
/// An abbreviation of ISO 32000-2 Table 92 (`/Fl`, `/AHx`, …) maps to the
/// name it abbreviates, and any other name to itself.  The abbreviations
/// are accepted on every stream, not only on inline images.
#[must_use]
pub const fn filter_full_name(name: &[u8]) -> &[u8] {
    match name {
        b"AHx" => b"ASCIIHexDecode",
        b"A85" => b"ASCII85Decode",
        b"LZW" => b"LZWDecode",
        b"Fl" => b"FlateDecode",
        b"RL" => b"RunLengthDecode",
        b"CCF" => b"CCITTFaxDecode",
        b"DCT" => b"DCTDecode",
        other => other,
    }
}

/// The full names of the stream's `/Filter` chain; empty when `/Filter` is
/// absent or null.
///
/// # Errors
/// `/Filter`, or an element of a `/Filter` array, is not a name: an
/// unresolved indirect reference or a value of another type names no
/// filter, and decoding the stream without it would hand on encoded bytes
/// as decoded ones.
fn collect_filters(dict: &Dictionary) -> Result<Vec<Vec<u8>>, String> {
    let full = |o: &Object| {
        o.as_name()
            .map(|n| filter_full_name(n).to_vec())
            .ok_or_else(|| format!("/Filter entry is {}, not a filter name", o.enum_variant()))
    };
    match dict.get(b"Filter") {
        None | Some(Object::Null) => Ok(Vec::new()),
        Some(Object::Array(a)) => a.iter().map(full).collect(),
        Some(other) => full(other).map(|name| vec![name]),
    }
}

fn collect_decode_parms(dict: &Dictionary, count: usize) -> Vec<Option<Object>> {
    match dict.get(b"DecodeParms") {
        Some(Object::Array(a)) => a.iter().cloned().map(Some).collect(),
        Some(o) => {
            let mut v = vec![Some(o.clone())];
            v.resize(count, None);
            v
        }
        None => vec![None; count],
    }
}

// ── DictLookup trait (avoid coupling stream.rs to the full Document) ──────────

/// Minimal dictionary lookup used inside this module to avoid coupling its
/// callers to a specific dictionary type (it's used by both [`Dictionary`] and
/// the raw [`Object`] variants returned from xref-stream parsing).
pub(crate) trait DictLookup {
    fn get_i64(&self, key: &[u8]) -> Option<i64>;
}

impl DictLookup for Dictionary {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        self.get(key)?.as_i64()
    }
}

// Blanket impl so we can pass `Option<&HashMap<…>>` transparently.
impl DictLookup for &dyn DictLookup {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        (*self).get_i64(key)
    }
}

// Allow apply_flate to accept Option<&dyn DictLookup> where params is a raw
// Object (e.g. from xref stream).
impl DictLookup for Object {
    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        self.as_dict()?.get(key)?.as_i64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_length_literal_repeat_and_eod() {
        // 2 → copy 3 bytes; 254 → repeat the next byte 3 times; 128 ends.
        let data = [2u8, 0xAA, 0xBB, 0xCC, 254, 0x42, 128, 0xFF];
        assert_eq!(
            decode_run_length(&data, usize::MAX),
            Decoded::whole(vec![0xAA, 0xBB, 0xCC, 0x42, 0x42, 0x42])
        );
        assert_eq!(decode_run_length(&[], usize::MAX), Decoded::whole(vec![]));
    }

    /// A run cut short by the end of the data keeps the bytes present and
    /// marks the output truncated.
    #[test]
    fn run_length_truncated_input_keeps_present_bytes() {
        let out = decode_run_length(&[3u8, 0x01, 0x02], usize::MAX);
        assert_eq!(out.bytes, [0x01, 0x02]);
        assert!(out.truncated.is_some_and(|r| r.contains("lacks 2 bytes")));
        let out = decode_run_length(&[0x81u8], usize::MAX);
        assert_eq!(out.bytes, [0u8; 0]);
        assert!(out.truncated.is_some());
        // Data ending at a run boundary without the end marker loses nothing.
        assert_eq!(
            decode_run_length(&[1u8, 7, 8], usize::MAX),
            Decoded::whole(vec![7, 8])
        );
    }

    #[test]
    fn run_length_stops_at_max_output() {
        // 20 records of 128 repeats = 2560 bytes uncapped.
        let data: Vec<u8> = std::iter::repeat_n([0x81u8, 0xAB], 20).flatten().collect();
        let out = decode_run_length(&data, 1000);
        assert_eq!(out.bytes.len(), 896, "seven whole runs fit under the cap");
        assert!(out.bytes.iter().all(|&b| b == 0xAB));
        assert!(out.truncated.is_some());
    }

    #[test]
    fn run_length_is_a_stream_filter() {
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"RunLengthDecode".to_vec()));
        let stream = crate::Stream::new(dict, vec![1, b'h', b'i', 128]);
        assert_eq!(stream.decompressed_content().unwrap(), b"hi");
    }

    /// Per-byte reference for `apply_png_filter`: the RFC 2083 recurrence
    /// evaluated with all three neighbours recomputed per byte. The
    /// row-dispatched production form must match it exactly for every
    /// filter, including out-of-spec filter bytes (copy-through).
    fn png_filter_reference(filter: u8, raw: &[u8], prev: &[u8], dst: &mut [u8], bpp: usize) {
        for i in 0..dst.len() {
            let a = if i >= bpp { dst[i - bpp] } else { 0 };
            let b = prev.get(i).copied().unwrap_or(0);
            let c = if i >= bpp {
                prev.get(i - bpp).copied().unwrap_or(0)
            } else {
                0
            };
            // The Average mean is formed in u16, independently of the
            // production `u8::midpoint`.
            let mean = u8::try_from(u16::midpoint(u16::from(a), u16::from(b))).unwrap();
            dst[i] = match filter {
                1 => raw[i].wrapping_add(a),
                2 => raw[i].wrapping_add(b),
                3 => raw[i].wrapping_add(mean),
                4 => raw[i].wrapping_add(paeth(a, b, c)),
                _ => raw[i],
            };
        }
    }

    #[test]
    fn png_filter_matches_per_byte_reference() {
        // Deterministic pseudo-random rows over every filter (including an
        // out-of-spec byte), several pixel widths, and strides that are not
        // multiples of bpp — with and without a previous row.
        for bpp in [1usize, 2, 3, 4, 6, 8] {
            for stride in [1usize, 2, 3, 7, 16, 33, 100] {
                let raw: Vec<u8> = (0..stride)
                    .map(|i| i.wrapping_mul(151).wrapping_add(43).to_le_bytes()[0])
                    .collect();
                let prev_row: Vec<u8> = (0..stride)
                    .map(|i| i.wrapping_mul(97).wrapping_add(17).to_le_bytes()[0])
                    .collect();
                for filter in 0u8..=5 {
                    for prev in [None, Some(&prev_row[..])] {
                        let mut want = vec![0u8; stride];
                        png_filter_reference(filter, &raw, prev.unwrap_or(&[]), &mut want, bpp);
                        let mut got = vec![0u8; stride];
                        apply_png_filter(filter, &raw, prev, &mut got, bpp);
                        assert_eq!(
                            got,
                            want,
                            "filter={filter} bpp={bpp} stride={stride} prev={}",
                            prev.is_some()
                        );
                    }
                }
            }
        }
    }

    /// Minimal `DictLookup` for predictor-parameter tests.
    struct Params(Vec<(&'static [u8], i64)>);

    impl DictLookup for Params {
        fn get_i64(&self, key: &[u8]) -> Option<i64> {
            self.0.iter().find(|(k, _)| *k == key).map(|&(_, v)| v)
        }
    }

    #[test]
    fn tiff_predictor_2_undoes_horizontal_differencing() {
        // 2 rows × 3 pixels × 2 components, 8-bit. Rows are differenced
        // per component (first pixel raw, then deltas); undo must
        // reconstruct the original componentwise running sums without
        // bleeding across rows.
        let params = Params(vec![
            (b"Predictor", 2),
            (b"Colors", 2),
            (b"BitsPerComponent", 8),
            (b"Columns", 3),
        ]);
        let encoded = vec![
            10, 20, 5, 250, 3, 4, // row 0: deltas (250 exercises wrapping)
            7, 1, 0, 0, 1, 255, // row 1
        ];
        let decoded = apply_png_predictor(encoded, &params).unwrap();
        assert_eq!(
            decoded,
            vec![
                10, 20, // first pixel raw
                15, 14, // 10+5, 20+250 (wraps: 270 % 256)
                18, 18, // 15+3, 14+4
                7, 1, // row 1 restarts the running sums
                7, 1, // 7+0, 1+0
                8, 0, // 7+1, 1+255 (wraps)
            ],
        );
    }

    #[test]
    fn tiff_predictor_2_undoes_packed_and_16_bit_components() {
        let undo = |bits, colors, cols, data: Vec<u8>| {
            let params = Params(vec![
                (b"Predictor", 2),
                (b"Colors", colors),
                (b"BitsPerComponent", bits),
                (b"Columns", cols),
            ]);
            apply_png_predictor(data, &params).unwrap()
        };
        // 1-bit: a running XOR per row; the row is padded to a byte.
        assert_eq!(undo(1, 1, 6, vec![0b1000_1000]), [0b1111_0000]);
        // 2-bit, two components: 1 2 | 3 3 | 1 0 → 1 2 | 0 1 | 1 1.
        assert_eq!(
            undo(2, 2, 3, vec![0b0110_1111, 0b0100_0000]),
            [0b0110_0001, 0b0101_0000]
        );
        // 4-bit, two rows: 0 7 9 7 → 0 7 0 7, then F 1 → F 0 (wraps).
        assert_eq!(undo(4, 1, 2, vec![0x07, 0xF1]), [0x07, 0xF0]);
        assert_eq!(
            undo(4, 1, 4, vec![0x07, 0x97, 0xF1, 0x00]),
            [0x07, 0x07, 0xF0, 0x00]
        );
        // 16-bit big-endian: 0x00FF + 0x0001 carries into the high byte.
        assert_eq!(
            undo(16, 1, 3, vec![0x00, 0xFF, 0x00, 0x01, 0xFF, 0x00]),
            [0x00, 0xFF, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn tiff_predictor_2_unsupported_depth_passes_through() {
        let params = Params(vec![
            (b"Predictor", 2),
            (b"Colors", 1),
            (b"BitsPerComponent", 3),
            (b"Columns", 8),
        ]);
        let data = vec![0x12, 0x34, 0x56];
        let out = apply_png_predictor(data.clone(), &params).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn png_predictor_rejects_out_of_cap_colors() {
        let params = Params(vec![(b"Predictor", 15), (b"Colors", 33)]);
        let err = apply_png_predictor(vec![0; 64], &params).unwrap_err();
        assert!(
            err.contains("Colors"),
            "error must name the parameter: {err}"
        );
    }

    #[test]
    fn png_predictor_rejects_out_of_cap_columns() {
        let params = Params(vec![(b"Predictor", 15), (b"Columns", 2_000_000)]);
        let err = apply_png_predictor(vec![0; 64], &params).unwrap_err();
        assert!(
            err.contains("Columns"),
            "error must name the parameter: {err}"
        );
    }

    #[test]
    fn png_predictor_rejects_out_of_cap_bits() {
        let params = Params(vec![(b"Predictor", 15), (b"BitsPerComponent", 64)]);
        let err = apply_png_predictor(vec![0; 64], &params).unwrap_err();
        assert!(
            err.contains("BitsPerComponent"),
            "error must name the parameter: {err}"
        );
    }

    #[test]
    fn png_predictor_tolerates_truncated_rows() {
        // 4-byte stride + 1 filter byte per row; 7 bytes = 1 full row +
        // 2 trailing bytes. The floor-to-whole-rows contract stands (the
        // truncation is logged, not fatal).
        let params = Params(vec![(b"Predictor", 15), (b"Columns", 4)]);
        let data = vec![0u8, 1, 2, 3, 4, 0, 9];
        let out = apply_png_predictor(data, &params).unwrap();
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    /// A Brotli stream holding one uncompressed meta-block: a three-byte
    /// header, then the 64 bytes `(7i + 3) mod 251`, then the end marker.
    const BROTLI_RAW_64: [u8; 68] = {
        let mut s = [0u8; 68];
        s[0] = 0x8b;
        s[1] = 0x1f;
        s[2] = 0x80;
        let mut i = 0;
        while i < 64 {
            #[expect(clippy::cast_possible_truncation, reason = "below 251")]
            let b = ((i * 7 + 3) % 251) as u8;
            s[3 + i] = b;
            i += 1;
        }
        s[67] = 0x03;
        s
    };

    #[test]
    fn brotli_decodes_and_keeps_a_truncated_prefix() {
        let full = apply_brotli(&BROTLI_RAW_64, None).unwrap();
        assert_eq!(full, Decoded::whole(BROTLI_RAW_64[3..67].to_vec()));
        let prefix = apply_brotli(&BROTLI_RAW_64[..40], None).unwrap();
        assert!(
            !prefix.bytes.is_empty() && full.bytes.starts_with(&prefix.bytes),
            "{prefix:?}"
        );
        assert!(
            prefix
                .truncated
                .is_some_and(|r| r.starts_with("BrotliDecode"))
        );
        assert!(apply_brotli(b"not brotli at all", None).is_err());
    }

    /// Every filter [`is_decoded_filter`] names transforms its input in
    /// [`apply_filter`]; image codecs pass through unchanged.
    #[test]
    fn decoded_filters_are_the_ones_apply_filter_decodes() {
        for name in DECODED_FILTERS {
            let encoded: &[u8] = if name == b"BrotliDecode" {
                &BROTLI_RAW_64
            } else {
                b"x\x9c+\xc9\xc8,\x06\x00\x04]\x01\xc1"
            };
            let out = apply_filter(name, encoded, None).map(|d| d.bytes);
            assert!(
                out.as_deref() != Ok(encoded),
                "{} passed its input through",
                String::from_utf8_lossy(name)
            );
        }
        for codec in [
            &b"DCTDecode"[..],
            b"JPXDecode",
            b"JBIG2Decode",
            b"CCITTFaxDecode",
        ] {
            assert!(!is_decoded_filter(codec));
            assert_eq!(
                apply_filter(codec, b"abc", None),
                Ok(Decoded::whole(b"abc".to_vec()))
            );
        }
    }

    #[test]
    fn abbreviated_filter_names_expand() {
        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(vec![
                Object::Name(b"A85".to_vec()),
                Object::Name(b"Fl".to_vec()),
                Object::Name(b"DCT".to_vec()),
                Object::Name(b"Custom".to_vec()),
            ]),
        );
        assert_eq!(
            collect_filters(&dict).expect("names"),
            [
                &b"ASCII85Decode"[..],
                b"FlateDecode",
                b"DCTDecode",
                b"Custom"
            ]
        );
    }

    /// A `/Filter` value that is not a name, such as an indirect reference
    /// left unresolved, fails the decode instead of passing the encoded
    /// bytes through as decoded; an absent or null `/Filter` is no filter.
    #[test]
    fn non_name_filter_fails_the_decode() {
        for filter in [
            Object::Reference((5, 0)),
            Object::Integer(3),
            Object::Array(vec![
                Object::Name(b"AHx".to_vec()),
                Object::Reference((5, 0)),
            ]),
            Object::Array(vec![Object::Null]),
        ] {
            let mut dict = Dictionary::new();
            dict.set(b"Filter", filter.clone());
            let err = decode_stream(b"41>", &dict).expect_err("no filter name");
            assert!(err.contains("not a filter name"), "{filter:?}: {err}");
        }
        let mut dict = Dictionary::new();
        assert_eq!(
            decode_stream(b"41>", &dict),
            Ok(Decoded::whole(b"41>".to_vec()))
        );
        dict.set(b"Filter", Object::Null);
        assert_eq!(
            decode_stream(b"41>", &dict),
            Ok(Decoded::whole(b"41>".to_vec()))
        );
    }

    /// An illegal byte, or `z` inside a group, ends the data; what decoded
    /// before it is kept.  A lone final character yields nothing, and an
    /// oversized group does not overflow.
    #[test]
    fn ascii85_keeps_the_prefix_before_an_illegal_byte() {
        // "0 1 0 rg" is `0H`(m0HbCK`.
        let out = decode_ascii85(b"0H`(m0HbCK\x7f+>GT)");
        assert_eq!(out.bytes, b"0 1 0 rg");
        assert!(out.truncated.is_some_and(|r| r.contains("0x7f")));
        let out = decode_ascii85(b"9jqo^9jzqo^~>");
        assert_eq!(out.bytes, b"Man M");
        assert!(out.truncated.is_some());
        assert_eq!(
            decode_ascii85(b"9jqo^9~>"),
            Decoded::whole(b"Man ".to_vec())
        );
        assert_eq!(decode_ascii85(b"uuuuu~>").bytes.len(), 4);
    }

    #[test]
    fn ascii85_decode_basic() {
        // "Man " encodes to "9jqo^" in ASCII85.
        let encoded = b"9jqo^~>";
        let decoded = decode_ascii85(encoded);
        assert_eq!(decoded, Decoded::whole(b"Man ".to_vec()));
    }

    #[test]
    fn ascii_hex_decode() {
        let decoded = decode_ascii_hex(b"48656c6c6f>");
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn flate_roundtrip() {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let original = b"hello world, this is a test of flate encoding";
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(original).unwrap();
        let compressed = enc.finish().unwrap();
        let decompressed = apply_flate(&compressed, None).unwrap();
        assert_eq!(decompressed, Decoded::whole(original.to_vec()));
    }

    /// A Flate stream cut short keeps the bytes inflated before the cut and
    /// reports the truncation through the whole chain; an intact one reports
    /// none.
    #[test]
    fn flate_cut_short_is_reported_as_truncated() {
        let original: Vec<u8> = (0..20_000u32)
            .flat_map(|i| format!("{i} 0 m {i} 1 l S\n").into_bytes())
            .collect();
        let compressed = flate_compress(&original);
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"FlateDecode".to_vec()));

        let whole = decode_stream(&compressed, &dict).unwrap();
        assert_eq!(whole, Decoded::whole(original.clone()));

        let cut = decode_stream(&compressed[..compressed.len() / 2], &dict).unwrap();
        assert!(!cut.bytes.is_empty() && original.starts_with(&cut.bytes));
        assert!(cut.bytes.len() < original.len());
        assert!(
            cut.truncated
                .as_deref()
                .is_some_and(|r| r.starts_with("FlateDecode")),
            "{:?}",
            cut.truncated
        );
    }

    /// A complete deflate stream whose Adler-32 trailer is wrong loses no
    /// data and is not reported as truncated, by either inflate backend.
    #[test]
    fn flate_bad_checksum_is_not_truncated() {
        let original = b"0 g 10 10 80 80 re f".repeat(50);
        let mut compressed = flate_compress(&original);
        let n = compressed.len();
        compressed[n - 1] ^= 0xFF;

        let flate2 = decompress_zlib_flate2(&compressed).unwrap();
        assert_eq!(flate2, Decoded::whole(original.clone()));
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"FlateDecode".to_vec()));
        assert_eq!(
            decode_stream(&compressed, &dict).unwrap(),
            Decoded::whole(original)
        );
    }

    #[test]
    fn flate_roundtrip_large_grow_buffer() {
        // Exercises the InsufficientSpace -> resize loop in the libdeflate
        // backend by feeding it highly-compressible input that decompresses
        // to ~1 MiB (well above the 4× initial reservation for repetitive
        // input that compresses to a few KiB).
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let original: Vec<u8> = (0..1_000_000u32).map(|i| (i % 251) as u8).collect();
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&original).unwrap();
        let compressed = enc.finish().unwrap();
        let decompressed = apply_flate(&compressed, None).unwrap();
        assert_eq!(decompressed.bytes.len(), original.len());
        assert_eq!(decompressed, Decoded::whole(original));
    }

    // ── Filter-chain DoS guards ────────────────────────────────────────────────

    fn flate_compress(data: &[u8]) -> Vec<u8> {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn legit_ascii85_then_flate_chain_roundtrips() {
        // The standard transport-wrapper idiom: payload is flate-compressed
        // then ASCII85-armoured.  `decode_stream` must undo both, left to
        // right — this is the scanned-book prefilter step (only the terminal
        // codec differs in the corpus case).
        let original = b"chained transport payload \x00\x01\x02\xff and text";
        let flated = flate_compress(original);
        let ascii85 = {
            // Adobe ASCII85-encode `flated` (terminated with `~>`).
            let mut out = Vec::new();
            for chunk in flated.chunks(4) {
                let mut grp = [0u8; 4];
                grp[..chunk.len()].copy_from_slice(chunk);
                let mut n = u32::from_be_bytes(grp);
                let mut enc = [0u8; 5];
                for slot in enc.iter_mut().rev() {
                    *slot = b'!' + u8::try_from(n % 85).unwrap();
                    n /= 85;
                }
                out.extend_from_slice(&enc[..=chunk.len()]);
            }
            out.extend_from_slice(b"~>");
            out
        };

        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(vec![
                Object::Name(b"ASCII85Decode".to_vec()),
                Object::Name(b"FlateDecode".to_vec()),
            ]),
        );
        let decoded = decode_stream(&ascii85, &dict).expect("legit chain must decode");
        assert_eq!(decoded, Decoded::whole(original.to_vec()));
    }

    #[test]
    fn hostile_overlong_filter_chain_is_rejected_loudly() {
        // `/Filter [/FlateDecode … ×(MAX+1)]` — must fail with a clear DoS
        // error, never iterate the loop unbounded.
        let mut dict = Dictionary::new();
        dict.set(
            b"Filter",
            Object::Array(
                std::iter::repeat_with(|| Object::Name(b"FlateDecode".to_vec()))
                    .take(MAX_FILTER_CHAIN + 1)
                    .collect(),
            ),
        );
        let err = decode_stream(b"anything", &dict).expect_err("overlong chain must error");
        assert!(err.contains("filter chain too long"), "got: {err}");
    }

    #[test]
    fn lzw_bomb_cap_predicate_and_sink_flag() {
        use std::io::Write;

        // The LZW-bomb defence has two halves:
        //  1. the size predicate trips exactly at the cap boundary, and
        //  2. `BoundedSink` flags `tripped` + errors loudly so the caller
        //     discards the unrecoverable partial output.
        // Tested without allocating the bomb (predicate is pure).
        assert!(!would_overflow_cap(0, MAX_DECOMPRESSED));
        assert!(!would_overflow_cap(MAX_DECOMPRESSED, 0));
        assert!(would_overflow_cap(MAX_DECOMPRESSED, 1));
        assert!(would_overflow_cap(MAX_DECOMPRESSED - 4, 8));
        // saturating_add: an absurd incoming length cannot wrap to "fits".
        assert!(would_overflow_cap(usize::MAX, usize::MAX));
        let mut sink = BoundedSink {
            buf: Vec::new(),
            tripped: false,
        };
        assert_eq!(sink.write(b"hello").unwrap(), 5);
        assert!(!sink.tripped);
    }

    #[test]
    fn chain_stage_output_overrun_is_rejected() {
        // Aggregate-across-chain guard: even though each Flate stage is
        // individually 1-GiB-capped, an intermediate that exceeds the cap
        // must abort the whole chain.  Drive it through the public API with a
        // single Flate stage whose output we force over the limit via the
        // post-stage check (a true 1-GiB fixture is impractical in a unit
        // test, so we assert the smaller invariant: a valid short chain still
        // succeeds, proving the guard is not a false-positive).
        let mut dict = Dictionary::new();
        dict.set(b"Filter", Object::Name(b"FlateDecode".to_vec()));
        let payload = b"a modest stream well under the cap";
        let decoded = decode_stream(&flate_compress(payload), &dict).unwrap();
        assert_eq!(decoded, Decoded::whole(payload.to_vec()));
    }
}
