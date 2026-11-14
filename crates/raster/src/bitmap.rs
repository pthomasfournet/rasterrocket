//! Pixel buffers.
//!
//! [`Bitmap<P>`] is a top-down pixel buffer, generic over the pixel type:
//! - Rows are stored top to bottom with a positive stride.
//! - Typed row access via [`bytemuck::cast_slice`] — zero-copy, no mode switch.
//! - Alpha is an optional separate plane, one byte per pixel.

use std::marker::PhantomData;

use crate::types::Pixel;

// ── internal helper ───────────────────────────────────────────────────────────

/// Convert a `u32` to `usize`.
///
/// On every platform where Rust runs, `usize` is at least 32 bits wide, so
/// this conversion is always exact. The compile-time `const` assertion below
/// enforces that invariant; any platform that violates it will fail to
/// compile rather than producing silently truncated values.
#[inline]
const fn u32_to_usize(v: u32) -> usize {
    const {
        assert!(
            u32::BITS <= usize::BITS,
            "platform has a narrower usize than u32"
        );
    }
    v as usize
}

/// A bitmap whose buffers cannot be allocated: their size overflows
/// `usize`, or the allocator refuses them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitmapAllocError {
    /// Width of the requested bitmap in pixels.
    pub width: u32,
    /// Height of the requested bitmap in pixels.
    pub height: u32,
}

impl std::fmt::Display for BitmapAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot allocate a {}×{} bitmap", self.width, self.height)
    }
}

impl std::error::Error for BitmapAllocError {}

// ── BitmapBuffers ─────────────────────────────────────────────────────────────

/// The storage of a [`Bitmap`], detached from it.
///
/// A later bitmap can be built in the same allocations: see
/// [`Bitmap::try_new_filled_in`] and [`Bitmap::into_buffers`].  The
/// contents are unspecified; only the capacities carry over.
#[derive(Debug, Default)]
pub struct BitmapBuffers {
    /// Colour buffer.
    pub data: Vec<u8>,
    /// Alpha plane buffer.
    pub alpha: Vec<u8>,
}

impl BitmapBuffers {
    /// The buffers a bitmap built by [`Bitmap::try_new_filled_in`] uses:
    /// the colour buffer, and the alpha buffer when `with_alpha`.  An alpha
    /// buffer the bitmap does not use stays in `self`.
    #[must_use]
    pub fn take_for(&mut self, with_alpha: bool) -> Self {
        Self {
            data: std::mem::take(&mut self.data),
            alpha: if with_alpha {
                std::mem::take(&mut self.alpha)
            } else {
                Vec::new()
            },
        }
    }

    /// Take the storage of `bitmap`: its colour buffer, and its alpha plane
    /// when it has one.  The alpha buffer held is kept when `bitmap` has
    /// no alpha plane, so a plane removed during a render
    /// ([`Bitmap::take_alpha_plane`]) and stored here survives.
    pub fn reclaim<P: Pixel>(&mut self, bitmap: Bitmap<P>) {
        self.data = bitmap.data;
        if let Some(alpha) = bitmap.alpha {
            self.alpha = alpha;
        }
    }
}

/// `buf` holding `len` copies of `value`, in its own allocation when its
/// capacity is at least `len` and in a new one otherwise.
fn refill(mut buf: Vec<u8>, len: usize, value: u8) -> Option<Vec<u8>> {
    buf.clear();
    if buf.capacity() < len {
        buf = Vec::new();
        buf.try_reserve_exact(len).ok()?;
    }
    buf.resize(len, value);
    Some(buf)
}

// ── Bitmap ────────────────────────────────────────────────────────────────────

/// A typed, top-down pixel buffer.
///
/// `P` determines the in-memory layout. The row stride is `width * P::BYTES`
/// rounded up to the next multiple of `row_pad` (default 4).
///
/// The optional alpha plane is always `width * height` bytes, one byte per pixel,
/// top-down. It is stored separately from the colour data.
pub struct Bitmap<P: Pixel> {
    /// Width of the bitmap in pixels.
    pub width: u32,
    /// Height of the bitmap in pixels.
    pub height: u32,
    /// Byte distance between the start of consecutive rows; always `≥ width * P::BYTES`
    /// and a multiple of `row_pad`.
    pub stride: usize,
    data: Vec<u8>,
    alpha: Option<Vec<u8>>,
    _marker: PhantomData<P>,
}

impl<P: Pixel> Bitmap<P> {
    /// Allocate a new zeroed bitmap.
    ///
    /// `row_pad` pads each row to a multiple of that many bytes (pass 1 for no padding).
    /// `with_alpha` allocates a separate alpha plane initialised to 0.
    ///
    /// # Panics
    ///
    /// Panics if `row_pad` is 0 or the buffer size overflows `usize`.
    #[must_use]
    pub fn new(width: u32, height: u32, row_pad: usize, with_alpha: bool) -> Self {
        Self::new_filled(width, height, row_pad, with_alpha, 0)
    }

    /// Allocate a new bitmap whose colour bytes are all `fill` (row padding
    /// included), written once as the buffer is allocated.
    ///
    /// `row_pad` and `with_alpha` are as for [`Bitmap::new`]; the alpha
    /// plane, when allocated, is initialised to 0.
    ///
    /// # Panics
    ///
    /// Panics if `row_pad` is 0 or the buffer size overflows `usize`.
    #[must_use]
    pub fn new_filled(width: u32, height: u32, row_pad: usize, with_alpha: bool, fill: u8) -> Self {
        let (stride, data_len, alpha_len) = Self::layout(width, height, row_pad, with_alpha)
            .expect("Bitmap::new: buffer size overflows usize");
        Self {
            width,
            height,
            stride,
            data: vec![fill; data_len],
            alpha: alpha_len.map(|len| vec![0u8; len]),
            _marker: PhantomData,
        }
    }

    /// [`Bitmap::new_filled`] that reports a buffer size overflowing
    /// `usize`, or an allocation the allocator refuses, as an error.  The
    /// buffer is written at once, so its memory is committed here: an
    /// allocator that overcommits (Linux by default) seldom refuses, and the
    /// out-of-memory killer may act instead when memory runs out.
    ///
    /// # Errors
    ///
    /// [`BitmapAllocError`] when either buffer cannot be allocated.
    ///
    /// # Panics
    ///
    /// Panics if `row_pad` is 0.
    pub fn try_new_filled(
        width: u32,
        height: u32,
        row_pad: usize,
        with_alpha: bool,
        fill: u8,
    ) -> Result<Self, BitmapAllocError> {
        Self::try_new_filled_in(
            BitmapBuffers::default(),
            width,
            height,
            row_pad,
            with_alpha,
            fill,
        )
    }

    /// [`Bitmap::try_new_filled`] built in `buffers`: a buffer whose
    /// capacity suffices is reused without allocating, a shorter one is
    /// replaced.  The alpha buffer is dropped when `with_alpha` is false.
    ///
    /// # Errors
    ///
    /// [`BitmapAllocError`] when a buffer that must be allocated cannot be.
    ///
    /// # Panics
    ///
    /// Panics if `row_pad` is 0.
    pub fn try_new_filled_in(
        buffers: BitmapBuffers,
        width: u32,
        height: u32,
        row_pad: usize,
        with_alpha: bool,
        fill: u8,
    ) -> Result<Self, BitmapAllocError> {
        let error = BitmapAllocError { width, height };
        let (stride, data_len, alpha_len) =
            Self::layout(width, height, row_pad, with_alpha).ok_or(error)?;
        let data = refill(buffers.data, data_len, fill).ok_or(error)?;
        let alpha = match alpha_len {
            Some(len) => Some(refill(buffers.alpha, len, 0).ok_or(error)?),
            None => None,
        };
        Ok(Self {
            width,
            height,
            stride,
            data,
            alpha,
            _marker: PhantomData,
        })
    }

    /// Row stride, colour buffer length and, when `with_alpha`, alpha
    /// plane length of a `width` × `height` bitmap with rows padded to
    /// `row_pad` bytes; `None` when a length overflows `usize`.
    fn layout(
        width: u32,
        height: u32,
        row_pad: usize,
        with_alpha: bool,
    ) -> Option<(usize, usize, Option<usize>)> {
        assert!(row_pad >= 1, "row_pad must be ≥ 1");
        let raw_stride = u32_to_usize(width).checked_mul(P::BYTES)?;
        let stride = if row_pad <= 1 {
            raw_stride
        } else {
            raw_stride.div_ceil(row_pad).checked_mul(row_pad)?
        };
        let data_len = stride.checked_mul(u32_to_usize(height))?;
        let alpha_len = if with_alpha {
            Some(u32_to_usize(width).checked_mul(u32_to_usize(height))?)
        } else {
            None
        };
        Some((stride, data_len, alpha_len))
    }

    // ── Row access ────────────────────────────────────────────────────────────

    /// Typed read-only access to row `y`.
    ///
    /// Returns a slice of exactly `width` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn row(&self, y: u32) -> &[P] {
        let bytes = self.row_bytes(y);
        bytemuck::cast_slice(&bytes[..u32_to_usize(self.width) * P::BYTES])
    }

    /// Typed mutable access to row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_mut(&mut self, y: u32) -> &mut [P] {
        let w = u32_to_usize(self.width) * P::BYTES;
        let bytes = self.row_bytes_mut(y);
        bytemuck::cast_slice_mut(&mut bytes[..w])
    }

    /// Shared bounds check used by every row accessor.
    #[inline]
    fn assert_row_in_bounds(&self, y: u32) {
        assert!(
            y < self.height,
            "row index {y} is out of bounds for bitmap height {}",
            self.height
        );
    }

    /// Raw byte read-only access to the full stride of row `y` (including padding).
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn row_bytes(&self, y: u32) -> &[u8] {
        self.assert_row_in_bounds(y);
        let off = u32_to_usize(y) * self.stride;
        &self.data[off..off + self.stride]
    }

    /// Raw byte mutable access to the full stride of row `y`.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_bytes_mut(&mut self, y: u32) -> &mut [u8] {
        self.assert_row_in_bounds(y);
        let off = u32_to_usize(y) * self.stride;
        &mut self.data[off..off + self.stride]
    }

    // ── Alpha access ──────────────────────────────────────────────────────────

    /// Alpha plane row `y`, if the alpha plane was allocated.
    ///
    /// Returns `None` if this bitmap was created without an alpha plane.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    #[must_use]
    pub fn alpha_row(&self, y: u32) -> Option<&[u8]> {
        self.assert_row_in_bounds(y);
        self.alpha.as_ref().map(|a| {
            let w = u32_to_usize(self.width);
            let off = u32_to_usize(y) * w;
            &a[off..off + w]
        })
    }

    /// Returns `true` if this bitmap has a separate alpha plane.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.alpha.is_some()
    }

    /// Read-only access to the full alpha plane (`width × height` bytes).
    ///
    /// Returns `None` if this bitmap was created without an alpha plane.
    #[must_use]
    pub fn alpha_plane(&self) -> Option<&[u8]> {
        self.alpha.as_deref()
    }

    /// Mutable access to the full alpha plane (`width × height` bytes).
    ///
    /// Returns `None` if this bitmap was created without an alpha plane.
    pub fn alpha_plane_mut(&mut self) -> Option<&mut [u8]> {
        self.alpha.as_deref_mut()
    }

    /// Remove and return the alpha plane, leaving the colour data: the
    /// bitmap is opaque from here on.  `None`, the bitmap unchanged, for a
    /// bitmap without one.
    pub const fn take_alpha_plane(&mut self) -> Option<Vec<u8>> {
        self.alpha.take()
    }

    /// Simultaneous mutable access to the pixel row and the alpha row.
    ///
    /// Returns `(pixel_row_bytes, Some(alpha_row))` or `(pixel_row_bytes, None)`.
    /// Provided because the borrow checker rejects holding two mutable
    /// references into the same `&mut self` (pixel data and alpha plane) via
    /// separate accessor calls.
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row_and_alpha_mut(&mut self, y: u32) -> (&mut [u8], Option<&mut [u8]>) {
        self.assert_row_in_bounds(y);
        let stride = self.stride;
        let w = u32_to_usize(self.width);
        let off = u32_to_usize(y) * stride;
        let pixels = &mut self.data[off..off + stride];
        let alpha = self.alpha.as_mut().map(|a| {
            let aoff = u32_to_usize(y) * w;
            &mut a[aoff..aoff + w]
        });
        (pixels, alpha)
    }

    // ── Bulk operations ───────────────────────────────────────────────────────

    /// Fill the entire bitmap with a solid colour and an alpha value.
    ///
    /// `alpha` is ignored if the bitmap has no alpha plane.
    pub fn clear(&mut self, color: P, alpha: u8) {
        let pixel_bytes: &[u8] = bytemuck::bytes_of(&color);
        if P::BYTES == 1 {
            // Optimise single-byte pixels: memset the entire data buffer.
            self.data.fill(pixel_bytes[0]);
        } else {
            debug_assert!(
                P::BYTES > 0,
                "clear: P::BYTES must be non-zero for chunked fill"
            );
            // Write one pixel-sized chunk per pixel in each row, then zero the
            // stride padding. Using `chunks_exact_mut` avoids manual index arithmetic.
            let w = u32_to_usize(self.width);
            let n = w * P::BYTES;
            let stride = self.stride;
            // A zero-width bitmap has no pixels and a zero stride.
            for row in self.data.chunks_exact_mut(stride.max(1)) {
                // `as_chunks_mut::<P::BYTES>()` would need an associated const as a
                // const-generic argument, which stable Rust does not accept.
                for dst in row[..n].chunks_exact_mut(P::BYTES) {
                    dst.copy_from_slice(pixel_bytes);
                }
                row[n..].fill(0);
            }
        }
        if let Some(ref mut a) = self.alpha {
            a.fill(alpha);
        }
    }

    /// Total byte size of the colour data buffer.
    #[must_use]
    pub const fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Access the full colour data buffer as a flat byte slice.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Mutable access to the full colour data buffer as a flat byte slice.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the bitmap and return its colour buffer (`stride * height`
    /// bytes); the alpha plane, if any, is dropped.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Consume the bitmap and return its colour buffer and alpha plane for
    /// reuse by [`Bitmap::try_new_filled_in`]; the alpha buffer is empty
    /// when the bitmap has no alpha plane.
    #[must_use]
    pub fn into_buffers(self) -> BitmapBuffers {
        BitmapBuffers {
            data: self.data,
            alpha: self.alpha.unwrap_or_default(),
        }
    }

    /// A bitmap without an alpha plane over `data`, rows packed with no
    /// padding (`stride == width * P::BYTES`).
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * P::BYTES`.
    #[must_use]
    pub fn from_packed(width: u32, height: u32, data: Vec<u8>) -> Self {
        let stride = u32_to_usize(width) * P::BYTES;
        assert_eq!(
            Some(data.len()),
            stride.checked_mul(u32_to_usize(height)),
            "Bitmap::from_packed: data must hold exactly width * height pixels",
        );
        Self {
            width,
            height,
            stride,
            data,
            alpha: None,
            _marker: PhantomData,
        }
    }
}

// ── Row interleaving ──────────────────────────────────────────────────────────

/// Interleave the packed RGB pixels of `rgb` with the alpha bytes of
/// `alpha` into the RGBA pixels of `out`, alpha not premultiplied; every
/// pixel is opaque when `alpha` is `None`.
///
/// The pixel count is that of the shortest input: `out.len() / 4`,
/// `rgb.len() / 3` and `alpha.len()`.
pub fn interleave_rgba(rgb: &[u8], alpha: Option<&[u8]>, out: &mut [u8]) {
    let rgb = rgb.as_chunks::<3>().0;
    let out = out.as_chunks_mut::<4>().0;
    match alpha {
        Some(alpha) => {
            for ((d, px), &a) in out.iter_mut().zip(rgb).zip(alpha) {
                *d = [px[0], px[1], px[2], a];
            }
        }
        None => {
            for (d, px) in out.iter_mut().zip(rgb) {
                *d = [px[0], px[1], px[2], u8::MAX];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rgb8;

    #[test]
    fn bitmap_stride_alignment() {
        let bm = Bitmap::<Rgb8>::new(7, 3, 4, false);
        // 7 * 3 = 21 bytes, next multiple of 4 = 24
        assert_eq!(bm.stride, 24);
    }

    #[test]
    fn try_new_filled_matches_new_filled() {
        let a = Bitmap::<Rgb8>::new_filled(7, 3, 4, true, 9);
        let b = Bitmap::<Rgb8>::try_new_filled(7, 3, 4, true, 9).unwrap();
        assert_eq!(a.stride, b.stride);
        assert_eq!(a.data(), b.data());
        assert_eq!(a.alpha_plane(), b.alpha_plane());
    }

    #[test]
    fn try_new_filled_reports_overflow_and_refusal() {
        let overflow = BitmapAllocError {
            width: u32::MAX,
            height: u32::MAX,
        };
        assert_eq!(
            Bitmap::<Rgb8>::try_new_filled(u32::MAX, u32::MAX, 1, false, 0).err(),
            Some(overflow)
        );
        // 13 PB: representable, but no allocator grants it.
        assert!(Bitmap::<Rgb8>::try_new_filled(u32::MAX, 1 << 20, 1, false, 0).is_err());
    }

    #[test]
    fn bitmap_row_len() {
        let bm = Bitmap::<Rgb8>::new(5, 2, 1, false);
        assert_eq!(bm.row(0).len(), 5);
        assert_eq!(bm.row(1).len(), 5);
    }

    #[test]
    fn bitmap_clear() {
        let mut bm = Bitmap::<Rgb8>::new(4, 2, 1, true);
        bm.clear(
            Rgb8 {
                r: 255,
                g: 0,
                b: 128,
            },
            200,
        );
        for y in 0..2u32 {
            for px in bm.row(y) {
                assert_eq!(
                    *px,
                    Rgb8 {
                        r: 255,
                        g: 0,
                        b: 128
                    }
                );
            }
            for &a in bm.alpha_row(y).unwrap() {
                assert_eq!(a, 200);
            }
        }
    }

    /// Clearing a zero-width bitmap of multi-byte pixels is a no-op.
    #[test]
    fn clear_zero_width_bitmap() {
        let mut bm = Bitmap::<Rgb8>::new(0, 3, 1, true);
        bm.clear(Rgb8 { r: 1, g: 2, b: 3 }, 9);
        assert_eq!(bm.data_len(), 0);
    }

    #[test]
    fn try_new_filled_in_reuses_a_large_enough_buffer() {
        let first = Bitmap::<Rgb8>::try_new_filled(8, 4, 1, true, 7).unwrap();
        let buffers = first.into_buffers();
        let (data_ptr, alpha_ptr) = (buffers.data.as_ptr(), buffers.alpha.as_ptr());
        let b = Bitmap::<Rgb8>::try_new_filled_in(buffers, 4, 4, 1, true, 255).unwrap();
        assert_eq!(b.data().as_ptr(), data_ptr, "colour buffer reused");
        assert_eq!(b.alpha_plane().unwrap().as_ptr(), alpha_ptr, "alpha reused");
        assert_eq!(b.data_len(), 4 * 4 * 3);
        assert!(b.data().iter().all(|&v| v == 255));
        assert!(b.alpha_plane().unwrap().iter().all(|&v| v == 0));
        let want = Bitmap::<Rgb8>::new_filled(4, 4, 1, true, 255);
        assert_eq!(b.stride, want.stride);
    }

    #[test]
    fn try_new_filled_in_grows_a_short_buffer() {
        let small = Bitmap::<Rgb8>::new_filled(2, 2, 1, false, 1).into_buffers();
        let b = Bitmap::<Rgb8>::try_new_filled_in(small, 5, 3, 4, false, 9).unwrap();
        assert_eq!(b.data_len(), Bitmap::<Rgb8>::new(5, 3, 4, false).data_len());
        assert!(b.data().iter().all(|&v| v == 9));
        assert!(b.alpha_plane().is_none());
    }

    #[test]
    fn reclaim_keeps_a_taken_alpha_plane() {
        let mut b = Bitmap::<Rgb8>::new(4, 4, 1, true);
        let alpha = b.take_alpha_plane().expect("an alpha plane");
        assert!(!b.has_alpha() && b.take_alpha_plane().is_none());
        let mut buffers = BitmapBuffers {
            data: Vec::new(),
            alpha,
        };
        buffers.reclaim(b);
        assert!(buffers.data.capacity() >= 48 && buffers.alpha.capacity() >= 16);
        buffers.reclaim(Bitmap::<Rgb8>::new(2, 2, 1, true));
        assert_eq!(
            buffers.alpha.len(),
            4,
            "a bitmap's own plane replaces the held one"
        );
    }

    #[test]
    fn take_for_an_opaque_bitmap_leaves_the_alpha_buffer() {
        let mut buffers = Bitmap::<Rgb8>::new(4, 4, 1, true).into_buffers();
        let alpha = buffers.alpha.as_ptr();
        let opaque =
            Bitmap::<Rgb8>::try_new_filled_in(buffers.take_for(false), 4, 4, 1, false, 0).unwrap();
        assert!(!opaque.has_alpha() && buffers.data.is_empty());
        assert_eq!(buffers.alpha.as_ptr(), alpha, "alpha buffer left in place");
        let taken = buffers.take_for(true);
        assert_eq!(taken.alpha.as_ptr(), alpha);
        assert_eq!(buffers.alpha, Vec::<u8>::new());
    }

    #[test]
    fn interleave_rgba_pairs_each_pixel_with_its_alpha() {
        let rgb = [1, 2, 3, 4, 5, 6];
        let mut out = [0; 8];
        interleave_rgba(&rgb, Some(&[7, 8]), &mut out);
        assert_eq!(out, [1, 2, 3, 7, 4, 5, 6, 8]);
        interleave_rgba(&rgb, None, &mut out);
        assert_eq!(out, [1, 2, 3, 255, 4, 5, 6, 255]);
        let mut short = [0; 4];
        interleave_rgba(&rgb, Some(&[7, 8]), &mut short);
        assert_eq!(short, [1, 2, 3, 7]);
    }
}
