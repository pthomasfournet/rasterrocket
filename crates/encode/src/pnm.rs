//! Netpbm images (PPM, PGM, PBM) written as their rows arrive, band by
//! band: the header first, then each band's rows.  The bytes are those
//! [`crate::write_ppm`], [`crate::write_pgm`] and [`crate::write_pbm`]
//! write for the same image.

use std::io::Write;

use crate::rows::{RowCount, RowImage, RowLayout};
use crate::{EncodeError, pack_bilevel_row, write_live_rows};

/// The Netpbm format of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PnmKind {
    /// P6: 8-bit RGB, from RGB bands; an alpha plane is dropped.
    Ppm,
    /// P5: 8-bit gray, from gray bands.
    Pgm,
    /// P4: one bit a pixel from gray bands, 1 for black (below
    /// [`crate::BLACK_BELOW`]).
    Pbm,
}

impl PnmKind {
    const fn layout(self) -> RowLayout {
        match self {
            Self::Ppm => RowLayout::Rgb,
            Self::Pgm => RowLayout::Gray,
            Self::Pbm => RowLayout::Bilevel,
        }
    }
}

/// A Netpbm image being written.
pub struct PnmWriter<W: Write> {
    out: W,
    kind: PnmKind,
    rows: RowCount,
    /// One packed bilevel row.
    packed: Vec<u8>,
}

impl<W: Write> PnmWriter<W> {
    /// Begin a `width × height` image of `kind` on `out`: write its header.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for an empty image;
    /// [`EncodeError::Io`] on a write failure.
    pub fn new(
        mut out: W,
        (width, height): (u32, u32),
        kind: PnmKind,
    ) -> Result<Self, EncodeError> {
        let rows = RowCount::new(width, height)?;
        match kind {
            PnmKind::Ppm => crate::ppm::write_ppm_header(&mut out, width, height)?,
            PnmKind::Pgm => crate::pgm::write_pgm_header(&mut out, width, height)?,
            PnmKind::Pbm => crate::pbm::write_pbm_header(&mut out, width, height)?,
        }
        Ok(Self {
            out,
            kind,
            rows,
            packed: Vec::new(),
        })
    }

    /// Write the rows of `band`, the next rows of the image.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] for a band of another width, of more
    /// rows than the image has left, or of colour for a gray image or the
    /// reverse; [`EncodeError::Io`] on a write failure.
    pub fn push(&mut self, band: RowImage<'_>) -> Result<(), EncodeError> {
        self.rows.take(&band, self.kind.layout())?;
        match (band, self.kind) {
            (RowImage::Rgb(b), _) => write_live_rows(b, &mut self.out)?,
            (RowImage::Gray(b) | RowImage::Bilevel(b), PnmKind::Pbm) => {
                let w = b.width as usize;
                self.packed.resize(w.div_ceil(8), 0);
                for y in 0..b.height {
                    pack_bilevel_row(&b.row_bytes(y)[..w], &mut self.packed);
                    self.out.write_all(&self.packed)?;
                }
            }
            (RowImage::Gray(b) | RowImage::Bilevel(b), _) => write_live_rows(b, &mut self.out)?,
        }
        Ok(())
    }

    /// Flush the image and return the sink.
    ///
    /// # Errors
    ///
    /// [`EncodeError::InvalidInput`] when rows are missing;
    /// [`EncodeError::Io`] on a write failure.
    pub fn finish(mut self) -> Result<W, EncodeError> {
        self.rows.finished()?;
        self.out.flush()?;
        Ok(self.out)
    }
}
