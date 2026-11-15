//! AVIF decode for comic pages: the HEIF container is read here
//! ([`container`]), the AV1 pictures by the workspace's `rav1d` fork, and
//! the planes are converted to 8-bit gray or RGB here ([`color`]).
//!
//! The page is the primary item: a coded picture (`av01`) or a grid of them
//! (`grid`), cropped, turned and mirrored as its `clap`, `irot` and `imir`
//! properties say, in the order they are associated.  An image sequence
//! (`avis`) yields its primary item; a file with tracks alone is an error.
//! Auxiliary items (alpha, depth) are not read, so a transparent picture
//! decodes as its colour planes.
//!
//! Colour: the matrix coefficients and range come from the item's `nclx`
//! colour box when it has one and from the AV1 sequence header otherwise.
//! ICC profiles are not applied (no codec of this crate applies one), and
//! neither are transfer characteristics: samples are taken as display-
//! referred 8-bit values after the matrix.

mod color;
mod container;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use rav1d::{StillLimits, StillPicture, StillSamples};

use self::color::{ColorSpec, Frame, Pixels, Plane, Sample, chroma_size, to_pixels};
use self::container::{Container, Grid, ItemInfo, Nclx};
use super::{
    CodecPixels, DecodeError, MAX_SAMPLES, Target, gray_from_tight, gray16_from_tight,
    guard_dimensions, rgb_from_tight, rgb16_from_tight,
};

pub(in crate::decode) use self::container::is_avif;

/// The most tiles a grid may have.
const MAX_TILES: u32 = 1024;

fn codec(what: impl std::fmt::Display) -> DecodeError {
    DecodeError::Codec(format!("avif: {what}"))
}

/// A geometric property of an item.
#[derive(Clone, Copy, Debug)]
enum Transform<'a> {
    /// `clap`, its payload parsed against the image size at application.
    Crop(&'a [u8]),
    /// `irot`: anticlockwise quarter turns.
    Turn(u8),
    /// `imir`: 0 exchanges top and bottom, 1 left and right.
    Mirror(u8),
}

/// The properties of one item this decoder acts on.
#[derive(Debug, Default)]
struct ItemProps<'a> {
    size: Option<(u32, u32)>,
    nclx: Option<Nclx>,
    /// Whether the item has an `av1C` codec configuration.
    configured: bool,
    /// At most one of each kind, in association order.
    transforms: Vec<Transform<'a>>,
}

impl<'a> ItemProps<'a> {
    /// Read the properties of item `id`.
    ///
    /// # Errors
    ///
    /// [`DecodeError::Codec`] for a malformed property, an essential
    /// property this decoder does not implement, a selection of an
    /// operating point or layer other than the default, or a transformative
    /// property (`clap`, `irot`, `imir`) associated twice, which HEIF
    /// forbids.
    fn read(container: &Container<'a>, id: u32) -> Result<Self, DecodeError> {
        let mut out = Self::default();
        for property in container.properties(id)? {
            let payload = property.payload;
            let transform = match &property.kind {
                b"clap" => Some(Transform::Crop(payload)),
                b"irot" => Some(Transform::Turn(container::parse_irot(payload)?)),
                b"imir" => Some(Transform::Mirror(container::parse_imir(payload)?)),
                _ => None,
            };
            if let Some(transform) = transform {
                let same = std::mem::discriminant(&transform);
                if out
                    .transforms
                    .iter()
                    .any(|t| std::mem::discriminant(t) == same)
                {
                    return Err(codec(format_args!(
                        "the transformative property '{}' associated twice with one item",
                        property.kind.escape_ascii()
                    )));
                }
                out.transforms.push(transform);
                continue;
            }
            match &property.kind {
                b"ispe" => out.size = Some(container::parse_ispe(payload)?),
                b"colr" => {
                    // The first `nclx` colour description of the item
                    // stands; an ICC profile is not applied.
                    if out.nclx.is_none() {
                        out.nclx = container::parse_colr(payload)?;
                    }
                }
                b"av1C" => {
                    container::parse_av1c(payload)?;
                    out.configured = true;
                }
                b"a1op" => {
                    if container::parse_a1op(payload)? != 0 {
                        return Err(codec("operating point selection is not supported"));
                    }
                }
                b"lsel" => {
                    if container::parse_lsel(payload)? != 0xFFFF {
                        return Err(codec("layer selection is not supported"));
                    }
                }
                // Descriptive properties with no effect on 8-bit output.
                b"pixi" | b"pasp" | b"auxC" | b"a1lx" | b"clli" | b"mdcv" | b"cclv" | b"amve" => {}
                other if property.essential => {
                    return Err(codec(format_args!(
                        "unsupported essential property '{}'",
                        other.escape_ascii()
                    )));
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// A decoded picture of either sample width.
#[derive(Debug)]
enum AnyFrame<'a> {
    U8(Frame<'a, u8>),
    U16(Frame<'a, u16>),
}

/// The colour description of a decoded picture's sequence header.
fn stream_color(picture: &StillPicture) -> ColorSpec {
    ColorSpec {
        primaries: u16::from(picture.color.primaries),
        matrix: u16::from(picture.color.matrix),
        full_range: picture.color.full_range,
    }
}

/// A frame of the size, depth and layout of `picture` over exactly three
/// `planes`.
fn frame_of<'a, T: Clone>(
    picture: &StillPicture,
    planes: Vec<Plane<'a, T>>,
) -> Option<Frame<'a, T>> {
    Some(Frame {
        width: picture.width as usize,
        height: picture.height as usize,
        depth: picture.bit_depth,
        layout: picture.layout,
        planes: planes.try_into().ok()?,
    })
}

/// The planes of `picture` as a frame, borrowed from the decoder's buffer.
fn frame_view(picture: &StillPicture) -> Result<AnyFrame<'_>, DecodeError> {
    let mut narrow = Vec::with_capacity(3);
    let mut wide = Vec::with_capacity(3);
    for index in 0..3 {
        let plane = picture.plane(index);
        let (width, height) = (plane.width as usize, plane.height as usize);
        match plane.samples {
            StillSamples::U8(data) => narrow.push(Plane {
                width,
                height,
                stride: plane.stride,
                data: Cow::Borrowed(data),
            }),
            StillSamples::U16(data) => wide.push(Plane {
                width,
                height,
                stride: plane.stride,
                data: Cow::Borrowed(data),
            }),
        }
    }
    // The three planes are of one sample width, so one list holds them all.
    frame_of(picture, narrow)
        .map(AnyFrame::U8)
        .or_else(|| frame_of(picture, wide).map(AnyFrame::U16))
        .ok_or_else(|| codec("decoded planes of mixed sample widths"))
}

/// Decode the coded picture item `item`, whose properties are `props`.
///
/// The item's `ispe` size passes the page limits before the decoder runs
/// and bounds what the decoder may allocate, with the chroma layout of the
/// stream's own sequence headers; the picture must then have exactly that
/// size.
fn decode_coded(
    container: &Container<'_>,
    item: ItemInfo,
    props: &ItemProps<'_>,
    file_len: usize,
    target: Target,
) -> Result<StillPicture, DecodeError> {
    if item.kind != *b"av01" {
        return Err(codec(format_args!(
            "unsupported item type '{}'",
            item.kind.escape_ascii()
        )));
    }
    if !props.configured {
        return Err(codec("a coded item without its codec configuration"));
    }
    let (w, h) = props
        .size
        .ok_or_else(|| codec("a coded item without its image size"))?;
    if w == 0 || h == 0 {
        return Err(codec(format_args!("zero dimensions {w}x{h}")));
    }
    guard_dimensions(w, h, target)?;
    let too_large = DecodeError::TooLarge {
        width: w,
        height: h,
    };
    let Some(max_pixels) = w.checked_mul(h) else {
        return Err(too_large);
    };
    let data = container.data(item.id, file_len)?;
    let layout = rav1d::still_layout(&data).map_err(codec)?;
    let (cw, ch) = chroma_size(layout, w as usize, h as usize);
    let samples = (cw as u64)
        .checked_mul(ch as u64)
        .and_then(|chroma| chroma.checked_mul(2))
        .and_then(|chroma| chroma.checked_add(u64::from(max_pixels)));
    if samples.is_none_or(|n| n > MAX_SAMPLES as u64) {
        return Err(too_large);
    }
    let picture = rav1d::decode_still(&data, StillLimits { max_pixels }).map_err(codec)?;
    if (picture.width, picture.height) != (w, h) {
        return Err(codec(format_args!(
            "a {}x{} picture in an item declared {w}x{h}",
            picture.width, picture.height
        )));
    }
    Ok(picture)
}

/// Copy `tile` into `canvas` with its top-left luma sample at
/// (`x`, `y`), clipped to the canvas.  Both have one layout, and `x`, `y`
/// are multiples of the chroma subsampling factors.
fn blit<T: Sample>(canvas: &mut Frame<'_, T>, tile: &Frame<'_, T>, x: usize, y: usize) {
    let (sx, sy) = canvas.layout.chroma_shift();
    for (index, (to, from)) in canvas.planes.iter_mut().zip(&tile.planes).enumerate() {
        let (x, y) = if index == 0 {
            (x, y)
        } else {
            (x >> sx, y >> sy)
        };
        if from.width == 0 || x >= to.width {
            continue;
        }
        let n = from.width.min(to.width - x);
        let rows = to.data.to_mut().chunks_exact_mut(to.stride).skip(y);
        for (row, to) in rows.enumerate().take(from.height) {
            to[x..x + n].copy_from_slice(&from.row(row)[..n]);
        }
    }
}

/// An all-zero frame of the layout and depth of `like`.
fn canvas_like<T: Sample>(like: &Frame<'_, T>, width: usize, height: usize) -> Frame<'static, T> {
    let (cw, ch) = chroma_size(like.layout, width, height);
    Frame {
        width,
        height,
        depth: like.depth,
        layout: like.layout,
        planes: [
            Plane::zeroed(width, height),
            Plane::zeroed(cw, ch),
            Plane::zeroed(cw, ch),
        ],
    }
}

/// Decode the grid item `id`: its tiles, row-major, composed on a canvas
/// of the grid's output size.  Returns the canvas and the colour
/// description of the first tile.
fn decode_grid(
    container: &Container<'_>,
    id: u32,
    file_len: usize,
    target: Target,
) -> Result<(AnyFrame<'static>, ColorSpec, Option<Nclx>), DecodeError> {
    let Grid {
        rows,
        columns,
        width,
        height,
    } = container::parse_grid(&container.data(id, file_len)?)?;
    if width == 0 || height == 0 {
        return Err(codec(format_args!("zero dimensions {width}x{height}")));
    }
    guard_dimensions(width, height, target)?;
    let tiles = container.references(id, *b"dimg");
    if rows * columns > MAX_TILES || tiles.len() != (rows * columns) as usize {
        return Err(codec(format_args!(
            "a {columns}x{rows} grid of {} tiles",
            tiles.len()
        )));
    }

    let mut canvas: Option<(AnyFrame<'static>, ColorSpec, Option<Nclx>)> = None;
    let mut tile_size = (0u32, 0u32);
    for (index, &tile_id) in (0u32..).zip(tiles) {
        let props = ItemProps::read(container, tile_id)?;
        if !props.transforms.is_empty() {
            return Err(codec("a transformed grid tile"));
        }
        let (tw, th) = props
            .size
            .ok_or_else(|| codec("a coded item without its image size"))?;
        if index == 0 {
            // Tiles abut, so their size is a multiple of the chroma
            // subsampling factors: even covers every layout.
            let even = |n: u32, count: u32| count == 1 || n.is_multiple_of(2);
            let covers = |n: u32, count: u32, total: u32| {
                u64::from(n) * u64::from(count) >= u64::from(total)
            };
            if !covers(tw, columns, width) || !covers(th, rows, height) {
                return Err(codec("grid tiles do not cover the grid"));
            }
            if !even(tw, columns) || !even(th, rows) {
                return Err(codec("odd-sized grid tiles"));
            }
            let samples = u64::from(tw) * u64::from(columns) * u64::from(th) * u64::from(rows);
            if samples.saturating_mul(3) > MAX_SAMPLES as u64 {
                return Err(DecodeError::TooLarge { width, height });
            }
            tile_size = (tw, th);
        } else if (tw, th) != tile_size {
            return Err(codec("grid tiles of different sizes"));
        }
        let picture = decode_coded(
            container,
            container.item(tile_id)?,
            &props,
            file_len,
            target,
        )?;
        let color = stream_color(&picture);
        let tile = frame_view(&picture)?;
        let (canvas, ..) = canvas.get_or_insert_with(|| {
            let (w, h) = (width as usize, height as usize);
            let canvas = match &tile {
                AnyFrame::U8(t) => AnyFrame::U8(canvas_like(t, w, h)),
                AnyFrame::U16(t) => AnyFrame::U16(canvas_like(t, w, h)),
            };
            (canvas, color, props.nclx)
        });
        let x = ((index % columns) * tw) as usize;
        let y = ((index / columns) * th) as usize;
        match (canvas, &tile) {
            (AnyFrame::U8(c), AnyFrame::U8(t)) if (c.layout, c.depth) == (t.layout, t.depth) => {
                blit(c, t, x, y);
            }
            (AnyFrame::U16(c), AnyFrame::U16(t)) if (c.layout, c.depth) == (t.layout, t.depth) => {
                blit(c, t, x, y);
            }
            _ => return Err(codec("grid tiles of different pixel formats")),
        }
    }
    canvas.ok_or_else(|| codec("a grid without tiles"))
}

/// Tight pixels of `channels` samples `O` each.
struct Packed<O> {
    width: u32,
    height: u32,
    channels: usize,
    data: Vec<O>,
}

impl<O: Copy> Packed<O> {
    /// Rebuild the image as `width × height`, output pixel (x, y) taken
    /// from source pixel `source(x, y)`.
    fn remap(&mut self, width: u32, height: u32, source: impl Fn(usize, usize) -> (usize, usize)) {
        let (c, old_w) = (self.channels, self.width as usize);
        let mut data = Vec::with_capacity(width as usize * height as usize * c);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let (sx, sy) = source(x, y);
                let at = (sy * old_w + sx) * c;
                data.extend_from_slice(&self.data[at..at + c]);
            }
        }
        (self.width, self.height, self.data) = (width, height, data);
    }

    /// The image after `transforms`, in order: its width, height, whether
    /// it is gray, and its samples.
    fn transformed(
        mut self,
        transforms: &[Transform<'_>],
    ) -> Result<(u32, u32, bool, Vec<O>), DecodeError> {
        for &transform in transforms {
            self.apply(transform)?;
        }
        Ok((self.width, self.height, self.channels == 1, self.data))
    }

    fn apply(&mut self, transform: Transform<'_>) -> Result<(), DecodeError> {
        let (w, h) = (self.width, self.height);
        let (last_x, last_y) = (w as usize - 1, h as usize - 1);
        match transform {
            Transform::Crop(payload) => match container::parse_clap(payload, w, h)? {
                Some((x0, y0, cw, ch)) => {
                    let (x0, y0) = (x0 as usize, y0 as usize);
                    self.remap(cw, ch, |x, y| (x0 + x, y0 + y));
                }
                None => log::warn!("avif: clean aperture off the pixel grid; not applied"),
            },
            Transform::Turn(0) => {}
            // One anticlockwise quarter turn: the right column becomes the
            // top row.
            Transform::Turn(1) => self.remap(h, w, |x, y| (last_x - y, x)),
            Transform::Turn(2) => self.remap(w, h, |x, y| (last_x - x, last_y - y)),
            Transform::Turn(_) => self.remap(h, w, |x, y| (y, last_y - x)),
            Transform::Mirror(0) => self.remap(w, h, |x, y| (x, last_y - y)),
            Transform::Mirror(_) => self.remap(w, h, |x, y| (last_x - x, y)),
        }
        Ok(())
    }
}

/// Decode the primary item of an AVIF file to a gray or RGB bitmap: 16-bit
/// for a picture of more than 8 bits on a 16-bit target, 8-bit otherwise.
///
/// # Errors
///
/// [`DecodeError::TooLarge`] when the declared size of the page or of a
/// tile exceeds the page limits; [`DecodeError::Codec`] on a malformed,
/// truncated or unsupported container or AV1 stream.
pub(super) fn decode(bytes: &[u8], target: Target) -> Result<CodecPixels, DecodeError> {
    let container = Container::parse(bytes)?;
    let primary = container.primary()?;
    let props = ItemProps::read(&container, primary.id)?;
    // The frame of a coded picture borrows the decoder's planes.
    let picture;
    let (frame, stream, tile_nclx) = if primary.kind == *b"grid" {
        decode_grid(&container, primary.id, bytes.len(), target)?
    } else {
        picture = decode_coded(&container, primary, &props, bytes.len(), target)?;
        (frame_view(&picture)?, stream_color(&picture), None)
    };
    let spec = props.nclx.or(tile_nclx).map_or(stream, |nclx| ColorSpec {
        primaries: nclx.primaries,
        matrix: nclx.matrix,
        full_range: nclx.full_range,
    });
    match frame {
        AnyFrame::U16(f) if target.deep() => {
            let packed = packed(f.width, f.height, to_pixels::<_, u16>(&f, spec)?)?;
            let (width, height, gray, data) = packed.transformed(&props.transforms)?;
            if gray {
                gray16_from_tight(width, height, data)
            } else {
                rgb16_from_tight(width, height, data)
            }
        }
        frame => {
            let (width, height, pixels) = match frame {
                AnyFrame::U8(f) => (f.width, f.height, to_pixels::<_, u8>(&f, spec)?),
                AnyFrame::U16(f) => (f.width, f.height, to_pixels::<_, u8>(&f, spec)?),
            };
            let (width, height, gray, data) =
                packed(width, height, pixels)?.transformed(&props.transforms)?;
            if gray {
                gray_from_tight(width, height, data)
            } else {
                rgb_from_tight(width, height, data)
            }
        }
    }
}

/// The `width × height` pixels as a [`Packed`] image.
fn packed<O>(width: usize, height: usize, pixels: Pixels<O>) -> Result<Packed<O>, DecodeError> {
    let (channels, data) = match pixels {
        Pixels::Gray(data) => (1, data),
        Pixels::Rgb(data) => (3, data),
    };
    let side = |v: usize| u32::try_from(v).map_err(|_| codec("side over 32 bits"));
    Ok(Packed {
        width: side(width)?,
        height: side(height)?,
        channels,
        data,
    })
}
