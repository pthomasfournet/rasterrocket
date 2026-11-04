//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.

use alloc::vec::Vec;

use super::bitplane::{
    BitPlaneDecodeBuffers, BitPlaneDecodeContext, Coefficient, CoefficientState,
};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{
    CodeBlockStyle, ComponentInfo, Header, QuantizationStyle, WaveletTransform,
};
use super::ht::{HtBlock, HtDecodeContext};
use super::idwt::IDWTOutput;
use super::progression::progression_iterator;
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use super::{ComponentData, bitplane, build, ht, idwt, mct, segment, tile};
use crate::error::{DecodingError, MarkerError, Result, TileError, ValidationError, bail};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::math::SimdBuffer;
use crate::reader::BitReader;
use core::ops::Range;

/// The per-sample transform [`decode`] left for the caller to fuse into its
/// own pass over the samples.
pub(crate) enum Deferred {
    /// Every transform has been applied.
    Nothing,
    /// The inverse MCT and the DC level shift of the first three components.
    Mct(mct::PendingMct),
    /// The DC level shift of the only component: this offset is to be added
    /// to each of its samples.
    Shift(f32),
}

/// Decode `data` into `ctx.channel_data`.
///
/// With `defer`, an inverse MCT the tiles call for is validated but not
/// applied: it is returned, and the first three components are left without
/// it and without their DC level shift.  Without an MCT, a single
/// component's DC level shift is likewise returned instead of applied.
/// Otherwise every transform is applied and [`Deferred::Nothing`] is
/// returned.
pub(crate) fn decode<'a>(
    data: &'a [u8],
    header: &'a Header,
    ctx: &mut DecoderContext<'a>,
    defer: bool,
) -> Result<Deferred> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;

    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }

    ctx.reset(header, &tiles[0])?;

    for tile in &tiles {
        decode_tile(
            tile,
            header,
            &mut ctx.tile_decode_context,
            &mut ctx.channel_data,
            &mut ctx.storage,
        )?;
    }

    // Note that this assumes that either all tiles have MCT or none of them.
    // In theory, only some could have it... But hopefully no such cursed
    // images exist!
    // The DC level shift of the first three components is fused into the
    // inverse MCT, which then touches each of their samples once instead of
    // twice.
    let mut deferred = Deferred::Nothing;
    let mut shifted = 0;
    if tiles[0].mct
        && let Some(transform) =
            mct::inverse_transform(&ctx.channel_data, &tiles[0].component_infos, header)?
    {
        let mct = mct::PendingMct {
            transform,
            offsets: core::array::from_fn(|i| {
                header.component_infos.get(i).map_or(0.0, sign_shift_offset)
            }),
        };
        if defer {
            deferred = Deferred::Mct(mct);
        } else {
            mct::apply_inverse(&mut ctx.channel_data, mct);
        }
        shifted = 3;
    } else if defer && ctx.channel_data.len() == 1 {
        deferred = Deferred::Shift(sign_shift_offset(&header.component_infos[0]));
        shifted = 1;
    }

    apply_sign_shift(
        &mut ctx.channel_data[shifted..],
        &header.component_infos[shifted..],
    );

    Ok(deferred)
}

/// A decoder context for decoding JPEG2000 images.
#[derive(Default)]
pub struct DecoderContext<'a> {
    tile_decode_context: TileDecodeContext,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    storage: DecompositionStorage<'a>,
}

impl DecoderContext<'_> {
    fn reset(&mut self, header: &Header, initial_tile: &Tile<'_>) -> Result<()> {
        self.tile_decode_context.reset();
        self.storage.reset();

        self.channel_data.clear();
        let sample_count = (header.size_data.image_width() as usize)
            .checked_mul(header.size_data.image_height() as usize)
            .ok_or(ValidationError::ImageTooLarge)?;
        // TODO: SIMD Buffers should be reused across runs!
        for info in &initial_tile.component_infos {
            self.channel_data.push(ComponentData {
                container: SimdBuffer::zeros(sample_count),
                bit_depth: info.size_info.precision,
            });
        }

        Ok(())
    }
}

fn decode_tile<'a>(
    tile: &Tile<'a>,
    header: &Header,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut [ComponentData],
    storage: &mut DecompositionStorage<'a>,
) -> Result<()> {
    storage.reset();

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    build::build(tile, storage, header.skipped_resolution_levels)?;
    // Next, we parse the layers/segments for each code block, in the packet
    // order of the tile. The iterator lists the precincts that `build` has
    // bounded, so it is created after it.
    let progression_iterator = progression_iterator(tile, header)?;
    segment::parse(tile, progression_iterator, header, storage)?;

    // Interleave bitplane decoding, IDWT, and storage for each component tile
    // so we can reuse the coefficient and IDWT buffers.
    // Every code block of a decoded sub-band writes all of its coefficients,
    // zeros included, and code blocks partition the sub-band, so the
    // coefficients a component leaves behind never reach the next one's
    // IDWT and the buffer is not cleared between components.
    for (idx, component_info) in header.component_infos.iter().enumerate() {
        // We then decode the bitplanes of each code block, yielding the
        // (possibly dequantized) coefficients of each code block.
        decode_component_tile_bit_planes(idx, tile, tile_ctx, storage, header)?;
        // Next, we apply the inverse discrete wavelet transform.
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
        );
        // Finally, we store the raw samples for the tile area in the correct
        // location. Note that in case we have MCT, we are not applying it yet.
        // It will be applied in the very end once all tiles have been processed.
        // The reason we do this is that applying MCT requires access to the
        // data from _all_ components. If we didn't defer this until the end
        // we would have to collect the IDWT outputs of all components before
        // applying it. By not applying MCT here, we can get away with doing
        // IDWT and store on a per-component basis. Thus, we only need to
        // store one IDWT output at a time, allowing for better reuse of
        // allocations.
        store(
            tile,
            header,
            tile_ctx,
            &mut channel_data[idx],
            component_info,
        );
    }

    Ok(())
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// A buffer so that we can reuse allocations for layers/code blocks/etc.
/// across different tiles.
#[derive(Default)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
}

impl DecompositionStorage<'_> {
    fn reset(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        // No need to clear the coefficients, as they will be resized
        // and then overridden.
        // self.coefficients.clear();
        self.precincts.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.tag_tree_nodes.clear();
    }
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HT code blocks.
    pub(crate) ht_decode_context: HtDecodeContext,
    /// The cleanup and refinement segments of the HT code block being
    /// decoded, gathered from its layers.
    pub(crate) ht_segments: [Vec<u8>; 2],
}

impl TileDecodeContext {
    fn reset(&mut self) {
        // This method doesn't do anything, just keeping it there in case
        // it's needed in the future.
        // Bitplane decode context and buffers will be reset in the
        // corresponding methods. IDWT output and scratch buffer will be
        // overridden on demand, so those don't need to be reset either.
    }
}

fn decode_component_tile_bit_planes<'a>(
    component_idx: usize,
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header,
) -> Result<()> {
    let component_info = &tile.component_infos[component_idx];

    // Only decode the resolution levels we actually care about.
    for resolution in 0..component_info.num_resolution_levels() - header.skipped_resolution_levels {
        let tile_composition = &storage.tile_decompositions[component_idx];
        let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

        for sub_band_idx in sub_band_iter {
            decode_sub_band_bitplanes(
                sub_band_idx,
                resolution,
                component_info,
                tile_ctx,
                storage,
                header,
            )?;
        }
    }

    Ok(())
}

fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header,
) -> Result<()> {
    let sub_band = &storage.sub_bands[sub_band_idx];

    let quantised =
        component_info.quantization_info.quantization_style != QuantizationStyle::NoQuantization;
    let irreversible = component_info.wavelet_transform() == WaveletTransform::Irreversible97;
    let dequantization_step = {
        if !quantised {
            1.0
        } else {
            let (exponent, mantissa) =
                component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;

            let r_b = {
                let log_gain = match sub_band.sub_band_type {
                    SubBandType::LowLow => 0,
                    SubBandType::LowHigh => 1,
                    SubBandType::HighLow => 1,
                    SubBandType::HighHigh => 2,
                };

                component_info.size_info.precision as u16 + log_gain
            };

            crate::math::pow2i(r_b as i32 - exponent as i32) * (1.0 + (mantissa as f32) / 2048.0)
        }
    };

    let num_bitplanes = {
        let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        // Equations (E-2) and (H-3).
        let num_bitplanes = (component_info.quantization_info.guard_bits as u16)
            .checked_add(exponent)
            .and_then(|x| x.checked_sub(1))
            .and_then(|x| x.checked_add(component_info.roi_shift as u16))
            .ok_or(DecodingError::InvalidBitplaneCount)?;

        if num_bitplanes > MAX_BITPLANE_COUNT as u16 {
            bail!(DecodingError::TooManyBitplanes);
        }

        num_bitplanes as u8
    };

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            let style = &component_info.coding_style.parameters.code_block_style;
            if style.mixed_high_throughput {
                bail!(MarkerError::UnsupportedHtj2k);
            }
            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;
            let base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;
            let reconstruction = Reconstruction {
                irreversible,
                roi_shift: component_info.roi_shift,
                dequantization_step,
                stride: sub_band.rect.width() as usize,
            };

            if style.high_throughput {
                decode_ht_code_block(code_block, num_bitplanes, style, tile_ctx, storage)?;
                let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
                reconstruction.store(
                    tile_ctx.ht_decode_context.coefficient_rows(),
                    &mut base_store[base_idx..],
                );
            } else {
                bitplane::decode(
                    code_block,
                    sub_band.sub_band_type,
                    num_bitplanes,
                    style,
                    tile_ctx,
                    storage,
                    header.strict,
                )?;
                let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
                reconstruction.store(
                    tile_ctx.bit_plane_decode_context.coefficient_rows(),
                    &mut base_store[base_idx..],
                );
            }
        }
    }

    Ok(())
}

/// How decoded signs and magnitudes become sub-band coefficients.
struct Reconstruction {
    irreversible: bool,
    roi_shift: u8,
    dequantization_step: f32,
    /// The sub-band's row length.
    stride: usize,
}

impl Reconstruction {
    /// Write a code block's rows into `out`, which starts at the code
    /// block's first coefficient within the sub-band.
    fn store<'a>(
        &self,
        rows: impl Iterator<Item = (&'a [Coefficient], &'a [CoefficientState])>,
        out: &mut [f32],
    ) {
        let mut base_idx = 0;
        for (coefficients, coefficient_states) in rows {
            let Some(out_row) = out.get_mut(base_idx..) else {
                return;
            };
            for ((output, coefficient), coefficient_state) in
                out_row.iter_mut().zip(coefficients).zip(coefficient_states)
            {
                *output =
                    coefficient.reconstructed(coefficient_state, self.irreversible, self.roi_shift)
                        * self.dequantization_step;
            }
            base_idx += self.stride;
        }
    }
}

/// Gather into `cleanup` and `refinement` the segments of the last HT set
/// in `segments` whose cleanup segment carries data.  Returns the set's
/// index and its number of refinement passes; the index is `None`, and the
/// buffers are empty, when no cleanup segment carries data.
fn gather_ht_set<'a>(
    segments: impl Iterator<Item = &'a Segment<'a>>,
    cleanup: &mut Vec<u8>,
    refinement: &mut Vec<u8>,
) -> (Option<u8>, u8) {
    cleanup.clear();
    refinement.clear();
    let mut refinement_passes = 0u8;
    let mut set = None;
    for segment in segments {
        let segment_set = segment.idx / 2;
        if segment.idx % 2 == 0 {
            // An empty cleanup segment opens an empty set, which leaves the
            // set before it in force.
            if !segment.data.is_empty() {
                set = Some(segment_set);
                cleanup.clear();
                cleanup.extend_from_slice(segment.data);
                refinement.clear();
                refinement_passes = 0;
            }
        } else if set == Some(segment_set) {
            refinement_passes = refinement_passes.saturating_add(segment.coding_pases);
            refinement.extend_from_slice(segment.data);
        }
    }
    (set, refinement_passes)
}

/// Decode an HT code block: the last of its HT sets whose cleanup segment
/// carries data, with the refinement segment that follows it.  Each HT set
/// codes the whole magnitude down to one bitplane below the set before it,
/// so the last such set supersedes the others.
fn decode_ht_code_block(
    code_block: &CodeBlock,
    num_bitplanes: u8,
    style: &CodeBlockStyle,
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
) -> Result<()> {
    let [cleanup, refinement] = &mut tile_ctx.ht_segments;
    let segments = storage.layers[code_block.layers.clone()]
        .iter()
        .filter_map(|layer| layer.segments.clone())
        .flat_map(|range| &storage.segments[range]);
    let (set, refinement_passes) = gather_ht_set(segments, cleanup, refinement);
    let block = HtBlock {
        width: code_block.rect.width(),
        height: code_block.rect.height(),
        cleanup,
        refinement,
        passes: if set.is_some() {
            1 + refinement_passes.min(2)
        } else {
            0
        },
        bitplanes: num_bitplanes,
        // Every HT set before the decoded one, placeholder or not, lowers
        // its cleanup by one bitplane.
        missing_msbs: code_block
            .missing_bit_planes
            .saturating_add(set.unwrap_or(0)),
        stripe_causal: style.vertically_causal_context,
    };
    ht::decode(&block, &mut tile_ctx.ht_decode_context)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok(())
}

/// The DC level shift of a component: `2^(precision - 1)`.
fn sign_shift_offset(component_info: &ComponentInfo) -> f32 {
    (1_u32 << (component_info.size_info.precision - 1)) as f32
}

fn apply_sign_shift(channel_data: &mut [ComponentData], component_infos: &[ComponentInfo]) {
    use crate::math::{Level, dispatch, f32x8};

    for (channel, component_info) in channel_data.iter_mut().zip(component_infos.iter()) {
        let offset = sign_shift_offset(component_info);
        dispatch!(Level::new(), simd => {
            let offset_v = f32x8::splat(simd, offset);
            for chunk in channel.container.as_chunks_mut::<8>().0 {
                let v = f32x8::from_slice(simd, chunk);
                (v + offset_v).store(chunk);
            }
        });
    }
}

fn store<'a>(
    tile: &'a Tile<'a>,
    header: &Header,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut ComponentData,
    component_info: &ComponentInfo,
) {
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    // Resolution levels can be empty (B-14), in which case there is nothing
    // to store.
    if idwt_output.rect.width() == 0 || idwt_output.rect.height() == 0 {
        return;
    }

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );

    let (image_x_offset, image_y_offset) = (
        header.size_data.image_x_offset(),
        header.size_data.image_y_offset(),
    );

    if scale_x == 1 && scale_y == 1 {
        // If no sub-sampling, use a fast path where we copy rows of coefficients
        // at once.

        // The rect of the IDWT output corresponds to the rect of the highest
        // decomposition level of the tile, which is usually not 1:1 aligned
        // with the actual tile rectangle. We also need to account for the
        // offset of the reference grid.

        let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0);

        // When the IDWT output is exactly the component's whole image, the
        // copy below would reproduce it verbatim: hand the buffer over
        // instead, and give the IDWT the channel's previous allocation to
        // reuse for the next component.
        let image_width = header.size_data.image_width() as usize;
        let image_height = header.size_data.image_height() as usize;
        if skip_x == 0
            && skip_y == 0
            && resolution_tile.rect.x0 <= image_x_offset
            && resolution_tile.rect.y0 <= image_y_offset
            && idwt_output.rect.width() as usize == image_width
            && idwt_output.rect.height() as usize == image_height
            && idwt_output.coefficients.len() == image_width * image_height
            && channel_data.container.truncated().len() == image_width * image_height
        {
            let coefficients = core::mem::take(&mut idwt_output.coefficients);
            let previous =
                core::mem::replace(&mut channel_data.container, SimdBuffer::new(coefficients));
            idwt_output.coefficients = previous.into_vec();
            return;
        }

        let input_row_iter = idwt_output
            .coefficients
            .chunks_exact(idwt_output.rect.width() as usize)
            .skip(skip_y as usize)
            .take(idwt_output.rect.height() as usize);

        let output_row_iter = channel_data
            .container
            .chunks_exact_mut(header.size_data.image_width() as usize)
            .skip(resolution_tile.rect.y0.saturating_sub(image_y_offset) as usize);

        for (input_row, output_row) in input_row_iter.zip(output_row_iter) {
            let input_row = &input_row[skip_x as usize..];
            let output_row = &mut output_row
                [resolution_tile.rect.x0.saturating_sub(image_x_offset) as usize..]
                [..input_row.len()];

            output_row.copy_from_slice(input_row);
        }
    } else {
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        // Otherwise, copy sample by sample.
        let first_x = (tile.rect.x0 == header.size_data.image_area_x_offset)
            .then_some(resolution_tile.rect.x0);
        let first_y = (tile.rect.y0 == header.size_data.image_area_y_offset)
            .then_some(resolution_tile.rect.y0);

        for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
            let relative_y = (y - idwt_output.rect.y0) as usize;
            let reference_grid_y = if first_y == Some(y) {
                image_y_offset
            } else {
                (scale_y as u32 * y) / y_shrink_factor
            };
            let y_end = (scale_y as u32 * (y + 1)) / y_shrink_factor;

            for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
                let relative_x = (x - idwt_output.rect.x0) as usize;
                let reference_grid_x = if first_x == Some(x) {
                    image_x_offset
                } else {
                    (scale_x as u32 * x) / x_shrink_factor
                };
                let x_end = (scale_x as u32 * (x + 1)) / x_shrink_factor;

                let sample = idwt_output.coefficients
                    [relative_y * idwt_output.rect.width() as usize + relative_x];

                for x_position in u32::max(reference_grid_x, image_x_offset)
                    ..u32::min(x_end, image_width + image_x_offset)
                {
                    for y_position in u32::max(reference_grid_y, image_y_offset)
                        ..u32::min(y_end, image_height + image_y_offset)
                    {
                        let pos = (y_position - image_y_offset) as usize * image_width as usize
                            + (x_position - image_x_offset) as usize;

                        channel_data.container[pos] = sample;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(idx: u8, coding_pases: u8, data: &[u8]) -> Segment<'_> {
        Segment {
            idx,
            coding_pases,
            data_length: data.len() as u32,
            data,
        }
    }

    fn gather(segments: &[Segment<'_>]) -> (Option<u8>, u8, Vec<u8>, Vec<u8>) {
        let (mut cleanup, mut refinement) = (Vec::new(), Vec::new());
        let (set, passes) = gather_ht_set(segments.iter(), &mut cleanup, &mut refinement);
        (set, passes, cleanup, refinement)
    }

    /// A later HT set whose cleanup carries data supersedes the one before
    /// it, refinement included.
    #[test]
    fn later_set_supersedes_earlier_one() {
        let segments = [
            segment(0, 1, &[1, 1]),
            segment(1, 2, &[2]),
            segment(2, 1, &[3, 3]),
            segment(3, 1, &[4]),
        ];
        assert_eq!(gather(&segments), (Some(1), 1, vec![3, 3], vec![4]));
    }

    /// An empty cleanup segment leaves the set before it in force, and its
    /// own refinement segment does not join it.
    #[test]
    fn empty_cleanup_keeps_previous_set() {
        let segments = [
            segment(0, 1, &[1, 1]),
            segment(1, 1, &[2]),
            segment(2, 1, &[]),
            segment(3, 2, &[]),
        ];
        assert_eq!(gather(&segments), (Some(0), 1, vec![1, 1], vec![2]));
    }

    /// Refinement passes of one set split over layers concatenate; leading
    /// placeholder sets carry no data.
    #[test]
    fn split_refinement_concatenates_after_placeholders() {
        let segments = [segment(2, 7, &[]), segment(3, 1, &[])];
        assert_eq!(gather(&segments), (None, 0, vec![], vec![]));
        let segments = [
            segment(4, 7, &[9, 9]),
            segment(5, 1, &[5]),
            segment(5, 1, &[6, 7]),
        ];
        assert_eq!(gather(&segments), (Some(2), 2, vec![9, 9], vec![5, 6, 7]));
    }
}
