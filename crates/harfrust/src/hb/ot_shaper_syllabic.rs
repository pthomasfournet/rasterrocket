use super::buffer::*;
use super::font_funcs::FontFuncsDispatch;
use super::ot_shape_plan::hb_ot_shape_plan_t;
use super::GlyphInfo;
use crate::BufferFlags;

pub fn insert_dotted_circles(
    font_funcs: &mut FontFuncsDispatch,
    buffer: &mut hb_buffer_t,
    broken_syllable_type: u8,
    dottedcircle_category: u8,
    repha_category: Option<u8>,
    dottedcircle_position: Option<u8>,
) -> bool {
    if buffer
        .flags
        .contains(BufferFlags::DO_NOT_INSERT_DOTTED_CIRCLE)
    {
        return false;
    }

    if (buffer.scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_BROKEN_SYLLABLE) == 0 {
        return false;
    }

    let dottedcircle_glyph = match font_funcs.nominal_glyph(0x25CC) {
        Some(g) => g.to_u32(),
        None => return false,
    };

    let mut dottedcircle = GlyphInfo {
        glyph_id: 0x25CC,
        ..GlyphInfo::default()
    };
    dottedcircle.set_ot_shaper_var_u8_category(dottedcircle_category);
    if let Some(dottedcircle_position) = dottedcircle_position {
        dottedcircle.set_ot_shaper_var_u8_auxiliary(dottedcircle_position);
    }
    dottedcircle.glyph_id = dottedcircle_glyph;

    buffer.clear_output();

    buffer.idx = 0;
    let mut last_syllable = 0;
    while buffer.idx < buffer.len {
        let syllable = buffer.cur(0).syllable();
        if last_syllable != syllable && (syllable & 0x0F) == broken_syllable_type {
            last_syllable = syllable;

            let mut ginfo = dottedcircle;
            ginfo.cluster = buffer.cur(0).cluster;
            ginfo.mask = buffer.cur(0).mask;
            ginfo.set_syllable(buffer.cur(0).syllable());

            // Insert dottedcircle after possible Repha.
            if let Some(repha_category) = repha_category {
                while buffer.idx < buffer.len
                    && last_syllable == buffer.cur(0).syllable()
                    && buffer.cur(0).ot_shaper_var_u8_category() == repha_category
                {
                    buffer.next_glyph();
                }
            }

            buffer.output_info(ginfo);
        } else {
            buffer.next_glyph();
        }
    }

    buffer.sync();

    true
}

// `next_syllable()` cuts syllables longer than `MAX_SYLLABLE_LENGTH`, but the
// cut is computed from the current glyph positions and is not recorded, so
// walks before and after substitutions see different pieces.  Record the cut
// in `syllable()`, which substituted glyphs inherit, by giving each piece its
// own serial.  The pieces are not syllables on their own and text must not be
// broken between them, so keep them in one cluster.
pub(crate) fn limit_syllable_length(buffer: &mut hb_buffer_t) {
    let mut syllable_serial: u8 = 1;
    let mut first_piece = 0;
    let mut start = 0;
    let mut end = buffer.next_syllable(0);
    while start < buffer.len {
        let syllable = buffer.info[start].syllable();
        if end >= buffer.len || buffer.info[end].syllable() != syllable {
            // Last piece of this syllable.
            if first_piece < start {
                buffer.merge_clusters(first_piece, end);
            }
            first_piece = end;
        }
        for info in &mut buffer.info[start..end] {
            info.set_syllable((syllable_serial << 4) | (syllable & 0x0F));
        }
        syllable_serial += 1;
        if syllable_serial == 16 {
            syllable_serial = 1;
        }
        start = end;
        end = buffer.next_syllable(start);
    }
}

pub(crate) fn syllabic_clear_var(
    _: &hb_ot_shape_plan_t,
    _: &mut FontFuncsDispatch,
    buffer: &mut hb_buffer_t,
) -> bool {
    for info in &mut buffer.info {
        info.set_syllable(0);
    }
    buffer.deallocate_var(GlyphInfo::SYLLABLE_VAR);

    false
}
