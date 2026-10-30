//! Text rendering — `show_text` and `show_text_type3`.

use super::super::gstate::{ctm_multiply, mat2x2_mul};
use super::super::text::TextState;
use super::MAX_FORM_DEPTH;
use super::PageRenderer;
use super::text_ops::{GlyphRecord, glyph_advance, text_to_device, translate_text_x};
use color::Rgb8;
use raster::{
    glyph::{GlyphBitmap, fill_glyph},
    pipe::PipeSrc,
};

/// The painting operations of a text rendering mode (PDF §9.3.6, Table 106).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TextModeOps {
    /// Fill the glyphs with the non-stroking colour.
    fill: bool,
    /// Stroke the glyph outlines with the stroking colour and parameters.
    stroke: bool,
    /// Add the glyph outlines to the clipping path.
    clip: bool,
}

impl TextModeOps {
    /// The operations of `Tr` operand `mode`; a value outside `0..=7` fills.
    const fn of(mode: i32) -> Self {
        let (fill, stroke, clip) = match mode {
            1 => (false, true, false),
            2 => (true, true, false),
            3 => (false, false, false),
            4 => (true, false, true),
            5 => (false, true, true),
            6 => (true, true, true),
            7 => (false, false, true),
            _ => (true, false, false),
        };
        Self { fill, stroke, clip }
    }
}

impl PageRenderer<'_> {
    /// Render a byte string using the current font, colour, and text matrix.
    ///
    /// Two-phase design to satisfy the borrow checker:
    /// 1. Rasterize all glyphs while holding the `font_cache` borrow (no bitmap access).
    /// 2. Blit the pre-rasterized bitmaps, using `bitmap` mutably (no `font_cache` access).
    #[expect(
        clippy::too_many_lines,
        reason = "two-phase glyph render (rasterize then blit) inherently interleaves font, matrix, and clip logic that cannot be cleanly split without adding costly intermediate allocations"
    )]
    pub(super) fn show_text(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.diag.has_vector_text = true;

        let TextModeOps {
            fill: do_fill,
            stroke: do_stroke,
            clip: do_clip,
        } = TextModeOps::of(self.gstate.current().text.render_mode);
        let do_paint = do_fill || do_stroke;

        // Mode 3 (invisible) — nothing to do.
        if !do_paint && !do_clip {
            return;
        }

        // Snapshot all state we need before any mutable borrow of font_cache.
        let font_name = self.gstate.current().text.font_name.clone();
        let font_size = self.gstate.current().text.font_size;
        let char_spacing = self.gstate.current().text.char_spacing;
        let word_spacing = self.gstate.current().text.word_spacing;
        // PDF Tz is a percentage (default 100 %).  A value of 0 % is degenerate
        // (zero horizontal advance); we map it to 100 % to avoid invisible text.
        // This overrides an explicit `Tz 0` instruction — rare in practice.
        let raw_hz = self.gstate.current().text.horiz_scaling;
        let horiz_scaling = if raw_hz.abs() < f64::EPSILON {
            // Tz=0 % is degenerate (zero-width advance = invisible text).
            // Substitute 100 % so text remains visible; warn because this
            // overrides explicit PDF authoring intent.
            log::warn!(
                "rasterrocket-interp: Tz is 0 %, substituting 100 % to avoid invisible text"
            );
            1.0
        } else {
            raw_hz / 100.0
        };
        let ctm = self.gstate.current().ctm;
        let rise = self.gstate.current().text.rise;
        let mut tm = self.gstate.current().text.text_matrix;

        let Some(font) = self.gstate.current().text.font.clone() else {
            log::debug!(
                "rasterrocket-interp: no font dict for /{}",
                String::from_utf8_lossy(&font_name)
            );
            return;
        };
        let descriptor = &*font.descriptor;

        // ── Type 3 fast path ──────────────────────────────────────────────────
        // Type 3 glyphs are PDF content streams executed at the current pen
        // position.  They paint directly into the page bitmap, so there is no
        // two-phase rasterise-then-blit loop — each CharProc runs inline.
        // A glyph description paints with its own operators and has no
        // outline to fill or stroke (PDF §9.6.5), so every painting mode
        // executes the CharProc unchanged.
        if let Some(t3) = &descriptor.type3 {
            self.show_text_type3(
                bytes,
                t3,
                font_size,
                ctm,
                tm,
                char_spacing,
                word_spacing,
                horiz_scaling,
                rise,
                do_paint,
                do_clip,
            );
            return;
        }

        // Phase 1: rasterize glyphs and collect the outlines the mode needs.
        //
        // Two-phase design: borrow `font_cache` here (read-only after load), then
        // release it before touching `self.bitmap` or `self.gstate` in phase 2.
        let mut records: Vec<GlyphRecord> = Vec::with_capacity(bytes.len());
        // Union of the device-space glyph outlines, built when `do_clip`.
        let mut clip_outline = raster::path::Path::new();
        // Union of the device-space glyph outlines, built when `do_stroke`.
        let mut stroke_outline = raster::path::Path::new();
        // Device-space outlines of glyphs painted as filled paths instead of
        // cached bitmaps: every glyph when the fill colour is a pattern, else
        // the glyphs of faces too large for glyph bitmaps.
        let mut fill_paths: Vec<raster::path::Path> = Vec::new();
        // A pattern fill colour paints through the path fill, which resolves
        // the pattern once over the union of the glyph outlines.
        let pattern_fill = do_fill && self.gstate.current().fill_pattern.is_some();

        {
            // Trm[2×2] = font_size × Tm[2×2] × CTM[2×2] — the text rendering
            // matrix in device pixels, encoding actual size and skew/rotation.
            // The 2×2 submatrix (indices 0–3) is stable across the glyph loop
            // because only the translation components (indices 4–5) change as
            // the pen advances — the orientation and size don't change mid-string.
            let tm2x2 = mat2x2_mul(&tm, &ctm);
            let trm = tm2x2.map(|v| v * font_size);

            // Clone the glyph-cache handle before borrowing `font_cache` mutably:
            // it is an `Arc` internally, so this shares one cache rather than
            // copying it, and it sidesteps holding two borrows of `font_cache`.
            let glyph_cache = self.font_cache.glyph_cache().clone();
            let glyph_cache = &glyph_cache;

            // Load (or retrieve cached) face — mutable borrow of font_cache.
            let Some(face) = self.font_cache.get_or_load(&font, &font_name, trm) else {
                return;
            };
            let fill_as_path = do_fill && (pattern_fill || face.prefers_path_fill());
            let fill_as_bitmap = do_fill && !fill_as_path;
            let need_outline = do_clip || do_stroke || fill_as_path;

            // Iterate character codes.  Simple fonts use one byte per character;
            // Type 0 composite fonts use 1–4 bytes determined by the Encoding CMap.
            let cid_enc = descriptor.cid_encoding.as_ref();

            // Shared helper: push a rasterized glyph into `records` (when
            // filling) and append its outline to `stroke_outline` (when
            // stroking) and `clip_outline` (when clipping); a face too large
            // for glyph bitmaps fills through `fill_paths`.
            //
            // `gid` is the resolved glyph index.  `char_code_for_path` is the
            // raw character code.  `by_gid` selects which one the face is
            // indexed by:
            //
            // * Simple fonts (`by_gid == false`): the char code is resolved
            //   through the face's `code_to_gid` table / Unicode charmap by
            //   `make_glyph` / `glyph_path` (handles `Differences`, the
            //   standard 14, etc.).
            // * Type 0 / CIDFonts (`by_gid == true`): the interpreter has
            //   already mapped the code through the Encoding CMap and
            //   `CIDToGIDMap` to a final glyph selector.  It must be
            //   used directly — re-resolving it through the Unicode charmap
            //   mis-maps every glyph to `.notdef` for a CID-keyed CFF subset
            //   (whose charmap is absent/unrelated), rendering a blank page.
            let mut push_glyph =
                |char_code_for_path: u32, gid: u32, by_gid: bool, pen_x: i32, pen_y: i32| {
                    let bmp = if fill_as_bitmap {
                        if by_gid {
                            face.make_glyph_by_gid_cached(glyph_cache, gid, 0)
                        } else {
                            face.make_glyph_cached(glyph_cache, gid, 0)
                        }
                    } else {
                        None
                    };
                    if let Some(bitmap) = bmp {
                        records.push(GlyphRecord {
                            pen_x,
                            pen_y,
                            bitmap,
                        });
                    }
                    if !need_outline {
                        return;
                    }
                    // The glyph outline in device space.  `glyph_path`
                    // returns pixels relative to the pen position with Trm
                    // applied, y up; translate by (pen_x, pen_y) and flip y
                    // (device y is positive-down).
                    let outline = if by_gid {
                        face.glyph_path_by_gid(gid)
                    } else {
                        face.glyph_path(char_code_for_path)
                    };
                    let Some(mut path) = outline else {
                        return;
                    };
                    let px = f64::from(pen_x);
                    let py = f64::from(pen_y);
                    for pt in &mut path.pts {
                        pt.x += px;
                        pt.y = py - pt.y;
                    }
                    if do_clip {
                        clip_outline.append(&path);
                    }
                    if do_stroke {
                        stroke_outline.append(&path);
                    }
                    if fill_as_path {
                        fill_paths.push(path);
                    }
                };

            if let Some(enc) = cid_enc {
                // Type 0 composite font: multi-byte character codes via Encoding CMap.
                // Identity-H/V (enc.encoding_cmap == None) uses 2-byte codes by convention
                // (PDF spec §9.7.4, Table 118); embedded CMaps specify their own width.
                let code_bytes = enc.encoding_cmap.as_ref().map_or(2, |cm| cm.code_bytes);

                let mut pos = 0usize;
                while pos + usize::from(code_bytes) <= bytes.len() {
                    let mut char_code = 0u32;
                    for &b in &bytes[pos..pos + usize::from(code_bytes)] {
                        char_code = (char_code << 8) | u32::from(b);
                    }
                    pos += usize::from(code_bytes);

                    // charcode → CID (identity if no CMap), then CID → GID.
                    let cid = enc
                        .encoding_cmap
                        .as_ref()
                        .and_then(|cm| cm.map.get(&char_code).copied())
                        .unwrap_or(char_code);
                    let gid = enc.code_to_gid(char_code);

                    let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);
                    push_glyph(char_code, gid, true, pen_x, pen_y);

                    // CID width lookup uses the CID, not the raw char code.
                    // Advance formula: w/1000 * font_size (PDF §9.7.4.3).
                    // Word spacing does not apply to composite fonts (PDF §9.3.3).
                    let cid_width = enc.width_for_cid(cid);
                    let advance_glyph = f64::from(cid_width) / 1000.0;
                    let tx_adv =
                        glyph_advance(advance_glyph, font_size, char_spacing, 0.0, horiz_scaling);
                    tm = translate_text_x(tm, tx_adv);
                }
            } else {
                // Simple font: one byte per character code.
                // PDF §9.2.4: the Widths array is the authoritative advance source
                // for simple fonts (within [FirstChar, LastChar]); MissingWidth
                // applies out-of-range.  The font program's advance is consulted
                // ONLY when Widths is absent entirely (e.g. the 14 standard fonts
                // pre-PDF 1.5, where a substitute face supplies metrics).
                for &byte in bytes {
                    let (pen_x, pen_y) = text_to_device(&ctm, &tm, 0.0, rise, self.height);
                    push_glyph(u32::from(byte), u32::from(byte), false, pen_x, pen_y);

                    // PDF §9.4.4: pen advances even for missing/invisible glyphs.
                    let advance_glyph = descriptor.width_for_code(u32::from(byte)).map_or_else(
                        || face.glyph_advance(u32::from(byte)).unwrap_or(0.0),
                        |w| f64::from(w) / 1000.0,
                    );
                    let extra = if byte == b' ' { word_spacing } else { 0.0 };
                    let tx_adv =
                        glyph_advance(advance_glyph, font_size, char_spacing, extra, horiz_scaling);
                    tm = translate_text_x(tm, tx_adv);
                }
            }
        } // font_cache borrow released here

        // Phase 2: fill the glyphs, then stroke their outlines.
        if do_fill {
            let fill_bytes = self.gstate.current().fill_color.as_slice().to_vec();
            let clip = self.gstate.current().clip.clone_shared();
            let pipe = Self::make_pipe(
                self.gstate.current().fill_alpha,
                self.gstate.current().blend_mode,
            );
            let src = PipeSrc::Solid(&fill_bytes);

            for rec in &records {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "glyph dimensions are always small (sub-pixel-sized); cannot exceed i32::MAX"
                )]
                let glyph = GlyphBitmap {
                    data: &rec.bitmap.data,
                    x: rec.bitmap.x_off,
                    y: rec.bitmap.y_off,
                    w: rec.bitmap.width as i32,
                    h: rec.bitmap.height as i32,
                    aa: rec.bitmap.aa,
                };
                // The ClipResult return value is only relevant for accumulating
                // text-as-clip paths (text render modes 4–7), which Phase 3 below
                // handles separately via glyph_path; per-glyph fill output is unused.
                let _ = fill_glyph::<Rgb8>(
                    &mut self.bitmap,
                    &clip,
                    &pipe,
                    &src,
                    rec.pen_x,
                    rec.pen_y,
                    &glyph,
                );
            }
            if pattern_fill {
                let mut union = raster::path::Path::new();
                for path in &fill_paths {
                    union.append(path);
                }
                if !union.pts.is_empty() {
                    self.fill_path(&union, false);
                }
            } else {
                for path in &fill_paths {
                    self.fill_path(path, false);
                }
            }
        }
        // The stroke parameters are user-space quantities under the CTM, as
        // for any stroked path (PDF §8.4.3.2); the text matrix and font size
        // scale the outline but not the line width.
        if do_stroke && !stroke_outline.pts.is_empty() {
            self.stroke_path(&stroke_outline);
        }

        // Phase 3: intersect glyph outlines into the clip path (modes 4–7).
        //
        // PDF §9.3.6: glyph outlines from a single text-showing operator are
        // unioned together first, then the union is intersected with the clip
        // under the non-zero winding rule.
        if do_clip && !clip_outline.pts.is_empty() {
            self.clip_path_into_gstate(&clip_outline, false);
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }

    // ── Type 3 font rendering ─────────────────────────────────────────────────

    /// Execute Type 3 `CharProcs` for all character codes in `bytes`.
    ///
    /// Each `CharProc` is a PDF content stream that paints the glyph using the
    /// full graphics pipeline.  This method sets up the CTM for each glyph
    /// position and executes the stream, then advances the text matrix.
    ///
    /// The `CharProc` coordinate system (glyph space) maps to device space via:
    /// `charproc_ctm = CTM × Tm_at_glyph × scale(font_size) × FontMatrix`
    ///
    /// `CharProcs` are re-rasterised on every call.  If Type 3 perf becomes a
    /// hot path, the right fix is to extend `font::GlyphCache` to key on Type
    /// 3 font instances and add an off-screen bitmap path here, not to
    /// reintroduce a separate Type-3-only cache.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors show_text() parameter set; all values are needed for glyph placement"
    )]
    #[expect(clippy::many_single_char_names, reason = "PDF matrix components a–f")]
    pub(super) fn show_text_type3(
        &mut self,
        bytes: &[u8],
        t3: &crate::resources::font::Type3Data,
        font_size: f64,
        ctm: [f64; 6],
        mut tm: [f64; 6],
        char_spacing: f64,
        word_spacing: f64,
        horiz_scaling: f64,
        rise: f64,
        do_paint: bool,
        do_clip: bool,
    ) {
        if do_clip {
            log::warn!(
                "rasterrocket-interp: text-clip mode (render_mode ≥ 4) is not supported for Type 3 \
                 fonts — clip ignored"
            );
        }
        for &byte in bytes {
            let Some(glyph) = t3.glyph(byte) else {
                // No CharProc for this code — advance by zero width, continue.
                let [a, b_m, c, d, e, f] = tm;
                let tx_adv = char_spacing * horiz_scaling;
                tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
                continue;
            };

            if do_paint && self.form_depth >= MAX_FORM_DEPTH {
                log::warn!(
                    "rasterrocket-interp: Type 3 CharProc depth {MAX_FORM_DEPTH} exceeded — \
                     glyph 0x{byte:02X} not painted (text position still advances)"
                );
            } else if do_paint {
                // Build the CharProc CTM:
                //   Tm_rise = Tm with rise shift applied (PDF §9.4.4)
                //   TRM     = [fs·fm[0], fs·fm[1], fs·fm[2], fs·fm[3], tx, ty]
                //             where [tx, ty] = Tm_rise × CTM translation
                //   charproc_ctm = TRM × CTM_current
                //
                // Concretely: we insert `(font_size × FontMatrix) × Tm` as the
                // new CTM component so that the charproc's user space maps
                // correctly to device pixels.
                let fm = t3.font_matrix;
                // font_size × FontMatrix (2×2 + translation)
                let fs_fm = [
                    font_size * fm[0],
                    font_size * fm[1],
                    font_size * fm[2],
                    font_size * fm[3],
                    fm[4],
                    fm[5],
                ];
                // Apply rise to text matrix (vertical shift in text space).
                let tm_rise = if rise.abs() > f64::EPSILON {
                    let [a, b_m, c, d, e, f] = tm;
                    [a, b_m, c, d, rise.mul_add(c, e), rise.mul_add(d, f)]
                } else {
                    tm
                };
                // TRM = fs_fm × tm_rise
                let trm = ctm_multiply(&fs_fm, &tm_rise);
                // charproc_ctm = trm × CTM
                let charproc_ctm = ctm_multiply(&trm, &ctm);

                self.gstate.save();
                self.form_depth += 1;
                self.gstate.current_mut().ctm = charproc_ctm;
                // Reset text state inside charproc (PDF §9.6.5 — charproc is a
                // fresh graphics state, not a text object).
                self.gstate.current_mut().text = TextState::default();

                let ops = crate::content::parse(&glyph.content);
                self.execute(&ops);

                self.form_depth -= 1;
                self.gstate.restore();
            }

            // Advance the text matrix by the glyph width (PDF §9.4.4).
            // width_units are in glyph space; scale by FontMatrix[0] × font_size.
            let glyph_advance = f64::from(glyph.width_units) * t3.font_matrix[0];
            let extra = if byte == b' ' { word_spacing } else { 0.0 };
            let tx_adv = (glyph_advance.mul_add(font_size, char_spacing) + extra) * horiz_scaling;
            let [a, b_m, c, d, e, f] = tm;
            tm = [a, b_m, c, d, tx_adv.mul_add(a, e), tx_adv.mul_add(b_m, f)];
        }

        // Write updated text matrix back.
        self.gstate.current_mut().text.text_matrix = tm;
    }
}

#[cfg(test)]
mod tests {
    use super::TextModeOps;

    #[test]
    fn text_mode_ops_follow_the_mode_table() {
        let ops = |m| {
            let o = TextModeOps::of(m);
            (o.fill, o.stroke, o.clip)
        };
        assert_eq!(ops(0), (true, false, false));
        assert_eq!(ops(1), (false, true, false));
        assert_eq!(ops(2), (true, true, false));
        assert_eq!(ops(3), (false, false, false));
        assert_eq!(ops(4), (true, false, true));
        assert_eq!(ops(5), (false, true, true));
        assert_eq!(ops(6), (true, true, true));
        assert_eq!(ops(7), (false, false, true));
        assert_eq!(ops(8), ops(0), "out of range fills");
        assert_eq!(ops(-1), ops(0), "out of range fills");
    }
}
