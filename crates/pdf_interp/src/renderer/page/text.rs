//! Text rendering — `show_text` and `show_text_type3`.

use super::super::gstate::mat2x2_mul;
use super::PageRenderer;
use super::pixel_rect::PixelRect;
use super::text_ops::{
    GlyphRecord, fit_to_advance, glyph_advance, glyph_placement, text_to_device, translate_text_x,
    translate_text_y,
};
use crate::resources::cmap::DecodedCode;
use color::Rgb8;
use font::face::FontFace;
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
        // Mode 3 paints nothing and clips nothing, but the pen still
        // advances by every glyph's width (PDF §9.4.4): the run goes through
        // the glyph loop below, which rasterises nothing it will not use.
        let do_paint = do_fill || do_stroke;

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
        // outline to fill, stroke or clip with (PDF §9.6.5): every mode but
        // the invisible one executes the CharProc unchanged, and none adds
        // to the clip.
        // The glyphs are one object under the soft mask in effect.
        if let Some(t3) = &descriptor.type3 {
            let region = PixelRect::of_clip(&self.gstate.current().clip, self.width, self.height);
            let alpha = self.gstate.current().fill_alpha;
            self.paint_object(region, alpha, |r| {
                r.show_text_type3(
                    bytes,
                    t3,
                    &font.key,
                    font_size,
                    ctm,
                    tm,
                    char_spacing,
                    word_spacing,
                    horiz_scaling,
                    rise,
                    do_paint || do_clip,
                );
            });
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
            // Trm[2×2] = [Tfs×Th 0 0 Tfs] × Tm[2×2] × CTM[2×2] — the text
            // rendering matrix in device pixels (PDF §9.4.4), encoding
            // actual size, horizontal scaling and skew/rotation.  The 2×2
            // submatrix (indices 0–3) is stable across the glyph loop
            // because only the translation components (indices 4–5) change as
            // the pen advances — the orientation and size don't change mid-string.
            let [ax, ay, cx, cy] = mat2x2_mul(&tm, &ctm);
            let trm = [
                ax * font_size * horiz_scaling,
                ay * font_size * horiz_scaling,
                cx * font_size,
                cy * font_size,
            ];

            // Clone the glyph-cache handle before borrowing `font_cache` mutably:
            // it is an `Arc` internally, so this shares one cache rather than
            // copying it, and it sidesteps holding two borrows of `font_cache`.
            let glyph_cache = self.font_cache.glyph_cache().clone();
            let glyph_cache = &glyph_cache;

            // Load (or retrieve cached) face — mutable borrow of font_cache.
            let Some(face) = self
                .font_cache
                .get_or_load_showing(&font, &font_name, trm, bytes)
            else {
                return;
            };
            let fill_as_path = do_fill && (pattern_fill || face.prefers_path_fill());
            let fill_as_bitmap = do_fill && !fill_as_path;
            let need_outline = do_clip || do_stroke || fill_as_path;

            // Iterate character codes.  Simple fonts use one byte per character;
            // Type 0 composite fonts use 1–4 bytes determined by the Encoding CMap.
            let cid_enc = descriptor.cid_encoding.as_ref();

            // A substitute's glyphs are fitted to the advances the document
            // specifies: condensed when wider, centred when narrower.  An
            // embedded program's glyphs are the document's own, and a
            // standard family's substitute already has its advances.
            let fit = face.substitute
                && !descriptor.substitute.standard
                && !cid_enc.is_some_and(|enc| enc.vertical);
            let face_width = |advance: Option<f64>| -> i32 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "advances are a few em at most; thousandths fit i32"
                )]
                let width = advance.map_or(0.0, |w| (w * 1000.0).round()) as i32;
                width
            };
            // The advance of a drawn glyph `(face, selector, by_gid)`.
            let advance_of = |(face, selector, by_gid): (&FontFace, u32, bool)| {
                if by_gid {
                    face.glyph_advance_by_gid(selector)
                } else {
                    face.glyph_advance(selector)
                }
            };

            // Shared helper: push a rasterized glyph into `records` (when
            // filling) and append its outline to `stroke_outline` (when
            // stroking) and `clip_outline` (when clipping); a face too large
            // for glyph bitmaps fills through `fill_paths`.  `(dx, dy)` is
            // the exact device pen position; `x_scale` condenses the glyph
            // to the document's advance.
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
            //
            // `face` is the face that draws the glyph: the font's own, or the
            // fallback face of a code its substitute lacks, whose glyph is
            // always selected by index.
            let mut push_glyph = |face: &FontFace,
                                  char_code_for_path: u32,
                                  gid: u32,
                                  by_gid: bool,
                                  dx: f64,
                                  dy: f64,
                                  x_scale: f64| {
                let (pen_x, x_frac, pen_y) = glyph_placement(dx, dy);
                let bmp = if fill_as_bitmap {
                    if by_gid {
                        face.make_glyph_by_gid_cached_fitted(glyph_cache, gid, x_frac, x_scale)
                    } else {
                        face.make_glyph_cached_fitted(glyph_cache, gid, x_frac, x_scale)
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
                // applied, y up; translate by the exact x and the bitmap's
                // row and flip y (device y is positive-down).
                let outline = if by_gid {
                    face.glyph_path_by_gid_fitted(gid, x_scale)
                } else {
                    face.glyph_path_fitted(char_code_for_path, x_scale)
                };
                let Some(mut path) = outline else {
                    return;
                };
                let py = f64::from(pen_y);
                for pt in &mut path.pts {
                    pt.x += dx;
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
                // Type 0 composite font: the Encoding CMap splits the string
                // into 1–4-byte codes and maps each to a CID (PDF §9.7.6.3).
                for DecodedCode {
                    code,
                    value: cid,
                    len,
                } in enc.codes(bytes)
                {
                    // A substitute face for a composite font carries a
                    // CID-indexed glyph table (vertical forms included); an
                    // embedded one is selected through `CIDToGIDMap` / its
                    // charset, with its vertical form substituted here.  A
                    // substitute's `.notdef` is never the document's glyph:
                    // a CID neither it nor a fallback draws leaves a gap.
                    let drawn = face
                        .fallback_glyph(code, cid)
                        .map(|(fallback, gid)| (fallback, gid, true))
                        .or_else(|| {
                            let by_gid = face.code_to_gid.is_empty();
                            let mut gid = if by_gid { enc.gid_for_cid(cid) } else { cid };
                            if by_gid && enc.vertical {
                                gid = face.program().vertical_selector(gid);
                            }
                            let notdef = if by_gid {
                                gid == 0
                            } else {
                                face.glyph_index(gid) == 0
                            };
                            (!(face.substitute && notdef)).then_some((&**face, gid, by_gid))
                        });
                    let (x_scale, centre) = match drawn {
                        Some(drawn) if fit => {
                            fit_to_advance(enc.width_for_cid(cid), face_width(advance_of(drawn)))
                        }
                        _ => (1.0, 0.0),
                    };
                    // In vertical writing mode the glyph's vertical origin
                    // sits at the pen: its horizontal origin is displaced
                    // by the position vector −v, in glyph units scaled by
                    // the font size (PDF §9.7.4.3).
                    let (dx, dy) = if enc.vertical {
                        let v = enc.vertical_metrics_for_cid(cid);
                        let tx = -f64::from(v.vx) / 1000.0 * font_size * horiz_scaling;
                        let ty = (-f64::from(v.vy) / 1000.0).mul_add(font_size, rise);
                        text_to_device(&ctm, &tm, tx, ty, self.height)
                    } else {
                        let tx = centre / 1000.0 * font_size * horiz_scaling;
                        text_to_device(&ctm, &tm, tx, rise, self.height)
                    };
                    if let Some((drawn, selector, by_gid)) = drawn {
                        push_glyph(drawn, cid, selector, by_gid, dx, dy, x_scale);
                    }

                    // The advance (PDF §9.4.4): horizontally
                    // `(w0 × Tfs + Tc + Tw) × Th`; vertically
                    // `w1 × Tfs + Tc + Tw`, along the text matrix's y axis.
                    // Word spacing applies to the byte 32 when the `CMap`
                    // reads it as a one-byte code, never to a multi-byte code
                    // (PDF §9.3.3).
                    let extra = if code == 32 && len == 1 {
                        word_spacing
                    } else {
                        0.0
                    };
                    if enc.vertical {
                        let w1 = f64::from(enc.vertical_metrics_for_cid(cid).w1y) / 1000.0;
                        let ty_adv = glyph_advance(w1, font_size, char_spacing, extra, 1.0);
                        tm = translate_text_y(tm, ty_adv);
                    } else {
                        let advance_glyph = f64::from(enc.width_for_cid(cid)) / 1000.0;
                        let tx_adv = glyph_advance(
                            advance_glyph,
                            font_size,
                            char_spacing,
                            extra,
                            horiz_scaling,
                        );
                        tm = translate_text_x(tm, tx_adv);
                    }
                }
            } else {
                // Simple font: one byte per character code.
                // PDF §9.2.4: the Widths array is the authoritative advance source
                // for simple fonts (within [FirstChar, LastChar]); MissingWidth
                // applies out-of-range.  The font program's advance is consulted
                // ONLY when Widths is absent entirely (e.g. the 14 standard fonts
                // pre-PDF 1.5, where a substitute face supplies metrics).
                for &byte in bytes {
                    let code = u32::from(byte);
                    let pdf_width = descriptor.width_for_code(code);
                    // A substitute's `.notdef` is never the document's
                    // glyph: a code neither it nor a fallback draws leaves
                    // a gap.
                    let drawn = face
                        .fallback_glyph(code, code)
                        .map(|(fallback, gid)| (fallback, gid, true))
                        .or_else(|| {
                            let notdef = face.glyph_index(code) == 0;
                            (!(face.substitute && notdef)).then_some((&**face, code, false))
                        });
                    let (x_scale, centre) = match (pdf_width, drawn) {
                        (Some(width), Some(drawn)) if fit => {
                            fit_to_advance(width, face_width(advance_of(drawn)))
                        }
                        _ => (1.0, 0.0),
                    };
                    let tx = centre / 1000.0 * font_size * horiz_scaling;
                    let (dx, dy) = text_to_device(&ctm, &tm, tx, rise, self.height);
                    if let Some((drawn, selector, by_gid)) = drawn {
                        push_glyph(drawn, code, selector, by_gid, dx, dy, x_scale);
                    }

                    // PDF §9.4.4: pen advances even for missing/invisible glyphs.
                    let advance_glyph = pdf_width.map_or_else(
                        || {
                            drawn
                                .map_or_else(|| face.glyph_advance(code), advance_of)
                                .unwrap_or(0.0)
                        },
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
        if do_fill && !self.gstate.current().fill_paints_nothing() {
            let fill_bytes = self.gstate.current().painted_fill_color().to_vec();
            let clip = self.gstate.current().clip.clone_shared();
            let blend_mode = self.gstate.current().blend_mode;
            let backdrop = self.blend_backdrop(blend_mode);
            let pipe = Self::make_pipe(
                self.gstate.current().fill_alpha,
                blend_mode,
                backdrop.as_deref(),
            );
            let src = PipeSrc::Solid(&fill_bytes);

            // The glyphs, as bitmaps or filled outlines, are one object under
            // the soft mask in effect; the outline fills lie within one pixel
            // of their points.
            let bitmap_corners = records.iter().flat_map(|rec| {
                let x = f64::from(rec.pen_x) + f64::from(rec.bitmap.x_off);
                let y = f64::from(rec.pen_y) + f64::from(rec.bitmap.y_off);
                [
                    (x, y),
                    (
                        x + f64::from(rec.bitmap.width),
                        y + f64::from(rec.bitmap.height),
                    ),
                ]
            });
            let outline_points = fill_paths
                .iter()
                .flat_map(|path| path.pts.iter().map(|p| (p.x, p.y)));
            let region = PixelRect::around(
                bitmap_corners.chain(outline_points),
                1.0,
                self.width,
                self.height,
            );
            let alpha = self.gstate.current().fill_alpha;
            self.paint_object(region, alpha, |r| {
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
                        &mut r.bitmap,
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
                        r.fill_path(&union, false);
                    }
                } else {
                    for path in &fill_paths {
                        r.fill_path(path, false);
                    }
                }
            });
        }
        // The stroke parameters are user-space quantities under the CTM, as
        // for any stroked path (PDF §8.4.3.2); the text matrix and font size
        // scale the outline but not the line width.
        if do_stroke && !stroke_outline.pts.is_empty() {
            self.stroke_text_outline(&stroke_outline);
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
