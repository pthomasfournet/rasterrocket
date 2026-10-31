//! `Do`: image and form `XObjects`, and the transparency groups of forms.

use std::sync::Arc;

use color::Rgb8;
use pdf::ObjectId;
use raster::{
    Bitmap,
    transparency::{GroupParams, begin_group, paint_group},
    types::BlendMode,
};

use super::super::gstate::{ctm_multiply, ctm_transform};
use super::pixel_rect::PixelRect;
use super::{GroupContext, MAX_FORM_DEPTH, PageRenderer, RenderAbort};

/// Device-pixel extent `(width, height)` of an image drawn under `ctm`: the
/// lengths of the images of its unit x and y vectors, rounded up.  Correct
/// under rotation and skew.  `None` for a degenerate or non-finite matrix.
pub(super) fn image_draw_size(ctm: &[f64; 6]) -> Option<(u32, u32)> {
    let extent = |dx: f64, dy: f64| {
        let len = dx.hypot(dy).ceil();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "len is finite, at least 1 and at most u32::MAX, checked first"
        )]
        (len.is_finite() && (1.0..=f64::from(u32::MAX)).contains(&len)).then_some(len as u32)
    };
    Some((extent(ctm[0], ctm[1])?, extent(ctm[2], ctm[3])?))
}

impl PageRenderer<'_> {
    // ── XObject rendering ─────────────────────────────────────────────────────

    /// Execute a `Do` operator: look up and render the named `XObject`.
    ///
    /// An `XObject` whose `/OC` is hidden is neither decoded nor drawn.
    /// Image `XObjects` are blitted directly.  Form `XObjects` are executed
    /// recursively via [`do_form_xobject`].
    pub(super) fn do_xobject(&mut self, name: &[u8]) {
        if let Some(oc) = self.resources.xobject_optional_content(name)
            && !self.oc_visible(&oc)
        {
            return;
        }
        // Try Form first (cheap dict lookup, no pixel decoding).
        if let Some(form) = self.resources.form_xobject(name) {
            self.do_form_xobject(&form);
            return;
        }
        // Try Image, decoding it only when this page has not already.
        let draw_size = image_draw_size(&self.gstate.current().ctm);
        let id = self.resources.xobject_id(name);
        let memoized = id.and_then(|id| self.image_memo.borrow().get(id, draw_size));
        if let Some(decoded) = memoized {
            self.draw_image_xobject(name, &decoded);
            return;
        }
        let request = crate::resources::ImageRequest { name, draw_size };
        let mut ctx = crate::resources::image::ImageDecodeCtx {
            doc: self.resources.doc(),
            color_spaces: self.resources.color_space_dict(),
            #[cfg(feature = "nvjpeg")]
            nvjpeg: self.nvjpeg.as_mut(),
            #[cfg(feature = "vaapi")]
            vaapi: self.vaapi_jpeg_queue.as_ref(),
            #[cfg(feature = "nvjpeg2k")]
            nvjpeg2k: self.nvjpeg2k.as_mut(),
            #[cfg(feature = "gpu-icc")]
            icc: crate::resources::image::IccHandles {
                gpu_ctx: self.gpu_ctx.as_deref(),
            },
            #[cfg(feature = "cache")]
            image_cache: self.cache_state.as_ref().map(|cs| &cs.cache),
            #[cfg(feature = "cache")]
            doc_id: self.cache_state.as_ref().map(|cs| cs.doc_id),
        };
        // Route to whichever GPU JPEG decoder is active (only one is Some per render).
        // Each branch is a separately monomorphised call; no decoder is wasted.
        #[cfg(all(feature = "gpu-jpeg-huffman", feature = "vulkan"))]
        let resolution = if self.jpeg_vk.is_some() {
            self.resources
                .image(request, &mut ctx, self.jpeg_vk.as_mut())
        } else {
            self.resources
                .image(request, &mut ctx, self.jpeg_gpu.as_mut())
        };
        #[cfg(not(all(feature = "gpu-jpeg-huffman", feature = "vulkan")))]
        let resolution = self.resources.image(
            request,
            &mut ctx,
            #[cfg(feature = "gpu-jpeg-huffman")]
            self.jpeg_gpu.as_mut(),
        );
        self.handle_image_resolution(name, resolution, id.map(|id| (id, draw_size)));
    }

    /// Dispatch on an [`ImageResolution`] returned from [`PageResources::image`].
    ///
    /// - `Ok` → blit the image, keeping it in the page's image memo under
    ///   `memo_key` (the stream object and the draw size it was decoded
    ///   for) when given.
    /// - `DecodeFailed` → record the failure in the page diagnostics and skip
    ///   the image; the pixel region keeps what was painted beneath it.
    /// - `Absent` → a non-image `XObject` subtype with no raster appearance;
    ///   log a debug message and skip.
    pub(super) fn handle_image_resolution(
        &mut self,
        name: &[u8],
        resolution: crate::resources::image::ImageResolution,
        memo_key: Option<(ObjectId, Option<(u32, u32)>)>,
    ) {
        use crate::resources::image::ImageResolution;
        match resolution {
            ImageResolution::Ok(decoded) => {
                if let Some(reason) = &decoded.partial {
                    self.record_partial_image(Some(name), reason.clone());
                }
                match memo_key {
                    Some((id, draw_size)) => {
                        let decoded = self.image_memo.borrow_mut().insert(id, draw_size, decoded);
                        self.draw_image_xobject(name, &decoded);
                    }
                    None => self.draw_image_xobject(name, &decoded),
                }
            }
            ImageResolution::DecodeFailed(msg) => {
                self.record_image_failure(Some(name), msg);
            }
            ImageResolution::Absent => {
                log::debug!(
                    "rasterrocket-interp: Do /{} skipped (no raster appearance)",
                    String::from_utf8_lossy(name)
                );
            }
        }
    }

    /// Blit the decoded image `XObject` `name`, attributing a blit failure
    /// to it.
    pub(super) fn draw_image_xobject(
        &mut self,
        name: &[u8],
        decoded: &crate::resources::DecodedImage,
    ) {
        self.blit_image(decoded, Some(name));
    }

    /// Execute a Form `XObject`'s content stream in the current graphics context.
    ///
    /// PDF §8.10.1: the form's `Matrix` is concatenated onto the current CTM,
    /// graphics state is saved/restored around execution, and the form's own
    /// `Resources` dict (if present) is used to resolve fonts and images inside
    /// the form.
    ///
    /// When the form carries a `Group /S /Transparency` entry, content is rendered
    /// into an intermediate group bitmap and composited back (PDF §11.6.6).
    ///
    /// A form invoked while it is already executing, directly or through
    /// other forms, draws nothing.
    pub(super) fn do_form_xobject(&mut self, form: &crate::resources::FormXObject) {
        if self.form_depth >= MAX_FORM_DEPTH {
            self.budget_exceeded = Some(RenderAbort::FormDepth);
            log::warn!(
                "rasterrocket-interp: Form XObject nesting depth {MAX_FORM_DEPTH} exceeded — aborting"
            );
            return;
        }
        // Propagate an already-tripped budget: a recursive form call after
        // the outer execute() set budget_exceeded must not execute further.
        if self.budget_exceeded.is_some() {
            return;
        }
        if self.active_forms.contains(&form.resources_id) {
            log::warn!(
                "rasterrocket-interp: Form XObject ({}, {}) invokes itself — recursive Do skipped",
                form.resources_id.0,
                form.resources_id.1
            );
            return;
        }
        self.active_forms.push(form.resources_id);

        // Capture the parent's compositing state before saving, so the group is
        // painted using the opacity/blend that was active when the form was invoked
        // (PDF §11.6.6: the group is composited using the surrounding context).
        let parent_fill_alpha = self.gstate.current().fill_alpha;
        let parent_blend_mode = self.gstate.current().blend_mode;

        // Save graphics state (equivalent to `q`).
        self.gstate.save();
        self.form_depth += 1;

        // Concatenate the form's Matrix onto the current CTM.
        // PDF §8.10.1: the form's /Matrix maps form space to the parent user
        // space in effect when `Do` was invoked; it pre-multiplies, exactly as
        // `cm` does (§8.3.4) — new_CTM = form.matrix × old_CTM, never the
        // reverse.  Post-multiplying scales the matrix's translation by the
        // wrong factor and pushes nested-form content off the device page.
        let old_ctm = self.gstate.current().ctm;
        let form_ctm = ctm_multiply(&form.matrix, &old_ctm);
        self.gstate.current_mut().ctm = form_ctm;
        let parent_pattern_ctm = std::mem::replace(&mut self.pattern_ctm, form_ctm);
        // PDF §8.10.1: the form's BBox, in form space, clips its content.
        if let Some(bbox) = form.bbox {
            self.clip_to_user_rect(&bbox);
        }

        // Switch to the form's resource context, keeping the parent for restore.
        let child_resources = self.resources.for_form(form);
        let parent_resources = std::mem::replace(&mut self.resources, child_resources);

        // Transparency group: render into an intermediate bitmap, then
        // composite it back onto the page.
        let mut group = form
            .transparency
            .and_then(|tg| self.group_params(form.bbox, tg))
            .map(|params| {
                let bitmap = self.group_bitmap();
                begin_group(
                    &self.bitmap,
                    self.group.backdrop.as_deref(),
                    &self.gstate.current().clip,
                    params,
                    bitmap,
                )
            });

        // Swap in the group bitmap when present.  PDF §11.6.6: the alpha
        // constants and blend mode start at their initial values inside a
        // transparency group; the parent's values apply once, when the group
        // is composited.
        // A group is one object under the soft mask in effect: the mask
        // applies when the group is composited, not to its content.
        let group_mask = group
            .as_ref()
            .and_then(|_| self.gstate.current_mut().soft_mask.take());

        // Blend modes inside the group see its own backdrop: the parent's
        // pixels for a non-isolated group, nothing for an isolated one; and
        // its objects knock out one another when it is a knockout group,
        // whatever the enclosing group is.
        let parent_group = group.as_mut().map(|g| {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
            let (x0, y0, x1, y1) = g.region();
            let gs = self.gstate.current_mut();
            gs.fill_alpha = 255;
            gs.stroke_alpha = 255;
            gs.blend_mode = BlendMode::Normal;
            // The content is clipped to the region, so the group bitmap is
            // painted nowhere else and clearing the region restores it.
            gs.clip
                .clip_to_rect(f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1));
            let context = GroupContext {
                backdrop: g.backdrop.take().map(Arc::new),
                knockout: form.transparency.is_some_and(|tg| tg.knockout),
            };
            std::mem::replace(&mut self.group, context)
        });

        // Parse and execute the form's content stream.  Its `Q`s cannot
        // pop the state saved above.
        let components = |name: &[u8]| self.resources.inline_image_components(name);
        let ops = crate::content::parse_with(&form.content, Some(&components));
        self.execute_content_stream(&ops);

        // Swap the group bitmap back and composite it as one object of the
        // parent group, using the parent's opacity/blend (captured before
        // gstate.save() — the form's own state is irrelevant here).
        if let (Some(mut g), Some(parent_group)) = (group, parent_group) {
            std::mem::swap(&mut self.bitmap, &mut g.bitmap);
            self.group = parent_group;
            let backdrop = self.blend_backdrop(parent_blend_mode);
            let pipe = Self::make_pipe(parent_fill_alpha, parent_blend_mode, backdrop.as_deref());
            let (x0, y0, x1, y1) = g.region();
            let region = PixelRect::new(x0, y0, x1, y1);
            // Residual Clip is unused — parent gstate is restored immediately below.
            let (_, bitmap) =
                self.paint_knocking_out(region, parent_fill_alpha, |r| match group_mask {
                    Some(mask) => r.paint_through_soft_mask(&mask, region, |r| {
                        paint_group(&mut r.bitmap, g, &pipe)
                    }),
                    None => paint_group(&mut r.bitmap, g, &pipe),
                });
            self.group_pool.push(bitmap);
        }

        // Restore resources and graphics state.
        self.resources = parent_resources;
        self.pattern_ctm = parent_pattern_ctm;
        self.form_depth -= 1;
        let _ = self.active_forms.pop();
        self.gstate.restore();
    }

    /// A transparent bitmap of the current bitmap's size for a transparency
    /// group: one from the pool, or a fresh one.
    pub(super) fn group_bitmap(&mut self) -> Bitmap<Rgb8> {
        let (width, height) = (self.bitmap.width, self.bitmap.height);
        self.group_pool
            .pop()
            .filter(|b| b.width == width && b.height == height)
            .unwrap_or_else(|| Bitmap::new(width, height, 1, true))
    }

    /// The parameters of a transparency group `tg` whose form has `bbox`:
    /// the box's extent under the current CTM, in device pixels, or the
    /// whole page when the form has no box.  `None` when the extent is
    /// non-finite; the form then paints directly onto the page.
    pub(super) fn group_params(
        &self,
        bbox: Option<[f64; 4]>,
        tg: crate::resources::TransparencyGroupParams,
    ) -> Option<GroupParams> {
        let Some(bbox) = bbox else {
            return Some(GroupParams {
                x_min: 0,
                y_min: 0,
                x_max: i32::try_from(self.width).unwrap_or(i32::MAX),
                y_max: i32::try_from(self.height).unwrap_or(i32::MAX),
                isolated: tg.isolated,
            });
        };
        let ctm = self.gstate.current().ctm;
        let page_h = f64::from(self.height);
        let [bx0, by0, bx1, by1] = bbox;
        let corners = [
            ctm_transform(&ctm, bx0, by0),
            ctm_transform(&ctm, bx1, by0),
            ctm_transform(&ctm, bx0, by1),
            ctm_transform(&ctm, bx1, by1),
        ];
        if corners
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            log::warn!(
                "rasterrocket-interp: transparency group BBox is non-finite — rendering without group"
            );
            return None;
        }
        let left = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::INFINITY, f64::min)
            .floor();
        let top = corners
            .iter()
            .map(|(_, y)| page_h - y)
            .fold(f64::INFINITY, f64::min)
            .floor();
        let right = corners
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil();
        let bottom = corners
            .iter()
            .map(|(_, y)| page_h - y)
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→i32 cast saturates in Rust ≥ 1.45 (no UB); \
                      begin_group clamps GroupParams to bitmap bounds"
        )]
        let params = GroupParams {
            x_min: left as i32,
            y_min: top as i32,
            x_max: right as i32,
            y_max: bottom as i32,
            isolated: tg.isolated,
        };
        Some(params)
    }
}
