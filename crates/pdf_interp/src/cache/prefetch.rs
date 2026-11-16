//! Background image-cache prefetcher.
//!
//! At session start the renderer typically has no idea which images
//! it'll need until it actually walks the content stream of each
//! page.  For multi-page or multi-pass workloads (OCR pipelines,
//! re-renders of the same PDF) priming the cache up front means the
//! first render hits the cache instead of paying the JPEG decode +
//! upload cost serially on the render hot path.
//!
//! This module:
//! 1. Walks `doc.get_pages()`; for each page walks its content stream, and
//!    the forms it paints, for the image `XObject`s drawn and the
//!    transformation in effect at each `Do`.
//! 2. Hands each image a draw of which the cache serves to a small worker
//!    pool that decodes + inserts into the cache.
//! 3. Dedupes by `ObjId` so an image referenced from N pages is
//!    decoded once, not N times.
//!
//! The draw sizes need the scale the pages render at
//! ([`PrefetchConfig::scale`]).  An image every draw of which reduces it
//! (see [`draw_reduces`]) is not prefetched: the render resamples such a
//! draw from host-resident samples and bypasses the cache, so its decode
//! would be wasted work.  Without one scale for every page nothing is
//! prefetched, since no draw can be told to be one the cache serves.
//!
//! Only `/DCTDecode` (JPEG) images are prefetched — those are the
//! decode-expensive case and the one the [`gpu::cache::ImageCache`]
//! was built for.  CCITT, JBIG2, and JPEG-2000 images decode on the
//! renderer's first touch.
//!
//! Workers are plain `std::thread`s, not rayon, so prefetch never
//! steals capacity from the page-render pool.  The default worker
//! count is intentionally low (2) — the goal is to be done before
//! the renderer reaches the image, not to saturate the CPU.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use gpu::cache::{DocId, ImageCache, ObjId};
use pdf::{Document, Object, ObjectId};

use crate::content::Operator;
use crate::renderer::PageView;
use crate::renderer::gstate::{Ctm, ctm_multiply};
use crate::renderer::page::{MAX_FORM_DEPTH, draw_reduces, image_draw_size};
use crate::resources::dict_ext::DictExt;
use crate::resources::image::{ImageFilter, ImageRequest, filter_name, validated_dims};
use crate::resources::{PageResources, resolve_image};
use crate::{PageBox, Rotation};

/// Upper bound on prefetcher worker threads.
///
/// Caps an arbitrary caller-supplied [`PrefetchConfig::workers`]
/// value so a misconfig (typo, overflow from `num_cpus * factor`)
/// can't spawn thousands of threads.  Beyond ~16 workers the
/// JPEG-decode pool would contend with the renderer's rayon pool
/// anyway.
pub const MAX_PREFETCH_WORKERS: usize = 16;

// `Ord::clamp` panics when min > max.
const _: () = assert!(MAX_PREFETCH_WORKERS >= 1);

/// A caller-supplied worker count clamped to `[1, MAX_PREFETCH_WORKERS]`.
fn clamp_workers(n: usize) -> usize {
    n.clamp(1, MAX_PREFETCH_WORKERS)
}

/// Tunables for [`spawn_prefetch`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct PrefetchConfig {
    /// Number of background worker threads.  Default 2; bumping this
    /// only helps if the renderer hasn't started yet (i.e. the
    /// session was opened well in advance of the first render call).
    /// Clamped to the range `1..=MAX_PREFETCH_WORKERS` at spawn time
    /// (see [`MAX_PREFETCH_WORKERS`]).
    pub workers: usize,
    /// Hard cap on the number of distinct images the prefetcher will
    /// decode.  Acts as a guardrail for adversarial PDFs that list
    /// tens of thousands of image `XObject`s; the renderer can still
    /// decode unprefetched images on first touch.  Default 4096.
    pub max_images: usize,
    /// Device pixels per point `(x, y)` along the device axes at which every
    /// page of the session renders, when that is one scale for all pages.
    /// An image is prefetched only when some draw of it on a page, or in a
    /// form the page paints, does not reduce it.  `None` (the default)
    /// prefetches nothing, as does a scale that is not finite and positive.
    pub scale: Option<(f64, f64)>,
    /// The session's rotation, added to each page's `/Rotate`; it decides
    /// which device axis each scale applies to.  Default
    /// [`Rotation::None`].
    pub rotation: Rotation,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            workers: 2,
            max_images: 4096,
            scale: None,
            rotation: Rotation::None,
        }
    }
}

impl PrefetchConfig {
    /// The scale of [`Self::scale`] when it is finite and positive along
    /// both axes.
    fn valid_scale(&self) -> Option<(f64, f64)> {
        self.scale.filter(|&scale| {
            <[f64; 2]>::from(scale)
                .into_iter()
                .all(|s| s.is_finite() && s > 0.0)
        })
    }
}

/// Handle to a running prefetch job.  Drop the handle to cancel
/// in-flight prefetch and join the workers.
pub struct PrefetchHandle {
    cancel: Arc<AtomicBool>,
    workers: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Drop for PrefetchHandle {
    fn drop(&mut self) {
        // Signal cancellation; workers check this between images and
        // exit promptly when set.  The mpsc receiver also drops with
        // the sender so any worker blocked on `recv` wakes up too.
        self.cancel.store(true, Ordering::Release);
        // Drain under the lock, then join unlocked — JoinHandle::join
        // blocks for the worker's full runtime, which is too long to
        // hold the workers mutex.
        let drained: Vec<_> = {
            let mut guard = self
                .workers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.drain(..).collect()
        };
        for handle in drained {
            let _ = handle.join();
        }
    }
}

/// Spawn a prefetch job for `doc` against `cache` keyed by `doc_id`.
///
/// Returns immediately with a [`PrefetchHandle`]; the actual work
/// happens on background threads.  The renderer can start
/// immediately — cache hits become available the moment each
/// worker finishes a decode.
///
/// # Concurrency
///
/// `Document` is `Send + Sync` and the cache is internally
/// thread-safe, so handing `Arc`s to multiple workers is safe.
/// Discovery dedupes via a single-threaded `HashSet<ObjId>` local
/// to a private `discover_pages` helper; an image referenced from
/// many pages is only decoded once.
///
/// Without a valid [`PrefetchConfig::scale`] no thread is started and
/// nothing is prefetched, which is reported at warn.
///
/// # Errors
///
/// The error of `std::thread::Builder::spawn` when the OS cannot create a
/// thread.  The threads already started are stopped and joined before it
/// is returned; the caller renders without prefetching.
pub fn spawn_prefetch(
    doc: Arc<Document>,
    cache: ImageCache,
    doc_id: DocId,
    config: PrefetchConfig,
) -> std::io::Result<PrefetchHandle> {
    let cancel = Arc::new(AtomicBool::new(false));
    let Some(scale) = config.valid_scale() else {
        log::warn!(
            "image prefetch: the pages render at no one scale, so no draw can be told to be \
             one the image cache serves; nothing is prefetched"
        );
        return Ok(PrefetchHandle {
            cancel,
            workers: Mutex::new(Vec::new()),
        });
    };
    let state = Arc::new(PrefetchState {
        cache,
        doc,
        doc_id,
        cancel,
        scale,
        rotation: config.rotation,
    });
    let workers = clamp_workers(config.workers);
    let mut handles = Vec::with_capacity(workers + 1);
    let spawned = spawn_threads(&mut handles, &state, workers, config.max_images);
    let handle = PrefetchHandle {
        cancel: Arc::clone(&state.cancel),
        workers: Mutex::new(handles),
    };
    // On failure the channel's sender is already dropped, so dropping the
    // handle wakes and joins every worker started.
    spawned.map(|()| handle)
}

/// Start `workers` decode workers and the discovery thread, pushing each
/// handle onto `handles`.
///
/// The job channel's sender lives only in the discovery thread (or in
/// this frame until it starts), so when this returns, by success or
/// failure, the workers end once the channel is drained.
fn spawn_threads(
    handles: &mut Vec<thread::JoinHandle<()>>,
    state: &Arc<PrefetchState>,
    workers: usize,
    max_images: usize,
) -> std::io::Result<()> {
    // Unbounded mpsc; the cap is `max_images`, enforced by the discovery
    // loop, so memory is bounded.
    let (tx, rx) = mpsc::channel::<PrefetchJob>();
    let rx = Arc::new(Mutex::new(rx));

    // Workers first, so they are ready when discovery begins.
    for worker_idx in 0..workers {
        let rx = Arc::clone(&rx);
        let state = Arc::clone(state);
        handles.push(
            thread::Builder::new()
                .name(format!("pdf-prefetch-{worker_idx}"))
                .spawn(move || worker_loop(&rx, &state))?,
        );
    }

    // Discovery walks the pages and pushes jobs; its exit drops `tx`,
    // closing the channel.
    let state = Arc::clone(state);
    handles.push(
        thread::Builder::new()
            .name("pdf-prefetch-walk".to_string())
            .spawn(move || discover_pages(&state, max_images, &tx))?,
    );
    Ok(())
}

/// Job payload passed from the discovery thread to the worker pool.
struct PrefetchJob {
    /// The image `XObject` stream.
    stream_id: ObjectId,
    /// The device-pixel extent of a draw of the image that does not reduce
    /// it; `None` for a full-resolution decode.
    draw_size: Option<(u32, u32)>,
}

/// Bundled state borrowed by both [`discover_pages`] and
/// [`worker_loop`] — keeps each function's arg list sane and means
/// the spawn site only owns one set of `Arc`s.
///
/// `seen` doesn't live here: discovery is single-threaded and is
/// the only writer, so the dedup set is a plain `HashSet` owned
/// inside [`discover_pages`].
struct PrefetchState {
    cache: ImageCache,
    doc: Arc<Document>,
    doc_id: DocId,
    cancel: Arc<AtomicBool>,
    /// The valid [`PrefetchConfig::scale`].
    scale: (f64, f64),
    /// The session rotation.
    rotation: Rotation,
}

/// Walk every page's draws ([`page_draws`]) and emit a [`PrefetchJob`]
/// for each `/DCTDecode`-filtered image `XObject` we haven't already
/// seen, with the size of a draw on the page that does not reduce it
/// (see [`cached_draw`]).  An image every draw of which on a page reduces
/// it is skipped there; a later page drawing it larger still emits it.
fn discover_pages(state: &PrefetchState, max_images: usize, tx: &mpsc::Sender<PrefetchJob>) {
    // Discovery is single-threaded so the dedup set is a plain
    // local — no Mutex / Arc.  An image referenced from N pages
    // therefore generates one PrefetchJob, not N.  Pre-sized so
    // logo-heavy decks (thousands of XObjects) skip the first few
    // rehashes; capped at max_images so the prefetcher never
    // allocates more dedup space than it'd ever populate.
    let mut seen: HashSet<ObjId> = HashSet::with_capacity(max_images.min(256));
    let mut emitted = 0usize;
    let doc = state.doc.as_ref();
    'pages: for (_page_num, page_id) in doc.get_pages() {
        if state.cancel.load(Ordering::Acquire) {
            break;
        }
        // A page without `XObject` resources paints none.
        if doc.get_page_resource_dict(page_id, b"XObject").is_err() {
            continue;
        }
        let draws = page_draws(doc, page_id, state.scale, state.rotation);
        for (stream_id, sizes) in &draws {
            if state.cancel.load(Ordering::Acquire) {
                break 'pages;
            }
            if emitted >= max_images {
                log::debug!("prefetch: hit max_images={max_images}, stopping discovery early");
                break 'pages;
            }
            let obj_id = ObjId(stream_id.0);
            if seen.contains(&obj_id) {
                continue;
            }
            // Only DCT images with valid dimensions are pushed; the
            // worker re-resolves the stream and owns the decode.
            let Some(dims) = dct_image_dims(doc, *stream_id)
                .filter(|_| state.cache.lookup_by_id(state.doc_id, obj_id).is_none())
            else {
                let _ = seen.insert(obj_id);
                continue;
            };
            let Prefetch::At(draw_size) = cached_draw(dims, sizes) else {
                continue;
            };
            let _ = seen.insert(obj_id);
            if tx
                .send(PrefetchJob {
                    stream_id: *stream_id,
                    draw_size,
                })
                .is_err()
            {
                // Receiver dropped — handle was cancelled.
                break 'pages;
            }
            emitted += 1;
        }
    }
}

/// Worker drain loop.  Pulls one job at a time off the shared
/// receiver and decodes it.  Returns when the channel is closed
/// (discovery thread exited) or cancellation flips.
fn worker_loop(rx: &Mutex<mpsc::Receiver<PrefetchJob>>, state: &PrefetchState) {
    loop {
        if state.cancel.load(Ordering::Acquire) {
            break;
        }
        // Each `recv` is a single-message critical section; multiple
        // workers contend on the same mpsc receiver here.  This is
        // fine for the prefetch hot path — JPEG decode (~ms per
        // image) dominates by orders of magnitude.
        let job = {
            let guard = rx.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.recv()
        };
        let Ok(job) = job else { break };

        // Under an unwinding panic strategy a panic in one decode is
        // contained here and the worker moves on; under `panic = "abort"`
        // (the release profile) it ends the process before reaching this.
        // `resolve_image` reports soft failures itself.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            decode_one(&state.doc, state.doc_id, &state.cache, &job);
        }));
        if result.is_err() {
            log::warn!(
                "prefetch: decode panicked for image object {}",
                job.stream_id.0
            );
        }
    }
}

/// Decode one image `XObject` into the cache via the shared
/// [`resolve_image`] entry point with all GPU back-ends disabled.
/// `resolve_image` logs its own failures at warn level; this fn
/// is fire-and-forget from the worker's perspective.
fn decode_one(doc: &Document, doc_id: DocId, cache: &ImageCache, job: &PrefetchJob) {
    // The stream under a name of its own: the decode resolves an image by
    // name in an `XObject` dictionary.
    const NAME: &[u8] = b"Im";
    let mut xobjects = pdf::Dictionary::new();
    xobjects.set(NAME, Object::Reference(job.stream_id));
    let mut ctx = crate::resources::image::ImageDecodeCtx {
        image_cache: Some(cache),
        doc_id: Some(doc_id),
        ..crate::resources::image::ImageDecodeCtx::cpu(doc)
    };
    let _ = resolve_image(
        &mut ctx,
        &xobjects,
        ImageRequest {
            name: NAME,
            draw_size: job.draw_size,
        },
    );
}

/// Cheap pre-filter: the `/Width` and `/Height` of `stream_id` when it
/// resolves to a stream with `/Subtype Image`, `/Filter DCTDecode` and
/// valid dimensions; `None` otherwise.  Avoids spinning up a worker for
/// non-image references, non-DCT images, or images the decode rejects.
///
/// Filter normalisation (single-name vs single-element array vs
/// chained filters vs empty array) is delegated to the shared
/// [`filter_name`] helper used by the image decode path so the
/// prefetcher and the renderer agree on what "is JPEG" means.
fn dct_image_dims(doc: &Document, stream_id: ObjectId) -> Option<(u32, u32)> {
    let obj_arc = doc.get_object(stream_id).ok()?;
    let dict = &obj_arc.as_stream()?.dict;
    if !matches!(dict.get(b"Subtype"), Some(Object::Name(n)) if n == b"Image") {
        return None;
    }
    let filter = ImageFilter::from_filter_str(filter_name(dict.get(b"Filter")?).as_deref());
    if !matches!(filter, ImageFilter::Dct) {
        return None;
    }
    validated_dims(dict.get_i64(b"Width")?, dict.get_i64(b"Height")?)
}

/// The prefetch of an image on one page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prefetch {
    /// No draw on the page reads the image cache.
    Skip,
    /// Decode for a draw of this device-pixel extent; `None` decodes at
    /// full resolution.
    At(Option<(u32, u32)>),
}

/// The prefetch of a `dims` image given `draws`, the device-pixel extents
/// (`image_draw_size`) of its draws on a page: at a draw that does not
/// reduce the image, or [`Prefetch::Skip`] when every draw reduces it or
/// the page draws it nowhere.  A draw of unknown extent decodes at full
/// resolution and does not reduce.
fn cached_draw((width, height): (u32, u32), draws: &HashSet<Option<(u32, u32)>>) -> Prefetch {
    draws
        .iter()
        .copied()
        .find(|draw| !draw.is_some_and(|draw| draw_reduces(width, height, draw)))
        .map_or(Prefetch::Skip, Prefetch::At)
}

/// The device-pixel extents (`image_draw_size`) of the draws of each
/// `XObject` that the content stream of page `page_id`, or a form it
/// paints, paints by name, keyed by the `XObject`'s stream, at `scale`
/// device pixels per point under the session rotation `rotation`.  Only
/// `q`, `Q`, `cm` and a form's `/Matrix` change the transformation
/// followed.  Forms are followed to the nesting depth the renderer
/// executes ([`MAX_FORM_DEPTH`]); a form is not entered from within
/// itself.  Empty when the page's geometry or content cannot be read.
fn page_draws(doc: &Document, page_id: ObjectId, scale: (f64, f64), rotation: Rotation) -> Draws {
    let mut walk = DrawWalk::default();
    // The extents depend on the linear part of the page CTM alone, which
    // the page box does not enter.
    let Ok(geom) = crate::page_size_pts_by_id(doc, page_id, PageBox::default()) else {
        return walk.draws;
    };
    let Ok(content) = doc.page_content(page_id) else {
        return walk.draws;
    };
    let ctm: Ctm = PageView::whole(&geom.rotated(rotation), scale).page_ctm();
    let ops = crate::parse_page_content(doc, page_id, &content.bytes);
    walk.content(&PageResources::new(doc, page_id), &ops, ctm, 0);
    walk.draws
}

/// The draw extents of each `XObject` stream (see [`page_draws`]).
type Draws = HashMap<ObjectId, HashSet<Option<(u32, u32)>>>;

/// The state of [`page_draws`]: the draws found and the forms being walked.
#[derive(Default)]
struct DrawWalk {
    draws: Draws,
    /// The forms whose content is being walked, outermost first.
    forms: Vec<ObjectId>,
}

impl DrawWalk {
    /// Record the draws of `ops`, a content stream whose names resolve in
    /// `resources`, entered under `ctm` at form nesting depth `depth`.
    fn content(
        &mut self,
        resources: &PageResources<'_>,
        ops: &[Operator<'_>],
        mut ctm: Ctm,
        depth: u32,
    ) {
        let mut saved: Vec<Ctm> = Vec::new();
        for op in ops {
            match op {
                Operator::Save => saved.push(ctm),
                Operator::Restore => {
                    if let Some(outer) = saved.pop() {
                        ctm = outer;
                    }
                }
                Operator::ConcatMatrix(m) => ctm = ctm_multiply(m, &ctm),
                Operator::PaintXObject(name) => self.paint(resources, name, ctm, depth),
                _ => {}
            }
        }
    }

    /// Record the draw of `XObject` `name` of `resources` under `ctm`, at
    /// form nesting depth `depth`: an image's extent, or a form's draws.
    fn paint(&mut self, resources: &PageResources<'_>, name: &[u8], ctm: Ctm, depth: u32) {
        let Some(id) = resources.xobject_id(name) else {
            return;
        };
        if resources.form_xobject_entry(name).is_none() {
            let _ = self
                .draws
                .entry(id)
                .or_default()
                .insert(image_draw_size(&ctm));
            return;
        }
        if depth >= MAX_FORM_DEPTH || self.forms.contains(&id) {
            return;
        }
        let Some(form) = resources.form_xobject(name) else {
            return;
        };
        let inner = resources.for_form(&form);
        let components = |name: &[u8]| inner.inline_image_components(name);
        let ops = crate::content::parse_with(&form.content, Some(&components));
        self.forms.push(id);
        self.content(&inner, &ops, ctm_multiply(&form.matrix, &ctm), depth + 1);
        let _ = self.forms.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_pdf_no_images() -> Vec<u8> {
        b"%PDF-1.4\n\
1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n\
2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n\
3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]>>\nendobj\n\
xref\n0 4\n\
0000000000 65535 f\r\n\
0000000009 00000 n\r\n\
0000000056 00000 n\r\n\
0000000111 00000 n\r\n\
trailer\n<</Size 4 /Root 1 0 R>>\n\
startxref\n180\n%%EOF"
            .to_vec()
    }

    #[test]
    fn prefetch_config_default_is_two_workers_4k_cap() {
        let c = PrefetchConfig::default();
        assert_eq!(c.workers, 2);
        assert_eq!(c.max_images, 4096);
    }

    #[test]
    fn clamp_workers_caps_high_and_lifts_zero() {
        // Pin the bound so an accidental bump shows up in review.
        assert_eq!(MAX_PREFETCH_WORKERS, 16);
        // Real call into the production helper — a regression that
        // drops the clamp in `spawn_prefetch` would still be caught
        // here so long as both this test and the helper aren't
        // deleted together.
        assert_eq!(clamp_workers(0), 1);
        assert_eq!(clamp_workers(1), 1);
        assert_eq!(clamp_workers(8), 8);
        assert_eq!(clamp_workers(MAX_PREFETCH_WORKERS), MAX_PREFETCH_WORKERS);
        assert_eq!(
            clamp_workers(MAX_PREFETCH_WORKERS + 1),
            MAX_PREFETCH_WORKERS
        );
        assert_eq!(clamp_workers(10_000), MAX_PREFETCH_WORKERS);
        assert_eq!(clamp_workers(usize::MAX), MAX_PREFETCH_WORKERS);
    }

    #[test]
    fn dct_image_dims_rejects_non_streams_and_missing_objects() {
        let doc = Document::from_bytes_owned(minimal_pdf_no_images()).expect("minimal PDF");
        // Object 1 is the catalog dict, not a stream.
        assert_eq!(dct_image_dims(&doc, (1, 0)), None);
        // Object 99 does not exist — must return None, not panic.
        assert_eq!(dct_image_dims(&doc, (99, 0)), None);
    }

    /// Page `(4, 0)`, 400 points square, whose content stream is `content`
    /// and whose `/Im` is a 400 × 400 JPEG image `XObject`.
    fn page_with_jpeg(content: &str) -> Document {
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /MediaBox [0 0 400 400] /Contents 5 0 R \
                 /Resources << /XObject << /Im 6 0 R >> >> >>"
                    .to_owned(),
            ),
            (
                5,
                format!(
                    "<< /Length {} >>\nstream\n{content}\nendstream",
                    content.len()
                ),
            ),
            (
                6,
                "<< /Type /XObject /Subtype /Image /Filter /DCTDecode /Width 400 \
                 /Height 400 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 0 >>\n\
                 stream\n\nendstream"
                    .to_owned(),
            ),
        ])
    }

    /// The prefetch of image `(6, 0)` given the draws `draws`.
    fn prefetch_of(doc: &Document, draws: &Draws) -> Prefetch {
        let dims = dct_image_dims(doc, (6, 0)).expect("a JPEG image");
        draws
            .get(&(6, 0))
            .map_or(Prefetch::Skip, |sizes| cached_draw(dims, sizes))
    }

    /// The prefetch of `/Im` on page `(4, 0)` at `scale`.
    fn prefetched_draw(doc: &Document, scale: (f64, f64)) -> Prefetch {
        prefetch_of(doc, &page_draws(doc, (4, 0), scale, Rotation::None))
    }

    #[test]
    fn an_image_drawn_below_half_its_size_is_not_prefetched() {
        let doc = page_with_jpeg("q 150 0 0 150 0 0 cm /Im Do Q");
        assert_eq!(prefetched_draw(&doc, (1.0, 1.0)), Prefetch::Skip);
        // At three times the scale the same draw covers 450 pixels.
        assert_eq!(
            prefetched_draw(&doc, (3.0, 3.0)),
            Prefetch::At(Some((450, 450)))
        );
    }

    #[test]
    fn an_image_is_prefetched_for_a_draw_that_does_not_reduce_it() {
        let doc = page_with_jpeg(
            "q 100 0 0 100 0 0 cm /Im Do Q q 2 0 0 2 0 0 cm 200 0 0 200 0 0 cm /Im Do Q",
        );
        assert_eq!(
            prefetched_draw(&doc, (1.0, 1.0)),
            Prefetch::At(Some((400, 400)))
        );
    }

    #[test]
    fn restore_returns_to_the_saved_transformation() {
        let doc = page_with_jpeg("q 0.1 0 0 0.1 0 0 cm Q 400 0 0 400 0 0 cm /Im Do");
        assert_eq!(
            prefetched_draw(&doc, (1.0, 1.0)),
            Prefetch::At(Some((400, 400)))
        );
    }

    #[test]
    fn the_session_rotation_turns_the_scales_onto_the_other_axes() {
        // 400 × 100 points, drawn at 1 pixel a point across and 4 down: the
        // image is reduced across.  Turned a quarter, its width lies down.
        let doc = page_with_jpeg("q 400 0 0 100 0 0 cm /Im Do Q");
        let at = |rotation| prefetch_of(&doc, &page_draws(&doc, (4, 0), (1.0, 4.0), rotation));
        assert_eq!(at(Rotation::None), Prefetch::At(Some((400, 400))));
        assert_eq!(at(Rotation::Cw90), Prefetch::Skip);
    }

    /// A page `(4, 0)` that paints form `/F` (object 7, `/Matrix`
    /// `matrix`, content `form`, `/Im` its own resource) with `page`.
    fn page_with_form(page: &str, matrix: &str, form: &str) -> Document {
        let stream = |dict: &str, content: &str| {
            format!(
                "<< {dict} /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            )
        };
        crate::test_helpers::make_doc_with_objects(&[
            (
                4,
                "<< /Type /Page /MediaBox [0 0 400 400] /Contents 5 0 R \
                 /Resources << /XObject << /F 7 0 R >> >> >>"
                    .to_owned(),
            ),
            (5, stream("", page)),
            (
                6,
                "<< /Type /XObject /Subtype /Image /Filter /DCTDecode /Width 400 \
                 /Height 400 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 0 >>\n\
                 stream\n\nendstream"
                    .to_owned(),
            ),
            (
                7,
                stream(
                    &format!(
                        "/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Matrix {matrix} \
                         /Resources << /XObject << /Im 6 0 R /F 7 0 R >> >>"
                    ),
                    form,
                ),
            ),
        ])
    }

    /// An image drawn only inside a form is sized under the form's matrix
    /// and the transformation at its `Do`.
    #[test]
    fn an_image_drawn_in_a_form_is_sized_by_the_forms_matrix() {
        let doc = page_with_form("q 5 0 0 5 0 0 cm /F Do Q", "[100 0 0 100 0 0]", "/Im Do");
        assert_eq!(
            prefetched_draw(&doc, (1.0, 1.0)),
            Prefetch::At(Some((500, 500)))
        );
        // At a fifth of that size the draw reduces it.
        let small = page_with_form("/F Do", "[100 0 0 100 0 0]", "/Im Do");
        assert_eq!(prefetched_draw(&small, (1.0, 1.0)), Prefetch::Skip);
    }

    /// A form that paints itself ends the walk, its draws recorded.
    #[test]
    fn a_form_painting_itself_ends_the_walk() {
        let doc = page_with_form("/F Do", "[1 0 0 1 0 0]", "/F Do 400 0 0 400 0 0 cm /Im Do");
        assert_eq!(
            prefetched_draw(&doc, (1.0, 1.0)),
            Prefetch::At(Some((400, 400)))
        );
    }

    #[test]
    fn an_image_the_page_does_not_draw_is_not_prefetched() {
        let doc = page_with_jpeg("");
        assert_eq!(prefetched_draw(&doc, (1.0, 1.0)), Prefetch::Skip);
    }

    #[test]
    fn a_scale_that_is_not_finite_and_positive_is_unknown() {
        let with = |scale| PrefetchConfig {
            scale: Some(scale),
            ..PrefetchConfig::default()
        };
        assert_eq!(with((2.0, 3.0)).valid_scale(), Some((2.0, 3.0)));
        for bad in [
            (0.0, 1.0),
            (1.0, -1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert_eq!(with(bad).valid_scale(), None, "{bad:?}");
        }
    }
}
