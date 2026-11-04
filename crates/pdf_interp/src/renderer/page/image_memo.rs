//! Decoded images of one page — image `XObjects` keyed by stream object
//! and default-colour-space context, inline images keyed by their bytes —
//! so an image drawn several times (by `Do` operators in the page, its
//! forms and its pattern cells, or by the same `BI … EI` block in a form
//! drawn repeatedly) is decoded once.
//!
//! A decode depends on the stream, on the default colour spaces of the
//! resources in effect and, for JPEG 2000 alone, on the requested draw
//! size, which selects the resolution level.  An entry is therefore
//! returned for any request in its context when it is not JPEG 2000 and
//! only for an equal draw size when it is, so every draw renders exactly
//! as a fresh decode would.  An inline image's key holds the resource
//! context it was decoded under with its parameters and data, since a
//! named colour space resolves through that context.
//!
//! The memo holds the document's [`DocumentImages`], through which image
//! `XObjects` drawn on several pages are decoded once for the document.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::Arc;

use pdf::ObjectId;

use crate::renderer::document_images::{self, DocumentImages, ImageKey, host_bytes};
use crate::resources::DecodedImage;

/// Host bytes of decoded pixels and masks the memo holds at most.  A decode
/// that would exceed it is drawn without being kept.  Decodes the document
/// store also holds count nothing.
const BUDGET_BYTES: usize = 64 << 20;

/// The memo shared by a page renderer and the renderers of its pattern
/// cells.
pub(super) type SharedImageMemo = Rc<RefCell<ImageMemo>>;

/// Decoded images keyed by stream object or by inline-image bytes.
pub(super) struct ImageMemo {
    entries: HashMap<ImageKey, Entry>,
    /// The reason each image failed to decode, by the draw size of the
    /// request: a failure at one resolution level of a JPEG 2000 image
    /// says nothing of another.
    failures: HashMap<(ImageKey, Option<(u32, u32)>), String>,
    inline: HashMap<Box<[u8]>, Entry>,
    /// Sum of the entries' [`Entry::bytes`].
    bytes: usize,
    budget: usize,
    /// The page the memo belongs to, as the document store knows it.
    page: ObjectId,
    /// The decodes shared by the pages of the document.
    document: Arc<DocumentImages>,
}

struct Entry {
    image: Arc<DecodedImage>,
    /// The draw size the image was decoded for.
    draw_size: Option<(u32, u32)>,
    /// Host bytes the entry holds beyond the document store.
    bytes: usize,
}

impl ImageMemo {
    /// An empty memo of page `page` holding at most [`BUDGET_BYTES`],
    /// sharing decodes with the other pages of `document`.
    pub(super) fn shared(page: ObjectId, document: Arc<DocumentImages>) -> SharedImageMemo {
        Rc::new(RefCell::new(Self::with_budget(
            BUDGET_BYTES,
            page,
            document,
        )))
    }

    fn with_budget(budget: usize, page: ObjectId, document: Arc<DocumentImages>) -> Self {
        Self {
            entries: HashMap::new(),
            failures: HashMap::new(),
            inline: HashMap::new(),
            bytes: 0,
            budget,
            page,
            document,
        }
    }

    /// The page the memo belongs to.
    pub(super) const fn page(&self) -> ObjectId {
        self.page
    }

    /// The decodes shared by the pages of the document.
    pub(super) const fn document(&self) -> &Arc<DocumentImages> {
        &self.document
    }

    /// The key of an inline image: `spaces` (the object identifying the
    /// `/ColorSpace` dictionary its colour-space names resolve in, `None`
    /// when there is none), its parameters and its data.
    pub(super) fn inline_key(spaces: Option<ObjectId>, params: &[u8], data: &[u8]) -> Box<[u8]> {
        let mut key = Vec::with_capacity(17 + params.len() + data.len());
        let (tag, (num, generation)) = spaces.map_or((0u8, (0, 0)), |id| (1, id));
        key.push(tag);
        key.extend_from_slice(&num.to_le_bytes());
        key.extend_from_slice(&generation.to_le_bytes());
        key.extend_from_slice(&(params.len() as u64).to_le_bytes());
        key.extend_from_slice(params);
        key.extend_from_slice(data);
        key.into_boxed_slice()
    }

    /// The decode kept for the inline image `key` (see
    /// [`Self::inline_key`]); an inline image has no draw size.
    pub(super) fn get_inline(&self, key: &[u8]) -> Option<Arc<DecodedImage>> {
        self.inline.get(key).map(|e| Arc::clone(&e.image))
    }

    /// Keep `image`, the decode of the inline image `key`, as
    /// [`insert`](Self::insert) does for an `XObject`.
    pub(super) fn insert_inline(
        &mut self,
        key: Box<[u8]>,
        image: DecodedImage,
    ) -> Arc<DecodedImage> {
        let image = Arc::new(image);
        let entry = Entry::new(Arc::clone(&image), None, host_bytes(&image));
        keep(&mut self.inline, &mut self.bytes, self.budget, key, entry);
        image
    }

    /// The decode of `key` kept for a request at `draw_size`, when it
    /// equals what decoding the stream for that request returns.
    pub(super) fn get(
        &self,
        key: ImageKey,
        draw_size: Option<(u32, u32)>,
    ) -> Option<Arc<DecodedImage>> {
        let entry = self.entries.get(&key)?;
        document_images::serves(&entry.image, entry.draw_size, draw_size)
            .then(|| Arc::clone(&entry.image))
    }

    /// Why decoding `key` for a request at `draw_size` failed, when it has.
    pub(super) fn failure(&self, key: ImageKey, draw_size: Option<(u32, u32)>) -> Option<&str> {
        self.failures.get(&(key, draw_size)).map(String::as_str)
    }

    /// Keep `reason`, why decoding `key` for a request at `draw_size`
    /// failed, so that a later request reports it without decoding again.
    pub(super) fn insert_failure(
        &mut self,
        key: ImageKey,
        draw_size: Option<(u32, u32)>,
        reason: String,
    ) {
        let _ = self.failures.insert((key, draw_size), reason);
    }

    /// Keep `image`, the page's decode of `key` for a request at
    /// `draw_size`, in place of any earlier decode of `key`, and, when
    /// `share`, offer it to the document store.  When it would exceed the
    /// budget, the earlier decode is dropped and `image` is not kept.
    /// Returns the image for drawing either way.
    pub(super) fn insert(
        &mut self,
        key: ImageKey,
        draw_size: Option<(u32, u32)>,
        image: DecodedImage,
        share: bool,
    ) -> Arc<DecodedImage> {
        let image = Arc::new(image);
        let entry = Entry::new(Arc::clone(&image), draw_size, host_bytes(&image));
        keep(&mut self.entries, &mut self.bytes, self.budget, key, entry);
        if share && self.entries.contains_key(&key) {
            self.document.offer(key, self.page, &image, draw_size);
        }
        image
    }

    /// Keep `image`, the document store's decode of `key` for a request at
    /// `draw_size`, in place of any earlier decode of `key`.  It charges the
    /// budget nothing when the store `kept` it, its bytes otherwise.
    pub(super) fn insert_shared(
        &mut self,
        key: ImageKey,
        draw_size: Option<(u32, u32)>,
        image: &Arc<DecodedImage>,
        kept: bool,
    ) {
        let bytes = if kept { 0 } else { host_bytes(image) };
        let entry = Entry::new(Arc::clone(image), draw_size, bytes);
        keep(&mut self.entries, &mut self.bytes, self.budget, key, entry);
    }
}

impl Entry {
    const fn new(image: Arc<DecodedImage>, draw_size: Option<(u32, u32)>, bytes: usize) -> Self {
        Self {
            image,
            draw_size,
            bytes,
        }
    }
}

/// Keep `entry` at `key` in `map` in place of any earlier entry there,
/// when its bytes fit `budget` beside the other entries counted in `used`;
/// otherwise drop the earlier entry and keep nothing.  `used` tracks the
/// change.
fn keep<K: Eq + Hash>(
    map: &mut HashMap<K, Entry>,
    used: &mut usize,
    budget: usize,
    key: K,
    entry: Entry,
) {
    let replaced = map.get(&key).map_or(0, |e| e.bytes);
    let old = if entry.bytes <= budget - (*used - replaced) {
        *used += entry.bytes;
        map.insert(key, entry)
    } else {
        map.remove(&key)
    };
    if let Some(old) = old {
        *used -= old.bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::document_images::tests::decoded;
    use crate::resources::image::ImageFilter;

    const KEY: ImageKey = ImageKey {
        id: (5, 0),
        context: None,
    };

    fn memo(budget: usize) -> ImageMemo {
        ImageMemo::with_budget(budget, (1, 0), Arc::default())
    }

    #[test]
    fn non_jpx_decode_serves_every_draw_size() {
        let mut memo = memo(BUDGET_BYTES);
        assert!(memo.get(KEY, None).is_none());
        let kept = memo.insert(KEY, Some((4, 4)), decoded(ImageFilter::Dct, 8), true);
        for size in [None, Some((4, 4)), Some((100, 3))] {
            let hit = memo.get(KEY, size).expect("hit");
            assert!(Arc::ptr_eq(&hit, &kept));
        }
        let other_object = ImageKey { id: (6, 0), ..KEY };
        assert!(memo.get(other_object, None).is_none(), "other objects miss");
        let other_context = ImageKey {
            context: Some((4, 0)),
            ..KEY
        };
        assert!(
            memo.get(other_context, None).is_none(),
            "other contexts miss"
        );
    }

    #[test]
    fn jpx_decode_serves_only_its_draw_size() {
        let mut memo = memo(BUDGET_BYTES);
        let _ = memo.insert(KEY, Some((4, 4)), decoded(ImageFilter::Jpx, 4), true);
        assert!(memo.get(KEY, Some((4, 4))).is_some());
        assert!(memo.get(KEY, Some((8, 8))).is_none());
        assert!(memo.get(KEY, None).is_none());

        // A decode for another size replaces it.
        let _ = memo.insert(KEY, Some((8, 8)), decoded(ImageFilter::Jpx, 8), true);
        assert!(memo.get(KEY, Some((4, 4))).is_none());
        assert_eq!(memo.get(KEY, Some((8, 8))).expect("hit").image.width, 8);
        assert_eq!(memo.bytes, 8);
    }

    #[test]
    fn decodes_beyond_the_budget_are_not_kept() {
        let mut memo = memo(10);
        let drawn = memo.insert(KEY, None, decoded(ImageFilter::Raw, 8), true);
        assert_eq!(drawn.image.width, 8);
        assert_eq!(memo.bytes, 8);
        let other = ImageKey { id: (6, 0), ..KEY };
        let over = memo.insert(other, None, decoded(ImageFilter::Raw, 3), true);
        assert_eq!(over.image.width, 3, "an image over budget is still drawn");
        assert!(memo.get(other, None).is_none());
        assert_eq!(memo.bytes, 8);
        // Replacing an entry frees its bytes first.
        let _ = memo.insert(KEY, None, decoded(ImageFilter::Raw, 10), true);
        assert_eq!(memo.bytes, 10);
    }

    #[test]
    fn shared_decodes_charge_nothing() {
        let mut memo = memo(10);
        let _ = memo.insert(KEY, None, decoded(ImageFilter::Raw, 10), true);
        let other = ImageKey { id: (6, 0), ..KEY };
        let shared = Arc::new(decoded(ImageFilter::Raw, 50));
        memo.insert_shared(other, None, &shared, true);
        assert!(Arc::ptr_eq(&memo.get(other, None).expect("hit"), &shared));
        assert_eq!(memo.bytes, 10);
    }

    #[test]
    fn a_decode_the_store_refused_is_charged() {
        let mut memo = memo(100);
        let refused = Arc::new(decoded(ImageFilter::Raw, 30));
        memo.insert_shared(KEY, None, &refused, false);
        assert_eq!(memo.bytes, 30);
        // Beyond the page budget, it is not kept at all.
        let other = ImageKey { id: (6, 0), ..KEY };
        memo.insert_shared(
            other,
            None,
            &Arc::new(decoded(ImageFilter::Raw, 200)),
            false,
        );
        assert!(memo.get(other, None).is_none());
        assert_eq!(memo.bytes, 30);
    }
}
