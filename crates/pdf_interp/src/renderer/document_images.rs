//! Decoded image `XObjects` shared by the pages of one document, so an
//! image drawn on many pages (a letterhead, a background scan) is decoded
//! once rather than once per page.
//!
//! An image is kept only once a second page requests it: the first
//! request records the page, and the decode that page keeps in its own
//! memo is offered here by weak reference.  When the second page arrives
//! while the first still holds that decode, it is kept without decoding
//! again; otherwise the second page decodes it here, and pages requesting
//! it meanwhile wait for that one decode.  A document whose images each
//! appear on one page therefore keeps nothing.
//!
//! Only host-resident decodes are kept; device-resident ones are shared
//! by the device image cache.  Kept decodes total at most the store's
//! byte budget; a decode beyond it, or a JPEG 2000 decode at a draw size
//! other than the one kept, is decoded by each page as before.
//!
//! A failure is shared only with requests at the draw size it was made
//! for: a JPEG 2000 decode can fail at one resolution level and succeed at
//! another.
//!
//! No decode path below the page fan-out runs on the thread pool, so a
//! page waiting for another page's decode cannot be asked to run that
//! page; a `decode` given to [`DocumentImages::resolve`] must keep it so.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use pdf::ObjectId;

use crate::resources::DecodedImage;
use crate::resources::image::ImageFilter;

/// Host bytes of decoded pixels and masks a session's store holds at most.
pub const DOCUMENT_IMAGE_BUDGET_BYTES: usize = 256 << 20;

/// The identity of a decode: the image stream and the resource context
/// whose default colour spaces replace its device space, `None` when the
/// resources in effect define none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ImageKey {
    pub(crate) id: ObjectId,
    pub(crate) context: Option<ObjectId>,
}

/// What decoding an image `XObject` produced, as kept for other pages.
#[derive(Clone)]
pub(crate) enum Outcome {
    /// The decode, made for a request at `draw_size`.
    Image {
        image: Arc<DecodedImage>,
        draw_size: Option<(u32, u32)>,
    },
    /// Why decoding failed for a request at `draw_size`.
    Failed {
        reason: String,
        draw_size: Option<(u32, u32)>,
    },
    /// The `XObject` has no raster appearance.
    Absent,
    /// The decode is device-resident and is not kept; each page decodes
    /// it through the device image cache.
    Device,
}

/// Whether `image`, decoded for a request at `kept`, is what decoding its
/// stream for a request at `draw_size` returns: the requested size selects
/// the resolution level of a JPEG 2000 decode alone.
pub(crate) fn serves(
    image: &DecodedImage,
    kept: Option<(u32, u32)>,
    draw_size: Option<(u32, u32)>,
) -> bool {
    image.image.filter != ImageFilter::Jpx || kept == draw_size
}

/// A decode some pages wait for while one page makes it.
pub(crate) type SharedCell = Arc<OnceLock<Outcome>>;

/// How a page obtains a decode it does not hold.
pub(crate) enum Lookup {
    /// Decode it on the page, as the page memo then keeps it.
    Local,
    /// Draw the store's decode, made by [`DocumentImages::resolve`] on the
    /// first request.
    Shared(SharedCell),
}

/// The decoded images shared by the pages of one document.  Entries are
/// keyed by object number, so a store serves one document only.
///
/// The store holds a slot per distinct image requested, for the session's
/// life; a slot not kept holds no pixels.
pub struct DocumentImages {
    state: Mutex<State>,
    budget: usize,
}

struct State {
    slots: HashMap<ImageKey, Slot>,
    /// Sum of the host bytes of the kept decodes.
    bytes: usize,
}

enum Slot {
    /// Requested by page `page` alone; `image` is the decode it keeps, for
    /// as long as it keeps it, made for a request at `draw_size`.
    Seen {
        page: ObjectId,
        image: Weak<DecodedImage>,
        draw_size: Option<(u32, u32)>,
    },
    /// Requested by two pages or more.
    Shared(SharedCell),
    /// Not kept: device-resident or beyond the budget.
    Refused,
}

impl Default for DocumentImages {
    /// A store holding at most [`DOCUMENT_IMAGE_BUDGET_BYTES`].
    fn default() -> Self {
        Self::new(DOCUMENT_IMAGE_BUDGET_BYTES)
    }
}

impl DocumentImages {
    /// A store holding at most `budget` host bytes of decodes.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            state: Mutex::new(State {
                slots: HashMap::new(),
                bytes: 0,
            }),
            budget,
        }
    }

    /// Host bytes of the decodes the store keeps.
    #[cfg(test)]
    pub(crate) fn kept_bytes(&self) -> usize {
        self.state().bytes
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// How page `page`, which holds no decode of `key`, obtains one.
    pub(crate) fn lookup(&self, key: ImageKey, page: ObjectId) -> Lookup {
        self.state().lookup(key, page, self.budget)
    }

    /// Offer `image`, the host-resident decode page `page` made of `key`
    /// for a request at `draw_size` and keeps, to a later page's
    /// [`lookup`](Self::lookup).
    pub(crate) fn offer(
        &self,
        key: ImageKey,
        page: ObjectId,
        image: &Arc<DecodedImage>,
        draw_size: Option<(u32, u32)>,
    ) {
        if image.image.data.as_cpu().is_none() {
            return;
        }
        if let Some(Slot::Seen {
            page: first,
            image: offered,
            draw_size: offered_size,
        }) = self.state().slots.get_mut(&key)
            && *first == page
        {
            *offered = Arc::downgrade(image);
            *offered_size = draw_size;
        }
    }

    /// The outcome in `cell`, the store's entry for `key`, made by
    /// `decode` when no page has made it yet, and whether the store keeps
    /// it.  A device-resident decode, or one beyond the budget, is dropped
    /// from the store once made; the pages holding `cell` still receive it,
    /// and keep it at their own charge.
    pub(crate) fn resolve(
        &self,
        key: ImageKey,
        cell: &SharedCell,
        decode: impl FnOnce() -> Outcome,
    ) -> (Outcome, bool) {
        let mut made = None;
        let outcome = cell
            .get_or_init(|| {
                let outcome = decode();
                made = Some(match &outcome {
                    Outcome::Image { image, .. } => Some(host_bytes(image)),
                    Outcome::Failed { .. } | Outcome::Absent => Some(0),
                    Outcome::Device => None,
                });
                outcome
            })
            .clone();
        let mut state = self.state();
        if let Some(size) = made {
            state.settle(key, cell, size, self.budget);
        }
        let kept =
            matches!(state.slots.get(&key), Some(Slot::Shared(kept)) if Arc::ptr_eq(kept, cell));
        drop(state);
        (outcome, kept)
    }
}

impl State {
    /// [`DocumentImages::lookup`] in a store holding at most `budget`.
    fn lookup(&mut self, key: ImageKey, page: ObjectId, budget: usize) -> Lookup {
        let Some(slot) = self.slots.get_mut(&key) else {
            let _ = self.slots.insert(
                key,
                Slot::Seen {
                    page,
                    image: Weak::new(),
                    draw_size: None,
                },
            );
            return Lookup::Local;
        };
        match slot {
            Slot::Seen { page: first, .. } if *first == page => Lookup::Local,
            Slot::Seen {
                image, draw_size, ..
            } => {
                let cell = SharedCell::default();
                if let Some(image) = image.upgrade() {
                    let size = host_bytes(&image);
                    if size > budget - self.bytes {
                        *slot = Slot::Refused;
                        return Lookup::Local;
                    }
                    self.bytes += size;
                    let draw_size = *draw_size;
                    let _ = cell.set(Outcome::Image { image, draw_size });
                }
                *slot = Slot::Shared(Arc::clone(&cell));
                Lookup::Shared(cell)
            }
            Slot::Shared(cell) => Lookup::Shared(Arc::clone(cell)),
            Slot::Refused => Lookup::Local,
        }
    }

    /// Count `size`, the host bytes of the decode just made in `cell` for
    /// `key`, against `budget`; drop the entry when it is `None` (a
    /// device-resident decode) or does not fit.  An entry no longer holding
    /// `cell` is left as it is.
    fn settle(&mut self, key: ImageKey, cell: &SharedCell, size: Option<usize>, budget: usize) {
        let Some(slot) = self.slots.get_mut(&key) else {
            return;
        };
        if !matches!(slot, Slot::Shared(held) if Arc::ptr_eq(held, cell)) {
            return;
        }
        match size {
            Some(size) if size <= budget - self.bytes => self.bytes += size,
            _ => *slot = Slot::Refused,
        }
    }
}

/// Host bytes of `image`'s pixels and mask; device-resident pixels count
/// nothing.
pub(crate) fn host_bytes(image: &DecodedImage) -> usize {
    let pixels = image.image.data.as_cpu().map_or(0, <[u8]>::len);
    let mask = image.image.smask.as_ref().map_or(0, |m| m.alpha().len());
    pixels + mask
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::resources::image::{ImageColorSpace, ImageData, ImageDescriptor};

    /// A `width` × 1 gray decode from `filter`.
    pub(crate) fn decoded(filter: ImageFilter, width: u32) -> DecodedImage {
        DecodedImage {
            image: ImageDescriptor {
                width,
                height: 1,
                color_space: ImageColorSpace::Gray,
                data: ImageData::Cpu(vec![0; width as usize]),
                smask: None,
                filter,
            },
            source_width: 8,
            interpolate: false,
            own_soft_mask: false,
            partial: None,
        }
    }

    const KEY: ImageKey = ImageKey {
        id: (5, 0),
        context: None,
    };
    const PAGE_1: ObjectId = (10, 0);
    const PAGE_2: ObjectId = (11, 0);
    const PAGE_3: ObjectId = (12, 0);

    fn image(width: u32) -> Outcome {
        Outcome::Image {
            image: Arc::new(decoded(ImageFilter::Raw, width)),
            draw_size: None,
        }
    }

    fn shared(lookup: Lookup) -> SharedCell {
        match lookup {
            Lookup::Shared(cell) => cell,
            Lookup::Local => panic!("expected a shared entry"),
        }
    }

    #[test]
    fn one_page_s_requests_stay_local() {
        let store = DocumentImages::default();
        assert!(matches!(store.lookup(KEY, PAGE_1), Lookup::Local));
        assert!(matches!(store.lookup(KEY, PAGE_1), Lookup::Local));
        assert_eq!(store.state().bytes, 0);
    }

    #[test]
    fn second_page_decodes_once_for_every_later_page() {
        let store = DocumentImages::default();
        assert!(matches!(store.lookup(KEY, PAGE_1), Lookup::Local));
        let decodes = Cell::new(0);
        let decode = || {
            decodes.set(decodes.get() + 1);
            image(8)
        };
        for page in [PAGE_2, PAGE_3, PAGE_1] {
            let cell = shared(store.lookup(KEY, page));
            assert!(matches!(
                store.resolve(KEY, &cell, decode),
                (Outcome::Image { .. }, true)
            ));
        }
        assert_eq!(decodes.get(), 1);
        assert_eq!(store.state().bytes, 8);
    }

    #[test]
    fn a_held_first_decode_is_kept_without_decoding_again() {
        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let first = Arc::new(decoded(ImageFilter::Raw, 8));
        store.offer(KEY, PAGE_1, &first, None);
        let cell = shared(store.lookup(KEY, PAGE_2));
        let (Outcome::Image { image, .. }, true) =
            store.resolve(KEY, &cell, || panic!("no decode"))
        else {
            panic!("expected the offered image");
        };
        assert!(Arc::ptr_eq(&image, &first));
        assert_eq!(store.state().bytes, 8);
    }

    #[test]
    fn a_dropped_first_decode_is_decoded_again() {
        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let first = Arc::new(decoded(ImageFilter::Raw, 8));
        store.offer(KEY, PAGE_1, &first, None);
        drop(first);
        let cell = shared(store.lookup(KEY, PAGE_2));
        assert!(cell.get().is_none(), "the first page no longer holds it");
    }

    #[test]
    fn decodes_beyond_the_budget_are_not_kept() {
        let store = DocumentImages::new(10);
        let _ = store.lookup(KEY, PAGE_1);
        let cell = shared(store.lookup(KEY, PAGE_2));
        // The decode is returned, but not kept: the page charges it.
        assert!(matches!(
            store.resolve(KEY, &cell, || image(11)),
            (Outcome::Image { .. }, false)
        ));
        assert_eq!(store.state().bytes, 0);
        assert!(matches!(store.lookup(KEY, PAGE_3), Lookup::Local));

        // An offered decode beyond the budget is refused at once.
        let other = ImageKey { id: (6, 0), ..KEY };
        let _ = store.lookup(other, PAGE_1);
        let big = Arc::new(decoded(ImageFilter::Raw, 11));
        store.offer(other, PAGE_1, &big, None);
        assert!(matches!(store.lookup(other, PAGE_2), Lookup::Local));
    }

    #[test]
    fn device_resident_decodes_are_not_kept() {
        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let cell = shared(store.lookup(KEY, PAGE_2));
        assert!(matches!(
            store.resolve(KEY, &cell, || Outcome::Device),
            (Outcome::Device, false)
        ));
        assert!(matches!(store.lookup(KEY, PAGE_3), Lookup::Local));
    }

    #[test]
    fn failures_are_shared() {
        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let cell = shared(store.lookup(KEY, PAGE_2));
        let _ = store.resolve(KEY, &cell, || Outcome::Failed {
            reason: "broken".to_owned(),
            draw_size: Some((4, 4)),
        });
        let cell = shared(store.lookup(KEY, PAGE_3));
        let (outcome, _) = store.resolve(KEY, &cell, || panic!("no decode"));
        assert!(matches!(
            outcome,
            Outcome::Failed { reason, draw_size: Some((4, 4)) } if reason == "broken"
        ));
    }

    #[test]
    fn concurrent_pages_wait_for_one_decode() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let decodes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for page in 0..8 {
                let (store, decodes) = (&store, &decodes);
                let _ = scope.spawn(move || {
                    let cell = shared(store.lookup(KEY, (100 + page, 0)));
                    let (outcome, kept) = store.resolve(KEY, &cell, || {
                        let _ = decodes.fetch_add(1, Ordering::Relaxed);
                        image(8)
                    });
                    assert!(matches!(outcome, Outcome::Image { .. }) && kept);
                });
            }
        });
        assert_eq!(decodes.load(Ordering::Relaxed), 1);
        assert_eq!(store.state().bytes, 8);
    }

    #[test]
    fn contexts_are_separate_entries() {
        let store = DocumentImages::default();
        let _ = store.lookup(KEY, PAGE_1);
        let other = ImageKey {
            context: Some((4, 0)),
            ..KEY
        };
        assert!(matches!(store.lookup(other, PAGE_2), Lookup::Local));
    }

    #[test]
    fn jpx_decode_serves_only_its_draw_size() {
        let jpx = decoded(ImageFilter::Jpx, 4);
        assert!(serves(&jpx, Some((4, 4)), Some((4, 4))));
        assert!(!serves(&jpx, Some((4, 4)), Some((8, 8))));
        let raw = decoded(ImageFilter::Raw, 4);
        assert!(serves(&raw, Some((4, 4)), None));
    }
}
