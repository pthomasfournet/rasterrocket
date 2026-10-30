//! Decoded image `XObjects` of one page, keyed by stream object, so an
//! image drawn by several `Do` operators (in the page, its forms and its
//! pattern cells) is decoded once.
//!
//! A decode depends on the stream and, for JPEG 2000 alone, on the
//! requested draw size, which selects the resolution level.  An entry is
//! therefore returned for any request when it is not JPEG 2000 and only
//! for an equal draw size when it is, so every draw renders exactly as a
//! fresh decode would.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use pdf::ObjectId;

use crate::resources::DecodedImage;
use crate::resources::image::ImageFilter;

/// Host bytes of decoded pixels and masks the memo holds at most.  A decode
/// that would exceed it is drawn without being kept.
const BUDGET_BYTES: usize = 64 << 20;

/// The memo shared by a page renderer and the renderers of its pattern
/// cells.
pub(super) type SharedImageMemo = Rc<RefCell<ImageMemo>>;

/// Decoded images keyed by stream object.
pub(super) struct ImageMemo {
    entries: HashMap<ObjectId, Entry>,
    /// Sum of the entries' [`Entry::bytes`].
    bytes: usize,
    budget: usize,
}

struct Entry {
    image: Rc<DecodedImage>,
    /// The draw size the image was decoded for.
    draw_size: Option<(u32, u32)>,
    /// Host bytes the entry holds.
    bytes: usize,
}

impl ImageMemo {
    /// An empty memo holding at most [`BUDGET_BYTES`].
    pub(super) fn shared() -> SharedImageMemo {
        Rc::new(RefCell::new(Self::with_budget(BUDGET_BYTES)))
    }

    fn with_budget(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
            budget,
        }
    }

    /// The decode of `id` kept for a request at `draw_size`, when it equals
    /// what decoding the stream for that request returns.
    pub(super) fn get(
        &self,
        id: ObjectId,
        draw_size: Option<(u32, u32)>,
    ) -> Option<Rc<DecodedImage>> {
        let entry = self.entries.get(&id)?;
        (entry.image.image.filter != ImageFilter::Jpx || entry.draw_size == draw_size)
            .then(|| Rc::clone(&entry.image))
    }

    /// Keep `image`, the decode of `id` for a request at `draw_size`, in
    /// place of any earlier decode of `id`.  When it would exceed the
    /// budget, the earlier decode is dropped and `image` is not kept.
    /// Returns the image for drawing either way.
    pub(super) fn insert(
        &mut self,
        id: ObjectId,
        draw_size: Option<(u32, u32)>,
        image: DecodedImage,
    ) -> Rc<DecodedImage> {
        let image = Rc::new(image);
        let bytes = host_bytes(&image);
        let replaced = self.entries.get(&id).map_or(0, |e| e.bytes);
        let old = if bytes <= self.budget - (self.bytes - replaced) {
            self.bytes += bytes;
            let entry = Entry {
                image: Rc::clone(&image),
                draw_size,
                bytes,
            };
            self.entries.insert(id, entry)
        } else {
            self.entries.remove(&id)
        };
        if let Some(old) = old {
            self.bytes -= old.bytes;
        }
        image
    }
}

/// Host bytes of `image`'s pixels and mask; device-resident pixels count
/// nothing.
fn host_bytes(image: &DecodedImage) -> usize {
    let pixels = image.image.data.as_cpu().map_or(0, <[u8]>::len);
    let mask = image.image.smask.as_ref().map_or(0, |m| m.alpha().len());
    pixels + mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::image::{ImageColorSpace, ImageData, ImageDescriptor};

    fn decoded(filter: ImageFilter, width: u32) -> DecodedImage {
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
        }
    }

    const ID: ObjectId = (5, 0);

    #[test]
    fn non_jpx_decode_serves_every_draw_size() {
        let mut memo = ImageMemo::with_budget(BUDGET_BYTES);
        assert!(memo.get(ID, None).is_none());
        let kept = memo.insert(ID, Some((4, 4)), decoded(ImageFilter::Dct, 8));
        for size in [None, Some((4, 4)), Some((100, 3))] {
            let hit = memo.get(ID, size).expect("hit");
            assert!(Rc::ptr_eq(&hit, &kept));
        }
        assert!(memo.get((6, 0), None).is_none(), "other objects miss");
    }

    #[test]
    fn jpx_decode_serves_only_its_draw_size() {
        let mut memo = ImageMemo::with_budget(BUDGET_BYTES);
        let _ = memo.insert(ID, Some((4, 4)), decoded(ImageFilter::Jpx, 4));
        assert!(memo.get(ID, Some((4, 4))).is_some());
        assert!(memo.get(ID, Some((8, 8))).is_none());
        assert!(memo.get(ID, None).is_none());

        // A decode for another size replaces it.
        let _ = memo.insert(ID, Some((8, 8)), decoded(ImageFilter::Jpx, 8));
        assert!(memo.get(ID, Some((4, 4))).is_none());
        assert_eq!(memo.get(ID, Some((8, 8))).expect("hit").image.width, 8);
        assert_eq!(memo.bytes, 8);
    }

    #[test]
    fn decodes_beyond_the_budget_are_not_kept() {
        let mut memo = ImageMemo::with_budget(10);
        let drawn = memo.insert(ID, None, decoded(ImageFilter::Raw, 8));
        assert_eq!(drawn.image.width, 8);
        assert_eq!(memo.bytes, 8);
        let over = memo.insert((6, 0), None, decoded(ImageFilter::Raw, 3));
        assert_eq!(over.image.width, 3, "an image over budget is still drawn");
        assert!(memo.get((6, 0), None).is_none());
        assert_eq!(memo.bytes, 8);
        // Replacing an entry frees its bytes first.
        let _ = memo.insert(ID, None, decoded(ImageFilter::Raw, 10));
        assert_eq!(memo.bytes, 10);
    }
}
