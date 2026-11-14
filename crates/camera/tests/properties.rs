//! Property tests: the geometric stages return exactly the pixels their
//! documentation names, and refuse, without panicking, what lies outside
//! the image.
//!
//! Each property runs the case count given to `config` unless
//! `PROPTEST_CASES` sets another.

use proptest::prelude::*;
use rasterrocket_camera::{CameraError, Detected, GrayView, Quad, Rect, crop, rectify, split_at};

#[path = "../../../tests/support/proptest_config.rs"]
mod proptest_config;
use proptest_config::config;

/// An image with row padding: width, height, stride and buffer.
#[derive(Debug, Clone)]
struct Image {
    w: usize,
    h: usize,
    stride: usize,
    data: Vec<u8>,
}

impl Image {
    fn view(&self) -> GrayView<'_> {
        GrayView {
            data: &self.data,
            width: self.w,
            height: self.h,
            stride: self.stride,
        }
    }

    fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.stride + x]
    }
}

fn image() -> impl Strategy<Value = Image> {
    (1usize..=96, 1usize..=96, 0usize..8).prop_flat_map(|(w, h, extra)| {
        let stride = w + extra;
        // The last row needs only its live pixels.
        prop::collection::vec(any::<u8>(), stride * (h - 1) + w).prop_map(move |data| Image {
            w,
            h,
            stride,
            data,
        })
    })
}

/// Coordinates around and far outside an image, and non-finite ones.
fn coordinate() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => -20.0f64..120.0,
        1 => any::<f64>(),
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
    ]
}

proptest! {
    #![proptest_config(config(1024))]

    /// A crop is the image's pixels inside the rectangle; a rectangle that
    /// is empty or reaches outside the image is refused.
    #[test]
    fn crop_copies_the_rectangle(
        img in image(),
        (x, y, width, height) in (0usize..110, 0usize..110, 0usize..110, 0usize..110),
    ) {
        let rect = Rect { x, y, width, height };
        let inside = width > 0 && height > 0 && x + width <= img.w && y + height <= img.h;
        match crop(img.view(), rect) {
            Ok(page) => {
                prop_assert!(inside);
                prop_assert_eq!((page.width, page.height), (width, height));
                for (j, row) in page.data.chunks_exact(width).enumerate() {
                    for (i, &v) in row.iter().enumerate() {
                        prop_assert_eq!(v, img.at(x + i, y + j));
                    }
                }
            }
            Err(e) => {
                prop_assert!(!inside);
                prop_assert!(matches!(e, CameraError::RectOutOfBounds { .. }), "{:?}", e);
            }
        }
    }

    /// A split's pages are the columns left and right of the cut; a cut at
    /// or past an edge is refused.
    #[test]
    fn split_partitions_the_columns(img in image(), x in 0usize..110) {
        match split_at(img.view(), x) {
            Ok([left, right]) => {
                prop_assert!(x > 0 && x < img.w);
                prop_assert_eq!((left.width, right.width), (x, img.w - x));
                prop_assert_eq!((left.height, right.height), (img.h, img.h));
                for y in 0..img.h {
                    for c in 0..img.w {
                        let got = if c < x {
                            left.data[y * x + c]
                        } else {
                            right.data[y * (img.w - x) + c - x]
                        };
                        prop_assert_eq!(got, img.at(c, y));
                    }
                }
            }
            Err(e) => {
                prop_assert!(x == 0 || x >= img.w);
                prop_assert!(matches!(e, CameraError::SplitOutOfRange { .. }), "{:?}", e);
            }
        }
    }

    /// A quad's bounding rectangle holds exactly the pixels whose centres
    /// lie within its corners' extent on both axes.
    #[test]
    fn bounding_rect_holds_the_centres_in_the_extent(
        corners in prop::array::uniform4(prop::array::uniform2(coordinate())),
        (w, h) in (1usize..100, 1usize..100),
    ) {
        let quad = Quad { corners };
        let rect = quad.bounding_rect(w, h);
        if !corners.iter().flatten().all(|v| v.is_finite()) {
            prop_assert_eq!(rect, None);
            return Ok(());
        }
        let extent = |axis: usize| {
            let values = corners.iter().map(|c| c[axis]);
            (values.clone().fold(f64::INFINITY, f64::min), values.fold(f64::NEG_INFINITY, f64::max))
        };
        let centres = |(lo, hi): (f64, f64), len: usize| -> Vec<usize> {
            (0..len)
                .filter(|&i| {
                    let c = f64::from(u32::try_from(i).expect("small"));
                    lo <= c && c <= hi
                })
                .collect()
        };
        let (xs, ys) = (centres(extent(0), w), centres(extent(1), h));
        match rect {
            None => prop_assert!(xs.is_empty() || ys.is_empty()),
            Some(r) => {
                prop_assert_eq!((r.x..r.x + r.width).collect::<Vec<_>>(), xs);
                prop_assert_eq!((r.y..r.y + r.height).collect::<Vec<_>>(), ys);
            }
        }
    }

    /// Rectifying any quad, its corners anywhere or not finite, gives a
    /// page of 16 px or more a side and at most twice the image, or a
    /// reason for declining.
    #[test]
    fn rectify_bounds_its_output(
        (img, corners) in image().prop_flat_map(|img| {
            // Image corners moved by up to a third of the image, or anywhere.
            #[expect(clippy::cast_precision_loss, reason = "sides of at most 96")]
            let (w, h) = (img.w as f64 - 0.5, img.h as f64 - 0.5);
            let reach = w.max(h) / 3.0;
            let near = |x: f64, y: f64| {
                (-reach..=reach, -reach..=reach).prop_map(move |(dx, dy)| [x + dx, y + dy])
            };
            let moved = (near(-0.5, -0.5), near(w, -0.5), near(w, h), near(-0.5, h))
                .prop_map(<[[f64; 2]; 4]>::from);
            let anywhere = prop::array::uniform4(prop::array::uniform2(coordinate()));
            (Just(img), prop_oneof![3 => moved, 1 => anywhere])
        }),
    ) {
        match rectify(img.view(), &Quad { corners }).expect("a valid view") {
            Detected::Found(page) => {
                prop_assert!(page.width >= 16 && page.height >= 16);
                prop_assert!(page.width * page.height <= 2 * img.w * img.h);
                prop_assert_eq!(page.data.len(), page.width * page.height);
            }
            Detected::Declined(reason) => prop_assert!(!reason.is_empty()),
        }
    }
}
