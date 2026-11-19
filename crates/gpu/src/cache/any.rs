//! A device image cache, and an image cached in one, of any backend.
//!
//! A session runs one backend, so it holds one [`ImageCache`]; every image
//! that cache returns is a [`CachedImage`] of the same backend.  The
//! operations dispatch to the [`DeviceImageCache`] of that backend.

use std::sync::Arc;

#[cfg(all(windows, feature = "d3d12"))]
use crate::backend::d3d12::D3d12Backend;
#[cfg(feature = "vulkan")]
use crate::backend::vulkan::VulkanBackend;

use super::{
    CacheError, CachedDeviceImage, ContentHash, DeviceImageCache, DocId, ImageLayout,
    InsertRequest, ObjId,
};

/// A shared device image cache on the CUDA, Vulkan or Direct3D 12 backend.
///
/// Cloning shares the cache.
#[derive(Clone)]
pub enum ImageCache {
    /// Slabs in CUDA device memory.
    Cuda(Arc<DeviceImageCache>),
    /// Slabs in Vulkan device buffers.
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<DeviceImageCache<VulkanBackend>>),
    /// Slabs in Direct3D 12 default-heap buffers.
    #[cfg(all(windows, feature = "d3d12"))]
    D3d12(Arc<DeviceImageCache<D3d12Backend>>),
}

/// An image held by an [`ImageCache`], of the cache's backend.
///
/// Cloning shares the image; the slab stays allocated while any clone
/// lives.
#[derive(Clone, Debug)]
pub enum CachedImage {
    /// An image in CUDA device memory.
    Cuda(Arc<CachedDeviceImage>),
    /// An image in a Vulkan device buffer.
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<CachedDeviceImage<VulkanBackend>>),
    /// An image in a Direct3D 12 default-heap buffer.
    #[cfg(all(windows, feature = "d3d12"))]
    D3d12(Arc<CachedDeviceImage<D3d12Backend>>),
}

/// Evaluates `$body` with `$c` bound to the [`DeviceImageCache`] of
/// `$cache` and `$wrap` naming the [`CachedImage`] variant of its backend.
macro_rules! with_cache {
    ($cache:expr, |$c:ident, $wrap:ident| $body:expr) => {
        match $cache {
            ImageCache::Cuda($c) => {
                let $wrap = CachedImage::Cuda;
                $body
            }
            #[cfg(feature = "vulkan")]
            ImageCache::Vulkan($c) => {
                let $wrap = CachedImage::Vulkan;
                $body
            }
            #[cfg(all(windows, feature = "d3d12"))]
            ImageCache::D3d12($c) => {
                let $wrap = CachedImage::D3d12;
                $body
            }
        }
    };
}

/// Evaluates `$body` with `$i` bound to the [`CachedDeviceImage`] of
/// `$image`.
macro_rules! with_image {
    ($image:expr, |$i:ident| $body:expr) => {
        match $image {
            CachedImage::Cuda($i) => $body,
            #[cfg(feature = "vulkan")]
            CachedImage::Vulkan($i) => $body,
            #[cfg(all(windows, feature = "d3d12"))]
            CachedImage::D3d12($i) => $body,
        }
    };
}

impl ImageCache {
    /// The backend's name in diagnostics: `"cuda"`, `"vulkan"`, `"d3d12"`.
    #[must_use]
    pub const fn backend_name(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => "vulkan",
            #[cfg(all(windows, feature = "d3d12"))]
            Self::D3d12(_) => "d3d12",
        }
    }

    /// Live entry count of the VRAM tier.
    #[must_use]
    pub fn len(&self) -> usize {
        with_cache!(self, |c, _wrap| c.len())
    }

    /// Whether the VRAM tier holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        with_cache!(self, |c, _wrap| c.is_empty())
    }

    /// Live VRAM usage in bytes.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        with_cache!(self, |c, _wrap| c.used_bytes())
    }

    /// Budget in bytes.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        with_cache!(self, |c, _wrap| c.budget_bytes())
    }

    /// [`DeviceImageCache::lookup_by_id`].
    #[must_use]
    pub fn lookup_by_id(&self, doc: DocId, obj: ObjId) -> Option<CachedImage> {
        with_cache!(self, |c, wrap| c.lookup_by_id(doc, obj).map(wrap))
    }

    /// [`DeviceImageCache::lookup_by_hash`].
    #[must_use]
    pub fn lookup_by_hash(&self, hash: &ContentHash) -> Option<CachedImage> {
        with_cache!(self, |c, wrap| c.lookup_by_hash(hash).map(wrap))
    }

    /// [`DeviceImageCache::lookup_by_hash_for_doc`].
    #[must_use]
    pub fn lookup_by_hash_for_doc(
        &self,
        doc: DocId,
        obj: ObjId,
        hash: &ContentHash,
    ) -> Option<CachedImage> {
        with_cache!(self, |c, wrap| c
            .lookup_by_hash_for_doc(doc, obj, hash)
            .map(wrap))
    }

    /// [`DeviceImageCache::insert`].
    ///
    /// # Errors
    /// As [`DeviceImageCache::insert`].
    pub fn insert(&self, req: InsertRequest<'_>) -> Result<CachedImage, CacheError> {
        with_cache!(self, |c, wrap| c.insert(req).map(wrap))
    }

    /// [`DeviceImageCache::download`].
    ///
    /// # Errors
    /// [`CacheError::BackendMismatch`] when `image` is of another backend;
    /// otherwise as [`DeviceImageCache::download`].
    pub fn download(&self, image: &CachedImage) -> Result<Vec<u8>, CacheError> {
        match (self, image) {
            (Self::Cuda(c), CachedImage::Cuda(i)) => c.download(i),
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(c), CachedImage::Vulkan(i)) => c.download(i),
            #[cfg(all(windows, feature = "d3d12"))]
            (Self::D3d12(c), CachedImage::D3d12(i)) => c.download(i),
            #[cfg(device_pages)]
            _ => Err(CacheError::BackendMismatch),
        }
    }
}

impl std::fmt::Debug for ImageCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageCache")
            .field("backend", &self.backend_name())
            .field("len", &self.len())
            .field("used_bytes", &self.used_bytes())
            .finish_non_exhaustive()
    }
}

impl CachedImage {
    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        with_image!(self, |i| i.width)
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        with_image!(self, |i| i.height)
    }

    /// Pixel layout.
    #[must_use]
    pub fn layout(&self) -> ImageLayout {
        with_image!(self, |i| i.layout)
    }

    /// Bytes in VRAM occupied by the image.
    #[must_use]
    pub fn vram_bytes(&self) -> u64 {
        with_image!(self, |i| i.vram_bytes())
    }
}
