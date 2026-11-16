//! Device timestamps between the stages of a page's command buffer.
//!
//! A [`StageClock`] owns a query pool of [`StageClock::COUNT`] timestamp
//! queries.  A page that is timed resets the pool and writes the queries in
//! order between its stages, each written once every earlier command of the
//! page has completed; once the page has been waited on,
//! [`StageClock::intervals`] reads the time between consecutive queries.

use std::sync::Arc;
use std::time::Duration;

use ash::vk;

use crate::backend::{BackendError, Result};

use super::device::DeviceCtx;
use super::error::vk_err;

/// A timestamp query pool and the device's timestamp scale.
pub(super) struct StageClock {
    device: Arc<DeviceCtx>,
    pool: vk::QueryPool,
    /// The bits of a timestamp the compute queue writes
    /// (`timestampValidBits`), as a mask.
    valid: u64,
    /// Nanoseconds per timestamp tick (`timestampPeriod`).
    period_ns: f64,
}

impl StageClock {
    /// Queries per page: one before the first stage and one after each of
    /// the three.
    pub(super) const COUNT: u32 = 4;

    /// The clock of `device`'s compute queue.
    ///
    /// # Errors
    ///
    /// When the compute queue family writes no timestamp bits, or the
    /// query pool cannot be created.
    pub(super) fn new(device: Arc<DeviceCtx>) -> Result<Self> {
        // SAFETY: the instance and the physical device outlive `device`.
        let (families, props) = unsafe {
            (
                device
                    .instance
                    .get_physical_device_queue_family_properties(device.phys),
                device.instance.get_physical_device_properties(device.phys),
            )
        };
        let bits = families
            .get(device.compute_queue_family as usize)
            .map_or(0, |f| f.timestamp_valid_bits);
        if bits == 0 {
            return Err(BackendError::msg(
                "render_page_timed: the device's compute queue writes no timestamps",
            ));
        }
        let valid = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(Self::COUNT);
        // SAFETY: `info` is a valid create info; the pool is destroyed in
        // `Drop`, after `VulkanBackend::drop` has waited for the device.
        let pool = unsafe { device.device.create_query_pool(&info, None) }
            .map_err(vk_err("vkCreateQueryPool"))?;
        Ok(Self {
            device,
            pool,
            valid,
            period_ns: f64::from(props.limits.timestamp_period),
        })
    }

    /// The query pool.
    pub(super) const fn pool(&self) -> vk::QueryPool {
        self.pool
    }

    /// The device time between each pair of consecutive queries of the
    /// last page that wrote them.
    ///
    /// # Errors
    ///
    /// When the results cannot be read.  The caller has waited for the
    /// page, so the results are available.
    pub(super) fn intervals(&self) -> Result<[Duration; Self::COUNT as usize - 1]> {
        let mut ticks = [0u64; Self::COUNT as usize];
        // SAFETY: the pool holds `COUNT` queries, all written by a page
        // that has completed.
        unsafe {
            self.device.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(vk_err("vkGetQueryPoolResults"))?;
        Ok(std::array::from_fn(|i| {
            let delta = ticks[i + 1].wrapping_sub(ticks[i]) & self.valid;
            #[expect(
                clippy::cast_precision_loss,
                reason = "a page's stage takes far fewer than 2^52 ticks"
            )]
            Duration::from_secs_f64(delta as f64 * self.period_ns * 1e-9)
        }))
    }
}

impl Drop for StageClock {
    fn drop(&mut self) {
        // SAFETY: `VulkanBackend::drop` waits for the device before the
        // page scratch holding the clock drops; no page uses the pool.
        unsafe { self.device.device.destroy_query_pool(self.pool, None) };
    }
}
