//! Device timestamps between the stages of a page's command list.
//!
//! A [`StageClock`] owns a timestamp query heap of [`StageClock::COUNT`]
//! queries and a read-back buffer their results resolve into.  A page that
//! is timed writes the queries in order between its stages, each written
//! once every earlier command of the page has completed, and resolves them
//! at its end; once the page has been waited on, [`StageClock::intervals`]
//! reads the time between consecutive queries at the compute queue's
//! timestamp frequency.

use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE_TIMESTAMP, ID3D12CommandQueue, ID3D12Device,
    ID3D12QueryHeap,
};

use super::error::device_err;
use super::memory::{self, HostBuffer, HostHeap};
use crate::backend::{BackendError, Result};

/// A timestamp query heap, the read-back buffer of its results, and the
/// queue's timestamp frequency.
pub(super) struct StageClock {
    heap: ID3D12QueryHeap,
    results: HostBuffer,
    /// Timestamp ticks per second (`GetTimestampFrequency`).
    frequency: u64,
}

impl StageClock {
    /// Queries per page: one before the first stage and one after each of
    /// the three.
    pub(super) const COUNT: u32 = 4;

    /// The clock of the compute queue `queue` of `device`.
    ///
    /// # Errors
    ///
    /// When the queue reports no timestamp frequency, or the heap or the
    /// read-back buffer cannot be created.
    pub(super) fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Result<Self> {
        // SAFETY: `queue` is live; a compute queue supports timestamps.
        let frequency = unsafe { queue.GetTimestampFrequency() }
            .map_err(device_err(device, "GetTimestampFrequency"))?;
        if frequency == 0 {
            return Err(BackendError::msg(
                "render_page_timed: the compute queue reports a timestamp frequency of 0",
            ));
        }
        let desc = D3D12_QUERY_HEAP_DESC {
            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
            Count: Self::COUNT,
            NodeMask: 0,
        };
        let mut heap: Option<ID3D12QueryHeap> = None;
        // SAFETY: `desc` outlives the call; `heap` is a valid out-pointer.
        unsafe { device.CreateQueryHeap(&raw const desc, &raw mut heap) }
            .map_err(device_err(device, "CreateQueryHeap"))?;
        let heap = heap.ok_or_else(|| BackendError::msg("CreateQueryHeap returned no heap"))?;
        let results = memory::alloc_host(
            device,
            Self::COUNT as usize * 8,
            HostHeap::Readback,
            "render_page_timed timestamps",
        )?;
        Ok(Self {
            heap,
            results,
            frequency,
        })
    }

    /// The query heap.
    pub(super) const fn heap(&self) -> &ID3D12QueryHeap {
        &self.heap
    }

    /// The read-back buffer the queries resolve into.
    pub(super) const fn results(&self) -> &HostBuffer {
        &self.results
    }

    /// The device time between each pair of consecutive queries of the
    /// last page that wrote and resolved them, which the caller has waited
    /// for.
    pub(super) fn intervals(&self) -> [Duration; Self::COUNT as usize - 1] {
        let mut ticks = [0u64; Self::COUNT as usize];
        for (t, c) in ticks
            .iter_mut()
            .zip(self.results.as_slice().as_chunks::<8>().0)
        {
            *t = u64::from_ne_bytes(*c);
        }
        intervals(&ticks, self.frequency)
    }
}

/// The time between consecutive ticks of `ticks`, of a clock of
/// `frequency > 0` ticks a second; zero where a tick precedes the one
/// before it.
fn intervals(ticks: &[u64; StageClock::COUNT as usize], frequency: u64) -> [Duration; 3] {
    std::array::from_fn(|i| {
        let delta = ticks[i + 1].saturating_sub(ticks[i]);
        #[expect(
            clippy::cast_precision_loss,
            reason = "a page's stage takes far fewer than 2^52 ticks"
        )]
        Duration::from_secs_f64(delta as f64 / frequency as f64)
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    #[test]
    fn intervals_scale_ticks_by_the_frequency() {
        assert_eq!(
            super::intervals(&[1_000, 1_500, 4_000, 3_000], 1_000_000),
            [
                Duration::from_micros(500),
                Duration::from_micros(2_500),
                Duration::ZERO
            ]
        );
    }
}
