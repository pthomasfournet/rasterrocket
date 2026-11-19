//! Where the pages of a GPU-page session went.

/// Where the pages of a GPU-page session went
/// ([`RasterSession::device_page_counts`](crate::RasterSession::device_page_counts)).
///
/// Every page the session rendered, or stopped rendering, is counted once,
/// under the first of these that applies to it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DevicePageCounts {
    /// Pages the devices rendered.
    pub rendered: u32,
    /// The pages each device rendered, in the order of
    /// [`SessionConfig::gpu_devices`](crate::SessionConfig::gpu_devices)
    /// (one entry for a session on one device): the device's name, or its
    /// selector while it is not open, and the count.  The counts sum to
    /// [`rendered`](Self::rendered).
    pub by_device: Vec<(String, u32)>,
    /// Pages painting with something the GPU page raster does not render,
    /// by reason, in the order each reason first occurred; rendered on the
    /// CPU.
    pub ineligible: Vec<(&'static str, u32)>,
    /// Pages rendered on the CPU because no device had opened by
    /// [`GPU_PAGE_OPEN_WAIT`](crate::GPU_PAGE_OPEN_WAIT) after the first
    /// page waited for one.
    pub opening: u32,
    /// Pages rendered on the CPU because no device opened.
    pub no_device: u32,
    /// Pages rendered on the CPU after device errors retired every device
    /// that opened.
    pub retired: u32,
    /// Pages beyond a device's limits or the memory it had left; rendered
    /// on the CPU, the device staying in service.
    pub device_limit: u32,
    /// Pages a device failed on, which retired it; rendered on the CPU.
    pub failed: u32,
    /// Pages whose recording stopped on their work budget, deadline or
    /// cancellation, or which were cancelled while they waited for a
    /// device to open; neither a device nor the CPU completed them.
    pub stopped: u32,
}

impl DevicePageCounts {
    /// Pages rendered on the CPU: every page counted but those the devices
    /// rendered and those stopped.
    #[must_use]
    pub fn cpu(&self) -> u32 {
        self.ineligible
            .iter()
            .fold(0u32, |n, &(_, k)| n.saturating_add(k))
            .saturating_add(self.opening)
            .saturating_add(self.no_device)
            .saturating_add(self.retired)
            .saturating_add(self.device_limit)
            .saturating_add(self.failed)
    }

    /// Count one page under `outcome`.
    pub(super) fn add(&mut self, outcome: Outcome) {
        let slot = match outcome {
            Outcome::Rendered => &mut self.rendered,
            Outcome::Ineligible(reason) => {
                let i = self
                    .ineligible
                    .iter()
                    .position(|&(r, _)| r == reason)
                    .unwrap_or_else(|| {
                        self.ineligible.push((reason, 0));
                        self.ineligible.len() - 1
                    });
                &mut self.ineligible[i].1
            }
            Outcome::Missed(Miss::Opening) => &mut self.opening,
            Outcome::Missed(Miss::NoDevice) => &mut self.no_device,
            Outcome::Missed(Miss::Retired) => &mut self.retired,
            Outcome::DeviceLimit => &mut self.device_limit,
            Outcome::Failed => &mut self.failed,
            Outcome::Stopped | Outcome::Missed(Miss::Cancelled) => &mut self.stopped,
        };
        *slot = slot.saturating_add(1);
    }
}

impl std::fmt::Display for DevicePageCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.by_device.len() > 1 {
            write!(f, "{} on the devices (", self.rendered)?;
            for (i, (name, n)) in self.by_device.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{name}: {n}")?;
            }
            write!(f, "), {} on the CPU", self.cpu())?;
        } else {
            write!(
                f,
                "{} on the device, {} on the CPU",
                self.rendered,
                self.cpu()
            )?;
        }
        let misses = [
            ("device opening", self.opening),
            ("no device", self.no_device),
            ("device retired", self.retired),
            ("device limit", self.device_limit),
            ("device failed", self.failed),
        ];
        let mut parts = self
            .ineligible
            .iter()
            .copied()
            .chain(misses)
            .filter(|&(_, n)| n > 0)
            .peekable();
        if parts.peek().is_some() {
            f.write_str(" (")?;
            for (i, (what, n)) in parts.enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{what}: {n}")?;
            }
            f.write_str(")")?;
        }
        if self.stopped > 0 {
            write!(f, "; {} stopped", self.stopped)?;
        }
        Ok(())
    }
}

/// Why [`DevicePages::claim`](super::DevicePages::claim) gave no device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Miss {
    /// No device opened within [`OPEN_WAIT`](super::OPEN_WAIT) of the
    /// first page that waited.
    Opening,
    /// The page was cancelled while it waited for a device to open.
    Cancelled,
    /// No device opened.
    NoDevice,
    /// Device errors retired every device that opened.
    Retired,
}

impl Miss {
    /// The miss as the page's debug log names it.
    pub(super) const fn describe(self) -> &'static str {
        match self {
            Self::Opening => "no device has opened yet",
            Self::Cancelled => "cancelled while waiting for a device to open",
            Self::NoDevice => "no device opened",
            Self::Retired => "every device is retired",
        }
    }
}

/// Where one page went, as [`DevicePageCounts`] counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Outcome {
    Rendered,
    Ineligible(&'static str),
    Missed(Miss),
    DeviceLimit,
    Failed,
    Stopped,
}

#[cfg(test)]
mod tests {
    use super::{DevicePageCounts, Miss, Outcome};

    /// Each outcome is counted under its own field, ineligible pages by
    /// reason in order of first occurrence, and the summary names every
    /// non-zero count.
    #[test]
    fn counts_tally_and_summarise_every_outcome() {
        let mut counts = DevicePageCounts::default();
        assert_eq!(counts.to_string(), "0 on the device, 0 on the CPU");
        for outcome in [
            Outcome::Rendered,
            Outcome::Rendered,
            Outcome::Ineligible("shading"),
            Outcome::Ineligible("translucent hairline"),
            Outcome::Ineligible("shading"),
            Outcome::Missed(Miss::Opening),
            Outcome::Missed(Miss::NoDevice),
            Outcome::Missed(Miss::Retired),
            Outcome::DeviceLimit,
            Outcome::Failed,
            Outcome::Stopped,
            Outcome::Missed(Miss::Cancelled),
        ] {
            counts.add(outcome);
        }
        assert_eq!(
            counts.ineligible,
            [("shading", 2), ("translucent hairline", 1)]
        );
        assert_eq!((counts.device_limit, counts.failed), (1, 1));
        assert_eq!((counts.rendered, counts.cpu(), counts.stopped), (2, 8, 2));
        assert_eq!(
            counts.to_string(),
            "2 on the device, 8 on the CPU (shading: 2, translucent hairline: 1, \
             device opening: 1, no device: 1, device retired: 1, \
             device limit: 1, device failed: 1); 2 stopped"
        );
    }

    /// With several devices the summary names each device's pages.
    #[test]
    fn the_summary_names_the_pages_of_each_device() {
        let counts = DevicePageCounts {
            rendered: 5,
            by_device: vec![("GPU A".to_owned(), 3), ("GPU B".to_owned(), 2)],
            ineligible: vec![("shading", 1)],
            ..DevicePageCounts::default()
        };
        assert_eq!(
            counts.to_string(),
            "5 on the devices (GPU A: 3, GPU B: 2), 1 on the CPU (shading: 1)"
        );
        let one = DevicePageCounts {
            rendered: 5,
            by_device: vec![("GPU A".to_owned(), 5)],
            ..DevicePageCounts::default()
        };
        assert_eq!(one.to_string(), "5 on the device, 0 on the CPU");
    }
}
