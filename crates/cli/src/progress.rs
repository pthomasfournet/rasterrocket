//! The per-page lines of `--timings` and `--progress`, shared by every
//! input type.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// The per-page report of one run: a `timing:` line per page under
/// `--timings`, and a progress line with an estimate of the time left under
/// `--progress`.  Lines are written to stderr as pages finish, in
/// completion order.
pub(crate) struct PageLog {
    timings: bool,
    progress: bool,
    pages: usize,
    done: AtomicUsize,
    start: Instant,
}

impl PageLog {
    /// The report of a run of `pages` pages, its clock started now.
    pub(crate) fn new(timings: bool, progress: bool, pages: usize) -> Self {
        Self {
            timings,
            progress,
            pages,
            done: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// Run `render`, the work of page `page_num`, and report it.
    pub(crate) fn page<R>(
        &self,
        page_num: impl std::fmt::Display,
        render: impl FnOnce() -> R,
    ) -> R {
        let t0 = self.timings.then(Instant::now);
        let result = render();
        if let Some(t0) = t0 {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let tid = rayon::current_thread_index().unwrap_or(99);
            note!("timing: page {page_num:4}  {ms:7.1}ms  thread {tid}");
        }
        if self.progress {
            let completed = self.done.fetch_add(1, Ordering::Relaxed) + 1;
            let elapsed = self.start.elapsed().as_secs_f64();
            note!(
                "rrocket: page {page_num} done  [{completed}/{}]  {elapsed:.1}s elapsed  {}",
                self.pages,
                eta(completed, self.pages, elapsed)
            );
        }
        result
    }
}

/// The estimate of the time left once `completed` of `pages` pages took
/// `elapsed` seconds, at their mean rate; unknown before two pages and
/// half a second.
fn eta(completed: usize, pages: usize, elapsed: f64) -> String {
    if elapsed <= 0.5 || completed < 2 {
        return "~?s remaining".to_owned();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "page counts are far below 2^52; the estimate is shown to 0.1 s"
    )]
    let eta_s = pages.saturating_sub(completed) as f64 * elapsed / completed as f64;
    if eta_s.is_finite() {
        format!("~{eta_s:.1}s remaining")
    } else {
        "~?s remaining".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::eta;

    #[test]
    fn the_estimate_needs_two_pages_and_half_a_second() {
        assert_eq!(eta(1, 10, 5.0), "~?s remaining");
        assert_eq!(eta(5, 10, 0.5), "~?s remaining");
        assert_eq!(eta(2, 10, 1.0), "~4.0s remaining");
        assert_eq!(eta(10, 10, 3.0), "~0.0s remaining");
    }
}
