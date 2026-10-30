# rasterrocket-camera

Cleanup of camera-captured pages on 8-bit grayscale images: page crop,
perspective rectification, double-page spread splitting and an experimental
dewarp. It has no dependencies beyond `std`, runs on the calling thread, keeps
no global state and returns on any input without panicking (MSRV 1.98).

Each stage detects before it resamples; a stage that is not confident leaves
the page unchanged and records why in the `Report`, so flatbed scans and
born-digital pages come back as given. Only malformed input is an error.

```rust
use rasterrocket_camera::{CameraError, CameraOptions, GrayView, process};

fn clean(pixels: &[u8], width: usize, height: usize) -> Result<(), CameraError> {
    let mut opts = CameraOptions::default(); // every stage off
    opts.rectify = true;
    opts.split_spreads = true;
    let out = process(GrayView::packed(pixels, width, height), &opts)?;
    for page in &out.pages {
        // page.width, page.height, page.data (packed, stride == width)
    }
    println!("split: {:?}", out.report.split);
    Ok(())
}
```

`Report::replay` applies the same geometry to another image of the same size,
such as each channel of the colour image whose luma was processed. `rrocket`
exposes the stages as `--crop`, `--rectify`, `--split-spreads` and `--dewarp`.
Stage criteria and types: the
[API reference](../../docs/api-reference.md#rasterrocket-camera-crate--camera-capture-page-cleanup).
