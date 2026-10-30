# OCR Integration

rasterrocket outputs 8-bit greyscale pixel buffers (`RenderedPage`) ready for direct consumption by any OCR engine. This page documents the optimal integration pattern for two offline engines usable from Rust: **Tesseract** (via the `tesseract` crate) and **ocrs**.

For cloud vision APIs (Google Cloud Vision, GPT-5), see [LLM Vision OCR Integration](LLM-Vision-OCR-Integration).

---

## Quick comparison

| | Tesseract (`tesseract` crate) | ocrs |
|---|---|---|
| **Accuracy** | Excellent — best on clean documents, wide language support | Good — strong on printed English, weaker on complex scripts |
| **System deps** | Requires `libtesseract`, `libleptonica` + language data files installed | Pure Rust — no system deps, WASM-compatible |
| **Ingestion** | Raw pixels through `set_frame`; Tesseract copies them once into its own image | One `u8→f32` normalisation pass |
| **Threading model** | One `Tesseract` instance per thread (`Send`, not `Sync`) | Single `Arc<OcrEngine>` shared across threads (`Send + Sync`) |
| **DPI required** | Yes — pass `page.effective_dpi` or accuracy degrades | No — character size inferred from pixel dimensions |

---

## DPI guidance

`RenderedPage` carries two DPI fields:

- `dpi` — the render resolution you passed in `RasterOptions`
- `effective_dpi` — `dpi × UserUnit`, accounting for PDF `UserUnit` scaling

**Always pass `effective_dpi` to downstream consumers.** For the vast majority of PDFs `UserUnit` is 1.0 and both fields are equal, but when they differ, using `dpi` instead of `effective_dpi` misreports the resolution and degrades OCR accuracy.

Recommended render DPI for OCR: **300**. Character x-height should be 20–30 px at that resolution. 150 DPI is acceptable for large-print documents; below 150 DPI accuracy drops sharply.

---

## Tesseract via the `tesseract` crate

### Add the dependency

```toml
# Cargo.toml
[dependencies]
tesseract = "0.15"
```

Requires the Tesseract and Leptonica development files and at least one language data package:

```bash
# Debian / Ubuntu
sudo apt install libtesseract-dev libleptonica-dev tesseract-ocr-eng

# macOS
brew install tesseract
```

### Feed raw pixels, not an encoded image

`RenderedPage::pixels` is raw 8-bit grayscale, one byte per pixel, rows packed
top to bottom (`stride == width`). Hand it to Tesseract with `set_frame`
(bytes per pixel `1`, bytes per line `width`). `set_image_from_mem` is the
wrong call: it decodes an image *file* held in memory (PNG, TIFF, …) and
rejects or misreads raw pixels.

### The footgun to avoid

`tesseract::ocr_from_frame` creates a **new Tesseract instance per call**:

```rust
// ❌ DO NOT USE IN A PIPELINE — engine initialisation on every page
let text = tesseract::ocr_from_frame(&page.pixels, w, h, 1, w, "eng")?;
```

Initialisation loads the language model from disk each time; on a
1000-page document that cost dominates the run.

### Correct pattern — single-threaded

Create one `Tesseract` and reuse it across all pages. Its setters take and
return the instance by value:

```rust
use rasterrocket::{RasterOptions, raster_pdf};
use std::path::Path;
use tesseract::Tesseract;

fn ocr_pdf(path: &Path, dpi: f32) -> anyhow::Result<Vec<String>> {
    // Create once. The language model is loaded here, not per page.
    let mut tess = Tesseract::new(None, Some("eng"))?;

    let opts = RasterOptions::new(dpi);
    let mut results = Vec::new();

    for (_page_num, result) in raster_pdf(path, &opts) {
        let page = result?;
        let width = i32::try_from(page.width)?;
        let height = i32::try_from(page.height)?;

        // effective_dpi includes the page's UserUnit — always use it.
        let ppi = page.effective_dpi.round() as i32;

        tess = tess
            .set_frame(&page.pixels, width, height, 1, width)?
            .set_source_resolution(ppi);
        results.push(tess.get_text()?);
    }

    Ok(results)
}
```

### Correct pattern — multi-threaded

A `Tesseract` instance is `Send` but not `Sync`: give each worker thread its
own. Feed pages via `render_channel`:

```rust
use rasterrocket::{RasterOptions, render_channel};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use tesseract::Tesseract;

fn ocr_pdf_parallel(path: &Path, dpi: f32, workers: usize) -> anyhow::Result<Vec<(u32, String)>> {
    let opts = RasterOptions::new(dpi);
    let rx = Arc::new(Mutex::new(render_channel(path, &opts, workers * 2)));

    let (tx_out, rx_out) = std::sync::mpsc::channel();

    // Distribute work across N threads, each holding its own instance.
    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let rx = Arc::clone(&rx);
            let tx_out = tx_out.clone();
            thread::spawn(move || {
                let mut tess = Tesseract::new(None, Some("eng")).expect("tesseract init");
                loop {
                    let item = rx.lock().unwrap().recv();
                    let Ok((page_num, result)) = item else { break };
                    let page = match result {
                        Ok(p) => p,
                        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
                    };
                    let (w, h) = (page.width as i32, page.height as i32);
                    tess = tess
                        .set_frame(&page.pixels, w, h, 1, w)
                        .expect("set_frame")
                        .set_source_resolution(page.effective_dpi.round() as i32);
                    let text = tess.get_text().unwrap_or_default();
                    tx_out.send((page_num, text)).ok();
                }
            })
        })
        .collect();

    drop(tx_out);
    let mut results: Vec<(u32, String)> = rx_out.iter().collect();
    for h in handles { h.join().ok(); }
    results.sort_by_key(|(n, _)| *n);
    Ok(results)
}
```

> **Note:** `render_channel` returns `std::sync::mpsc::Receiver`, which is not `Clone`. The example above wraps it in `Arc<Mutex<_>>` so multiple threads can drain the same channel. For higher throughput, `crossbeam-channel` eliminates the per-recv mutex.

### Pages with missing content

A page whose image or content stream could not be decoded still renders, with
that part missing, and lists it in `page.diagnostics.image_failures` /
`content_failures` (an entry with `partial: true` was drawn in part). Check
them before trusting the OCR text of a page, or set `opts.strict = true` to get
an `Err` for such pages instead.

---

## ocrs

### Add the dependency

```toml
# Cargo.toml
[dependencies]
ocrs = "0.7"
rten = "0.13"        # ocrs runtime dependency
```

Download model weights (run once):

```bash
# Detection model
curl -L https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten -o text-detection.rten
# Recognition model
curl -L https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten -o text-recognition.rten
```

### The footgun to avoid

Loading model weights inside the page loop is expensive (~seconds on first call):

```rust
// ❌ DO NOT DO THIS — reloads models on every page
for page in pages {
    let engine = OcrEngine::new(OcrEngineParams { .. })?;  // expensive
    engine.get_text(&input)?;
}
```

### Correct pattern — shared engine

`OcrEngine` is `Send + Sync`. Construct once, wrap in `Arc`, share across threads:

```rust
use ocrs::{OcrEngine, OcrEngineParams, ImageSource};
use rasterrocket::{RasterOptions, raster_pdf};
use rten::Model;
use std::path::Path;
use std::sync::Arc;

fn ocr_pdf(path: &Path, dpi: f32) -> anyhow::Result<Vec<String>> {
    // Load models once. This is the expensive step (~100s ms).
    let detection_model = Model::load_file("text-detection.rten")?;
    let recognition_model = Model::load_file("text-recognition.rten")?;

    let engine = Arc::new(OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?);

    let opts = RasterOptions::new(dpi);
    let mut results = Vec::new();

    for (_page_num, result) in raster_pdf(path, &opts) {
        let page = result?;

        // Zero allocation: from_bytes creates a tensor view over page.pixels.
        // One f32 buffer is allocated inside prepare_input (u8→f32 normalisation).
        let source = ImageSource::from_bytes(&page.pixels, (page.width, page.height))?;
        let input = engine.prepare_input(source)?;

        let words = engine.detect_words(&input)?;
        let lines = engine.find_text_lines(&input, &words);
        let text_items = engine.recognize_text(&input, &lines)?;

        let text = text_items
            .iter()
            .filter_map(|item| item.as_ref())
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        results.push(text);
    }

    Ok(results)
}
```

### Multi-threaded pattern

```rust
use ocrs::{OcrEngine, OcrEngineParams, ImageSource};
use rasterrocket::{RasterOptions, render_channel};
use rten::Model;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

fn ocr_pdf_parallel(path: &Path, dpi: f32, workers: usize) -> anyhow::Result<Vec<(u32, String)>> {
    let detection_model = Model::load_file("text-detection.rten")?;
    let recognition_model = Model::load_file("text-recognition.rten")?;
    let engine = Arc::new(OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?);

    let opts = RasterOptions::new(dpi);
    let rx = Arc::new(Mutex::new(render_channel(path, &opts, workers * 2)));
    let (tx_out, rx_out) = std::sync::mpsc::channel();

    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let rx = Arc::clone(&rx);
            let tx_out = tx_out.clone();
            thread::spawn(move || {
                loop {
                    let item = rx.lock().unwrap().recv();
                    let Ok((page_num, result)) = item else { break };
                    let page = match result {
                        Ok(p) => p,
                        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
                    };
                    let source = ImageSource::from_bytes(
                        &page.pixels, (page.width, page.height)
                    ).expect("ImageSource");
                    let input = engine.prepare_input(source).expect("prepare_input");
                    let words = engine.detect_words(&input).expect("detect_words");
                    let lines = engine.find_text_lines(&input, &words);
                    let text_items = engine.recognize_text(&input, &lines).expect("recognize");
                    let text = text_items.iter()
                        .filter_map(|i| i.as_ref())
                        .map(|line| line.to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    tx_out.send((page_num, text)).ok();
                }
            })
        })
        .collect();

    drop(tx_out);
    let mut results: Vec<(u32, String)> = rx_out.iter().collect();
    for h in handles { h.join().ok(); }
    results.sort_by_key(|(n, _)| *n);
    Ok(results)
}
```

> **Note:** ocrs uses Rayon internally for detection and layout parallelism. Setting `workers` above the number of physical cores may cause over-subscription. Start with `workers = num_cpus::get() / 2` and tune from there.

---

## Choosing between Tesseract and ocrs

| Situation | Recommendation |
|-----------|---------------|
| Need wide language support or high accuracy on degraded scans | Tesseract |
| Deploying to a container or WASM with no system deps | ocrs |
| Need word-level bounding boxes | Tesseract (`get_hocr_text` / `get_tsv_text`) |
| Simplest possible Rust-only build | ocrs |
| Mixed scripts (Arabic, CJK, RTL) | Tesseract |
