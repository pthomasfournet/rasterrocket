//! Render a single PDF page to a PPM file using the native render stack.
//!
//! Usage: `cargo run -p rasterrocket-interp --example render_page -- <pdf> [page] [dpi] [out.ppm]`

/// Maximum pixel dimension per axis.  Matches `rasterrocket::MAX_PX_DIMENSION`;
/// the example doesn't depend on that crate so the value is duplicated here.
const MAX_PX: u32 = 32_768;

fn main() {
    let _ = env_logger::try_init();
    let mut args = std::env::args().skip(1);
    let pdf = args
        .next()
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let page = args.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
    let dpi = args
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(72.0);
    let out = args
        .next()
        .unwrap_or_else(|| "/tmp/render_out.ppm".to_string());

    let doc = rasterrocket_interp::open(&pdf).unwrap_or_else(|e| {
        eprintln!("error: failed to open {pdf}: {e}");
        std::process::exit(1);
    });
    let total = rasterrocket_interp::page_count(&doc);
    println!("Opened {pdf}: {total} pages");

    if page == 0 || page > total {
        eprintln!("error: page {page} out of range (document has {total} pages)");
        std::process::exit(1);
    }
    let page_id = doc.get_page(page - 1).unwrap_or_else(|e| {
        eprintln!("error: could not resolve page {page}: {e}");
        std::process::exit(1);
    });

    let geom = rasterrocket_interp::page_size_pts(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: could not determine size of page {page}: {e}");
        std::process::exit(1);
    });
    println!(
        "Page {page} size: {:.1} × {:.1} pt (rotate {}°)",
        geom.width_pts, geom.height_pts, geom.rotate_cw
    );

    let scale = dpi / 72.0;
    let (w, h) = geom.size_px(scale, scale);
    if w == 0 || h == 0 || w > MAX_PX || h > MAX_PX {
        eprintln!(
            "error: page {page} dimensions out of range ({w}×{h} px at {dpi} dpi; \
             max {MAX_PX} per axis)"
        );
        std::process::exit(1);
    }

    let ops = rasterrocket_interp::parse_page(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: failed to parse page {page}: {e}");
        std::process::exit(1);
    });
    println!("Page {page}: {} operators", ops.len());

    let view = rasterrocket_interp::renderer::PageView::whole(
        w,
        h,
        (scale, scale),
        geom.rotate_cw,
        geom.origin_x,
        geom.origin_y,
    );
    let mut renderer = rasterrocket_interp::renderer::PageRenderer::with_view(&view, &doc, page_id);
    renderer.execute(&ops);
    let (bitmap, _diag) = renderer.finish();

    let file = std::fs::File::create(&out).unwrap_or_else(|e| {
        eprintln!("error: could not create output file {out}: {e}");
        std::process::exit(1);
    });
    let writer = std::io::BufWriter::new(file);
    encode::write_ppm(&bitmap, writer).unwrap_or_else(|e| {
        eprintln!("error: failed to write PPM to {out}: {e}");
        std::process::exit(1);
    });
    println!("Written {out} ({w}×{h} px at {dpi} dpi)");
}
