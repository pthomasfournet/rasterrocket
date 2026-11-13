//! Start-up cost of each GPU backend, step by step, in a fresh process.
//!
//! Run:
//! ```text
//! cargo build --release -p rasterrocket-gpu --bin startup_bench [--features ...]
//! target/release/startup_bench cuda
//! target/release/startup_bench vulkan [DEVICE]
//! ```
//!
//! Each step is timed once, in the order a session reaches it; run the
//! binary several times for a distribution.  A step whose feature is not
//! compiled in is skipped.

use std::time::Instant;

/// Time `f`, print `label` with the elapsed milliseconds, and return its
/// value; exit with status 1 on an error.
fn step<T, E: std::fmt::Display>(label: &str, f: impl FnOnce() -> Result<T, E>) -> T {
    let t = Instant::now();
    let v = f().unwrap_or_else(|e| {
        eprintln!("startup_bench: {label}: {e}");
        std::process::exit(1);
    });
    println!("{label:<48} {:>8.2} ms", t.elapsed().as_secs_f64() * 1e3);
    v
}

/// A 64 × 64 conversion through a 17-node CLUT on `backend`: the first
/// kernel launch of a session.
fn first_clut<B: rasterrocket_gpu::backend::GpuBackend>(backend: &B) -> Result<Vec<u8>, String> {
    let table = vec![128_u8; 17_usize.pow(4) * 3];
    let cmyk = vec![64_u8; 64 * 64 * 4];
    rasterrocket_gpu::backend::icc_clut(backend, &cmyk, &table).map_err(|e| e.to_string())
}

fn cuda() {
    use std::sync::Arc;
    step("driver load + cuInit", cudarc::driver::result::init);
    let _primary = step("primary context (CudaContext::new)", || {
        cudarc::driver::CudaContext::new(0)
    });
    let ctx = step("GpuCtx::init (context + PTX modules)", || {
        rasterrocket_gpu::GpuCtx::init().map(Arc::new)
    });
    let _stream = step("stream (first GpuCtx::stream)", || {
        Ok::<_, String>(ctx.stream())
    });
    #[cfg(feature = "cache")]
    {
        let stream = step("image cache stream", || ctx.new_stream());
        let _cache = step("DeviceImageCache::new", || {
            Ok::<_, String>(rasterrocket_gpu::cache::DeviceImageCache::<
                rasterrocket_gpu::backend::cuda::CudaBackend,
            >::new(
                stream,
                rasterrocket_gpu::cache::VramBudget::DEFAULT,
                rasterrocket_gpu::cache::HostBudget::DEFAULT,
            ))
        });
    }
    let backend = rasterrocket_gpu::backend::cuda::CudaBackend::from_ctx(Arc::clone(&ctx));
    let _ = step("first ICC CLUT call (64 x 64)", || first_clut(&backend));
    let _ = step("second ICC CLUT call (64 x 64)", || first_clut(&backend));
    #[cfg(feature = "nvjpeg2k")]
    for i in 1..=2 {
        let _d = step(&format!("nvJPEG2000 decoder #{i}"), || {
            rasterrocket_gpu::nvjpeg2k::NvJpeg2kDecoder::new(0)
        });
    }
    #[cfg(feature = "nvjpeg")]
    for i in 1..=2 {
        let _d = step(&format!("nvJPEG decoder #{i}"), || {
            rasterrocket_gpu::nvjpeg::NvJpegDecoder::new(0)
        });
    }
    let t = Instant::now();
    drop(backend);
    drop(ctx);
    println!(
        "{:<48} {:>8.2} ms",
        "drop (context kept by the primary handle)",
        t.elapsed().as_secs_f64() * 1e3
    );
}

#[cfg(feature = "vulkan")]
fn vulkan(device: Option<&str>) {
    use rasterrocket_gpu::backend::vulkan::{DeviceSelector, VulkanBackend};
    let selector: Option<DeviceSelector> = device.map(|d| {
        d.parse().unwrap_or_else(|e| {
            eprintln!("startup_bench: {e}");
            std::process::exit(1);
        })
    });
    let backend = step("VulkanBackend (instance, device, pipeline cache)", || {
        VulkanBackend::with_selector(selector.as_ref())
    });
    let _ = step("ICC CLUT exactness probe (first kernel)", || {
        Ok::<_, String>(backend.icc_clut_is_exact())
    });
    let _ = step("second ICC CLUT call (64 x 64)", || first_clut(&backend));
    let t = Instant::now();
    drop(backend);
    println!("{:<48} {:>8.2} ms", "drop", t.elapsed().as_secs_f64() * 1e3);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let t = Instant::now();
    match args.next().as_deref() {
        Some("cuda") => cuda(),
        #[cfg(feature = "vulkan")]
        Some("vulkan") => vulkan(args.next().as_deref()),
        _ => {
            eprintln!("usage: startup_bench cuda | vulkan [DEVICE]");
            std::process::exit(2);
        }
    }
    println!(
        "{:<48} {:>8.2} ms",
        "total",
        t.elapsed().as_secs_f64() * 1e3
    );
}
