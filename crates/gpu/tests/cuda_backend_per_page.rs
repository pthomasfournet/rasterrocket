//! End-to-end: alloc, record two ops, submit, wait, verify the result.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   `cargo test -p gpu --features gpu-validation --test cuda_backend_per_page`

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_per_page_zero_fill_then_icc_clut() {
    use rasterrocket_gpu::backend::cuda::CudaBackend;
    use rasterrocket_gpu::backend::{GpuBackend, params};

    const GRID_N: u32 = 2;
    let backend = CudaBackend::new()
        .expect("CudaBackend::new failed — is a CUDA device available and the driver loaded?");
    let n_pixels: u32 = 1024;
    let n = n_pixels as usize;

    let cmyk_host: Vec<u8> = (0..n * 4)
        .map(|i| u8::try_from(i * 37 % 256).expect("below 256"))
        .collect();
    let clut_host: Vec<u8> = (0..(GRID_N as usize).pow(4) * 3)
        .map(|i| u8::try_from(i * 11 % 256).expect("below 256"))
        .collect();

    let mut cmyk = backend
        .alloc_device(cmyk_host.len())
        .expect("alloc_device(cmyk) failed — VRAM exhausted or driver error");
    let mut clut = backend
        .alloc_device(clut_host.len())
        .expect("alloc_device(clut) failed — VRAM exhausted or driver error");
    let mut rgb = backend
        .alloc_device(n * 3)
        .expect("alloc_device(rgb) failed — VRAM exhausted or driver error");
    backend.upload(&mut cmyk, &cmyk_host).expect("upload(cmyk)");
    backend.upload(&mut clut, &clut_host).expect("upload(clut)");
    backend
        .upload(&mut rgb, &vec![0xAB; n * 3])
        .expect("upload(rgb)");

    backend
        .begin_page()
        .expect("begin_page failed — recorder did not enter page-recording state");
    backend
        .record_zero_buffer(&mut rgb, n * 3)
        .expect("record_zero_buffer failed — memset launch error");
    backend
        .record_icc_clut(params::IccClutParams {
            cmyk: &cmyk,
            rgb: &mut rgb,
            clut: &clut,
            n_pixels,
        })
        .expect("record_icc_clut failed — kernel launch error or arg mismatch");
    let fence = backend
        .submit_page()
        .expect("submit_page failed — could not record CudaEvent on the stream");
    backend
        .wait_page(fence)
        .expect("wait_page failed — event.synchronize() reported a CUDA error");

    // Both operations run in recording order under the one page fence:
    // the transform overwrites the zero fill, which overwrote the 0xAB
    // fill, so the buffer holds the transform's output.
    let mut got = vec![0u8; n * 3];
    backend.download(&rgb, &mut got).expect("download(rgb)");
    let want = rasterrocket_gpu::icc_cmyk_to_rgb_cpu(&cmyk_host, Some((&clut_host, GRID_N)));
    let max_delta = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| g.abs_diff(w))
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= 1,
        "GPU CLUT output differs from the CPU reference by {max_delta} LSB"
    );
    assert!(
        got.iter().any(|&b| b != 0),
        "output is all zero: the zero fill ran after the transform"
    );

    backend.free_device(cmyk);
    backend.free_device(clut);
    backend.free_device(rgb);
}

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_record_zero_buffer_round_trip() {
    // begin_page → record_zero_buffer → submit_page → wait_page must
    // complete without an `unimplemented!`, a kernel-launch error, or
    // a CUDA error.  The Vulkan-side smoke test does the readback
    // verification; here we only care that the trait method dispatches
    // through `cuMemsetD8Async` cleanly.
    use rasterrocket_gpu::backend::GpuBackend;
    use rasterrocket_gpu::backend::cuda::CudaBackend;

    const SIZE: usize = 4096;
    let backend = CudaBackend::new().expect("CudaBackend::new");
    let mut buf = backend.alloc_device(SIZE).expect("alloc_device");

    backend.begin_page().expect("begin_page");
    backend
        .record_zero_buffer(&mut buf, SIZE)
        .expect("record_zero_buffer");
    let fence = backend.submit_page().expect("submit_page");
    backend.wait_page(fence).expect("wait_page");

    backend.free_device(buf);
}
