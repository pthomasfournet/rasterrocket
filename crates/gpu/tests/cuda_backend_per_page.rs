//! End-to-end: alloc, record two ops, submit, wait, verify the result.
//!
//! Gated on `gpu-validation` so CI without a CUDA device skips it.
//! Run with:
//!   `cargo test -p gpu --features gpu-validation --test cuda_backend_per_page`

/// Runs one page on a fresh CUDA backend: the grid conversion of
/// `cmyk_host` into an output buffer pre-filled with `0xAA`, after a zero
/// fill of that buffer when `zero_first`; returns the output's device
/// bytes, `n × 4` (`n × 8` with `wide`).
#[cfg(feature = "gpu-validation")]
fn icc_clut_page(
    cmyk_host: &[u8],
    grid: &color::grid::CmykGrid,
    wide: bool,
    zero_first: bool,
) -> Vec<u8> {
    use rasterrocket_gpu::backend::cuda::CudaBackend;
    use rasterrocket_gpu::backend::layout::device_pixel_words;
    use rasterrocket_gpu::backend::{GpuBackend, params};

    let backend = CudaBackend::new()
        .expect("CudaBackend::new failed — is a CUDA device available and the driver loaded?");
    let n = cmyk_host.len() / 4;
    let n_pixels = u32::try_from(n).expect("a 32-bit pixel count");
    let rgb_len = n * 4 * device_pixel_words(if wide { 2 } else { 1 });
    let clut_host = grid.as_bytes();

    let mut cmyk = backend
        .alloc_device(cmyk_host.len())
        .expect("alloc_device(cmyk) failed — VRAM exhausted or driver error");
    let mut clut = backend
        .alloc_device(clut_host.len())
        .expect("alloc_device(clut) failed — VRAM exhausted or driver error");
    let mut rgb = backend
        .alloc_device(rgb_len)
        .expect("alloc_device(rgb) failed — VRAM exhausted or driver error");
    backend.upload(&mut cmyk, cmyk_host).expect("upload(cmyk)");
    backend.upload(&mut clut, clut_host).expect("upload(clut)");
    backend
        .upload(&mut rgb, &vec![0xAA; rgb_len])
        .expect("upload(rgb)");

    backend
        .begin_page()
        .expect("begin_page failed — recorder did not enter page-recording state");
    if zero_first {
        backend
            .record_zero_buffer(&mut rgb, rgb_len)
            .expect("record_zero_buffer failed — memset launch error");
    }
    backend
        .record_icc_clut(params::IccClutParams {
            cmyk: &cmyk,
            rgb: &mut rgb,
            clut: &clut,
            n_pixels,
            wide,
        })
        .expect("record_icc_clut failed — kernel launch error or arg mismatch");
    let fence = backend
        .submit_page()
        .expect("submit_page failed — could not record CudaEvent on the stream");
    backend
        .wait_page(fence)
        .expect("wait_page failed — event.synchronize() reported a CUDA error");

    let mut words = vec![0u8; rgb_len];
    backend.download(&rgb, &mut words).expect("download(rgb)");
    backend.free_device(cmyk);
    backend.free_device(clut);
    backend.free_device(rgb);
    words
}

/// 1 024 pseudo-random CMYK pixels.
#[cfg(feature = "gpu-validation")]
fn cmyk_pixels() -> Vec<u8> {
    (0..1024 * 4)
        .map(|i| u8::try_from(i * 37 % 256).expect("below 256"))
        .collect()
}

#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_per_page_zero_fill_then_icc_clut() {
    use rasterrocket_gpu::backend::layout::unpad_pixels;

    let cmyk_host = cmyk_pixels();
    let grid = rasterrocket_gpu::backend::random_grid(5, [1; 4], color::grid::NodeRange::Extended);
    let words = icc_clut_page(&cmyk_host, &grid, false, true);

    // Both operations run in recording order under the one page fence:
    // the transform overwrites the zero fill, which overwrote the 0xAA
    // fill, so the buffer holds the transform's output.
    let mut got = vec![0u8; cmyk_host.len() / 4 * 3];
    unpad_pixels(&words, &mut got, 1);
    assert_eq!(got, grid.convert(&cmyk_host), "the grid's conversion");
    assert!(
        got.iter().any(|&b| b != 0),
        "output is all zero: the zero fill ran after the transform"
    );
}

/// The kernel writes whole words: the bytes past each pixel's samples are
/// zero on an output buffer it alone writes, at either width.
#[cfg(feature = "gpu-validation")]
#[test]
fn cuda_backend_icc_clut_writes_the_pad_bytes() {
    use rasterrocket_gpu::backend::layout::unpad_pixels;

    let cmyk_host = cmyk_pixels();
    let n = cmyk_host.len() / 4;
    let grid = rasterrocket_gpu::backend::random_grid(5, [1; 4], color::grid::NodeRange::Extended);

    let words = icc_clut_page(&cmyk_host, &grid, false, false);
    assert!(
        words.as_chunks::<4>().0.iter().all(|p| p[3] == 0),
        "the byte past each 8-bit pixel's samples is zero"
    );
    let mut got = vec![0u8; n * 3];
    unpad_pixels(&words, &mut got, 1);
    assert_eq!(got, grid.convert(&cmyk_host), "the grid's conversion");

    let words = icc_clut_page(&cmyk_host, &grid, true, false);
    assert!(
        words.as_chunks::<8>().0.iter().all(|p| p[6..] == [0, 0]),
        "the bytes past each 16-bit pixel's samples are zero"
    );
    let mut got = vec![0u8; n * 6];
    unpad_pixels(&words, &mut got, 2);
    let expected: Vec<u8> = grid
        .convert16(&cmyk_host)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(got, expected, "the grid's 16-bit conversion");
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
